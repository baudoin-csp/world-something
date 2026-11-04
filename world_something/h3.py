"""MiniMax-H3 with Mundo's conditioning: proxy video + context photos + cameras (Phases 7-8).

Built on diffusers' `MiniMaxH3Transformer3DModel`, which runs one packed sequence with full
attention. The packed sequence, in order:

  text rows        Qwen3-VL layer-50 features of the presentation: a "<Picture i>: " label and
                   a vision block per context photo, then the prompt (diffusers' Ref2VA encoder)
  context rows     each context photo's clean latent (t = 0.999, H3's anchor level), on its own
                   rotary slot after the text, exactly where Ref2VA puts reference images
  proxy rows       the proxy video's latents (t = 0.999) on the *target's* (t, h, w) grid, so
                   every proxy token sits at the same rotary position as the pixels it describes
  target rows      the video being generated; with a prefix (chunks after the first), its first
                   latent frames are the previous chunk's last 34 frames, held clean
There are no audio rows: the render is silent and the audio heads are unused.

Camera conditioning: per video row, a 4x4 grid of Plücker rays (o x d, d) of the pixels it
covers, in the frame of the chunk's first context photo, goes through a zero-initialised MLP and
is added to the row's input embedding, together with a zero-initialised role embedding (target,
proxy, context, prefix). Zero initialisation leaves the released model's behaviour unchanged
until the adapter is trained (Phase 8).

H3's flow convention: t = 1 is clean, x_t = t x0 + (1 - t) noise, and the transformer predicts
the data-ward velocity v = x0 - noise. Video sigmas use the exponential shift 12.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

FPS = 24
PATCH = (1, 2, 2)
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
COND_T = 0.999
VIDEO_SHIFT = 12.0
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32
PIXELS_PER_ROW = 32          # VAE 16x, patch 2x
ROLE_TARGET, ROLE_PROXY, ROLE_CONTEXT, ROLE_PREFIX = 0, 1, 2, 3
RAYS_PER_ROW = 4             # Plücker rays per row side: 4 x 4 x 6 = 96 numbers


def latent_frames(num_frames):
    if num_frames % 17 != 5:
        raise ValueError(f"H3 encodes 17n + 5 frames, got {num_frames}")
    return (num_frames - 5) // 17 * 5 + 2


def prefix_latent_frames(prefix_frames):
    if prefix_frames % 17:
        raise ValueError(f"a prefix must be whole 17-frame VAE clips, got {prefix_frames} frames")
    return prefix_frames // 17 * 5


def latent_frame_spans(num_latent_frames):
    """Pixel frames each latent frame covers: 1, 4, 4, 4, 4, 1, 4, ... (per 17-frame clip)."""
    return [ROPE_FRAMES_PER_LATENT[i % 5] for i in range(num_latent_frames)]


# --- packed layout ----------------------------------------------------------------------------------


def _spatial_grid(dim, patch, sqrt_area):
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def frame_grid(latent_h, latent_w):
    """(h, w) rotary coordinates of one latent frame, as diffusers' `_frame_position_grid`."""
    sqrt_area = np.sqrt(latent_h * latent_w)
    gh = _spatial_grid(latent_h, PATCH[1], sqrt_area)
    gw = _spatial_grid(latent_w, PATCH[2], sqrt_area)
    hh, ww = torch.meshgrid(gh, gw, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)


def temporal_grid(num_latent_frames, origin):
    spans = torch.tensor([ROPE_FRAME_RESCALE * s for s in latent_frame_spans(num_latent_frames)], dtype=torch.float64)
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def patchify(latents):
    """(1, C, T, H, W) -> (T * H/2 * W/2, C * 4) rows, frame-major then row-major."""
    b, c, t, h, w = latents.shape
    x = latents.reshape(b, c, t, 1, h // 2, 2, w // 2, 2).permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * 4).contiguous()


def unpatchify(rows, shape):
    c, t, h, w = shape
    x = rows.reshape(1, t, h // 2, w // 2, c, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(1, c, t, h, w).contiguous()


@dataclass
class Layout:
    position_ids: torch.Tensor   # (S, 3) float64
    token_tags: torch.Tensor     # (S,)
    video_indices: torch.Tensor  # positions of all video rows (context, proxy, target), in that order
    text_indices: torch.Tensor
    audio_indices: torch.Tensor  # empty
    segments: dict               # name -> slice into the video rows
    roles: torch.Tensor          # (num_video_rows,) role per video row
    target_shape: tuple          # (C, T, H, W) of the target latents
    num_prefix_rows: int


def build_layout(text_tags, context_shapes, proxy_shape, target_shape, num_prefix_latents=0):
    """Positions and tags of [text | context photos | proxy | target] (shapes are (C, T, H, W))."""
    n_text = len(text_tags)
    _, T, H, W = target_shape
    rows_per_frame = (H // 2) * (W // 2)
    n_ctx = [s[1] * (s[2] // 2) * (s[3] // 2) for s in context_shapes]
    n_proxy = 0 if proxy_shape is None else proxy_shape[1] * rows_per_frame
    n_target = T * rows_per_frame
    S = n_text + sum(n_ctx) + n_proxy + n_target
    pos = torch.zeros(S, 3, dtype=torch.float64)
    pos[:n_text, 0] = torch.arange(n_text, dtype=torch.float64)
    cursor, clock = n_text, float(n_text)
    segments, roles = {}, []
    for k, (shape, n) in enumerate(zip(context_shapes, n_ctx)):
        pos[cursor:cursor + n, 0] = clock
        pos[cursor:cursor + n, 1:] = frame_grid(shape[2], shape[3]).repeat(shape[1], 1)
        segments[f"context_{k}"] = slice(cursor - n_text, cursor - n_text + n)
        roles.append(torch.full((n,), ROLE_CONTEXT))
        cursor += n
        clock += 1.0  # an image takes one integer rotary slot, as in Ref2VA
    target_time = temporal_grid(T, clock).repeat_interleave(rows_per_frame)
    target_hw = frame_grid(H, W).repeat(T, 1)
    if n_proxy:
        if proxy_shape[1] != T:
            raise ValueError("the proxy video must have as many latent frames as the target")
        pos[cursor:cursor + n_proxy, 0] = target_time
        pos[cursor:cursor + n_proxy, 1:] = target_hw
        segments["proxy"] = slice(cursor - n_text, cursor - n_text + n_proxy)
        roles.append(torch.full((n_proxy,), ROLE_PROXY))
        cursor += n_proxy
    pos[cursor:, 0] = target_time
    pos[cursor:, 1:] = target_hw
    segments["target"] = slice(cursor - n_text, cursor - n_text + n_target)
    n_prefix_rows = num_prefix_latents * rows_per_frame
    target_roles = torch.full((n_target,), ROLE_TARGET)
    target_roles[:n_prefix_rows] = ROLE_PREFIX
    roles.append(target_roles)
    tags = torch.full((S,), VIDEO_TAG, dtype=torch.long)
    tags[:n_text] = torch.as_tensor(text_tags, dtype=torch.long)
    return Layout(position_ids=pos, token_tags=tags, video_indices=torch.arange(n_text, S),
                  text_indices=torch.arange(n_text), audio_indices=torch.zeros(0, dtype=torch.long),
                  segments=segments, roles=torch.cat(roles), target_shape=tuple(target_shape),
                  num_prefix_rows=n_prefix_rows)


def row_timesteps(layout, t_target):
    """(distinct timesteps, index per sequence row): conditioning and prefix rows sit at COND_T."""
    S = layout.position_ids.shape[0]
    t = torch.full((S,), float(t_target), dtype=torch.float32)
    n_text = len(layout.text_indices)
    cond = layout.roles != ROLE_TARGET
    t[n_text:][cond] = max(float(t_target), COND_T)
    return torch.unique(t, sorted=True, return_inverse=True)


# --- cameras -----------------------------------------------------------------------------------------


def plucker_rows(K, c2ws, width, height, latent_spans=None, rays=RAYS_PER_ROW):
    """Plücker features (num_rows, 6 * rays^2) for a video (c2ws: one per pixel frame) or an image
    (one c2w), one entry per transformer row. Each latent frame uses its last pixel frame's camera."""
    from world_something.cameras import plucker_rays

    c2ws = [np.asarray(c, float) for c in (c2ws if isinstance(c2ws, (list, tuple)) else [c2ws])]
    if latent_spans is None:
        frames = [len(c2ws) - 1]
    else:
        ends = np.cumsum(latent_spans) - 1
        frames = [min(int(e), len(c2ws) - 1) for e in ends]
    gh, gw = height // PIXELS_PER_ROW, width // PIXELS_PER_ROW
    step = PIXELS_PER_ROW // rays
    out = []
    for f in frames:
        sub_h, sub_w = gh * rays, gw * rays
        Ks = np.asarray(K, float).copy()
        Ks[0] /= step
        Ks[1] /= step
        p = plucker_rays(Ks, c2ws[f], sub_h, sub_w)                    # (6, gh*r, gw*r)
        p = p.reshape(6, gh, rays, gw, rays).transpose(1, 3, 0, 2, 4)  # (gh, gw, 6, r, r)
        out.append(p.reshape(gh * gw, 6 * rays * rays))
    return torch.from_numpy(np.concatenate(out)).float()


class CameraRoleAdapter(nn.Module):
    """Adds camera (Plücker) and role embeddings to the video rows' input embeddings."""

    def __init__(self, hidden_size, plucker_dim=6 * RAYS_PER_ROW ** 2, width=512, num_roles=4):
        super().__init__()
        self.camera = nn.Sequential(nn.Linear(plucker_dim, width), nn.SiLU(), nn.Linear(width, hidden_size))
        self.role = nn.Embedding(num_roles, hidden_size)
        nn.init.zeros_(self.camera[2].weight)
        nn.init.zeros_(self.camera[2].bias)
        nn.init.zeros_(self.role.weight)

    def forward(self, plucker, roles):
        return self.camera(plucker) + self.role(roles)


class ProjInWithCamera(nn.Module):
    """Wraps the transformer's `proj_in`: rows -> proj_in(rows) + adapter(cameras, roles)."""

    def __init__(self, proj_in, adapter):
        super().__init__()
        self.proj_in, self.adapter = proj_in, adapter
        self.plucker, self.roles = None, None

    @property
    def weight(self):  # diffusers reads the parameter dtype of `proj_in`
        return self.proj_in.weight

    def forward(self, x):
        y = self.proj_in(x)
        if self.plucker is not None:
            extra = self.adapter(self.plucker.to(y.device, self.adapter.camera[0].weight.dtype),
                                 self.roles.to(y.device))
            y = y + extra.to(y.dtype).unsqueeze(0)
        return y


def attach_adapter(transformer, adapter=None):
    if isinstance(transformer.proj_in, ProjInWithCamera):
        return transformer.proj_in.adapter
    adapter = adapter or CameraRoleAdapter(transformer.config.hidden_size)
    adapter.to(transformer.proj_in.weight.device, torch.float32)
    transformer.proj_in = ProjInWithCamera(transformer.proj_in, adapter)
    return adapter


# --- encoding helpers --------------------------------------------------------------------------------


def frames_to_pixels(frames, device):
    """uint8 (T, H, W, 3) -> ImageNet-normalised (1, 3, T, H, W) float32."""
    x = torch.from_numpy(np.array(frames, dtype=np.uint8)).to(device).permute(3, 0, 1, 2)[None].float() / 255.0
    mean = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def vae_encode(vae, frames, seed=42):
    """H3's conditioning recipe: sampled posterior (fixed seed), fp16 rounding, normalised latents."""
    dev = next(vae.parameters()).device
    posterior = vae.encode(frames_to_pixels(frames, dev).to(next(vae.parameters()).dtype), return_dict=False)[0]
    z = posterior.sample(generator=torch.Generator(device=dev).manual_seed(seed)).to(torch.float16).float()
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    return (z - mean) / std


@torch.no_grad()
def vae_decode(vae, latents):
    """Normalised latents -> uint8 (T, H, W, 3)."""
    dev = next(vae.parameters()).device
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    video = vae.decode((latents.to(dev) * std + mean).to(next(vae.parameters()).dtype), return_dict=False)[0]
    pm = torch.tensor(PIXEL_MEAN, device=dev).view(1, 3, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=dev).view(1, 3, 1, 1, 1)
    video = (video.float() * ps + pm).clamp(0, 1)
    return (video[0].permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)


def encode_presentation(components, prompt, context_images):
    """Text rows of a Ref2VA-style presentation with the context photos as "<Picture i>" blocks,
    through diffusers' own MiniMax-H3 Ref2VA encoder (needs the Qwen3-VL text encoder)."""
    from diffusers.modular_pipelines.minimax_h3.encoders import (MiniMaxH3Ref2VATextEncoderStep,
                                                                  get_qwen3vl_prompt_embeds)
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

    refs = [MiniMaxH3ImageReference(image=np.asarray(img)) for img in context_images]
    step = MiniMaxH3Ref2VATextEncoderStep()
    vision, n_img, n_vid, ts = step._gather_vision_features(components.processor, refs, FPS)
    token_ids, token_tags = step._build_presentation(components.tokenizer, prompt, refs, n_img, n_vid, ts,
                                                     text_tag=TEXT_TAG, video_tag=VIDEO_TAG)
    emb = get_qwen3vl_prompt_embeds(components.text_encoder, components.processor, token_ids, vision,
                                    text_encoder_layer=50, device=components.text_encoder.device,
                                    dtype=components.text_encoder.dtype)
    return emb, torch.tensor(token_tags, dtype=torch.long)


# --- the conditioned model ---------------------------------------------------------------------------


@dataclass
class Condition:
    text: torch.Tensor           # (1, n_text, text_dim)
    text_tags: torch.Tensor      # (n_text,)
    context: list                # [(1, C, 1, h, w) latents] per context photo
    proxy: torch.Tensor          # (1, C, T, H, W) or None
    plucker: torch.Tensor        # (num_video_rows, 96) in layout order (context, proxy, target)
    prefix: torch.Tensor = None  # (1, C, T_prefix, H, W) clean latents of the prefix, or None


def condition_rows(cond, generator=None):
    rows = []
    for c in cond.context + ([cond.proxy] if cond.proxy is not None else []):
        noise = torch.randn(c.shape, generator=generator, device=c.device, dtype=torch.float32)
        rows.append(patchify(COND_T * c.float() + (1.0 - COND_T) * noise))
    return torch.cat(rows) if rows else None


def forward_velocity(transformer, layout, cond_rows, target_rows, t_target, cond):
    """Velocity of the target rows at timestep t_target."""
    timestep, idx = row_timesteps(layout, t_target)
    dev = target_rows.device
    video = target_rows if cond_rows is None else torch.cat([cond_rows.to(dev), target_rows])
    inner = getattr(transformer, "module", transformer)  # DDP / FSDP wrappers
    proj = inner.proj_in
    if isinstance(proj, ProjInWithCamera):
        proj.plucker, proj.roles = cond.plucker, layout.roles
    audio = torch.zeros(1, 0, inner.config.audio_in_channels, device=dev)
    out = transformer(hidden_states=video[None], audio_hidden_states=audio,
                      encoder_hidden_states=cond.text.to(dev), timestep=timestep.to(dev),
                      timestep_indices=idx.to(dev), token_tags=layout.token_tags.to(dev),
                      position_ids=layout.position_ids.to(dev), video_indices=layout.video_indices.to(dev),
                      audio_indices=layout.audio_indices.to(dev), text_indices=layout.text_indices.to(dev),
                      return_dict=False)[0][0]
    seg = layout.segments["target"]
    return out[seg].float()


def make_layout(cond, target_shape):
    n_prefix = 0 if cond.prefix is None else cond.prefix.shape[2]
    return build_layout(cond.text_tags.tolist(), [tuple(c.shape[1:]) for c in cond.context],
                        None if cond.proxy is None else tuple(cond.proxy.shape[1:]), target_shape, n_prefix)


@torch.no_grad()
def sample(transformer, cond, target_shape, steps=20, shift=VIDEO_SHIFT, seed=0, device=None):
    """Euler sampling of the target latents (1, C, T, H, W) with H3's scheduler."""
    from diffusers.schedulers import MiniMaxH3Scheduler

    device = device or next(transformer.parameters()).device
    layout = make_layout(cond, target_shape)
    sched = MiniMaxH3Scheduler(shift=shift)
    sched.set_timesteps(steps, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    C, T, H, W = target_shape
    x = patchify(torch.randn((1, C, T, H, W), generator=g)).to(device)
    prefix_rows = None if cond.prefix is None else patchify(cond.prefix.float()).to(device)
    crow = condition_rows(cond, g)
    for t in sched.timesteps:
        if prefix_rows is not None:  # the prefix is held at the anchor level, never generated
            noise = torch.randn(prefix_rows.shape, generator=g).to(device)
            x[:layout.num_prefix_rows] = COND_T * prefix_rows + (1.0 - COND_T) * noise
        v = forward_velocity(transformer, layout, crow, x, float(t), cond)
        x = sched.step(v, t, x, return_dict=False)[0]
    if prefix_rows is not None:
        x[:layout.num_prefix_rows] = prefix_rows
    return unpatchify(x.cpu(), target_shape)


def training_loss(transformer, cond, target, shift=VIDEO_SHIFT, generator=None):
    """Rectified-flow loss on the generated target rows (video loss only, as in Code World Model)."""
    shape = tuple(target.shape[1:])
    layout = make_layout(cond, shape)
    dev = target.device
    x0 = patchify(target.float())
    noise = torch.randn(x0.shape, generator=generator, device="cpu").to(dev)
    u = float(torch.rand((), generator=generator))
    sigma = shift * u / (1.0 + (shift - 1.0) * u)  # H3's exponential shift of a uniform sigma
    t = 1.0 - sigma
    xt = t * x0 + (1.0 - t) * noise
    if layout.num_prefix_rows:
        xt[:layout.num_prefix_rows] = COND_T * x0[:layout.num_prefix_rows] + (1 - COND_T) * noise[:layout.num_prefix_rows]
    crow = condition_rows(cond, generator)
    v = forward_velocity(transformer, layout, crow, xt, t, cond)
    keep = slice(layout.num_prefix_rows, None)
    return torch.mean((v[keep] - (x0 - noise)[keep]) ** 2), t


def tiny_components(text_dim=48):
    """Random-weight stand-ins with H3's real classes and shapes (for CPU tests of this path)."""
    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    vae = AutoencoderKLMiniMaxH3(block_out_channels=(8, 8, 8, 8, 8, 16), layers_per_block=1, norm_num_groups=4,
                                 decoder_num_layers=1, decoder_num_attention_heads=2, decoder_attention_head_dim=16)
    transformer = MiniMaxH3Transformer3DModel(num_attention_heads=2, attention_head_dim=32, hidden_size=64,
                                              num_layers=2, num_refiner_layers=1, ffn_dim=128, text_dim=text_dim,
                                              freq_dim=32, time_embed_hidden_dim=64, time_embed_dim=32,
                                              rope_freq_dim=4)
    return vae.eval(), transformer


def tiny_text(prompt, num_context, text_dim=48):
    """Deterministic stand-in for the Qwen3-VL presentation: hashed tokens, H3's tag layout."""
    words = prompt.split()[:64] or ["render"]
    rng = np.random.default_rng(abs(hash(prompt)) % (2 ** 31))
    n = len(words) + 4 * num_context
    tags = [VIDEO_TAG if (i < 4 * num_context and i % 4) else TEXT_TAG for i in range(n)]
    return torch.from_numpy(rng.standard_normal((1, n, text_dim)).astype(np.float32)), torch.tensor(tags)


def latent_size(width, height):
    if width % PIXELS_PER_ROW or height % PIXELS_PER_ROW:
        raise ValueError(f"H3 canvases are multiples of {PIXELS_PER_ROW} px, got {width}x{height}")
    return width // 16, height // 16


def context_size(width, height, short_edge):
    s = short_edge / min(width, height)
    w = max(PIXELS_PER_ROW, int(round(width * s / PIXELS_PER_ROW)) * PIXELS_PER_ROW)
    h = max(PIXELS_PER_ROW, int(round(height * s / PIXELS_PER_ROW)) * PIXELS_PER_ROW)
    return w, h


def math_check():
    """Sanity numbers for the README: tokens per 124-frame chunk at 640x480."""
    lw, lh = latent_size(640, 480)
    return {"latent_frames": latent_frames(124), "rows_per_frame": (lh // 2) * (lw // 2),
            "target_rows": latent_frames(124) * (lh // 2) * (lw // 2),
            "prefix_latent_frames": prefix_latent_frames(34)}
