"""Phase 8b: fine-tune MiniMax-H3 to render proxy videos: LoRA + camera/role adapter.

  python -m world_something.phase8_train precompute --pairs outputs/pairs [--model MiniMaxAI/MiniMax-H3]
  accelerate launch --config_file configs/accelerate_fsdp.yaml -m world_something.phase8_train train \\
      --pairs outputs/pairs --out outputs/phase8
  python -m world_something.phase8_train precompute --pairs ... --tiny     # CPU test of the whole path
  python -m world_something.phase8_train train --pairs ... --out ... --tiny --steps 20

Recipe (Code World Model's, as in the plan): rank-128 LoRA on every attention and feed-forward
projection of all 50 blocks (~0.6B trainable parameters) plus the zero-initialised camera/role
adapter, video loss only, AdamW, learning rate 2e-5 with 100 warm-up steps and a cosine decay to
1e-6 over 3,534 steps, one pair per GPU per step (8 GPUs in Code World Model), bf16, gradient
checkpointing. Half of the steps hold the first 10 latent frames clean (a chunk's 34-frame
prefix), so the same weights render first chunks and continuations.

precompute writes pair_XXX/latents.pt: VAE latents of the real video (target), the proxy video
and the context photos, the text presentation (Qwen3-VL layer 50, with the context photos as
"<Picture i>" vision blocks) and every row's Plücker features. Training then needs only the
transformer.

Outputs of train: lora/pytorch_lora_weights.safetensors (diffusers format), adapter.pt,
train_log.jsonl; checkpoints every --save-every steps.
"""

import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np

LORA_TARGETS = r".*transformer_blocks\.\d+\.(attn\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.0\.proj|ff\.net\.2)"


def pair_dirs(root):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "**", "pair_*"), recursive=True) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no pair_* folders under {root}; run phase8_pairs first")
    return dirs


def precompute(args):
    import torch

    from world_something import h3
    from world_something.renderer import MundoH3, h3_prompt, load_video, resize_frames

    backend = MundoH3(model=args.model, tiny=args.tiny, device=args.device)
    for d in pair_dirs(args.pairs):
        out = os.path.join(d, "latents.pt")
        if os.path.exists(out) and not args.overwrite:
            continue
        chunk = {"dir": d, "proxy": load_video(os.path.join(d, "proxy.mp4"))}
        with open(os.path.join(d, "cameras.json")) as f:
            chunk["cameras"] = json.load(f)
        with open(os.path.join(d, "context.json")) as f:
            chunk["context"] = json.load(f)
        with open(os.path.join(d, "caption.txt")) as f:
            caption = f.read().strip()
        H0, W0 = chunk["proxy"].shape[1:3]
        size = (max(32, W0 // 5 // 32 * 32), max(32, H0 // 5 // 32 * 32)) if args.tiny else None
        cond, shape, (W, H) = backend.condition(chunk, caption, None, size=size)
        real = resize_frames(load_video(os.path.join(d, "real.mp4")), W, H)
        target = h3.vae_encode(backend.vae, real)
        torch.save({"target": target.cpu(), "proxy": cond.proxy.cpu(), "context": [c.cpu() for c in cond.context],
                    "text": cond.text.cpu(), "text_tags": cond.text_tags, "plucker": cond.plucker,
                    "caption": caption, "size": [W, H]}, out)
        print(f"  {d}: target {tuple(target.shape)}, {len(cond.context)} context photos, {cond.text.shape[1]} text rows")


class Pairs:
    def __init__(self, root):
        self.files = [os.path.join(d, "latents.pt") for d in pair_dirs(root) if os.path.exists(os.path.join(d, "latents.pt"))]
        if not self.files:
            raise SystemExit("no latents.pt; run `phase8_train precompute` first")

    def get(self, i, device, with_prefix):
        import torch

        from world_something import h3

        z = torch.load(self.files[i % len(self.files)], map_location="cpu", weights_only=False)
        target = z["target"].to(device)
        prefix = target[:, :, :h3.prefix_latent_frames(34)] if with_prefix and target.shape[2] > 10 else None
        cond = h3.Condition(text=z["text"].to(device), text_tags=z["text_tags"], context=[c.to(device) for c in z["context"]],
                            proxy=z["proxy"].to(device), plucker=z["plucker"], prefix=prefix)
        return cond, target


def lr_at(step, base, final, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model_state_dict

    from world_something import h3

    accel = None
    try:
        from accelerate import Accelerator

        accel = Accelerator(mixed_precision="no" if args.tiny else "bf16")
        device = accel.device
        is_main = accel.is_main_process
    except ImportError:
        device, is_main = torch.device(args.device or "cpu"), True
    if args.tiny:
        _, transformer = h3.tiny_components()
    else:
        from diffusers import MiniMaxH3Transformer3DModel

        transformer = MiniMaxH3Transformer3DModel.from_pretrained(args.model, subfolder=args.subfolder,
                                                                  torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    adapter = h3.attach_adapter(transformer)
    transformer.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights=True,
                                       target_modules=LORA_TARGETS))
    for p_ in adapter.parameters():
        p_.requires_grad_(True)
    if not args.tiny:
        transformer.enable_gradient_checkpointing()
    n_lora = sum(p_.numel() for n, p_ in transformer.named_parameters() if p_.requires_grad and "lora_" in n)
    n_adapter = sum(p_.numel() for p_ in adapter.parameters())
    transformer.train()
    if accel is not None:  # wrap first (FSDP), then build the optimizer over the wrapped parameters
        transformer = accel.prepare(transformer)
    params = [p_ for p_ in transformer.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    if accel is not None:
        opt = accel.prepare(opt)
    data = Pairs(args.pairs)
    os.makedirs(args.out, exist_ok=True)
    if is_main:
        print(f"Phase 8: {len(data.files)} pairs, LoRA rank {args.rank}: {n_lora / 1e6:.1f}M parameters, "
              f"adapter {n_adapter / 1e6:.2f}M; {args.steps} steps")
    log = open(os.path.join(args.out, "train_log.jsonl"), "a") if is_main else None
    rng = random.Random(args.seed + (accel.process_index if accel else 0))
    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.lr_final, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        cond, target = data.get(rng.randrange(len(data.files)), device, with_prefix=rng.random() < 0.5)
        loss, t = h3.training_loss(transformer, cond, target, shift=args.shift, generator=gen)
        if accel is not None:
            accel.backward(loss)
            accel.clip_grad_norm_(params, 1.0)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if is_main and (step % args.log_every == 0 or step == args.steps - 1):
            rec = {"step": step, "loss": round(loss.item(), 5), "t": round(float(t), 3), "lr": lr,
                   "prefix": cond.prefix is not None, "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"  step {step}: loss {rec['loss']:.4f} (t={rec['t']}, lr={lr:.2e})")
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save(transformer, adapter, args.out, get_peft_model_state_dict, accel)  # every rank (FSDP gathers)
    if is_main:
        print(f"Phase 8: saved LoRA and adapter to {args.out}")


def _clean(key):
    for junk in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module."):
        key = key.replace(junk, "")
    return key.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")


def save(transformer, adapter, out, get_state, accel=None):
    """LoRA in diffusers' format (keys 'transformer.<module>.lora_A.weight') and the adapter."""
    import torch
    from safetensors.torch import save_file

    if accel is not None and accel.distributed_type.name == "FSDP":
        full = accel.get_state_dict(transformer)  # collective: every rank must call it
        if not accel.is_main_process:
            return
        lora = {f"transformer.{_clean(k)}": v for k, v in full.items() if "lora_" in k}
        adapter_state = {_clean(k).split("proj_in.adapter.", 1)[1]: v for k, v in full.items() if "proj_in.adapter." in k}
    else:
        if accel is not None and not accel.is_main_process:
            return
        model = accel.unwrap_model(transformer) if accel is not None else transformer
        lora = {f"transformer.{k}": v for k, v in get_state(model).items()}
        adapter_state = adapter.state_dict()
    os.makedirs(os.path.join(out, "lora"), exist_ok=True)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in lora.items()},
              os.path.join(out, "lora", "pytorch_lora_weights.safetensors"))
    torch.save({k: v.detach().float().cpu() for k, v in adapter_state.items()}, os.path.join(out, "adapter.pt"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 8b: LoRA fine-tune of MiniMax-H3 on real/proxy pairs.")
    p.add_argument("cmd", choices=["precompute", "train"])
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="outputs/phase8")
    p.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--subfolder", default="transformer_ref", help="the Ref2VA transformer in the diffusers repo")
    p.add_argument("--tiny", action="store_true", help="random tiny weights: tests the path on a CPU")
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--steps", type=int, default=3534)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lr-final", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--shift", type=float, default=12.0)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.tiny and args.steps == 3534:
        args.steps, args.warmup, args.lr, args.log_every = 30, 3, 1e-3, 5
    precompute(args) if args.cmd == "precompute" else train(args)


if __name__ == "__main__":
    main()
