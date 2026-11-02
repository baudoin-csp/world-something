"""Phases 1-2 of the Mundo rebuild: metric geometry (Depth Anything 3) and 3D boxes (Boxer)."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIRD_PARTY = os.environ.get("WS_THIRD_PARTY", os.path.join(REPO_ROOT, "third_party"))
BOXER_DIR = os.environ.get("BOXER_DIR", os.path.join(THIRD_PARTY, "boxer"))
DA3_MODEL = os.environ.get("DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")


def add_boxer_to_path():
    """Boxer is a plain repo (top-level `loaders`, `utils`, ...), not an installable package."""
    if not os.path.isdir(BOXER_DIR):
        raise FileNotFoundError(f"Boxer not found at {BOXER_DIR}; run ./setup.sh first")
    if BOXER_DIR not in sys.path:
        sys.path.insert(0, BOXER_DIR)
