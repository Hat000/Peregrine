"""Recon: what are the local detector models + real frames? (spike data-availability)."""
import sys, glob, os
import numpy as np

print("=== models in ./models ===")
for p in sorted(glob.glob("models/*.pt")):
    try:
        import torch
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        # ultralytics stores a dict with 'model' (nn.Module) carrying .yaml / .kpt_shape / .names
        m = ckpt.get("model") if isinstance(ckpt, dict) else None
        yaml = getattr(m, "yaml", None) if m is not None else None
        kpt = None
        names = getattr(m, "names", None) if m is not None else None
        if isinstance(yaml, dict):
            kpt = yaml.get("kpt_shape")
        task = ckpt.get("train_args", {}).get("task") if isinstance(ckpt, dict) else None
        print(f"  {os.path.basename(p):40s} kpt_shape={kpt} names={names} task={task} size={os.path.getsize(p)//1024}KB")
    except Exception as e:
        print(f"  {os.path.basename(p):40s} ERROR {type(e).__name__}: {e}")

print("\n=== real frames (worktree handoff) ===")
fr = sorted(glob.glob(".claude/worktrees/*/handoff/shadowpc-followups-2026-06-05/task2_frames/*.png"))
print(f"  count={len(fr)}")
for f in fr[:3] + fr[-2:]:
    try:
        import cv2
        img = cv2.imread(f)
        print(f"  {os.path.basename(f):30s} shape={None if img is None else img.shape}")
    except Exception as e:
        print(f"  {os.path.basename(f)} ERR {e}")
