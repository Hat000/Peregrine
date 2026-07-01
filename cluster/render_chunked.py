r"""Chunked, blackout-RESUMABLE Blender render wrapper.

``render_entry.py`` renders straight through with NO resume, so a ShadowPC reboot mid-render (the box
powers off on disconnect) loses the whole run. This wrapper renders ``--n`` frames in CHUNKS into ONE
contiguous dataset dir (``images/train/000000..`` + ``labels/train/``), skipping any chunk already on
disk -- so re-running after a reboot resumes at the first missing chunk. Each chunk uses a distinct
seed for viewpoint/appearance variety.

  <venv>\Scripts\python.exe cluster/render_chunked.py --preset vq2_photoreal_dark_red \
      --out C:\Users\Shadow\vq2_darkred_1k_2026-07-01 --n 1000 --chunk 100 --seed0 1000

NOTE: this drives Blender (its own Python), so it sets PYTHONPATH for Blender's isolated interpreter
(see blender_gen/RUN_GUIDE.md). Defaults target this ShadowPC; override with flags/env if they move.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

_BL = r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
_ENTRY = Path(__file__).resolve().parents[1] / "src" / "racer" / "vision" / "blender_gen" / "render_entry.py"
_PYPATH = r"C:\Users\Shadow\AppData\Roaming\Python\Python313\site-packages"

_DATA_YAML = """\
# {name}: chunked photoreal render (cluster/render_chunked.py). {n} frames, preset {preset}.
path: {path}
train: images/train
val: images/train
names:
  0: gate
kpt_shape: [8, 3]
flip_idx: [1, 0, 3, 2, 5, 4, 7, 6]
"""


def _chunk_done(out: Path, off: int, chunk: int) -> bool:
    """A chunk is complete iff its LAST frame already sits at the right offset in the merged dir."""
    return (out / "images" / "train" / f"{off + chunk - 1:06d}.png").exists()


def _render_chunk(args, off: int, seed: int, env: dict) -> bool:
    tmp = Path(args.out) / f"_chunk_{off:06d}"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    cmd = [args.blender, "--background", "--python", str(_ENTRY), "--",
           "--preset", args.preset, "--out", str(tmp),
           "--n-train", str(args.chunk), "--n-val", "0", "--seed", str(seed)]
    if args.no_augment:
        cmd.append("--no-augment")
    if args.masks:
        cmd.append("--masks")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    imgs = sorted(glob.glob(str(tmp / "images" / "train" / "*.png")))
    if len(imgs) < args.chunk:
        sys.stderr.write(f"[render_chunked] chunk@{off} FAILED (got {len(imgs)}/{args.chunk}); "
                         f"blender tail:\n{(r.stdout or '')[-800:]}\n")
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    for i in range(args.chunk):
        os.replace(tmp / "images" / "train" / f"{i:06d}.png",
                   Path(args.out) / "images" / "train" / f"{off + i:06d}.png")
        os.replace(tmp / "labels" / "train" / f"{i:06d}.txt",
                   Path(args.out) / "labels" / "train" / f"{off + i:06d}.txt")
        if args.masks:
            mp = tmp / "masks" / "train" / f"{i:06d}.png"
            if mp.exists():
                (Path(args.out) / "masks" / "train").mkdir(parents=True, exist_ok=True)
                os.replace(mp, Path(args.out) / "masks" / "train" / f"{off + i:06d}.png")
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chunked resumable Blender render")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000, help="seed of chunk 0; each chunk = seed0 + index")
    ap.add_argument("--no-augment", action="store_true", default=True)
    ap.add_argument("--augment", dest="no_augment", action="store_false")
    ap.add_argument("--masks", action="store_true")
    ap.add_argument("--blender", default=_BL)
    ap.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH") or _PYPATH)
    a = ap.parse_args(argv)

    out = Path(a.out)
    (out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out / "labels" / "train").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = a.pythonpath

    n_chunks = (a.n + a.chunk - 1) // a.chunk
    print(f"[render_chunked] {a.n} frames in {n_chunks} chunks of {a.chunk} -> {out}", flush=True)
    for c in range(n_chunks):
        off = c * a.chunk
        if _chunk_done(out, off, a.chunk):
            print(f"[render_chunked] chunk {c+1}/{n_chunks} @{off} already done -- skip", flush=True)
            continue
        print(f"[render_chunked] chunk {c+1}/{n_chunks} @{off} seed={a.seed0 + c} ...", flush=True)
        if not _render_chunk(a, off, a.seed0 + c, env):
            print(f"[render_chunked] STOPPED at chunk {c+1} -- re-run to resume", flush=True)
            return 2
        print(f"[render_chunked] chunk {c+1}/{n_chunks} @{off} done", flush=True)

    (out / "data.yaml").write_text(_DATA_YAML.format(
        name=out.name, n=a.n, preset=a.preset, path=str(out).replace("\\", "/")))
    total = len(glob.glob(str(out / "images" / "train" / "*.png")))
    print(f"[render_chunked] DONE: {total} frames in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
