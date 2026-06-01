#!/usr/bin/env python3
"""Ship local files to the Adroit scratch workspace through the running `serve` daemon, with
no extra Duo and no token on the shared cluster: tar+gzip the inputs, base64, stream the text
in <=20 KB chunks via `adroit.py x` (each fits a Windows command line), then decode + untar on
the cluster. Small + reusable -- re-run whenever the source changes. [Peregrine]"""
from __future__ import annotations

import base64
import io
import subprocess
import tarfile
from pathlib import Path

ADROIT = r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\adroit.py"
REMOTE = "/scratch/network/fl3689/peregrine"
ROOT = Path(r"C:\Users\Fengy\Downloads\Projects\Anduril")
INCLUDE = [   # what to ship to the cluster
    "src/racer",
    "scripts/gen_synthetic_dataset.py",
    "cluster/yolo_fetch.sh",
    "cluster/yolo_smoke.sbatch",
    "cluster/yolo_train.sbatch",
]
CHUNK = 20000


def x(cmd: str) -> str:
    r = subprocess.run(["py", "-3.13", ADROIT, "x", cmd], capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def _add_file(tf: tarfile.TarFile, path: Path, arcname: str) -> None:
    data = path.read_bytes()
    if path.suffix in (".sh", ".sbatch"):
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")   # LF only, or bash chokes on \r
    info = tarfile.TarInfo(arcname.replace("\\", "/"))
    info.size = len(data)
    info.mode = 0o755 if path.suffix in (".sh", ".sbatch") else 0o644
    tf.addfile(info, io.BytesIO(data))


def build_tar_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for inc in INCLUDE:
            p = ROOT / inc
            if p.is_file():
                _add_file(tf, p, inc)
                continue
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and "__pycache__" not in sub.parts and sub.suffix not in (".pyc", ".pyo"):
                    _add_file(tf, sub, str(sub.relative_to(ROOT)))
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    b64 = build_tar_b64()
    parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    print(f"tarball: {len(b64)} b64 chars in {len(parts)} chunk(s)")
    print(x(f"mkdir -p {REMOTE} && rm -f {REMOTE}/_up.b64").strip())
    for i, p in enumerate(parts):
        x(f"printf %s {p} >> {REMOTE}/_up.b64")
        print(f"  sent chunk {i + 1}/{len(parts)}")
    out = x(
        f"base64 -d {REMOTE}/_up.b64 | tar xzf - -C {REMOTE} && rm -f {REMOTE}/_up.b64 "
        f"&& echo EXTRACTED && find {REMOTE}/src/racer -name '*.py' | wc -l "
        f"&& grep -c V_OFF {REMOTE}/src/racer/vision/synthetic.py"
    )
    print(out.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
