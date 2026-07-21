"""Push a local file to Adroit THROUGH the already-authenticated daemon session.

The connector ships `pull_file.py` (remote -> local, chunked base64) but no upload that
reuses the live session: `adroit.py upload` opens a FRESH connection, which means a new
Duo push to the pilot's phone.  This is the mirror image of pull_file.py -- base64 in
bounded chunks over `adroit.py x`, so one existing approval covers the whole transfer.

Verifies the remote sha256 against the local one before declaring success, and refuses
to leave a half-written file in place (it assembles into <remote>.part and only moves it
across once the hash matches).

Usage:
    py push_via_daemon.py <local_path> <remote_path> [--chunk 7000]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADROIT = Path(r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\adroit.py")
PY = Path(r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\.venv\Scripts\python.exe")


def x(cmd: str, timeout: int = 300) -> str:
    r = subprocess.run([str(PY), str(ADROIT), "x", cmd],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"daemon call failed: {(r.stderr or r.stdout).strip()[:400]}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("remote")
    ap.add_argument("--chunk", type=int, default=7000)
    a = ap.parse_args()

    data = Path(a.local).read_bytes()
    want = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    part, tmp = a.remote + ".part", a.remote + ".b64"
    chunks = [b64[i:i + a.chunk] for i in range(0, len(b64), a.chunk)]

    print(f"pushing {a.local} -> {a.remote}")
    print(f"  {len(data)} bytes, {len(b64)} b64 chars, {len(chunks)} chunk(s)")

    x(f"rm -f {tmp} {part}")
    for i, c in enumerate(chunks, 1):
        # single quotes are safe: base64's alphabet is A-Za-z0-9+/= -- no quote, no backslash.
        x(f"printf '%s' '{c}' >> {tmp}")
        print(f"  [{i}/{len(chunks)}] {min(i * a.chunk, len(b64))} / {len(b64)}")

    x(f"base64 -d {tmp} > {part} && rm -f {tmp}")
    got = x(f"sha256sum {part}").strip().split()[0]
    if got != want:
        x(f"rm -f {part}")
        print(f"FAIL: sha mismatch\n  local  {want}\n  remote {got}\n  (remote .part removed)")
        return 1

    x(f"mv {part} {a.remote}")
    print(f"OK  {a.remote}  sha256={got[:16]}... (verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
