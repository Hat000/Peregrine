"""Push the v1.6 changed files to Adroit via the daemon, chunked b64 (argv-safe), md5-gated.
CRLF-normalizes to LF before push (cluster copies are LF; python+bash both want LF)."""
import base64
import hashlib
import os
import subprocess
import sys

ADROIT = r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\adroit.py"
SRC = r"C:\Users\Fengy\Downloads\Projects\wt-fix"
DST = "/scratch/network/fl3689/peregrine_repo"
TMP = "/scratch/network/fl3689/_push_v16.b64"
CHUNK = 18000  # raw b64 chars per call, under the 24KB argv ceiling

FILES = [
    "rl/peregrine_racing.py",
    "rl/ego_reward.py",
    "rl/peregrine_racing_ego.py",
    "rl/peregrine_train_ego.py",
    "rl/vq2_ego_curriculum.py",
    "rl/launch_v16.sh",
]


def x(cmd: str) -> str:
    r = subprocess.run([sys.executable, ADROIT, "x", cmd],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"daemon cmd failed rc={r.returncode}: {r.stderr[:400]}")
    return r.stdout


def main() -> int:
    for rel in FILES:
        lp = os.path.join(SRC, rel.replace("/", os.sep))
        with open(lp, "rb") as f:
            data = f.read()
        if b"\r\n" in data:
            data = data.replace(b"\r\n", b"\n")
            print(f"  ({rel}: CRLF->LF normalized)", flush=True)
        want = hashlib.md5(data).hexdigest()
        b64 = base64.b64encode(data).decode()
        x(f"rm -f {TMP}")
        for i in range(0, len(b64), CHUNK):
            x(f"printf %s {b64[i:i+CHUNK]} >> {TMP}")
        remote = f"{DST}/{rel}"
        got = x(f"base64 -d {TMP} > {remote} && md5sum {remote}").split()[0]
        status = "OK" if got == want else "FAIL"
        print(f"{status} {rel}: local {want} remote {got} ({len(data)} B, {len(b64)//CHUNK + 1} chunks)",
              flush=True)
        if got != want:
            return 1
    x(f"rm -f {TMP}")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
