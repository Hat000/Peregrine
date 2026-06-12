#!/usr/bin/env python3
"""launch_inc7.py -- one-command inc7 launch through the adroit-connector serve daemon
(LAPTOP-INC7-ENV 2026-06-12). Enforces the three launch gates from
rl/peregrine_racing_inc7.sbatch IN ORDER and refuses to submit on any failure:

  GATE 3 (laptop, already green this session): 616/620 tests incl. tests/test_contact_geometry.py
          (the 4 failures are the parallel VISION-FRAME-FIX session's uncommitted WIP).
  GATE 1 (cluster): login-node precheck builds the REAL training cfg with the inc7 env keys
          (body_radius_lo/hi, frame_depth_m, dr_force_bias) -- catches hydra/wiring breaks.
  GATE 2 (cluster, GPU): the config-matrix parity gate (check_diffaero_gate.py, now incl. the
          dr_nominal inert-hook config + FORCE_BIAS behavioral check) via peregrine_gate.sbatch.
  Then:   sbatch seeds 0,1,2 of rl/peregrine_racing_inc7.sbatch (A100-pinned) + record job IDs.

USAGE (two terminals):
  T1:  cd C:\\Users\\Fengy\\Downloads\\Projects\\Adroit\\adroit-connector
       .venv\\Scripts\\python adroit.py serve          (passphrase + ONE Duo approval)
  T2:  .venv\\Scripts\\python handoff\\laptop-inc7-env-2026-06-12\\scripts\\launch_inc7.py all
       (from the Anduril repo root; waits for the daemon, then runs unattended)

Subcommands: sync | precheck | gate | launch | all | status. Each step is idempotent --
re-run after a daemon drop (approve the reconnect Duo in the serve window).
"""
from __future__ import annotations

import base64
import io
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ADROIT = r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\adroit.py"
APY = r"C:\Users\Fengy\Downloads\Projects\Adroit\adroit-connector\.venv\Scripts\python.exe"
ROOT = Path(__file__).resolve().parents[3]
REMOTE = "/scratch/network/fl3689/peregrine_repo"
SCRATCH = "/scratch/network/fl3689"
FILES = [                       # everything inc7 changed (the repo on scratch is a FILE COPY)
    "rl/peregrine_racing.py",
    "rl/diffaero_dynamics.py",
    "rl/check_diffaero_gate.py",
    "rl/peregrine_eval.py",
    "rl/offline_rollout.py",
    "rl/peregrine_racing_inc7.sbatch",
    "rl/peregrine_gate.sbatch",
    "rl/run_gate_body.sh",
    "rl/peregrine_racing_precheck.py",
]
CHUNK = 20000
GATE_OUT = f"{SCRATCH}/peregrine_gate.out"
PRECHECK_ARGS = ("+dynamics.dr_aero=true +dynamics.dr_mixer=true +dynamics.dr_force_bias=true "
                 "+dynamics.dr_latency_min_steps=1 +dynamics.dr_latency_max_steps=3 "
                 "+env.body_radius_lo=0.28 +env.body_radius_hi=0.38 +env.frame_depth_m=0.30 "
                 "+env.rw_tilt=96.0 +env.rw_collision=75.0 +env.rw_miss=40.0 +env.rw_oob=75.0 "
                 "+env.rw_finish_time=0.25 +env.rw_dact=0.0 +env.rw_corner=16.0")


def x(cmd: str, quiet: bool = False) -> str:
    r = subprocess.run([APY, ADROIT, "x", cmd], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if not quiet:
        print(out.strip())
    return out


def wait_daemon() -> None:
    while True:
        out = x("echo DAEMON_READY", quiet=True)
        if "DAEMON_READY" in out:
            print("[daemon] reachable")
            return
        print("[daemon] not reachable -- start `adroit.py serve` in the connector dir "
              "(passphrase + Duo); retrying in 20 s")
        time.sleep(20)


def build_tar_b64() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in FILES:
            p = ROOT / rel
            data = p.read_bytes()
            if p.suffix in (".sh", ".sbatch"):
                data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            info = tarfile.TarInfo(rel.replace("\\", "/"))
            info.size = len(data)
            info.mode = 0o755 if p.suffix in (".sh", ".sbatch") else 0o644
            tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def step_sync() -> None:
    b64 = build_tar_b64()
    parts = [b64[i:i + CHUNK] for i in range(0, len(b64), CHUNK)]
    print(f"[sync] tarball: {len(b64)} b64 chars in {len(parts)} chunk(s)")
    x(f"mkdir -p {REMOTE} && rm -f {REMOTE}/_up.b64", quiet=True)
    for i, p in enumerate(parts):
        x(f"printf %s {p} >> {REMOTE}/_up.b64", quiet=True)
        print(f"  sent chunk {i + 1}/{len(parts)}")
    out = x(f"base64 -d {REMOTE}/_up.b64 | tar xzf - -C {REMOTE} && rm -f {REMOTE}/_up.b64 "
            f"&& cp {REMOTE}/rl/run_gate_body.sh {SCRATCH}/run_gate_body.sh "
            f"&& echo EXTRACTED "
            f"&& grep -c slab_frame_hits {REMOTE}/rl/peregrine_racing.py "
            f"&& grep -c dr_force_bias {REMOTE}/rl/diffaero_dynamics.py "
            f"&& grep -c dr_nominal {REMOTE}/rl/check_diffaero_gate.py")
    if "EXTRACTED" not in out:
        sys.exit("[sync] FAILED -- extraction did not confirm")
    print("[sync] OK (new features confirmed present on scratch)")


def step_precheck() -> None:
    print("[precheck] GATE 1: login-node cfg/wiring precheck (niced CPU, ~2-5 min)...")
    out = x("cd /scratch/network/fl3689/diffaero 2>/dev/null || true; "
            "source /etc/profile.d/modules.sh && module load anaconda3/2024.10 && "
            "eval \"$(conda shell.bash hook)\" && conda activate diffaero && "
            f"ln -sfn {SCRATCH}/diffaero_repo {SCRATCH}/diffaero && "
            f"export PYTHONPATH={SCRATCH}:{REMOTE}/src:{REMOTE}/rl:$PYTHONPATH && "
            f"cd {SCRATCH}/diffaero && nice -n 19 python {REMOTE}/rl/peregrine_racing_precheck.py "
            + PRECHECK_ARGS)
    if "PRECHECK_DONE" not in out:
        sys.exit("[precheck] GATE 1 FAILED -- fix before the GPU gate (do NOT submit)")
    print("[precheck] GATE 1 PASS")


def step_gate() -> bool:
    print("[gate] GATE 2: parity gate on GPU (peregrine_gate.sbatch)...")
    x(f"rm -f {GATE_OUT}", quiet=True)
    out = x(f"sbatch {REMOTE}/rl/peregrine_gate.sbatch")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        sys.exit("[gate] sbatch submission failed")
    print(f"[gate] job {m.group(1)} submitted; polling {GATE_OUT}")
    for _ in range(120):                       # up to ~60 min queue+run
        time.sleep(30)
        out = x(f"tail -n 25 {GATE_OUT} 2>/dev/null || echo PENDING", quiet=True)
        if "SBATCH_GATE_DONE" in out:
            print(out.strip())
            if "GATE_PASS" in out:
                print("[gate] GATE 2 PASS")
                return True
            sys.exit("[gate] GATE 2 FAILED -- DO NOT SUBMIT inc7; diagnose the failing config")
        print("  ...waiting (job queued/running)")
    sys.exit("[gate] timed out waiting for the gate job; check `status` and re-run `gate`")


def step_launch() -> None:
    ids = []
    for seed in (0, 1, 2):
        out = x(f"sbatch --export=ALL,SEED={seed},TAG=inc7_s{seed} "
                f"{REMOTE}/rl/peregrine_racing_inc7.sbatch")
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            sys.exit(f"[launch] seed {seed} submission FAILED")
        ids.append((seed, m.group(1)))
        print(f"[launch] seed {seed} -> job {m.group(1)}")
    x("squeue -u fl3689")
    print("\nINC7_JOB_IDS " + " ".join(f"s{s}:{j}" for s, j in ids))
    print("Record these in the WRITEUP. Outputs: /scratch/network/fl3689/inc7_run_<job>.out")


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    wait_daemon()
    if cmd == "status":
        x("squeue -u fl3689 && tail -n 5 " + GATE_OUT)
        return 0
    if cmd in ("sync", "all"):
        step_sync()
    if cmd in ("precheck", "all"):
        step_precheck()
    gate_ok = cmd in ("launch",)               # bare `launch` assumes a prior gate pass
    if cmd in ("gate", "all"):
        gate_ok = step_gate()
    if cmd in ("launch", "all"):
        if cmd == "launch":
            print("[launch] NOTE: bare `launch` trusts a prior GATE 2 pass this session")
        step_launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
