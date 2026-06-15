"""run_inc7_lap.py — one inc7 RL lap with --debug-obs for the R_y(pi) frame-residual canary
(and a high-speed ~17 m/s head-on-ish recording for an independent at-speed sigma anchor).

Uses the proven drive->launch->GO->wait chain. fly_rl defaults to the inc7 checkpoint + bridge OFF;
--no-auto-reset avoids the §7 sim-reset DQ behavior (safe for a recording).
"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from simops_helper import drive_to_waiting, send_keys, n_sim_procs, classify

label = sys.argv[1] if len(sys.argv) > 1 else "rl_canary"
flights = sys.argv[2] if len(sys.argv) > 2 else "1"
assert n_sim_procs() == 1, "need exactly 1 DCGame"
p = drive_to_waiting()
assert classify(p) == "WAITING", f"not WAITING: {classify(p)}"
print(f"WAITING pos_off={p['pos_off_m']}"); time.sleep(1.0)
cmd = [PYTHON, "rl/fly_rl.py", "--flights", flights, "--no-auto-reset", "--debug-obs", "--label", label]
proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1)
ready = False; t0 = time.monotonic()
while time.monotonic() - t0 < 30:
    line = proc.stdout.readline()
    if not line:
        if proc.poll() is not None:
            break
        time.sleep(0.05); continue
    print("  " + line.rstrip())
    if "waiting" in line.lower() and ("pos=yes" in line or "started=False" in line):
        ready = True; break
    if "timed out" in line.lower():
        break
if ready:
    print("GO"); send_keys("enter:1.0")
while True:
    line = proc.stdout.readline()
    if not line:
        if proc.poll() is not None:
            break
        time.sleep(0.05); continue
    print("  " + line.rstrip())
print("exit", proc.wait())
