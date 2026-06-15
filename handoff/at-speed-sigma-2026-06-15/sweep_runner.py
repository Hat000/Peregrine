"""sweep_runner.py — at-speed sigma sweep: head-on CTBR laps at a list of target speeds.

Drives the proven simops chain per pass (drive->WAITING, launch fly_vq1 --faithful with a speed
override, send GO, wait for finish), records video+telemetry, and flags gate contact (a contact lap
is INVALID per the mission validity rule -> not counted; we keep recording for diagnosis but mark it).

Derived verbatim from run_boresight_batch.py (the proven lap chain, commit on claude/trusting-maxwell);
the ONLY change is the speed override (--faithful-cruise/--faithful-max-speed) + a target-speed loop.

Usage:
  PYTHONPATH=src .venv/Scripts/python.exe handoff/at-speed-sigma-2026-06-15/sweep_runner.py \
      --speeds 8,14,20,26 --passes 4 --max-gates 2 --tag s1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from simops_helper import drive_to_waiting, send_keys, n_sim_procs, classify  # noqa: E402

MAP = "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
COMMON = ["--faithful", "--force-saved-map", "--map", MAP,
          "--gate-corner-to-center", "--rate", "100", "--sim-build", "1.0.3364"]

CONNECT_WAIT_S = 12
LAP_TIMEOUT_S = 200


def run_lap(speed: float, max_gates: int, label: str, lap_num: int) -> dict:
    print(f"\n{'='*60}\nLAP {lap_num}: speed={speed} m/s  label={label}\n{'='*60}")
    n = n_sim_procs()
    print(f"  DCGame instances: {n}")
    if n != 1:
        return {"ok": False, "reason": f"zombie: {n} DCGame procs", "speed": speed, "label": label}
    p = drive_to_waiting()
    st = classify(p)
    if st != "WAITING":
        return {"ok": False, "reason": f"drive failed: state={st}", "speed": speed, "label": label}
    print(f"  Sim WAITING (pos_off={p['pos_off_m']} m).")
    time.sleep(1.0)

    args = COMMON + ["--faithful-cruise", str(speed), "--faithful-max-speed", str(speed),
                     "--max-gates", str(max_gates), "--finish-hold-s", "1.0", "--label", label]
    cmd = [PYTHON, "scripts/fly_vq1.py"] + args
    print(f"  Launching fly_vq1 (cruise/max-speed={speed}, max-gates={max_gates}) ...")
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    t_start = time.monotonic()
    lines: list[str] = []
    ready = False
    while time.monotonic() - t_start < CONNECT_WAIT_S + 5:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            time.sleep(0.05)
            continue
        lines.append(line.rstrip())
        if "waiting: started=False pos=yes" in line:
            ready = True
            print(f"  fly_vq1 ready after {time.monotonic()-t_start:.1f}s. GO.")
            break
        if "timed out" in line.lower() or "aborting" in line.lower():
            print(f"  fly_vq1 aborted early: {line.strip()}", file=sys.stderr)
            break
    if not ready:
        proc.kill()
        return {"ok": False, "reason": "never ready", "speed": speed, "label": label, "lines": lines}

    send_keys("enter:1.0")
    t_lap = time.monotonic()
    while True:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            if time.monotonic() - t_lap > LAP_TIMEOUT_S:
                print(f"  LAP TIMEOUT. Killing.", file=sys.stderr)
                proc.kill()
                break
            time.sleep(0.05)
            continue
        lines.append(line.rstrip())
        if any(k in line for k in ["state]", "summary", "collision", "ABORT", "col=", "recording ->"]):
            print(f"  {line.rstrip()}")
    exit_code = proc.wait()
    print(f"  fly_vq1 exited: code={exit_code}")

    result = {"ok": exit_code == 0, "exit_code": exit_code, "speed": speed, "label": label,
              "lap_num": lap_num}
    import re
    for ln in lines:
        if "final state:" in ln:
            result["final_state"] = ln.split(":", 1)[1].strip()
        if "gate_index:" in ln:
            result["gate_index"] = ln.split(":", 1)[1].strip()
        if "recording ->" in ln or "recording:" in ln:
            result["recording"] = ln.split("->", 1)[-1].strip() if "->" in ln else ln.split(":", 1)[1].strip()
        if "col=" in ln:
            m = re.findall(r"col=(\d+)", ln)
            if m:
                result["last_col"] = int(m[-1])
    col = result.get("last_col", 0)
    result["contact"] = col > 0
    if col > 0:
        print(f"  *** COLLISION col={col} -> INVALID lap. (sleeping 3s spawn-artefact guard)")
        result["ok"] = False
        result["reason"] = f"collision col={col}"
        time.sleep(3.0)
    print(f"  Result: final={result.get('final_state')} gate={result.get('gate_index')} "
          f"col={col} rec={result.get('recording','?')}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default="8", help="comma list of target speeds m/s")
    ap.add_argument("--passes", type=int, default=3, help="passes per speed point")
    ap.add_argument("--max-gates", type=int, default=2)
    ap.add_argument("--tag", default="sweep", help="label tag -> data/runs/<stamp>_atspd_<tag>_s<speed>_p<n>")
    ap.add_argument("--stop-on-contact-streak", type=int, default=3,
                    help="abort a speed point after this many consecutive contacts (controller lost it)")
    args = ap.parse_args()
    speeds = [float(x) for x in args.speeds.split(",") if x.strip()]

    results = []
    lap = 0
    for sp in speeds:
        contact_streak = 0
        for pn in range(1, args.passes + 1):
            lap += 1
            label = f"atspd_{args.tag}_s{sp:g}_p{pn}"
            r = run_lap(sp, args.max_gates, label, lap)
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "lines"}, indent=2))
            if r.get("contact"):
                contact_streak += 1
            elif r.get("ok"):
                contact_streak = 0
            if contact_streak >= args.stop_on_contact_streak:
                print(f"\n  speed {sp}: {contact_streak} consecutive contacts -> controller lost the "
                      f"course at this speed. Moving on.")
                break
            time.sleep(2.0)

    out = ROOT / "handoff" / "at-speed-sigma-2026-06-15" / "logs" / f"sweep_{args.tag}_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print("\n=== SWEEP COMPLETE ===")
    for r in results:
        print(f"  lap {r['lap_num']:2d} s={r['speed']:>4g} {'OK ' if r['ok'] else 'INV'} "
              f"final={r.get('final_state','?'):<10s} col={r.get('last_col',0)} "
              f"rec={r.get('recording','?')}")
    print(f"\nResults -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
