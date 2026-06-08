"""scripts/fit_twin.py -- fit a sim-faithful CtbrPlantConfig from the ShadowPC sysid extract and
validate it against the recorded gate0_course1 flight (Task B).

Usage:  python scripts/fit_twin.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from racer.twin_fit import fit_plant, load_run, validate

_EXTRACT = Path(__file__).resolve().parent.parent / "handoff/shadowpc-followups-2026-06-05/sysid"
_RUNS = ("rate1", "rate2", "hsweep1", "hsweep2", "hsweep3")     # fit inputs (course1 is validation)


def main() -> int:
    runs = [load_run(_EXTRACT / f"{name}.json") for name in _RUNS]
    course = load_run(_EXTRACT / "gate0_course1.json")
    cfg, rep = fit_plant(runs, drag_run=course)
    rf, tf = rep["rate"], rep["thrust"]

    print("=== FAITHFUL CtbrPlantConfig fit (ShadowPC sysid extract) ===\n")
    print("RATE LOOP (quaternion finite-diff realized rate; %d step segments, tau from %d):"
          % (rf["n_segments"], rf["tau_n"]))
    for ax in ("roll", "pitch", "yaw"):
        pa = rf["per_axis"][ax]
        print(f"  {ax:5s}  signed gain {pa['gain']:+.3f}  (r2={pa.get('r2', float('nan')):.3f}, n={pa['n']})")
    print(f"  -> rate_gain = [{', '.join(f'{g:.3f}' for g in rf['gain'])}]   (|steady realized/commanded|)")
    print(f"  -> rate_sign = [{', '.join(f'{s:+.0f}' for s in rf['sign'])}]   (roll/yaw command-inverted)")
    print(f"  -> rate_tau_s = {rf['tau']:.4f} s")

    print(f"\nTHRUST (pooled {tf['n']} near-level constant-collective windows; hsweeps excluded -- "
          "unreliable vz):")
    for th, ac in tf["points"]:
        print(f"  thrust {th:.3f}  net up-accel {ac:+.3f} m/s^2")
    print(f"  -> hover_thrust = {tf['hover']:.4f}   slope = {tf['slope']:.2f} m/s^2 per unit thrust")

    print(f"\nDRAG (linear, world-frame; estimated from course1's sustained forward flight):")
    print(f"  -> linear_drag = {rep['drag']:.4f} /s   (the real sim's terminal velocity the "
          "ideal-rotor twin otherwise lacks)")

    print("\n=== CtbrPlantConfig (FAITHFUL) ===")
    print(f"  CtbrPlantConfig(hover_thrust={cfg.hover_thrust:.4f}, rate_tau_s={cfg.rate_tau_s:.4f},")
    print(f"                  rate_gain=[{', '.join(f'{g:.3f}' for g in cfg.rate_gain)}],")
    print(f"                  rate_sign=[{', '.join(f'{s:+.0f}' for s in cfg.rate_sign)}],")
    print(f"                  linear_drag={cfg.linear_drag:.4f})")

    # -- validate against the clean closed-loop course1 run --
    v = validate(cfg, course)
    print(f"\n=== VALIDATION vs gate0_course1 (open-loop rollout, {v['n']} steps) ===")
    if v["n"]:
        print("  attitude RMS (deg)  roll/pitch/yaw:")
        print(f"    one-step-ahead : {_fmt(v['step_att_rms_deg'])}")
        print(f"    open-loop      : {_fmt(v['ol_att_rms_deg'])}")
        print("  velocity RMS (m/s)  vx/vy/vz:")
        print(f"    one-step-ahead : {_fmt(v['step_vel_rms_mps'])}")
        print(f"    open-loop      : {_fmt(v['ol_vel_rms_mps'])}   (speed RMS {v['ol_speed_rms_mps']:.3f})")
    return 0


def _fmt(a) -> str:
    return "  ".join(f"{x:+.3f}" for x in np.asarray(a))


if __name__ == "__main__":
    raise SystemExit(main())
