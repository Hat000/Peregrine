"""PRE-FLIGHT twin predictions for the SHADOWPC-TWIN-FALSIFY campaign (2026-06-10/11).

Every probe in the campaign gets its map-ON faithful-twin prediction COMMITTED BEFORE the
flight so the live comparison is an honest falsification test, not a post-hoc fit. The twin
under test: ``twin_fit.faithful_config(super_rate=True)`` --

  * world-frame LINEAR drag, isotropic, d = 0.2111 1/s (same coefficient fwd/lat/back/vert)
  * thrust map a_up = g * (collective / 0.2656), linear over the whole [0,1] stick
  * inner rate loop: static super-rate map g(|c|) = G0/(1-0.30*min(|c|,pi)/pi), tau=19 ms,
    slew 260/260/80 rad/s^2 -- INDEPENDENT of airspeed
  * no battery model (hover collective constant forever)
  * no control-rate dependence beyond integration step

Analytic closed forms (the twin's translational dynamics are exactly linear):
  * alt-held steady tilt theta -> horizontal terminal speed v_term = g*tan(theta)/d
  * accel from rest under held tilt: v(t) = v_term*(1 - exp(-d*t))
  * level coast (collective=hover): v(t) = v0*exp(-d*t), i.e. a(v) = -d*v, tau = 1/d = 4.74 s
    -- a STRAIGHT LINE through the origin in the (v, a) plane; any curvature = quadratic term
    = twin falsified
  * vertical: a_up(T, vz) = g*(T/0.2656 - 1) - d*vz  (climb terminal vz = a0/d)

Usage (repo root):  .venv\\Scripts\\python.exe handoff\\shadowpc-twin-falsify-2026-06-10\\predict.py
Output: predictions.md next to this script (committed before the flights).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

from racer.twin import CtbrPlant  # noqa: E402
from racer.twin_fit import faithful_config  # noqa: E402

G = 9.80665
CFG = faithful_config(super_rate=True)
D = float(CFG.linear_drag)            # 0.2111 1/s
H = float(CFG.hover_thrust)           # 0.2656


def vterm(theta_deg: float) -> float:
    """Alt-held steady tilt -> horizontal terminal speed (m/s)."""
    return G * np.tan(np.radians(theta_deg)) / D


def v_accel(theta_deg: float, t: float) -> float:
    """Speed after t seconds of held tilt from rest (alt-held)."""
    return vterm(theta_deg) * (1.0 - np.exp(-D * t))


def gain(c: float, axis: int = 0) -> float:
    g0 = float(np.asarray(CFG.rate_gain)[axis])
    s = float(np.asarray(CFG.super_rate_s)) if np.ndim(CFG.super_rate_s) == 0 else float(np.asarray(CFG.super_rate_s)[axis])
    return g0 / (1.0 - s * min(abs(c), np.pi) / np.pi)


def vertical_accel(T: float, vz_up: float = 0.0) -> float:
    """Net world-up accel at collective T and up-speed vz_up (m/s)."""
    return G * (T / H - 1.0) - D * vz_up


def twin_coast_check() -> float:
    """Numeric CtbrPlant sanity check of the analytic coast tau (catches wrong closed forms)."""
    from racer.contracts import ControlCommand, ControlMode
    p = CtbrPlant(CFG, velocity_ned=[10.0, 0.0, 0.0])
    cmd = ControlCommand(mode=ControlMode.BODY_RATE, body_rate=np.zeros(3), thrust=H)
    t, dt = 0.0, 0.01
    while p.vel[0] > 10.0 / np.e:
        p.step(cmd, dt)
        t += dt
    return t


def main() -> None:
    L: list[str] = []
    a = L.append
    a("# Twin predictions (map-ON faithful config) -- COMMITTED BEFORE FLIGHT\n")
    a(f"config: hover={H}  drag={D}/s (linear, world-frame, isotropic)  "
      f"G0={np.asarray(CFG.rate_gain).tolist()}  s={CFG.super_rate_s}  "
      f"alpha_max={np.asarray(CFG.alpha_max_rps2).tolist()}  tau={CFG.rate_tau_s}\n")

    a("## P1 aero/drag at speed")
    a("| tilt (deg) | v_term (m/s) | v @3s held | v @5s held |")
    a("|---|---|---|---|")
    for th in (8, 10, 12, 17, 25, 32):
        a(f"| {th} | {vterm(th):.2f} | {v_accel(th, 3):.2f} | {v_accel(th, 5):.2f} |")
    tau_num = twin_coast_check()
    a(f"\n- coast decay: v(t)=v0*exp(-{D}*t), tau = {1/D:.2f} s (numeric CtbrPlant check: "
      f"{tau_num:.2f} s); a(v) = -{D}*v EXACTLY LINEAR through origin, no v^2 term")
    a("- per-axis: IDENTICAL forward / backward / lateral / vertical (isotropic world drag);")
    a("  body-frame drag or any fwd/lat asymmetry falsifies the model form")
    a("- attitude during coast does not matter (world-frame drag): a coast at level attitude")
    a("  and a coast while tilted (alt-held) decay identically\n")

    a("## P2 rate loop at airspeed (twin: NO airspeed dependence)")
    a("| |cmd| (rad/s) | sustained rate roll/pitch (rad/s) | gain |")
    a("|---|---|---|")
    for c in (0.3, 1.0, 3.14):
        a(f"| {c} | {gain(c)*c:.3f} | {gain(c):.3f} |")
    a("- identical at 0 m/s and 12 m/s; slew 260 rad/s^2; tau 19 ms. Any shift with airspeed")
    a("  falsifies the static map's completeness.\n")

    a("## P3 collective map (twin: linear, a_up = g*(T/0.2656 - 1) - d*vz)")
    a("| collective | a_up @vz=0 (m/s^2) | climb v_term (m/s) |")
    a("|---|---|---|")
    for T in (0.0, 0.10, 0.20, 0.2656, 0.32, 0.40, 0.55, 0.70, 0.85, 1.00):
        a0 = vertical_accel(T)
        vt = a0 / D if a0 > 0 else float("nan")
        a(f"| {T:.4g} | {a0:+.2f} | {vt:.1f} |" if np.isfinite(vt) else f"| {T:.4g} | {a0:+.2f} | -- |")
    a("- T=0 is exactly free fall (-g); no thrust floor, no nonlinearity anywhere on [0,1]")
    a("- zero collective->attitude coupling (collective is a pure body -Z force)\n")

    a("## P4 long-duration drift (twin: NONE)")
    a("- hover collective is 0.2656 at t=0 and at t=8 min; any monotonic drift falsifies\n")

    a("## P5 control-rate sensitivity (twin: discretization only)")
    a("- same maneuver at 50/100/200 Hz differs only via integration step of the SAME ODE;")
    a("  realized sustained rates / speeds shift < ~1%. A systematic shift (e.g. the 1.0.3364")
    a("  tick-phase coupling) falsifies.\n")

    a("## P6 determinism (twin: exact)")
    a("- twin is deterministic; live run-to-run spread = the measurement noise floor that all")
    a("  other comparisons inherit. No prediction, this CALIBRATES.\n")

    a("## P7 mixed-axis coordinated turn")
    a("- predicted offline by replaying the recorded command stream through CtbrPlant")
    a("  (replay_twin.py); banked turn at 20 deg / ~6 m/s -> radius v^2/(g*tan20) ~ 10.1 m,")
    a("  turn rate ~0.59 rad/s. Error table comes from the replay, same as the sweep style.\n")

    out = Path(__file__).parent / "predictions.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
