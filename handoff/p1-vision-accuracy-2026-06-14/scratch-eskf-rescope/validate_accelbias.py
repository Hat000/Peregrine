"""Validate the SUB-STEPPED accel-bias observability instrument (the faithful one: production 90 Hz Q,
n*Q_pv(imu_dt) over a gap, NOT the 72x-inflated one-big-step Q_pv(tau)).

Rules out a residual bug by checking the instrument behaves as observability theory demands:
  (1) DENSE-FIX limit: more fixes within the same flight -> sigma_ba must drop (else the instrument is dead).
  (2) MANY-LAP convergence (q_ba=0, constant bias): sigma_ba must decrease with total fixes if observable.
  (3) MANEUVER vs STRAIGHT at high N: attitude diversity must help (accel-bias needs rotational excitation,
      COWORK-2 / Mirzaei-Roumeliotis).
If all three behave, the realized-rate poor observability is an HONEST finding, not a bug.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scratch-eskf"))
import eskf_geometry as G
import accel_bias_observability as A

BUD = A.G_SIN_BUDGET


def sd_horiz(recs, **kw):
    P, _ = A.cov_recursion_accelbias(recs, **kw)
    s = np.sqrt(np.diag(P[6:9, 6:9]))
    return float(np.hypot(s[0], s[1])), float(s[2])


def main():
    gates, _ = G.load_course()
    print(f"budget |b_a_horiz| < {BUD:.3f} m/s^2\n")

    print("(1) DENSE-FIX limit (1 lap, 6 gates, q_ba=0): does sigma_ba drop as fixes densify?")
    for npg in (7, 15, 30, 60, 120):
        rng = np.random.default_rng(100 + npg)
        recs = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                 n_fix_per_gate=npg, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=1)
        h, v = sd_horiz(recs, q_ba=0.0)
        print(f"   npg={npg:>3} (N={len(recs):>3}) -> horiz {h:.3f}  vert {v:.3f}  {'<bud' if h<BUD else ''}")

    print("\n(2) MANY-LAP convergence (npg=7 ~ fr0.07, and npg=25 ~ fr0.25, q_ba=0):")
    for npg, tag in ((7, "fr0.07"), (25, "fr0.25")):
        row = []
        for nl in (1, 3, 6, 12, 24):
            rng = np.random.default_rng(7 * nl + npg)
            recs = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                     n_fix_per_gate=npg, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=nl)
            h, _ = sd_horiz(recs, q_ba=0.0)
            row.append(f"{nl}lap(N{len(recs)}):{h:.3f}")
        print(f"   {tag}: " + "  ".join(row))

    print("\n(3) MANEUVER vs STRAIGHT at high N (5-lap-equiv, q_ba=0):")
    rng = np.random.default_rng(99)
    straight = G.make_fix_stream(rng, gates_ned=gates, gate_ids=[4], r_min=5.0, r_max=30.0,
                                 n_fix_per_gate=210, crab_mean_deg=37.5, crab_spread_deg=1.0, n_laps=1)
    maneuver = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                 n_fix_per_gate=35, crab_mean_deg=37.5, crab_spread_deg=8.0, n_laps=1)
    hs, vs = sd_horiz(straight, q_ba=0.0)
    hm, vm = sd_horiz(maneuver, q_ba=0.0)
    print(f"   straight 1-gate  (N={len(straight)}): horiz {hs:.3f}  vert {vs:.3f}")
    print(f"   maneuver 6-gate  (N={len(maneuver)}): horiz {hm:.3f}  vert {vm:.3f}")
    print(f"   -> maneuvering helps by {hs/max(hm,1e-9):.1f}x" if hm < hs else "   -> maneuvering did NOT help")

    print("\n(4) Best realistic case: many laps + dense fixes -> floor of accel-bias observability:")
    for nl in (6, 12):
        rng = np.random.default_rng(5 * nl)
        recs = G.make_fix_stream(rng, gates_ned=gates, gate_ids=list(range(6)), r_min=5.0, r_max=30.0,
                                 n_fix_per_gate=30, crab_mean_deg=37.5, crab_spread_deg=10.0, n_laps=nl)
        h, v = sd_horiz(recs, q_ba=0.0)
        print(f"   {nl} laps npg=30 (N={len(recs)}): horiz {h:.3f}  vert {v:.3f}  {'<budget' if h<BUD else 'OVER budget'}")


if __name__ == "__main__":
    main()
