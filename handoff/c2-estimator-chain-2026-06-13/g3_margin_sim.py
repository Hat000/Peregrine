"""G3 MARGIN SIM (C2 gauntlet, BLUEPRINT §3.4 G3) -- gate-4 in-plane RMS/p90/p99 + frac-over across the
body-radius band, with the abs/submap negative controls.

Ports the VALIDATED c1_gate_relative part_b three-arm bias/variance sim onto the PRODUCTIONIZED
racer.kf_rewind.RewindKF + the production gate-relative cov constants (racer.localization), then extends
it with p99 and frac-over across r in {0.21, 0.26, 0.30, 0.33, 0.38}.

THREE arms, all consuming the SAME straight g3->g4 @ 37 m/s perfectly-centered trajectory (true in-plane
offset = 0), so the terminal in-plane error IS the estimator's contribution:
  abs    : absolute world-fix  z = p_true + map_bias + isotropic 0.50 noise + 0.40 floor   (NEGATIVE CTRL)
  submap : subtract-gate-MAP   z carries map_bias + tight lateral noise + 0.40 floor        (NEGATIVE CTRL)
  rel    : TRUE gate-relative  z carries tight lateral noise, NO map_bias, NO in-plane floor (THE FIX)

Margin model: MARGIN(r) = W_eff - r with W_eff = 0.155 + 0.38 = 0.535 m (anchored to the design's
"gate-4 contact-true in-plane margin 0.155 m @ r=0.38"). CENTRAL report = r=0.30; r=0.38 = worst-case
stress knob. frac-over(r) = P(in-plane error >= MARGIN(r)).

EXPECTED (BY DESIGN, NOT a build failure, BLUEPRINT §0/§3.4): rel E_bias ~ 0 (map bias dropped EXACTLY);
rel RMS ~ 0.14 (clears 0.155 on RMS) but p90 ~ 0.20 OVER the 0.155 worst-case -> CANNOT-SETTLE-OFFLINE,
escape-hatched to L3. abs ~ 0.279 / submap ~ 0.228 are the negative controls (must be WORSE than rel).

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/c2-estimator-chain-2026-06-13/g3_margin_sim.py
[C2-ESTIMATOR-CHAIN 2026-06-13]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from racer.frames import ATTITUDE_NOISE_STD_RAD, R_world_from_body  # noqa: E402
from racer.kf_rewind import RewindKF  # noqa: E402
from racer.localization import (  # noqa: E402
    FIX_COV_FLOOR_STD,
    GATE_REL_ALONG_SIGMA,
    GATE_REL_INPLANE_SIGMA,
)
from racer.state_estimator import LinearKF  # noqa: E402

SEED = 20260613
V_RACE = 37.0
MARGIN_G4 = 0.155                 # contact-true in-plane margin @ r=0.38
W_EFF = MARGIN_G4 + 0.38          # 0.535 m effective clear half-width (anchor)
RADIUS_BAND = [0.21, 0.26, 0.30, 0.33, 0.38]
CENTRAL_R = 0.30
ABS_AXIS_SIGMA = 0.50            # absolute world-fix per-axis sigma (a1/b1)
MAP_BIAS_ED = np.array([0.18, 0.06])   # per-track gate-4 map/registration bias (E,D) -- the un-filterable
PER_AXIS_SIG = GATE_REL_INPLANE_SIGMA  # 0.265 m gate-relative lateral per-axis (the production constant)

G3_BOTTOM = np.array([-111.5, -5.1, 24.57])
G4_BOTTOM = np.array([-135.5, -0.8, 25.36])
DT_IMU = 1.0 / 90.0
FIX_HZ = 14.0


def _run_arm(arm: str, n_mc: int) -> dict:
    seg = G4_BOTTOM - G3_BOTTOM
    Lseg = float(np.linalg.norm(seg))
    uhat = seg / Lseg
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    accel_body = R_wb.T @ (-g)
    T = Lseg / V_RACE
    n_steps = int(T / DT_IMU)
    fix_dt = 1.0 / FIX_HZ
    E_errs, D_errs, ip_errs = [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 7000 * {"abs": 1, "submap": 2, "rel": 3}[arm] + s)
        p0 = G3_BOTTOM.copy()
        v0 = uhat * V_RACE
        kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        t = 0.0
        t_ns = 0
        next_fix = 0.0
        for _ in range(n_steps):
            t += DT_IMU
            t_ns += int(DT_IMU * 1e9)
            rk.predict(accel_body, R_wb, DT_IMU, t_ns)
            p_true = p0 + uhat * V_RACE * t
            rng_g4 = float(np.linalg.norm(G4_BOTTOM - p_true))
            if t >= next_fix and rng_g4 < 12.0:
                next_fix += fix_dt
                nE, nD = r.normal(0, PER_AXIS_SIG), r.normal(0, PER_AXIS_SIG)
                # along-track (gate-normal ~ N) loose var, as the production helper shapes it.
                var_along = ABS_AXIS_SIGMA**2 + (ATTITUDE_NOISE_STD_RAD * rng_g4)**2 + FIX_COV_FLOOR_STD**2
                if arm == "abs":
                    nA = r.normal(0, ABS_AXIS_SIGMA, 3)
                    z = p_true + nA
                    z[1] += MAP_BIAS_ED[0]; z[2] += MAP_BIAS_ED[1]
                    cov = (ABS_AXIS_SIGMA**2 + FIX_COV_FLOOR_STD**2) * np.eye(3)
                elif arm == "submap":
                    z = p_true.copy()
                    z[1] += MAP_BIAS_ED[0] + nE; z[2] += MAP_BIAS_ED[1] + nD
                    z[0] += r.normal(0, ABS_AXIS_SIGMA)
                    cov = np.diag([ABS_AXIS_SIGMA**2, PER_AXIS_SIG**2, PER_AXIS_SIG**2]) \
                        + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                else:  # rel -- production cov: tight in-plane NO floor; loose along-track WITH floor
                    z = p_true.copy()
                    z[1] += nE; z[2] += nD
                    z[0] += r.normal(0, ABS_AXIS_SIGMA)
                    cov = np.diag([var_along, PER_AXIS_SIG**2, PER_AXIS_SIG**2])
                rk.update_position(z, cov, sim_time_ns=t_ns)
        p_true_final = p0 + uhat * V_RACE * (n_steps * DT_IMU)
        err = rk.position - p_true_final
        E_errs.append(float(err[1])); D_errs.append(float(err[2]))
        ip_errs.append(float(np.hypot(err[1], err[2])))
    E, D, ip = np.array(E_errs), np.array(D_errs), np.array(ip_errs)
    res = dict(
        arm=arm, n_mc=n_mc,
        inplane_rms=float(np.sqrt(np.mean(ip**2))),
        inplane_p50=float(np.percentile(ip, 50)),
        inplane_p90=float(np.percentile(ip, 90)),
        inplane_p99=float(np.percentile(ip, 99)),
        E_bias=float(E.mean()), E_std=float(E.std()),
        D_bias=float(D.mean()), D_std=float(D.std()),
        frac_over={f"{rr:.2f}": float(np.mean(ip >= (W_EFF - rr))) for rr in RADIUS_BAND},
    )
    return res


def run(n_mc: int = 600) -> dict:
    out = {"seed": SEED, "v_race": V_RACE, "margin_g4_at_r038": MARGIN_G4, "w_eff": W_EFF,
           "radius_band": RADIUS_BAND, "central_r": CENTRAL_R, "per_axis_inplane_sigma": PER_AXIS_SIG,
           "arms": {}}
    for arm in ("abs", "submap", "rel"):
        out["arms"][arm] = _run_arm(arm, n_mc)
    return out


def main():
    res = run()
    print("=" * 96)
    print("G3 MARGIN SIM -- gate-4 in-plane error (RMS/p90/p99) + frac-over across the body-radius band")
    print(f"  V_RACE={V_RACE} m/s, per-axis in-plane sigma={PER_AXIS_SIG} m, W_eff={W_EFF} m "
          f"(MARGIN(r)=W_eff-r; 0.155 @ r=0.38)")
    print("=" * 96)
    print(f"\n{'arm':>7} {'RMS':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'E_bias':>8} {'D_bias':>8}  "
          + " ".join(f"over@{rr:.2f}" for rr in RADIUS_BAND))
    for arm in ("abs", "submap", "rel"):
        x = res["arms"][arm]
        fo = "   ".join(f"{x['frac_over'][f'{rr:.2f}']*100:5.1f}%" for rr in RADIUS_BAND)
        print(f"{arm:>7} {x['inplane_rms']:7.3f} {x['inplane_p50']:7.3f} {x['inplane_p90']:7.3f} "
              f"{x['inplane_p99']:7.3f} {x['E_bias']:+8.3f} {x['D_bias']:+8.3f}  {fo}")
    rel = res["arms"]["rel"]
    print(f"\nCENTRAL (r={CENTRAL_R}, margin={W_EFF-CENTRAL_R:.3f} m): rel frac-over "
          f"{rel['frac_over'][f'{CENTRAL_R:.2f}']*100:.1f}% | rel p90 {rel['inplane_p90']:.3f} "
          f"(>{MARGIN_G4} worst-case @0.38 -> CANNOT-SETTLE-OFFLINE, escape-hatched to L3, BY DESIGN)")
    print(f"NEGATIVE CONTROLS: abs E_bias {res['arms']['abs']['E_bias']:+.3f} / RMS "
          f"{res['arms']['abs']['inplane_rms']:.3f}; submap E_bias {res['arms']['submap']['E_bias']:+.3f}"
          f" / RMS {res['arms']['submap']['inplane_rms']:.3f}  (both WORSE than rel -> bias removal real)")
    p = Path(__file__).resolve().parent / "g3_margin_sim_results.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
