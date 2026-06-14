"""G3 margin-sim invariants (C2 gauntlet, BLUEPRINT §3.4 G3) -- the in-suite guard.

A compact, reduced-N replica of handoff/c2-estimator-chain-2026-06-13/g3_margin_sim.py (the full report
with the radius-band frac-over). Asserts the STRUCTURAL invariants that pin the gate-relative fix as
real (NOT the p90 pass -- the p90 is OVER the 0.155 worst-case BY DESIGN, escape-hatched to L3):

  - gate-relative (rel) E_bias ~ 0 -> the per-track map/registration bias drops EXACTLY;
  - abs / submap E_bias ~ +0.17 -> the negative controls RE-INJECT the map bias (must FAIL);
  - rel in-plane RMS < submap RMS < abs RMS -> bias removal + tight in-plane is the win;
  - rel RMS < 0.155 (clears the margin on RMS) while abs & submap RMS are OVER it.

Drives the PRODUCTIONIZED racer.kf_rewind.RewindKF + the production gate-relative cov constants.
Torch-free (numpy only). [C2-ESTIMATOR-CHAIN 2026-06-13]
"""
import numpy as np

from racer.frames import ATTITUDE_NOISE_STD_RAD, R_world_from_body
from racer.kf_rewind import RewindKF
from racer.localization import FIX_COV_FLOOR_STD, GATE_REL_ALONG_SIGMA, GATE_REL_INPLANE_SIGMA
from racer.state_estimator import LinearKF

SEED = 20260613
V_RACE = 37.0
ABS_SIG = 0.50
MAP_BIAS_ED = np.array([0.18, 0.06])
SIG = GATE_REL_INPLANE_SIGMA
MARGIN_G4 = 0.155
G3 = np.array([-111.5, -5.1, 24.57])
G4 = np.array([-135.5, -0.8, 25.36])
DT = 1.0 / 90.0


def _arm(arm, n_mc=200):
    seg = G4 - G3
    Lseg = float(np.linalg.norm(seg))
    uhat = seg / Lseg
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    accel_body = R_wb.T @ (-np.array([0.0, 0.0, 9.80665]))
    n_steps = int((Lseg / V_RACE) / DT)
    fix_dt = 1.0 / 14.0
    E, D, ip = [], [], []
    for s in range(n_mc):
        r = np.random.default_rng(SEED + 7000 * {"abs": 1, "submap": 2, "rel": 3}[arm] + s)
        p0 = G3.copy()
        rk = RewindKF(kf=LinearKF.initialize(p0 + r.normal(0, 0.3, 3), uhat * V_RACE,
                                             pos_std=0.5, vel_std=0.5), horizon_s=0.5)
        t, t_ns, next_fix = 0.0, 0, 0.0
        for _ in range(n_steps):
            t += DT; t_ns += int(DT * 1e9)
            rk.predict(accel_body, R_wb, DT, t_ns)
            p_true = p0 + uhat * V_RACE * t
            rng_g4 = float(np.linalg.norm(G4 - p_true))
            if t >= next_fix and rng_g4 < 12.0:
                next_fix += fix_dt
                nE, nD = r.normal(0, SIG), r.normal(0, SIG)
                var_along = ABS_SIG**2 + (ATTITUDE_NOISE_STD_RAD * rng_g4)**2 + FIX_COV_FLOOR_STD**2
                if arm == "abs":
                    z = p_true + r.normal(0, ABS_SIG, 3)
                    z[1] += MAP_BIAS_ED[0]; z[2] += MAP_BIAS_ED[1]
                    cov = (ABS_SIG**2 + FIX_COV_FLOOR_STD**2) * np.eye(3)
                elif arm == "submap":
                    z = p_true.copy(); z[1] += MAP_BIAS_ED[0] + nE; z[2] += MAP_BIAS_ED[1] + nD
                    z[0] += r.normal(0, ABS_SIG)
                    cov = np.diag([ABS_SIG**2, SIG**2, SIG**2]) + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                else:
                    z = p_true.copy(); z[1] += nE; z[2] += nD; z[0] += r.normal(0, ABS_SIG)
                    cov = np.diag([var_along, SIG**2, SIG**2])
                rk.update_position(z, cov, sim_time_ns=t_ns)
        err = rk.position - (p0 + uhat * V_RACE * (n_steps * DT))
        E.append(err[1]); D.append(err[2]); ip.append(float(np.hypot(err[1], err[2])))
    E, D, ip = np.array(E), np.array(D), np.array(ip)
    return dict(rms=float(np.sqrt(np.mean(ip**2))), p90=float(np.percentile(ip, 90)),
                p99=float(np.percentile(ip, 99)), E_bias=float(E.mean()), D_bias=float(D.mean()))


def test_g3_gate_relative_drops_map_bias_and_beats_negative_controls():
    a, sm, rel = _arm("abs"), _arm("submap"), _arm("rel")
    # rel removes the per-track map bias EXACTLY; the negative controls re-inject it.
    assert abs(rel["E_bias"]) < 0.03 and abs(rel["D_bias"]) < 0.03, rel
    assert a["E_bias"] > 0.12 and sm["E_bias"] > 0.12, (a, sm)        # map bias survives (the controls)
    # bias removal + tight in-plane => rel is the tightest by a wide margin.
    assert rel["rms"] < sm["rms"] < a["rms"], (rel, sm, a)
    # rel clears the margin on RMS; abs & submap (the negative controls) do NOT.
    assert rel["rms"] < MARGIN_G4 < sm["rms"] and a["rms"] > MARGIN_G4, (rel, sm, a)


def test_g3_rel_p90_is_over_the_worstcase_margin_by_design():
    # The make-or-break finding (BLUEPRINT §0/§3.4): even bias-free, rel p90 is OVER 0.155 @ r=0.38
    # -> CANNOT-SETTLE-OFFLINE, escape-hatched to L3. This is the EXPECTED result, not a failure.
    rel = _arm("rel")
    assert rel["p90"] > MARGIN_G4, f"rel p90 {rel['p90']:.3f} unexpectedly cleared the 0.155 worst-case"
