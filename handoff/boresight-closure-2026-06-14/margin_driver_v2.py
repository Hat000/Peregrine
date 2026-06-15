"""BORESIGHT-CLOSURE Deliverable-3 margin ENGINE (v2) -- anisotropic in-plane sigma + constant vertical
fix bias, on top of the PRODUCTION margin stack (margin-closure-envelope-2026-06-14/margin_envelope.py).

WHY this file exists
--------------------
The production engine ME.fly_lap models (a) an UNBIASED fix (z = true_pos + zero-mean in-plane+depth
noise) -- i.e. the POST-bake estimator -- and (b) an ISOTROPIC in-plane sigma (one `inplane_sigma` on
BOTH in-plane axes). Deliverable 3 must show the boresight bake's effect and use the REAL anisotropic
gate-4 sigma. So this module adds exactly TWO knobs and changes NOTHING else:

  (i)  vert_fix_bias_m : a CONSTANT vertical fix bias added to the fix z along the vertical in-plane axis
       e2. This models the PRE-bake estimator (the measured epsilon_vert) vs the POST-bake estimator
       (vert_fix_bias_m = 0). The production bake on main (frames.BoresightCorrection.vert_offset_m via
       localization._apply_camera_vert_offset) makes the deployed +L fix UNBIASED -- which is precisely
       what ME.fly_lap already models -- so POST-bake == vert_fix_bias_m=0 == ME.fly_lap. PRE-bake ==
       vert_fix_bias_m set to the measured bias.
  (ii) sigma_lat / sigma_vert : ANISOTROPIC in-plane sigma. sigma_lat on the horizontal/lateral in-plane
       axis e1, sigma_vert on the more-vertical in-plane axis e2. (ME.fly_lap uses one inplane_sigma on
       both.) Setting sigma_lat == sigma_vert recovers the isotropic production behaviour exactly.

This file IMPORTS margin_envelope as ME and REUSES ME.get_truth, ME constants, and the EXACT production
KF stack (racer.state_estimator.LinearKF + racer.kf_rewind.RewindKF). It does NOT edit the production
file. fly_lap_v2 is a byte-for-byte copy of ME.fly_lap with ONLY the three documented changes.

THE IN-PLANE BASIS (load-bearing for the bias sign)
---------------------------------------------------
For the binding gate-3 -> gate-4 approach axis u34 = ME.GATES[4]-ME.GATES[3] (normalised), the
production helper ME._inplane_basis(u34) returns (e1, e2):
  e1 = [ 0.17636,  0.98433,  0.0     ]  -> |z|=0.000  == the HORIZONTAL / LATERAL in-plane axis
  e2 = [-0.03188,  0.00571, -0.99948 ]  -> |z|=0.999  == the (near-pure) VERTICAL in-plane axis
e2 is unambiguously "the more vertical" (|e2_z| 0.999 >> |e1_z| 0.000). We therefore put sigma_vert and
the vertical fix bias on e2, sigma_lat on e1 -- matching the prompt. (For ANY u-hat, fly_lap_v2 picks e2
as whichever of ME._inplane_basis(u) has the larger |z| and e1 as the other, then re-derives the matched
covariance basis, so the (i)<->(ii) axis labelling is always self-consistent even off the g3->g4 axis.)

SIGN of vert_fix_bias_m (read before you pass a number)
-------------------------------------------------------
The term is added to the fix as `+ vert_fix_bias_m * e2`. e2 has e2.z = -0.99948 (NED +z = DOWN), so e2
points world-UP. The measured boresight convention is rel_vert = (vision_fix - GT) projected on
gate-DOWN = -0.25 m (the vision chain places the gate ~0.25 m gate-DOWN of truth; FORM_RESOLUTION.md).
A fix offset of d metres along world-DOWN equals a coefficient on e2 of d * (down . e2) = d * (-0.99948),
i.e. coefficient ~= -d. So:

   to inject the measured gate-DOWN bias of -0.25 m (PRE-bake)  ->  pass vert_fix_bias_m = +0.25
   POST-bake (bake applied, fix unbiased)                        ->  pass vert_fix_bias_m =  0.0   (== ME.fly_lap)

(|down.e2| = 0.99948, so the projected magnitude is 0.25*0.99948 = 0.2499 m -- the 0.05% off-vertical
tilt of e2 is physical: e2 is the in-plane axis perpendicular to the slightly-climbing u34, not exactly
world-down. The Deliverable-3 caller chooses the numeric value + sign; this engine only mechanises the
+e2 term. The default 0.0 is byte-identical to the production post-bake fix.)

WHAT IS DELIBERATELY UNCHANGED (every other channel is ME.fly_lap verbatim)
---------------------------------------------------------------------------
- depth/along-track noise + cov: still RADIAL_SIGMA on target_u (ME.RADIAL_SIGMA = 0.50).
- the LinearKF init, RewindKF horizon, accel bias model (accel_bias_mag/bias_mode), accel white noise,
  fix-rate scheduling, latency queue, warm/weakvel branches, the final in-plane miss reduction onto u34.
- the per-step PRECOMPUTED truth driver from ME.get_truth(v_race) (s_p/s_v/s_R/s_ab/s_e1/s_e2/s_u/...).
  NOTE: the cov + bias use the SAME per-step e1/e2 (s_e1[k], s_e2[k]) the production loop uses, so on the
  g3->g4 axis the (e1,e2) here are identical to ME's; we only RE-LABEL which is lateral vs vertical and
  apply distinct sigmas + the bias. With sigma_lat==sigma_vert and vert_fix_bias_m=0 the draw, the cov,
  and z are algebraically identical to ME.fly_lap (verified to MC noise in the regression below).

[BORESIGHT-CLOSURE 2026-06-14]
"""
from __future__ import annotations

import os

# Match the production BLAS-thread pinning (set before numpy import; harmless if numpy already imported
# via ME -- these are no-ops then, but we keep them so this file run standalone behaves like ME).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np

# Import the PRODUCTION engine as ME (do NOT edit it). It lives in the sibling envelope handoff dir.
_HERE = Path(__file__).resolve().parent
_ME_DIR = _HERE.parents[0] / "margin-closure-envelope-2026-06-14"
if str(_ME_DIR) not in sys.path:
    sys.path.insert(0, str(_ME_DIR))
# ME itself inserts <repo>/src for racer.* -- importing it wires the production KF stack.
import margin_envelope as ME  # noqa: E402

# Re-export the production KF symbols ME pulled in, so fly_lap_v2's body reads like ME.fly_lap's.
from racer.state_estimator import LinearKF  # noqa: E402  (same class ME.fly_lap uses)
from racer.kf_rewind import RewindKF        # noqa: E402  (PRODUCTIONIZED OOSM -- the g3 engine's filter)

# Production constants, referenced by name (NOT re-defined) so this file tracks ME exactly.
ACCEL_NOISE_STD = ME.ACCEL_NOISE_STD
RADIAL_SIGMA = ME.RADIAL_SIGMA
DETECTOR_HZ = ME.DETECTOR_HZ
FIX_WINDOW_M = ME.FIX_WINDOW_M


def _label_inplane_axes(uhat):
    """Return (e1, e2) from ME._inplane_basis(uhat) ordered so that e2 is the MORE-VERTICAL axis
    (larger |z| component) and e1 the other (horizontal/lateral). For the g3->g4 axis this returns
    ME._inplane_basis(u34) unchanged (e2 already the vertical one); the swap guard only ever fires for
    a hypothetical near-vertical approach axis. Documented in the module header."""
    e1, e2 = ME._inplane_basis(uhat)
    if abs(e1[2]) > abs(e2[2]):
        e1, e2 = e2, e1
    return e1, e2


# =================================================================================================
# fly_lap_v2 -- ME.fly_lap with EXACTLY three changes (anisotropic draw, anisotropic cov, +e2 bias).
# Everything else is byte-for-byte ME.fly_lap (handoff/margin-closure-envelope-2026-06-14).
# =================================================================================================
def fly_lap_v2(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode, fix_rate, vel_mode,
               latency_ms, vert_fix_bias_m=0.0, horizon_s=0.5, fix_window_m=FIX_WINDOW_M):
    """One Monte-Carlo cold lap g0->g4 -- v2 of ME.fly_lap with anisotropic in-plane sigma
    (sigma_lat on e1, sigma_vert on e2) and an optional constant vertical fix bias vert_fix_bias_m
    added to the fix z along the vertical in-plane axis e2.

    fly_lap_v2(..., sigma_lat=s, sigma_vert=s, vert_fix_bias_m=0.0, ...) == ME.fly_lap(..., inplane_sigma=s, ...).

    See module header for the e1/e2 identity (e2 = vertical, |z|=0.999) and the vert_fix_bias_m sign
    (pass +0.25 to inject the measured gate-DOWN -0.25 m PRE-bake bias; 0.0 == POST-bake/ME.fly_lap)."""
    L_s = latency_ms / 1e3
    tr = ME.get_truth(v_race)
    u34 = tr["u34"]

    # constant body-frame accel bias for this run  (UNCHANGED from ME.fly_lap)
    if accel_bias_mag <= 0.0:
        accel_bias_body = np.zeros(3)
    elif bias_mode == "inplane":
        e1, e2 = ME._inplane_basis(u34)
        ang = rng.uniform(0, 2 * np.pi)
        d_world = np.cos(ang) * e1 + np.sin(ang) * e2     # in the gate-4 in-plane (world)
        accel_bias_body = tr["R_wb_g4"].T @ (accel_bias_mag * d_world)
    else:  # random3d (d3 default)
        d = rng.normal(0, 1, 3)
        d /= (np.linalg.norm(d) + 1e-12)
        accel_bias_body = accel_bias_mag * d

    # init at g0 with a realistic prior; COLD = velocity prior poorly known.  (UNCHANGED)
    p_init = tr["p_init_true"] + rng.normal(0, 0.5, 3)
    if vel_mode == "warm":
        v_init = tr["v_init_true"].copy()
    else:
        v_init = tr["v_init_true"] + rng.normal(0, 1.5, 3)
    kf = LinearKF.initialize(p_init, v_init, pos_std=1.0, vel_std=1.5)
    rk = RewindKF(kf=kf, horizon_s=horizon_s)

    s_p, s_v, s_R, s_ab = tr["s_p"], tr["s_v"], tr["s_R"], tr["s_ab"]
    s_dt, s_ts, s_tns = tr["s_dt"], tr["s_ts"], tr["s_tns"]
    s_tgt, s_e1, s_e2, s_u = tr["s_tgt"], tr["s_e1"], tr["s_e2"], tr["s_u"]
    n_steps = tr["n_steps"]

    fix_apply_queue = []
    fix_dt = 1.0 / (DETECTOR_HZ * fix_rate)
    next_fix_t = 0.0
    last_fix_z = None
    last_fix_t = None
    warm = (vel_mode == "warm")
    weakvel = (vel_mode == "weakvel")
    # ---- CHANGE 1of3: anisotropic in-plane variances (was: sig2_ip = inplane_sigma**2 on both axes) ----
    sig2_lat = sigma_lat ** 2
    sig2_vert = sigma_vert ** 2
    sig2_rad = RADIAL_SIGMA ** 2

    for k in range(n_steps):
        t = s_ts[k]
        t_ns = int(s_tns[k])
        # TRUE specific force + constant body BIAS + white noise.  (UNCHANGED)
        accel_meas = s_ab[k] + accel_bias_body + rng.normal(0, ACCEL_NOISE_STD, 3)
        rk.predict(accel_meas, s_R[k], s_dt[k], t_ns)
        if warm:
            rk.update_velocity(s_v[k], (0.1 ** 2) * np.eye(3), sim_time_ns=t_ns)

        if s_tgt[k] >= 0 and t >= next_fix_t:
            next_fix_t = t + fix_dt
            # production per-step in-plane basis (s_e1 horizontal, s_e2 vertical on the g3->g4 axis) +
            # target_u depth axis. RE-LABEL by |z| so sigma_vert/bias always ride the vertical axis even
            # off-axis; on the binding axis this is a no-op (s_e1==e1_lat, s_e2==e2_vert already).
            e1_raw, e2_raw, target_u = s_e1[k], s_e2[k], s_u[k]
            if abs(e1_raw[2]) > abs(e2_raw[2]):
                e1, e2 = e2_raw, e1_raw   # e2 := more-vertical
            else:
                e1, e2 = e1_raw, e2_raw
            # ---- CHANGE 2of3: anisotropic noise draw (sigma_lat on e1, sigma_vert on e2) +
            #      CHANGE 3of3: constant vertical fix bias vert_fix_bias_m along e2 added to z ----
            n_lat = (rng.normal(0, sigma_lat) * e1
                     + rng.normal(0, sigma_vert) * e2
                     + rng.normal(0, RADIAL_SIGMA) * target_u)
            z = s_p[k] + n_lat + vert_fix_bias_m * e2
            # ---- matching anisotropic fix cov in the (e1,e2,u) basis (was: sig2_ip*(e1e1+e2e2)) ----
            cov = (sig2_lat * np.outer(e1, e1)
                   + sig2_vert * np.outer(e2, e2)
                   + sig2_rad * np.outer(target_u, target_u))
            fix_apply_queue.append((t + L_s, t_ns, z.copy(), cov.copy()))

        if fix_apply_queue:
            fix_apply_queue.sort(key=lambda e: e[0])
            while fix_apply_queue and fix_apply_queue[0][0] <= t + 1e-12:
                _, cap_ns, z, cov = fix_apply_queue.pop(0)
                rk.update_position_at(cap_ns, z, cov)
                if weakvel:
                    if last_fix_z is not None and last_fix_t is not None:
                        dt_fix = (cap_ns - last_fix_t) / 1e9
                        if dt_fix > 1e-3:
                            v_meas = (z - last_fix_z) / dt_fix
                            # NOTE: ME.fly_lap's weakvel uses inplane_sigma here; v2 uses sigma_lat as
                            # the scalar stand-in (weakvel is a context-only REF branch, not used by the
                            # cold decision cells; with sigma_lat==sigma_vert it is identical to ME).
                            sig_v = np.sqrt(2.0) * sigma_lat / dt_fix
                            rk.update_velocity(v_meas, (sig_v ** 2) * np.eye(3), sim_time_ns=t_ns)
                    last_fix_z = z.copy()
                    last_fix_t = cap_ns

    err = rk.position - tr["p_final_true"]
    along = float(np.dot(err, u34))
    inplane_vec = err - along * u34
    return float(np.linalg.norm(inplane_vec))


# =================================================================================================
# run_cell_v2 -- thin MC wrapper mirroring ME.run_cell, exposing the two new knobs. Seed formula is
# ME.run_cell's EXACT formula EXTENDED with sigma_vert + vert_fix_bias_m terms (so isotropic/zero-bias
# cells reuse ME's seeds bit-for-bit when sigma_vert==sigma_lat and vert_fix_bias_m==0; see note).
# =================================================================================================
def _seed_for(s, v_race, att_bias_deg, sigma_lat, sigma_vert, fix_rate, vel_mode, bias_mode,
              latency_ms, vert_fix_bias_m, match_me):
    """Per-lap seed. If match_me is True AND sigma_vert==sigma_lat AND vert_fix_bias_m==0, returns the
    IDENTICAL integer ME.run_cell uses (so the two engines draw the SAME random stream -> the regression
    matches to ~0, not just within MC noise). Otherwise it folds the two extra knobs in so distinct v2
    cells get distinct streams."""
    base = (ME.SEED + 101 * s + int(round(v_race)) * 13 + int(round(att_bias_deg * 100)) * 7
            + int(round(sigma_lat * 1000)) * 17 + int(round(fix_rate * 1000)) * 23
            + {"cold": 1, "warm": 2, "weakvel": 3}[vel_mode] * 1009
            + {"random3d": 0, "inplane": 1}[bias_mode] * 3001 + int(round(latency_ms)) * 53)
    if match_me and abs(sigma_vert - sigma_lat) < 1e-12 and abs(vert_fix_bias_m) < 1e-12:
        return base
    return base + int(round(sigma_vert * 1000)) * 131 + int(round(vert_fix_bias_m * 1000)) * 211


def run_cell_v2(v_race, sigma_lat, sigma_vert, att_bias_deg, bias_mode, fix_rate, vel_mode,
                latency_ms=15.0, vert_fix_bias_m=0.0, n_mc=400, match_me_seeds=False):
    """Monte-Carlo one operating-space cell with anisotropic sigma + optional vertical fix bias.
    Returns the same summary dict shape ME.run_cell returns, plus the two new knobs. Set
    match_me_seeds=True for the isotropic/zero-bias regression so seeds == ME.run_cell exactly."""
    accel_bias_mag = ME.att_deg_to_accel_bias(att_bias_deg)
    misses = []
    for s in range(n_mc):
        seed = _seed_for(s, v_race, att_bias_deg, sigma_lat, sigma_vert, fix_rate, vel_mode,
                         bias_mode, latency_ms, vert_fix_bias_m, match_me_seeds)
        rng = np.random.default_rng(seed)
        misses.append(fly_lap_v2(rng, v_race, sigma_lat, sigma_vert, accel_bias_mag, bias_mode,
                                 fix_rate, vel_mode, latency_ms, vert_fix_bias_m=vert_fix_bias_m))
    m = np.array(misses)
    p50 = float(np.percentile(m, 50))
    p90 = float(np.percentile(m, 90))
    p99 = float(np.percentile(m, 99))
    frac_over = {f"{r:.2f}": float(np.mean(m >= ME.margin_at(r))) for r in ME.RADIUS_BAND}
    clears = {f"{r:.2f}": bool(p90 < ME.margin_at(r) and p99 < ME.margin_at(r)) for r in ME.RADIUS_BAND}
    return dict(
        v_race=v_race, sigma_lat=sigma_lat, sigma_vert=sigma_vert, att_bias_deg=att_bias_deg,
        accel_bias_mag=accel_bias_mag, bias_mode=bias_mode, fix_rate=fix_rate, vel_mode=vel_mode,
        latency_ms=latency_ms, vert_fix_bias_m=vert_fix_bias_m, n_mc=n_mc,
        inplane_rms=float(np.sqrt(np.mean(m ** 2))),
        inplane_p50=p50, inplane_p90=p90, inplane_p99=p99, inplane_max=float(m.max()),
        frac_over=frac_over, clears_p90_and_p99=clears,
    )


if __name__ == "__main__":
    # Smoke: prints the in-plane basis labelling + a single isotropic vs anisotropic cell so a human can
    # eyeball that the engine wires up. The MANDATORY regression lives in regression_proof.py.
    import json
    seg, L, u34 = ME.seg_axes(ME.GATES[3], ME.GATES[4])
    e1, e2 = _label_inplane_axes(u34)
    print(f"u34={u34}")
    print(f"e1 (lateral) ={e1}  |z|={abs(e1[2]):.4f}")
    print(f"e2 (vertical)={e2}  |z|={abs(e2[2]):.4f}  (down.e2={np.dot([0,0,1.0],e2):+.4f})")
    iso = run_cell_v2(37.0, 0.10, 0.10, 0.0, "random3d", 0.07, "cold", n_mc=200, match_me_seeds=True)
    print("isotropic(0.10,0.10) b0 fr0.07:", json.dumps(
        {k: iso[k] for k in ("inplane_p50", "inplane_p90", "inplane_p99")}, indent=0))
    aniso = run_cell_v2(37.0, 0.19, 0.10, 0.0, "random3d", 0.07, "cold", n_mc=200, vert_fix_bias_m=0.25)
    print("aniso(lat0.19,vert0.10) b0 fr0.07 bias+0.25:", json.dumps(
        {k: aniso[k] for k in ("inplane_p50", "inplane_p90", "inplane_p99")}, indent=0))
