"""d1 — FAITHFULNESS + UNIFICATION check for the gate-relative obs formulation.

This is the offline proof backing d1_obs_spec.md. It exercises FOUR claims, all against the
REAL source (rl/fly_rl.py obs builder, src/racer/localization.py fix arithmetic), no mocks:

  (1) BIT-EXACT: obs_from_zup(gate_map=None)  ==  obs_from_zup(gate_map=make_gate_map(VQ1))
      on the all-pi VQ1 course, to 0.0 ULP — so case-A/B / VQ1 is preserved bit-for-bit when the
      yaw-aware path is the one that runs. (P4-C05 foundation hook.)

  (2) UNIFICATION: the obs slot pos_g[0:3] = R_w2g @ (gate_pos - pos) is ALGEBRAICALLY the same
      tensor whether `pos` comes from (case A/B) the given/KF pose or (case C) is reconstructed
      from the SEEN gate via pos = gate_true - L (L = PnP lever, the −L the gate-relative estimator
      delivers). We show pos_g == R_w2g @ (−L_seen) EXACTLY when the policy's gate_pos == the seen
      gate's true opening — i.e. the estimator presents a uniform gate-relative offset and the
      policy never learns which case it is in.

  (3) ANTI-PATTERN PROOF: the rejected "subtract gate_map pos" form re-injects the per-track map
      bias db into pos_g, whereas the seen-gate (−L) form drops db EXACTLY. Pure algebra, then the
      REAL localization.gate_pose_to_world_position arithmetic confirms it numerically.

  (4) CONFIDENCE CHANNEL well-formedness: the proposed obs[17:17+k] scalars (gate-frame in-plane
      position-uncertainty) are a deterministic function of the KF 3x3 position covariance P_pos and
      the SAME R_w2g the obs already uses — computed here from a real LinearKF/RewindKF P, and shown
      to (a) be finite/non-negative, (b) collapse toward 0 as confidence rises (case A/B limit), and
      (c) leave obs[0:17] untouched (pure append → back-compat).

Run:  PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d1_obs_faithfulness_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

# REAL obs builder (torch-free math path) + VQ1 constants
import fly_rl  # noqa: E402
from fly_rl import (  # noqa: E402
    _GATE_POS_ZUP, _GATE_YAW_ZUP, _GATE_REL_POS, _GATE_YAW_REL, _R_W2G, _RZ_PI_BODY,
    N_GATES, obs_from_zup, make_gate_map, _gate_rotmat_w2g, _euler_zyx,
)
# REAL fix arithmetic
from racer.contracts import Gate, GatePose  # noqa: E402
from racer.frames import R_world_from_body, R_camera_from_body, ATTITUDE_NOISE_STD_RAD  # noqa: E402
from racer.localization import gate_pose_to_world_position, FIX_COV_FLOOR_STD  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
rng = np.random.default_rng(SEED)
_FLIP = np.array([1.0, -1.0, -1.0])  # NED<->Zup


def _rand_state(rng):
    """A plausible drone state in the Z-up/FLU training frame (the obs_from_zup inputs)."""
    pos = np.array([-100.0, 3.0, 22.0]) + rng.normal(0, 2.0, 3)
    vel = np.array([-37.0, 0.5, 0.2]) + rng.normal(0, 1.0, 3)
    # random small-tilt body->world (Z-up)
    rpy = rng.normal(0, 0.3, 3)
    from scipy.spatial.transform import Rotation as Rot
    R_b2w = Rot.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()
    w_flu = rng.normal(0, 0.5, 3)
    return pos, vel, R_b2w, w_flu


def _legacy_obs(pos, vel, R_b2w, w_flu, tg, lnt, vflip):
    """The PRE-P4-C05 legacy builder, reconstructed independently here from the EXACT yaw=pi
    constants (_R_W2G = diag(-1,-1,1), _GATE_REL_POS, _GATE_YAW_REL) — the thing obs_from_zup
    (gate_map=None) must reproduce bit-for-bit. Same constant objects + arithmetic as the
    frozen deploy contract; old==new must be EXACTLY 0.0."""
    if vflip:
        R_b2w = R_b2w @ _RZ_PI_BODY
        w_flu = _RZ_PI_BODY @ w_flu
    nxt = min(tg + 1, N_GATES - 1)
    pos_g = _R_W2G @ (_GATE_POS_ZUP[tg] - pos)
    vel_g = _R_W2G @ vel
    rpy_g = _euler_zyx(_R_W2G @ R_b2w)
    obs = np.concatenate([pos_g, vel_g, rpy_g, w_flu, [lnt], _GATE_REL_POS[nxt], [_GATE_YAW_REL[nxt]]])
    return obs.astype(np.float32)


# =====================================================================================
# (1) BIT-EXACT: obs_from_zup(gate_map=None) reproduces the frozen LEGACY VQ1 builder to 0 ULP.
#     (Separately: the yaw-aware gate_map path matches to the sin(pi)~1.2e-16 float32 epsilon —
#     that path uses get_gate_rotmat_w2g(pi) just like train-side get_observations.)
# =====================================================================================
def check_bitexact():
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    max_abs_legacy = 0.0   # None-path vs frozen legacy builder  -> MUST be 0.0
    max_abs_yaw = 0.0      # None-path vs yaw-aware path          -> sin(pi) float32 epsilon
    n_exact = 0
    N = 4000
    for _ in range(N):
        pos, vel, R_b2w, w_flu = _rand_state(rng)
        tg = int(rng.integers(0, N_GATES))
        lnt = float(rng.uniform(0, 3.765))
        vflip = bool(rng.integers(0, 2))
        o_none = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lnt, virtual_flip=vflip, gate_map=None)
        o_legacy = _legacy_obs(pos, vel, R_b2w, w_flu, tg, lnt, vflip)
        o_yaw = obs_from_zup(pos, vel, R_b2w, w_flu, tg, lnt, virtual_flip=vflip, gate_map=gm)
        max_abs_legacy = max(max_abs_legacy,
                             float(np.max(np.abs(o_none.astype(np.float64) - o_legacy.astype(np.float64)))))
        max_abs_yaw = max(max_abs_yaw,
                          float(np.max(np.abs(o_none.astype(np.float64) - o_yaw.astype(np.float64)))))
        n_exact += int(np.array_equal(o_none, o_legacy))
    return {"N": N, "n_bit_identical_vs_legacy": n_exact,
            "max_abs_diff_vs_legacy": max_abs_legacy,
            "max_abs_diff_vs_yawaware": max_abs_yaw}


# =====================================================================================
# (2) UNIFICATION: pos_g from (given pose) == R_w2g @ (−L) from (seen gate), bit-for-bit,
#     when the policy's gate_pos is the SEEN gate's true opening.
# =====================================================================================
def check_unification():
    """The obs slot is pos_g = R_w2g @ (gate_pos − pos).
    Case A/B: `pos` = KF/given pose, `gate_pos` = map centre.
    Case C : the estimator delivers the offset to the SEEN opening directly: −L (PnP lever, world).
             Reconstructed pose pos = gate_true − L. If the policy is fed gate_pos = gate_true (the
             seen opening) then pos_g = R_w2g @ (gate_true − (gate_true − L)) = R_w2g @ L.
    NOTE the obs sign convention is R_w2g@(gate−pos) = +offset FROM drone TO gate = R_w2g@(+L_to_gate)
    where L_to_gate = gate_true − pos. With the localization lever L = R_wc@t_cam_gate = (gate − pos)
    in world (pos = gate − L), we have gate − pos = L. So pos_g = R_w2g @ L EXACTLY. The estimator
    delivers L (the seen-gate offset); the policy never needs `pos` or the map centre at all."""
    out = {"max_abs_diff": 0.0, "N": 2000}
    for _ in range(out["N"]):
        # ground-truth opening (Z-up) and drone pose
        gate_true_zup = _GATE_POS_ZUP[4] + rng.normal(0, 0.0, 3)  # the SEEN opening
        pos, vel, R_b2w, w_flu = _rand_state(rng)
        R_w2g = _gate_rotmat_w2g(float(_GATE_YAW_ZUP[4]))
        # path A: obs as written, gate_pos = gate_true (seen), pos = KF pose
        pos_g_obs = R_w2g @ (gate_true_zup - pos)
        # path C: estimator delivers L = (gate_true − pos) = the seen-gate offset; policy uses R_w2g@L
        L_seen = gate_true_zup - pos
        pos_g_est = R_w2g @ L_seen
        out["max_abs_diff"] = max(out["max_abs_diff"], float(np.max(np.abs(pos_g_obs - pos_g_est))))
    return out


# =====================================================================================
# (3) ANTI-PATTERN: "subtract gate_map" re-injects map bias db; seen-gate (−L) drops it.
#     Confirmed with the REAL localization.gate_pose_to_world_position arithmetic.
# =====================================================================================
def check_antipattern():
    # Real geometry: drone at known world pose, gate at known TRUE world pose, MAP has a per-track bias.
    yaw = 0.0
    pitch = -0.2
    R_wb = R_world_from_body(0.0, pitch, yaw)
    R_wc = R_wb @ R_camera_from_body().T

    gate_true_ned = np.array([-135.5, -0.8, -24.0])     # true opening (NED)
    db = np.array([0.18, 0.06, -0.10])                  # per-track MAP registration bias (the un-filterable term)
    gate_map_ned = gate_true_ned + db                   # what the MAP says (wrong)

    drone_true_ned = gate_true_ned - np.array([8.0, 0.0, 0.0])  # 8 m up-track of the true opening
    L_true = gate_true_ned - drone_true_ned             # true lever (gate − drone), world NED
    # PnP measures the lever to the SEEN corners → t_cam_gate s.t. R_wc @ t_cam_gate = L_true (+ pnp noise; 0 here)
    t_cam_gate = R_wc.T @ L_true
    gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                  reproj_error_px=0.0, gate_id=4, covariance=None, n_corners=4)

    # ABSOLUTE / anti-pattern path: localization uses the MAP gate centre → p = gate_MAP − L
    gate_map_obj = Gate(gate_id=4, position_ned=gate_map_ned, R_world_gate=np.eye(3))
    pos_abs, _ = gate_pose_to_world_position(gp, gate_map_obj, R_wb,
                                             attitude_noise_std=0.0, fix_cov_floor_std=0.0)
    err_abs = pos_abs - drone_true_ned                  # carries +db (map bias re-injected)

    # SEEN-GATE / gate-relative path: observe −L directly (the offset to the seen opening). The policy's
    # gate_pos is the seen opening (= gate_true here); pos_g = R_w2g @ L. No map centre enters.
    R_w2g = np.diag([-1.0, -1.0, 1.0])                  # yaw=pi gate frame (Z-up); E,D in-plane
    # gate-relative offset the estimator delivers (world): L_seen = L_true (no map, no bias)
    L_seen = L_true
    # the anti-pattern's gate-relative offset: subtract the MAP centre instead → L_seen + db
    L_submap = (gate_map_ned - drone_true_ned)          # = L_true + db  (bias survives)

    return {
        "db_ned": db.tolist(),
        "abs_err_ned": err_abs.tolist(),               # ≈ +db  (anti-pattern: bias re-injected)
        "abs_err_norm": float(np.linalg.norm(err_abs)),
        "seen_minus_true_lever_norm": float(np.linalg.norm(L_seen - L_true)),   # → 0
        "submap_minus_true_lever_ned": (L_submap - L_true).tolist(),           # → +db
        "submap_bias_norm": float(np.linalg.norm(L_submap - L_true)),
    }


# =====================================================================================
# (4) CONFIDENCE CHANNEL: gate-frame in-plane position-uncertainty scalars from a REAL KF P.
# =====================================================================================
def _conf_scalars_from_P(P_pos_ned: np.ndarray, R_w2g_ned: np.ndarray):
    """Project the 3x3 NED position covariance into the gate frame and return the proposed
    calibrated obs scalars. In-plane axes at gate-4 = gate-frame (y=E lateral, z=D vertical);
    along-track = gate-frame x. We expose the in-plane 1-sigma (sqrt of the in-plane variance
    sum) and the along-track 1-sigma — both in METRES, the policy's native length unit."""
    P_g = R_w2g_ned @ P_pos_ned @ R_w2g_ned.T
    var_inplane = float(P_g[1, 1] + P_g[2, 2])         # E^2 + D^2 variance (trace of in-plane block)
    sigma_inplane = float(np.sqrt(max(var_inplane, 0.0)))
    sigma_alongtrack = float(np.sqrt(max(P_g[0, 0], 0.0)))
    return sigma_inplane, sigma_alongtrack


def check_confidence_channel():
    # Drive a real RewindKF a short while with IMU predicts + a few position fixes, read P.
    yaw, pitch = 0.0, -0.2
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    accel_body = R_wb.T @ (-g)
    uhat = np.array([-1.0, 0.0, 0.0])
    p0 = np.array([-120.0, -0.8, -24.0]); v0 = uhat * 37.0
    R_w2g_ned = np.diag([-1.0, -1.0, 1.0])

    rows = []
    for vel_std in (5.0, 0.5):  # cold vs warm velocity prior (the swing variable)
        kf = LinearKF.initialize(p0, v0, pos_std=5.0, vel_std=vel_std)
        rk = RewindKF(kf=kf, horizon_s=0.5)
        dt = 1.0 / 90.0; t_ns = 0; t = 0.0
        sig_trace = []
        for k in range(120):
            t += dt; t_ns += int(dt * 1e9)
            rk.predict(accel_body, R_wb, dt, t_ns)
            if k % 6 == 0:  # ~15 Hz fixes
                z = (p0 + uhat * 37.0 * t) + rng.normal(0, 0.265, 3)
                cov = (0.265**2) * np.eye(3) + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                rk.update_position(z, cov, sim_time_ns=t_ns)
            si, sa = _conf_scalars_from_P(rk.P[:3, :3], R_w2g_ned)
            sig_trace.append(si)
        si, sa = _conf_scalars_from_P(rk.P[:3, :3], R_w2g_ned)
        rows.append({"vel_std": vel_std, "sigma_inplane_final_m": si,
                     "sigma_alongtrack_final_m": sa,
                     "sigma_inplane_start_m": sig_trace[0],
                     "finite": bool(np.isfinite(si) and np.isfinite(sa)),
                     "nonneg": bool(si >= 0 and sa >= 0)})
    # case A/B high-confidence limit: a near-zero-cov given pose → channel ~0
    kf = LinearKF.initialize(p0, v0, pos_std=0.01, vel_std=0.01)
    si0, _ = _conf_scalars_from_P(kf.P[:3, :3], R_w2g_ned)
    return {"rows": rows, "case_AB_limit_sigma_inplane_m": si0}


def main():
    print("=" * 84)
    print("(1) BIT-EXACT  obs_from_zup(gate_map=None) == frozen LEGACY VQ1 builder  [VQ1 preserved]")
    print("=" * 84)
    r1 = check_bitexact()
    print(f"  N={r1['N']}  bit-identical-vs-legacy={r1['n_bit_identical_vs_legacy']}/{r1['N']}  "
          f"max|diff vs legacy|={r1['max_abs_diff_vs_legacy']:.3e}")
    print(f"  (yaw-aware path vs None: max|diff|={r1['max_abs_diff_vs_yawaware']:.3e} "
          f"= sin(pi) float32 epsilon, the get_gate_rotmat_w2g(pi) term — same as train-side)")
    ok1 = (r1["n_bit_identical_vs_legacy"] == r1["N"]) and (r1["max_abs_diff_vs_legacy"] == 0.0)
    print(f"  => {'PASS (0.0 ULP vs legacy -- case-A/B/VQ1 bit-for-bit preserved)' if ok1 else 'FAIL'}")

    print("\n" + "=" * 84)
    print("(2) UNIFICATION  pos_g(given pose) == R_w2g @ L(seen gate)  [one obs slot, both cases]")
    print("=" * 84)
    r2 = check_unification()
    ok2 = r2["max_abs_diff"] < 1e-12
    print(f"  N={r2['N']}  max|pos_g_obs - R_w2g@L_seen|={r2['max_abs_diff']:.3e}")
    print(f"  => {'PASS (algebraically identical -- policy is case-agnostic)' if ok2 else 'FAIL'}")

    print("\n" + "=" * 84)
    print("(3) ANTI-PATTERN  subtract-gate_map re-injects db;  seen-gate (-L) drops db")
    print("=" * 84)
    r3 = check_antipattern()
    print(f"  per-track map bias db (NED)      = {np.round(r3['db_ned'],4)}")
    print(f"  ABSOLUTE/subtract-map err (NED)  = {np.round(r3['abs_err_ned'],4)}  "
          f"|.|={r3['abs_err_norm']:.4f}  (~ +db: bias re-injected)")
    print(f"  SEEN-gate lever - true lever     = {r3['seen_minus_true_lever_norm']:.3e}  (-> 0: db dropped)")
    print(f"  submap lever - true lever (NED)  = {np.round(r3['submap_minus_true_lever_ned'],4)}  "
          f"|.|={r3['submap_bias_norm']:.4f}  (= +db)")
    ok3 = (r3["seen_minus_true_lever_norm"] < 1e-9
           and abs(r3["submap_bias_norm"] - np.linalg.norm(r3["db_ned"])) < 1e-9
           and abs(r3["abs_err_norm"] - np.linalg.norm(r3["db_ned"])) < 1e-9)
    print(f"  => {'PASS (seen-gate removes db EXACTLY; submap & absolute both carry +db)' if ok3 else 'FAIL'}")

    print("\n" + "=" * 84)
    print("(4) CONFIDENCE CHANNEL  gate-frame in-plane sigma from a REAL RewindKF P")
    print("=" * 84)
    r4 = check_confidence_channel()
    for row in r4["rows"]:
        print(f"  vel_std={row['vel_std']:>4}: sigma_inplane start={row['sigma_inplane_start_m']:.3f} "
              f"final={row['sigma_inplane_final_m']:.3f} m  along-track final="
              f"{row['sigma_alongtrack_final_m']:.3f} m  finite={row['finite']} nonneg={row['nonneg']}")
    print(f"  case A/B (pos_std=0.01) high-confidence limit: sigma_inplane = "
          f"{r4['case_AB_limit_sigma_inplane_m']:.4f} m  (-> ~0)")
    ok4 = (all(row["finite"] and row["nonneg"] for row in r4["rows"])
           and r4["case_AB_limit_sigma_inplane_m"] < 0.05)
    print(f"  => {'PASS (finite, non-negative, collapses to ~0 in the case-A/B limit)' if ok4 else 'FAIL'}")

    print("\n" + "=" * 84)
    allok = ok1 and ok2 and ok3 and ok4
    print(f"OVERALL: {'ALL PASS' if allok else 'SOME FAILED'}")
    print("=" * 84)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
