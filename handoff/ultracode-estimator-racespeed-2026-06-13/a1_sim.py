"""a1_sim.py -- closed-loop estimator sim for the gate-4 race-speed window.

MISSION (estimator-racespeed): MEASURE the achieved filtered position 1-sigma at the
gate-4 plane crossing for a VALID (zero-contact) run, fed by the realistic case-C vision
fix stream, using the REAL racer.state_estimator.LinearKF + the REAL RewindKF OOSM wrapper.

Composes (does NOT re-implement) the filter math:
  - racer.state_estimator.LinearKF                       (predict / update_position, Joseph form)
  - handoff/.../kf_rewind_buffer.RewindKF                (OOSM rewind/replay)
  - racer.localization.gate_pose_to_world_position       (REAL fix-cov: 0.40 floor + 1.4deg
                                                          attitude lever + PnP block)

TRUTH trajectory: g3 -> g4, 24.4 m, ~level, ~37 m/s. Baseline = constant velocity along the
segment; +1 mild-accel sensitivity case. Body specific force at ~90 Hz drives KF.predict;
accel meas noise 0.3 m/s^2 (matches accel_noise_std). Vision fixes at 30Hz*acceptance; each
fix = truth_pos + RESAMPLED measured off_ned residual (range-band conditioned), applied at
CAPTURE time via RewindKF.update_position_at(t_fix_ns, z, cov) with latency L; a naive
in-place LinearKF.update_position(now) arm runs for contrast.

SWEEP: L in {6,16,112} ms x eff-fix-rate {47%(~14Hz),25%(~7Hz),10%(~3Hz)} x error-model
{RAW=with measured bias, DEBIASED=zero-mean noise}. Monte-Carlo >=300 seeds/cell.

MEASURE at the gate-4 plane crossing, per axis (along-track N, in-plane E, in-plane D):
  (i) filter claimed 1-sigma = sqrt(diag(P)); (ii) empirical bias + RMS of (est-truth) over
  seeds; (iii) NEES (chi2(3) consistency); (iv) in-plane miss=sqrt(E^2+D^2). Reports whether
  filtered in-plane 1-sigma AND empirical RMS clear <0.05 m, + rewind-vs-naive delta.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/a1_sim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.state_estimator import LinearKF, GRAVITY_NED  # noqa: E402
from racer.localization import gate_pose_to_world_position  # noqa: E402
from racer.contracts import Gate, GatePose  # noqa: E402
from racer.frames import (  # noqa: E402
    R_world_from_body,
    R_camera_from_body,
    ATTITUDE_NOISE_STD_RAD,
)
from kf_rewind_buffer import RewindKF  # noqa: E402

HERE = Path(__file__).resolve().parent
PERCEPTION = ROOT / "handoff" / "perception-char-2026-06-08" / "characterize_course_60s.json"

# ----------------------------------------------------------------------------- #
# Geometry (track_map, FACTS) -- gate-4 window
# ----------------------------------------------------------------------------- #
G3 = np.array([-111.5, -5.1, 24.57])      # bottom-centre NED
G4 = np.array([-135.5, -0.8, 25.36])
SEG = G4 - G3                              # g3->g4 segment
SEG_LEN = float(np.linalg.norm(SEG))       # ~24.4 m
SEG_HAT = SEG / SEG_LEN                     # unit along-track
SPEED = 37.0                                # m/s post-gate-3 cruise

# In-plane axes at gate-4: motion ~along -N => gate normal ~ -N => in-plane = E (lateral),
# D (vertical); along-track = N. (FACTS gate-4 geometry block.) We report per-axis N/E/D and
# label N=along-track, E&D=in-plane; the binding in-plane miss = sqrt(E^2 + D^2).
AX_LABEL = {0: "N (along-track)", 1: "E (in-plane lateral)", 2: "D (in-plane vertical)"}

# ----------------------------------------------------------------------------- #
# Attitude assumption (STATED): ~level forward cruise at 37 m/s. A constant-velocity cruise
# needs thrust to balance gravity + aero drag; with the measured linear_drag ~0.21/s the
# drag accel at 37 m/s is ~7.77 m/s^2, requiring a forward pitch ~38 deg (atan(7.77/9.807))
# to hold level constant velocity. We hold that drag-hold attitude (pitch nose-forward, zero
# roll/yaw=along -N). This is the realistic high-speed cruise posture; it makes the predict
# step's skew(specific_force) attitude-noise injection NON-trivial (tilted gravity injects
# phantom horizontal accel), which is the honest worst-ish case for blind-coast growth.
# Sensitivity: we also report a near-level (pitch=5 deg) attitude variant in the findings.
DRAG_ACCEL = 0.21 * SPEED
DRAG_HOLD_PITCH_RAD = float(np.arctan2(DRAG_ACCEL, GRAVITY_NED[2]))   # ~0.67 rad ~38.4 deg
# Yaw: heading along the segment in the N-E plane. SEG points mostly -N (and slightly +E).
YAW_RAD = float(np.arctan2(SEG_HAT[1], SEG_HAT[0]))   # atan2(E, N); ~pi-ish (heading -N, slight +E)

IMU_HZ = 90.0
IMU_DT = 1.0 / IMU_HZ
DETECTOR_HZ = 30.0

# Acceptance / effective-fix-rate cells (acceptance fraction of the 30 Hz detector stream)
ACCEPT_CELLS = {
    "47pct_14Hz": 0.47,    # canonical acceptance -> ~14.1 Hz effective
    "25pct_7Hz": 0.25,     # ~7.5 Hz
    "10pct_3Hz": 0.10,     # ~3.0 Hz
}
LATENCY_CELLS_MS = {"L6_edge_p50": 6.0, "L16_edge_p90": 16.0, "L112_cpu_ub": 112.0}
ERROR_MODELS = ["raw", "debiased"]
N_SEEDS = 400              # >=300 required
BASE_SEED = 20260613

# Buffer horizon: must exceed the largest latency (112 ms) with margin. Default 0.5 s is fine
# for edge L; we set 0.5 s so the CPU-upper-bound L=112 ms is comfortably inside (FACTS warns
# horizon_s < true L drops ALL fixes -> divergence). State this explicitly.
HORIZON_S = 0.5


# ----------------------------------------------------------------------------- #
# Measured per-fix residual pool, range-band conditioned (RAW + DEBIASED)
# ----------------------------------------------------------------------------- #
def load_residual_pools():
    """Return per-band off_ned residual arrays (RAW) + per-band means (for DEBIASED).

    Band on the LIVE range-to-current-gate field `range_m` (the field that reproduces the
    FACTS per-band stats exactly). Near the gate-4 crossing the range-to-gate-4 is small, so
    we draw from the [0,8) and [8,16) m bands as FACTS directs. GOOD cut = |off_ned|<3 m
    (the in-loop-accepted-like cut, n=109 total).
    """
    d = json.loads(PERCEPTION.read_text())
    rows = d["rows"]
    bands = {"[0,8)": [], "[8,16)": [], "[16,24)": []}
    for r in rows:
        o = r.get("off_ned")
        rg = r.get("range_m")
        if o is None or rg is None:
            continue
        o = np.asarray(o, dtype=np.float64)
        if np.linalg.norm(o) >= 3.0:
            continue
        if 0.0 <= rg < 8.0:
            bands["[0,8)"].append(o)
        elif 8.0 <= rg < 16.0:
            bands["[8,16)"].append(o)
        elif 16.0 <= rg < 24.0:
            bands["[16,24)"].append(o)
    pools = {k: np.asarray(v) for k, v in bands.items() if len(v) > 0}
    means = {k: v.mean(axis=0) for k, v in pools.items()}
    stds = {k: v.std(axis=0, ddof=1) for k, v in pools.items()}
    return pools, means, stds


def band_for_range(rng_to_g4: float) -> str:
    """Pick the residual band for the current range-to-gate-4. Near crossing uses [0,8);
    farther in the approach uses [8,16). We cap at 16 m (the [8,16) band) per FACTS direction
    to use the [0,8) and [8,16) near-gate-4 bands."""
    if rng_to_g4 < 8.0:
        return "[0,8)"
    return "[8,16)"


# ----------------------------------------------------------------------------- #
# Fix covariance via the REAL model
# ----------------------------------------------------------------------------- #
def fix_cov_real(drone_pos: np.ndarray, R_wb: np.ndarray) -> np.ndarray:
    """Build the world-fix covariance using the REAL racer.localization model.

    We synthesize the geometry the model needs: the gate (gate-4) world pose, the trusted
    attitude R_wb, and the gate-in-camera translation t_cam_gate consistent with the true
    lever arm L = G4 - drone_pos (world). gate_pose_to_world_position then assembles:
        Cov = PNP_FIX_COV_INFLATION * R_wc Cov(t_cam_gate) R_wc^T
              + sigma_theta^2 (|L|^2 I - L L^T)               (attitude lever, grows w/ range)
              + FIX_COV_FLOOR_STD^2 I                         (0.40 m systematics floor)
    PnP block: we pass covariance=None so the model uses its analytic default
    (default_position_std=0.3 m isotropic for the camera-frame translation). This is the real
    shipped behaviour when a per-fix analytic PnP cov is unavailable; the dominant terms at
    these ranges (<=16 m) are the floor (0.40 m) and the lever (1.4deg * |L|).
    """
    R_wc = R_wb @ R_camera_from_body().T
    lever_world = G4 - drone_pos                      # gate rel. drone, world NED
    t_cam_gate = R_wc.T @ lever_world                 # gate origin in camera frame
    gate = Gate(gate_id=4, position_ned=G4.copy(), R_world_gate=np.eye(3))
    gp = GatePose(
        frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3),
        t_cam_gate=t_cam_gate, reproj_error_px=0.3, gate_id=4,
        covariance=None, n_corners=4,
    )
    _pos, cov = gate_pose_to_world_position(gp, gate, R_wb)
    return cov


# ----------------------------------------------------------------------------- #
# One Monte-Carlo run of one cell
# ----------------------------------------------------------------------------- #
def truth_pos(t: float, accel_alongtrack: float) -> np.ndarray:
    """Truth position at time t (s) since gate-3 crossing. Constant velocity baseline along
    SEG_HAT at SPEED; +0.5*a*t^2 along-track for the mild-accel case."""
    s = SPEED * t + 0.5 * accel_alongtrack * t * t
    return G3 + SEG_HAT * s


def truth_vel(t: float, accel_alongtrack: float) -> np.ndarray:
    return SEG_HAT * (SPEED + accel_alongtrack * t)


def run_cell(L_ms: float, accept: float, error_model: str, accel_alongtrack: float,
             pools, means, n_seeds: int, base_seed: int):
    """Monte-Carlo a single sweep cell. Returns per-axis stats at the gate-4 plane crossing
    for BOTH the rewind (OOSM) arm and the naive in-place arm.

    Timeline: t=0 at gate-3, advance at IMU_DT to the gate-4 crossing (t_cross = SEG_LEN/SPEED
    for the constant-v case; for the accel case we solve for plane crossing along-track).
    """
    L_s = L_ms / 1e3
    # gate-4 plane crossing time (when along-track distance == SEG_LEN)
    if abs(accel_alongtrack) < 1e-9:
        t_cross = SEG_LEN / SPEED
    else:
        # solve 0.5 a t^2 + v t - L = 0
        a, b, c = 0.5 * accel_alongtrack, SPEED, -SEG_LEN
        t_cross = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    R_wb = R_world_from_body(0.0, DRAG_HOLD_PITCH_RAD, YAW_RAD)
    g = GRAVITY_NED

    # Vision-fix schedule: detector ticks at 30 Hz; each tick is ACCEPTED w.p. `accept`.
    # Effective fix rate = 30 * accept. Fix capture time = tick time; it is *applied* at
    # tick_time + L (when the pipeline finishes), but RewindKF rewinds to capture time.
    det_dt = 1.0 / DETECTOR_HZ

    # error-model bias handling
    debias = (error_model == "debiased")

    # accumulators (per arm: rewind, naive)
    err_rw = []   # (est - truth) at crossing, NED, per seed
    err_nv = []
    P_rw_diag = []   # claimed variance diag at crossing (rewind arm)
    P_nv_diag = []
    nees_rw = []
    nees_nv = []
    dropped_counts = []

    n_imu = int(np.ceil(t_cross / IMU_DT)) + 2

    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + 1009 * s + int(L_ms) * 31 + int(accept * 100) * 7
                                    + (1 if debias else 0))

        # --- initialise both filters at gate-3 with the SAME prior ---
        p0 = truth_pos(0.0, accel_alongtrack)
        v0 = truth_vel(0.0, accel_alongtrack)
        # case-C: position from vision only, velocity from IMU integration. Initialise with a
        # realistic post-gate-3 prior uncertainty (we are mid-race, recently fixed): pos_std
        # ~0.6 m (one fix's worth), vel_std ~0.5 m/s. Add a sampled init error so seeds differ.
        pos_std0, vel_std0 = 0.6, 0.5
        p_init = p0 + rng.normal(0, pos_std0, 3)
        v_init = v0 + rng.normal(0, vel_std0, 3)

        kf_rw_inner = LinearKF.initialize(p_init, v_init, pos_std=pos_std0, vel_std=vel_std0)
        rkf = RewindKF(kf=kf_rw_inner, horizon_s=HORIZON_S)
        kf_nv = LinearKF.initialize(p_init.copy(), v_init.copy(), pos_std=pos_std0, vel_std=vel_std0)

        # schedule of detector ticks within [0, t_cross]
        det_times = np.arange(0.0, t_cross + 1e-9, det_dt)
        # which ticks are accepted
        accepted = rng.random(det_times.shape[0]) < accept
        # pre-sample each accepted tick's residual + its capture-time truth & cov
        # We process events (IMU predicts + fix applications) in time order. A fix captured at
        # tick time tc is APPLIED at tc+L; we apply it at the first IMU step whose time >= tc+L.
        fix_apply_events = []   # (apply_time, capture_time, z, cov)
        for tc, ok in zip(det_times, accepted):
            if not ok:
                continue
            p_true_cap = truth_pos(tc, accel_alongtrack)
            rng_to_g4 = SEG_LEN - float(np.dot(p_true_cap - G3, SEG_HAT))
            band = band_for_range(max(0.0, rng_to_g4))
            pool = pools[band]
            resid = pool[rng.integers(0, pool.shape[0])].copy()
            if debias:
                resid = resid - means[band]
            z = p_true_cap + resid
            cov = fix_cov_real(p_true_cap, R_wb)
            apply_t = tc + L_s
            fix_apply_events.append([apply_t, tc, z, cov])
        fix_apply_events.sort(key=lambda e: e[0])
        ev_i = 0
        n_dropped = 0

        # --- march IMU forward, applying fixes when their apply_time is reached ---
        t = 0.0
        for k in range(1, n_imu + 1):
            t_prev = t
            t = k * IMU_DT
            if t > t_cross:
                # final partial step exactly to t_cross
                dt = t_cross - t_prev
                t = t_cross
            else:
                dt = IMU_DT
            if dt <= 0:
                # apply any remaining fixes whose apply time <= t_cross, then stop
                pass
            else:
                # body specific force for this step: a_world (truth) measured w/ noise.
                a_world = SEG_HAT * accel_alongtrack    # truth world accel (0 for const-v)
                sf_world = a_world - g                  # specific force in world = a_world - g
                accel_body_true = R_wb.T @ sf_world
                accel_body_meas = accel_body_true + rng.normal(0, kf_rw_inner.accel_noise_std, 3)
                t_ns = int(round(t * 1e9))
                rkf.predict(accel_body_meas, R_wb, dt, sim_time_ns=t_ns)
                kf_nv.predict(accel_body_meas, R_wb, dt)

            # apply fixes whose apply_time has now passed
            while ev_i < len(fix_apply_events) and fix_apply_events[ev_i][0] <= t + 1e-12:
                apply_t, cap_t, z, cov = fix_apply_events[ev_i]
                t_fix_ns = int(round(cap_t * 1e9))
                res = rkf.update_position_at(t_fix_ns, z, cov)
                if not res.applied:
                    n_dropped += 1
                # naive arm: apply at CURRENT state (no rewind)
                kf_nv.update_position(z, cov)
                ev_i += 1

            if t >= t_cross:
                break

        # any fixes not yet applied (apply_time > t_cross) are simply not used (they arrive
        # after the crossing) -- correct: the crossing estimate can only use info available by
        # then. They are NOT dropped-as-error.

        p_true_cross = truth_pos(t_cross, accel_alongtrack)
        e_rw = rkf.position - p_true_cross
        e_nv = kf_nv.position - p_true_cross
        err_rw.append(e_rw)
        err_nv.append(e_nv)
        P_rw_diag.append(np.diag(rkf.P)[:3].copy())
        P_nv_diag.append(np.diag(kf_nv.P)[:3].copy())
        # NEES over the 3-DOF position block
        Ppos_rw = rkf.P[:3, :3]
        Ppos_nv = kf_nv.P[:3, :3]
        nees_rw.append(float(e_rw @ np.linalg.solve(Ppos_rw, e_rw)))
        nees_nv.append(float(e_nv @ np.linalg.solve(Ppos_nv, e_nv)))
        dropped_counts.append(n_dropped)

    err_rw = np.asarray(err_rw)
    err_nv = np.asarray(err_nv)
    P_rw_diag = np.asarray(P_rw_diag)
    P_nv_diag = np.asarray(P_nv_diag)

    def arm_stats(err, Pd, nees):
        bias = err.mean(axis=0)
        rms = np.sqrt((err ** 2).mean(axis=0))            # RMS error per axis (incl bias)
        std = err.std(axis=0, ddof=1)                      # pure scatter
        claimed_sigma = np.sqrt(Pd.mean(axis=0))           # filter claimed 1-sigma per axis
        inplane_miss = np.sqrt(err[:, 1] ** 2 + err[:, 2] ** 2)
        return dict(
            bias=bias.tolist(),
            rms=rms.tolist(),
            std=std.tolist(),
            claimed_sigma=claimed_sigma.tolist(),
            inplane_rms=float(np.sqrt((inplane_miss ** 2).mean())),
            inplane_claimed_sigma=float(np.sqrt(claimed_sigma[1] ** 2 + claimed_sigma[2] ** 2)),
            inplane_miss_mean=float(inplane_miss.mean()),
            inplane_miss_p90=float(np.percentile(inplane_miss, 90)),
            nees_mean=float(np.mean(nees)),
        )

    out = dict(
        L_ms=L_ms, accept=accept, error_model=error_model,
        accel_alongtrack=accel_alongtrack, t_cross_s=float(t_cross),
        eff_fix_hz=float(DETECTOR_HZ * accept),
        n_seeds=n_seeds,
        mean_dropped_fixes=float(np.mean(dropped_counts)),
        rewind=arm_stats(err_rw, P_rw_diag, nees_rw),
        naive=arm_stats(err_nv, P_nv_diag, nees_nv),
    )
    # clears<0.05 m flags on the in-plane axes (rewind arm = the deployed path)
    rw = out["rewind"]
    out["clears_005_inplane_claimed_sigma"] = bool(
        rw["claimed_sigma"][1] < 0.05 and rw["claimed_sigma"][2] < 0.05)
    out["clears_005_inplane_rms"] = bool(rw["inplane_rms"] < 0.05)
    out["clears_005_E_rms"] = bool(rw["rms"][1] < 0.05)
    out["clears_005_D_rms"] = bool(rw["rms"][2] < 0.05)
    return out


def main():
    np.seterr(all="raise")
    pools, means, stds = load_residual_pools()

    meta = dict(
        geometry=dict(
            g3=G3.tolist(), g4=G4.tolist(), seg_len_m=SEG_LEN,
            seg_hat=SEG_HAT.tolist(), speed_mps=SPEED, yaw_rad=YAW_RAD,
            drag_hold_pitch_deg=float(np.rad2deg(DRAG_HOLD_PITCH_RAD)),
        ),
        imu_hz=IMU_HZ, detector_hz=DETECTOR_HZ, horizon_s=HORIZON_S,
        accel_noise_std=0.3, attitude_noise_std_rad=float(ATTITUDE_NOISE_STD_RAD),
        n_seeds=N_SEEDS, base_seed=BASE_SEED,
        residual_bands={k: dict(n=int(v.shape[0]), bias=means[k].tolist(),
                                std=stds[k].tolist()) for k, v in pools.items()},
        axis_labels=AX_LABEL,
        accept_cells=ACCEPT_CELLS, latency_cells_ms=LATENCY_CELLS_MS,
        error_models=ERROR_MODELS,
        notes=(
            "In-plane axes at gate-4 = E (lateral) + D (vertical); along-track = N. "
            "Binding in-plane miss = sqrt(E^2+D^2). Fix-cov via REAL "
            "localization.gate_pose_to_world_position (covariance=None -> 0.3m PnP default "
            "+ 1.4deg attitude lever + 0.40m floor). Residuals RESAMPLED from MEASURED off_ned "
            "(perception-char-2026-06-08), range-band conditioned on range_m [0,8)+[8,16)."
        ),
    )

    cells = []
    accel_cases = [("const_v", 0.0), ("mild_accel_5", 5.0)]   # baseline + mild accel sensitivity
    for accel_name, accel in accel_cases:
        for L_name, L_ms in LATENCY_CELLS_MS.items():
            for acc_name, acc in ACCEPT_CELLS.items():
                for em in ERROR_MODELS:
                    res = run_cell(L_ms, acc, em, accel, pools, means, N_SEEDS, BASE_SEED)
                    res["cell_id"] = f"{accel_name}|{L_name}|{acc_name}|{em}"
                    res["accel_case"] = accel_name
                    res["L_name"] = L_name
                    res["accept_name"] = acc_name
                    cells.append(res)
                    rw = res["rewind"]
                    print(f"{res['cell_id']:>44} | eff {res['eff_fix_hz']:4.1f}Hz | "
                          f"claimed sig E/D {rw['claimed_sigma'][1]:.3f}/{rw['claimed_sigma'][2]:.3f} | "
                          f"RMS E/D {rw['rms'][1]:.3f}/{rw['rms'][2]:.3f} | "
                          f"inplane RMS {rw['inplane_rms']:.3f} | NEES {rw['nees_mean']:.2f} | "
                          f"drop {res['mean_dropped_fixes']:.2f} | "
                          f"clrSig {int(res['clears_005_inplane_claimed_sigma'])} "
                          f"clrRMS {int(res['clears_005_inplane_rms'])}")

    out = dict(meta=meta, cells=cells)
    (HERE / "a1_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'a1_results.json'}  ({len(cells)} cells)")
    return out


if __name__ == "__main__":
    main()
