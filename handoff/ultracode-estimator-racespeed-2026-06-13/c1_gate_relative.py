"""c1 — GATE-RELATIVE observation branch (estimator-racespeed).

Mission slice: can a GATE-RELATIVE observation (track drone position RELATIVE to the upcoming
gate-4 via the PnP translation t_cam_gate, rotated to world) thread the gate-4 0.155 m in-plane
margin at the ~37 m/s post-gate-3 approach -- where the ABSOLUTE world-fix CANNOT (a1/a2/b1/b2:
filtered variance floor ~0.24 m, per-gate bias floor ~0.19-0.52 m, deployable ~0.55 m in-plane)?

The architectural claim under test (FACTS section 2, point 2 / project-estimator-robustness layer-3c):
the absolute world-fix is  p_drone = gate.position_ned - L  with  L = R_world_camera @ t_cam_gate.
Its error = (gate MAP error) + (PnP lever error). The in-plane miss the drone PHYSICALLY must hit is
its offset from the ACTUAL gate opening, i.e. the lateral part of  -L  in the gate plane. That carries
PnP noise (and attitude-lever noise) but NOT the map/registration bias. So a gate-relative observation
REMOVES the map-bias term for the binding in-plane miss.

This file:
  (a) Quantifies the achievable in-plane miss = the PnP-relative-translation LATERAL noise only, from
      the MEASURED data (t_cam_err_m decomposed into radial(=range_err) vs lateral), at the gate-4
      transit band, in-loop-accepted (maha<=16.27) fixes only.
  (b) Proves the bias-removal arithmetic: relative-state KF (track e = p_drone - gate_true) vs the
      'subtract gate map pos' anti-pattern (re-injects map bias). Runs the REAL LinearKF both ways.
  (c) Determinism-per-track: the relative observation has NO per-track map constant -> nothing to
      re-survey; determinism preserved trivially.
  (d) Visibility through the 37 m/s approach: camera up-tilt 20deg, VFoV 58.7deg, HFoV 90deg --
      does gate-4 stay in frame / how close before it clips to <4 corners (P3P).

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c1_gate_relative.py
(it also needs the case-C protos on the path; it inserts them itself.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

from racer.frames import (  # noqa: E402
    ATTITUDE_NOISE_STD_RAD,
    CAMERA_INTRINSICS_K,
    R_camera_from_body,
    R_world_from_body,
)
from racer.localization import FIX_COV_FLOOR_STD  # noqa: E402
from racer.state_estimator import LinearKF  # noqa: E402
from kf_rewind_buffer import RewindKF  # noqa: E402

SEED = 20260613
CHI2_GATE = 16.27  # navigator chi2 99.9% / 3 DOF
BAR = 0.05         # m, the <0.05 m 1-sigma variance target (= margin/3)
MARGIN_G4 = 0.155  # m, gate-4 contact-true in-plane margin @ r=0.38
V_RACE = 37.0      # m/s post-gate-3 cruise (ASSUMED, memory/plan)

# Gate-4 geometry (FACTS / track_map). Motion ~along -N, gate normal ~-N.
# => in-plane = (E lateral, D vertical); along-track = N.
G4_BOTTOM_NED = np.array([-135.5, -0.8, 25.36])
G3_BOTTOM_NED = np.array([-111.5, -5.1, 24.57])
G4_OPENING_NED = G4_BOTTOM_NED + np.array([0.0, 0.0, -1.36])  # opening ~1.36 m above bottom in -D


def load_rows(path):
    d = json.load(open(path))
    return [r for r in d["rows"] if r.get("associated") and "world_fix_err_m" in r]


def lateral_pnp_err(r):
    """Lateral (cross-LOS) PnP translation error magnitude from t_cam_err_m and range_err.

    t_cam_err_m = ||t_cam_solved - t_cam_pred|| (full 3D PnP-relative error, CAMERA frame, NO map
    error -- both terms use the SAME map gate pos). range_err_m = pose.range_m - true_range_m is
    the radial(depth) component along LOS. At small bearing (gate near optical axis) range~=depth,
    so lateral = sqrt(t_cam_err^2 - range_err^2). This is the gate-relative in-plane noise."""
    tce = float(r["t_cam_err_m"])
    re = float(r["range_err_m"])
    return float(np.sqrt(max(tce * tce - re * re, 0.0)))


# ======================================================================================
# (a) ACHIEVABLE in-plane miss = PnP-relative lateral noise at the gate-4 transit band
# ======================================================================================
def part_a():
    out = {}
    # Use BOTH the per-gate g4 bundle (60 frames, dense near g4) and the course_60s gate-4 subset.
    g4 = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_g4.json")
    g4 = [r for r in g4 if r.get("gate_id") == 4]
    course = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_course_60s.json")
    course_g4 = [r for r in course if r.get("gate_id") == 4]

    def analyze(rows, label):
        # in-loop-accepted set (the fixes the navigator would actually use): maha<=CHI2_GATE,
        # 4-corner. These are the only fixes a gate-relative update would consume too.
        acc = [r for r in rows if np.isfinite(r["maha"]) and r["maha"] <= CHI2_GATE
               and r["n_corners"] == 4]
        res = {"label": label, "n_all": len(rows), "n_accepted": len(acc), "bands": {}}
        for lo, hi in [(0, 6), (6, 9), (9, 12), (0, 12), (0, 16)]:
            band = [r for r in acc if lo <= r["true_range_m"] < hi]
            if not band:
                continue
            lat = np.array([lateral_pnp_err(r) for r in band])
            rad = np.abs([r["range_err_m"] for r in band])
            wfix = np.array([r["world_fix_err_m"] for r in band])  # ABSOLUTE (for contrast)
            res["bands"][f"[{lo},{hi})"] = dict(
                n=len(band),
                lat_pnp_rms=float(np.sqrt(np.mean(lat**2))),
                lat_pnp_mean=float(np.mean(lat)),
                lat_pnp_std=float(np.std(lat)),
                lat_pnp_p50=float(np.percentile(lat, 50)),
                lat_pnp_p90=float(np.percentile(lat, 90)),
                radial_pnp_rms=float(np.sqrt(np.mean(rad**2))),
                abs_worldfix_rms=float(np.sqrt(np.mean(wfix**2))),
            )
        return res

    out["g4_bundle"] = analyze(g4, "characterize_g4 (dense)")
    out["course_g4"] = analyze(course_g4, "course_60s gate-4 subset")
    return out


# ======================================================================================
# (b) BIAS-REMOVAL PROOF: relative-state KF vs 'subtract map pos' anti-pattern.
#     Runs the REAL LinearKF. Truth: drone flies straight g3->g4 at 37 m/s, perfectly centered
#     (true in-plane offset from gate-4 opening = 0). We inject:
#       - a per-track MAP/registration BIAS on gate-4 (the un-filterable absolute term, b2 ~0.19 m
#         in-plane at the near band, a2 ~0.45 m at full window) as a CONSTANT,
#       - per-fix PnP LATERAL noise (zero-mean) resampled from the measured accepted set.
#     Three observation models, all consuming the SAME sightings:
#       (A) ABSOLUTE world-fix: z = p_true + map_bias + noise  -> KF tracks p_true+map_bias.
#       (B) GATE-RELATIVE via 'subtract gate MAP pos': e = z_world - gate_map = (p_true-gate_map)
#           + map_bias + noise. The drone-relative-to-gate-OPENING error STILL carries map_bias
#           because gate_map != gate_true. (anti-pattern; re-injects the bias.)
#       (C) TRUE gate-relative: observe the lever directly. e_obs = -(L_est) measured rel to the
#           ACTUAL seen opening; truth e_true = p_true - gate_true. map_bias DROPS OUT (the seen
#           gate corners ARE the true opening). KF tracks the true relative offset.
#     We measure the IN-PLANE (E,D) error of each at the gate-4 plane crossing.
# ======================================================================================
def part_b(part_a_out):
    rng = np.random.default_rng(SEED)
    # per-fix lateral PnP noise pool: accepted gate-4 fixes, decomposed lateral, near band.
    g4 = load_rows(_REPO / "handoff/perception-char-2026-06-08/characterize_g4.json")
    g4 = [r for r in g4 if r.get("gate_id") == 4 and np.isfinite(r["maha"])
          and r["maha"] <= CHI2_GATE and r["n_corners"] == 4 and r["true_range_m"] < 12.0]
    # Build a per-axis (E,D) lateral noise model. The decomposition gives magnitude; we split it
    # isotropically in the perp plane (the PnP lateral law is ~isotropic across the two tangential
    # axes). lat_rms is the per-fix lateral MAGNITUDE; per-axis sigma = lat_rms/sqrt(2).
    lat = np.array([lateral_pnp_err(r) for r in g4])
    lat_rms = float(np.sqrt(np.mean(lat**2)))
    per_axis_sigma = lat_rms / np.sqrt(2.0)

    # absolute per-fix model from a1/b1: per-fix world-fix per-axis sigma ~0.50 m near gate.
    abs_axis_sigma = 0.50
    # per-track gate-4 MAP bias (the un-filterable absolute term). Use b2 near-band in-plane 0.19 m,
    # split E-dominant per a2 (E carries it): E=0.18, D=0.06 -> |inplane|~0.19.
    map_bias_ED = np.array([0.18, 0.06])  # (E, D) per-track constant, gate-4

    # trajectory: straight g3->g4 at 37 m/s, perfectly centered (true in-plane offset = 0).
    dt_imu = 1.0 / 90.0
    seg = G4_BOTTOM_NED - G3_BOTTOM_NED
    L_seg = float(np.linalg.norm(seg))
    uhat = seg / L_seg
    T = L_seg / V_RACE
    n_steps = int(T / dt_imu)
    # fixes at 30 Hz * 47% acceptance ~ 14 Hz over the approach; only those within ~12 m of g4.
    fix_hz = 14.0
    fix_dt = 1.0 / fix_hz

    # attitude: drag-hold along -N. roll0, NOSE-DOWN pitch ~-38.4deg (forward thrust vectoring to
    # counter drag at 37 m/s; sign verified: this is the posture that keeps gate-4 at bearing ~16deg,
    # consistent with the measured bearing ~12deg), yaw along -N (atan2(E,N) of uhat).
    yaw = float(np.arctan2(uhat[1], uhat[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))
    R_wb = R_world_from_body(0.0, pitch, yaw)
    g = np.array([0.0, 0.0, 9.80665])
    # specific force for constant velocity (a_world=0): f_world = -g; accel_body = R_wb.T @ (-g)
    accel_body = R_wb.T @ (-g)

    N_MC = 600
    res = {"per_axis_lat_sigma_m": per_axis_sigma, "lat_rms_m": lat_rms,
           "map_bias_ED_m": map_bias_ED.tolist(), "abs_axis_sigma_m": abs_axis_sigma,
           "n_mc": N_MC, "v_race": V_RACE, "seg_len_m": L_seg, "transit_s": T}

    def run_arm(arm, n_mc=N_MC):
        inplane_errs = []
        nees_list = []
        for s in range(n_mc):
            r = np.random.default_rng(SEED + 1000 * {"abs": 1, "submap": 2, "rel": 3}[arm] + s)
            p0 = G3_BOTTOM_NED.copy()
            v0 = uhat * V_RACE
            kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
            rk = RewindKF(kf=kf, horizon_s=0.5)
            t_ns = 0
            next_fix_t = 0.0
            t = 0.0
            for k in range(n_steps):
                t += dt_imu
                t_ns += int(dt_imu * 1e9)
                rk.predict(accel_body, R_wb, dt_imu, t_ns)
                p_true = p0 + uhat * V_RACE * t
                rng_to_g4 = float(np.linalg.norm(G4_BOTTOM_NED - p_true))
                if t >= next_fix_t and rng_to_g4 < 12.0:
                    next_fix_t += fix_dt
                    # per-fix lateral noise in (E,D) plane
                    nE = r.normal(0, per_axis_sigma)
                    nD = r.normal(0, per_axis_sigma)
                    if arm == "abs":
                        # absolute world-fix: bigger isotropic noise + map bias, full 3D
                        nA = r.normal(0, abs_axis_sigma, 3)
                        z = p_true + nA
                        z[1] += map_bias_ED[0]; z[2] += map_bias_ED[1]
                        cov = (abs_axis_sigma**2) * np.eye(3) + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                        rk.update_position(z, cov, sim_time_ns=t_ns)
                    elif arm == "submap":
                        # subtract MAP pos then add back map pos == identical to abs in absolute frame
                        # (proves the anti-pattern: e_obs uses gate_map; bias survives). We model the
                        # SAME measurement but with the SMALLER PnP lateral noise on E,D (since the
                        # relative obs IS lower-noise) but the map bias STILL present (gate_map wrong).
                        z = p_true.copy()
                        z[1] += map_bias_ED[0] + nE
                        z[2] += map_bias_ED[1] + nD
                        # along-track (N) carries radial PnP noise; small contribution at gate-4
                        z[0] += r.normal(0, abs_axis_sigma)
                        cov = np.diag([abs_axis_sigma**2, per_axis_sigma**2, per_axis_sigma**2]) \
                            + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                        rk.update_position(z, cov, sim_time_ns=t_ns)
                    else:  # rel
                        # TRUE gate-relative: observe e = p - gate_true (map bias drops out). Implement
                        # as a position pseudo-fix z = p_true + zero-mean lateral noise ONLY (no bias).
                        # NO 0.40 m bias-absorption floor needed (no bias to absorb) -> tighter R.
                        z = p_true.copy()
                        z[1] += nE
                        z[2] += nD
                        z[0] += r.normal(0, abs_axis_sigma)  # along-track radial PnP noise
                        # tight R: pure PnP lateral, NO floor (the floor was bias-absorption)
                        cov = np.diag([abs_axis_sigma**2, per_axis_sigma**2, per_axis_sigma**2])
                        rk.update_position(z, cov, sim_time_ns=t_ns)
            # at end (~at gate-4 plane), in-plane error = (E,D) of estimate vs true
            p_true_final = p0 + uhat * V_RACE * (n_steps * dt_imu)
            err = rk.position - p_true_final
            ip = np.array([err[1], err[2]])  # (E,D)
            inplane_errs.append(float(np.linalg.norm(ip)))
            P = rk.P[:3, :3]
            try:
                nees = float(err @ np.linalg.solve(P, err))
            except np.linalg.LinAlgError:
                nees = np.nan
            nees_list.append(nees)
        ip = np.array(inplane_errs)
        # decompose mean (bias) vs std (variance) of the in-plane error vector per axis
        return dict(
            inplane_rms=float(np.sqrt(np.mean(ip**2))),
            inplane_p50=float(np.percentile(ip, 50)),
            inplane_p90=float(np.percentile(ip, 90)),
            nees_mean=float(np.nanmean(nees_list)),
        )

    # per-axis bias/variance: re-run capturing signed E,D
    def run_arm_signed(arm, n_mc=N_MC):
        E_errs, D_errs = [], []
        for s in range(n_mc):
            r = np.random.default_rng(SEED + 7000 * {"abs": 1, "submap": 2, "rel": 3}[arm] + s)
            p0 = G3_BOTTOM_NED.copy()
            v0 = uhat * V_RACE
            kf = LinearKF.initialize(p0 + r.normal(0, 0.3, 3), v0, pos_std=0.5, vel_std=0.5)
            rk = RewindKF(kf=kf, horizon_s=0.5)
            t_ns = 0; next_fix_t = 0.0; t = 0.0
            for k in range(n_steps):
                t += dt_imu; t_ns += int(dt_imu * 1e9)
                rk.predict(accel_body, R_wb, dt_imu, t_ns)
                p_true = p0 + uhat * V_RACE * t
                rng_to_g4 = float(np.linalg.norm(G4_BOTTOM_NED - p_true))
                if t >= next_fix_t and rng_to_g4 < 12.0:
                    next_fix_t += fix_dt
                    nE = r.normal(0, per_axis_sigma); nD = r.normal(0, per_axis_sigma)
                    if arm == "abs":
                        nA = r.normal(0, abs_axis_sigma, 3)
                        z = p_true + nA; z[1] += map_bias_ED[0]; z[2] += map_bias_ED[1]
                        cov = (abs_axis_sigma**2 + FIX_COV_FLOOR_STD**2) * np.eye(3)
                        rk.update_position(z, cov, sim_time_ns=t_ns)
                    elif arm == "submap":
                        z = p_true.copy(); z[1] += map_bias_ED[0] + nE; z[2] += map_bias_ED[1] + nD
                        z[0] += r.normal(0, abs_axis_sigma)
                        cov = np.diag([abs_axis_sigma**2, per_axis_sigma**2, per_axis_sigma**2]) \
                            + (FIX_COV_FLOOR_STD**2) * np.eye(3)
                        rk.update_position(z, cov, sim_time_ns=t_ns)
                    else:
                        z = p_true.copy(); z[1] += nE; z[2] += nD; z[0] += r.normal(0, abs_axis_sigma)
                        cov = np.diag([abs_axis_sigma**2, per_axis_sigma**2, per_axis_sigma**2])
                        rk.update_position(z, cov, sim_time_ns=t_ns)
            p_true_final = p0 + uhat * V_RACE * (n_steps * dt_imu)
            err = rk.position - p_true_final
            E_errs.append(float(err[1])); D_errs.append(float(err[2]))
        E = np.array(E_errs); D = np.array(D_errs)
        return dict(E_bias=float(E.mean()), E_std=float(E.std()),
                    D_bias=float(D.mean()), D_std=float(D.std()))

    for arm in ("abs", "submap", "rel"):
        res[arm] = run_arm(arm)
        res[arm].update(run_arm_signed(arm))
    return res


# ======================================================================================
# (d) VISIBILITY through the 37 m/s approach. Camera tilts up 20deg; VFoV 58.7, HFoV 90.
#     Does gate-4 stay in frame, and at what range does it clip to <4 corners (P3P only)?
# ======================================================================================
def part_d():
    K = CAMERA_INTRINSICS_K
    W, H = 640.0, 360.0
    R_cb = R_camera_from_body()
    yaw = float(np.arctan2((G4_BOTTOM_NED - G3_BOTTOM_NED)[1], (G4_BOTTOM_NED - G3_BOTTOM_NED)[0]))
    pitch = float(-np.arctan(0.21 * V_RACE / 9.80665))  # NOSE-DOWN drag-hold (verified sign)
    R_wb = R_world_from_body(0.0, pitch, yaw)
    R_wc = R_wb @ R_cb.T  # world<-camera

    seg = G4_BOTTOM_NED - G3_BOTTOM_NED
    uhat = seg / np.linalg.norm(seg)
    # gate-4 opening corners (square, side 1.5 m inner) in the gate plane. Gate normal ~-N;
    # plane spanned by E (lateral) and D (vertical). Centre at opening.
    half = 1.5 / 2.0
    centre = G4_OPENING_NED
    corners_w = np.array([
        centre + np.array([0, +half, +half]),
        centre + np.array([0, +half, -half]),
        centre + np.array([0, -half, -half]),
        centre + np.array([0, -half, +half]),
    ])
    out = {"camera": {"W": W, "H": H, "HFoV_deg": 90.0, "VFoV_deg": 58.7, "uptilt_deg": 20.0},
           "pitch_deg": float(np.degrees(pitch)), "yaw_deg": float(np.degrees(yaw)),
           "ranges": []}
    for rng_to_centre in [20, 16, 12, 10, 8, 6, 5, 4, 3, 2]:
        # drone position = centre - uhat*rng (approaching along uhat from g3 side); but centre is the
        # opening; approach so the along-N distance ~ rng. Place drone at centre - uhat*rng.
        p_drone = centre - uhat * rng_to_centre
        in_frame = 0
        margins = []
        for c in corners_w:
            d_world = c - p_drone
            d_cam = R_wc.T @ d_world  # camera optical frame
            if d_cam[2] <= 0.05:
                margins.append(None)
                continue
            u = K[0, 0] * d_cam[0] / d_cam[2] + K[0, 2]
            v = K[1, 1] * d_cam[1] / d_cam[2] + K[1, 2]
            inside = bool((0 <= u <= W) and (0 <= v <= H))
            in_frame += int(inside)
            margins.append([round(float(u), 1), round(float(v), 1), inside])
        out["ranges"].append(dict(range_m=rng_to_centre, n_corners_in_frame=in_frame,
                                  corners_uv=margins))
    return out


def main():
    np.random.seed(SEED)
    results = {"seed": SEED, "bar_m": BAR, "margin_g4_m": MARGIN_G4, "v_race_mps": V_RACE}
    print("=" * 80)
    print("(a) ACHIEVABLE in-plane miss = PnP-relative LATERAL noise at gate-4 transit band")
    print("=" * 80)
    a = part_a()
    results["part_a"] = a
    for key in ("g4_bundle", "course_g4"):
        r = a[key]
        print(f"\n{r['label']}: accepted(maha<={CHI2_GATE},4-corner)={r['n_accepted']}/{r['n_all']}")
        print(f"  {'band':>10} {'n':>3} {'lat_rms':>8} {'lat_p50':>8} {'lat_p90':>8} "
              f"{'radial':>8} {'ABS_wfix':>9}")
        for band, b in r["bands"].items():
            print(f"  {band:>10} {b['n']:3d} {b['lat_pnp_rms']:8.3f} {b['lat_pnp_p50']:8.3f} "
                  f"{b['lat_pnp_p90']:8.3f} {b['radial_pnp_rms']:8.3f} {b['abs_worldfix_rms']:9.3f}")

    print("\n" + "=" * 80)
    print("(b) BIAS-REMOVAL PROOF: abs vs subtract-map (anti-pattern) vs TRUE gate-relative")
    print("=" * 80)
    b = part_b(a)
    results["part_b"] = b
    print(f"\n per-fix lateral PnP per-axis sigma = {b['per_axis_lat_sigma_m']:.3f} m "
          f"(lat_rms {b['lat_rms_m']:.3f}); map_bias(E,D)={b['map_bias_ED_m']} m; "
          f"abs per-axis sigma={b['abs_axis_sigma_m']:.2f} m")
    print(f" trajectory: straight g3->g4 {b['seg_len_m']:.1f} m @ {b['v_race']:.0f} m/s, "
          f"transit {b['transit_s']:.2f} s, fixes 14 Hz within 12 m, N_MC={b['n_mc']}")
    print(f"\n {'arm':>8} {'inplane_rms':>11} {'p50':>7} {'p90':>7} {'E_bias':>7} {'E_std':>6} "
          f"{'D_bias':>7} {'D_std':>6} {'NEES':>6} {'clears_0.05?':>12} {'<margin?':>9}")
    for arm in ("abs", "submap", "rel"):
        x = b[arm]
        clears = "YES" if x["inplane_rms"] < BAR else "NO"
        marg = "YES" if x["inplane_rms"] < MARGIN_G4 else "NO"
        print(f" {arm:>8} {x['inplane_rms']:11.3f} {x['inplane_p50']:7.3f} {x['inplane_p90']:7.3f} "
              f"{x['E_bias']:+7.3f} {x['E_std']:6.3f} {x['D_bias']:+7.3f} {x['D_std']:6.3f} "
              f"{x['nees_mean']:6.2f} {clears:>12} {marg:>9}")

    print("\n" + "=" * 80)
    print("(d) VISIBILITY through 37 m/s approach (corners in 640x360 frame)")
    print("=" * 80)
    d = part_d()
    results["part_d"] = d
    print(f" camera up-tilt {d['camera']['uptilt_deg']}deg, VFoV {d['camera']['VFoV_deg']}deg, "
          f"HFoV {d['camera']['HFoV_deg']}deg; cruise pitch {d['pitch_deg']:.1f}deg")
    print(f" {'range_m':>8} {'corners_in_frame':>16}")
    for rr in d["ranges"]:
        print(f" {rr['range_m']:8d} {rr['n_corners_in_frame']:16d}")

    outpath = Path(__file__).resolve().parent / "c1_gate_relative_results.json"
    outpath.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
