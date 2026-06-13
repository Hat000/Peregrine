"""Validate the range-anisotropic R against the CURRENT (K2 + lever + isotropic-floor) model on a
HELD-OUT split of the perception-char fixes, and run the two adversarial self-checks.

CALIBRATION METRIC: Normalized Innovation Squared (NIS) / whitened residuals. For a well-calibrated
covariance R, the fix error e = off_ned is distributed N(0, R), so the whitened residual w = R^{-1/2} e
is ~N(0, I): each axis has mean-square ~1, and the 3-DOF NIS = e^T R^{-1} e has mean ~3 (chi^2_3).
- NIS mean >> 3  => R is TOO TIGHT (over-confident) -> the chi2 gate over-rejects good fixes.
- NIS mean << 3  => R is TOO LOOSE (under-confident) -> the filter under-weights vision.
A model whose held-out NIS mean is closer to 3 AND whose per-axis whitened variances are closer to 1
is better calibrated.

We compute NIS in the LOS-aligned frame too (radial / 2 tangential whitened components) to expose WHICH
axis each model mis-calibrates -- the whole point of the anisotropic build.

ACCEPTANCE/LEAK CHECK: the live navigator gates a fix on maha = nu^T S^-1 nu < chi2_3,0.999 = 16.27,
where S = P_pos + R (P_pos is the KF position covariance). Here we approximate S ~ R (the fix-vs-truth
innovation, P_pos small under given-pose VQ1) and report, on ALL pooled fixes (good + catastrophic),
the fraction of GOOD fixes ACCEPTED (should stay ~ the 47% race-window / not collapse) and the fraction
of CATASTROPHIC (|off|>=3 m) fixes that LEAK through (should stay ~ the 0.53-1.6% bound, not blow up).

ADVERSARIAL SELF-CHECKS:
  (1) MC r^4 depth law: synthesize square corners via project_gate_corners at r in {5,10,15,20,25,30},
      add sigma_px pixel noise, run estimate_gate_pose, measure depth variance vs range -> confirm
      std(depth) ~ r^2 (var ~ r^4) and matches the derived coefficient.
  (2) Overfit guard: report HELD-OUT (test gates 1,3,5) NIS, never train; also report a train-vs-test
      gap so any overfit is visible.

Run (after fit_range_R.py):
    .venv/Scripts/python.exe handoff/ultracode-vision-case-c-2026-06-13/validate_R.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from racer.frames import R_world_from_odo_quat_wxyz, R_camera_from_body  # noqa: E402
from racer.vision.gate_pose import project_gate_corners, estimate_gate_pose, GATE_INNER_SIZE_M  # noqa: E402
from racer.frames import CAMERA_INTRINSICS_K  # noqa: E402
from racer.contracts import GateObservation  # noqa: E402

import range_anisotropic_R as RA  # noqa: E402
import fit_range_R as FIT  # noqa: E402

CHI2_GATE = 16.27   # chi2_3,0.999, the live navigator vision_gate_chi2


# ---------------------------------------------------------------------------
# Build the per-fix R for each model. We need R_wc per fix. We reconstruct it from the per-gate
# frame bundle's raw ODOMETRY quaternion (the TRUE attitude via R_world_from_odo_quat_wxyz) and the
# camera mount -- exactly what the navigator does live. t_cam_gate is reconstructed from the SOLVED
# lever (gate_pos - world_fix) expressed back in camera frame.
# ---------------------------------------------------------------------------
def _frames_pergate(g: int) -> dict[int, dict]:
    p = FIT.CHAR_DIR / "pg" / f"course_g{g}" / "frames.json"
    return {f["frame_id"]: f for f in json.loads(p.read_text())["frames"]}


def build_fix_records() -> list[dict]:
    """Each good/bad fix joined to R_wc + the SOLVED camera-frame lever t_cam_gate, plus the world
    fix error off_ned (the innovation vs truth) and its radial/tangential split."""
    gmap = FIT.load_map()
    recs: list[dict] = []
    for g in range(6):
        fbid = _frames_pergate(g)
        d = json.loads((FIT.CHAR_DIR / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("pose_range_m") is not None):
                continue
            f = fbid.get(r["frame_id"])
            if f is None:
                continue
            gid = r["gate_id"]
            drone = np.asarray(f["drone_position_ned"], float)
            off = np.asarray(r["off_ned"], float)
            R_wb = R_world_from_odo_quat_wxyz(f["odo_q_wxyz"])
            R_wc = R_wb @ R_camera_from_body().T
            # world fix = drone + off ; solved lever (world) = gate - fix ; t_cam = R_wc^T @ lever_world
            lever_world = gmap[gid] - (drone + off)
            t_cam = R_wc.T @ lever_world
            recs.append(dict(
                source_gate=g, gate_id=gid, frame_id=r["frame_id"],
                true_range=r["true_range_m"], pose_range=r["pose_range_m"],
                off=off, R_wc=R_wc, t_cam=t_cam, n_corners=r["n_corners"],
                reproj=r["reproj_px"], wfe=r["world_fix_err_m"],
                lever_true=gmap[gid] - drone,
            ))
    return recs


def _whiten_stats(recs: list[dict], cov_fn) -> dict:
    """NIS (3-DOF) + per-axis whitened variance (world NED) + LOS-frame (radial/tangential) whitened
    variance, over the given records using cov_fn(rec) -> 3x3 R."""
    nis = []
    w_world = []      # whitened world-NED residual per fix (3,)
    w_los = []        # whitened LOS-frame residual (radial, tang1, tang2)
    for rec in recs:
        R = cov_fn(rec)
        e = rec["off"]
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            continue
        nis.append(float(e @ Rinv @ e))
        # world whitened: L = chol(R); w = L^{-1} e
        try:
            L = np.linalg.cholesky(R)
            w = np.linalg.solve(L, e)
            w_world.append(w)
        except np.linalg.LinAlgError:
            pass
        # LOS frame: build orthonormal [Lhat, t1, t2] from the TRUE lever, rotate R into it, whiten
        lv = rec["lever_true"]; nL = np.linalg.norm(lv)
        if nL > 1e-9:
            Lhat = lv / nL
            a = np.array([1.0, 0, 0]) if abs(Lhat[0]) < 0.9 else np.array([0, 1.0, 0])
            t1 = np.cross(Lhat, a); t1 /= np.linalg.norm(t1)
            t2 = np.cross(Lhat, t1)
            B = np.column_stack([Lhat, t1, t2])     # world<-LOS
            R_los = B.T @ R @ B
            e_los = B.T @ e
            try:
                Ll = np.linalg.cholesky(R_los)
                w_los.append(np.linalg.solve(Ll, e_los))
            except np.linalg.LinAlgError:
                pass
    nis = np.array(nis)
    w_world = np.array(w_world) if w_world else np.zeros((0, 3))
    w_los = np.array(w_los) if w_los else np.zeros((0, 3))
    return dict(
        n=len(nis), nis_mean=float(nis.mean()) if nis.size else float("nan"),
        nis_median=float(np.median(nis)) if nis.size else float("nan"),
        w_world_var=w_world.var(axis=0).tolist() if w_world.size else [float("nan")] * 3,
        w_los_var=w_los.var(axis=0).tolist() if w_los.size else [float("nan")] * 3,  # [radial,t1,t2]
        nis=nis,
    )


def _cov_new(rec: dict) -> np.ndarray:
    return RA.R_aniso(rec["t_cam"], rec["R_wc"], reproj_px=rec["reproj"],
                      range_m=rec["pose_range"], n_corners=rec["n_corners"])


def _cov_baseline(rec: dict) -> np.ndarray:
    return RA.R_baseline_k2_lever_floor(rec["t_cam"], rec["R_wc"], range_m=rec["pose_range"],
                                        n_corners=rec["n_corners"])


def _cov_live_analytic(rec: dict) -> np.ndarray:
    """The ACTUAL shipped model: localization.gate_pose_to_world_position's cov, using the live
    analytic PnP Fisher block. The detector's true corner pixels are not in the dumps, so we
    reconstruct the analytic block's GEOMETRIC SHAPE: project the SOLVED pose to noise-free corners,
    run estimate_gate_pose(compute_covariance=True) to get the analytic [t,rvec] Fisher cov, then run
    it through the live localization formula (K2 inflation + lever + 0.40 floor). This is the fairest
    'what ships today' baseline -- it carries the live code's actual radial/tangential shape, only the
    pixel-noise SCALE (WEIGHTED_SIGMA_PX) is nominal rather than the true per-frame detector noise."""
    from racer.localization import gate_pose_to_world_position
    from racer.contracts import Gate, GatePose
    t = rec["t_cam"]; R_wc = rec["R_wc"]
    r = float(np.linalg.norm(t))
    Rcg = np.eye(3)  # frontal proxy: shape is dominated by depth/range, not the small gate rotation
    try:
        corners = project_gate_corners(Rcg, t)
    except ValueError:
        return _cov_baseline(rec)
    obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=corners, gate_id=rec["gate_id"],
                          corner_ids=np.array([0, 1, 2, 3]), corner_confidence=np.ones(4))
    pose = estimate_gate_pose(obs, compute_covariance=True, corner_sigma_px=1.5)
    if pose is None or pose.covariance is None:
        return _cov_baseline(rec)
    # R_world_body for the live formula: R_wc = R_wb @ R_cam_from_body().T -> R_wb = R_wc @ R_cam_from_body()
    R_wb = R_wc @ R_camera_from_body()
    gate = Gate(gate_id=rec["gate_id"], position_ned=np.zeros(3), R_world_gate=np.eye(3),
                inner_size_m=GATE_INNER_SIZE_M)
    _, cov = gate_pose_to_world_position(pose, gate, R_wb)
    if rec["n_corners"] < 4:
        cov = cov * 9.0
    return 0.5 * (cov + cov.T)


def _cov_live_isotropic(rec: dict) -> np.ndarray:
    """The SHIPPED model's actual behaviour: the analytic PnP block is replaced by an isotropic
    estimate (~the K2-inflated cov collapses to roughly isotropic + lever + floor in practice). We
    approximate it as the live floor (0.40) + lever only -- the 'no radial-shape' control."""
    t = rec["t_cam"]; R_wc = rec["R_wc"]; L = R_wc @ t; nL = np.linalg.norm(L)
    st = float(RA.ATTITUDE_NOISE_STD_RAD)
    cov = (RA.FIX_COV_FLOOR_STD ** 2) * np.eye(3)
    if nL > 1e-9:
        cov = cov + (st**2) * (nL * nL * np.eye(3) - np.outer(L, L))
    if rec["n_corners"] < 4:
        cov = cov * 9.0
    return 0.5 * (cov + cov.T)


def acceptance_leak(recs: list[dict], cov_fn) -> dict:
    """Fraction of GOOD (|off|<3) fixes accepted + CATASTROPHIC (|off|>=3) fixes that leak, under the
    chi2 gate with S ~ R (P_pos negligible under given-pose). This is the conservative (tightest-gate)
    view; live S = P_pos + R is looser so acceptance >= this."""
    good_acc = good_tot = bad_leak = bad_tot = 0
    for rec in recs:
        R = cov_fn(rec)
        try:
            d2 = float(rec["off"] @ np.linalg.solve(R, rec["off"]))
        except np.linalg.LinAlgError:
            continue
        if rec["wfe"] < 3.0:
            good_tot += 1
            good_acc += int(d2 <= CHI2_GATE)
        else:
            bad_tot += 1
            bad_leak += int(d2 <= CHI2_GATE)
    return dict(good_accept=good_acc / max(good_tot, 1), good_tot=good_tot,
                bad_leak=bad_leak / max(bad_tot, 1), bad_tot=bad_tot,
                bad_leak_n=bad_leak)


def mc_depth_law() -> None:
    """Adversarial check (1): direct Monte-Carlo of the PnP depth variance vs range -> r^4."""
    print("\n=== ADVERSARIAL CHECK 1: Monte-Carlo PnP depth law (should be var~r^4 / std~r^2) ===")
    rng = np.random.default_rng(0)
    K = CAMERA_INTRINSICS_K; f = K[0, 0]; s = GATE_INNER_SIZE_M; sigpx = 1.5
    c2_theory = RA._c2_from_pixels(sigpx)
    a1_theory = RA._a1_from_pixels(sigpx)
    print(f"  f={f} s={s} sigma_px={sigpx}  derived c2(std=c2 r^2)={c2_theory:.5f}  a1(std=a1 r)={a1_theory:.5f}")
    print(f"  {'r(m)':>5} {'depth_std':>10} {'lat_std':>9} {'c2_emp=std/r^2':>15} {'a1_emp=std/r':>13} {'log-slope':>10}")
    rs = [5, 10, 15, 20, 25, 30]
    ds_all = []
    for r in rs:
        base = project_gate_corners(np.eye(3), np.array([0, 0, float(r)]))
        ts = []
        for _ in range(500):
            noisy = base + rng.normal(0, sigpx, base.shape)
            obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=noisy, gate_id=0,
                                  corner_ids=np.array([0, 1, 2, 3]), corner_confidence=np.ones(4))
            p = estimate_gate_pose(obs, compute_covariance=False, weighted_refine=False)
            if p is not None:
                ts.append(p.t_cam_gate)
        ts = np.array(ts)
        ds, ls = ts[:, 2].std(), ts[:, 0].std()
        ds_all.append((r, ds, ls))
        print(f"  {r:5d} {ds:10.4f} {ls:9.4f} {ds / r**2:15.5f} {ls / r:13.5f}")
    rr = np.array([x[0] for x in ds_all]); dd = np.array([x[1] for x in ds_all])
    slope = np.polyfit(np.log(rr), np.log(dd), 1)[0]
    print(f"  --> depth-std log-log slope = {slope:.2f}  (theory 2.0; var slope = {2*slope:.2f}, theory 4.0)")
    pass_law = abs(slope - 2.0) < 0.25
    print(f"  --> r^4 DEPTH LAW {'CONFIRMED' if pass_law else 'NOT confirmed'} (|slope-2|<0.25)")


def main() -> None:
    recs = build_fix_records()
    good = [r for r in recs if r["wfe"] < 3.0]
    train = [r for r in good if r["source_gate"] in FIT.TRAIN_GATES]
    test = [r for r in good if r["source_gate"] in FIT.TEST_GATES]
    coeffs = RA.load_coeffs()
    print(f"loaded coeffs source: {coeffs['source']}")
    print(f"pooled={len(recs)} good={len(good)} train={len(train)} test(HELD-OUT g{sorted(FIT.TEST_GATES)})={len(test)}")

    print("\n=== NIS / WHITENED-RESIDUAL CALIBRATION (target: NIS mean ~ 3.0; per-axis whitened var ~ 1.0) ===")
    print(f"{'set':<22}{'model':<26}{'N':>4}{'NIS_mean':>10}{'NIS_med':>9}{'w_world_var(N,E,D)':>26}{'w_los_var(rad,t,t)':>26}")
    for label, subset in (("HELD-OUT test g1,3,5", test), ("(ref) train g0,2,4", train), ("(ref) all good", good)):
        for mname, fn in (("LIVE analytic (ships)", _cov_live_analytic),
                          ("CURRENT K2+lever+floor", _cov_baseline),
                          ("LIVE iso-floor+lever", _cov_live_isotropic),
                          ("NEW R_aniso (calib)", _cov_new)):
            s = _whiten_stats(subset, fn)
            ww = ",".join(f"{v:.2f}" for v in s["w_world_var"])
            wl = ",".join(f"{v:.2f}" for v in s["w_los_var"])
            print(f"{label:<22}{mname:<26}{s['n']:>4}{s['nis_mean']:>10.2f}{s['nis_median']:>9.2f}{ww:>26}{wl:>26}")
        print()

    print("=== ACCEPTANCE / LEAK under chi2_0.999=16.27 (S~R conservative; pooled good+catastrophic) ===")
    print(f"{'model':<26}{'good_accept':>12}{'good_N':>8}{'bad_leak':>10}{'bad_N':>7}{'leak_n':>8}")
    for mname, fn in (("LIVE analytic (ships)", _cov_live_analytic),
                      ("CURRENT K2+lever+floor", _cov_baseline),
                      ("LIVE iso-floor+lever", _cov_live_isotropic),
                      ("NEW R_aniso (calib)", _cov_new)):
        al = acceptance_leak(recs, fn)
        print(f"{mname:<26}{al['good_accept']*100:>11.1f}%{al['good_tot']:>8}{al['bad_leak']*100:>9.1f}%{al['bad_tot']:>7}{al['bad_leak_n']:>8}")

    mc_depth_law()

    # Adversarial check (2): overfit guard -- train vs test NIS gap for the NEW model.
    print("\n=== ADVERSARIAL CHECK 2: overfit guard (train vs HELD-OUT NIS gap for NEW model) ===")
    st_tr = _whiten_stats(train, _cov_new); st_te = _whiten_stats(test, _cov_new)
    print(f"  NEW R_aniso: train NIS mean {st_tr['nis_mean']:.2f} (N={st_tr['n']}) vs held-out {st_te['nis_mean']:.2f} (N={st_te['n']})")
    gap = abs(st_tr['nis_mean'] - st_te['nis_mean'])
    print(f"  --> |train-test| NIS gap = {gap:.2f}  ({'small, no overfit' if gap < 1.5 else 'LARGE -- inspect'})")


if __name__ == "__main__":
    main()
