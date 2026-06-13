"""INDEPENDENT adversarial verification of piece C (range-anisotropic R).

This script does NOT import the prototype's RA/FIT helpers for the headline checks; it
re-derives the depth law from scratch, re-runs an independent Monte-Carlo, re-checks the
held-out split integrity (no leakage between train/test), and re-checks the leak gate with
an independent computation. Where it does reuse RA/FIT it is only to read the SAME data.
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

from racer.frames import (R_world_from_odo_quat_wxyz, R_camera_from_body,
                          CAMERA_INTRINSICS_K, ATTITUDE_NOISE_STD_RAD)
from racer.vision.gate_pose import project_gate_corners, estimate_gate_pose, GATE_INNER_SIZE_M
from racer.contracts import GateObservation
from racer.localization import FIX_COV_FLOOR_STD

CHAR_DIR = REPO / "handoff" / "perception-char-2026-06-08"
MAP_PATH = REPO / "handoff" / "shadowpc-firstcontact-2026-06-02" / "track_map.json"

print("=" * 78)
print("INDEPENDENT VERIFICATION — piece C")
print("=" * 78)

# -----------------------------------------------------------------------------
# (1) RE-DERIVE the depth law analytically and via a CLEAN independent MC.
# -----------------------------------------------------------------------------
f = float(CAMERA_INTRINSICS_K[0, 0])
s = float(GATE_INNER_SIZE_M)
sigpx = 1.5
N = 4
c2_derived = 2.0 * sigpx / (f * s * np.sqrt(N))     # std_depth = c2 r^2
a1_derived = sigpx / (f * np.sqrt(N))               # std_lat   = a1 r
print(f"\n[1] DERIVATION: f={f} s={s} sigma_px={sigpx} N={N}")
print(f"    c2 (depth std=c2 r^2) = {c2_derived:.6f}   a1 (lat std=a1 r) = {a1_derived:.6f}")
print(f"    prototype coeffs file c2 = {json.loads((HERE/'range_R_coeffs.json').read_text())['c2']}")

# CLEAN MC, fresh rng seed, MORE trials, MORE ranges incl. beyond 24 m to test extrapolation.
rng = np.random.default_rng(20260613)
rs = np.array([5, 8, 12, 16, 20, 25, 30, 40, 50], float)
depth_std = []
lat_std = []
for r in rs:
    base = project_gate_corners(np.eye(3), np.array([0.0, 0.0, r]))
    zs, xs = [], []
    for _ in range(2000):
        noisy = base + rng.normal(0, sigpx, base.shape)
        obs = GateObservation(frame_id=0, sim_time_ns=0, corners_px=noisy, gate_id=0,
                              corner_ids=np.array([0, 1, 2, 3]), corner_confidence=np.ones(4))
        p = estimate_gate_pose(obs, compute_covariance=False, weighted_refine=False)
        if p is not None:
            zs.append(p.t_cam_gate[2]); xs.append(p.t_cam_gate[0])
    depth_std.append(np.std(zs)); lat_std.append(np.std(xs))
depth_std = np.array(depth_std); lat_std = np.array(lat_std)
slope_d = np.polyfit(np.log(rs), np.log(depth_std), 1)[0]
slope_l = np.polyfit(np.log(rs), np.log(lat_std), 1)[0]
print(f"\n    INDEPENDENT MC (2000 trials/range, seed 20260613):")
print(f"    {'r':>4} {'depth_std':>10} {'c2_emp':>9} {'lat_std':>9} {'a1_emp':>9}")
for i, r in enumerate(rs):
    print(f"    {r:4.0f} {depth_std[i]:10.4f} {depth_std[i]/r**2:9.5f} {lat_std[i]:9.4f} {lat_std[i]/r:9.5f}")
print(f"    depth log-log slope = {slope_d:.3f} (theory 2.0)   lat slope = {slope_l:.3f} (theory 1.0)")
print(f"    --> depth r^4 law {'CONFIRMED' if abs(slope_d-2)<0.25 else 'FAILED'};"
      f" lat r^2 law {'CONFIRMED' if abs(slope_l-1)<0.30 else 'FAILED'}")
# Compare empirical c2 to derived
c2_emp = np.median(depth_std / rs**2)
print(f"    empirical c2 (median std/r^2) = {c2_emp:.5f} vs derived {c2_derived:.5f} "
      f"(ratio {c2_emp/c2_derived:.2f})")

# -----------------------------------------------------------------------------
# (2) HELD-OUT INTEGRITY: confirm the fit NEVER saw the test gates, and that the
#     coeffs on disk match a fresh fit on TRAIN only.
# -----------------------------------------------------------------------------
import fit_range_R as FIT
fixes = FIT.load_fixes()
good = [x for x in fixes if x["wfe"] < FIT.GOOD_FIX_MAX_M]
train = [x for x in good if x["source_gate"] in FIT.TRAIN_GATES]
test = [x for x in good if x["source_gate"] in FIT.TEST_GATES]
train_gates = set(x["source_gate"] for x in train)
test_gates = set(x["source_gate"] for x in test)
print(f"\n[2] HELD-OUT INTEGRITY:")
print(f"    train source gates = {sorted(train_gates)}  test source gates = {sorted(test_gates)}")
print(f"    overlap = {train_gates & test_gates}  (must be empty)")
# But note: associated gate_id may differ from source_gate. Check the ASSOCIATED gate ids too.
train_assoc = set(x["gate_id"] for x in train)
test_assoc = set(x["gate_id"] for x in test)
print(f"    train ASSOCIATED gate_ids = {sorted(train_assoc)}  test ASSOCIATED = {sorted(test_assoc)}")
print(f"    associated-id overlap = {train_assoc & test_assoc}  (LEAKAGE if nonempty!)")
# Re-fit and compare to disk
coeffs_fresh, _ = FIT.fit(train)
disk = json.loads((HERE / "range_R_coeffs.json").read_text())
match = all(abs(coeffs_fresh[k] - disk[k]) < 1e-9 for k in ("c2", "c1", "sig0_rad", "a1", "sig0_tan"))
print(f"    fresh fit matches disk coeffs: {match}")

# -----------------------------------------------------------------------------
# (3) INDEPENDENT NIS on held-out test for the live and new model — recompute by hand.
# -----------------------------------------------------------------------------
import range_anisotropic_R as RA

def build_recs():
    gmap = FIT.load_map()
    recs = []
    for g in range(6):
        p = CHAR_DIR / "pg" / f"course_g{g}" / "frames.json"
        fbid = {fr["frame_id"]: fr for fr in json.loads(p.read_text())["frames"]}
        d = json.loads((CHAR_DIR / f"characterize_g{g}.json").read_text())
        for r in d["rows"]:
            if not (r.get("associated") and r.get("pose_range_m") is not None):
                continue
            fr = fbid.get(r["frame_id"])
            if fr is None:
                continue
            gid = r["gate_id"]
            drone = np.asarray(fr["drone_position_ned"], float)
            off = np.asarray(r["off_ned"], float)
            R_wb = R_world_from_odo_quat_wxyz(fr["odo_q_wxyz"])
            R_wc = R_wb @ R_camera_from_body().T
            lever_world = gmap[gid] - (drone + off)
            t_cam = R_wc.T @ lever_world
            recs.append(dict(source_gate=g, gate_id=gid, off=off, R_wc=R_wc, t_cam=t_cam,
                             n_corners=r["n_corners"], reproj=r["reproj_px"], wfe=r["world_fix_err_m"],
                             pose_range=r["pose_range_m"], lever_true=gmap[gid]-drone,
                             range_err=r["range_err_m"], true_range=r["true_range_m"]))
    return recs

recs = build_recs()
test_recs = [r for r in recs if r["source_gate"] in FIT.TEST_GATES and r["wfe"] < 3.0]

def nis_new(rec):
    R = RA.R_aniso(rec["t_cam"], rec["R_wc"], reproj_px=rec["reproj"],
                   range_m=rec["pose_range"], n_corners=rec["n_corners"])
    return float(rec["off"] @ np.linalg.solve(R, rec["off"]))

nis_vals = np.array([nis_new(r) for r in test_recs])
print(f"\n[3] INDEPENDENT held-out NIS (NEW R_aniso): mean={nis_vals.mean():.3f} "
      f"median={np.median(nis_vals):.3f} N={len(nis_vals)} (report claims 2.84)")

# -----------------------------------------------------------------------------
# (4) LEAK-GATE attack: does R_aniso pass the SAME catastrophic fixes the live model
#     rejects? Re-derive leak counts independently, and check it does NOT just inflate
#     R to pass everything (which would defeat the gate).
# -----------------------------------------------------------------------------
CHI2 = 16.27
def leak(cov_fn):
    gacc=gtot=blk=btot=0
    for rec in recs:
        R = cov_fn(rec)
        d2 = float(rec["off"] @ np.linalg.solve(R, rec["off"]))
        if rec["wfe"] < 3.0:
            gtot += 1; gacc += int(d2 <= CHI2)
        else:
            btot += 1; blk += int(d2 <= CHI2)
    return gacc/gtot, blk/btot, blk, btot

def cov_new(rec):
    return RA.R_aniso(rec["t_cam"], rec["R_wc"], reproj_px=rec["reproj"],
                      range_m=rec["pose_range"], n_corners=rec["n_corners"])
def cov_base(rec):
    return RA.R_baseline_k2_lever_floor(rec["t_cam"], rec["R_wc"], range_m=rec["pose_range"],
                                        n_corners=rec["n_corners"])

ga_n, bl_n, blk_n, btot = leak(cov_new)
ga_b, bl_b, blk_b, _ = leak(cov_base)
print(f"\n[4] LEAK-GATE (independent): NEW accept={ga_n*100:.1f}% leak={bl_n*100:.1f}% ({blk_n}/{btot})"
      f"  | CURRENT accept={ga_b*100:.1f}% leak={bl_b*100:.1f}% ({blk_b}/{btot})")
# Is R_aniso bigger or smaller than baseline (det ratio)? If it just inflates everything,
# leak would rise AND accept would rise -- the "cheat". Compare trace of R over good fixes.
tr_new = np.mean([np.trace(cov_new(r)) for r in recs if r["wfe"]<3.0])
tr_base = np.mean([np.trace(cov_base(r)) for r in recs if r["wfe"]<3.0])
print(f"    mean trace(R) over good fixes: NEW={tr_new:.4f}  CURRENT={tr_base:.4f}  ratio={tr_new/tr_base:.3f}")

# -----------------------------------------------------------------------------
# (5) range_err channel grounding: is dump range_err_m really the LOS-radial error?
#     reconstruct radial component of off_ned along TRUE LOS and correlate.
# -----------------------------------------------------------------------------
rad_recon, range_err = [], []
for rec in recs:
    lv = rec["lever_true"]; nL = np.linalg.norm(lv)
    if nL < 1e-6: continue
    Lhat = lv/nL
    rad_recon.append(float(rec["off"] @ Lhat))
    range_err.append(rec["range_err"])
rad_recon = np.array(rad_recon); range_err = np.array(range_err)
corr = np.corrcoef(rad_recon, range_err)[0,1]
# sign-corrected residual
resid_same = np.median(np.abs(rad_recon - range_err))
resid_flip = np.median(np.abs(rad_recon + range_err))
print(f"\n[5] range_err channel grounding: corr(radial_recon, dump range_err) = {corr:.4f}")
print(f"    median|radial - range_err| = {resid_same:.4f} m ; median|radial + range_err| = {resid_flip:.4f} m")
print(f"    (report claims corr -0.9986, residual 3.4 cm under sign flip)")

# -----------------------------------------------------------------------------
# (6) The CENTRAL question: within VQ1 range, is depth Fisher term or floor dominant?
#     Re-derive the crossover where c2 r^2 = 0.40.
# -----------------------------------------------------------------------------
r_cross = np.sqrt(FIX_COV_FLOOR_STD / c2_derived)
print(f"\n[6] depth Fisher overtakes 0.40 floor at r = sqrt(0.40/c2) = {r_cross:.2f} m")
print(f"    -> within VQ1 (<=24 m) depth is floor-dominated until ~{r_cross:.0f} m. (report claims ~16 m)")

# Extrapolation table (eigen-std of R_aniso head-on)
print(f"\n    R_aniso depth-axis std vs iso-floor std (extrapolation, case-C value):")
for r in (10, 20, 24, 30, 40, 50):
    cov = RA.R_aniso(np.array([0,0,float(r)]), np.eye(3), range_m=float(r))
    depth = np.sqrt(np.max(np.linalg.eigvalsh(cov)))
    iso = FIX_COV_FLOOR_STD
    print(f"    r={r:3d}  R_aniso depth_std={depth:.3f}  iso-floor={iso:.3f}  ratio={depth/iso:.2f}")
print("\nDONE.")
