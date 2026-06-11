"""Per-frame 3-DOF vision-chain attitude error from SOLVED PnP rotations (instrumented dumps).

The map gate orientation R_world_gate is exact (axis-aligned sim gates, quaternion verified
2026-06-04), and PnP solves R_cam_gate from the image alone. So per frame:

    R_wc_true  = R_world_gate @ R_solved.T          (what the renderer actually used)
    R_wc_model = R_wb(odo) @ R_camera_from_body().T (what the chain assumes)
    E          = R_pred @ R_solved.T  ==  R_wc_model.T @ R_wc_true   (camera-frame error)

rotvec(E) is the chain attitude error, independent of gate POSITION (no lever-arm division,
no map-centre contamination). PnP out-of-plane rotation noise is degrees-level per frame, so
systematics are extracted by regression over many frames, weighted by the per-frame analytic
rotation information (recomputed offline from the dumped corners + confidences).

Steps: stats -> roll/pitch regressions -> closed-form composition hypotheses (tilt-before-roll
etc.) fitted by least squares -> residual spread = the honest attitude noise (rotation channel).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from racer import frames as F                                  # noqa: E402
from racer.navigator import load_track_map                     # noqa: E402
from racer.vision.gate_pose import (                           # noqa: E402
    CONF_FLOOR,
    TUKEY_C_LO_PX,
    WEIGHTED_SIGMA_PX,
    gate_object_points,
)

HERE = Path(__file__).resolve().parent
MAP = ROOT / "handoff/shadowpc-firstcontact-2026-06-02/track_map.json"
_NO_DIST = np.zeros((4, 1))
SWAP = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
TAU = F.CAMERA_PITCH_RAD


def rodr(v):
    return cv2.Rodrigues(np.ascontiguousarray(v, dtype=np.float64))[0]


def rot_cov_rvec(corners, conf, rvec, tvec):
    """Recompute the analytic pose covariance (rvec block, camera frame) at the solved pose:
    same Fisher-information form as gate_pose._refine_pose's final iteration (Tukey c=lo)."""
    obj = gate_object_points().reshape(-1, 1, 3)
    proj, jac = cv2.projectPoints(obj, rvec.reshape(3, 1), tvec.reshape(3, 1),
                                  F.CAMERA_INTRINSICS_K, _NO_DIST)
    resid = proj.reshape(-1, 2) - corners
    u = np.linalg.norm(resid, axis=1) / TUKEY_C_LO_PX
    robust = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
    w_conf = 1.0 / (WEIGHTED_SIGMA_PX / np.clip(conf, CONF_FLOOR, 1.0)) ** 2
    J = np.asarray(jac, float)[:, 0:6]
    W = np.repeat(w_conf * robust, 2)
    try:
        cov = np.linalg.inv((J.T * W) @ J)
    except np.linalg.LinAlgError:
        return None
    return cov[0:3, 0:3]                                       # rvec block


def load_rows(pattern="before_g%d.json"):
    gates = {g.gate_id: g for g in load_track_map(MAP, corner_to_center=True)}
    rows = []
    for gi in range(6):
        p = HERE / (pattern % gi)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for r in d["rows"]:
            if not (r.get("associated") and "rvec_cam_gate" in r and r["n_corners"] == 4):
                continue
            R_solved = rodr(np.asarray(r["rvec_cam_gate"], float))
            R_pred = rodr(np.asarray(r["rvec_cam_gate_pred"], float))
            E = R_pred @ R_solved.T
            e_cam = Rotation.from_matrix(E).as_rotvec()
            roll, pitch, yaw = (np.radians(a) for a in r["rpy_deg"])
            corners = np.asarray(r["corners_px"], float)
            conf = (np.asarray(r["corner_conf"], float) if r.get("corner_conf") is not None
                    else np.ones(4))
            covr = rot_cov_rvec(corners, conf, np.asarray(r["rvec_cam_gate"], float),
                                np.asarray(r["t_cam_solved"], float))
            rows.append(dict(
                gate_id=r["gate_id"], err=r["world_fix_err_m"], range_m=r["true_range_m"],
                e_cam=e_cam, roll=roll, pitch=pitch, yaw=yaw,
                R_solved=R_solved, R_wg=gates[r["gate_id"]].R_world_gate,
                q=np.asarray(r["odo_q_wxyz"], float), cov_rr=covr,
                off=np.asarray(r["off_ned"], float),
                t_solved=np.asarray(r["t_cam_solved"], float), maha=r["maha"],
                good=r["world_fix_err_m"] < 1.5))
    return rows


# ---- composition hypotheses: R_wc(roll,pitch,yaw; params) ------------------------------
def Rz(a): return Rotation.from_euler("Z", a).as_matrix()
def Ry(a): return Rotation.from_euler("Y", a).as_matrix()
def Rx(a): return Rotation.from_euler("X", a).as_matrix()


def make_candidates():
    """name -> (n_params, fn(r, p) -> R_wc_model_candidate)."""
    def c0(r, p):     # production chain (sanity: zero-param)
        return Rz(r["yaw"]) @ Ry(r["pitch"]) @ Rx(r["roll"]) @ Ry(TAU) @ SWAP.T

    def c_mount(r, p):  # fixed 3-dof mount correction after the tilt (classic calibration)
        return Rz(r["yaw"]) @ Ry(r["pitch"]) @ Rx(r["roll"]) @ Ry(TAU) @ rodr(p[:3]) @ SWAP.T

    def c_tilt_before_roll(r, p):  # tilt inserted before roll, tilt value free
        return Rz(r["yaw"]) @ Ry(r["pitch"] + p[0]) @ Rx(r["roll"]) @ SWAP.T

    def c_tbr_mount(r, p):  # tilt-before-roll + residual fixed mount
        return Rz(r["yaw"]) @ Ry(r["pitch"] + p[0]) @ Rx(r["roll"]) @ rodr(p[1:4]) @ SWAP.T

    def c_roll_scale(r, p):  # roll under/over-applied + free tilt after roll
        return Rz(r["yaw"]) @ Ry(r["pitch"]) @ Rx(p[1] * r["roll"]) @ Ry(p[0]) @ SWAP.T

    def c_tbr_scale(r, p):  # tilt-before-roll with roll scale + mount
        return (Rz(r["yaw"]) @ Ry(r["pitch"] + p[0]) @ Rx(p[1] * r["roll"]) @ rodr(p[2:5]) @ SWAP.T)

    return {
        "C0 production (0p)": (0, c0),
        "C5 fixed-mount (3p)": (3, c_mount),
        "C1 tilt-before-roll (1p: tau)": (1, c_tilt_before_roll),
        "C6 tilt-before-roll+mount (4p)": (4, c_tbr_mount),
        "C3 roll-scale+tilt (2p)": (2, c_roll_scale),
        "C7 tbr+rollscale+mount (5p)": (5, c_tbr_scale),
    }


def fit_candidate(rows, nfn, x0=None):
    n, fn = nfn

    def resid(p):
        out = []
        for r in rows:
            R_true = r["R_wg"] @ r["R_solved"].T
            Ecand = fn(r, p).T @ R_true
            out.append(Rotation.from_matrix(Ecand).as_rotvec())
        return np.concatenate(out)

    p0 = np.zeros(n) if x0 is None else np.asarray(x0, float)
    if n == 0:
        rv = resid(p0)
        return p0, rv
    sol = least_squares(resid, p0, method="lm", max_nfev=200)
    return sol.x, resid(sol.x)


def roll_slope(rows, rv):
    """Residual-rotvec (body frame) slope vs roll per axis + r^2 of the strongest."""
    e_b = np.array([(Ry(TAU) @ SWAP.T) @ v for v in rv.reshape(-1, 3)])   # cam -> body
    roll = np.array([r["roll"] for r in rows])
    out = []
    for ax in range(3):
        s = np.polyfit(roll, e_b[:, ax], 1)[0]
        r2 = np.corrcoef(roll, e_b[:, ax])[0, 1] ** 2 if roll.std() > 0 else 0.0
        out.append((s, r2))
    return out


def main():
    rows = load_rows()
    good = [r for r in rows if r["good"]]
    print(f"rows: {len(rows)} solved 4-corner; {len(good)} good (|fix|<1.5 m)")

    e = np.degrees([r["e_cam"] for r in good])
    mag = np.linalg.norm(e, axis=1)
    print(f"\n== raw chain attitude error (camera frame, deg; good fixes) ==")
    print(f"  |e| p50 {np.percentile(mag,50):.2f}  p90 {np.percentile(mag,90):.2f}  max {mag.max():.2f}")
    print(f"  mean [{e[:,0].mean():+.2f} {e[:,1].mean():+.2f} {e[:,2].mean():+.2f}]  "
          f"std [{e[:,0].std():.2f} {e[:,1].std():.2f} {e[:,2].std():.2f}]   "
          f"(cam X~pitch, Y~yaw, Z~optical roll)")
    # rotation-noise floor implied by the analytic covs (how much of the std is PnP noise?)
    sig = np.array([np.sqrt(np.diag(r["cov_rr"])) for r in good if r["cov_rr"] is not None])
    print(f"  analytic PnP rotation 1-sig (deg): median "
          f"[{np.degrees(np.median(sig[:,0])):.2f} {np.degrees(np.median(sig[:,1])):.2f} "
          f"{np.degrees(np.median(sig[:,2])):.2f}]")

    print("\n== regressions of e (BODY frame) on attitude (good fixes) ==")
    e_b = np.array([(Ry(TAU) @ SWAP.T) @ r["e_cam"] for r in good])
    roll = np.array([r["roll"] for r in good]); pitch = np.array([r["pitch"] for r in good])
    for nm, x in (("roll", roll), ("pitch", pitch)):
        line = f"  vs {nm:>5}: "
        for ax, an in enumerate(("x(roll)", "y(pitch)", "z(yaw)")):
            s = np.polyfit(x, e_b[:, ax], 1)[0]
            r2 = np.corrcoef(x, e_b[:, ax])[0, 1] ** 2
            line += f"e_{an} {s:+.3f} (r2 {r2:.2f})  "
        print(line)

    print("\n== composition hypothesis fits (good fixes; residual rms + roll coupling) ==")
    for name, nfn in make_candidates().items():
        x0 = None
        if "tilt-before-roll" in name or "tbr" in name:
            x0 = np.zeros(nfn[0]); x0[0] = TAU
        if name.startswith("C3"):
            x0 = np.array([TAU, 1.0])
        if name.startswith("C7"):
            x0 = np.zeros(5); x0[0] = TAU; x0[1] = 1.0
        p, rv = fit_candidate(good, nfn, x0)
        rms = np.degrees(np.sqrt(np.mean(rv ** 2)))
        slopes = roll_slope(good, rv)
        worst = max(slopes, key=lambda t: abs(t[0]))
        ptxt = np.array2string(np.degrees(p), precision=2, suppress_small=True) if len(p) else "-"
        print(f"  {name:32s} rms/axis {rms:5.2f} deg   params(deg) {ptxt:28s} "
              f"worst roll-slope {worst[0]:+.3f} (r2 {worst[1]:.2f})")

    print("\nNOTE: C3 params = [tilt_after_roll, roll_scale(NOT deg)]; "
          "C7 = [tilt, roll_scale(NOT deg), mount rotvec x3]")


if __name__ == "__main__":
    main()
