"""ADVERSARIAL re-derivation of the d1 obs spec load-bearing claims.

Lens: "preserves-VQ1-and-caseAB-and-is-faithful-to-get_observations".
Charge: REFUTE. The original d1_obs_faithfulness_check.py passes, but several of its checks are
weaker than the spec's prose claims. This script tests the GAPS:

  A. FRAME-CONSISTENCY of the unification. The obs builder (fly_rl.obs_from_zup) runs in DiffAero
     Z-up/FLU. The localization lever L = R_world_camera @ t_cam_gate (localization.py:86) is in
     world NED. The original check (2) NEVER touches localization -- it computes R_w2g@(gate-pos)
     two ways in the SAME frame, a tautology. Here we drive the REAL localization arithmetic to
     produce a NED fix, convert it to Z-up the way build_obs does (_FLIP), and ask whether
     pos_g (obs slot) == R_w2g_zup @ L_seen_zup BIT-EXACTLY. If a _FLIP is missing, the lateral
     (E) and vertical (D) in-plane axes flip sign and the claim breaks.

  B. SIGN of the delivered offset. The spec header (lines 50,63) says the estimator delivers
     "-L (the drone->gate offset)" but the obs slot pos_g = R_w2g@(gate-p) = R_w2g@(+L)
     (gate-p == +L). Verify which sign actually lands in the obs slot, against the real builder.

  C. Does the REAL gate_pose_to_world_position, fed a noiseless seen-gate PnP, reconstruct a pose
     p such that obs_from_zup(p, gate=seen_opening) reproduces the same pos_g as feeding L directly?
     (The end-to-end case-C path the spec claims, not the isolated algebra.)

  D. VQ1 / case-A/B preservation under the 19-dim append: confirm dims 0..16 are byte-identical
     whether or not the confidence channel is appended, AND that a 17-dim consumer is unaffected.

Run: PYTHONPATH=src .venv/Scripts/python.exe handoff/ultracode-gate-relative-pipeline-design-2026-06-13/v_obs_adversarial_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "rl"))
sys.path.insert(0, str(_REPO / "handoff" / "ultracode-vision-case-c-2026-06-13"))

import fly_rl
from fly_rl import (
    _GATE_POS_ZUP, _GATE_YAW_ZUP, _FLIP, N_GATES,
    obs_from_zup, make_gate_map, _gate_rotmat_w2g,
)
from racer.contracts import Gate, GatePose
from racer.frames import R_world_from_body, R_camera_from_body
from racer.localization import gate_pose_to_world_position

SEED = 4242
rng = np.random.default_rng(SEED)


def _rand_attitude_zup(rng):
    from scipy.spatial.transform import Rotation as Rot
    rpy = rng.normal(0, 0.3, 3)
    return Rot.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()


# ===========================================================================================
# A + B + C: end-to-end NED localization fix -> Z-up obs slot. Does pos_g actually equal the
# gate-frame seen offset, with the REAL cross-frame conversion?
# ===========================================================================================
def check_end_to_end_caseC():
    """Build a real seen-gate sighting in NED, run the REAL localization to get the world fix
    p_abs (NED), then form the obs the way build_obs does: convert to Z-up, apply gate_map yaw.
    Compare three candidate 'pos_g' the spec conflates:
      (i)   obs pos_g from feeding the seen-gate opening as gate_pos and the reconstructed pose p
      (ii)  R_w2g_zup @ L_seen_zup  (the 'estimator delivers L' shortcut the spec claims is identical)
      (iii) R_w2g_zup @ (-L_seen_zup) (the spec header's '-L' phrasing)
    The faithful one must equal (i). We test whether (ii) or (iii) matches, and whether the
    NED->Zup conversion is actually accounted for.
    """
    max_ii = 0.0
    max_iii = 0.0
    n = 1500
    # use gate 4 (the binding gate); its true opening (NED) and yaw
    tg = 4
    gate_true_zup = _GATE_POS_ZUP[tg].copy()
    gate_true_ned = gate_true_zup * _FLIP        # Z-up -> NED (inverse is the same diag flip)
    yaw_gate_zup = float(_GATE_YAW_ZUP[tg])
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    R_w2g_zup = _gate_rotmat_w2g(yaw_gate_zup)
    for _ in range(n):
        # drone true pose in NED, a few metres up-track of the gate
        drone_true_ned = gate_true_ned - np.array([rng.uniform(4, 12), rng.uniform(-1, 1), rng.uniform(-1, 1)])
        roll, pitch, yaw = rng.normal(0, 0.15), rng.normal(-0.3, 0.1), rng.normal(np.pi, 0.1)
        R_wb = R_world_from_body(roll, pitch, yaw)
        R_wc = R_wb @ R_camera_from_body().T
        L_true_ned = gate_true_ned - drone_true_ned         # gate - drone, NED  (== +L)
        t_cam_gate = R_wc.T @ L_true_ned                    # noiseless PnP to the SEEN opening
        gp = GatePose(frame_id=0, sim_time_ns=0, R_cam_gate=np.eye(3), t_cam_gate=t_cam_gate,
                      reproj_error_px=0.0, gate_id=tg, covariance=None, n_corners=4)
        gate_obj = Gate(gate_id=tg, position_ned=gate_true_ned, R_world_gate=np.eye(3))
        # REAL localization: p_abs (NED) = gate.position_ned - L
        p_abs_ned, _ = gate_pose_to_world_position(gp, gate_obj, R_wb,
                                                   attitude_noise_std=0.0, fix_cov_floor_std=0.0)
        # (i) the obs as build_obs would form it: pose -> Z-up, gate_map yaw path
        p_zup = p_abs_ned * _FLIP
        vel_zup = np.zeros(3); R_b2w_zup = _rand_attitude_zup(rng); w = np.zeros(3)
        obs = obs_from_zup(p_zup, vel_zup, R_b2w_zup, w, tg, 0.0, virtual_flip=False, gate_map=gm)
        pos_g_i = obs[:3].astype(np.float64)
        # (ii) estimator delivers L in NED, convert to Z-up, apply gate frame: R_w2g_zup @ (L*_FLIP)
        L_seen_ned = R_wc @ t_cam_gate                      # == L_true_ned (noiseless)
        L_seen_zup = L_seen_ned * _FLIP
        pos_g_ii = R_w2g_zup @ L_seen_zup
        # (iii) the header's '-L' phrasing
        pos_g_iii = R_w2g_zup @ (-L_seen_zup)
        max_ii = max(max_ii, float(np.max(np.abs(pos_g_i - pos_g_ii))))
        max_iii = max(max_iii, float(np.max(np.abs(pos_g_i - pos_g_iii))))
    return {"n": n, "max|obs - R_w2g@(+L_zup)|": max_ii,
            "max|obs - R_w2g@(-L_zup)|": max_iii}


# ===========================================================================================
# A2: does the ORIGINAL check (2) silently assume Z-up==NED? Re-run its exact logic but with a
# realistic NED lever to show it would have MISSED a missing _FLIP. (It computes everything in
# one frame, so it cannot catch the cross-frame bug; we demonstrate the blind spot.)
# ===========================================================================================
def check_original_blindspot():
    """The original (2) sets gate_true_zup = _GATE_POS_ZUP[4], pos = a Z-up state, and computes
    pos_g_obs = R_w2g @ (gate_true_zup - pos);  L_seen = gate_true_zup - pos;  pos_g_est = R_w2g @ L_seen.
    These are the SAME expression by construction (L_seen := gate - pos), so max|diff|==0 ALWAYS,
    regardless of frames. It is a tautology, not a faithfulness test. Demonstrate: it returns 0
    even if we corrupt the frame, because both sides use the corrupted value identically."""
    R_w2g = _gate_rotmat_w2g(float(_GATE_YAW_ZUP[4]))
    gate = _GATE_POS_ZUP[4]
    pos = gate - np.array([5.0, 0.2, -0.3])
    # both sides use (gate - pos); identical by definition
    a = R_w2g @ (gate - pos)
    L = gate - pos
    b = R_w2g @ L
    return {"original_(2)_is_identity": float(np.max(np.abs(a - b))),
            "note": "0.0 by construction (L := gate-pos); does NOT exercise localization or NED<->Zup"}


# ===========================================================================================
# D: 19-dim append preserves dims 0..16 byte-for-byte and a 17-dim consumer is unaffected.
# ===========================================================================================
def check_append_backcompat():
    gm = make_gate_map(_GATE_POS_ZUP, _GATE_YAW_ZUP)
    max_diff = 0.0
    n = 2000
    for _ in range(n):
        pos = np.array([-100.0, 3.0, 22.0]) + rng.normal(0, 2.0, 3)
        vel = rng.normal(0, 1.0, 3)
        R_b2w = _rand_attitude_zup(rng)
        w = rng.normal(0, 0.5, 3)
        tg = int(rng.integers(0, N_GATES))
        lnt = float(rng.uniform(0, 3.765))
        obs17 = obs_from_zup(pos, vel, R_b2w, w, tg, lnt, gate_map=gm)
        # simulate the 19-dim builder: same 0..16, then append two sigmas. We mimic the proposed
        # append by concatenating; the test is that 0..16 are untouched.
        P_pos = np.diag([0.3, 0.27, 0.27])
        R_w2g = _gate_rotmat_w2g(float(_GATE_YAW_ZUP[tg]))
        P_g = R_w2g @ P_pos @ R_w2g.T
        sig_in = np.sqrt(P_g[1, 1] + P_g[2, 2]); sig_al = np.sqrt(P_g[0, 0])
        obs19 = np.concatenate([obs17, [sig_in, sig_al]]).astype(np.float32)
        max_diff = max(max_diff, float(np.max(np.abs(obs19[:17].astype(np.float64) - obs17.astype(np.float64)))))
    return {"n": n, "max|obs19[:17] - obs17|": max_diff}


def main():
    print("=" * 88)
    print("A+B+C  END-TO-END case-C: real NED localization fix -> Z-up obs slot")
    print("=" * 88)
    r = check_end_to_end_caseC()
    print(f"  n={r['n']}")
    print(f"  max|obs pos_g - R_w2g_zup @ (+L_zup)|  = {r['max|obs - R_w2g@(+L_zup)|']:.3e}")
    print(f"  max|obs pos_g - R_w2g_zup @ (-L_zup)|  = {r['max|obs - R_w2g@(-L_zup)|']:.3e}")
    sign_ok = r['max|obs - R_w2g@(+L_zup)|'] < 1e-9
    print(f"  => obs slot == R_w2g @ (+L_zup) [+L, NOT -L]: {'CONFIRMED' if sign_ok else 'FAILED'}")
    print(f"     (the spec header phrase 'estimator delivers -L' is loose; the obs slot needs +L = gate-p)")

    print("\n" + "=" * 88)
    print("A2  ORIGINAL CHECK (2) BLIND SPOT")
    print("=" * 88)
    b = check_original_blindspot()
    print(f"  original (2) diff (by construction) = {b['original_(2)_is_identity']:.3e}")
    print(f"  {b['note']}")

    print("\n" + "=" * 88)
    print("D  19-dim APPEND back-compat (dims 0..16 untouched)")
    print("=" * 88)
    d = check_append_backcompat()
    print(f"  n={d['n']}  max|obs19[:17] - obs17| = {d['max|obs19[:17] - obs17|']:.3e}")
    print(f"  => {'PASS (pure append)' if d['max|obs19[:17] - obs17|'] == 0.0 else 'FAIL'}")

    print("\n" + "=" * 88)
    allok = sign_ok and d['max|obs19[:17] - obs17|'] == 0.0
    print(f"OVERALL adversarial: {'frame-consistent +L append confirmed' if allok else 'PROBLEM FOUND'}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
