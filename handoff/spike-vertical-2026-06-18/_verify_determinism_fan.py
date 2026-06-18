"""Determinism (2x identical) + range-fan emul-vs-real obs delta characterization."""
import sys, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "rl"))
os.chdir(ROOT)
import spike_vertical_slice as S

np.set_printoptions(precision=8, suppress=True)

# ---- determinism: run the full slice twice, diff every frozen quantity ----
r1 = S.run_slice_once(); r2 = S.run_slice_once()
def dmax(a, b): return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
print("=== DETERMINISM (run1 vs run2, max|delta|) ===")
print("  gate_pose.t_cam_gate :", dmax(r1["gate_pose"].t_cam_gate, r2["gate_pose"].t_cam_gate))
print("  nav_state.position   :", dmax(r1["nav_state"].position_ned, r2["nav_state"].position_ned))
print("  obs20                :", dmax(r1["obs20"], r2["obs20"]))
print("  rate_frd             :", dmax(r1["rate_frd"], r2["rate_frd"]))
print("  collective/thrust    :", abs(r1["collective"]-r2["collective"]), abs(r1["normed_thrust"]-r2["normed_thrust"]))
print("  n_rel_applied/n_vision_fixes :", r1["nav"].vision_diag.n_rel_applied, r1["nav"].n_vision_fixes)

# ---- range fan: drone stays near the ORIGIN SEED (so the navigator range-consistency gate
# accepts the fix); MOVE THE GATE to vary range. Characterizes obs17 parity + the [17:20] triple
# AND the range-dependence of the confidence channel. ----
print("\n=== RANGE FAN (gate moved, drone near origin seed; emul-vs-real obs) ===")
print(f"  {'range_m':>8} {'reproj_px':>10} {'obs17 bo-vs-fz':>14} {'sig_ip':>8} {'sig_al':>8} {'c_ip':>7} {'c_al':>7} {'fid_delta':>10} {'act_ok':>7} {'nrel':>5}")
for gate_n in [5.0, 8.0, 11.0, 14.0]:
    drone = [0.5, -0.3, -2.0]                                  # near the origin seed (|drone| < pos_std 5)
    nav, gate, gp, ns = S.drive_navigator(gate_pos_ned=(gate_n, 0.0, -2.5), drone_true_ned=drone, n_ticks=120)
    obs20, info = S.build_obs20(nav, gate, ns)
    rate, coll, nt, ok, _ = S.actor_step(obs20)
    bo_fz = float(np.max(np.abs(info["obs17_buildobs"].astype(float) - info["obs17_fromzup"].astype(float))))
    sig_ip, sig_al = info["sigmas"]
    emul_t, dep_t = S.emul_confidence_fidelity(nav.kf.P[:3, :3], gate.position_ned, ns.time_since_vision_update_s)
    fid = float(np.max(np.abs(np.asarray(emul_t) - np.asarray(dep_t))))
    print(f"  {gp.range_m:8.3f} {gp.reproj_error_px:10.2e} {bo_fz:14.2e} {sig_ip:8.4f} {sig_al:8.4f} "
          f"{info['triple'][0]:7.4f} {info['triple'][1]:7.4f} {fid:10.2e} {str(ok):>7} {nav.vision_diag.n_rel_applied:5d}")
print("\nDONE")
