import json, numpy as np
base = "handoff/perception-char-2026-06-08"

# ---- Confirm: does the predict-step Q model SYSTEMATIC accel/attitude bias? It models NOISE only. ----
# A constant attitude BIAS b_theta (not noise) injects a CONSTANT phantom accel g*b_theta that
# integrates DETERMINISTICALLY: pos error = 0.5 * (g*b_theta) * T^2 (NOT covered by Q -> KF is overconfident).
g=9.80665
print("=== Systematic drift the Q does NOT model (constant attitude bias -> deterministic pos error) ===")
for b_deg in [0.5, 1.0]:
    b=np.deg2rad(b_deg); a=g*b
    print(f"  attitude bias {b_deg}deg -> phantom accel {a:.3f} m/s^2 -> pos err at T: " +
          "  ".join(f"{T}s:{0.5*a*T*T:.3f}m" for T in [0.5,1.0,2.0]))
# accel bias directly:
print("  accel bias 0.1 m/s^2 -> pos err: " + "  ".join(f"{T}s:{0.5*0.1*T*T:.3f}m" for T in [0.5,1.0,2.0]))

# ---- gate-3 registration: live crosses 1.46 m above map D-center yet PASS-CLEAN. Confirm the geometry ----
tm = json.load(open("handoff/shadowpc-firstcontact-2026-06-02/track_map.json"))
g3 = np.array(tm["gates"][3]["position_ned"])
outer_half = 2.72/2
print(f"\n=== gate-3 registration (case-C danger amplifier) ===")
print(f"  map g3 position_ned = {g3.round(3)} (this is BOTTOM-CENTRE per schema; D={g3[2]:.3f})")
print(f"  outer_half={outer_half:.3f} m; finding: live crosses 1.46m off in D yet clean => |map D error|~1.46m > {outer_half:.2f}")
print(f"  CASE-C IMPLICATION: a vision fix anchored to a gate whose MAP centre is 1.46m off will pull")
print(f"  the KF position 1.46m toward the wrong place, AND there is no given-pos to correct it.")

# ---- Mahalanobis gate behaviour in case C: prior is vision-dead-reckoned. Does a biased map slip through? ----
# A systematic per-gate map bias is SHARED between the prior (built from past biased fixes) and the new fix,
# so innovation nu = z - x_prior is SMALL -> chi2 passes. The gate cannot catch a CONSISTENT map bias.
print("\n=== Mahalanobis gate vs a CONSISTENT map bias (case C) ===")
print("  In case A/B given-pos anchors x, so a biased fix has LARGE innovation -> chi2 may reject / given-pos overrides.")
print("  In case C x is built FROM vision; a consistent map bias appears in BOTH prior and fix ->")
print("  innovation ~0 -> chi2 PASSES -> bias enters the estimate undetected (the case-C-specific hole).")

# ---- what feeds predict() when given pos/vel OFF: confirm attitude R_wb still GIVEN ----
print("\n=== predict() input in case C ===")
print("  navigator.update: R_wb = R_world_from_odo_quat_wxyz(ds.orientation_ned_wxyz)  [attitude STILL given]")
print("  kf.predict(ds.accel_body, R_wb, dt)  <-- UNCONDITIONAL, runs in case C identically.")
print("  Only the two `if use_given_position/velocity` blocks are skipped. So KF coasts on IMU+given-attitude,")
print("  corrected ONLY by vision fixes. Confirmed: prediction source is intact in case C.")
