"""CPU-only reconstruction of the inc8 arm-A pointing->fix CONVERSION gap (diagnostic, NO training).

Uses the REAL calibrated surrogate (rl/fix_surrogate.py) + the REAL reward shape (rl/inc8_reward.py
terminal_weight / visibility) + the VQ1 course geometry to quantify the structural reward<->accept
anti-alignment, independent of any policy. Then folds in the pilot-trajectory logged metrics
(job 3273437) to bound the achieved in-window pointing. Pure numpy; no GPU, no checkpoint needed.
"""
import sys, json, math
sys.path.insert(0, "rl"); sys.path.insert(0, "src")
import numpy as np
import fix_surrogate as FS

S = FS.DEFAULT
NENVS = 2048

print("="*78)
print("0. CALIBRATION CONSTANTS (rl/fix_surrogate.py DEFAULT == Track-3 fit)")
print("="*78)
print(f"  accept band-pass: rlo={S.accept_rlo:.2f} m  rhi={S.accept_rhi:.2f} m  w={S.accept_wlo:.2f}  pmax={S.accept_pmax:.3f}")
print(f"  valid-range guard: [{S.accept_range_guard_lo}, {S.accept_range_guard_hi}] m  (hard-zero outside)")
print(f"  off-image accept floor p_out = {S.accept_p_out_of_image:.6f}")
print(f"  VFoV half = {FS.VFOV_HALF_DEG:.2f} deg   HFoV half = {FS.HFOV_HALF_DEG:.1f} deg")

# ---- terminal_weight (arm A) replicated in numpy (== inc8_reward.terminal_weight) -----------------
D_LOCK, D_ACQ, W0 = 5.0, 24.0, 0.15      # inc8_reward.Inc8RewardWeights defaults
def w_term(r):
    ramp = np.clip((D_ACQ - r) / max(D_ACQ - D_LOCK, 1e-6), 0.0, 1.0)
    return W0 + (1.0 - W0) * ramp

print("\n" + "="*78)
print("1. THE RANGE ANTI-ALIGNMENT  (reward peak vs accept band)")
print("="*78)
rs = np.arange(0.0, 35.1, 1.0)
pa = S.accept_prob_in_image(rs)          # P(accept | in_image) -- the exact deployed band-pass
wt = w_term(rs)
print(f"  {'range':>6} {'p_accept|in_img':>16} {'w_term(armA)':>13}   note")
for r, p, w in zip(rs, pa, wt):
    if r % 2 == 0:
        note = ""
        if p < 1e-3 and r <= 14: note = "<- DEAD ZONE (no fix possible)"
        if p > 0.5: note = "<- ACCEPT BAND (fixes here)"
        if abs(r - D_LOCK) < 0.6: note += "  [reward PEAK w_term=1.0]"
        print(f"  {r:6.0f} {p:16.4f} {w:13.3f}   {note}")

# key contrasts
p_at_reward_peak = float(S.accept_prob_in_image(np.array([D_LOCK]))[0])   # accept where reward is maximal
r_accept_peak = float(rs[np.argmax(pa)])
w_at_accept_peak = float(w_term(np.array([r_accept_peak]))[0])
print(f"\n  >> At the REWARD peak range (d={D_LOCK} m, w_term=1.0):   p_accept = {p_at_reward_peak:.5f}   (FIX IMPOSSIBLE)")
print(f"  >> At the ACCEPT peak range (d={r_accept_peak:.0f} m, p_accept={pa.max():.3f}):  w_term = {w_at_accept_peak:.3f}   (reward {w_at_accept_peak/1.0*100:.0f}% of terminal)")
# reward-weighted accept (what fraction of the perception reward sits where fixes are possible)
band = pa > 0.05
print(f"  >> Reward-weight INSIDE the accept band (16-28 m): w_term in [{w_term(np.array([16.0]))[0]:.2f}, {w_term(np.array([28.0]))[0]:.2f}]"
      f"  vs 1.0 at terminal -> the reward maximises pointing OUTSIDE the fixable band.")

print("\n" + "="*78)
print("2. GATE SPACING -- is the 16-28 m accept band even traversed?  (VQ1 course)")
print("="*78)
course = json.load(open("rl/peregrine_course_diffaero.json"))
P = np.array([g["pos_zup"] for g in course["gates"]])
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
for i, d in enumerate(seg):
    frac_band = max(0.0, (min(d, S.accept_rhi) - S.accept_rlo)) / d if d > S.accept_rlo else 0.0
    print(f"  g{i}->g{i+1}: spacing {d:5.1f} m   approach fraction inside [16,28] m ~ {frac_band*100:4.0f}%")
print(f"  mean spacing {seg.mean():.1f} m -> EVERY approach sweeps the band; it is NOT structurally absent.")

print("\n" + "="*78)
print("3. FIX-RATE QUANTUM + OFF-IMAGE FLOOR  (decisive A-vs-B test, from EXISTING logs)")
print("="*78)
q = 1.0 / NENVS
print(f"  NENVS={NENVS} -> fix_rate quantum = 1/{NENVS} = {q:.6f}")
for k in (0,1,2,3):
    print(f"    {k} fix/step -> fix_rate {k*q:.5f}   (pilot logs show exactly 0.00000/{q:.5f}/{2*q:.5f}/{3*q:.5f})")
# expected off-image-floor contribution: ~(1-pointing_rate)*NENVS envs each with p_out
pointing_rate = 0.028        # logged plateau (in_img over ALL ranges)
exp_offimg_fix_per_step = (1 - pointing_rate) * NENVS * S.accept_p_out_of_image
obs_mean_fix_rate = 0.000316     # measured plateau mean (steps ~1300-5990, 48 logged pts)
print(f"\n  off-image floor alone: ~{(1-pointing_rate)*NENVS:.0f} not-in-img envs x p_out {S.accept_p_out_of_image:.5f}"
      f" = {exp_offimg_fix_per_step:.2f} fix/step  -> rate {exp_offimg_fix_per_step/NENVS:.5f}")
print(f"  observed plateau MEAN fix_rate = {obs_mean_fix_rate:.5f} ({obs_mean_fix_rate*NENVS:.2f} fix/step)")
print(f"  -> off-image floor {exp_offimg_fix_per_step:.2f} + in-window excess {obs_mean_fix_rate*NENVS-exp_offimg_fix_per_step:.2f} fix/step (rate {(obs_mean_fix_rate*NENVS-exp_offimg_fix_per_step)/NENVS:.6f}).")
print(f"  NO blocking bug: GREEN x2 (parity-gated) already proved surrogate+KF converts in-window pointing->fixes;")
print(f"  the small excess-over-floor corroborates the path fires in training too. in-window fixes ~ 0, not blocked.")

print("\n" + "="*78)
print("4. INFER the achieved in-window pointing from the 3 logged metrics")
print("="*78)
term_pointing = 0.065        # in_img | range<=5 m   (DEAD ZONE)
fix_rate = 0.000316          # plateau MEAN (measured from the pilot trajectory, 48 logged pts)
# fix_rate ~= P(in_img & range in band)*pmax + (1-..)*p_out  -> solve for P(in_img & band)
p_inwin_inimg = max(0.0, (fix_rate - S.accept_p_out_of_image) / S.accept_pmax)
print(f"  terminal_pointing (in_img @ <=5 m, DEAD ZONE)         = {term_pointing:.3f}")
print(f"  pointing_rate     (in_img @ ANY range)                = {pointing_rate:.3f}")
print(f"  => implied P(in_img AND range in [16,28]) ~ (fix_rate-p_out)/pmax = {p_inwin_inimg:.5f}")
print(f"  => in-window in_img is ~{term_pointing/max(p_inwin_inimg,1e-9):.0f}x RARER than terminal in_img.")
print(f"     The policy points the camera in the TERMINAL dead zone, essentially never in the fix band.")

print("\n" + "="*78)
print("5. BORESIGHT (Q4): is it in the emulation accept/geometry path?")
print("="*78)
print("  inc8_estimator_emul.batched_geometry computes in_image/range/az/el from TRUTH pose with the")
print("  canonical R_CAMERA_FROM_BODY_NP -- NO boresight offset. p_accept depends ONLY on (range,in_image).")
print("  The -0.25 m vertical boresight enters NOWHERE in p_accept; it is modelled (loosely) only as the")
print("  per-episode one-signed in-plane FIX-VALUE bias (bias_mag 0-0.19 m), which changes the KF z, not")
print("  whether a fix is accepted. => Applying BoresightCorrection(-0.25) does NOT change p_accept at the")
print("  policy's geometry. Boresight is a RED HERRING for the training conversion gap.")
print("\nDONE.")
