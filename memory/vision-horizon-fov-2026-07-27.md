---
name: vision-horizon-fov-2026-07-27
description: "ROOT CAUSE of the gate-side slam: the drone is BLIND for the final ~1.8 m because the gate overflows the 58.7 deg VERTICAL field of view. Zero of 899 confirmed passes ever got a fix inside 1.19 m. Structural — not tunable."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T07:30:33.846Z
---

# The vision horizon is FIELD-OF-VIEW GEOMETRY — 2026-07-27

**This is the root cause of the failure Fengyou has been pointing at since the start** ("the drone
flies into the SIDE of the gate even though the vision was fine"). The vision *is* fine. It just
stops, always, at the same place, for a reason no knob can reach.

## 🟢🟢 THE HARD FLOOR — 0 of 899, measured non-circularly

Over **671 flights / 899 CONFIRMED PASSES**: **not one pass ever received a vision fix closer than
1.19 m.** Last-sighted range p10 **1.48** · median **1.78** · p90 **2.80** m. Fraction of passes
with any fix inside 1.0 m: **0.0%**.

🚩 **THE FIRST VERSION OF THIS STATISTIC WAS CIRCULAR AND I CAUGHT IT MYSELF.** Binning per-TICK by
`norm(rel_flu)` is circular: with no fresh fix, `rel_flu` is the PROPAGATED belief, so a long coast
marches the *believed* range down through the low bins and fills them with precisely the not-seen
ticks. **The non-circular instrument is per-APPROACH:** the LAST-SIGHTED RANGE, read off a
measurement (a fresh fix) and never off a coast, restricted to CONFIRMED PASSES so "the log ended
early" cannot masquerade as "vision stopped". `scripts/vision_horizon/last_sighted_range.py`.
It confirmed the claim and **sharpened it from a falloff into a wall**. Zero-out-of-899 is a
mechanism, not a statistic.

## 🟢🟢 THE MECHANISM: THE GATE OVERFLOWS THE VERTICAL FRAME

Camera (`src/racer/frames.py:243-258`): fx=fy=320, W=640, H=360 ⇒ **HFOV 90.0°** but
**VFOV only 58.7°**, optical axis pitched **+20° UP**, body elevation band **(−9.4°, +49.4°)**.
**Vertical is barely two-thirds of horizontal — it is the axis that clips first.**

Predict the FOV-limited range per pass by solving `|elevation| + atan(0.75/R) = VFOV/2` using each
pass's own last-fix elevation, then compare to the range actually observed (n=899):

| | p10 | median | p90 |
|---|---|---|---|
| **predicted by FOV geometry** | 1.39 | **1.76** | 2.78 |
| **observed last-fix range** | 1.48 | **1.78** | 2.80 |

**median(observed − predicted) = +0.04 m.** The whole distribution matches within 0.09 m.
At the last fix `|el| + half_angle` = **28.9°** against a frame half-height of **29.4°** — the gate
is sitting exactly on the vertical frame edge. It overflows vertically on **47%** of passes at the
last fix and **70%** one tick later; horizontally only 13% → 29%.
🚩 **CONTROL that makes it causal, not coincidental:** last-fix **|azimuth| is 13.1° vs 5.2°** over
all fresh ticks (2.5×), and |elevation| 6.2° vs 5.4° — the final fix is taken at *unusually
off-axis* geometry, exactly as a frame-edge cutoff requires.
🛑 **HONEST CAVEAT:** per-pass correlation between predicted and observed is only **+0.113**. Both
distributions are tight around the same value because the cutoff is dominated by the constant
(VFOV/2 and the 0.75 m half-width) with elevation as a small perturbation, so there is little
spread left to correlate. The distribution match + the overflow fractions + the azimuth control
carry this finding; the per-pass correlation does NOT.

## ⇒ WHAT THIS MEANS FOR THE CAMPAIGN

**The final ~1.8 m (~0.22 s at 8 m/s) is flown BLIND, always, by construction.** Combined with the
already-established fact that at the last sighted fix passes and gate-strikes are nearly the same
distribution (|lat| AUC 0.582), the chain is now complete:

> the drone loses the gate at ~1.8 m because the gate leaves the frame → it commits open-loop on a
> propagated belief → the terminal scatter that decides pass-vs-strike is generated in that
> interval → **every deploy-side aim/trim/bias knob acts BEFORE the blind interval and therefore
> cannot touch the thing that kills.**

## 🛑🛑 "TRAIN IT TO COMMIT BLIND" IS **REFUTED** — TRAINING ALREADY MODELS THIS BLACKOUT, EXACTLY

My own leading hypothesis, killed before it cost a 6-hour GPU run. **Training is not naive about
visibility.** `rl/gate_visibility.py:225 gate_detectable` projects **8 gate keypoints** through the
**SAME intrinsics imported from `racer.frames`** (fx=fy=320, 640×360, +20°) and requires
**`MIN_VISIBLE_CORNERS = 4`** in-frame and unoccluded, plus a 30 m far cap. Ran it directly
(torch 2.9.1 CPU, `scripts/vision_horizon/train_vs_wire_horizon.py`) — closest range still
detectable in TRAINING, by geometry:

| approach | training horizon |
|---|---|
| pitched **+24°** (the flown median) | **1.72 m** |
| pitched +10° | **1.78 m** |
| level, 0.3 m high | **1.19 m** |
| head-on level centred | 1.29 m |
| level 0.3 m low / 0.3 m lateral | 1.55 / 1.40 m |

**Wire: median 1.78 m, minimum-ever 1.19 m.** Training +10° gives 1.78; training 0.3-m-high gives
1.19. **They agree to centimetres across the geometry range.** ⇒ **THE POLICY HAS ALREADY BEEN
TRAINED ON THIS EXACT BLACKOUT. DO NOT RUN THAT ARM.** 🟢 A useful corollary: the deploy YOLO is
performing AT the level training assumes (median observed − predicted = +0.04 m, and the p90 tail
matches too) — **the vision stack is not underperforming; this is geometry.**

## 🛑🛑 …BUT I CHECKED THE RIGHT VARIABLE ON THE WRONG CHANNEL. THE MISMATCH IS REAL AND INVERTED.

**Correction to the refutation above, from the 48-agent training audit.** I verified that training
goes blind at the same RANGE as the wire — true, and it stands. What I never asked was **what the
observation looks like ONCE blind**, and there the two could not differ more:

* **TRAINING MASKS `obs[11:16]` TO ZEROS** — 100% of ticks below 1.0 m, 99.3% below 1.5 m
  (independent replay of the env's own `gate_detectable` over 26 092 real flown attitudes; 50%
  crossing at ~1.83 m). `peregrine_racing_ego.py:521-523`, the `keep &= det` line.
* **THE WIRE FEEDS A FILLED, COASTED LEVER** — 94.0% of ticks at the last tick before a gate
  advance, 97.2% four ticks back (465 confirmed advances across 188 rate-30 flights, measured in
  ticks-before-advance so it needs no range estimate and is immune to the advance seam).

⇒ **The policy has never once been trained on the input state in which every gate outcome is
decided.** And the asymmetry runs **OPPOSITE to my hypothesis**: deploy hands the policy *more*
information in the blind band than training ever did, not less. A replay of v19Ws0/v20Vs0 over
28 936 real observations with slot0 zeroed vs real moves `|Δroll cmd|` by a median **0.47–0.56
rad/s** (p95 2.0) at 1.5–2.5 m — it is the channel the policy reads hardest, exactly there.

🟢 **THE ARM IS ONE TOKEN: `+env.ego_obs_coast=true`** (reaches `peregrine_racing_ego.py:1254`,
drops the `keep &= det` mask). No code change, no warm-start break. 🚩 The 2026-07-09 coast
precedent read neutral, but that stage ran at `ego_noise_scale: 0.0` where the curriculum's own
comment calls the coasted estimate "nearly free (~0.01 m error over 1.5 s)" — a *perfect* lever
through the blackout, which is not the test. The shipped stage runs real noise.
🛑 **Note the deploy knob `ego_obs_coast` already exists and is OFF; do NOT "fix" this by turning it
on at deploy — that would move the WIRE toward training's zeros, i.e. throw information away.
The change belongs in TRAINING.**

⇒ **The terminal scatter is therefore NOT explained by the blind interval alone.** Levers,
re-ranked:
1. **Make the policy better at the blind commit** — reward/termination pressure on terminal L-inf
   SCATTER (not its mean). A legitimate training arm; see [[aim-cohort-and-loop-rate-2026-07-27]].
2. **Shrink the blind interval in TIME** rather than distance — i.e. gate-approach speed. Trades
   against the speed north-star; measure before assuming.
3. **Detect PARTIAL gates** to push the horizon closer than 4-of-8 corners. Note training would
   then be the OPTIMISTIC side, so it must move too, or a new mismatch is created.
4. **Better belief propagation** across the blind interval (`ego_fix_gain`,
   `seeker_propagate_range` are both still at pre-2026-07-25 defaults).
5. Camera FOV / mount pitch — hardware; the +20° up-pitch is why the *vertical* axis binds, worth
   telling whoever owns the mount.

Instruments: `scripts/vision_horizon/last_sighted_range.py` · `scripts/vision_horizon/fov_cutoff.py`.
Related: [[aim-cohort-and-loop-rate-2026-07-27]] · [[gate-strike-last-3m-2026-07-27]].
