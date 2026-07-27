---
name: vision-horizon-fov-2026-07-27
description: "ROOT CAUSE of the gate-side slam: the drone is BLIND for the final ~1.8 m because the gate overflows the 58.7 deg VERTICAL field of view. Zero of 899 confirmed passes ever got a fix inside 1.19 m. Structural — not tunable."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T06:24:31.013Z
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

**NOT FIXABLE BY TUNING.** The candidate responses, in order of expected value:
1. **TRAIN THE POLICY TO COMMIT BLIND** — if training feeds an accurate lever to 0 m while the wire
   goes blind at 1.8 m, the policy has never experienced the situation that kills it. That is a
   `rate`-class train/deploy contract mismatch. **VERIFY BEFORE ACTING** (this is the open question
   handed to the training-contract workflow).
2. **DETECT PARTIAL GATES.** The detector needs the whole gate; a detector that fires on bars or
   corners would keep tracking closer. Vision-training change, not a knob.
3. **IMPROVE THE PROPAGATED BELIEF** over the blind interval (`ego_fix_gain`,
   `seeker_propagate_range` are both still at pre-2026-07-25 defaults).
4. Camera FOV / mount pitch — hardware, likely unavailable, but the +20° up-pitch is why the
   *vertical* axis binds and is worth stating to whoever owns the mount.

Instruments: `scripts/vision_horizon/last_sighted_range.py` · `scripts/vision_horizon/fov_cutoff.py`.
Related: [[aim-cohort-and-loop-rate-2026-07-27]] · [[gate-strike-last-3m-2026-07-27]].
