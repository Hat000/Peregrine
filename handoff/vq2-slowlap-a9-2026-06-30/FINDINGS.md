# VQ2 self-localized SLOW-lap — ATTEMPT 9 (A8 accel-reject fix) — 2026-06-30

**Branch flown:** `claude/loving-galileo-92f020` @ `75ace7d` (acceleration-aware accel rejection +
`vq2_case_c` arms `ahrs_accel_motion_reject=True`). **Sim:** AI-GP 1.0.3379, **R2-TRAINING = VQ2**
(deliberately selected; verified lit-warehouse via onboard red-glow 5517 px). 3 runs (1 fresh GO, 2 RESTART).
New accel-motion-reject tests pass (38/38).

## TL;DR — the fix did NOT work, and the root cause is now (b), not (a)
| | Result |
|---|---|
| A8 accel-reject fix effective? | **NO.** runs 2 & 3 reproduce the exact A8 failure (user-watched): fly forward briefly, then **hard pitch UP, throttle up, fly BACKWARD facing the ceiling.** run1 over-rotated and hit the floor — same mechanism. gates=0, all 3. |
| Loop rate | ~13–16 Hz / CHOKED (33–97 ticks); frame drops 0.0% (perf fix holds). |
| **Root cause** | **(b) the LIVE VQ2 gyro PITCH polarity is inverted vs the code's FRD assumption.** This REVISES the A8 (a) verdict (see below). |

## The decisive evidence: commanded vs realized pitch rate, anchored to the human's eyeball
Estimator-independent, all three runs, over the controlled-flight pitch maneuver (crash impact excluded):

| run | user saw (live) | net pitch | pitch-pulse realized `ygyro` | commanded pitch rate |
|---|---|---|---|---|
| run1 (floor)   | —        | −57° | **−1.49 rad/s** | **+1.50** |
| run2 (ceiling) | nose-UP  | −36° | **−1.43 rad/s** | **+1.50** |
| run3 (ceiling) | nose-UP  | −43° | **−1.43 rad/s** | **+1.50** |

- The controller **commands** pitch rate **+1.50**; the gyro consistently **realizes −1.4** (opposite sign,
  every run).
- The pilot's direct visual read anchors the sign: the nose physically pitched **UP**, and for that nose-up
  rotation the gyro reads **negative**. In standard FRD (which the code/fix assume) nose-up is **positive**
  pitch rate. **=> the live VQ2 HIGHRES_IMU pitch (y) polarity is inverted relative to the code's FRD
  assumption.** (Reproduce: `python gyro_sign_check.py <run_dir>`.)

## Why this produces the runaway, and why the A9 fix couldn't help
1. The drone (commanded +1.50) really pitches nose-up. The **gyro reports it as negative**.
2. The AHRS integrates the inverted gyro → the **attitude estimate pitches the WRONG way** (believes
   nose-DOWN when the body is nose-UP).
3. The controller, seeing "nose-down," commands **more nose-up** to correct → the drone pitches up further
   → gyro reports more-negative → estimate believes more nose-down → **runaway** → nose all the way up,
   facing the ceiling, flying backward. (run1: over the top → floor.)
4. The A8 **accel-reject** fix targets the *downstream* symptom — accel contamination once the drone is
   *already pinned* nose-up (the A8 "gyro-quiet, accel-drifting" phase). It does nothing about the
   **inverted-gyro pulse that drives it there**, so the runaway is unchanged.

## Revision of the A8 verdict (honest correction)
A8 concluded **(a) accel-leveling bias** and stated "(b) is refuted." That was incomplete. The A8 analysis
compared accel-tilt vs gyro-integrated tilt and found the *sustained* divergence happens while the gyro is
quiet — true, but that is the **aftermath**, not the cause. A8 **explicitly flagged that it could not verify
the gyro SIGN** (the flight had no ground-truth rotation to check against). A9 supplies exactly that missing
check — the pilot's live nose-up observation paired with a negative realized gyro — and it points to **(b)**.
The accel contamination (a) is a real secondary/sustaining effect, but the **primary driver is the inverted
gyro (b)**. Lesson reinforced: the human's direct view supplied the one fact the telemetry alone could not.

## Non-determinism note
Same code, three runs, different endpoints (floor / ceiling / ceiling) and different net pitch (−57 / −36 /
−43°). Consistent with a runaway whose final resting attitude depends on the initial perturbation and how
far past vertical it rotates — not a contradiction of (b), just the open-loop runaway landing differently.

## Recommended next step (commander's call; stack NOT touched)
1. **Flip the live VQ2 gyro PITCH sign before it enters the AHRS** (or fix the HIGHRES_IMU body-frame
   adapter for the live wire). The offline sign-audit passed because synthetic IMU is self-consistent in the
   code's assumed convention — it cannot see a LIVE wire-convention mismatch. Validation must use the live
   wire (or a recording), not synthetic data.
2. **Determine if it's a single-axis pitch flip or a full frame handedness mismatch.** These flights were
   ~pure pitch (roll/yaw net ≈ 0), so only the y-gyro sign is established. A short controlled roll and yaw
   test on the live wire would tell whether x/z are also inverted (full frame flip) or only y.
3. Re-evaluate the accel-reject fix afterward — it may still be wanted for genuine sustained-accel robustness,
   but it is not this bug.

## Artifacts
- `gyro_sign_check.py` — per-run: controlled-flight realized `ygyro` (sign + net pitch) vs the commanded
  +1.50, with the pilot's nose-up ground truth. `python gyro_sign_check.py <run_dir>`.
- `fly_run1.log`, `fly_run2.log`, `fly_run3.log` — the three flight stdout logs (commanded `rate=[...,+1.50,...]`).
- Recordings (gitignored, on ShadowPC): `data/runs/20260630_023659_*` (run1 floor),
  `_024248_*` (run2 ceiling), `_024427_*` (run3 ceiling).
- Flight stack untouched; only this handoff dir committed.
