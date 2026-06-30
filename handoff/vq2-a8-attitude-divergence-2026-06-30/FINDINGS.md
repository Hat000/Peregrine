# A8 VQ2 attitude-divergence — telemetry localization — 2026-06-30

**Question:** the A8 drone pitches nose-UP + throttle-up + flies up-and-back (footage-confirmed: onboard
camera on the ceiling). The seeker is fine; the bug is a WRONG live ATTITUDE ESTIMATE fed to the
controller. Offline sign-audit found all estimator seams sign-clean, so it is either:
- **(a)** accel-leveling bias under sustained linear acceleration, or
- **(b)** the LIVE VQ2 HIGHRES_IMU gyro polarity/frame not matching the code's FRD assumption.

**VERDICT: (a) — accel-leveling bias under sustained linear acceleration. (b) is refuted.** Both runs,
identical signature. Script: [`extract_attitude.py`](extract_attitude.py); raw output: [`extract_output.txt`](extract_output.txt).

## What's in the recording (and what isn't)
`mavlink.tlog` has **HIGHRES_IMU** (xacc/yacc/zacc, x/y/zgyro) and **ACTUATOR_OUTPUT_STATUS** (4 motors).
It does **NOT** have `ATTITUDE` (the VQ2 wire blocks it — the estimator's reported pitch is *not on the
wire*, and the navigator's internal `_ahrs` quat is in-memory only, never logged) nor `ATTITUDE_TARGET`
(fly_rl's outgoing SET_ATTITUDE_TARGET is not tlogged). So the **estimator pitch is unavailable** — Q2 is
answered by inference, and the controller's intent is read from the **motor sum** (high-rate thrust proxy).
`position_ned` is ignored entirely (VQ2 returns `pos=NO` → estimator fiction).

## The decisive, convention-free test
A gyro measures rotation directly. So compare, over the sustained post-launch flight, the **accel-derived
apparent pitch** vs the **gyro-integrated pitch**:
- If accel-pitch DRIFTS while the gyro reads ~0 → the drift is **linear-acceleration contamination of the
  accelerometer**, not rotation → **(a)**. (A zero gyro has no polarity to be wrong, so this refutes (b).)
- If the gyro shows a sustained rotation whose integrated sign opposes truth → **(b)**.

## Result — both runs, the longest sustained gyro-quiet, throttle-on segment
```
run         segment        accel-pitch drift   gyro-integrated drift   max|ygyro|   mean|a|
run1   t+1.4 .. t+5.6 s         -23.6 deg            -0.8 deg          0.089 rad/s   12.0
run2   t+1.2 .. t+5.7 s         -23.6 deg            -1.1 deg          0.085 rad/s   12.3
```
~24° of apparent-pitch drift with the gyro essentially **flat** (<1° integrated, peak rate <0.09 rad/s).
The per-tick trace makes it unmistakable (run2):
```
 t-lnch   |a|  ygyro  accel_p gyro_intp   sumM
  0.00   14.8  +0.00     +0.1      +0.2   1.39   <- launch, throttle up, levels to ~0
  0.90    1.2  -1.37    +17.7     -21.5   0.41   <- ONE brief real rotation pulse (gyro_int jumps to ~-44)
  1.81   15.7  -0.01     -0.7     -43.8   1.49   <- thereafter ygyro ~= 0 ...
  2.71   13.1  -0.00     -5.3     -43.9   1.49      gyro_int HOLDS flat at -44 (no rotation)
  3.61   11.2  -0.00    -13.0     -43.9   1.49      while accel_p marches DOWN ...
  4.52   10.4  -0.00    -19.4     -43.9   1.49
  5.43   10.0  -0.00    -22.9     -43.9   1.49   <- accel says -23 deg; gyro says it never moved
  5.76  608.0 -31.55    +10.3      -0.3   1.66   <- crash impact (excluded)
```
`gyro_intp` jumps to ~−44° during the single launch pulse, then **holds flat at −44° for 4+ seconds**
(gyro = 0) while `accel_p` drifts 0 → −23°. Over that window `xa` grows 0 → +4 m/s² (sustained forward
specific force) and `|a|` **stays ≈10–12 ≈ g**. So the specific-force vector keeps magnitude ≈ g while its
*direction* tilts ~23° — exactly the case a **magnitude-based accel-gate cannot reject**.

## Answers to the three questions
1. **Accel-tilt vs gyro-tilt sign:** they **DIVERGE**, but not because the gyro inverts — the gyro reads
   **~0** through the whole divergence (integrates to <1° over 4+ s), while the accel-apparent pitch drifts
   ~−24°. The drift is linear-accel contamination of the accelerometer, **not** rotation. → not (b).
2. **Estimator vs IMU-truth pitch:** the estimator pitch isn't logged (VQ2 blocks ATTITUDE), but the IMU
   truth (gyro) says the body is **not rotating**, while the contaminated accel reads progressively
   **nose-down**. An accel-leveling estimator tracks that false nose-down — which is exactly what makes the
   controller command **nose-up**, matching the footage (up + back). Inferred, consistent, direction = the
   estimate reads MORE nose-down than truth (the A8 signature).
3. **Controller fighting it:** motor `sumM` rises to ~2.1–2.4 and **holds** (throttle up / sustained thrust)
   — the sustained thrust is what produces the sustained linear acceleration that contaminates the accel.
   (Commanded body_rate[1] itself is only ~1 Hz in stdout and not in the tlog; the motor sum is the
   high-rate, estimator-independent proxy.)

## Verdict and why the offline audit missed it
**(a) accel-leveling bias under sustained linear acceleration.** The estimate diverges during a window
where the gyro reads ~0 and `|a| ≈ g`, so a magnitude-gated accel-leveler is fooled by a *tilted-but-
g-magnitude* specific force and levels the attitude to the wrong direction. The offline sign-audit was
right that the code is sign-clean — this is not a code-seam sign bug; it's a **physical estimator dynamic**
(thrust → linear accel → tilted specific force at ≈g magnitude → accel-gate evaded) that **self-consistent
synthetic data cannot reproduce**, because synthetic IMU is generated from the true state with no
independent sustained-acceleration contamination.

**(b) is refuted as the driver:** the sustained divergence occurs while the gyro reads ≈0 — a zero gyro
cannot inject a polarity error. (There is ONE real rotation pulse at launch, `gyro_int`→−44°; this analysis
does not independently certify that pulse's polarity because the flight offers no clean low-`|a|` rotation
to cross-check, but it is not the source of the sustained up-and-back divergence.)

## Suggested fix direction (commander's call)
The ESKF accel-gate is **magnitude-only**, so it passes a g-magnitude vector that is mis-oriented by
sustained accel. Candidates:
1. **Thrust-aware accel rejection:** the controller knows its commanded collective → predict the expected
   body specific force and subtract it (or hard down-weight the accel correction) whenever sustained thrust
   implies significant linear accel — don't level to the accelerometer during powered pursuit.
2. **Innovation/direction gating, not just magnitude:** reject/deweight the accel update when its *direction*
   disagrees with the gyro-propagated gravity beyond a threshold, not only when `|a|` departs g.
3. **Lower the accel-correction gain during high-thrust segments** and lean on gyro propagation (which the
   data shows is well-behaved and quiet here).
Validation should use a flight (or a sim case) with **sustained translational acceleration**, not
self-consistent synthetic IMU, or this class of bug stays invisible.

## Artifacts
- `extract_attitude.py` — the extractor (both runs; robust launch/crash windows; longest gyro-quiet segment).
- `extract_output.txt` — full per-tick tables + answers + verdict for both runs.
- Recordings (gitignored, on ShadowPC): `data/runs/20260630_005945_*` (run1), `20260630_010404_*` (run2).
