# COMMANDER independent cross-check — gate-4 margin under cold case-C velocity

Run inline by the commander (opus-4.8) during ULTRACODE-GATE-RELATIVE-PIPELINE-DESIGN, INDEPENDENT of
the d3 workflow agent (written before reading d3's sim). Real `LinearKF` + `RewindKF` + measured noise.
Artifacts: `margin_xcheck_v2.py` / `margin_xcheck_v2_results.json` (the credible sim);
`margin_xcheck.py` (v1, RETRACTED — corner artifact, see below).

## What I tested
The straight g3->g4 binding leg (c1 geometry), perfectly centered (true in-plane offset = 0 at g4),
gate-relative POSITION fixes only (measured lateral sigma 0.265 m/axis, 14 Hz within 12 m, NO map bias,
NO 0.40 m floor), driven through the real RewindKF (OOSM at capture time). The one change vs c1: instead
of seeding velocity to truth (c1's WARM assumption), I sweep a **COLD velocity prior** v0 = uhat*V +
N(0, sigma_v) with NO velocity update during the window (case C is position-only). sigma_v is the
lap-accumulated velocity error entering gate-4 — the load-bearing swing variable.

## Two sim bugs I caught and corrected (honesty trail)
1. **v1 corner artifact (RETRACTED).** A piecewise-linear multi-leg g0->g4 path has instantaneous
   velocity-direction changes at each gate; feeding the KF a constant gravity-comp accel_body across legs
   means it never turns its velocity vector, so the cold arm lagged catastrophically (4-5 m, vel_err ~11
   m/s). That is a sim artifact, not case-C physics (the real policy flies smooth turns the IMU measures).
   v2 reverts to c1's single straight leg + a cold velocity PRIOR instead.
2. **v2a mis-timing (FIXED).** I first generated the fix `z` from the CURRENT-time truth but stamped it at
   capture time t-L. That fed a too-fresh fix to a too-old timestamp -> a v*L along-track error that the
   14deg g3->g4 heading (u_E~0.24) projected onto E, falsely blowing the L=115 ms arm up to ~0.86 m.
   RewindKF was correct; I fed it mis-timed data. With `z` = capture-time truth, L=15 vs L=115 nearly
   match -> **RewindKF compensates latency correctly (independently re-confirms the case-c Piece-B verdict).**

## Headline numbers (N=600/cell, gate-4 in-plane (E,D) error at the plane crossing, V=37 m/s)
| sigma_v (m/s) | L=15ms RMS | L=115ms RMS | p90 (L=115) | p95 | clears 0.155 RMS / p90 |
|---|---|---|---|---|---|
| 0.0 (WARM, = c1)   | 0.132 | 0.136 | 0.204 | 0.228 | YES / **NO** |
| 0.3               | 0.160 | 0.188 | 0.286 | 0.315 | **NO** / NO |
| 0.6               | 0.203 | 0.267 | 0.412 | 0.466 | NO / NO |
| 1.0               | 0.234 | 0.324 | 0.495 | 0.547 | NO / NO |

Speed-flat in RMS for fixed sigma_v (25/30/37 m/s all ~0.13 warm, ~0.26 at sigma_v=0.6): the in-plane
miss is set by the per-fix lateral sigma + sigma_v, NOT speed directly. Speed couples in THROUGH sigma_v
(less time to converge velocity) and through per-fix blur inflation (HELD FIXED here = the modeled-not-
measured caveat — all sigma are best-case lower bounds at 37 m/s).

## Verdict (corroborates AND sharpens the prior CONDITIONAL-GO / straddle)
1. **Warm baseline corroborated:** ~0.13-0.14 m RMS = c1's 0.139. The gate-relative fix + RewindKF is the
   right architecture and works in principle.
2. **SHARPENING (the c1 single-number headline understated this):** even WARM, the **p90/p95 tail
   (~0.20-0.24 m) is OVER the 0.155 m margin** at every speed. A contact margin is a worst-case gate, not
   an RMS gate, so "RMS 0.139 < 0.155" overstates the clearance. Honest read: **RMS straddles, the tail
   exceeds, even at the best (warm) velocity prior.**
3. **The verdict pivots on sigma_v** (velocity-prior error entering gate-4): at sigma_v ~ 0.3 m/s the RMS
   itself crosses the margin; at sigma_v >= 0.6 it is decisively over on both RMS and tail.
4. **RewindKF is NOT the binding term** — latency compensates (L=15 vs 115 differ only by the cold-velocity
   re-propagation residual). The binding term is sigma_v.
5. **sigma_v in real case C cannot be pinned offline** — it depends on at-speed accel-bias/attitude realism
   and the velocity-acquisition method (IMU-only vs position-fix-differencing vs vision-velocity). The
   bounding argument (attitude-bias phantom accel ~0.24 m/s^2 over the coast, partially corrected) admits
   sigma_v anywhere from ~0.1 (well-corrected, clears RMS) to ~0.6+ (poorly corrected, decisively over).
   **=> ESCAPE HATCH: margin closure at 37 m/s is genuinely UNSETTLED offline; it requires the ShadowPC
   at-speed (~37 m/s) gate-4 vision recording to measure (a) the real gate-relative per-fix lateral sigma
   under 37 m/s blur and (b) the achievable sigma_v entering gate-4.**

## Implication for the blueprint
The gate-relative pipeline is the RIGHT build (necessary; the absolute path is hopeless). But the
speed-ladder portfolio must treat the gate-4 margin as a WORST-CASE (p90/p95) gate and SELECT THE SPEED
RUNG where the achieved-at-that-speed gate-relative per-fix sigma + sigma_v keep p90 < 0.155 m — which on
this evidence is likely BELOW 37 m/s until the live data + a tighter velocity prior (lower per-fix sigma
via sub-pixel corner refine, and/or a position-fix-difference/vision-velocity assist) are in hand.
