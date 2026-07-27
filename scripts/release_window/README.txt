RELEASE-WINDOW / "LAUNCH DIVE" REFUTATION  (2026-07-27)
=======================================================

WHAT THIS IS
    The instrument that was built to design a v2.1 training arm against the
    census's #1-ranked target -- "release-dive retrain", 20.1% of 572 deaths --
    and instead REFUTED it. No arm was launched. Nothing here changes training.

    Run order (each reads the flight corpus READ-ONLY; needs numpy only):
        release_discrim.py     -> release_rows.json
        where_do_they_die.py   -> g0_deaths.json
        arm_level_test.py      (consumes both)
        validate_and_replace.py
        speed_seamfree.py

    Corpus: C:\Users\Fengy\Downloads\Projects\wt-arrest\data\runs  (573 sessions,
    563 parseable). Edit RUNS at the top of each file if it moves.

THE CLAIM UNDER TEST
    docs/failure-profile-2026-07-27/PROFILE.md ranks "release-dive retrain" #1:
    20.1% exposure, "highest confidence bucket", "universal nose-down command at
    release (min pitch cmd p50 -1.07 rad/s)". v2.0 regressed that command ~2x by
    open-loop replay (v20Vs0 mean -1.064 vs v19Ws0 -0.547) and was held back partly
    on that basis.

WHAT THE MEASUREMENTS SHOW
  1. THE BUCKET IS NOT A LAUNCH MODE. Its definition (classify.py rule #1) is purely
     temporal -- `t_end <= 2.5 s AND max_gate == 0` -- and it is the FIRST rule, so it
     pre-empts every mid-leg and at-gate rule. Measured: gate 0 sits at range p50
     10.70 m at release; these flights die at range p50 1.50 m, 81.2% within 3.0 m
     (v19 subset: 100%, p90 1.90 m). They fly the whole leg and die AT GATE 0.
     81.2% would classify at-gate under the profile's own 3.0 m cut.
     Its "high confidence" label (93/115) is `t_end <= 2.2` -- a restatement of the
     clock that defined the bucket, not evidence about the mode.

  2. THE BUCKET'S SIZE IS EXPOSURE, NOT DIFFICULTY. Gate 0 has the BEST conditional
     survival of any gate (72.6%; g1 65.8, g3 56.1, g5 19.6) but is attempted by all
     563 flights, so it collects the most absolute deaths (142).

  3. THE NAMED MARKER DOES NOT DISCRIMINATE. Four independent tests:
       flight level, within v19 : deaths -1.379 vs survivors -1.378 (p50, worst
                                  nose-down cmd in the 4 ticks after release);
                                  tails identical (MIN -1.385 vs -1.387)
       deep/shallow, within v19 : 22.2% vs 19.4% gate-0 death (n=36/36)
       arm level, 6 lineages    : corr(median dive, gate-0 death rate) = +0.338
                                  -- the WRONG SIGN. v19 dives hardest and dies
                                  least (20.8%); v1 never dives and dies most (53.3%).
       dose-response, n=563     : dive <= -1.2 (v2.0's territory) 22.2% vs
                                  shallower 25.9%, z = -0.76. Deepest band
                                  (<= -1.5, n=20) is the LOWEST at 20.0%.
                                  Deep-diving flights reach MORE gates (2.11 vs 1.75).
     And the realized motion at release is CLIMBING, not descending -- consistent
     with the profile's own caveat that "the policy asks for a dive; the fence and
     the thrust mostly prevent it, and the drone dies anyway."

  4. THE WIRE EVIDENCE IS SIX FLIGHTS. 3/6 v2.0 flights dying at gate 0 has
     p = 0.11-0.17 against the 21-25% baseline. Not a signal.

  5. CORROBORATED BY THE SIMULATOR, NOT BY VISION. For the gate-0 deaths dying
     <=3 m out, the sim's own collision counter reads >=1 on 100% of them (mean
     9.9). They collide at short range on the gate-0 approach; they do not depart
     at launch. This uses no vision quantity at all.

  6. THE MARKER IS A WEATHERVANE. The recorded v1.9 -> v2.0 delta is EXACTLY six
     override lines (ego_vision_frame_hz 30->25, +course_drop_lo/hi,
     +course_gates_ceiling, cadence seed, init_from, runname) -- NOT ONE of which
     touches pitch. Yet the release pitch command moved ~2x (-0.547 -> -1.064). An
     output that swings 2x on six lines that never mention it is unpriced and
     wanders with any training change. Confirmed from the rendered configs:
     rw_att_pitch and att_pitch_limit_rad are ABSENT from all of v20V_s0, v20V_s1
     and v19W_s0 .hydra/config.yaml -- v2.0 prices pitch AMPLITUDE (rw_pitch_duty
     0.2 beyond a 0.6 band) and pitch JERK (0.03), both sign-blind, and nothing
     else.

TWO FOOTGUNS THIS WORK ESTABLISHED (both nearly banked a false result)
  A. THE GATE-SEAM TELEPORT MANUFACTURES SEPARATION. Measuring approach speed over a
     window that straddles the RACE_STATUS advance gives PASSES a spurious negative
     closing speed (p10 -28.8 m/s) because the seeker drops the passed gate and
     rel_flu jumps to the next one. That artifact alone produced AUC 0.723 for
     "speed kills at gate 0" -- matching the profile's best real separators.
     Measured seam-free over a fixed 6->3 m range band (speed_seamfree.py), the
     effect VANISHES: AUC 0.565, died p50 8.12 vs passed 8.60 m/s, and the
     conditional death rate FALLS with speed (30.4% at 4-6 m/s -> 20.8% above 10).
     Any approach-speed statistic must be measured over a fixed range band, never
     over a window that can contain an advance.

     WHY THE 6->3 m BAND IS SOUND, stated so a future reader does not downgrade a
     proof to an unswept assumption: the defence is STRUCTURAL, not empirical.
     Both edges lie strictly BEFORE the gate plane, so a RACE_STATUS advance
     CANNOT occur inside the window for either group -- the artefact is ruled out
     by construction rather than shown not to bite at a chosen cut. A structural
     guarantee is STRONGER than a threshold sweep; you sweep when you have no such
     guarantee. (Point made by d1-velocity, and correct -- I had understated my own
     evidence.)
     THE RESIDUAL: that argument covers the ADVANCE seam only. It does NOT cover
     the broader mis-lock seam (the seeker re-locking onto a different gate
     mid-approach while the index holds). That one is caught here by a separate
     guard -- windows are rejected if any consecutive-tick lever jump exceeds
     1.5 m -- which gates on the VISION LEVER ALONE and so does not touch the
     variable under test.

     THAT 1.5 m GATE IS NOW SWEPT AND MEASURED (d1-velocity, 6c3587fe, 43,885
     same-gate consecutive fix pairs; leak = teleport-class events MISSED, reject
     = share of all pairs discarded):
         |d|/dt > 20 m/s  (a speed gate)   leak 1.45%   reject 9.76%
         |d|    > 1.5 m   (this one)       leak 0.14%   reject 3.72%   <-- best
         |d| > 20*dt + 1.0 m               leak 4.78%   reject 2.58%
         |d| > 15*dt + 1.0 m               leak 2.03%   reject 3.00%
     ~10x less leakage while discarding 2.6x LESS good data. Sweep of the distance
     itself: 1.0 -> 0.00%/7.59%, 1.5 -> 0.14%/3.72%, 2.0 -> 0.60%/2.50%,
     3.0 -> 11.3%/1.49%. 1.5 m is the knee.
     MECHANISM, which is the transferable part: DIVIDING BY dt DILUTES A TELEPORT
     that lands across a long detection gap, while its DISTANCE stays large.
     Teleport-class apparent speed has p5 = 28 m/s -- that slow tail is exactly
     what a speed gate lets through. A DISTANCE gate has no such tail, because a
     re-lock is bounded below by the gate spacing. Corroborating the value: honest
     inter-fix motion is bounded by corpus p99 speed 14.2 m/s over the p99 fix gap
     124 ms = 1.76 m.
     HONEST PROVENANCE: 1.5 m was chosen here as a physically-motivated bound on
     one tick of real motion (~0.33 m at 10 m/s and 30 Hz, so ~4.5x margin). It
     was NOT derived from the p99xp99 calculation above, and it was not swept by
     me -- d1-velocity did both, post hoc. Right variable, roughly right value,
     less rigour than the number now carries.
  B. A CLASSIFIER'S RULE ORDER IS A CAUSAL CLAIM. Putting a cheap temporal rule
     first silently relabels every mode that happens to be fast. Rank modes only
     after checking what the earlier rules absorbed.

THE RULE THAT GENERALISES BOTH (and two other 07-27 withdrawals)
    Premises that failed on this project recently, with what "established" them:
      "release-dive is the #1 mode"   <- classify.py rule #1, t_end <= 2.5
                                         => measured WHEN a flight ended, not WHAT killed it
      "speed kills at gate 0" AUC .72 <- a speed window containing the advance
                                         => measured the GATE SEAM, not the drone
      "no launcher arms ego_faithful" <- a grep over *.sbatch + launch_v1[6789].sh
                                         => measured which FILES MENTION a flag, not
                                            which CONFIG RAN (the stage dict in
                                            rl/vq2_ego_curriculum.py:1390 owns it)
      "obs lateral vel carries no info" <- a lever difference without de-rotation
      "vertical undershoot dominates"   <- rel_flu[2] read at +23.8 deg pitch
    NONE of these was a wrong number. Every one was a CORRECT number carrying a
    mechanism claim it could not support.

      => BEFORE ADOPTING A PREMISE, NAME THE ARTIFACT THAT ESTABLISHED IT, AND
         CHECK THAT THE ARTIFACT MEASURED THE MECHANISM RATHER THAN A CORRELATE.

    "Adjudicate from a run's own .hydra/overrides.yaml, never the launcher source"
    and "check what a classifier's rule actually tests before ranking on its
    output" are corollaries, not separate lessons.

  RANKING OF EVIDENCE, in the order you should prefer it:
      1. RIGHT VARIABLE       -- the quantity is chosen against the physics
      2. RULED OUT BY CONSTRUCTION -- the artefact cannot occur in your window
      3. HOLDS ACROSS A SWEPT THRESHOLD
      4. HOLDS AT THE CHOSEN VALUE
    Do not let a reflex to self-criticise demote (2) to (4) -- a structural
    guarantee is STRONGER than a sweep; you sweep when you have no guarantee.
    And (1) DOMINATES (3), which is the counter-intuitive one and was measured,
    not asserted: a swept 20 m/s SPEED gate leaked 1.45% of teleports while
    discarding 9.76% of good data; an unswept 1.5 m DISTANCE gate leaked 0.14%
    while discarding 3.72%. The sweep was thorough and the variable was wrong,
    because dividing by dt dilutes a teleport across a long detection gap while
    its distance stays large. A SWEEP TELLS YOU WHETHER A THRESHOLD IS ROBUST; IT
    TELLS YOU NOTHING ABOUT WHETHER YOU PICKED THE RIGHT VARIABLE, and it can
    launder a bad variable into a confident-looking result.
    (Established by d1-velocity, 6c3587fe, over 43,885 fix pairs -- they set out
    to find a leak in the distance gate and found the opposite.)

  COROLLARY THAT BIT TWICE: THE GUARD YOU REACH FOR CAN BE THE SECOND ARTEFACT.
    Having found the seam, the obvious fix is to reject windows where the vision
    displacement and the dead-reckoned displacement disagree. That gates on the
    KF velocity UNDER TEST: it keeps exactly the windows where dead reckoning
    already agrees, flatters the control arm, and can invert the verdict.
    (d1-velocity, 8d5badb9 -- gated on vision+gyro apparent speed ALONE instead,
    and pinned it with a test feeding a drone whose DR velocity is wrong by
    9.5 m/s but whose lever moves smoothly, asserting the window is still used.)
    Same shape as the AUC 0.723 above: a measurement artefact and the guard
    against it are both artefacts until each is validated independently of the
    quantity being measured. NEVER build a data-quality gate out of the variable
    under test.

  AND THE SEAM IS BIGGER THAN gate_index. Breaking windows on a gate_index change
    is NOT sufficient: the seeker can re-lock onto a DIFFERENT gate while
    RACE_STATUS still reports the OLD index, so gate_index looks constant across a
    landmark change. Measured: 1.57% of SAME-index consecutive fix pairs move >3 m
    more than any plausible velocity explains (p99 4.4 m, max 29.1 m)
    (d1-velocity, corpus). Gate on APPARENT SPEED, not on the index.

WHAT SURVIVES, AND WHERE TO GO INSTEAD
    The deaths are real; only the mode name and the proposed lever are wrong. Those
    that die close (<=3 m) read INSIDE the 1.5 m opening at the last observable tick
    (levelled |lat| p50 0.27 m, |vert| p50 0.35 m) -- the profile's own terminal-
    blindness signature. That points at the census's #2 target (close-in coast +
    bbox-range preference / keeping the recorder alive past impact), not at a
    release-pitch reward term.

INSTRUMENT VALIDATION (validate_and_replace.py, the project's standard)
    On 323 CONFIRMED gate passes (a RACE_STATUS advance certifies |miss| < 0.75 m):
    levelled |vert| > 0.75 m false-positive 4.0% (the profile's VG instrument: 6.3%);
    raw body-frame control 5.0% on this near-plane subset.
    CAVEAT, stated because it bounds the vertical read: this levelling is a simple
    roll/pitch unrotation of rel_flu, NOT the profile's gyro-propagated VG fit. It
    leaves residual pitch coupling -- corr(levelled vert, range*sin(pitch)) = +0.444
    against the profile's +0.112. So the levelled MISS numbers here are indicative
    only. THE REFUTATION DOES NOT REST ON THEM: range-at-death, range-at-release,
    closing speed and the commanded pitch rate are all frame-independent (vector
    norms and a logged command), and every discrimination test above uses only those.
