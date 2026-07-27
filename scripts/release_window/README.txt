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
  B. A CLASSIFIER'S RULE ORDER IS A CAUSAL CLAIM. Putting a cheap temporal rule
     first silently relabels every mode that happens to be fast. Rank modes only
     after checking what the earlier rules absorbed.

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
