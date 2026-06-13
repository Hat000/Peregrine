# SHARED CONTEXT — Gate-Relative Case-C VQ2 Pipeline DESIGN (ultracode)

You are a worker agent on project **Peregrine** (Anduril AI Grand Prix autonomous drone racing).
Repo root: `C:\Users\Fengy\Downloads\Projects\Anduril`. HEAD = `186a692`, branch `main`, 692 tests green.
Address the user as **Fengyou** if you surface anything user-facing (you usually won't — your output
is structured data for the commander).

## MISSION (the whole workflow)
Produce the **DESIGN BLUEPRINT** for the gate-relative case-C VQ2 pipeline — the convergent workstream
the project now rests on. This is **DESIGN + offline validation, NOT the build**. Five prior ultracodes
established: the binding VQ2 risk is the **ESTIMATOR**; absolute world-frame vision nav is **NO-GO at any
speed**; **gate-relative observation is the only fix** (CONDITIONAL-GO, velocity-prior-sensitive). We
architect the full pipeline coherently: detector → PnP → RewindKF → gate-relative pose+confidence → the
inc8 policy (retrained gate-relative) → CTBR control.

## HARD SAFETY RULES (every agent)
- **DESIGN + OFFLINE PROTOTYPE/SIM ONLY.** No live sim. No SLURM. **No edits or commits to `src/` or
  anything under version control on `main`.** No memory edits (do NOT touch `memory/`).
- Write ALL artifacts under `handoff/ultracode-gate-relative-pipeline-design-2026-06-13/`. Use a filename
  prefix matching your agent label (e.g. `d1_obs_*.py`, `v3_margin_*.md`).
- You MAY read anything, and run offline Python in the existing venv. You may build on / import the prior
  prototypes (read-only) but copy any code you modify into the new handoff dir — do NOT edit the originals.
- This is a PLAN. Recommend; do not build the production chain. Prototypes are for VALIDATING the design.

## HOW TO RUN OFFLINE PYTHON (Windows)
- Venv: `.venv\Scripts\python.exe`. From repo root. Most prototypes self-insert `src/` on `sys.path`;
  if not, run with `PYTHONPATH=src` or insert `sys.path.insert(0,'src')`.
- The KF / localization / frames stack imports WITHOUT torch (numpy+scipy only). `diffaero`/`pytorch3d`
  exist ONLY on the training cluster — the policy obs *math* is mirrored torch-free in `rl/fly_rl.py`.
- Reproduced-working example (the load-bearing margin sim): from repo root,
  `.venv\Scripts\python.exe handoff/ultracode-estimator-racespeed-2026-06-13/c1_gate_relative.py`
  → reproduces rel-arm in-plane RMS **0.139 m** (clears 0.155 m margin), p90 0.203 m (over). CONFIRMED
  reproducible by the commander 2026-06-13.

## THE LOAD-BEARING FACTS (re-derive, don't just trust)
- **Gate-4 contact-true in-plane margin = 0.155 m @ r=0.38** (E lateral + D vertical in-plane; N along-track).
  This is THE primary margin guard. Gate-5 clean (0.314 m). Faster speed worsens the margin/σ ratio.
- **Absolute world-frame case-C nav = NO-GO at any speed:** deployable in-plane ~0.55 m (≈3.5× margin),
  floor-dominated + SPEED-FLAT (invalid even at 8 m/s). Binding terms: per-fix VARIANCE floor (~0.50 m/axis,
  velocity unobservable in case C → KF averages only ~3–9 fixes) + un-filterable per-track BIAS (range-
  collapsing: ≥0.19 m near-band, 0.11 m at last fix).
- **GATE-RELATIVE observation = THE FIX:** observe the offset to the **SEEN gate-4 opening** (−L from PnP),
  which removes the per-track map/registration bias EXACTLY (`db` drops out of the arithmetic). "Subtract
  gate_map pos" is a proven ANTI-PATTERN (it re-injects the bias). AUGMENT the absolute KF, don't replace.
- **The straddle (make-or-break):** gate-relative in-plane ≈ **0.11 m RMS warm (lap-converged) velocity
  prior / 0.17–0.21 m cold prior** — straddles the 0.155 m margin. Velocity is the swing variable. Case-C
  velocity is observable ONLY through position-fix differencing (vision is position-only). CONDITIONAL-GO.
- **RewindKF = DEFAULT** (in-loop latency L ≈ 115 ms median CPU / ~15–25 ms GPU; covers both eval-HW cases).
  Horizon must be sized > L or it inverts (drops all fixes, diverges to ~21 m). HARD-blocked on TIMESYNC.
- **Measured noise model (DR doctrine — train DR on MEASURED error classes, NOT arbitrary noise):**
  VISION-PKG2 world-fix σ≈[0.73, 0.47, 0.29] m (N,E,D); accepted-fix bias [−0.42,+0.06,−0.28] m; +0.67 m
  systematic N-bias (corroborates gate-relative — per-track world bias drops out). Gate-relative per-fix
  LATERAL per-axis σ ≈ 0.265 m (lat_rms 0.375) at the near band (measured, accepted maha≤16.27 4-corner).
  Detector is sub-pixel (reproj p50 0.50 px) → in-plane noise is attitude-lever(1.4°)+floor limited, NOT
  pixel-limited. NEVER reward damping. Select on ≥5-seed generalization.
- **The 3 P0 case-C bugs (from the vision-case-c report):**
  1. `navigator.py:255-271` `_initialize` reads `ds.position_ned` with NO `use_given_position` guard →
     every case-C "test" is secretly case A (hidden GT seed). True case C: position_ned=None → origin seed,
     pos_std=5.0. THE gate to validating anything.
  2. TIMESYNC: `frame.sim_time_ns` (server UNIX epoch) vs `DroneState.sim_time_ns` (IMU sim-boot epoch) are
     distinct + unreconciled. HARD prereq for RewindKF (else indexed by a garbage epoch); already corrupts
     `time_since_vision_update_s` (`navigator.py:426`).
  3. Velocity unobservable in true case C (vision position-only; vel = IMU integration of accel_body only).

## THE P4-C05 FOUNDATION (already shipped, 692 green — build on it)
- `rl/fly_rl.py` now has `GateMap` / `make_gate_map(gate_pos, gate_yaw)` + `obs_from_zup(..., gate_map=)`.
  The yaw-aware path reproduces `rl/peregrine_racing.py get_observations` BIT-EXACTLY (`get_gate_rotmat_w2g
  (gate_yaw[tg])`). Default `gate_map=None` = exact VQ1 yaw=π specialization (bit-exact). This is THE
  foundation hook for the gate-relative estimator rebuild — same code path, same rotation matrix.
- The 17-dim obs is ALREADY gate-relative: `pos_g = R_w2g @ (gate_pos − pos)`, `vel_g = R_w2g @ vel`,
  `rpy_g`, body_rates, `prev_normed_thrust`, next-gate lookahead, next_relyaw. In case A/B `pos` is the
  given/KF pose & `gate_pos` is the MAP centre. **The crux of the unification:** in case C the policy must
  consume `pos_g` derived from the SEEN gate (PnP −L lever) — i.e. the estimator delivers gate-relative
  offset directly — NOT `R_w2g @ (gate_map − p_KF_absolute)` (the anti-pattern that re-injects map bias).
- 🚩 **Current deployable stack is VQ1-only:** three loud guards (`_assert_vq1_constants_consistent`
  import-time, `_assert_live_course_is_vq1` deploy-time, `assert_gate_map_allpi`) fail-loud-abort on any
  non-π/VQ2 course. Intended + correct; the gate-relative rebuild removes them. Judged path = `rl/submit_rl.py`.

## INC8 RETRAIN INPUTS (from the S2 / planning report)
- S2 DECIDED = `staged_monolithic_then_decomposed`. Speed gap = TILT ENVELOPE, not architecture.
- inc8 sub-tasks: (1) graft arc-length progress reward over a REBUILT corrected-aero contact-safe line
  (`rl/reference_line_vq1.json` is drag-infeasible + 170° inverted → must be rebuilt; `reference_line.py`
  loader valid); (2) envelope relaxation rw_tilt 96→48 then free-cone 60→~70°; (3) BSR3 spin-margin gate
  MANDATORY before retrain (widen spin_rate_abort→~9–10, spin_time_abort→3.0 s); (4) ≥5 seeds (narrow
  basin: inc7 was 2/3 viable); (5) hybrid-cast arc-length reward.
- Selection metric = the **speed-ladder portfolio**: fastest contact-valid policy the gate-relative
  estimator can SUPPORT. `rl/contact_true_eval.py` is the instrument (gate-4 0.155 m margin guard,
  S_stable, per-gate margins, D-offset probe; already has per-gate `gate_yaw` support).
- The reward contract lives in `rl/peregrine_racing.py` (R1–R7, T1–T4, contact-true geometry). NEVER reward
  damping; honest contact geometry + structured DR.

## TWO COMMANDER-ENDORSED DESIGN INPUTS TO EVALUATE
- (i) **UNCERTAINTY-AWARE OBS** — feed the KF per-tick position covariance/confidence into the policy obs
  so it can be fast-when-confident / careful-when-uncertain (HIGHEST leverage for the speed-vs-validity
  coupling). Needs a CALIBRATED covariance + a SPEED-PRESERVING reward (NOT the rejected logstd-as-
  uncertainty). This expands the 17-dim obs — design the new channel + how the policy/critic consume it +
  how training DR exercises variable confidence.
- (ii) **MEASURED-ERROR DR** (above) — train on the measured error classes, not arbitrary noise.
- Cheap ablation to note: all-gate look-ahead vs next-gate-only.

## PRIOR ARTIFACTS YOU BUILD ON (read-only; copy if you extend)
- `handoff/ultracode-estimator-racespeed-2026-06-13/` — `c1_gate_relative.py` (THE fix prototype: abs/submap/
  rel arms), `c5_speed_coupling.py`, `a1_sim.py`, `a2_bias.py`, `REPORT.md`, `FACTS.md`.
- `handoff/ultracode-vision-case-c-2026-06-13/` — `kf_rewind_buffer.py` (RewindKF OOSM, bit-exact),
  `range_anisotropic_R.py` + `range_R_coeffs.json`, `latency_harness.py`, `vision_cal.py`, `REPORT.md`.
- `src/racer/navigator.py` (KF chain + 3 P0 bug sites), `src/racer/state_estimator.py` (LinearKF),
  `src/racer/localization.py` (`gate_pose_to_world_position`, FIX_COV_FLOOR_STD), `src/racer/frames.py`.
- `rl/fly_rl.py` (GateMap/obs builder), `rl/peregrine_racing.py` (get_observations + reward), `rl/
  contact_true_eval.py` (selection instrument).

## OUTPUT DISCIPLINE
Return STRUCTURED data per your schema. Put long prose / specs in markdown files under the handoff dir and
reference them by path; keep the structured return tight. Flag every load-bearing claim and HOW it was
verified (re-derived number? ran a sim? read the source?). Distinguish MEASURED vs EXTRAPOLATED vs ASSUMED.
