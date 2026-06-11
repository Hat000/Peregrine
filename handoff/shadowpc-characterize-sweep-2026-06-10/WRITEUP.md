# Characterization sweep: windup curve + anomaly boundary (ShadowPC, 2026-06-10/11)

**Session: SHADOWPC-CHARACTERIZE-SWEEP.** Two payloads gating the 2nd-order integration +
env redesign (memory `project_phase2_rl_vision_decisions` "NEXT CHEAP EXPERIMENT"):
(1) map the inner-loop amplitude dependence from a controlled magnitude sweep instead of
the n=1 tumble (cc6921d); (2) map where the sim's "±180° yaw-spin anomaly" engages.

**Both payloads landed, and both REWRITE the cc6921d picture:**

1. **The inner loop is NOT PI-windup. It is a STATIC amplitude-dependent gain map**
   (RC-rates / "super-rate" style) in front of a fast, well-damped (~critically damped at
   saturation) loop. Measured sustained gains: **2.50 → 3.50/3.53 (roll/pitch) from
   |cmd| 0.3 → 3.14**. The S1.2 "9.7 rad/s vs 7.85 ceiling" was unmodeled DC gain, not
   transient overshoot. cc6921d's "G=3.8 rejected, DC anchored at 2.5" was the wrong call —
   the optimizer was seeing the real saturated gain.
2. **There is NO open-loop physics anomaly.** The drone rotates rigid-body-clean through
   full inversion at every rate we could command (up to 11.2 rad/s sustained, single-axis
   AND the exact S1.2 tumble command vector). The S1.2 "blew through ±180° tilt and
   yaw-spun" decomposes into: (a) the unmodeled super-rate gain (real divergence driver),
   (b) an Euler-representation artifact (yaw flips wildly near inversion while the
   quaternion is smooth), (c) **a gate-post COLLISION at sim_t≈11.145 (threat 2) that
   imparted −63 rad/s roll** — the "spin" was contact dynamics.
   **⇒ The planned "crash-termination at the measured anomaly boundary" should be a
   COLLISION-based termination (DiffAero already has it), NOT a tilt threshold. There is
   no anomaly boundary to encode. VQ2 open item (A) likely dissolves the same way.**

Reproduce: `fit_sweep.py` (model-free tables + per-mag LTI fits), `fit_windup.py`
(PI-windup hypothesis — superseded but kept: it documents why windup was rejected),
`check_fits.py` (trace overlays + tumble cross-eval), `static_gain_curve.py` (sustained
gains + the tumble post-90° / collision re-examination), `refit_sat.py` (saturated LTI
with the measured gain), `anomaly_analyze.py` (anomaly-run divergence scan).
All from repo root with `.venv\Scripts\python.exe`. The tumble parts need
`C:\Users\Shadow\Downloads\stage1_inc1_actor.pth` (policy-replay command reconstruction).

## 0. Harness (durable; in `scripts/rate_sysid.py`, this commit)

- **Unattended chaining**: `_wait_fresh_go` = fly_vq1 fresh-GO mechanics + escalating
  recovery: every `--reset-after` (8 s) fire MAV_CMD 31000; after two ineffective resets,
  Win32-focus the `AI-GP` window + Enter ×2 (`_kick_sim_from_home`). 15+ probe runs were
  chained with zero GUI touching.
- **🚩 NEW SIM FACT**: after ~90 min idle post-race, the sim parks on an off-race screen
  where **31000 is a no-op even though telemetry still streams** (stale RACE_STATUS
  `started=True`, frozen sim_time; 22×31000 ignored). This is a third state beyond
  HOME/no-telemetry. The window kick recovers it (fresh countdown in 0.2 s).
- **Velocity-damped station-keeping hold** (`--vel-damp`, `--pos-pull`): the pure attitude
  hold ratchets horizontal velocity (each step's tilt = a kick only drag τ≈5 s removes →
  18 m drift abort). Mapping measured from flight data: `a_body ≈ −g·reported_tilt`
  (corr −0.95..−0.99) ⇒ damping tilt target = +(c/g)·(v_body − v_des). Probed axes stay
  pure open-loop.
- **`--mode anomaly`**: init-level → climb (altitude headroom) → ONE sustained open-loop
  step, multi-axis capable (`--axes roll,yaw --mag 3.14,3.14`), zero command on
  uncommanded axes, tilt abort off, collision/position/altitude/time aborts kept.

## 1. Payload 1 — the amplitude curve (15 runs, `data/runs/20260610_22*_sweep_*` + `20260611_02*_anom_*`)

### Static gain curve (sustained rotations; the n=1 → n=11 upgrade)

| \|cmd\| (rad/s) | roll g | pitch g | yaw g | super-rate `2.5/(1−0.30·|c|/3.14)` |
|---|---|---|---|---|
| 0.30 (0.8 s steps) | 2.505 | 2.506 | 2.231 | 2.57 |
| 0.50 | 2.555 | — | — | 2.63 |
| 1.00 | 2.712 | 2.712 | 2.405* | 2.76 |
| 2.00 | 3.043 | — | 2.705* | 3.09 |
| 3.14 | **3.499–3.542** | **3.523–3.543** | 3.10–3.16 (tumbling) / 2.35 (level)* | 3.57 |

Roll ≡ pitch to 3 digits. The one-parameter super-rate form fits within ~3% (slightly flat
at the low end). DC structure for the plant: `target = g(|cmd|)·rate_sign·cmd`,
`g(|c|) = G0/(1 − s·|c|/π)`, **G0 = shipped [2.501, 2.504, 2.231], s ≈ 0.29–0.30**
(roll/pitch; refit s on the table above if a better form is wanted).

*Yaw caveat: yaw's gain is maneuver-dependent — yaw-only at level attitude plateaus
~2.35–2.7 (looks like a ~7.4 rad/s cap), but while tumbling it tracks ~3.1 (and the S1.2
tumble reconstruction is consistent with ~3.16). Yaw was always the worst-modeled axis;
treat s_yaw ≈ s_roll with the level-attitude cap as a caveat, or give yaw its own pass.

### Transient on top of the static map

- Model-free, in-step (`fit_sweep.py` raw table): rise t80 ≈ 31–52 ms at every magnitude;
  peak/sustained ≈ 1.10 at 0.3 → 1.02 at 3.14 (the big "peaks" vs the OLD 2.5 anchor —
  1.45× at 3.14 — were the static gain, not overshoot).
- Saturated LTI re-fit with the measured gain (`refit_sat.py`): **roll wn=50.3 ζ=0.84,
  pitch wn=82.7 ζ=1.04 — overshoot 0–1%, τ_eq = 2ζ/wn ≈ 25–33 ms** (vs shipped τ=19 ms).
  Small-signal (0.3): wn≈71–75, ζ≈0.48–0.53 (~10% overshoot, real but small).
- Slew limit: saturated rise ≈ 250–280 rad/s² (roll/pitch), ~78 (yaw, level).
- Input delay: 5–15 ms (don't double-count with the S1.3 transport-delay DR).

### Why PI-windup was rejected (the cc6921d "amplitude-dependent ωₙ/ζ" story)

`fit_windup.py` fit a PI+integrator-clamp+torque-limit mechanism jointly across all
magnitudes: it matches the per-mag LTI fits AND reproduces the tumble peaks (9.57/9.58/8.03
vs measured 9.77/9.72/8.14) — but it predicts the saturated response DECAYS back to the
2.5-gain target, and the sustained-rotation probes show **no decay over 1.2+ s** (11.12
rad/s flat). A static map + fast loop explains everything the windup model did, plus the
sustained data. The cc6921d two-regime (ωₙ 21–28, ζ 0.39–0.47) fit was the LTI shadow of
the wrong DC anchor: per-mag LTI fits do not cross-generalize (sweep-3.14 params on the
tumble: RMSE 6.5 vs its own fit 0.25, and vice versa) — nonlinearity misattributed to
dynamics lands on whatever maneuver you fit.

## 2. Payload 2 — anomaly boundary: THERE ISN'T ONE (open-loop)

Sustained rotations through full inversion, tilt abort off (`*_anom_*` runs,
`anomaly_analyze.py`):

| probe | result |
|---|---|
| roll +1.0 (2.7 rad/s), 2 s, through 180° | clean; off-axis rates ≈ 0.00; model dev ≤ 0.23 |
| roll +3.14 (11.1 rad/s), 1.2 s, ~2 revs | clean rigid-body spin, no runaway |
| pitch +3.14, 1.2 s, 3× through inversion | clean (tilt cycles 0→180→0, off-axis 0.00) |
| **[+3.14, −3.14, +3.14] = the exact S1.2 tumble cmd**, 1.2 s | all 3 axes settle (11.05/−11.05/−9.91) and HOLD; clean tumble |
| roll+yaw [3.14, 3.14] | clean (10.77/−9.72 sustained) |
| roll 0.5 / roll 2.0 / pitch 1.0 sustained | clean (the static-curve points) |

S1.2 tumble re-examination (`static_gain_curve.py`): the fit window cut at 10.99 s
("anomaly regime") was over-cautious — through 10.77–11.13 (tilt 90–118°) rates stay
magnitude-consistent with the static-gain model; **COLLISION id=1001 threat=2 at
sim_t≈11.145; roll = −62.9 rad/s two samples later.** The "yaw-spin" was a gate-post
strike plus Euler yaw wrapping near inversion.

**Env-redesign implication (for the fable+adroit ENV-COHERENCE session):**
- Crash-termination = the sim's collision signal analog (twin gate/wall geometry contact),
  which DiffAero's racing env already provides. Do NOT add a tilt-threshold
  pseudo-anomaly termination — it would forbid attitudes the sim handles fine and that
  fast racing may legitimately visit.
- Tilt/jerk regularization (S1.3 item ③) remains justified for SMOOTHNESS/VQ2 style and
  for keeping the policy in well-modeled regimes — but not as anomaly avoidance.
- The real transfer killer to fix is the plant: see §3.

## 3. Integration recommendation (supersedes cc6921d §5; twin → rl_plant → torch parity)

Replace the rate-update target computation, keep everything else:

```
g       = G0 / (1 - s * min(|cmd|, pi) / pi)        # s ≈ 0.30 roll/pitch (DR over [0.25, 0.35])
target  = g * rate_sign * cmd                        # G0 = shipped [2.501, 2.504, 2.231]
omega  += (target - omega) * (1 - exp(-dt/tau))      # tau ≈ 0.020 s (shipped 0.019 fine)
domega  clamped to ±alpha_max * dt                   # alpha_max ≈ 260 rad/s² r/p, ~80 yaw
```

- The existing first-order lag stays (the 2nd-order refinement buys ~10% small-signal
  overshoot fidelity, ζ→1 at saturation — optional; the static map is the load-bearing fix).
- **DR**: s ∈ [0.25, 0.35], τ ∈ [0.015, 0.03], alpha_max ∈ [200, 320] — this REPLACES both
  the disproven "+30% rate_gain band" AND the cc6921d ωₙ/ζ-envelope plan (memory already
  flags the former; the latter is this session's correction).
- Yaw: same form, plus the level-attitude ~7.4 rad/s cap as a caveated extra clamp (or
  give yaw its own measurement pass; it matters less — racing yaw cmds are small).
- The twin's existing `max_body_rate` clamp must be ≥ ~11.5 rad/s for roll/pitch or it
  will re-introduce the exact ceiling artifact we just removed.
- Keep delay in the transport-delay mechanism (measured 5–15 ms here), not in τ.
- Checkpoint impact: policies trained on the 2.5-flat plant (stage1_inc1, the running
  S1.3) under-predict their own authority at large commands by up to 42% — retrain after
  integration, per the already-banked gating.

## 4. Caveats

- All probes start near hover at low airspeed; gain/slew at racing airspeed (aero
  coupling) is unmeasured — the live course data (RMSE 0.22–0.41 under the shipped model
  at ≲3 rad/s cmds) suggests the small-signal regime is unaffected, and racing rarely
  holds saturated rates, but the DR band covers residual mismatch.
- The static curve has n=1–2 per (axis, magnitude) sustained point; the 4-point roll curve
  + exact pitch agreement at 2 points + cross-checks (multi-axis runs) make the shape
  solid, but s is good to ~±0.02, not ±0.001.
- Tumble command reconstruction phase ambiguity (±13–33 ms) still limits pointwise RMSE
  there; magnitudes and peaks are the reliable comparisons.
- Yaw maneuver-dependence (level cap vs tumbling) is measured but unexplained.

## 5. Files / data

- Harness: `scripts/rate_sysid.py` (auto-reset + home-kick, vel-damp hold, anomaly mode).
- Analysis (this dir): `fit_sweep.py`, `fit_windup.py`, `check_fits.py`,
  `static_gain_curve.py`, `refit_sat.py`, `anomaly_analyze.py`.
- Runs (gitignored, on ShadowPC): sweep `data/runs/20260610_22{28,34,52,53,54,55,57,58}*,
  20260610_2300*` (sweep_r03..r314y), anomaly `data/runs/20260611_02*_anom_*`.
  `20260610_222846_sweep_r03` is an empty no-GO artifact; `sweep_r20`/`r20r`/`r314p`/
  `r314r` aborted mid-schedule (data still used; see `fit_sweep.RUNS` for the per-run map).
