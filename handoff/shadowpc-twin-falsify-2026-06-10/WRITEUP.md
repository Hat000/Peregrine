# Twin falsification campaign: aero at speed, collective map, rate-at-airspeed (ShadowPC, 2026-06-10/11)

**Session SHADOWPC-TWIN-FALSIFY.** Predict-with-the-twin-first (predictions committed at
9c240df BEFORE any probe flew), fly, compare. Sim build **1.0.3364**; map-ON twin under test =
`twin_fit.faithful_config(super_rate=True)`. 40+ recordings, all probes unattended-chained.

## Headline (ranked by RL-transfer impact)

1. **Drag is QUADRATIC (and body-direction-dependent), not linear — twin falsified.**
   `a = -c2*|v|*v` with c2 ≈ 0.052 /m pooled (body-frame split: **0.042 nose-first / 0.058
   tail-first / 0.055 lateral**; vertical **0.076 climb / 0.054 descend**). The shipped
   `linear_drag = 0.2111` over-brakes at low speed and *massively* under-brakes at race speed:
   at 9 m/s real drag is ~4.2 m/s² vs the twin's 1.9 — **2.2× wrong exactly in the VQ1+ speed
   band**, and worse beyond. Coast-replay speed RMS: legacy 0.81 m/s → candidate 0.24-0.29.
2. **The collective→accel map is strongly CONVEX above hover — twin falsified.** Measured
   `K(thr)` vs the twin's linear `g*thr/0.2656`: ratio 0.38 @0.15 stick, 1.00 @hover,
   1.47 @0.40, 1.91 @0.55, **2.12 @1.0** (78.3 m/s² full-stick vs 36.9 modeled). Motor witness
   tracks the stick 1:1, so this is motor→thrust physics (≈ thrust^1.6-1.9), not a stick remap.
   Punch-out / climb authority is ~2× what the twin trains against.
3. **The super-rate gain map HOLDS at airspeed (twin survives).** Sustained |rate| at ~6 m/s:
   ratios 0.95-0.98 of the hover map at |cmd| 0.3 / 1.0 / 3.14, roll+pitch (same 2-5%
   flatness the map shows at hover). No airspeed term needed in the rate loop.
4. **No control-rate sensitivity (twin survives).** Same probe at 50/100/200 Hz: raw-odo
   sustained rate 2.679/2.683/2.688 rad/s (±0.2%), euler-slope confirms. The *apparent* Hz
   effect we first measured was OUR quat-FD pipeline artifact (see §6 — important gotcha).
5. **No battery sag (twin survives).** 320 s hover: alt-hold integrator slope −0.00003/min
   (−0.02% collective over a full 8-min VQ2 race). No battery model needed.
6. **Determinism/noise floor:** 5 repeats of one closed-loop maneuver: across-run SD ≈
   0.03 m/s vel, ≤0.10 m pos (upper bound incl. our command jitter). All deltas above are
   far over the floor.

## 0. Sim-ops incidents (read before the next live session)

- **TWO sim instances were running** (launched 7:01 PM and 9:15 PM on 6/10), BOTH speaking
  MAVLink to 127.0.0.1:14550. pymavlink's UDP peer re-learns from the last received packet, so
  our arm/rate commands alternated between a live race and a frozen parked instance →
  **armed=True but motors pinned at 0.05 idle, telemetry an interleave of two sim_t streams**
  (one advancing, one frozen at 2114.30 s). Diagnostic that cracked it: sim_t flip-flopping
  between two values in one pump loop. Cure: kill the windowless pair — NOTE killing either
  pair killed BOTH (shared parent/launcher); full relaunch via
  `...\AIGP_3364\FlightSim.exe` → any-key → Enter (cached login) → HOME → harness Enter×2
  chain works unattended (proven again tonight).
- **Stale-collision abort**: `client.collisions` accumulates pre-GO residue (the previous
  run's post-disarm fall) — rate_sysid now snapshots the count at GO (committed).
- **31000 behaves** once exactly one instance runs: fresh countdown ~2.8 s after each reset;
  ~27 s per chained probe.

## 1. Probe inventory

| probe | runs (data/runs/, gitignored, on ShadowPC) |
|---|---|
| P6 determinism | 20260611_0429..0431 det1..det5 (fixmnvr @100 Hz) |
| P5 rate-Hz | 20260611_0656* fix50a/b fix200a/b + 20260611_0705-0706 nh50/nh100/nh200 (near-hover legacy rate mode) |
| P1 drag | 20260611_0432* recon_lat12, drag_back08; 0641-0648 drag_back17/25/32, drag_fwd17/25, drag_lat17p/17n/25p |
| P1/P3 vertical | 20260611_0648* vert_mid, vert_high (vert_high aborted at the vz-17 guard as designed) |
| P3 collective | 20260611_0649* coll_hover; 0651 coll_speed2 |
| P2 rate@speed | 20260611_0651-0654 rspd_{r,p}{03,10,314} |
| P7 mixed | 20260611_0655 turn20 |
| P4 drift | 20260611_0657 drift320 (320 s) |
| dead runs | 0414 fix100a (dual-instance deaf), 0417 fix100b (no GO), 0426 fix100b2 (tilt-abort tuning), 0650 coll_speed (stale-collision abort) |

Analysis (this dir, all re-runnable from repo root with `.venv\Scripts\python.exe`):
`predict.py` (+ `predictions.md`, committed pre-flight), `gen_profiles.py` → `profiles/*.json`,
`runs.py` (loader/registry), `fit_aero.py`, `fit_vertical_joint.py` (→ `vert_fit_coef.npy`),
`fit_rates_misc.py`, `replay_twin.py`, `replay_coast.py`.

## 2. P1 — drag at speed

**Pre-flight twin prediction:** v_term = g·tan(tilt)/0.2111 (8.2/14.2/21.7/29.0 m/s at
10/17/25/32°), coast decay exponential τ=4.74 s, a(v) exactly linear, isotropic.

**Measured:** top speed barely grows with tilt (5.5 → 6.7 → 7.2 m/s at 17/25/32°) and coast
decay is slow at low speed — both linear-model killers. Binned along-track decel on the pooled
coast cloud (3121 samples, 10 runs):

| v band (m/s) | a (m/s²) | a/v (linear test) | a/v² (quad test) |
|---|---|---|---|
| 2.0-2.75 | 0.42 | 0.159 | 0.061 |
| 2.75-3.5 | 0.56 | 0.172 | 0.053 |
| 3.5-4.25 | 0.82 | 0.215 | 0.056 |
| 4.25-5.0 | 1.18 | 0.254 | 0.055 |
| 5.0-5.75 | 1.47 | 0.276 | 0.052 |
| 5.75-6.5 | 1.63 | 0.272 | 0.046 |

a/v rises 75%, a/v² flat ⇒ **quadratic**. Pooled fits: pure quad c2=0.0515 (rms 0.153),
linear d1=0.2484 (rms 0.181 — and the d1 estimate is just the cloud's mean v × c2), mixed
d1=0.070 + c2=0.037 (rms 0.148). Per-family quad c2: **back 0.058, lat 0.055, fwd 0.042**
(nose-first ~25-30% cleaner — body-frame anisotropy; the fwd runs are the one family the
isotropic candidate misfits in §5's coast replay). Vertical (joint fit, §3): c_up 0.076
(climb), c_dn 0.054 (descend). Same order as horizontal ⇒ a body-frame quad-drag diag
**[0.042 (+x), 0.058 (−x), ~0.055 (±y), 0.076 (−z up-flow), 0.054 (+z)]** covers everything
measured. (+x/−x asymmetry from coast windows where |v_bx| dominates; ±y symmetric within 2%.)

## 3. P3 — collective map + vertical drag (joint fit, 4812 samples, rms 1.16 m/s²)

`a_z − g = −K(thr)·cosθ + D(vz)`, K piecewise-linear, D directional-quadratic, fit jointly
across vert_mid/vert_high/coll_hover/det1-3 (hover knot recovers g to 2.3% — method check):

| thr | K meas (m/s²) | twin g·thr/h | ratio |
|---|---|---|---|
| 0.15 | 2.1 | 5.5 | 0.38 |
| 0.20 | 4.7 | 7.4 | 0.64 |
| 0.2656 | 9.6 | 9.8 | 0.98 |
| 0.32 | 13.6 | 11.8 | 1.15 |
| 0.40 | 21.7 | 14.8 | 1.47 |
| 0.55 | 38.8 | 20.3 | 1.91 |
| 0.80 | 58.4 | 29.5 | 1.98 |
| 1.00 | 78.3 | 36.9 | 2.12 |

- Knots 0.0/0.10 are NOT trustworthy (short descent windows, drag colinearity; the raw c000
  step shows near-free-fall as expected). Use 0.15-1.0.
- Power-law equivalent: a_up ≈ g·(thr/0.2656)^p with p≈1.9 mid-range, flattening to ~1.6 at
  the top — the knot table is the deliverable, the power law a convenience.
- **Motor witness:** commanded collective → motor outputs 1:1 (idle floor 0.05; ~2% sag at
  1.0) ⇒ the convexity is thrust physics, not input mapping.
- **No translational-lift coupling:** the hover-fit map predicts coll_speed (6-8 m/s
  horizontal) with rms 0.59, bias −0.03 m/s² (EVAL, not fit).
- **No collective→attitude coupling:** full-stick steps leave attitude within ~2° at
  hold effort ≤0.03 rad/s.
- Envelope actually reached: climb 17 m/s @0.55 (guard abort), descent 10.7 m/s @0.15.

## 4. P2 — rate loop at airspeed (map holds)

Raw-odo sustained |rate| during steps at ~6 m/s (euler-slope cross-check agrees below
inversion):

| probe | cmd | v_h | meas | map pred | ratio |
|---|---|---|---|---|---|
| rspd_r03 | 0.30 | 6.0 | 0.747 | 0.772 | 0.97 |
| rspd_p03 | 0.30 | 6.0 | 0.746 | 0.773 | 0.97 |
| rspd_r10 | 1.00 | 6.1 | 2.660 | 2.765 | 0.96 |
| rspd_p10 | 1.00 | 6.1 | 2.635 | 2.768 | 0.95 |
| rspd_r314 | 3.14 | 6.0 | 11.026 | 11.216 | 0.98 |
| rspd_p314 | 3.14 | 6.0 | 11.035 | 11.230 | 0.98 |

Same 2-5% the map over-predicts at hover (S14 integration report) ⇒ **no airspeed term**.

## 5. P7 — integrated replay (turn20: roll 20° + 720° yaw ramp + alt-hold, 2 circles @~6 m/s)

Open-loop command replay, full trace: speed RMS legacy/map_on 13.7/13.6 → **candidate 9.4**
(final-pos drift 277→178 m over 26 s — yaw integration drift dominates attitude RMS; the
one-step vel RMS 0.05 m/s is the per-step fidelity). Drag-run coast-only replay (seeded from
live state at coast start — isolates the drag model from accel-phase thrust errors):

| variant | mean coast speed RMS (9 runs) |
|---|---|
| legacy (linear 0.2111) | 0.810 m/s (grows with speed: 0.28@8° → 1.65@32°) |
| candidate quad (c2h 0.052 iso) | 0.289 m/s |
| candidate mixed (d1 0.070 + c2 0.037) | **0.240 m/s** |

Candidate plant implementation: `replay_twin.CandidatePlant` (quad drag + knot collective
map on top of the map-ON rate loop) — ready to port.

## 6. Measurement gotcha that cost an hour (and previously poisoned tonight's first P2/P5 read)

The harness's quat-FD `true_rate` (finite-diff of ODOMETRY quaternions over **sim_time**) is
**biased high, loop-rate-dependently** (read 3.0-3.6 rad/s where truth was 2.68): sim_time in
`client.state` advances on BOTH the ~75 Hz LPN and ~97 Hz ODOMETRY streams, while attitude
only changes with ODOMETRY ⇒ the finite-diff sometimes divides a full attitude step by a
short LPN-LPN gap. The aliasing pattern shifts with the control-loop rate — it *looked* like
a 100 Hz-specific gain boost AND a 1.2× at-speed gain boost. Ground truth: raw ODOMETRY
angular_rate magnitude == euler-angle slope at every loop rate. Yesterday's sweep fits used
raw odo (tlog) — unaffected. **Rule: never use the live quat-FD column for DC magnitudes;**
it exists for hold damping (where lag, not magnitude, matters).

## 7. Proposed twin changes (measure-only session: NOT applied to src/)

Ranked by RL-transfer impact, with DR bands:

1. **Replace `linear_drag` with body-frame quadratic drag** (impact: every fast segment —
   terminal speed, braking distance, energy):
   `f_b = -C2_b ⊙ |v_b| ⊙ v_b`, nominal C2 = [0.042 fwd; 0.058 back (use sign split on x),
   0.055 lat, 0.076 up-flow / 0.054 down-flow (sign split on z)].
   Minimum viable: isotropic c2 = 0.052 + optional d1 = 0 (pure quad; the mixed form's small
   d1=0.07 buys 0.05 m/s on back/lat but slightly worsens fwd).
   **DR: c2 ∈ [0.040, 0.065] per axis (resampled per env), d1 ∈ [0, 0.08].**
   The legacy world-frame-isotropic option is strictly dominated.
2. **Replace the linear collective map with the measured K(thr) knot table** (impact: any
   climb/punch-out/recovery; 2× authority error at full stick): `a_up = interp(thr, knots, K)`
   with the §3 table (hold 0.0/0.10 at the linear extrapolation of 0.15-0.20 until a dedicated
   low-stick pass). **DR: scale the K table by U[0.9, 1.1]; keep the hover point pinned ±2%**
   (hover error moves the trim the policy lives on).
3. **Rate loop: NO change** beyond S14 (map validated at airspeed; no Hz term; the existing
   s/alpha_max/tau DR stands).
4. **No battery-sag model** (measured < 0.03% over 8 min). Drop VQ2 sag from the worry list.
5. Checkpoint impact: anything trained on the linear-drag plant has learned wrong braking
   distances at speed (under-braking by ~2× at 9 m/s) and wrong climb authority. Same
   retrain-after-integration gating as the S14 map.

## 8. Caveats

- Drag measured to 7.6 m/s horizontal (coast peaks; accel phases corroborate the same c2 to
  ~7.5). VQ2 may exceed this; quad extrapolation is principled but unverified beyond 8 m/s.
  A higher-speed pass needs a longer corridor (the start arena is clear ≥43 m +X, ≥40 m ±Y at
  6-8 m up; collision aborts armed; no strikes all night).
- The fwd/back c2 split rests on 2 fwd runs (clean, consistent) — one more fwd run at 32°
  would pin it; the isotropic-0.052 fallback under-fits ONLY the fwd family (by ~0.9 m/s end
  speed over a 6 s coast).
- K(thr) below 0.15 is extrapolation; the c000 raw step looked like proper free fall, so the
  bottom is qualitatively sane, just not fit-grade.
- Vertical drag knots and K are partially colinear in the climb steps (mitigated by multi-run
  pooling + the hover-knot==g check at 2.3%); c_up vs c_dn split is robust in sign, ±20% in
  magnitude.
- turn20 attitude RMS (19/32/11°) is open-loop drift over 26 s, not per-step infidelity
  (one-step vel RMS 0.05 m/s); use coast/eval replays for model selection.
