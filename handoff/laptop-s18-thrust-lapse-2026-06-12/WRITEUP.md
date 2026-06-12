# LAPTOP-S18-THRUST-LAPSE — refit, integrate, and a verdict that overturns the premise

**Session:** LAPTOP-S18-THRUST-LAPSE (2026-06-12). **Model:** claude-opus-4-8.
**Input:** the inc6 live residual from `shadowpc-inc6-diag` / `shadowpc-bridge-retest` —
"thrust map overpredicts 15–25% at 3–12 m/s; refit and re-run the deploy matrix."
**Commits:** `fb99636` (lapse integration), this writeup.
**Dataset:** `handoff/shadowpc-refit-dataset-2026-06-12/debug_obs_17runs.zip` (commit c846054).

---

## TL;DR — the headline reverses the working hypothesis

1. **The thrust lapse is REAL** and is now fit, integrated into all three plants (parity-gated,
   defaults OFF = bit-identical legacy), and ready for inc7 DR. The early standing-start forward
   speed deficit is confirmed: live reaches 5.9 m/s by tick 12 where the no-lapse plant reaches
   9.5; the lapse closes part of it.

2. **But the thrust lapse is NOT the cause of inc6's live standing-start failure, and the refit
   plant does NOT reproduce that failure.** I proved this three ways (no lapse curve of any depth
   reproduces it; the live failure is a yaw spin the policy *commands*; open-loop replay).

3. **The true cause, localized precisely: a residual LATERAL (roll-handedness) sign inversion in
   the thrust→world force projection, surfacing only in the hard-banked (roll ≳ 50°) gate-0
   flare.** Given the live-recorded true attitude, our force model computes the East thrust
   acceleration as **−33 m/s²** where the sim measured **+30** (both rollfix flights, identical).
   North and vertical match; only the left-right (East) axis is mirrored. This is a sibling of the
   diag's roll-mirror, one level deeper, that the diag's near-level / pitched quat-FD validation
   could not exercise.

4. **Consequence:** the offline twin self-mirrors (so inc6 finishes 6/6 offline on every plant),
   which is exactly why the laptop matrix gave a false 16/16 while live was 0/15. **Fly-as-is is
   wrong** (the failure is unresolved) and **inc7-retrain-now is wrong** (it would bake the mirror
   deeper). The next step is a focused convention diagnostic, not a plant refit — STOP and report
   per the escape hatch. The commander decides on a fable escalation.

5. The lapse work ships regardless: it is correct physics, costs nothing (defaults OFF), and is
   needed once the convention is fixed. 589 tests green.

---

## 1. FIT — the lapse is real and well-identified where it matters

Method (`scripts/s18_characterize.py`, `s18_identify.py`, `s18_fit.py`, `s18_lock.py`): smooth
ticks (|ω|<1 rad/s) across all 17 runs; pristine `vel_ned` finite-differenced for the measured
specific force; raw quat as-is for attitude (bcc93f9); `collective` (sign-mirror does not touch
the thrust scalar) shifted by the d=2 transport delay. `K_eff = (a_meas − g − a_drag)·b3`,
ratio = `K_eff / K(collective)`. Pre-fix and post-fix runs agree (POST n=96 tracks PRE within
noise), so all 17 are usable for the thrust magnitude.

**The curve (median ratio vs |v|, measured-drag attribution):**

| |v| band (m/s) | 3–6 | 6–9 | 9–12 | 12–15 | 15–18 |
|---|---|---|---|---|---|
| ratio | 0.74 | 0.82 | 0.88 | 1.00 | 0.99 |

**Identifiability — the good news is that the lapse is well-identified exactly in the
standing-start band.** Re-attributing the residual under drag-scale {0.5, 1.0, 1.5}× gives a
spread of only **0.08–0.09 at 3–9 m/s** (drag is negligible there, so the deficit is
drag-independent), versus 0.37–1.07 above 12 m/s (where lapse and drag are NOT separable — but
lapse ≈ 1 there regardless). **So the fit is NOT degenerate with drag in the regime that
matters.** No disambiguating ShadowPC probe is required for the standing-start lapse. (If VQ2
high-speed work later needs the >12 m/s thrust curve, the clean probe is a constant-collective,
pure-vertical-sag sweep — hold collective at hover, push forward by pitch only, read vertical
sag vs |v| — which isolates the thrust axis from the in-plane drag.)

**Locked model** (multiplicative on `a_up`, np.interp end-clamped; L(0)=1 hover-anchored,
L≥15 = 1):
```
LAPSE_SPEED_MEASURED  = [0.0, 4.0, 8.0, 12.0, 15.0]   # m/s
LAPSE_FACTOR_MEASURED = [1.0, 0.78, 0.80, 0.92, 1.0]
```
3-fold CV (held-out, `s18_lock.py`) reduces the out-of-sample b3-accel bias in every low-speed
band (−4.5→−2.1, −3.3→−1.2, −5.3→−3.5 m/s²) and is neutral at 12–18 (−0.4→+0.0). Conservative
floor 0.78–0.80 matches the diag's independent "15–25% below" estimate.

**Deliberately excluded — fast descent.** At v_axial < −2 m/s the ratio falls below 0 (L → −0.9
in fast descent; vortex-ring-like). This is OUTSIDE the climbing standing-start manifold
(v_axial > 0 throughout the gate-0 approach) and concentrated in failure-flight transients. It is
folded into the inc7 DR band, NOT modeled as a map term — a 2-D axial-inflow thrust map would be
structurally new and is not warranted by the standing-start failure.

## 2. INTEGRATE — all three plants, parity-gated, defaults OFF (`fb99636`)

S14/S16/S17 recipe exactly: a multiplicative `a_up *= interp(|vel|, lapse_speed, lapse_factor)`
on the OLD world velocity (like drag), set-together + range-validated, None → bit-identical
legacy.

* `src/racer/rl_plant.py` — `LAPSE_SPEED/FACTOR_MEASURED`; `lapse_speed/lapse_factor` PlantParams
  fields + `__post_init__` validation; applied in `step` after the collective map.
* `src/racer/twin.py` — config fields + inline set-together apply (uses `np.linalg.norm(self.vel)`).
* `src/racer/twin_fit.py` — `faithful_config(..., lapse=True)` (requires `measured_aero`); the
  fully sim-faithful twin is now `(super_rate, measured_aero, mixer, lapse)=True`.
* `rl/diffaero_dynamics.py` — torch mirror in `_step_torch` (`torch.sqrt((v*v).sum(-1))`,
  op-mirror of rl_plant); `+dynamics.dr_lapse` opt-in (requires dr_aero; scales the lapse DEPTH
  1−L per-env by [lapse_lo, lapse_hi]=[0.5,1.5]) for inc7.
* `rl/check_diffaero_gate.py` — `_lapse` rebuild + `{lapse, lapse_full}` configs.
* `rl/offline_rollout.py`, `rl/peregrine_eval.py` — `--plant lapse` (= mixer + lapse).

**Gates:**
* **Test suite 562 → 589 green.** +20 twin↔rl_plant parity cases (`lapse`/`lapse_full` × 2 dt × 5
  sequences) **bit-identical**; +7 lapse unit/torch/DR/validation tests (`test_measured_aero.py`).
* Negative controls in-test: lapse OFF reads the full map value at every speed; biased/missing
  knots raise; dr_lapse requires dr_aero.
* `test_defaults_are_off_everywhere` extended — lapse OFF in every legacy/mixer config.
* **CPU config-matrix gate (`check_diffaero_gate.py`) could NOT run on this laptop** — the
  DiffAero base package is not importable here (pre-existing; fails identically on clean HEAD).
  The numpy twin↔rl_plant parity battery + the `_t_interp1d` bitwise test cover the algebra
  locally; **🚩 the V100 config-matrix gate (now carrying `lapse`/`lapse_full`) MUST run at next
  Adroit contact** before trusting the torch backend for an inc7 train.

## 3. VERDICT — the refit plant does not reproduce the live failure; the cause is elsewhere

### 3a. No thrust-lapse curve reproduces the live signature (`s18_repro_sweep.py`)

inc6 on `--plant mixer` finishes 6/6 @ 9.49 s (matches the deploy matrix). On `--plant lapse`
(as-fit): 6/6 @ 9.49 s — **the lapse barely moves the outcome.** Swept depth/persistence at
latency {0,2,3}:

| lapse curve | outcome | E at gate-0 plane |
|---|---|---|
| as-fit (recover@15) | FINISHED | −0.2 m |
| persist-0.72 (no recovery) | FINISHED | −0.2 m |
| **deep-early floor 0.55, recover@14** | FINISHED | −0.2 m |
| **very-deep floor 0.45, recover@14** | FINISHED | −0.2 m |
| deep-0.60 persistent | COLLISION (vertical sag) | −0.2 m |

The live target is **gate-0 plane crossed at East ≈ +5…+7 m, then wander.** Shallow lapses finish
cleanly centred; deep persistent lapses cause a *vertical* crash, still laterally centred. **No
thrust-axis curve — any depth, any speed profile — bends the trajectory into the lateral
excursion.**

### 3b. The live failure is a yaw spin the policy COMMANDS (`s18_live_yaw_decomp.py`, `s18_rate_track_check.py`)

Both rollfix flights (reproducible to the degree): the +East drift develops at tick 42–66 at
**16–18 m/s** — exactly where the lapse is ≈ 1. There the policy commands a hard pitch-up flare
and a yaw turn; sideslip blows up to −72°; the heading spins (135°→175°→−115°). Comparing the
realized true rate to the plant's command-tracking prediction `g(|cmd|)·sign·cmd`:

* **Yaw: prediction matches realized almost exactly** (tick 60: pred +2.12 vs actual +2.08). The
  policy is *commanding* the spin — it is NOT an unmodeled disturbance torque.
* Pitch magnitudes match too (|pred| ≈ |actual|), only a sign-label difference in the logged
  `rate_frd`. **The rate loop faithfully tracks commands; there is no missing torque.**

So the divergence is driven by what the policy *commands*, which is set by the obs trajectory.

### 3c. Open-loop replay localizes it to a LATERAL force mirror (`s18_openloop_replay.py`)

Seed the twin at a live pre-divergence state (tick 36) and drive it with the **exact recorded
live wire commands** for 27 ticks:

| | live | twin |
|---|---|---|
| roll / pitch / yaw (deg, tick 60) | 57 / 7 / −115 | 56 / 7 / −113 |
| yaw rate (tick 60) | +2.08 | +2.14 |
| speed (tick 60) | 14.8 | 14.9 |
| **East velocity (tick 60)** | **+11.1** | **−11.1** |

**The twin reproduces attitude, rates, and speed exactly — but the East velocity comes out
exactly opposite-signed.** Same attitude + same thrust → opposite lateral force. Only the
left-right axis flips; North and vertical match (speed is preserved).

### 3d. The mirror, pinned (`s18_force_frame_selfcheck.py`)

Using the live-recorded true attitude, compute our force model's East acceleration vs the live
measured East acceleration (both rollfix flights, identical):

| tick | roll | measured a_E | model a_E | model thrust_E |
|---|---|---|---|---|
| 42 | 46° | +5.4 | −6.8 | −6.9 |
| 51 | 59° | +30.3 | −35.1 | −33.2 |
| 60 | 57° | +2.9 | −17.3 | −12.4 |

The model's East acceleration is the **negative** of the measured, dominated by the thrust term.
**Flipping the thrust East sign makes model match measurement** (tick 51: −35 → +31 ≈ measured
+30; drag would need an absurd +65 m/s², ruling out a real sideforce). The discrepancy is already
present at tick 39–42 in *normal* banked flight (roll 44–46°, sideslip 14–20°) — not a
degenerate-regime artifact.

**Interpretation (convention-robust at the localization level):** our plant's thrust→world
lateral projection is roll-handedness-mirrored relative to the sim, given the same attitude
quaternion. The most likely root is that the raw ODOMETRY quat is roll-mirrored relative to the
true *physical* attitude — the diag established "quat = true attitude" from quat-FD (rates) and a
near-level v_body→position-derivative check, both of which a roll-mirror passes (the diag's own
lesson). The hard-roll gate-0 flare is the first maneuver that exercises the lateral handedness.
In closed loop the twin mirrors self-consistently → inc6 finishes offline on *every* plant
(mixer, lapse, all start modes/latencies/perturbed seams — verified, no regression); live, the
policy banks the wrong way at the flare and diverges. This is precisely why the laptop matrix
could not catch the live failure.

### 3e. Deliverable fork — neither clean branch applies

* **NOT "robust on refit plant → fly as-is":** the offline "robust" result is the same
  self-mirroring harness that already produced a false 16/16. It cannot expose the live failure,
  so its green is uninformative. The failure is unresolved.
* **NOT "breaks on refit plant → spec inc7 with lapse-DR":** the refit plant does NOT break inc6,
  and retraining against a self-mirrored twin would bake the mirror deeper, not fix it.

**Verdict: STOP the refit→retrain loop here. The live failure's root cause is a residual lateral
convention mismatch, not a plant coefficient.** This is the escape-hatch case ("the failure
demands something the S16/S17 recipe doesn't cover — STOP, characterize, report"). The lapse
ships as correct, defaults-off physics; the next step is a focused convention diagnostic.

## 4. mixer_probe2 consistency check (`s18_mixer_probe2_check.py`) — CONTRADICTED, flagged

The 4 new settled-spin rows vs the S17 model (`u_i = clip(c ± d, idle, 1)`,
`d = κ_err(target−ω) + κ_hold·ω`):

| row | c | model d | measured d | note |
|---|---|---|---|---|
| c60_r31 (clean, no clip) | 0.6 | 0.523 | **0.132** | model over-predicts roll diff **~4×** |
| zhov_r31 | 0.27 | 0.528 | 0.217 | |
| c100_r31 | 1.0 | 0.524 | 0.356 | model predicts up-motor saturated; measured 0.999 borderline |
| c100_y31 | 1.0 | 0.122 | **0.349** | model under-predicts yaw top-rail diff **~3×** (zeta over-suppresses at c=1) |

The clean c60_r31 row implies **roll κ_hold ≈ 0.012**, not the yaw-derived 0.046 the S17 model
applies to all axes. **Contradicted on two axes** (roll hold ~4× too high; yaw top-rail
effectiveness ~3× too low at c=1.0). NOT a trivial coefficient update (per-axis κ_hold +
top-rail yaw effectiveness — structural), so **NOT integrated this session; flagged for a
dedicated follow-up.** Bounded impact: these are sustained full-stick settled-spin measurements;
the inc6 corner-tax policy avoids that regime entirely (thr_p95 0.061, 0% saturation, 0% yaw
flips), so the training-side effect is likely small.

## 5. Recommendations (commander decides)

1. **Do NOT fly inc6 as-is** — the live standing-start failure is unresolved and is NOT the lapse.
2. **Do NOT train inc7 yet** — the twin's lateral mirror would be baked in. The lapse-DR infra is
   built and ready (`+dynamics.dr_lapse`, `--plant lapse`) for *after* the convention is fixed.
3. **NEXT (likely fable-grade, like the roll-mirror diag): a focused lateral-convention
   diagnostic.** The precise tools are in this handoff: `s18_force_frame_selfcheck.py` (live
   measured accel vs model-from-attitude, per-component — the smoking gun) and
   `s18_openloop_replay.py`. Extend the diag's quat-FD method to **hard-ROLL** phases (the
   original validated level + pitched). Candidate roots to test: (a) the raw ODOMETRY quat is
   roll-mirrored vs true physical attitude; (b) a sign in the deploy thrust path. Rule out a real
   aero sideforce (the −33↔+30 magnitude match makes it implausible, but confirm). One live probe
   would settle it: a deliberate sustained-roll-bank at moderate speed, comparing measured lateral
   accel to the attitude-derived prediction.
4. **mixer_probe2 roll κ_hold / yaw top-rail:** separate small follow-up; low priority (policy
   avoids the regime).
5. **V100 config-matrix gate** must run at next Adroit contact (lapse configs added; CPU gate
   un-runnable on this laptop).

## 6. Index

* Scripts: `handoff/laptop-s18-thrust-lapse-2026-06-12/scripts/` (fit: `s18_characterize`,
  `s18_identify`, `s18_fit`, `s18_lock`; verdict: `s18_repro_sweep`, `s18_live_yaw_decomp`,
  `s18_rate_track_check`, `s18_openloop_replay`, `s18_force_frame_selfcheck`; mixer:
  `s18_mixer_probe2_check`).
* Dataset: `handoff/shadowpc-refit-dataset-2026-06-12/` (zip in repo; `extracted/` gitignored).
* Code: `fb99636`.

---

MEMORY-DELTA:
- 🚩 NEW (project, HEADLINE): inc6's live standing-start failure is NOT the thrust lapse — it is a
  residual LATERAL (roll-handedness) thrust→world force MIRROR, surfacing only in the hard-banked
  (roll ≳50°) gate-0 flare. Pinned: model East-thrust accel −33 vs sim measured +30 m/s² at the
  live true attitude; open-loop replay reproduces live attitude+rates+speed but opposite-sign East
  velocity. Offline twin self-mirrors → inc6 finishes on every plant offline (why laptop matrix
  gave false 16/16). SUPERSEDES the "thrust-lapse residual blocks both start modes" framing — the
  lapse is real but secondary. Next step = focused lateral-convention diag (fable-grade), NOT a
  refit/retrain. Tools: s18_force_frame_selfcheck.py / s18_openloop_replay.py.
- NEW (project): S18 airspeed thrust lapse fit + integrated all 3 plants (LAPSE_SPEED/FACTOR_
  MEASURED = [0,4,8,12,15]/[1.0,0.78,0.80,0.92,1.0]; multiplicative on a_up vs |vel|), parity-gated
  defaults OFF, faithful_config(lapse=True), --plant lapse, +dynamics.dr_lapse for inc7; 562→589
  tests; commit fb99636. Well-identified vs drag in the 3–9 m/s band (spread ≤0.09); NOT separable
  >12 m/s but ≈1 there. Fast-descent thrust loss folded into DR, not modeled.
- 🚩 DECISION: do NOT fly inc6 as-is; do NOT train inc7 until the lateral convention is fixed
  (retraining bakes the mirror in). inc7 lapse-DR infra is built and ready for after.
- AMEND aigp-mixer-coupling-law: mixer_probe2 CONTRADICTS S17 on roll κ_hold (clean c60_r31 →
  ~0.012, not the yaw-derived 0.046; ~4× over-predict) and yaw top-rail (zeta over-suppresses ~3×
  at c=1.0). NOT integrated (structural, not trivial); flagged follow-up; policy avoids the regime.
- 🚩 V100 config-matrix gate must run at next Adroit contact (lapse/lapse_full configs added; CPU
  gate un-runnable on this laptop — DiffAero base not importable here).
