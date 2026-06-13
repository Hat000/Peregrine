# LAPTOP-INC8-BINDING-GATE-VERIFY — verifying the inc8 binding-gate finding

**Date:** 2026-06-13 · **Model:** opus-4.8 / medium · **Machine:** laptop (dev)
**Instrument:** `rl/contact_true_eval.py` (extended this session) · **Checkpoint:** `rl/checkpoints/stage1_inc7_actor.pth` (md5 2AFF8D…50E3)
**Plant:** `mixer` (default, map-ON, fully-measured) — confirmed in every run's `CONTACT_TRUE_SUMMARY` line. ✅

---

## TL;DR VERDICT

**The binding gate is GATE-4, but only under a competition-representative SIMSTART (high-speed post-gate-3 approach).** At the inc7-training-consistent radius r=0.38, gate-4's simstart contact-true margin is **0.155 m** — the tightest of all six gates, by a clear gap (next tightest gate-2 at 0.219 m). The binding gate does **not** shift with radius (margins are `(0.75−r) − linf`; the constant shift preserves the ranking). It **does** shift with start type: in synthetic TRAINRESET resets gate-4 relaxes (linf 0.215→0.199) and gate-2 marginally binds instead — which is itself the proof that **the high-speed post-gate-3 approach is what stresses gate-4.** The "post-gate-3 risk zone" framing is **CONFIRMED but localizes to gate-4 specifically, not the whole 4–5 segment** (gate-5 is clean, linf 0.056). Gate-3's nominal margin is usable as a *relative/ranking* signal only; its *absolute* value carries an unresolved ~1.46 m track_map D-registration caveat (see Task 4) and the same unverified-registration risk contaminates gate-4.

---

## Code changes (committed)

1. **`gate_d_offset_probe(pos_traj, delta_d_m, gate_id=3, …)`** — generic parametric probe (was gate-3-hardcoded). `gate3_d_offset_probe` retained as a thin backward-compat wrapper (same g3-prefixed keys) so existing callers/tests are untouched.
2. **`per_gate_margin_stats(results, start_filter=None, pass_band=PASS_BAND_NOM)`** — (a) `start_filter` prefix-selects simstart vs trainreset episodes; (b) **`pass_band` parameter — measurement-correctness fix** (see below).
3. **CLI** — prints POOLED + SIMSTART-only + TRAINRESET-only margin tables, and runs the D-offset probe on gates **3, 4, 5**.
4. **Tests** — `tests/test_contact_true_eval.py`: +10 tests (generic-probe keys/gate-id/per-gate-nominal/D-offset-direction/gate3-wrapper-equivalence; pass_band-shift; start_filter). **657 passed** (was 647).

### 🚩 Measurement bug found & fixed (pre-existing, in scope for Task 1)
`per_gate_margin_stats` hard-coded `pass_band = PASS_BAND_NOM` (0.42, the r=0.33 band) **regardless of `--body-radius`**. Recorded `linf` is pure trajectory geometry (radius-independent); only the pass band shifts with radius. So the original tool reported **identical margins at every radius** — the multi-radius sweep would have been meaningless. Now `pass_band = _HALF_OPEN − args.body_radius` flows through. (`GateCrossing.margin` property still reports the nominal-band margin by definition; it is only used for the probe's nominal-row display, where r=0.33, so no inconsistency surfaces.)

---

## TASK 1 — Multi-radius baseline (SIMSTART, competition-representative)

`linf` is identical across radii; margin = `(0.75 − r) − linf`. **r=0.38 is the inc7-training-consistent primary row.**

| gate | simstart linf | margin r=0.28 | margin r=0.33 | **margin r=0.38 (primary)** |
|:----:|:-------------:|:-------------:|:-------------:|:---------------------------:|
| 0 | 0.109 | 0.361 | 0.311 | 0.261 |
| 1 | 0.079 | 0.391 | 0.341 | 0.291 |
| 2 | 0.151 | 0.319 | 0.269 | 0.219 |
| 3 | 0.051 | 0.419 | 0.369 | 0.319 |
| **4** | **0.215** | **0.255** | **0.205** | **0.155 ← BINDING** |
| 5 | 0.056 | 0.414 | 0.364 | 0.314 |

**Binding gate = gate-4 at every radius.** Because all gate margins shift by the same constant `(0.75−r)`, the tightest-margin gate is invariant to radius — radius changes only the *absolute* slack, never *which* gate binds. S_stable = 1.000 (7/7), all seeds FINISHED, zero collisions, at all three radii.

## TASK 3 — Disaggregation by start type (the decisive cut)

Pooling simstart with the 6 synthetic trainreset seeds **hides the real signal.** Tightest gate per start type @ r=0.38:

| start type | tightest gate | its margin @ r=0.38 | runner-up | note |
|---|---|---|---|---|
| **SIMSTART** (full-course, real high-speed post-g3 approach) | **gate-4** | **0.155** | gate-2 @ 0.219 | gate-4 binds by a clear 0.064 m gap |
| TRAINRESET (synthetic, at-rest 1 m-back) | gate-2 | 0.166 | gate-4 @ 0.171 | gate-4 *relaxes* (linf 0.215→0.199) |

**Only SIMSTART is competition-representative for the post-gate-3 binding question** (trainreset spawns at rest 1 m before each gate → never builds the high-speed descending approach into gate-4). The fact that gate-4's linf drops from 0.215 (simstart) to 0.199 (trainreset) is direct evidence that **the high-speed post-gate-3 approach is the stressor** that makes gate-4 bind. In the synthetic resets gate-4 is gentle enough that gate-2 edges it out — so a pooled or trainreset-only view would mis-identify the binding gate.

## TASK 2 — Generalized map-offset probe (gates 3, 4, 5)

`gate_d_offset_probe` run at ±1.5 m on gates 3/4/5 (simstart trajectory, identical at all radii):

| gate | nominal | D−1.5 m | D+1.5 m |
|:----:|:-------:|:-------:|:-------:|
| 3 | FINISHED (g3_margin +0.369) | **COLLISION** | **COLLISION** |
| 4 | FINISHED (g4_margin +0.205) | **COLLISION** | **COLLISION** |
| 5 | FINISHED (g5_margin +0.364) | **COLLISION** | **COLLISION** |

**Fragility is universal, not gate-4-specific.** Every gate flips pass→collision under a 1.5 m map shift. So the probe does **not** single out gate-4 as map-confounded — it only shows the contact-true metric is generically sensitive to a 1.5 m map error at *any* gate. This is fragility (what *would* happen if the map were off), **not** evidence any specific gate's map *is* off. The only gate with positive evidence of a real offset is gate-3 (Task 4) — and that evidence comes from live data, not this probe.

## TASK 4 — Gate-3 reconciliation (the framing correction)

**The simple "0.051 ≈ 0.06 → track_map accurate" cross-check does NOT hold as stated, and the raw live data refutes it.**

- Offline simstart gate-3 L-inf = **0.051 m** — but this is measured **vs track_map** (offline uses track_map for the policy target, the obs, AND scoring → self-consistent, says nothing about truth).
- Live gate-3 crossing (`shadowpc-inc7-live` debug_obs, 5/5 flights): drone at D ≈ **23.10 m** vs track_map center D = **24.568 m** → **−1.46 to −1.48 m above center on the D (vertical) axis**; E (lateral) offset only −0.07 m. n_coll = 0, PASS-CLEAN, no saturation.
- The "~0.06 m" live figure agrees with offline **only on the E axis** (~0.07 m). The D axis disagrees by **1.46 m** and the gate plane includes D — so the track_map-relative live L-inf is ~1.46 m, **not** 0.06 m.

**Why this forces a registration conclusion:** track_map outer_half = 1.36 m. If track_map equalled the true gate, a crossing 1.46 m (> 1.36 m) off-center would be a MISS/outside, not a clean pass. The sim registered a clean gate-3→4 transition (n_coll=0) → the drone *is* inside the sim's TRUE gate → the TRUE gate-3 center is ~1.46 m from our track_map center. **track_map gate-3 D-center is mis-registered by ~1.46 m.** The agreement of two small offsets measured against two *different* gates does not prove the gates coincide; the raw D-coordinate gap proves they don't.

**Is gate-3's nominal margin usable now?**
- **As a relative / ranking signal: YES.** In the offline self-consistent frame the policy centers well on gate-3 (0.051) and gate-3 is decisively **non-binding** (largest margin of all gates). Live confirms gate-3 passes clean with wide true-gate margin every flight. Phase-1 does not need gate-3's absolute number to know it isn't the constraint.
- **As an absolute clearance number: NO, not yet.** The ~1.46 m track_map↔true-gate D-registration error means the offline gate-3 *position* is wrong by ~1.46 m. Any tight gate-3-specific decision must wait for **SHADOWPC-VISION-CAL** to resolve the registration. Do **not** harden the earlier draft claim "track_map ~accurate" — the live data does not support it.

**Carry-over risk for Phase-1:** the binding gate (gate-4) inherits the **same** track_map-registration uncertainty and currently has **NO live per-gate offset cross-check** (gate-4/5 live offsets were never extracted — the LOW-priority residual noted for the next ShadowPC touch). So gate-4's *absolute* 0.155 m margin is a track_map-frame number; live PASS-CLEAN confirms gate-4 is *safe*, but not the exact live clearance. The binding-gate **ranking** is robust; the binding-gate **absolute margin** is provisional until VISION-CAL.

---

## What Phase-1 arm selection can safely trust

1. **Binding gate = gate-4** (ranking, from realistic simstart). Use it as the gate whose clearance the envelope-relaxation ladder must not erode. ✅
2. **Use SIMSTART margins, not pooled/trainreset** for the post-gate-3 budget. Pooling hid gate-4. ✅
3. **Absolute margins are track_map-frame** with a confirmed ~1.46 m gate-3 D-registration error and unverified gate-4/5 registration. Treat 0.155 m as a relative budget, keep headroom, and close registration via SHADOWPC-VISION-CAL + ShadowPC gate-4/5 live-offset extraction before betting tight absolute numbers. ⚠️
4. **The "gates 4–5 risk zone" is really gate-4.** Gate-5 is clean (0.056). Two independent instruments now agree the stressor is the *high-speed approach into gate-4* immediately after gate-3. ✅

---

## MEMORY-DELTA
- BINDING GATE = **gate-4** (SIMSTART, all radii); margin @ r=0.38 (inc7-training-consistent) = **0.155 m**, tightest by 0.064 m over gate-2. Binding gate is radius-INVARIANT (margins shift by const `0.75−r`).
- Binding gate is START-TYPE-dependent: gate-4 binds ONLY under simstart (high-speed post-g3 approach, linf 0.215). In synthetic trainreset gate-4 relaxes (linf 0.199) and gate-2 marginally binds → trainreset/pooled views MIS-RANK. Always use SIMSTART for the post-g3 question.
- "post-gate-3 risk zone" CONFIRMED but localizes to **gate-4 specifically** (gate-5 clean, linf 0.056), not the whole 4–5 segment.
- 🚩 contact_true_eval bug FIXED: `per_gate_margin_stats` hard-coded nominal pass_band → margins were radius-invariant; now takes `pass_band`. Multi-radius sweep was meaningless before this.
- 🚩 Task-4 framing CORRECTED: "0.051≈0.06 → track_map accurate" is UNSOUND. Live gate-3 crosses 1.46 m above track_map D-center yet PASS-CLEAN (>outer_half 1.36 m) → track_map gate-3 IS mis-registered ~1.46 m (D-axis). Agreement holds only on E-axis (0.07 m). Gate-3 nominal margin USABLE for ranking (non-binding, live-clean) but NOT as absolute clearance until SHADOWPC-VISION-CAL.
- Gate-4 (binding) shares the same registration risk, NO live cross-check yet → extract gate-4/5 live offsets next ShadowPC touch. Ranking robust; absolute margins provisional.
- D-offset probe now parametric (gates 3/4/5); ALL flip under ±1.5 m → fragility is UNIVERSAL, not gate-4-specific (not evidence any map IS off). Tests 647→657 green.
