# VQ2 Deploy — Opt-in Ensemble Support in the Flight Detector — REPORT

**Date:** 2026-06-19 · **Session:** vq2-ensemble-support (Opus 4.8, ShadowPC) · **Branch:** `vq2-ensemble-support` (off `main`/ea71423; push for selective-merge)

## VERDICT: DONE — opt-in `EnsembleGateDetector` + per-gate dedup delivered, unit-tests GREEN, single-model path proven byte-identical, latency in budget. Unwired (go-live still gated on inc8).

This is **deploy-plan step 1**: the detector can run the adopted clean ensemble *when selected*; the VQ1 single-model flight path is untouched. No closed-loop flight, no live wiring, no weight-pointing (all GO-LIVE, held on the inc8 case-C controller).

---

## What this branch is (and why it exists)

The ensemble detector itself was **already built + offline-validated** on the deferred `claude/vq2-flight-validation-2026-06-19` branch (commit 17da0ba). But that branch is **blocked on inc8**, entangled with the flight-gate driver + an `AsyncGateDetector`, **never pushed**, and the detector **had no unit tests** (`tests/test_detector.py` there is byte-identical to `main` — it tests only the legacy single-model core). So this branch does the missing deploy-prep work as a **clean, focused, selectively-mergeable unit off `main`**:

- **Ports** the proven plumbing into `src/racer/vision/detector.py`: the 8→4 inner-corner slice, the `++` opt-in routing in `GateDetector.load`, and `EnsembleGateDetector` (union + per-gate dedup). Logic is faithful to the validated version (identical `detect`/`_dedup`/`_fuse`).
- **Excludes `AsyncGateDetector`** — that is the flight-gate's async-vision finding (sync `detect()` starves the 100 Hz loop), a *separate* concern gated on inc8, mergeable on its own. Keeping it out keeps this branch tightly scoped to ensemble support (and drops the `threading`/`time` imports).
- **Adds the real unit-test suite** (`tests/test_ensemble_detector.py`, 12 tests) + a reproducible latency script.

---

## Design — where the dedup sits, and why

**The dedup lives in `EnsembleGateDetector._dedup`, at the detector boundary, BEFORE association** (not in `association.py`, not in the navigator).

- **The problem (deploy ≠ offline eval).** The offline `eval_ensemble.py` unions detections with **no dedup** — safe *there* because its metrics self-select per gate (course = `any(obs<3m)`, good-fix = `min di`). The deploy path feeds the union into `navigator._maybe_run_vision` → per-obs `_process_observation` → `KF.update_position`. That loop applies **one fix per observation** and does **not** dedup per `gate_id` (verified in `navigator.py:446-454`). So two members both detecting gate-3 → two gate-3 fixes → the KF **double-counts** (its position covariance shrinks ~2× too fast, over-trusting that gate).
- **Why the detector, not association.** Deduping at the detector means `detect()` returns one observation per gate, so the navigator receives exactly one fix per gate **with zero navigator edits** — the ensemble is a pure drop-in detector and the VQ1-proven live path (association → PnP → KF → innovation gates) stays byte-unchanged. A post-association dedup keyed on `gate_id` would be more "semantic," but it would require editing the live `_maybe_run_vision` loop — exactly the path the guardrail says to leave untouched.
- **The dedup rule (light, geometric).** Greedy by detection score; cluster a later detection with a kept one iff its 4-corner **centroid is within `dedup_px`** AND its **apparent span is comparable**, then **fuse** the cluster (score-weighted corner average of the full-4-corner members; `corner_confidence` = elementwise max). This maps cleanly to "one per gate" because the ensemble's duplicates are **near-coincident by construction** — both members localize the same physical square in the same image (sub-px-to-few-px centroid agreement, near-identical span). The **span guard** is the one safety that pure-pixel NMS lacks: on the receding-collinear course a near (big) and far (small) gate can share a centroid, and a centroid-only merge would drop a real gate; requiring comparable span keeps them separate (same overfit-GEOMETRY spirit as `association.py`).
- **Opt-in toggle.** `dedup_px=12` is the deploy default (dedup ON — never double-count by default). `dedup_px=0` recovers the pure union (the offline reference / an A/B knob). Members ≥2 only: a single weight never enters this code.
- **Fuse vs keep-best (honest note).** Fusing gives the KF a noise-reduced corner set and lets a member whose PnP is better than its box-score implies still contribute (the offline A/B favoured fuse over keep-highest-score). One minor edge: a mixed cluster with <2 full-4-corner members falls back to the **top-score** member, which can be a 3-corner (P3P) obs even when a lower-score 4-corner exists. It is harmless (the navigator handles 3-corner via P3P; the validated good-fix metric filters to 4-corner so it was insensitive to this) and is left faithful to the validated implementation rather than changed unvalidated.

## Opt-in / byte-identical guardrail

`GateDetector.load("weights.pt")` → a plain `GateDetector` (the `++` branch is not taken) → `detect()` is the **unchanged** method. Only an `a++b` spec (or constructing `EnsembleGateDetector` directly) engages the union+dedup. The 8→4 slice in `observations_from_results` is a no-op for native 4-keypoint models and only subsets the inner-4 for 8-keypoint models (the deployed champion is `vq2_pose_8kp`). Net: the single-model path is additive-only and byte-identical; ensemble + dedup run **only** for N≥2.

---

## Unit-test results — GREEN

`tests/test_ensemble_detector.py` — **12/12 pass**, offline (duck-typed fakes, no ultralytics / no weights / no GPU):

| Objective | Test(s) | Asserts |
|---|---|---|
| **A. Union** | `test_ensemble_unions_member_detections`, `test_pure_union_keeps_duplicates_when_dedup_off` | members' detections concatenate; `dedup_px=0` keeps duplicates (= offline eval) |
| **B. Per-gate dedup** | `test_dedup_collapses_to_one_observation_per_gate`, `test_dedup_toggle_is_the_only_difference`, `test_dedup_size_guard_does_not_merge_overlapping_collinear_gates`, `test_dedup_passthrough_for_empty_and_single` | union=4 → **2 (one per gate)**; toggle 0↔12 is the only difference; **span guard keeps overlapping collinear gates separate**; empty/single passthrough |
| **B. Fuse** | `test_fuse_is_score_weighted_corner_average`, `test_fuse_single_member_is_identity` | fused corners = score-weighted average, conf = elementwise max, stays 4-corner |
| **C. Opt-in / byte-identical** | `test_single_weight_load_stays_plain_gatedetector`, `test_single_model_detect_is_byte_identical`, `test_ensemble_does_not_mutate_member_observations` | single-weight load → `GateDetector` (not Ensemble); `a++b` → Ensemble, 2 models, `dedup_px=12`; **ensemble-of-one detect == GateDetector.detect field-for-field** |
| **D. 8→4 adapter** | `test_8kp_results_subset_to_inner_four_and_pnp_recovers`, `test_ensemble_of_8kp_models_feeds_pnp_four_corners` | 8-kpt → **exactly 4 inner corners** to PnP; `estimate_gate_pose` recovers pose (n_corners=4); ensemble obs are never 8-corner |
| **E. Dedup latency** | `test_dedup_overhead_is_submillisecond` | 12-obs dedup < 1 ms/frame |

**No regression:** `tests/test_detector.py` 12/12 (unchanged single-model core) + downstream `test_navigator` / `test_navigator_gate_relative` / `test_mission` / `test_association` / `test_gate_pose` **69/69**. Total **93 GREEN**.

## Latency — measured (objective E)

`handoff/vq2-ensemble-support-2026-06-19/measure_latency.py` on the worktree detector + real weights, 40-frame `task2_frames` bundle, ShadowPC RTX 2000 Ada:

| config | mean | p95 | obs/frame |
|---|---|---|---|
| champion (single model) | 18.8 ms | 23.4 ms | 2.80 |
| ensemble pure-union (`dedup_px=0`) | 38.0 ms | 43.0 ms | 6.53 |
| **ensemble fuse-dedup (`dedup_px=12`, deploy)** | **39.4 ms** | **43.7 ms** | **3.58** |

→ **p95 43.7 ms ≤ 50 ms budget = PASS.** Dedup collapses **6.53 → 3.58 obs/frame (45% fewer KF updates/frame)** — an exact match to the flight-val figure, confirming the port behaves identically. (2-model inference ≈ 2× single; the dedup itself adds ~1 ms.)

---

## Guardrails honored

Round-1 champion + legacy single-model path **untouched** (byte-identical, proven by test + 2.80 obs/frame unchanged); ensemble **NOT wired** into the live stack, no default flipped, no weight-pointing (go-live held on inc8); worked in the harness-provided worktree on **feature branch `vq2-ensemble-support`** (not `main`); weights stayed local (gitignored, referenced by absolute path); `memory/` not edited (MEMORY-DELTA below).

## For the commander (selective merge)

- **Merge target:** `src/racer/vision/detector.py` (8→4 slice + `++` routing + `EnsembleGateDetector`) and `tests/test_ensemble_detector.py`. Both apply cleanly on `main`.
- **Not included by design:** `AsyncGateDetector` (flight-gate async-vision concern) — pull from `claude/vq2-flight-validation-2026-06-19` when the inc8 flight gate is taken up.
- **When inc8 lands:** the flight gate runs immediately — select the ensemble via `GateDetector.load("…precision_L107.pt++…course_L110.pt")` (dedup ON by default), wrap in the async detector for the live loop, run `validate_ensemble_detector.py` to re-confirm offline, then the paired closed-loop gate.

---

## MEMORY-DELTA (≤10 lines; for the commander to bank — memory/ not edited here)

1. **VQ2 deploy ensemble plumbing DELIVERED** on focused branch `vq2-ensemble-support` (off main): opt-in `EnsembleGateDetector` (union + per-gate fuse-dedup) + 8→4 inner-corner slice + `GateDetector.load("a++b")` routing. Single-model path **byte-identical** (proven by test). **93 tests GREEN** (12 new ensemble + 12 detector + 69 downstream). Latency p95 43.7 ms ≤ 50 ms; dedup 6.53→3.58 obs/frame (45% fewer KF updates).
2. **Dedup sits in the DETECTOR, before association** (not in navigator/association) — so the VQ1 live path (assoc→PnP→KF) is untouched; the ensemble is a drop-in detector. The KF double-counts a gate if the raw union reaches it (navigator applies one fix per obs, no per-gate_id dedup), which is exactly why deploy needs the dedup the offline eval omits.
3. **The ensemble was already built+validated on the DEFERRED `vq2-flight-validation` branch but never unit-tested or pushed**; this branch extracts it clean for selective merge and adds the missing pytest suite. `AsyncGateDetector` deliberately left on the flight-val branch (separate async-vision concern, gated on inc8).
4. Go-live unchanged: ensemble unwired, weights not pointed, no closed-loop flight — all held on the inc8 case-C controller retrain.
