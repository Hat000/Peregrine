# Productionize the deploy obs[17:20] confidence channel — REPORT

**Session:** deploy-obs20 · **Branch:** `p2-deploy-obs20` (worktree `C:\Users\Fengy\Downloads\Projects\Anduril-obs20`, branched off `main` @ 1e500ee)
**Date:** 2026-06-19 · **Status:** DONE — all definition-of-done items hold (see Verdict).

---

## TL;DR
The deployed case-C stack now emits the **full 20-dim** inc8 policy observation. A production confidence-triple
builder (`confidence_triple` / `confidence_triple_from_sigmas` / `estimator_obs20` / `estimator_obs_auto`) was
promoted into `src/racer/estimator_obs.py`, **byte-faithful** to the trained encoder
(`EstimatorEmulator.confidence_channel`) — production-vs-trained worst delta **2.93e-08** (float32 epsilon),
production-vs-spike **0.0** (bit-identical), across a 360-case pose/cov/staleness/gate-frame sweep. The √2
reconciliation (R3 footgun) is applied **once, in the obs builder**; the NavState wire export stays
un-reconciled. **inc7 17-dim stays byte-identical** (R1 == 0.0). **Design = Path A** (NavState contract only) —
and it required **no NavState/navigator/contracts change at all**: the contract already carried every input.

---

## Design choice: Path A (NavState contract) — and why no contract extension was needed
The leading hypothesis (Path A: source the triple from the `NavState` export) is the production design. The key
discovery: **`NavState` already exports all three inputs** the builder needs, so Objective B's "extend NavState
additively" collapsed to **"no extension required"** — the cleanest possible outcome (zero blast radius on
`contracts.py` / `navigator.py` / `state_estimator.py`):

| obs[17:20] element | source on NavState | reconciliation |
|---|---|---|
| `c_inplane` | `nav_inplane_sigma` = √(P_E+P_D) | **÷√2** → σ̂_ip = √((P_E+P_D)/2) (the d5 average) |
| `c_along`   | `nav_along_sigma` = √(P_along) | none (identical definition) |
| `age_norm`  | `time_since_vision_update_s` | clip(t/τ, 0, 1) |

The **√2 lives in the obs builder** (`confidence_triple`, one named line), per footgun #3 / R3's stated intent
("the OBS BUILDER applies the correction"). The navigator export stays the sum-sqrt (R3 pin intact). This keeps
`NavState` agnostic of the policy obs encoding and keeps the deploy seam reading only the contract (consistent
with how it already reads `position_ned`/`velocity_ned`). Path B (raw KF `P` + `_last_fix_gate_R`) would couple
the seam to Navigator internals; we use it only as a **cross-check** (and confirm Path A == Path B to 0.0 on a
real navigator run).

---

## Per-objective results

**A. Production confidence-triple builder (pure numpy, torch-free, √2-correct).** DONE.
`confidence_triple_from_sigmas` (the d5 encoder core) + `confidence_triple` (NavState path, applies ÷√2).
Cross-validated vs BOTH the spike and the trained emul over a 360-case sweep (cov scale 0.005→0.5, staleness
0→0.3 s, gate yaw −π→π):
- production vs **trained emul** (`EstimatorEmulator.confidence_channel`): **2.93e-08** (float32 epsilon)
- production vs **spike** (`deploy_confidence_triple`): **0.0** (bit-identical)
- spike vs trained emul: **2.93e-08**
The √2 is load-bearing: feeding NavState raw (no ÷√2) mis-reports c_inplane by **0.103** in a representative case
(0.248 raw vs 0.3508 correct == the emul's value); ÷√2 recovers it exactly. Consistent with R3's "up to ~0.29".

**B. Clean input seam.** DONE — **no NavState extension needed** (see Design above). All existing NavState pins
stay green (`test_sysid_confidence_constants`, `test_estimator_obs_wiring`).

**C. 20-dim production obs == trained 20-dim obs (elementwise).** DONE. `estimator_obs20` = [0:17] via the
existing seam ++ [17:20] via `confidence_triple`, assembled exactly like `EstimatorEmulator.obs(obs_dim=20)`
(concat, float32 cast). Over a 120-case pose/vel/attitude/cov/staleness sweep: **obs[0:17] delta == 0.0**
(byte-identical) AND **obs[17:20] ≤ 1e-6** (measured float32 epsilon).

**D. inc7 17-dim byte-identical.** DONE. `estimator_obs` / `estimator_state_for_obs` are UNCHANGED (only
additions below them). R1 == 0.0; `test_inc7_17dim_path_byte_identical` green; the 20-dim path is strictly opt-in.

**E. Deploy SELECTION seam.** DONE as a **tested, documented seam, not a live-loop rewire**. Added
`estimator_obs_auto(..., actor_obs_dim)` → dispatches 20-dim for a ≥20-wide actor, 17-dim otherwise (the
obs-layer dispatch, ready for the deploy loop). **Deliberately did NOT touch `fly_rl.fly_once`**: that live loop
is the **case-A wire-position** path (`build_obs(s, ...)`, hardcoded 17-dim probe at `fly_rl.py:1305`), NOT the
case-C estimator path — wiring the navigator + 20-dim selection into it is the separate POC-flight deploy
integration and would touch the JUDGED inc7 path (footgun #1). The selection logic lives cleanly in the obs
layer; the loop only needs to choose by width when case-C self-localization is wired.

**F. Tests.** DONE.
- `test_sysid_production_obs.py` **R2 flipped** absent→present+faithful (drives production vs trained emul,
  ≤1e-6); **R1 (byte-identity) and R3 (√2 wire pin) kept green**.
- New `test_deploy_obs20.py` (11 tests): constants pin · three-way builder parity sweep · from-sigmas core ==
  spike bit-exact · full-20 elementwise parity · inc7 byte-identity · cold-start [0,0,1] · stale age-ramp ·
  degenerate/over-converged (σ→0 → c=1) · last-fix gate-frame (Path A == Path B on a real navigator run) ·
  dispatch-by-width · torch-free subprocess.

**G. green_gate.** Load-bearing **invariants GREEN** (OFF==inc7 AST parity, +L sign-faithfulness — actually
ran, VQ1 import guard) and **sentinel GREEN** (collected 1048 ≥ 933; final run: diff-subset `4 failed, 832
passed, 6 skipped`). The diff-scoped pytest is **RED ONLY on 4 pre-existing failures in `tests/test_measured_aero.py`** (`TypeError: object.__init__() takes exactly one
argument` at `rl/diffaero_dynamics.py:306`) — a **diffaero-absent stub-ordering artifact** entirely unrelated
to obs[17:20]. **PROVEN pre-existing:** running the identical module batch on the **pristine `main` checkout
(my change absent)** fails with the SAME 4 tests (`4 failed, 826 passed`); `test_measured_aero` passes in
isolation in both checkouts; my new tests don't pollute it (`39 passed` together). The 63-module subset is
pulled in only because green_gate maps a change to `estimator_obs.py` onto the parent package name `racer`
(≈ every test imports it). **My diff introduces NO regression** — every obs/estimator/confidence/spike module
+ all 3 invariants + the sentinel are GREEN. Recommendation: the pre-existing `test_measured_aero` isolation
flake is out of scope here (diffaero dynamics-test infrastructure, footgun #7); flag it separately.

---

## Parity / test evidence
- New file `tests/test_deploy_obs20.py`: **11 passed**.
- `tests/test_sysid_production_obs.py` (R1/R2/R3): **3 passed**.
- Focused regression set (estimator_obs / confidence / spike / emul / +L sign / OFF-identity): **49 passed, 1
  skipped** (the skip = `test_full_slice_action_golden`, needs the gitignored inc8 `.pth`).
- Baseline collection 1036 → +11 new tests; sentinel floor 933 (≥, stays green).

## Edge-case semantics (all match the d5 encoder / spike)
- **Cold-start / no fix yet:** navigator exports `inf` sigmas + `inf` clock → triple `[0, 0, 1]` (zero
  confidence, fully stale), no NaN. *Note (not a bug):* the trained emul's cold-start uses a FINITE `pos_std=1.0`
  cold P → c≈0.05, because the emul always knows the gate frame; the deploy navigator has **no gate frame until
  the first accepted fix** so it honestly reports `inf`→c=0. This is an **estimator-state** difference at
  cold-start only (reconciles after the first fix), NOT a builder-faithfulness gap — the builder is byte-faithful
  given the same inputs (the parity sweep drives matched inputs).
- **Over-converged cov (inplane-floor OFF, #74 regime):** σ→0 → c=1.0 via the `if σ>0 else 1.0` guard
  (byte-identical to emul + spike).
- **Stale fix:** `age_norm = clip(t/0.10, 0, 1)`.

## Footguns honored
inc7 byte-identity (R1==0.0); √2 applied in the builder, NOT reconciled in NavState (R3 intact); degenerate
σ≤0→c=1.0; +L convention untouched; src/racer stays torch-free (subprocess-pinned); memory/ untouched; only own
paths staged; no push.

## Adversarial review (ultracode workflow — 4 lenses, 9 agents)
Ran a 4-lens adversarial review (faithfulness/√2 · byte-identity/regression · edge-cases/degenerate ·
contract/torch-free/consumers) with per-finding adversarial verification. **20 praise findings** independently
confirmed the core claims: byte-faithfulness to the trained encoder; the ÷√2 reconciliation is exact and the
sqrt(s)/√2-vs-sqrt(0.5·s) float pitfall is negligible (0.0 after float32 pack); the per-axis-vs-sum clamp
**cannot diverge for a real PSD KF cov**; inc7 17-dim seam unchanged + the triple strictly appended; torch-free
preserved; Path A correct (no contract change needed); R2 flipped correctly + R3 intact; leaving `fly_rl.main`
(case-A) untouched is the right Objective-E call.

**3 confirmed issues, all NON-BLOCKING — all addressed:**
1. *(minor/major across two lenses; agreed minor-bounded)* **Cold-start train/deploy gap**: pre-first-fix the
   deploy navigator exports `inf` → triple `[0,0,1]`, while the trainer's cold KF (finite P=I) → `[0.05,0.05,1]`.
   An **estimator-state** semantic (navigator has no gate frame until the first fix), **not a builder bug** —
   byte-faithfulness holds *given the same inputs*. Both flag the channel stale (`age_norm=1`); converges on the
   first fix. **Action taken (Option B = document + pin, the recommended proportionate response):** added a
   docstring note to `confidence_triple` and a characterization test
   `test_cold_start_train_deploy_divergence_documented` that drives the emul cold start (asserts `[0.05,0.05,1]`)
   and locks the deploy `[0,0,1]` divergence as explicit/known. *Option A (give the navigator a finite cold
   sigma) was NOT taken* — it changes the deploy obs and should be gated on flight evidence that the 0.05 cold
   gap matters; surfaced in the MEMORY-DELTA as a deploy-integration carry-forward.
2. *(minor, doc-only)* **Torch-free test docstring** overstated coverage (claimed it exercises
   `estimator_obs_auto(17-dim)`, which the snippet never calls and which is *not* torch-free). **Fixed** the
   docstring.
3. *(minor)* **Floor cross-pin comment** cited a pin the test didn't add. **Fixed** by adding
   `INPLANE_POS_FLOOR_STD == SIGMA_REF_M` to the constants pin test, making the comment accurate.

Noted-and-accepted (faithful-by-design, no change): a NaN sigma reads as `c=1.0` via the `if σ>0 else 1.0`
guard — byte-identical to the trained emul (faithfulness wins; a NaN sigma cannot arise from the navigator,
which clamps). `estimator_obs_auto` uses `>=20` (a wider-actor-gets-the-channel semantic; only 17/20 exist).
`estimator_obs20`'s float32 double-cast is a verified bit-idempotent no-op.

---

## MEMORY-DELTA (≤10 lines, for the commander to bank)
- ✅ **DEPLOY obs[17:20] PRODUCTIONIZED & MERGE-READY** (branch `p2-deploy-obs20`, off 1e500ee): `confidence_triple`/`confidence_triple_from_sigmas`/`estimator_obs20`/`estimator_obs_auto` in `src/racer/estimator_obs.py` — the inc8 20-dim deploy obs now builds. The obs[17:20] gap (sys-id R2) is CLOSED.
- **Design = Path A (NavState contract); NO contracts/navigator/state_estimator change needed** — NavState already exports `nav_inplane_sigma`(√(P_E+P_D))/`nav_along_sigma`(√P_along)/`time_since_vision_update_s`. The **÷√2 lives in the obs builder** (R3 export stays un-reconciled, intact).
- **Parity (empirical):** production-vs-trained-emul **2.93e-8** (float32 ε), production-vs-spike **0.0** (bit-identical), full-20 obs[0:17]=**0.0** byte-id. inc7 17-dim BYTE-IDENTICAL (R1==0.0). 12 new tests (`tests/test_deploy_obs20.py`) + R2 flipped present+faithful.
- 🚩 **CARRY-FORWARD (deploy-integration): cold-start gap.** Pre-first-fix, deploy obs[17:18]=**0** (navigator exports inf until a gate frame anchors) vs trained **~0.05** (emul cold P=I). BOUNDED, `age_norm=1` flags stale in BOTH, converges on first fix; characterized+pinned. Fix only if flight shows the cold window matters = give navigator a finite cold gate-frame sigma (Option A, changes deploy obs — gate on evidence).
- 🚩 **Objective E (selection) = tested obs-layer seam (`estimator_obs_auto`), NOT a live-loop rewire.** `fly_rl.main`/`fly_once` is the case-A wire-position path (hardcoded 17-dim); wiring case-C self-loc + 20-dim selection into it is the POC-flight deploy-integration step (touches the JUDGED inc7 path — deferred deliberately).
- 🚩 **green_gate RED is a PRE-EXISTING UNRELATED FLAKE, not this diff:** `tests/test_measured_aero.py` (4 fails, `object.__init__` at `diffaero_dynamics.py:306`) is a **diffaero-absent stub-ordering** artifact — PROVEN by running the same batch on pristine `main` (same 4 fail / 826 pass, change absent). Invariants + sentinel GREEN; all obs/estimator modules GREEN. (green_gate over-maps `estimator_obs`→parent pkg `racer`→63 modules.) Worth a separate cleanup task.
- Adversarial review (4 lenses/9 agents): 20 praise, 3 non-blocking issues ALL addressed (cold-start doc+test, torch-free docstring, floor cross-pin). No bugs found in the √2/byte-identity/torch-free core.
