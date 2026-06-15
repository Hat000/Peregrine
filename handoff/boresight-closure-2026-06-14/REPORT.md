# BORESIGHT CLOSURE + σ-RECAL → GATE-4 CLOSE/NO-CLOSE VERDICT

**Pathway:** boresight-closure (overall-commander-authored, run as an ultracode/Workflow session, laptop).
**Date:** 2026-06-14 → 15. **Mode:** offline analysis on banked data (NO sim, NO flight).
**Engine:** the production case-C cold-margin stack — `racer.state_estimator.LinearKF` + `racer.kf_rewind.RewindKF`
+ Catmull-Rom truth, via `handoff/margin-closure-envelope-2026-06-14/margin_envelope.py` (`ME`, sourced from
branch `claude/charming-jemison-b111c5`) and the anisotropic/bias-injecting extension
`margin_driver_v2.py` (regression-proven == `ME.fly_lap` to 3.3e-16 in the iso / zero-bias limit).
**Bake branch:** `worktree-wf_43064dee-ce7-1` @ `e345dad` → merged into `claude/jovial-gagarin-65931d`.

---

## 0. VERDICT

> **NO-CLOSE at the measured operating point. The gate-4 cold r=0.30 close/no-close question remains
> CANNOT-SETTLE-OFFLINE — but for SHARPER, now-actionable reasons.** The −0.25 m boresight bake and the
> σ-recal are BOTH load-bearing wins: they CLOSE the **bias axis** (ε_vert removed, validated) and the
> **σ-floor axis** (real σ ≪ the modeled 0.265). They do NOT close gate-4 — they **relocate** the binding
> constraints to **(1) TERMINAL GATE-LOCK** and **(2) at-speed per-fix σ**, neither of which is settleable offline.

**Concretely (post-bake, honest anisotropic σ = [lat 0.191, vert 0.101], COLD case-C, bootstrap-90%-CI-honest):**

| operating point | r=0.30 (M=0.235) | r=0.38 (M=0.155) |
|---|---|---|
| **measured fix-rate 0.07, bias 0** | **NO_CLOSE** (p99 ≈ 0.45–0.47) | NO_CLOSE |
| fix-rate 0.35, bias 0, v=37 | CLOSE (p99 ≈ 0.21) | NO_CLOSE |
| fix-rate 0.50, bias 0 | CLOSE (p99 ≈ 0.20) | NO_CLOSE (closes only at iso-0.10) |
| any fix-rate **with a 0.30 s terminal fix-drought** | **NO_CLOSE** | NO_CLOSE |
| fix-rate 0.50, **bias 0.6°** | NO_CLOSE | NO_CLOSE |

**r=0.30 closes IFF ALL of:** (a) **terminal gate-lock** — accepted gate-4 fixes through the last ~6 m / ~0.15 s
of approach (NOT merely a high pooled fix-rate); (b) effective fix-rate **≥ 0.50** in that terminal window;
(c) at-speed σ_lat **≤ ~0.23–0.245 m** (measured 0.191 at 18 m/s — only ×1.25 headroom, and motion-blur
extrapolation to 30 m/s is ×1.67–2.06, i.e. **past the ceiling**); (d) residual attitude/accel bias **≤ ~0.6°**.
**r=0.38 (worst-case halo) does not close** anywhere in the realistic envelope. The bake is a **necessary,
load-bearing** step (it removes ~0.08–0.15 m of p99) but it is **not sufficient**.

---

## 1. D1 — APPLY + VALIDATE THE BAKE  ✅ (done, suite green, byte-identical)

`frames.BORESIGHT = BoresightCorrection(vert_offset_m=-0.25)` (metric form, +L lever) — committed at
`e345dad`. The `localization._apply_camera_vert_offset` helper reads `frames.BORESIGHT` **live** and applies
`-R_world_body @ [0,0,vert_offset_m]` at BOTH +L sites. Metric is a **translation** → `R_camera_from_body()`
is untouched.

- **Sign — CONFIRMED −0.25 (load-bearing).** Corrected `rel_vert` → **+0.004** (P3 head-on, N=13,308) and
  **+0.035** (L3 gate-4, N=81), both ≈ 0. `vert_offset_m=+0.25` would *double* the bias (→ ≈ −0.50) — so the
  sign is empirically pinned. (+L itself is unchanged; the bake corrects ε, not +L — `test_obs_sign_faithfulness` intact.)
- **Mechanism** — the production lever fix shifts by exactly `-R_wb@[0,0,voff]` at both head-on-level and
  tilted-at-speed attitudes; abs and rel levers shift identically.
- **Attitude-projection nuance** — head-on, the body-Z offset projects ≈ 0.25 onto gate-vertical; at the L3
  tilted attitude (pitch −25°/roll 20°) it attenuates to **≈ 0.21** (crab/yaw do not attenuate the vertical).
  This is why the head-on-calibrated −0.25 leaves a small +0.035 residual at-speed — within the per-fix σ (0.10).
- **Byte-identical** — `R_camera_from_body()` is `np.array_equal` to the 20°-only mount with the bake live;
  the case-A given-pose obs is unchanged → inc7/VQ1/case-A and inc8-OFF stay byte-identical (the bake is case-C only).
- **Suite — 786 passed / 42 skipped (unchanged from baseline).** 9 tests that pinned the *unbiased* +L lever
  output (exact drone-position recovery / +L identity) were updated to monkeypatch a zero `BORESIGHT` for that
  sub-assertion (they test lever mechanics, not the deployed calibration); the default-zero byte-identity
  invariant is preserved as "a ZERO correction → byte-identical mount."

---

## 2. D2 — σ-RECAL: surrogate pooled σ_vert 0.28 vs L3 gate-4 σ_vert 0.10  ✅

**Reconciliation = POOLING-dominant (not axis-convention, not speed-in-band).** Variance-ladder
attribution of the 0.28 → 0.10 vertical gap:

| step | σ_vert | driver |
|---|---|---|
| surrogate pooled floor (10–26 m) | 0.282 | baseline |
| its own 20–26 m sub-band | 0.235 | drop the noisy near-band (strong-perspective PnP) — ×1.20 |
| L3 gate-4 20–24 m | 0.101 | range-match + gate-specific |

≈ **35 %** of the removed variance is near-band pooling, **≈ 65 %** is range-matching to 20–24 m + gate-specific.
- **Axis convention contributes ~0** to the *vertical* gap — surrogate diag `[lat=gateX, vert=gateY, depth=gateZ]`
  matches shadow `rel_cross`/`rel_vert` in the same `gate.R_world_gate` frame. What axis DOES explain is the
  **opposite anisotropy**: surrogate vert(0.28) ≫ lat(0.10), but **L3 gate-4 lat(0.191) > vert(0.101)** — because
  **gate-4 is a high-crab gate (crab 36–54°)** that loads yaw/crab noise onto the cross/lateral axis. (At the
  lower-crab g2, vert > cross, surrogate-like.) Real geometry, not a labelling bug.
- **Speed** explains ~0 of the gap in 0–18 m/s (P3 static-vs-moving rel_vert differ by 0.007 m → speed-flat);
  it is the **OPEN risk above 18 m/s**, not part of the measured gap.
- **Boresight-wander** is a secondary inflator (few cm; the surrogate vertical `bias_band` is non-stationary
  0.01→0.186→0.197 across range bands, mixing into the pooled scatter).
- **Effective-N**: the 81 fixes are from 13 gate-4 windows, but a session-cluster bootstrap gives ICC ≈ 0
  (fixes ~independent) → the σ CIs are essentially iid: σ_vert CI90 **[0.085, 0.117]**, σ_lat CI90 **[0.166, 0.210]**.
- **Recommended margin σ**: anisotropic **[lat 0.191, vert 0.101]** (the honest gate-4 shape), iso-central 0.153.
  New gate-4 checkpoint written to `sigma_gate4_l3.json`. **Do NOT use iso-0.10** — it ignores the dominant
  lateral axis and over-claims closure by 2.3–3.3× in fix-rate.

---

## 3. D3 — GATE-4 COLD-MARGIN RE-RUN  (post-bake + real anisotropic σ)

`MARGIN(r) = W_EFF − r = (0.75 − r) − 0.215`: r0.21→0.325, r0.26→0.275, **r0.30→0.235**, r0.33→0.205, **r0.38→0.155**.
Closure = p90 < M AND p99 < M (CI-honest = upper-CI(p99) < M). Decision cells nmc=5000 + 2000× bootstrap.

**Post-bake, anisotropic σ [0.191, 0.101], COLD, per-radius p99 at the operating points:**

| v | fix-rate | bias | p99 | r0.21 | r0.26 | r0.30 | r0.33 | r0.38 |
|---|---|---|---|---|---|---|---|---|
| 30 | 0.07 | 0 | 0.45–0.47 | n | n | n | n | n |
| 30 | 0.25 | 0 | 0.26 | Y | Y | n | n | n |
| 30 | 0.35 | 0 | 0.234 | Y | Y | **knife** | n | n |
| 30 | 0.50 | 0 | 0.196 | Y | Y | Y | Y | n |
| 37 | 0.35 | 0 | 0.21 | Y | Y | **CLOSE** | n | n |
| 37 | 0.50 | 0 | 0.197 | Y | Y | Y | Y | n |
| 30/37 | 0.50 | 0.6° | 0.25–0.30 | Y | n | n | n | n |

**The bake DELTA (pre ε=+0.25 vs post ε=0), anisotropic σ:** the bake removes **+0.08 to +0.15 m off p99**
(e.g. v37/fr0.25/b0: pre p99 0.401 → post 0.265). Pre-bake fails much harder at every cell — confirming the
ε_vert removal is genuinely load-bearing, not cosmetic. Baseline anchor (iso 0.265, fr0.07): p99 0.69 (sanity vs banked).

---

## 4. ADVERSARIAL VERIFICATION  (2 verdict-moving lenses, fresh independent seeds)

**Lens A — terminal-drought / clustering — REFUTED the mean-fix-rate framing (high confidence).**
The regular-cadence closure is an **artifact of evenly-spaced fixes**. Imposing a realistic **terminal fix-drought**
(no accepted gate-4 fix in the last D s of approach — the camera losing the gate centre on final approach):
- fr=0.50 stops closing r=0.30 at a terminal gap of **D = 0.15 s** (3 of 4 cells); the most optimistic breaks at **0.30 s**.
- **0 of 5 closing cells survive a 0.30 s drought**; at a realistic 0.3–0.6 s gap p99-CIhi runs 0.29–0.49 (25–110 % over margin).
- Dropping fixes only in the **last 4 m** (≈0.11 s @ 37 m/s) already breaks fr=0.50 closure.
- **Burstiness alone does NOT break closure** (bursts that keep firing near the gate retain lock) — the damage is
  specifically the *terminal* drought. → **terminal gate-lock is the true requirement, not a high pooled mean.**
  (Engine `fly_lap_drought(D=0)` reproduces `fly_lap_v2` to exactly 0.0 over 240 laps.)

**Lens B — σ honesty / at-speed ceiling — REFUTED the "fr≥0.35 robust" sub-claim (high confidence).**
- Effective-N survives (ICC≈0); but at the **upper-CI σ**, fr=0.35 fails r=0.30 (both speeds) while fr=0.50 survives →
  the honest fix-rate floor is **0.50, not 0.35**.
- **At-speed σ_lat ceiling for r=0.30 @ fr=0.50 ≈ 0.23–0.245 m** — only **×1.25** above the measured 0.191. A linear
  motion-blur extrapolation 18→30 m/s (×1.67) / 18→37 (×2.06) puts at-speed σ_lat at 0.32–0.39, **far past the ceiling**;
  a mere ×1.3 inflation breaks closure. The crab axis loads blur onto the already-dominant lateral axis.
- Lateral dominance confirmed: naive iso-0.10 would falsely close at fr=0.15 (2.3–3.3× over-claim, +30–41 % p99 understatement).

(The other two designed lenses — bake sign/projection, and CI/knife-edge — are already answered by D1's
`bake_validate` and the bootstrap CIs embedded in the D3 decision cells, respectively.)

---

## 5. CAVEATS / WHAT STILL BLOCKS

1. **Terminal gate-lock (the load-bearing inc8 lever)** — closure needs accepted gate-4 fixes through the last
   ~6 m / 0.15 s, not just a high average fix-rate. Camera-pointing is empirically validated (P3 head-on
   fix-rate 67–77 % vs inc7 1.4 %), but the *terminal* lock is an inc8 training/flight deliverable, not yet demonstrated.
2. **At-speed per-fix σ (the binding unmeasured risk)** — L3 is ~17–18 m/s (inc7); the binding gate-4 is ~30 m/s
   (doctrine) / older 37. σ is speed-flat ≤18 m/s but the ceiling has only ×1.25 headroom and blur extrapolation
   exceeds it. **This genuinely cannot be settled offline — it needs a 30 m/s gate-4 at-speed recording.**
3. **Residual attitude/accel bias ≤ ~0.6°** — the boresight (vertical) bias is now baked out, but the velocity-drift
   accel-bias channel still breaks r=0.30 at bias 0.6° even at fr=0.50 → the ESKF 2-axis accel-bias lever remains relevant.
4. **r=0.38 worst-case halo** — does not close in the realistic envelope; central reporting radius 0.30 is the operative target.

**Net:** the bake + σ-recal are correct, validated, and load-bearing, and they retire the *bias* and *σ-floor*
sub-questions. But "CANNOT-SETTLE-OFFLINE" does **not** flip to a clean CLOSE — it is **refined** into two
concrete, mostly-empirical conditions (terminal gate-lock; at-speed σ ≤ ~0.24), plus the standing ESKF-bias lever.

---

## 6. ARTIFACTS (this pathway, under `handoff/boresight-closure-2026-06-14/`)

- `bake_validate.py` (in the bake commit) — D1 mechanism + sign + byte-identical validation.
- `sigma_recal.py` / `sigma_recal_results.json` / `sigma_gate4_l3.json` — D2 reconciliation + recalibrated checkpoint.
- `margin_driver_v2.py` (anisotropic σ + ε-injection; regression-proven) / `regression_proof.py`.
- `margin_rerun.py` / `margin_rerun_results.json` / `margin_rerun_table.md` / `fine_fixrate_results.json` — D3.
- `verify_clustering*.{py,json,md}` — adversarial Lens A (terminal-drought).
- `verify_sigma*.{py,json,md}` — adversarial Lens B (σ ceiling / effective-N).
- `data/` — materialized raw rows + engine (NOT committed; raw rows live on `hardcore-lehmann` / `blissful-kalam`,
  margin engine on `charming-jemison`).

---

## MEMORY-DELTA (boresight-closure pathway, for the overall commander to bank — NOT written to memory/ by this session)

```
- BORESIGHT BAKE APPLIED + VALIDATED (e345dad, merged to claude/jovial-gagarin-65931d): frames.BORESIGHT=
  BoresightCorrection(vert_offset_m=-0.25), +L lever (case-C only); 786 green, byte-identical (mount array_equal,
  case-A unchanged), sign pinned (corrected rel_vert P3 +0.004 / L3 +0.035; +0.25 would double). Load-bearing:
  removes +0.08..0.15 m off gate-4 p99. Attitude-projection nuance: -0.25 -> ~0.21 gate-vert at the at-speed tilt.
- VERDICT: CANNOT-SETTLE-OFFLINE does NOT flip to CLOSE -- REFINED, not resolved. Bake CLOSES the bias axis +
  sigma-recal CLOSES the sigma-floor axis, but gate-4 r=0.30 is NO_CLOSE at measured fix-rate 0.07 (p99 ~0.45-0.47).
- sigma-RECAL: surrogate 0.28 vs L3 0.10 = POOLING-dominant (35% near-band + 65% range-match/gate); axis labels
  MATCH (gate-4 crab 36-54deg flips anisotropy -> honest gate-4 sigma = ANISO [lat 0.191, vert 0.101], NOT iso 0.10;
  iso-0.10 over-claims closure 2.3-3.3x). Speed-flat <=18 m/s. eff-N fine (ICC~0).
- BINDING LEVER #1 = TERMINAL GATE-LOCK (adversarial-confirmed dominant): a 0.15-0.30 s terminal fix-drought breaks
  r=0.30 at EVERY closing cell (0/5 survive 0.30 s); burstiness alone is benign. Need accepted g4 fixes through the
  last ~6 m, not a high mean fix-rate. Honest fix-rate floor = 0.50 (fr=0.35 fails upper-CI sigma).
- BINDING LEVER #2 = AT-SPEED sigma (UNSETTLEABLE OFFLINE): sigma_lat ceiling ~0.23-0.245 for r=0.30 @ fr0.50, only
  x1.25 over measured 0.191; blur extrapolation 18->30 m/s is x1.67-2.06 -> likely past ceiling. NEEDS a 30 m/s g4 recording.
- ALSO: residual accel/attitude bias must be <=~0.6deg (ESKF lever still live); r=0.38 halo never closes; central r=0.30.
- See handoff/boresight-closure-2026-06-14/REPORT.md. -> [[index-vision-estimator]] [[index-control-sim]]
```
