> 🚩 **SUPERSEDED 2026-06-19 (511e85c) re: the σ_p0 bar.** Any "σ_p0 ≲ 0.08 / NO-GO vs 0.08 / near-field-gate-estimator pivot" conclusion in this report is OVERTURNED — the 0.08 bar was a `margin_envelope.py` double-count of the drone (real gate clearance 0.75−r ≈ 0.37–0.47 m). The REAL bar is **σ_p0_lat ≲ 0.15 (p99 ≲ 0.45)**; measured 0.15–0.20 = MARGINAL-PASSING, NOT NO-GO. Gate-4 is a REACH/PASS-RATE problem; RL stays the tool (the "pivot off RL" is RETRACTED). The engineering + measurements below STAND; only the bar and its GO/NO-GO verdict are corrected. → MEMORY.md:8 + memory/project_rl_increment_history.md §inc8-2026-06-19.
# PORT look-at into the σ_p0 eval harness — REPORT (2026-06-18)

**Session:** make `contact_true_eval` faithful for inc8 (port the look-at primitive), re-fly rc1, re-measure σ_p0.
**Branch:** `p2-eval-lookat` (off main @ 6c7e6bb). green_gate GREEN (947 tests; OFF==inc7 parity + +L sign GREEN).
**Files (committed on branch, NOT memory/):** `rl/contact_true_eval.py`, `rl/estimator_emul.py`, `rl/inc8_sigmap0_eval.py`.

---

## RETURN (the two asks)

**Does inc8 fly with look-at ported? → PARTIALLY / NO as a clean win.** The port is correct, but it revealed the
binding gap is **NOT the look-at omission — it is the START distribution** (+ a pitch-axis plant-convention
issue). With the fix:
- From the **training-native `trainreset` start**, **seed 0 flies and reaches gate-4 (136/200 = 0.68)**.
- From the **`simstart`** the σ_p0 eval hardcoded (a *deploy* nose-first spawn the policy never trained on),
  **NO seed flies** (0/200), with or without look-at.
- Seeds 1 & 2 die at gate-1 even from `trainreset` (eval generalization, see §4).

**σ_p0 read (seed 0, trainreset, yaw-only look-at, 200 ep):** `σ_p0_lat = 0.177 m` (bias −0.095, p90 0.323,
p99 0.375) → **NO-GO vs the 0.08 target**. 🚩 **PROVISIONAL & likely pessimistic**: measured under DEGRADED
pointing (pitch look-at blocked — see §2 — so gate-4 band fix-rate ≈ 0; the KF coasts into gate-4). The
faithful 2-axis σ_p0 is not yet measurable.

---

## 1. THE LOOK-AT PORT (what landed)

Training (`peregrine_racing_inc8.step:233-246`) composes a camera→gate body-rate correction onto the CTBR
action *before* the dynamics, gated to the approach band. The eval omitted it. The port replicates it exactly:

- **Source (`rl/estimator_emul.py`):** expose `self._last_geom = geom` in `EstimatorEmulator.step()` (+ init/
  reset to `None`). `geom` is the `FS.geometry` already computed each step; its `.t_cam` (gate centre in the
  camera frame) and `.range_m` are byte-identical in formula to the torch emulator training reads
  (`R_cb @ Rwb^T @ lever`). No other emulator behaviour changes.
- **Composition (`rl/contact_true_eval.py`):** after `policy_step`, reuse the **pinned** `inc8_reward.
  lookat_correction` (+ baked `r_body_from_camera` 20° matrix — NOT reimplemented; pinned by
  `tests/test_inc8_lookat`). Gate on band `r∈[8,30] & t_cam_z>0`. Convention: `policy_step` returns
  `rate_frd = (virtual_flip(act[1:4]))·_FLIP`, so `rate_frd·_FLIP` recovers the **real FLU action** training
  adds `dlook` to → compose in FLU, clamp to `[_ACT_MIN,_ACT_MAX]`, map back to FRD. wf=1 (warmup complete).
- **inc8-only gate:** `apply_lookat = estim_emul & not emul_passive & obs_dim>=20`. inc7 (17-dim) and the
  truth/passive paths get NO look-at.

Diff is small and self-contained (3 files). The eval now also accepts `--start {simstart,trainreset,racestart}`
and `--lookat {auto,off,yaw}`.

## 2. 🚩 PITCH-AXIS BLOCKER (the convention trap the prompt warned about)

Sweeping all look-at sign combos from `trainreset` (seed 0): **yaw-only look-at flies (13/40 reach g4, matches
no-lookat 12/40); ANY pitch correction (g_pitch=±3, either sign) collapses flight to 0/40 (dies after gate-0).**

This is the documented **"CTBR/VQ1 LEGACY SIGN CONFIG is a self-consistent alias"** footgun. The eval's
`plant_step` carries a legacy rate-sign alias (`_RATE_SIGN_LIVE=[1,1,-1]`) that the *policy's own output* is
self-consistent with (so inc7 + inc8-no-lookat fly), but an **externally-injected physical pitch correction
does not compose through it** on the body-Y axis. Yaw (body-Z) survives ("body z is body z under both flip and
FLU→FRD", policy_step:541); pitch does not. `lookat_correction` itself is correct (pinned) — the mismatch is
between training's diffaero plant convention and the eval's alias on the pitch axis. **Resolving it needs the
diffaero↔plant_step pitch-rate map; I did NOT force a sign (both break) — flagged for follow-up.**

🚩 **Why this matters for the GO:** gate-4 is the **vertical-σ-dominated** gate (MEMORY), and **pitch is the
elevation-pointing DoF**. With pitch look-at blocked, gate-4 band fix-rate ≈ 0 (the camera isn't elevation-
pointed) → the KF coasts → the σ_p0 above is measured WITHOUT the elevation pointing the policy trained with.
A faithful gate-4 σ_p0 REQUIRES the pitch injection. This is the load-bearing follow-up.

## 3. inc7 BYTE-IDENTICAL (proof)

inc7 (17-dim) eval signature over 30 episodes (simstart + trainreset, estim_emul, mixer), sha256 of all
crossings/outcomes/final-positions:
- branch `p2-eval-lookat`: `c705e39d3d23c209a896d90eb123645e39d581b5f49b5ca843bd58c5a16a00d8`
- main (stashed): `c705e39d3d23c209a896d90eb123645e39d581b5f49b5ca843bd58c5a16a00d8`  → **IDENTICAL.**

Structurally guaranteed (apply_lookat=False for obs_dim<20; `_last_geom` is written-but-never-read for inc7) +
empirically confirmed. green_gate `OFF==inc7 AST parity` also GREEN.

## 4. DISAMBIGUATION (look-at / start-dist / plant-DR)

| factor | finding | binding? |
|--------|---------|----------|
| **look-at (yaw)** | ported faithfully; flies; inc7 byte-identical | NO (resolved) |
| **look-at (pitch)** | breaks the eval plant (alias, both signs); blocks faithful gate-4 fixes | YES (2nd) |
| **start-distribution** | `simstart` (deploy spawn) = 0 flight all seeds; `trainreset` (training-native) flies seed 0 | **YES (1st)** |
| **plant / generalization** | inc7-on-emul also degraded (4/40 to g4); inc8 seeds 1&2 die at gate-1 from trainreset; only seed 0 generalizes to the fixed-start + estim_emul + mixer slice | YES (3rd) |

**The eval is a narrow slice** (one fixed start + estim_emul + deterministic mixer plant) of training's
2048-randomized-start + full-DR distribution. The previous "look-at omission" diagnosis was incomplete: look-at
(yaw) was necessary but the **dominant** reason σ_p0 was unmeasurable on `simstart` is that the policy was never
trained on that deploy spawn.

## 5. σ_p0 TABLE (trainreset, yaw-only look-at, estim_emul, 200 ep)

| seed | reach@g4 | σ_p0_lat | bias_lat | lat p90/p99 | σ_p0_vert | term_lock | g4_fix | GO |
|------|----------|----------|----------|-------------|-----------|-----------|--------|-----|
| 0 | 136/200 (0.68) | **0.177** | −0.095 | 0.323/0.375 | 0.169 | 0.072 | ~0.000 | NO-GO |
| 1 | 0/200 | — | — | — | — | — | — | NO-DATA |
| 2 | 0/200 | — | — | — | — | — | — | NO-DATA |

Best seed = **seed 0**. σ_p0_lat 0.177 m ≈ 2.2× the 0.08 target → NO-GO **but PROVISIONAL** (degraded pointing,
gate-4 fix-rate ≈ 0; the true 2-axis σ_p0 awaits the pitch-injection fix). From `simstart`: all seeds 0/200.

## 6. NEXT (recommended — for the commander)

1. **Fix the pitch-axis look-at injection** through the eval's `plant_step` alias (map the diffaero↔plant_step
   body-Y rate convention) — this is load-bearing for a faithful gate-4 (vertical-σ-dominated) σ_p0; until then
   gate-4 fix-rate ≈ 0 and σ_p0 is pessimistic.
2. **Standardize the σ_p0 start on a training-representative distribution** (trainreset / randomized + standing
   mix), NOT the deploy `simstart`. simstart is a separate deploy-generalization question.
3. **Seeds 1 & 2 don't generalize** to the fixed-start eval slice (die at gate-1) though they trained to
   success_rate ~0.5 — investigate start/plant-DR alignment (eval is deterministic mixer; training had full
   DR + randomized starts). σ_p0 across-seed needs this.
4. INTERPRETATION FRAME (unchanged): even a future GO = "policy centers to σ_p0=X under the EMUL's fix quality";
   the real-8-kpt-detector per-fix accuracy (best.pt not staged) is the final confirm.

---

## MEMORY-DELTA (text only — do NOT commit memory/)

```
EVAL LOOK-AT PORT + σ_p0 re-measure (2026-06-18, branch p2-eval-lookat, green_gate GREEN 947):
- PORTED look-at into contact_true_eval (reuse pinned inc8_reward.lookat_correction; _last_geom exposed in
  numpy EstimatorEmulator.step; compose in real-FLU via rate_frd*_FLIP, clamp, back to FRD; inc8-only gate
  obs_dim>=20). inc7 BYTE-IDENTICAL proven (sha c705e39d branch==main, 30 ep) + green_gate OFF==inc7 GREEN.
- 🚩 BINDING GAP WAS THE START, not look-at: simstart (DEPLOY nose-first spawn, virtual_flip) = 0 flight all
  seeds; trainreset (TRAINING-native gate-relative) = seed 0 flies, reach@g4 136/200 (0.68). Prior 'look-at
  omission' diagnosis incomplete.
- 🚩 PITCH look-at BREAKS the eval plant (g_pitch=±3 both -> 0/40; yaw-only flies) = the LEGACY-ALIAS footgun
  (plant_step rate-sign self-consistent for policy output, NOT for an injected physical pitch correction;
  body-Z/yaw survives, body-Y/pitch doesn't). lookat_correction itself correct (pinned). DID NOT force a sign.
- σ_p0 (seed 0, trainreset, YAW-ONLY, 200ep) = lat 0.177 bias -0.095 p99 0.375 -> NO-GO vs 0.08, but
  PROVISIONAL/pessimistic: pitch blocked => gate-4 band fix-rate ~0 (elevation pointing is the gate-4-binding
  axis). Seeds 1,2 die at gate-1 from trainreset (don't generalize to the fixed-start+emul+mixer slice);
  inc7-on-emul also degraded (4/40). NEXT = fix pitch injection (diffaero<->plant_step body-Y map) + training-
  representative start + seed1/2 start/plant-DR alignment, THEN re-measure faithful σ_p0.
```
