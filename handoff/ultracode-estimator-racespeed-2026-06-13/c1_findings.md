# c1 — GATE-RELATIVE observation branch (estimator-racespeed)

Agent **c1** (estimator/perception, opus-4.8). BRANCH: gate-relative observation — track drone
position RELATIVE to the upcoming gate-4 from the PnP translation `t_cam_gate`, so the absolute
world-fix BIAS (global + per-gate map error) becomes IRRELEVANT to the in-plane miss the drone
physically must hit. project-estimator-robustness layer-3(c) (sanctioned). Offline analysis +
sim only. Composes the REAL `racer.state_estimator.LinearKF` + REAL `RewindKF`; measured inputs.

Files: `c1_gate_relative.py` (+`_results.json`), `c1_variance_and_adversarial.py` (+`_results.json`).
Reproducible: seed 20260613, numpy RNG seeded per-arm/per-seed; numbers reproduce on re-run.

---

## HEADLINE

**Gate-relative observation REMOVES the un-filterable map/registration BIAS that kills the absolute
path (a2/b1: ~0.45–0.55 m in-plane), and pulls the gate-4 in-plane miss to ~0.11–0.14 m RMS —
CLEARING the 0.155 m contact-true margin, but NOT the 0.05 m 1-sigma "variance" bar.** It converts
the gate-4 problem from INFEASIBLE (absolute) to FEASIBLE-WITH-CONDITIONS (gate-relative, on the
*margin* criterion). The bar is met only if the per-fix lateral PnP noise reaches ~0.08 m/axis (vs
the ~0.03 m the absolute path needs — a 2.6× easier target, and the near-band data already shows
~0.10–0.20 m).

| path | binding in-plane error at gate-4 | clears 0.155 m margin? | clears 0.05 m bar? |
|---|---|---|---|
| ABSOLUTE world-fix (a1/b1 deployable) | ~0.55 m (bias-dominated) | NO (3.5×) | NO (11×) |
| GATE-RELATIVE, per-fix lateral 0.265 m (0–12 m avg) | **0.135 m RMS** (1σ 0.095) | **YES** | NO |
| GATE-RELATIVE, per-fix lateral 0.20 m (near-band <9 m) | **0.114 m RMS** (1σ 0.080) | **YES** | NO |
| GATE-RELATIVE, per-fix lateral 0.08 m (achievable goal) | 0.070 m RMS (1σ 0.049) | YES | **YES** |

**This makes the cross-cutting reprioritization actionable: the estimator is the binding VQ2 risk,
and the gate-relative observation is the highest-leverage fix — it is the difference between
"absolute world-fix cannot thread gate-4 at race speed" and "the in-plane miss fits inside the
contact margin."**

---

## (a) Achievable in-plane miss = PnP-relative LATERAL noise (NOT map bias, NOT abs variance)

The realistic fix stream decomposes (EXACT arithmetic, proven, no MC):
- ABSOLUTE world-fix error = `db` (MAP registration bias, per-track constant) − `R_wc @ dt_pnp` (PnP
  solve noise + attitude lever). **The KF tracks `db` as truth ⇒ un-filterable** (a2/b1).
- GATE-RELATIVE obs error = `−R_wc @ dt_pnp` (**PnP ONLY; `db` DROPS OUT entirely**), because the
  target is the offset to the *seen* opening `e_true = −L_true = −R_wc @ t_cam_gate_true`, and the
  seen corners ARE the true opening — there is no `gate_map` in the loop.

**Measured PnP-relative lateral noise** (decomposed from the data: `t_cam_err_m` is the full 3D
PnP-relative error magnitude in the CAMERA frame — uses the SAME map gate pos on both sides, so it
carries NO map error; `range_err_m` is the radial/depth component; lateral = √(t_cam_err² − range_err²)).
Accepted set only (`maha ≤ 16.27`, 4-corner — the fixes the navigator actually uses):

| band (true range) | n | lat PnP RMS (m) | lat min (m) | ABS world-fix RMS (m) |
|---|---|---|---|---|
| ≤ 9 m (binding transit) | 3 | **0.201** | 0.105 | 0.34–0.40 |
| 8–10 m | 7 | 0.312 | 0.105 | — |
| 9–12 m | 7 | 0.415 | — | 0.56 |
| 0–12 m (avg) | 9 | 0.375 | — | 0.52 |

The lateral PnP noise at the binding transit band (≤ 9 m, the last-usable accepted fix is ~8.25 m
per b2) is **~0.10–0.20 m**, vs the ABSOLUTE world-fix RMS of ~0.52 m at the same fixes — the
gate-relative observation strips out the ~0.4 m of map-bias + lever error. At the gate-4 last-fix
(range 8.25 m, the literal binding fix), lateral PnP = **0.105 m** (vs world-fix 0.34 m).

## (b) Wiring: does it REMOVE the map-bias term or just relocate it? (the precise question)

THREE observation models, all consuming the same sightings, run through the REAL LinearKF+RewindKF
on the straight g3→g4 approach (24.4 m, 37 m/s, 0.66 s transit, 14 Hz fixes within 12 m, N_MC=600):

| arm | in-plane RMS | E_bias | D_bias | what it does to the map bias `db` |
|---|---|---|---|---|
| (A) ABSOLUTE world-fix | 0.279 | **+0.176** | +0.062 | tracks `db` as truth — bias SURVIVES |
| (B) "subtract gate MAP pos" | 0.228 | **+0.174** | +0.063 | bias RELOCATED to control (see below) — SURVIVES |
| (C) TRUE gate-relative | **0.139** | **−0.000** | −0.004 | bias REMOVED — E/D bias → 0 |

- **(B) is the anti-pattern the FACTS flagged and it is REAL: subtracting the gate MAP position
  re-injects the bias.** If you form `z = p_fix − gate_map` and let the filter track "offset-from-map,"
  the filter is unbiased *about offset-from-map* — but the controller then centers on `gate_map`,
  which is `gate_true + db`, so the drone flies to `gate_true + db`: the **bias is RELOCATED from the
  estimator to the control target, not removed.** (B)'s estimator E_bias is still +0.174 m.
- **(C) is the correct wiring: a true relative state / IBVS centering term — observe the offset to the
  SEEN opening (`−L` from PnP), never reference `gate_map`.** Then `db` drops out of the arithmetic
  entirely and the estimator E/D bias → 0 (measured −0.000 / −0.004 m). The map error is REMOVED.

**Mechanically (does not require new filter math):** the cleanest implementation is a controller-side
IBVS/centering term that drives the seen-gate image error (or the PnP lateral offset `−L`) to zero —
the LinearKF keeps estimating absolute world position for *planning/feed-forward*, and the relative
centering term owns the *terminal in-plane miss*. A KF-side alternative (augment the state with the
relative offset, observe `−L` directly with cov = the PnP lateral block, NO 0.40 m bias-absorption
floor since there is no bias to absorb) gives the same in-plane numbers above; both are valid. The
controller-side IBVS is preferable because it does not need a per-gate relative state plumbed through
the estimator and degrades gracefully if a fix drops.

## (b-variance) The filtered VARIANCE sweep (the remaining binding term after bias is gone)

With `db` removed, the gate-relative path is purely VARIANCE-limited (good news: averageable). REAL
KF, TRUE gate-relative obs, TIGHT R (no bias-absorption floor), per-axis lateral sigma swept:

| per-fix lateral 1σ | filtered E 1σ | filtered D 1σ | filtered in-plane 1σ | in-plane RMS | <0.05 m? | <0.155 m? |
|---|---|---|---|---|---|---|
| 0.40 m | 0.131 | 0.132 | 0.132 | 0.186 | NO | NO |
| 0.30 m | 0.108 | 0.111 | 0.109 | 0.155 | NO | YES (edge) |
| **0.265 m** (0–12 m avg) | 0.094 | 0.096 | **0.095** | **0.135** | NO | **YES** |
| **0.20 m** (near-band) | 0.080 | 0.081 | **0.080** | **0.114** | NO | **YES** |
| 0.105 m (last-fix) | 0.058 | 0.063 | 0.061 | 0.086 | NO | YES |
| **0.08 m** | 0.050 | 0.048 | **0.049** | 0.070 | **YES** | YES |
| 0.05 m | 0.035 | 0.035 | 0.035 | 0.050 | YES | YES |

The KF averages ~9 fixes over the approach (vel partially observable through the position fixes), so
filtered 1σ ≈ per-fix lateral / ~2.8. **To clear the 0.155 m margin, per-fix lateral ≤ ~0.30 m
(already met). To clear the 0.05 m 1-sigma bar, per-fix lateral ≤ ~0.08 m** — vs the ABSOLUTE path's
~0.03 m requirement (a1/b1). The near-band data (0.10–0.20 m) sits between these: **margin: cleared
today; bar: needs a modest per-fix lateral improvement (better corner localization / sub-pixel refine).**

## (c) Determinism per-track — PRESERVED, trivially

The gate-relative observation references the *seen* gate, NOT the map. There is no per-track absolute
map constant in the loop, so **there is nothing to re-survey and no calibration lap dependency** — the
determinism-per-track property is preserved without exploiting it (unlike the absolute path, whose only
bias lever was a privileged-pose cal lap, a2 §4, which a pure case-C race cannot run). The relative
obs is the same every lap because the gate geometry is fixed; the only per-track variability is the
zero-mean PnP scatter, which is exactly what the KF averages.

## (d) Visibility through the 37 m/s approach — REFUTED (gate stays in frame)

Camera up-tilt 20° (verified from `R_camera_from_body`), VFoV 58.7°, HFoV 90°, 640×360. Drag-hold
cruise pitch **−38.4° (nose-down**, verified: the posture that vectors thrust forward to counter
0.21/s drag at 37 m/s; it puts gate-4 at bearing ~16°, consistent with the measured ~12°). Gate-4
opening corners projected at each approach range:

| range to opening | corners in frame |
|---|---|
| 20, 16, 12, 10, 8, 6, 5, 4 m | **4 (all)** |
| 3, 2 m | 2 (clips top/bottom → P3P) |

**Gate-4 holds all 4 corners in frame down to ~4 m**, and the last *usable accepted* 4-corner fix in
the data is at ~8.25 m — comfortably before the clip. So the visibility objection does NOT refute the
branch: the gate is in frame, 4-corner, through the entire usable fixing window of the approach. (The
≤3 m clip happens AFTER the last useful fix; the terminal cm of the transit coasts on IMU + the last
locked relative fix — exactly when the KF/IBVS estimate is already converged.)

---

## ADVERSARIAL — the strongest case AGAINST, and whether it survives

1. **"Loses the absolute anchor for planning."** PARTLY VALID, fully mitigable. A pure relative
   centering term has no global position for the min-snap/TOPP line or for the g4→g5 hand-off. BUT
   the recommended wiring KEEPS the absolute LinearKF running for planning/feed-forward and uses the
   relative term ONLY for the terminal in-plane miss — you do not give up the anchor, you add a
   higher-accuracy terminal observation on top of it. The absolute anchor's ~0.55 m error is fine for
   *planning* (it sets which gate and the gross approach); it is only fatal for the *terminal thread*,
   which the relative term now owns. **Survives: do not replace absolute, AUGMENT it.**

2. **"PnP depth-flip at close range re-poisons the relative fix."** REFUTED by data. The frontal-PnP
   depth-flip / wrong-gate fixes show up as gross PnP-relative errors (>3 m): there are 21 of them at
   gate-4, and **0/21 survive the chi2 16.27 gate.** The depth-flip is an ALONG-LOS (radial/depth)
   ambiguity → at gate-4 the LOS is ~along-track (N), so a depth-flip perturbs the ALONG-track
   estimate, NOT the in-plane (E,D) miss — the binding axis is the one the flip *least* corrupts.
   CAVEAT: the gross errors are caught here by the *world-fix* maha gate; a relative-obs pipeline needs
   its own relative-innovation / reproj gate. Reproj alone does NOT separate flips (flip p50 0.71 px
   vs clean 1.02 px — flips can fit tighter), so the gate must be a *relative innovation* test, not a
   reproj threshold. **Survives WITH the condition that the relative-innovation gate is implemented.**

3. **"Gate clipping at transit ⇒ 3-corner P3P only ⇒ degenerate."** REFUTED for the binding window.
   Only 4/58 gate-4 sightings are 3-corner, 3 of them at far range (24–40 m), and only 1 ever survived
   the chi2 gate. The clip to <4 corners happens at ≤3 m (part d) — AFTER the last usable 4-corner fix
   (~8 m). So at the moment the relative obs locks the terminal centering, the gate is full 4-corner.
   The ≤3 m P3P regime is the IMU-coast tail, by which point the relative estimate is converged.
   **Survives: the P3P regime is downstream of the binding fix.**

**Net adversarial verdict: the branch survives all three. The one genuine new requirement is a
RELATIVE-innovation outlier gate (item 2) to replace the world-fix maha gate; the depth-flip is
otherwise benign because it lands on the along-track axis, not the in-plane miss.**

---

## VARIANCE-vs-BIAS classification (the FACTS crux)

The gate-relative observation **attacks the BIAS term** (removes `db`, the per-track map/registration
bias — the term a1/a2/b1 showed the absolute path cannot filter). It ALSO improves VARIANCE as a
side-effect (PnP lateral noise ~0.2 m near-band vs absolute world-fix ~0.5 m, because it drops the
attitude-lever and the 0.40 m bias-absorption floor that the absolute fix needs). After the fix, the
RESIDUAL binding term is VARIANCE (the per-fix lateral PnP scatter), which is averageable and reaches
the 0.155 m margin today and the 0.05 m bar at per-fix lateral ≤ 0.08 m. So: **touches BOTH**, with
the *decisive* contribution being BIAS removal (that is what crosses the margin), and a residual
VARIANCE wall standing between "clears margin" and "clears 0.05 m bar."

## Honest caveats / what is NOT measured

- **Per-fix pool is ~5.35 m/s, range ≤ 23.3 m data** (FACTS) — there is NO measured PnP-relative
  lateral data at 37 m/s. Motion blur / rolling-shutter at race speed could WIDEN the ~0.2 m near-band
  lateral noise. So the gate-relative in-plane miss (0.11–0.14 m) is a **best-case lower bound**; the
  margin-clearing claim must be re-verified with a ShadowPC at-speed gate-4 recording (same resolver
  a1/b1 flagged). At per-fix lateral ≥ ~0.30 m the margin is still cleared (edge); above ~0.40 m it is
  not — so there is headroom but it is not unlimited.
- **Residual relative bias (the honest floor a2 §4.2 named):** a gate-relative obs removes the MAP
  registration bias but NOT a chain-correlated PnP/extrinsic systematic (a consistent sub-pixel
  centroid bias or extrinsic mis-cal that biases `dt_pnp` one-signed). I can bound it only by the
  per-fix lateral MEAN MAGNITUDE at the near band (~0.19 m, n=3 — tiny sample, magnitude not sign), so
  the *signed* residual relative bias is UNMEASURED. If a future extrinsic/Bayesian-IoU re-cal
  (SHADOWPC-VISION-CAL) shows the PnP lateral is genuinely zero-mean, the variance sweep above is the
  full story; if it carries a one-signed systematic, that adds to the floor. This is the single
  highest-value follow-up measurement for this branch.
- **The relative-innovation gate (adversarial item 2) is NOT yet built** — the numbers above assume
  the gross depth-flips are rejected (which the world-fix maha gate does in the VQ1 data); a case-C
  relative pipeline needs the equivalent gate proven.
- **CASE CONDITIONALITY:** this whole branch is the **case-C worst case**. If VQ2 streams
  LOCAL_POSITION_NED/ODOMETRY (case A/B) the pose is pristine and the gate-relative fix is unnecessary.
  Conditional on organizer Q①.

---

## Recommendation (for the commander)

1. **ADOPT gate-relative as the gate-4 terminal observation (controller-side IBVS / relative-centering
   term), AUGMENTING — not replacing — the absolute LinearKF.** It is the only measured path that puts
   the gate-4 in-plane miss (0.11–0.14 m RMS) inside the 0.155 m contact margin at 37 m/s. The absolute
   path cannot (0.55 m).
2. **It clears the MARGIN, not the 0.05 m 1-sigma bar.** If the program insists on the 0.05 m bar as a
   hard gate, the additional requirement is per-fix lateral PnP ≤ ~0.08 m (2.6× easier than the
   absolute path's 0.03 m) — a corner-localization / sub-pixel-refine improvement, not an architecture
   change. Recommend treating 0.155 m margin (with the variance headroom of the relative obs) as the
   operative gate, with 0.05 m as a stretch.
3. **Build a RELATIVE-innovation outlier gate** (not a reproj threshold — reproj does not separate
   depth-flips). This is the one genuinely new pipeline element.
4. **The decisive open measurement is a ShadowPC at-speed (37 m/s) gate-4 recording** to (a) confirm
   the near-band lateral PnP noise does not blow up under motion blur, and (b) test for a one-signed
   residual relative bias (extrinsic systematic). Until then 0.11–0.14 m is a best-case lower bound.

---

## MEMORY-DELTA:
- c1 GATE-RELATIVE branch DONE: gate-relative obs REMOVES the per-track MAP bias (proven exact: rel
  obs error = −R_wc@dt_pnp, `db` drops out) that kills the absolute path; pulls gate-4 in-plane miss to
  **0.11–0.14 m RMS (1σ 0.08–0.095) — CLEARS the 0.155 m contact margin, NOT the 0.05 m bar.** Converts
  gate-4 from INFEASIBLE (absolute ~0.55 m) to FEASIBLE-WITH-CONDITIONS.
- Measured PnP-relative LATERAL noise (from t_cam_err decomposed vs range_err, accepted maha≤16.27 set):
  near-band (≤9 m) ~0.10–0.20 m, last-fix (8.25 m) 0.105 m — vs absolute world-fix ~0.52 m same fixes.
- WIRING: "subtract gate MAP pos" is an ANTI-PATTERN (bias RELOCATED to control target, E_bias still
  +0.17 m). CORRECT = controller-side IBVS/relative-centering on the SEEN opening (`−L` from PnP),
  AUGMENTING the absolute KF (keep absolute for planning). Touches BOTH bias (decisive) + variance.
- 0.05 m bar needs per-fix lateral ≤ ~0.08 m (2.6× easier than abs 0.03 m); margin needs ≤ ~0.30 m
  (met today). Determinism-per-track PRESERVED trivially (no map reference, nothing to re-survey).
- VISIBILITY refuted: gate-4 holds 4 corners to ~4 m (drag-hold pitch −38.4° nose-down), last usable
  fix ~8 m — clip to P3P (≤3 m) is downstream of the binding fix. ADVERSARIAL survives: (1) keep abs
  anchor for planning; (2) 21/21 depth-flips caught by chi2, flip lands on along-track N not in-plane;
  (3) P3P only 1 accepted, all far. NEW REQ: a RELATIVE-innovation gate (reproj does NOT separate flips).
- CAVEATS: per-fix pool ~5 m/s (37 m/s blur UNMEASURED ⇒ 0.11–0.14 m is best-case LB); residual
  one-signed PnP/extrinsic bias UNMEASURED (bound ≤0.19 m mag, n=3). Decisive resolver = ShadowPC
  at-speed gate-4 recording. Conditional on organizer Q① (case C).
