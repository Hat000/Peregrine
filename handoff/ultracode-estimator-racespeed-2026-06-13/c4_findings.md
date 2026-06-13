# c4 — Lowering the 0.40 m FIX_COV_FLOOR_STD: real lever or trap?

Agent **c4** (estimator/perception, opus-4.8). Offline sim only. Composes the REAL
`racer.localization.gate_pose_to_world_position` (floor overridden), the REAL `LinearKF` + `RewindKF`,
the REAL navigator chi2 gate (16.27), and the a1 sim geometry/residual pools. Builds on a1/a2/b1/b2.

Files: `c4_floor_lower.py` (the floor sweep: variance + bias re-admission + leak MC) +
`c4_floor_lower_results.json`; `c4_chi2_overrej.py` (analytic over-rejection / leak companion) +
`c4_chi2_overrej_results.json`. Reproducible: numpy RNG seeded per-seed/per-floor (base 20260613),
600 seeds/cell; the headline rows reproduce bit-for-bit on re-run.

---

## VERDICT: **TRAP, not a lever.** Lowering the floor is NET NEGATIVE at gate-4.

Lowering `FIX_COV_FLOOR_STD` below ~0.20 m buys **no usable variance** (filtered in-plane 1σ stays
~0.31 m, flat across the whole 0.40→0.00 sweep — best is 0.307 m at floor 0.20, a **4.6 %** improvement
over the 0.40 baseline 0.322 m, nowhere near the 0.05 m bar), while it **re-grows the per-track bias
2.5×** (filtered in-plane bias 0.035 → 0.086 m), **re-opens the chi2 over-rejection wound the floor was
added to cure** (good-fix rejection 4–7 % → 21–36 %, i.e. back to the documented 13–35 % failure), and
**doubles the damage of any leak that passes** (state-pull 0.164 → 0.332 m in-plane). The floor is doing
its job; touching it makes gate-4 worse, not better. **Do not lower the floor as a speed/accuracy lever.**

---

## Why the variance lever is bounded (the decisive structural fact)

The floor enters as `+ σ_floor² I` in quadrature. But the per-fix world-fix covariance also carries the
**PnP-default block** and the **attitude lever**. Critically, when `gate_pose.covariance is None`
(the shipped case-C path — no analytic PnP cov available), the model uses `default_position_std = 0.3 m`
isotropic **and does NOT apply `PNP_FIX_COV_INFLATION`** (the inflation only multiplies an analytic cov;
see `localization.py` lines 88–94). So the per-fix 1σ **floors at ~0.30 m even with σ_floor = 0**:

| floor | per-fix σ near gate (r=2 m, NED) | per-fix σ at r=8 m | per-fix σ at r=16 m |
|---|---|---|---|
| 0.40 (shipped) | [0.500, 0.502, 0.502] | [0.501, 0.536, 0.537] | [0.505, 0.631, 0.635] |
| 0.20 | [0.361, 0.364, 0.364] | [0.362, 0.409, 0.410] | [0.367, 0.527, 0.532] |
| 0.00 | [0.300, 0.304, 0.304] | [0.302, 0.356, 0.358] | [0.308, 0.488, 0.493] |

**The floor can buy at most a 0.50 → 0.30 m per-fix reduction near the gate.** a1's `floor_probe` swept
the *whole* per-fix isotropic σ down to 0.03 m and found you'd need ~0.03 m per-fix to clear 0.05 m
filtered. The **floor lever cannot reach there** — it bottoms out at the 0.30 m PnP-default block, which
a1's own probe maps to ~0.226 m filtered (4.5× over the bar). So even the *idealised* floor-to-zero is
~4.5× over. This is the ceiling on the lever before any risk is counted.

---

## (A) Variance benefit of lowering the floor — essentially zero, then it reverses

VARIANCE arm (global de-bias applied, per-gate residual removed — the *pure* variance question), L=6 ms,
47 % accept (~14 Hz), 600 seeds:

| floor | E std | D std | **in-plane std** | over-rej | NEES |
|---|---|---|---|---|---|
| 0.40 (shipped) | 0.253 | 0.199 | **0.322** | 4.9 % | 3.55 |
| 0.30 | 0.247 | 0.191 | **0.312** | 6.8 % | 4.34 |
| **0.20** | 0.266 | 0.154 | **0.307** (best) | 9.3 % | 5.66 |
| 0.15 | 0.270 | 0.154 | 0.311 | 12.9 % | 6.12 |
| 0.10 | 0.291 | 0.158 | 0.331 | 14.9 % | 7.42 |
| 0.05 | 0.272 | 0.161 | 0.316 | 15.6 % | 7.11 |
| 0.00 | 0.281 | 0.170 | 0.329 | 15.9 % | 7.46 |

The filtered in-plane variance is **flat at ~0.31 m** across the entire sweep. Lowering the floor lets
each fix pull harder (higher Kalman gain), but it injects more of the ~0.30 m irreducible per-fix noise
per fix, so the converged variance barely moves and then *worsens* below 0.20 m. **Best-case variance
benefit = 0.322 → 0.307 m (15 mm, 4.6 %).** That is sub-resolution against the 0.155 m gate-4 margin and
6× over the 0.05 m bar. The floor is **not** the thing holding variance up — the per-fix measurement
floor and case-C velocity-unobservability (only ~2–4 effective fixes averaged) are, exactly as a1 found.

Note D std *does* improve (0.199 → 0.154) as the floor drops — but E std *worsens* (0.253 → 0.281),
because the residual scatter the floor was masking starts pulling the estimate. The in-plane combination
nets out flat.

---

## (B) THE TRAP, part 1 — bias re-admission (the floor is bias-absorption by design)

DEPLOYABLE arm (global de-bias applied, **per-gate residual LEFT IN** — modeled as a per-track-constant
offset ~N(0, [0.21, 0.24, 0.03] m) NED, the case-C residual sigma per FACTS):

| floor | in-plane std | **in-plane bias** | **in-plane RMS** | over-rej | NEES |
|---|---|---|---|---|---|
| 0.40 (shipped) | 0.397 | **0.035** | **0.398** | 5.2 % | 5.42 |
| 0.30 | 0.387 | 0.064 | 0.392 | 7.0 % | 6.36 |
| 0.20 | 0.390 | 0.068 | 0.395 | 10.0 % | 8.74 |
| 0.15 | 0.422 | 0.072 | 0.428 | 12.5 % | 9.80 |
| 0.10 | 0.398 | 0.078 | 0.405 | 14.9 % | 9.95 |
| 0.05 | 0.395 | 0.086 | 0.404 | 16.1 % | 10.54 |
| 0.00 | 0.407 | 0.079 | 0.414 | 16.7 % | 10.21 |

**The filtered in-plane BIAS grows 2.5× (0.035 → 0.086 m) as the floor drops 0.40 → 0.05.** This is the
mechanism the floor was named for: the more the KF trusts a fix, the harder it tracks that fix's
*constant per-track residual* as if it were real motion. A higher floor distrusts the fix and leans on
the IMU prior, which damps how fast the bias is absorbed; lowering it lets the bias in. The
**deployable in-plane RMS is flat ~0.40 m and never improves** — the tiny variance gain is exactly
cancelled by the bias regrowth. And NEES climbs 5.4 → 10.5: the filter becomes **more overconfident**
(P shrinks with the floor, but the bias P cannot see grows) — the chi2 monitor's live tell that an
unmodeled registration bias is being absorbed. **Lowering the floor converts a damped bias into a
tracked bias for zero variance gain.**

---

## (C) THE TRAP, part 2 — chi2 over-rejection regrowth + leak amplification

The navigator gate is `d2 = ν^T (P + R)^-1 ν > 16.27 → reject`. Lowering the floor shrinks R → shrinks
S → raises d2 for the same innovation.

**(C-i) Over-rejection of GOOD biased fixes regrows to the pre-floor failure level.** Analytic MC
(`c4_chi2_overrej.py`), a good fix carrying the per-gate residual + near-band noise, range 4 m:

| floor | P(reject) prior σ=0.24 | prior σ=0.15 | prior σ=0.10 |
|---|---|---|---|
| 0.40 (shipped) | 4.2 % | 5.9 % | 6.7 % |
| 0.20 | 13.8 % | 20.1 % | 23.1 % |
| 0.10 | 19.2 % | 28.1 % | 32.2 % |
| 0.05 | 20.6 % | 30.3 % | 35.0 % |
| 0.00 | 21.2 % | 31.2 % | 36.0 % |

This **reproduces the exact failure vision-pkg2 added the floor to fix** (13 % → 0–2 % over-rejection at
<1–3 m for gate-0 close fixes; 35 % at gate-0 close range). Below floor 0.20 m the good-fix over-
rejection climbs back to 14–36 %. The over-rejection is *worse* the tighter the prior — and the prior
*is* tighter at a converged gate-4 approach — so the regression bites hardest precisely where you'd want
the close fixes. The MC sweep's measured `over_rej_frac` (4.9 % → 16–17 %) corroborates the analytic
curve. **Fewer accepted close fixes → slower convergence → the variance lever you were chasing gets
*worse*, not better.** This is a self-defeating loop.

**(C-ii) Leaks that pass do ~2× the damage.** A bounded wrong-gate leak (in-plane 0.9 m offset),
range 4 m, prior σ=0.24:

| floor | leak d2 | passes gate? | state-pull (in-plane) |
|---|---|---|---|
| 0.40 | 2.58 | yes | 0.164 m |
| 0.20 | 4.16 | yes | 0.265 m |
| 0.10 | 4.91 | yes | 0.312 m |
| 0.00 | 5.22 | yes | **0.332 m** |

Lowering the floor *does* raise the leak's d2 (so a slightly grosser leak gets rejected — a small
benefit), but a sub-gross leak that **still passes** is fused with a higher Kalman gain, so it yanks the
state **2× further** (0.164 → 0.332 m in-plane). Against a 0.155 m gate-4 margin, a single passing leak
at the low floor is a margin-blowing event. The MC `leak_pass_frac` also stays high (29–86 %) because
the bounded-leak class is, by construction, inside the gate. **Net leak effect of lowering the floor:
marginally fewer gross leaks pass, but each one does double the damage — strictly worse for the binding
margin.**

---

## Synthesis: the three terms move the WRONG way together

| floor | variance benefit | bias cost | over-rej cost | deployable in-plane RMS |
|---|---|---|---|---|
| 0.40 (shipped) | baseline 0.322 | 0.035 | 5 % | **0.398** |
| 0.20 | −0.015 (best) | +0.033 | 10 % | 0.395 |
| 0.05 | −0.006 | +0.051 | 16 % | 0.404 |
| 0.00 | +0.007 (worse) | +0.044 | 17 % | 0.414 |

The variance benefit (≤15 mm) is **smaller than** the bias regrowth (+51 mm at floor 0.05) and is
swamped by the over-rejection cost (which *slows* convergence, undoing even the 15 mm). The deployable
in-plane RMS is flat-to-rising. **There is no floor value that improves the deployable gate-4 error.**

### What about the premise "IF VISION-CAL removes the global bias, the floor can come down"?
b1/b2 confirm the global de-bias is real and not circular (LOGO hold-out). But the floor does NOT exist
to absorb the *global* bias — it exists to absorb the **per-gate / per-track residual + close-range
depth** systematics, which b2 showed *survive* global de-bias (range-collapsing ~0.11–0.34 m in-plane).
Removing the global bias therefore **does not justify lowering the floor**: the residual it absorbs is
still there. The floor could only safely come down if the **per-gate** residual were also removed — that
needs a privileged-pose (case A/B) per-gate cal lap (a2 §4: pulls residual to ~0.07 m, σ-limited), and
**a pure case-C cal lap cannot reach it** (chain-circularity, a2/b2). And even with a perfect per-gate
de-bias the floor lever still only buys 0.50→0.30 per-fix → ~0.23 m filtered (4.5× over the bar). **The
premise's "if" is not satisfiable in pure case C, and even if it were, the lever is too small to matter.**

---

## Assumptions — MEASURED vs ASSUMED (flagged)

- **Per-gate residual model (ASSUMED magnitude, from FACTS):** drawn as a per-track-constant
  ~N(0, [0.21, 0.24, 0.03] m) NED — the case-C residual sigma the cross-cutting note carries. b2 corrected
  a2's "0.52 m constant" to a range-collapsing 0.11–0.34 m in-plane; using the FACTS residual sigma is the
  conservative-middle choice and matches the memory headline. Sensitivity: the *direction* of every
  finding (variance flat, bias regrows, over-rej regrows) is independent of this magnitude; only the
  absolute bias numbers scale with it.
- **Per-fix cov (REAL model):** `gate_pose_to_world_position(covariance=None, floor=<swept>)` — the
  shipped case-C path. The 0.30 m PnP-default *not* being inflated (inflation is analytic-cov-only) is a
  CODE FACT verified by reading `localization.py`, not an assumption — it is what bounds the lever.
- **Trajectory / attitude / IMU / acceptance / latency:** identical to a1 (const-v 37 m/s g3→g4,
  drag-hold pitch 38.4°, accel_noise 0.3, attitude 1.4°, 90 Hz IMU, 47 % accept, L=6 ms, rewind
  horizon 0.5 s). a1/b1 already showed these are not the binding terms.
- **Leak model (ASSUMED, bounded):** a chi2-gate-bounded wrong-gate offset (~gate-edge class at the
  0.40 floor), NOT the 139 m gross outliers the gate trivially rejects. This isolates the floor's effect
  on the borderline leak class — the only class whose gate-pass changes with the floor.
- **Speed caveat (inherited, UNRESOLVED):** the per-fix pool is ~5.35 m/s data; 37 m/s motion blur is
  unmeasured and would WIDEN per-fix σ, making the floor lever even less useful (a wider per-fix σ means
  the 0.30 m PnP floor is a smaller fraction of the total, so removing the 0.40 floor buys even less).
  The verdict is robust to this — it only hardens.

---

## Implications / recommendation (for the commander)

1. **Do NOT lower `FIX_COV_FLOOR_STD` as an accuracy/speed lever.** It is a TRAP: ≤15 mm variance
   benefit, +51 mm bias regrowth, 14–36 % good-fix over-rejection regression (the exact vision-pkg2
   failure), and 2× leak amplification. Deployable gate-4 in-plane RMS is flat ~0.40 m at every floor.
2. **The floor is correctly sized.** 0.40 m (shipped, jointly MLE-fit to 0.407 in vision-pkg2) sits at
   the over-rejection knee (4–7 %) and the NEES knee (~5). Anything below 0.20 m re-opens both wounds.
   If anything, the data says the floor is doing exactly what it should — leave it.
3. **The premise is unsatisfiable in case C.** Lowering the floor would only be justified if the
   *per-gate* residual were removed (not just the global bias), which needs a privileged-pose cal lap
   (case A/B) — and even then the lever bottoms at the 0.30 m PnP-default block (~0.23 m filtered, 4.5×
   over the bar). The floor is not the binding term; the per-fix measurement floor + case-C
   vel-unobservability are.
4. **This reinforces a1/a2/b1/b2 and the gate-relative recommendation.** The absolute case-C path has no
   clearing knob — not acceptance, not latency, not the cov floor. The architectural fix is
   **gate-relative observation** (close on the seen gate-4 corners), which sidesteps both the per-fix
   absolute variance and the per-track residual bias the floor is absorbing.
5. **CONDITIONAL on organizer Q①** — if VQ2 streams LOCAL_POSITION_NED/ODOMETRY (case A/B), pose is
   pristine, the floor question is moot. This is the case-C worst case.

---

## MEMORY-DELTA:
- c4 floor-lowering branch DONE: **lowering FIX_COV_FLOOR_STD is a TRAP, not a lever** (NET NEGATIVE at
  gate-4). Best variance benefit 0.322→0.307 m in-plane (15 mm, 4.6 %) at floor 0.20; below that it
  reverses. Deployable in-plane RMS FLAT ~0.40 m at every floor 0.40→0.00.
- WHY bounded: case-C PnP-default (covariance=None) = 0.30 m isotropic and is NOT inflated by
  PNP_FIX_COV_INFLATION (code fact, localization.py L88–94) → per-fix σ floors at ~0.30 m even at
  floor=0 → ~0.23 m filtered (4.5× over 0.05 m bar). Floor can only buy 0.50→0.30 per-fix.
- TRAP-1 (bias): filtered in-plane bias regrows 0.035→0.086 m (2.5×) as floor 0.40→0.05; NEES climbs
  5.4→10.5 (more overconfident). The floor IS bias-absorption; lowering it tracks the per-track residual.
- TRAP-2 (chi2): good-fix over-rejection regrows 4–7 %→21–36 % (reproduces the exact 13–35 % vision-pkg2
  failure the floor cured); a bounded leak that passes does 2× damage (state-pull 0.164→0.332 m in-plane).
- PREMISE FAILS: global de-bias does NOT justify lowering the floor — the floor absorbs the PER-GATE
  residual (b2: survives global de-bias), removable only by a privileged-pose case-A/B cal lap. Leave the
  floor at 0.40. Reinforces gate-relative as the only real fix. CONDITIONAL on organizer Q①.
