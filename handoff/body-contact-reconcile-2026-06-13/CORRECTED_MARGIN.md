# CORRECTED MARGIN — body-contact-radius reconciliation & gate-4 verdict re-read (2026-06-13)

Fengyou — this is the **reconcile leg**. It fuses the three sub-legs (provenance, tilt-projected
geometry, empirical) into a single best-estimate gate-4 effective contact radius, recomputes the error
budget, and re-reads the "gate-4 margin does NOT close offline" headline against the corrected budget.
No `src/` or `memory/` edits. Flags on every number: **[M]** measured/logged-verbatim ·
**[X]** derived-here · **[A]** assumed.

Sub-legs (all read & cross-checked this session):
`provenance.md` · `geom_radius.py`/`geom_radius_result.json`/`geom_radius_findings.md` · `empirical_radius.md`.
Margin-closure source data independently read & re-derived from:
`handoff/ultracode-gate-relative-pipeline-design-2026-06-13/{d3_margin_closure.py, d3_margin_closure_results.json, d3_margin_findings.md, v3_margin_VERIFY.md}`.

---

## TL;DR (the corrected verdict)

The prompt's hypothesis — "0.38 is too big, the budget doubles, the cold case closes" — is **directionally
right about 0.38 but wrong about the conclusion**, for two compounding reasons:

1. **0.38 is NOT a geometric body radius** (geometry tops out at **0.213 m** even tilted). BUT it is also
   **not** the right number to *replace with geometry*: the posture-matched live contacts (gate-3 steep
   crashes at L-inf 0.37–0.49 m) prove the *effective* contact envelope at the gate-4 posture exceeds
   rigid-body geometry by an unmodeled rotor/prop/aero halo. **Reconciled best estimate ≈ 0.30 m
   (band 0.26–0.33), worst-case tail 0.38.** [X]
2. **The radius correction DOES roughly double the clearance** (budget 0.155 → ~0.235 m at r=0.30) —
   exactly as the prompt predicted — **but it does not move the binding case.** The headline failure was
   never the nominal/warm pass; it is the **cold case-C estimator p90 under the MEASURED 1.4° attitude
   bias (p90 = 0.338 m)**, which **fails at every radius at or above the geometric floor** (it needs
   budget > 0.338 → r < 0.197, i.e. *below even the flat half-diagonal 0.198 m* — physically impossible).

**VERDICT: CLOSES_EXCEPT_WORST_BIAS.** With the corrected radius the warm and the cold@zero-bias p90s
now clear, but the cold@1.4°-attitude-bias p90 (0.338, the realistic case-C tail) **stays open**. The
"cannot settle offline" conclusion of d3/v3 **survives the radius correction**, now for a *sharper*
reason: the open quantity is the **effective accel/attitude bias**, not the contact radius. [X]

---

## 1. The reconciliation arithmetic — the two framings are ONE inequality (this is the load-bearing insight)

The three sub-legs and the d3 margin-closure sim describe the **same** geometric gate with two different
variables held fixed. Reconciling them requires seeing they are the *same inequality rearranged*: [X]

- **`HALF_OPEN` = 0.75 m** [M] — spec inner opening 1500 mm / 2, confirmed against the PDF §3.7 and
  `offline_rollout.py:65`. (Frame depth 0.30 m and outer 1.36 m do NOT touch the in-plane pass band —
  provenance.md §1; confirmed.)
- **`r`** = effective contact radius (the only body-extent knob; `contact_true_eval` models the body as
  an L-inf halo of half-width `r` that shrinks the pass band to `half_in = 0.75 − r`). [M]
- **Nominal in-plane crossing offset `linf₀` = 0.215 m** [X] — the *radius-invariant* geometry of where
  the gate-4 simstart trajectory actually crosses the plane (the recorded `linf` is radius-independent;
  only the band shifts with `r`). Back-out: `linf₀ = (0.75 − 0.38) − 0.155`.
- **Estimator in-plane error `m`** — the perpendicular distance between the *estimated* and *true*
  position at the gate-4 plane (d3 `inplane_miss`, lines 365–374). The p90s {warm 0.203, cold 0.234,
  cold@1.4° 0.338} are p90s **of `m`**, compared directly to a budget. [M, from d3 JSON]

The contact-true non-collision condition the drone must satisfy is `linf₀ + m < half_in = 0.75 − r`,
i.e.

> **budget(r) = (0.75 − r) − linf₀ = (0.75 − r) − 0.215**, and the gate clears iff **m < budget(r)**.

This is **exactly** d3's `MARGIN_G4` test (`clears_margin_p90 = p90(m) < MARGIN_G4`, lines 415–417):
at r = 0.38, budget = (0.75 − 0.38) − 0.215 = **0.155 m** — reproducing d3's hardcoded `MARGIN_G4 = 0.155`
**bit-for-bit** (verified by script). So d3's 0.155 m budget *is* the r=0.38 instance of this identity;
the prompt's "0.155 derives from a 0.38 radius" is **confirmed correct**, and correcting `r` shifts the
budget the estimator-error p90 is tested against. [X]

**Why the prompt's "budget doubles" intuition is arithmetically right but operationally inert:** dropping
r from 0.38 to ~0.20 raises budget from 0.155 to ~0.335 (≈ 2.2×) — a real doubling. But the *binding*
p90 (cold@1.4° = 0.338) sits *above even that doubled budget*. The lever moved; the rock didn't.

---

## 2. Reconciled best-estimate effective contact radius at gate-4

Weighting per the task: tilt-projected geometry = the **principled** number; empirical = **cross-check
respecting posture**; 0.38 = **incumbent to beat-or-justify**.

| Input | r (m) | Flag | Weight for gate-4 | What it is |
|---|---|---|---|---|
| Flat half-diagonal | 0.198 | [X] | floor only | level chassis, strict lower bound |
| **Tilt-projected L-inf (gate-4 posture)** | **0.213** | [X] HIGH-conf | **principled floor** | 280×280×160 box @ pitch−38°/roll55°, L-inf on gate plane, yaw-swept; cross-checked 2 ways. Tilt adds only +0.015 m over flat. |
| Empirical near-level (corner-pass, gate 0) | 0.18–0.20 | [M/X] | **DOWN-WEIGHT** | clean@0.50 / contact@0.64 → r_eff bracket (0.11,0.25]. **Wrong posture** (near-level, ~3 m/s) — NOT gate-4. |
| **Empirical posture-matched (gate-3 steep crashes)** | **≥0.26 … ≥0.38** | [M/X] | **HIGHEST** | contacts at L-inf 0.37–0.49 on the trained tilt≈55°/12–18 m/s climb — the SAME family as gate-4. Lower bounds (body reached frame *by* that L-inf). |
| Incumbent worst-case (DR top) | 0.38 | [M-in-code] | conservative tail | steep-crash-fit DR upper bound `peregrine_racing.py:160–164`; **not** geometric. |

**Principled reconciliation [X]:** geometry sets a hard floor at **0.213 m**, but the *only
posture-matched* empirical evidence (gate-3 steep crashes) logs contacts implying **r_eff ≥ 0.26 m** at
the gate-4 posture — i.e. the real effective envelope at this posture is **provably larger than rigid-body
geometry** by an unmodeled rotor-wash / prop-disk / blade-strike halo (the spec documents *no* prop
extent — provenance.md §3, so this halo is real but undocumented). Therefore:

- **Do NOT adopt the geometric 0.213** as the contact radius — it ignores real posture-matched contacts
  above it (would be optimistic).
- **Do NOT adopt the near-level 0.18–0.20** — wrong posture (the prompt's empirical anchor is gate-0,
  not gate-4; this is the prompt's one genuine misread, corrected by `empirical_radius.md` §2).
- **Reconciled best estimate = 0.30 m central, band 0.26–0.33 m**, worst-case tail **0.38 m**.
  [X — central is the loosest posture-matched lower bound (0.26) up through DR nominal (0.33); 0.30 ≈
  the codebase's own "rotor halo ~0.3 m" prose and the band midpoint.]

**Confidence:** geometry HIGH; empirical posture-matched MED (4 events, lower bounds only); reconciled
best estimate **MED**. The residual is irreducibly posture-dependent and only a live gate-4 offset sweep
at the trained tilt/speed would pin it (we have none — `empirical_radius.md` §5).

**Should `contact_true_eval --body-radius` 0.38 be corrected? → NO (keep 0.38 as the worst-case knob);
report central 0.30.** 0.38 is *defensible as conservative DR* (it is a posture-matched steep-crash lower
bound, and contacts only prove `r_eff ≥`, so the true value could even exceed it). What should change is
**not the constant but the reporting discipline**: stop quoting a single margin tied to one radius; report
margin **as a function of r ∈ {0.21(geom-floor), 0.26, 0.30, 0.33, 0.38}**. The honest central planning
radius is **0.30 m** (budget 0.235), with 0.38 retained as the worst-case stress knob. [X]

---

## 3. Corrected error budget

**budget(r) = HALF_OPEN(0.75) − r − linf₀(0.215)** [X]

| r (source) | corrected budget (m) | vs incumbent 0.155 |
|---|---|---|
| 0.213 (geom floor) | **0.322** | +0.167 (≈2.1×) |
| 0.26 (best-est lo) | 0.275 | +0.120 |
| **0.30 (best-est central)** | **0.235** | **+0.080 (≈1.5×)** |
| 0.33 (best-est hi / DR nominal) | 0.205 | +0.050 |
| 0.38 (incumbent worst-case) | 0.155 | — |

**Corrected best-estimate budget = 0.235 m** (at the reconciled central r = 0.30). It is ~1.5× the
incumbent 0.155 m — a real loosening, just short of the prompt's "doubles" (which holds only at the
optimistic, posture-wrong r≈0.20). [X]

---

## 4. Re-read of the margin verdict against the corrected budget

The gate-4 in-plane **estimator-error p90s** (MEASURED, d3 seed 20260613; v3-reproduced on a different
seed) re-tested against budget(r) = `m < (0.75 − r) − 0.215`: [M for the p90s, X for PASS/FAIL]

| p90 case (what it is) | p90 (m) | r=0.213 (b=0.322) | r=0.26 (b=0.275) | **r=0.30 (b=0.235)** | r=0.33 (b=0.205) | r=0.38 (b=0.155) |
|---|---|---|---|---|---|---|
| **warm** (vel≈truth — NOT case C; optimistic ceiling) | 0.203 | PASS | PASS | **PASS** (+0.032) | PASS (+0.002) | FAIL |
| **cold @ bias 0** (case-C, vel IMU-only, zero accel bias) | 0.234 | PASS | PASS | **PASS** (+0.001) | FAIL | FAIL |
| **cold @ 1.4° attitude bias** (case-C, MEASURED 1-σ att err) | 0.338 | FAIL | FAIL | **FAIL** (−0.103) | FAIL | FAIL |

Robustness of the FAIL: the cold@1.4° case needs budget > 0.338, i.e. **r < 0.197 m — below the flat
chassis half-diagonal (0.198 m)**. No physically admissible radius (≥ the geometric floor 0.213) clears
it. v3's *strengthened* in-plane-forced phantom (cold@1.4° p90 = **0.348**) fails even harder. So the
worst-bias FAIL is **radius-invariant** within the admissible range — it cannot be rescued by any honest
radius correction. [X]

### Which cases now CLEAR the corrected (best-estimate r=0.30, budget 0.235) budget
- **warm 0.203 → CLEARS** (+0.032 m). *Caveat:* warm is the c1-style ceiling (velocity continuously
  re-seeded to truth); **unreachable in case C** — vision is position-only, velocity dead-reckoned.
  It is shown for contrast, not as the planning case.
- **cold @ bias 0 = 0.234 → CLEARS, but only just** (+0.001 m, knife-edge). This is the *idealised*
  case-C tail (perfect attitude, zero accel bias). Newly-cleared by the radius correction — this **is**
  the prompt's hypothesis vindicated, but only for the zero-bias idealisation.
- **cold @ 1.4° = 0.338 → STILL FAILS** (−0.103 m). This is the **realistic case-C tail** (the measured
  1-σ attitude error maps to a 0.240 m/s² phantom accel — v3 confirms the g·sinθ mapping is exact).
  **This is the binding case and it does not close at any admissible radius.**

---

## 5. VERDICT

### CLOSES_EXCEPT_WORST_BIAS

With the reconciled best-estimate radius (0.30 m, budget 0.235 m), the gate-4 margin **closes for the
warm and cold@zero-bias p90s** — newly so, thanks to the radius correction (this is the kernel of truth
in the prompt's hypothesis). **It does NOT close for the cold@1.4°-attitude-bias p90 (0.338 m)**, which
is the *realistic* case-C tail under the MEASURED 1-σ attitude error. The radius correction loosens the
budget by ~1.5× but **does not overturn the headline**, because the binding failure was always the
worst-bias tail, not the nominal pass — and that tail fails at **every admissible radius**.

**Does this soften the "CANNOT-SETTLE-OFFLINE" headline?** Partially and honestly:
- **Softened:** the *zero-bias idealised* cold case-C now clears (it failed at r=0.38: 0.234 > 0.155).
  The verdict is no longer "cold fails even at zero bias for any radius" — at the corrected radius,
  zero-bias cold is a (thin) PASS. The prompt was right that 0.38 was doing real work in the failure.
- **NOT overturned:** the operative case-C tail carries the measured 1.4° attitude bias (→ 0.240 m/s²
  phantom accel → p90 0.338), and that **fails at every radius ≥ the geometric floor**. The thing that
  is genuinely unsettled offline is the **true effective accel/attitude bias entering gate-4** — exactly
  d3/v3's residual — and the radius reconciliation **does not touch it**. If anything it *sharpens* the
  headline: the open question is provably the bias, not the contact radius.

### Should `contact_true_eval`'s body-radius be corrected, and to what?
- **Keep `--body-radius 0.38` as the worst-case stress knob** (it is a posture-matched steep-crash
  lower bound; defensible as conservative DR; contacts only prove `r_eff ≥`).
- **Adopt `0.30 m` as the best-estimate / central reporting radius** (was effectively 0.33 nominal in
  code; 0.30 ≈ the band midpoint and the codebase's own "rotor halo ~0.3 m"). The `BODY_RADIUS_NOM = 0.33`
  is fine to leave; the *correction is in interpretation*, not the constant.
- **The real fix is reporting discipline:** quote the gate-4 margin **as a function of r** ∈
  {0.21, 0.26, 0.30, 0.33, 0.38}, not a single number, and **always against p90/p99** (a margin is a
  worst-case gate). The headline should read: *"closes for r ≲ 0.33 at zero accel bias; the worst-case
  cold tail under the measured 1.4° attitude bias fails at every admissible radius — closure is gated by
  the effective accel/attitude bias, which only the live ≥5-lap at-speed gate-4 recording can pin."*

---

## 6. Caveats / what could still move this (none rescues the worst-bias case)
- **`linf₀ = 0.215 m` is reused from the contact-true simstart back-out** [X], not re-run here; all three
  sub-legs and provenance.md §5 carry the same value, and it is *radius-invariant* by construction, so it
  is the right pivot. A different nominal crossing offset would shift every budget equally but not change
  which p90s clear relative to each other.
- **The p90s are estimator-error p90s** [M, d3 JSON] — `m`, not the absolute crossing position. The
  reconciliation correctly tests `linf₀ + m` against `0.75 − r` via the budget identity. Confirmed the
  d3 `inplane_miss` is the perpendicular estimated-vs-true error at the g4 plane (lines 365–374).
- **The worst-bias FAIL is robust to *strengthening***: v3's physically-faithful in-plane-forced phantom
  gives cold@1.4° p90 = 0.348 (worse), and v3 reproduced the zero-bias straddle on an independent seed.
  Direction is not a seed/RNG artifact.
- **Posture-matched empirical r_eff are lower bounds** [X]: the true gate-4 effective radius could exceed
  0.38, which would only *tighten* the budget — never loosen it. So 0.30 central is, if anything, mildly
  optimistic; the worst-bias verdict is safe against that error direction.
- **Spec-vs-code discrepancies** (provenance.md §3, re-confirmed): outer 1.36 vs spec 1.350 (+0.010 m);
  frame-depth 0.30 vs spec 0.260 (+0.040 m); inner 0.75 == spec 0.750 (match). None touch the in-plane
  pass band or the headline.

---

### Files (absolute)
- THIS file: `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\body-contact-reconcile-2026-06-13\CORRECTED_MARGIN.md`
- Sub-legs: `...\body-contact-reconcile-2026-06-13\{provenance.md, geom_radius.py, geom_radius_result.json, geom_radius_findings.md, empirical_radius.md}`
- Margin-closure source (independently read): `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\ultracode-gate-relative-pipeline-design-2026-06-13\{d3_margin_closure.py (MARGIN_G4=0.155 line 59; inplane_miss 365–374; clear-flags 415–418), d3_margin_closure_results.json, d3_margin_findings.md, v3_margin_VERIFY.md}`
- Geometry consts: `C:\Users\Fengy\Downloads\Projects\Anduril\rl\contact_true_eval.py:51–53`; `rl\offline_rollout.py:65–66`; DR-band provenance `rl\peregrine_racing.py:153–164`
- Spec: `C:\Users\Fengy\Downloads\Projects\Anduril\260508_Technical_Spec_0002.pdf` §3.6 (chassis 280×280×160), §3.7 (gate inner 1500 / outer 2700 / depth 260)
