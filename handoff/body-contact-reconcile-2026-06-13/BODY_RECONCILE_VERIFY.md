# BODY-CONTACT RECONCILE — adversarial VERIFY leg (2026-06-13)

Fengyou — this is the **refute/verify** leg. Charge: try hardest to show the gate-4 margin does **NOT**
close — i.e. that 0.38 m (or something close) is the right effective contact radius and the
CANNOT-SETTLE-OFFLINE headline stands. I re-derived every load-bearing number from source and the data,
attacked the reconciliation on the four flagged angles (a)-(d), and report a verdict.

Flags: **[M]** measured/logged-verbatim · **[X]** derived-here · **[A]** assumed.
No `src/` or `memory/` edits. Sub-legs cross-read: `provenance.md`, `geom_radius.py`,
`empirical_radius.md`, `CORRECTED_MARGIN.md`; margin source `d3_margin_closure*.{py,json}`,
`v3_margin_refute*.{py,json}`.

---

## BOTTOM LINE: **QUALIFIED**

The reconciliation's **headline conclusion is UPHELD and is robust**: the gate-4 contact-true margin does
**NOT** settle offline, because the binding case — cold case-C with the **MEASURED 1.4 deg attitude bias**
— **fails at EVERY physically admissible radius**, p90 and p99 alike. I could not construct an honest
radius that rescues it; the attempt to do so is what this leg exists to test, and it fails on the geometry.

But I **QUALIFY** (not UPHOLD outright) the *reconciliation document* on two material points where it is too
generous to the "it closes" side:

1. **The synth verdict label `CLOSES_EXCEPT_WORST_BIAS` over-claims.** Its one genuinely-newly-cleared
   *realistic* case — cold @ zero accel bias — clears **only at p90 and only knife-edge** (+0.001 m at
   r=0.30). At **p99 that same case FAILS at every admissible radius** (cold@bias0 p99 = 0.322;
   r_max-to-clear = 0.213, i.e. exactly the geometric floor — so it needs the *most optimistic possible*
   rigid-body radius even to straddle). The task explicitly demands p90/p99 ("a margin is a worst-case
   gate"). On a p99 reading, the radius correction clears **nothing in case C** — only the non-case-C warm
   ceiling and the idealized cold@bias0-at-p90 move. The headline is therefore *more* robust than
   "CLOSES_EXCEPT_WORST_BIAS" implies, not less. The correct label is **QUALIFIED-NO-CLOSE**: closes only
   for non-case-C/idealized tails at p90; the realistic case-C tail does not close at any radius at p90 or
   p99.

2. **0.38 is correctly diagnosed as NON-geometric, but the geometry leg slightly UNDER-states the prop
   contribution and over-rounds the "yaw-invariant" claim** — neither changes the verdict, but both are
   places the prompt's "0.38 might be partly justified" intuition is *more* alive than the reconciliation
   admits. See angle (a).

---

## 1. Re-derivation of the load-bearing chain (all independently reproduced this session)

| Quantity | Value | Source verified | Flag |
|---|---|---|---|
| Chassis W×L×H | 280×280×160 mm | spec PDF §3.6 (`spec_text_dump.txt:162-166`) | [M] |
| Gate inner half-opening `_HALF_OPEN` | 0.750 m | spec §3.7 (1500/2) == `offline_rollout.py:65` | [M] |
| `MARGIN_G4` (= budget at r=0.38) | 0.155 m | `d3_margin_closure.py:59` + JSON `margin_g4_m` | [M] |
| `body_radius` worst/nom | 0.38 / 0.33 | `contact_true_eval.py:51-52` (hardcoded; DR top / mid) | [M] |
| DR band provenance | "[0.28,0.38] brackets measured contact offsets" | `peregrine_racing.py:163-164` | [M] |
| Nominal gate-4 crossing offset `linf₀` | 0.215 m | back-out (0.75−0.38)−0.155; radius-invariant | [X] |
| budget identity | `budget(r)=(0.75−r)−0.215` | reproduces MARGIN_G4 bit-for-bit at r=0.38 | [X] |
| Tilt-projected L-inf (gate-4 posture) | **0.2127 m** | re-ran `geom_radius.py` (0.2127) | [X] |
| **3D box half-diagonal (absolute ceiling)** | **0.2135 m** | `sqrt(.14²+.14²+.08²)` — see §2 | [X] |
| cold@bias0 p90 / p99 @37 m/s | 0.234 / 0.322 | d3 `race_speed_verdict`; v3 `R0_d3_cold` | [M] |
| cold@1.4°-bias p90 / p99 (random) | 0.338 / 0.453 | v3 `R2.bias_0.24` (g·sin1.4°=0.240) | [M] |
| cold@1.4°-bias in-plane (worst) p90/p99 | 0.348 / 0.476 | v3 `R3.inplane` | [M] |

The two-framings-are-one-inequality insight (`budget(r) = (0.75−r) − linf₀`, clears iff estimator
percentile < budget) is **correct** and I confirm it reproduces `MARGIN_G4=0.155` at r=0.38 exactly. The
prompt's premise — "0.155 is the r=0.38 instance of the budget" — is **CONFIRMED**.

---

## 2. The geometric kill-shot (strengthens the NON-geometric finding beyond the reconciliation)

The reconciliation says geometry "tops out at ~0.213 m" via yaw-swept silhouette. I push this to the
**absolute** bound: the **full 3D half-diagonal of the chassis box is 0.2135 m**
(`√(0.140²+0.140²+0.080²)`). The L-inf (and even the Euclidean) in-plane radius can never exceed the 3D
half-diagonal, for **any** rotation, **any** approach normal, **any** in-plane basis — projection only
shrinks. So:

> **No rigid-body orientation of the documented chassis can present more than 0.2135 m.** 0.38 is
> 1.78× that hard ceiling. 0.38 is provably NOT chassis geometry — to the strongest possible standard,
> not just "at the gate-4 posture." [X]

I also confirmed the yaw-invariance is **real, not a grid artifact** (fine 0.05° grid: L-inf spread = 1e-5
m; the worst corner sits essentially in the gate plane, along-normal component only −0.0115 m, so the
silhouette captures its full 0.2135 m extent). The reconciliation's 0.2127 is right.

**Consequence the reconciliation states and I confirm:** the ~0.17 m gap between geometry (0.213) and the
budget radius (0.38) is **necessarily** something other than rigid-chassis geometry — prop/rotor/blade-
strike halo (undocumented: spec lists no prop extent, `spec_text_dump.txt:162-166` has chassis only) and/or
deliberate estimator/registration conservatism stacked into the constant. This is the crux of the verdict
and is treated in §3-§4.

---

## 3. Attacking the four flagged angles (the refutation attempt)

### (a) Does the tilt-projection UNDER-count? — PARTIALLY YES, but not enough to reach 0.38.

I tested every way the bare silhouette could be too small:

- **Crab/yaw misalignment** (body heading ≠ gate normal): swept extra body-yaw ±45° AND tested the 55° as a
  *heading crab* `Rz(55)Ry(−38)` vs a *body roll* `Rx(55)`. **L-inf is invariant at 0.2127** across all of
  it (worst = best). Yaw misalignment adds **nothing** — the box's in-plane extent is set by the corner,
  which the gate-aligned L-inf already captures. **This refutation angle FAILS.** [X]
- **Off-center body** (the in-plane miss): the contact-true model inflates the *aperture* by r around the
  **point** trajectory; the recorded `linf₀=0.215` already **is** the off-center crossing offset, and
  estimator error `m` already perturbs it. Off-center is captured by `linf₀ + m`, **not** by `r` — adding
  it to `r` would be double-counting. **This refutation angle FAILS (and exposes a double-count trap that
  the reconciliation correctly avoids).** [X]
- **Prop disc** — the **one angle with real teeth.** The geometry leg ran props *off* (PROP_TIP_FULL_SPAN
  = None) and reported 0.213 as "the geometric max." That is the **chassis-only** max. A realistic racing
  prop on a 280 mm-class frame extends beyond the chassis. Adding 4 prop tips:
  - full-span 0.30 m → L-inf **0.213** (tips inside the tilted box silhouette; no change)
  - full-span 0.36 m → L-inf **0.233**
  - full-span 0.40 m → L-inf **0.259**
  - full-span 0.45 m → L-inf **0.291**

  So a plausible prop envelope lifts the effective rigid radius into the **0.23–0.29 m** band — **partway
  to the empirical 0.26–0.38, but still short of 0.38** even at a 0.45 m span. **This is the legitimate
  residue of the prompt's "0.38 might be partly justified" intuition**, and the geometry leg under-states
  it by labeling 0.213 "the geometric max" when it is only the *chassis* max. It does **not**, however,
  reach 0.38 — and it is undocumented [A] (spec has no prop dimension). **Partial success; insufficient to
  overturn.** [X/A]

### (b) Does 0.38 deliberately STACK control-tracking + gate-registration error onto the bare body, so shrinking it double-discounts? — **THE STRONGEST PRO-0.38 ARGUMENT, and it is real.**

Provenance (`peregrine_racing.py:153-164`, `contact_true_eval.py:51`) shows 0.38 is the **upper tail of a
DR band fit to live-crash L-inf (0.37–0.49 steep; corner-pass 0.60)** — i.e. an **effective envelope** that
already absorbs whatever real-world slop (rotor wash, control tracking at the steep tilt, gate registration)
produced those contacts. **If 0.38 already contains the control/registration budget, then shrinking r to
"bare geometry" 0.213 and *separately* adding estimator error `m` could double-count** — the contacts that
calibrated 0.38 happened *under closed-loop control with its own tracking error*. This is the best case that
0.38 captures something the geometry misses, and the reconciliation's "do NOT change the 0.38 constant; keep
it as the worst-case stress knob" is the correct response to it.

**Why it still does not save the close-case:** the d3/v3 estimator error `m` is **estimator-only** (position
estimate vs truth at the plane; `d3:365-374`), and `linf₀=0.215` is the *true* nominal crossing under the
trained controller — so the controller's own tracking is **already in `linf₀`**, not in `m`. The legitimate
worry is whether 0.38's halo and `m` overlap. But the binding failure is at the **measured attitude bias**,
which is an **estimator/IMU** error feeding `m`, conceptually distinct from the **aero/prop** halo in 0.38.
The honest exposure is: **the cold@1.4° p90 0.338 fails even against the FULL r=0.38 budget at p99 (0.453 ≫
0.155) and against the most generous geometry-floor budget (0.322) at p90.** Stacking question moot — it
fails with or without the stack. [X]

### (c) Is the 0.5/0.6 clip from a non-comparable (near-level) posture? — **YES. The prompt's empirical anchor is disqualified.**

`empirical_radius.md` + the primary record (`shadowpc-inc5-live-2026-06-11/WRITEUP.md §5`) confirm the
corner-pass probe is **gate-0, CTBR-bridge, ~3 m/s, near-level** — clean at actual 0.50, contact at actual
0.64 (the 0.60 *label* overshot). That yields near-level `r_eff ≈ 0.18–0.20`, **but it is the wrong posture
for gate-4**. The *only* posture-matched data (gate-3 steep crashes, trained tilt ≈55°/12–18 m/s) logs
contacts at **L-inf 0.37–0.49 → r_eff ≥ 0.26–0.38**. So the prompt's "0.18–0.20 → budget doubles" chain is
**arithmetically right but applies a near-level radius to the steep gate**. **This refutation angle SUCCEEDS
for the pro-0.38 side**: the empirical evidence at the right posture corroborates 0.26–0.38, *not* 0.20, and
the steep-crash numbers are **lower bounds** (true r_eff could exceed 0.38). One caveat that cuts the other
way: those 4 gate-3 events are crashes under a *failed* policy, so they are lower bounds on an effective
envelope that **includes control failure** — i.e. they may overstate r_eff for a *well-tracked* gate-4 pass.
Net: posture-matched data defensibly supports r in [0.26, 0.38]; it does not pin 0.38 as central, but it
**forbids dropping to 0.20 at gate-4**. [M/X]

### (d) Does the corrected budget also clear at p99 (not just p90)? — **NO. This is the decisive caveat.**

Full clear matrix (estimator percentile < budget(r); Y=clears):

| case (gate-4 in-plane, 37 m/s) | quantile | r=0.213 (b=0.322) | r=0.30 (b=0.235) | r=0.33 (b=0.205) | r=0.38 (b=0.155) |
|---|---|---|---|---|---|
| warm (vel=truth — **NOT case C**) | p90 | Y | Y | Y | Y |
| warm | **p99** | Y | Y | Y | **N (+0.023)** |
| cold@bias0 (case-C **ideal**) | p90 | Y | Y (+0.001) | **N** | **N** |
| cold@bias0 | **p99** | Y (−0.000) | **N (+0.087)** | **N** | **N** |
| cold@1.4° random (**REALISTIC**) | p90 | **N (+0.016)** | **N** | **N** | **N** |
| cold@1.4° random | **p99** | **N (+0.131)** | **N** | **N** | **N** |
| cold@1.4° in-plane (**WORST**) | p90/p99 | **N** | **N** | **N** | **N** |

**Reading at p99 (the proper worst-case gate):**
- The realistic case-C tail (cold@1.4°) **fails at p90 AND p99 at every admissible radius** — to clear it
  even at p90 you need r < 0.197 m, **below the flat half-diagonal 0.198 m** (physically impossible);
  at p99 you need r < 0.082 m (absurd). **Radius-invariant FAIL. UPHELD.**
- The *idealized* cold@bias0 — the synth's one "newly-cleared realistic" case — **clears at p90 only
  knife-edge (r≤0.30) and clears at p99 only at r=0.213 (the geometric floor, −0.000 m, i.e. it does not
  truly clear)**. At the synth's own central r=0.30 it **FAILS at p99 (+0.087)**. So even the case the synth
  credits to the radius correction **does not survive a p99 reading at the central radius.**
- Even the **warm** ceiling fails p99 at r=0.38.

**Conclusion on (d): the corrected budget does NOT clear at p99 for any case-C tail.** The synth verdict's
"CLOSES" is a p90-only artifact for the idealized cold case; the realistic case never closes. This makes the
headline **stronger**, and is why the verdict is QUALIFIED toward the reconciliation being **too soft on the
close-side**, while its top-line "does not settle offline" is **upheld**.

---

## 4. The one decisive caveat

> **The realistic case-C gate-4 tail carries the MEASURED 1.4° attitude bias, which maps (g·sin1.4° = 0.240
> m/s² phantom horizontal accel, mapping confirmed exact by v3) to an in-plane estimator p90 of 0.338 m
> (0.348 m in-plane-forced). Clearing it requires budget > 0.338 → r < 0.197 m — below the flat chassis
> half-diagonal (0.198 m) and below the absolute 3D box half-diagonal (0.2135 m). No physically admissible
> contact radius — not the geometric floor, not any value the spec or the empirics permit — closes this
> case, at p90 or p99. The radius is therefore NOT the load-bearing unknown; the effective accel/attitude
> bias entering gate-4 is, and only a live ≥5-lap at-speed gate-4 recording can pin it.**

Corollary the prompt asked me to test: **does the margin close if 0.38 is wrong?** Partially, and only for
cases that don't matter — the radius correction (0.38→0.30) is real (budget 0.155→0.235, ≈1.5×, *not* the
prompt's "doubles", which needs the posture-wrong r≈0.20) and it newly clears the **warm** ceiling and the
**idealized zero-bias cold at p90**. It does **not** clear the realistic case-C tail at any radius, and at
p99 it clears **no** case-C tail at all. The prompt's hypothesis is **vindicated in mechanism for the
idealized case** (0.38 was doing real work in the zero-bias p90 failure) but **refuted in conclusion**: the
CANNOT-SETTLE-OFFLINE headline survives, sharpened.

---

## 5. Where I disagree with / sharpen the reconciliation (for the commander)

1. **Relabel.** `CLOSES_EXCEPT_WORST_BIAS` → **`DOES-NOT-CLOSE (case-C); closes only non-case-C/idealized
   tails at p90`**. The synth's framing credits the radius correction with clearing "the only genuinely-
   newly-cleared realistic case (cold@zero-bias)" — but that case **fails at p99** and is **knife-edge at
   p90**. Calling it "CLOSES" is p90-cherry-picking the one quantile where the idealized case squeaks by.
2. **Geometry leg under-states props.** 0.213 is the **chassis** max, not "the geometric max"; a realistic
   prop span (0.36–0.45 m) lifts rigid r_eff to 0.23–0.29 m. This narrows (does not close) the gap to the
   empirical 0.26–0.38 and is the legitimate kernel of the prompt's "0.38 partly justified." Still ≠ 0.38.
3. **Strengthen the non-geometric proof:** cite the **3D box half-diagonal 0.2135 m as the absolute
   ceiling** for any orientation — a cleaner, posture-free refutation of "0.38 is geometry" than the
   yaw-swept silhouette.
4. **Agree fully** with: keep `--body-radius 0.38` as the worst-case knob (do NOT edit the constant);
   report margin **as a function of r AND quantile (p90 and p99)**; the open quantity is the attitude/accel
   bias, not the radius. The recommended reporting headline should read: *"gate-4 closes only for the
   non-case-C warm ceiling and the idealized zero-bias cold at p90; the realistic case-C tail under the
   measured 1.4° attitude bias fails at every admissible radius at both p90 and p99 — closure is gated by
   the effective bias, pinnable only by a live ≥5-lap at-speed gate-4 recording."*

---

### Files (absolute)
- THIS: `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\body-contact-reconcile-2026-06-13\BODY_RECONCILE_VERIFY.md`
- Sub-legs: `...\body-contact-reconcile-2026-06-13\{provenance.md, geom_radius.py, geom_radius_result.json, empirical_radius.md, CORRECTED_MARGIN.md}`
- Margin source: `...\ultracode-gate-relative-pipeline-design-2026-06-13\{d3_margin_closure.py (MARGIN_G4=0.155:59; inplane_miss 365-374), d3_margin_closure_results.json, v3_margin_refute.py (g·sin1.4°=0.240:225-241), v3_margin_refute_results.json}`
- Code: `rl\contact_true_eval.py:51-52,99-147`; `rl\offline_rollout.py:65-66,121-148`; `rl\peregrine_racing.py:153-164`
- Spec: `260508_Technical_Spec_0002.pdf` §3.6 (280×280×160), §3.7 (inner 1500 / outer 2700 / depth 260)
