---
name: aim-cohort-and-loop-rate-2026-07-27
description: "First 7 flights on the aim-offset branch: the offset test did NOT run (1 activation), but the post-impact recorder captured 7/7 gate impacts and a loop-rate mismatch surfaced (recipe pins rate=40, training dt=30)."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T05:47:50.495Z
---

# The 7 aim-offset flights (2026-07-27 04:38-04:40) — what they did and did not settle

Cohort: `20260727_0438{40,58}` / `0439{17,44}` / `0440{03,17,29}`, all `v19Ws0`, all
`ego_aim_offsets = "4:0,3;5:0,3"`, release 12.0, fade 0.0. Logs on
`origin/ratchet-arrestor-2026-07-18` (`8770defa`). All 7 CRASH.

## 🛑 THE AIM TEST DID NOT RUN — n=1. "No improvement" is NOT evidence against it.

`aim_off` non-null on **one flight only** (043917, 22 ticks at gate 4). The other six never
reached gate 4, and `aim_off` is null on **every tick** of all six ⇒ the offset is **provably
inert** there. Gates reached: `[0,0,0,1,2,2,4]`.
🟢 Arming verified from `meta.json`, not from the launcher: the spec, release and fade are all
exactly as briefed, and `recipe_drift: ["ego_aim_offsets"]` shows the panel flagged the override.

## 🟢🟢 THE POST-IMPACT RECORDER WORKED — 7/7, first impact data in the project

`ego_postimpact.jsonl` exists **only** in these 7 runs (checked: no earlier run has it).
21 ticks / ~0.513 s past the terminal tick, every flight.
🚩 **`COLLISION id 1001 = GATE, 1002 = ENVIRONMENT`** (`src/racer/race_outcome.py:11`,
`GATE_COLLISION_ID`). **All 7 flights end on a 1001 = GATE contact**, impulse 4.06–7.32, landing
at terminal **+0.000 to +0.056 s**. Including the three that died at 2.2 s at gate 0 — those are
**gate strikes at gate 0**, not a launch-window failure (corroborates the banked correction).
🚩 The accel spike is only resolvable in 3 of 7 (40 Hz IMU vs a <25 ms impact); in those 3 the
levelled reaction is **HEAD-ON 92–98%**. In the other 4 the nearest tick reads ~1.5 g = no impact
sampled — **do not read those as impact directions.**

## 🎯 THE GATE-4 FLIGHT IS THE SIDE-STRIKE, ON TAPE — and it contradicts "last 3 m" for THIS flight

043917, gate 4, levelled + z_bias-corrected, drone MINUS gate (+lat = drone LEFT):

| rng | 7.73 | 5.42 | 4.07 | 2.95 | 2.22 | 2.53 (terminal) |
|-----|------|------|------|------|------|------|
| lat | +1.82 | +1.49 | +1.43 | +1.26 | +1.34 | **+1.57** |

It flew a line **1.3–1.8 m left of centre from acquisition to impact** and never converged.
The SAME flight passed gates 0–3 with |lat| at 3 m = **0.24 / 0.54 / 0.47 / 0.84 m**. So gate 4's
lateral bias was **2–5× every other leg in the same flight and established before 8 m**.
🛑 **CANNOT BE ATTRIBUTED — n=1 and confounded.** That leg is also the only one with the offset
armed; it released at **11.66 m with a 3.17 m step** in the perceived lever. "Gate 4 is hard" and
"the release step perturbed it" are indistinguishable here. `ego_aim_fade` > 0 is the one-knob
test that separates them (fade has NEVER been flown ⇒ treat as a new arm).
🚩 Terminal levers are trustworthy: `aim_off = None` at the terminal tick on all 7 (release is
`rng <= 12 m`, `ego_obs.py:_aim_offset_now`), and 043917/043944 are `pose_seen=True, age=0.000,
conf=1.00`. 🚩 **My >1.5 m mis-lock guard FLAGS THE AIM RELEASE** (the 3.17 m step is the offset
coming off) — exclude the release tick before applying it to any aim-offset flight.

## 🚩 LOOP-RATE MISMATCH — the strongest actionable lead, ranked honestly

**BY CONSTRUCTION (certain, needs no statistic):** the panel knob `rate` documents itself
*"Control loop Hz (**training dt = 30**)"* with knob default **30.0**, but **`_V1_RECIPE` pins
`"rate": 40.0`** (`tools/pilot_panel.py:427`) and every later recipe spreads from it. `rate` IS in
`_recipe_managed_keys()` ⇒ **every model-pick sets the control loop to 40 Hz against a 30 Hz-trained
policy.** 🛑 My own "pick the model FIRST" instruction is what re-applied it.

**CORRELATIONAL (suggestive, CONFOUNDED WITH DATE — all slow flights are 07-25/26, all fast are
07-27):** achieved loop rate (measured from `sim_time_ns`, not the request — 30 requested → ~21 Hz,
40 requested → ~25 Hz):

| achieved | n | gates reached | mean | died at g0 | reached ≥4 |
|---|---|---|---|---|---|
| <23 Hz | 5 | 2,3,4,4,6 | 3.8 | 0 | 3 |
| ≥23 Hz | 13 | 0,0,0,0,0,0,1,1,1,2,2,3,4 | 1.08 | 6 | 1 |

Fisher: reached ≥ g4 **p = 0.044**; died at g0 p = 0.114; v19Ws0-only 30-vs-40 requested p = 0.067.
Mechanism is specific and not merely correlational: `det_hold`/`stale_horizon` are in **seconds**, so
a faster loop at an unchanged vision rate lowers the fresh-fix fraction (81–98% → 69–91%).
🚩 **The 07-27 v20Vs0 session shows the same gate-0 cluster (3/6) with NO offset set** ⇒ whatever it
is, it is not the aim offset.

## 🟢🟢 RATE 30 FLOWN — CONFIRMED, AND IT IS A TRUE SINGLE-VARIABLE TEST

`20260727_0509-0513`, n=9, same v19Ws0, same branch, **31 minutes after** the rate-40 set. The two
runs' recorded `meta.json` differ on **exactly one substantive key: `rate_hz` 40.0 → 30.0**
(`recipe_drift` is the panel's own bookkeeping of that same override). ⇒ **the date confound is
dead.**

| | n | gates reached | mean | died at g0 | ≥ g3 | ≥ g4 |
|---|---|---|---|---|---|---|
| rate 40 | 7 | 0,0,0,1,2,2,4 | 1.14 | 3 | 1 | 1 |
| **rate 30** | 9 | 1,2,3,3,4,5,7,7,8 | **4.44** | **0** | **7** | **5** |

Fisher: reached ≥ g3 **p = 0.041**; died at g0 p = 0.063; ≥ g4 p = 0.145. Achieved loop 21–22 Hz
(vs 24.5–26). **Fresh-fix fraction 83–99% (median 96%) vs 69–91%** — the predicted mechanism.
Best flights now reach **gate 7 twice and gate 8 once**, i.e. past both invisible obstacles.
🚩 The aim offset armed on **5 of 9** flights (21–52 ticks, gates 4 and 5) — the offset test is now
live for the first time, but still unadjudicated.

## 🚩 A SYSTEMATIC **LEFT** BIAS — the side slams are the TAIL OF A SHIFTED DISTRIBUTION

Kill axis is now read **at the impact** (strongest `id=1001` GATE contact → last fresh fix on that
gate → levelled), not at the last log line: `scripts/postimpact/`.
**Every classified lateral kill across both cohorts puts the drone LEFT of the gate**: +1.07, +1.10,
+1.28, +1.57, +0.92 m (5/5; fair-coin p = 0.031).

Calibrated on **CONFIRMED PASSES** (the surface where a strike is impossible), closest fresh fix
inside 5 m, aim-armed ticks excluded — 07-27 v19Ws0, n=49:
* lateral **mean +0.285 m, median +0.238, t = +3.99, 36 left / 13 right** (sign test p = 0.0016)
* vertical mean −0.097 — **the bias is LATERAL-ONLY**
* pass |lat| p90 = **0.84 m**; the kills are 0.92–1.57 ⇒ **continuous, one distribution, not a
  separate failure mode**

🟢 **NOT A LEVELLING ARTIFACT — it survives removing the instrument**: on the RAW unlevelled
`rel_flu` it is +0.238, 37/12; roll-sign-flipped +0.179. Sign anchored to CODE, not memory
(`parse_aim_offsets`: `lateral_m` is +RIGHT, applied `rel_flu[1](LEFT) -= lateral_m` ⇒ drone-minus-
gate lateral = `−rel_flu[1]`, positive = drone LEFT).

🛑 **NOT YET SHOWN TO BE STANDING.** Split by cohort: v19Ws0 rate30 07-27 **+0.246 (t 2.96, 29/11)**
and rate40 07-27 **+0.462 (t 3.95, 7/2)** — so it is **not the rate** — but v19Ws0 rate30 **07-25 is
+0.080 (t 0.35, 6/10)**. That 07-25 null is **UNDERPOWERED, not a refutation**: n=16, sd 0.888,
SE 0.222 ⇒ it could hide the entire +0.25 effect at 1.1σ. Established for tonight; one more cohort
decides whether it is a standing property.

🛑 **My range-scaling → "9° yaw boresight" read is WITHDRAWN.** Lateral offset shrinking as the
drone closes in is just the loop working; the regression slope identifies no mount angle. What
survives is **the SIGN** and that the offset is still **+0.115 m at 1–2 m** (t = 3.8, n = 128).

**THE TRIM IS EXPRESSIBLE TODAY, ZERO CODE** (verified against the real builder): `ego_aim_offsets`
= `0:0.25,0;1:0.25,0;…;8:0.25,0` with **`ego_aim_release` = 0.0** — `_aim_offset_now` returns None
only when `rng <= release`, so 0.0 keeps it armed **through the gate** (12.0 switches it OFF at
exactly the range that matters). 🛑 **One global release ⇒ the lateral trim and the gate-4/5
vertical dodge are MUTUALLY EXCLUSIVE.** An always-on lateral trim (the analogue of
`ego_gate_z_bias`, which has **no lateral counterpart** — checked the whole knob list) is the small
default-OFF build that would let both run.

## 🛑🛑 THIRD COHORT (05:25, n=9) — MY DIRECTIONAL CLAIM DIES, THE DISTRIBUTION SURVIVES

Same config as 05:09 (`rate` 30, offset `4:0,3;5:0,3`). Gates `[0,0,1,3,3,5,5,5,6]`, mean 3.11.
**Pooled rate-30 (n=18): mean 3.78 vs rate-40's 1.14** — the rate result is untouched.

🛑🛑 **"ALL 5 LATERAL KILLS PUT THE DRONE LEFT (p = 0.031)" IS WITHDRAWN.** This cohort's four
lateral kills are **−0.48, +0.71, −0.77, −0.64 — three on the RIGHT.** Pooled across all three
cohorts: **6 left / 3 right, sign-test p = 0.51.** A five-sample direction count was never
evidence; it took nine more flights to say so. **Never bank a direction off n=5.**

🟢 **The pass-lateral shift SURVIVES and tightens as n grows** — the opposite behaviour to an
artifact:

| cohort | n | mean lat | t | L/R | sign p |
|---|---|---|---|---|---|
| 05:09 rate30 | 40 | +0.246 | +2.96 | 29/11 | 0.0064 |
| 05:25 rate30 | 28 | +0.119 | +1.29 | 19/9 | 0.087 |
| **pooled rate30** | **68** | **+0.193** | **+3.13** | **48/20** | **0.0009** |
| pooled 07-27 v19 | 77 | +0.225 | +3.94 | 55/22 | 0.0002 |

🟢 **The model is self-consistent**: μ = +0.193, σ = 0.511 predicts **3.0 : 1** left-vs-right
beyond ±0.45 m; observed 2.0 : 1. So the slams ARE the tail of a shifted distribution — the claim
just lives at the level of **the distribution**, never the individual kill. Trim target is now
**+0.20** (from n = 68, not n = 40).

🟡 **Gate-5 attrition looks broken open: 4 of 9 flights that reached gate 4 passed gate 5 (44%)
vs the census's 9 of 101 (8.9%), Fisher p = 0.011.** 🛑 **CONFOUNDED** — rate AND the aim offset
both differ from the census; there is **no control arm** (every flight in all three cohorts has
the offset configured), so the vertical dodge is still **unadjudicated**.

🚩 **META PROVENANCE HOLE:** `recipe_drift` flags `ego_assist_thrust` from flight 3 of this set
onward, but the meta block in `fly_rl.py` **never writes its value** (it writes fix_gain,
propagate_range, vel_fuse, aim_*, obs_coast, rate_scale, z_bias, clamps — not assist). The flown
value is unrecoverable from the log. Trivial fix; until then a drifted assist silently confounds
a cohort. Ask the pilot what changed at 05:26.

## 🟢🟢 THE LATERAL TRIM FLEW (05:31–05:40, n=5) — IT ACTUATES, THEN THE AUTHORITY EVAPORATES

`0:0.25,0;…;8:0.25,0`, `ego_aim_release 0.0`, armed 50–235 ticks/flight. **De-inject before any
geometry**: logged `rel_flu` includes the offset, so `rel_true = rel_logged + [0, +aim_lat, −aim_vert]`.
🚩 **The pass-point statistic (n=16) said the trim went the WRONG way (+0.13). It was mixing
ranges.** Range-MATCHED at tick level it is unambiguous — the trim should give −0.25:

| range | baseline (n) | trim (n) | delta |
|---|---|---|---|
| 8–12 m | +0.639 (848) | −0.146 (182) | **−0.785 ± 0.124** |
| 5–8 m | +0.385 (627) | +0.082 (161) | **−0.303 ± 0.067** ← full authority |
| 3–5 m | +0.384 (413) | +0.193 (96) | −0.191 ± 0.047 |
| 2–3 m | +0.302 (248) | +0.209 (70) | −0.093 ± 0.068 |
| 1–2 m | +0.153 (170) | +0.057 (42) | **−0.096 ± 0.049** ← ~40% left |

## 🛑🛑 VISION BLACKS OUT INSIDE 1.5 m — AND THE OUTCOME IS DECIDED THERE

Fresh-fix rate vs range (n≈4200 ticks): 99% beyond 4 m · 95% at 2.5–4 · **82% at 2–2.5** ·
**80% at 1.5–2** · **24% at 1–1.5** · **0% inside 1 m** (median fix age there **0.285 s** ≈ 2.3 m of
travel at 8 m/s). `pass_drop_range_m = 2.5` does not make a cliff at 2.5; the collapse is ~1.5 m.

🛑🛑 **AT THE LAST SIGHTED FIX, PASSES AND GATE-STRIKES ARE NEARLY THE SAME DISTRIBUTION**
(fresh fix inside 3 m; 93 passes vs 23 gate-strikes):

| discriminator at last sight | pass p50 | strike p50 | **AUC** |
|---|---|---|---|
| **\|lateral\|** | 0.35 | 0.42 | **0.582** ← barely above chance |
| \|vertical\| | 0.19 | 0.39 | 0.676 |
| **max(\|lat\|,\|vert\|) = the L-inf** | 0.39 | 0.71 | **0.736** ← the real discriminator |
| range at last sight (BLIND METRES) | 1.80 | 1.73 | 0.420 (none) |
| speed · \|roll\| · \|pitch\| · \|yaw\| rate | — | — | 0.53 / 0.49 / 0.55 / 0.47 (none) |

**40% of CONFIRMED PASSES are already outside 0.45 m laterally at last sight and pass anyway**;
strike |lat| includes 0.02, 0.05, 0.06, 0.16 — dead-centre at last sight and still hit.
⇒ **No aim/trim/offset knob can fix this: they move a MEAN, and what kills is the SCATTER, which
already straddles the aperture before the drone goes blind.** This is a policy-precision problem,
i.e. TRAINING, not deploy.

🛑 **SELF-CORRECTION: "the reward prices L-inf, which is the v2.1 defect" is UNDERMINED BY MY OWN
DATA.** `cross_offset = pass_linf` prices **exactly the quantity that best predicts survival**
(0.736 vs 0.582 lateral). Lateral scatter IS ~1.8× vertical at last sight (p50 0.35 vs 0.19), but
it discriminates WORSE because passes tolerate it. The defect is **scatter magnitude**, not the
choice of norm. → supersedes the v2.1 candidate line in [[gate-strike-last-3m-2026-07-27]].

🚩 **Trim cohort died 3/5 on ENVIRONMENT at gate 4, not on a gate** — expected: `release 0.0` for
the lateral trim removes the gate-4/5 vertical dodge (the mutual exclusivity, realised). Put the
dodge back; the trim buys nothing at the gate.

🚩 **PILOT OBSERVATION (Fengyou, 07-27): takeoff assist was turned OFF deliberately — the
transition OUT of assist SINKS the drone right before a gate.** That is the `ego_assist_thrust`
drift from 05:26 onward, and it is a real handoff-transient defect worth its own look; it sits next
to the TERMINAL DIVE open defect (`fly_rl.py:766` fences nose-DOWN only).

Related: [[gate-strike-last-3m-2026-07-27]] · [[failure-census-2026-07-27]].
