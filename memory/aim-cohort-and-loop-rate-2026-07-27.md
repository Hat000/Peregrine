---
name: aim-cohort-and-loop-rate-2026-07-27
description: "First 7 flights on the aim-offset branch: the offset test did NOT run (1 activation), but the post-impact recorder captured 7/7 gate impacts and a loop-rate mismatch surfaced (recipe pins rate=40, training dt=30)."
metadata: 
  node_type: memory
  type: project
  originSessionId: b85130f9-ac88-4db2-8c68-0e28b966cf80
  modified: 2026-07-27T05:21:30.151Z
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

Related: [[gate-strike-last-3m-2026-07-27]] · [[failure-census-2026-07-27]].
