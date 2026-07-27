# Gate 5: what the obstacle actually is, what dodge it needs, and what lies past gate 6

Worker report, 2026-07-27. Corpus: 680 recorded flights (671 with a parseable `ego_obs.jsonl`).
All geometry is levelled and aim-de-injected; every instrument guard from the brief is applied and
two of them caught real errors in my own first pass (§7).

**Headline, and it reverses the standing verdict: the gate-5 obstacle is already solved by the
`5:0,3` dodge that is currently pinned.** Band deaths at gate 5 fall from 5/6 to 3/23
(Fisher **p = 0.0025**). What is *not* solved at gate 5 is the gate itself, and that is the ordinary
FOV-blind strike, on which no aim knob can act.

---

## 0. Conventions and instrument

| quantity | definition |
|---|---|
| `p = -(R_level @ rel_true)` | drone **minus** gate, levelled. `p[1] > 0` = drone LEFT of gate, `p[2] > 0` = drone ABOVE gate |
| `R_level` | `Ry(-obs[4]) @ Rx(-obs[3])`, then `q[2] += z_bias·cos20°` |
| de-injection | `rel_true = rel_logged + [0, +aim_lat, -aim_vert]` |
| BAND death | died chasing gate g with terminal range ≥ 8 m |
| ATGATE death | died chasing gate g with terminal range < 3 m |
| MID death | died chasing gate g at 3–8 m |
| dose | read from the LOG (`aim_off[1]`), never the config string |

**De-injection verified by a known-answer test, not assumed.** Across all n=39 aim-release
transitions in the corpus, the *perceived* lever jumps by |Δup| median **3.345 m** while the
*de-injected true* lever stays continuous at |Δup| median **0.234 m**. Sign convention confirmed
against `src/racer/ego_obs.py:_channels` (`rel_flu += [0, -aim_lat, +aim_vert]`) and
`parse_aim_offsets` (`lateral +RIGHT`, `vertical +UP`).

**Release semantics confirmed from the log, not from memory.** `_aim_offset_now` returns `None`
when `rng <= aim_release_m`, so `release` is the range at which the offset is **let go**. Measured:
`release=12.0` → armed over 12.00–22.17 m, unarmed over 1.39–11.99 m. The dodge is therefore armed
*across the whole obstacle band*, which is the thing that matters.

---

## 1. Prior work I did not know about: the pilot already swept gate 5

The brief describes a single gate-5 arm (`5:0,3`, 4/8). The corpus contains **six more**, flown by
the pilot on 07-24/25 — a lateral × vertical sweep, 15 flights, that has never been adjudicated:

| gate-5 offset (lat, vert) | n reached g5 | passed |
|---|---|---|
| `(0, +10)` | 5 | 2 |
| `(-1, +10)` | 2 | 0 |
| `(-1, +5)` | 1 | 1 |
| `(-5, +5)` | 2 | 0 |
| `(-2, +4)` | 3 | 1 |
| `(-3, +4)` | 2 | 1 |
| `(0, +3)`  (07-27, the pinned one) | 8 | 4 |
| **none** | **6** | **0** |

Those 07-24/25 sessions do **not** record `ego_aim_offsets` in `meta.json` (the pilot's ShadowPC
build only emits `aim_off` per tick), which is why they were invisible to a config-string scan.

---

## 2. (a) Where the gate-5 obstacle is

### Range

| | n far deaths | median | IQR | min–max | % within ±2 m of median |
|---|---|---|---|---|---|
| **gate 4** | 27 | **14.49 m** | 1.66 | 12.69–18.39 | **0.96** |
| **gate 5** | 17 | **15.12 m** | 2.73 | 12.9–17.4 (2 stale-fix outliers at 8.1, 2.0) | **0.76** |
| gate 0 | 19 | 17.78 | 5.25 | — | 0.58 |
| gate 1 | 21 | 13.26 | 7.56 | — | 0.24 |

Gate 5's obstacle sits **~0.6 m further from its gate than gate 4's, and its band is ~1.6× wider**
(IQR 2.73 vs 1.66). Per-engagement lethality is *higher*: 26.2 % of gate-5 engagements end in a far
death vs 18.8 % at gate 4.

The two-axis discriminator matters: gates 4 and 5 have both a **high** far-death fraction and a
**tight** mode. Gates 0 and 1 have a low fraction and a wide smear — those are mis-locks, not
obstacles. This is the first quantitative separation of the two.

### Confirmed at the impact, not inferred from the last log line

The 07-27 block (30 flights) carries the post-impact recorder. Using the established discriminator
(`threat_level == 2 AND impulse > 1.0`):

| death class | id 1001 GATE | id 1002 ENV |
|---|---|---|
| band death at **gate 4 or 5** | **0** | **4** (imp 5.2, 5.3, 6.6, 7.4) |
| band death at gate 0, 1, 2 | **3** | **0** |
| at-gate death, any gate | 18 (+2 mixed) | 1 |

4/4 vs 0/3, **Fisher p = 0.0286**. The gate-4/5 band deaths are environment impacts; the far deaths
at gates 0–2 are gate strikes with a bad terminal lever. The single gate-5 case is flight
`20260727_052902`, ENV, impulse 5.3, at 18.3 m from gate 5.

### Offset in the levelled body frame — measurable at gate 4, NOT at gate 5

**Gate 4 (v19Ws0 @30, un-dodged, n=55 legs crossing 12.5–17.0 m):**

| | survived the band (n=37) | band-dead (n=18) | Δ | t |
|---|---|---|---|---|
| levelled **vertical** | **+0.290** | **−0.286** | **−0.577** | **−2.88** |
| levelled lateral | +2.963 | +2.407 | −0.556 | −0.92 |
| speed \|v\| | 9.757 | 9.699 | −0.057 | −0.14 |

Gate 4's obstacle kills the **low** ones; its top edge sits at roughly gate-centre height
(vert ≈ 0). Lateral does not separate — the obstacle spans the corridor the drone flies.

**Gate 5: I cannot measure this, and I am not going to pretend otherwise.** The matched un-dodged
cohort is **n = 5 legs with a fix in the band (6 flights total)**. Every axis returns `nan` or
|t| < 1. Pooling checkpoints raises n to 22 and gives vertical −0.654 (dead) vs −0.274 (alive),
t = −0.96 — same *direction* as gate 4, not significant, and checkpoint-pooling is exactly what
manufactured a false signal elsewhere in this analysis (§7).

**Is it the same shape?** Weakly for: the same +3 m UP dose clears both, and the medians differ by
only 0.6 m. Weakly against: gate 5's band is wider and more lethal. **Verdict: undetermined at the
n available.** Settling it needs ~25 un-dodged flights reaching gate 5; we have 6.

---

## 3. (b) What dodge gate 5 should use

### `ego_aim_offsets = "4:0,3;5:0,3"`, `ego_aim_release = 12.0`, `ego_aim_fade = 0.0`

That is **exactly what is already pinned on `aim-offsets-pin-2026-07-27`. Change nothing.** Four
independent lines say so, and one of them is a warning against the obvious next move.

**1. The obstacle is already solved.** v19Ws0 @30, conditioned on reaching gate 5:

| | n | PASS | BAND | MID | ATGATE |
|---|---|---|---|---|---|
| no dodge | 6 | 0 | **5** | 0 | 1 |
| any vertical dodge | 23 | 9 | **3** | 4 | 7 |

Band-death rate **0.833 → 0.130, Fisher p = 0.0025**. Robust to the band window (the window is set
by the gate-4 ENV impacts and applied unchanged to gate 5 — never gate on the variable under test):
8–25 m → p = 0.0025; 11–20 m, 12–19 m and 12.5–18 m → p = 0.0180.

**Difference-in-differences against the drift.** The 07-24/25 session splits cleanly at 22:59:37
into a pre-dodge era and a gate-5-dodge era. Gate 4 was *never* configured in either, so its
band-death change measures the era drift:

| | era A | era B | log-odds change |
|---|---|---|---|
| gate 4 (untreated control) | 9/19 = 0.474 | 9/36 = 0.250 | −0.993 |
| gate 5 (treated) | 5/6 = 0.833 | 2/15 = 0.133 | −3.481 |

**DiD = −2.488 log-odds → odds ratio 0.083, a 12× reduction** after the drift is removed.
Gates 0–3 placebo across the same eras: p = 0.18, **0.014**, 0.60, 1.00 — gate 1 drifts (that is
the drift the DiD subtracts); the other three are null.

**2. Lateral is NOT the answer, and it has already been tested.** The pilot flew 10 gate-5 flights
with a LEFT lateral of −1 to −5 m:

| arm | passed |
|---|---|
| LEFT + UP (lat ≤ −1) | 3/10 = 0.30 |
| PURE UP (lat = 0) | 6/13 = 0.46 |

Fisher **p = 0.67**. No evidence lateral helps; the point estimate runs the wrong way. The two
largest left doses, `(-1,+10)` and `(-5,+5)`, went 0/2 each. *This is the valuable negative the
brief asked for: gate 5 does not need a lateral dodge.*

**3. Dose-response says 3 m is right and more is actively worse.**

| dose | n | PASS | BAND | MID (3–8 m) | ATGATE |
|---|---|---|---|---|---|
| +3 | 8 | 4 | 1 | **0** | 3 |
| +4 | 5 | 2 | 1 | **0** | 2 |
| +5 | 3 | 1 | 1 | **1** | 0 |
| +10 | 7 | 2 | 0 | **3** | 2 |

MID deaths: **0/13 at dose 3–4 m vs 4/10 at dose 5–10 m, Fisher p = 0.0237**, while band deaths are
flat (2/13 vs 1/10, p = 1.00). **Overdosing buys nothing on the obstacle and creates a new failure
at 3–8 m.** The instinct to "give gate 5 more vertical because +3 wasn't enough" is the one move the
data specifically forbids.

**4. The residual at gate 5 is not the obstacle.** With the dodge armed, of 14 deaths in 23
engagements, **11 are inside 8 m** (7 at-gate, 4 mid). See §5.

### The brief's premise that +3 m "is not enough at gate 5" does not survive

That rested on one flight — `20260727_052902`, band death at 18.3 m with the dodge armed. Its
gate-5 leg trace shows the drone **stalled**: 12 consecutive fresh fixes oscillating between 16.4
and 18.9 m without closing, parked 7.2–8.3 m LEFT of the gate line, then an ENV contact. A
converging leg from the same block (`052652`, PASS) closes 19.7 → 15.3 m monotonically with lateral
shrinking 7.2 → 4.1 m. `052902` is a non-converging approach, not an under-dosed dodge.

### Caveat I will not paper over: the *mechanism* is not established

The +3 UP command does **not** measurably lift the drone inside the band. Matched, range-matched,
v19Ws0 @30:

| | gate 4 (dodge n=9 vs none n=55) | gate 5 (dodge n=22 vs none n=5) |
|---|---|---|
| levelled vertical | −0.028 m, t = −0.17 | −0.190 m, t = −0.29 |
| forward speed | **−0.677 m/s, t = −2.65** | +0.483, t = +0.51 |
| band-death rate | 0.327 → 0.000 | 0.800 → 0.136 |

Across range the dodged cohort does climb relative to a flat baseline (+0.72 m from 22 m to 12 m at
gate 4), but only ~0.2–0.4 m of it has accumulated by the obstacle. **The outcome is established;
how ~0.3 m of lift and a 7 % speed reduction flip a 33 % band-death rate to 0/9 is not.** Do not
bank a mechanism story. The speed channel (t = −2.65) is the most promising lead and is testable
with an `ego_speed_gov` arm at gate 4/5 with no offset at all.

---

## 4. (c) Obstacles before gates 6, 7, 8?

**No positive evidence anywhere. Gates 6 and 8 are geometrically excluded. Gate 7 cannot be settled
at the n available and I am not going to guess.**

### Structural argument: leg length

Measured from the slot-1 lever at the advance tick, so it is **not** capped by
`max_acquire_range_m = 22` the way slot-0 first acquisition is:

| leg into gate | n | median | p10 | p90 | max |
|---|---|---|---|---|---|
| 1 | 497 | 17.4 | 15.8 | 19.2 | 32.9 |
| 2 | 345 | 8.5 | 7.1 | 18.0 | 24.8 |
| 3 | 224 | 11.7 | 9.8 | 18.1 | 30.0 |
| **4** | 144 | **20.6** | 19.0 | 22.1 | 26.6 |
| **5** | 64 | **20.7** | 19.7 | 22.1 | 23.7 |
| 6 | 16 | **15.4** | 14.4 | 16.2 | **16.7** |
| 7 | 8 | 17.9 | 1.9 | 26.7 | 27.1 |
| 8 | 3 | **10.7** | 10.5 | 11.0 | **11.1** |

An obstacle ~14.5–15 m short of gate g requires leg(g) ≫ 15 m. **The two long legs, 20.6 and
20.7 m, are exactly the two that carry obstacles.** An obstacle 14.5 m short of gate 6 would sit
0.9 m past gate 5 — that is not a mid-leg obstacle, that is gate 5's own aperture. Gate 8's leg
(max 11.1 m) cannot host one at all. Gate 7's median 17.9 m *could*.

### Empirical

| gate | engaged | far deaths (≥8 m) | what they actually were |
|---|---|---|---|
| 6 | 16 | 1 | `042836` at 18.95 m — beyond its own 16.7 m max leg length |
| 7 | 8 | 2 | both artifacts (below) |
| 8 | 3 | 0 | — |

Gate 7's two far deaths on inspection: `230235` died at 25.66 m, beyond the 22 m acquire cap → a
re-lock, not a leg position. `233031`'s gate-7 leg has fresh fixes only at 2.1–3.5 m yet a terminal
lever of 14.29 m → a mis-lock seam.

The two gate-7 **ENV** impacts in the recorder block are not mid-leg obstacles either:
- `051018` — **never acquired gate 7 at all**. `pose_seen` false and `rel_flu` null for the entire
  leg; it flew the whole leg blind and took 3 ENV contacts (imp 1.67–4.04).
- `051237` — ENV, impulse 5.47, at **3.2 m** from the gate while **3.49 m BELOW** it. That is a
  terrain/ground signature at the gate, not an obstacle 14.5 m out.

### Power — the honest bound

One-sided 95 % upper bounds on the per-engagement band-death rate:

| gate | observed | 95 % upper bound | gate-5-sized obstacle (0.26) would be ruled out? |
|---|---|---|---|
| 6 | 1/16 | **0.264** | marginally, yes (P(≤1 \| p=0.26) = 0.054) |
| 7 | 2/8 | **0.600** | **no** |
| 8 | 0/3 | **0.632** | **no** |

**Gate 7 is the open question and the data cannot answer it.** An obstacle as lethal as gate 5's
would be completely invisible at n=8. **What would settle it: ~30 flights reaching gate 7 with the
post-impact recorder armed.** At gate-5 lethality that yields ~8 ENV band deaths vs ~0 if clean —
separable at p < 0.01. We currently have 8 engagements ever, and 0 of 671 flights have passed
gate 9.

---

## 5. (d) Gate-5 at-gate strikes are the ordinary FOV-blind strike

v19Ws0 @30, geometry at the **last fresh fix inside 8 m**, aim de-injected, calibrated against
**confirmed passes** at the same gate (footgun 6):

| | last-sighted range (median) | \|lat\| AUC | \|vert\| AUC | **L-inf AUC** |
|---|---|---|---|---|
| gates 0–3 (156 pass / 74 strike) — the reference | 1.82 m | 0.399 | 0.556 | **0.492** |
| **gate 5 (4 pass / 8 strike)** | **1.95 m** | 0.469 | 0.344 | **0.500** |
| gate 4 (8 pass / 15 strike) | 1.86 m | **0.792** | 0.500 | **0.850** |

**Gate 5 matches gates 0–3 on every axis.** Same last-sighted range (1.95 vs 1.82 m — the
established 1.78 m FOV cutoff), and **L-inf AUC 0.500 against its own passes: literally zero
discrimination.** At the last fix, a gate-5 strike and a gate-5 pass are the same distribution.
That is the defining signature of the blind interval. 6 of the 8 gate-5 strikes are inside the
0.45 m half-aperture on **both** axes at the last fix. The recorder confirms the object: all three
gate-5 at-gate deaths in the 07-27 block hit **id 1001 GATE** (impulses 5.4, 2.3–4.7, 6.4).

**Gate 4 is the odd one, not gate 5.** Gate 4's at-gate strikes carry a real lateral signature
(L-inf AUC 0.850) — plausibly obstacle-disturbed approaches arriving off-line. Gate 5's do not.

**Operational consequence.** The offset releases at 12 m; the last fix is at ~1.9 m. **No aim knob
acts anywhere inside the interval that kills at gate 5.** Gate 5's residual is the vision-horizon
problem, and it belongs to the `+env.ego_obs_coast=true` training arm, not to a deploy knob.

---

## 6. Where gate 5's remaining 60 % actually goes

23 engagements with the dodge armed → 9 PASS (0.391), 3 BAND, 4 MID, 7 ATGATE.

| failure | n | owner |
|---|---|---|
| at-gate strike | 7 | FOV-blind interval — training (`ego_obs_coast`), not a deploy knob |
| MID (3–8 m) | 4 | **iatrogenic — 4/4 are dose ≥ 5 m; zero at dose 3–4 m** |
| band | 3 | 1 is a stalled non-converging approach; 2 predate the pinned config |

At the pinned `5:0,3` dose specifically: 8 engagements → 4 PASS, 1 BAND, **0 MID**, 3 ATGATE. Every
remaining death is at the gate.

---

## 7. Two instrument errors I made and caught — bank these

**(1) Reading arm membership from the log conditions on surviving to the dodge gate.** Footgun 1
says read the *dodge* from the log. I extended that to *arm membership* and the by-construction
placebo detonated: gates 0–3 came out 24/24 = 1.000 for the "dodge" arm vs 0.71–0.80 for "none",
p = 0.0021–0.0150 — impossible, since a gate-4/5 offset cannot touch gates 0–3. Cause: a flight
configured for a gate-5 dodge that dies at gate 1 logs no `aim_off`, so it is silently sorted into
"none". **Rule: read the DOSE from the log; read the ARM from the config, or from the session time
block when the build does not record it; and condition on reaching the gate.** Conditioned on
`gmax ≥ 5` the log-derived arm is exact — every dodge-era flight that reached gate 5 logs its
offset — which is why the §3 numbers stand while the naive placebo did not.

**(2) Pooling checkpoints manufactured a 3 m lateral effect that does not exist.** Pooled over all
checkpoints, the gate-5 dodge cohort sat **+3.13 m further LEFT** inside the band than the un-dodged
cohort, **t = +4.64** — a beautiful, entirely fake result. Restricted to v19Ws0 @30 it is **−0.45 m,
t = −0.46**. v15/v16 fly the gate-5 approach on a different line. Related: the pooled free-space
carving at gate 5 gave a lateral separation of −2.48 m (t = −2.73) that vanished on stratification
(t = −0.04). **At gate 5 the carving instrument has n=5 survivors in the un-dodged stratum and
cannot localise anything.**

Also worth noting: the levelled "lateral" in the drone's own yaw frame is approach *geometry*, not
cross-track error — every 07-27 gate-5 leg acquires the gate at lat +3.4 to +8.6 m because the leg
turns. Do not read it as a bias.

---

## 8. Recommended actions

1. **Ship `4:0,3;5:0,3` unchanged.** Do not raise the gate-5 dose. Do not add lateral. Both moves
   are contraindicated by data already in hand (p = 0.024 and p = 0.67 respectively).
2. **Do not spend flights on a gate-5 lateral arm.** It is already n=10 and null. Spend them on
   gate 7 instead.
3. **Fly ~30 flights to gate 7 with the recorder armed** — the only open obstacle question, and the
   only one that is answerable with flights.
4. **Gate 5's residual is the FOV-blind gate strike**, identical to gates 0–3 (L-inf AUC 0.500).
   It belongs to `+env.ego_obs_coast=true`, not to any deploy knob.
5. **Optional, cheap, and it would settle the mechanism:** an `ego_speed_gov` arm at gates 4/5 with
   *no* aim offset. The dodge's only significant in-band effect at gate 4 is a −0.68 m/s forward
   speed reduction (t = −2.65), not a vertical lift. If speed alone reproduces the band-death drop,
   the dodge is mis-named and the real knob is slower approach.

---

## MEMORY-DELTA

- 🟢🟢 **GATE 5's OBSTACLE IS SOLVED by `5:0,3`** — band deaths **5/6 → 3/23, Fisher p=0.0025** (robust across windows, p≤0.018); DiD vs the untreated gate-4 control in the same session = **12× odds reduction**. The brief's "gate 5 is NOT solved" is WITHDRAWN.
- 🛑 **GATE 5 DOES NOT NEED LATERAL — already swept, n=10 (lat −1..−5): LEFT+UP 3/10 vs PURE UP 6/13, p=0.67.** Pilot's 07-24/25 sweep (15 flights, 6 doses) was invisible because that build writes no `ego_aim_offsets` to meta.
- 🛑 **DO NOT RAISE THE DOSE: MID (3–8 m) deaths 0/13 at +3/+4 vs 4/10 at +5/+10, p=0.0237**, with band deaths flat. Overdosing is iatrogenic. `+3` is the right dose.
- 🚩 **"+3 IS NOT ENOUGH AT GATE 5" WITHDRAWN** — that one flight (`052902`) STALLED at 16.4–18.9 m for 12 fixes, 7–8 m off-line, then hit ENV. Non-converging approach, not an under-dose.
- 🟢 **GATE-5 AT-GATE STRIKES = THE ORDINARY FOV-BLIND STRIKE**: last-sighted 1.95 m (gates 0–3: 1.82) and **L-inf AUC 0.500 vs its own passes = ZERO discrimination**. Gate 4 is the outlier (AUC 0.850, lateral). Release is 12 m, last fix 1.9 m ⇒ **no aim knob acts in the killing interval.**
- 📏 **OBSTACLE POSITIONS:** g4 median **14.49 m** short (IQR 1.66, 96% within ±2 m); g5 **15.12 m** (IQR 2.73, 76%). Confirmed at the impact: **4/4 g4+g5 band deaths = id 1002 ENV; 0/3 g0–g2 = ENV** (p=0.029). g4 kill axis = VERTICAL (alive +0.290 vs dead −0.286, t=−2.88); lateral null. **g5's cross-section is UNDETERMINED — n=5 un-dodged legs.**
- 🚩 **LEG LENGTHS (slot1 at the advance, uncapped):** g4 **20.6 m**, g5 **20.7**, g6 15.4 (max 16.7), g7 17.9, g8 10.7 (max 11.1), g1 17.4, g2 8.5, g3 11.7. **A 14.5 m obstacle needs a >15 m leg ⇒ gates 6 and 8 are geometrically EXCLUDED.**
- 🛑 **GATE 7 CANNOT BE SETTLED (n=8, 95% upper bound on band-death rate 0.60).** Its 2 "far" deaths are artifacts (one at 25.7 m = re-lock past the 22 m cap; one a mis-lock seam) and its 2 ENV impacts are a never-acquired blind leg and a contact 3.5 m BELOW the gate. **Needs ~30 gate-7 engagements with the recorder.**
- 🛑🛑 **NEW FOOTGUN — ARM MEMBERSHIP FROM THE LOG CONDITIONS ON SURVIVING TO THE DODGE GATE.** It put the placebo at 24/24=1.000 on gates 0–3 (p=0.002). **Read the DOSE from the log, the ARM from the config/time-block, and condition on REACHING the gate.**
- 🛑🛑 **CHECKPOINT-POOLING FAKED A +3.13 m LATERAL EFFECT AT GATE 5 (t=+4.64) THAT IS −0.45 (t=−0.46) WITHIN v19Ws0@30.** v15/v16 fly the g5 approach on a different line. Levelled "lateral" at long range is APPROACH GEOMETRY (every 07-27 g5 leg acquires at +3.4..+8.6 m), not bias.
- 🚩 **THE DODGE'S MECHANISM IS UNPROVEN — it does NOT lift the drone in the band** (g4 Δvert −0.028, t=−0.17). Its only significant in-band effect is **forward speed −0.68 m/s, t=−2.65**. Cheap test: `ego_speed_gov` at g4/5 with NO offset.
- ✅ **RELEASE SEMANTICS CONFIRMED FROM THE LOG:** offset armed while `rng > release` (release=12 → armed 12.0–22.2 m). De-injection verified by a known-answer test: true lever continuous across release (|Δup| 0.234 m) vs perceived 3.345 m, n=39.
