# DEPLOY FLIGHT FORENSICS — velocity-runaway/pitch-down + yaw-oscillation mechanism (analysis-only)

2026-07-18, laptop RL-commander delegation. Reads the v1 pick-flights (`wt-arrest`) vs the
champion-ckpt P0.3 population + champion tape (`wt-ratchet`); pure read of `data/runs/*/ego_obs.jsonl`
+ `meta.json`. No shipping-code / config / data edits. Script: `flight_forensics.py` (reproduces every
number). Field conventions established in `G2-FORENSICS.md` / `OFFSET-FIT.md` and re-verified here.

**Sign conventions (code-authoritative + empirical):** `rate_frd=[roll,pitch,yaw]` wire FRD post-clamp;
`rate_frd[1]=+a_pitch` ⇒ **a_pitch<0 = nose-DOWN**; `rate_frd[2]` = yaw cmd, **clamp ±0.7 rad/s**.
Toward-gate yaw sign = `−sign(bearing_FLU_left)` (gate-right ⇒ +yaw), fixed empirically off the champion.
`obs[4]`=body pitch (rad, leveled); horizontal speed = robust windowed diff of `kf_pos_ned` N,E
(kf-horizontal trustworthy per OFFSET-FIT strapdown match −0.22 m; **kf-vertical WEAK, no baro — altitude
claims excluded**). Telemetry dt is jittery 21–105 ms w/ duplicate timestamps ⇒ all derivatives use an
adaptive ±window central difference (g2_forensics convention).

## Run roster — configs isolate the variables

| run | ckpt | pitch_clamp | rate_scale | det_hold | gates | final | dur | note |
|---|---|---|---|---|---|---|---|---|
| **v1_A** | v1As0 | **0.0 = OFF** | 1.0 | 0.3 | 1 | CRASH (coll 2) | 5.5 s | arm A: no pitch fence |
| **v1_R1** | v1Rs0 | 1.0 | 1.2 | 0.2 | 1 | STALLED (coll 1) | 7.6/9.2 s | arm R survivor |
| **v1_R2/R3/R4** | v1Rs0 | 1.0 | 1.2 | 0.2 | **0** | CRASH (coll 2/1/2) | ~2.6 s | die AT gate 0 |
| **champ** | vpeffs0 | 1.0 | 1.2 | 0.2 | 5 | CRASH (coll 1) | 10.8 s | tape source |
| **p03_r1..r5** | vpeffs0 | 1.0 | 1.2 | 0.2 | {2,2,1,2,2} | STALL/CRASH | 5–9 s | settled pop |

🎯 **Arm R and champ share an IDENTICAL deploy recipe** (pitch 1.0 / rate 1.2 / det_hold 0.2 / yaw_clamp
0.7) and differ **only by checkpoint** ⇒ R-vs-champ isolates the *policy*. Arm A additionally turns the
pitch fence OFF and runs rate 1.0 / det_hold 0.3.

---

## Q1 — VELOCITY RUNAWAY + PITCH-DOWN

**Speed profiles (kf N,E diff):**

| run | v_end | v_peak (t) | ratchet¹ | pitch_med | pitch_min | %t<−20° | %t<−30° | accel-phase mean pitch |
|---|---|---|---|---|---|---|---|---|
| v1_A | 12.3 | **12.4** (5.4) | 0.74 | **−29.0°** | **−64.7°** | 0.72 | **0.47** | **−38.0°** |
| v1_R1 | 6.5 | 7.1 (5.8) | 0.90 | −17.1° | −24.5° | 0.28 | 0.00 | −19.9° |
| v1_R2/3/4 | 5.2/7.2/5.4 | 5.2/7.2/5.7 | 0.95–1.0 | ~−18° | −19..−23° | 0–0.39 | 0.00 | ~−19° |
| **champ** | 12.0 | **15.2** (9.1) | 0.78 | −26.5° | −42.7° | 0.70 | 0.31 | −26.5° |
| **p03_r1** | 17.1 | **20.5** (8.1) | 0.88 | −20.5° | −44.8° | 0.55 | 0.33 | −22.6° |
| p03_r2..r5 | 8.1–9.0 | 8.4–9.0 | 0.85–0.98 | ~−17° | −21..−48° | 0.10–0.22 | 0–0.12 | −17..−19° |

¹ratchet = fraction of post-1 s ticks with non-decreasing (smoothed) speed.
**Champion per-gate speed: g0 6.4 → g1 9.0 → g2 7.8 → g3 11.5 → g4 15.2 m/s** (ramps, crashes at 15).

**Findings:**
- **Velocity runaway is REAL and is the SHARED terminal mode of the whole lineage, not a v1-only defect.**
  Speed ratchets nearly monotonically (0.74–0.88) in every committed flyer; the champion itself ramps
  6→15 m/s across 5 gates and crashes at ~15, and **p03_r1 hits 20.5 m/s** (independently reproduces
  G2-FORENSICS's 7→20.9). No policy ever executes a **braking phase**: the sustained-decel+nose-up+low-
  collective test finds a max streak of 0.34 s (v1_A, and speed doesn't actually drop); the only decels
  anywhere are transient gate-pass/turn dips. **The lineage has no learned brake.**
- **"Flies pitch down super hard" is an ARM-A pathology, gated by `pitch_clamp=0.0`.** With the fence OFF,
  v1_A commands and achieves a hard dive: median −29°, **47 % of the flight below −30°, min −65°**, and it
  **accelerates at a mean pitch of −38°** (nose-down converts to forward accel). Arm R and champ (fence
  1.0) are held to −17..−27° and **never** exceed −30° for arm R. So the pitch-down-driven acceleration is
  specific to the un-fenced arm A; arm R cannot dive and dies a different way (Q2/Q3).
- Nuance vs the memory "flat-plant=wall" thread: acceleration is dominated by nose-down attitude
  (accel-phase pitch −38° v1_A / −26° champ), consistent with gravity-projection thrust, not a plant knob.

---

## Q2 — YAW-OSCILLATION MECHANISM (discriminate a / b / c)

**Magnitude, saturation, frequency, gain:**

| run | \|yaw\|_med | **%satur (±0.7)** | flip/s | **a_yaw@10°err** | med\|a_yaw\| | **gain ×champ** | limit-cyc Hz² |
|---|---|---|---|---|---|---|---|
| v1_A | 0.700 | 0.82 | 5.1 | 2.29 | 2.80 | **15.8×** | 1.6 |
| v1_R1 | 0.700 | 0.90 | 5.2 | 2.03 | 2.09 | **14.0×** | 4.5 |
| v1_R2 | 0.700 | 0.68 | 6.8 | 1.88 | 1.25 | **13.0×** | **6.0** |
| v1_R3 | 0.700 | 0.84 | 6.1 | 2.03 | 1.84 | **14.0×** | **6.0** |
| v1_R4 | 0.700 | 0.76 | 6.1 | 1.72 | 1.46 | **11.8×** | **6.0** |
| **champ** | **0.106** | **0.00** | 3.7 | **0.14** | **0.10** | 1.0× | (noise) |
| p03_r1..r5 | 0.06–0.23 | 0.01–0.06 | 3.4–4.8 | 0.11–0.18 | 0.05–0.15 | 0.7–1.2× | (noise) |

²first-negative-lag of `rate_frd[2]` autocorrelation ⇒ implied fundamental; only meaningful when saturating.

**Hypothesis (a) SCANNING current↔next gate — REFUTED.**
- Flip-conditional slot-1 state is **flat**: age_s1@flip ≈ age_s1@non (v1_R2 0.061 vs 0.034; v1_A 0.265 vs
  0.208) and %pose_seen1@flip ≈ @non (v1_R2 0.72 vs 0.74). Flips are **not** triggered by slot-1 staleness.
- Sweep-direction match: each half-cycle points toward the **active** gate (slot0) **76–84 %** for the
  arm-R runs, but toward the **next** gate (slot1) only **53 % = chance**. The oscillation tracks the gate
  the drone is flying at, not an alternation to the next one.

**Hypothesis (b) LATENCY LIMIT-CYCLE — CONFIRMED (the mechanism).**
- The v1 policies command **~2 rad/s of yaw for a 10° bearing error — 12–16× the champion's 0.14 rad/s**
  (p95 desired a_yaw ≈ 3.1 rad/s). That is ~3× the ±0.7 clamp, so the wire **saturates 68–90 %** of ticks
  and **bang-bangs** at the rail. Champion desires <0.7 always ⇒ **0 % saturation**, smooth.
- The three independent arm-R short runs limit-cycle at a **near-constant ~6 Hz** (autocorr first-neg = 3
  ticks), the signature of a self-sustained gain×latency oscillation; amplitude = the clamp; **sign follows
  the active-gate bearing** (match0 76–84 %) and is **independent of slot-1** (per a). All four (b)
  predictions hold.
- **Why the champion is quiet = low gain, not easier geometry.** Bearing separation \|b1−b0\| is similar
  (champ 16° vs v1 ~20°) and champ's slot-1 is not fresher; the sole difference is the ~15× yaw gain.
  Arm R vs champ share the entire deploy recipe, so the oscillation is 100 % attributable to the
  **v1 checkpoint** (a retrain regression), not a clamp/rate/det knob.

**Hypothesis (c) SELF-INDUCED perception-loss feedback — REFUTED / consequence.**
- Event study at pose_seen True→False onsets: prior-5-tick net yaw rotation @loss ≈ baseline
  (ratio 0.58–1.13 for arm R; champ 0.92) — **yaw excursions do NOT precede detection loss**.
- Whole-run detection availability is **normal**: v1 arm-R sees the active gate 76–82 % of ticks,
  ≈ champ 79 % / p03 78–84 %. The dropouts are ambient detector noise shared by all policies, not caused
  by the sweeping. Lag-correlations of \|yaw\| with subsequent Δage are weak (<0.25) and sign-inconsistent
  (and low-power, since \|yaw\| is a saturated near-constant). The yaw oscillation's harm is on **control/
  geometry** (off-aligned arrival), not perception.

---

## Q3 — LAST ~1.5 s FAILURE SIGNATURE + coupling

| run | final@gate | v_last (dV over 1.5 s) | pitch_last | yaw satur last-6 | det@death | signature |
|---|---|---|---|---|---|---|
| **v1_A** | CRASH g1 | 12.3 (**+3.1, still accel**) | **−52°** | **1.00** | LOST (area 0.02) | nose-down dive + accel + **attitude-driven gate loss** (cam +20° at −52° body ⇒ pointing down; %see0 0.58) |
| **v1_R2/3/4** | CRASH **g0** | 5.2–7.2 (+1.5..+4.0) | −8..−18° | **0.70–1.00** | lost ≤0.15 s prior | reach **1.6–2.7 m** from g0, gate centered (b0 −9..+6°), **yaw limit-cycle ⇒ off-aligned arrival ⇒ gate CONTACT** (coll 2) |
| **champ** | CRASH g5 | 12.0 (−2.8) | −28° | **0.00** | ~fresh | pure **speed** (peaked 15) overwhelms at a late gate |
| **p03_r2/4/5** | CRASH g2 | 8.1–9.0 (+0.4..+2.5) | −20..−37° | **0.00–0.11** | **fresh to end** | slower+lower, under-descend (G2-FORENSICS); **no yaw pathology** |

**The v1 death signature is categorically different from the champion-ckpt population.** v1 deaths
uniquely coincide with **yaw saturation (last-6 = 0.7–1.0)** and detection loss; every p03/champ death has
**yaw saturation ≈ 0** and detection **fresh to the end**. The p03 population dies from speed/geometry
with clean yaw (G2-FORENSICS: slower+lower, under-descend). The v1 arms add two NEW earlier failure modes:
arm A = hard-dive runaway + nose-down FOV loss; arm R = yaw-limit-cycle gate contact at g0. Neither v1 arm
reproduces the champion line; both regress on yaw.

---

## Q4 — VERDICT TABLE

| # | claim | verdict | carrying number |
|---|---|---|---|
| Q2-b | **latency limit-cycle from excessive policy yaw gain** | **CONFIRMED** | **12–16× champion yaw gain** (a_yaw@10° = 2.0 vs 0.14 rad/s) ⇒ **68–90 % clamp saturation**, constant **~6 Hz**, sweep-matches-active-gate 76–84 % |
| Q2-a | scanning current↔next gate | **REFUTED** | sweep matches slot-1 only **53 % (chance)**; flips independent of slot-1 staleness (age_s1@flip ≈ @non) |
| Q2-c | self-induced perception-loss feedback | **REFUTED** (consequence) | net-rotation@detection-loss ≈ baseline (**0.6–1.1×**); %see0 76–82 % ≈ champ 79 % |
| Q1 | velocity runaway | **CONFIRMED — lineage-wide** | champ 6→15 crash · p03_r1 **20.5** · v1_A 12.4; **no braking** anywhere (ratchet 0.74–0.88, max brake-streak 0.34 s) |
| Q1 | "flies pitch down super hard" | **CONFIRMED — arm A only** | v1_A median **−29°, min −65°, 47 % <−30°**, accel at −38°; = `pitch_clamp=0.0`. Arm R fenced ⇒ never <−30° |

**Data-limitation honesty (per hypothesis):**
- (b) rests partly on three **short** arm-R runs (67 ticks / 2.6 s each) — but the ~6 Hz saturated limit-
  cycle is **consistent across all three** and reproduced in the longer v1_R1 (240 ticks) and v1_A; the
  yaw-gain measure uses 51–226 fresh ticks/run. Robust.
- (a)/(c) are **cleanly discriminated** in every run (flip-conditional + event-study), not gated by run
  length; the longer champ/p03 (128–287 ticks) strengthen the baselines.
- Q3 arm-R "off-aligned arrival vs altitude at the gate" cannot be split from telemetry alone (kf-vertical
  weak); det-fresh + centered bearing point to alignment/attitude ⇒ **video adjudication recommended**
  (telemetry = defendant, per G2-FORENSICS).

**Reconciliation for the RL commander:** the deploy flip rate (5.1–6.8/s) matches the v1 training in-run
flips (5.5–7.1) almost exactly ⇒ the training-metric "flips>4" flag was a **real yaw instability**, not a
gate-throughput artifact. The v1 retrain's anti-dither/yaw handling produced a policy that saturates the
yaw clamp; **vpeffs0 does not.** v1 fails the /goal on both counts (velocity runaway present in arm A;
severe yaw oscillation in both arms). The champion's yaw is quiet purely by lower gain — the property to
preserve/retrain toward, not the clamp.
