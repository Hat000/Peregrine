# Gate-2 descend-wall forensics (analysis-only) — the floor does NOT bind; the wall is a launch-line / chaotic-divergence problem (d) + an insufficient-descend command (c), NOT a mechanical floor (a) and NOT perception loss (b)

2026-07-18, laptop RL-commander side, branch `ratchet-tape-2026-07-17`. Reads the N=5 settled-GO
champion-recipe policy flights + the champion tape source (`data/runs/*/ego_obs.jsonl`); no
shipping-code / config / test / data edits. Script: `g2_forensics.py` (reproduces every number).

**Leg map** (active `gate_index==2` == the g1→g2 descend leg, map row `[0,-1]`): r1 passes g1 then
hunts g2 for 151 ticks (survivor); r2/r4/r5 die 12/24/26 ticks into the leg; **r3 never reached it**
(died targeting g1); champ threads the leg in 27 ticks and PASSES g2 at 6.189 s.

**Semantics established from code** (`rl/fly_rl.py`, `src/racer/ego_obs.py`):
- `--ego-assist-thrust 1.3` sets BOTH (i) the **takeoff-assist** emitted-collective floor (1.3 g →
  wire 0.345; hover 1.0 g → 0.266) which is **PERMANENT-handover on first |gyro|>1 rad/s OR climb>0.5 m
  OR timeout**, and (ii) the `EgoFloorClamp` arrest target — but `EgoFloorClamp` fires only with
  `--ego-floor-clamp>0`, which is **ABSENT from the champion recipe ⇒ OFF (bit-identical passthrough)**.
- 21-dim obs: `obs[4]`=body pitch (rad); `obs[10]`=coarse-map vert (−1=descend); `obs[13]`=UP-offset to
  ACTIVE gate (+=gate above), `obs[18]`=UP-offset to NEXT gate; slot rel/conf/area **mask to 0 once
  age≥det_hold(0.2 s)** (coast OFF = champion default). `act_raw[0]`=policy-desired thrust (g, pre-clamp);
  `normed_thrust`=EMITTED thrust after floors; `rate_frd[1]=+act_raw[2]` (=+a_pitch).

---

## Q1 — FLOOR HYPOTHESIS (top priority): **REFUTED, decisively**

| run | nTk | assistF | floorF | coll_min(g→wire) | norm_min(g) | polThr_min | polThr_med | %pol<hover | **%floorBinds** |
|---|---|---|---|---|---|---|---|---|---|
| r1 | 151 | 0.00 | 0.00 | 0.019 | 0.071 | 0.071 | 1.497 | 0.13 | **0.00** |
| r2 | 12 | 0.00 | 0.00 | 0.026 | 0.099 | 0.099 | 0.464 | 0.75 | **0.00** |
| r4 | 24 | 0.00 | 0.00 | 0.239 | 0.899 | 0.899 | 1.168 | 0.12 | **0.00** |
| r5 | 26 | 0.00 | 0.00 | 0.194 | 0.730 | 0.730 | 1.226 | 0.12 | **0.00** |
| champ | 27 | 0.00 | 0.00 | 0.020 | 0.075 | 0.075 | 0.731 | 0.67 | **0.00** |

- **No floor is active anywhere on the descend leg** (`assistF=floorF=0.00`), and `normed_thrust`
  never exceeds `act_raw[0]` (`%floorBinds=0.00`) — the emitted thrust equals the policy-desired
  thrust tick-for-tick. The takeoff-assist 1.3 g floor **hands over permanently at t=0.056–0.118 s**
  (during lift-off), ~5.3 s *before* the leg. `EgoFloorClamp` is OFF.
- Descent-via-thrust is **fully available and used**: collective is driven to near-zero (`coll_min`
  0.019–0.026 on r1/champ = policy thrust 0.07–0.10 g), and both r2 (75%) and champ (67%) spend most
  of the leg below hover. **The wall is not mechanical — nothing prevents the drone from descending.**

## Q2 — PERCEPTION: loss is a **consequence**, not the driver

| run | last g2 fix t | src | death t | **gap (fix→death) s** | up_off@fix | lateral@fix | alt drift fix→death |
|---|---|---|---|---|---|---|---|
| r1 | 9.068 | slot0 | 9.068 | 0.000 (never loses it) | +7.48 | +6.27 | +0.00 |
| r2 | 5.865 | slot0 | 5.997 | 0.132 | +2.39 | +1.31 | −0.22 |
| r4 | 6.164 | slot0 | 6.324 | 0.160 | +0.82 | +0.99 | −0.12 |
| r5 | 6.295 | slot0 | 6.413 | 0.118 | +0.85 | +1.02 | −0.11 |
| champ | 5.779 | slot0 | 10.822 | 5.043 (lived on) | +1.23 | +1.02 | −1.36 |

g2 up-offset trend approaching (last-6 slot1 fixes; `>0`=g2 above nose): r1 `[7.1,6.6,6.8,6.7,6.0,4.7]`,
r2 `[7.6,7.7,7.5,7.7,8.1,8.1]`, r4 `[6.9,6.9,6.1,5.4,3.4,2.5]`, r5 `[7.3,7.1,6.9,6.7,5.8,4.9]`,
champ `[3.7,2.9,2.2,1.7,1.1,1.3]`.

- Crashers **die ≤0.16 s (one det_hold) after the last fresh g2 fix**, with g2 just-seen and
  dead-centered/close (r4/r5 up_off +0.8, lateral +1.0, area 1.0) — they die essentially **AT** g2,
  not from losing it and drifting. **No "drift up" death** (post-fix altitude drift is ±0.1–0.2 m).
- **g2 does NOT drop out the bottom**: up-offset stays **positive** (g2 above/level) for everyone.
  champ converges g2 smoothly to centre (3.7→1.1); r2 STAYS HIGH (7.6→8.1) with area collapsing
  (g2 exits high/edge, tied to its −38° nose-down camera-down attitude, Q4); r1 keeps g2 **+5.8 m
  above** (never lines up). Perception is downstream of geometry/attitude, not the cause.

## Q3 — STATE AT G1-EXIT: settled arrive **SLOWER + LOWER** than champion (not higher)

| run | speed \|v\| | v_horiz | alt(−kfD) | v_vert(dn+) | leg alt[min..max] | leg speed[min..max] |
|---|---|---|---|---|---|---|
| r1 | 7.02 | 6.79 | 2.63 | +1.28 | [−0.19..+2.63] | **[7.0..20.9]** |
| r2 | 7.92 | 7.50 | 2.57 | +0.93 | [+2.11..+3.42] | [7.9..9.1] |
| r4 | 6.60 | 6.45 | 1.43 | +9.23* | [+0.50..+1.43] | [6.6..8.2] |
| r5 | 6.41 | 6.28 | 3.70 | +0.98 | [+1.94..+3.70] | [6.4..9.1] |
| **champ** | **8.98** | **8.55** | **5.67** | −1.21 | **[+4.88..+6.24]** | [8.5..9.4] |

champion kf-altitude per pass: g0 +4.69 → **g1 +5.67 → g2 +4.77** → g3 +2.76 → g4 −0.10 (a gentle,
steepening descent; g2 is ~0.9 m below g1). Champion holds ~9 m/s / ~5–6 m through the leg. (*r4's
+9.23 is a transient kf edge; the leg alt-drop of 0.93 m is the robust descent measure.)

- **The settled population is SLOWER (6.4–7.9 vs 9.0 m/s) and LOWER (kf-alt 1.4–3.7 vs 5.7 m)** at the
  descend entry — the opposite of "arrives too high." (kf-alt is weakly observed / no baro, but the
  independent detector up-offset agrees: g2 sits above the settled runs, at eye-level for champ.)
- **r1 is a velocity RUNAWAY**: it dives from alt 2.63 → ~0 (min −0.19) and accelerates **7→20.9 m/s**
  while g2 stays +5.8 m (up to +11.8 m) ABOVE it — it flies low & fast **under** g2, never lining up.
  🚩 This **nuances REPORT4's "too high" pilot call**: the instrument reads r1 as ending too LOW and
  too FAST. Telemetry=defendant — recommend video adjudication (kf-vertical is the weak axis).

## Q4 — COMMAND vs ACHIEVED: pitch frozen (shared) + **3/4 crashers don't command the descend**

| run | pitch(deg) | %fence_on | a_pitch mean | %nose-down floored | %thr<hover | leg alt drop(m) |
|---|---|---|---|---|---|---|
| r1 | −29.9 | 1.00 | +0.02 | 0.23 | 0.13 | +1.31 |
| r2 | −38.5 | 1.00 | −0.09 | 0.75 | 0.75 | −0.62 (climbs) |
| r4 | −19.9 | 1.00 | −0.62 | 1.00 | 0.12 | +0.93 |
| r5 | −20.0 | 1.00 | −0.49 | 1.00 | 0.12 | +1.76 |
| champ | −29.7 | 1.00 | −0.35 | 0.89 | 0.67 | +0.79 |

- **Pitch fence is 100% active on every leg** (flying pitch −20…−38° ≫ the 1.0° cap) and floors the
  policy's nose-down command 89–100% of ticks — the pitch axis is effectively **open-loop/frozen**.
  This is **SHARED with the champion**, so it is not the pass/fail differentiator, but it removes the
  "dive" tool ⇒ descent must be thrust-only.
- **Command heterogeneity is the real story**: r1/r4/r5 hold thrust **HIGH** (median 1.17–1.50 g, only
  12–13% below hover) — they **do not command the descend** and stay powered/level; r2 over-cuts
  (median 0.46 g, 75% below hover) + pitches −38°; champ is moderate (0.73 g, 67% below). The settled
  runs **scatter** — none reproduces champion's smooth throttle-down. (r2 even *climbs* 0.62 m,
  attitude-driven.)
- 🚩 **Pitch-sign correction:** code is authoritative — `rate_frd[1]=+a_pitch`, and the fence floors
  `rate_frd[1]≥0` to "allow nose-UP recovery" ⇒ **`a_pitch<0` = nose-DOWN** (empirical
  corr(rate_frd1, d·pitch/dt)=+0.37 agrees; champ physically pitches −30° nose-down to fly forward).
  This **contradicts the MEMORY flag "a_pitch=−0.10→nose-up"** — the deployed pipeline reads it the
  other way. (Conclusions unaffected: the fence blocks `a_pitch<0` regardless of label.)
- 🚩 **`--ego-pitch-clamp 1.0°` is ~30× tighter than the code docstring's own recommendation (~30°,
  "set BELOW the −17.8° resting tilt so hover is never fenced").** Champion used it too, so it is not
  the differentiator — but it is the cheapest dive-authority lever (below).

## Q5 — VERDICT: ranked mechanisms + what the arrestor fixes + cheapest levers

| rank | mechanism | evidence | strength |
|---|---|---|---|
| 1 | **(d) speed / geometry / chaotic divergence** | champ uniquely fast (9 vs 6.4–7.9) + high (5.7 vs 1.4–3.7) + committed; settled slower + (OFFSET-FIT) +2 m right + lower, then diverge (OFFSET-FIT e-fold 0.9 s) into varied bad states — r1 dive+runaway(20.9 m/s), r2 over-cut, r4/r5 under-descend; none reproduces the champion line | **STRONGEST** |
| 2 | **(c) policy doesn't command enough descent** | r1/r4/r5 hold ~1.2–1.5 g (12–13% below hover), don't drop onto g2; pitch-fence freeze (shared) removes the dive tool so r4/r5, using neither, can't descend | **SUPPORTED (3/4)** |
| 3 | **(b) perception loss below FOV** | crashers die ≤0.16 s after last g2 fix (no drift time); g2 stays *above*, not out the bottom; r1 never loses it; r2 area-collapse is downstream of −38° pitch | **WEAK / consequence** |
| 4 | **(a) collective floor binds** | assistF=floorF=%bind=0.00; takeoff-only (handover ≤0.12 s), EgoFloorClamp OFF | **REFUTED** |

**Arrestor (post-g1 stare-brake, collective [0.7,1.4] g, level-out) — would / would not fix:**
- **(d) — FIXES (highest value):** brake+level+face-g2 collapses the chaotic divergence and caps r1's
  20.9 m/s runaway, resetting the descend to a champion-like, in-distribution calm start. Matches
  OFFSET-FIT's independent "arrestor + <3 s segment tapes" conclusion.
- **(c) — PARTIAL:** the 0.7 g floor *allows* descent and leveling cuts the forward-pitch sink, but a
  brake-and-hold does not itself descend onto g2 — the policy still executes the drop, but now from an
  in-distribution state (better).
- **(b) — incidental:** a stable, gate-facing state aids re-acquisition; perception wasn't the killer.
- **Does NOT fix:** the pitch-fence freeze (a recipe knob, not the arrestor), nor the policy's
  descend-command gap itself (it only resets the state the policy acts from).

**Single cheapest lever per mechanism:**
- **(d):** the arrestor at g1 (planned) + upstream OFFSET-FIT launch-pose settle-match to land the g1
  arrival on champion's line; a *generous* ~10 m/s speed cap (above champion's 9) bounds r1's runaway
  without penalizing the champion line.
- **(c):** raise `--ego-pitch-clamp 1.0→~30°` — a one-number recipe change, no retrain — to restore the
  nose-down/dive authority the policy is asking for (a_pitch −0.3…−0.6, 89–100% floored); secondary =
  the arrestor to reset to an in-distribution descend state.
- **(b):** none needed (det_hold 0.2 / stale 0.5 fine; slot1 g2 tracking works) — arrestor covers it.
- **(a):** none (refuted) — keep `--ego-floor-clamp 0`.

---

## MEMORY-DELTA (≤12 lines)

1. **G2 descend-wall forensic (P0.3, N=5+champ) → SSOT `handoff/ratchet-p0-2026-07-17/G2-FORENSICS.md`
   + `g2_forensics.py`.** Wall is NOT mechanical and NOT perception.
2. **Q1 FLOOR = REFUTED:** on the g1→g2 leg assistF=floorF=0.00, emitted thrust never exceeds
   policy-desired (%bind=0.00); takeoff-assist 1.3 g floor hands over PERMANENTLY at t≤0.12 s (~5.3 s
   before the leg); `EgoFloorClamp` OFF (no `--ego-floor-clamp`). Policy drives collective to ~0.02
   when it wants — descent-via-thrust fully available/used (champ 67%, r2 75% of leg below hover).
3. **Q2 PERCEPTION = consequence:** crashers die ≤0.16 s (one det_hold) after last fresh g2 fix, g2
   dead-centred (up_off +0.8, area 1.0); g2 never drops out the bottom (up_off stays +); no drift-up.
4. **Q3:** settled arrive SLOWER (6.4–7.9 vs 9.0 m/s) + LOWER (kf-alt 1.4–3.7 vs 5.7 m), not higher.
   r1 "stall" is really a **dive to alt~0 + velocity RUNAWAY 7→20.9 m/s** with g2 +5.8 m ABOVE it —
   🚩 nuances REPORT4's "too high"; kf-vertical weak → recommend video adjudication.
5. **Q4:** pitch fence 100% active (fly −20…−38° ≫ 1.0° cap), floors 89–100% of nose-down cmds — pitch
   frozen, SHARED w/ champ (not the differentiator). 3/4 crashers hold thrust HIGH (1.2–1.5 g, 12–13%
   below hover) = don't command the descend; runs SCATTER (no champion-line reproduction).
6. 🚩 **Pitch-sign correction (code-authoritative):** `rate_frd[1]=+a_pitch`; fence allows nose-UP ⇒
   **a_pitch<0 = nose-DOWN** — CONTRADICTS the MEMORY flag "a_pitch=−0.10→nose-up." Fix the note.
7. 🚩 **`--ego-pitch-clamp 1.0°` is ~30× tighter than the code's own ~30° recommendation** (docstring:
   set below the −17.8° rest tilt). Cheapest (c)-lever: raise to ~30° (no retrain) to restore dive.
8. **Q5 rank:** (d) launch-line/chaotic-divergence STRONGEST > (c) insufficient descend (3/4) > (b)
   perception (consequence) > (a) floor REFUTED. **Arrestor fixes (d) fully** (brake+level+face-g2
   resets to in-distribution calm, caps r1 runaway; matches OFFSET-FIT), **partial on (c)**, does NOT
   fix the pitch-fence freeze or the policy descend-gap.
