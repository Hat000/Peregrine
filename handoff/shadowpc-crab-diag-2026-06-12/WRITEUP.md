# SHADOWPC-CRAB-DIAG — the "crab" is inc6's trained posture, not a deploy bug

**Session:** SHADOWPC-CRAB-DIAG (2026-06-12). **Model:** opus-4.8.
**Repo:** pulled to `d94899e` (fast-forward over 421bb15). **Checkpoint:** `rl/checkpoints/stage1_inc6_actor.pth`.
**Scripts:** `handoff/shadowpc-crab-diag-2026-06-12/scripts/{crab_live_extract,crab_twin_rollout,rate_track,traj_compare}.py`.
**Branch outcome: 2b** (twin reproduces the crab → no live-only deploy fix). **No flights flown; no jobs submitted.**

---

## TL;DR

1. **The crab is REAL but it is NOT a bug.** Live cruise holds +38–43° roll, +128–131° heading, ~+54° sideslip (operator's read confirmed to the degree). **The offline twin holds the IDENTICAL posture** (roll +41.5°, yaw +130.9°, sideslip +51°) and **finishes at 9.62 s.** It is inc6's genuine trained flight style — a ~55° high-tilt racing cruise present in **training, twin, and live alike**. The commander's prime suspect (an observation-chain rotation error from the `Rz(π)` virtual-flip ∘ `R_y(π)` conjugation) is **falsified**: the twin has no telemetry chain and produces the same posture; step-0 obs is identical to <0.003.
2. **The "~70% slowdown" DISSOLVES — it was a clock artifact.** Live bridge "16.24 s" is the RACE_STATUS clock, which **includes the ~8 s CTBR bridge-launch before RL recording starts.** The actual RL segment (gate-0→gate-5) is **8.96 s at 17 m/s median**, matching the twin handoff's 8.16 s segment-by-segment (≤0.15 s/segment).
3. **The "0.85 rate-gain droop" DISSOLVES.** That number was the *quat-FD-vs-`w_raw` telemetry-consistency* canary, not command tracking. Realized/commanded rate gain is **~2.5 (the super-rate map), identical live vs twin.** Rate tracking is faithful.
4. **The one genuine residual is the standing-start gate-3 clip.** Live standing tracks the twin within 1–2 m the whole way, then arrives at the gate-3 plane **~0.5 m high and ~0.5 m short**, clipping the upper frame; the twin threads it (vert-off 0.00). The divergence lives almost entirely in **N (along-course) and D (vertical); lateral E is tracked to 0.04 m** → a small **thrust/drag plant-model residual on the high-collective gate-2→3 climb, NOT a frame issue.** Confirmed by `frame_residual_report`: systematic +2.0–2.8 m/s² N/D residual in the 12–18 m/s banked-climb bin, mirror canary clean (+0.97).
5. **Both bridge finishes are zero-contact** (`n_coll`=0 across all ticks). Bridge laps are a legitimate zero-contact submission posture.

---

## 1. Discriminator — twin reproduces the crab (cruise, speed > 3 m/s)

Reconstruction: TRUE physical attitude = euler of conjugate(`q_raw`) (FRAME-AUDIT `R_y(π)`); course from pristine `vel_ned` (external invariant); `tilt` = angle of thrust axis from vertical (the quantity `rw_tilt` penalizes, yaw-invariant).

| source | roll | pitch | yaw (hdg) | **tilt** | sideslip(vs vel) | outcome |
|---|---|---|---|---|---|---|
| **Live std_f1** | +42.6 | −38.3 | +129.0 | ~53 | +54.1 | gate-3 clip |
| **Twin racestart** (flip) | +41.5 | −34.0 | +130.9 | **+54.8** | +51.2 | **FINISH 9.62 s** |
| **Live brg_f1** | +38.7 | −35.2 | +130.9 | ~52 | +51.4 | finish (8.96 s RL) |
| **Twin handoff** (flip) | +39.5 | −34.3 | +134.7 | ~54 | +47.4 | **FINISH 8.16 s** |

The twin matches live on **every attitude axis within a few degrees.** Step-0 obs identity (the task's explicit check): for the live std_f1 spawn pose, twin `simstart` step-0 obs = `[23.30, −0.40, 1.41 … rpy_g(0, −0.31, +3.14)]` vs live `[23.2975, −0.40, 1.4125 … rpy_g(0, −0.31, −3.14)]` — match <0.003, yaw ±π identical. `check_build_obs` = **0.00e+00**. The deploy obs chain is byte-faithful.

### Why the crab is trained posture, not the virtual flip

Control experiment — `trainreset` + **`--no-virtual-flip`** (the exact training-native regime where inc6 scored 0.97 under `rw_tilt=96`):

| regime | roll | pitch | yaw | **tilt** | outcome |
|---|---|---|---|---|---|
| **trainreset, NO flip** (training-native) | −40.6 | +37.0 | −51.6 | **+55.5** | FINISH 9.12 s |
| racestart, flip (deploy default) | +41.5 | −34.0 | +130.9 | +54.8 | FINISH 9.62 s |
| racestart, NO flip (mismatched) | −61.6 | +42.7 | +124.4 | +69.0 | OOB@gate0 |

inc6 flies at **~55° tilt in its own native training frame.** The virtual flip is *correct* for the deploy start (removing it sends racestart OOB at 69° tilt); it re-expresses the same ~55°-tilt flight as +40° roll / +130° heading in NED. The "crab posture" and the "yaw bias that never corrects" are this trained high-tilt racing style, faithfully transferred. `rw_tilt=96` did not produce a low-tilt policy — the speed/progress optimum sits at ~55° tilt.

### Non-commutation note (no escape hatch needed)
The commander flagged that `Rz(π)` (virtual flip) and `R_y(π)` (telemetry conjugation) don't commute. They never compose into a heading offset: the conjugation fully resolves telemetry→true attitude (a fixed sim property) **before** the flip is applied, and the flip then acts on the true attitude **identically** in twin (`obs_from_truth`) and deploy (`build_obs`). Empirical proof of non-interference: step-0 obs identical to <0.003, full-trajectory attitude match within degrees, `frame_residual` mirror canary +0.97 on all post-fix flights. The composition is verified correct — it did not resist proof.

---

## 2. The slowdown dissolves (clock artifact)

| | RL-segment dur | median speed | RACE_STATUS clock |
|---|---|---|---|
| Live brg_f1 | **8.96 s** (269 ticks @30 Hz) | 17.2 m/s | 16.24 s (incl. ~8 s CTBR launch) |
| Live brg_f2 | 8.90 s | — | 17.74 s |
| Twin handoff | 8.16 s | ~18–20 m/s | — |

Per-segment (gate-pass times, s from RL t0): live brg_f1 `{g0:0.68, g1:2.42, g2:4.17, g3:6.41, g4:7.67, fin:8.96}` vs twin handoff `{g0:0.30, g1:2.03, g2:3.60, g3:5.69, g4:6.86, g5:8.16}` — segment durations match within **≤0.15 s** (longest leg = gate-2→3 climb, ~2.1–2.2 s in both). Live cruise speed equals the twin's. There is **no transfer slowdown**; the 16.24 s vs 9.5 s comparison mixed the full-race clock (CTBR + RL) against an RL-only time.

## 3. Rate tracking is faithful (no droop)

Realized (`−w_raw`) vs commanded (`rate_frd`), lag-aligned, |cmd|>0.3 rad/s, cruise bin 15–22 m/s:

| axis | live std_f1 | live brg_f1 | live brg_f2 | twin simstart | twin handoff |
|---|---|---|---|---|---|
| roll | 2.52 | 2.54 | 2.58 | 2.61 | 2.60 |
| pitch | 2.48 | 2.48 | 2.47 | 2.60 | 2.60 |

The gain is the **super-rate map** (stick→rate expo), **identical live vs twin** — the mixer-throttling Q is already in the twin plant and matches. The "0.85" was `frame_residual`'s `quat-FD(true) ~ −w_raw` telemetry-consistency canary (0.86–0.99), a different quantity. No airspeed-dependent command-tracking droop exists.

---

## 4. The genuine residual — standing gate-3 clip = thrust/drag plant gap on the climb

Closed-loop **twin `simstart` vs live std_f1**, tick-aligned (identical start, shared obs/action code → divergence = pure plant gap):

```
 k    t    live_ned                twin_ned               |dpos| dN    dE    dD
 60  2.03  [-24.9, -0.2, -1.1]     [-25.3, -0.2, -0.9]     0.4   -0.4 +0.0  +0.2
120  4.03  [-59.1, -3.2,  8.1]     [-59.4, -3.0,  8.5]     0.6   -0.3 +0.1  +0.5
180  6.03  [-94.9, -0.0, 16.1]     [-95.9, -0.0, 17.2]     1.5   -1.0 +0.0  +1.1
210  7.03  [-110.6,-5.0, 22.5]     [-112.4,-5.0, 23.4]     2.0   -1.8 -0.0  +0.8
```
Gate-3 centre NED `[-111.5, -5.1, 23.2]`, opening L-inf half-width 0.75 m:
- **live** nearest gate-3 plane: pos `[-111.0, -5.06, 22.72]`, lateral-off(E) **+0.04**, vert-off(D) **−0.49** (0.5 m high), ~0.5 m short → clips upper frame.
- **twin** nearest gate-3 plane: pos `[-111.8, -4.89, 23.21]`, lateral-off(E) +0.21, vert-off(D) **0.00** → threads.

The divergence is **N + D only; E (the frame/handedness-sensitive axis) tracks to 0.04 m.** It accumulates on the **gate-2→gate-3 climb** (+11 m of altitude; dD grows +0.2→+1.3). `frame_residual_report` localizes the mechanism: a systematic **+2.0–2.8 m/s² model−meas residual in N and D in the 12–18 m/s × tilt-35–90 bin** (mirror canary TRUE +0.97 / AS-IS ≤−0.68 → frame clean). Integrated over the ~2 s climb, ~2 m/s² → the observed ~1–2 m offset. This is the collective-accel / drag model slightly off in the high-collective banked-climb regime — under the audit's "≤2 m/s² = noise" bar per-tick, but **systematic and trajectory-bending** when integrated. The bridge clears gate 3 (live brg_f1 passes at `[-108.5, -4.6, +21.7]`, more vertical margin from the flatter/faster approach) and finishes **zero-contact**.

## 5. Contacts (race validity)

`n_coll` (= count of sim `client.collisions` with threat_level ≥ 2): **0 across all 269 / 267 ticks** for **brg_f1 and brg_f2** → both bridge finishes are zero-contact (caveat: the sim can omit very light clips — `race-outcome-recording-gotchas`; but the collision client read 0 every tick and RACE_STATUS finished). std_f1 `n_coll` is also 0 up to the terminating gate-3 HARD COLLISION (the crash itself is the race-ending event). **Bridge laps qualify as a zero-contact submission posture** at ~9 s RL-segment.

## 6. `frame_residual_report.py --replay 36` (standing post-session check)

```
std_f1 : 12-18/35-90 N+2.79 E+1.57 D+2.36 | 18-40 N+0.02 E+0.90 D+0.88 | mirror TRUE+0.97 AS-IS-0.68 OK | rate[.988,.994,.939] | replay vel-err[1.24,.46,1.10]
brg_f1 :  8-12 N+1.12 E+0.13 D+0.87 | 12-18 N+2.08 E+1.21 D+2.13 | 18-40 N-0.22 E+1.24 D+0.48 | mirror TRUE+0.97 AS-IS-0.75 OK | rate[.855,.956,.842] | replay[-.70,.21,-.10]
brg_f2 :  8-12 N+1.81 E+0.14 D+1.27 | 12-18 N+2.37 E+0.76 D+1.97 | 18-40 N+0.18 E+1.42 D+1.03 | mirror TRUE+0.97 AS-IS-0.75 OK | rate[.857,.981,.851] | replay[-.72,.23,-.10]
```
Frame remains clean on all post-fix flights. The N/D residual structure is the gate-3 mechanism above.

---

## 7. Branch-2b deliverable — measured cause + train-side fix spec (NO jobs this session)

**Measured cause of the only live deficit (standing gate-3 clip):** a systematic ~2.0–2.8 m/s² collective-accel/drag force-model residual in the 12–18 m/s banked-climb regime (frame-clean: E tracked to 0.04 m, mirror canary +0.97), integrating to a ~1–2 m N+D trajectory divergence over the gate-2→3 climb that moves the *standing* approach from the twin's clean thread into the gate-3 upper frame. The crab posture, the bridge "slowdown", and the "rate-gain droop" are **not** deficits — they dissolve (trained style / clock artifact / wrong canary). **No reward change is warranted** (the ~55° tilt is the winning trained style and finishes zero-contact on the bridge).

**Spec for inc7 (laptop S-step; do NOT submit here):**
1. **Plant refit (banked-climb regime):** re-identify `COLL_MAP_ACCEL` / `QUAD_DRAG_C2` against pristine-`vel_ned` FD in the 12–18 m/s × tilt-35–90 × high-collective bin, using the **8 post-fix zero-contact frame-audit recordings + 2 bridge finishes** (all clean-frame, no new flights). Target: drive the N/D median residual from +2–2.8 to <1 m/s².
2. **Robustness DR for gate-3 margin:** retrain with ±12% DR on high-collective `coll_map_accel` (and drag), so the policy carries ~0.5–0.7 m of vertical margin through gate 3 — robust to the residual without depending on an exact refit. Do **not** use `--plant lapse` / `dr_lapse` (VOIDED).
3. **Selection:** ≥3-seed generalization averaging (durable protocol — single-seed gen is volatile, inc6 was 0.741 vs 0.982).
4. **Gate:** bundle the pending **V100 config-matrix gate** at next Adroit contact (mixer/aero/map configs; lapse present but VOIDED — do not select).

**Predicted outcome (on record, for the eventual inc7 confirm):** standing start clears gate 3 with margin; standing finish rate → bridge parity; no posture change (tilt stays ~55°, the trained style).

---

MEMORY-DELTA:
- **CRAB = trained posture, NOT a bug. Branch 2b.** Twin reproduces it exactly (roll +41.5°/yaw +130.9°/tilt +54.8°, FINISH 9.62 s) and the training-native `trainreset` flies +55.5° tilt — inc6's genuine ~55° high-tilt racing cruise, faithful in training/twin/live. **Supersedes** the commander's "twin finishes CENTERED at 9.5 s" premise and the "observation-chain rotation error" hypothesis (falsified: step-0 obs identical <0.003, mirror canary +0.97).
- **"~70% slowdown" DISSOLVED (clock artifact).** Live bridge RL-segment = 8.96 s @17 m/s ≈ twin handoff 8.16 s, per-segment ≤0.15 s. The 16.24 s was RACE_STATUS (incl. ~8 s CTBR launch). **Supersedes** LIVE-CONFIRM fact 5.
- **"0.85 rate-gain droop" DISSOLVED.** Realized/commanded gain = ~2.5 super-rate, identical live vs twin. The 0.85 was the quat-FD-vs-`w_raw` telemetry-consistency canary, not tracking. **Supersedes** LIVE-CONFIRM fact 4 as a "deficit."
- **Standing gate-3 clip = thrust/drag plant residual on the high-collective climb** (~2–2.8 m/s² N/D in 12–18 m/s bin; E tracked to 0.04 m → frame-clean). ~1–2 m divergence puts standing into gate-3 upper frame; bridge clears it. Fix is train-side: refit collective/drag in the climb regime + ±12% DR margin, ≥3-seed, V100 gate. No reward change.
- **Bridge finishes brg_f1/brg_f2 are zero-contact** (`n_coll`=0 all ticks) → legitimate ~9 s RL-segment submission posture.
- **Deploy chain (fly_rl + tools) is FAITHFUL** — do not "fix" the virtual flip or conjugation; verified against twin + external invariant on post-fix data.
