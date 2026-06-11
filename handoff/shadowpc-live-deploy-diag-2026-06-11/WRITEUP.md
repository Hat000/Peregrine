# SHADOWPC-LIVE-DEPLOY-DIAG — Root Cause of the RL Transfer Failure + Re-flight

**Session:** SHADOWPC-LIVE-DEPLOY-DIAG  
**Date:** 2026-06-11  
**Input failure:** `handoff/shadowpc-inc5-live-2026-06-11/WRITEUP.md` — inc5 0/10, inc4 0/10,
identical signature (collective=0.000 from step 0, yaw pinned ±3.14), while
`offline_rollout --plant aero` finishes 6/6 from the exact handoff state.

---

## TL;DR

**H1 (garbage live obs) and H2 (action-path mangling) are both FALSE — the live obs/action
pipeline is byte-for-byte correct.** The "all-rail" outputs are the policy's normal learned
bang-bang behavior (it flies the twin the same way). The root cause is an **unmodeled
actuator coupling in the sim's motor mixer**, measured live in isolation:

> At near-zero commanded collective, large body-rate demands force motor-pair
> differentials; the mixer clips the low pair at idle (~0.05) and the **mean thrust rises
> to ≈ the differential amplitude — up to ~0.4 collective ≈ hover lift (2 g transient)
> that was never commanded.** Symmetrically, at full-rail collective (≈1.0) there is no
> differential headroom and **rate authority collapses during every thrust pulse.**

The policy lives at both rails (duty-cycled 0↔1 collective + per-tick ±3.14 yaw dither),
so the twin-trained strategy lands exactly in the two corners the twin models wrong
(thrust and rates are independent in `rl_plant`). H3 ("sim treats collective 0 as motor
cutoff") is also false — collective 0 is honored faithfully when rate demands are small.

**Deploy-side mitigation shipped:** `--yaw-scale 0` (drop the yaw command — net-zero
motion in-twin, the dominant parasitic-floor source live). Result: from **0 clean gate
passes in 20 flights** yesterday to **14 clean gate passes in 10 flights** (gate 0: 7/7 of
RL-engaged flights clean; median 2 gates). Lap completion is still 0/10: the residual
killers (rate-authority holes during thrust pulses + 67 ms measured latency) are
train-side problems. **inc6 must train against a mixer-aware plant.**

**inc4 probe verdict: the corrected-aero twin is RIGHT about inc4.** Under the identical
mitigation, inc4 slams sustained full collective and rockets 3.1 m above gate 0 in 0.37 s,
3/3 flights — the falsified-map 2× thrust error, live. inc5 remains the only transfer
candidate.

---

## 1. Forensic method (no blind fixes)

New permanent tool: **`rl/replay_obs.py`** — re-parses a recorded session's `mavlink.tlog`,
replicates fly_rl's 30 Hz loop on the recorded telemetry through the *same* `build_obs` +
`policy_step` code objects, and dumps per-tick JSONL (telemetry, labeled 17-dim obs, actor
mean/tanh/rescale, wire command), plus velocity self-consistency columns and an offline
aero-twin rollout from the same handoff state for side-by-side diff.

Run on `20260611_184326_rl_inc5_live_f1` (f5 and inc4 f1 cross-checked):

### 1a. Step-0 obs: live == offline exactly

Live-reconstructed tick-0 obs `pos_g=[+2.97,−0.04,+0.07] vel_g=[+5.07,−0.04,+0.34]
rpy_g=[0,+0.02,+3.14] w=[−0.01,+0.44,+0.01] o12=0` → action `rate_frd=[+1.43,−2.20,+3.14]
thr=0.000` — element-for-element the offline twin's step-0 row, and identical to the
flight log's printed first action. Obs builder, sidecar rescale, virtual flip, action
pipeline: all correct live.

### 1b. The policy is a bang-bang controller BY DESIGN

Offline from the handoff state the policy flies gate 0 with collective 0.000 on most ticks
and **full-rail pulses (thr = 1.000, normed 3.765) every ~3 ticks** (obs[12] flips
0 → 3.76 → 0), while the **yaw command dithers ±3.14 every tick** (realized net yaw ≈ 0).
Yesterday's "saturation signature" is the policy's normal operating mode.

### 1c. Live divergence is PHYSICS, not telemetry

From tick 1 the live state diverges from the twin **opposite to free fall**: climbing
+3.2 m/s and accelerating 5.1 → 9.8 m/s by tick 12, at commanded collective 0.000 — net
specific force ≈ 2 g, uncommanded. The twin (same actions) free-falls and the policy
pulses thrust to catch itself; live it sees "climbing" and never pulses. The drone lofts
~1 m above gate-0 center and strikes the TOP frame band at tick ~13.

Self-consistency: `velocity_ned` (the client's raw-quat rotation of the ODOMETRY body
twist) matches the position finite-difference to 0.1–0.9 m/s throughout, including
tumbles — pos/vel/attitude/rate channels are mutually consistent. No NaNs, no staleness
pre-crash (odo age 0–14 ms). (Side-validation: the artifact-undone rotation `R(−roll,p,y)`
does NOT match d(pos)/dt — the ODOMETRY twist lives in the *reported*-quat body frame, so
`mavlink_client`'s existing rotation is the right one. No action needed.)

### 1d. Motor outputs during yesterday's failures

`ACTUATOR_OUTPUT_STATUS`, commanded collective = 0.000 the whole RL phase:

| flight | phase | motors mean | motors max |
|---|---|---|---|
| inc5 f1 | bridge (last 1.5 s, cmd ~0.3–0.5) | 0.377 | 0.56 |
| inc5 f1 | RL 0–0.5 s (cmd 0.000) | **0.42–0.56** | 0.85 |
| inc5 f5 | RL 1.5–4 s (cmd 0.000) | 0.37 | 0.88 |
| inc5 f9 | RL 1.5–4 s (cmd 0.000) | **[0.3, 0.2, 1.0, 0.3] locked** | 1.0 |

f9's locked pattern (motor 3 pinned at 1.0 for minutes) is the "frozen at gate-0,
spinning" state: a sustained max yaw demand against the mixer.

---

## 2. The mixer law, measured live in isolation

`mixer_probe.json` (rate_sysid profile mode; recording `20260611_194826_mixer_probe`):
open-loop (thrust × body-rate) corners with re-level catches between.

| phase | cmd thr | cmd rate (rad/s) | motors mean | motors max | a_up (m/s²) | verdict |
|---|---|---|---|---|---|---|
| z00_r0 | 0.000 | 0 | **0.064** | 0.26 | **0.64** | free fall — collective 0 honored |
| z00_y12 | 0.000 | yaw 1.2 | 0.126 | 0.54 | 1.73 | ≈ free fall (twin map a_up(0.126) ≈ 1.1 — consistent) |
| z00_y31 | 0.000 | yaw 3.14 | **0.258 (0.40 early)** | 0.77 | **9.36** | ≈ HOVER LIFT at commanded ZERO |
| zhov_y31 | 0.266 | yaw 3.14 | 0.287 | 0.76 | 12.1 | mean preserved when headroom exists |
| z00_rp15 | 0.000 | roll+pitch 1.5 | 0.243 | 0.64 | (tumbles) | parasitic ≈ hover, flips the drone |
| (c100, twin-falsify) | 1.000 | ~0 | 0.874 | 1.0 | 58.9 | only ~0.13 headroom at the top rail |

Motor pattern at the yaw rail, zero collective: `[0.08, 0.73, 0.73, 0.08]`.
**Law: motors = clip(collective ± rate-PID differentials, idle ≈ 0.05, 1.0); when
commanded collective < differential demand, mean thrust ≈ differential amplitude; when
collective ≈ 1, differentials (rate authority) ≈ 0.** Yaw is the most expensive axis
(weakest authority: slew 80 vs 260 rad/s², sim super-gain ~3.2× at the rail → realized
target ~10 rad/s), and the policy's per-tick sign-flipping dither keeps the yaw PID error
permanently maxed.

The 2026-06-10/11 characterization grid measured (thr sweep × rates≈0) and (thr=hover ×
one-axis 3.14) — **never (thr≈0 × |rate| large), the corner the policy occupies.** That is
why the twin is faithful everywhere it was fit and wrong exactly where the policy operates.
Why both checkpoints failed identically: rail-riding is a property of tanh-squashed PPO at
the bounds, shared by inc4/inc5 — the failure is action-space × mixer, not physics
knowledge.

---

## 3. Mitigation search (twin-validated before flying)

`offline_rollout --plant aero --start handoff --handoff-speed 5.1`:

| config (inc5) | lat 0 | lat 1 | lat 2 | lat 3 |
|---|---|---|---|---|
| baseline | 6/6 8.23 s | 6/6 8.69 s | — | — |
| max-rate 1.2 | 6/6 8.29 s | 6/6 8.13 s | 6/6 8.33 s | COLLISION @g3 |
| **yaw-scale 0** | 6/6 8.89 s | 6/6 8.89 s | **6/6 8.76 s** | MISS @g2 |
| yaw-scale 0 + max-rate 1.2 | 6/6 | COLLISION @g1 | COLLISION @g1 | — |
| yaw-scale 0 + max-thrust 3.0 | 6/6 | 6/6 | COLLISION @g2 | COLLISION @g2 |
| yaw0 + obs-extrapolation | — | — | (lat2+ex2 6/6) | lat3+ex3 1 gate — REJECTED |

Lessons: mitigations do NOT stack (the policy needs full roll/pitch authority once yaw is
gone, and needs its full thrust pulses always); first-order obs extrapolation is garbage at
8–11 rad/s body rates. **Final live config: `--yaw-scale 0` alone.**

`--max-rate 1.2` was flown first (4 flights, `rl_inc5_mr12_f1–f4`: 3× CRASH g0, 1× g0
PASS+CONTACT) and abandoned on evidence: the capped yaw dither (±1.2) still never
converges → residual ~1 g parasitic climb → same top-frame loft, slower.

## 4. Live latency, measured

Cross-correlating per-step commanded `rate_frd` (debug_obs.jsonl) against realized
ODOMETRY rates (flight ys0_f1): **best lag = 2 ticks (67 ms), corr 0.86 roll / 0.77
pitch** — effective command→response latency including the ~19 ms rate-loop constant,
transport, telemetry age, and 30 Hz ZOH. The twin at lat 2 with yaw-zero finishes 6/6, so
latency alone does not explain the residual live gap — the §2 top-rail authority holes do.

## 5. Harness hardening (committed, fly_rl.py)

* **`--debug-obs` (default ON):** per-step JSONL — telemetry, labeled obs, actor
  mean/tanh/rescale, wire command, odo staleness → `<session>/debug_obs.jsonl`.
* **Live-reset guard:** sim-autoreset detection (reset_counter bump, race_start change,
  active_gate_index drop, >10 m teleport) → commands CUT immediately (`SIM_RESET`).
  Yesterday fly_rl commanded through autoresets → minutes of throttle-latched spinning.
* **Spin guard:** |body rate| > 6 rad/s sustained 2 s with no gate progress → `SPIN_ABORT`.
* **`--full-reset` (default ON):** between flights ESC+Down×3+Enter → HOME → Enter×2 →
  fresh countdown (no 31000 residue); `wait_fresh_go` escalates to the home-kick after 3
  ineffective 31000s.
* **Verified-foreground window control:** the fullscreen sim AUTO-MINIMIZES on focus loss
  (discovered mid-batch: every later keybd_event went to the wrong window → NO_GO chain).
  All key sends now SW_RESTORE + ALT-wrapped SetForegroundWindow + verify
  GetForegroundWindow, else fall back to 31000.
* `--yaw-scale` (mitigation knob, also in offline_rollout for twin parity).

Zero SIM_RESET/SPIN incidents in 17 hardened flights; every crash ended in a clean abort,
disarm, and an auto-chained fresh race.

## 6. Re-flight: inc5 ×10, `--yaw-scale 0`

Command: `fly_rl.py --checkpoint rl/checkpoints/stage1_inc5_actor.pth --yaw-scale 0
--flights ...` (batches `rl_inc5_ys0` ×6 + `rl_inc5_ys0b` ×4; the batch split was the
window-minimize discovery, §5).

| # | session | fly_rl | race_outcome gates (clean/contact) | failure |
|---|---------|--------|-------------------------------------|---------|
| 1 | `195907_…ys0_f1` | CRASH | **2 (2/0)** | gate-2 frame, dy −1.7 m |
| 2 | `195938_…ys0_f2` | CRASH | **2 (2/0)** | gate-2 frame, dy −1.0 m |
| 3 | `195959_…ys0_f3` | CRASH | 0 | bridge-seam contact; 0 RL steps |
| 4 | `200016_…ys0_f4` | CRASH | **2 (2/0)** | terrain after gate 1 |
| 5 | `200107_…ys0_f5` | CRASH | 0 | bridge-seam contact; 0 RL steps (139 env coll) |
| 6 | `200123_…ys0_f6` | CRASH | **1 (1/0)** | gate-1 frame, near-center clip |
| 7 | `200733_…ys0b_f1` | CRASH | **2 (2/0)** | gate-2 frame, dy −1.3 m |
| 8 | `200804_…ys0b_f2` | CRASH | 0 | bridge-seam contact; 0 RL steps |
| 9 | `200821_…ys0b_f3` | CRASH | **1 (1/0)** | gate-1 frame, near-center clip |
| 10 | `200839_…ys0b_f4` | CRASH | **2 (2/0)** | gate-2 frame, dy −1.8 m |

**Distribution: 14 clean gate passes / 10 flights (yesterday: 0 clean / 20). Of the 7
flights where the RL actually engaged (3 were bridge-contact aborts at the seam, caught at
RL step 0 by the collision guard): gate 0 = 7/7 clean, gate 1 = 5/7, gate 2 = 0/7.
Finishes 0/10 (twin@lat2 predicted 6/6, ~8.8 s).**

The gate-2 failure is uniform and diagnostic: **vertical error ≈ 0 (dz ±0.3 m), lateral
dy = −1.0…−1.8 m short, mid-bank** — the +3.7 m lateral swing toward gate 2 arrives late.
Vertical control transferred (the parasitic floor is gone); lateral control is starved by
the §2 top-rail authority holes (each ~100 ms thrust pulse = no turning) compounded by the
67 ms latency. Both are twin-blind-spots → train-side.

## 7. inc4 twin-validation probe ×3, same config

Twin predictions (ys0, corrected-aero plant): lat 0 → 6/6; lat 1–2 → collide after 3–4
gates. The alternative world ("aero twin too pessimistic; inc4 threads ~7 s") predicted
clean laps.

| # | session | result | race_outcome | RL steps | mechanism |
|---|---------|--------|--------------|----------|-----------|
| 1 | `200914_rl_inc4_ys0_f1` | CRASH | g0 PASS+CONTACT | 11 | thr 0,0,0 then SUSTAINED 1.0; ends 3.1 m ABOVE gate 0 |
| 2 | `200942_rl_inc4_ys0_f2` | CRASH | 0 gates | 11 | identical |
| 3 | `200959_rl_inc4_ys0_f3` | CRASH | 0 gates | 11 | identical |

**Verdict: the corrected-aero world is the real one.** inc4 (trained on the falsified
~3.77 g linear collective map) commands sustained full collective expecting ~half the
authority the real ~8 g map delivers → rockets 3 m high in 0.37 s, 3/3 deterministic. It
is strictly worse than inc5 live (gate 0: 0/3 clean vs 7/7 clean), in the direction and
for the mechanism the corrected twin predicts. The twin's absolute numbers were optimistic
for inc4 (its lat-1/2 runs reach gates 3–4) because the saturation holes punish inc4's
even-heavier rail-riding hardest — but the discrimination inc5 ≫ inc4 on corrected aero is
**confirmed live**. inc4 is retired as a transfer candidate.

## 8. Recommendations (inc6)

1. **Train against a mixer-aware plant.** Minimal faithful model in `rl_plant.step`:
   compute the per-motor demands `m_i = thrust_cmd ± d_roll ± d_pitch ± d_yaw` from the
   rate-loop torque demands, clip to [0.05, 1.0], then realized collective = mean(m_i) and
   realized torques from the clipped differentials. Fit the differential gains from
   `20260611_194826_mixer_probe` (+ the rspd_*314 runs). This kills both rails' artifacts
   in training and lets the policy LEARN pulse-and-turn scheduling.
2. **Action regularization:** the per-tick yaw rail dither buys nothing in-twin and is the
   single largest live killer; an action-rate penalty (or yaw exclusion) removes it for free.
3. **Latency in training:** measured live = 2 control steps (67 ms); train with
   transport_delay ≥ 2 (inc5 tolerates lat 2 in-twin only with yaw-zero — make it native).
4. **Characterization:** future plant fits must cover (low-thrust × high-rate) and
   (full-thrust × rate) corners; one mixer_probe flight covers both.
5. Keep `--yaw-scale 0` for any live flight of rail-riding checkpoints; do NOT stack caps.
6. The bridge clips the gate-0 approach in ~3/13 fresh races (seam contacts with 0 RL
   steps). Consider widening the handoff lateral tolerance or a slightly earlier seam.

## 9. Recording index

* Mixer probe: `data/runs/20260611_194826_mixer_probe` (+ `commands.jsonl`)
* mr12 (abandoned): `20260611_1950..52_rl_inc5_mr12_f1–f4`
* inc5 ys0: `20260611_1959..2001_rl_inc5_ys0_f1–f6`, `20260611_2007..2008_rl_inc5_ys0b_f1–f4`
* inc4 ys0: `20260611_2009..2010_rl_inc4_ys0_f1–f3`
* Per-step dumps: `<session>/debug_obs.jsonl` (all hardened flights)
* Replay forensics on yesterday's failures: `<old session>/replay_obs.jsonl` (+ `_offline.jsonl`)

Sim exited to HOME at session end (verified by screenshot).
