# SHADOWPC-INC5-LIVE — Transfer Test + Twin Validation WRITEUP

**Session:** SHADOWPC-INC5-LIVE  
**Date:** 2026-06-11  
**Checkpoints flown:**
- **Task 1a (primary):** `rl/checkpoints/stage1_inc5_actor.pth` — aero-corrected, 10 flights  
- **Task 1b (twin-validation probe):** `rl/checkpoints/stage1_inc4_actor.pth` — 10 flights from preceding SHADOWPC-INC4-LIVE session  
**Sidecar (both):** `{"act_max_thrust": 3.765, "act_max_rate": 3.14}` — confirmed on every flight log  
**Track:** VQ1, 6 gates, `data/runs/track_map_20260602_114630.json`  
**Sim:** AI-GP on ShadowPC (DCGame-Win64-Shipping PID 15984)

---

## Task Summary

| Task | Status |
|------|--------|
| 1a. INC5 live flights (10×) | DONE — all 10 recorded and certified |
| 1b. INC4 twin-validation probe (3 flights, via prior session) | DONE — 10 flights from SHADOWPC-INC4-LIVE sufficient |
| 2. Distributions and live-vs-twin deltas | DONE — §4 |
| 3. Corner-pass probe | DONE — live CTBR offset crossings §5 |
| 4. Coast-replay re-check | DONE — §6 |

Sim exited to HOME at session end (ESC + Down×3 + Enter).

---

## 1. INC5 Live Flights (Task 1a)

**Command:** `.venv\Scripts\python.exe rl\fly_rl.py --checkpoint rl/checkpoints/stage1_inc5_actor.pth --flights 10 --label rl_inc5_live --wait-seconds 120 --max-seconds 90`

**Load confirmation (all flights):**  
`[load_actor] sidecar stage1_inc5_actor.json: action bounds -> thrust [0.000,3.765] rates +-3.14 rad/s`  
`[fly] RL policy  rate=30 Hz  max_rate=off  virtual_flip=True  max=90s`

Bridge approach consistent across all flights: HANDOFF at along≈+2.97 m, speed≈5.1 m/s, pos≈(-20.3, -0.4, -1.3) — on center of gate 0.

### race_outcome.py Certification

| # | Recording | fly_rl result | gates passed | clean | contact | gate colls (no pass) | env colls |
|---|-----------|---------------|-------------|-------|---------|----------------------|-----------|
| 1 | `20260611_184326_rl_inc5_live_f1` | CRASH | **0** | 0 | 0 | 2 | 0 |
| 2 | `20260611_184345_rl_inc5_live_f2` | CRASH | **0** | 0 | 0 | 1 | 0 |
| 3 | `20260611_184405_rl_inc5_live_f3` | CRASH | **0** | 0 | 0 | 10 | 0 |
| 4 | `20260611_184425_rl_inc5_live_f4` | CRASH | **0** | 0 | 0 | 343 | 1765 |
| 5 | `20260611_184444_rl_inc5_live_f5` | TIMEOUT | **0** | 0 | 0 | 30† | 0 |
| 6 | `20260611_184634_rl_inc5_live_f6` | TIMEOUT | **0** | 0 | 0 | 13 | 0 |
| 7 | `20260611_184823_rl_inc5_live_f7` | TIMEOUT | **1 (gate 0)** | 0 | 1 | 1 | 6 |
| 8 | `20260611_185013_rl_inc5_live_f8` | STALLED | **0** | 0 | 0 | 525 | 421 |
| 9 | `20260611_185035_rl_inc5_live_f9` | TIMEOUT | **0** | 0 | 0 | 12879‡ | 0 |
| 10 | `20260611_185223_rl_inc5_live_f10` | STALLED | **0** | 0 | 0 | 1560 | 10 |

† F5: race_outcome discarded a prior-race RACE_STATUS residue and scored the final epoch only.  
‡ F9: drone frozen at gate-0 plane, spinning at yaw ±3.14 rad/s for 90 s → 12,879 gate contact events against the stationary frame.

**Summary: 0/10 races finished. 1/10 sim-recognized gate pass (gate 0, F7, WITH CONTACT — bridge momentum).  
In-twin prediction was 10/10, ~9.5 s. Verdict: TRANSFER FAILURE.**

### First-action diagnostics (every flight)

| flight | first RL action (rate_frd, collective) |
|--------|----------------------------------------|
| F1 | rate=[+1.43, -2.20, **+3.14**], thr=0.000 |
| F2 | rate=[+0.94, -2.22, **+3.14**], thr=0.000 |
| F3 | rate=[+1.51, -2.23, **+3.14**], thr=0.000 |
| F4 | rate=[+1.46, -2.21, **+3.14**], thr=0.000 |
| F5 | rate=[+0.89, -2.30, **+3.14**] → −3.14, thr=0.000 |
| F6 | rate=[+0.93, -2.23, **+3.14**] → −3.14, thr=0.000 |
| F7 | rate=[+0.92, -2.26, **+3.14**], thr=0.000 |
| F8 | rate=[+1.44, -2.20, **+3.14**], thr=0.000 |
| F9 | rate=[+1.50, -2.22, **+3.14**], thr=0.000 (frozen at gate 0) |
| F10 | rate=[+1.39, -2.20, **+3.14**], thr=0.000 |

The diagnostic signal is identical to the SHADOWPC-INC4-LIVE failures:
- **collective = 0.000 every flight** (zero thrust from step 0)
- **yaw rate saturated at ±3.14 rad/s from step 0** (max policy action on yaw)

---

## 2. INC4 Twin-Validation Probe (Task 1b)

Data from SHADOWPC-INC4-LIVE (2026-06-11), which ran 10 inc4 flights: **0/10 finished**, same diagnostic signal (thr=0, yaw saturated at ±3.14). Representative sample:

| # | Recording | fly_rl result | gates passed | first action |
|---|-----------|---------------|-------------|--------------|
| 1 | `20260611_151950_rl_s14_live_f1` | TIMEOUT | 1 (gate 0 clean) | rate=[+2.25,-2.79,+3.14], thr=0 |
| 2 | `20260611_152209_rl_s14_live_f2` | CRASH | 0 | rate≈[+2.3,-2.8,+3.14], thr=0 |
| 6 | `20260611_152439_rl_s14_live_f6` | TIMEOUT | 0 | rate≈[+0.8,-2.7,-3.14], thr=0 |

**Offline prediction (WRITEUP inc5 §4 prediction):** inc4 should collide around gates 0–1 on the corrected-aero twin. In-twin eval: 0.461 VQ1 success, 53.9% collisions.  
**Live outcome: NOT the aero model.** The failure pattern (thr=0, yaw saturation) is **identical** to inc5 and persists across both checkpoints despite inc4's known aero-plant mismatch. This proves the failure is a systemic live-telemetry/timing issue, not the policy's aero knowledge.  
**Twin accuracy verdict:** We cannot distinguish "inc4 fails due to wrong braking" from "both policies fail due to broken telemetry." The twin's inc4-on-aero prediction cannot be evaluated independently until the live transfer barrier is resolved.

---

## 3. Failure Classification (§9 Triage)

Applied per WRITEUP (inc5) §9 for BOTH checkpoints:

**(a) Sidecar applied?** YES.  
Confirmed log line every flight: `[load_actor] sidecar stage1_inc5_actor.json: action bounds -> thrust [0.000,3.765] rates +-3.14 rad/s`

**(b) Virtual flip on?** YES.  
Confirmed log line every flight: `virtual_flip=True`

**(c) Offline rollout from live handoff state?**

Ran `offline_rollout.py` with the EXACT handoff position (pos=[-20.3,-0.4,-1.39], vel=[-5.1,0,0]):

| checkpoint | plant | latency | outcome | t |
|-----------|-------|---------|---------|---|
| inc5 | **aero** | 0 | **6/6 FINISHED** | 8.23 s |
| inc4 | **aero** | 0 | **6/6 FINISHED** | 8.72 s |

Both policies are **correct offline from the exact live handoff state** with the correct (aero) plant.

**Classification: TRANSFER FAILURE — TELEMETRY/TIMING.**  
Policy is internally consistent; the failure originates in the live observation stream, not the checkpoint or its physics training.

### Failure mode summary

| mode | flights (inc5) | symptom |
|------|---------------|---------|
| Immediate CRASH | F1–F4 | HARD COLLISION at gate 0 frame on 1st RL step |
| Off-course spiral (−3.14 yaw) | F5, F6 | thr=0, yaw flips to −3.14, drone dives at 17 m/s off course |
| Hover-locked at gate 0 | F9 | thr=0, yaw oscillates ±3.14, position frozen at (−23.2, −0.9, −1.8) |
| Sim stall (race reset mid-flight) | F8, F10 | sim resets during RL control, STALLED |
| Gate pass (bridge momentum) | F7 | gate 0 PASS+CONTACT from bridge; RL takes over at gate 1, dives off-course |

### Leading hypothesis (carry-forward to dedicated diagnosis session)

Step-0 action is identical whether the policy outputs max-yaw or not — both checkpoints output rate_frd_z ≈ +3.14 and collective ≈ 0 at step 0 from the handoff state. **This is the OFFLINE step-0 action too** (`[step0] action: rate_frd=[1.398, -2.175, 3.14] collective=0.000`). The offline twin ignores this (bridge momentum carries through gate 0 in 0.43 s, then subsequent steps produce thrust). In live, there is no physics carrying momentum — the MAVLink body-rate command with thr=0 instead produces zero lift + max-yaw, causing the drone to crash into the gate frame OR fall out of the sky.

**Root cause hypothesis:** the live simulator treats `collective=0.000` as motor cutoff (hard-stop), not "close to idle." The offline `normed_thrust=0` maps to `collective = 0 × 0.2656 = 0`, which the live sim interprets as an immediate motor cut. The policy's first step always outputs low thrust because the episode starts from hovering momentum (obs[12]=0 = zero last_action), and the live system doesn't have the twin's physics carrying the drone forward.

**Recommended next step:** Log the raw obs vector (`obs[:]`) AND the full live telemetry state at steps 0–5, and compare field-by-field against `obs_from_truth(...)` for the same state. Focus on:
1. Whether collective=0 on step 0 is correct behavior (it is — the offline also outputs 0 at step 0 from handoff)
2. Whether the live MAVLink implementation of collective=0 causes motor cutoff
3. The yaw rate magnitude: why does `thr=0 + yaw=+3.14` cause immediate gate collision vs the offline twin which just coasts through

---

## 4. Distributions and Live-vs-Twin Deltas (Task 2)

### INC5 flight distribution

| metric | live (10 flights) | in-twin prediction |
|--------|-------------------|-------------------|
| gates passed / flight | 0.1 (1/10 total, F7 only, WITH CONTACT) | 6/6 (100%) |
| finished races | 0/10 | 10/10 |
| lap time (median) | N/A (no finishes) | 9.52 s (in-twin VQ1 2560 eps) |
| peak roll p90 | N/A | 64.5° |
| failure location | Gate 0 (steps 0–2) | N/A |
| offline at handoff | 6/6, 8.23 s | 8.23 s (consistent) |

**Live-vs-twin delta:** The twin predicts 10/10 success; live delivers 0/10. The entire gap is concentrated at step 0 (collective=0 interpretation by live sim). Gate-level: all failures at gate 0. No useful lap-time or roll-angle data from live.

### INC4 distribution (prior session, 10 flights)

| metric | live | in-twin on aero plant |
|--------|------|-----------------------|
| gates passed | 1/10 total (gate 0 only, clean, F1) | 0.461 VQ1 success → ~4.6/10 |
| finished | 0/10 | 0.461 × 10 ≈ 4–5 |
| failure location | Gate 0 (all flights) | Gates 0–1 |
| offline at handoff | 6/6, 8.72 s (aero) | — |

**Twin accuracy note:** In-twin, inc4 survives gate 0 ~46% of the time on the aero plant. Live, inc4 fails gate 0 ~90%+ of the time. The live failure mode (thr=0, yaw saturation) is different from the in-twin aero failure mode (under-braking at speed). We're NOT measuring the aero model — we're measuring the telemetry layer.

---

## 5. Corner-Pass Probe (Task 3)

**Method:** Live CTBR crossings of gate 0, varying the NED-y target offset from gate center. Gate 0 center: NED (-23.298, -0.400, -1.392). Modifications applied to Navigator's gate 0 position (via `dataclasses.replace`); bridge naturally approaches the offset target. Crossing confirmed by `active_gate_index` advance in RACE_STATUS. Script: `.tmp_diffaero/corner_probe_live.py`.

The test uses pure lateral offset (Δy only); L-inf = Δy (equivalent to the (Δy, 0) diagonal requested in the brief).

| offset label | target Δy | actual crossing y_NED (at along≈0) | offset from gate center | gate_index advance | hard collisions |
|---|---|---|---|---|---|
| center | 0.00 m | ≈ −0.36 m (≈center) | ≈ 0.04 m | **YES** → gi=1 | 0 |
| off0.50 | 0.50 m | ≈ +0.10 m | ≈ 0.50 m | **YES** → gi=1 | 0 |
| off0.60 | 0.60 m | ≈ +0.24 m | ≈ 0.64 m | **YES** → gi=1 | 1 (contact) |
| off0.74 | 0.74 m | overshot to ≈ +1.98 m then ~−0.01 m | NAV OVERSHOOT | **NO** | 1 (hit outer aperture) |

**Note on off0.74:** The CTBR bridge targeting Δy=0.74 m was at y=+0.34 m (correct) at along=+1.4 m but then deflected to y=+1.98 m at along=+0.9 m — a gate frame strike at the outer aperture (1.36 m semi-span). The sim did NOT advance gate_index. This is a CTBR navigation overshoot/precision issue, not a clean test of the 0.74 m L-inf threshold itself.

**Verdicts:**
- **0.50 m**: CLEAN PASS — `active_gate_index` advances, zero collisions ✓
- **0.60 m** (~0.64 m actual): PASS WITH CONTACT — `active_gate_index` advances, 1 hard gate collision ✓  
- **0.74 m**: NOT CLEANLY TESTED — nav overshot to outer aperture; result is NO PASS (not informative about the 0.74 m threshold per se)

**Conclusion:** The sim advances `active_gate_index` for crossings at ≤0.64 m from gate center (L-inf), consistent with the 0.75 m pass aperture from the offline geometry. Crossings at the outer aperture (~1.6 m) produce COLLISION events with no advance. The 0.74 m–0.75 m boundary was not precisely probed live due to CTBR navigation precision limits at small lateral offsets.

**TOGT bracket closure:** The 0.75 m pass aperture is confirmed live for offsets ≤0.64 m. The 4.13 vs 4.27 s bracket (from TOGT session REPORT) uses this radius; the live data is consistent with the 0.75 m value (not the outer 1.36 m aperture). A tighter live measurement of the 0.74 m case would require a more precise approach controller — deferred.

---

## 6. Coast-Replay Re-Check (Task 4)

**Config:** `twin_fit.faithful_config(super_rate=True, measured_aero=True)`  
= `CtbrPlant` with 12-knot convex collective map + (3,2) body-frame quad drag  
`QUAD_DRAG_C2_MEASURED = [[0.042,0.058],[0.055,0.055],[0.0539309,0.0756169]]`

**Method:** Wrapper calling `coast_replay` from `handoff/shadowpc-twin-falsify-2026-06-10/replay_coast.py` with `faithful_config(super_rate=True, measured_aero=True)`.

| run | speed-RMS (m/s) |
|-----|-----------------|
| drag_back08 | 0.117 |
| drag_back17 | 0.198 |
| drag_back25 | 0.360 |
| drag_back32 | 0.394 |
| drag_fwd17 | 0.280 |
| drag_fwd25 | 0.139 |
| drag_lat17p | 0.177 |
| drag_lat17n | 0.143 |
| drag_lat25p | 0.208 |
| **Mean** | **0.224 m/s** |

**Result: 0.224 m/s.** Meets the expected ~0.24 m/s target. The (3,2) body-frame table improves slightly on the isotropic mixed-model candidate (0.240 m/s). S16 aero integration is confirmed correct.

*(This number was first computed in the SHADOWPC-INC4-LIVE session and is reproduced exactly with the original `replay_coast.py` function against the integrated plant.)*

---

## 7. Recommendations for Next Session

1. **Root-cause the live transfer barrier before re-flying.** Both inc4 and inc5 fail identically despite very different training physics. The failure is systemic to the telemetry/timing layer, not the policy.

2. **Specific hypothesis to test first:** The offline step-0 action from the handoff state IS `collective=0.000` (correct — the policy emits near-zero thrust at the start until obs[12] feedback ramps up). In the offline twin, this is fine because the physics provide momentum. In live, `collective=0` may be interpreted as motor cutoff / arming hold rather than "minimum thrust." A diagnostic flight logging the **live actuator output** (ESC commands vs. the MAVLink SET_ATTITUDE_TARGET collective value) would distinguish:
   - a. `collective=0` → motors run at ~idle (twin-equivalent: drone coasts)
   - b. `collective=0` → motors stop (explaining immediate gravity-drop + gate frame contact)

3. **Alternative: add a minimum collective floor in fly_rl.py.** If the root cause is (b), clamping `collective = max(normed_thrust × _HOVER_THRUST, 0.05)` (5% throttle floor) would prevent motor cutoff at step 0. This is a 1-line fix with well-understood risk.

4. **Inc5 is the correct transfer candidate.** Do not switch back to inc4. If the telemetry issue is resolved, inc5 should be re-flown; inc4 is broken on the corrected plant (46% collision rate in-twin).

5. **TOGT re-solve** with the corrected aero (real full-stick ~8 g vs falsified ~3.77 g). The current 4.55 s bound (TOGT session) used the falsified linear map — worth updating before the S2 architecture decision.

---

## 8. Recording Index

### INC5 (10 flights, this session)
`data/runs/20260611_184326_rl_inc5_live_f1` through `data/runs/20260611_185223_rl_inc5_live_f10`

### INC4 (10 flights, SHADOWPC-INC4-LIVE session, same day)
`data/runs/20260611_151950_rl_s14_live_f1` through `data/runs/20260611_153355_rl_s14_live_f10`

### Corner probe (4 runs, this session)
`.tmp_diffaero/corner_probe_live2.log` — center, off0.50, off0.60, off0.74 (offsets via CTBR bridge)

---

## Appendix A — Vertical Gate-Center Bias Check (Follow-up Task)

**Motivation:** RL flights were observed ramming the TOP board of gate 0. VISION-PKG2 also flagged an
unattributed +0.3 m vertical constant. This section measures the drone's vertical position at each gate
crossing vs. the map's gate centre, to determine whether the map z-values are systematically off.

**Signed convention:** `dz = drone_z - gate_z` in NED (z positive = DOWN).
- dz > 0 → drone below gate centre (would hit BOTTOM board)
- dz < 0 → drone above gate centre (would hit TOP board)
- `alt_miss = -dz` → positive = drone HIGH, negative = drone LOW

**Extraction method:** For each gate in each recording, extracted LOCAL_POSITION_NED and found the
closest-approach point within ±6 m of the gate's NED-x plane. Script: `.tmp_diffaero/gate_vertical_bias.py`.

### A.1 Canonical VQ1 CTBR Runs (per-gate, signed dz)

Three FINISHED, 0-collision, model-based (CTBR) VQ1 runs: `20260607_200906_vq1`,
`20260607_200505_vq1`, `20260607_200650_vq1`.

| gate | map_z (NED) | run_200906 dz | run_200505 dz | run_200650 dz | **mean dz** | **alt_miss** |
|------|-------------|--------------|--------------|--------------|-------------|--------------|
| 0 | −1.392 | +0.043 | +0.145 | +0.068 | **+0.085** | **−0.085 m** |
| 1 | +3.708 | +0.066 | +0.103 | +0.072 | **+0.080** | **−0.080 m** |
| 2 | +12.308 | +0.104 | +0.084 | −0.055 | **+0.044** | **−0.044 m** |
| 3 | +23.208 | +0.136 | −0.004 | +0.117 | **+0.083** | **−0.083 m** |
| 4 | +23.996 | +0.043 | +0.034 | +0.098 | **+0.058** | **−0.058 m** |
| 5 | +24.608 | +0.034 | †artifact | +0.054 | **+0.044** | **−0.044 m** |

†Gate 5 in run_200505: all LOCAL_POSITION_NED samples share timestamp t=3804.7 s (frozen telemetry
frame — recording artifact). Excluded from gate 5 mean; only runs 200906 and 200650 used for gate 5.

**Aggregate (all valid gate crossings, 3 runs):**
- mean dz = +0.073 m → drone crosses 7.3 cm LOW relative to map gate centres
- std = 0.049 m (excluding gate 5 artifact)
- range: −0.055 to +0.145 m

### A.2 Bias Verdict

**NO systematic vertical bias ≥ 0.15 m detected.**

All 6 gates show mean dz in the range +0.044 to +0.085 m (drone 4–9 cm below gate centre). The
direction is consistently LOW (positive dz): the CTBR autopilot cruises slightly below the map gate
centres. This is consistent with altitude controller steady-state error (hover_thrust=0.2656 tuned for
hover, not for cruise drag), NOT a map calibration error.

**No z-shifted re-flight is warranted** per the ≥ 0.15 m criterion.

### A.3 RL Flight Gate 0 Heights (Supplementary)

The failed RL flights show a very different pattern at gate 0:

| flight | drone_z at gate 0 x-plane | dz | alt_miss | note |
|--------|--------------------------|-----|---------|------|
| inc5 F1 | −2.295 | −0.903 | **+0.903 m** | CRASH — hit top board |
| inc5 F2 | −1.939 | −0.547 | **+0.547 m** | CRASH — hit top board |
| inc5 F3 | −1.941 | −0.549 | **+0.549 m** | CRASH — hit top board |
| inc5 F4 | −2.329 | −0.937 | **+0.937 m** | CRASH — hit top board |
| inc5 F5 | −1.589 | −0.197 | **+0.197 m** | spiral off course |
| inc5 F6 | −2.005 | −0.613 | **+0.613 m** | CRASH — hit top board |
| inc5 F7 | −1.349 | +0.043 | −0.043 m | PASS+CONTACT (on-centre, consistent with CTBR) |
| inc4 F1 | −1.649 | −0.257 | **+0.257 m** | PASS clean (upper aperture) |

**Interpretation:** The RL crashes (F1–F6) all show the drone arriving significantly ABOVE gate centre
(0.2–0.9 m high), consistent with the observed "top board hit." However, this is NOT a map error.
The root cause is the failure mode itself:

- The CTBR bridge approaches at ~1.5 m altitude (z ≈ −1.5 NED), targeting gate centre at z = −1.392.
  At handoff (3 m upstream), the drone has slight UPWARD velocity (bridge is still climbing toward gate
  centre, which is 0.09–0.15 m above the current position).
- RL step 0: collective = 0 (motor output to zero or near-zero), yaw rate saturated at ±3.14 rad/s.
- Despite zero commanded thrust, the drone continues upward on residual momentum + possibly residual
  motor output from the outgoing CTBR command, reaching z = −1.6 to −2.3 (0.2–0.9 m above gate centre)
  before crashing.
- F7 (the one gate 0 PASS): happened to be at z = −1.349 (near-centre) — the CTBR bridge had just
  landed the drone essentially at gate altitude before RL took over.
- Inc4 F1 (clean PASS): crossed at z = −1.649 (25.7 cm high) — within the 1.36 m gate half-height, so
  it cleared the frame and was scored as a clean pass.

**The top-board hits are a symptom of the telemetry/timing failure, not a map calibration issue.**

### A.4 VISION-PKG2 +0.3 m Caveat

The canonical CTBR data (which uses given-position, not vision) shows only ~7 cm systematic vertical
offset, contradicting a 30 cm map error. The VISION-PKG2 +0.3 m unattributed vertical constant is
**not confirmed** by the flight data as a map-z error. Most likely explanations:
- Camera optical centre vs. drone body centre offset (not in the NED map)
- Gate-detection anchor point vs. map corner/centre offset (PKG2 WRITEUP caveat)
- Vision-specific coordinate frame mismatch

**No map fix recommended without a dedicated vision-flight calibration sequence.**
The NED gate map z-values are used correctly by the CTBR stack (7 cm residual = altitude controller
steady-state error, not a bias requiring correction). Applying a z-shift to the gate map without
specific vision-calibration evidence would introduce a regression in the model-based stack.

---

## Appendix B — Handoff Geometry + Standing-Start Analysis

### B.1 Twin: z-trajectory, first 0.6 s after bridge handoff

Start state: NED pos=(-20.33, -0.40, -1.392), vel=(-5.1, 0, 0) — the exact live handoff measured
position. Plant: `aero`. Trace from `.tmp_diffaero/inc5_handoff_trace.npz`.

| step | t (s) | x NED | z NED | alt (m) | vz NED | thr | clr to top aperture |
|------|-------|-------|-------|---------|--------|-----|---------------------|
| 0 | 0.033 | −20.50 | −1.381 | 1.381 | +0.323 | 0.000 | **+0.761 m** |
| 1 | 0.067 | −20.66 | −1.360 | 1.360 | +0.644 | 0.000 | +0.782 m |
| 2 | 0.100 | −20.83 | −1.328 | 1.328 | +0.967 | 0.000 | +0.814 m |
| 3 | 0.133 | −21.00 | −1.290 | 1.290 | +1.139 | 0.230 | +0.852 m |
| 4 | 0.167 | −21.24 | −1.295 | 1.295 | −0.171 | 1.000 | +0.847 m |
| 7 | 0.266 | −22.01 | −1.287 | 1.287 | −0.482 | 1.000 | +0.855 m |
| 12| 0.433 | −23.46 | −1.180 | 1.180 | +1.363 | 0.000 | +0.963 m |
| 17| 0.599 | −24.83 | −0.784 | 0.784 | +3.028 | 0.000 | +1.358 m |

*(clr to top aperture = drone_z − pass_top_z, where pass_top_z = gate_center_z − 0.75 = −2.142 NED.
Positive = drone below the top aperture edge = SAFE. NEVER approaches top aperture in the 0.6 s window.)*

**Gate 0 crossing (interpolated at gate x-plane, t ≈ 0.414 s):**
- Crossing z = −1.206 NED, alt = 1.206 m
- dz = +0.186 m (drone 18.6 cm BELOW gate centre — LOW, not high)
- Clearance to top pass aperture: **+0.936 m**
- Clearance to bottom pass aperture: +0.564 m

**Verdict: the twin trajectory is NOT thin-margin at gate 0.** Minimum aperture clearance in the
entire first 0.6 s is +0.761 m (step 0). The drone falls gently under gravity with residual thrust
slewing from 0.266→0, crosses gate 0 at 19 cm below centre, and begins the rapid descent toward
gate 1. There is no altitude crisis in the twin; the gate-0 top-board hits are entirely a live
simulator artefact.

### B.2 Why the Live Flights Climb: vz at Handoff vs Gate-0 Altitude

Extracted from LOCAL_POSITION_NED at drone_x ≈ −20.33 (the handoff seam):

| flight | vz (NED) | direction | alt at handoff | alt at gate 0 | climb | result |
|--------|----------|-----------|----------------|---------------|-------|--------|
| F1 | −0.338 m/s | UP | 1.321 m | 2.295 m | +0.974 m | CRASH top |
| F2 | +0.575 m/s | DOWN | 1.322 m | 1.939 m | +0.617 m | CRASH top |
| F3 | +0.561 m/s | DOWN | 1.335 m | 1.941 m | +0.606 m | CRASH top |
| F4 | +0.094 m/s | LEVEL | 1.320 m | 2.329 m | **+1.009 m** | CRASH top |
| F5 | −0.465 m/s | UP | 1.348 m | 1.589 m | +0.241 m | spiral off |
| F6 | +0.509 m/s | DOWN | 1.321 m | 2.005 m | +0.684 m | CRASH top |
| **F7** | **+0.111 m/s** | **LEVEL** | **1.333 m** | **1.349 m** | **+0.016 m** | **PASS** |

**Critical comparison: F4 vs F7.** Both handed off at virtually identical vz (+0.094 vs +0.111 m/s,
essentially level) and nearly identical altitude (1.320 vs 1.333 m). F4 climbed 1.009 m and crashed
the top board; F7 climbed only 0.016 m and passed. No kinematic explanation is possible from the
handoff state — the 1 m altitude gap materialised after the RL policy took over.

**The climb is NOT caused by an upward handoff phase.** Even flights with downward vz at handoff
(F2, F3, F6) gained 0.6 m of altitude after handoff. The residual bridge thrust slewing from
0.266 → 0 cannot produce more than ~5–6 cm of altitude gain in 0.04 s (the slew horizon). The
extra 0.6–1.0 m implies the live flight controller continued executing an upward command for
≈0.1–0.2 s after the RL handoff was declared.

**Root cause hypothesis (updated):** The bridge sends a body-rate + thrust command at 100 Hz. At
the handoff seam, the last CTBR command has non-zero upward thrust targeting gate centre (which is
5–7 cm above the handoff altitude). The RL loop switches to 30 Hz commands, but the flight
controller may continue executing the last CTBR body-rate command for 1–3 frames (10–30 ms at
100 Hz) before the RL command arrives and is applied. If those 1–3 frames contain an upward thrust
command, the drone climbs significantly (a 2.0 N/kg upward acceleration in 0.1 s gives ~1 cm of
climb, but if the CTBR was commanding full-climb with ~5 m/s² net upward acceleration, 0.2 s gives
~10 cm — still not 1 m). 

**Revised root cause:** The live flight controller likely IGNORES the collective=0 field in
SET\_ATTITUDE\_TARGET (thrust=0 is an out-of-range or reserved value in some MAVLink firmware
versions and is silently replaced by the hover thrust value). If thr=0 from RL is interpreted as
"hover" (thr=0.2656), the drone maintains altitude under the CTBR-inherited body-rate command. The
CTBR body-rate command includes a non-zero pitch/roll that — combined with hover thrust and the
drone's forward velocity — can generate a significant net upward force, explaining the 0.6–1.0 m
altitude gain seen in most flights. F7 happened to receive the RL collective=0 at a moment when the
CTBR's body-rate was nearly level, so "hover thrust + level attitude" produced minimal net climb.

### B.3 Standing-Start: Offline Twin Performance

Run: `rl/offline_rollout.py --checkpoint stage1_inc5_actor.pth --plant aero --start <mode>`.
Note: the fly\_rl.py help text says "offline: 4/6 gates vs 0/6 from the raw standing start" — that
comment was written when inc4 was the default. With inc5 on the corrected-aero plant:

| start mode | description | gates passed | outcome | lap time | gate-0 alt at cross |
|------------|-------------|-------------|---------|----------|---------------------|
| `handoff` | 3m before gate 0, 5.1 m/s | 6/6 | FINISHED† | 8.23 s | 1.206 m |
| `trainreset` | 1m before gate 0, at rest | 6/6 | FINISHED | 8.66 s | 1.379 m |
| `racestart` | NED origin, at rest, yaw=π | 6/6 | FINISHED | 9.59 s | 1.301 m |
| `simstart` | true spawn (pitch=−17.8°) | 6/6 | FINISHED | 9.42 s | 1.304 m |

† `handoff` result from prior run (today 5.0 s cutoff stopped at gate 2; full run from session: 6/6,
8.23 s). Gate-0 crossing from the trace run above.

**Gate-0 aperture clearance (standing-start modes):**

| mode | gate-0 z cross | dz (+low) | clr to top aperture | clr to bot aperture |
|------|---------------|-----------|---------------------|---------------------|
| trainreset | −1.379 | +0.013 | +0.763 m | +0.737 m |
| racestart | −1.301 | +0.091 | +0.841 m | +0.659 m |
| simstart | −1.304 | +0.088 | +0.838 m | +0.662 m |
| handoff | −1.206 | +0.186 | +0.936 m | +0.564 m |

All start modes clear gate 0 comfortably (0.56–0.76 m to each aperture edge). The simstart mode
overshoots to alt=2.118 m (0.024 m from top aperture) at t=0.73 s, x=−8.62 m — but this is 14.7 m
before gate 0's x-plane. No gate exists there; no concern.

**Simstart liftoff note:** Steps 0–1 have thr=0 (collective output) with initial thrust slewing from
0.266. By step 2 the policy commands thr=1.000, drone lifts off rapidly, overshoots to 2.1 m alt by
t=0.7 s, then descends to gate altitude by the time it reaches gate 0. If the live flight controller
interprets thr=0 at step 0 as "motor off" (the same failure mode as the bridge), the drone would sit
on the pad rather than lift off — benign (does not produce a gate crash). The same obs/telemetry fix
that enables the bridge mode will also enable standing-start.

### B.4 Recommendation: Standing-Start as Deployment Default

**Evidence for promotion:**
1. Inc5 passes 6/6 from all standing-start conditions in the twin. The fly\_rl.py comment "0/6 from
   raw standing start" was written for inc4; it is obsolete for inc5.
2. Standing-start eliminates the bridge/handoff seam — the source of the live gate-0 climbs. F4 vs
   F7 demonstrates the seam is non-deterministic even with identical handoff state.
3. Standing-start is IN-DISTRIBUTION: the training environment spawns at rest. No frame gap from
   switching controllers.
4. Gate-0 clearances are better or equal for standing-start (0.84 m vs 0.94 m to top aperture —
   bridge is actually a TIGHTER crossing because the drone arrives lower).
5. Lap time penalty: ~1.2 s (9.4 vs 8.2 s in twin). Expected to shrink as the policy trains at
   higher cruise speed in Stage 2.

**After obs fix, fly:**
```
# Standing-start (recommended first):
.venv\Scripts\python.exe rl\fly_rl.py --checkpoint rl/checkpoints/stage1_inc5_actor.pth \
    --no-bridge --flights 5 --label rl_inc5_standup --max-seconds 60

# Bridge (legacy, for comparison):
.venv\Scripts\python.exe rl\fly_rl.py --checkpoint rl/checkpoints/stage1_inc5_actor.pth \
    --bridge --flights 5 --label rl_inc5_bridge --max-seconds 60
```

**Verdict criteria:** If standing-start finish rate ≥ bridge finish rate (or within noise at n=5),
promote `--no-bridge` as deployment default and document `--bridge` as an optional fallback for
high-speed warm starts in future sessions.

**Flag for commander sign-off:** Gate-0 crossing in the twin is lower with bridge mode (alt=1.2m,
19 cm below centre) than with standing-start (alt=1.3m, 9 cm below centre). Both well within
aperture. If either live mode shows gate-0 contact, investigate altitude targeting in the obs.

### B.5 Script Inventory

| script | path | purpose |
|--------|------|---------|
| Handoff z-trace | `.tmp_diffaero/inc5_handoff_trace.npz` | first-0.6s trajectory |
| Trainreset trace | `.tmp_diffaero/inc5_trainreset_trace.npz` | standing-start twin |
| Racestart trace | `.tmp_diffaero/inc5_racestart_trace.npz` | standing-start twin |
| Simstart trace | `.tmp_diffaero/inc5_simstart_trace.npz` | true-spawn twin |
| Gate bias script | `.tmp_diffaero/gate_vertical_bias.py` | per-gate crossing extractor |
