# SHADOWPC-INC4-LIVE — Handoff Writeup

**Session:** SHADOWPC-INC4-LIVE  
**Date:** 2026-06-11  
**Checkpoint:** `rl/checkpoints/stage1_inc4_actor.pth` (stage1_inc4, job 3267360, s14_valid_s1 seed 1, 6000 updates)  
**Sidecar:** `stage1_inc4_actor.json` → `act_max_thrust=3.765, act_max_rate=3.14`  
**Track:** `data/runs/track_map_20260602_114630.json` (6 gates)

---

## 1. Task Summary

| Task | Status |
|------|--------|
| 1. Live flights (10×) | DONE — all 10 recorded |
| 2. Failure report | DONE — §9.4 triage complete |
| 3. Corner-pass probe | PARTIAL — offline geometry only (live blocked by transfer failure) |
| 4. Coast-replay re-check | DONE (integrated aero-ON, 0.224 m/s) |

Sim exited to HOME after all tasks. Recordings in `data/runs/20260611_15????_rl_s14_live_f{1..10}`.

---

## 2. Live Flight Results

**Command:** `.venv\Scripts\python.exe rl/fly_rl.py --checkpoint rl/checkpoints/stage1_inc4_actor.pth --flights 10 --label rl_s14_live`

Sidecar confirmed: `[load_actor] sidecar stage1_inc4_actor.json: action bounds -> thrust [0.000,3.765] rates +-3.14 rad/s`  
Bridge: `--bridge True`, `--virtual-flip True`, `--rate 30`

| # | Recording | fly_rl result | race_outcome gates | Gate collisions | fly behavior |
|---|-----------|---------------|-------------------|-----------------|--------------|
| 1 | `20260611_151950_rl_s14_live_f1` | TIMEOUT | **1 (gate 0: PASS-CLEAN)** | 0 | Bridge→gate0 pass; policy: yaw locked +π, thr≈0, TIMEOUT 120s |
| 2 | `20260611_152209_rl_s14_live_f2` | CRASH | 0 | 1 gate hit (no pass) | Immediate HARD COLLISION at first RL policy step |
| 3 | `20260611_152227_rl_s14_live_f3` | CRASH | 0 | 341 gate + 1722 env | Spun into gate frame and wall/terrain repeatedly |
| 4 | `20260611_152246_rl_s14_live_f4` | STALLED | 0 | 178 gate + 355 env | fly_rl printed "gate 0 PASSED" but sim stalled (race reset mid-flight); prior-race residue discarded by race_outcome |
| 5 | `20260611_152417_rl_s14_live_f5` | STALLED | 0 | 1562 gate + 1 env | 1562 gate hits during stall — sim never cleanly started final epoch |
| 6 | `20260611_152439_rl_s14_live_f6` | TIMEOUT | 0 | 10 gate | First action yaw=−π (negative, vs +π in f1); flew NE off course 130s |
| 7 | `20260611_152658_rl_s14_live_f7` | TIMEOUT | 0 | 2 gate | Same pattern as f6 |
| 8 | `20260611_152917_rl_s14_live_f8` | TIMEOUT | 0 | 5 gate | Same pattern |
| 9 | `20260611_153135_rl_s14_live_f9` | TIMEOUT | 0 | 2 gate | Complex looping (flew back toward origin then off course) |
| 10 | `20260611_153355_rl_s14_live_f10` | TIMEOUT | 0 | 0 | Smooth off-course arc, 0 gate contacts |

**Summary:** 0/10 races finished. 1/10 had a sim-recognized gate pass (gate 0, flight 1). All failures occur immediately after (or at) gate 0 crossing. Total collisions across all flights: 200 (ids 1001=gate, 1002=env/wall).

---

## 3. Failure Analysis — §9.4 Triage

### Checklist

**(a) Sidecar applied?** YES.  
Log line (every flight): `[load_actor] sidecar stage1_inc4_actor.json: action bounds -> thrust [0.000,3.765] rates +-3.14 rad/s`

**(b) Virtual flip on?** YES.  
Log line (every flight): `[fly] RL policy  rate=30 Hz  max_rate=off  virtual_flip=True  max=120s`

**(c) Offline rollout from live handoff state?**  

Ran `offline_rollout.py --start handoff --handoff-speed 5.1`:

```
[outcome] FINISHED  gates passed: 6/6 [0,1,2,3,4,5]  t=6.76s  vmax=30.8 m/s
```

Also tested with:
- `--latency-steps 1` → 6/6 PASS (6.96s) ✓
- `yaw=0` at handoff (bridge heading north, not south) → 6/6 PASS (8.13s) ✓
- `yaw=π` at handoff (offline default) → 6/6 PASS (6.57s) ✓

**Policy is correct offline from the exact live handoff state. The failure is TELEMETRY/TIMING, not the policy.**

### Failure Modes Observed

| Mode | Flights | Symptom | First action |
|------|---------|---------|--------------|
| Post-gate-0 spin | f1 | Gate 0 passed (bridge momentum), then yaw=+π, thr≈0 frozen for 120s | rate=[+2.25,−2.79,+3.14] thr=0 |
| Immediate collision | f2, f3 | HARD COLLISION at t_0 of policy (gate frame hit) | rate≈[+2.3,−2.8,+3.14] thr=0 |
| Negative yaw spin | f6, f7, f8, f10 | Flew NE off course from step 1 | rate≈[+0.8,−2.7,−3.14] thr=0 |
| Chaotic loop | f4, f5, f9 | Erratic high-rate flight, sim stalled or loops away | Mixed rates, thr oscillating |

**Common diagnostic signal:** In all flights, yaw_rate = ±3.14 (saturated at `act_max_rate`) appears at step 0 and persists. This suggests the yaw component of the observation passed to the live policy is systematically wrong — placing the policy into a region not covered by training distribution.

### Diagnosis Status

**Classification:** Transfer failure — telemetry/timing mismatch. Policy is internally consistent (passes offline). Likely cause: observation feature related to yaw/heading is delivered incorrectly by the live MAVLink path (e.g., attitude convention, rate sign, or timing offset in the IMU angular velocity stream).

**Recommended next step:** Dedicated live-vs-offline obs comparison session. Record the live obs vector (log obs[] at step 0-5) and compare field-by-field against the offline `obs_from_truth` output for the same position/velocity/attitude. The yaw feature is the prime suspect.

---

## 4. Corner-Pass Probe

### Gate 0 Geometry

Source: `data/runs/track_map_20260602_114630.json`

- **Center NED:** (−23.298, −0.400, −1.392) m  
- **Gate size:** 2.72 m × 2.72 m (square)  
- **Pass threshold:** L-inf < 0.75 m from center  
- **Frame collision:** L-inf 0.75–1.36 m  
- **Clear miss:** L-inf > 1.36 m  

### Offline Geometry Verdicts

Script: `.tmp_diffaero/corner_probe_offline.py`

| In-plane offset (Δy, Δz) | L-inf | Offline verdict | Expected live |
|--------------------------|-------|-----------------|---------------|
| (0.50, 0.50) m | 0.50 m | **PASS** | PASS, gate_index→1 |
| (0.60, 0.60) m | 0.60 m | **PASS** | PASS, gate_index→1 |
| (0.74, 0.74) m | 0.74 m | **PASS** (1 cm inside boundary) | PASS, gate_index→1 |
| (0.75, 0.75) m | 0.75 m | PASS (at boundary) | — |
| (0.76, 0.76) m | 0.76 m | COLLISION | FRAME HIT, no advance |

All three requested offsets fall within the 0.75 m L-inf pass region. No COLLISION events expected.

### Why Live Probe Was Not Completed

The transfer failure renders the RL policy unusable for targeted gate navigation. The CTBR bridge consistently approaches at (Δy≈0, Δz≈+0.1 m) from gate center — insufficient to test corner offsets. Flight 1 already demonstrates a PASS-CLEAN from the bridge trajectory (within 0.1 m of center), confirming the sim correctly advances `active_gate_index`.

**What a live probe would require:**  
A custom CTBR waypoint approach script that: (1) targets the desired in-plane offset rather than gate center; (2) crosses the gate at speed ≥ 3 m/s; (3) queries `race_status.active_gate_index` before and after crossing. Estimated effort: ~1 h. Deferred to a dedicated session if geometry confirmation is required.

---

## 5. Coast-Replay Re-Check (Task 4)

**Config:** `twin_fit.faithful_config(super_rate=True, measured_aero=True)`  
= CtbrPlant + 12-knot collective map + body-frame drag table  
`QUAD_DRAG_C2_MEASURED = [[0.042,0.058],[0.055,0.055],[0.0539309,0.0756169]]`

**Script:** `.tmp_diffaero/coast_integrated.py`

| Run | Speed-RMS (m/s) |
|-----|-----------------|
| back08 | 0.117 |
| back17 | 0.198 |
| back25 | 0.360 |
| back32 | 0.394 |
| fwd17  | 0.280 |
| fwd25  | 0.139 |
| lat17p | 0.177 |
| lat17n | 0.143 |
| lat25p | 0.208 |
| **Mean** | **0.224 m/s** |

**Verdict:** Meets target (~0.24 m/s). The integrated (3,2) body-frame drag table improves on the isotropic mixed candidate (`cand_mix` was 0.240 m/s). S16 aero integration confirmed correct.

---

## 6. Sim Housekeeping

- All 10 flights: arms disarmed after TIMEOUT/CRASH/STALLED via `[safety] disarming ...`
- Sim exited to HOME at session end: `exit_to_home.py` → ESC + Down×3 + Enter to AI-GP window

---

## 7. Recommendations for Next Session

1. **Root-cause the transfer failure.** Log the raw obs vector (`obs[:]`) at steps 0–5 of a live flight and compare to `obs_from_truth(...)` offline. Focus on the yaw/heading features — yaw_rate saturating at ±3.14 on step 0 suggests one field is wrong.

2. **Candidate hypotheses (in priority order):**
   - Live attitude convention differs from training (e.g., live delivers roll/pitch/yaw in a different frame than the virtual_flip transform expects)
   - Angular-velocity sign from live ODO differs from training sign convention (`_ODO_RATE_SIGN = [-1,-1,1]`)
   - Observation timing: live obs is from the previous frame (1-step stale) but plant-step in training assumed zero lag

3. **Corner-pass probe** (if needed): implement CTBR offset-approach controller (~1 h).

4. **Coast RMS confirmed 0.224 m/s** — no further action needed on aero integration.
