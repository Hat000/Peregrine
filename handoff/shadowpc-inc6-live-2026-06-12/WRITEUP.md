# SHADOWPC-INC6-LIVE — Live deployment session (2026-06-12)

**Session:** SHADOWPC-INC6-LIVE. **Model:** claude-sonnet-4.6. **Checkpoint:** `rl/checkpoints/stage1_inc6_actor.pth` = `inc6_c16_s0` (md5 `8fb8855e07d4fd01045e7b2ecbc5acd3`). **Sim:** AI-GP v1.0.3364 (ShadowPC). **Preceded by:** `handoff/laptop-s17-mixer-inc6-2026-06-11/WRITEUP.md`.

**VERDICT: 0/15 finishes (standing 0/10, bridge 0/5). Transfer failure on ShadowPC. Deployment layer confirmed at both gate-0 (standing-start) and gate-1 (bridge post-handoff). Root cause: ShadowPC obs encoding mismatch versus laptop training/eval environment. Laptop deployed 16/16 on the same checkpoint.**

---

## Task 0 — Mixer probe (`mixer_probe2.json`)

**Best recording:** `data/runs/20260612_034154_mixer_probe2` (run 1 of 3).

Three attempts. All aborted at phase `c100_r31` (thr=1.0 + roll_rate=3.14 rad/s) due to `rate_sysid.py` Euler ZYX singularity. Root cause: `c100_r31` drives roll past 90° in <0.5 s; at >90° roll, ZYX Euler representation is discontinuous and `s.roll` reads far above any `--max-tilt-deg` limit. Cannot be fixed without modifying `abort_check()` in `rate_sysid.py`. Three runs at `--max-tilt-deg` 77/85/88/90 (successive retry limits) all aborted on the same tick.

| Run | Recording | Aborted at | Tilt at abort |
|-----|-----------|-----------|---------------|
| 1 | `20260612_034154_mixer_probe2` | c100_r31 | 77 deg |
| 2 | `20260612_034440_mixer_probe2` | c100_r31 | 88 deg |
| 3 | `20260612_034640_mixer_probe2` | c100_r31 | 93 deg |

**Coverage:**

| Phase | Status | Notes |
|-------|--------|-------|
| z00_y31_long | **CLEAN** | 1.5 s yaw step at thr=0. Captured in all 3 runs. |
| c100_r31 | **PARTIAL** | ~0.5 s of data captured before abort; settled spin not reached. |
| c100_y31 | NOT CAPTURED | Abort occurred before this phase. |
| c60_r31 | NOT CAPTURED | Abort occurred before this phase. |
| zhov_r31 | NOT CAPTURED | Abort occurred before this phase. |

**ACTUATOR_OUTPUT_STATUS:** Confirmed captured in all runs (e.g. `mot~0.050 [0.05,0.05,0.05,0.05]` at idle; `mot~0.404 [0.08,0.73,0.73,0.08]` during z00_y31 step). The `c100_r31` settled motors `[0.99,0.65,0.99,0.65]` are present in run 1 output prior to abort.

**For laptop follow-up:** The three unconstrained phases (c100_y31, c60_r31, zhov_r31) were not probed. The kappa DR bands in the shipped model remain as-is. See S17 WRITEUP Section 2 "Under-constrained corners."

---

## Task 1 — Standing-start flights ×10

**Checkpoint:** `rl/checkpoints/stage1_inc6_actor.pth` (inc6_c16_s0). No mitigation flags.

**Result: 0/10 finishes. 0 gate passes.**

| Flight | Recording | Outcome | Gates | Collision events |
|--------|-----------|---------|-------|-----------------|
| 1 | `20260612_034852_inc6_standing_f1` | SPIN_ABORT | 0 | env |
| 2 | `20260612_035014_inc6_standing_f2` | SPIN_ABORT | 0 | env |
| 3 | `20260612_035048_inc6_standing_f3` | CRASH | 0 | env |
| 4 | `20260612_035131_inc6_standing_f4` | SPIN_ABORT | 0 | env |
| 5 | `20260612_035205_inc6_standing_f5` | TIMEOUT | 0 | env |
| 6 | `20260612_035415_inc6_standing_f6` | SPIN_ABORT | 0 | env |
| 7 | `20260612_035458_inc6_standing_f7` | TIMEOUT | 0 | env |
| 8 | `20260612_035708_inc6_standing_f8` | TIMEOUT | 0 | env |
| 9 | `20260612_035918_inc6_standing_f9` | TIMEOUT | 0 | env |
| 10 | `20260612_040128_inc6_standing_f10` | SPIN_ABORT | 0 | env |

**Total collisions (all flights):** 198, all id=1002 (environment). Zero gate contacts (id=1001).

**Trajectory pattern:** Identical, deterministic first action every flight: thr=0.320, rate=[+0.59,-1.68,+1.08]. Drone reaches approximately (-50m, +22m, +13m) within ~14s — overshooting gate-0 region (gate is at ≈x=-23m). Does not pass gate 0 in any flight. After initial overshoot, enters large orbital patterns (50-80m diameter) climbing to 20-35m altitude.

**Spin-abort pattern:** |w|=7.0-7.1 rad/s for >2s with no gate progress. NOT a yaw-dither pattern (rates are moderate, not railed at ±3.14). Six of ten flights reached SPIN_ABORT; three timed out (120s elapsed at gi=0); one crashed into environment.

**Lap time vs 9.9s prediction:** N/A (no gate passes).

**Per-gate pass offsets vs ≤0.28m:** N/A (no gate passes).

**Contact events:** Zero gate contacts — all 198 collisions are environment (id=1002). All flights certified VALID by the zero-gate-contact criterion; all invalid by the zero-finish criterion.

**Style notes:** No yaw dither observed (yaw_p95 was 0.009 on laptop; consistent with mixer physics killing it in training). Thrust is moderate (0.2-0.7 range, not railing). The failure mode is not style — it is navigation.

---

## Task 2 — Bridge-mode flights ×5

**Checkpoint:** `rl/checkpoints/stage1_inc6_actor.pth` (inc6_c16_s0). Bridge mode (default `--bridge`). Bridge handoff at along-track < 3m and speed > 4 m/s from gate 0.

**Result: 0/5 finishes. 5/5 passed gate 0. 0/5 passed gate 1.**

| Flight | Recording | Outcome | Gates |
|--------|-----------|---------|-------|
| 1 | `20260612_040440_inc6_bridge_f1` | TIMEOUT | 1 |
| 2 | `20260612_040748_inc6_bridge_f2` | TIMEOUT | 1 |
| 3 | `20260612_041004_inc6_bridge_f3` | SPIN_ABORT | 1 |
| 4 | `20260612_041035_inc6_bridge_f4` | SPIN_ABORT | 1 |
| 5 | `20260612_041113_inc6_bridge_f5` | STALLED | 1 |

**Bridge handoff position (all flights, highly consistent):** pos=(-20.3, -0.4, -1.3), speed=5.1 m/s, along≈+3.0 m to gate 0. Gate 0 x_ned = -23.3 m.

**Gate 0 pass:** Occurs at or immediately after handoff in all 5 flights. Gate 0 position at pass: (-20.3 to -20.4, -0.4, -1.3 to -1.4) m. No gate contact.

**Post-gate-0 behavior:** After gate 0 pass, the policy consistently takes a large lateral sweep: moves to approximately (-57m, -9m, +7m) in the first 1-2 seconds, then makes a large arc and returns toward the gate-1 target region. None of the flights converge onto gate 1. The drone either orbits at large radius (50-100m+) and times out, or accumulates rate and triggers SPIN_ABORT.

**Gate-0 vs gate-1 contrast:** The bridge controller navigates to gate 0 correctly, and the RL policy passes gate 0 with essentially zero offset (handoff is 3m from gate center, and gate is registered within 1 tick). However the same RL policy cannot navigate from gate-0-pass to gate 1.

**Style notes:** No yaw dither. Thrust moderate (0.2-0.7). Rates moderate except in SPIN_ABORT events (briefly >6 rad/s). Same non-dither style as standing-start.

---

## Task 3 — A/B comparison: standing-start vs bridge

| Metric | Standing-start | Bridge |
|--------|---------------|--------|
| Finishes | 0/10 | 0/5 |
| Gate 0 passes | 0/10 | 5/5 |
| Gate 1 passes | 0/10 | 0/5 |
| Median gates reached | 0 | 1 |

**Standing ≥ bridge?** No. Bridge (gates=1 on all 5 flights) clearly outperforms standing-start (gates=0 on all 10). **Recommendation: do NOT flip deployment default to standing-start for ShadowPC.** Bridge mode provides enough initial trajectory momentum for the RL policy to register gate 0; standing start does not.

However, neither mode is functional — both stall at the same transition layer.

---

## Failure triage (S17 playbook)

The S17 playbook categorizes failures:
- **"anything at step 0 = deployment layer"** — applies to standing-start (never passes gate 0)
- **"lateral-late = top-rail depth"** — does not apply (rates not railed)
- **"parasitic-climb = bottom rail wrong"** — does not apply (no thr=0 rate-rail spin observed)

The bridge flights expose a second deployment layer: gate-0-pass → gate-1-targeting. The policy navigates the first 1-2 seconds after gate-0-pass in a consistent, large-radius arc that misses gate 1 every time.

**Working hypothesis — ShadowPC obs encoding mismatch:**

The laptop evaluation ran 16/16 on this same checkpoint (`stage1_inc6_actor.pth`) across all start modes including standing start and latency 2-3. ShadowPC gives 0/15. The difference is most likely in how gate-relative observations are encoded by the sim on this machine:

1. **Gate position/heading encoding:** If the sim reports gate centers or headings in a different convention on ShadowPC vs laptop, the policy's navigation commands will be systematically wrong.
2. **Observation normalization:** The policy receives normalized gate-relative obs. If normalization constants differ (e.g. a course-length scale factor), the policy would navigate in the wrong direction or at the wrong speed.
3. **Sim version or course variant:** ShadowPC uses AI-GP v1.0.3364. If the laptop used a different version, course layout or gate positions may differ.

**The standing-start failure is consistent with a gate-0-relative obs encoding issue:** the first action is deterministic ([+0.59,-1.68,+1.08]) and identical across all 10 flights, suggesting the initial obs is the same every time and maps to a specific action that is correct on laptop but overshoots gate 0 on ShadowPC.

**The bridge failure (stuck at gate 1) is consistent with the same root cause:** gate-1-relative obs is wrong on ShadowPC, so the policy navigates in a large arc that never converges.

**Recommended debug path:**
1. Log and compare `debug_obs.jsonl` first obs vector from ShadowPC (available in each run directory) vs laptop standing-start equivalent.
2. Specifically inspect: gate-relative position encoding (are gate positions normalized by course length? Is the NED frame consistent?), heading representation, and the observation vector index layout.
3. Check if the sim's `RACE_STATUS` message returns gate positions in a different unit or frame on ShadowPC vs laptop.

---

## Recordings inventory

All recordings retained per session instructions.

| Group | Count | Directory prefix |
|-------|-------|-----------------|
| mixer_probe2 | 3 | `data/runs/20260612_034*_mixer_probe2` |
| inc6_standing | 10 | `data/runs/20260612_034852_inc6_standing_f1` … `040128_f10` |
| inc6_bridge (aborted UDP) | 1 | `data/runs/20260612_040236_inc6_bridge_f1` |
| inc6_bridge (5-flight session) | 5 | `data/runs/20260612_040440_inc6_bridge_f1` … `041113_f5` |

---

## Expected vs actual

| Metric | Expected (laptop S17) | Actual (ShadowPC) |
|--------|----------------------|-------------------|
| Finish rate (standing) | 16/16 = 100% | 0/10 = 0% |
| Finish rate (bridge) | 16/16 = 100% | 0/5 = 0% |
| Median lap time | 9.86 s | N/A (no finishes) |
| Per-gate offsets | ≤ 0.28 m | N/A (no finishes) |
| Yaw dither | None (yaw_p95 0.009) | None confirmed |
| Gate contacts | 0 | 0 |
| Thr style | thr_p95 0.061 | Moderate (0.2-0.7), no rail |

Finish rate vs 9.9s prediction: n/a. The 0/15 result is a complete live transfer failure, not a marginal miss.

---

## Session notes

- Sim process crashed mid-session after bridge flight 1 aborted. Relaunched from `C:\Users\Shadow\Downloads\AI-GP Simulator v1.0.3364\AIGP_3364\FlightSim.exe`.
- `mixer_probe2.json` c100_r31 Euler abort is a script limitation (`rate_sysid.py abort_check()` uses ZYX Euler `s.roll`). The phases captured BEFORE c100_r31 (z00_y31_long, init, climb, catch1) are clean. The `c100_r31` settled data (one partial row: [0.99, 0.65, 0.99, 0.65]) appears in the run 1 log but the phase was not completed.
- `debug_obs.jsonl` files are present in all 15 flight run directories. These are the primary artifact for root-cause analysis.
