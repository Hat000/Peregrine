# VQ2 plant sysID — open-loop rate-response capture (2026-07-16)

**For:** RL-commander (DiffAero plant tune / retrain).
**Headline:** the live VQ2 sim's inner rate loop delivers **~2.7× the commanded body rate**, and the gain is **amplitude-progressive** (≈2.2× for small commands rising to ≈2.9× for large ones), **symmetric and consistent across roll/pitch/yaw**. Training assumed wire-scale **1.0×**. If DiffAero does not reproduce this, it is the strongest single explanation for the chronic deploy over-banking / roll-runaway.

---

## 1. The finding

### 1a. Yaw amplitude sweep (run #1, `SYSID_DONE`, full 399-row capture)
Yaw steps ±0.1/0.2/0.4/0.6/0.8 held 0.5 s each, drift-free (spins in place). Perfectly **odd-symmetric** (+/− identical to 3 d.p.) → a real static **expansive nonlinearity**, not noise:

| \|a_yaw\| | cmd_wz (rad/s) | settled \|gyro_z\| | **\|gain\|** |
|---|---|---|---|
| 0.1 | 0.314 | 0.70 | **2.23×** |
| 0.2 | 0.628 | 1.45 | **2.31×** |
| 0.4 | 1.256 | 3.11 | **2.48×** |
| 0.6 | 1.884 | 5.02 | **2.67×** |
| 0.8 | 2.512 | 7.26 | **2.89×** |

### 1b. Cross-axis (doublet peaks at |a|=0.6)
| axis | run | \|gain\| @ \|a\|=0.6 |
|---|---|---|
| roll | #3 | **2.74×** |
| pitch | #4 | **2.71×** |
| yaw | #1 | **2.67×** |

All three axes over-deliver ~2.7× **identically** at the same command amplitude → the nonlinearity is a shared property of the inner rate loop, not axis-specific.

**Implication:** a single scalar `cmd_rate_scale` cannot correct this — the effective gain rises with command magnitude, so the hardest maneuvers (e.g. the 50–61° g2→g3 bank) are over-rotated *most*. Match the curve in the DiffAero plant, or retrain against the corrected plant.

**Caveat:** the amp-sweep holds are 0.5 s; part of the amplitude-rise could be second-order overshoot rather than pure static gain. The **doublet + chirp** runs (#3/#4/#5) are included specifically to separate transient from steady-state — please fit those.

---

## 2. How to read the captures

**Two files per session** under `data/runs/<session>/`:
- `sysid_vq2_log.csv` — 40 Hz loop log. Columns: `k, phase(boot|prog), sim_time_ns, a_thrust,a_roll,a_pitch,a_yaw, cmd_wx,cmd_wy,cmd_wz, cmd_thrust, collective, gyro_x,gyro_y,gyro_z, accel_x,accel_y,accel_z`. Use `phase=="prog"` rows (bootstrap is `boot`).
- `sysid_vq2_imu_raw.csv` — raw 117 Hz HIGHRES_IMU straight off the wire: `sim_time_ns, gyro_x/y/z (RAW), accel_x/y/z`. Use this for Bode/bandwidth fits.

**Action → wire mapping** (bounds PINNED to [0,5]/[±3.14]; a loaded ego ckpt's 3.765 g thrust bound is bypassed for the sysID):
- `cmd_wx = 3.14·a_roll`, `cmd_wy = −3.14·a_pitch`, `cmd_wz = 3.14·a_yaw`  (FLU→FRD = [1,−1,1])
- `normed_thrust = 2.5·(a_thrust+1)`; `collective = clip(normed·0.2656, 0, 1)`; hover (1 g) ≈ collective 0.2656.

**SIGN FOOTGUN (A9):** raw wire gyro is **sign-inverted** vs the command (`gyro_sign=(−1,−1,−1)`). So raw `gyro_z` is negative for a positive `cmd_wz`. In code-FRD (after the sign flip) the ratio is **+2.7×**. The `gyro_*` columns in the log are RAW (pre-flip), matching the imu_raw file. Apply the flip before comparing to command direction.

**No ODOMETRY:** the deploy wire (3379) does not emit ODOMETRY/pose — response is **gyro + accel only** (no position; do not expect double-integrated pose).

---

## 3. Capture inventory

Battery-2 (this session, `sysid/seg2_*.csv` programs):

| # | program | session | rows (prog/total) | outcome | content |
|---|---|---|---|---|---|
| 1 | `seg2_yaw_ampsweep` | `20260716_014656_sysid_yaw_ampsweep_f1` | 399/399 | SYSID_DONE | full yaw gain-vs-amplitude (§1a) |
| 2 | `seg2_thrust_curve` | `20260716_015043_sysid_thrust_curve_f1` | 175/280 | ceiling crash | thrust→accel_z at 1 / 1.5 / 2 / 2.5 g (missed 3 g + free-fall) |
| 3 | `seg2_roll_dyn` | `20260716_015421_sysid_roll_dyn_f1` | 104/224 | wall ~2.6 s | **full** ±0.6 roll doublet + low-freq chirp |
| 4 | `seg2_pitch_dyn` | `20260716_015652_sysid_pitch_dyn_f1` | 97/224 | wall ~2.4 s | **full** ±0.6 pitch doublet + low-freq chirp |
| 5 | `seg2_yaw_bandwidth` | `20260716_015754_sysid_yaw_bw_f1` | 126/220 | wall ~3.1 s | yaw chirp ~0.5→7 Hz (hi-freq tail cut) |

Battery-1 (earlier this cycle, `sysid/seg_*.csv` — single-level steps, corroborate the settled gain):

| program | session | content |
|---|---|---|
| `seg_yaw_thrust` | `20260716_012911_sysid_yawthrust_f1` | yaw+thrust **full** (483 prog rows), single-level steps |
| `seg_pitch` | `20260716_013241_sysid_pitch_f1` | pitch step |
| (roll step) | `20260716_011527_sysid_vq2_f1` | roll step |

**Why the lateral axes are partial:** open-loop excitation with no position hold drifts the drone into a wall in the confined race scene in ~1–3 s. Yaw (spins in place) and thrust (vertical) don't translate → they capture longest. Roll/pitch translate → the doublet lands but the high-freq chirp tail is clipped. All doublets captured the full ±0.6 amplitude before impact.

---

## 4. The ask

1. **Overlay vs DiffAero.** Command the same battery to the DiffAero training plant. If DiffAero also yields ~2.2→2.9× → it's a shared convention (no bug, deploy matches training). If DiffAero is ~1:1 → this ~2.7× is a **train↔deploy mismatch** = the bug. This is the decisive test.
2. If it's a mismatch: reproduce the **amplitude-progressive** curve (not a single scalar) in the DiffAero plant, or retrain against it.
3. Reproduce/extend: tooling is on this branch — `rl/fly_rl.py --sysid-replay sysid/<prog>.csv ... --no-virtual-flip --seeker-detector red_glow`, extractor `scripts/sysid_tlog_to_imu_raw.py`. `--no-virtual-flip` is mandatory (the FLU→FRD map is sign-correct only with flip OFF).

Tooling origin: `claude/ego-deploy-2026-07-09` (commits `0c07074`, `e3c5060`), merged onto this branch.
