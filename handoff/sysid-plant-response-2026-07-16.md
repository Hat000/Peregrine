# VQ2 plant sysID — open-loop rate-response capture (2026-07-16)

**For:** RL-commander (DiffAero plant tune / retrain).
**Headline:** the live VQ2 sim's inner rate loop delivers **~2.7× the commanded body rate**, and the gain is **amplitude-progressive** (≈2.2× for small commands rising to ≈2.9× for large ones), **symmetric and consistent across roll/pitch/yaw**.

> **FRAMING CORRECTED (commander overlay, 2026-07-16):** the gap is NOT "training assumed 1.0×." DiffAero already models the base over-delivery (flat ~2.50× roll/pitch, 2.23× yaw). **The real mismatch is FLAT (DiffAero) vs EXPANSIVE (VQ2)** — they agree at small/mid command and diverge at high command, which is exactly why aggressive banks (g2→g3) over-rotate in deploy but not in training. Battery-3 (below) directly measured the per-axis roll/pitch curves and a free-fall-first thrust curve to close this. Fix being wired: `super_rate_s=0.315` + small-signal `rate_gain=[2.222, 2.198, 2.164]` reproduces the curve to ±0.2%.

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

Tooling origin: `claude/ego-deploy-2026-07-09` (commits `0c07074`, `e3c5060`, `a2f1bc8`), merged onto this branch.

---

## 5. Battery-3 (2026-07-16) — per-axis roll/pitch curves + free-fall-first thrust

Programs `sysid/seg3_*.csv`, priority-ordered (0.1 & 0.6 first, sign-paired to bound drift). **Each row now carries an explicit `seg` label** (new `seg` column in `sysid_vq2_log.csv`; loader reads an optional 6th program column). Same footguns (raw-gyro A9 sign-flip, gyro+accel only, `--no-virtual-flip`, clamps off).

### 5a. Roll & pitch rate-gain curves (directly measured, no longer inferred)
| axis | \|a\|=0.1 (small-signal) | \|a\|=0.6 (high) | session |
|---|---|---|---|
| roll | **2.47×** (cmd_wx ±0.314) | **2.92×** (cmd_wx +1.884) | `20260716_024734_sysid_roll_sweep_f1` |
| pitch | **2.47×** (cmd_wy ∓0.314) | **2.90×** (cmd_wy ∓1.884) | `20260716_025059_sysid_pitch_sweep_f1` |
| yaw (battery-2) | 2.23× | 2.67× | `…_014656` |

Roll & pitch are **identical** to each other, and their small-signal points (2.47) sit right on your DiffAero base (2.50). Both rise expansively to ~2.9 at \|a\|=0.6 — **flat-vs-expansive confirmed on all three axes, now with direct roll/pitch points** (previously one doublet each). Both crashed into the wall ~2.3 s in (roll/pitch translate); the front-loading captured the low + high priority points cleanly before impact. Missing 0.2/0.4/0.8/−0.6 — re-flyable if you want denser curvature, but the shape is established.

### 5b. Free-fall-first thrust curve — session `20260716_025331_sysid_thrust_ff_f1` (96/112, both endpoints)
Order: free-fall(0 g) → 3 g → 2.5/2/1.5 g. accel_z is body-frame specific force → thrust/weight in g.
- **Free-fall (collective 0) → ≈0 g** (weightless ✓ — confirms 0 collective = 0 thrust).
- **Full-ish stick (collective 0.797, my linear "3 g" label) → 4.6–5.6 g MEASURED.** The sim delivers far more than a linear collective→g map (which predicts 3 g). **This directly supports the ~2.1× full-stick under-prediction** — if training's map is the legacy under-scaled one, deploy gets ~1.5–1.9× more thrust than trained → a second contributor to deploy over-acceleration, on top of the rate expansion.
- **NEW: thrust is climb-velocity dependent (inflow).** Holding collective 0.797 constant, accel_z ramps to −5.6 g (spin-up, low velocity) then **decays to −3.9 g over 0.4 s** as the drone accelerates upward. A static thrust map misses this ~30% falloff. Worth reproducing/checking in the DiffAero aero model.
- Clean static-ish points: (col 0, 0 g), (col 0.2656 hover, 1 g — from the zero-velocity bootstrap hover), (col 0.797 peak, ~5.6 g) → a **convex** thrust curve.
- **Caveat:** an open-loop drop-test can't isolate *static* thrust at high collective (the drone is always moving vertically), and the descending mids (2.5/2/1.5 g) are carryover+velocity muddied. Use battery-2's **ascending** `seg2_thrust_curve` (`…_015043`, clean 1/1.5/2/2.5 g) for the low/mid static points, and this run for the 0 g + full-stick endpoints and the velocity-dependence. Best fit = velocity-dependent thrust from the raw 117 Hz IMU.

### 5c. Battery-3 capture inventory
| program | session | rows | landed |
|---|---|---|---|
| `seg3_roll_ampsweep` | `20260716_024734_sysid_roll_sweep_f1` | 93/300 | roll ±0.1, +0.6 |
| `seg3_pitch_ampsweep` | `20260716_025059_sysid_pitch_sweep_f1` | 90/300 | pitch ±0.1, +0.6 |
| `seg3_thrust_curve_ff` | `20260716_025331_sysid_thrust_ff_f1` | 96/112 | 0 g + 3 g endpoints + mids |
