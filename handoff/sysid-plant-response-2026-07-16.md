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

---

## 6. Battery-4 (2026-07-16) — FULL-RANGE coverage (fills the battery-3 gaps)

Battery-3 sweeps only landed 0.1 & 0.6 (translate → wall ~2.3 s). Rate settles by ~tick 4 (0.1 s), so battery-4 uses **shorter 0.3 s holds** → less attitude windup → survives longer → covers more amplitudes. `sysid/seg4_*.csv`, seg-labeled. **Read window: gains measured on ticks 4–12 (~0.1–0.3 s plateau), skipping the 2–3-tick spin-up.**

### 6a. COMPLETE roll & pitch rate-gain curves (both axes, both signs where landed)
| \|a\| | cmd (rad/s) | roll \|gain\| | pitch \|gain\| |
|---|---|---|---|
| 0.1 | 0.314 | 2.47 | 2.47 |
| 0.2 | 0.628 | 2.46 | 2.46 |
| 0.4 | 1.256 | 2.65 | 2.64 |
| 0.6 | 1.884 | 2.92 | 2.90 |
| 0.8 | 2.512 | 3.07 | 3.04 |

Roll ≡ pitch across the whole range. **Flat ~2.46 at small signal (0.1–0.2), rising expansively to ~3.05 at 0.8.** The small-signal plateau (2.46) sits right on your DiffAero base (2.50). Negatives captured for 0.1/0.2/0.4 (0.8 partial) — all within ~2% of the positive, confirming odd symmetry. (yaw, battery-2 ampsweep: 2.23→2.31→2.48→2.67→2.89 over 0.1→0.8.)

> **Consistency note:** the rate has a slow ~4–9 % upward creep *within* a hold (e.g. roll +0.6: 2.71 @0.1 s → 2.96 @0.5 s). Battery-4 (0.3 s holds) reads the ~0.15–0.3 s plateau; battery-3 (0.5 s holds) read a touch higher on the creep — that's the ~2.9-vs-2.92 spread, not noise. All per-tick values are in the logs; fit the plateau or the creep as you prefer.

### 6b. CLEAN static thrust curve — `20260716_031128_sysid_thrust_hover3g_f1` (SYSID_DONE, full)
3 g applied **first from the v≈0 bootstrap hover** (cleanest static point), mids interleaved with 1 g. Peak accel_z per collective (peak = spin-up complete, lowest velocity):
| collective | thrust (g) | linear label |
|---|---|---|
| 0 (free-fall, batt-3) | ~0 | 0 g |
| 0.2656 (hover) | 1.0 | 1 g |
| 0.398 | ~1.2 | 1.5 g |
| 0.531 | ~2.2 | 2 g |
| 0.664 | ~3.6 | 2.5 g |
| 0.797 | **~5.2** | 3 g |

**Convex, ≈ thrust ∝ collective^1.5** (0.797 = 3× hover-collective → 5.2 g ≈ 3^1.5 = 5.2). At full-ish stick the sim delivers **~1.7× the linear thrust** → confirms the ~2.1× full-stick under-prediction, cleaner than the free-fall run's velocity-smeared 4.6–5.6.

**Velocity-dependence re-confirmed independently:** the 1 g-recovery segments (drone climbing fast) read only **~0.4 g at hover-collective** vs 1 g at v=0 — same collective, less thrust when climbing (inflow). So the thrust map is **f(collective, climb-velocity)**, not static. Fit both from the raw 117 Hz IMU (this run = climbing velocities; battery-3 free-fall-first = falling velocities → together they bracket the velocity range).

### 6c. Battery-4 capture inventory
| program | session | rows | landed |
|---|---|---|---|
| `seg4_roll_fill` | `20260716_030723_sysid_roll_fill_f1` | 95/144 | roll ±0.2, ±0.4, +0.8 |
| `seg4_pitch_fill` | `20260716_031011_sysid_pitch_fill_f1` | 89/144 | pitch ±0.2, ±0.4, +0.8 |
| `seg4_thrust_hover3g` | `20260716_031128_sysid_thrust_hover3g_f1` | 85/85 FULL | static 3/2.5/2/1.5/1 g + velocity |

> ⚠ **§6b static thrust superseded by §7** — the hover-3g "static" mids were measured while the drone was *climbing* (velocity-reduced), so they read LOW. The battery-5 vz-sweeps (below) pass through vz≈0 and give the TRUE static, which is higher.

---

## 7. Battery-5 (2026-07-16) — CLIMB-VELOCITY THRUST SWEEP (inflow lapse)

The one unmodeled piece: thrust falls with climb-rate (1 g rest → ~0.4 g climbing). **vz is not on the wire** (ODOMETRY blocked) → reconstructed as **`vz(t) = −cumsum(accel_z + 9.81)·dt`** from vz≈0 at the post-arrest hover (clean: no roll/pitch → pure-vertical motion → no attitude coupling). Programs `sysid/seg5*.csv` hold a fixed collective while the drone accelerates through a vz range (free-fall reset between collectives to sweep through zero). seg-labeled.

### 7a. Inflow lapse — thrust(g) vs climb-rate vz(m/s), by fixed collective
| collective | @ vz≈0 (static) | mid vz | high vz | lapse slope |
|---|---|---|---|---|
| 0.266 (hover) | 1.0 | 0.47 @ +10 | 0.24 @ +13 | steep near hover |
| 0.398 (1.5g) | ~2.0 | — (slow climb, vz≤+1) | — | — |
| 0.531 (2g) | **3.15** | 2.99 @ +2.8 | 2.64 @ +7.2 | −0.07 g/(m/s) |
| 0.664 (2.5g) | ~4.0 | 3.50 @ +9.8 | 2.60 @ +16 | −0.12 g/(m/s) |
| 0.797 (3g) | ~5.2–5.6 | 3.30 @ +17 | 2.20 @ +22 | −0.17 g/(m/s) |

Thrust peaks near vz≈0 and falls ~linearly with climb-rate; **the lapse slope steepens with collective** (higher disk loading → stronger inflow). The **hover-collective** curve reaches the commander's cited ~0.4 g (≈ vz +11) and keeps dropping — legacy lapse floor 0.78 badly under-models this. (Hover-col high-vz entry, vz +14–18, is spin-down-contaminated from the launch; use the +10–13 tail.)

### 7b. CORRECTED static thrust curve (from the vz≈0 crossings — supersedes §6b)
| collective | TRUE static (vz≈0) | §6b climbing-contaminated |
|---|---|---|
| 0 | ~0 | ~0 |
| 0.266 | 1.0 | 1.0 |
| 0.398 | **~2.0** | 1.2 |
| 0.531 | **~3.15** | 2.2 |
| 0.664 | **~4.0** | 3.6 |
| 0.797 | ~5.2–5.6 | 5.2 |
Convex **≈ collective^1.6** — steeper than §6b thought; the static full-stick over-delivery is *larger* (col 0.797 → ~5.2–5.6 g vs linear 3 g ≈ 1.8×). Fit thrust(collective, vz) = static(collective) · lapse(vz, collective) from the raw 117 Hz IMU across all vz-sweep runs (falling + climbing velocities bracket the range).

### 7c. Battery-5 capture inventory
| program | session | rows | landed |
|---|---|---|---|
| `seg5_thrust_vzsweep` | `20260716_032909_sysid_vzsweep_f1` | 131/131 FULL | 2/2.5/3 g lapse through vz≈0 |
| `seg5b_hover_vzsweep` | `20260716_033817_sysid_hover_vzsweep_f1` | 70/112 | hover-col lapse to 0.4 g |
| `seg5_thrust_vzsweep_lo` | `20260716_034013_sysid_vzsweep_lo_f1` | 52/88 | 1.5 g lapse + static |

**Sim-ops note:** the hover-col sweep launches UP → hit the race gate at low climb (climb_s 0.7); climb_s 1.5 cleared it. Free-fall-first sweeps drop away from the gate → safe at climb_s 1.0.

---

## 8. Battery-6 (2026-07-16) — ⚠ THE 17° START PITCH recontextualizes all thrust data

**The race start block is 17° nose-down-forward, and the sysID commands zero rates → the drone HELD ~17° pitch through EVERY thrust test.** I was implicitly treating it as level. Pilot's direct visual is the ground truth here (this wire is pose-blocked — I can't see attitude in the IMU, and my gyro-integrated attitude estimate had the sign backward until the pilot corrected it).

**What this does and doesn't change:**
- **Rate curves (§1, §5a, §6a): unaffected.** Gyro reads body rate regardless of base attitude.
- **Thrust MAGNITUDE (§6b, §7b): unaffected.** The accelerometer reads thrust along body-Z whatever the pitch, so thrust-vs-collective (convex ~coll^1.6, ~1.8× linear) stands.
- **The inflow lapse (§7): recontextualized.** At 17°, sin17° = 0.29 of thrust is horizontal → the drone drifted **forward** through every "vertical" thrust hold. So the reconstructed `vz` was NOT pure axial climb — it mixed vertical + forward motion. **The lapse data is the pitched-with-forward-drift (racing-ish) regime, not clean vertical.** The lapse's existence + rough magnitude (thrust falls as speed rises) hold; the precise slope-vs-pitch comparison does not (all runs had forward velocity).

**Pitch-controlled probe (open-loop pitch-rate → hold thrust; achieved pitch from gyro magnitude + pilot visual):**
| session | cmd | actual pitch | note |
|---|---|---|---|
| `20260716_040219_sysid_pitch_trulevel_f1` | a_pitch −0.10 ×16t | **~0° (pilot: "levels out")** | but carries forward momentum from the 17° drift |
| `20260716_035550_sysid_pitch_30_f1` | a_pitch −0.10 ×13t | ~4° (near level) | |
| (`seg5` vz-sweeps) | — | 17° | the battery-5 lapse baseline |
| `20260716_035325_sysid_pitch_level_f1` | a_pitch +0.10 ×14t | ~32° forward | forward-drift-contaminated vz |

**Pitch sign (pilot-confirmed, use this):** `a_pitch = −0.10` pitches nose **UP/back toward level**; `+0.10` pitches **down/forward**; ~1.04°/tick at |a_pitch|=0.10; `θ = 17 − cumsum(−gyro_y_raw)·dt`.

**Key limitation:** a clean **pure-vertical** thrust test is not achievable from the 17° start in this confined scene — the start imparts forward momentum that persists even after leveling (pilot: "carries forward momentum, flies up, hits the ceiling"). 

**Recommendation:** model thrust as **f(collective, full 3-D velocity relative to the rotor disk)**, not f(collective, vertical-vz). The battery-3/4/5 lapse data is a valid sample of that function **at ~17° pitch with forward drift** — tag it as such. To isolate pure axial vs edgewise inflow you'd need either a velocity-nulling maneuver (pitch back to kill forward momentum before the hold) or a bench/tethered test; flag if you want me to attempt the former.
