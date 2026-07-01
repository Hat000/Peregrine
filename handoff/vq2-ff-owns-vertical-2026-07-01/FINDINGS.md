# VQ2 ff_owns_vertical — the A19c alt-hold bang-bang fix

**Branch:** `vq2-ff-owns-vertical-2026-07-01` (off `db4e016`, pushed to origin — HEAD `e2cdc01`)
**Date:** 2026-07-01 · **Author:** opus-4.8 (commander delegation)

## The fix form (chosen) + why

Confirmed diagnosis: on the state-denied VQ2 wire the decoupled alt-hold
`thrust = hover + kp_alt*(z - z_t) + kd_alt*(vel[2] - vz_t)` had `kp_alt*(z-z_t)` collapse to the
CONSTANT `kp_alt*alt_offset_m` during pursuit (`sp.position_ned` is None; `alt_offset_m=0` here), so
the ONLY time-varying thrust driver was `kd_alt*(vel[2] - vz_t)` with `kd_alt=3.0`. `vel[2] =
nav.velocity_ned[2]` is the estimator's DEAD-RECKONED vertical velocity — pure IMU-accel integration
(no baro, ODOMETRY blocked) — noisy + drifting. Damping the collective against it rail-slammed the
thrust 0.05<->0.60 for the whole active flight -> a net asymmetric climb OVER the acquired gate -> gate
lost -> the 360 yaw-search.

**Chosen form — the principled guardrail-2+3 fix** (a new default-off `Controller.ff_owns_vertical`
flag, the vertical twin of `ff_owns_horizontal`). When ON **and** `sp.accel_ned is not None` (same
trigger as the horizontal twin):

1. **Keep a POSITION loop on the trustworthy z.** `kp_alt*(z - z_target)` on `pos[2]`
   (floor_height-corrected, semi-trustworthy) — unlike `vel[2]`.
2. **Vertical-align works THROUGH a ramping z_target.** `z_target` latched to the current `z` on the
   first ff-owns-vertical tick, then ramped by the commanded rate: `z_target += vz_t*dt` (dt from the
   setpoint sim-time delta). A commanded sink/climb becomes a moving position target the robust loop
   tracks — NOT a velocity-error term on the fictional `vel[2]`. Preserves the A5-blocker-1
   vertical-align intent without the fictional-velocity feedback.
3. **Damp against a TRUSTWORTHY vertical rate.** A low-passed FINITE DIFFERENCE of the floor-corrected
   `z` (`ff_vertical_kd_alt` on the LP fd-vz), NOT raw `vel[2]`. The fd of a BOUNDED corrected z is
   itself bounded, so it can't drift the collective into the rails the way integrated `vel[2]` did; the
   LP keeps per-tick z-jumps from injecting thrust noise. A pure-P loop on noisy z + the sim's
   sense->act lag can relay-ring (`twin.py` `cmd_latency_s`/`thrust_tau_s`), so real damping matters —
   this is why I kept a modest dedicated damping term, not the bare-P fallback.

New knobs: `ff_vertical_kd_alt=0.5` (dedicated damping gain on the fd-vz — NOT the hot `kd_alt=3.0`,
which is tuned for the OFF-path integrated vel[2]) and `ff_vertical_vz_lp_alpha=0.5` (fd-vz low-pass).
Both are default fields; the profile does not need to set them.

## Gate A — offline repro (before/after thrust traces)

Harness: `handoff/vq2-ff-owns-vertical-2026-07-01/repro_vertical_bangbang.py`, feeding the REAL
recorded A19c arc `nav_estimate_a19c.jsonl` (76 ticks) through the decoupled alt-hold.

**vel[2] PROXY CAVEAT:** `velocity_ned` is NOT logged in the arc, and the estimator's dead-reckoned vz
is not directly recoverable. Reconstructed a faithful noisy-vz proxy as the finite difference of the
logged floor-corrected `position_ned[2]` (`diff(z)/dt`). A PROXY, not the true integrated-IMU vz, but
it captures the jumpy/drifting vz and reproduces the rail-slam.

### On the REAL A19c arc (open-loop replay — the no-rail-slam mechanism proof)
```
BEFORE (current code, OFF):  rail lo=41  hi=24  rail-slam flips=19 / 76  std=0.248
  thrust ticks 10..44: 0.600 0.050 0.600 0.224 0.050 0.600 0.268 0.050 0.050 0.050 0.600 0.050 ...
AFTER  (ff_owns_vertical ON, vz_t=0):  rail lo=66  hi=3  rail-slam flips=3 / 76  std=0.129
```
AFTER-on-the-arc pins low because it is an OPEN-LOOP replay of the BROKEN trajectory (z diverges to
-4.5 m while z_target holds) — it proves the rail-slam is gone (flips 19->3), not the hover band.

### On a SYNTHETIC held altitude (a WORKING closed loop — the hover-band proof)
z ~= const -2.0 m + small noise, level attitude, 40 ticks @ 12 Hz:
```
OFF (kd_alt=3.0 vs the drifting/noisy vz):  flips=17  mean=0.352 std=0.254   (RAIL-SLAMS)
ON  (ff_owns_vertical):                     flips= 0  mean=0.261 std=0.077   median=0.267 ~= hover 0.2656
```
Graded vertical-align tracking (held z, modest vz_t): `descend(vz_t=+0.3)=0.052 < hold=0.266 <
climb(vz_t=-0.3)=0.589` — sign-correct.

**Gate A: PASS.** Bang-bang reproduced on current code (arc 19 flips; held 17 flips); fix holds thrust
around hover (median 0.267 ~= 0.2656) with zero rail-slam, holds at vz_t=0, tracks descent/climb.

## Gate B — green_gate
```
1. invariants:  GREEN  OFF==inc7 AST parity . +L sign-faithfulness . VQ1 import guard
2. sentinel:    GREEN  collected 1597 >= baseline 1091
3. full suite (unmapped tools/ flight-stack changes forced --full):
                1 failed, 1523 passed, 73 skipped in 514 s
   the 1 fail = tests/test_diagnose_session.py::test_real_bundles_diagnose_end_to_end
              = the KNOWN pre-existing handoff-slimming fixture gap (NOT mine).
```
(green_gate widened to FULL because the flight stack has `tools/blender_pipeline/*` + `tools/sysid/*`
changes vs main's merge-base — pre-existing flight-stack lineage, NOT in my diff. My diff vs db4e016 =
the 5 files below.)

**Gate B: PASS** (GREEN-except-the-ONE-known-unrelated-RED).

## Gate C — pinning test
`tests/test_ff_owns_vertical.py` (6 tests) — FAILS on old code (no field / no profile override), PASSES
on new: OFF-path rail-slams on a drifting vz (the bug); ON holds around hover with <=1 flip, far calmer
than OFF; sink<hold<climb; OFF-path thrust == the EXACT legacy formula tick-by-tick; default OFF;
vq2_case_c ships it ON. **Gate C: PASS.**

## The seam (default-off, byte-identical)
- `Controller.ff_owns_vertical: bool = False` — OFF path reads `vel[2]` with `kd_alt` EXACTLY as today
  (VQ1 / case-A / every offline path byte-identical; pinned by `test_off_path_byte_identical` + the
  OFF==inc7 AST parity invariant + the VQ1 constants guard).
- `deploy_profile.vq2_case_c` `controller_overrides` sets `ff_owns_vertical=True` (beside
  `ff_owns_horizontal=True`) — the ONLY place it is turned on.

## Files changed (vs db4e016)
```
src/racer/controller.py               +76  (flag + _ff_owns_vertical_thrust helper + branch)
src/racer/deploy_profile.py           +10  (vq2_case_c: ff_owns_vertical=True)
tests/test_ff_owns_vertical.py       +146  (pinning test)
handoff/.../repro_vertical_bangbang.py +192 (offline repro harness)
handoff/.../nav_estimate_a19c.jsonl   +76  (recorded arc, for reproducibility)
```

## Residual fly-risk (for the NEXT fly, A20/A21)
- **Offline-validated, not yet flown.** No-rail-slam + hover-band + tracking proven offline on the real
  arc + synthetic held-z; the CLOSED-LOOP gains (`kp_alt=2.0`, `ff_vertical_kd_alt=0.5`, lp=0.5) still
  need a live confirm. `ff_vertical_kd_alt` may want a small live retune if the alt channel is
  soft/ringy under real sim lag — a dedicated knob, safe to tune.
- **z_target latch** happens on the first ff-owns-vertical tick and is NOT reset if pursuit drops out
  and re-engages (z_target persists across the gap — usually desirable). A re-latch per pursuit
  re-entry is a future refinement (not needed for the A19c fix).
- **Proxy caveat** (above): the fix removes the dependence on ANY dead-reckoned vz, so the
  proxy-vs-true distinction does not change the conclusion — but the live fly is the real confirmation.
- Nice-to-have not done: per-tick `vel[2]`+`vz_t` logging in fly_rl's nav-estimate logger (deferred).
