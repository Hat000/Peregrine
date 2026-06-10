# VQ2 follow-ups from the live re-fly (ShadowPC, 2026-06-09 GUI observations)

Logged from the pilot's direct GUI read during the 4× launch-ramp verify (all CLEAN 6/6 — these
do NOT affect VQ1 validity; they are time/polish for VQ2). VQ1 is banked; none of these is shipped
blind. Ordered by how much they matter.

## (A) Start YAW-spin — a real bug (VQ2-fatal, VQ1-cosmetic)

**Observed:** 3 of 4 runs (#1, #2, #4) — after the takeoff hover the drone yawed ~180° to "look
behind" (nose toward the start), flew on for ~1.5 s, then yawed back to face the gates. Confirmed
from run #1 ODOMETRY (yaw, relative to control engaging):

```
 t=1.11s  yaw -179°  pitch  +2°   facing the gates
 t=1.27s  yaw  -99°               starts spinning
 t=1.43s  yaw   -8°  pitch +15°   facing BACKWARD (+X, toward start)
 t=1.6-2.4 yaw  ~-5°  pitch +43°  flies on, nose pointed behind it
 t=2.86s  yaw +179°  pitch -21°   swung back to face the gates
 t=3.5s+  yaw ±180°  pitch  -8°   settled, normal forward flight
```

**Why it's invisible in the trajectory:** the decoupled CTBR controller translates via WORLD-frame
tilt — yaw is cosmetic — so the body can point backward and still fly to the gate (it compensates
with a +43° "backward" pitch). Position stayed clean → valid 6/6, and the position-only telemetry
never showed it. The pilot's GUI read caught it.

**Mechanism (hypothesis):** a geometric-attitude-law instability at LARGE TILT near the yaw=±180°
wrap. As the launch lean ramps pitch up past ~10-15° while yaw sits at ±180°, the rotvec attitude
error `R_cur^T R_des` develops a spurious yaw component and the nose runs 180° around, then
recovers. SEPARATE from the tilt-step tumble the launch ramp fixed — likely pre-existing, just
masked by the tumble on 1.0.3364.

**Impact:** cosmetic for VQ1 (valid pass). **VQ2-fatal**: a forward camera is blind / pointed
backward for ~1.5 s right at launch.

**Fix path (twin-first, cheapest → cleanest):**
1. **Reproduce offline first** — `scripts/twin_launch_phase_sweep.py` spawns LEVEL at exactly π and
   did NOT show it; seed the real **−17.8° pad pitch + the ±π yaw wrap** (both #4 spawned +π, #1-3
   −π) to reproduce, then tune against it.
2. Gentler launch accel so pitch stays modest during the wrap-sensitive window (tuning).
3. A dedicated yaw-rate channel decoupled from the tilt rotvec (small controller change).
4. The RL policy (Phase 2) subsumes it.

## (B) Decelerates before every gate (VQ2 time)

**Observed:** the drone slows on approach to each gate.

**Cause:** velocity-targeting geometry — along-track desired speed = `kp_pos·(dist-to-carrot)`
capped at `max_speed`, with `kp_pos=0.6`, `lookahead=5 m`. Within ~10 m of the carrot (≈5 m before
the gate) the commanded speed ramps below the cap → it decelerates into each gate.

**Fix path:** bigger `lookahead` / higher `kp_pos` reduces it — but measured TRADEOFF: more speed
degrades cross-track centring (memory: `max_speed 8 → 0.39 m`, 10 worse), and g1 is already
marginal. Belongs to the VQ2 racing-line / RL work, not a blind bump.

## (C) Speed capped ~20 km/h — could go faster (VQ2 time)

**Cause:** `FAITHFUL_TUNED_GAINS max_speed=6 m/s` (21.6 km/h) / `cruise_speed=8`.

**Fix path:** raising them is a config one-liner, but NOT free: measured `max_speed 8 → 0.39 m`
miss, `10` worse, vs the 0.75 m half-opening; g1 already ~0.59-0.8 m. **VQ1 is pass/fail (speed
irrelevant)**; for VQ2 (fastest-valid-time) this is THE lever — but it's the racing-line/RL effort
with the centring tradeoff in view, not a cap bump in isolation.

---
**Net:** VQ1 (pass/fail) is banked and robust. (A) is a genuine bug to fix before VQ2 vision (do
it twin-first); (B)/(C) are racing-line/RL tuning where speed trades against centring.
