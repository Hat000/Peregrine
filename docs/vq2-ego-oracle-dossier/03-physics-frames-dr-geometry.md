# 03 — Physics, Frames, Domain Randomization, Geometry

## The simulated plant

Dynamics adapter: `rl/diffaero_dynamics.py`, class `PeregrinePlantDynamics` (`dynamics.name =
peregrine_plant`). It wraps a **system-ID'd** Peregrine plant (`racer.rl_plant`) — i.e. the nominal
(DR-off) dynamics are our best estimate of the real drone. 13-dim state `[p, q(xyzw), v, w]`,
action_dim 4 `[normed_thrust, roll, pitch, yaw]` (CTBR / body-rate). Internally NED; DiffAero world
is Z-up. `dynamics.controller.max_normed_thrust = 3.765`, `g = 9.80665`.

## Domain randomization (all gated by `dynamics.dr`)

`self._dr_enabled = bool(cfg.dr)`. **Every** DR component is `_dr_enabled and <flag>`, so
`dynamics.dr=False` turns the **entire** DR path off at once. Components (training used all ON):

- **`dr_aero`** — forces the quad-drag + **convex-collective aero** (parasitic-lift channel) ON. This
  is a *modeled aerodynamic effect that is otherwise absent from the base integrator.* With `dr` off,
  aero is off → the lift/thrust response is materially different from what the policy trained on.
- **`dr_mixer`** — motor-mixer randomization (requires `dr_aero`); per-env mixer params resampled at
  reset; runs the clipped motor mean through a knot table.
- **`dr_lapse`** — thrust-lapse deficit (requires `dr_aero`).
- **`dr_force_bias`** — a per-env external force bias. **Verified zero-mean:** `bias = (isotropic
  random unit direction from randn) × (magnitude ~ U[0, bias_max])` in world/NED, plus a random
  speed×tilt *regime bin* (`FORCE_BIAS_SPEED_BINS`, `FORCE_BIAS_TILT_BINS_DEG`) gating when it's
  active. Because the direction is isotropic and zero-mean, **nominal (bias=0) is the mean of the
  distribution** — a policy trained across it should handle nominal, and does *not* have a systematic
  bias to "compensate" (we initially hypothesised it did; that was wrong).
- **`dr_latency`** — a per-env action ring buffer applying a randomized actuator delay in
  `{min..max}` control steps. Training used `dr_latency_min_steps=1, dr_latency_max_steps=3` → delay
  ∈ {1,2,3}. **With `dr` off, latency = 0 — which is OUT of the training distribution** (the policy
  never saw zero delay). This is a prime suspect for the DR-off overshoot (see [07]).

### The consequence for "deployment" evaluation

`dynamics.dr=False` is **not** a faithful deployment proxy: it removes modeled aero *and* pushes
latency out-of-distribution. The real plant has aero and some latency, so the **DR-ON** evaluation is
the closer proxy. There is a middle option we did **not** run: "dr_nominal" (DR code path ON but every
per-env DR tensor pinned at its nominal, force-bias zeroed) — the branch `check_diffaero_gate.py`
exercises — which would keep the aero structure at nominal params. **We have no evaluation at the
actual competition-sim dynamics, and we don't know exactly how they differ from our nominal plant.**

## Coordinate frames & conventions (a nest of sign traps)

- **DiffAero world = Z-up. Plant = NED (Z-down).** Bridged by `_FLIP = [1, -1, -1]` (keep x, negate
  y,z). The injected gate layout is expressed in Z-up.
- **Gate frame:** +x = exit/down-course = world `[cos yaw, sin yaw, 0]`; z stays world-z. `is_passed`
  is a plane crossing on gate-local +x. `world_to_gateframe(d, yaw)`: `x = c·dx + s·dy`,
  `y = −s·dx + c·dy`, `z = dz`.
- **Tail-first:** training spawns tail-first w.r.t. the gate (`spawn_yaw = gate_yaw + π`). The deploy
  path `fly_rl.py` runs the policy behind a **virtual π body-z flip** to compensate. This CTBR/VQ1
  legacy sign config is a **self-consistent alias — do NOT "fix" it.**
- **Camera flip (RC1):** the emulated camera looks *backward* (along −body-x), default-ON in
  `peregrine_racing_ego.py`. Only the emulated camera is rotated; control frame, `rel_pos`, velocity,
  rates keep the unflipped body frame.
- **ODOMETRY quaternion is R_y(π)-conjugated** on the live wire (a known deploy-side gotcha).

## Gate & body geometry (the numbers that define "thread")

- Inner opening **1.5 m** → **half-opening 0.75 m** (the aperture; `gate_half_opening_m`).
- Outer physical frame **2.72 m** → **half 1.36 m** (`gate_half_outer_m`). Frame band = L-inf in
  `[0.75, 1.36]` = a clip = contact = invalid.
- **Body radius 0.28–0.38 m** (`body_radius_lo/hi`), randomized per env. The contact bands are
  **inflated by the body radius**: `half_inner_eff = 0.75 − r`, `half_outer_eff = 1.36 + r`. So the
  **effective clean-pass window is ~0.75 − 0.33 ≈ 0.42 m** (centre-of-mass must land inside ~0.42 m).
- Gate spec (from the VQ2 technical spec PDF, re-confirmed this session): height 2.72 m, opening
  1.5 m, frame depth ~0.26 m. Drone 280×280×160 mm (→ the 0.28–0.38 m halo is conservative).
- `frame_depth_m` makes the frame band *material over gate-frame |x| ≤ depth* (a slab, not a plane) —
  the inc7 contact contract.

## Sensor wire (VQ2, live-confirmed) — why the obs is what it is

- **No magnetometer, no barometer.** HIGHRES_IMU `fields_updated=63` = accel + gyro only. **Yaw and
  altitude must come from vision.** Any code path assuming mag/baro is dead.
- 30 Hz JPEG camera; HIGHRES_IMU ~117 Hz; RACE_STATUS 4 Hz (**includes `active_gate_index`**);
  ATTITUDE/LPN/ODOM/GLOBAL/GATE_INFO blocked in both training and competition (byte-identical).
- Track is deterministic per load (not randomized); training wire == competition wire.

This is the justification for the whole egocentric design: the drone genuinely cannot see its world
position or the gate normal reliably; it has body-frame vision geometry + IMU. The RL obs is a
faithful (noised) model of that.
