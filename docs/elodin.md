# Elodin practice rig — dev/validation surrogate

The official DCL/Anduril sim isn't released yet. [Elodin's open-source harness](https://github.com/elodin-sys/ai-grand-prix)
(Apache-2.0) is a faithful-enough stand-in to develop the whole stack against **today**:
real Betaflight SITL for the inner loop, deterministic 1 kHz physics, and a camera whose
intrinsics match VADR-TS-002 exactly (640×360, fx=fy=320, cx=320, cy=180, +20° tilt, 30 Hz).

It is **not** the scoring sim. Use it to de-risk perception, estimation, planning, and
control; final validation happens on the official MAVLink sim.

## Where to run it
`elodin` has **no Windows wheel** (verified: `pip install elodin==0.17.2` → "no matching
distribution"), and the rig is Linux/macOS/WSL only. Options, best first:

1. **Azure VM via WSL2** (once provisioned) — A10 GPU does the camera render properly. Same
   box we'll run the official sim on.
2. **Laptop via WSL2 (Ubuntu)** — fine for non-camera work (control/estimation/planning vs
   ground truth). The Intel iGPU may struggle with the FPV render; see the far-plane note.
3. **Adroit** — Linux + GPU, but Slurm/non-interactive; awkward for an interactive sim. Better
   reserved for offline training.

## Setup (Linux/WSL)
```bash
git clone --recurse-submodules https://github.com/elodin-sys/ai-grand-prix.git
cd ai-grand-prix
bash scripts/install_elodin.sh          # elodin + elodin-db v0.17.2 CLI
bash scripts/fetch_betaflight.sh        # betaflight submodule
bash scripts/build_betaflight.sh        # SITL ELF with ENABLE_SIMULATOR_GYROPID_SYNC (~5 min)
uv sync                                  # python deps (elodin==0.17.2, jax, ...)
uv run pytest                            # 35 tests, no runtime needed (<1s) — sanity check
elodin editor sim/main.py               # interactive; or: elodin run sim/main.py (headless)
```

## How our stack plugs in
Elodin loads a contestant module via `RACE_SOLVER=<module>` exposing
`autopilot(update: SensorUpdate) -> RCCommand`. Our integration lives in
[`src/racer/elodin_adapter.py`](../src/racer/elodin_adapter.py) (pure conversions, no
`elodin` import, unit-tested in our normal venv):

- `sensorupdate_to_state(...)` → our `DroneState` (+ ground-truth for eval). Handles
  ENU→NED, FLU→FRD, scalar-last quaternion, and `baro` (altitude) → pressure.
- `frame_from_rgba(...)` → our BGR `Frame`.
- `controlcommand_to_rc(...)` → RC PWM channels (**provisional** — gains/signs need rig
  calibration; see below).

The thin glue solver (`autopilot()` that imports Elodin's `SensorUpdate`/`RCCommand`, calls
the adapter, runs our stack, returns an `RCCommand`) lives on the Linux box and is **TBD until
the controller exists** — there's no closed loop to run yet.

## Gotchas (learned from reading the source)
- **GT discipline.** Elodin hands you ground-truth `world_pos`/`world_vel`; the official sim
  does **not**. The adapter hides GT from `DroneState` by default (`position_ned`/
  `velocity_ned` = `None`) and returns it separately, so the stack is forced to localise from
  vision. Don't flip `expose_ground_truth=True` except for debugging / easy-mode experiments.
  GT is the perfect **oracle to score the estimator** offline.
- **Camera far-plane.** `sim/camera.py` registers the FPV camera with `far=0.65` (metres). A
  0.65 m far plane makes 10 m gates invisible. Patch `far` to ~100 m before any vision work,
  and confirm rendered gates actually appear. (Flag upstream / keep a local patch.)
- **`baro` is altitude, not pressure** (the adapter converts via standard atmosphere).
- **`mag` is the body direction of ENU-North**, normalised.
- **No position/velocity control** — Betaflight angle/acro only. `controlcommand_to_rc` raises
  `NotImplementedError` for POSITION/VELOCITY. The position-setpoint "easy mode" is an
  official-sim-only question.
- **Default config is ANGLE mode** (AUX2), roll/pitch sticks = angle, yaw = rate, AUX1 = arm
  (≥1700). Control-stick gains and **signs** (esp. pitch: `api.py` says <1500 = forward, but
  the baseline's effective sign looks opposite) MUST be calibrated on the running rig.
- **Physics ≠ official sim**: single drag coefficient, no battery sag, generic 5″ quad. Tune
  numbers against the official sim once available; treat Elodin as structural, not exact.
- Elodin independently flags the same **VFoV mislabel** we found (VFoV ≈ 58.72°, prose says 90°
  = actually HFoV).

## Validation loop this unlocks (no official sim needed)
Run a trajectory in Elodin → feed frames through `detector → gate_pose` and IMU/baro/mag
through the estimator → compare the estimate to Elodin's ground-truth pose. That measures
blind-segment drift on our single riskiest subsystem, today.
