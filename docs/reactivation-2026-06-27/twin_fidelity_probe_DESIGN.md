# Twin Fidelity Probe — Design Spec

*Answers: "Does our diffaero twin reproduce VQ2's dynamics faithfully enough for RL training to transfer?"*

Status: **DESIGN ONLY** — this spec must be reviewed before any code is written.
The measurements it defines feed the go/no-go gate for RL-on-twin training.

Related prior art: `scripts/frame_residual_report.py` (plant-force residual + R_y(π) canary),
`scripts/verify_bundle.py` (stream presence), the system-id spike (#37 numpy↔torch byte-faithful
emulator-fidelity check), and `fit_twin.py` (offline plant-parameter estimation).

---

## 1. Framing and the Twin's Role

Our **diffaero twin** (`racer.rl_plant`, `PlantParams`, `plant_step`) is the environment
model in which RL trains.  Transfer happens if VQ2's plant dynamics are inside the twin's
DR (domain randomisation) envelope.  If they're not, the policy learns the wrong physics
and gate-4 reach/pass-rate gains evaporate on the real sim.

The twin is an analytical model with measured constants (`COLL_MAP_ACCEL_MEASURED`,
`QUAD_DRAG_C2_MEASURED`, `MIXER_*_MEASURED`, etc.) calibrated on VQ1.  VQ2 may use
different physics parameters, a different flight controller model, or a different IMU
driver.  The probe answers this systematically.

The central residual to minimise is the **velocity residual** — the difference between
the twin's predicted NED velocity and VQ2's ground-truth NED velocity, given the same
commanded inputs.  Velocity is chosen because:
- It integrates all plant forces (thrust, drag, gravity, controller) into one observable.
- It is directly measurable from `LOCAL_POSITION_NED.{vx,vy,vz}` (if Training exposes it)
  or from IMU double-integration (Competitive fallback, higher noise).
- `frame_residual_report.py` already implements the velocity-FD → force-residual pipeline;
  this probe extends it into a **commanded-input replay**.

---

## 2. Tier Structure (gated by C5 finding on load day)

### BEST TIER — Training leaks GT (LOCAL_POSITION_NED + ODOMETRY present)

**What's available:** world-frame GT position + velocity (LPN), attitude quaternion (ODO),
angular rates (ODO), IMU specific-force (HIGHRES_IMU), commanded inputs (the outputs of
our policy or a fixed test command sequence).

**Approach: commanded-input replay against GT trajectory.**

```
record: commands(t), IMU(t), LPN_vel(t), ODO_quat(t)

For each segment [t0, t0+N*dt]:
  seed twin at (pos_GT, vel_GT, q_TRUE, omega) from VQ2 at t0
  replay recorded commands through twin for N steps
  compare twin.vel vs LPN_vel (and twin.q vs ODO_quat)
```

This is a direct extension of `frame_residual_report.py`'s `replay()` function, which
already does 27-tick open-loop velocity comparison.  Key extension: vary the command
sequence (hover, roll sweep, pitch sweep, full-speed cruise) to cover the policy's
operating range rather than just one incidental segment.

**Residual metrics:**
- `delta_vel_NED` (m/s): twin-predicted minus GT velocity after N=27 ticks (~0.9 s)
- `delta_vel_norm` (m/s): `|delta_vel_NED|`
- Per-axis breakdown: N (thrust coupling), E (mirror canary), D (drag + gravity)
- `delta_att_deg`: attitude quaternion angle error (axis-angle) after N ticks

**Go/no-go thresholds:**
- `delta_vel_norm` p90 ≤ 1.5 m/s at N=27 ticks: **twin ADEQUATE for RL**
  (rationale: 1.5 m/s over 0.9 s ≈ 1.7 m position error; gate is 1.5 m wide; policy's
   built-in robustness from DR should absorb this)
- `delta_vel_norm` p90 > 3.0 m/s: **twin INADEQUATE — re-calibrate or widen DR**
  (rationale: 3 m/s over 0.9 s ≈ 2.7 m; exceeds the gate half-width; the policy cannot
   be expected to bridge this gap via DR alone)
- Between 1.5–3.0 m/s: **marginal — widen DR envelope and re-test**

**Plant calibration (if twin is inadequate):**
Run `fit_twin.py` on the VQ2 recording (it already fits `PlantParams` by minimising the
velocity residual over recorded segments).  The key parameters likely to shift between
VQ1 and VQ2: `COLL_MAP_ACCEL_MEASURED` (thrust map), `QUAD_DRAG_C2_MEASURED` (drag),
`MIXER_*_MEASURED` (attitude-rate bandwidth).

**What this tier can validate:**
- Full position/velocity/attitude state residual
- Thrust map and drag model accuracy
- Attitude-rate bandwidth (via attitude residual)
- DR envelope coverage check (does the real dynamics stay inside our DR?)

---

### FALLBACK TIER — Competitive only (HIGHRES_IMU + vision + commands; no GT pos/vel)

**What's available:** raw IMU (`accel_body`, `mag_body`) from HIGHRES_IMU; our own
commanded inputs; 30 Hz JPEG frames (vision-derived gate-relative positions); NO world-frame
GT position or velocity.

This is harder.  We cannot compare `vel_twin` to `vel_GT` directly.  Two sub-approaches:

#### Sub-approach A: IMU-response residual

Commanded input `u(t)` → twin predicts `a_body_twin(t)` (specific force in FRD body frame).
VQ2 delivers `a_body_real(t)` via HIGHRES_IMU.  The residual `a_body_twin - a_body_real` is
the direct force-model error in the body frame.

This is essentially what `frame_residual_report.py` already computes (the TRUE-attitude
force residual), but using the COMPETITIVE wire (no GT attitude, so attitude must come from
our AHRS filter rather than ODO.q).  The AHRS-attitude error contaminates the residual —
this makes the fallback a lower-confidence check.

**Residual metric:** median body-frame accel residual per axis (m/s^2) over a 5-second
hover + sweep sequence.  Threshold: |median residual| ≤ 0.5 m/s^2 per axis (small vs g=9.8).

**Complication:** without GT attitude, the "TRUE attitude" used in `frame_residual_report`
is replaced by our AHRS estimate.  A large residual could be twin error OR AHRS error —
the two are conflated.  Run AHRS attitude bias check (C9 / `sample_attitude_bias`) first to
bound the AHRS error before interpreting this residual.

#### Sub-approach B: Vision-derived gate-relative trajectory match

Use the C2 gate-relative estimator chain to build a gate-relative trajectory
`t_cam_gate(t)` from PnP on 30 Hz frames.  Replay the same commands through the twin
starting from the first fix; compare twin's gate-relative position to the PnP-derived track.

This is less precise (PnP noise ≈ 0.1–0.2 m per fix) but captures the gross dynamics
(does the drone move as the twin predicts, trajectory-wise?).

**Residual metric:** gate-frame position RMS at each PnP fix time, over 10+ consecutive fixes.
Threshold: RMS ≤ 0.3 m (≈ 1.5× PnP noise; a clean twin should do ≈ 0.15 m).

**What the fallback tier can and cannot validate:**
- CAN: gross force-magnitude error, thrust-to-weight ratio, drag sign
- CAN: trajectory shape (does the twin curve the same way as the real drone?)
- CANNOT: precise drag coefficients (noisy IMU integration, AHRS-attitude contamination)
- CANNOT: attitude-rate bandwidth (no GT angular rate in Competitive)
- CANNOT: feed plant calibration (`fit_twin.py` needs GT velocity)

---

## 3. Inputs, Tools, and Code Structure

### Inputs (BEST tier)
- `mavlink.tlog` from a Training-mode recording (produced by `record_session.py`)
- Commands log: either `debug_obs.jsonl` (if running the RL policy with `--debug-obs`)
  or a purpose-built command-sequence sweep (constant collective, roll/pitch steps,
  cruise). The command sweep is preferable for calibration because it covers known
  operating points cleanly.

### Inputs (FALLBACK tier)
- `mavlink.tlog` from a Competitive-mode recording (HIGHRES_IMU only)
- `video.bin` + `video_index.jsonl` (30 Hz frames for PnP)
- Commands log (same as above)

### Implementation map
The harness should be a NEW script `scripts/vq2_loadday/twin_fidelity_probe.py` that:

1. **Loads** data via `racer.recording.RecordingReader` (already handles tlog + video).
2. **Decodes GT** from `iter_mavlink()` into arrays (reuse `frame_residual_report.load()`
   for the `debug_obs.jsonl` path; or add a `load_from_tlog()` variant for tlog-native).
3. **Seeds** `PlantState` from GT at each of K evenly-spaced anchor ticks.
4. **Replays** `plant_step()` for N=27 ticks per anchor using the recorded `rate_frd` +
   `collective` commands (already in `debug_obs.jsonl`; or from the tlog for BODY_RATE
   SET_ATTITUDE_TARGET messages).
5. **Computes** per-anchor residuals; reports p50/p90/p99 by speed × tilt bin.
6. **Runs** the R_y(π) mirror canary (import from `diagnose_session._frame_residual()`).
7. **Emits** a JSON verdict: `{"tier": "BEST"|"FALLBACK", "p90_vel_residual_ms": ...,
   "go_nogo": "GO"|"MARGINAL"|"NO_GO", "detail": {...}}`.

### Dependencies
- `racer.recording.RecordingReader` — for tlog + video index
- `racer.rl_plant.plant_step`, `PlantState`, `PlantParams` — the twin
- `racer.frames.ODO_QUAT_TRUE_CONJ_WXYZ` — for the R_y(π) correction
- `scipy.spatial.transform.Rotation` — quaternion ops
- `frame_residual_report.model_accel`, `.load` — reuse directly (don't reimplement)
- FALLBACK only: `racer.vision.gate_pose` + C2 estimator chain for PnP trajectory

---

## 4. Go/No-Go Summary Table

| Metric | GO | MARGINAL | NO_GO |
|---|---|---|---|
| p90 vel residual @ N=27 ticks (BEST) | ≤ 1.5 m/s | 1.5–3.0 m/s | > 3.0 m/s |
| Median body-accel residual (FALLBACK A) | ≤ 0.5 m/s^2/axis | 0.5–1.5 | > 1.5 |
| PnP-trajectory RMS (FALLBACK B) | ≤ 0.3 m | 0.3–0.6 m | > 0.6 m |
| R_y(π) mirror canary | PASS | — | TRIPPED |
| Drag sign (E-axis at bank, BEST) | ≥ 0.7 corr | — | < 0.5 |

If NO_GO: run `fit_twin.py` on the VQ2 tlog to re-estimate `PlantParams`, retrain RL
with the new constants + widened DR, then re-probe.

If MARGINAL: widen `QUAD_DRAG_C2_MEASURED` DR range by 2× and proceed; add a speed-cap
(reduce max speed in training by 20%) to keep the policy in the well-calibrated regime.

---

## 5. Review Questions Before Coding

1. **Command log format in Competitive mode**: `debug_obs.jsonl` is only written by `fly_rl.py`.
   In Competitive mode, are we flying with the RL policy?  If not, how do we capture `rate_frd`
   + `collective`?  Answer this before implementing the command-replay loader.

2. **AHRS attitude source in FALLBACK**: which AHRS filter will we have running at load day?
   The `LinearKF` in `racer.estimator`?  The ESKF (gated off currently)?  The fallback residual
   quality depends critically on AHRS attitude accuracy — document the expected AHRS error bound
   before interpreting fallback results.

3. **Anchor spacing**: N=27 ticks = ~0.9 s at 30 Hz.  Is this enough to see significant velocity
   divergence?  `frame_residual_report.replay()` uses N=27 with a 36-tick step.  VQ2 is faster
   than VQ1 — at 30 m/s the drone travels 27 m in 0.9 s; the residual grows fast.  Consider
   N=15 (0.5 s) for the high-speed regime to avoid the twin integrating too far off-track.

4. **`RecordingReader` dependency**: `verify_bundle.py` uses `RecordingReader` but it pulls in
   `cv2` (OpenCV).  Confirm OpenCV is in `.venv` before relying on video-index loading.

5. **Calibration write-back**: if `fit_twin.py` produces new `PlantParams`, where do the new
   constants live?  They should go in `racer/rl_plant.py` as new `_VQ2_MEASURED` constants (not
   overwriting the VQ1 ones), with a flag to select them at training time.  Spec this write-back
   path before calibration begins.

---

*Author: VQ2 pre-staging agent, 2026-06-28.  Fengyou to review §5 before coding starts.*

---
## PRE-ANSWERS to the 5 review Qs (2026-06-28, autonomous — from codebase)
- **(a) Command capture in Competitive:** commands are `SET_ATTITUDE_TARGET` (attitude or body-rate/CTBR) sent via `mavlink_client.py:418/430`; `recording.py:record_mavlink()` logs raw MAVLink. We generate our own commands → the command log is available regardless of telemetry blocking. (Confirm whether the deployed loop also writes a debug_obs.jsonl, but raw-MAVLink record is sufficient.)
- **(b) AHRS attitude source at load — CRITICAL:** current `state_estimator.LinearKF` TRUSTS the given `ATTITUDE` message and does NOT estimate orientation (docstring: "the sim HANDS us attitude"). **VQ2 blocks ATTITUDE → LinearKF cannot run as-is; there is NO self-attitude estimator built.** The ESKF is documented-but-unbuilt. ⇒ the FALLBACK tier's attitude source MUST be a newly-built ESKF or learned AHRS; this is the same gap the T1 AHRS bench targets. Twin-fidelity FALLBACK validation and the AHRS build are coupled.
- **(d) RecordingReader cv2 dep:** YES — `recording.py:242 RecordingReader` uses `cv2.imdecode` + `import cv2`. Plan for the cv2 dependency in the harness env.
- (c) anchor spacing + (e) calibration write-back: JUDGMENT calls — leave for Fengyou.
