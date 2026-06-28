# Gyro Plumbing Audit — `angular_rate_body` source-to-consumers map

**Author:** background audit agent (sonnet-4.6) · **Date:** 2026-06-28 · **Status:** AUDIT ONLY (no code touched)
**Audience:** the build agent who wires GAP #4 + GAP #2 (gyro→DroneState + AHRS→NavState).

---

## 0. Executive summary

`angular_rate_body` today originates **exclusively from the ODOMETRY MAVLink message**
(`mavlink_client.py:275`), which is **VQ2-blocked** in qualification (VADR-TS-003 §9.3).
The raw HIGHRES_IMU gyro (`msg.xgyro / ygyro / zgyro`) is **never parsed** in the
current ingest path — only `accel` and `mag` from HIGHRES_IMU reach `DroneState`.
There is **no `gyro_body` field** on `DroneState` at all.

The AHRS stack (`src/racer/ahrs/eskf.py`) needs FRD body-frame gyro as its primary
propagation input, and that raw field is wholly missing from the ingest→contract seam.

Minimal fix: add one field `gyro_body` to `DroneState`, parse `xgyro/ygyro/zgyro`
in the HIGHRES_IMU branch, and flip the AHRS+NavState consumers to use it (behind
`use_ahrs=True`). Off by default → zero regression.

---

## 1. Where `angular_rate_body` is defined and assigned today

### 1.1 `DroneState.angular_rate_body` — contracts.py:81

```
src/racer/contracts.py:81
angular_rate_body: np.ndarray = _vec(3)   # (rollspeed,pitchspeed,yawspeed) rad/s FRD, from ODOMETRY
```

The comment is unambiguous: "from ODOMETRY". No other source.

### 1.2 Assignment on the live wire — mavlink_client.py:275-277

```python
elif t == "ODOMETRY":
    ...
    angular_rate_body=np.array(
        [msg.rollspeed, msg.pitchspeed, msg.yawspeed], dtype=np.float64
    ),
```
`mavlink_client.py` lines 246–278. `rollspeed/pitchspeed/yawspeed` are the ODOMETRY
body-twist fields (MAVLink `child_frame_id=MAV_FRAME_BODY_NED=8`, i.e. body FRD).
This is the **only write** of this field in the live wire path.

**Key fact:** The HIGHRES_IMU branch (`mavlink_client.py:227-236`) parses only
`msg.xacc / yacc / zacc` (→ `accel_body`) and `msg.xmag / ymag / zmag` (→ `mag_body`).
`msg.xgyro / ygyro / zgyro` (the raw HIGHRES_IMU gyro fields) are **silently dropped**.
No field on `DroneState` receives them.

### 1.3 Assignment in the twin (training/dev) — twin.py:372

```python
angular_rate_body=self.omega * np.asarray(self.cfg.odo_rate_report_sign),
```
`twin.py:372`. The twin emulates the ODOMETRY frame aliasing via `odo_rate_report_sign`
(same convention as the live sim). `self.omega` is the plant's TRUE FRD body rate; the
report_sign converts it to the wire "ODOMETRY convention" that the controller tuned against.

### 1.4 Assignment in the Elodin adapter — elodin_adapter.py:101

```python
angular_rate_body=R_FRD_FROM_FLU @ np.asarray(gyro, dtype=np.float64),
```
`elodin_adapter.py:101`. Here `gyro` IS the raw FLU body-frame gyro from Elodin's
`SensorUpdate`. The adapter converts FLU→FRD and writes it as `angular_rate_body`.
This is NOT ODOMETRY-blocked (the Elodin rig feeds the field directly), but it is also
**not the VQ2 path** — the Elodin rig is a dev surrogate only. The important observation:
the Elodin adapter proves that raw FLU gyro → R_FRD_FROM_FLU → FRD is the correct
pre-existing recipe for sourcing rates from a raw gyro.

### 1.5 On-wire frame convention: ODOMETRY vs. TRUE FRD

The ODOMETRY body-twist is in `child_frame_id = MAV_FRAME_BODY_NED (8)` = body FRD,
but is **sign-conjugated** relative to the TRUE physical body rate. The conjugation
mirrors the quat conjugation:

```
frames.py:183-186
def true_rate_from_odo_angular_rate(w_body) -> np.ndarray:
    """ODOMETRY rollspeed/pitchspeed/yawspeed -> TRUE FRD body rate:
    all-axes negation (quat-FD of the conjugated attitude == -w_raw,
    gain 0.999/0.999/0.996; FRAME-AUDIT)."""
    return -np.asarray(w_body, dtype=np.float64)
```

`fly_rl.py:87`:
```python
_ODO_RATE_SIGN = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
```

`fly_rl.py:380`:
```python
w_frd = w_raw * _ODO_RATE_SIGN    # raw ODOMETRY -> true FRD body rates
```

So `DroneState.angular_rate_body` carries the **ODOMETRY-convention sign**, NOT the
true physical FRD rate. Consumers that need TRUE FRD must negate all three axes.

### 1.6 Twin's rate report sign

`twin.py:372` applies `cfg.odo_rate_report_sign` (default `[1,1,1]`; live sim value
calibrated as `[+1,-1,+1]` in controller). The twin reproduces the wire's ODOMETRY-sign
alias so the controller's `odo_rate_sign` correction cancels it correctly, exactly as live.

---

## 2. The HIGHRES_IMU raw gyro — availability on the wire

The MAVLink `HIGHRES_IMU` message carries fields `xgyro / ygyro / zgyro` (rad/s, body
FRD). They are on the wire but **currently unparsed**. Evidence:

- `mavlink_client.py:227-236`: the HIGHRES_IMU branch reads only `xacc/yacc/zacc`,
  `xmag/ymag/zmag`, `abs_pressure`. No `xgyro/ygyro/zgyro` read.
- `contracts.py:83-86`: `DroneState` has `accel_body`, `mag_body`, `baro_pressure_hpa`
  from HIGHRES_IMU; there is no `gyro_body` field.
- `scripts/vq2_loadday/imu_profile.py` probes HIGHRES_IMU for accel/mag/baro but does
  not read or record gyro (its `samples` dict stores `acc`, `mag`, `baro_hpa` only).

**Frame of the raw HIGHRES_IMU gyro:** body FRD, same frame as `accel_body`. The
`imu_gen.py` IMUSequence contract (`imu_gen.py:14-18`) and the AHRS README confirm the
AHRS expects `gyro: (N,3) rad/s body FRD`. No rotation needed between HIGHRES_IMU and
ESKF input — they share the FRD convention.

**MAVLink field names (standard):** `HIGHRES_IMU.xgyro`, `.ygyro`, `.zgyro` (rad/s).

---

## 3. Every consumer of `angular_rate_body` — file:line, convention expected

### 3.1 `build_obs` in fly_rl.py — the primary obs builder

```
rl/fly_rl.py:373    w_raw = np.asarray(state.angular_rate_body, dtype=np.float64)
rl/fly_rl.py:380    w_frd = w_raw * _ODO_RATE_SIGN   # raw -> true FRD body rates
rl/fly_rl.py:381-382 obs_from_zup(... w_frd * _FLIP ...)
```

**What it gets:** `DroneState.angular_rate_body` in ODOMETRY-convention (sign-conjugated).
**What it needs:** The function APPLIES the `_ODO_RATE_SIGN = [-1,-1,-1]` correction to
get TRUE FRD, then converts FRD→FLU via `_FLIP=[1,-1,-1]` to produce `w_flu` for
`obs_from_zup`. So `build_obs` expects the ODOMETRY-convention value in `state.angular_rate_body`.

**obs[9:12] (`w_flu`):** These are the TRUE FLU body rates consumed by the policy.
`obs_from_zup` (fly_rl.py:307) passes them straight into the obs vector.

### 3.2 `estimator_state_for_obs` in estimator_obs.py — the C2 seam

```
src/racer/estimator_obs.py:42-52
def estimator_state_for_obs(ds: DroneState, nav_state: NavState) -> DroneState:
    return dataclasses.replace(
        ds,
        position_ned=...,   # overwritten with NavState's pos
        velocity_ned=...,   # overwritten with NavState's vel
    )
```

This function **does not touch `angular_rate_body`** — it passes through from `ds`
(the live DroneState). So `estimator_obs` → `build_obs` paths still read the
ODOMETRY-convention rate from `DroneState`. Comment at line 47 says:
> "The trusted GIVEN attitude/rates stay from the wire (the estimator does not estimate attitude)."

This is the seam the build agent must re-route: when `use_ahrs=True`, the attitude
(for the obs[6:9] `rpy_g` channel) AND the rates (for obs[9:12] `w_flu`) must come
from the AHRS output, not `ds.angular_rate_body`.

### 3.3 `make_nav_state` in state_estimator.py — NavState assembler

```
src/racer/state_estimator.py:216
angular_rate_body=np.asarray(drone_state.angular_rate_body, dtype=np.float64).copy(),
```

**What it gets:** ODOMETRY-convention value from `DroneState`.
**What NavState.angular_rate_body is:** The docstring in `contracts.py:237` has no
extra annotation — it inherits the ODOMETRY-convention of the DroneState field.
Consumers of `NavState.angular_rate_body` must know this convention.

### 3.4 Controller — `_decoupled_body_rate` in controller.py

```
src/racer/controller.py:331
rate = np.asarray(nav.angular_rate_body, dtype=np.float64) * np.asarray(self.odo_rate_sign, ...)
omega = (self.kp_att * rotvec - self.kd_att * rate) / max(self.ff_gain, 1e-6)
```

**What it gets:** `NavState.angular_rate_body` (ODOMETRY-convention, from `DroneState` via
`make_nav_state`).
**What it needs:** The controller applies its OWN `odo_rate_sign` correction (measured:
`[+1,-1,+1]` on the live sim) to convert from ODOMETRY-convention to TRUE FRD for the rate
damping term. So it explicitly depends on receiving the ODOMETRY-convention value.

Also `controller.py:245`:
```
omega = omega - self.kd_att * np.asarray(nav.angular_rate_body, dtype=np.float64)
```
(Non-decoupled path, no `odo_rate_sign` correction here — this path is for sysid/bench only,
not the deployed CTBR flight path; `decoupled=True` is the deployed mode.)

And `controller.py:51-74` (`level_hold_body_rate`, a sysid helper):
```
omega = (kp * rotvec - kd * np.asarray(angular_rate_body, dtype=np.float64)) / max(ff_gain, 1e-6)
```
Called with `angular_rate_body` directly. This is a standalone function used only in sysid probes.

### 3.5 Spin guard in fly_rl.py

```
rl/fly_rl.py:1079
w_mag = float(np.linalg.norm(np.asarray(s.angular_rate_body, dtype=np.float64)))
```

Here `s` is the raw `DroneState`, NOT NavState. This reads `DroneState.angular_rate_body`
in ODOMETRY-convention. Used purely for spin detection (|w| threshold), so sign doesn't
matter — a flip to raw-gyro magnitudes would be fine.

### 3.6 `telemetry_health` in fly_rl.py

```
rl/fly_rl.py:402,413
if ... or state.angular_rate_body is None:
    return "no_fix", ...
if ... and np.all(np.isfinite(np.asarray(state.angular_rate_body, ...))):
    ...
```

Checks that `angular_rate_body` is not None and is finite. This is the ODOMETRY health gate:
it returns "no_fix" if ODOMETRY has not arrived, because `angular_rate_body` defaults to
`_vec(3)` (a zero numpy array), NOT None. Actually `_vec(3)` returns `np.zeros(3)`, which
would PASS the None check. The None check here guards `orientation_ned_wxyz` (which IS
optional/None pre-ODOMETRY). Note: `angular_rate_body` has a non-None default (`_vec(3)`),
so in VQ2 (no ODOMETRY) it would read as zeros, not None — it would pass the None gate
but the `orientation_ned_wxyz is None` check would catch it first. The health check is
ODOMETRY-gated via `orientation_ned_wxyz`, not via `angular_rate_body` directly.

### 3.7 Debug logging in fly_rl.py

```
rl/fly_rl.py:1114
"w_raw": np.asarray(s.angular_rate_body).round(4).tolist(),
```

Logs the raw ODOMETRY-convention value. Informational only.

### 3.8 `_deprecated/replay_obs.py`

`_deprecated/replay_obs.py:190` uses `angular_rate_body=o["w_raw"]`. Deprecated file,
not on any active path.

---

## 4. Summary: source → consumers map

```
Wire message             Field written             Convention
─────────────────────────────────────────────────────────────────────
ODOMETRY.rollspeed       DroneState.angular_rate_body   ODOMETRY-sign
 .pitchspeed              (mavlink_client.py:275-277)    (= -1 × TRUE FRD
 .yawspeed                                               per _ODO_RATE_SIGN)
HIGHRES_IMU.xgyro        (NOT PARSED — field absent)    raw FRD / BLOCKED VQ2
 .ygyro
 .zgyro

DroneState.angular_rate_body
  ├── make_nav_state (state_estimator.py:216)
  │     └── NavState.angular_rate_body
  │           ├── Controller._decoupled_body_rate (controller.py:331)
  │           │     applies odo_rate_sign → TRUE FRD for damping
  │           └── Controller._body_rate_command (controller.py:245)
  │                 no sign correction (bench/sysid path)
  ├── build_obs (fly_rl.py:373-380)
  │     applies _ODO_RATE_SIGN × _FLIP → w_flu → obs[9:12]
  │     (via estimator_state_for_obs, which passes angular_rate_body through unchanged)
  ├── telemetry_health (fly_rl.py:413)  [finiteness check, sign-agnostic]
  └── fly loop spin guard (fly_rl.py:1079)  [|w| only, sign-agnostic]
```

---

## 5. The minimal plumbing change

### 5.1 New field on DroneState

**File:** `src/racer/contracts.py`

Add after line 85 (after `mag_body`):
```python
gyro_body: np.ndarray | None = None    # rad/s, FRD; raw HIGHRES_IMU gyro (xgyro/ygyro/zgyro).
                                       # None until the first HIGHRES_IMU message.
                                       # For VQ2 AHRS: this is the non-blocked gyro source.
                                       # DroneState.angular_rate_body stays as the ODOMETRY-derived
                                       # value (blocked in VQ2 qualification).
```

**Why `None` default:** `angular_rate_body` uses `_vec(3)` (non-None) because old code
assumed it always valid (ODOMETRY always present in VQ1). The new `gyro_body` should be
`None` pre-first-HIGHRES_IMU tick to let consumers gate on it cleanly, matching `mag_body`.
This is additive; no existing field changes → zero regression.

### 5.2 Parse raw gyro in HIGHRES_IMU branch

**File:** `src/racer/mavlink_client.py` — HIGHRES_IMU branch (lines 229-236)

Add `gyro_body` to the `replace()`:
```python
elif t == "HIGHRES_IMU":
    self.state = replace(
        self.state,
        sim_time_ns=int(msg.time_usec) * 1_000,
        recv_monotonic_ns=recv,
        accel_body=np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64),
        gyro_body=np.array([msg.xgyro, msg.ygyro, msg.zgyro], dtype=np.float64),   # NEW
        mag_body=np.array([msg.xmag, msg.ymag, msg.zmag], dtype=np.float64),
        baro_pressure_hpa=float(msg.abs_pressure),
    )
```

This is a **pure addition** — the ODOMETRY branch still writes `angular_rate_body`.
Frame: raw `HIGHRES_IMU.xgyro/ygyro/zgyro` is body FRD (same convention as `accel_body`).
No rotation needed. The ESKF `step(gyro, accel, dt)` takes FRD input directly.

**Elodin adapter parity:** `elodin_adapter.py:101` already writes
`angular_rate_body=R_FRD_FROM_FLU @ gyro` because Elodin sends FLU. For the Elodin
path, `gyro_body` should also be populated:
```python
gyro_body=R_FRD_FROM_FLU @ np.asarray(gyro, dtype=np.float64),
```
This keeps the two ingest paths symmetric so AHRS works against both.

### 5.3 AHRS consumption of gyro_body (the re-route)

When `use_ahrs=True`, the Navigator calls:
```python
q_wxyz = self._ahrs.step(
    ds.gyro_body,          # <-- raw FRD, no sign correction needed
    ds.accel_body,
    dt,
    ds.mag_body,           # optional, None-guarded inside ESKF
)
```

**NOT** `ds.angular_rate_body`. The AHRS takes the raw physical gyro;
`angular_rate_body` is the ODOMETRY-convention aliased rate and must NOT be fed to
the AHRS (wrong sign and wrong provenance).

### 5.4 AHRS output → NavState re-route (rates for obs[9:12])

When `use_ahrs=True`, `make_nav_state` (or the Navigator's `_nav_state`) must override
`angular_rate_body` in the `NavState` with the bias-corrected AHRS estimate:

```python
# After AHRS step:
w_body_true_frd = ds.gyro_body - self._ahrs.gyro_bias   # bias-corrected TRUE FRD

# Then in make_nav_state (or an override dict):
NavState(
    ...
    roll=..., pitch=..., yaw=...,            # from AHRS Euler
    angular_rate_body=w_body_true_frd,       # TRUE FRD, bias-corrected
    ...
)
```

**CRITICAL:** When `use_ahrs=True`, the `NavState.angular_rate_body` written here is
**TRUE FRD** (not ODOMETRY-convention). This changes the convention the consumer expects.

### 5.5 Consumer impact of the convention change

This is where the frame footgun lives. Two consumer classes:

**A) The obs builder (`build_obs` / `estimator_state_for_obs`):**
`build_obs` (`fly_rl.py:373-380`) reads `state.angular_rate_body` and applies
`_ODO_RATE_SIGN = [-1,-1,-1]`. If the AHRS path writes TRUE FRD directly into
`NavState.angular_rate_body`, then when `estimator_state_for_obs` passes it through
to `build_obs`, the builder will NEGATE it again — wrong by a factor of -1 on all axes.

**Fix option A:** When `use_ahrs=True`, negate the bias-corrected gyro before writing
to `NavState.angular_rate_body` so `build_obs` un-negates it back to TRUE FRD:
```python
angular_rate_body = -(ds.gyro_body - self._ahrs.gyro_bias)  # ODOMETRY-convention compatible
```
This preserves the `build_obs` convention contract (it always expects ODOMETRY-sign).
Down the chain: `obs[9:12] w_flu` will be computed correctly.

**Fix option B (cleaner, but touches more consumers):** Add a dedicated
`angular_rate_body_frd_true` field to `NavState` or pass the bias-corrected rate
separately, and update all consumers to use the right field. This is the right
long-term shape but is a larger change.

**Recommendation: use Fix option A for the minimal change.** Store
`-(gyro_body - gyro_bias)` into `NavState.angular_rate_body` when `use_ahrs=True`.
This preserves the ODOMETRY-convention that all existing consumers expect, requiring
zero changes downstream.

**B) The controller (`_decoupled_body_rate`, `controller.py:331`):**
It applies `odo_rate_sign = [+1,-1,+1]` (the measured live-sim correction). This
`odo_rate_sign` was measured empirically against the ODOMETRY rate — it corrects
the ODOMETRY-convention alias back to TRUE FRD (or rather, to the sign the sim's
inner loop expects). If we write ODOMETRY-convention-compatible data into `NavState`
(Fix option A), the controller continues to work without changes.

However, note: the controller's `odo_rate_sign` was calibrated against ODOMETRY's
specific per-axis sign quirks. The raw HIGHRES_IMU gyro may have a different per-axis
sign relationship to truth. This must be empirically verified on the sim — the audit
cannot settle it without a live recording. This is flagged in Section 7 (Risks).

### 5.6 Gating flag

Add `use_ahrs: bool = False` to `NavigatorConfig` (alongside the existing
`use_rewind_kf`, `use_gate_relative`, etc. in `navigator.py`). When `False`, the
code path is byte-identical to today (ODOMETRY attitude, ODOMETRY `angular_rate_body`
into NavState). When `True`, the AHRS instance is stepped each tick, its Euler/rates
override the NavState fields.

---

## 6. Test plan

### 6.1 Contracts/parse unit test (OFF-path regression)

**File:** `tests/test_gyro_plumbing.py` (new)

**Test A — `gyro_body` populated from HIGHRES_IMU:**
Feed a synthetic HIGHRES_IMU message with known `xgyro/ygyro/zgyro` values via a
`MavlinkClient._handle()` call or direct `DroneState` construction.
Assert `ds.gyro_body` equals `[xgyro, ygyro, zgyro]` as float64. Assert
`ds.angular_rate_body` is still zeros (no ODOMETRY yet) or unchanged from before.

**Test B — ODOMETRY path unchanged:**
Feed an ODOMETRY message. Assert `ds.angular_rate_body` equals the expected ODOMETRY
values. Assert `ds.gyro_body` is `None` (ODOMETRY does not set it).

**Test C — `gyro_body` is None until first HIGHRES_IMU:**
Fresh `DroneState()` → `gyro_body is None`. After an ODOMETRY message, `gyro_body`
is still `None`. Only after HIGHRES_IMU is it set.

**Test D — OFF == byte-identical:**
With `use_ahrs=False` (default), the entire navigator+obs stack must be byte-identical
to before. Run existing `tests/test_deploy_obs20.py` and `tests/test_spike_golden.py`
unchanged — they must stay green. Run `scripts/green_gate.py` — sentinel 1091 must hold.

### 6.2 Rate convention regression test

**File:** (same `tests/test_gyro_plumbing.py` or a new `tests/test_gyro_convention.py`)

**Test E — raw-gyro path matches ODOMETRY-derived rate within sensor-noise budget:**
Construct a synthetic `DroneState` with:
- `angular_rate_body = [r, p, y]` (ODOMETRY-convention, e.g. `[0.5, 0.3, 0.1]`)
- `gyro_body = -(np.array([r, p, y]))` (raw FRD true = negation of ODOMETRY-convention)

Assert that `build_obs(ds, ...)` with the ODOMETRY-convention `angular_rate_body`
yields the same `obs[9:12]` as `build_obs(ds_ahrs, ...)` where `ds_ahrs` has
`angular_rate_body = -(true_frd)` (i.e. the negated-back value that Fix option A writes).

Delta must be below float32 epsilon (1.2e-7). This confirms that Fix option A's
`-(gyro - bias)` round-trips correctly through `build_obs`.

**Test F — AHRS vs ground-truth rate on a synthetic trajectory:**
Use `ahrs/traj6dof.py` to generate a known angular rate sequence.
Feed `gyro_body` samples (from `imu_gen.IMUSequence.gyro`) to the AHRS ESKF.
Compare ESKF `gyro_bias`-corrected output against the true rate.
Pass criterion: per-axis `|rate_ahrs - rate_true|` p90 ≤ gyro_noise_std × sqrt(N)
after a convergence window (~100 ticks at 200 Hz = 0.5 s).

### 6.3 End-to-end OFF-path regression

Run `scripts/green_gate.py --full` (sentinel 1091 / 1047 passed / 44 skipped).
Must be GREEN with zero new failures after adding `gyro_body` field and the HIGHRES_IMU
parse. The existing tests never construct a HIGHRES_IMU message that sets `gyro_body`,
so all `DroneState` fixtures will have `gyro_body=None` — consistent with the `None`
default.

---

## 7. Frame / convention risks and footguns

### 7.1 The sign footgun (highest risk)

`DroneState.angular_rate_body` carries ODOMETRY-convention (= `-1 × TRUE FRD`).
The AHRS must consume `gyro_body` (TRUE FRD), NOT `angular_rate_body`.
Mixing them silently gives a `−2×` rate (ODOMETRY sign + AHRS negate = double-negative)
on acrobatic maneuvers, which would corrupt the attitude estimate.

**Pin this with a test:** assert that feeding `angular_rate_body` to the ESKF (instead
of `gyro_body`) gives attitude error > 5° after 1 s on a synthetic spinning trajectory,
while `gyro_body` stays below the pass criterion.

### 7.2 Controller odo_rate_sign calibration gap

The `odo_rate_sign = [+1,-1,+1]` in the controller was tuned empirically against
ODOMETRY rates. When `use_ahrs=True`, `NavState.angular_rate_body` comes from
`-(gyro_body - gyro_bias)` (Fix option A). The net mapping from raw HIGHRES_IMU gyro
to the controller's rate input is then:

```
rate_to_controller = odo_rate_sign × angular_rate_body_nav
                   = odo_rate_sign × (-(gyro_body - bias))
                   = odo_rate_sign × (-true_frd)
```

For `odo_rate_sign=[+1,-1,+1]` and true FRD `w=[r,p,y]` this yields `[-r, p, -y]`.
The ODOMETRY path would give `[-r, p, -y]` too (ODOMETRY-convention `[−r,−p,−y]`
multiplied by `[+1,−1,+1]` = `[−r,+p,−y]`... wait — let's be explicit:

- ODOMETRY raw (from wire): `[-r_true, -p_true, -y_true]` (all-negative, `_ODO_RATE_SIGN`)
- Fix option A writes: `-(gyro_body - bias) ≈ -(true_frd) = [-r_true, -p_true, -y_true]`

So Fix option A and the ODOMETRY path write the **same convention** into
`NavState.angular_rate_body`. The `odo_rate_sign` correction therefore applies
identically. **No controller recalibration needed** under Fix option A.

### 7.3 The +L obs sign is unaffected

`obs[0:3] pos_g = R_w2g @ (gate_pos - nav_state.position_ned)`. This depends on
attitude (via `R_wb` used by the localization lever and KF predict), but NOT on
`angular_rate_body`. The +L pinning (`tests/test_obs_sign_faithfulness.py`) is unaffected
by the gyro re-route. No action required.

### 7.4 obs[17:20] confidence/age is unaffected

`confidence_triple` reads only `nav_inplane_sigma`, `nav_along_sigma`, and
`time_since_vision_update_s` from `NavState`. None of these come from `angular_rate_body`
or `gyro_body`. The `/√2` reconciliation at `estimator_obs.py:163` is unaffected.
No action required.

### 7.5 The #37 obs-faithfulness implication

`estimator_state_for_obs` (`estimator_obs.py:42-52`) copies `angular_rate_body` through
from `DroneState` unchanged. In the `use_ahrs=True` path, the AHRS-sourced value must
arrive in `DroneState.angular_rate_body` (via a new assignment in the navigator or
an override in `estimator_state_for_obs`) so that `build_obs` sees the right value.

Two options:
- A) Navigator writes the AHRS-derived rate back into a synthetic DroneState before
  calling `estimator_state_for_obs` (cleanest: the seam is explicit).
- B) `estimator_state_for_obs` accepts an optional `rate_override` kwarg.

Either way, the test `tests/test_deploy_obs20.py` uses fixture DroneStates without
AHRS — the `use_ahrs=False` path → same `DroneState.angular_rate_body` → byte-identical
to today. #37 is preserved.

### 7.6 HIGHRES_IMU gyro frame in the real VQ2 sim

The audit assumes `HIGHRES_IMU.xgyro/ygyro/zgyro` is body FRD in the VQ2 sim, matching
the accel convention and the MAVLink spec. This has NOT been confirmed on the VQ2 wire
(VQ2 sim was not available at audit time). The load-day probe
(`scripts/vq2_loadday/imu_profile.py`) must be extended to log raw gyro fields and check:
1. The fields are non-zero during motion (not silently zeroed by the sim).
2. The gyro values correlate with ODOMETRY body rates with the expected sign relationship
   (`gyro_frd ≈ -angular_rate_body` per `_ODO_RATE_SIGN`).

This is a **VQ2 load-day check**, not a blocker for writing the plumbing code.

---

## 8. The one-line minimal-change summary

**Add one field (`gyro_body`) to `DroneState` (contracts.py after line 85). Parse
`msg.xgyro/ygyro/zgyro` in the HIGHRES_IMU branch of `mavlink_client.py` (after
line 234). Feed `ds.gyro_body` (NOT `ds.angular_rate_body`) to the ESKF. Write the
AHRS output as `-(gyro_body - gyro_bias)` into `NavState.angular_rate_body` (Fix
option A) so all downstream consumers stay convention-compatible. Gate everything
behind `NavigatorConfig.use_ahrs=False` (default OFF → byte-identical).**

---

## Appendix A — file:line index for this audit

| Item | Location |
|---|---|
| `DroneState.angular_rate_body` definition | `src/racer/contracts.py:81` |
| `DroneState.gyro_body` — MISSING (add here) | `src/racer/contracts.py:85-86` (after `mag_body`) |
| HIGHRES_IMU parse (no gyro today) | `src/racer/mavlink_client.py:227-236` |
| ODOMETRY parse → angular_rate_body | `src/racer/mavlink_client.py:275-277` |
| Twin angular_rate_body assignment | `src/racer/twin.py:372` |
| Elodin adapter angular_rate_body | `src/racer/elodin_adapter.py:101` |
| ODOMETRY sign function | `src/racer/frames.py:183-186` |
| R_wb from ODOMETRY quat (GAP #1) | `src/racer/navigator.py:416` |
| NavState assembler (GAP #2) | `src/racer/state_estimator.py:209-221` |
| angular_rate_body → NavState | `src/racer/state_estimator.py:216` |
| build_obs rate consumer | `rl/fly_rl.py:373-383` |
| `_ODO_RATE_SIGN = [-1,-1,-1]` | `rl/fly_rl.py:87` |
| obs_from_zup w_flu input | `rl/fly_rl.py:307-357` |
| estimator_state_for_obs (pass-through) | `src/racer/estimator_obs.py:42-52` |
| Controller decoupled rate consumer | `src/racer/controller.py:331` |
| Controller body_rate_command | `src/racer/controller.py:245` |
| level_hold_body_rate sysid helper | `src/racer/controller.py:49-76` |
| Spin guard consumer | `rl/fly_rl.py:1079` |
| telemetry_health finiteness check | `rl/fly_rl.py:402-413` |
| ESKF step (gyro input = FRD) | `src/racer/ahrs/eskf.py:191-216` |
| ESKF gyro_bias property | `src/racer/ahrs/eskf.py:173-175` |
| IMUSequence (gyro contract) | `src/racer/ahrs/imu_gen.py:52` |
| imu_profile.py (no gyro today) | `scripts/vq2_loadday/imu_profile.py:59-73` |
| AHRS test infra | `src/racer/ahrs/traj6dof.py`, `ahrs/metrics.py` |
| +L obs sign test (unaffected) | `tests/test_obs_sign_faithfulness.py` |
| #37 fidelity tests (must stay green) | `tests/test_deploy_obs20.py`, `tests/test_spike_golden.py` |

---

## MEMORY-DELTA

```
GYRO-PLUMBING-AUDIT (2026-06-28):
- angular_rate_body origin: ODOMETRY.rollspeed/pitchspeed/yawspeed (mavlink_client.py:275).
  ODOMETRY-convention = -1 × TRUE FRD (frames.py _ODO_RATE_SIGN=[-1,-1,-1]).
- Raw HIGHRES_IMU gyro (xgyro/ygyro/zgyro FRD) is NOT parsed anywhere; no gyro_body field exists.
- Consumers + their expected convention:
    build_obs (fly_rl.py:373-380): ODOMETRY-sign → applies _ODO_RATE_SIGN internally → obs[9:12] w_flu
    make_nav_state (state_estimator.py:216): pass-through to NavState (ODOMETRY-sign)
    Controller._decoupled_body_rate (controller.py:331): ODOMETRY-sign, applies odo_rate_sign=[+1,-1,+1]
    spin guard (fly_rl.py:1079): |w| only, sign-agnostic
    telemetry_health: finiteness only, sign-agnostic
- Minimal change: add DroneState.gyro_body (None default), parse HIGHRES_IMU.xgyro/ygyro/zgyro,
  feed raw FRD gyro_body to ESKF, write -(gyro-bias) into NavState.angular_rate_body (Fix opt A
  = ODOMETRY-convention-compatible → zero controller/obs recalibration), gate use_ahrs=False default.
- Staged doc: docs/reactivation-2026-06-27/gyro-plumbing-audit.md
```
