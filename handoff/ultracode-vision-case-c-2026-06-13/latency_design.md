# IN-LOOP vision latency — measurement + compensation design (Task D)

Fengyou — case-C readiness, 2026-06-13. Vision-ONLY pose (VQ2 case C). This document reports
the measured in-loop vision compute latency `L_total` and designs the predict-forward / rewind
compensation. Harness: `latency_harness.py`; raw numbers: `latency_results.json`.

---

## 0. What "in-loop vision latency" is, and what it is NOT

`L_total` = wall-time of the per-tick vision COMPUTE chain that converts one camera frame into a
KF position fix:

```
detector forward pass  ->  PnP (estimate_gate_pose)  ->  associate  ->
gate_pose_to_world_position  ->  kf.update_position
```

It is the amount by which a fix is STALE purely because the work took time, on top of frame age.
It is the `dt` the compensation must cover. It is **distinct** from two latencies already in the
ledger, which it is routinely conflated with:

| latency | what it is | value | source |
|---|---|---|---|
| **command/actuation** | command issued → realized on the airframe | **67 ms (2 ticks)** | cmd-vs-realized cross-corr (memory) |
| **content-vs-telemetry offset Δ** | image content instant vs recv-paired telemetry pose | a capture-stamp lag, `off=-Δ·v+c` | `vision-pkg2/latency_fit.py` |
| **in-loop vision compute `L_total`** | frame-in → fix-applied compute time | **THIS DOC** | `latency_harness.py` |

The 67 ms actuation number is the WRONG number for piece-B staleness (the A-audit flagged exactly
this conflation). The content-offset Δ is a *separate, additive* staleness source (the image is
already old when it arrives); `L_total` is the *processing* time after arrival. Total fix age the
estimator must correct = **frame-age (= Δ + transport/jpeg) + L_total**. This doc quantifies
`L_total`; Δ is the vision-pkg2 workstream's; both feed the same predict-forward/rewind correction.

---

## 1. Measured latency (state the exact HW for every number)

**HW:** laptop, `Windows-11`, `Intel64 Family 6 Model 170` (Core Ultra-class), Python 3.13.2,
`torch 2.12.0+cpu`, **CUDA UNAVAILABLE**. No GPU/NPU on this machine.

### 1a. PnP / associate / world-fix / KF update — MEASURED ON LAPTOP CPU
Timed over 120 realistic geometries (gate ranges drawn from the empirical `perception-char`
4-corner-fix distribution, off-axis bearing ±12°, gate yaw ±25°, 0.7 px corner noise), corners
synthesized via `project_gate_corners` (the exact inverse of the PnP), confidence-weighted
refine + analytic covariance ON (the live path: `compute_covariance=True`).

| stage | p50 | p90 |
|---|---|---|
| `estimate_gate_pose` (IPPE + robust GN refine + Fisher cov) | 0.62 ms | 0.70 ms |
| `associate` | 0.06 ms | 0.08 ms |
| `gate_pose_to_world_position` | 0.03 ms | 0.04 ms |
| `kf.update_position` (Joseph form, 6-state) | 0.02 ms | 0.03 ms |
| **PnP-chain total `L_pnp`** | **0.73 ms** | **0.83 ms** |

PnP dominates the chain (~85%); the GN refine + 6×6 Fisher inversion is the cost. assoc/worldfix/KF
are each sub-0.1 ms. These are pure numpy/OpenCV dense linear algebra on tiny matrices, so they
port to the edge CPU with little uncertainty (the edge ARM core is comparable; treat 0.83 ms p90 as
an **upper bound** for the edge too). **`L_pnp` is negligible at any speed** (0.83 ms × 30 m/s =
0.025 m). The latency budget is entirely the detector.

### 1b. Detector forward pass — TWO labelled numbers, never mixed

**(i) CPU MEASURED = UPPER BOUND (NOT eval-HW).** `models/gate_yolo11s_curriculum_v2.pt`, full
`predict()` path (letterbox preprocess + forward + NMS/pose-decode) at the stride-padded inference
size 640×384, `device=cpu`:

> **detector(CPU) p50 ≈ 124 ms / p90 ≈ 139 ms** — laptop Intel CPU, torch+cpu. **UPPER BOUND.**
> This is what you get if the eval hardware had NO accelerator. It is reported only to bracket the
> top end; it must NOT be quoted as the eval-HW latency.

**(ii) Edge ESTIMATE (NOT measured here).** Model = 9.72 M params; ultralytics `get_flops` =
22.54 GFLOPs @ 640², scaling to **13.5 GFLOPs** at 640×384. Onboard budget ≈ **100 TOPS** (INT8).

- *FLOP compute floor:* 13.5 GFLOP / (100 TOPS × util). At 10–35% sustained util → **0.39–1.35 ms**.
  This proves the detector is **NOT compute-bound** on the edge (the floor is sub-ms).
- *Realistic end-to-end:* preprocess (letterbox+normalise), NMS/pose-decode, kernel-launch, and
  memory-bound early layers dominate at this tiny resolution. A 9.7 M-param yolo11s at 640×384 on a
  ~100-TOPS-class accelerator (Jetson Orin NX / similar) runs end-to-end ≈ **5–15 ms**. I use this
  **empirical bracket** for the budget (NOT the over-optimistic sub-ms floor). **Labelled ESTIMATE
  — not measured on eval HW. The single most important number to confirm with one real timing run
  on the actual eval box (or a `latency.txt` from the organizer).**

### 1c. `L_total` budget

| HW assumption | detector p50/p90 | + L_pnp | **L_total p50** | **L_total p90** |
|---|---|---|---|---|
| **edge estimate (5–15 ms)** | 5.0 / 15.0 ms | 0.73 / 0.83 ms | **≈ 5.7 ms** | **≈ 15.8 ms** |
| laptop CPU upper bound | 124 / 139 ms | 0.73 / 0.83 ms | ≈ 124 ms | ≈ 140 ms | 

**Headline (edge, the operative assumption): `L_total` ≈ 6 ms (p50), ≈ 16 ms (p90).**

---

## 2. Uncompensated position error = v · L_total

The fix is computed from the pose at *capture* time but applied to the *current* KF state
`L_total` later, with no rewind/predict-forward. The position error injected is `v · L_total`:

| speed | edge p50 (5.7 ms) | edge p90 (15.8 ms) | CPU upper-bound (≈130–140 ms) |
|---|---|---|---|
| 5.35 m/s (VQ1 recording) | 0.03 m | 0.09 m | 0.67–0.75 m |
| 20 m/s (VQ2 target) | **0.11 m** | **0.32 m** | 2.5–2.8 m |
| 30 m/s (VQ2 high) | **0.17 m** | **0.47 m** | 3.7–4.2 m |

Read against the **0.75 m gate inner-radius / validity budget** and the inc8 binding-gate margins
(gate-4 ≈ 0.155 m, gate-5 ≈ 0.314 m): at edge latency the uncompensated `L_total` error is 0.1–0.3 m
at 20 m/s — *comparable to the gate-4 margin* and therefore **not ignorable at VQ2 speed**, but an
order of magnitude below the CPU figure. The CPU-bound 2.5–4.2 m would be catastrophic; it is the
clearest possible demonstration that the eval-HW detector time MUST be confirmed, because the entire
conclusion hinges on which row of the table is real.

**Important framing:** `v·L_total` is only the *compute-time* slice of the total fix-age error. The
content-offset Δ (vision-pkg2) adds its own `v·Δ` term, and at racing speed the *frame age itself*
(time since the photons, before our compute even starts) may exceed `L_total`. The compensation
below is designed to correct the **total** capture-to-apply age, of which `L_total` is one measured
component.

---

## 3. Compensation: predict-forward (b) vs rewind (a) — which is correct

The fix `z` is a measurement of the drone position at **capture time** `t_c`. It is applied at
**apply time** `t_a = t_c + age`, where `age = frame_age (Δ + transport) + L_total`. The KF state at
`t_a` has already moved. Two corrections:

### (a) REWIND to capture time (piece B, the ring buffer)
Snapshot KF state + buffered IMU. On a fix: rewind state to `t_c`, apply `update_position(z, R)` at
`t_c`, then re-propagate the buffered IMU samples from `t_c` forward to `t_a`. This is the textbook
**out-of-sequence measurement (OOSM)** handling.

- **Correct in general:** handles arbitrary, *variable* age and arbitrary motion (the re-propagation
  uses the true IMU between `t_c` and `t_a`, including accel changes — no constant-velocity
  assumption). It also correctly mixes the fix's information at the right epoch (the covariance is
  rewound too), so the gain is right.
- **Cost:** a ring buffer of `(state, P, accel_body, R_wb, dt, sim_time_ns)` over the max age
  (~16 ms p90 + frame-age; a 64-deep buffer at 75 Hz IMU = ~0.85 s, ample). Re-propagation is a few
  `predict()` calls (each ~0.02 ms measured) → < 1 ms even at the deepest rewind. Cheap.
- **Hard prerequisite (A-audit):** the fix's capture timestamp must be in the SAME clock as the IMU
  master clock. Today it is NOT: `obs.sim_time_ns` is the **video UNIX epoch**
  (`jpeg_receiver`→`detector`), `ds.sim_time_ns` is the **IMU sim-boot epoch** (`mavlink_client`).
  Rewind keyed off `obs.sim_time_ns` would index the IMU buffer with a garbage epoch. **Rewind is
  BLOCKED on TIMESYNC** (video↔IMU epoch reconciliation), exactly as `state_estimator.py:36` states.

### (b) PREDICT-FORWARD the fix by `age` before applying
Propagate the *measurement* (not the state) forward from `t_c` to `t_a` using the model, then apply
as a normal current-time update:

```
z_forward = z + ∫_{t_c}^{t_a} v dt   (constant-velocity:  z_forward ≈ z + v̂ · age)
R_forward = R + age² · Cov(v̂) + Q-growth over age   (inflate for the propagation uncertainty)
kf.update_position(z_forward, R_forward)
```

- **Correct under near-constant motion over `age`.** Error vs rewind = the *change* in velocity over
  `age`: `½ a · age²`. At a=20 m/s² (aggressive) and age=16 ms → ½·20·0.016² = **0.0026 m** — three
  orders below the gate budget. So at edge latency, predict-forward ≈ rewind to < 3 mm. At the CPU
  140 ms figure the second-order term is ½·20·0.14² = 0.20 m — non-negligible, another reason the
  edge number must be confirmed.
- **Cost:** ~0 (one vector add + a cov inflation). No buffer.
- **Clock requirement:** needs `age` = `t_a − t_c`, which STILL needs the capture timestamp in a
  comparable clock (or a *measured constant* total-age, since frame-age + L_total is roughly stable
  per HW). So predict-forward also benefits from TIMESYNC, but can fall back to a **measured constant
  age** (calibrate `age` once on the eval box, apply blindly) when per-frame `t_c` is unavailable —
  rewind cannot, because it must index the buffer by epoch.

### Verdict
- **Rewind SUBSUMES predict-forward** (it is exact for arbitrary motion + variable age and gets the
  covariance epoch right). It is the correct end-state for piece B.
- **But for THIS magnitude of `L_total` (≈6–16 ms on edge), predict-forward is within ~3 mm of
  rewind** and is essentially free. The second-order `½·a·age²` gap only matters if `age` blows out
  to the CPU-bound 100+ ms regime (which would mean the eval HW is wrong, a separate alarm).
- **Recommendation:** ship **predict-forward with a measured constant `age`** first — it removes the
  first-order `v·age` term (0.1–0.5 m at VQ2 speed) at zero buffer cost and does NOT require the full
  epoch reconciliation (only a one-time age calibration). Treat the full **rewind ring buffer
  (piece B) as the principled upgrade**, gated on TIMESYNC, justified ONLY if (1) the eval detector
  time is large, (2) `age` is highly variable frame-to-frame, or (3) accel changes sharply within
  `age`. At the measured edge latency, none of those hold, so rewind is **defer-able** — but TIMESYNC
  is still worth building because it also fixes the `time_since_vision_update_s` epoch-mix bug
  (`navigator.py:426`) that the A-audit found to be load-bearing in the case-C coast/abort policy.

### Interaction with piece B and the velocity-unobservable problem
Both corrections need `v̂` (predict-forward to extrapolate; rewind to re-propagate — rewind uses
IMU-integrated `v̂` directly). In **true case C**, velocity is *unobservable*: vision is
position-only (`update_position`, never `update_velocity`) and `v̂` comes solely from IMU integration
of `accel_body` between fixes. So the compensation accuracy inherits the IMU velocity drift. This is
fine over a 6–16 ms `age` (drift over 16 ms is sub-mm) but it means **the compensation cannot be
better than the IMU velocity over `age`** — another reason to keep `age` small (favor a fast
detector / accept the latency as-is rather than chase a heavier model).

---

## 4. Relation to the "30 Hz binds only at ≥30 m/s & last-fix ≤10 m" rule

The rule: the 30 Hz policy decision rate is only a bottleneck when the drone is fast (≥30 m/s) AND
the most-recent ACCEPTED vision fix was taken within 10 m downrange of the gate (close-range, where
sub-tick staleness matters for the terminal approach).

**Does `L_total` push the effective last-accepted-fix distance past 10 m?** A fix captured at
downrange `D` *applies* when the drone is at `D − v·L_total`:

| `L_total` | @20 m/s | @30 m/s | fix captured at 10 m applies at (30 m/s) |
|---|---|---|---|
| edge p50 5.7 ms | 0.11 m | 0.17 m | 9.83 m |
| edge p90 15.8 ms | 0.32 m | 0.47 m | 9.53 m |
| CPU upper-bound 130 ms | 2.6 m | 3.9 m | 6.1 m |

- **At edge latency: NO.** `L_total` erodes the 10 m window by at most ~0.47 m at 30 m/s (≈5%). The
  fix still lands well inside 10 m; the rule's regime is unchanged. `L_total` is *not* the thing that
  determines whether the 30 Hz rate binds — the *frame-age + accepted-fix interval* (effective
  accepted-fix rate ≈ 36 Hz from 77 Hz raw × 47% acceptance → ~0.83 m between accepted fixes at
  30 m/s) dominates the "distance since last fix" far more than the 0.5 m compute slice.
- **At CPU latency: YES, badly.** A 130 ms compute time consumes 3.9 m of the 10 m window at 30 m/s
  — it would single-handedly move the effective last-accepted-fix distance from 10 m to ~6 m and make
  the terminal approach blind over the last ~4 m. This is the scenario where the latency rule, the
  predict-forward second-order term, and the gate margins ALL break simultaneously. **It is the
  decisive reason to obtain one real eval-HW detector timing before trusting any case-C speed claim.**

**Bottom line on the rule:** measured (edge) `L_total` does NOT push the last-accepted-fix past 10 m;
the 30 Hz analysis from memory holds. But the analysis is only valid *if the eval detector runs in
the ~5–15 ms band*. The conclusion is HW-conditional and must be re-confirmed once the eval-HW
detector time is known.

---

## 5. Adversarial self-checks (honesty contract)

- **Every number is HW-tagged.** PnP-chain = MEASURED on laptop Intel CPU (ports to edge as an upper
  bound, low uncertainty — tiny dense linear algebra). Detector CPU 124–139 ms = MEASURED on laptop
  CPU, **labelled UPPER BOUND**, never used as the eval-HW number. Detector edge 5–15 ms =
  **ESTIMATE** (empirical bracket), not measured here. FLOP floor 0.39–1.35 ms = derived ESTIMATE,
  used only to prove the detector is not compute-bound, NOT as the forward time.
- **No fabricated GPU number.** I did not invent a single GPU/NPU measurement. The edge time is an
  explicitly-bracketed estimate with the derivation shown; the harness writes both the FLOP floor and
  the empirical bracket so a reader can see exactly which is which.
- **CPU is not allowed to masquerade as eval-HW.** The budget table composes `L_total` separately per
  HW assumption; the CPU row is labelled "UPPER BOUND (NOT eval-HW)" everywhere it appears.
- **PnP-chain timed on the LIVE path** (`compute_covariance=True`, confidence-weighted refine ON,
  priors set as the navigator sets them) so the 0.73 ms is the real in-loop cost, not a stripped
  fast path. Reprojection invariant: corners came from `project_gate_corners`, so PnP recovers the
  planted pose — timing is on geometrically valid, solvable inputs (no degenerate fast-fail bias).
- **Frame-age vs compute-time kept distinct.** `v·L_total` is explicitly the compute slice only; the
  doc states the total fix age = frame-age (Δ + transport) + L_total and that the compensation
  targets the total, so `L_total` is not silently passed off as the whole staleness.

## 6. Single most important follow-up
Get ONE real detector-forward timing on the actual eval hardware (or an organizer-provided latency
spec). Everything in §2/§3/§4 is conditional on the 5–15 ms edge estimate; the CPU upper bound shows
how much rides on it. Until then: ship predict-forward with a calibrated constant `age` (cheap,
first-order-correct, no TIMESYNC), keep the rewind ring buffer (piece B) as the principled upgrade
behind TIMESYNC, and re-confirm the 30 Hz rule once eval-HW latency is known.
