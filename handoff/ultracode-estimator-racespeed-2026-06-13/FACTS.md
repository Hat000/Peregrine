# ESTIMATOR-RACESPEED — shared facts (read first)

Mission: can the vision-fed KF reach **<0.05 m 1-sigma position error at the post-gate-3 ~37 m/s
gate-4 window** for a VALID (zero-contact) run? The gate-4 contact-true margin is **0.155 m @ r=0.38**
(binding gate, registration-confirmed). Vision world-fix unfiltered East sigma ~0.47 m = 3.0x the
margin. CONDITIONAL on case C (vision-only pose). If VQ2 streams LOCAL_POSITION_NED/ODOMETRY (case A/B)
pose is pristine and the risk is moot — build the worst-case (case-C) readiness, flag the conditionality.

## Environment (ALL agents)
- Repo root: `C:\Users\Fengy\Downloads\Projects\Anduril`
- Python: `.venv/Scripts/python.exe` (3.13, has numpy/scipy). ALWAYS set `PYTHONPATH=src` so
  `import racer...` works. Example (PowerShell): `$env:PYTHONPATH="src"; .venv\Scripts\python.exe foo.py`
  or bash: `PYTHONPATH=src .venv/Scripts/python.exe foo.py`.
- Write ALL outputs under `handoff/ultracode-estimator-racespeed-2026-06-13/` with a UNIQUE filename
  prefix per agent (a1_*, a2_*, b1_*, c1_* ...) so concurrent agents never collide.
- SAFETY: offline analysis + prototype/sim ONLY. NO live sim, NO edits to `src/`, NO commits, NO SLURM,
  NO memory edits. Build on the case-C prototypes; do NOT re-derive them — measure their accuracy.

## The real KF (compose it; do NOT re-implement) — `src/racer/state_estimator.py`
- `LinearKF`: 6-state [pos(3), vel(3)] world-NED. `LinearKF.initialize(pos, vel, pos_std, vel_std)`.
- `predict(accel_body, R_world_body, dt)`: a_world = R_wb @ accel_body + g (g=[0,0,9.80665] NED).
  Q = process noise from `accel_noise_std=0.3` m/s^2 (isotropic) + `attitude_noise_std` (=1.4deg,
  `ATTITUDE_NOISE_STD_RAD`) projected through skew(specific_force) (PSD). `max_dt_s=0.2` guard.
- `update_position(z, cov)` / `update_velocity(z, cov)`: Joseph-form, SPD-preserving.
- In case C: predict on IMU, update_position from vision world-fix ONLY (no given pos/vel).
  Velocity is UNOBSERVABLE in case C (vision is position-only -> vel = IMU integration only).

## Fix covariance — `src/racer/localization.py` (`gate_pose_to_world_position`)
Cov(p) = PNP_FIX_COV_INFLATION(=2.0) * R_wc Sigma_pnp R_wc^T   (pixel-noise PnP block)
         + sigma_theta^2 (|L|^2 I - L L^T)                      (attitude lever, 1.4deg, grows with range)
         + FIX_COV_FLOOR_STD^2 I  (FIX_COV_FLOOR_STD = 0.40 m isotropic floor)
3-corner (P3P) fix: x9 inflation (`P3P_FIX_COV_INFLATION`). The 0.40 m floor is BIAS-ABSORPTION
(covers constant systematics: +0.3 m vertical, per-gate lateral, close-range depth) — keep this in mind
when asked "can the floor come down."

## Navigator in-loop gates — `src/racer/navigator.py`
- `vision_max_range_m = 32.0` (cap), association (shape), depth-sanity (range_consistent), Mahalanobis
  chi2 gate 16.27 (chi2 99.9% / 3 DOF). Acceptance ~47% of race-window frames; catastrophic leak 0.53%.
- Detector cadence 30 Hz (inherited DiffAero default). IMU/ODOMETRY ~75-97 Hz.

## Case-C prototypes to BUILD ON (handoff/ultracode-vision-case-c-2026-06-13/) — measure their accuracy
- `kf_rewind_buffer.py` — `RewindKF` OOSM wrapper around LinearKF. `update_position_at(t_fix_ns, z, cov)`
  rewinds to capture time, applies fix, replays IMU forward. BIT-EXACT vs in-order oracle (verified).
  SHARPEST RISK: horizon_s < true L -> drops ALL fixes, diverges to ~21 m. Default horizon_s=0.5.
- `range_anisotropic_R.py` — `R_aniso(t_cam_gate, R_wc, ...)` range-anisotropic 3x3 NED cov (r^4 depth
  law). In-range benefit ~ZERO (shipped analytic cov already carries r^4); value at >24 m + off-nominal.
  Coeffs in `range_R_coeffs.json` (c2=0.003125 depth, a1=0.026 lateral, sigma_theta=1.4deg, floor 0.40).
- `latency_harness.py` / `latency_results.json` — in-loop L: edge 6 ms p50 / 16 ms p90 (ESTIMATE,
  never measured on eval HW); laptop CPU 112-139 ms (upper bound, no GPU); PnP->KF chain 0.77 ms.
- `vision_cal.py` — robust per-gate median/MAD re-survey; recovers global offset = -MEASURED_FIX_BIAS.

## MEASURED per-fix world-fix error (the realistic fix stream) — GROUND TRUTH for the sim
Source: `handoff/perception-char-2026-06-08/characterize_course_60s.json` -> `rows` (240). Each row has
`off_ned` (= pos_fix - true_drone, world NED, m), `range_m`, `pose_range_m`, `true_range_m`,
`range_err_m`, `reproj_px`, `n_corners`, `world_fix_err_m`, `maha`, `associated`, `score`.
Computed natively (commander, .venv) over GOOD fixes (|off|<3 m, n=109, the in-loop-accepted-like cut):
- **per-fix BIAS NED = [-0.285, +0.064, -0.346] m** (N, E, D)
- **per-fix NOISE std NED = [0.816, 0.577, 0.436] m** (N, E, D)
- range bands (good): [0,8) n=37 bias[-.376,-.003,-.255] std[.97,.55,.57]; [8,16) n=60 bias[-.177,+.017,-.422]
  std[.73,.56,.29]; [16,24) n=12 bias[-.548,+.509,-.249] std[.55,.55,.51].
- range only reaches **23.3 m max** in the data (recorded at ~5.35 m/s median). n_corners: 190x4-corner,
  13x3-corner. VISION-PKG2 canonical headline (KF-accepted post-gate cut): sigma~[0.73,0.47,0.29],
  bias~[-0.42,+0.06,-0.28], range-flat to ~24 m, acceptance ~47%, leak 0.53%.

## GATE-4 GEOMETRY (the window) — from track_map (handoff/shadowpc-firstcontact-2026-06-02/track_map.json)
Gate positions (bottom-centre, NED; opening-centre ~1.36 m above in -D):
g0[-23.3,-0.4,-0.03] g1[-46.9,-2.5,5.07] g2[-74.6,1.2,13.67] g3[-111.5,-5.1,24.57] g4[-135.5,-0.8,25.36] g5[-159.2,-4.4,25.97].
- **g3->g4 = 24.4 m, nearly LEVEL (0.8 m descent).** g2->g3 = 39.0 m (10.9 m descent) = where speed builds.
  Post-gate-3 the drone is at ~37 m/s on the level g3->g4 straight approaching gate-4.
- Motion at gate-4 approach is ~along -N (g3 N=-111.5 -> g4 N=-135.5). Gate-4 normal ~ -N.
  => IN-PLANE axes at gate-4 = **E (lateral)** and **D (vertical)**; ALONG-TRACK axis = **N**.
  => The binding in-plane miss = sqrt(E_err^2 + D_err^2). The N error is along-track (affects WHEN you
     cross the plane, not the in-plane miss). Latency staleness v*L is ~along-track (N) at gate-4.

## THE VARIANCE vs BIAS SPLIT (the crux — every measurement MUST separate these)
The <0.05 m "1-sigma" bar is a VARIANCE target (margin/3). But total in-plane error = BIAS + NOISE.
- A KF crushes ZERO-MEAN noise by averaging fixes; it does NOT remove a consistent BIAS (it tracks it
  as real drift). VISION-CAL global de-bias removes the GLOBAL offset; the PER-GATE/PER-TRACK residual
  (case-C reports residual registration sigma ~[0.21,0.24,0.03] m N,E,D ACROSS gates) is a CONSTANT
  offset within one track for one gate -> does NOT average out within the gate-4 approach.
- So the binding question is THREE-fold: (1) does the filtered VARIANCE reach <0.05 m? (averaging; depends
  on fix rate, sigma_meas, process-noise growth between fixes); (2) is the residual per-track in-plane
  BIAS at gate-4 < ~0.05 m after de-bias? (un-filterable floor); (3) latency staleness (mostly along-track
  at gate-4). Report all three. Gate-RELATIVE observation (close on the SEEN gate corners for centering)
  sidesteps (2) entirely and is the likely highest-leverage fix if absolute can't reach the bar.

## Determinism note
Determinism is PER-TRACK (the course/seed fixed within an attempt; offline line-iteration legal). A
per-track CONSTANT bias is the same every lap of that track — relevant to whether a 2nd calibration lap
can remove it. Speed > keeping gate in view.
