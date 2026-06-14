# D2 — CASE-C ESTIMATOR CHAIN SPEC (build-ready)

Fengyou — this is the build-ready chain spec for **Component 2: the case-C estimator chain**. It
productionizes the vision-case-c prototypes into the live `Navigator` and resolves the 3 P0 bugs.
Scope: DESIGN + offline validation only — NO edits to `src/` were made; every load-bearing number
below was re-derived this run (sim re-run or source re-read), labelled MEASURED / EXTRAPOLATED /
ASSUMED. Offline checks live next to this file:
- `d2_relinnov_gate_check.py` (+ `_results.json`) — relative-innovation outlier gate vs reproj.
- `d2_horizon_cov_check.py` (+ `_results.json`) — RewindKF horizon sizing + the horizon<L trap +
  the calibrated covariance handed to component 1.

Reproduced load-bearing facts (this run):
- Gate-relative in-plane RMS **0.139 m** (clears 0.155 m margin), p90 0.203 m (over); absolute 0.279 m,
  submap anti-pattern 0.228 m — both FAIL. `c1_gate_relative.py` re-run. **MEASURED (sim).**
- PnP→KF chain **0.77 ms p50 / 0.90 ms p90** (laptop CPU); detector CPU 112 ms p50 / 124 ms p90 (upper
  bound, no GPU); edge detector 5–15 ms (ESTIMATE). `latency_results.json` re-read. **MEASURED/ESTIMATE.**
- Reproj does NOT separate depth-flips: flip p50 **0.445 px** < clean p50 **0.657 px** (perception-char
  bundle, re-derived). Relative-innovation gate at chi2(2)=13.82 rejects **99.8%** of flips while
  keeping 99.9% clean; a reproj gate lets **93.2%** of flips through. **MEASURED (data) + MC.**
- horizon<L → 100% fixes dropped → in-plane RMS 0.423 m (divergence); horizon>L → 0% drop, RMS 0.146 m,
  calibrated in-plane sigma 0.131 m. `d2_horizon_cov_check.py`. **MEASURED (sim).**

---

## 1. THE FULL LIVE CHAIN (where each piece plugs in)

The chain is an AUGMENT of the existing `Navigator.update` → `_maybe_run_vision` →
`_process_observation` path. The absolute `LinearKF` stays exactly as-is for planning/feed-forward and
the g4→g5 hand-off; the gate-relative term is added as a SECOND, in-plane-only correction that owns the
terminal centering miss. Line numbers are against the current `src/racer/navigator.py` (HEAD 186a692).

### 1.0 Wrap the KF in RewindKF (case-C only)
- **Where:** `Navigator._initialize` (navigator.py:255-271) builds `self.kf = LinearKF.initialize(...)`.
  In case C, wrap it: `self.kf = RewindKF(kf=LinearKF.initialize(...), horizon_s=0.5)`.
- **Interface:** `RewindKF` (handoff/ultracode-vision-case-c-2026-06-13/`kf_rewind_buffer.py`) proxies
  `x/P/position/velocity` so `make_nav_state` and `_mahalanobis_position` read it unchanged. The ONLY
  call-site changes: `predict(accel_body, R_wb, dt, sim_time_ns)` now takes the IMU stamp (navigator.py:313),
  and `update_position` takes an optional `sim_time_ns` (navigator.py:316/321). Vision fixes call
  `update_position_at(t_fix_ns, z, cov)` instead of `update_position` (navigator.py:397). **Verified:**
  bit-identical to bare KF for zero-latency fixes (prior REPORT, piece B).
- **Data needed:** the IMU master clock `ds.sim_time_ns` (already in hand) for predict/given updates; the
  vision **capture** time `t_fix_ns` on the SAME clock (this is what P0-bug-2 TIMESYNC delivers).

### 1.1 detector → PnP (`estimate_gate_pose`) — UNCHANGED
- **Where:** `_process_observation` navigator.py:363 already calls
  `estimate_gate_pose(obs, prior=prior, compute_covariance=True)`. The PnP `prior` is the fresh
  map+attitude+KF prediction `predicted[gate_id]` (navigator.py:360-362) — keep this exactly; it is what
  breaks the IPPE 2-fold tie. **No change.** Output: `GatePose.t_cam_gate` (gate origin in camera frame),
  `R_cam_gate`, `reproj_error_px`, `covariance` (6×6), `n_corners`.

### 1.2 association — UNCHANGED
- **Where:** `_associate` navigator.py:402-405 (`associate(obs, predicted, ...)` shape gate). Keep. It
  binds `obs → gate_id` so the lever is referenced to the RIGHT map gate (and, for the relative term,
  the right SEEN opening). The depth-sanity `range_consistent` (navigator.py:374) stays — it is an
  upstream defense the relative-innovation gate does NOT replace.

### 1.3 THE GATE-RELATIVE FIX — observe −L to the SEEN opening (NEW)
- **Math.** The PnP lever (gate rel. drone, world NED) is `L = R_wc @ t_cam_gate`, exactly as
  `localization.gate_pose_to_world_position` builds it (localization.py:85-86). The drone's offset from
  the **seen** gate-4 opening is `e_obs = -L` projected into the gate plane. Because `e_obs` is referenced
  to the SEEN corners (the true opening) and NOT to `gate.position_ned` (the map), the per-track
  map/registration bias `db` drops out of the arithmetic — this is the whole fix. **DO NOT** compute
  `R_w2g @ (gate_map − p_KF)` ("subtract gate map pos"): that re-injects `db` (the submap anti-pattern,
  MEASURED 0.228 m vs rel 0.139 m, c1_gate_relative).
- **Where:** in `_process_observation`, AFTER `gate_pose_to_world_position` (navigator.py:380) and AFTER
  the existing absolute `update_position` (navigator.py:397, kept), add a second correction:
  1. `L = R_wc @ pose.t_cam_gate` (`R_wc = R_wb @ R_camera_from_body().T`).
  2. Build the gate-frame in-plane basis from the associated `gate.R_world_gate` (X=lateral, Y=vertical;
     gate +Z = through-axis). At gate-4 the in-plane axes are world (E, D), along-track is N (FACTS).
  3. `e_obs_ip = P_ip @ (-L)` where `P_ip` projects onto the gate plane (drop the along-track component).
     The OBSERVATION the KF consumes is a world-position pseudo-fix that constrains ONLY the in-plane
     axes: `z_rel = gate.position_ned - L` but with the along-track (gate-normal) component left
     un-constrained via the covariance (see 1.5). Equivalently: a 2-DOF in-plane update with
     `H_ip = [in-plane basis | 0_{2x3}]`. **Recommended implementation:** keep it a 3-DOF
     `update_position` with an ANISOTROPIC cov that is tight in-plane (PnP lateral sigma) and LOOSE
     along-track (gate-normal), so no new KF method is needed — the existing `LinearKF.update_position`
     already does the general linear update. The along-track stays owned by the absolute fix + IMU.
- **Interface:** a new helper `gate_relative_inplane_fix(pose, gate, R_wb) -> (z_ned, cov_ned)` (mirror
  of `gate_pose_to_world_position` but cov shaped in the gate-plane frame; see 1.5). It must NOT add the
  0.40 m `FIX_COV_FLOOR_STD` to the in-plane axes — that floor is BIAS-ABSORPTION, and the relative obs
  has no bias to absorb (MEASURED: rel arm uses no floor and reaches 0.139 m; with the floor it would not
  clear the margin). Keep the floor on the along-track axis (where range/depth bias still lives).
- **Data needed:** `pose.t_cam_gate`, `R_wb` (true attitude, `R_world_from_odo_quat_wxyz`), the
  associated `gate.R_world_gate` (gate-plane basis), `gate.position_ned`.

### 1.4 RewindKF OOSM update at capture time (NEW call-site)
- **Where:** replace `self.kf.update_position(position_ned, cov)` (navigator.py:397) with
  `self.kf.update_position_at(t_fix_ns, position_ned, cov)` for BOTH the absolute and the gate-relative
  fix (each is an OOSM correction stamped at the same `t_fix_ns`). `t_fix_ns` = the vision **capture**
  time on the IMU clock (P0-2). **Verified:** OOSM exact vs in-order oracle, SPD-preserving (prior REPORT).
- **Fallback (predict-forward):** if TIMESYNC (P0-2) is not yet wired, set `t_fix_ns = now - L_const`
  using a single calibrated constant age `L_const` (= measured L). This is the cheap 80% of the rewind
  win and needs no capture timestamp — ship it first (prior REPORT P1-1). Residual gap to full rewind is
  `0.5*a*age^2` (<3 mm at edge L). **MEASURED/EXTRAPOLATED.**

### 1.5 range-anisotropic R (cov shaping)
- **Where:** the cov fed to `update_position_at`. For the ABSOLUTE fix keep
  `gate_pose_to_world_position`'s cov (unchanged; it already carries the r⁴ depth law in-range — R_aniso
  is a re-parameterization with ~zero in-range win, prior REPORT piece C). For the GATE-RELATIVE in-plane
  fix, shape the cov in the gate-plane frame:
  - in-plane (E,D at gate-4): `sigma_lat = 0.265 m/axis` (MEASURED accepted gate-4 lateral, c1) growing
    as `a1*r` at longer range (`range_anisotropic_R.py` lateral law) — NO 0.40 m floor.
  - along-track (gate-normal): LOOSE — the absolute fix + IMU own it. Set ~the absolute radial sigma
    plus the 0.40 m floor so the relative term never fights the absolute term on depth.
  - assemble via the LOS/gate-plane projectors `R_aniso` already provides (`_perp_projector`,
    `outer(Lhat,Lhat)`); pass `coeffs=load_coeffs()` (`range_R_coeffs.json`: c2=0.003125 depth,
    a1=0.026 lateral, sigma_theta=1.4°, floor 0.40). **Caveat (carry):** a1=0.026 is an 11× band-floor
    that EXTRAPOLATES BADLY past ~24 m — the depth axis extrapolates faithfully, the lateral does not
    (prior REPORT piece C). The gate-relative fix is consumed only inside ~12 m of gate-4, so this is
    in-band; flag it for the >24 m regime.

### 1.6 latency predict-forward fallback
- Already covered in 1.4: predict-forward = `update_position_at(now - L_const, ...)`, the TIMESYNC-free
  path. RewindKF degenerates to in-place when `t_fix >= now`, so a single code path handles both
  (zero-latency, predict-forward constant age, and full OOSM capture time).

---

## 2. THE 3 P0 BUGS — concrete fixes

### (a) `_initialize` case-C cold-start crutch (navigator.py:255-271)
- **Bug.** `_initialize` reads `ds.position_ned` with NO `use_given_position` guard (navigator.py:257-261,
  267). If the sim still streams LOCAL_POSITION_NED, flipping `config.use_given_position=False` does NOT
  produce a vision-only cold-start — tick 1 seeds a hidden ground-truth pose at `pos_std=0.05`, so every
  case-C "test" is secretly case A. **CONFIRMED by source read.**
- **Fix.** Gate the seed on the config flag, not on message presence:
  ```
  use_gp = self.config.use_given_position and ds.position_ned is not None
  pos     = np.asarray(ds.position_ned, float) if use_gp else np.zeros(3)
  pos_std = self.config.given_pos_std if use_gp else 5.0
  vel     = np.asarray(ds.velocity_ned, float) if (self.config.use_given_velocity
            and ds.velocity_ned is not None) else None
  ```
  True case C (`use_given_position=False`) then seeds **origin @ pos_std=5.0** (P[0,0]=25) regardless of
  whether LPN is on the wire — every case-C test becomes genuinely vision-only. Same guard belongs on the
  per-tick given updates (navigator.py:315, 320) — those ALREADY have the `config.use_given_position and
  ds.position_ned is not None` guard, so only `_initialize` is the leak. **This is the gate to validating
  everything else** (prior REPORT P0-1).
- **Test rider.** Add a case-C test that passes `position_ned=NOT None` on the wire WITH
  `use_given_position=False` and asserts the seed is origin@5.0 (catches the regression directly).

### (b) TIMESYNC epoch reconciliation (navigator.py:426, RewindKF prereq)
- **Bug.** `frame.sim_time_ns` is the server **UNIX epoch**; `DroneState.sim_time_ns` is the IMU
  **sim-boot epoch** (contracts.py:18-24: sim_time_ns driven SOLELY by HIGHRES_IMU.time_usec). They are
  distinct and unreconciled. Two consequences: (i) `time_since_vision_update_s` (navigator.py:423-426)
  subtracts the two epochs → garbage tsv (benign at VQ1 where tsv is unused, load-bearing in the case-C
  coast/abort policy); (ii) RewindKF indexed by `obs.sim_time_ns` would rewind to a garbage epoch.
  **CONFIRMED by source read + contracts docstring.**
- **Fix.** Reconcile ONCE to the IMU master clock. On the first frame, learn the offset
  `delta_epoch = frame.sim_time_ns - ds.sim_time_ns` (captured at near-simultaneous arrival, using
  `recv_monotonic_ns` to pair the closest IMU sample), then index everything by
  `t_fix_ns = frame.sim_time_ns - delta_epoch` (= the capture time on the IMU clock). Store `delta_epoch`
  on the Navigator; re-learn on `reset()` (epoch restart). For tsv (navigator.py:426): set
  `self._last_vision_sim_time_ns = t_fix_ns` (IMU-clock) at navigator.py:400 instead of
  `obs.sim_time_ns`, so the subtraction at navigator.py:426 is same-clock and correct.
- **Verification (live, escape-hatch).** `delta_epoch` must be confirmed against a live wire trace — the
  offset and any drift between the two clocks cannot be checked offline. Until verified, the predict-forward
  constant-age fallback (1.4/1.6) sidesteps this entirely (uses `now - L_const`, no capture timestamp).
  **This makes TIMESYNC a HARD prereq for full RewindKF but NOT for predict-forward** (ship order: P0-1 →
  predict-forward → TIMESYNC → full RewindKF).

### (c) velocity-unobservable (hand off to component 3/4)
- **Fact.** In true case C, vision is position-only; velocity is observed ONLY through position-fix
  differencing inside the KF (the `LinearKF` couples pos/vel through F), driven by IMU integration of
  `accel_body` between fixes (state_estimator.py predict). There is NO direct velocity measurement —
  accel bias integrates into both position AND the planner velocity (prior REPORT P3-1). **CONFIRMED.**
- **Decision.** Do NOT build a vision-velocity measurement channel (over-build; drift is bounded by the
  RewindKF + fix rate, prior REPORT defers it). Velocity stays the KF's pos/vel-coupled estimate.
- **INTERFACE to component 3/4 (the velocity-prior design).** The estimator EXPORTS, per tick, on the
  NavState: (1) `velocity_ned = kf.velocity` (the coupled estimate, already on NavState); (2) the velocity
  block of the covariance `P[3:6,3:6]` (already on `NavState.pos_vel_covariance`). Component 3/4 owns the
  **velocity prior** that swings the gate-relative margin (warm/lap-converged 0.11 m vs cold 0.17–0.21 m,
  CONTEXT): the policy obs `vel_g = R_w2g @ vel` (fly_rl `obs_from_zup`) consumes `kf.velocity` directly.
  The estimator's contract is: deliver the best pos/vel-coupled velocity it can AND its covariance; the
  policy/DR design decides how much to trust it (uncertainty-aware obs, component 1). **No new estimator
  surface** — the existing `NavState.velocity_ned` + `pos_vel_covariance` ARE the interface.

---

## 3. THE GATE-RELATIVE FIX AS AN AUGMENT + the relative-innovation outlier gate

### 3.1 AUGMENT, not replace
- Keep the absolute `LinearKF` (now RewindKF-wrapped) running on the absolute world-fix for: planning
  carrots, feed-forward, and the **g4→g5 hand-off** (the absolute frame is what carries the drone between
  gates). The gate-relative in-plane term owns ONLY the **terminal in-plane centering miss** at the
  active gate, applied inside the visibility window (~12 m of gate-4, MEASURED visibility holds 4-corner
  to ~4 m, c1 part_d). Both corrections are OOSM updates to the SAME KF state — the relative term is a
  tighter in-plane measurement layered on top of the looser absolute one. **MEASURED:** the absolute arm
  alone misses (0.279 m); adding the relative term reaches 0.139 m, clearing the 0.155 m margin.

### 3.2 The REQUIRED relative-innovation outlier gate (math + threshold)
- **Why a new gate.** Reprojection error does NOT separate depth-flips: MEASURED flip p50 **0.445 px** is
  BELOW clean p50 **0.657 px** (perception-char bundle, re-derived this run). A reproj threshold tuned to
  keep 99% of clean fixes lets **93.2%** of depth-flips survive (MC, d2_relinnov_gate_check). The existing
  absolute Mahalanobis gate (navigator.py:391, chi2(3)=16.27 on world position) catches gross
  teleports but a depth-flip that lands near the absolute prior can slip it — the BINDING quantity is the
  in-plane lever, so the gate must test the **relative innovation**.
- **Math.** Let `e_obs = P_ip @ (-L)` be the observed in-plane offset to the seen opening (2-vector in the
  gate plane), and `e_pred = P_ip @ (gate.position_ned - x_KF[:3])` the KF-propagated relative prior.
  Innovation `nu_ip = e_obs - e_pred`. Innovation cov `S_ip = H_ip P H_ip^T + R_ip` (the 2×2 in-plane
  block of `P_pos + R`, with `R_ip` the in-plane gate-relative cov from 1.5, NO floor). Gate:
  ```
  d2_rel = nu_ip^T S_ip^-1 nu_ip   ;   ACCEPT iff d2_rel <= chi2(2, 0.999) = 13.82
  ```
- **Threshold.** chi2 99.9% / **2 DOF** = **13.82** (in-plane is 2-DOF; the absolute gate is 3-DOF=16.27).
  **MEASURED (MC):** at 13.82 the gate keeps 99.9% of clean fixes and rejects 99.8% of depth-flips
  (d2 clean p99.9 = 13.30 sits below d2 flip p1 = 16.9 — clean separation). This is the gate that earns
  the relative term its keep; ship it WITH the relative fix, never without.
- **Order.** Apply the absolute fix (existing gates: range cap, assoc, depth-sanity, abs-maha) FIRST;
  then compute the relative innovation against the just-updated prior and apply the relative in-plane fix
  ONLY if `d2_rel <= 13.82`. A rejected relative fix leaves the absolute estimate intact (graceful).

---

## 4. RewindKF HORIZON SIZING + the horizon<L trap + the calibrated covariance output

### 4.1 Horizon sizing vs measured L
- **Measured L** (latency_results.json, re-read): edge L_p50 **6 ms** / L_p90 **16 ms** (ESTIMATE — eval
  HW); laptop CPU **112–125 ms** (UPPER BOUND, no accelerator). PnP→KF chain 0.77 ms is negligible.
- **The trap (MEASURED, d2_horizon_cov_check).** horizon `< L` → the oldest buffered op is newer than the
  fix → EVERY fix is `applied=False` (dropped) → the KF dead-reckons → divergence. Re-derived: at
  horizon 8 ms < L 16 ms, **100% of fixes dropped**, in-plane RMS **0.423 m** and post-sigma inflates to
  0.598 m; at horizon ≥ L, **0% dropped**, RMS **0.146 m**. The prior verifier's ~21 m divergence is the
  same trap over a longer track.
- **Off-by-equality sharpening (NEW, from this run).** At `horizon == L` exactly (100 ms == 100 ms) the
  check STILL shows 100% dropped: `_prune` drops ops where `(ref_ns - op_ns) > horizon`, and after the
  latest predict advances `now`, a fix exactly at the horizon edge falls just outside. **Sizing rule must
  be STRICT:** `horizon_s > L_p99 + frame_age + pad`, not `>=`.
- **RECOMMENDED horizon = 0.5 s.** Covers edge L (16 ms) with vast margin, AND covers the CPU-class
  upper bound (125 ms) so the buffer is safe even if eval HW lacks an accelerator. Cost is negligible
  (~0.19 ms/fix, ~30 KB; prior REPORT). The 0.5 s default in `kf_rewind_buffer.py` is correct AS LONG AS
  real L < ~0.45 s — which both the edge estimate and the CPU upper bound satisfy. **Do not shrink it.**
- **HARD-blocked on:** P0-2 (measure L on eval HW to confirm < 0.45 s) and P0-3 (TIMESYNC, so the buffer
  is indexed by the IMU clock not a garbage epoch). Until both land, ship predict-forward (constant age),
  which has no horizon and no timestamp dependency.

### 4.2 Calibrated covariance output → component 1's uncertainty channel
- **What to export.** After each (rewind) fix, the KF posterior `P[:3,:3]` is the per-tick position
  covariance — already carried on `NavState.pos_vel_covariance` (state_estimator.py make_nav_state). For
  the gate-relative speed-vs-validity coupling, component 1 wants the **in-plane** confidence:
  ```
  P_ip = P[[1,2],[1,2]]   (gate-4: E,D block; general: project P[:3,:3] into the gate plane)
  inplane_sigma = sqrt(trace(P_ip) / 2)     # scalar 1-sigma, m
  ```
- **Calibration (MEASURED, d2_horizon_cov_check).** When fixes land (horizon>L), `inplane_sigma` = 0.131 m
  tracks the true terminal in-plane error (RMS 0.146 m) — i.e. P is a USABLE, well-calibrated confidence
  channel (slightly conservative, the safe direction; consistent with the prior REPORT's NEES ~2.75
  under-confident finding for RewindKF). When fixes DROP (mis-sized horizon / coast), `inplane_sigma`
  inflates to 0.598 m — the channel correctly signals "uncertain → be careful." This is exactly the
  fast-when-confident / careful-when-uncertain signal component 1 feeds the policy.
- **Interface to component 1.** Per tick, alongside the existing NavState, expose:
  `inplane_sigma` (scalar, m), and optionally the (E,D) per-axis std `sqrt(diag(P_ip))`. Component 1 owns
  the obs-channel normalization + the speed-preserving reward; the estimator's contract is just to deliver
  a calibrated `P_ip`. **No retrain coupling here** — the estimator simply makes P honest.

---

## 5. RESIDUAL RISKS / OPEN QUESTIONS (estimator scope)
- **L on eval HW is UNMEASURED** (P0-2). If eval HW is CPU-class (~125 ms), v*L staleness is 2.3–4.2 m at
  20–30 m/s and the speed thesis flips to NO-GO. Resolvable by ONE eval-HW timing run. **EXTRAPOLATED.**
- **TIMESYNC offset/drift** (P0-3) cannot be checked offline — needs a live wire trace. Predict-forward
  sidesteps it for the cheap 80%. **ASSUMED until live.**
- **Velocity prior swings the margin** (CONTEXT): warm 0.11 m vs cold 0.17–0.21 m straddles 0.155 m. The
  estimator delivers the coupled velocity + its covariance; component 3/4 owns the prior. CONDITIONAL-GO.
- **Lateral cov a1 extrapolates badly past ~24 m** (prior REPORT piece C) — in-band for the ~12 m
  gate-relative window, flag for long range.
- **The gate-relative fix is validated in SIM only** — the end-to-end true-vision-only × VQ2-speed ×
  eval-HW operating point has never run on anything (prior REPORT). The physics is proven; the operating
  point is not.
