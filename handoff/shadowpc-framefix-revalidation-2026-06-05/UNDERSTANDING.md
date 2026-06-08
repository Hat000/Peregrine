# Frame-fix re-validation — PRE-REGISTRATION (Phase 0)

**Date:** 2026-06-05 · **Branch:** red-team-tier-a (worktree at `5075434`, contains fix `c3b5a8e`)
**Mode:** OFFLINE replay of existing recordings. No live sim, no flight, no controller change, no re-tuning.
**One hypothesis:** the ODOMETRY body→world velocity frame bug (fixed in `c3b5a8e`) is the root cause of
the gate-0 saga's *"KF velocity lags 4×"* (`project_ctbr_control_sysid.md` §5: truth −0.92 m/s, KF
reported −0.22) and the lateral oscillation. Confirm or refute against a real saga recording.

A result of *"the fix doesn't fully explain it"* (Partial) is a SUCCESS if true. Report numbers, not hopes.

---

## 1. The fix under test (`c3b5a8e`)

- `frames.world_vec_from_body_quat(v_body, q_wxyz)` → `R(q) @ v_body` (q scalar-first w,x,y,z → scipy
  [x,y,z,w]); degenerate-quat guard returns input unchanged.
- `mavlink_client._handle` ODOMETRY branch (line 263–265): **was** `velocity_ned = [msg.vx,msg.vy,msg.vz]`
  (raw body twist stored as world); **now** `velocity_ned = world_vec_from_body_quat([vx,vy,vz], q)`.
- `LOCAL_POSITION_NED` branch (line 244): `velocity_ned = [msg.vx,msg.vy,msg.vz]` (already world) — **unchanged**.
- Regression test `test_odometry_velocity_rotated_body_to_world`: yaw=180° + body vx=+1 ⇒ world `[-1,0,0]`.

**Why it could corrupt:** ODOMETRY (75 Hz) and LOCAL_POSITION_NED (96 Hz) both write the *same*
`DroneState.velocity_ned`, **last-writer-wins**. Pre-fix, ODOMETRY wrote a body-frame velocity and LPN a
world one. At the −X / yaw≈−180° course heading the two are sign-flipped (`Rz(−180)` negates x and y), so
`velocity_ned` flickered between `+v` and `−v` → the consumers saw a partially-cancelled small magnitude.

## 2. Data-flow facts already established from the code (grounds Phase 3 / Check 0)

- **navigator.py:281–285** — the KF velocity update consumes `ds.velocity_ned` directly, tight
  (`given_vel_std = 0.10`, "authoritative for VQ1"). So the KF measurement *is* the mixed field.
- **fly_vq1.py:355–358** — default (not `--use-kf-state`) control path:
  `gs = client.state` (line 355, the raw mavlink-client `DroneState`); `rawv = gs.velocity_ned` (356);
  `vel = [rawv[0], rawv[1], kf_vz]` (357). So the **"raw given horizontal velocity" workaround consumed
  `ds.velocity_ned[0:2]` — the identical interleaved mix.** It only swaps in the KF *vertical* velocity.
  ⇒ Working hypothesis for Phase 3: the lateral-oscillation workaround was *also* corrupted at −180°
  (fighting a phantom), not a clean pristine velocity. To be quantified on the real stream.

## 3. ADD #1 — the sim-declared frame enums (definitive, not inferred)

Read directly off every ODOMETRY message in the recordings (`select_probe.py`):

| field | value | MAV_FRAME name | meaning |
|---|---|---|---|
| `ODOMETRY.frame_id` | **1** | `MAV_FRAME_LOCAL_NED` | the **world** frame the *pose* (x/y/z) lives in |
| `ODOMETRY.child_frame_id` | **8** | `MAV_FRAME_BODY_NED` | the **body** frame the *twist* (vx/vy/vz + rates) lives in |

Constant across **100%** of ODOMETRY messages in every run checked. This is the definitive, sim-specific
confirmation that the velocity is declared in a **body** child-frame distinct from the world pose frame —
stronger than inferring it from the vx sign-flip. (Note: the sim uses the *deprecated* `BODY_NED`=8 rather
than `BODY_FRD`=12 that the fix comment names; both are body-fixed, z-down. The exact body convention is
validated empirically in Phase 1 ADD #2 — the quaternion that rotates it is the same ODOMETRY quaternion,
so the rotation is correct as long as twist and quaternion share the body frame.)

## 4. Recordings selected

Selection probe over all gate-0 candidates (`scratch/select_probe.py`). Picked **two**, with distinct roles:

### PRIMARY — `20260604_025347_gate0_course1` (the decisive KF replay)
- FINISHED, gate_index=1, 0 collisions, 77.5 s. Dense dual-stream: **LPN 96 Hz + ODOMETRY 75 Hz** (+ IMU
  119 Hz), 0 drops → faithful interleaving + clean replay.
- Yaw locked at the course heading **−180°…−175°** the whole approach (x: −23.5→0). No wandering.
- Bug present: at −180°, body vx +0.31 → rotated-world vx −0.30 (sign-flip), 482 flips.
- Velocity signal: **world vx −4.69 m/s** (forward, heavily sign-flipped) + **world vy max 0.49 m/s**
  (lateral, modest). Attitude: pitch to −17.8°, **roll only ±1.6°** (near-zero roll).
- *Role:* cleanest possible decisive replay (minimal confounds). Forward axis gives an enormous OLD-vs-NEW
  signal; the modest lateral exercises the §5 cross-track axis.

### SECONDARY — `20260604_145110_gate0_given1` (ADD #2 cross-attitude validation + §5 magnitude)
- FINISHED, gate_index=1, 1022.8 s, full rate (LPN 95 Hz + ODOMETRY 75 Hz), 0 drops.
- **Attitude-varied: 73,207 samples with |roll|>5° AND |pitch|>5° simultaneously; peak roll 53.4°,
  peak pitch 17.8°.** The only run that exercises roll *and* pitch together (course1/roll1/roll3 barely roll).
- Lateral reaches **world vy 2.40 m/s** → brackets §5's −0.92 m/s figure on a genuine cross-track move.
- Contains a ~49 s near-180° segment (3647 ODOMETRY samples) for a second KF-replay cross-check.
- Caveat: yaw wanders (−180°…+142°) over the long run, so the −180° segment must be sliced out.
- *Role:* validate the full-quaternion rotation OFF level (all 3 axes), and reproduce the §5-magnitude
  lateral undercount.

Rejected: roll2 (chaotic 47 m runaway, full yaw spin), roll3-132220 (aborted, 5 collisions, short),
given4/5 (diverged to −400/−37 m), given3 (3–5 Hz, no lateral). **roll1** is a near-miss (FINISHED, lateral
to 1.92 m/s) but **sparse (10–13 Hz)** — itself a recording-cleanliness data point; kept as a fallback.

> **Which run is §5's literal "−0.92 vs −0.22"?** Not positively identified. course1's lateral peaks at
> only 0.49 m/s, so the −0.92 figure came from a bigger-lateral run (given1/roll1 bracket it). The
> *mechanism* is testable on all of them; given1 carries the matching magnitude.

---

## 5. Pre-registered checks + numeric predictions

**Truth reference (both runs):** LPN world velocity, cross-checked independently against central-difference
Δposition/Δt of the LPN positions — they MUST agree (same triangulation discipline as the velocity-fork).

### Phase 1 / Check 1 — bug present in the saga data
- (P1a) Frame enums constant: frame_id=1, child_frame_id=8 across 100% of ODOMETRY. *[observed in selection;
  re-confirm formally]*
- (P1b) RAW: at −180°, ODOMETRY body velocity vs LPN world velocity **disagree, sign-flipped** on x (and y);
  |raw difference| ≈ 2|v|.
- (P1c) ROTATED: `world_vec_from_body_quat(body_v, q)` vs LPN world velocity **agree to RMS ≤ 0.05 m/s**
  over the −180° segment (course1).
- (P1d) Truth self-consistency: LPN world v vs Δpos/Δt agree to **RMS ≤ 0.05 m/s**.
- (P1e) **ADD #2** — across the attitude-varied segment (given1, |roll|>5 & |pitch|>5, peak roll 53°):
  rotated ODOMETRY vs LPN agree to **RMS ≤ 0.10 m/s** (looser, for 75/96 Hz time-alignment under faster
  dynamics). **Falsification:** if the off-level RMS is ≫ the at-level RMS, or shows a bias that grows with
  roll/pitch, the body convention / rotation is wrong → report it, do not paper over it.

### Phase 2 / Check 3 — decisive replay: does the fix make the KF track truth?
Re-run `Navigator` on the recorded message stream **twice**, identical except `velocity_ned`:
(a) **OLD** = raw body ODOMETRY interleaved with world LPN (last-writer-wins, reproducing the bug);
(b) **NEW** = ODOMETRY rotated body→world (the fix). Compare each KF velocity output to truth over the
−180° lateral segment. *(Reconstructed from raw messages; no pre-fix checkout needed.)*
- (P2a) **OLD** undershoots truth on the sign-flipped axes. Predict: forward axis (course1 truth ≈ −4.7
  cruise) OLD-KF **|error| ≳ 2 m/s** (magnitude collapses, may oscillate / approach zero); lateral-axis
  **|KF|/|truth| < 0.5**, qualitatively reproducing §5's −0.22/−0.92 ≈ 0.24.
- (P2b) **NEW** tracks truth to **RMS ≤ 0.15 m/s** on all axes (mild KF smoothing lag only; no 4× gap).
- **DECISION:** NEW tracks (≤0.15) AND the undercount appears *only* under OLD ⇒ **Confirmed** (§5 = frame
  bug → retire). NEW still ~4× / **≥0.5 m/s** error ⇒ **Partial** (a residual real lag remains).

### Phase 3 / Check 0 — data flow
- Restate with line refs (navigator.py:282-283, fly_vq1.py:355-358) that **both** the KF and the
  "raw given" horizontal workaround consumed `ds.velocity_ned` (the mix). Quantify on the recorded stream:
  fraction of −180° `velocity_ned[0:2]` samples that are ODOMETRY-sourced (corrupted) vs LPN-sourced, and
  the resulting error vs truth. Predict: the workaround velocity was **also corrupted at −180°** (fighting a
  phantom), so the lateral-oscillation fix was not operating on a pristine signal.

## 6. Pre-registered OUTCOMES (commit before analysis)

- **Confirmed** *(expected most likely)* — with the fix the re-run KF velocity tracks truth (no 4× gap), raw
  ODOMETRY velocity was body-frame (sign-flipped at −180°), and the rotation validates across pitch/roll
  ⇒ §5 was the frame bug → **retire §5's "KF lags 4×" finding** and recast the lateral workaround.
- **Partial** — the frame bug is real, but NEW-KF velocity still lags even on clean rotated velocity
  ⇒ a residual real lag remains (the fix helps but isn't the whole story); keep a reduced §5.
- **Refuted** — the recording shows no frame mismatch ⇒ the bug didn't affect this run; investigate.
  *(Already unlikely: the frame enums + observed sign-flips show the bug is present in the data.)*

The honest open question is **Confirmed vs Partial** (does NEW-KF fully track, or leave residual lag) — and
whether the rotation holds off-level (ADD #2). I will report whichever the numbers show.

## 7. Guardrails
Offline only. One hypothesis, measured. No controller changes / re-tuning / live sim / flight. If no clean
lateral-move-at-−180° segment exists, say so rather than forcing a weak one.
