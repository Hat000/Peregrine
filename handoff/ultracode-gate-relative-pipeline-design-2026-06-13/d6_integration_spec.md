# D6 — INTEGRATION + VALIDATION PLAN (gate-relative case-C VQ2 pipeline)

Component 6 of the gate-relative pipeline design. Scope: how the gate-relative estimator chain
(C2) + the gate-relative / uncertainty-aware obs (C1) + the inc8 policy (C5) wire into the live
autonomous stack behind `rl/submit_rl.py` / `rl/fly_rl.py` / `src/racer/navigator.py`; the ordered
**offline gauntlet** that must be green before any live run; the **live ShadowPC** validation order;
and the **dependency graph + build order** across all 6 components.

This is a PLAN. It edits nothing on `main`. Every load-bearing number below was re-derived this
session (sim re-run / source read) — flagged MEASURED / EXTRAPOLATED / ASSUMED inline.

Verification anchors read this session (paths absolute under repo root
`C:\Users\Fengy\Downloads\Projects\Anduril`):
- `rl/submit_rl.py` (authoritative judged entry — pins inc7, `--no-bridge --no-auto-reset
  --no-debug-obs --flights 1`; safety pins appended LAST so they win; NO `MAV_CMD 31000` ever).
- `rl/fly_rl.py` — the deploy loop: `obs_from_zup` / `build_obs` / `make_gate_map` / `GateMap`
  (P4-C05 foundation), the 3 loud guards (`_assert_vq1_constants_consistent` import-time,
  `_assert_live_course_is_vq1` deploy-time @ line 925, `assert_gate_map_allpi` public), the RL
  control loop (`_fly_armed`, build_obs call @ line 1083), `telemetry_health` F-C gate, finally-disarm.
- `src/racer/navigator.py` — KF chain + the 3 P0 case-C bug sites (`_initialize` 255-271,
  `time_since_vision_update_s` 426, velocity unobservable).
- `src/racer/state_estimator.py` (`LinearKF`), `rl/contact_true_eval.py` (selection instrument,
  already gate_yaw-aware), `scripts/frame_residual_report.py` (post-session frame audit),
  `rl/peregrine_racing.py` (`get_observations`, `obs_dim=17`, `world_to_gateframe`, `rel_tables`),
  `tests/test_train_deploy_obs_elementwise.py` + `tests/test_confirmed_p4_c05.py` (the obs-parity
  test pattern the gauntlet extends).

---

## 0. THE CORE ARCHITECTURAL FACT (drives every wiring decision below)

**Today the standing-start judged path NEVER runs the Navigator/KF.** Read `rl/fly_rl.py`
`_fly_armed`: with `--no-bridge` (the submission default + `submit_rl` pin), PATH B (the
`Navigator + Mission` bridge) is skipped, and the RL loop calls
`build_obs(s, gate_index, last_normed, virtual_flip=...)` directly on the **raw `DroneState` `s`**
(MEASURED — read `fly_rl.py:1083`; `build_obs` takes `state.position_ned / velocity_ned /
orientation_ned_wxyz / angular_rate_body` straight off the wire). The KF (`src/racer/navigator.py`)
appears ONLY inside `build_bridge` (PATH B). So the judged pose today is the sim's GIVEN pose —
which is exactly why VQ1 works (pos/vel pristine) and exactly why case C breaks (no given pose).

The gate-relative rebuild therefore **inserts an estimator stage that does not currently exist in
the judged path.** That is the integration's central change: a per-tick estimator (`RewindKF` +
gate-relative fix) must sit between telemetry and `build_obs`, and `build_obs` must consume the
estimator's gate-relative offset — NOT the raw given pose, and NOT `R_w2g @ (gate_map − p_KF_abs)`
(the proven anti-pattern; re-derived this run: `submap` arm 0.228 m vs `rel` arm 0.139 m, §2).

---

## 1. WIRING — the gate-relative chain into the live stack

### 1.1 The seam map (where each component attaches)

```
                          ┌─────────────────────── JUDGED PATH (submit_rl.py) ───────────────────────┐
 wire telemetry (DroneState s)                                                                        │
   pos/vel given*  ──►  [C2 ESTIMATOR]  ──►  gate-relative offset e_g + confidence c  ──►  [C1 OBS]  ──►  [C5 inc8 actor]  ──►  policy_step  ──►  CTBR wire
   accel_body            RewindKF + gate-rel    (pos_g from SEEN gate, not map)         build_obs(...,        (18-dim)         (unchanged tanh→rescale,    SET_ATTITUDE_TARGET
   odo quat (TRUE)       fix (PnP −L lever)                                              gate_map, est, conf)                   FLU→FRD, virtual_flip)     (BODY_RATE)
   camera Frame  ──────────────────────────────►  detector → PnP → associate ──────────────┘
   * given pose present in case A/B; ABSENT (origin seed, pos_std 5.0) in true case C
```

- **C2 estimator** = `RewindKF` (handoff/ultracode-vision-case-c/`kf_rewind_buffer.py`, bit-exact
  OOSM wrapper around `LinearKF`) driven by the Navigator chain, with a **gate-relative position
  fix**: observe the offset to the SEEN gate-4 opening (`−L`, `L = R_world_cam @ t_cam_gate`), which
  removes the per-track map bias EXACTLY (re-derived §2). Augments the absolute KF; does not replace it.
- **C1 obs** = `build_obs` extended to (a) consume `pos_g`/`vel_g` from the estimator's
  gate-relative offset, and (b) append the **uncertainty channel** (calibrated KF covariance →
  1 scalar), making the obs **18-dim**.
- **C5 inc8 actor** = retrained on the 18-dim gate-relative + uncertainty obs under measured-error DR.
- **C3 margin sim / C4 line** feed selection (`contact_true_eval`), not the live wire.

### 1.2 Estimator → obs handoff (the unification crux)

The 17-dim obs is ALREADY gate-relative: `obs[0:3] = pos_g = R_w2g @ (gate_pos − pos)` (MEASURED —
`peregrine_racing.get_observations:569` and `fly_rl.obs_from_zup:348`). The unification is:

- **Today (VQ1):** `pos = p_given`, `gate_pos = MAP centre`, `R_w2g` from `gate_map` yaw. `pos_g` is
  map-relative.
- **Gate-relative (case C):** the estimator delivers `e_g` = the drone's offset from the **seen
  opening** directly (gate-frame). `build_obs` must set `pos_g = e_g` (equivalently `pos_g = −R_w2g @
  L` rotated into the gate frame), NOT recompute `R_w2g @ (gate_map − p_KF_abs)`. Doing the latter
  re-injects `db` (map error) — the anti-pattern, re-derived `submap` 0.228 m fail vs `rel` 0.139 m
  pass this run (§2).

**Concrete `build_obs` change (C1's deliverable; C6 specifies the contract):** add an `estimator`
arg carrying `(e_g_pos, vel_g, confidence)`; when present, `obs[0:3] = e_g_pos`, `obs[3:6] = vel_g`
(estimator's gate-frame velocity), and append `obs[17] = confidence_channel`. When absent (VQ1
fallback / unit tests), the existing `gate_map=None` path is unchanged and bit-exact (the
`tests/test_confirmed_p4_c05.py` LENS2a invariant still holds).

**Velocity (P0 bug #3, ASSUMED-then-MEASURED):** case-C velocity is observable ONLY through
position-fix differencing (vision is position-only; `LinearKF` `update_velocity` is fed given vel
today — `navigator.py:320`). With given vel off, `vel_g` comes from the KF velocity state driven by
IMU + the gate-relative position fixes. The straddle (0.11 m warm / 0.17–0.21 m cold) is
velocity-prior-sensitive (CONTEXT) — so C1 must feed the **uncertainty channel** that lets the
policy slow when the velocity prior is cold. This is the load-bearing coupling.

### 1.3 Removing/replacing the 3 P4-C05 loud guards SAFELY

The guards (MEASURED — `fly_rl.py:219-285`, `:925`) currently **fail-loud-abort on any non-π / VQ2
course**. They are intentional and correct for a VQ1-only stack. The rebuild **replaces** them with
gate_map threading + an estimator-health gate. Do NOT just delete them — convert each into a positive
assertion that the *new* path is wired:

| Guard (current) | Behaviour today | Replacement under gate-relative |
|---|---|---|
| `_assert_vq1_constants_consistent()` (import-time, `:238`/called `:287`) | asserts hardcoded `_R_W2G`/`_GATE_REL_POS` == per-gate `rel_tables` on the all-π course | **KEEP** as-is. It guards a half-migrated *VQ1* edit and is course-independent (only exercises the hardcoded constants). Costs nothing; still true. |
| `_assert_live_course_is_vq1(track_gates)` (deploy-time, `:257`, called `:925`) | LOUD-ABORTS if live course ≠ hardcoded VQ1 (≠6 gates or >2 m pos deviation) | **REPLACE** with `gate_map = make_gate_map(live_gate_pos_zup, live_gate_yaw)` built from `TRACK_INFO`, threaded into `build_obs(..., gate_map=gate_map)`. The new guard asserts (a) a `gate_map` was successfully built from `TRACK_INFO` (not None on a non-VQ1 course), and (b) **estimator readiness** (RewindKF initialized, TIMESYNC reconciled — P0 #2). On any non-π course with NO gate_map → still abort (never silently corrupt). |
| `assert_gate_map_allpi(gate_yaw)` (public, `:219`) | helper the above two call | **KEEP** as a *negative* guard on the `gate_map=None` default path only. The yaw-aware path (`gate_map` passed) must NOT call it. |

**Net:** the loud-abort moves from "is this VQ1?" to "is the gate-relative pipeline actually wired
(gate_map built + estimator ready + timesync ok)?". A non-π course with the pipeline wired runs; a
non-π course with anything unwired still aborts loudly. This preserves the project's "fail loud, never
silently corrupt the obs by metres" doctrine (the whole point of P4-C05).

### 1.4 Keeping the judged path clean (no `MAV_CMD 31000` on the wire)

- `submit_rl.py` is UNCHANGED structurally: it still pins `--no-auto-reset` (R1 DQ guard),
  `--no-bridge`, `--no-debug-obs`, `--flights 1`, and `--checkpoint <inc8>` (the pin updates from
  inc7 → the selected inc8 ckpt). Safety pins still appended LAST (MEASURED — `submit_rl.py:40-51`),
  so a forwarded flag can never re-enable resets.
- The estimator runs **client-side only** (reads telemetry + video, writes the obs). It emits NO
  MAVLink command — the only wire traffic remains `SET_ATTITUDE_TARGET (BODY_RATE)` from
  `policy_step` + arm/disarm. Verify by the same independent relay decode used for the F-A merge
  (CONTROL-SIM §4 live-verify: "ZERO MAV_CMD 31000 on wire, 118 s independent decode").
- `--debug-obs` stays OFF on the judged path (it is the only per-tick blocking file I/O). The
  estimator must not add per-tick file I/O either — its diagnostics buffer in memory and flush at
  session end (mirror the `debug_obs.jsonl` flush-every-30 pattern, `fly_rl.py:1115`).
- The checkpoint sidecar (`stage1_inc8_actor.json`) MUST ship with the `.pth` so `_apply_checkpoint_
  sidecar` (`fly_rl.py:448`) sets the inc8 action bounds (`act_max_thrust`) — a missing sidecar
  silently overdrives thrust up to 33% (MEASURED warning path `fly_rl.py:459`).

---

## 2. OFFLINE VALIDATION GATES — the ordered gauntlet (cheapest fails first)

Nothing live runs until every gate below is green. Ordered so a cheap, deterministic check fails
before an expensive sim. All run in the laptop venv (`.venv\Scripts\python.exe`) — the KF /
localization / RewindKF / gate-relative stack imports **torch-free** (MEASURED this session:
`LinearKF`, `RewindKF`, `gate_pose_to_world_position`, `FIX_COV_FLOOR_STD=0.4`,
`ATTITUDE_NOISE_STD_RAD=0.0244 rad ≈ 1.4°` all import with numpy+scipy only).

| # | Gate | Instrument | EXACT pass criterion |
|---|---|---|---|
| **G0** | Static import + sidecar | `python -c` import of estimator chain; `_apply_checkpoint_sidecar` | estimator chain imports torch-free; inc8 `.json` sidecar present beside `.pth` and parses `act_max_thrust`/`act_max_rate`. |
| **G1** | **Obs bit-exactness vs `get_observations`** | extend `tests/test_train_deploy_obs_elementwise.py` + `tests/test_confirmed_p4_c05.py` | On the all-π VQ1 course with `estimator=None`, the 18-dim deploy obs `[0:17]` == train `get_observations` to **≤2e-4** (recording round-off; the existing bound). On a **non-π** course, yaw-aware `build_obs` matches `world_to_gateframe`/`rel_tables` ground truth to **float32 (≤1e-5 m)**. NEW: the gate-relative path with a synthetic estimator delivering `e_g` reproduces `R_w2g @ (gate_pos − pos)` when fed the matching pose (the unification identity). Negative control (sign-flip) MUST break. obs[17] (confidence) appended deterministically; obs[0:17] byte-identical to pre-channel build. |
| **G2** | **Estimator chain unit checks vs measured noise model** | extend `kf_rewind_buffer` validators; new gate-relative-fix unit test | (a) `update_position_at(t_fix≥now)` BIT-IDENTICAL to naive `update_position` (zero-latency degeneracy — already an invariant). (b) OOSM fix NEES over ≥600 MC ≈ **chi²(3) mean ≈ 3** (re-run: `c1` part_b NEES = **1.96** MEASURED — note: that is the in-plane (2-DOF-dominated) NEES at the gate plane; the 3-DOF buffer-level NEES is the ≈3 target). (c) RewindKF horizon **> L**: with L=115 ms (CPU) the horizon (default 0.5 s) must exceed it or the filter inverts/diverges to ~21 m (CONTEXT footgun) — assert `horizon_s > L_max`. (d) gate-relative fix uses the measured per-axis lateral σ = **0.265 m** (re-derived this run) and NO 0.40 m bias floor (the floor is bias-absorption; absent in `rel`). |
| **G3** | **Gate-4 margin sim (C3) pass** | C3's at-speed margin sim (builds on `c1_gate_relative.py`) | gate-relative `rel` arm in-plane **RMS < 0.155 m** at ~37 m/s. **Re-derived this run: rel RMS = 0.139 m (PASS), p90 = 0.203 m (OVER margin).** Pass criterion for the *gate*: RMS < 0.155 m AND E_bias ≈ 0 (map bias dropped). **p90 > margin is the residual risk → escape-hatch to live G3-live recording (§3).** `abs` 0.279 / `submap` 0.228 m both FAIL — these are the negative controls. |
| **G4** | **`contact_true_eval` selection (C5 policy)** | `rl/contact_true_eval.py --ckpt <inc8> --plant mixer` | gate-4 `margin_min ≥ 0` at `--body-radius 0.38` (the contact-true 0.155 m guard is `PASS_BAND_NOM − linf`; gate-4 must not flip to collision). Run yaw-aware (pass per-gate `gate_yaw`) so a non-π course is scored correctly (MEASURED — instrument already supports `gate_yaw`, `_score_gate:121`). Report per-gate `margin_p10`/`margin_min` + the D-offset probe on gates 3/4/5 (gate-4 verdict must NOT flip under ±1.5 m). |
| **G5** | **S_stable + BSR3 spin-margin gate** | `contact_true_eval` `compute_s_stable`; BSR3 spin check | `S_stable ≥ 2/3` across the ≥5 seeds (inc7 was 1/3-viable → narrow-basin warning fires below 2/3, `contact_true_eval:580`). **BSR3 spin-margin gate MANDATORY before retrain ships** (inc8 input #3): widened `spin_rate_abort → ~9–10`, `spin_time_abort → 3.0 s` must not trip on any of the ≥5 selected seeds' nominal rollouts (no false SPIN_ABORT) AND must still catch an injected sustained spin (negative control). |
| **G6** | **Speed-ladder selection** | `contact_true_eval` portfolio | pick the FASTEST contact-valid policy the estimator can SUPPORT: among ≥5-seed-stable inc8 ckpts passing G3/G4, choose max cruise speed whose gate-4 `margin_min` stays ≥0 at the estimator's delivered σ (warm-prior 0.11 m, not the cold 0.17–0.21 m unless the uncertainty channel demonstrably recovers it). |
| **G7** | **Full suite green** | `pytest` from repo root | **692 → ≥692 + new tests, 0 failures** (MEASURED baseline this session: `692 tests collected`). The guard-replacement + obs-channel + estimator tests ADD tests; none of the 692 may regress. Run from repo root (CONTEXT ops lesson: pytest-from-root). |

**Gate ordering rationale:** G0–G2 are sub-second deterministic unit checks (fail before any MC).
G3 is a ~seconds MC sim. G4–G6 are the full closed-loop offline rollouts (~minutes, ≥5 seeds). G7 is
the regression backstop. A failure at G1 (obs drift) makes every downstream rollout meaningless, so
it gates first — this is the project's hardest-won lesson (4 frame/convention bugs; the
`test_train_deploy_obs_elementwise.py` preamble enumerates them).

---

## 3. LIVE SHADOWPC VALIDATION (only after the offline gauntlet is fully green)

Ordered cheapest-first; zero-gate-contact is THE validity rule throughout (🚩 gate contact =
INVALID run). Run via `rl/fly_rl.py` with `--dev-auto-reset` (dev rig needs it; the JUDGED path
uses `submit_rl.py` with NO reset). `--debug-obs` ON for forensics on these dev runs.

1. **L0 — single dev smoke (case A, given pose).** One `fly_rl --dev-auto-reset --debug-obs` standing
   run with the inc8 ckpt on the VQ1 course, estimator in **passthrough** (given pose). Confirms the
   18-dim obs + new guard wiring doesn't regress the LIVE-CONFIRMED VQ1 5/5 baseline. Pass:
   clean 6/6 finish, zero gate contact, exit 0. **Run `scripts/frame_residual_report.py
   data/runs/<session>` immediately after** — East force-residual correlation must stay ~+0.97..+0.99
   (true conjugation intact); the mirror-canary must show as-is WORSE at bank. (MEASURED tool
   contract — `frame_residual_report.py` reads `q_raw_wxyz/w_raw/collective/rate_frd/vel_ned/pos_ned`
   from `debug_obs.jsonl`; it is robust to obs-dim growth because it never reads the obs vector
   itself, only wire columns — verified by reading its `load()`.)

2. **L1 — case-C cold-start activation.** Flip the estimator to true case C (P0 #1 fix:
   `position_ned=None` → origin seed, `pos_std=5.0`; `navigator.py:267`). Same VQ1 course. This is
   the FIRST run that exercises the gate-relative fix end-to-end on hardware. Pass: 6/6, zero contact,
   AND the recorded per-fix gate-relative offset σ ≤ the modeled 0.265 m/axis near-band. Frame-residual
   report after.

3. **L2 — at-speed fresh-reset batch (~37 m/s).** N≥10 fresh full-reset runs (ESC→HOME→Enter×2 between,
   `full_sim_reset()` — every flight an identical fresh countdown) at the inc8 cruise speed. Pass:
   **zero gate contact across the batch** (any contact = that run invalid; the batch needs a clean
   zero-contact majority with gate-4 specifically clean). Watch the SPAWN-ARTEFACT footgun (gate-3
   hard-collision reset → next respawn 0-tick crash is NOT a policy failure; exclude from stats).

4. **L3 — at-speed gate-4 vision recording (resolves the C3 escape-hatch).** This is the run that
   buys down the **p90 = 0.203 m > 0.155 m margin** residual risk (G3 MEASURED). Capture, at the real
   ~37 m/s post-gate-3 approach: (a) the **per-fix gate-relative lateral error** (PnP `t_cam` lateral
   decomposition) at the gate-4 transit band 0–12 m — confirm the modeled 0.265 m/axis σ holds at
   real speed/motion-blur; (b) the **corner-count vs range** transition — re-derived this run:
   gate-4 holds **4 corners in frame down to 4 m, drops to 2 corners (P3P only) at 3 m** (MEASURED,
   `c1` part_d) — confirm the live detector matches (if it clips to <4 corners earlier than 4 m, the
   4-corner accepted-fix rate at the binding band drops and the warm-prior σ degrades); (c) the **fix
   acceptance rate** (maha≤16.27, 4-corner) at the band — modeled ~47% → 14 Hz effective. Pass: live
   gate-4 lateral σ ≤ 0.265 m/axis AND ≥4-corner coverage to ≤4 m AND zero gate-4 contact. **If p90
   margin is breached live → fall back to a slower speed-ladder rung (G6) until gate-4 zero-contact.**

5. **L4 — RewindKF latency confirmation.** Confirm the in-loop vision latency on the eval host
   (CPU ~115 ms / GPU ~15–25 ms, CONTEXT) and that the RewindKF horizon exceeds it live (no fix-drop
   cascade). HARD-blocked on TIMESYNC (P0 #2) — if `frame.sim_time_ns` (server epoch) vs
   `DroneState.sim_time_ns` (IMU boot epoch) are still unreconciled, the rewind indexes a garbage
   epoch and this run is INVALID by construction; reconcile before L4.

**After EVERY live session:** `scripts/frame_residual_report.py <session>` (🚩 mandatory; internal
consistency cannot catch a conjugation). Zero-gate-contact gates each step — a contacted run is
discarded, not "passed with an asterisk".

---

## 4. DEPENDENCY GRAPH + BUILD ORDER (all 6 components)

### 4.1 Dependency edges (X → Y: Y needs X)

```
   P0-bug-fixes ───────────────┐         (P0#1 case-C seed, P0#2 TIMESYNC, P0#3 velocity-from-fixes)
        │                       │
        ▼                       ▼
  [C2 ESTIMATOR] ──► [C1 OBS] ──► [C5 inc8 POLICY] ──► [C6 INTEGRATION/VALIDATION]
        │  (RewindKF +    │ (gate-rel +    │ (retrain on        │  (wiring + gauntlet + live)
        │   gate-rel fix) │  uncertainty)  │  C1 obs, DR)       │
        │                 │                ▲                    │
        ▼                 │                │                    │
  [C3 MARGIN SIM] ────────┘         [C4 corrected LINE] ────────┘
   (gate-4 0.155 m       (validates C1   (arc-length reward over
    at est. σ)            obs delivers    contact-safe rebuilt line;
                          the margin)     inc8 reward input)
```

- **P0 bugs gate everything case-C.** P0#1 (case-C seed, `navigator.py:255-271`) is THE gate to
  validating anything (without it every "case C" run is secretly case A). P0#2 (TIMESYNC) is a HARD
  prereq for RewindKF (C2). P0#3 (velocity unobservable) shapes C1's uncertainty channel + C2's
  fix-differencing velocity. → fix P0 FIRST, in C2's branch.
- **C2 → C1:** the obs can only consume `e_g` once the estimator delivers it.
- **C1 → C5:** the policy can't be retrained until the obs contract (18-dim, channel layout) is frozen.
- **C3 → C5 selection / C6 G3:** the margin sim sets the speed ceiling the policy is selected against.
- **C4 → C5:** the arc-length progress reward grafts over the rebuilt corrected-aero contact-safe
  line (`reference_line_vq1.json` is drag-infeasible + 170° inverted → must be rebuilt; inc8 input #1).
- **C6 depends on ALL** (it wires them and runs the gauntlet) but its *spec* (this doc) is written in
  parallel; its *execution* is last.

### 4.2 Clean build order (what to build, in order)

1. **P0 bug fixes** (in the C2 estimator branch): case-C seed guard (`use_given_position` honored),
   TIMESYNC reconciliation (`frame.sim_time_ns` ↔ `DroneState.sim_time_ns`), velocity-from-fixes
   path. Unit-tested torch-free. *Blocks all case-C work.*
2. **C2 estimator chain**: RewindKF + gate-relative position fix (observe `−L` to the seen opening;
   no `subtract gate_map` anti-pattern). Validated by gauntlet **G2**. Builds on
   `kf_rewind_buffer.py` (read-only) + `c1_gate_relative.py` part_b arithmetic.
3. **C3 margin sim**: at-speed gate-4 0.155 m sim at the estimator's delivered σ (extends
   `c1_gate_relative.py`). Validated by **G3**. Produces the speed ceiling for C5 selection. (C3 can
   proceed in parallel with C2 once the noise model is fixed — it only needs the measured σ, 0.265 m.)
4. **C4 corrected line**: rebuild the contact-safe corrected-aero reference line (loader
   `reference_line.py` valid; JSON must be rebuilt). Feeds C5's arc-length reward. (Parallel with C2/C3.)
5. **C1 obs**: extend `build_obs`/`obs_from_zup` to consume `e_g` + append the calibrated-confidence
   channel → **18-dim**; bump `peregrine_racing.obs_dim 17→18` + actor input layer 17→18. Freeze the
   channel layout. Validated by **G1**. *Needs C2 (for `e_g`) frozen.*
6. **C5 inc8 retrain**: staged_monolithic_then_decomposed; arc-length reward over C4's line; envelope
   relaxation (rw_tilt 96→48, free-cone 60→~70°); BSR3 spin-margin gate (G5) MANDATORY before retrain;
   ≥5 seeds; measured-error DR (the C1/CONTEXT error classes). Ship `.pth` + `.json` sidecar.
   Validated by **G4/G5/G6**. *Needs C1 obs frozen + C3 ceiling + C4 line.*
7. **C6 integration**: thread the live `gate_map` + estimator into `_fly_armed`; replace the 3 P4-C05
   guards (§1.3); pin the inc8 ckpt in `submit_rl.py`. Run the full **offline gauntlet G0–G7** →
   then the **live L0–L4** ladder. *Last; depends on all.*

**Critical path:** P0 → C2 → C1 → C5 → C6-gauntlet → C6-live. C3 and C4 are off the critical path
(parallel feeders into C5/selection), so they should be built concurrently with C2 to avoid stalling
C5. The single longest pole is **C5 retrain** (≥5 seeds, narrow basin) — everything upstream exists
to de-risk that one expensive training run before it starts.

---

## 5. RESIDUAL RISKS CARRIED INTO INTEGRATION

- **p90 gate-4 margin OVER (0.203 m > 0.155 m), MEASURED this run.** RMS passes (0.139 m) but the
  tail does not. Mitigation: the uncertainty channel (slow-when-uncertain) + the speed-ladder
  fallback (G6) + the live L3 recording that resolves whether the modeled σ holds at real
  speed/blur. **This is the make-or-break and is correctly velocity-prior-sensitive** (warm 0.11 m
  clears comfortably; cold 0.17–0.21 m straddles). HIGH.
- **TIMESYNC (P0#2) HARD-blocks RewindKF.** If unreconciled, the rewind buffer indexes a garbage
  epoch and the whole C2 chain is invalid — not degraded, invalid. Must be fixed in step 1, verified
  live at L4. HIGH.
- **Cold-prior straddle.** Case-C velocity is observable only via position-fix differencing; a cold
  lap-entry prior pushes in-plane to 0.17–0.21 m (over margin). The uncertainty channel is the
  designed mitigation but is unproven until C5 trains on it. MEDIUM.
- **Obs-dim growth breaking a silent consumer.** `frame_residual_report.py` is safe (reads wire
  columns, not the obs vector — verified). But any tooling that hardcodes obs length 17 (replay_obs,
  debug dumps) must be audited when obs goes 18-dim. LOW (caught by G1 + G7).
- **Detector 4→2 corner clip at 3 m.** If the live detector clips below 4 corners earlier than the
  modeled 4 m, the 4-corner accepted-fix rate at the binding gate-4 band drops and the warm σ
  degrades. Resolved by L3. MEDIUM.
- **Narrow inc8 basin (S_stable).** inc7 was 1/3-viable; ≥5 seeds + the 2/3 S_stable gate are the
  guard. A retrain that lands <2/3 must not ship. MEDIUM.
```
