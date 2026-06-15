# AT-SPEED σ + TERMINAL-WINDOW RECORDING — REPORT (gate cold-margin, 30 m/s blur risk)

**Date:** 2026-06-15 · **Machine:** ShadowPC (sim host) · **Operator:** opus-4.8 (high), worker
**Branch:** `worker/at-speed-sigma-2026-06-15` (off `origin/main`) · **Sim:** AI-GP v1.0.3364 (`DCGame-Win64-Shipping`), `udp:127.0.0.1:14550`
**Vehicle:** CTBR head-on bridge `scripts/fly_vq1.py --faithful` (FAITHFUL_TUNED gains/planner) + inc7 `rl/checkpoints/stage1_inc7_actor.pth` (one canary lap)
**Detector:** `models/gate_yolo11s_curriculum_v2.pt` · **Map:** surveyed `track_map.json` (corner_to_center)

> **Headline.** The one offline-unsettleable margin risk was: does σ_lat blow up at 30 m/s (a LINEAR
> motion-blur MODEL extrapolated 0.19 → 0.32–0.39 m, past the 0.245 m ceiling → closure FAILS)? That
> model assumes the sim renders speed-dependent image blur. **It does not.** Phase 0 settles it: **the
> AI-GP sim is BLUR-FREE** — at 14.5 m/s gate/wall edges are razor-sharp with zero directional streak,
> identical to hover. With no blur, in-sim σ is determined by RANGE/geometry, not speed: across **22
> clean head-on CTBR laps (0 contact)**, range-controlled σ_lat is **DEAD FLAT at ~0.13–0.17 m over
> 6→15 m/s** (the CTBR clean-speed ceiling), bias_lat ≈ 0. Projected flat to 30 m/s, **σ_lat ≈ 0.15 m
> CLEARS the 0.245 m ceiling (×1.6 headroom); the linear-blur model is FALSIFIED in-sim.** The
> at-speed-σ risk therefore **cannot be settled in our blur-free sim** — it is a SIM-TO-REAL concern
> (real cameras blur ∝ exposure × image-velocity) → **escalate to the physical drone (Sept/Nov).**
> Pivoting to the blur-independent deliverable (3): the **terminal last-6 m fix-rate is ~5 %, NOT the
> required ≥0.50** — but the cause is ASSOCIATION failure at close range (gate fills the frame, corners
> clip out), **not blur and not speed**; fixes die at ~8 m. (Scope caveat: head-on CTBR posture,
> gate-agnostic — NOT the inc8 race posture the gate-4 requirement is ultimately about.) Both canaries
> PASS (R_y(π) ODO convention intact; video↔LPN 2.51 ms).

---

## PHASE 0 — SIM-BLUR DETERMINATION: **BLUR-FREE (NO)** ✅ decisive

**Static (render config):** the packaged shipping UE build exposes **no** external motion-blur / shutter
/ exposure config — settings are baked into `Content/Paks/*.pak` (only `pgos_res/pgos_config.ini`
exists; no `Saved/Config`). The §3.8 camera spec is **pinhole, no distortion, 640×360, fx=fy=320** —
the signature of a SceneCapture2D render target, which in UE does not apply temporal motion blur.

**Empirical (decisive):** a gate at range R looks identical whichever gate it is (every gate is the same
red-square prop), so motion blur — a temporal effect ∝ closing speed at fixed range — would smear the
gate/wall edges more at higher speed.
- **Direct visual:** a 14.2 m/s frame (gate-3 @ 24 m) vs a 0.5 m/s hover frame (gate-0 @ 23 m), matched
  range. The fast frame's block-wall and gate edges are **crisp with zero directional streak** — even
  the closest blocks (largest image motion) — visually identical to the hover frame.
  (`analysis/_frame_fast.png`, `_frame_hover.png`.) Full-frame Laplacian var is *higher* for the fast
  frame (545 vs 316) — content-driven, the opposite of blur.
- **Gate-box Laplacian-variance ratio (fast/slow) at matched range** (`blur_probe.py`, 12 194 in-FoV
  gate boxes pooled across all gates/laps): **0.68 / 1.06 / 1.20 / 1.55** across range bins — scattered
  around 1 with **no monotonic speed trend**, and FAST is *sharper* in 3 of 4 bins. If blur were
  rendered, FAST would never be sharper. ⇒ no speed-dependent image degradation.

**Implication:** in-sim σ(speed) is flat by construction; the linear-blur extrapolation cannot manifest
here. The σ-at-speed risk is a **SIM-TO-REAL** question our sim cannot answer → physical-drone item.

---

## PHASE 1 — RECORDING (proven CTBR head-on bridge) ✅ 22 clean laps, 0 contact

`handoff/at-speed-sigma-2026-06-15/sweep_runner.py` drives the proven simops chain per lap
(`drive_to_waiting` → launch `fly_vq1 --faithful` with a speed override → GO → wait → contact-flag).
**Speed override:** a 3-line surgical patch to `fly_vq1.py` adds `--faithful-cruise/--faithful-max-speed`
to dial the FAITHFUL planner cruise + controller velocity cap while keeping the validated faithful gains.

| batch | config | laps | result | gate contact | speed |
|---|---|---|---|---|---|
| smoke | faithful, max-gates 2 | 1 | FINISHED | 0 | ~7 m/s |
| **s1** | targets 14/20/26/32, max-gates 2 | **16** | **16× FINISHED** | **0** | ~7–11 m/s (accel-limited on 47 m leg) |
| **s2** | targets 20/30, **max-gates 6** (full course) | **6** | **6× FINISHED** | **0** | ramp 0→**14.5 m/s** |
| canary | inc7 RL, `--debug-obs --no-auto-reset` | 1 | FINISHED 6/6 | 0 | p50 17.1 / **max 21.4 m/s** |

- **Head-on geometry SOLVES the camera-pointing limit.** Course-yaw keeps the nose on the gate axis →
  crab p50 **~5°** (vs inc7's 66°) → gate in-FoV **75–97 %** → ~300 accepted fixes/lap (vs inc7's ~28).
- **Max clean CTBR speed = 14.5 m/s.** The faithful controller is **drag-limited** there — the velocity
  loop saturates the **45° tilt cap** and forward thrust = drag (target 20 vs 30 both top out at 14.2–
  14.5; not cruise/accel-cap limited). Reaching 30 m/s clean would require raising the tilt cap (contact
  risk) → **NOT chased** (escape hatch).
- **inc7 reaches 17–21 m/s clean**, but with heavy crab → its fixes are all **oblique (bearing > 25°)**;
  it yields **no head-on σ point above 14.5 m/s** (matched geometry unavailable at high speed).

---

## PHASE 2 — CANARIES ✅ both PASS

- **`verify_bundle.py`** (CTBR smoke lap): **ALL_PASS**. video 30.1 Hz, LPN 95.7 Hz, ODO 74.6 Hz;
  **video↔LPN recv-clock gap p50 = 2.51 ms** (hits the ~2.5 ms target), p99 5.6 ms; GT velocity LIVE
  (max 7.6 m/s here; live to 21.4 m/s on the inc7 lap); ODO body-twist cross-checks LPN velocity to <2 %.
- **`scripts/frame_residual_report.py`** (inc7 `--debug-obs` lap, 282 ticks): **mirror canary TRUE
  +0.97 / AS-IS −0.81 → OK** — the internal-consistency-invisible **R_y(π) ODOMETRY conjugation is
  intact** on this instance/build (identical to the 2026-06-14 L3 result), so the σ/bias measured below
  (which use `R_world_from_odo_quat_wxyz`) are real, not frame artifacts. Rate canary gain
  [0.92, 0.80, 0.93] ≈ 1.0; force residual N +3.58 / D +2.33 m/s² = the known unchanged plant-gap baseline.

---

## PHASE 3 — ANALYSIS (σ-vs-speed, terminal window, bias)

Offline shadow chain `analysis/shadow_allgates.py` (the production C2 chain verbatim — GateDetector →
associate_scored → estimate_gate_pose → gate_pose_to_world_position + gate_relative_inplane_fix →
2-DOF GT-anchored relinnov gate; one detect() per frame scoring all gates), `analyze_atspeed.py`.
Pool: **6 full-course CTBR laps (29 600 frame×gate rows) + inc7 canary lap**, gates 1–5.

### (A) σ vs SPEED — **FLAT** (range-controlled, accepted fixes, head-on bearing ≤ 25°, range 10–22 m)

| speed bin | N | **σ_lat (m)** | σ_vert (m) | bias_lat (m) | med range |
|---|---|---|---|---|---|
| 6–9 m/s | 118 | **0.169** | 0.116 | −0.007 | 21.0 |
| 9–11 m/s | 490 | **0.164** | 0.089 | +0.002 | 13.0 |
| 11–13 m/s | 196 | **0.170** | 0.126 | −0.025 | 16.3 |
| 13–15 m/s | 46 | **0.129** | 0.136 | +0.102 | 20.2 |

**σ_lat is flat (0.13–0.17 m) across 6→15 m/s with no monotonic trend; bias_lat ≈ 0.** (A naïve
speed-binned fit without range control gives a *negative* slope — an artifact of the speed↔range
coupling along the course, where slow bins sit at far range; controlling range removes it.)

- **Projection to 30 m/s:** blur-free ⇒ flat ⇒ **σ_lat(30) ≈ 0.15 ± 0.04 m → CLEARS the 0.245 m
  ceiling (×1.6 headroom).** The linear-blur model (σ_lat 0.32–0.39 at 30) is **FALSIFIED in-sim.**
- **Caveat (load-bearing):** "in-sim" only. Real cameras blur ∝ exposure × image-velocity; that risk is
  not testable here → physical drone.

### (B) TERMINAL-WINDOW fix density — **~5 %, NOT ≥0.50** (association-limited, speed-independent)

Close-range funnel (all gates pooled; `analyze_atspeed`, inline diagnosis):

| range | in-FoV % | detected % | **accepted / frames** | accept-rate (of frames) |
|---|---|---|---|---|
| 14–18 m | 50 % | 99 % | 289 / 640 | 45 % |
| 10–14 m | 55 % | 99 % | 318 / 656 | 48 % |
| **8–10 m** | 58 % | 99 % | **97 / 357** | **27 %** |
| **6–8 m** | 62 % | 99 % | **15 / 380** | **4 %** |
| **4–6 m** | 62 % | 99 % | **13 / 433** | **3 %** |
| 2–4 m | 56 % | 97 % | 0 / 536 | 0 % |

- **Per-frame accept-prob in the last 4–8 m (in-FoV) = 0.054.** At 30 m/s the last 6 m lasts 200 ms ≈ 6
  frames @ 30 Hz → **~0.3 accepted fixes; effective terminal fix-rate ≈ 0.05 ≪ 0.50.** Speed-independent.
- **Cause = ASSOCIATION, not blur / FoV-center-exit / the relinnov gate.** At 6–8 m the gate is still
  ~62 % in-FoV and **99 % detected**, but associations collapse (103 → 39 → 37 at 8–10 / 6–8 / 4–6 m):
  the gate fills the frame and its corners clip out, so shape-consistency association + ≥3-corner PnP
  fail. Min accepted-fix range = **4.9 m** (rare). This reproduces the prior verdict that the binding
  limit is association/coverage, not detector/PnP precision.
- **Scope caveat:** head-on CTBR posture, gate-agnostic. The gate-4 requirement is about the **inc8 race
  posture** (not yet crowned); its terminal corner-visibility may differ (oblique approach can keep
  corners in-frame longer, or worse). **This is evidence the ≥0.50 terminal lock is hard even in the
  favourable head-on posture — flag for the inc8 race-posture test.**

### (C) effective bias + one-signed PnP depth bias (vs speed)

- **Lateral chain bias ε_lat ≈ 0** at every speed (±0.02 m; the +0.10 at 13–15 m/s is a 46-sample
  fluctuation) — consistent with the L3 finding of a well-calibrated yaw/roll attitude.
- **Depth (PnP one-signed) bias is POSITIVE (fix placed BEYOND the gate) and RANGE-dependent, NOT
  speed-dependent:** +2.1 m @ 26 m → +0.6 m @ 24 m → +0.25 m @ 14 m (the classic monocular far-range
  depth inflation; σ_depth 0.5–1.1 m). It shrinks as the gate is approached and is owned by the absolute
  fix + IMU, not the in-plane centering term.

---

## VERDICT (raw numbers for synthesis)

1. **Sim is BLUR-FREE** → the σ-at-speed margin risk is **SIM-TO-REAL, unsettleable in our sim** →
   physical-drone item (Sept/Nov). In-sim, σ_lat is **flat ~0.15 m to 30 m/s → CLEARS 0.245**; the
   linear-blur model is falsified here.
2. **Max clean matched-geometry (head-on) speed = 14.5 m/s** (CTBR drag/tilt-cap). inc7 reaches 21.4 but
   oblique only. A pristine 30 m/s head-on race recording needs the inc8 envelope-relaxed policy (not
   crowned) — out of scope, as the brief anticipated.
3. **Terminal last-6 m fix-rate ≈ 5 % (≪ 0.50)** — association failure at close range, speed-independent.
   Binding for the gate-4 TERMINAL GATE-LOCK requirement, but measured in the head-on CTBR posture; the
   inc8 race-posture terminal test is the true arbiter.
4. Canaries clean (R_y(π) intact; video↔LPN 2.51 ms).

---

## Tooling / deliverables (`handoff/at-speed-sigma-2026-06-15/`)
- **`sweep_runner.py`** — speed-sweep lap runner (proven simops chain + `--faithful-cruise/-max-speed`).
- **`run_inc7_lap.py`** — one inc7 `--debug-obs` lap (canary + high-speed anchor).
- **`analysis/shadow_allgates.py`** — multi-gate shadow replay, one detect() per frame (extends `shadow_gate4.py`).
- **`analysis/analyze_atspeed.py`** — σ-vs-speed (range-controlled) + terminal-window + bias.
- **`analysis/blur_probe.py`** — gate-edge sharpness vs speed at matched range (blur Y/N).
- **`analysis/verify_bundle.py`**, `scripts/frame_residual_report.py` — canaries.
- **`analysis/atspeed_summary.json`**, `_frame_fast.png`, `_frame_hover.png` (blur evidence), `logs/`.
- **Code change:** `scripts/fly_vq1.py` +3 lines (`--faithful-cruise/--faithful-max-speed` overrides).
- **Raw recordings (LOCAL, gitignored, ~20 MB/lap):** `data/runs/*_atspd_s1_*` (16), `*_atspd_s2_*` (6),
  `*_atspd_smoke_*` (1), `*_rl_canary_f1` (1). Per-frame rows JSONs (`s2_allgates_rows.json` 9.8 MB,
  `inc7_allgates_rows.json`) stay LOCAL in `analysis/` (gitignored); the distilled `atspeed_summary.json` is committed.

## Validity
- ZERO gate contact on all 22 CTBR laps + the inc7 lap (the validity rule); no run discarded.
- inc7 pinned to `stage1_inc7_actor.pth` (the live-confirmed best, now the fly_rl default); `--no-auto-reset` (no §7 sim-reset).
- fly-map ≡ analysis-map (surveyed track_map.json, corner_to_center); no estimator drove control (pure shadow/offline).
- WIRE CAVEAT: GT/LPN/ODOMETRY tooling is OFFLINE-CALIBRATION-ONLY (our sim streams position; the official VQ spec may not). The deployed stack must self-localize.

## MEMORY-DELTA (≤10 lines)
- **SIM IS BLUR-FREE (decisive):** at 14.5 m/s gate/wall edges razor-sharp, zero directional streak,
  = hover; gate-box Laplacian ratio fast/slow 0.68–1.55 (no speed trend, fast often sharper). UE
  SceneCapture2D (pinhole, no distortion) ⇒ no temporal blur. The σ-at-speed margin risk is **SIM-TO-
  REAL, UNSETTLEABLE in our sim → PHYSICAL-DRONE item (Sept/Nov).**
- **σ_lat is FLAT vs speed (range-controlled, head-on): 0.13–0.17 m over 6→15 m/s, bias_lat≈0.**
  Projected flat to 30 m/s ≈ **0.15 m → CLEARS the 0.245 m ceiling (×1.6); the LINEAR-BLUR MODEL
  (0.32–0.39) is FALSIFIED in-sim.** σ_vert 0.09–0.14; depth PnP bias POSITIVE, range-dependent
  (+2.1@26m→+0.25@14m), speed-independent.
- **Terminal last-6 m fix-rate ≈ 5 % (≪ 0.50 req)** — ASSOCIATION failure at close range (detect 99 %,
  in-FoV 62 %, assoc collapses as gate fills frame/corners clip), NOT blur/FoV/relinnov; min accepted
  range 4.9 m, speed-independent. **CAVEAT: head-on CTBR posture, gate-agnostic — inc8 race-posture
  terminal test is the true arbiter.** Binding for the gate-4 TERMINAL GATE-LOCK requirement.
- **Max clean head-on speed = 14.5 m/s** (faithful CTBR drag/45°-tilt-cap; targets 20=30 both top here).
  inc7 reaches 21.4 m/s clean but oblique-only (crab 66°, no head-on high-speed point). 30 m/s head-on
  race recording needs inc8 (uncrowned).
- **Canaries PASS:** R_y(π) ODO mirror TRUE +0.97/AS-IS −0.81 OK; video↔LPN p50 2.51 ms; GT-vel live to 21.4 m/s.
- **NEW PARKED (commander):** the ≥0.50 terminal-6 m fix-lock looks geometrically hard (association/
  close-range corner-clip) even in the favourable head-on posture — needs an inc8 race-posture terminal
  test to confirm/refute for gate-4. Head-on CTBR (crab ~5°) is a clean σ-vs-range rig for future calibration.
