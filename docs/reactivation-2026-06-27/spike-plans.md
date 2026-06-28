# Spike Plans — executable experiments to settle the open forks (2026-06-27)

*Designed by the "brains" agents, grounded in the live codebase. Execute when the no-build window lifts. Each: question · procedure · decision rule · our prediction · key confound. Full designs in the session transcript.*

---

## SPIKE A — Gate-4 discriminator: realizability vs calibration vs reach
**Gates:** Tier-4 SITT fork (#13) — should we invest in Student-Informed-Teacher co-training, or is gate-4 a boresight-calibration + reach problem?

**Sharper fact found (on unmerged branch `inc8-deterministic-retrain`):** an AIRTIGHT NEGATIVE — **fix-seating and deterministic flight are mutually exclusive** (`fix_rate=0` at every gain that flies on the mean; the gains that seat fixes crash the mean). This is the real wall, and it's diagnostically ambiguous between R/C/P.

**Procedure (all offline in the estimator emulator, no renderer):**
- **Arm 1 — cheap 2×2 counterfactual** (runs in hours, may close it alone): add 3 eval-only knobs to `EmulConfig` — `force_accept` (hand fixes regardless of pointing = tests **Realizability**), `bias_off` (zero ε_vert = tests **Calibration**), `fix_density_scale` (dose-response). Run the σ_p0 ensemble (`inc8_sigmap0_eval.py`, n=300) on the best det-flyable ckpt across the factorial. **R and C make OPPOSITE predictions:** if handing free fixes fixes it → R; if it changes nothing but removing bias does → C.
- **Arm 2 — SITT proxy-KL oracle** (only if Arm 1 implicates R): train a privileged teacher on the 36-dim `get_state` truth (⚠ needs `algo=appo`, see confound), BC-distill a proxy on the 20-dim deploy obs, read `KL(teacher‖proxy)` binned by range-to-gate-4. KL spike on the pitch-pointing segment = realizability localized. Then test a camera-aware teacher: if it kills the KL spike AND still passes gate-4 → SITT validated; if it can no longer pass → SITT ruled out (geometric FoV-vs-line conflict).

**Decision rule:** largest-leverage intervention wins. `+fix-density` ≥40% p99 drop → R→run Arm 2. `−bias` ≥40% drop (and fix-density doesn't) → **C: bake ε_vert offline, skip SITT.** Neither moves but reach≪1 with conditional-miss≤0.45 → **P: RL reward levers.**

**Our prediction (~60%):** primarily **Calibration + Reach**, with a real realizability component that SITT *can't* fix (camera-aware teacher likely can't pass off-line gate-4 under the 20° mount + 58.7° VFoV). Net: bake ε_vert + RL reach levers, **don't invest in SITT.** Highest value of the spike = killing the SITT fork cheaply.

**Key confound:** the symmetric-critic footgun corrupts the teacher — the privileged teacher MUST use `algo=appo`+`state_dim=36` (verify `critic.input_dim==36`), else the proxy-KL compares two equally-blind nets. Also: the empirical-opposite-sign look-at footgun (verify pointing by measured `band_el_abs`, never by gain sign); single-seed branch data (run ≥2 ckpts).

---

## SPIKE B — Hybrid RL+MPPI at racing speed under estimator noise
**Gates:** control fork D3b (RL+MPPI) / D3a (AC-MPC) vs RL-only spine. The referee for the regime no paper tested.

**Procedure (DiffAero on Adroit; reuses the existing noise model + GT metric):**
- Noise enters at ONE existing seam (`peregrine_racing_inc8.step` → `BatchedEstimatorEmulator`; σ_lat∈U[0.05,0.15], one-signed bias∈U[0,0.19] = "σ_p0≈0.15"). MPPI plans on the **KF estimate, never truth** — that's what makes the test honest.
- Baseline = PPO/CTBR **with the asymmetric-critic fix first** (prereq, not optional — comparing to a broken baseline is a confound).
- MPPI = thin residual layer at the existing look-at-residual seam: seed = PPO mean action, model = DiffAero torch backend (zero in-sim mismatch = MPPI's best case), horizon sweep H∈{8,15,30}, K∈{256,896}, hard gate-contact cost (the one thing RL reward can't guarantee). A 1-sample/zero-noise MPPI ≡ RL-only (degenerate-equivalence sanity cell).
- 6 core cells (2 controllers × 15/25/30 m/s), **paired** noise seeds, n=300/cell, 3 training seeds (9 shared PPO trainings — MPPI reuses each policy).
- **Independent VRAM/latency kill switch on an 8 GB box** (not Adroit): MPPI peak VRAM + solve time *concurrent with the 26–44 ms detector* — the residual control budget is only ~5–15 ms, not 33 ms.

**Decision rule:** hybrid lives ONLY if at 25 AND 30 m/s it beats RL-only on contact-rate by ≥5pp (paired-significant) AND is not worse on reach AND not worse on σ_p0 AND the winning (K,H) fits 8 GB alongside the detector. Else **parked.**

**Our prediction (~70%):** RL-only matches/beats hybrid at racing speed → **parked**; ~30% upside is a clean contact-rate win at 25 m/s. Reasoning: MPPI rolls a one-signed-biased state forward 0.3–0.5 s → optimizes in an offset frame → likely *amplifies* the error; no paper shows MPPI >12 m/s or under non-mocap state.

**Key confound:** within-sim = zero model-mismatch = MPPI's BEST case (a within-sim win is necessary-not-sufficient; the sim-to-sim gap is AC-MPC's home turf, a separate spike). VRAM masking if probed on a big GPU. λ/Σ tuned once at 15 m/s then frozen.

---

## SPIKE C — VGGT recon-map moat validation
**Gates:** Tier-3 #9 (recon-map = the single biggest strategic bet: no-map → known-map).

**Procedure (render → VGGT offline → Sim(3)-anchor → reloc):**
- Render a conservative recon-lap keyframe set (N≈120–200, each gate from ≥3 viewpoints) from Blender (`blender_gen/`, frozen contract 640×360 fx=fy=320) with GT poses; + two held-out hot-lap sets (on-map + "desert"/gate-sparse).
- Run VGGT (chunked via VGGT-Long) **offline on A100/H100** (8 GB can't); Sim(3)-anchor the cloud to gate-PnP fixes (`gate_pose.py` IPPE_SQUARE) via Umeyama+RANSAC; freeze the metric map; derive gate ordering from the trajectory.
- Measure: (a) map metric accuracy + inter-gate scale drift vs sim GT, (b) gate-pose/plane-normal recovery, (c) **relocalization error on held-out hot-lap frames split near-gate vs desert**, (d) explicit thin-gate-rim depth-error check.

**Decision rule:** GREEN (gate-center p90 ≤0.15 m, scale ≤1.5%, desert reloc p90 ≤0.30 m) → build the moat. YELLOW (rims smear but corridor+scale hold) → **hybrid: VGGT for inter-gate structure + ordering, PnP authoritative at gates** (still a win). RED (geometry/scale fail sim-gap, or reloc worse than coasting) → fall back to gate-relative reactive + DPVO sparse map.

**Our prediction (~60%):** **YELLOW** — corridor + gate-ordering win, thin rims smear → keep PnP authoritative. Still strategically large (removes gate-ordering + desert blindness, which no champion solves map-free).

**Key confounds:** spike only tests the **Blender** domain — the real VQ2 3D-scanned domain is untested until VQ2 exposes imagery (necessary-not-sufficient). Moat *applicability* is gated on VQ2-load check C3 (track-fixed-within-load). GT poses used to isolate map-question from estimator-question → needs a follow-up with estimated poses before deploy. **Cheaper first probe to consider: run the DPVO sparse-map path first** (Fast config 2.5 GB fits 8 GB) as the viability gate before the A100 VGGT dependency.

---

## Post-review refinements (our agents vs Gemini adversarial review, 2026-06-27)
**Doors-open principle (Fengyou): confident predictions do NOT close forks — cheap TESTS do. Gemini's "one thing to change" was a CHEAPER test in every case → adopt them as step 0, keep every fork alive until data (not prediction) closes it.**

- **Spike A cheapest-first = KINEMATIC FRUSTUM OVERLAY (Gemini, 15 min, no training):** fly the det-mean trajectory on the state oracle; project the known 3D gate-4 corners into the image using the actual pitch/roll over the **12–28 m fix-seating window**; check whether gate-4 is in-frame long enough for the ESKF to converge. Run BEFORE the emulator counterfactuals (which then separate residual C/P). Gemini predicts **REACHABILITY failure (90%)**: at ~45° racing pitch + fixed 20° mount the optical axis is ~25° below horizon → gate-4 exits the 58.7° VERTICAL frustum.
  - **DOOR OPENED (not closed):** mount is FIXED 20° (can't do Scaramuzza 40–45° uptilt) → if reachability, fix = (a) terminal blind-zone coast [we have it] + (b) **NEW: reshape gate-4 approach to use the 90° HORIZONTAL FoV (yaw-to-gate) not the narrow vertical (pitch-to-gate)**. SITT likely still the wrong tool, but confirm via the cheap overlay, don't assume.
- **Spike B SPLIT (Gemini + our agent): B1 = plant-mismatch (drag/battery-sag — MPPI's HOME TURF) FIRST, B2 = estimator-noise (worst case).** If MPPI fails even B1 on GPU latency-jitter (Gemini: >50 ms detector contention violates the Markov assumption, 95% parked), it's genuinely dead.
  - **DOOR OPEN: AC-MPC (D3a) ≠ MPPI (D3b)** — Gemini's GPU-rollout-contention critique targets parallel sampling; AC-MPC is ONE iLQR solve (light VRAM/latency) → stays a separate live fork. CBF-useless-single-step (true) → reframe to "short-horizon funnel/reachability check that fits budget?" — door cracked.
- **Spike C cheapest-first = DPVO-FIRST (Gemini + our agent):** DPVO (2.5 GB, tracks corners/edges) Sim(3)-anchored on gate corners — if even the sparse tracker can't hold gate geometry, no dense model will. Gemini: patch-regularization (gate tube < one ViT patch at 20 m) smears rims to background, YELLOW/RED 85%.
  - **DOOR OPEN: Gemini's RED overstates OUR use** — we never needed VGGT for the GATE (PnP does, from known size); we need the CORRIDOR + ordering + desert reloc. Smeared rims don't kill the corridor map → the YELLOW path (VGGT corridor + PnP-authoritative gates) survives.

## NEW IDEAS generated this pass (turn-inward synthesis)
1. **Track-wide perception-reachability map** — generalize the frustum overlay to EVERY gate: when is each gate in-frame along the racing line (given pitch/roll + fixed 20° mount + FoV)? Sizes the LIO/GRU coast; shapes reward (penalize blinding approaches). Converts the gate-4 one-off into a systematic design tool.
2. **Horizontal-FoV-aware approach shaping** — 90° HFoV ≫ 58.7° VFoV; prefer yaw-to-gate over pitch-to-gate where geometry allows → attacks reachability WITHOUT touching the fixed mount. A reward/planning lever.
3. **Promote blind-zone-coast + Cioffi LIO + GRU belief-state to LOAD-BEARING** (was Tier-3) — if gates routinely exit the vertical FoV under racing pitch, dead-reckoning through the blind window is core to passing any gate at speed, not optional.
