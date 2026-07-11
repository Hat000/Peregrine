# VQ2 Egocentric Generation — DESIGN (SSOT)

**Working name:** inc9 / env alias `peregrine_vq2_ego` (name TBD by Fengyou)
**Status:** APPROVED for build 2026-07-06 (Fengyou). New generation, trained from scratch. Breaks the frozen obs-20 contract — coordinated with the vision-stack commander for deploy.
**Supersedes:** the gate-frame single-gate obs of inc8 (`d5_inc8_spec.md`) for VQ2.

---

## 1. What & why

Replace the single-gate, gate-frame, **world-position-dependent** observation with an **egocentric, multi-gate, position-free** representation fed by a **realistic keypoint-gated vision model**. This:
- kills the handoff frame-teleport (the dual-gate failure root cause),
- removes the drift-prone world position the policy cannot trust at deploy (no mag/baro/GPS, monocular, case-C),
- makes training match the real VQ2 course (10–20 m spacing, emergent ~close-range blackout, occlusion).

## 2. The invariant that makes it coherent

**OBS is position-free and relative** (exactly what the policy can obtain at deploy). **REWARD and CRITIC use sim ground truth** (privileged, training-only). This is existing doctrine (`d5_inc8_spec.md` §1.3/§2.3: "reward never sees the injected error; the world model owns margin; critic sees truth"). It lets us keep the proven reward while making the obs honest.

## 3. Grounding for the frame decision (why egocentric, settled)

- **Two orthogonal axes** were being conflated: (1) reference frame [gate vs body vs world], (2) cardinality/continuity [one gate w/ jump vs rolling window]. The handoff problem lives mostly on axis 2.
- **Continuity across a handoff comes from only two sources:** (a) a persistent state estimate that remembers gate positions [Swift's VIO], or (b) continuously re-detecting the gate [Jung 2018 reactive LOS]. We **cannot use (a)** — no trustworthy world state — so continuity must come from (b), which the real VQ2 10–20 m spacing makes possible (co-visibility was geometrically impossible at VQ1's 23.7–38.5 m — the `gate_mapper` finding; it flips at VQ2 spacing).
- **Swift's policy never sees absolute position** — it consumes velocity + attitude + rates + relative gate geometry (`reference_prior_art.md`; `project_phase2_rl_vision_decisions.md` L373). Our *original* I/O design said the same. The gate-frame single-gate obs was an **estimator convenience** for the case-C PnP pipeline, not a control choice — and the emulator even launders it through a world-NED KF (`obs = gate_world − kf_world_pos`), so gate-frame never gave the position-immunity we credited it.
- **Absolute position is unobservable (case-C) or redundant+fragile (case-A/B via map+association, 5–15% wrong-gate). Velocity + gate-relative geometry is the minimal sufficient, observable set.** Drop position from both the obs AND the estimator (relative-state filter).

## 4. Architecture (data flow)

```
detector (per-gate keypoints)
  → keypoint-visibility + occlusion gate            [B]  (from GT geometry — a physical fact)
  → per-gate relative-fix emulator                  [A]  (noise + smoothing + brief ego-propagation, NO world position)
  → egocentric obs                                  [C]  velocity(IMU-primary) + attitude + rates + last-thrust
                                                          + rolling window of gate-relative pos/normal + coarse-map designation
  → MLP policy → CTBR (body rates + collective)
critic = privileged (GT god-view)     reward = privileged (GT)
```

## 5. Components

### A. Relative-state estimator emulator — `rl/inc8_estimator_emul.py` (core rewrite)
- **Out:** 6-state world-NED KF `[pos,vel]`; the `gate_world − kf_world_pos` laundering.
- **In:** per-gate relative-measurement emulator. Each visible gate → noisy relative position (measured anisotropic σ) → per-gate smoothing filter → smoothed relative pos + confidence. Brief ego-motion propagation through gaps; **mask past ~0.5–1 s stale**. No world position anywhere.
- **Velocity = IMU-primary (MEASURED model, run 024203, 82 s stationary @ 143.5 Hz):** integrate accel → velocity. **The deploy KF corrects velocity ONLY INDIRECTLY through gate-relative POSITION fixes (pos–vel cross-covariance behind a Mahalanobis gate) — there is NO direct frame-to-frame vision-velocity observation on the VQ2 stack (vision commander, 2026-07-06; `update_velocity` is telemetry-only, wire carries none).** So model velocity as IMU-integrated slow drift with NO velocity snap at fixes. **Drift is BIAS-dominated, not noise.** Accel white noise is tiny (per-axis σ: y 0.0086 / z 0.0063 / x 0.0007 m/s²; x's 10× quiet is a sim artifact → model accel noise UNIFORM at ~0.008 m/s², do NOT bake the quirk). The drift lever is the per-flight **residual accel bias (post-pad-cal) × Δt-since-fix**; DR over it. Because vision corrects at 30 Hz and gaps are short (~1.3 m blackout ≈ 0.13 s @ 10 m/s), velocity stays well-anchored regardless — velocity is a CHEAP estimate.
- **Gyro noise is COLORED (AR(1), lag-1 autocorr 0.75), NOT white** (~5e-5 rad/s ≈ 0.003°/s) — model as AR(1); white gyro noise would add high-freq content the real wire lacks. Tiny magnitude, but colored if modeled at all.
- **No absolute yaw needed:** egocentric relative gate vectors are body-frame straight from the detection; only roll/pitch (gravity-leveled from accel) + body rates + relative geometry are required. The estimator holds NO world position and NO world heading.
- **Keeps:** anisotropic σ (lat ~0.10 / vert 0.28 / depth 0.85 m), per-episode bias, FIX-B latency inflation (relative frame), stochastic miss, random-in-frame teleport outlier.

### B. Keypoint visibility + occlusion — NEW `rl/gate_visibility.py`
- 8 corners/gate (4 inner @1.5 m + 4 outer @2.7 m, front face); project through camera (640×360, fx=fy=320, cx=320, cy=180, 20° pitch-up).
- **Occlusion = image-space projected-annulus test** (equivalent to raycast, cheaper/batched): a keypoint is occluded iff it projects inside a *nearer* gate's projected frame annulus (between projected inner & outer quads). Optional Blender depth-render spot-check for fidelity.
- **≥4 of 8 in-frame AND unoccluded → gate detectable.** 30 m far cap. Per-gate independent. **Blackout is emergent** (close-range corner exit under the 20° tilt) — exact range computed, not hardcoded. **No dedicated blackout stage.**

### C. Egocentric multi-gate obs + coarse map — `rl/peregrine_racing_inc8.py`
- Fixed **21 dim** (`WINDOW=2`, Fengyou 2026-07-07): velocity(3, body) + roll/pitch(2) + body rates(3) + last collective(1) + **coarse-map sector for the current target(2)** + **2-gate slider** `[current · next]`, each = relative pos(3) + confidence(1) + **visible-area(1)**. Critic = 16.
- **Passed slot + next-next both DROPPED.** Passed: a passed gate is behind the forward-facing camera → no fix → would just be masked; handoff continuity comes from the next→current promotion (the next gate was already tracked). Merit-checked (overshoot-recovery/heading/exit-line all redundant with velocity + next-gate; Swift & champion racers feed forward-looking gates only) → no merit; logged as a cheap A/B (briefly-ego-propagated passed slot) if the handoff-exit ever looks wrong.
- **Gate normal/yaw DROPPED** (measured too wide, §9) — replaced by **visible-area**: apparent opening area ÷ head-on area at that range ∈ [0,1]. ROBUST — from the detected corners' area, NOT the fragile PnP normal. Head-on ≈ 1 (big square, easy thread); sharp angle → small (foreshortened diamond) → the policy learns to square up first. Finer alignment inferred from the rel-pos trajectory. Deploy: projected inner-quad area ÷ expected head-on area at the range from rel_pos. **2026-07-07 RECALIBRATION (Fengyou):** the code now computes exactly this — `gate_visibility.gate_apparent_area` projects the 4 inner corners through the verified camera + K and normalizes by the square-on area at the range (range-invariant, square-on = 1), replacing the earlier `|cos(view, normal)|` stand-in (which diverged close-in where perspective matters and would not match the deployed detector's reported area). Same quantity for the obs channel (noised) and the reward's privileged area, computed with the emulated nose-first camera.
- **KILL ON CONTACT (absolute):** any gate contact → immediate episode termination + large negative reward. Zero contact is THE validity rule; never a soft penalty. Reuse DiffAero's collision-based termination on the true frame geometry.
- **Target-relative indexed window**, advances on pass (no teleport — next gate pre-tracked in adjacent slot); masked slots when stale. **Zero absolute position.**
- **Coarse map = a STATIC PREBUILT per-gate sector index** `sector[G]` — each gate → a 3×3 bucket (horiz ∈ {L,C,R} = {-1,0,1} × vert ∈ {D,C,U} = {-1,0,1}), in the drone's **gravity-leveled heading frame**. The obs FEEDS the **current target's sector (2 dims)**; as the target advances (0→1→2→…) the fed sector switches per the prewritten index (e.g. upper-right → upper-center → upper-left). Role = **acquisition/anticipation prior**: when the current target is not yet visible (out of FOV / pre-handoff) the sector says roughly where to look; when visible the `current`-slot rel_pos refines it (low-dim, changes only at handoffs → not a competing position signal). Auto-filled from course geometry (true nominal bearing bucketed) + human-overridable. At DEPLOY it also drives designation (which detection = current); this first build's training designation is clean (GT-correct), the sector-matching mechanism wired for the deferred misassociation layer. NOT the deferred gate-ID tracker.
- **Critic:** privileged GT — all gates true relative geometry + true velocity + per-gate confidence.

### D. Scene / curriculum — `peregrine_racing.py`, `peregrine_course.py`, `vq2_curriculum.py`
- `standing_start_frac = 1.0` (remove trivial dash).
- Spawn-attitude jitter (gate off-center in FOV).
- Gate spacing **10–20 m**.
- Gate-1 info flow emergent from visibility (no artificial cut).
- **Stages: `single_gate → handoff_drill → dual_gate_full → multi_gate`** (blackout_pass DELETED).

### E. Training schedule — sbatch
- Longer: hard stages ~5–6k updates, easy ~1.5k; optional `n_envs 4096` on multi. `algo=appo` (privileged critic).
- **Early-abort FLIGHTCHECK** wired + active human watch (don't burn compute on a failed run).

## 6. Verification plan (each gate blocks the next)
1. **Visibility:** hand-computed cases; report emergent blackout range; **assert next gate detectable through the current gate's blackout at 10–20 m / 30 m cap.**
2. **Estimator:** relative-fix noise matches measured σ; velocity IMU-primary drift character; propagation bounded; **assert no world position in obs path.**
3. **Obs:** fixed shape; handoff advances window w/o teleport; masked-when-stale; **assert position-free.**
4. **Coarse map:** correct gate flagged from hint; ambiguity path.
5. **Scene:** spawns varied/off-center; spacing in range.
6. **Local smoke → Adroit 3-update smoke** before ladder.
7. **`single_gate` must recover ≈0.8 before advancing** (B2b lesson: validate before `_COMMON`).

## 7. Build order (critical path)
B (visibility) → A (relative estimator) → C (egocentric obs) → coarse map → D (scene/curriculum) → reward/critic tweak → E (sbatch) → local smoke → Adroit smoke → ladder. Unit-test each before integration; `single_gate` gate before `multi_gate`. **New env alias — never edit the diffaero clone.**

## 8. Deferred (out of scope)
Misassociation/wrong-gate modeling (needs more thinking); gate-ID/ByteTrack; occlusion refinements beyond annulus; redesigned noise-anneal + economy-on-hard-stages; portfolio arms (lookat_max_rate, rw_centering, γ 0.9975).

## 9. Deploy dependencies (vision-commander)
**Per-gate yaw/normal accuracy — MEASURED (2026-07-06, cross-domain: VQ2 dark-red detector on VQ1 orange frames, 214 poses 0–23 m, map-free):** gate-normal median **19°** (p90 32°, growing 12°→22° with range); yaw/azimuth median **~11°** (p90 25°); two-fold flip **~6%** map-free (→~0% with a coarse orientation prior; a flip throws the normal 40–70° off). **DECISION: DROP the normal from the policy obs** — too wide to be load-bearing; infer alignment from the relative-position trajectory. Caveat: cross-domain measurement inflates keypoint noise (normal is hypersensitive to corner error); true in-domain VQ2 normal is likely better — **recalibrate + re-enable (low-confidence) when VQ2 GT (recon map) lands.** Flip-fix lever for re-enable: feed the coarse-map orientation prior to the PnP solver → flip ~0%.
- **Velocity — CONFIRMED** IMU-primary, corrected ONLY indirectly via position fixes (no direct velocity obs). Applied in A (`vel_correct_gain`).
- **Deploy per-frame contract:** per-detected-gate body-frame relative POSITION + confidence + **visible-area (range-free ratio = canonical)** (normal deferred). Coarse map = static offline course annotation (leveled-heading sectors), not a per-frame output.
- **Visible-area (foreshortening) — MEASURED (2026-07-06, 213 four-corner detections):** robust (no flips, low-order in corners — survives the corner noise that killed the normal). **Range-FREE form `area / max-edge²` = canonical (no range/PnP coupling), σ ≈ 0.054 (~0.03 near → ~0.10 at 20–40 m).** COARSE alignment cue: cos-like, departs from 1 by ~0.07 @22° / ~0.13 @30° → clears the noise floor only at ≳20–25° misalignment (excellent "am I badly angled?" signal; won't resolve <~15° — trajectory-inferred handles fine alignment). A models `visible_area = |cos(approach angle)|` + range-growing σ 0.03→0.10. Cross-domain caveat until VQ2 GT.

## 10. Risks
- Learnability (egocentric multi-gate is a harder basin than gate-frame single — mitigate: curriculum, clear target flag, ≥5 seeds).
- Perceptual aliasing (mitigated by coarse-map designation).
- Estimator rewrite is the heaviest single piece.
- Deploy contract break needs vision-commander coordination (§9).

## §P PERCEPTION-HONESTY PACKAGE + HARD NO-SPIN (2026-07-10)

**Code:** `rl/gate_visibility.py` (gate_los_perp_rate, blur_extra_miss_prob) · `rl/ego_estimator.py`
(step `blur_extra_miss=`) · `rl/peregrine_racing_ego.py` (yaw clamp, fatal spin abort, blur wiring,
diagnostics) · `rl/vq2_ego_curriculum.py` (`dual_gate_boot_floor_percept` →
`dual_gate_fullstack_floor_percept`) · `rl/peregrine_vq2_ego.sbatch` (dead-key removal, UPD defaults).
**ALL default-off**: every knob's default reproduces prior behavior exactly (pinned by
`tests/test_perception_honesty.py` + the stage snapshot pins in `tests/test_vq2_ego_curriculum.py`).

### P.1 Motivation (measured, 2026-07-10)
Champion vn16 (DET 0.545 @ real noise) is a **SPINNER**: 79% of training ticks yaw-railed at ±3.14,
cruising |w_z|~9.3 rad/s with roll coning ±130° (a corkscrew), steering by phase-modulated roll/pitch,
catching a visibility fix ~once per revolution (40.8% detectable duty in its own training trace); A2
confirmed the gait executes on the wire. Two root gaps: **(a) PERFECT SHUTTER** — `gate_detectable`
is instantaneous geometry, so a camera sweeping at 500°/s detects as well as one holding steady
(spin-scan is free in sim; the real detector starves under that blur); **(b) NO FRAMING PREFERENCE** —
vn16's config had rw_perception=0 and rw_rate=1e-3 is negligible (~0.009/tick at 9 rad/s vs ~2.0/m
progress).

### P.2 Owner directive → guarantee-by-construction
Fengyou: *"I don't want the system to be spinning at all. Even if it works in sim, it's not good...
REGARDLESS of how the vision system behaves in the simulator."* The PRIMARY mechanisms are FATAL
constructions, not dis-incentives — but the guarantee has an honest CEILING, stated here so nobody
reads "non-spin by construction" as absolute: **at the stage defaults the construction guarantees no
SUSTAINED rotation above ~2.36 rad/s** (= 2π·rev_abort/window_s, the leaky-integral fixed point);
constant rotation at ~1.8–2.35 rad/s (a slow roll/pitch corkscrew, one rev every ~2.7–3.5 s, ~82%
detection duty) survives BOTH triggers forever, is blur-free below lo=2.0, and is priced only by the
r_perc opportunity cost (~0.02/tick vs ~2.0/m progress). See the exploit ledger (P.8) and the
first-run trace check (P.6). The mechanisms:
1. **FATAL SPIN ABORT** (all-axis, realized ||ω||): sustained-rate clock (reused inc8 `bsr3_update`;
   3.5 rad/s for 0.4 s) **OR** leaky accumulated-rotation trigger (1.5 rev-equivalent over a 4.0 s
   window; steady-state ≈ ||ω||·window ⇒ sustained >~2.36 rad/s eventually fatal, a single ~130° bank
   impulse decays legally). Termination is **COLLISION-CLASS**: the env composes
   `lethal = below_floor | spin_abort` and rides it through the parabola path's `floor_contact=` /
   `forfeit_mask=` kwargs — **the non-obvious load-bearing wiring**: under the champion parabola
   regime the terminal fires on floor+oob ONLY (`ego_reward.py:839-843`), so a spin abort routed only
   through `gate_collision` would exit PENALTY-FREE with banked progress kept (spin-to-bail strictly
   cheaper than a miss). Pinned by
   `test_ego_reward.py::test_spin_abort_through_gate_collision_alone_is_free_under_parabola_THE_TRAP`.
2. **YAW-COMMAND CLAMP** at the point of application (`clamp_yaw_command`, top of ego `step()` before
   `dynamics.step`, so a_norm/last_action/plant all see the applied command). Action space stays
   ±3.14 (invariant 3 — deploy's `load_ego_actor` hardcodes the rescale). 0.35 rad/s commanded ×
   ~3.5 realized-gain (empirical, A2 — verify from the first _percept trace) ≈ 1.2 rad/s realized.
NO energy penalty and NO rw_rate retune (owner's no-energy-penalty directive; fatality replaces
dis-incentive; the <1% smoothness pin stays green).

### P.3 Blur model (SECONDARY: honesty/alignment)
`rate_eff = ||ω − (ω·r̂)r̂||` — the ROTATION-induced LOS-perpendicular angular rate, computed in the
UNFLIPPED Z-up/FLU body frame (magnitude invariant under the rigid π-about-body-z camera flip; no new
frame math).
Two-part gate, env-side (`gate_detectable` itself untouched — the estimator's standalone fallback is
unaffected):
- **HARD deterministic cutoff:** `detectable &= rate_eff < hi`, applied at BOTH env call sites
  (`_step_estimator` + `_current_detectable`). Deterministic-per-state ⇒ the obs mask and the
  estimator fix stream agree within a step by construction (the `:546` pure-stateless contract).
- **SOFT band [lo, hi), stochastic, ESTIMATOR-SIDE ONLY:** `blur_extra_miss_prob(rate, lo, hi,
  miss_max)` → folded into the EXISTING miss draw (`miss_thresh = miss_prob·noise_scale +
  blur_extra_miss`, clamped). Obs layout/semantics untouched (invariant 2): blur only changes WHEN
  fixes arrive; missed fixes → confidence decays → the existing masking, exactly like miss_prob.
- **PLACEHOLDER TABLE** (invariant 6, pending the A2 measured detect-vs-angular-rate curve):
  lo=2.0 rad/s (free below), hi=4.0 rad/s (hard cut), miss_max=1.0 (continuous with the hard cut).
  **A2 slot-in procedure:** replace ONLY `blur_extra_miss_prob` (lo := rate where detection starts
  degrading; hi := rate where detection ~0; non-linear shape = swap that one function body). If the
  curve is measured vs plain |ω| instead of LOS-perp rate, swap the abscissa (one line in
  `_blur_los_rate`); if it shows roll-vs-sweep anisotropy, add a roll weight there. **Ask the
  calibration owner which abscissa the measurement uses — AND whether the measured curve conflates
  TRANSLATION-induced sweep (the model is rotation-only, so lo/hi must not be fit to data the model
  cannot reproduce).**
- **NOISE-SCALE INDEPENDENCE RULE (load-bearing):** blur is CAMERA PHYSICS, not estimator corruption
  — it is NOT multiplied by ego_noise_scale, so the noise-0 calibration boot still sees blur (a
  blur-free boot would re-discover spin-scan and the fullstack would inherit it). Pinned by
  `test_perception_honesty.py::test_blur_survives_noise_scale_zero_the_boot_stage_property`.
- **KNOWN DIVERGENCE from wire semantics:** (i) in the soft band the obs det flag can read 1 while
  fixes rarely arrive (on the wire the det flag comes from real detections). Kept acceptable by
  miss_max=1.0 + a tight band; if A2 shows a WIDE soft region, the future fix is a unified
  per-keypoint gate — not this package. (ii) **ROTATION-ONLY abscissa:** the model omits the
  TRANSLATION-induced LOS rate (||v_perp||/range — no velocity input), so a fast close flyby (e.g.
  8 m/s at 3 m lateral offset ≈ 2.7 rad/s true sweep) blurs the real detector but never the sim one
  — residual sim-optimism in exactly the near-gate final approach, and the A2 detect-vs-ANGULAR-rate
  curve cannot capture it either. Accepted for the anti-spin purpose (spin is the target gait; the
  spin triggers are rate-based); sits next to the roll-about-LOS first-order-blur-free limitation.

### P.4 Threshold ordering (blind-policy defense)
`clamped realized yaw (~1.2) < blur-free lo (2.0) < rate abort (3.5) [< hi 4.0]` — a full-authority
pointing sweep is never blur-punished and never fatal, so "point the camera at the gate" remains the
constructive information strategy (gate-ward spawn + camera flip give detectability at birth; r_perc
supplies the gradient; 10–20 m spacing covers the handoff). hi=4.0 sits ABOVE the 3.5 abort:
harmless, but NOT because the [3.5,4.0) band is "fatal anyway" — the two thresholds live on
DIFFERENT abscissas (blur cut: instantaneous per-gate LOS-PERP rate; abort: all-axis ||ω|| SUSTAINED
>0.4 s), so a 0.3 s transient at ||ω||=3.8 is blur-cut but never fatal (clock resets), while a
sustained roll-about-LOS at 3.7 is fatal but never blur-cut (LOS-perp ≈ 0). Keeping hi=4.0 is
correct anyway: blur-blinding brief aggressive transients is desirable honesty, sustained rotation
in the band is priced by the abort, and the value preserves the A2 calibration structure — but
[rate_abort, hi) is NOT a dead band for the recalibration owner to ignore. Pinned by
`test_vq2_ego_curriculum.py::test_percept_threshold_ordering_blind_policy_defense`. If
target_detectable_duty collapses on the first run, the FIRST lever is raising
ego_yaw_cmd_clamp_rad_s (config), NOT softening the abort.

### P.5 r_perc (framing preference; config-only — the wiring existed end-to-end)
`rw_perception=0.02, exponent 4` in the _percept stages. **FARM-NEUTRALITY BOUND (construction
rule):** rw_perception ≤ rw_time (=0.02) ⇒ hover-and-stare nets ≤ 0/tick — the prior 0.05 warm-start
enablement detonated as a farmable fly-away (curriculum `:829-830`), and Geles' field value is 0.025.
Introduced at the FRESH boot (perception + progress learned jointly — the safer mode). Monitors:
`perception_reward` component, exit_timeout / exit_front (farming shows there). **Pre-identified
escalation path (NOT built):** a clip_pen_anneal-style plain-float mutation on
`env._egorw.perception` (pattern at `ego_reward.py:231-240`) — if ever built, L16 applies: the hook
MUST print `[perc-anneal] ON` and the print must be verified in a COMPLETED log or TB (stdout is
SLURM-buffered on running jobs).

### P.6 Stage variants + launch box
`dual_gate_boot_floor_percept` → `dual_gate_fullstack_floor_percept` = the live floor chain VERBATIM
(snapshot-pinned) + the package knobs; fullstack carries NO `+init_from` (the sbatch chain appends
it). Launch:
```
sbatch --export=ALL,SEED=0,RUNTAG=vperc0,STAGES="dual_gate_boot_floor_percept dual_gate_fullstack_floor_percept",UPD_dual_gate_boot_floor_percept=4000,UPD_dual_gate_fullstack_floor_percept=12000,PRECHECK=1 rl/peregrine_vq2_ego.sbatch
```
**PRECHECK=1 is MANDATORY on the first launch** (512 envs × 3 updates — the env class is
diffaero-gated cluster-only, so step()-wiring is exercised only there), **AND the precheck log must
show the new loss_components keys emitting** (`spin_abort_rate`, `spin_rot_accum_mean`,
`target_detectable_duty`) — the L16 silently-inert guard: absent keys == the package did not arm.
Optional cheap rung (Fengyou's call given the July clock): a ~2k-upd single-gate smoke
(`EXTRA="++env.course_n_gates=1"` on the boot stage alone) before the dual-gate spend.
**Success reads (training metrics, never renders):** DET completion vs the vdflr0 twin ·
spin_abort_rate/exit_spin → ~0 by convergence (early nonzero = the gate is teaching) ·
`target_detectable_duty` **≫ 40.8%** (vn16's spin-scan duty) · realized |w_z| from the trace ≤ ~1.5
(verifies the 3.5× gain estimate) · **SUB-ABORT CONSTANT-ROTATION check (the P.8 unpriced band):
the realized-rate trace's sustained ||ω|| distribution must NOT plateau in ~1.8–2.35 rad/s — a slow
corkscrew there is a spinner by the owner's standard while spin_abort_rate reads 0; corroborate with
spin_rot_accum_mean sitting near its fixed point (~||ω||·window ≈ 7–9.4 rad) rather than decaying
between banks** · exit_timeout/exit_front flat (r_perc not farmed). **STATS
CAVEAT:** ego_collision_rate/collision_rate INCLUDE spin aborts on _percept stages — subtract
spin_abort_rate (logged separately; exit_spin is its own box-exit class) before comparing gate
contact vs non-percept twins.

### P.7 DEPLOY-PARITY FLAGS (🚩 no code changes in THIS repo; #1 requires a DEPLOY-REPO code change at flight time)
1. **Yaw cap at flight time — a deploy-repo CODE CHANGE, not a launch argument:** the training-side
   clamp does NOT bind the deployed network (`load_ego_actor` hardcodes ±3.14). Any flight of a
   _percept checkpoint requires a **yaw-only clamp option added to `fly_rl.policy_step`**
   (`clip(rate_flu[2], ±ego_yaw_cmd_clamp)` = ±0.35 after the rescale, roll/pitch untouched).
   **NO existing fly_rl argument implements this.** Do NOT reach for the existing knobs:
   `max_rate` clips ALL THREE axes (`np.clip(rate_flu, ±max_rate)`, fly_rl.py:615-616) — roll/pitch
   trained at full ±3.14 authority, so max_rate=0.35 cuts them ~9× = catastrophically OOD;
   `yaw_scale` is MULTIPLICATIVE (`rate_flu[2] *= yaw_scale`, :617-619), not a clamp — it mis-scales
   the whole yaw transfer function (a railed 3.14 command → 1.1 rad/s commanded ≈ 3.8 rad/s realized
   at the ~3.5× gain — 3× the trained 1.2 and ABOVE the 3.5 rad/s abort the policy trained under,
   while a legal sub-clamp 0.30 command, passed through unchanged in training, is under-realized ~3×).
   Forgetting the mirror re-opens the spin door on the wire.
2. **Blur needs NO deploy mirror** — the real camera blurs physically. This package touches neither
   `ego_actor_obs`'s body nor EgoEstimatorConfig behavioral defaults nor area normalization, so no
   deploy obs drift is possible by construction (the deploy parity pin `test_ego_deploy_obs.py`
   lives in the Anduril-ego-deploy repo).

### P.8 EXPLOIT LEDGER (one line per mechanism: its degenerate optimum → what prices it)
- **Blur gate → freeze-and-drift** (hold perfectly still to keep fixes, never commit): priced by
  rw_time (−0.02/tick), progress/parabola/finish dominating completion, exit_timeout monitoring.
- **r_perc → hover-and-stare / fly-away farming**: priced by the farm-neutrality bound
  (r_perc ≤ rw_time ⇒ net ≤ 0/tick) + completion terms ~5× larger; watched via exit_timeout/exit_front.
- **Yaw clamp → roll/pitch tumble-scan** (scan with the axes the clamp doesn't touch): priced by the
  ALL-AXIS spin abort (||ω||, never w_z alone; `test_spin_abort_is_all_axis_not_yaw_only`).
- **Rate abort → sub-threshold slow scan** (rotate at 3.4 rad/s forever): priced by the leaky
  accumulated-rotation trigger (sustained >~2.36 rad/s eventually fatal).
- **BOTH spin triggers → constant sub-ceiling corkscrew (UNPRICED — the honest residual):** constant
  rotation at ≤ 2π·rev_abort/window_s (~2.36 rad/s at defaults) has its leaky fixed point exactly at
  the bar and NEVER triggers; the yaw clamp binds only channel 3, so a 1.8–2.35 rad/s roll/pitch
  tumble-scan (one rev every ~2.7–3.5 s, ~82% detection duty) is reachable, blur-free below lo=2.0
  (only ~17% extra miss at 2.35), and priced ONLY by r_perc's ~0.02/tick opportunity cost vs ~2.0/m
  progress. spin_abort_rate reads 0 while this gait lives entirely inside the legal band — the
  first-run trace check (P.6) is the detector. Knob mitigation if it appears: rev_abort~0.75 or
  window 8 s binds at ~0.6–1.2 rad/s sustained, but re-check the P.4 ordering first (those values
  sit near the clamped realized yaw ~1.2).
- **Spin abort → spin-as-FREE-EXIT** (the losing design's defect): priced by the CRASH-FOLD — the
  lethal mask rides `floor_contact=`/`forfeit_mask=` on the parabola path, so a spin abort costs
  terminal_base + full banked-progress forfeit (collision-class), never a free bail. THE single most
  important wiring in the package; pinned in test + here.
- **Spin abort → spin-as-CHEAPER-exit vs OOB (uncovered by the crash-fold entry above):** the spin
  terminal pays terminal_base (100) + banked forfeit while OOB pays 200, and unlike the floor dive a
  spin burst is reachable from ANY state in ~0.4 s via full-authority roll/pitch — a failing env with
  banked < 100 about to exit a wall prefers the brief >3.5 rad/s burst over the 200 wall. Would keep
  exit_spin nonzero at convergence and re-teach spinning exactly in failure states; diagnosable via
  the mandated exit_spin/exit_side reads (a converged exit_spin plateau correlated with wall
  proximity = this exploit, not a teaching phase).
- **Leaky-window residual hole:** a pathological on-off duty-cycled rotation can sit under both
  triggers (exponential leak ≠ sliding sum) — both thresholds are knobs; the first run's realized-
  rate trace is the check.

### P.9 Cleanup ledger (2026-07-10)
- sbatch BASE dead keys `+env.spin_rate_abort=10.0 +env.spin_time_abort=3.0` REMOVED (inc8-only,
  unread by peregrine_racing_ego; 10 rad/s would not have caught the 9.3 corkscrew anyway). New ego
  knobs use ego_spin_* names so stale sbatch copies can never silently arm the gate.
- `max_w_xy`/`max_w_z`: ZERO occurrences in rl/, src/, tests/, sbatch — adjudicated NONEXISTENT
  in-repo (only a cluster-side diffaero hydra yaml could define env.max_w_*; PeregrineRacing
  implements its own step/reward and never reads them). The only real rate limits are the ±3.14
  action bounds; realized rate exceeds the command via the plant super-rate map. The
  "|w_z|~9.3 vs max_w_z=1.0" confusion is retired.
- `BatchedEgoEstimator._rel_var` dead writes DELETED (written, never read); `latency_cov_inflate` +
  `rel_pos_std_init` marked INERT (fields kept so cfgs don't crash; wiring variance into confidence
  would be a behavior change = a future default-off item).
- Stale docstrings corrected (comment-only): ego_actor_obs 26-dim → 21, slots 3× → 2×; ego_critic
  20 → 16, slots {0,1,2} → {0,1}; the '(ego 26-dim...)' section comment.
- `_occluder_projected_quads` double projection: NOTE-ONLY comment (verified geometry — do not
  refactor without re-running the visibility battery).
