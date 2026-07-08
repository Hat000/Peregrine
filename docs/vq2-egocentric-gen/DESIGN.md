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
