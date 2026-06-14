# BLUEPRINT — Gate-Relative Case-C VQ2 Pipeline (SYNTHESIS)

**Status: ONE coherent build plan, reconciled from the 6 component specs (d1–d6) + the completeness
critic + the adversarial verifiers (v3, v3b, v6, v_obs).** This file supersedes the component specs
where they conflict; conflicts are resolved explicitly in §0 and the component sections cite the
winning value.

Scope discipline: DESIGN + offline validation only. Nothing here edits `src/` or `main`. Every
load-bearing number was re-derived THIS session (sim re-run / source read / verifier re-run) and is
tagged MEASURED / EXTRAPOLATED / ASSUMED.

---

## HEADLINE (read this first — do not skip to the optimistic parts)

**The gate-relative OBSERVATION fix is SOLID and must be built: it removes the per-track map/registration
bias EXACTLY** (re-run this session: rel arm E_bias **−0.000 m** / D_bias **−0.004 m** vs abs **+0.176**
/ submap anti-pattern **+0.174**; rel in-plane RMS **0.139 m** vs abs 0.279 / submap 0.228). This is
necessary, proven, and the single highest-value change. Build it regardless of organizer Q① (it degrades
gracefully to case A/B as the high-confidence limit).

**But the gate-4 0.155 m WORST-CASE margin does NOT close offline in ANY regime.** A margin is a p90/p99
gate (a contact rule), not an RMS gate. **Re-run this session, `v3b_closure_refute_results.json`: every
single cell across the entire {σ_v ∈ 0.3–0.5} × {accel-bias 0–0.24 m/s²} grid has `p90_clear = false`.**
The best cell (σ_v=0.3, bias=0) is RMS 0.126 but **p90 0.190, 24% contact**. The only "GO" numbers
anywhere in the component specs (c1-warm RMS 0.139, d4v-visvel RMS 0.144) are RMS-clears that are
**simultaneously p90-FAILS** (0.203 / 0.213). **The entire speed-at-37-m/s thesis is escape-hatched to a
live recording (L3) that has not happened.** See §4 and §3.

**The velocity-channel "cheap lever" (d4v) is REFUTED and is demoted to insurance** (§1.4): at an honest
inter-frame PnP-delta σ_v ≈ **2.81 m/s** (not the assumed 0.3–1.0) and the mandated RewindKF latency
**L≈115 ms**, the cold case-C in-plane RMS is **0.19–0.35 m (NO-GO)** at every smoothing window
(`verify_velchannel_v6.py`, re-run this session). Margin closure must rest on **accel-bias / attitude
control (≤~0.6°) + the uncertainty-aware speed-down**, NOT on this channel.

**Margin-closure verdict: DOES-NOT-CLOSE offline → CANNOT-SETTLE-OFFLINE.** The cold-vs-warm velocity
prior and the true effective accel bias are the binding swing variables, and both are pinnable ONLY by the
ShadowPC at-speed gate-4 recording (§4, §6). Build the whole pipeline (it is the right architecture and
the observation fix is real), but **gate the speed ladder on the live recording, not on the offline
sims**, and **select policies on a p90 gate, not RMS**.

---

## 0. RECONCILIATION — the conflicts the critic raised, RESOLVED

These are FROZEN decisions. The component specs disagreed; this section is the single source of truth.

### 0.1 OBS LAYOUT — FROZEN at 20-dim, bounded-ratio confidence triple (resolves 18/19/20)

The three specs carried three incompatible widths (d6=18 +1 scalar; d1/d3/d4v=19 +2 raw-metre σ;
d5=20 +3 bounded ratios). Source of truth today: `rl/peregrine_racing.py:518 self.obs_dim = 17`
(verified this session). **FREEZE = d5's 20-dim, bounded-[0,1]-ratio triple** — it is the richest, is the
one the retrain actually consumes, and bounded ratios are stable PPO inputs (raw σ in metres has no
natural scale and destabilizes the obs-normalizer). d1's "raw σ in metres" representation and d6's "+1
scalar" are **superseded**.

```
obs_dim = 20  (was 17; +3 append, dims 0..16 byte-for-byte unchanged)
[0:17]  unchanged frozen contract (see §1.1)
[17]    c_inplane = clip(sigma_ref / sigma_inplane_hat, 0, 1)   in-plane (E,D gate-frame) confidence
[18]    c_along   = clip(sigma_ref / sigma_along_hat,  0, 1)   along-track (N gate-frame) confidence
[19]    age_norm  = clip(t_since_last_accepted_fix / TAU_STALE, 0, 1)   staleness clock
        sigma_ref = 0.05 m (the 1-σ confidence bar);  TAU_STALE = 0.10 s
```

`sigma_*_hat` is the **calibrated** gate-frame KF σ (component 1/2 deliver `sqrt(diag(P_gate))`). The
critic (`get_state`, asymmetric) gets TRUE gate-relative state + the SAME triple → critic dim 33→36.
`peregrine_racing.obs_dim`, the `_ActorMean` input layer, and the critic input layer all move 17→20 / 33→36
**in lockstep**. The deploy gate keys off the checkpoint sidecar obs-dim (17 → no append, existing path;
20 → pass live `RewindKF.P`). Back-compat is a **pure append** (verified this session: `obs20[:17] ==
obs17` to 0.0, `v_obs_adversarial_check.py` check D).

### 0.2 SIGN CONVENTION — the obs slot is +L = (gate_pos − pos), NOT −L (resolves a 24 m flip)

d1's header and d2 §1.3 say "the estimator delivers **−L** (the drone→gate offset)". **That sign is WRONG
for the obs slot.** Re-derived end-to-end this session (`v_obs_adversarial_check.py`, real
`localization.gate_pose_to_world_position` + real NED↔Z-up `_FLIP`):

```
max| obs pos_g  −  R_w2g_zup @ (+L_zup) |  =  4.8e-07   (recoverable float epsilon — CORRECT)
max| obs pos_g  −  R_w2g_zup @ (−L_zup) |  =  24.0       (FULL SIGN FLIP — the spec text is wrong)
```

The estimator must deliver `L_seen = gate_seen − p` (the +L lever, == `R_world_cam @ t_cam_gate` in NED,
converted to Z-up), and the obs builder applies `pos_g = R_w2g @ (+L_seen_zup)`. The d1 "0.0 bit-exact"
unification proof is a **tautology** (its check (2) computes `R_w2g@(gate-pos)` two ways in the same frame
with `L := gate-pos`, identical by construction; it never exercises localization or the NED↔Z-up
conversion). **The real faithfulness gate is `v_obs_adversarial_check.py`'s end-to-end check WITH a
sign-flip negative control** — fold it into G1 (§3). This is the project's hardest-won bug class (4 prior
frame/convention bugs); the +L convention is now pinned.

### 0.3 MARGIN HONESTY — selection is a p90 gate, not an RMS gate (resolves the "papered-over RMS")

By the blueprint's OWN sims, **no offline regime clears the gate-4 p90/p99 worst-case margin** (§0, §4;
`v3b` all cells `p90_clear=false`). Therefore:
- The headline finding is **DOES-NOT-CLOSE offline → CANNOT-SETTLE-OFFLINE**, escape-hatched to L3.
- **The selection metric (d5 §5.3 criterion 2, d6 G4/G6) is changed from RMS to p90:** a candidate is
  portfolio-eligible only if **gate-4 SIMSTART in-plane p90 < 0.155 m @ r=0.38** (not RMS < 0.155).
- Stop citing c1-warm RMS 0.139 or d4v-visvel RMS 0.144 as "clears" — both are p90-fails (0.203 / 0.213).

### 0.4 VELOCITY CHANNEL — DEMOTED to "unverified, likely insufficient" (resolves d4v vs d3/v3/v6)

d4v headlined a cold→GO flip (0.215 → 0.144) and recommended "lean BUILD". **Refuted this session
(`verify_velchannel_v6.py`):** (a) d4v injected GROUND-TRUTH velocity + noise; an honest differenced-
position σ_v is **≈2.81 m/s** (4-frame@30 Hz), not 0.3–1.0; (b) d4v ran at L=0 while the chain mandates
RewindKF L≈115 ms. Honest results:

```
honest PnP-delta σ_v = 2.81 m/s (4-frame)  -> cold in-plane RMS 0.268, p90 0.398   NO-GO
honest PnP-delta σ_v = 1.25 m/s (9-frame)  -> cold in-plane RMS 0.287, p90 0.430   NO-GO
L=115 ms, GT-cheat visvel (recorded path)  -> RMS 0.189, p90 0.285                  NO-GO
L=115 ms, cold pos-fix-diff ONLY (baseline)-> RMS 0.350, p90 0.526                  NO-GO
cold, ZERO velocity offset (no stress)     -> RMS 0.201, p90 0.299                  NO-GO
```

**Decision:** position-fix-differencing IS the KF (free, ship as baseline). The direct vision-velocity
channel is **insurance, not a margin lever** — design it, do NOT make it load-bearing, and do NOT rate the
cold-prior risk low on the strength of it. Margin closure rests on accel-bias/attitude control + the
uncertainty-aware speed-down (§4).

### 0.5 COLD-PRIOR RISK — reconciled to HIGH everywhere (resolves d5 R-LOW vs d2/d3/d4v/v3 HIGH)

d5 §7 rated cold-vs-warm R-LOW ("0.17–0.21 m, prior is warm by gate-4"). Re-derived this session
(`v3_margin_refute_results.json`): cold @ zero bias is **RMS 0.154, p90 0.234, p99 0.322, 37% contact**;
cold @ the MEASURED 1.4° attitude bias (0.24 m/s² phantom accel) is **RMS 0.229, p90 0.338, p99 0.453,
65% contact**. Cold is insensitive to the init error (vinit_sigma 1.5/3.0/6.0 all p90 ≈ 0.227) → the
failure is **variance/window-driven, not init-driven**, so "warm by gate-4" does NOT hold (the full lap
does not pre-converge velocity to warm quality). **Cold-prior risk = HIGH.** inc8 DR and selection are
tuned to the COLD design point (§2.5), not the warm one.

### 0.6 NEES — quote the right figure (resolves 1.96 vs 3.3–4.3 vs χ²(3)=3)

c1's **NEES = 1.96 is the IN-PLANE (2-DOF-dominated, gate-plane) NEES**, not a 3-DOF figure — it sits
below χ²(2)=2, i.e. the gate-relative filter is **mildly OVER-confident in-plane** (full 3-DOF NEES 1.96
is also < χ²(3)=3). d1 §3.2's separate "NEES 3.3–4.3" is the deployed-rewind buffer-level number. These
are different quantities and must be labeled as such. **Calibration requirement stands and is unchanged:**
component 1/2 must deliver a gate-frame `P` whose realized NEES → χ²(3)=3 (a `c_cal` trim, fit on L3),
asserted by the §2.2 unit test `mean(e_g²/diag(P_hat)) ∈ [0.8, 1.3]`. The over-confidence direction makes
this MORE important, not less (an over-confident channel teaches the policy to trust a lie).

### 0.7 R4' confidence-gated cone — acknowledge the real code surface

`peregrine_racing.py:342 tilt_free_rad` is a **scalar config constant** used as `math.cos(w.tilt_free_rad)`
in `compute_reward_terms` (line 365, verified). Making it per-env and wired to the live `c_inplane` inside
the reward is a **larger change than "replace the scalar"** (tensor broadcast of a per-env free-cone,
plumbing `c_inplane` into `compute_reward_terms`). Carry the **fixed-cone variant as the default L0** and
the confidence-gated cone as a portfolio ablation (§2.4) — if the gated cone shows no eval benefit, the
fixed cone is simpler and deploys identically.

### 0.8 ESTIMATOR-EMULATION EVAL WRAPPER — a hard prerequisite of G4/G6

`contact_true_eval.py:260` runs `obs_from_truth` (verified) — perfect pose, no perception noise, no
gate-relative error, no confidence channel. Selecting on it crowns policies on a fiction. **The §2.6
estimator-emulation obs wrapper + a `v*` (achieved gate-4 approach speed) extraction instrument are HARD
prerequisites of G4/G6.** Until both exist, G4/G6 results are upper bounds only and cannot crown a policy.

---

## 1. THE INTEGRATED PIPELINE (component by component, build-ready)

```
 wire telemetry (DroneState s)
   pos/vel given*  ──►  [C2 ESTIMATOR]  ──►  +L_seen offset + calibrated P  ──►  [C1 OBS]  ──►  [C5 inc8 actor] ──► CTBR wire
   accel_body            RewindKF(0.5s) +     (pos_g from SEEN gate +L,           build_obs 20-dim    (20-dim)        SET_ATTITUDE_TARGET
   odo quat (TRUE)       gate-rel in-plane     map bias dropped EXACTLY)          gate_map + est +                    (BODY_RATE only)
   camera Frame ─────────► detector → PnP → associate → relative-innov gate ──────┘ confidence triple
   * present in case A/B; ABSENT (origin seed, pos_std 5.0) in true case C
```

### 1.1 Detector → PnP → association (UNCHANGED)
- `_process_observation` (navigator.py:363) already calls `estimate_gate_pose(obs, prior=predicted[...],
  compute_covariance=True)`. The PnP `prior` (map+attitude+KF prediction) breaks the IPPE 2-fold tie —
  keep exactly. Output: `t_cam_gate`, `R_cam_gate`, `reproj_error_px`, 6×6 covariance, `n_corners`.
- `_associate` (navigator.py:402) shape-gate + `range_consistent` depth-sanity (navigator.py:374) — keep.
- **Visibility (MEASURED, c1 part_d re-run):** gate-4 holds **4 corners in frame down to 4 m, drops to 2
  corners (P3P only) at 3 m** at the 37 m/s drag-hold posture (cruise pitch −38.4°). L3 must confirm the
  live detector matches; earlier 4→<4 clipping drops the 4-corner accepted-fix rate at the binding band.

### 1.2 The gate-relative in-plane fix — observe +L to the SEEN opening (the core fix)
- **Math.** `L = R_wc @ t_cam_gate` (`R_wc = R_wb @ R_camera_from_body().T`), the world-NED lever to the
  **seen** corners. The obs offset is `+L` (gate_seen − p); the per-track map bias `db` never enters `L`,
  so it drops out EXACTLY (re-run: rel E_bias −0.000 vs submap/abs +0.174/+0.176). **DO NOT compute
  `R_w2g @ (gate_map − p_KF)`** — that re-injects `db` (submap anti-pattern, 0.228 m FAIL).
- **Where.** In `_process_observation`, AFTER `gate_pose_to_world_position` (navigator.py:380) and the
  existing absolute `update_position` (navigator.py:397, KEPT), add a second in-plane correction. New
  helper `gate_relative_inplane_fix(pose, gate, R_wb) -> (z_ned, cov_ned)`. Implement as a 3-DOF
  `update_position_at` with an **anisotropic cov**: tight in-plane (PnP lateral σ), loose along-track —
  no new KF method needed.
- **Cov shaping (MEASURED).** In-plane (E,D at gate-4): per-axis σ = **0.265 m** (accepted maha≤16.27
  4-corner near band, re-run), growing `a1*r` at range — **NO 0.40 m `FIX_COV_FLOOR_STD`** in-plane (that
  floor is bias-absorption; the relative obs has no bias to absorb). Along-track: loose (absolute fix +
  IMU own it), keep the floor there. Caveat: `a1=0.026` extrapolates badly past ~24 m — in-band for the
  ~12 m gate-4 window, flag for long range.
- **AUGMENT, not replace.** The absolute KF stays for planning/feed-forward/g4→g5 handoff; the relative
  term owns ONLY the terminal in-plane centering miss inside the visibility window.

### 1.3 The REQUIRED relative-innovation outlier gate
Reprojection error does NOT separate depth-flips (MEASURED: flip p50 **0.445 px** < clean p50 **0.657 px**;
a reproj gate lets **93.2%** of flips through). The relative-innovation gate is REQUIRED:
```
nu_ip = e_obs_ip − e_pred_ip   ;   d2_rel = nu_ip^T S_ip^-1 nu_ip   ;   ACCEPT iff d2_rel <= chi2(2,0.999) = 13.82
```
Keeps 99.9% clean, rejects 99.8% of flips (MEASURED, `d2_relinnov_gate_check`). **Caveat (carry, from the
d4v/v3 adversarial):** the 99.8%-reject headline holds for throws ≥1.5 m; flip survival rises at small
throws (48.5% @1.0 m, 95.5% @0.5 m). Small-throw flips that survive are bounded by the absolute Maha gate
+ range-consistency upstream; the relative gate is the in-plane backstop, not a sole defense.

### 1.4 Velocity (P0 #3) — KF pos/vel coupling, vision-velocity is INSURANCE not a lever
Case-C velocity is observable ONLY via position-fix differencing inside the KF (vision is position-only).
**Do NOT build a load-bearing vision-velocity channel** (§0.4 refutation). Export `kf.velocity` +
`P[3:6,3:6]` on the existing `NavState.velocity_ned` / `pos_vel_covariance` — that IS the interface; the
obs `vel_g = R_w2g @ vel` consumes `kf.velocity` directly. A weak inter-frame PnP-delta `update_velocity`
MAY be added later as insurance, but it does not close the cold margin at honest σ_v + mandated latency,
so it is **deferred P2**, not the cheap lever d4v claimed.

### 1.5 RewindKF horizon + the 3 P0 bugs
- **RewindKF wrap (case C only):** `self.kf = RewindKF(kf=LinearKF.initialize(...), horizon_s=0.5)`.
  Horizon MUST be **strictly > L** (`horizon == L` drops 100% of fixes — 1-ns quantization edge, MEASURED;
  diverges to RMS 0.423 m / ~21 m over a track). 0.5 s covers edge L (16 ms) AND CPU-class (125 ms). HARD-
  blocked on measuring L on eval HW (P0-2) and TIMESYNC (P0-3).
- **P0-a `_initialize` case-C seed (navigator.py:255-271):** gate the seed on `config.use_given_position`,
  NOT on `ds.position_ned` presence. True case C seeds **origin @ pos_std=5.0** even with LPN on the wire.
  The per-tick given updates (navigator.py:315,320) already have this guard — `_initialize` is the sole
  leak. THE gate to validating anything (else every "case C" test is secretly case A).
- **P0-b TIMESYNC (navigator.py:426):** learn `delta_epoch = frame.sim_time_ns − ds.sim_time_ns` once
  (paired via `recv_monotonic_ns`), index RewindKF + tsv by the IMU master clock. Predict-forward
  (constant calibrated age = L, `update_position_at(now − L_const, ...)`) sidesteps TIMESYNC for the cheap
  80% — **ship it first**. Full RewindKF (capture-time OOSM) is HARD-blocked on TIMESYNC, verified live at
  L4 (else the buffer indexes a garbage epoch and the whole C2 chain is INVALID by construction).
- **P0-c velocity:** as §1.4 — no new estimator surface.

### 1.6 Calibrated covariance → the confidence channel (C1 §0.1 triple)
After each (rewind) fix, project `P[:3,:3]` into the gate frame with the SAME `R_w2g` the obs uses:
`sigma_inplane_hat = sqrt(P_g[1,1]+P_g[2,2])`, `sigma_along_hat = sqrt(P_g[0,0])`. MEASURED: tracks true
in-plane error when fixes land (0.131 m est vs 0.146 m true, slightly conservative), inflates to ~0.6 m
when fixes drop — a faithful fast-when-confident channel. **Calibration is binding:** deliver NEES→χ²(3)=3
(the §0.6 over-confidence direction makes this mandatory); `c_cal=1.0` is a PLACEHOLDER until L3.

---

## 2. THE inc8 RETRAIN SPEC (Adroit-ready)

### 2.1 Obs (consumes §0.1): 20-dim, critic 36-dim
Train env populates `pos_g` (obs[0:3]) as **ground-truth gate-relative +L offset + the MEASURED estimator
residual error** (§2.2), NOT pristine truth and NEVER `R_w2g@(gate_map − p_KF)` (forbidden anti-pattern).
`vel_g`, `rpy_g` ride the same pose. Append the §0.1 confidence triple. Critic (`get_state`) gets TRUE
gate-relative state + the same triple (33→36).

### 2.2 MEASURED-error DR (`+env.estim_dr=case_c`) — on the OBS only, true dynamics/contact unperturbed
- **(a) In-plane per-fix lateral noise:** per-accepted-fix `n_E,n_D ~ N(0, σ²)`, **σ ~ U[0.08, 0.30] m/axis**
  per episode (measured 0.265). KF-smoothed via a low-pass with **N_eff ~ U[4,9]** so realized gate-plane
  1σ = σ/√N_eff. **VALIDATION TARGET: emulated `e_E,e_D` 1σ at the gate-4 plane matches the c1 rel table
  (0.265→0.095, 0.20→0.080, 0.08→0.049).**
- **(b) Range-collapse structure** (NOT flat): `σ(r) = σ * clip(r/9.0, 0.5, 2.2)` (matches measured
  0.20 m near / 0.41 m far). **Coherence note (critic gap 3):** sub-spec (a)'s flat-target validation
  (0.265→0.095) and (b)'s range model independently yield different filtered σ (≈0.107). FREEZE (a)'s
  flat 0.095 @0.265 as the validation target; (b) is the within-episode range shape whose EPISODE-MEAN
  must match (a). Assert both in one unit test (mean-over-range == (a) target).
- **(c) Bias:** MAP bias DROPS OUT — do NOT inject a per-track in-plane bias. Inject the residual
  one-signed PnP/extrinsic bias the gate-relative path does NOT remove. **Critic correction (d5 adversarial,
  ACCEPTED):** the c1 caveat bounds this at ≤0.19 m magnitude of UNKNOWN sign; a real +0.19 m one-signed
  → RMS 0.242 / p90 0.337 (OVER). The spec's ±0.10 m ZERO-MEAN draw does NOT build robustness to a one-
  signed systematic. **Change: inject `b_inplane ~ U[0, 0.19] m` with a PER-EPISODE RANDOM SIGN held
  constant within the episode** (not zero-mean-within-episode) so the policy must be robust to a constant
  unobserved offset of either sign. Magnitude bound is ASSUMED until L3/SHADOWPC-VISION-CAL.
- **(d) Latency staleness — ALONG-TRACK only:** `e_N += v_along*L`, **L ~ U[0.006, 0.125] s**; in-plane
  leak tiny (`v*L*sin(0.5°)` = 5–41 mm). RewindKF removes first-order along-track staleness → model
  residual `~ U[0, 0.05] m` with `+env.rewind_on=true` (DEFAULT), full `v*L` for the no-rewind ablation.

### 2.3 Reward (3 changes; everything else frozen)
- **R1' arc-length progress over the REBUILT contact-safe line** (`ReferenceLine.progress`,
  `src/racer/reference_line.py` — VERIFIED loader; `reference_line_vq1.json` is drag-infeasible + 170°
  inverted, MUST be rebuilt = C4). Closes corner-cut M-2 by construction. `rw_progress = 10.0`.
- **T4 finish-time pressure KEPT** (`rw_finish 20 + rw_finish_time 1.0/s`) — the speed reward,
  confidence-blind by design (cannot earn time by being timid).
- **R4' tilt free-cone** — DEFAULT = **fixed relaxed cone** (the simpler change, §0.7); confidence-gated
  cone `tilt_free(c_inplane) = lo + (hi−lo)*c_inplane` is a PORTFOLIO ABLATION (larger code surface:
  per-env tensor cone + `c_inplane` plumbed into `compute_reward_terms`). Inside the cone R4'=0 (NOT
  damping). `lo=60°`, `hi`=ladder target.
- **Anti-damping mechanism:** the asymmetric critic sees TRUE gate-relative state, so needless braking
  under HIGH true-margin gets negative advantage and PPO unlearns it. Caution is an EMERGENT best-response
  to honest perception noise + honest contact penalties — never a shaped term. **NEVER reward damping.**

### 2.4 Envelope ladder (BSR3 spin-gate MANDATORY first)
- **BSR3 (HARD prereq before any retrain):** widen `spin_rate_abort → 9–10 rad/s`, `spin_time_abort → 3.0 s`
  (super-rate plant legitimately commands ~11 rad/s; else the relaxed cone auto-aborts).
- **Ladder (rw_tilt FIRST, then free-cone):** L0 baseline (R1'+T4+fixed-cone 60°, rw_tilt 96) →
  L1 (rw_tilt 96→48) → L2 (free-cone 60→70°, cap 70 — 75–80° is the M-1 corner-cut regime). Each rung
  GATED on the §2.6 p90 gate at the rung's achieved speed.

### 2.5 Seeds + selection (p90 gate per §0.3; design point = COLD per §0.5)
- **≥5 seeds per rung** (inc7 was 2/3 viable; gate-relative + confidence + cone-gating is a harder basin).
  Portfolio = {L0,L1,L2}×5 + the fixed-vs-gated-cone and all-gate-lookahead ablations as 5-seed side-runs.
- **SELECT = the FINISHED policy with the lowest finish time among those whose gate-4 SIMSTART in-plane
  p90 < 0.155 m @ r=0.38 (NOT RMS) AND S_stable ≥ 2/3 (prefer ≥3/5) AND whose achieved v\* clears the c5
  σ-gate at component-2's DELIVERED per-fix σ at v\*.** The c5 σ-gate (MEASURED): per-fix σ ≤0.10 m →
  >55 m/s; 0.20 m → 17 m/s; 0.265 m (today's measured) → ~13–15 m/s. **The ESTIMATOR sets the ceiling, not
  the policy.** Tune DR/selection to the COLD velocity-error class (§0.5), not warm.
- Crown ONLY after a ≥3–5-lap fresh-reset live batch.

### 2.6 Eval extension (HARD prereq of G4/G6, §0.8)
- **Estimator-emulation obs wrapper:** inject the §2.2 `e_g(range, N_eff, L)` into the obs inside
  `contact_true_eval.run_episode` AFTER `obs_from_truth`, BEFORE `policy_step`, so the eval drone flies on
  NOISED gate-relative pose. Keep truth-pose eval as the optimistic upper-bound control.
- **`v*` extraction instrument:** read `|v|` at the last pre-gate-4 fix window from the SIMSTART
  trajectory (the σ-gate input). Without both, G4/G6 are upper bounds and cannot crown a policy.
- Eval invocation: `contact_true_eval.py --ckpt <inc8> --plant mixer --body-radius 0.38 --frame-depth 0.30`
  (map-ON mixer plant; r=0.38 = worst-case body radius; `PASS_BAND = 0.75 − 0.38 = 0.37 m`).

---

## 3. INTEGRATION + VALIDATION PLAN

### 3.1 The central architectural change
Today's standing-start judged path (`submit_rl.py --no-bridge`) NEVER runs the Navigator/KF — `fly_rl.
_fly_armed` skips PATH B and calls `build_obs(s,...)` on the raw wire pose (MEASURED, `fly_rl.py:1083`).
The gate-relative rebuild **inserts an estimator stage that does not currently exist** in the judged path.

### 3.2 Guard replacement (SAFE — loud-abort moves from "is this VQ1?" to "is the pipeline wired?")
- `_assert_vq1_constants_consistent` (import-time, course-independent) — **KEEP**.
- `_assert_live_course_is_vq1` (deploy-time) — **REPLACE** with `gate_map = make_gate_map(live TRACK_INFO)`
  threaded into `build_obs(..., gate_map=gate_map)` + an **estimator-readiness assertion** (RewindKF
  initialized, TIMESYNC reconciled). Any non-π course that is NOT fully wired still LOUD-ABORTS — never
  silently corrupts the obs by metres.
- `assert_gate_map_allpi` — **KEEP** as a negative guard on the `gate_map=None` default path only.

### 3.3 Judged path stays clean
`submit_rl.py` structurally unchanged: only the `--checkpoint` pin moves inc7→inc8 (+ ship the `.json`
sidecar, else thrust silently overdrives to 33%). Safety pins (`--no-bridge --no-auto-reset --no-debug-obs
--flights 1`) still appended LAST. The estimator is **client-side only** — emits NO MAVLink command, so NO
`MAV_CMD 31000` reaches the judged wire (verify with the independent relay decode). No per-tick file I/O
(buffer + flush at session end).

### 3.4 Offline gauntlet (cheapest fails first; all torch-free in `.venv`)
| # | Gate | Pass criterion (RECONCILED) |
|---|---|---|
| **G0** | Import + sidecar | estimator chain imports torch-free; inc8 `.json` sidecar present, parses `act_max_thrust`/`act_max_rate`. |
| **G1** | Obs bit-exactness + **sign control** | dims[0:17] == `get_observations` to ≤2e-4 (VQ1) / ≤1e-5 (non-π yaw-aware); 20-dim append leaves [0:17] byte-identical (0.0, verified). **NEW: the end-to-end `v_obs_adversarial_check.py` `+L` identity passes (4.8e-7) AND the `−L` sign-flip negative control BREAKS (24 m).** |
| **G2** | Estimator unit checks | `update_position_at(t≥now)` bit-identical to `update_position`; OOSM 3-DOF NEES → χ²(3)≈3 (note: c1's 1.96 is the in-plane 2-DOF figure, §0.6); horizon **strictly > L_max** (assert, else diverge to ~21 m); gate-rel fix uses σ=0.265 m, NO 0.40 m in-plane floor. |
| **G3** | Margin sim (C3) | **p90 gate, not RMS:** report gate-4 in-plane RMS / **p90** / p99 + frac-over at the estimator's delivered σ. rel arm E_bias≈0 (map bias dropped). **EXPECT p90 > 0.155 m → escape-hatch to L3 (this is the make-or-break, by design unresolved offline).** abs 0.279 / submap 0.228 = negative controls (must FAIL). |
| **G4** | `contact_true_eval` selection | **REQUIRES the §2.6 emulation wrapper + v\* instrument first (§0.8).** gate-4 SIMSTART in-plane **p90 < 0.155** @ r=0.38; D-offset probe (gates 3/4/5) does not flip pass→collision under ±1.5 m; yaw-aware. |
| **G5** | S_stable + BSR3 | S_stable ≥ 2/3 across ≥5 seeds; widened spin aborts do NOT trip on nominal rollouts AND still catch an injected sustained spin (negative control). |
| **G6** | Speed-ladder | fastest policy whose v\* clears the c5 σ-gate at component-2's σ AND gate-4 p90 < 0.155. The estimator sets the ceiling. |
| **G7** | Full suite | 692 → ≥692 + new tests, 0 regressions (pytest from repo ROOT). |

### 3.5 Live ShadowPC ladder (only after the gauntlet is green; zero gate-contact = THE validity rule)
1. **L0** — case-A dev smoke (estimator passthrough/given pose) on VQ1: confirms 20-dim obs + new guards
   don't regress the LIVE-CONFIRMED 5/5 VQ1 baseline. Run `frame_residual_report.py` after (East residual
   ~+0.97..0.99; tool is robust to obs-dim growth — reads wire columns, not the obs vector).
2. **L1** — case-C cold-start activation (P0-a: origin seed, pos_std=5.0) on VQ1: FIRST end-to-end gate-
   relative fix on hardware. Pass: 6/6, zero contact, recorded per-fix σ ≤ 0.265 m/axis.
3. **L2** — at-speed fresh-reset batch (~37 m/s), N≥10: zero gate contact (gate-4 specifically clean).
4. **L3 — at-speed gate-4 vision recording — RESOLVES THE ESCAPE-HATCH (see §4).** The run that buys down
   the p90 residual. **HARD-blocked on TIMESYNC (P0-b) being on one clock.**
5. **L4** — RewindKF latency confirmation on the eval host (CPU ~115 ms / GPU ~15–25 ms); confirm horizon
   > L live. HARD-blocked on TIMESYNC (else the chain is INVALID by construction).

---

## 4. RANKED RESIDUAL RISKS — does the velocity-prior margin close?

**It does NOT close offline.** `v3b` (re-run): **every cell** in the {σ_v × accel-bias} grid is
`p90_clear=false`; best cell RMS 0.126 / p90 0.190 / 24% contact. The make-or-break is escape-hatched to
L3. Ranked:

| # | Risk | Sev | Mitigation / resolver |
|---|---|---|---|
| 1 | **Gate-4 p90/p99 worst-case margin does NOT clear offline in any regime** (cold p90 0.234 @ bias=0 rising to 0.338 @ measured 1.4° bias; even the best low-σ_v/zero-bias cell p90 0.190). RMS-clears are p90-fails. | **HIGH** | p90 selection gate (§0.3) + uncertainty-aware speed-down + speed-ladder fallback (slower rung). **DECISIVE resolver = L3** (the offline answer is "cannot settle"). |
| 2 | **Cold case-C velocity prior is unpinned and HIGH-risk** (§0.5): variance/window-driven, not init-driven; "warm by gate-4" does NOT hold. The dominant swing is the **effective accel/attitude bias** (1.4° measured → 0.24 m/s² → cold p90 0.338, 65% contact). `dr_force_bias_max=3.0` is a FORCE disturbance, NOT an IMU bias — no measured accel-bias number exists. | **HIGH** | Attitude pipeline / ESKF-bias-state is in the critical path for the MARGIN (not just absolute nav); keep effective accel bias ≤~0.1 m/s² (attitude ≤~0.6°). L3 open-loop KF-on-recorded-IMU-vs-GT pins the bias slope. |
| 3 | **In-loop vision latency L on EVAL HW is UNMEASURED.** CPU-class (~125 ms) → v*L staleness 2.3–4.2 m at VQ2 speed → speed thesis flips to NO-GO; horizon sizing hinges on L<~0.45 s. | **HIGH** | One eval-HW timing run (L4). Predict-forward (constant age) is the TIMESYNC-free fallback for the cheap 80%. |
| 4 | **TIMESYNC `delta_epoch` cannot be checked offline.** Full RewindKF indexed by it; if unreconciled the C2 chain is INVALID by construction (not degraded). | **HIGH** | Predict-forward sidesteps it (ship first); full rewind HARD-blocked until a live wire trace (L4). |
| 5 | **Per-fix σ at 37 m/s is UNMEASURED** (pool ≤8.4 m/s; 1.0×→2.0× motion-blur MODELED). Every σ (0.265, the 0.139 RMS, the whole ladder ceiling) is a best-case LOWER bound; `c_cal=1.0` is a placeholder. | **HIGH** | L3 at-speed gate-4 recording is the sole resolver. |
| 6 | **Velocity channel is NOT the cheap lever d4v claimed** (§0.4): honest σ_v ≈ 2.81 m/s + L=115 ms → cold 0.19–0.35 m NO-GO. If green-lit as load-bearing it does nothing at the operating point. | **MED** | DEMOTED to deferred P2 insurance. Margin closure rests on bias control + speed-down, not this. |
| 7 | **Residual one-signed PnP/extrinsic in-plane bias** (≤0.19 m, sign unmeasured) that gate-relative does NOT remove. A real +0.19 m one-signed → p90 0.337 (OVER). | **MED** | DR now injects U[0,0.19] m with per-episode constant random sign (§2.2c). SHADOWPC-VISION-CAL resolves the true magnitude+sign. |
| 8 | **Confidence-channel calibration** — gate-relative P is mildly OVER-confident (in-plane NEES 1.96 < χ²(2), §0.6). An over-confident channel teaches the policy to trust a lie. | **MED** | NEES→χ²(3) `c_cal` trim fit on L3; §2.2 unit test `mean(e_g²/diag(P_hat)) ∈ [0.8,1.3]` is the gate. |
| 9 | **Narrow inc8 basin** (inc7 2/3 viable; this is harder). A retrain <2/3 S_stable must not ship. | **MED** | ≥5 seeds + S_stable gate + fixed-cone fallback if <2/5 viable at L0. |
| 10 | **Detector 4→2 corner clip earlier than modeled 4 m at real speed/blur** → 4-corner accepted-fix rate drops, warm σ degrades. | **MED** | L3 corner-count-vs-range recording. |
| 11 | **Obs-dim 17→20 silently breaks a length-17 consumer** (replay_obs, debug dumps). `frame_residual_report.py` verified safe. | **LOW** | G1 + G7 backstop; audit tooling. |

---

## 5. BUILD SEQUENCE (dependency-ordered)

```
P0 bug fixes ──► C2 estimator ──► C1 obs ──► C5 inc8 retrain ──► C6 gauntlet ──► C6 live
       │                              ▲                ▲
       └──► C3 margin sim ────────────┘      C4 corrected line
            (parallel feeder)                (parallel feeder)
```

Critical path: **P0 → C2 → C1 → C5 → C6-gauntlet → C6-live.** C3 (margin sim) and C4 (corrected line) are
OFF the critical path — parallel feeders into C5 selection. The longest pole is **C5 (≥5-seed retrain,
narrow basin)** — everything upstream exists to de-risk it before it starts. (See `build_sequence` in the
structured return for the per-step depends-on/effort table.)

---

## 6. THE ESCAPE HATCH — exact live data that resolves what cannot be settled offline

The headline finding (margin DOES-NOT-CLOSE offline) is resolved ONLY by the **ShadowPC at-speed (~37 m/s)
gate-4 vision recording (L3)**, time-synced on ONE clock (the TIMESYNC P0 prereq):
- **Trajectory:** full g0→g4 lap (or ≥ a g2→g3→g4 segment reaching ~37 m/s on the g3→g4 straight), flown
  by the current best policy, **≥5 laps** (a p90/worst-case distribution, not a point estimate).
- **Fields (one clock):** per-frame (≥30 Hz) 4-corner pixel detections + PnP `t_cam_gate` + reproj error;
  `HIGHRES_IMU accel_body` (≥90 Hz); given ATTITUDE quaternion; **ground-truth position + velocity** at
  IMU rate. ~60–100 s.
- **What it pins:** (a) the true accel-bias/attitude-error magnitude entering g4 (KF open-loop on recorded
  IMU+attitude vs GT velocity → drift slope = effective bias); (b) the realized COLD velocity-prior error
  distribution entering the g4 window; (c) the achievable σ_v from real position-fix differencing; (d) the
  per-fix gate-relative lateral σ at real speed/blur (confirm or break the 0.265 m model); (e) 4-corner
  coverage vs range; (f) the residual one-signed PnP bias magnitude AND sign. Plug (a)+(b)+(d) into the §4
  / `v3b` tables → the p90/p99 margin verdict becomes MEASURED, not ASSUMED.
- **Also needed (cheap, separate):** one **eval-HW timing run** (L4) to measure L and confirm horizon > L,
  and a **live wire trace** to confirm `delta_epoch` (TIMESYNC).

---

## Confidence ledger
- **MEASURED (re-run this session):** rel margin RMS 0.139 / p90 0.203, E/D_bias −0.000/−0.004, abs 0.279
  / submap 0.228 (`c1_gate_relative.py`); the +L sign identity 4.8e-7 vs −L 24.0 and the d1-check-(2)
  tautology (`v_obs_adversarial_check.py`); every `v3b` cell `p90_clear=false` (best 0.126/0.190/24%);
  cold p90 0.234 @ bias0 / 0.338 @ 1.4° (`v3_margin_refute`); velchannel honest σ_v 2.81 m/s, L=115 ms
  cold 0.19–0.35 NO-GO (`verify_velchannel_v6`); `obs_dim=17` source, `tilt_free_rad` scalar,
  `contact_true_eval` on `obs_from_truth` (source reads); 20-dim append leaves [0:17] at 0.0.
- **EXTRAPOLATED:** per-fix σ at 37 m/s (blur 1.0→2.0 MODELED, best-case lower bound); `c_cal=1.0`; L on
  eval HW; cold velocity prior entering g4.
- **ASSUMED:** const-v 37 m/s cruise, 90 Hz IMU, 47% acceptance / ~14 Hz fix rate, residual one-signed
  PnP bias magnitude (≤0.19 m) and sign.
