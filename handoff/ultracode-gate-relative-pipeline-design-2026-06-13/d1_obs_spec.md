# d1 — GATE-RELATIVE OBS FORMULATION (build-ready spec)

**Component 1 of the gate-relative case-C VQ2 pipeline.** Defines exactly what the policy
observes for the upcoming gate, the case-A/B-vs-C **unification** (one policy, case-agnostic),
the **uncertainty-aware confidence channel** (commander-endorsed, highest-leverage), and the
**faithfulness** proof that VQ1 / case-A/B is preserved bit-for-bit.

Every load-bearing claim here is backed by a reproducible offline check:
`handoff/ultracode-gate-relative-pipeline-design-2026-06-13/d1_obs_faithfulness_check.py`
(**ALL PASS**, exit 0; run `PYTHONPATH=src .venv/Scripts/python.exe …/d1_obs_faithfulness_check.py`).
Source of truth read in full: `rl/fly_rl.py` (obs builder + P4-C05 GateMap), `rl/peregrine_racing.py`
`get_observations` / `world_to_gateframe` / `rel_tables` (the bit-exact train-side definition),
`src/racer/localization.py` (`gate_pose_to_world_position` — the lever arithmetic), and
`handoff/ultracode-estimator-racespeed-2026-06-13/c1_gate_relative.py` + REPORT §2–3 (the rel-arm fix).

MEASURED / EXTRAPOLATED / ASSUMED are tagged inline.

---

## 0. The existing 17-dim obs (the frozen contract — do not change the first 17)

Train-side `peregrine_racing.PeregrineRacing.get_observations` (lines 563–599), mirrored torch-free
in `fly_rl.obs_from_zup` (lines 307–357). `tg = target_gate`, `R_w2g = get_gate_rotmat_w2g(gate_yaw[tg])`
(rows `[c,s,0; -s,c,0; 0,0,1]`):

| idx | name | definition | frame |
|---|---|---|---|
| 0:3 | `pos_g` | `R_w2g @ (gate_pos[tg] − p)` | gate (x=down-course, y=lateral, z=world-up) |
| 3:6 | `vel_g` | `R_w2g @ v` | gate |
| 6:9 | `rpy_g` | ZYX-euler of `R_w2g @ R_b2w` | gate-relative attitude |
| 9:12 | `w` | body rates | FLU body |
| 12 | `prev_normed_thrust` | last RESCALED `act[0]` in `[0, act_max]` g | — |
| 13:16 | `next_relpos` | `gate_rel_pos[nxt]` (precomputed `rel_tables`) | gate (tg)'s frame |
| 16 | `next_relyaw` | `gate_yaw_rel[nxt]` | — |

**Already gate-relative.** The whole obs is translation-invariant in world: it depends on the drone's
pose *only through its offset to the target gate* (`gate_pos[tg] − p`) and gate-frame rotations. This is
the structural fact the entire pipeline exploits — **the policy never sees an absolute world position**,
so the gate-relative estimator and the case-A/B given-pose path can feed the *same* slot.

The P4-C05 foundation (`fly_rl.GateMap` / `make_gate_map` / `obs_from_zup(gate_map=)`) already threads a
**runtime per-gate yaw** through this builder, reproducing `get_observations` exactly; `gate_map=None` is
the bit-exact yaw=π VQ1 specialization. **This spec changes only how `pos_g`/`vel_g` are *sourced* in
case C, and appends a confidence channel — it does NOT touch the layout of dims 0–16.**

---

## (1) What the policy observes for the upcoming gate in case C

### 1.1 The estimator delivers a gate-relative offset for the SEEN gate (−L lever, map bias dropped)

The localization fix (`localization.gate_pose_to_world_position`, lines 85–87) is, in world NED:

```
L      = R_world_camera @ t_cam_gate          # lever: gate origin relative to drone (= gate_true − p)
p_abs  = gate.position_ned − L                # the ABSOLUTE fix (re-injects gate.position_ned = MAP centre)
```

`t_cam_gate` is the PnP translation to the **physically seen** gate corners; `L` is therefore the offset
from the drone to the **true seen opening**, carrying PnP + attitude-lever noise but **NOT** the
map/registration bias `db` (the map centre never enters `L`). The gate-relative estimator delivers
**`L` (or equivalently `−L`, the drone→gate offset)** directly to the obs builder and **drops the
`gate.position_ned − L` step**.

### 1.2 Mapping onto the 17-dim layout via `R_w2g(gate_yaw[tg])`

The obs slot is `pos_g = R_w2g @ (gate_pos − p)`. Substitute the seen-gate offset:

```
gate_pos − p  ==  L_seen   (world Z-up; gate_pos is the seen opening, p = gate_true − L_seen)
=> pos_g = R_w2g @ L_seen
```

**This is exact, not approximate** (verified: check (2), `max|pos_g_obs − R_w2g@L_seen| = 0.0` over 2000
random states). The estimator hands the obs builder the world-frame seen-gate offset `L_seen`; the builder
applies the **same `R_w2g = get_gate_rotmat_w2g(gate_yaw[tg])`** it already uses (the P4-C05 yaw-aware
path, `obs_from_zup(gate_map=make_gate_map(pos, yaw))`). No new rotation, no new frame convention.

- `vel_g = R_w2g @ v` is unchanged. **Velocity stays IMU/KF-sourced** (vision is position-only — case-C
  velocity is unobservable except through position-fix differencing; REPORT §2 refinement). This is the
  swing variable behind the CONDITIONAL-GO and is addressed by the confidence channel + the warm-prior
  full-lap follow-up, not by the obs *form*.
- `rpy_g`, `w`, `prev_normed_thrust`, `next_relpos`, `next_relyaw` are **identical** to get_observations.

### 1.3 How `pos_g` is materialized in practice (RewindKF-augmented, not a parallel filter)

Per REPORT §2 condition 3, **augment, do not replace** the absolute `LinearKF`:

1. The absolute KF + RewindKF (`kf_rewind_buffer.RewindKF`) runs as today (planning/feed-forward/g4→g5
   handoff). Its position estimate `p_KF` and covariance `P` are produced exactly as now.
2. For the obs, build the seen-gate offset from the **fix-side lever**, not from `gate_map − p_KF`:
   `L_seen = R_world_camera @ t_cam_gate` at the most recent accepted fix, **propagated to now by the IMU**
   (`L_seen(t_now) = L_seen(t_fix) − ∫v dt` over the rewind horizon — the same propagation RewindKF
   already does on the state). `pos_g = R_w2g @ L_seen(t_now)`.
3. When **no fresh fix is in-band** (the drone is between fixes or the gate has clipped <4 corners),
   fall back to `pos_g = R_w2g @ (gate_pos − p_KF)` — i.e. the absolute KF *is* the high-uncertainty
   continuation of the same quantity, and the confidence channel (§3) tells the policy which regime it is in.

This is the IBVS/relative-centering term REPORT §3 rank-1 calls for. The **+1 relative-innovation outlier
gate** (REPORT §2 condition 2 — reproj alone does not separate depth-flips: flip p50 0.71 px < clean 1.02 px)
belongs to the detector→PnP component, not the obs; this spec assumes accepted fixes are already relative-
innovation-gated.

---

## (2) THE UNIFICATION — one policy, case-agnostic

**Claim.** A single policy serves case A/B (given/KF pose, confidence ≈ certain) and case C (vision-
estimated seen-gate offset, variable confidence) **without the policy knowing which case it is in**,
because the estimator presents a *uniform gate-relative offset + a confidence*, and case A/B is just the
high-confidence limit of the same channel.

**Mechanism (the obs slot is identical in form):**

```
pos_g  =  R_w2g @ offset_to_gate
   case A/B :  offset_to_gate = gate_map_centre − p_given     (p_given is pristine; db ≈ 0)
   case C   :  offset_to_gate = L_seen                        (PnP lever to the SEEN opening)
```

In **both** cases the policy receives one vector `pos_g` in the gate frame; it cannot and need not
distinguish the source. Case A/B is the limit where (i) the offset is exact (no PnP noise, no map bias)
and (ii) the confidence channel (§3) reads ≈0 uncertainty. **Verified:** check (2) shows
`pos_g(given pose) == R_w2g @ L_seen` to **0.0** (bit-for-bit) when the policy's `gate_pos` is the seen
opening — the two cases are the *same tensor*, not merely similar.

**This is NOT the "subtract gate_map" anti-pattern.** The anti-pattern forms the gate-relative offset as
`gate_map_centre − p_KF_absolute`, which re-injects the per-track map bias `db`:

```
submap offset = gate_map − p = (gate_true + db) − p = L_true + db      # db SURVIVES
seen   offset = L_seen       = L_true                                  # db DROPS OUT
```

**Verified (check (3), real `localization.gate_pose_to_world_position` arithmetic):** with a per-track
`db=[0.18, 0.06, −0.10] m`,
- absolute / subtract-map path error `= +db` exactly (‖·‖ = **0.2145 m**) — bias re-injected;
- seen-gate lever − true lever `= 0.000e+00` — **db removed exactly**.

This is the same `db`-drops-out arithmetic the estimator-racespeed REPORT §2/§3 and `c1_gate_relative.py`
part (b) prove (rel arm E_bias `−0.000 m` vs abs/submap `+0.176/+0.174 m`; reproduced this run). The case
A/B↔C unification is therefore *exactly* the gate-relative fix, generalized: **always feed the seen-gate
offset; in case A/B the "seen-gate offset" is just the given pose's offset and the map bias is already ~0.**

**Sourcing summary (the contract the estimator must honor):**

| | source of `pos_g` offset | source of `vel_g` | confidence (§3) |
|---|---|---|---|
| case A/B | `gate_map − p_given` (pristine; ≡ seen offset, db≈0) | given/KF `v` | ≈ certain (σ→0) |
| case C, fresh fix | `L_seen` (PnP lever, IMU-propagated) | KF `v` (IMU + fix-differencing) | KF `P` in gate frame |
| case C, coasting | `gate_map − p_KF` (absolute KF continuation) | KF `v` | KF `P` (inflated by coast) |

---

## (3) THE UNCERTAINTY-AWARE OBS CHANNEL (the new dimensions)

**Goal (commander-endorsed, highest leverage).** Feed the KF **per-tick position covariance** into the
policy so it can be *fast-when-confident / careful-when-uncertain* — directly attacking the speed↔validity
coupling that makes case C CONDITIONAL-GO. This is **NOT** the rejected logstd-as-uncertainty (a policy
output): it is a **measured filter covariance** (an obs *input*), derived from the real `RewindKF.P`.

### 3.1 The scalar(s)

Project the KF 3×3 position covariance `P_pos = P[:3,:3]` into the **gate frame** with the SAME `R_w2g`
the obs already uses, then expose the **in-plane** and **along-track** 1-σ (in metres — the policy's native
length unit, so it is directly comparable to `pos_g`):

```
P_g            = R_w2g @ P_pos @ R_w2g.T           # gate-frame position covariance (3x3)
sigma_inplane  = sqrt( P_g[1,1] + P_g[2,2] )       # E(lateral)+D(vertical) — the BINDING gate-4 axes
sigma_along    = sqrt( P_g[0,0] )                  # along-track (N) — affects crossing timing, not the miss
```

`sigma_inplane` is the load-bearing scalar: it is the 1-σ of the exact quantity the gate-4 margin guards
(REPORT/FACTS: gate-4 in-plane = gate-frame y,z). **k = 2** (in-plane + along-track) is the recommended
channel; a **k = 1** ablation (in-plane only) is the cheap fallback if the along-track scalar proves inert.

**Verified well-formed (check (4), real RewindKF P over the g3→g4 window):** `sigma_inplane` is finite,
non-negative, **velocity-prior-sensitive** (0.311 m cold-prior `vel_std=5` → 0.282 m warm `vel_std=0.5` —
it *moves with the swing variable*, exactly the signal we want the policy to see), and **collapses to
0.014 m in the case-A/B high-confidence limit** (`pos_std=0.01`). So "certain" in case A/B reads as ≈0
uncertainty, and the same channel carries real information in case C.

### 3.2 Calibration ("confident" must mean actually-accurate)

The raw KF `P` is only honest if its NEES tracks χ²(3). From the REPORT (§1 filter-honesty): the deployed
**rewind + de-biased gate-relative** path is consistent-to-mildly-overconfident (NEES 3.3–4.3 vs χ²(3)=3.0);
the RAW absolute arm is overconfident (4–7.5, because P does not model the per-fix bias). Two calibration
requirements, both MEASURED-grounded:

1. **Use the gate-relative R, not the absolute R.** The gate-relative fix has **no 0.40 m bias-absorption
   floor** (the floor existed to absorb the map bias that gate-relative removes — `c1_gate_relative.py`
   part (b) uses `cov = diag([σ_abs², σ_lat², σ_lat²])` with **no** `FIX_COV_FLOOR_STD`). Per-fix lateral
   per-axis σ ≈ **0.265 m** (MEASURED, accepted maha≤16.27 4-corner near band; reproduced this run). With
   the bias removed and the tighter R, NEES sits at the consistent end → `sigma_inplane` is a *calibrated*
   accuracy proxy, not a damping artifact.
2. **One calibration scalar `c_cal` (a multiplicative trim on `sigma_inplane` before it enters the obs),
   fit so empirical NEES → 3.0** on the at-speed recording (SHADOWPC-VISION-CAL). Until that recording
   exists `c_cal = 1.0` (EXTRAPOLATED: the 37 m/s motion-blur multiplier ×1.0→×2.0 is MODELED, so the
   channel is a best-case lower bound on uncertainty, same caveat as every REPORT σ).

**Anti-gaming (the reward must not let the policy buy a low σ by slowing down).** σ is an *input*, not a
reward term; the speed-preserving reward (inc8: arc-length progress, NEVER reward damping — handoff inc8
contract) means the policy is rewarded for *fast valid* flight, and the σ channel only lets it *modulate*
aggressiveness with measured confidence. **The DR must exercise variable confidence** (next subsection) so
the policy cannot assume σ is constant.

### 3.3 Exact obs index layout change (17 → 17+k), critic, and DR

```
NEW obs (k=2):   [ 0:17 ]  unchanged (frozen contract, bit-identical — see (4) below)
                 [ 17  ]   sigma_inplane   (gate-frame in-plane 1-sigma, m, calibrated)
                 [ 18  ]   sigma_along     (gate-frame along-track 1-sigma, m, calibrated)
```

- **Builder:** extend `obs_from_zup` / `build_obs` with an optional `pos_cov` (3×3 NED) arg; when provided,
  append `[sigma_inplane, sigma_along]` computed with the SAME `R_w2g`. `OBS_LABELS` gains
  `["sigma_inplane","sigma_along"]`. Network input dim `_ActorMean` 17→19 (and the critic, below).
- **Critic (`get_state`, line 602):** the asymmetric critic already sees a privileged 3-gate window. Append
  the **same** two calibrated scalars (or, since the critic is privileged, the *true* gate-frame position
  error magnitude as a value-shaping aid). Recommended minimal change: append `[sigma_inplane, sigma_along]`
  to the critic input too, so value and policy share the uncertainty signal. (Both grow by k; keep them equal
  unless an ablation shows the privileged-true-error critic helps.)
- **Training DR (MEASURED-error DR, REPORT doctrine — train on measured error classes, not arbitrary
  noise):** at train time, generate `pos_g` by corrupting the privileged true gate offset with the
  **measured gate-relative per-fix lateral noise** (per-axis σ ≈ 0.265 m near band, the c1 pool) at the
  **measured cadence/acceptance** (~14 Hz, 47%), run it through a sim `LinearKF`+`RewindKF`, and feed the
  filter's actual `P` into the channel. Sweep the velocity prior cold↔warm (the swing variable) and the
  fix availability (coast windows) so the policy sees the **full σ range** it will meet in case C. In the
  **case-A/B fraction** of training (the high-confidence limit), feed pristine offset + σ≈0 — this is what
  preserves case-A/B behavior while teaching the policy that low σ ⇒ safe-to-push. Select on
  **≥5-seed generalization** (inc8 narrow-basin lesson), `rl/contact_true_eval.py` gate-4 0.155 m guard.

### 3.4 Back-compat (current 17-dim checkpoints + fly_rl deploy)

- The channel is a **pure append**: dims 0–16 are byte-for-byte unchanged (check (1) and (4)). A current
  17-dim inc7 checkpoint is unaffected by *the existence* of the design; it simply isn't fed dims 17–18.
- **Deploy switch in `fly_rl`:** gate the new dims on the checkpoint's obs-dim (read from the sidecar, like
  `act_max_thrust`). A 17-dim actor → `obs_from_zup(pos_cov=None)` (no append), the existing path. A 19-dim
  actor → pass the live `RewindKF.P`. `submit_rl.py` (the judged path) continues to pin inc7 (17-dim) until
  the gate-relative inc8 (19-dim) is selected and validated; the VQ1 loud-abort guards
  (`_assert_live_course_is_vq1` etc.) are removed only at that rebuild, as CONTEXT specifies.
- No retrain of inc7 is implied; inc7 stays the standing-best until inc8 supersedes it on the speed-ladder.

### 3.5 All-gate look-ahead vs next-gate-only (the cheap ablation)

The obs feeds the policy the **target gate** offset (+ a single precomputed next-gate lookahead, dims
13:16). The critic's `get_state` already uses a **3-gate window** (lines 602–617). **Ablation to run in
inc8:** add the confidence scalar(s) for the **next gate too** (one more `R_w2g(gate_yaw[tg+1])` projection
of `P`), i.e. a 2-gate uncertainty look-ahead, vs **next-gate-only** (the k=2 above, target gate only).
Hypothesis (EXTRAPOLATED): at 37 m/s the reaction window shrinks ~1/v (166 ms @8 → 36 ms @37 — REPORT §3),
so seeing the *next* gate's confidence early may let the policy pre-commit speed before gate-4's fix
stabilizes. Cost is +k dims and one rotation; decide on the speed-ladder.

---

## (4) FAITHFULNESS — where this overlaps `get_observations`, and the bit-exact proof

**Overlap (must be identical):** dims 0–16 in *all* cases. The gate-relative change touches only the
*value fed into* `pos_g`/`vel_g` (the offset source), via the **already-shipped, already-bit-exact** P4-C05
`R_w2g = get_gate_rotmat_w2g(gate_yaw[tg])` path. The confidence channel is appended at 17+, leaving 0–16
untouched.

**Divergence (intended):**
- In **case C**, the *numerical value* of `pos_g`/`vel_g` differs from the absolute-map path (that is the
  entire point — it removes `db`). The *form* (the obs layout, the rotation, the frame) is identical.
- Dims 17–18 are new (the confidence channel). No existing dim moves.

**Demonstrated bit-exact, offline (the check the prompt asks for):**
`d1_obs_faithfulness_check.py` — **ALL PASS**:

1. **BIT-EXACT (0.0 ULP):** `obs_from_zup(gate_map=None)` reproduces the frozen legacy VQ1 builder exactly
   over 4000 random states (all gates, both `virtual_flip` branches) — `max|diff| = 0.000e+00`,
   bit-identical 4000/4000. The yaw-aware `gate_map=make_gate_map(VQ1)` path differs by only
   `2.384e-07` = the `sin(π) ≈ 1.2e-16` float32 epsilon from `get_gate_rotmat_w2g(π)` — **the same term
   train-side `get_observations` carries**, so the yaw-aware path is faithful to the *trained* obs, and
   the `None` path is faithful to the *frozen deploy* obs. (This confirms VQ1 / case-A/B is preserved
   bit-for-bit when confidence is certain.)
2. **UNIFICATION (0.0):** `pos_g(given pose) == R_w2g @ L_seen` to `0.000e+00` over 2000 states.
3. **ANTI-PATTERN (real localization arithmetic):** seen-gate lever drops `db` to `0.000e+00`; absolute and
   subtract-map both carry `+db` (‖·‖ = 0.2145 m for `db=[0.18,0.06,−0.10]`).
4. **CONFIDENCE CHANNEL (real RewindKF P):** `sigma_inplane` finite, non-negative, velocity-prior-sensitive
   (0.311→0.282 m), collapses to 0.014 m in the case-A/B limit; pure append leaves 0–16 untouched.

---

## Confidence ledger (MEASURED / EXTRAPOLATED / ASSUMED)

- **MEASURED (re-derived/re-run this session):** bit-exactness (0.0 ULP) and the sin(π) epsilon; the
  unification identity (0.0); the anti-pattern db-drop (real `localization` arithmetic, 0.0 vs +0.2145 m);
  the rel-arm margin clear (in-plane RMS **0.139 m** < 0.155 m, p90 0.203 m over — `c1_gate_relative.py`
  reproduced); per-fix lateral per-axis σ **0.265 m**; the KF `P` gate-frame projection is finite/calibratable
  and case-A/B→0.
- **EXTRAPOLATED:** the channel's value at 37 m/s (blur ×1.0→×2.0 MODELED; σ is a best-case lower bound);
  `c_cal=1.0` until the at-speed recording; the all-gate-lookahead benefit; warm-vs-cold velocity prior
  entering gate-4 (REPORT swing variable — resolved by a full-lap case-C sim).
- **ASSUMED:** const-v 37 m/s cruise, 90 Hz IMU, 47% acceptance, ~14 Hz fix rate (REPORT §5).

**The binding residual** (inherited from REPORT, unchanged by this spec): the margin-clear is
velocity-prior-sensitive and gated upstream by **organizer Q①** (if VQ2 is case A/B, pose is pristine and
the confidence channel simply reads ≈0 forever — the unification means *the same policy still flies*, which
is the insurance value of building gate-relative regardless, per the organizer-pivot directive).
