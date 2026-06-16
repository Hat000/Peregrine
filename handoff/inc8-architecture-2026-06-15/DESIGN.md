# inc8 architecture pivot — DESIGN (for review BEFORE any GPU)

**Status:** DRAFT for Fengyou's review. No code written, no GPU spent. Supersedes weight-tuning
(closed after 4 NO-GOs: window → entropy → shape → incentive).

## 1. Why the four weight iterations failed (the diagnosis we're building on)
The inc8 thesis was *learn camera-pointing from a reward*. It bracketed the whole weight space:
- low-pass terminal_weight → exploits the easy terminal <5 m zone (points point-blank, 0 fixes)
- band-pass terminal_weight → abandons pointing (perc 0.5 ≈ 5% of progress 10; not worth the line)
- fix-driven (conf_shape 0.3 + fix_bonus 1.5) → points NOWHERE: conf_shape is a ~constant −0.29/step
  penalty (no incremental gradient), fix_bonus never fires (fix_rate≈0, sparse binary).

Root cause is **not** a weight: **pointing→fix is a hard, all-or-nothing behaviour, and PPO cannot
discover it from a non-pointing start** (the geometric-pointing proxy is gameable; the honest fix
signal is sparse). This is a sparse-reward / exploration / credit-assignment wall.

**The fix is architectural: stop hoping PPO discovers pointing — ENGINEER the pointing (a primitive),
and reward the OUTCOME densely (true centering), not a proxy.** The primitive breaks the exploration
wall; the dense reward gives the primitive's strength an honest, smooth objective to be tuned against.

## 2. The two pieces

### 2A. LOOK-AT PRIMITIVE — *the load-bearing change* (active perception in the action space)
The camera-frame angular errors to the gate are **already computed every step** — `geom["t_cam"]`
(gate centre in camera frame) → α = atan2(X,Z) (azimuth), β = atan2(Y,√(X²+Z²)) (elevation), exactly
as `inc8_reward.visibility_2axis` does. The primitive drives α→0 and β→0 by adding a proportional
body-rate correction to the policy's CTBR command:

```
ω_cmd = ω_policy + g · Δω_lookat,     Δω_lookat = K · R_body_from_camera · [ -β ; -α ]  (pitch,yaw)
action_executed = [ collective_policy , ω_cmd ]      # collective & the racing line stay the policy's
```

- **Intercept point:** top of `PeregrineRacingInc8.step()`, before `self.dynamics.step(action)`
  (line 206). Compose, then use the *composed* action for `dynamics.step` AND for `a_norm`
  (so R5 dact / R6 rate penalties tax the realised command; BSR3 already gates the realised rate).
  From PPO's view the primitive is just part of the transition (residual-action shaping — standard).
- **Azimuth (yaw) = cheap.** A quad's heading is a redundant DoF: yawing to point the camera
  horizontally at the gate barely perturbs the thrust/translation. Apply with **high authority**.
- **Elevation (pitch) = costly.** The 20° fixed mount + steep race tilt (~55° @ 30 m/s) is why the
  gate sits below the VFoV ~90% of the time at fixable range (the known CONTROL CAVEAT). Pointing
  vertically means briefly *flattening* (reducing tilt) in the band — a real racing-line cost. This
  is the genuine inc8 trade; it is the axis the gain exists to modulate.
- **Range-gated:** primitive active only for range ∈ ≈[r_lo−δ, r_hi] (the approach/fix band ~[10,30] m);
  zero outside, so it never fights the line during transitions or far approach.
- **The gain g:**
  - **Phase 1 — FIXED schedule** (e.g. g_yaw=1 in band, g_pitch small/0). **No new action channel,
    smallest possible change**, action_dim unchanged. Tests the cheapest hypothesis directly.
  - **Phase 2 — LEARNED gain**: add g (or [g_yaw, g_pitch]) as a 5th/6th action channel (changes
    action_dim → policy head) so the policy learns *when/how hard* to dwell-and-look vs race. Only if
    a fixed schedule can't trade speed-vs-fix-rate well.

### 2B. DENSE TERMINAL-σ_p0 CENTERING REWARD — *the honest objective*
σ_p0 (the terminal lateral miss, the single binding number for gate closure) ≈ the **estimator
in-plane error at crossing** (the drone tracks Γ-through-centre on its KF estimate, so
miss = truth − KF, in-plane). That error, `err_ip`, is **already computed every step**
(`self._emu.gate_frame_error_inplane`, line 227). The existing `gt_estimerr_anchor` penalises it
FLAT (and is "coast-satisfied" — a decent prior keeps it moderate everywhere with no extra push to
get it *tight near the gate*). The new term is a **range-weighted** version that ramps the penalty up
as range→0:

```
r_centering = − rw_centering · w_near(range) · err_ip,     w_near(r) = sigmoid((r_near − r)/w)  (≈1 at crossing, ≈0 far)
reward = ... + r_centering          # NEW additive term in inc8_reward, default rw_centering = 0 (byte-identical)
```

- **Dense + GT-anchored** (not a proxy): smooth in the error, hard to game (it reads truth).
- **Targets σ_p0 directly** and is **not coast-satisfiable without pointing**: near the gate the
  policy has dead-reckoned since the last fix (≥~12 m, the PnP floor), so low near-gate err_ip
  *requires* an accurate fix in the band → requires pointing. Coast-drift (<0.02 m / 0.4 s) is the
  cheap part; the fix is the lever.
- **Why it succeeds where conf_shape/fix_bonus failed:** by itself it has the *same* cold-start
  problem — which is exactly why it is paired with 2A. The primitive makes pointing HAPPEN; the dense
  reward makes the policy KEEP doing it (and, in Phase 2, set the gain). Neither piece alone is enough.

## 3. Staged plan (start simplest; escalate only on evidence). Each stage gated behind a cfg flag,
default OFF == byte-identical inc7.

| Stage | Change | Hypothesis under test | GO signal |
|-------|--------|----------------------|-----------|
| **S0** | FIXED **yaw-only** look-at primitive + existing rewards | cheap free-ish yaw pointing alone lifts band fix-rate | inc8_lockband_pointing ↑, inc8_fix_rate off the ~0.001 floor, racing line intact (finish_time, success_rate not wrecked) |
| **S1** | + dense terminal-σ_p0 reward (range-weighted err_ip) | rewarding the honest outcome tightens near-gate err_ip | inc8_estim_err_inplane_m near gate ↓ toward σ_p0 ≲ 0.08 m; pass_offset_m p99 ↓ |
| **S2** | + FIXED **pitch** (elevation) look-at component | flattening in-band closes the VFoV/elevation gap | vertical fix density ↑ (the σ_vert axis), σ_p0 holds with the line cost acceptable |
| **S3** | **LEARNED gain** (5th/6th action channel) | adaptive speed-vs-fix trade beats any fixed schedule | better σ_p0 *at* lower finish_time than S2's fixed schedule |

Stop at the earliest stage that hits σ_p0 ≲ 0.08 m without destroying the line. Most of the value
may land at S0–S1.

## 4. Code touch-points (all training-only; never touches the deployed policy/estimator)
- `rl/peregrine_racing_inc8.py` `step()`: action interception + compose (S0), new reward term wire-in
  (S1), action_dim bump + gain decode (S3). New diagnostics: in-band yaw/pitch error, gain value.
- `rl/inc8_reward.py`: `lookat_correction(t_cam, R_cam_from_body, gains)` (pure, unit-testable) +
  `centering_reward(err_ip, range, rw_centering, r_near, w)` + weights `rw_centering`, `lookat_*`.
- `frames.py`: reuse `R_camera_from_body` for the camera→body rate transform (the boresight infra
  already there). The exact angular-error→body-rate map verified by a unit test (camera α,β →
  Δω reduces α,β) + a tiny sim check (primitive ON drives the gate to frame centre on a static approach).
- `cfg`: `use_lookat_primitive`, `lookat_g_yaw`, `lookat_g_pitch`, `lookat_r_lo/r_hi`, `rw_centering`,
  `rw_centering_r_near` — all default to the OFF/zero values.

## 5. Risks & mitigations
- **Look-at/racing coupling** (the main risk): yaw is near-free, pitch is the costly axis. Mitigated
  by range-gating + keeping pitch authority low/learned. If a fixed schedule fights the line → S3
  learned gain, or fall back to a **pointing curriculum** (easy gate-in-frame spawns, harden) for the
  cold-start instead of the primitive.
- **Gameability:** the dense reward is GT-anchored (reads truth, not a proxy) and the primitive is
  deterministic — neither is gameable the way the proxy/confidence terms were.
- **"Engineered, not learned"** (vs the original inc8 thesis): we still *learn* the racing line and
  (Phase 2) the gain; we only *engineer* the geometric pointing direction, which is closed-form from
  the map + estimate. This is standard perception-aware planning, and it transfers to VQ2 (any gate
  geometry given a rough map) — arguably MORE robust than a learned-from-scratch pointing policy.
- **VQ2 calibration:** the architecture is general; only the tunables (band edges, gains, r_near)
  depend on the real perception model. Build + smoke on VQ1 now; final-tune when the VQ2 sim drops
  (~2026-06-29).

## 6. What I want signed off before implementation
1. **The staged plan** (start with S0 = fixed yaw primitive; escalate on evidence).
2. **Phase-1 fixed gain vs jumping to a learned gain** (S3). Rec: fixed first — it's the smallest
   change and isolates the "does forced pointing help" question before touching the policy head.
3. **The "engineered pointing" deviation** from the pure learned-pointing thesis (rec: accept it —
   4 NO-GOs say learned-from-scratch pointing is the wrong bet; active-perception is the standard fix).
