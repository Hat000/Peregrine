# VQ2 A26 — vertical brake fix (de-clip washout + gate-offset-rate fusion) — BUILD DELTAS

Follows A25 (gate-relative ẑ_off altitude term, commit 71e8939). A25 flew (run 20260702_221235,
TRT engine, fresh pose) and WORKED directionally — the ẑ_off term reversed the climb and sought gate
height (no ceiling-lock, operator-confirmed). But the drone OVERSHOT: a smooth, level descent
(operator's eyes; the logged −47° pitch is a DIVERGED-AHRS artifact under 225 g collision spikes,
NOT real) that sailed past gate-1 height to the floor. Root cause (Fable, off the trustworthy WIRE
signals z_off/vz_t/vert_vz_est/thrust — NOT the untrustworthy integrated trajectory):

**The washout `vert_vz_est` is not a brakeable velocity signal.** (1) It RAILS to its ±3 clip (pinned
18–28% of ticks) → the damping term `0.25·(vz−vz_t)` swings ±0.52 collective = **5× the authority of
the A25 position term** `−0.06·z_off` (±0.065) → A25 is correct-but-outgunned, thrust bang-bangs the
0.05/0.60 clamps 48% of ticks. (2) It is BLIND to real descent — during genuine sinking it reads
**~0** → zero braking → the smooth descent never arrests → overshoot to the floor. A25 itself is
validated correct in flight: `corr(offset_z_world, z_off)=+0.80`, right sign, right size (ω_n≈1.49).

## FIX 1 (ship first, one line) — de-saturate the damper
`src/racer/vertical_estimator.py`: `export_clip_mps: 3.0 → 1.5`. At ±3 the damping term reaches ±1.0
collective (4× hover) → guaranteed clamp-slam; at ±1.5 the worst-case swing is `0.25·(1.5−(−1))=±0.625`
— still strong, no longer dwarfing the position term. De-saturates regardless of oscillation-vs-one-off.

## FIX 2 (the real fix) — gate-offset-rate fusion (implement §6 of the A25 spec, previously DEFERRED)
Ref: `handoff/vq2_gate_relative_altitude_spec_2026-07-02.md` §6 (the v2 hook, designed but not built in
A25). It was deferred on the rationale "the position term seeks, so vz accuracy doesn't matter for DC" —
the overshoot PROVES that wrong: **vz accuracy matters for BRAKING.** Build it now, default ON for
vq2_case_c. Mechanism (fuse a REAL gate-relative descent velocity into the washout so it stops reading
~0 on descent):
- Track the previous FRESH-pose `offset_z_world` + its timestamp inside the estimator (the latch path,
  `latch_offset()`, already receives offset_z_world + obs_age each fresh pose).
- On a fresh pose whose previous fresh pose is `Δt ∈ (0.05, 0.8] s` old: `vz_gate = (offset_z_now −
  offset_z_prev)/Δt` — this is the drone's TRUE vertical velocity relative to the (static) gate, NED
  down-positive (gate moving DOWN in frame as the drone climbs, etc. — VERIFY the sign against the
  z_off/vz conventions and PIN it with a test: drone descending toward gate ⇒ vz_gate > 0). Reject if
  `|vz_gate| > 2.5` m/s or the point-blank guard (<2 m range) is active.
- Blend into the washout state: `vz ← vz + k_g·(vz_gate − vz)`, `k_g = 0.15`. At ~118 ms (8–9 Hz TRT
  cadence) this is an effective ~0.6 rad/s correction — a complementary crossover just above the
  washout leak. No gate visible ⇒ pure washout, no mode switch.
- Add a flag `use_gate_vz_fusion` (default OFF on the Controller/estimator for byte-identical VQ1;
  ON in vq2_case_c) OR gate it on the same use_vertical_estimator path — keep VQ1/case-A byte-identical.

## FIX 3 — DO NOT do yet
Do NOT raise `ff_vertical_kd_alt` (hold 0.25). Raising damping gain on a signal that rails/reads-0 just
amplifies the problem. Revisit only AFTER 1+2 make vert_vz_est trustworthy (then ~0.30–0.35 is safe).

## Housekeeping (do)
- **Fix the `pose_age_s = 0.0` logging bug** — it logs 0 every tick; wire the real seeker pose age
  (fly_rl's own async-detect obs age was 118 ms) into the nav_estimate record so we can confirm freshness.
- **Add a per-tick `contact_frozen` bool** to nav_estimate.jsonl (from VerticalEstimator.contact_frozen())
  so we can verify the |f|-spike contact-gate actually fired on the 458/2211 m/s² impacts (Fable could
  not confirm in-loop firing from the current log).

## Tests + validation
- Unit tests for the fusion: vz_gate sign (descending-toward-gate ⇒ vz_gate>0, PINNED), Δt window
  reject, |vz_gate|>2.5 reject, blend math, no-gate ⇒ pure washout, VQ1 byte-identical (flag OFF).
- Full vertical/controller/navigator/deploy_profile/gate_seeker suites green; VQ1/case-A byte-identical.
- COMMANDER offline-validates FIX 1 separately: replay run 221235's logged vz_t/z_off through the clipped
  law at export_clip 1.5 and confirm reconstructed pre-clip thrust stops hitting the 0.05/0.60 rails 48%.
- Commit on vq2-gate2-turn-dive: `feat(vq2): vertical brake — de-clip washout + gate-offset-rate fusion (A26)`.
