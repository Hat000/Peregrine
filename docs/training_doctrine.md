# Peregrine RL training doctrine

Durable, in-repo. Established 2026-06-12 (LAPTOP-TRAINING-DOCTRINE session,
`handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md` — every rule below cites the
measurement that earned it). Amend only with new evidence; record amendments here with dates.

## 0. The one-line doctrine

**Train the fastest possible policy against the most honest possible world; buy robustness with
world-model structure (geometry, DR, diversity) and never with reward damping.**

## 1. Reward composition

The reward is the c16 contract (`rl/peregrine_racing.py` module docstring) and changes only when
a term is shown to price a *measured* physical effect wrongly.

What the reward must NOT contain, and why:
- **No aggression/action damping beyond the c16 corner tax.** S17 (inc6 sweep): blunt ‖Δa‖²
  trades generalization monotonically (gen 0.727→0.571 as dact 1→16); the targeted joint tax
  `|thr−mid|·‖rate‖` decouples them because it prices a measured actuator nonlinearity (the
  mixer rails), not a style preference. Live evidence (2026-06-12 crash forensics): failures use
  ~16% of rate authority — damping aims at a non-problem.
- **No gate-proximity / margin penalty.** Margin is a geometry property (§2). A proximity term
  taxes legal corner efficiency on every gate to fix a mis-modeled contact envelope on two.
- **No style terms for measured-near-optimal styles.** The ~55° crab costs 0.43 m/s² along-track
  drag (3%, ≈0.15 s/lap) vs the trajectory-preserving optimum — a flat basin, not a pathology.
  The tilt envelope (rw_tilt) is the one deliberate style constraint and it is a *speed ladder
  lever* (measured 2.3 s/lap), relaxed stepwise with live verification, not a robustness device.
- **Terminal contact penalties stay terminal and big** (T1/T2/T3 sizing rationale in the env
  docstring; gate contact = invalid run by rules).

## 2. Contact geometry: margin comes from the world, not the reward

The training env must encode the MEASURED contact envelope, not the geometric aperture:
- **Body-radius inflation:** pass requires point L-inf ≤ 0.75 − r_body; collision band
  (0.75 − r_body, 1.36 + r_body]. r_body sampled per env ∈ [0.28, 0.38] m (the halo is not
  precisely known and is unobservable → the policy trains to the sampled worst case).
  Evidence: live crashes at L-inf 0.37–0.49 scored as "pass" by the 0.75 point-mass model
  (4/4 standing gate-3 clips); corner-pass probe contact at 0.60 m.
- **Frame extrusion:** the collision band is volumetric over gate-frame |x| ≤ 0.30 m. Evidence:
  live strikes occur up to 0.5 m BEFORE the plane on slope-0.45 approaches.
- A trained policy's margin claims are stated against this contact-true geometry only. Margins
  quoted against the 0.75 point aperture are inflated by the body halo and are inadmissible.

## 3. Domain randomization policy

- **Standing axes (never removed):** super-rate map params, rate_tau, alpha_max, aero (quad-drag
  c2 relative band, collective-table delta scale ±10%, residual d1), mixer (idle, κ_err, κ_hold,
  ζ_yaw), transport latency centered on the measured value ({1,2,3} ticks for 67 ms).
- **Principled widths:** each force axis's band must be ≥ the certified residual bound from the
  regime-binned `scripts/frame_residual_report.py` over all post-fix live recordings (currently
  ≤ ~3 m/s²). When a refit shrinks the residual, the band may shrink to the new bound — never
  below measurement uncertainty.
- **Structured (regime-binned) force-bias DR:** per-env random world-frame bias ‖b‖ ≤ the
  certified bound, active in a per-env random speed×tilt bin. Rationale: global scale DR cannot
  represent "sysid is wrong in one regime" — the exact error class that produced the gate-3
  displacement. Global-scale robustness was measured present (±12% coll, ±25% drag absorbed);
  the structured axis closes the remaining class.
- **DR buys closed-loop disturbance absorption, not crossing margin** (measured: force-scale
  sweeps leave crossing offsets unchanged). Do not widen DR to fix a margin problem — fix §2.
- **Dead axes:** `dr_lapse` / `--plant lapse` (VOIDED by frame-audit 2026-06-12).

## 4. Resets, courses, curriculum

- `course_mode=random` (procedural 6-gate courses, VQ1-derived ranges, turns ±60°/segment),
  **VQ1 held out** as the eval course. Widen sampler ranges only on actual VQ2 information.
- `standing_start_frac=0.3`; both spawn modes get pose jitter. This coverage is SUFFICIENT for
  recovery: measured 12/12 re-threads from ±1.5 m displaced restarts at 18.6 m/s, 48/48 finishes
  under start-jitter × latency. Recovery curricula / perturbation injection are optional
  insurance, not load-bearing — do not add them as a fix for a transfer failure without first
  checking §2 and §3.
- One change family per increment (attribution discipline): geometry, DR axes, reward, and rate
  changes do not land together unless one is byte-identical-verified inert.

## 5. Checkpoint selection + validation gauntlet (before ANY live flight)

1. **≥3-seed generalization averaging** (gen is seed-volatile: inc6 0.741–0.982). Style/validity
   metrics (thr_p95, saturation, yaw flips) must reproduce across seeds.
2. **Held-out VQ1 (mixer plant, map-ON):** sr, median lap. (`peregrine_eval.py` defaults to the
   legacy flat plant — always pass the plant explicitly.)
3. **Offset tails + corridor:** per-gate crossing L-inf p95 under start-jitter × latency {0–3}
   (n ≥ 48) scored against CONTACT-TRUE aperture (§2); corridor clearance at gate-frame x=−1 m
   reported per gate.
4. **Robustness probes** (`handoff/laptop-training-doctrine-2026-06-12/scripts/doctrine_probes.py`):
   global force scales (coll ×0.88/×1.12, drag ×0.77/×1.25), regime-binned residual injection at
   the current certified bound, displaced-restart recovery. All must finish.
5. **V100 config-matrix parity gate** whenever dynamics-side code changed since the last gate.
6. **Deploy matrix** (latency 0–3 × start modes × perturbed seams) with fixed tools — and the
   matrix is only as good as its conventions: `tests/test_frame_conventions.py` green is a
   prerequisite, not a substitute.
7. **After every live session:** `scripts/frame_residual_report.py` (mirror canary, regime bins,
   rate canary) + crash forensics on any failure BEFORE iterating flags or retraining.

## 6. Inadmissible twin evidence (hard-won; do not relearn)

- **Lap time without attitude extraction** (CRAB-DIAG): never interpret live posture as a bug
  without extracting the twin's TRUE attitude at the same start state.
- **Internal-consistency convention checks** (FRAME-AUDIT): proper-rotation conjugations pass
  every internal test; only external invariants (force vs pristine vel_ned FD) discriminate.
- **Margins against the point-mass 0.75 aperture** (this session): inflated by the body halo.
- **Single-seed generalization numbers** (R2): seed-volatile.
- **Emulation harnesses that share convention constants with deploy code** (S18/FRAME-AUDIT):
  mirrors round-trip to false passes; the golden-file tests pin against recorded live data.
- **RACE_STATUS clock for RL-segment timing** (CRAB-DIAG): includes the CTBR launch (~8 s).

## 7. Where robustness comes from (priority order)

1. Honest contact geometry (§2) — margins from the optimum itself.
2. Structured + global force DR (§3) — disturbance absorption.
3. Reset/course diversity (§4) — recovery coverage.
4. Reward shaping — never for robustness; only for measured actuator pricing (c16).

Aggression is not penalized. Speed is the objective; the world model is the constraint.
