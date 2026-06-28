# Design Draft — RL-only vs RL + Model-Based Layer (2026-06-27)

*Status: DESIGN DRAFT for discussion, not a build commitment. Research phase; forks stay open.*

## The question the hybrid is trying to answer
Under **per-load track randomization** + **monocular self-localization noise** + **8 GB onboard**, is a pure RL/CTBR policy enough, or do we add a model-based optimization layer (differentiable-MPC or sampling-MPPI) on top of an RL prior for online adaptation / robustness / hard constraints?

---

## Plan A — RL-only (the current inc8 lineage, the spine)
Detector → gate-relative estimator (ESKF/RewindKF) → **PPO policy → CTBR**. One generalist policy trained with **layout domain-randomization**; reactive/gate-relative (no executed optimal-line tracking).

- **Pros:** proven (Swift, MonoRace, One-Net, E2E-RL all ship this); tiny net, <1 ms inference (all VRAM free for the detector); gate-relative obs already generalizes across layouts; lowest integration + code-audit risk; it's what we already have.
- **Cons:** robustness to dynamics/OOD is only what DR buys; no hard constraint guarantee (zero-gate-contact is reward-shaped, not enforced); peak time-optimality bounded by reward shaping.
- **Evidence:** Reaching-the-Limit (RL ≥ OC at the limit); E2E-RL (γ=0.999 → near-time-optimal); our own inc8 flies.
- **Highest-certainty wins inside Plan A (do these regardless):** ① fix the **symmetric critic** (triple-confirmed: symmetric = catastrophic) → privileged asymmetric critic; ② add **perception-aware reward** (PAMPC projection-centering + anti-blur + Fisher-info-over-corners); ③ add the **IMO learned-inertial gap-filler** (TCN Δp from commanded-thrust+gyro) for occlusion gaps; ④ per-gate **initial-state buffer** for gate-4 reach.

## Plan B — RL + online model-based layer (the hybrid)
Same perception/estimator; replace/augment the controller. Two concrete variants:

**B1 — AC-MPC (differentiable MPC as the actor's last layer).** Neural cost-map → diagonal Q,p → iLQR solve → CTBR; PPO backprops through the solver (needs DiffAero's differentiable model).
- *Pros:* nearly our exact stack; **deploy-light (one deterministic solve, 50 Hz/13.5 ms, leaves VRAM for detector)**; zero-shot 21 m/s; robust to **dynamics changes without retraining** (mass +27%, inertia ±80%, wind 1.5×); reuses DiffAero.
- *Cons:* adapts **dynamics, not new tracks online**; **NO state constraints** (still can't hard-enforce gate-contact); robustness shown to dynamics, **not estimator noise** (our binding risk); **30× training cost**; diagonal-Q mandatory.

**B2 — RL-prior + sampling MPPI.** RL mean action seeds the MPPI nominal; MPPI samples around it on the DiffAero model, scored by a gate/contact/perception cost.
- *Pros:* gradient-free → **arbitrary non-diff costs incl. a hard gate-contact term**; re-plans per-load online; compute proven feasible (100 Hz, 896×15, 8 GB Jetson).
- *Cons:* **racing-speed UNPROVEN (only ≤12 m/s in the literature)**; parallel-rollout **VRAM competes with the detector** on 8 GB; static hand-tuned sampling noise; rollouts amplify estimator noise over a 1.5 s horizon.

## Shared / additive components (belong to BOTH plans, not differentiators)
IMO gap-filler · recon-lap gate-map + relocalization (the "moat") · perception-aware reward · multi-gate PnP + de-rotated fallback + IMU-saturation guard · AHRS · GS-of-real-imagery offline (gated on VQ2 Training exposing real frames).

## Comparison
| Axis | A: RL-only | B1: AC-MPC | B2: RL+MPPI |
|---|---|---|---|
| Per-load track adaptation | DR-generalist (good) | dynamics only (≈A on tracks) | online re-plan (best, **but unproven @ speed**) |
| Estimator-noise robustness | DR-trained | **not addressed** | **not addressed** |
| Hard contact constraint | reward only | **none** | **yes (cost term)** |
| Deploy VRAM (vs 8 GB + detector) | trivial | light (1 solve) | **heavy (rollouts)** |
| Racing-speed evidence | strong | 21 m/s | **≤12 m/s only** |
| Training cost | baseline | **30×** | baseline (+ MPPI tuning) |
| Integration / audit risk | lowest | medium | medium-high |

## Honest assessment — what does the hybrid actually buy?
Less than my first pitch implied. The headline "online per-load adaptation without retraining" is **overstated**: AC-MPC adapts dynamics not tracks; MPPI re-plans per-load but isn't shown at racing speed and fights the detector for VRAM; and a **well-trained gate-relative generalist PPO already absorbs layout variation**. **Neither hybrid touches the estimator-noise binding risk.** The hybrid's *genuine* value is narrower: **(B1)** dynamics/OOD robustness for the sim-to-real (DiffAero→competition-sim→Neros) gap, and **(B2)** a *hard* zero-contact constraint the RL reward can't guarantee.

## Proposed spike (design-only now; ~1–2 days of DiffAero work when we build)
A single discriminating experiment, all in DiffAero, no new sim:
1. Baseline = current PPO/CTBR with the symmetric-critic fix + perception reward.
2. Inject realistic estimator noise (σ_p0 ≈ 0.15, the measured value) into the policy's state input.
3. Compare against (B2) a thin MPPI layer (seed = PPO mean, short horizon, gate-contact cost) under the *same* noise, at 15 / 25 / 30 m/s.
4. Metric: gate-4 reach/pass-rate and contact-rate vs speed.
- **Decision rule:** the hybrid earns a place only if it improves pass-rate-at-speed OR contact-rate *under noisy state* by a margin worth the VRAM/complexity. If RL-only matches it, the hybrid stays parked.

## Recommendation
Keep **RL/CTBR as the spine** and bank the four Plan-A certain wins first. Treat the hybrid as a **Phase-2 robustness/constraint lever**, with **B2 (RL+MPPI, for the hard contact constraint)** as the first thing to spike and **B1 (AC-MPC, for dynamics robustness)** as a fast-follow since it's almost our exact stack. Do **not** rebuild the controller around either until the spike shows a margin under *noisy* state at *racing* speed — the one regime no paper has tested and the one that actually decides it.
