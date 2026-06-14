# MEMO — Perception-Aware / Active-Vision Reward Design for RL Drone Racing

**To:** Fengyou · **Project:** Peregrine (feeds path P2) · **Date:** 2026-06-14
**Re:** A policy-driven reward that keeps the *next gate* centered in the fixed camera through the **terminal approach**, without slowing the racer down.
**Method:** deep-research harness (5-angle fan-out, ~45 sources searched, ~30 primary PDFs read, key/recent citations adversarially verified). Honesty flags inline: **[read]** = extracted from primary text; **[proposed]** = our synthesis, not found implemented in a drone paper; **[qual]** = supported qualitatively, not quantified in the literature.

---

## 0. Bottom line first (escape hatch)

There **is** a clear best-practice spine in the literature, and it is almost exactly your problem. SWIFT (Nature 2023) and Geles et al. (RSS 2024) both add a single, small, smooth reward term that rewards aligning the camera optical axis with the **next gate center**, on top of a dominant **potential-based gate-progress** reward, with a **privileged (asymmetric) critic**. That is the proven baseline. Your two genuine deltas beyond the literature are (a) making the term **2-axis** (azimuth + elevation) to match your asymmetric 90°H/59°V FoV and pitch coupling, and (b) **terminal-weighting** it so visibility is enforced hardest in the final approach — which is what actually breaks your localization. The terminal weighting is a real, defensible novelty: the only drone precedent for proximity-weighting "keep-in-view" is a 2025 competition planner (Azhari et al.), and it is a *planner*, not an RL reward.

### Recommended reward shape

Per-step reward (speed stays dominant):

```
r_t = r_prog      (dominant, potential-based — farm-proof)
    + r_perc      (small, terminal-weighted, 2-axis — the new term)
    + r_smooth    (body-rate + action-rate penalties / CAPS)
    + r_pass      (sparse gate-pass bonus, optional)
    - r_crash     (terminal penalty)
```

**Progress (unchanged, keep it dominant) [read — SWIFT/Song]**
`r_prog = λ1 · ( d_{t-1}^gate − d_t^gate )`  — or the gate-to-gate line-projection increment `Δs = s(p_t) − s(p_{t-1})`. This is a potential difference (Φ = −distance), so it is provably un-farmable by loitering (Ng et al. 1999).

**Perception (the new term — 2-axis, terminal-locked, progress-gated):**

```
r_perc = λp · w_term(d) · v(α, β) · max(Δs, 0)
```

where, with the gate center expressed in the camera frame `p^c = R_bc^T R^T (g − p) = [X, Y, Z]`:

- **azimuth** `α = atan2(X, Z)`, **elevation** `β = atan2(Y, √(X²+Z²))`  (Qin et al. 2026 parameterization) **[read]**
- **separable quartic visibility** `v(α,β) = exp[ −((α/σα)^4 + (β/σβ)^4) ]`, with `σα ≈ α_max ≈ 45°`, `σβ ≈ β_max ≈ 29.5°` (= your H/V half-angles, optionally ×0.8 to bias toward center). Quartic-in-angle inside an `exp` = flat plateau while the gate is safely in frame, sharp cliff as it nears either edge. This is the SWIFT/Geles `exp(−δ⁴)` shape, generalized to two independent axes so the **narrower vertical FoV and the 20°-tilt/pitch coupling are penalized more tightly than azimuth**. **[read + extended]**
- **terminal weight** `w_term(d) = w0 + (1−w0)·clip( (d_acq − d)/(d_acq − d_lock), 0, 1 )`, with `d_lock ≈ 5 m` (full lock, w=1, through your terminal window), `d_acq ≈ inter-gate spacing`, floor `w0 ≈ 0.1–0.2` (acquire early, lock late). Azhari et al. λ(d) ramp. **[read for the ramp; terminal-as-RL-reward is proposed]**
- **progress-gating** `· max(Δs, 0)`: the bonus is only payable *while making forward progress*, so "slow down / orbit to farm a fix" earns ≈0. **[proposed — see §3]**

**Smoothness [read — CAPS / Song]:** keep `−λω‖ω‖²` and add `−λΔa‖a_t − a_{t-1}‖²`; penalize **yaw-rate and pitch-rate** specifically. If chatter appears, add CAPS temporal+spatial smoothness losses (`L_T`, `L_S`) — reward-agnostic, sits on top of PPO. Feed `a_{t-1}` into the observation if you use an action-rate penalty.

**Critic [read — SWIFT/Pinto]:** train an **asymmetric** PPO — give the *critic* privileged full state (drone pose/vel, exact gate pose, true α/β), the *actor* only the deployable observation. Let the actor **discover** pointing and fix-timing; do **not** hardcode a fix schedule you have no evidence for.

**Starting weights [read]:** set `λp ≈ 5% of λ1` (Geles: 0.025 vs 0.5; Song ICRA'23: 0.05 vs 5.0). Expect ~0% lap-time cost where yaw-pointing is free and up to ~5–10% only where geometry forces a pitch/visibility conflict (Song ICRA'23 measured 7.78 s → 8.52 s on their tightest track).

**Axis routing (your 2-axis question) [read + inference]:** route **azimuth via yaw** — yaw is a flat output, translationally ~free, so reducing your 64° crab toward the gate-bearing costs almost no time. Be gentle on **elevation via pitch** — pitch *is* the forward-accel axis, so a hard elevation term fights the racing attitude (PAMPC documents this exactly: to center a target the optimizer pitches *less* and gains altitude). Note your **20°-up tilt helps**: level flight looks 20° up, and nose-down pitch θ rotates the optical axis to (20−θ)°, so ~20° nose-down puts forward gates near boresight — keep `σβ` generous so the term doesn't blunt aggressive braking into gates.

### Minimal ablation set (run in this order)

| # | Config | Question it answers |
|---|--------|---------------------|
| **A0** | progress + smoothness + crash (no perception) | Baseline terminal fix-rate (your current ~1.4–7%). |
| **A1** | A0 + **single-axis** boresight `λp·exp(−δ⁴)`, uniform | Does the literature-standard term help at all? |
| **A2** | A1 → **2-axis** `exp[−((α/σα)⁴+(β/σβ)⁴)]`, uniform | Does separating elevation from azimuth matter? |
| **A3** | A2 + **terminal weight** `w_term(d)` | **The core bet:** does terminal-locking fix the drought? |
| **A4** | A3 + **progress-gating** (or PBRS variant) | Anti-farming: kills lap-time loss / loiter exploit? |
| **(orthogonal)** | asymmetric critic ON vs OFF; `λp ∈ {2%, 5%, 10%}` | Critic value + weight sensitivity. |

**Metrics on every run:** lap time (true objective), **terminal-window fix-rate** (final 5 m / ≥60% of approach), **worst-gate fix-rate** (your 1.4% gate), pointing-error trace, yaw-rate & pitch-rate RMS (chatter), gate-success rate. Plot lap-time (true) and pointing-score (proxy) on **separate axes** and watch for a divergence cliff (Pan et al. phase transition).

The rest of this memo is the evidence and the option space behind each choice.

---

## 1. Active-vision / perception-aware rewards (drone racing & RL)

**The dominant pattern is a small additive "point the optical axis at the next target" term.** Three concrete, near-drop-in formulations, in order of relevance:

- **SWIFT — Kaufmann et al., Nature 2023 [read, verified].** `r_t = r_prog + r_perc + r_cmd − r_crash`, with `r_perc = λ2·exp(λ3·δ_cam⁴)` where **δ_cam = angle between the camera optical axis and the direction to the next gate center**. Their stated rationale is verbatim your problem: *"seeing the next gate is rewarded because it increases the accuracy of the pose estimate."* Camera fixed; action = collective-thrust + body-rates, so **yaw is a free DOF the term spends to aim**. Localization = VIO + neural gate-corner detector fused in a Kalman filter — the perception reward exists *to feed that estimator*, exactly your dependency.
- **Geles et al., RSS 2024 [read].** End-to-end *from pixels*, same family: `r_perc = λ2·exp(−δ_cam⁴)`, `λ2 = 0.025` vs progress `λ1 = 0.5` (perception ≈ 5% of progress). Observation is a gate-edge segmentation mask; the perception reward keeps that mask populated. Uses an **asymmetric actor-critic** (privileged critic). This is your closest end-to-end analog.
- **Learning Perception-Aware Agile Flight — Song et al., ICRA 2023 [read].** `r_pa = exp(−‖θ_yaw − θ_dir‖)` (align yaw with flight direction). Weights: progress `kp=5.0` vs perception `kpa=0.05`, raised to `0.1` only on the hardest track. Motivated explicitly by the failure that RL controllers "completely ignore the perception constraint induced by the camera's field of view."

**Control-theoretic ancestor (for the exact "centering" geometry):**

- **PAMPC — Falanga et al., IROS 2018 [read].** Perception enters the MPC *cost* (not a hard constraint): penalize the target's **image-plane projection** `s=(u,v)` distance from image center **and** the image-plane *velocity* `ṡ` (anti motion-blur), reference `z=[s,ṡ]=0`. Borrow the second idea: penalizing `‖ṡ‖` discourages whipping the gate across the frame, a subtler aid than centering alone.
- **Asymmetric FoV (your exact geometry) — Qin et al., arXiv:2603.04305, 2026 [read, verified].** The only source that splits FoV into **independent azimuth `α_max` and elevation `β_max` half-angles** with a fixed tilted camera — i.e., your 90°H/59°V. Hard constraint `|α|≤α_max ∧ |β|≤β_max ∧ Z>Z_min`, relaxed to a **soft slack penalty** `L_FOV = w_FOV·(1ᵀS + ½‖S‖²)`. This is the template for the 2-axis `v(α,β)` above.
- **Other formulations worth knowing:** symmetric **cone** constraint that is convex in the squared-speed profile (Spasojevic et al., ICRA 2020); **tanh-frustum product** smooth visibility `Π ½(1+tanh(n_j·z/s))` (Murali et al., ACC 2019); **raised-cosine** in bearing angle `½(cos aθ+1)` (Frey & How, 2019). All are alternative shapes for `v`; the quartic-exp is smoother-plateaued and matches the proven drone-racing terms.

**Takeaway:** use the SWIFT/Geles `exp(−·⁴)` shape, generalized to the Qin azimuth/elevation pair. Everything else is a shape variant.

---

## 2. Preventing the degenerate "slow down to look" exploit

This is the crux, and the literature is unusually clear.

**(a) Your exploit is textbook, and the canonical paper describes it exactly. [read]** Ng, Harada & Russell (ICML 1999) — their motivating bugs are a bicycle agent that "rode in tiny circles" and a soccer bot that "vibrated next to the ball" to farm a non-potential progress bonus. The theorem: a shaping term preserves the optimal policy **iff** it is *potential-based*, `F = γΦ(s′) − Φ(s)`. **Your progress reward already is potential-based** (Φ = −distance-to-gate), so speed cannot be farmed by hovering — the danger is concentrated entirely in the *non-potential* perception term. Krakovna et al. (DeepMind 2020) give the drone-shaped cautionary tale: a boat rewarded with a shaping bonus "drove in circles… never finishing the race." Skalse et al. (NeurIPS 2022) prove an under-weighted/omitted-term proxy is almost always hackable; Pan et al. (ICLR 2022) show hacking *intensifies* with finer action resolution (your per-rotor/CTBR action space) and arrives as a sudden **phase transition** with "little prior warning" — so early correlation between pointing-score and lap-time is **not** evidence of safety.

**Three mitigations, in increasing strength:**

1. **Keep it small and bounded (literature-standard) [read].** `λp ≈ 5%` of progress; this is what SWIFT/Geles/Song do and it mostly works. Weakest guarantee; relies on tuning + monitoring.
2. **Potential-based perception shaping (strongest formal guarantee) [read + proposed].** Set `Φ(s) = c·w_term(d)·v(α,β)` and use `r = γΦ(s′) − Φ(s)`. By Ng et al. this **cannot change the speed-optimal policy** — zero farming risk. Caveat: by the same token it will *never* trade time for visibility, so use this only if you believe pointing is ~free on the optimal line (often true for yaw-azimuth, rarely for pitch-elevation).
3. **Progress-gated perception (recommended middle path) [proposed].** `r_perc ∝ max(Δs,0)·v(α,β)·w_term(d)`. The bonus is payable only while moving forward, so slowing to farm yields ≈0, yet — unlike strict PBRS — it still lets the policy buy a *little* visibility with attitude when geometry demands. The components are individually citable (potential-shaping; multiplicative "soft gating"), but **we did not find a drone paper that multiplies the perception bonus by progress** — flag it as our construction, validate in ablation A4.

**(b) Annealing / curriculum schedules [read].** Song et al. (IROS 2021) add the body-rate penalty "at an initial training stage" then relax it — a smoothness curriculum. Recent work anneals an auxiliary weight from 0 → target after the base task is learned. Practical recipe: train A0 to competent racing first, then **ramp `λp` from 0**; optionally ramp `w_term`'s lock distance inward over training. This avoids the early "slow-to-look" local optimum.

**(c) Asymmetric / privileged critic — directly applicable, low-risk [read, verified].** Pinto et al. (RSS 2018): critic sees full state, actor sees only observations; better value estimates → better, lower-variance policy gradients, and the actor learns robust behavior from impoverished input. **SWIFT explicitly does this** ("the value network … can access privileged information … not accessible to the policy," citing Pinto). This is precisely your stated intent: give the critic the gate pose / true α,β, and let the actor *discover* when and how to point — **no prescribed fix-timing schedule required.** (Teacher-student distillation — Learning by Cheating; Song ICRA'23 — is a heavier alternative; asymmetric-critic matches "let the actor discover" most directly.)

---

## 3. Terminal / approach-phase visibility specifically

**Your instinct — weight visibility near the gate, not uniformly — is correct and under-explored.** [qual]

- **Almost all target-keeping rewards are uniform over the trajectory** (Luo et al. active tracking `r = A − (√(x²+(y−d)²)/c + λ|a|)`; aerial-cinematography `R_pr`; Wang/Gao visibility-aware planner). None weight by approach phase.
- **The one drone precedent for proximity-weighting is a planner, not a reward [read, verified]:** Azhari et al. (arXiv:2512.20475, A2RL×DCL 2025, podium system) blend heading to the current vs next gate with a **distance ramp** `λ_i = 1 if d<d_min; 0 if d>d_max; (d_max−d)/(d_max−d_min) otherwise` — i.e., **lock onto the gate as you close in.** They show the time-optimal baseline "loses sight of the upcoming gate until the last moment" on sharp turns, and their anticipatory heading raised gate visibility from 51.8%→60.2% at a realistic 120°×90° FoV. That is your terminal-drought, observed and fixed — port the λ(d) ramp into `w_term(d)`.
- **Soft, distance-tightening FoV machinery [read]:** Qin et al.'s slack penalty `w_FOV·(1ᵀS+½‖S‖²)` "asymptotically approximates a hard constraint" as `w_FOV↑`. Make `w_FOV` (or our `w_term`) grow on approach: relaxed far (don't wreck the racing line during gate transitions), near-hard inside 5 m.
- **Transferable concept from terminal guidance [qual, concept only]:** missile FoV-constrained guidance keeps the seeker look-angle inside ±FoV through terminal approach via a **time-to-go-weighted** gain/bias. The *structure* — a keep-in-view weight that scales with time-to-go — is exactly `w_term(τ_go)`; the closed-form gains assume a 2-body intercept and don't transfer.

**Honest caveat:** no paper ablates *terminal vs. uniform* visibility weighting and shows terminal wins on **localization** — that framing (terminal fix-droughts dominate a fine average) is yours. It is well-motivated (Azhari's last-moment loss; Gschwindt et al.'s finding that even a *momentary* loss of the subject is rated catastrophic and should be "punished even more harshly"), but ablation **A3 is where you generate the evidence**, since you have the fix-rate instrumentation the literature lacks.

---

## 4. Failure modes & mitigations

- **Reward hacking / "slow-to-look" loiter [read].** Covered in §2. Defenses: potential-based or progress-gated perception, small `λp`, anneal from 0, and monitor lap-time vs pointing-score on separate axes for the phase-transition cliff (Pan et al.).
- **Oscillatory / chattering pointing [read].** A pointing term competing with a (bang-bang-prone) time-optimal objective creates the singular-arc conditions for chatter — yaw/pitch "hunting" around the centered set-point (Seyde et al., NeurIPS 2021: minimum-time → boundary actions → chattering/Zeno). CAPS (Mysore et al., ICRA 2021) is the clean fix: temporal `L_T = D(π(s_t),π(s_{t+1}))` + spatial `L_S = D(π(s_t),π(s̄_t))` smoothness losses; ~96% smoother control on a real quad at marginal reward cost. **Do not** post-filter a trained policy's outputs (CAPS reports catastrophic loss of control from naïve FIR filtering). Penalize **rates**, not just magnitudes; quartic-exp `v` (flat plateau) also helps by giving ~zero gradient when the gate is comfortably centered, so there's nothing to hunt against.
- **Conflict with aggressive racing attitude [read].** PAMPC: centering a target makes the optimizer **pitch less / gain altitude** — i.e., perception competes with the thrust-tilt that produces forward acceleration. Song ICRA'23 quantifies the resulting cost (≤ ~10% lap time, only on the tightest track; ~0% elsewhere) and the payoff (70%→100% success). Mitigations: route azimuth→yaw (free), keep elevation/pitch term gentle (`λp` small, `σβ` generous), accept a bounded time cost only where geometry forces it. Expect — and welcome — **emergent crab reduction**: a yaw-pointing reward will pull your 64° crab toward the gate-bearing; that *is* the fix. Bound residual sideslip only if it hurts aero/tracking.
- **Multi-objective fragility [read].** Linear scalarization (`progress − λ·pointing`) can't reach non-convex Pareto regions and is weight-sensitive (Song had to 2× the weight for a hard track). Manage via the `λp` sweep in the ablation, the curriculum ramp, and treating perception as a *soft* cost, never a hard constraint (hard FoV constraints render the racing problem infeasible — Qin et al.).
- **Goodhart on the proxy [read].** "Fraction of frames gate-centered" is a proxy for "perceive well enough to localize." Optimized hard it diverges from the goal (Amodei et al.; Skalse et al.). Keep the *true* metric (terminal fix-rate / localization health) in your eval loop, not just the reward.

---

## 5. References

**Drone-racing & perception-aware RL**
- Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza. *Champion-level drone racing using deep reinforcement learning* (SWIFT). Nature 620:982–987, 2023. https://www.nature.com/articles/s41586-023-06419-4 · open: https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/
- Geles, Bauersfeld, Romero, Xing, Scaramuzza. *Demonstrating Agile Flight from Pixels without State Estimation*. RSS 2024. https://www.roboticsproceedings.org/rss20/p082.pdf · https://arxiv.org/abs/2406.12505
- Song, Shi, Penicka, Scaramuzza. *Learning Perception-Aware Agile Flight in Cluttered Environments*. ICRA 2023. https://arxiv.org/abs/2210.01841
- Song, Steinweg, Kaufmann, Scaramuzza. *Autonomous Drone Racing with Deep Reinforcement Learning*. IROS 2021. https://arxiv.org/abs/2103.08624
- Song, Romero, Müller, Koltun, Scaramuzza. *Reaching the limit in autonomous racing: optimal control versus reinforcement learning*. Science Robotics 8(82), 2023. https://arxiv.org/abs/2310.10943
- Loquercio, Kaufmann, Ranftl, Müller, Koltun, Scaramuzza. *Learning high-speed flight in the wild*. Science Robotics 6(59), 2021. https://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf (tilts sensor 30° fwd to keep targets in FoV — hardware analog of the reward)
- Ferede, De Wagter, Izzo, de Croon. *End-to-end RL for Time-Optimal Quadcopter Flight*. 2023. https://arxiv.org/abs/2311.16948

**Perception-aware control / trajectory optimization (FoV cost formulations)**
- Falanga, Foehn, Lu, Scaramuzza. *PAMPC: Perception-Aware Model Predictive Control for Quadrotors*. IROS 2018. https://arxiv.org/abs/1804.04811 · code https://github.com/uzh-rpg/rpg_mpc
- Qin, Xing, Reiter, Romero, Lin, Liu, Scaramuzza. *Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight* (independent azimuth/elevation FoV; soft slack). arXiv:2603.04305, 2026. https://arxiv.org/abs/2603.04305
- Spasojevic, Murali, Karaman. *Perception-Aware Time Optimal Path Parameterization for Quadrotors*. ICRA 2020. https://arxiv.org/abs/2005.13986
- Murali, Spasojevic, Guerra, Karaman. *Perception-aware trajectory generation for aggressive quadrotor flight using differential flatness*. ACC 2019. https://dspace.mit.edu/handle/1721.1/132953
- Frey, Steiner, How. *Towards Online Observability-Aware Trajectory Optimization for Landmark-based Estimators*. arXiv:1908.03790, 2019. https://arxiv.org/abs/1908.03790
- Penin, Spica, Robuffo Giordano, Chaumette. *Vision-Based Minimum-Time Trajectory Generation for a Quadrotor UAV*. IROS 2017. https://ieeexplore.ieee.org/document/8206522 *(primary text not retrievable; method from secondary descriptions)*

**Terminal / proximity-weighted visibility & target-centering**
- Azhari et al. *Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing* (λ(d) distance ramp; A2RL×DCL). arXiv:2512.20475, 2025. https://arxiv.org/abs/2512.20475
- Luo, Sun, Zhong, Liu, Zhang, Wang. *End-to-end Active Object Tracking via RL*. ICML 2018 / TPAMI 2019. https://arxiv.org/abs/1705.10561 · https://arxiv.org/abs/1808.03405
- Gschwindt, Camci, Bonatti, Wang, Kayacan, Scherer. *Can a Robot Become a Movie Director?* IROS 2019. https://arxiv.org/abs/1904.02579 · JFR 2020 ext. https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21931
- Wang, Gao, Ji, Xu, Gao. *Visibility-aware Trajectory Optimization with Application to Aerial Tracking*. IROS 2021. https://arxiv.org/abs/2103.06742 · code https://github.com/ZJU-FAST-Lab/visPlanner
- (concept only) Missile terminal guidance with seeker FoV / time-to-go-weighted look-angle, e.g. *Look-angle-tracking 3D impact-time guidance with FoV constraint*, IJRNC 2023. https://onlinelibrary.wiley.com/doi/10.1002/rnc.6886

**Reward shaping, anti-exploit, privileged critic, smoothness**
- Ng, Harada, Russell. *Policy Invariance Under Reward Transformations* (potential-based shaping). ICML 1999. https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Pinto, Andrychowicz, Welinder, Zaremba, Abbeel. *Asymmetric Actor Critic for Image-Based Robot Learning*. RSS 2018. https://arxiv.org/abs/1710.06542
- Chen, Zhou, Koltun, Krähenbühl. *Learning by Cheating*. CoRL 2019. https://arxiv.org/abs/1912.12294
- Skalse, Howe, Krasheninnikov, Krueger. *Defining and Characterizing Reward Hacking*. NeurIPS 2022. https://arxiv.org/abs/2209.13085
- Pan, Bhatia, Steinhardt. *The Effects of Reward Misspecification*. ICLR 2022. https://arxiv.org/abs/2201.03544
- Krakovna et al. *Specification gaming: the flip side of AI ingenuity*. DeepMind, 2020. https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Amodei, Olah, Steinhardt, Christiano, Schulman, Mané. *Concrete Problems in AI Safety*. 2016. https://arxiv.org/abs/1606.06565
- Mysore, Mabsout, Mancuso, Saenko. *Regularizing Action Policies for Smooth Control* (CAPS). ICRA 2021. http://ai.bu.edu/caps/
- Seyde et al. *Is Bang-Bang Control All You Need?* NeurIPS 2021. https://arxiv.org/abs/2111.02552

*Verification note: SWIFT reward structure + privileged critic, and the existence/titles of the two 2025–26 preprints (2603.04305, 2512.20475) were independently re-checked. Exact SWIFT λ-values live in the paper's Extended Data; the functional forms above are confirmed. Penin et al. exact FoV equations could not be retrieved (publisher block).*

---

## MEMORY-DELTA

- **Best-practice spine exists:** SWIFT/Geles `r_perc = λ·exp(−δ_cam⁴)` (point optical axis at next-gate center) + potential-based gate-progress + asymmetric privileged critic. Adopt as baseline.
- **Recommended new term:** `r_perc = λp · w_term(d) · exp[−((α/σα)⁴+(β/σβ)⁴)] · max(Δs,0)` — 2-axis (Qin α/β, σα≈45°, σβ≈29.5°), terminal-locked (Azhari λ(d), d_lock≈5 m), progress-gated.
- **Anti-"slow-to-look":** progress-gating (proposed) and/or potential-based shaping (Ng 1999) make loitering net ≈0; keep λp≈5% of progress; anneal from 0. Progress-gating is OUR construction (not in a drone paper) → validate in ablation A4.
- **Critic:** asymmetric — privileged state to critic, deployable obs to actor; let actor discover fix-timing (no prescribed schedule).
- **Axis routing:** azimuth→yaw (free); elevation→pitch is the conflict axis (PAMPC); 20°-up tilt partially offsets nose-down pitch. Expect emergent crab reduction.
- **Failure modes:** chatter (CAPS + rate penalties; quartic plateau gives zero gradient when centered); ~0–10% lap-time cost only on tight geometry (Song); watch lap-time vs pointing-score for phase-transition cliff (Pan).
- **Open gap = our novelty:** no paper ablates terminal-vs-uniform visibility weighting on localization; ablation A3 generates that evidence (we have the fix-rate instrumentation the literature lacks).
- **Min ablations:** A0 none → A1 1-axis → A2 2-axis → A3 +terminal → A4 +progress-gate; orthogonal critic on/off and λp∈{2,5,10}%.
- **Do-not:** hard FoV constraint (infeasibility, Qin); post-hoc output filtering (CAPS); trusting early proxy↔truth correlation (Pan).
