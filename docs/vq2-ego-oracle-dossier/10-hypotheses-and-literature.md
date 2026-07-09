# 10 — Deep Hypothesis Analysis: the single-gate floor vs. what the field knows

Written 2026-07-09 overnight (RL commander), at Fengyou's instruction to **stop reflex-patching**
(e.g. capping the speed reward) and instead *connect the symptoms we observe to the things we know*,
with literature research. This document is the reasoning; [04](04-experiment-log.md) is the data.

---

## 1. The symptom, stated as precisely as we can

On a **single** gate (depth 8–15 m, height ±6 m, tail-first spawn), the champion `vglpan`:

- **reaches the gate plane ~100%** of the time (oob≈0, floor≈0 — the reach-rate "problem" was a
  broken-harness artifact, [00](00-CORRECTION-eval-harness.md));
- **crosses at ~0.88 m L-inf offset** from center (deterministic 22.6% thread; **no determinism gap**);
- was thought **control-limited** (more training, tighter reward zero, sharper *actor* noise all failed)
  — **but that is now REFUTED** (see next bullet). Those pushes never touched the *estimator* noise.
- The **speed lever is refuted**: capping the rewarded closing rate (`rw_vmax_mps` 39→5→3) made
  centering *monotonically worse* (0.88→1.57→3.29 m), because it flattens the far-field progress
  gradient rather than forcing a slower crossing ([04](04-experiment-log.md) Era 6).
- **PERCEPTION is the dominant cap (ablation resolved, 2026-07-09, FINAL).** `vglpns0`
  (`ego_noise_scale=0`, a *perfect estimator*, else the champion recipe) reached **thread 0.63 / xoff
  0.32 m** (deterministic **0.641**) — vs champion 0.20 / 0.88 m. A perfect estimator **3×'d thread and
  put the crossing inside the 0.42 m aperture.** So the floor is **perception-limited, not control-
  limited**; my ≤0.3 m prior was too low because the **closed loop amplifies** ~0.28 m of measurement
  noise into ~0.5 m of crossing offset (noisy `rel_pos` every step → hedged control).
- **A control clip-residual sits underneath.** With perfect perception the *remaining* 36% failures are
  almost all **collisions/frame-clips** (det: collision 0.35, miss 0.008, oob 0.0001) — the drone crosses
  near-centred but clips ~⅓ of the time. So: perception first (0.88→0.32), then a control/precision layer.

So: the drone arrives at the plane, but **noisy perception drives it ~0.9 m off-center** — cleaning the
perception (not the reward) is what tightens it (Fengyou's idea (a) confirmed). `vglpns0` is a *ceiling*
(perfect estimator, not deployable); the deployable question is how far `r_perc` closes the gap with real
noise (in flight).

---

## 2. What the field's champion actually does — and where WE differ

**Swift** (Kaufmann et al., *Champion-level drone racing using deep RL*, Nature 2023) is the north
star. Its exact design (pulled from the open-access Methods):

**Reward** `r_t = r_prog + r_perc + r_cmd − r_crash`:
- `r_prog = λ₁ (d_{t−1}^Gate − d_t^Gate)` — progress **toward the center of the *next* gate** (a
  distance-reduction potential). When a gate is passed, the target switches to the next gate's center.
- `r_perc = λ₂ exp(λ₃ · δ_cam/4)` — a **perception reward**: keep the camera optical axis pointed at
  the next gate center (improves the pose estimate, and implicitly keeps you square-on).
- `r_cmd = λ₄‖a_ω‖ + λ₅‖a_t − a_{t−1}‖²` — body-rate + action-smoothness penalty.
- `r_crash = 5.0` on `p_z<0` or gate collision — a **small terminal**, terminating the episode.

**Observation** `o ∈ ℝ³¹`: robot state (pos, vel, attitude *rotation matrix*) = 15; **the next gate as
the relative positions of its 4 corners = 12**; previous action = 4. **Only ONE next gate is
observed**, but as a *rich 12-dim corner cloud* (which encodes center, size, AND orientation/normal).

### The gaps between Swift and us

| | **Swift (proven)** | **Us (`vglpan`)** |
|---|---|---|
| **Speed reward** | **none** — speed emerges from progress + γ-discounting | explicit: progress clipped at `vmax=39 m/s` + `−0.02/step` time + finish-time bonus |
| **Centering** | **none** — emerges from progress-to-*center* | an **artificial parabola** centering bonus + passage-centering + magnitude "I" term |
| **Perception reward** | **yes** (`r_perc`, camera-on-gate) | **none** (we have a `visible_area` *observation* but no reward for pointing at the gate) |
| **Terminal penalty** | tiny (**5.0**) | large (**100–200**, progress-scaled forfeit) |
| **Gate observation** | **4 corners, relative (12-dim)** — full pose every frame | **noised center-vector (3) + confidence (1) + coarse area scalar (1)**; corners/normal **deliberately dropped** (vision too inconsistent) |
| **Gate slots** | 1 next gate (always filled) | WINDOW=2; on single-gate **one slot is permanently empty** |

Two things jump out. **(i)** We *added* machinery Swift never needed (a centering bonus, a large
terminal, an explicit speed incentive) and we are now fighting that machinery. **(ii)** Our per-gate
**observation is geometrically far poorer** than Swift's: a single noised vector-to-center + a coarse
scalar, versus 4 corners that *over-determine* the gate pose. This is the concrete form of Fengyou's
worry that *"the policy doesn't have every input it needs."*

---

## 3. Fengyou's two ideas, analyzed against this

### Idea (a): lower / remove the speed reward — "would that reduce the noise arriving to the policy?"

- **The field agrees with the instinct.** Swift has **no explicit speed reward** — speed is an
  *emergent* consequence of progress + discounting + the small time cost. Our stack, by contrast,
  explicitly pays closing-rate up to 39 m/s plus a time penalty plus a finish-time bonus. So *removing
  the explicit speed incentive and letting speed emerge* is **field-aligned, not a hack.**
- **But the naive version is already refuted.** `vglpsl` (capping `rw_vmax_mps`) made centering worse
  — because it flattened the far-field gradient (the drone dawdles, arrives worse), *not* because a
  slower crossing is bad. So the *lever* was wrong, not the idea.
- **The "reduce noise arriving to the policy" mechanism** is subtle and worth stating: a slower
  approach → **more accepted fixes per meter + less ego-propagation through no-fix gaps + less
  control latency effect** → a cleaner `rel_pos` at the plane. That is a *real* channel, distinct from
  the reward-gradient one that `vglpsl` broke.
- **Partly under test already:** `vglpns0` removes the injected measurement noise directly. **If a
  perfect estimator does *not* tighten the crossing, then "noise arriving to the policy" is not the
  binding constraint, and slowing-to-reduce-noise won't help either.** *Caveat:* `vglpns` zeroes
  injected *measurement* noise but **not** staleness/propagation or the control-latency DR, so a
  genuine slow-lap could still help via *latency* even if measurement noise is not the cap. → If
  `vglpns` says control-limited, the clean idea-(a) experiment is a **Swift-style reward** (progress-
  to-center as the sole driver, drop the explicit speed/time terms) — **not** another `vmax` cap.

### Idea (b): more gates / more filled inputs → "maybe it reacts better with multiple gates"

This splits into two claims, and the literature comes down hard on one of them:

- **(b2) The optimal crossing point through a gate is defined by the *next* gate.** **Strongly
  supported.** Time-optimal racing theory (Foehn et al. CPC, *Science Robotics* 2021; and the TOGT
  gate-traversing planner, arXiv:2309.06837) shows the minimum-time line traverses gates **near their
  inner edges — corner-cutting — not through the center** (TOGT is 7.2% faster than centering by
  hugging inner edges). The crossing point is a function of the **gate sequence**. **A single isolated
  gate therefore has an *underspecified* optimal crossing point** — the racing task doesn't care where
  in the aperture you cross a lone gate; only *safety margin* weakly favors center. **We may be
  grinding to nail a metric (dead-center a lone gate to 0.42 m) that the real multi-gate task
  *redefines* (cross on the racing line, often off-center).** The real VQ2 — 10–20 m spacing, multi-
  gate co-visibility — is exactly where the crossing point becomes well-defined.
- **(b1) More *filled slots* help. NOW EVIDENCE-BACKED WITH A DIRECT ABLATION.** Song, Steinweg,
  Kaufmann & Scaramuzza (IROS 2021 — the Swift precursor) ran a controlled gate-count ablation
  (Table IV), and it is dramatic:

  | gates observed N | lap time (s) | **crash ratio** |
  |---|---|---|
  | 1 | 9.92 | **23.0 %** |
  | 2 | 8.32 | **2.5 %** |
  | 3 | 8.36 | 2.3 % |

  **N=1 → N=2 cuts the crash rate ~10× (23% → 2.5%) and is faster.** N≥2 is field-standard: Song
  SciRob 2023 uses N=2 (4 corners each); Song IROS 2021 default N=2 (center + normal-angle); MonoRace
  (A2RL 2025 winner) uses current-gate + next-gate. **Our `WINDOW=2` architecture *is* Song's N=2 — but
  on a single gate the second slot is permanently empty, i.e. we are training in exactly the crash-prone
  N=1 regime the field abandoned.** The paper gives no mechanism, but it lines up with (b2): with only
  the current gate the policy cannot compute a crossing point/approach that respects what comes next.
  **This is the strongest single piece of evidence that the single-gate problem may be intrinsically
  harder than the real multi-gate task** — and that we should validate at N=2, not grind N=1.
  (The deeper lever it *also* points to is **richer per-gate observation** — 4 corners vs our noised
  center-vector — hypothesis H2 below.)

  **Observation representations across the field (for reference):** Swift N=1, 4 corners relative;
  Song SciRob N=2, 4 corners (later gates in the previous gate's frame); Song IROS N=2, spherical
  center + gate-normal-angle; Geles N=1, pixel mask (actor) + gate-*center* vector (privileged critic);
  MonoRace current-gate-frame state + next-gate center+yaw. **Ours:** noised center-vector(3) + conf(1)
  + visible-area(1) per slot, ×2 — center-based like Geles/MonoRace, but poorer (no corners, a coarse
  area scalar for orientation). No one else uses a lone `visible_area` scalar as the orientation cue.

**Net:** Idea (b) is pointing at two real, well-motivated levers — **(i) richer per-gate observation**
(corners/normal, matching Swift) and **(ii) evaluating/training in the real multi-gate regime where
the crossing point is actually defined** — and at a genuine possibility that **the single-gate 0.42 m
target is the wrong target.**

---

## 4. Refined hypothesis set (ranked) and how each is discriminated

| # | Hypothesis | Prediction if true | Discriminator | Status |
|---|---|---|---|---|
| H1 | **Perception-noise magnitude** caps centering | `vglpns0` tightens | `ego_noise_scale` 0.0/0.5 dose-response | ✅ **CONFIRMED** (0.88→0.52 m, falling) |
| H3 | **Control** (plant+CTBR+net can't null the last ~0.9 m) | `vglpns0` still floors ~0.88 m | `vglpns0` result | ❌ **REFUTED** (perfect estimator tightens) |
| H4 | **Missing perception objective** (`r_perc`) — the *deployable* form of H1 | adding `r_perc` tightens with real noise | `single_gate_varied_gvf_lpara_perc` | **BUILT + deployed; launch next** |
| H2 | **Observation representation** (noised center-vector + area ≪ 4 corners) | corner-obs tightens beyond `r_perc` | add gate-corner/normal obs channel | untested (fallback if `r_perc` under-delivers) |
| H5 | **Task underspecification** — lone-gate crossing point not well-defined (idea b2) | gate-1 offset *tighter* with a 2nd gate | dual-gate: measure gate-1 cross_offset | untested |
| H6 | **Input regime** — permanently-empty 2nd slot degrades single-gate (idea b1) | dual-gate tightens gate-1 | dual-gate vs single-gate gate-1 offset | untested |

**H1 confirmed → the fix is perception, not reward-shaping.** The clean deployable move is H4 (`r_perc`,
built) — it works *with* the real noise by keeping the gate detectable near the plane. H2 (richer obs)
and the estimator itself (vision-commander domain) are the deeper perception levers if `r_perc` caps out.

H5 and H6 are both tested by the **same** dual-gate experiment (measure the *gate-1* crossing offset
when a second gate is present and co-observed). H2 and H4 are the Swift-alignment fixes.

---

## 5. The disciplined plan (what NOT to rush)

> **STRATEGIC BOTTOM LINE (the one thing to take away).** Two independent literature facts say the
> single-gate 90% target may be *the wrong problem*: (1) Song IROS-2021 shows **N=1 gate observation is
> ~10× more crash-prone than N=2** (23% vs 2.5%) — and our single-gate run is exactly N=1 (empty second
> slot); (2) time-optimal theory says a gate's crossing point is defined by the *sequence*, so a lone
> gate's "correct" crossing is underspecified and dead-centering it optimizes a metric the racing task
> redefines. **Meanwhile our actual single-gate symptom is a *centering* offset (~0.88 m ≈ the field's
> no-`r_perc` baseline), for which the field's proven fix is the perception reward `r_perc` (→0.15 m).**
> So the two highest-value moves are: **(A) add `r_perc`** (fixes the centering symptom we have), and
> **(B) validate at N=2 / dual-gate** (the regime the field actually uses, and the real VQ2). Everything
> below is the sequencing of those two against the running `vglpns` result.


1. **Let `vglpns0/5` finish** — it cleanly splits H1 (noise) from H3 (control). *(running; watcher
   `bginllvbb`.)*
2. **Launch the perception-aware reward `r_perc` (H4) — the #1 evidence-backed lever.** Geles shows it
   takes gate-passing error from **~0.5 m → ~0.15 m** (2–4×), which is *precisely our gap*, and it is
   the clean realization of Fengyou's idea (a) ("reduce the noise arriving to the policy"): a per-step
   `λ·exp(−δ_cam⁴)` bonus for keeping the camera axis on the gate center → a better estimate on
   approach. Implemented + unit-tested + deployed tonight, **ready to launch** (warm from champion, as a
   λ dose-response). *Interpretation depends on `vglpns`:* if `vglpns0` tightens, `r_perc` is the exact
   fix (improves perception); if it floors, `r_perc` still has an attention/centering channel (weaker)
   and H2/H5 rise. Gate the *launch* on `vglpns` reading, not the implementation.
3. **Dual-gate gate-1-offset test (idea b / H5+H6).** Field-motivated: Song's champion runs `N=2` gate
   obs (our exact architecture) and *always* has both slots filled; our single-gate empty-slot regime
   is out-of-design. Measure the **gate-1 crossing offset with a real second gate present**. If it
   tightens → the lone-gate floor is partly an out-of-regime artifact and the crossing point is now
   task-defined → **advance to the real multi-gate regime rather than over-fitting a lone gate.**
4. **Richer per-gate obs (H2)** — add gate-corner/normal channels (Swift/Song use 4 corners; we use a
   noised center-vector). Higher-effort (obs-dim + estimator change); hold unless H1 says perception and
   `r_perc` under-delivers.

**The thing to resist:** another single-knob *speed/parabola* patch. The evidence says lone-gate reward
shaping is spent; the open, field-proven levers are **the perception reward (`r_perc`)**, **the
observation regime (multi-gate / richer per-gate)**, and possibly **the target metric itself** (0.42 m
dead-center may be tighter than champions achieve without `r_perc`, and off-center is the racing line).

---

## 6. Literature notes (sources) — the three champion reward functions

The reward designs of the three leading works, verified against primary sources (Geles read directly
from the PDF; Swift/Song via extraction with the caveats noted):

**Geles et al., "Agile Flight from Pixels without State Estimation" (RSS 2024)** — γ=0.995:
`r = r_prog + r_perc + r_pass − r_cmd − r_crash`
- `r_prog = 0.5 (d_{t−1} − d_t)` — potential, distance to next gate center. **No speed term.**
- `r_perc = 0.025 · exp(−δ_cam⁴)` — **angular perception reward**, δ_cam = angle(camera optical axis,
  drone→next-gate-center). **Every step, not proximity-gated.**
- `r_pass = 1.0 − d_t` **at the crossing** (d_t = distance to gate center) — a **sparse centering-at-
  pass** reward (max at center). *(This is the field's centering term — modest, sparse, for safety.)*
- `r_cmd = 0.0005‖a‖ + 0.0002‖Δa‖²`; `r_crash = −4.0` (floor/collision).

**Song et al., "Reaching the Limit: Optimal Control vs RL" (Science Robotics 2023):**
`r_k = ‖g−p_{k−1}‖ − ‖g−p_k‖ − 0.01‖ω_k‖`; collision −10, finish **+10**. No separate centering term
(implicit in distance-to-center); **no speed term.**

**Swift — Kaufmann et al. (Nature 2023):** `r = r_prog + r_perc + r_cmd − r_crash`; `r_prog =
λ₁(d_{t−1}^Gate − d_t^Gate)`; `r_perc` almost certainly `λ₂ exp(−δ_cam⁴)` (matches Geles — the
extracted `exp(−λ₃δ/4)` is a likely OCR artifact); crash 5.0; **no speed term.** Obs ℝ³¹ = state(15) +
**next-gate 4 corners relative(12)** + prev action(4); **one** next gate.
<https://www.nature.com/articles/s41586-023-06419-4> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/>
· <https://arxiv.org/abs/2406.12505> (Geles) · <https://arxiv.org/abs/2310.10943> (Song)

**Gate-count ablation** — Song, Steinweg, Kaufmann, Scaramuzza, *Autonomous Drone Racing with Deep RL*
(IROS 2021): N=1 → N=2 gates cuts crash ratio 23.0% → 2.5% (Table IV). Center + gate-normal-angle obs.
<https://ar5iv.labs.arxiv.org/html/2103.08624>
**MonoRace** (A2RL 2025 winner, arXiv:2601.15222): current-gate-frame state + next-gate center+yaw.
<https://arxiv.org/html/2601.15222v1>

**Time-optimal / CPC** — Foehn, Romero, Scaramuzza, Science Robotics 2021 (arXiv:2108.04537, waypoint
tolerance-ball `d_tol`=0.3 m — a *region*, not a point); **TOGT** gate-traverser (arXiv:2309.06837):
the min-time line **corner-cuts to gate inner edges, not centers** (7.2% faster); the crossing point is
a function of the gate *sequence*. <https://arxiv.org/abs/2108.04537> · <https://arxiv.org/html/2309.06837v3>

### The quantitative centering baseline (the most important new anchor)

We finally have field numbers for how off-center champion policies actually cross (Geles Tables I/II,
metric = mean gate-passing error, distance from gate center):

| policy | gate-passing error |
|---|---|
| state-based (Song baseline) | **0.45–0.53 m** (sim) / 0.37 m (real) |
| **state-based + perception-aware reward** | **0.12–0.22 m** (sim) |
| pixel-based (Geles) | 0.15–0.38 m |
| Swift "gate margin" | 0.09 ± 0.08 m *(unverified — possibly a fine-tuning-safety context)* |

**This recalibrates the whole problem.** Our **0.88 m L-inf (~0.6 m Euclidean)** is *roughly the field's
baseline* (no perception-aware reward) — we are not catastrophically bad, we are at the ordinary
precision of a gate-progress policy. **The single lever that takes the literature from ~0.5 m down to
~0.12–0.22 m is the perception-aware reward `r_perc`** (Geles: 0.5 → 0.15, a 2–4× tightening). Our
~0.42 m target is *achievable in the field, but only with `r_perc`.* And Song's tradeoff — *"deviation
from the center makes crashes more likely but yields faster lap times"* — confirms centering is a
*speed–safety* tradeoff the racing line deliberately spends (corner-cutting), reinforcing §3 idea (b2).

### Two field facts that reshape the plan

1. **No champion rewards speed explicitly** (Swift, Geles, Song all: progress + discounting only). Our
   explicit `vmax`-clipped progress + `−0.02/step` time + finish-time bonus is **non-standard**. The
   clean form of **idea (a)** is *"delete the explicit speed machinery, match the field"* — not another
   `vmax` cap. (Caveat: our γ=0.9975 is higher/more far-sighted than Geles' 0.995 — load-bearing for
   terminal dominance; keep it.)
2. **Two of three carry an angular perception reward `r_perc` we entirely lack** (H4). It keeps the
   gate centered in view every step → a *better estimate near the plane, exactly where precision
   matters* → this is the concrete mechanism behind Fengyou's *"reduce the noise arriving to the
   policy"* (idea a), realized as a reward rather than a speed change. **Adding `r_perc` (warm from
   champion) is now a top-ranked next lever** — field-proven, cheap, and it targets the plane-approach
   estimate quality that H1/H3 are probing. Our centering-at-pass (the parabola / passage term) is
   already field-aligned (≈ Geles' `1 − d_center`), so that is *not* the gap; the missing piece is the
   *approach-time attention/observation quality*, not the crossing bonus.
