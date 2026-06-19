# VQ2 Vision Model — Commander Report

**Date:** 2026-06-19 · **From:** vision project (manager) · **Re:** first consolidated update — the gate-detector effort, start to now

---

## BLUF

This is the **first full update on the VQ2 vision effort** since it began. Bottom line: the work was much less linear than a single result implies — we spent most of the time clearing **pipeline and methodology failures that were quietly corrupting our own experiments**, not tuning the model. Having cleared them, we now have a **validated new gate detector** (the "clean ensemble") that **beats the Round-1 champion on the two statistically reliable accuracy metrics** (course 76%→82%, p=0.0005; realistic/deployed 62%→82%), trained on clean data, with the win **correctly attributed to a training-objective change, not a data trick**. It is **adopted and deploy-ready**. One gate remains before competition go-live: a **closed-loop flight validation** — your call.

The most important takeaway is not the +6 points. It's that **we now trust the result** — and getting to "trustworthy" is what the journey below was actually about.

---

## The mission

Build the gate detector for the autonomous drone-racing competition (A2RL × DCL). It's **sim-to-sim**: we train on photoreal Blender renders and deploy on the VQ simulator, whose video arrives as degraded, JPEG-compressed frames over the network. The detector finds each gate's corners → a pose solver turns those into a distance/heading → that feeds navigation. **Accuracy of that distance fix is the whole game.** Our incumbent was the "Round-1 champion," which scored ~80% accurate fixes and had proven hard to beat.

---

## The journey — the walls we hit, and how we cleared them

Most of these were **self-inflicted, silent failures** — the experiment looked fine while measuring the wrong thing or training on corrupted inputs. Catching them was the real work.

| # | The wall | Why it mattered | How we cleared it |
|---|---|---|---|
| 1 | **Our scoreboard lied.** We were selecting models by the standard detection score (mAP). | A model with a great 0.835 mAP scored **28%** on actual pose accuracy. We were optimizing the wrong number entirely. | Built a custom **"good-fix" eval** (is the pose-derived distance within tolerance? miss = fail) on a fixed 40 + 204-frame test set. Everything is judged on it now — never mAP. |
| 2 | **We poisoned our own training data.** | New renders meant to *improve* the model came out blurry / low-res / too dim, and a silent fallback rendered procedural junk whenever assets were missing — so bad images entered training unnoticed. | Quarantined the bad render sets; made the non-photoreal fallback a **hard error** instead of a silent one; documented the photoreal pipeline so it's repeatable. |
| 3 | **A hidden bug made our own champion irreproducible.** | A shared random seed in the data splitter **reshuffled existing train/val splits every time we added data**. The champion's 80% silently dropped to ~52% on rebuilds, and we couldn't tell why. | Per-set deterministic seeding, so adding data never disturbs a frozen split. Pinned the champion's exact training list. |
| 4 | **The model couldn't even be scored or flown.** | The detector was upgraded to 8 keypoints, but the scoring and flight code still expected 4 — it crashed on the real model. | An adapter that feeds the **4 inner corners** (the gate opening — exactly what the pose solver uses) to the solver. The 8-keypoint model now scores and deploys, and reproduces the champion's numbers exactly. |
| 5 | **A full week where every improvement lever lost.** | More data, lighter augmentation, heavier augmentation, a tighter loss, a bigger network, test-time tricks — **all failed**. The champion was a stubborn local optimum and we were burning days. | Imposed discipline: **one change per experiment, every "win" must clear a confidence interval.** Learned a key fact — the heavy augmentation is *load-bearing* (it bridges the gap to the degraded VQ video; removing it collapses accuracy to 22%). |
| 6 | **A "win" that turned out to be a mirage.** | An overnight run finally beat the champion — but credited the wrong cause: a batch of substandard images it *believed* were the secret "precision data." Acting on that belief would have institutionalized bad data. | **Human ground-truth caught it**, and timestamp forensics proved it: those images were rendered ~10 hours *after* the champion finished training — it never saw them. The false explanation was retracted, the confounded model retired. |
| 7 | **The champion's famous "80%" was partly luck.** | Its headline precision is a **40-frame, high-variance single draw** — even byte-for-byte identical retrains land ~55%. Training isn't bit-reproducible even with determinism flags set. | Re-anchored the whole program on the **statistically stable metrics** (the 204-frame course score + the realistic/deployed score), which *do* reproduce. Those became the real bar — and they're exactly where the new model wins. |

---

## The breakthrough — how we landed here

Once the pipeline was trustworthy, the actual model improvement was a **training-objective change**, validated cleanly:

- **The insight.** The standard pose loss normalizes corner error by gate size, which *under-weights big, nearby gates* — precisely the frames where you most need sub-pixel corners for an accurate distance. We re-weighted the loss toward the inner corners and added an absolute-pixel accuracy term to fix that blind spot.
- **The trick.** That change creates a tradeoff — dial it one way for pinpoint precision up close, the other for reliable gate acquisition across the course; no single setting won both. So we run **two complementary models together (an ensemble)** and combine their detections, capturing both ends.
- **The proof (confound-free).** On the clean 2,000 images, the new loss beats the standard-loss baseline (course 167 vs 159, p=0.0078), the ensemble beats the champion (below), and — critically — it is **statistically identical to the earlier confounded version** (167 vs 168, p=1.0). That last comparison is the clincher: removing the substandard data changed *nothing*, so the win is the **loss lever**, not the data.

---

## The validated result (independently re-verified on a fresh run today)

| metric | Round-1 champion | Clean ensemble (adopted) | |
|---|---|---|---|
| **Course accuracy** (204 frames — *reliable*) | 76% (155/204) | **82% (167/204)** | **+12, p=0.0005, 12–0** ✅ |
| **Realistic / deployed accuracy** | 62% (25/40) | **82% (33/40)** | **+20 points** ✅ |
| Near-range (<1 m / <2 m) | 95% / 98% | 100% / 100% | no regression |
| Precision <0.5 m (40 frames — *noisy*) | 80% | 87.5% | directional only |

The credible, durable wins are the **course** and **realistic/deployed** rows. The 40-frame <0.5 m number is high-variance for *every* model (see wall #7) and is **not** the basis of the verdict.

---

## Where we are now

- **Adopted:** the clean ensemble is the official VQ2 vision model — `models/gate_clean_ens_precision_L107.pt` + `models/gate_clean_ens_course_L110.pt`; stripped weights, verified to load in a clean flight environment.
- **Cleaned up:** the confounded model is retired; the scientific record (recipe, the test that disproved the bad-data theory) is preserved.
- **Guardrails honored throughout:** the Round-1 champion weights were **never touched** (still dated 2026-06-16); the legacy flight model was never overwritten; no live flight-stack code was changed.

---

## Deploy plan + the one go-live gate

The model is deploy-ready but **not yet wired into the live flight stack** — deliberately, as that's the go-live decision:

1. **Ensemble support in the flight detector** — it runs one model today; the ensemble needs union-of-detections (proven offline). Small, contained change; recommend light dedup before the Kalman filter.
2. **Closed-loop flight validation (the gate)** — the win is validated *per-frame / offline*; confirm it holds in a full sim flight before competition. Latency ~26 ms, well inside the 50 ms budget.
3. **Point the flight stack at the adopted weights.**

---

## Bottom line

The headline is a clean +6-point accuracy gain over Round-1 — but the real deliverable is a **vision pipeline we can finally trust**: an honest scoreboard, reproducible data, a deployable model, and a win we've stress-tested against our own worst-case (a confound we caught and killed ourselves). **Recommend approving closed-loop validation → competition go-live.**

*Drill-downs:* clean-data forensics + A/B + CI in [`REPORT_CLEAN.md`](REPORT_CLEAN.md); model recipe in [`clean_champion/README.md`](clean_champion/README.md); the corrected provenance record in [`new_champion/README.md`](new_champion/README.md).
