# a4 — LATENCY GEOMETRY: along-track vs in-plane at the gate-4 window

**Role:** reconcile the two parent reports that appear to contradict on latency.
**Method:** `a4_latency.py` (offline, seed 20260613). Inputs: `latency_results.json` (case-C),
gate-4 geometry from `track_map.json`. Output: `a4_latency_results.json`. Numbers reproduce on re-run.

---

## TL;DR — both parents are right; they measured different latencies AND different axes

| | planning-togt-s2 (B_judge §1.3) | vision case-C (latency_results.json) |
|---|---|---|
| latency **source** | **command-side actuation** = 67 ms (2 ticks) | **in-loop vision-compute** L: edge 5.8–15.9 ms, CPU 112.6–125.2 ms |
| what it delays | the **control response** | the **estimate** (fix is stamped older than now by L) |
| error **axis** reported | **cross-track / in-plane** | **full |v·L| magnitude** (≈ all along-track at gate-4) |
| number | **0.77–1.93 mm** | **2.25–3.76 m** (20–30 m/s); **4.25–4.72 m** at 37.7 m/s |
| verdict | in-plane → benign | along-track → big but not a validity threat |

There is **no contradiction**. planning quoted the *cross-track* component of a *command-side*
held-attitude error; case-C quoted the *full magnitude* of the *vision-compute* staleness, which at
gate-4 is **98.4% along-track**. Different latency, different axis. My script reproduces planning's
0.77–1.93 mm exactly (command-side `0.5·g·tan θ·L_cmd²`, θ=2–5°, L_cmd=67 ms) to confirm the anchor.

---

## Gate-4 geometry (from track_map)

g3→g4 leg = **24.39 m**, descent 0.79 m, **tilt 1.9° from horizontal** (≈ level). Along-track unit
(NED) = `[−0.9838, +0.1763, +0.0324]` → **98.4% along −N**. So at the gate-4 window the velocity
(≈37.7 m/s) is essentially along −N, the gate-4 normal. **In-plane (binding-miss) axes = E + D;
along-track axis = N.**

---

## (1) ALONG-TRACK staleness — `v·L` projected on −N (v = 37.7 m/s)

The staleness vector `s = v·L·(along)` is **100.00% along-track** by construction (velocity ∥ leg);
the geometric in-plane projection of `s` is **0.0 m** to machine precision.

| band | L | along-track N | |v·L| |
|---|---|---|---|
| edge p50 | 5.77 ms | **0.217 m** | 0.217 m |
| edge p90 | 15.90 ms | **0.599 m** | 0.599 m |
| CPU p50 | 112.65 ms | **4.247 m** | 4.247 m |
| CPU p90 | 125.23 ms | **4.721 m** | 4.721 m |

This is the case-C "2.3–4.2 m" number recast onto the correct axis. It corrupts **when the plane is
crossed (phase) + the final-approach command**, NOT the in-plane miss. It is also the "last-fix"
distance: even CPU p90 (4.72 m) sits **comfortably under the 10 m last-fix rule** — the rule is fine.

---

## (2) IN-PLANE / cross-track leak of `v·L` (the only way it touches the binding miss)

On a straight leg the staleness leaks in-plane only via (a) a **velocity-heading mis-pointing θ**
(first-order, `v·L·sin θ`) or (b) **path curvature** (second-order, `0.5·v²·κ·L²`). Both are tiny:

| scenario | edge p90 (15.9 ms) | CPU p90 (125 ms) |
|---|---|---|
| well-tracked heading 0.5° | **5.2 mm** | **41.2 mm** |
| curvature R=500 m (≈straight) | 1.8 mm | 22.3 mm |
| **naive-uncompensated worst** (2° held, no rewind) | 20.9 mm | **164.8 mm** |

For comparison, planning's command-side held-attitude cross-track (2–5° over 67 ms) =
**0.77–1.93 mm** — reproduced exactly. (Note: command-side is a *second-order* `0.5·g·tanθ·L²`
acceleration effect → mm; the estimator heading-leak is *first-order* `v·L·sinθ` → cm. They are
physically distinct mechanisms; do not conflate the two θ numbers.)

---

## (3) IN-PLANE VERDICT (the deliverable answer)

**For the in-plane miss, well-tracked (the contact-free flight case: sub-degree velocity heading +
≈straight leg):**

- **Edge latency: BENIGN. ~5.2 mm = 10.5% of the 0.05 m 1-σ bar.** No concern.
- **CPU latency: still sub-bar but NOT negligible. ~41.2 mm = 82% of the 0.05 m bar** (≈27% of the
  0.155 m margin). It does not by itself bust validity, but at the CPU upper bound it eats most of
  the variance budget — which is one more reason the CPU path is unacceptable for racing and the
  edge/GPU path (or rewind) is required.
- **Naive-uncompensated worst case** (a 2° sustained velocity mis-pointing applied in-place, no
  rewind, at CPU latency): **~165 mm = 106% of the 0.155 m margin.** This is the *cost of not
  compensating*, not the realistic flight case (a 2° sustained heading error is 1.3 m/s of lateral
  velocity at 37.7 m/s — itself a tracking failure the contact-free plan forbids).

**Bottom line:** the in-plane miss is set by the **estimator VARIANCE + BIAS** (the FACTS three-fold
question), NOT by latency. Latency at edge is in-plane-benign; at CPU it is in-plane-marginal but the
CPU path is ruled out on the along-track ground anyway. Latency's real cost is **along-track**.

---

## (4) Rewind (horizon ≥ L) / predict-forward — which question does it serve?

- **PRIMARILY the ALONG-TRACK question.** It re-places the fix at capture time, correcting the N
  (phase) coordinate by ~`v·L` (up to **4.72 m at CPU p90**). This fixes plane-crossing timing and
  the final-approach command. This is the headline win.
- **For the in-plane miss it buys little when the leg is well-tracked** (E,D change <~1 cm over L on
  a straight leg). It *additionally* removes the first-order heading-leak term — the naive worst
  case up to ~165 mm in-plane at CPU latency — because the replay uses the true buffered velocity
  heading. So the in-plane benefit is **worst-case insurance** that only materialises under CPU
  latency *and* a mis-pointed velocity; at edge latency the in-plane residual is sub-cm either way.
- **Net:** rewind is justified by the along-track correction. If VQ2 runs on edge HW (5.8–15.9 ms)
  the along-track staleness is only 0.22–0.60 m and even predict-forward (constant-velocity
  extrapolation) would suffice; rewind is the rigorous version and is cheap (<0.5 ms/fix).

### horizon < L divergence risk, tied to the L bands

`RewindKF` (default `horizon_s = 0.5`) **drops any fix older than the horizon and diverges to ~21 m**
if it drops a run of fixes (FACTS / `kf_rewind_buffer.py` docstring: sharpest risk). All four L bands
sit **far below 0.5 s** — margin ≥ **0.37 s even at CPU p90**:

| band | L | horizon margin (0.5 − L) | covered? |
|---|---|---|---|
| edge p50 | 5.8 ms | 0.494 s | ✅ |
| edge p90 | 15.9 ms | 0.484 s | ✅ |
| CPU p50 | 112.7 ms | 0.387 s | ✅ |
| CPU p90 | 125.2 ms | 0.375 s | ✅ |

So horizon<L divergence is **NOT a risk at these L** (0.5 s is ~4× the CPU upper bound). It would
only bite if L spiked past ~0.5 s — a detector stall / dropped-frame burst — which the navigator
should handle with a **max-coast / re-init guard**, NOT by widening the horizon (a wider horizon just
delays the same divergence and costs replay). The horizon requirement is simply **horizon > realized
vision-compute L**, which is satisfied with 3–4× headroom on both HW assumptions.

---

## How this feeds the estimator mission

The <0.05 m 1-σ in-plane bar at gate-4 is **owned by the estimator (variance + per-track bias)**, not
by latency: latency's `v·L` is 98.4% along-track and the in-plane leak is sub-bar when well-tracked
(5 mm edge / 41 mm CPU). Latency does NOT relax or tighten the in-plane variance target. The CPU
latency path is independently unacceptable (4.25–4.72 m along-track staleness + 41 mm in-plane), so
the readiness case requires the edge/GPU detector **plus** rewind-or-predict-forward; with edge HW the
along-track staleness drops to 0.22–0.60 m and latency stops being a first-order concern on any axis.

**Conditionality:** all of this is the case-C (vision-only pose) worst case. If VQ2 streams
LOCAL_POSITION_NED/ODOMETRY (case A/B), pose is pristine and the entire latency question is moot.
