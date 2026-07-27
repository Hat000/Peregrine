# Deploy-side GEOMETRIC gate detectability — `--ego-det-geometric` (DEFAULT OFF, UNFLOWN)

Branch `geo-det-2026-07-27`, cut from `aim-offsets-pin-2026-07-27` (`5a0af798`).

---

## TL;DR for the pilot

**Panel knob:** `ego_det_geometric` — a checkbox in the **Ego perception** group, right under
`ego_obs_coast`. **Tick it ON to arm.** CLI equivalent `--ego-det-geometric`.

Everything else stays as it is. It is pinned OFF in `_V1_RECIPE`, so **picking any model clears
it** — arm it *after* choosing the checkpoint, same ordering rule as the aim offset.

It is **not** mutually exclusive with the gate-4/5 obstacle dodge; the two are independent by
construction and there is a test pinning that.

---

## What it does

Training decides every 33 ms whether the gate is *geometrically* visible — it projects 8 gate
keypoints through the camera and requires ≥ 4 in frame — and **zeros `obs[11:16]` the moment the
answer is no**. Deploy substitutes a stopwatch: `det_proxy = (age < ego_det_hold)`, default 0.2 s.

That stopwatch does not reproduce the blackout that matters, because **every fresh fix resets
`age`**. Measured over 899 seam-guarded confirmed passes:

| | |
|---|---|
| passes that ever got a fix inside 1.19 m | **0 of 899** |
| last-sighted range | p10 1.48 · **median 1.78** · p90 2.80 m |
| TRAINING slot0 filled inside 1.0 m | **0 %** |
| WIRE slot0 filled through the whole blind run-in | **65.7 % of approaches** (mean filled fraction 0.872; 94 % at the last tick before the advance) |

So the policy flies the final ~1.8 m — the interval in which every gate outcome is decided — reading
a filled, coasted, possibly mis-locked lever, **in a state it has never been trained on**.

Armed, the det used for masking becomes

```
det  =  (age < ego_det_hold)   AND   gate_detectable_geometric(held_lever, attitude)
```

and slot0 zeros at ~1.8 m instead, which is what training shows the policy.

---

## AND, not REPLACE — the design decision

The brief left this open. **AND**, for a reason that is not aesthetic:

The age proxy is not merely a bad stand-in for geometry. It also carries the one masking event
geometry *cannot* see: **loss of lock**. Training's `det` sits next to an estimator that always has
truth available, so geometry alone is a complete answer there. The wire's detector genuinely fails
— motion blur, dropped frames, a mis-lock — and the gate can be perfectly in frame while nothing
has been measured for a second.

REPLACING the proxy would therefore **delete the blackout cliff at loss-of-lock** and feed an
indefinitely coasted lever whenever the geometry is happy. That is strictly worse than today.

ANDing can only ever mask **more**, never less. Every masking event that fires today still fires;
the blind-run-in blackout is added. There is a test (`test_armed_never_unmasks_a_tick_the_default_path_masked`)
that asserts this over 500 mixed ticks × 2 slots.

---

## Verified

**The training source was available** (`gate_visibility.py` + torch 2.12 in the repo venv), so this
is not a mirror of documented behaviour — it was run head to head against training's own function.

**1. Exact parity with training on straight-in approaches — delta 0.000 m on every case.**
Reproduces the banked training-horizon table to the millimetre:

| geometry | training | this implementation |
|---|---|---|
| head-on, level, centred | 1.290 | **1.290** |
| level, 0.3 m high | 1.185 | **1.185** |
| level, 0.3 m low | 1.550 | **1.550** |
| pitch +10° | 1.780 | **1.780** |
| pitch +24° (flown median) | 1.715 | **1.715** |
| pitch −10° / −24° | 0.930 / 0.705 | **0.930 / 0.705** |
| roll 30° | 1.800 | **1.800** |

All eight are pinned in `tests/test_ego_det_geometric.py` at 5 mm tolerance. Because ±24° pitch is
1.715 vs 0.705 and ±0.3 m offset is 1.185 vs 1.550, **the table also pins every sign** — an inverted
attitude or lever cannot pass it.

**2. Unbiased over realistic random geometry.** 398 random approaches (roll ~N(0,15)°,
pitch ~N(24,10)°, gate yaw vs approach ~N(0,12)°, lateral/vertical offsets ~N(0,0.4) m), each
bisected against training's own `gate_detectable`:

- median cutoff **TRAINING 1.861 m vs DEPLOY 1.861 m**
- per-approach delta **median +0.009 m**, mean +0.002, p10 −0.256 / p90 +0.217 m

**3. Calibrated against the wire.** At the flown median attitude with the shipped boresight live,
the cutoff is **1.72 m** against the measured wire median of **1.78 m**. Pinned to the band
(1.30, 2.40) m — wide enough to survive a boresight recalibration, narrow enough that "fires at
5 m" or "fires at 0.2 m" trips it.

**4. Default-off is proven, not asserted.** `test_default_path_never_calls_the_geometry` **poisons**
`gate_detectable_geometric` to raise, then runs a 200-tick mixed sequence. Comparing numbers could
not prove this; poisoning does. Plus bitwise obs equality over 400 mixed ticks (fixes, gaps past
`det_hold`, re-acquisitions, gate advances, both slots) and an assertion that no new `last_diag`
key appears when off.

**5. No ground truth.** Pure function of the held lever + AHRS attitude. No world pose, no gate map,
no gate id. Pinned.

**6. Round-trip exactness.** At a fresh fix the reconstruction recovers `pose.t_cam_gate` to 1e-12,
so the test evaluates precisely the geometry the detector itself was working with.

**7. Frames.** Uses the TRUE unflipped body throughout — never `obs[3:5]` / `obs[11:14]`. World-up
is `R_b2w_zup[2, :]` (= `Rᵀe_z`), which removes attitude exactly and is yaw-free by construction;
there is a test that pre-multiplying the attitude by any `Rz` returns the identical answer.

**8. REPLAYED OVER THE REAL CORPUS — 564 recorded flights, 36 967 ticks that carried a genuine
detector fix.** This is the check that decides whether the model is fit to ship, because if the
detector produced a fix then the gate demonstrably *was* in frame, and any tick where the model
disagrees is a tick where arming the knob would throw a real measurement away.

*Frames handled per doctrine: `rel_flu` de-injected of the aim offset (`+[0, +lat, −vert]`), true
attitude recovered as `−obs[3] / −obs[4]`, and the test is yaw-free so `Rz` never enters.*

| | |
|---|---|
| ticks with a real detector fix | 36 967 |
| model disagrees ("would mask") | **1 136 = 3.07 %** |
| disagreement at **3 – 23 m** (≈30 000 ticks) | **0.0 – 0.2 % in every single bin** |
| disagreement at 2–3 m | 12.7 % |
| disagreement at 1–2 m | 83.2 % |
| median range of a disagreeing tick | **1.82 m**, median corner count 3.0 of 8 |

**The disagreement is not error — it is the deploy↔training gap itself, and it lands exactly where
training's own replay puts it.** Training masks 100 % of ticks below 1.0 m, 99.3 % below 1.5 m, with
its 50 % crossing at **1.83 m**; the median disagreeing tick here is at **1.82 m** with 3 of 8
corners, i.e. sitting on the 4-of-8 threshold. Beyond 3 m the model and the real detector agree
essentially perfectly across ~30 000 ticks, so **the knob does not blind the policy anywhere except
the band it was built for.**

And the headline effect, on confirmed approaches:

| inside 1.0 m — where training feeds ZEROS on 100 % of ticks | |
|---|---|
| ticks with a filled slot0 **today** | **571** |
| ticks still filled **when armed** | **0 (0.0 %)** |

Overall it newly masks 9.7 % of currently-fed ticks on confirmed approaches (2 850 of 29 393) — a
targeted intervention, not a blanket blackout.

🚩 **AN INSTRUMENT ERROR I MADE AND CAUGHT.** My first replay metric was "closest range at which the
geometric test is still true, per approach", and it returned a median of 3.01 m with a **p90 of
18 m** — which would have meant the mask fires absurdly early on 10 % of approaches. It was the
metric that was broken, not the model: it read the range off the *last-in-time* detectable tick,
and near the gate `rel_flu` is a coasted (sometimes mis-locked) belief whose range can jump, so
drifted and re-locked levers dominated the tail. The per-tick check against `pose_seen` above is
the sound instrument — it compares the model against an actual measurement rather than against
another propagated belief. **Do not quote the 3.01 / 18.01 numbers; they are withdrawn.**

**9. Test suite.** See the run summary at the bottom.

---

## 🚩 Three findings the commander should bank

**(a) THE CLOSED FORM GETS THE RIGHT NUMBER FROM THE WRONG MECHANISM.** I instrumented which corner
leaves through which frame edge at the threshold. **The four BOTTOM corners have already overflowed
the bottom edge long before the cutoff.** The surviving four are 2 inner-TOP + 2 **outer**-TOP, and
the cutoff fires when the **outer-top pair** leaves — through the *top* edge when the gate is high,
and through the *left/right* edges when it is low (the outer ring is 2.7 m wide and overflows
sideways). So what binds at 4-of-8 is the **outer ring**, not the 0.75 m inner half-width that
`|el| + atan(0.75/R) = VFOV/2` uses. That closed form predicted the wire median to +0.04 m, but it
does so by numerical coincidence of scale. **Do not extend it to new geometry and expect it to
hold** — and note its "vertical clips first" story is only half right: it is the top edge and the
horizontal edges that decide, once the bottom has already gone.

**(b) THE VERTICAL ASYMMETRY RUNS THE OPPOSITE WAY TO THE OBVIOUS READING.** Gate LOW (drone above)
stays detectable to **1.185 m**; gate HIGH (drone below) goes blind at **1.550 m**. The intuition
"the camera looks up, so a low gate clips first" is **wrong** — it is the high gate that clips
first, because raising it pushes the outer-top corners out through the top edge. Training does the
same thing, so this is a fact about the shared visibility model, not a deploy bug.

**(c) THE ARMED PATH COSTS +972 µs/tick = 3.30 % of the p50 29.5 ms tick.** Measured, after a hoist
that already cut the per-call cost 477 → 264 µs. This matters because the loop is compute-bound and
**vision freshness is what pays** (94.9 % vs 75.6 % is the whole rate-30 result). A 3.3 % tick
increase is small but it is a **real confound**: compare this arm against a *matched, same-session*
cohort, never against the historical corpus.

---

## Could NOT verify

- **No flight.** This has never been flown. The claim "the policy does better when the blind band is
  masked" is **unmeasured** — this knob is the instrument for measuring it, not evidence for it.
- **No policy replay.** I replayed the *mask* over the corpus (item 8) but did NOT push the
  resulting observations through v19Ws0 / v20Vs0 to measure how far the commanded rates actually
  move. That is the obvious next step before a flight, and the prior says it will be large: an
  earlier replay over 28 936 observations found zeroing slot0 moves `|Δroll cmd|` by a median
  0.47–0.56 rad/s (p95 2.0) at 1.5–2.5 m. **This knob will change the commands materially — it is
  not a subtle trim.**
- **Gate orientation is modelled, not measured.** The wire has no map and PnP `R_cam_gate` is
  documented as the ambiguous term of the fit (IPPE's two solutions differ in tilt sign; it measured
  a near-constant 0.99 apparent area in flight, i.e. no information). I use a square-on
  world-vertical gate whose normal is the horizontal LOS. Cost measured above: unbiased, ±0.25 m at
  the deciles. **This is the single largest modelling assumption.**
- **Occlusion by other gates is not reproduced** (training runs an image-space annulus test; the
  wire has no other-gate geometry). Omitting it can only make deploy *more* permissive, so it never
  fires spuriously — the safe direction.
- **The boresight costs a measured +0.123 m median vs training** (masks that much earlier). It is
  included because undoing it reconstructs `t_cam_gate` exactly; it is a real property of the deploy
  chain, not an approximation. Smaller than the p10/p90 scatter.
- **Outer half-width 1.35 m (training's 2.70) vs the deploy detector's `GATE_OUTER_SIZE_M = 2.72`** —
  a 1 cm difference in half-width, ~1 cm on the cutoff. I chose training's value for exact parity.

---

## ⚠️ Tension with banked memory — read before arming

`[[vision-horizon-fov-2026-07-27]]` says, of this exact asymmetry:

> the deploy knob `ego_obs_coast` already exists and is OFF; do NOT "fix" this by turning it on at
> deploy — that would move the WIRE toward training's zeros, i.e. throw information away. The change
> belongs in TRAINING.

Two things to say honestly:

1. **That sentence has an internal error.** Turning deploy `ego_obs_coast` *ON* **drops** the mask
   and feeds *more*, moving the wire *away* from training's zeros. The direction described is the
   direction of *this* knob, not of `ego_obs_coast`. So the warning, read for its intent, does apply
   to what I built.
2. **The intent stands and I have not refuted it.** Arming this knob does throw information away.
   The banked judgement is that the training arm (`+env.ego_obs_coast=true`, already launched as
   `launch_v21.sh`) is the right side to move.

So what is this for? **It is the hedge and the cheap measurement.** If v2.1 lands and training
coasts, the wire's filled lever becomes correct and this knob should stay off forever. If v2.1 does
not land, deploy's `det` is simply broken and this is the fix. Either way it costs one wire cohort
to find out instead of a 6-hour GPU run, and the deploy contract's *stated* intent — `obs_coast` is
OFF, i.e. "mask like training" — is currently not being honoured by the proxy that implements it.

**I did not change any default. Nothing flies differently until someone ticks the box.**

---

## Confidence

| claim | confidence |
|---|---|
| default-off is byte-identical | **very high** — proven by poisoning, not by comparison |
| the geometry mirrors training's `gate_detectable` | **very high** — 0.000 m on 8 straight-in cases, +0.009 m median over 398 random ones, against the real training source |
| the mask fires at the right range (~1.8 m) | **very high** — agrees with training's own function, with the 899-pass wire measurement, *and* with 36 967 real detector fixes (0.0–0.2 % disagreement everywhere beyond 3 m) |
| the square-on gate model is adequate | **high** — unbiased vs training (±0.25 m per approach) and confirmed on real data: 0 % disagreement with the detector from 3–23 m, disagreement confined to the 1–3 m threshold band where training also masks |
| **arming it will improve gate rate** | **low / untested** — this is a hypothesis with a real prior *against* it (see the tension above). Fly it as an experiment, not as a fix. |

---

## Files

| path | change |
|---|---|
| `src/racer/ego_obs.py` | `gate_detectable_geometric()`, the `GATE_VIS_*` constants, `det_geometric` config field, the AND in `_channels`, `det_geom` / `det_corners` diag |
| `rl/fly_rl.py` | `--ego-det-geometric`, config passthrough, startup print, `n_masked` honesty fix, `ego_obs.jsonl` fields, `meta.json` value |
| `tools/pilot_panel.py` | knob schema entry + `_V1_RECIPE` anti-ride-in pin |
| `tests/test_ego_det_geometric.py` | 26 tests (new file) |

**In flight, watch `det_geom` and `det_corners` in `ego_obs.jsonl`** — the corner count should decay
smoothly from 8 and cross 4 near 1.8 m. If it crosses at 5 m or at 0.3 m, land and report: the mask
is on the wrong geometry and is worse than none. `det_proxy` deliberately still means the age proxy
alone, so every existing analysis of that field is unchanged; the det actually used is
`det_proxy AND det_geom`.
