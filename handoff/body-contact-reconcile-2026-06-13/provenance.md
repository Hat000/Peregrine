# Body-contact reconcile — PROVENANCE (2026-06-13)

Fengyou — this is the provenance leg of the body-radius / gate-4-margin re-derivation. Scope: read
`rl/contact_true_eval.py` end to end, establish EXACTLY how the drone body is modeled for contact and
where every load-bearing constant (0.38 / 0.33 body_radius, 0.75 HALF_OPEN, 0.30 frame-depth) comes
from; cross-check against the spec PDF `260508_Technical_Spec_0002.pdf`. No `src/` or `memory/` edits.

Flags: **[M]** measured/extracted-verbatim, **[X]** extrapolated/derived-here, **[A]** assumed.

---

## 1. How `contact_true_eval` models the body for contact — VERBATIM

The body is **NOT a box or a volume**. It is a **single scalar radius `body_radius` that inflates the
gate aperture**, applied as an **L-inf (square) halo in the gate plane** — never as a disk, never swept
through frame depth as a body. The drone's own translation is a **point** (`prev_ned`, `cur_ned`); all
"size" lives in the band inflation. **[M]**

Core geometry, `rl/contact_true_eval.py:_score_gate` (lines 99–147), the load-bearing lines:

```python
# rl/contact_true_eval.py:121-130
w2g = _R_W2G if gate_yaw is None else _gate_rotmat_w2g(float(gate_yaw[gate]))
half_in  = _HALF_OPEN  - body_radius          # inner pass half-width shrinks by r
half_out = _HALF_OUTER + body_radius          # outer frame edge grows by r
prev_rel = w2g @ (prev_ned * _FLIP - gate_pos_zup[gate])
cur_rel  = w2g @ (cur_ned  * _FLIP - gate_pos_zup[gate])
if frame_depth > 0.0 and slab_frame_hit_np(prev_rel, cur_rel, half_in, half_out, frame_depth):
    return "collision", max(abs(cur_rel[1]), abs(cur_rel[2]))
```

And the plane-crossing branch (lines 138–147): interpolate the segment to the gate plane (`x=0` in gate
frame), take `linf = max(|y|, |z|)`, then classify `pass` iff `linf < half_in (= 0.75 - r)`, `collision`
iff `half_in <= linf <= half_out`, else `miss`. **[M]**

**Exact body model statement:** the body is a **square (L-inf) bounding halo of half-width `body_radius`
in the gate's (y,z) plane**, equivalent to inflating the gate frame inward by `r` on each side. It is an
L-inf disk (a square of half-side r), **not** a Euclidean disk and **not** a 3-D box. There is **no
attitude/silhouette term** — the model is attitude-agnostic; `body_radius` is the *only* knob that
encodes body extent, and the recorded `linf` is **radius-invariant geometry** (only the pass band
shifts with r — confirmed by the docstring of `per_gate_margin_stats`, lines 338–341). **[M]**

**How frame-depth enters** (`slab_frame_hit_np`, imported from `offline_rollout.py:121-148`): the gate
frame is treated as **material occupying `{ |x| <= frame_depth, half_in <= Linf(y,z) <= half_out }`** in
the gate frame. The straight segment prev_rel→cur_rel is tested for intersection with that volumetric
slab exactly (clip the segment to `|x|<=depth`, then check whether the L-inf(y,z) range over that
sub-segment intersects `[half_in, half_out]` — via convexity of L-inf(t), candidate t's at the
axis/diagonal crossings). This catches **strikes that happen BEFORE the plane** (steep approaches) and
crossings that thread the inflated pass band in-plane but clip frame material off-plane. **frame_depth
inflates the gate-NORMAL (x) extent of the *frame*, not the body**; it does NOT change `half_in`, does
NOT change the recorded plane-crossing `linf`, and therefore does NOT move the headline in-plane pass
margin — it only widens the volumetric pre-plane catch window (strictly more conservative). **[M/X]**

Cross-validation already in the codebase: `_score_gate(body_radius=0, frame_depth=0)` reproduces the
legacy `gate_event()` exactly (`tests/test_contact_true_eval.py` legacy-parity), and the numpy
`slab_frame_hit_np` is cross-checked against the torch `slab_frame_hits` in
`tests/test_contact_geometry.py`. **[M]**

---

## 2. Provenance of every constant

| Constant | Value | Where defined | Provenance |
|---|---|---|---|
| `BODY_RADIUS_NOM` | 0.33 | `rl/contact_true_eval.py:51` | **Hardcoded** = "nominal mid-point of [0.28, 0.38] DR range" (comment). **[M]** |
| `--body-radius` worst | 0.38 | CLI default uses NOM; 0.38 is the **DR upper bound** | Hardcoded DR band, see below. **[M]** |
| DR band `[0.28, 0.38]` | — | `rl/peregrine_racing.py:160-164` (`body_radius_lo/hi`, `U[lo,hi]` per reset) | **EMPIRICALLY CALIBRATED, not geometric.** **[M]** |
| `FRAME_DEPTH_NOM` | 0.30 | `rl/contact_true_eval.py:52`; default `--frame-depth 0.30` | Hardcoded "inc7 contact-true frame extrusion"; doctrine value (`peregrine_racing.py:169`). **[M]** |
| `_HALF_OPEN` | 0.75 | `rl/offline_rollout.py:65` | "1.5 m inner opening, L-inf half-width" — **= spec inner 1500mm / 2. Confirmed.** **[M]** |
| `_HALF_OUTER` | 1.36 | `rl/offline_rollout.py:66` | "2.72 m outer frame" — spec says **2700mm (1.350)**; code is 2720mm. +0.010 m discrepancy. **[M]** |
| `PASS_BAND_NOM` | 0.42 | `rl/contact_true_eval.py:53` | Derived `0.75 - 0.33`. **[X]** |

**The justifying comment for the body radius (the crux), `rl/peregrine_racing.py:160-164` VERBATIM:**

> `+env.body_radius_lo` / `+env.body_radius_hi` — per-env body radius r ~ U[lo, hi], resampled at
> every reset (the halo is not precisely known and is unobservable, so the policy trains to the sampled
> worst case). Pass band L-inf < 0.75 - r; frame band [0.75 - r, 1.36 + r]. **Doctrine band [0.28, 0.38]
> m brackets the measured live contact offsets (corner-pass probe contact at 0.60; live crashes
> 0.37-0.49 on steep descents).**

And `rl/peregrine_racing.py:153-156`:

> The live sim collides the drone's BODY (**rotor halo ~0.3 m**) with a VOLUMETRIC frame — **live
> standing crashes terminated at L-inf 0.37-0.49 m**, positions the 0.75 point-mass model scores as
> comfortable passes, and strikes occur up to 0.5 m BEFORE the plane on slope-0.45 approaches.

**Conclusion on 0.38:** it is **NOT a flat chassis half-diagonal**. It is an **effective contact halo
fit to live crash data** — chosen as the upper edge of a DR band whose stated purpose is to *bracket*
the L-inf offsets at which the real sim logged contacts (live steep crashes 0.37–0.49; corner-pass
probe contact at 0.60). The "rotor halo ~0.3 m" language is why the *band* sits near 0.3; the 0.38 cap
is the worst-case tail of that empirical band, deliberately conservative. **[M]**

---

## 3. Spec PDF extraction — `260508_Technical_Spec_0002.pdf` (VADR-TS-002, issue 00.02, 2026-05-08)

Extracted via `pdftotext -layout` (Bash, version 4.00). Verbatim:

**§3.6 Drone chassis [M]:** Width 280mm · Length 280mm · Height 160mm. **Confirmed 280×280×160 mm.**

**§3.7 Gate dimension [M]:**
- Gate boundaries (OUTER): Width **2700mm** · Height **2700mm** · Depth **260mm**.
- Gate inner square: Width **1500mm** · Height **1500mm** · Depth **260mm**.

**Propellers / rotors:** **NOT MENTIONED ANYWHERE in the spec.** §3.6 lists only the chassis box; there
is no rotor diameter, motor-to-motor span, or prop-tip envelope. So **prop_span is UNKNOWN from the
spec** — the spec gives no body extent beyond the 280×280 chassis footprint. The code's "rotor halo
~0.3 m" is an *internal empirical inference*, not a spec figure. **[M for "absent"; A for any prop value]**

**Spec-vs-code discrepancies found (flag these — none rescue or break the headline by itself):**
- Inner opening: spec 1500mm ⇒ half 0.750 m == code `_HALF_OPEN 0.75`. **MATCH.** **[M]**
- Outer: spec 2700mm ⇒ half 1.350 m vs code `_HALF_OUTER 1.36` (2720mm). **+0.010 m, code slightly
  generous on the outer frame; affects only the outer collision/miss boundary, not the inner pass
  margin.** **[M/X]**
- Frame depth: spec 260mm vs eval default `--frame-depth 0.30` (300mm). **+0.040 m, code more
  conservative; widens only the volumetric pre-plane catch slab, not the in-plane pass band.** **[M/X]**

---

## 4. Geometry re-derivation — does 0.38 survive as a *geometric* radius? (NO)

All computed here from the confirmed 280×280×160 chassis + gate-4 posture. **[X]**

- Flat half-width W/2 = **0.140 m**. Flat half-diagonal √((0.14)²+(0.14)²) = **0.198 m**.
- **Tilted projected L-inf half-extent on the gate plane** (8 box corners rotated by the gate-4 approach
  posture roll≈55°, pitch≈−38°, projected out the gate normal, L-inf of the (y,z) footprint):
  **0.213 m** (worst over crab-yaw too: 0.213 m). Pitch-only −38°: 0.149 m; roll-only 55°: 0.161 m.

So **pure rigid-body geometry, even at the tilted gate-4 posture, tops out at ~0.21 m** — the tilt
inflates the flat 0.198 m to only 0.213 m (+0.015 m). **The 0.38 cannot be justified as chassis
geometry; ~0.40–0.45 of it would have to be rotor/prop halo + collision-margin, which the spec does not
document.** The legitimate tilt argument is REAL but SMALL (≈0.015 m), nowhere near the 0.18 m gap
between 0.21 and 0.38. **[X]**

**Independent empirical cross-check of the *effective* radius (these are CONTACTS, so they set a lower
bound on r_eff):**
- Corner-pass probe: `memory/...` table shows ≤0.64 m → gate advances *clean*; **0.60 m → gate advances
  WITH a contact event**. Under the rules (contact = invalid run) the usable boundary is the *contact*
  threshold. A contact at L-inf 0.60 ⇒ r_eff ≈ 0.75 − 0.60 = **0.15 m** at that (likely near-level)
  geometry; the ≤0.5 m clean / 0.6 m clipped framing ⇒ boundary ~0.55 ⇒ r_eff ≈ **0.20 m**. **[M/X]**
- Live steep-descent crashes terminated at L-inf **0.37–0.49** ⇒ r_eff ≈ 0.75 − 0.49 = **0.26** up to
  0.75 − 0.37 = **0.38** at the *steep* posture. **[M/X]**

The empirical r_eff is **posture-dependent**: ~0.15–0.20 m near level (corner-pass probe), but **0.26–
0.38 m on steep descents** — which is exactly the gate-4 regime (drag-hold pitch −38°, crab/roll 55°).
This is why the doctrine put 0.38 at the *top* of the band: the steep-approach contacts are the binding
ones, and they sit far above what rigid-body projection predicts (the extra is rotor wash / prop disk /
aero-blade strike envelope, unmodeled and unobservable). **[X]**

---

## 5. Bottom line for the margin re-derivation (handed to the margin-closure leg)

Gate-4 simstart recorded in-plane `linf` is **radius-invariant = 0.215 m** (back-out: pass_band 0.37 −
margin 0.155). Re-scored against candidate radii (margin = (0.75 − r) − 0.215): **[X]**

| r | source/character | gate-4 margin | verdict |
|---|---|---|---|
| 0.198 (flat half-diag) | geometry | +0.337 m | closes easily |
| 0.213 (tilted silhouette) | geometry, gate-4 posture | +0.322 m | closes easily |
| 0.25 | geometry + small halo | +0.285 m | closes |
| 0.30 ("rotor halo ~0.3") | code prose | +0.235 m | closes |
| 0.33 (nominal) | DR mid | +0.205 m | closes |
| **0.38 (worst-case)** | **DR top, steep-crash-fit** | **+0.155 m** | **closes, thin (headline)** |

NB at the *recorded simstart linf* the gate-4 margin is **positive at every radius** (the headline
"+0.155 m" is itself already positive). The "does NOT close offline" headline is therefore about the
**worst-case p90 tail under estimator error**, not the nominal pass. Against the p90 in-plane tails:
**[X]**

| r | warm p90 0.203 | cold p90 0.234 (0 bias) | cold p90 0.338 (1.4° bias) |
|---|---|---|---|
| 0.213 (geom) | +0.332 | +0.301 | +0.197 |
| 0.30 | +0.247 | +0.216 | +0.112 |
| 0.33 | +0.217 | +0.186 | +0.082 |
| **0.38** | **+0.167** | **+0.136** | **+0.032** |

**So the prompt's hypothesis is directionally CORRECT but its mechanism is wrong:** halving the radius
roughly *doubles* the pass band slack (0.37→0.55), and at a geometry-justified r≈0.21 even the
worst cold+attitude-bias p90 (0.338) closes with **+0.197 m** instead of the **+0.032 m** at r=0.38.
The CANNOT-SETTLE-OFFLINE verdict is **load-bearing only on the choice r=0.38**, and **0.38 is an
empirically-fit steep-crash halo, not a geometric body size** — the spec documents no body extent
beyond 280×280 (no props), so 0.38 is *defensible as conservative DR* but *not derivable from spec
geometry*. The margin-closure leg should re-run `contact_true_eval --body-radius` across {0.21, 0.25,
0.30, 0.33, 0.38} and report the verdict as a *function of which radius you trust*, NOT as a single
number. The honest statement is "closes for any r ≲ 0.30; thin-but-positive at 0.33; knife-edge at the
steep-crash-fit 0.38 only under measured attitude bias." Distrust 0.38 as geometry; do NOT adopt 0.20
as truth either — the steep-descent contacts (0.37–0.49) are real and posture-matched to gate-4. **[X]**

---

### Files (absolute)
- Code: `C:\Users\Fengy\Downloads\Projects\Anduril\rl\contact_true_eval.py` (`_score_gate` 99–147;
  consts 51–53; CLI 528–531)
- Geometry helpers: `C:\Users\Fengy\Downloads\Projects\Anduril\rl\offline_rollout.py` (`_HALF_OPEN`
  65, `_HALF_OUTER` 66, `slab_frame_hit_np` 121–148)
- Provenance prose: `C:\Users\Fengy\Downloads\Projects\Anduril\rl\peregrine_racing.py` (153–169 body
  model + DR band justification; `slab_frame_hits` torch 286–313)
- Spec: `C:\Users\Fengy\Downloads\Projects\Anduril\260508_Technical_Spec_0002.pdf` (§3.6, §3.7);
  text dump `C:\Users\Fengy\Downloads\Projects\Anduril\handoff_spec_dump.txt`
