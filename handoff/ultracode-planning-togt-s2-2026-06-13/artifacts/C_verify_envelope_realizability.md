# Phase C — Adversarial Verification: ENVELOPE + REALIZABILITY lens

**Verifier role:** refute every achievable-time claim from the C1 / C2 prototypes that does not hold
under an honest envelope-realizability accounting. Lens: is the lateral-accel cap correctly tied to
tilt (`a_lat = g·tan(tilt)`)? Is the point-mass time achievable by ANY real tracker, or does it ignore
the k=1.85 structural tracking gap (8.3 s reality vs 4.55 s line)? Is the "achievable" time honestly a
PLANNING bound, not a flight time?

**Fengyou — headline result:** C1's corrected-aero *unconstrained* ceiling (~4.57–4.71 s) is SOUND and
well cross-validated. But C1's **style-respecting (60°) bound of 5.348 s is REFUTED**: the tilt envelope
was applied only to cornering acceleration, not to the total specific-force direction that actually sets
body tilt. When the cap is enforced honestly, the 60° achievable bound is **~9.8 s**, not 5.35 s — which
in turn flips three downstream C1 conclusions (the "0.77 s tilt tax", the "inc5 tax is reward-shaping"
reading, and the "ladder step 1 needs no cone relaxation" claim). C2 is honest as written (it makes no
style-respecting claim; its numbers are 90°-equivalent point-mass bounds).

All prototype numbers reproduced bit-for-bit on rerun (`.venv/Scripts/python.exe`, PYTHONPATH=src).

---

## 1. Constants audit (live from `src/racer/rl_plant.py`) — PASS

| constant | value read | used correctly? |
|---|---|---|
| `COLL_MAP_ACCEL_MEASURED[-1]` (full-stick) | 78.2828 m/s² = 7.983 g | yes (A_UP_MAX) |
| hover-knot accel @ thr 0.2656 | 9.5804 m/s² | yes (≈g for cancel) |
| `QUAD_DRAG_C2_POOLED` | 0.052 /m | yes |
| `QUAD_DRAG_C2_MEASURED` | [[.042,.058],[.055,.055],[.054,.076]] | yes (per-axis in C2 lens 2) |
| `g` | 9.80665 | yes |

Both prototypes read constants live; no hard-coded drift. LAPSE correctly left OFF in both. Drag-wall
arithmetic checks: level top speed `sqrt(A²−g²)=77.7 = c2·v² → v_top = 38.6 m/s` (C1 reports 39.2, TOGT
39.26 — agreement within the n_hat/g_n level-vs-descent bookkeeping). **Drag wall is robust.**

---

## 2. Reproduction — both prototypes rerun, exact

- **C1** (`proto_envelope_topp.py`): line dev ≤ 0.0125 m, length 164.65 m, min radius 29.97 m.
  Tilt sweep: 60→**5.348**, 65→**5.076**, 75→**4.686**, 80→**4.574**, 90→**4.574 s**. Twin probe diverges 37.9 m. ✔ matches C1 result.
- **C2** (`proto_togt_corrected.py`): lens-3 corrected retime **4.996 s**, corrected-no-drag 3.651 s,
  old-linear 5.465 s, expl-TW8 4.990 s; shipped refined laps and 8.48× thrust headroom all reproduce. ✔

---

## 3. THE CENTRAL FLAW — C1's tilt cap is applied to the wrong quantity

### 3.1 What C1 did
C1 caps **only the centripetal (cornering) acceleration**: `a_c = κ·v² ≤ a_lat_max = g·tan(tilt)`. The
**tangential** acceleration (forward push AND braking) is left free to consume the full 78.3 m/s² thrust
ball. The tilt envelope therefore never constrains how hard the drone decelerates into a gate.

### 3.2 Why that is physically wrong
Body tilt is the angle of the **total specific-force vector** `f = a_des − grav` from world-vertical —
NOT the angle of the lateral component alone. Braking and forward accel are horizontal-ish specific
forces too; they tilt the body just like cornering does. `tan(tilt) = |f_horizontal| / |f_vertical|`,
where `|f_horizontal|` combines lateral AND tangential demand.

### 3.3 The smoking gun — C1's own feasibility block
The prototype already computes the realized tilt and reports it but does not act on the contradiction:

| C1 tilt CAP | C1 lap (s) | realized max tilt (deg) | frac over 78.3 ceiling |
|---|---|---|---|
| **60** | 5.348 | **113.8** | 0.022 |
| **65** | 5.076 | **137.0** | 0.014 |
| 75 | 4.686 | 153.4 | 0.000 |
| 80 | 4.574 | 153.4 | 0.000 |
| 90 | 4.574 | 153.4 | 0.000 |

A "60° style-respecting" plan that requires the body to tilt to **113.8°** (past horizontal, inverted)
is not respecting a 60° envelope. The realized tilt comes from braking from ~38 m/s into the κ-limited
corners — exactly the regime the style cone is meant to forbid. The 75/80/90 rows are *consistent*
(their caps are ≥75° or off, so 153° braking tilt is allowed) — only the **60° and 65°** rows are
self-contradictory.

### 3.4 The honest recompute
I rebuilt the TOPP with the tilt cap enforced on the **total** specific-force direction
(`|f_horiz| ≤ g·tan(cap)` AND `|f| ≤ A`, jointly limiting cornering, accel, AND braking):

| tilt cap | **honest** lap (s) | C1 lap (s) | realized max tilt (honest) | error in C1 |
|---|---|---|---|---|
| **60** | **9.83** | 5.348 | 54.5° ✔ | C1 too fast by **4.48 s** |
| **65** | **8.84** | 5.076 | 59.8° ✔ | C1 too fast by 3.76 s |
| 75 | **6.70** | 4.686 | 78.8° (≈cap) | C1 too fast by 2.0 s |
| 80 | **5.44** | 4.574 | 107° (braking still over) | C1 too fast by 0.87 s |
| 90 | **4.60** | 4.574 | 153.6° (cap off) | ✔ matches (no cap) |

At 90° the two agree (4.60 vs 4.574 s) because there is no tilt cap to mis-apply — **this is why C1's
unconstrained ceiling survives**. The divergence grows monotonically as the cap tightens, exactly as a
mis-placed cap would.

---

## 4. Downstream consequences — three C1 findings collapse

### 4.1 "Tilt tax = 0.77 s" — REFUTED → corrected ~5.2 s
C1: 60° tax = 5.348 − 4.574 = 0.77 s. Honest: 60° tax = 9.83 − 4.60 = **~5.2 s/lap**. C1 understates the
kinematic cost of the style cone by ~7×.

### 4.2 "inc5's 2.3–2.9 s RL tax is mostly reward-SHAPING, not kinematic" — REFUTED (backwards)
C1 argued the kinematic cost of 60° is small (0.77 s), so inc5's larger RL tax must be reward-shaping
artifact. The honest number inverts this: honest 60° TOPP = **9.83 s** ≈ inc5 measured **rw_tilt=96 →
9.52 s**. The RL tilt tax is overwhelmingly **real kinematic cost**. C1's reasoning had the causality
backwards — the close agreement between an independent point-mass tilt-capped optimum (9.83 s) and the
RL-flown 9.52 s is strong mutual corroboration that ~60° flight genuinely costs ~9–10 s on this course.

### 4.3 "Envelope-ladder step 1 (rw_tilt 96→48) recovers time WITHOUT relaxing the 60° free-cone" — REFUTED
If 60° genuinely caps the lap near ~9.8 s, then reducing the *weight* of the tilt penalty while keeping
the 60° free-cone hard cannot buy back the time — the cone itself is the binding kinematic constraint.
Honest sweep shows you must **open the cone** to go faster: 65°→8.8 s, 75°→6.7 s, 80°→5.4 s. This is
precisely the inc5 ladder rationale (free-cone 60°→75–80°), and the verification supports the ladder's
existence rather than C1's claim that step 1 alone suffices.

### 4.4 "80° and 90° are identical (drag wall is the sole ceiling)" — PARTIALLY UPHELD
True in C1's model (both 4.574 s). But the honest model shows 80° still forces 107° braking tilt, i.e. an
80° cap is *not* cleanly feasible on hard decels either; honest 80° = 5.44 s ≠ 90° 4.60 s. The "any cap
≥79° never binds" claim holds for **cornering** but not for **corner-entry braking**.

---

## 5. PLANNING-BOUND vs FLIGHT-TIME (the k=1.85 structural gap) — conflation flagged

All four headline times — C1 `lap_time_s` 5.348, C1 90° 4.574, C2 `lap_time_s` 4.996 — are **point-mass
PLANNING bounds**: instantaneous attitude, no body-rate/tilt-rate dynamics, no 67 ms latency, no finite
tracker gain. They are not flight times.

- The known structural datum: the existing geometric tracker needs **k=1.85 (8.3 s)** to track the 4.55 s
  line, and a 54-combo gain sweep finds nothing faster — a *structural* tracking gap, not a tuning gap.
- C1's own twin probe **diverges 37.9 m** on the 60° plan. C1 reads this correctly ("a capable tracker
  is REQUIRED") but applies **no k-penalty** to the reported bound.
- The only realized flight time on the corrected plant is **inc7 monolithic RL: ~9.76 s twin / ~11.45 s
  fresh** — roughly **2× the C1/C2 point-mass bound**. That gap is the unrealized tracker headroom.

**Verdict on conflation:** C1 and C2 are *internally* honest that these are point-mass bounds (limitations
sections say so explicitly). The risk is the `lap_time_s` field name and the "achievable bound" framing:
any consumer who reads 4.6–5.35 s as a *lap they can fly* is conflating a planning bound with a flight
time. **The honest corrected-aero envelope claim is: lower bound ≈ 4.6 s (90° planning), currently realized
≈ 9.76 s (inc7); the tracker-achievable time is unknown and gated on a capable tracker.** The 5.348 s "60°
achievable" headline is doubly wrong — wrong cap (should be ~9.8 s as a planning bound) AND a planning
bound mislabeled as achievable.

---

## 6. What SURVIVES (upheld claims)

1. **Corrected-aero unconstrained ceiling ~4.57–4.71 s is ROBUST.** C1 90° 4.574 s vs independent C++
   TOGT corrected-aero refined 4.714 s (within 3%); v_max 39.2 ≈ 39.26. Thrust binds, drag wall caps
   ~39 m/s; the doubled T/W does NOT buy sub-4 s. This corrects the falsified-linear 4.27 s / 4.55 s.
2. **C2 corrected retime ~4.7–5.0 s for the existing geometry is honest** (it is a 90°-equivalent
   point-mass bound; C2 makes NO style-respecting claim, so no tilt-cap inconsistency). The 4.27 s
   planning bound being 0.4–0.7 s optimistic (linear-plant artifact) is upheld.
3. **Shipped 4.55 s reference line is drag-INFEASIBLE** in its high-speed segments (plans 46–55 m/s;
   corrected caps 37–39 m/s). The decomposed-RL implication (rebuild the offline line on corrected aero
   before decomposition) is upheld and important.
4. **Margin tax ~0.18–0.30 s/lap, plant-independent** (C2) — consistent, upheld.

---

## 7. Corrected number table (honest figures to carry forward)

| quantity | prototype claim | honest corrected | status |
|---|---|---|---|
| corrected unconstrained (90°) planning bound | 4.574 s | 4.57–4.71 s | UPHELD |
| **style-respecting 60° planning bound** | **5.348 s** | **~9.8 s** | **REFUTED** |
| 65° planning bound | 5.076 s | ~8.8 s | REFUTED |
| 75° planning bound | 4.686 s | ~6.7 s | REFUTED |
| 60° kinematic tilt tax | 0.77 s | ~5.2 s | REFUTED |
| inc5 RL tax interpretation | "mostly reward-shaping" | mostly KINEMATIC (9.83≈9.52) | REFUTED |
| C2 corrected geometry retime | 4.996 s | 4.996 s (90°-equiv bound) | UPHELD |
| drag-wall top speed | ~39 m/s | ~38.6–39.3 m/s | UPHELD |
| shipped 4.55 s line | drag-infeasible | drag-infeasible | UPHELD |
| **all reported times are FLIGHT times?** | implied by `lap_time_s` | PLANNING BOUNDS only | flag conflation |

---

## 8. Caveats on my own honest recompute
- My tilt-capped TOPP enforces `|f_horiz| ≤ g·tan(cap)` against an assumed vertical specific force ≈ g
  (altitude-hold). On the 26 m descent the vertical reserve is slightly less than g, so the honest 60°
  number (9.83 s) is itself a mild *lower* bound — the true 60° kinematic cost could be marginally higher,
  which only strengthens the refutation.
- It is still a point-mass bound (no attitude-rate transient), so like C1/C2 it is optimistic vs a flown
  time. The 9.83 s ≈ inc5 9.52 s agreement is therefore notable — and consistent with inc5 being a real
  flight that nonetheless lands near the point-mass tilt-capped optimum (the tilt cone, not tracking, is
  the dominant cost in that regime).
- Pooled isotropic c2 used; per-axis (climb 0.076) would shift the descent braking slightly. Second order.

---

## 9. Bottom line for the S2 architecture decision
The verification does NOT change the monolithic-first recommendation, but it sharpens the stakes:
- The corrected-aero **prize is ~4.6 s (90° planning bound)**, ~2× faster than inc7's 9.76 s — a real
  structural prize, confirmed.
- BUT capturing it **requires opening the style cone past 60°** (honest 60° ≈ 9.8 s ≈ where inc7 already
  is). Speed below ~9 s is gated on envelope relaxation, not just a better tracker. This is the inc5
  ladder, and the verification says the ladder is load-bearing, not optional.
- Whichever architecture is chosen, the headline "achievable" must be reported as a **planning bound with
  the tilt cap honestly applied**, never as a flight time, and never as a 60°-respecting 5.35 s.
