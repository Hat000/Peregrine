# Empirical effective contact radius — re-derivation from flight/contact data (2026-06-13)

Fengyou — this is the EMPIRICAL-RADIUS leg of the body-radius / gate-4-margin re-derivation. Question:
from our OWN flight/contact data, what is the empirically-implied effective contact radius `r_eff`, and
is that data POSTURE-COMPARABLE to the gate-4 tilted approach (drag-hold pitch ≈ −38°, crab/roll ≈ 55°)?

Flags on every number: **[M]** measured / logged-verbatim · **[X]** extrapolated/derived-here · **[A]** assumed.
No `src/` or `memory/` edits. Sibling leg: `provenance.md` (geometry + spec extraction).

`r_eff` is defined so that the live sim logs a body↔frame **contact** when the point-mass in-plane
crossing offset `L-inf` reaches `HALF_OPEN − r_eff`, i.e. **`r_eff = HALF_OPEN − L-inf_at_contact`**,
with `HALF_OPEN = 0.75 m` (spec inner opening 1500 mm / 2; confirmed in `provenance.md` §3). **[M]**

---

## TL;DR (the headline finding of THIS leg)

**`r_eff` is strongly posture-dependent, and the prompt's empirical anchor is the WRONG posture for gate-4.**

| Data source | Gate | Posture | Speed | `r_eff` implied | Posture-comparable to gate-4? |
|---|---|---|---|---|---|
| Corner-pass probe (0.6 m clip / 0.5 m clean) | **gate 0** | **near-level CTBR bridge** | **low (~3 m/s)** | **≈0.18–0.20 m** (bracket 0.11–0.25) | **NO** |
| Standing gate-3 steep crashes (L-inf 0.37–0.49) | **gate 3** | **banked/climbing, tilt ≈55°** | **high (12–18 m/s)** | **≥0.26–0.38 m** | **YES (closest available)** |

- The "0.6 m clipped, ≤0.5 m clean" anchor that motivated the whole re-derivation is a **gate-0,
  near-level, low-speed CTBR-bridge** event. At THAT posture `r_eff ≈ 0.18–0.20 m` — which does indeed
  roughly HALVE 0.38 and would roughly double the error budget. **But it is not posture-comparable to
  gate-4.** **[M for the events, X for r_eff]**
- The **only posture-matched empirical evidence** we have for the gate-4 regime is the set of 4 standing
  **gate-3** crashes, which logged contacts at **L-inf 0.37–0.49 m** on a high-tilt (~55°) high-speed
  approach. Those contacts imply `r_eff ≥ 0.26 m`, and the tightest one (0.37) implies `r_eff ≥ 0.38 m`.
  **At the relevant posture the data CORROBORATES the 0.38 worst-case, not the 0.18–0.20 one.** **[X]**
- **Net verdict for this leg: the empirical data does NOT support halving the radius at gate-4.** It
  supports `r_eff ≈ 0.18–0.20 m` *near level* and `r_eff ≈ 0.26–0.38 m` *at the tilted, fast posture that
  gate-4 actually flies*. So the optimistic-radius hypothesis is **real but mis-applied**: it would
  rescue a near-level gate, but gate-4 is the steep one.

---

## 1. PRIMARY SOURCE — the corner-pass probe (the "0.6 m clip / 0.5 m clean" anchor)

The memory line that motivated the prompt
(`memory/project_phase2_rl_vision_decisions.md` §CORNER-PASS-PROBE, and the doctrine restatement
"corner-pass 2026-06-11: contact logged at 0.60 m") traces to a LIVE probe. The authoritative primary
record is **`handoff/shadowpc-inc5-live-2026-06-11/WRITEUP.md` §5** (the live run; the inc4 writeup §4 is
offline geometry only — superseded by the live measurement). VERBATIM from §5: **[M]**

| offset label | target Δy | **actual** crossing offset from gate centre | gate_index advance | **hard collisions** |
|---|---|---|---|---|
| center | 0.00 m | ≈ 0.04 m | YES | **0** |
| off0.50 | 0.50 m | **≈ 0.50 m** | YES | **0 (CLEAN)** |
| off0.60 | 0.60 m | **≈ 0.64 m** | YES | **1 (CONTACT)** |
| off0.74 | 0.74 m | NAV OVERSHOOT to ≈1.98 m | NO | 1 (outer-aperture strike — nav precision artefact, NOT a clean threshold test) |

**Three load-bearing facts about this probe (all [M], from §5 method + table):**
1. **Gate = GATE 0.** "Live CTBR crossings of gate 0, varying the NED-y target offset from gate center.
   Gate 0 center: NED (−23.298, −0.400, −1.392)."
2. **Driver = the CTBR geometric BRIDGE, not the RL policy.** "the bridge naturally approaches the offset
   target." The RL policy was unusable for targeted offsets (transfer had failed). The bridge is the
   slow, near-level launch controller.
3. **It is a pure lateral (Δy-only) offset; L-inf = Δy.** Near-level, low approach speed (gate-0 is the
   first gate off the standing/bridge launch, ~3 m/s class, NOT the 12–18 m/s mid-course regime).

### `r_eff` from the corner-pass probe (gate 0, near-level) — `r_eff = 0.75 − L-inf` [X]

- **CLEAN at L-inf ≈ 0.50 m** (0 collisions) ⇒ body did **not** reach frame ⇒ **`r_eff ≤ 0.75 − 0.50 = 0.25 m`**.
- **CONTACT at L-inf ≈ 0.64 m** (1 collision) ⇒ body **did** reach frame ⇒ **`r_eff ≥ 0.75 − 0.64 = 0.11 m`**.
- ⇒ near-level gate-0 `r_eff` is **bracketed to (0.11, 0.25] m**. The clean/contact boundary lies in
  (0.50, 0.64]; its midpoint 0.57 ⇒ **`r_eff ≈ 0.18 m`**. The prompt's own framing (≤0.5 clean / 0.6
  clip → boundary ≈0.55) gives **`r_eff = 0.75 − 0.55 = 0.20 m`** — consistent. **[X]**

**⚠️ The prompt mis-states the probe slightly (does not change the conclusion).** The probe did NOT show
"0.6 m clipped while ≤0.5 m clean" as a tight pair; it showed **CLEAN at actual 0.50** and **CONTACT at
actual 0.64** (the 0.60 LABEL overshot to 0.64 actual). So the clean/contact transition is bracketed to
**(0.50, 0.64]**, not pinned at 0.55. `r_eff` near level is therefore **0.18–0.20 m central, 0.11–0.25 m
bracket** — still ~half of 0.38, still near the rigid-body geometric prediction (`provenance.md` §4:
flat half-diagonal 0.198 m). **[M/X]**

---

## 2. POSTURE — is the corner-pass probe comparable to the gate-4 approach? (NO)

This is the decisive question the prompt flagged, and the answer flips the naive reading.

| Dimension | Corner-pass probe (gate 0) | Gate-4 approach (the binding gate) |
|---|---|---|
| Pitch | ≈ level (CTBR bridge, low launch) **[X]** | drag-hold **≈ −38°** (memory) **[M]** |
| Roll / crab | ≈ level **[X]** | crab/roll **≈ 55°** (frame-audit confirmed; trained racing style, tilt ≈54.8°) **[M]** |
| Speed | low (~3 m/s, gate-0 launch) **[X]** | high (mid-course; 12–18 m/s climb-bin) **[M]** |
| Driver | CTBR bridge geometric controller **[M]** | RL policy at trained tilt cruise **[M]** |
| Projected silhouette | ≈ flat footprint (half-diag 0.198 m) **[X]** | tilted silhouette inflated to ≈0.213 m (`provenance.md` §4) **[X]** |

A near-level body presents its **flat footprint** (half-diagonal ≈ 0.198 m) to the gate plane; a body at
pitch −38° / roll 55° presents a **larger tilted silhouette** (≈0.213 m geometric, `provenance.md` §4) AND
— critically — flies into **rotor-wash / prop-disk / blade-strike** regimes that the rigid chassis box does
not capture. The corner-pass probe samples the *small-silhouette, low-energy* corner of the contact-radius
surface; gate-4 sits at the *large-silhouette, high-energy* corner. **They are not interchangeable. [X]**

---

## 3. POSTURE-MATCHED empirical evidence — the standing gate-3 steep crashes

The closest posture-matched data we own is the gate-3 standing-crash set. Primary source
**`memory/project_rl_increment_history.md:383`** (root-cause reframe) VERBATIM: **[M]**

> "All 4 standing gate-3 crashes terminate at L-inf **0.37–0.49 m** with mid-range commands (rate p95 ≈
> 0.5 rad/s vs 3.14 cap, 84% authority unused, zero saturation) — positions the trainer calls comfortable
> passes."

Cross-confirmed in `handoff/laptop-training-doctrine-2026-06-12/WRITEUP.md:98-99` ("live crashes at
0.37–0.49 on a 0.45-slope descent") and `rl/peregrine_racing.py:153-156` prose ("live standing crashes
terminated at L-inf 0.37-0.49 m"). **[M]**

**Posture of these events** (why they ARE the gate-4-comparable set): mid-course, **banked/climbing**, the
trained **crab ≈55° / tilt ≈54.8°** racing posture (`memory/index_control_sim.md:32,51`;
`project_phase2_rl_vision_decisions.md:1150` "Twin reproduces roll +41.5°/tilt +54.8°"), in the **12–18 m/s
climb-bin** (`project_phase2_rl_vision_decisions.md:1149`). This is the SAME high-tilt, high-speed family
as gate-4 (gate-3 and gate-4 are adjacent post-launch climb gates). **[M/X]**

### `r_eff` from the steep crashes (gate 3, tilted/fast) — these are CONTACTS ⇒ LOWER bounds on `r_eff` [X]

- contact at **L-inf 0.49** ⇒ `r_eff ≥ 0.75 − 0.49 = 0.26 m`
- contact at **L-inf 0.37** ⇒ `r_eff ≥ 0.75 − 0.37 = 0.38 m`  ← the tightest crash implies the 0.38 cap

So at the tilted/fast posture, `r_eff` is **bounded below by 0.26–0.38 m**. (These are lower bounds: the
true `r_eff` could be even larger — these crashes only tell us the body reached the frame *by* that
L-inf, not that it would have been clean any tighter.) **This is exactly why the inc7 doctrine put 0.38 at
the TOP of the DR band [0.28, 0.38]** — the steep-approach contacts are the binding ones and they sit far
above the rigid-body projection (`provenance.md` §4: geometry tops out at 0.213 m). The extra ≈0.17 m is
unmodeled rotor/aero envelope, undocumented in the spec (spec lists chassis 280×280×160 only; no props —
`provenance.md` §3). **[M/X]**

---

## 4. Reconciliation — does the empirical anchor overturn the gate-4 headline? (NO, at gate-4)

The naive chain in the prompt ("0.6 m clip ⇒ r_eff ≈ 0.20 ⇒ budget doubles ⇒ gate-4 closes") is
**arithmetically correct but applies the wrong-posture radius to gate-4.**

Sensitivity of the gate-4 SIMSTART margin (recorded in-plane `linf = 0.215 m`, radius-invariant;
`margin = (0.75 − r) − 0.215`): **[X]**

| `r_eff` (source) | posture it came from | gate-4 margin @ recorded linf 0.215 |
|---|---|---|
| 0.18–0.20 (corner-pass) | **near-level gate 0 — NOT gate-4** | +0.335 to +0.355 m (would close easily) |
| 0.21 (tilted geometry) | gate-4 silhouette, rigid body | +0.325 m |
| 0.26 (steep-crash, loosest) | **gate-4-comparable** | +0.275 m |
| 0.33 (DR nominal) | DR mid | +0.205 m |
| **0.38 (steep-crash, tightest)** | **gate-4-comparable, worst-case** | **+0.155 m (the headline)** |

**Reading:** if you trust the *near-level* corner-pass radius (0.18–0.20), gate-4 closes with huge slack.
But the *posture-matched* steep-crash data says `r_eff ≥ 0.26 m` and the tightest crash says `≥ 0.38 m`,
so at gate-4 you should NOT drop below ~0.26 m, and the conservative 0.38 is **empirically anchored, not a
geometry guess.** The empirical data of THIS leg therefore **does not overturn** the worst-case headline at
gate-4; it explains WHY 0.38 was chosen (steep-crash fit) and shows the optimistic radius belongs to a
different posture.

**What this leg DOES legitimately soften** (hand to the margin-closure leg, do not over-claim here):
- The 0.38 is a **lower-bound-derived worst case** (contacts only prove `r_eff ≥` that). It is defensible
  as conservative DR but is **one tightest-of-four** crash, not a tight central estimate. A more honest
  central gate-4 `r_eff` from posture-matched data is **~0.26–0.33 m** (loosest steep crash up to DR
  nominal), where gate-4 margin is **+0.205 to +0.275 m** — comfortably positive at the *recorded*
  nominal crossing. **[X]**
- The "does NOT close offline" verdict is about the **p90 tail under estimator error** (warm 0.203 /
  cold 0.234 / cold+1.4°-bias 0.338), NOT the nominal pass. Even there, the radius choice is the lever:
  at posture-matched r≈0.26 the worst cold+bias p90 0.338 gives margin **+0.152 m**; at r=0.38 it gives
  **+0.032 m** (knife-edge). The honest framing is **"closes for any posture-matched r ≲ 0.30; knife-edge
  only at the worst-case-fit r=0.38 under measured attitude bias."** Re-derived in detail by the
  margin-closure / refute legs in this same handoff dir. **[X]**

---

## 5. Caveats / what would tighten this (none available offline today)

- **No DIRECT gate-4 contact event exists in our logs.** The binding empirical record is gate-3
  (posture-comparable, adjacent climb gate), plus the gate-0 corner-pass (not comparable). A live gate-4
  offset-sweep at the trained tilt/speed would measure `r_eff` at the exact binding posture — we do not
  have it. **[M: absence]**
- The steep-crash L-inf values are **lower bounds** on `r_eff` (the body reached the frame *by* that
  L-inf; it may have been in contact earlier). So the true posture-matched `r_eff` could exceed 0.38; the
  data cannot rule that out, only rule out `r_eff < 0.26` at the tilted posture. **[X]**
- The corner-pass `r_eff` upper bound (0.25, from the CLEAN 0.50 pass) is the cleanest *single* number in
  the whole dataset (a clean pass at known offset directly caps `r_eff`), but it is **near-level only**. **[M/X]**
- All gate-4 margin numbers reuse the recorded radius-invariant in-plane `linf = 0.215 m` (simstart) from
  the contact-true eval; `provenance.md` §5 carries the same. This leg does not re-run the rollout. **[X]**

---

## Files (absolute)
- THIS file: `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\body-contact-reconcile-2026-06-13\empirical_radius.md`
- Sibling (geometry+spec): `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\body-contact-reconcile-2026-06-13\provenance.md`
- Corner-pass LIVE primary: `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\shadowpc-inc5-live-2026-06-11\WRITEUP.md` §5 (gate 0; off0.50 clean / off0.60→0.64 contact)
- Corner-pass offline (superseded): `C:\Users\Fengy\Downloads\Projects\Anduril\handoff\shadowpc-inc4-live-2026-06-11\WRITEUP.md` §4
- Steep-crash primary: `C:\Users\Fengy\Downloads\Projects\Anduril\memory\project_rl_increment_history.md:383`; cross-ref `handoff\laptop-training-doctrine-2026-06-12\WRITEUP.md:98-99`; `rl\peregrine_racing.py:153-156`
- Posture (crab ~55°/tilt ~54.8°): `C:\Users\Fengy\Downloads\Projects\Anduril\memory\index_control_sim.md:32,51`; `memory\project_phase2_rl_vision_decisions.md:1150`
- Gate-4 posture (pitch −38°, roll 55°): prompt KNOWN-CONSTANTS, sourced from memory gate-4 approach posture
- Geometry consts: `C:\Users\Fengy\Downloads\Projects\Anduril\rl\contact_true_eval.py:51-53`; `rl\offline_rollout.py:65-66`
