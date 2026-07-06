# Wire gate-credit proximity probe (2026-07-05)

**Question (RL-commander shared-knowledge flag):** does `RACE_STATUS.active_gate_index` credit a
gate ~9 m BEFORE the physical plane (A34/A35-era memory) or at/after crossing
(shadowpc-inc5-live-2026-06-11 WRITEUP, ≤0.64 m)? Load-bearing for how much OOD the old
wire-anchored RL flights had, and for the f63b4d9 deploy-parity fix's rationale.

**Method** (`probe_wire_advance.py`): offline over every recording on this box with BOTH a race
wire and a position stream (VQ1-era wires carry LOCAL_POSITION_NED/ODOMETRY). RACE_STATUS is
decoded from the tlog's ENCAPSULATED_DATA(data_type=1); at each within-race forward
`active_gate_index` transition, position is interpolated to the RACE_STATUS receive time (same
recv-clock base) and expressed in the credited gate's frame. `depth < 0` = credited BEFORE the
plane. KEY ASYMMETRY: RACE_STATUS staleness only biases depth POSITIVE (late detection), so a
negative depth would be a hard "early-credit" verdict — and depth_min bounds the credit point.

**Result — 447 transitions, 30+ recordings (2026-06-07 → 06-14, incl. full-course laps):**

- depth min = **−0.01 m** (one transition; interpolation noise), max = +4.41 m, mean = +0.99 m
- **The wire credits AT the plane. Zero early credits in 447 events.**
- RACE_STATUS cadence measured **4 Hz** (rs_gap ≈ 250 ms on every transition) → the positive
  tail is pure detection latency (250 ms × speed ≈ 1–2 m; the +4 m outliers pair with high
  post-crossing lateral drift, linf up to 1.0).

**Verdicts:**
1. inc5's "advance at/after crossing, ≤0.64 m" — **CONFIRMED** (our latency-debiased bound is
   tighter: credit ≤ ~0.0–0.1 m).
2. "~9 m early" — **REFUTED for wire credit timing on the VQ1 wire.** The A34-era "wire leads
   plane 9 m" figure cannot be about RACE_STATUS credit; it was measured in the VQ2 classical
   stack's ESTIMATE frame (multi-meter KF drift era) and/or described the turn-trigger's own
   geometry. The VQ2 wire carries NO position, so VQ2 credit timing is not probeable this way —
   the pending race-wire capture flight (nav-estimate + video method) is the instrument for that.
3. f63b4d9 deploy-parity fix survives the premise flip, with the rationale corrected: physical
   anchoring buys an ON-TIME re-anchor (wire = 0–250 ms / up to ~2 m late at 4 Hz) plus
   independence from uncaptured miss/forfeit wire semantics — not protection from a 9 m-early jump.

Files: `probe_wire_advance.py` (the tool), `probe_output_full.txt` (all 447 transitions).
