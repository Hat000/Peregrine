# v3 — ADVERSARIAL VERIFICATION of d3 (margin / cold-velocity faithfulness)

LENS: *is the cold-velocity sim FAITHFUL to case-C reality (accel bias + no given velocity)?*
Charged to REFUTE. All numbers below independently re-derived: I imported d3's OWN `fly_lap`
(the real `racer.state_estimator.LinearKF` + `RewindKF` + measured 0.265 m/axis σ), re-ran with a
DIFFERENT seed base, and added faithfulness knobs d3 hard-codes. Artifacts:
`v3_margin_refute.py` / `v3_margin_refute_results.json`, `v3b_closure_refute.py` /
`v3b_closure_refute_results.json`. d3 + d4v + c1-anchor all reproduced this run.

## VERDICT: UPHELD-WITH-CORRECTION
The CANNOT-SETTLE-OFFLINE verdict and claims 1–4 SURVIVE. Claim 5(i) (velocity channel σ_v≤0.3
"holds the margin at p90") is REFUTED — it is an **RMS-clear, not a p90-clear**. The correction makes
the design's bottom line *stronger* (margin even harder to close offline), so the blueprint's
"build all three + verify on live data" conclusion stands and is reinforced.

## SURVIVED

- **Anchor / shared machinery (REPRODUCED bit-for-bit):** c1 rel-arm RMS 0.139 / p90 0.203 / p99 0.311.
  The sim genuinely composes the real LinearKF + RewindKF; no re-implemented physics.
- **Claim 2 — cold fails the margin at ZERO bias (RNG-independent):** under d3's seed, cold p90 0.234 /
  p99 0.322 / 37% contact. Under MY different seed base (R0/R1): p90 0.227–0.234, p99 0.322–0.325, 37%.
  The straddle direction (warm clears, cold fails) is not a seed artifact.
- **Claim 2's no-given-vel FAITHFULNESS (the lens's core) — d3 VINDICATED.** d3 seeds velocity near
  truth at g0 (N(0,1.5)); I suspected this flatters the cold case. **R1 REFUTES the suspicion:** the g4
  cold verdict is INSENSITIVE to the velocity init — p90 0.227→0.228 as σ_init 1.5→3.0→6.0, and
  unchanged (0.226) under a +2 m/s SYSTEMATIC init bias. The full lap pre-converges velocity to its
  steady-state IMU-only quality regardless of init; the cold number is set by between-fix IMU drift, not
  the start. d3's "moderate prior at g0" is NOT optimistic. **The cold sim is faithful on no-given-vel.**
- **Claim 4 — att→accel mapping is physically EXACT.** g·sin(1.4°)=0.2396 (=d4v 0.2396); |specific
  force| on the drag-hold g3→g4 approach = 9.8066 = g, so the g·sinθ mapping is exact, and a fixed
  attitude misalignment ≡ a constant-per-run body accel bias (both rotate into world through R). The
  bias-sweep composition is HONEST, not double-counted: cold@bias=0 is a best case (att-as-Q only);
  realistic cold = cold@bias≈0.24. My independent **cold@0.240 → p90 0.338, p99 0.453, 65% contact** —
  at the HIGH end of the design's claimed 0.26–0.40. Claim 4 holds (if anything its lower bound is mild).

## FAILED → CORRECTED

- **Claim 5(i) REFUTED: the velocity channel does NOT hold the margin at p90.** d3 §5 says "At σ_v≤~0.3
  m/s the margin is held at p90 … p99 still grazes 0.18." This conflates d4v's RMS column with p90.
  - d4v's OWN table: visvel σ_v 0.3 → RMS 0.118 but **p90 0.187**; σ_v 0.5 → RMS 0.124, **p90 0.185** —
    all marked "clears" on the RMS column (0.118/0.124 < 0.155) while **p90 0.185–0.213 > 0.155**.
  - MY independent multi-gate closure (v3b, n_mc 1500, idealised velocity update at fixed σ_v): the
    BEST cell σ_v 0.3 / bias 0 → **RMS 0.131 (RMS-clears) but p90 0.196 (does NOT clear), p99 0.285,
    25.5% contact.** Adding the realistic 1.4° bias (0.240): σ_v 0.3 → **p90 0.275**; σ_v 0.5 → p90 0.312.
  - This contradicts d3's own (correct) thesis that a margin is a p90/p99 gate. **Corrected statement:**
    the velocity channel + low accel bias is an **RMS-clear only**; p90 (0.18–0.20 at best, 0.27–0.31
    with measured attitude) stays OVER the 0.155 worst-case margin. The conditional closure does **not**
    clear the worst-case margin offline; it needs velocity channel AND att≤~0.6° AND the
    uncertainty-aware speed-down — and even then must be verified on the live at-speed recording.

- **Minor conservatism gap (strengthens the REFUTE):** d3 draws the accel bias in a RANDOM 3D direction,
  wasting ~1/3 of its energy along-track (which does not count against the in-plane margin). The
  physically faithful attitude-tilt phantom is HORIZONTAL/in-plane. Forcing the bias in-plane:
  cold@0.240 → p90 **0.348** / p99 0.476 / 71% contact vs d3's random-direction 0.338 / 0.453 / 65%.
  d3 slightly UNDER-states the realistic cold failure. (Mode is irrelevant at bias 0, as expected.)

## RESIDUAL RISK
Same as d3's: the TRUE effective accel-bias / attitude-error entering g4 is unpinned offline (only the
ShadowPC ≥5-lap at-speed recording pins it). My correction tightens — not loosens — the requirement:
even the optimistic offline closure is a p90-FAIL, so the live recording is mandatory, not advisory.
