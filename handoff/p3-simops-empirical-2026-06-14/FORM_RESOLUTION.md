# BORESIGHT ε_vert — FORM RESOLUTION (P3 closeout, 2026-06-14)

**Verdict: ε_vert is a METRIC vertical offset (range-INDEPENDENT), NOT an angular boresight.**
**Magnitude −0.25 m (refined chain) / −0.27 m (raw IPPE) at the ~22 m operating band · sign gate-DOWN · cross ≈ 0.**

Operator: opus-4.8 · Data: pooled head-on gate-0, batch1 (main `data/runs`) + batch2 (`trusting-maxwell` worktree),
N = 13,308 accepted fixes (fix-rate 67–77% head-on vs inc7 1.4% — camera-pointing validated). Map = tracked
`shadowpc-firstcontact-2026-06-02/track_map.json` (fly ≡ analysis). Chain = production C2 shadow (shadow_gate4.py).

## The three-part resolution

### 1. Range-balanced, regime-separated (no re-PnP; `boresight_form.py` over pooled b1/b2 rows)
Single moving-approach regime (el ≈ −1°), range-binned `rel_vert` (gate-down):

| range | N | rel_vert | note |
|---|---|---|---|
| 10–14 m | 301 | −0.134 | near (suspect; see §3) |
| 14–18 m | 317 | −0.276 | plateau |
| 18–22 m | 256 | −0.273 | plateau |
| 22–24 m | 5439 | −0.244 | plateau (refined) |

**Matched-range 22–24 m: static hover (el +20°) −0.251 vs moving approach (el −1°) −0.244 → Δ = −0.007 m.**
⇒ NO elevation/regime dependence → rules OUT a principal-point (cy) / distortion / motion-blur mechanism; the
bias is a pure extrinsic, identical in both regimes.

### 2. IPPE both-flip (`headon_bothflip.py`, raw `SOLVEPNP_IPPE_SQUARE` candidates, GT-anchored, 3 approach laps)
| range | N | rv (lo-reproj) | rv (GT-correct) | flip-spread | ambig e2/e1 | sel==GT |
|---|---|---|---|---|---|---|
| 10–14 | 44 | −0.104 | −0.094 | 0.024 | 1.4 | 43% |
| 14–18 | 62 | −0.279 | −0.275 | 0.013 | 1.5 | 73% |
| 18–22 | 72 | −0.270 | −0.270 | 0.010 | 1.7 | 97% |
| 22–26 | 662 | −0.268 | −0.266 | 0.011 | **1.0** | 67% |

**Flip is IMMATERIAL:** spread < 0.024 m at every range — INCLUDING the genuinely-ambiguous 23 m anchor
(e2/e1 ≈ 1.0). `rv_loreproj ≈ rv_GTcorrect` everywhere ⇒ a deploy estimator WITHOUT a GT prior reads the same
vertical bias as the GT-disambiguated shadow chain. **The −0.27 m anchor is NOT flip-tainted** (refutes the
weak-perspective-far-anchor concern). The shadow rows (GT-prior disambiguated) ≡ GT-correct flip — confirmed.

### 3. FORM verdict — METRIC
Over the binding operating band **14–26 m, `rel_vert` is FLAT at −0.27 m** (−0.275 → −0.270 → −0.266; if anything
a slight DECREASE). An **angular** boresight ε = R·tanθ pinned to −0.27 m at 23 m would read −0.16 m at 14 m and
**−0.51 m at 26 m** (×1.9 growth) — emphatically contradicted. **The bias does NOT scale with range ⇒ METRIC.**
Corroboration: L3 cross-gate g2 −0.211 @ 20 m ≈ g4 −0.215 @ 22 m is also flat-with-range.
The only sub-plateau point is **10–14 m (−0.09 to −0.13)**: low-N (44), strongest-perspective, fast near-field
(8 m/s, latency-coupling largest) — outside the margin-binding band and second-order; it does NOT make the bias
angular (angular is wrong across the whole 14–26 m plateau regardless).

## DELIVERABLE TO P1 (overall commander applies one-line)
- **FORM = METRIC** → `BoresightCorrection(vert_offset_m = −0.25 at ref_range ≈ 22 m)` into the **+L lever**
  (NOT a pitch/extrinsic rotation — angular is refuted).
- **Sign convention:** `rel_vert = (vision_fix − GT)` projected on gate-DOWN = **−0.25 m** → the vision chain
  places the gate ~0.25 m gate-DOWN of truth (drone read ~0.25 m gate-UP). Correction NEGATES this. cross ≈ 0
  (lateral well-calibrated). Use **−0.25 m (refined/production chain)**; raw-IPPE pre-refine is −0.27 m.
- **Range-independent** over 14–26 m (the operating band) — a single constant offset suffices; no range schedule.

## CAVEAT for the margin verdict (flag, not in P3 scope)
A METRIC (constant-metre) vertical term is, mechanically, the kind of term that CAN cancel in the +L
gate-relative obs IF it lives in the map gate-position (δ_map) rather than the vision lever (ε). The δ_map
discriminator (d7c592e) pinned δ_map_vert ≈ 0, assigning this to ε (perception → does NOT cancel → margin OPEN).
This head-on result confirms the magnitude/form but is all bearing≈0; the discriminator's bearing-correlation
(+0.44, from oblique inc7) is untestable here. Recommend the overall commander reconcile FORM=metric-ε against
δ_map≈0 before locking the close/no-close direction.
