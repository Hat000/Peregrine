# ADVERSARIAL VERIFICATION — gate-4 sigma honesty + r=0.30 closure robustness

**Role:** adversarial verifier (refute, not confirm). Offline, no sim/flight.
**Claim under test:** *"Using the L3-measured gate-4 sigma (anisotropic [lat 0.19, vert 0.10]) for the margin is honest, and r=0.30 closure at fix-rate ≥ 0.35–0.50 is robust to the sigma uncertainty."*

**Engine:** `margin_driver_v2.run_cell_v2` — regression-proven **byte-identical to `ME.fly_lap` to 1e-9** in the iso/zero-bias limit (re-verified this session: ME 0.09802/0.13124 == v2 0.09802/0.13124). COLD case-C, bias=0, latency 15 ms, POST-bake (vert_fix_bias_m=0), random3d. CLOSURE = p90 < MARGIN(r) AND p99 < MARGIN(r); MARGIN(0.30)=0.235.

Reproduced the published per-fix sigma exactly from the 81 accepted fixes / 13 sessions: **rel_vert sd 0.1008, rel_cross sd 0.1914** (ddof=1), mean_vert −0.2147, range 20.1–24.0 m, speed 17.0–18.8 m/s.

---

## VERDICT: claim PARTIALLY REFUTED (confidence HIGH)

| Sub-claim | Result |
|---|---|
| "sigma is honest" (no labelling/effective-N error) | **SURVIVES** — the within-batch per-fix sigma CI is genuinely tight (ICC≈0) |
| "r=0.30 closes at fr ≥ **0.35**" | **REFUTED** — at the honest aniso sigma fr=0.35 only closes at **v37**, not v30; at the upper-CI sigma fr=0.35 fails at both speeds |
| "r=0.30 closes at fr = **0.50**" | **SURVIVES** the within-batch sigma CI, but **rests on an UNMEASURED at-speed sigma**; the headroom before it breaks is only **+0.04–0.05 m (×1.2–1.28)** |
| "robust to the sigma uncertainty" | **REFUTED for the binding uncertainty** (at-speed motion-blur), which the within-batch CI does not bound |

**One-line:** σ_lat 0.191 is a **17–18 m/s lower bound, not a safe floor**. The r=0.30 closure margin is **single-digit-cm thin** and the dominant uncertainty (at-speed lateral sigma at 30–37 m/s) is unmeasured; a ×1.3 inflation breaks closure even at the best fix-rate.

---

## Attack 1 — effective-N / cluster-bootstrap CI  → claim SURVIVES (this lens is a weak refutation)

Cluster bootstrap = resample the **13 whole sessions** with replacement (B=20000), recompute the pooled per-fix SD. Contrast with an iid per-fix bootstrap, plus one-way ANOVA ICC.

| axis | point sd | **cluster CI90** | iid CI90 | ICC | cluster/iid width |
|---|---|---|---|---|---|
| vert | 0.1008 | [0.0847, 0.1168] | [0.0836, 0.1182] | −0.029 | 0.93 |
| lat  | 0.1914 | [0.1664, 0.2103] | [0.1615, 0.2165] | +0.030 | 0.80 |

- The published "cluster CI **narrower** than iid" is **real, not a bug**: within-window correlation is ~0 (ICC −0.03 vert / +0.03 lat), so the 81 fixes are effectively independent and resampling whole sessions does not inflate the SD CI. n_eff ≈ 70–81. **The per-fix sigma is well-pinned for this batch.**
- **Upper-edge (pessimistic) sigma:** aniso CI90 **[0.210, 0.117]**, CI95 [0.214, 0.120]; iso-radial-equiv CI90 **0.170**.
- **Margin at the upper-CI sigma (r=0.30):**
  - **fr=0.50:** CLOSES for **all** upper-CI sigma sets, both speeds (p99 0.201–0.225 < 0.235). ✅ robust.
  - **fr=0.35:** FAILS — honest aniso 0.210/0.117 (p99 0.256 v30 / 0.250 v37) and 0.214/0.120 (0.258/0.240); even iso-0.170 fails at v30 (p99 0.238). ❌

**Caveat that matters:** 13 clusters is a small cluster count, and this CI only bounds the **within-batch per-fix scatter at 18 m/s**. It says nothing about how the sigma moves at the deployment speed — which is Attack 2, the binding uncertainty.

## Attack 2 — at-speed sigma growth  → claim REFUTED (headroom is small)

L3 is 17–18 m/s; the binding gate-4 is **30 m/s (doctrine) / 37 m/s (older)**. Per-fix motion-blur sigma at speed is **UNMEASURED** (memory's own standing caveat). Swept σ_lat at fr=0.50 (the best realistic fix-rate), bias 0, two vert policies.

**σ_lat CEILING for r=0.30 closure @ fr=0.50:**

| policy | v30 | v37 |
|---|---|---|
| vert held 0.10 | bracket [0.24, 0.25] → **~0.245** (×1.28 L3, +0.054 m) | [0.24, 0.25] → **~0.245** |
| vert scaled (L3 ratio) | [0.22, 0.24] → **~0.23** (×1.20 L3, +0.039 m) | [0.22, 0.24] → **~0.23** |

**Headroom over the L3-measured 0.191 is only +0.04 to +0.05 m (×1.2–1.28).**

Linear motion-blur extrapolation (pixel-blur ∝ speed at fixed ~22 m range): 18→30 m/s = ×1.67 → **σ_lat ≈ 0.32**; 18→37 m/s = ×2.06 → **σ_lat ≈ 0.39**. Both are **far above the ~0.23–0.245 ceiling**. Even a modest ×1.3 at-speed inflation (0.191→0.249) breaks closure. The crab axis (gate-4 crab 36–54°) carries the apparent motion, so blur loads the **already-dominant lateral axis** — the pessimization is physically the right one.

## Attack 3 — lateral-axis dominance  → CONFIRMED (iso-0.10 is grossly optimistic)

From the published `margin_rerun` A-grid (identical engine, post-bake, att0, random3d): honest aniso [0.19,0.10] vs naive iso-0.10 (the clean "sigma_vert" used isotropically).

| | honest aniso [0.19,0.10] | naive iso-0.10 |
|---|---|---|
| first fr that closes r=0.30 (v30) | **0.50** | **0.15** |
| first fr that closes r=0.30 (v37) | **0.35** | **0.15** |
| p99 over-optimism (naive understates by) | — | **+30–41% uniformly** |

Using iso-0.10 falsely claims r=0.30 closure at a **2.3–3.3× lower fix-rate** than honest. The bake fixes the vertical **bias** (−0.25 m), not the lateral **scatter** (0.19, crab-coupled). Closure rides on the **0.19 lateral + fix-rate**, exactly as the recal's own `honesty_flag` states. The correct single-sigma engine surrogate is the in-plane-radial-equiv **0.153** (upper-CI **0.170**), never 0.10.

---

## Answers to the posed questions

- **Is σ_lat 0.191 a safe floor, or could the true at-speed/effective-N sigma be materially higher and break closure?**
  σ_lat 0.191 is a **17–18 m/s LOWER BOUND, not a safe deployment floor.** Effective-N does NOT inflate it (clusters are ~independent; within-batch CI tight). The **at-speed** sigma can be materially higher and **does break closure**: a linear-blur model puts it at 0.32–0.39 at 30–37 m/s, vs a ceiling of ~0.23–0.245.

- **σ_lat ceiling for r=0.30 closure at fr=0.50:** **≈ 0.245 m** (vert held 0.10) / **≈ 0.23 m** (vert scaled with the L3 anisotropy), consistent at both 30 and 37 m/s. Headroom ×1.2–1.28 (+0.04–0.05 m) over the measured 0.191.

- **Effective-N-widened sigma CI (cluster bootstrap, 13 sessions):** vert **[0.085, 0.117]**, lat **[0.166, 0.210]** (CI90); iso-radial-equiv upper edge **0.170**. (Essentially equal to iid — ICC≈0.)

- **Does closure survive the upper-CI sigma?** **At fr=0.50: YES** (r=0.30 closes for all upper-CI sigma sets, both speeds). **At fr=0.35: NO** (the upper-CI aniso fails both speeds; iso-0.170 fails at v30).

## Standing recommendation for the closure analysis
1. State σ_lat 0.191 as an **18 m/s lower bound**; do **not** claim it holds at 30–37 m/s.
2. **MEASURE at-speed σ_lat** (a 30–37 m/s gate-4 shadow) before asserting r=0.30 closure — it is the single binding uncertainty and the ceiling is only ×1.25 away.
3. Drop the "fr ≥ 0.35" floor → use **fr ≥ 0.50** (the honest aniso sigma needs fr=0.50 at v30; fr=0.35 only works at v37 and fails at any upper-CI sigma).
4. Never report the iso-0.10 closure — it is ×2.3–3.3 optimistic on fix-rate.

## Artifacts (all under `handoff/boresight-closure-2026-06-14/`, prefix `verify_sigma_`)
- `verify_sigma_results.json` — consolidated (all three attacks)
- `verify_sigma_cluster_boot_results.json` + `verify_sigma_cluster_boot.py` — Attack 1 CI
- `verify_sigma_ckpt.json` — A1 margin cells + A2 at-speed sweep (raw)
- `verify_sigma_serial.py` / `verify_sigma_min.py` / `verify_sigma_margin.py` — drivers
- `verify_sigma_consolidate.py` — consolidation (reads cluster-boot + ckpt + published margin_rerun)
