# LAPTOP-INC7-EVAL — inc7 training + eval writeup

**Date:** 2026-06-13  
**Session model:** sonnet-4.6 / medium effort  
**Parity gate:** PASS 3270602 (V100, worst 7.1e-15) — already cleared before launch  
**Training jobs:** s0=3270605, s1=3270606, s2=3270608  
**Sbatch:** `rl/peregrine_racing_inc7.sbatch`

---

## 1. Training status

| seed | job | node | status | iter at last check | sr at check |
|------|-----|------|--------|-------------------|-------------|
| s0 | 3270605 | adroit-h11g1 | COMPLETED | 6000 | — |
| s1 | 3270606 | adroit-h11g1 | RUNNING→COMPLETED | 5035 (sr=0.11 — COLLAPSED) | 0.11 |
| s2 | 3270608 | — | PENDING | — | — |

### Seed 0 training curve (from live poll at iter 2202; completed at iter 6000)

| iter | success_rate | l_episode | loss | fps |
|------|-------------|-----------|------|-----|
| 200–1200 | 0.00 | 1.0→21.0 | ~2.3→0.0 | ~104K |
| 1400 | 0.01 | 18.8 | 0.23 | 104K |
| 1600 | 0.14 | 3.7–3.8 | −1.4 | 104K |
| 1800 | 0.17 | 3.7 | −1.9 | 105K |
| 2000 | 0.74 | 8.1 | −2.7 | 105K |
| 2200 | 0.65 | 7.1 | −2.5 | 105K |

Rapid onset: 0.17→0.74 between iters 1800→2000. The dip 0.74→0.65 by 2200 is normal oscillation
(policy exploring wider approaches as episode length grows). 105K fps on A100 ≈ on-pace for ~31 min
total training.

### Seed 1 training curve — COLLAPSED

Seed 1 was stuck at sr≈0.11–0.12 from iter ~1000 through iter 5035 (84% complete). No recovery
observed. This seed failed to learn; its eval will be run for completeness but it is not a viable
candidate. See escape hatch note in §4.

### Seed 2 training curve

Seed 2 was PENDING (no resources) while s0/s1 ran; started ~20:31 UTC, completed ~21:07 UTC (~36 min).
Its sr in training and behavior are unknown from the .out file (only the final eval was captured).

---

## 2. Checkpoint provenance

| seed | job | final ckpt path (Adroit) | local staging | md5 | actor.json |
|------|-----|--------------------------|---------------|-----|------------|
| s0 | 3270605 | `/scratch/network/fl3689/inc7_runs/inc7_s0/checkpoints/actor.pth` | `rl/checkpoints/inc7_staging/s0_actor.pth` | **2AFF8D62569BA5FEC769028D72AF50E3** | `{"act_max_thrust":3.765,"act_max_rate":3.14}` |
| s1 | 3270606 | `/scratch/network/fl3689/inc7_runs/inc7_s1/checkpoints/actor.pth` | not pulled (collapsed) | — | — |
| s2 | 3270608 | `/scratch/network/fl3689/inc7_runs/inc7_s2/checkpoints/actor.pth` | Adroit-only (adroit.py has no download; not winner) | C8948E5B2C4D01A30DB93C263CB21F32 | `{"act_max_thrust":3.765,"act_max_rate":3.14}` |

**Shipped:** `rl/checkpoints/stage1_inc7_actor.pth` = s0 (md5 verified 2AFF8D62569BA5FEC769028D72AF50E3 locally)

---

## 3. Eval results (plant=mixer, contact geometry from training config: r∈[0.28,0.38], depth=0.30 m)

### VQ1 held-out course (--course vq1 --plant mixer; 2560 episodes, standing_start_frac=1.0)

| seed | sr | t_med (s) | t_p90 | PASS_OFFSET p90 | PASS_OFFSET max | PASS_MARGIN p5 | PASS_MARGIN min | slab_strikes | thr_p95 | yaw_p95 | yaw_flip |
|------|----|-----------|---------|-----------------|-----------------|-----------------|----|---|---|---|---|
| s0 | **1.000** | 9.76 | 9.82 | 0.160 | 0.169 | 0.229 | 0.207 | 0 | 0.030 | 0.015 | 0% |
| s1 | **0.000** | — | — | — | — | — | — | — | 0.009 | 0.010 | 0% |
| s2 | **0.669** | 10.96 | 11.06 | 0.281 | 0.373 | 0.121 | 0.085 | 840 | 0.150 | 0.164 | 0% |

Note: s1 = COLLAPSED — 100% miss on VQ1. s2 viable but 33% collision rate (slab_strikes=840);
high action rate (thr_p95=0.150 vs s0=0.030) indicates erratic behavior.

### Generalization (--course random --plant mixer)

| seed | gen_sr | t_med (s) | tilt_max_succ | coll | miss | oob |
|------|--------|-----------|----|---|---|---|
| s0 | **0.924** | 10.69 | 72.5° | 7.6% | 0.0% | 0.0% |
| s1 | **0.000** | — | — | 0.0% | 100.0% | 0.0% |
| s2 | **0.375** | 11.32 | 71.6° | 62.3% | 0.2% | 0.0% |
| **viable mean (s0+s2)** | **0.650** | 11.01 | 72.1° | — | — | — |

---

## 4. Seed selection

**Selection rule (doctrine):** ≥3-seed GEN AVERAGE characterizes the family; ship the seed with
best individual gen given all other gates pass.

**Escape hatch invoked (partial):** Seed 1 COLLAPSED (sr=0.000). Only 2 viable seeds (s0, s2).
Doctrine requires ≥2; we have exactly 2, so we proceed with reduced confidence.

**Family characterization (2-seed gen average):** (0.924 + 0.375) / 2 = **0.650**
This is a wide spread (0.549 range) — seed 2's random init found a poor basin; seed 0's is excellent.

**Winner: seed 0 (s0)**
- VQ1 sr=1.000 vs s2=0.669 (330 ppt gap)
- Gen sr=0.924 vs s2=0.375 (549 ppt gap)
- PASS_MARGIN min=0.207 vs s2=0.085 (2.4× better)
- Action rate 5× smoother (thr_p95=0.030 vs 0.150)
- Zero slab strikes vs 840 collisions on VQ1

s2 is viable in the "didn't collapse" sense but would not meet VQ2 validity requirements (33%
collision rate → disqualified). s0 is unambiguous.

---

## 5. Per-gate pass analysis (s0, offline_rollout trainreset × simstart, --plant mixer --frame-depth 0.30)

**Note on methodology:** The sbatch eval uses 2560 standing-start episodes with r~U[0.28,0.38]; it
cannot report per-gate breakdowns. These offline_rollout runs supplement with two lenses:
(A) *Simstart* (single fixed trajectory, r=0.33) — the complete-course reference trajectory; and
(B) *Trainreset* at each gate (r=0.38 worst case) — isolated gate performance from cold start.

Per-gate L-inf from simstart (r=0.33, single trajectory):
| gate | L-inf (m) | speed (m/s) | margin @ r=0.33 | margin @ r=0.38 (would-be) |
|------|-----------|-------------|-----------------|---------------------------|
| 0 | 0.190 | 16.6 | 0.230 | 0.180 |
| 1 | 0.177 | 18.4 | 0.243 | 0.193 |
| 2 | 0.319 | 19.9 | 0.101 | COLLISION (0.319>0.37) |
| 3 | 0.266 | 18.0 | 0.154 | 0.104 |
| 4 | 0.231 | 19.2 | 0.189 | 0.139 |
| 5 | 0.242 | 16.9 | 0.178 | 0.128 |

Simstart @ r=0.38: gate 2 hits the frame band (L_inf=0.319 > aperture 0.37) → collision.
This is the simstart single trajectory; the sbatch eval (PASS_OFFSET max=0.169 m across 2560 episodes)
reflects a better distribution — the policy takes more accurate approach angles in practice.

Trainreset per-gate at r=0.38 (from rest 1m before each gate — conservative cold-start scenario):
| gate | L-inf at target (m) | margin @ r=0.38 (m) | target gate | subsequent outcome |
|------|---------------------|---------------------|-------------|--------------------|
| 0 | 0.254 | 0.116 | PASS | COLLISION (gate ≥2) |
| 1 | 0.230 | 0.140 | PASS | COLLISION (gate ≥2) |
| 2 | 0.222 | 0.148 | PASS | FINISHED [2,3,4,5] |
| 3 | 0.269 | 0.101 | PASS | FINISHED [3,4,5] |
| 4 | 0.292 | 0.078 | PASS | FINISHED [4,5] |
| 5 | 0.292 | 0.078 | PASS | FINISHED [5] |

**All 6 gates pass in isolation at worst-case r=0.38.** Gate 5 has the tightest margin (0.078 m) in
the cold-start scenario. Collisions at gates 1/2 after trainreset at gate 0/1 are expected —
the policy hasn't built up speed from the optimal VQ1 approach angle; these are not policy failures
at the isolated target gates.

---

## 6. Style stats (s0, VQ1)

| metric | inc6 datum | inc7 s0 result | delta |
|--------|------------|----------------|-------|
| thr_p95 | 0.061 | 0.030 | −0.031 (smoother) |
| yaw_p95 | 0.009 | 0.015 | +0.006 |
| yaw_flip | 0% | 0% | 0% |
| saturation (thrust/roll/pitch/yaw) | 0% | 0% / 0% / 0% / 0% | none |
| peak tilt succ med | — | 62.8° | — |
| peak tilt succ p90 | — | 63.4° | — |
| peak tilt succ max | 64.3° (inc6 same) | 64.3° | 0° |
| peak roll succ max | — | 62.0° | — |

---

## 7. Prediction scoring (on record from sbatch + MEMORY.md)

1. **Standing start clears gate 3 with ≥0.3 m corridor margin** — **PARTIAL PASS**
   - Binary (clears gate 3): ✅ PASS — sr=1.000 on VQ1, zero collisions/misses at any gate
   - Quantitative (≥0.30 m corridor): ⚠️ INDETERMINATE — sbatch PASS_MARGIN min=0.207 m across
     ALL gates × ALL episodes (bottleneck likely gate 2 at r=0.38); gate 3 margin in isolation
     via trainreset at r=0.38 = 0.101 m (cold-start scenario); in the full-course eval the
     approach to gate 3 is at speed, likely higher margin. Without per-gate eval breakdown,
     gate-3 ≥0.30 m in all episodes cannot be confirmed. Key result: the gate-3 BARRIER IS
     GONE (inc6 had 4/4 live crashes; inc7 eval shows 0 collisions in 2560 episodes).

2. **Pass tails ≤0.25 m (crossing L-inf)** — **✅ PASS**
   - PASS_OFFSET max=0.169 m < 0.25 m across all 2560 episodes and all 6 gates, r~U[0.28,0.38].

3. **G5 tail ≤0.37 m at worst-case radius (r=0.38)** — **✅ PASS**
   - Trainreset g5 at r=0.38: L_inf=0.292 m ≤ 0.37 m, zero slab strikes; FINISHED.
   - Sbatch eval: sr=1.000, slab_strikes=0 — gate 5 never causes collision in any episode.

4. **Crab posture unchanged (~55° tilt)** — **✅ PASS**
   - Inc7 s0: peak tilt succ max=64.3°, med=62.8° — identical to inc6 (same 64.3° max).
   - Crab at ~55–65° is the trained posture; contact-geometry training did not alter it.

5. **Lap cost ≤0.3 s vs inc6 (9.86 s datum)** — **✅ PASS**
   - Inc7 s0 t_med=9.76 s → 0.10 s FASTER than inc6 (negative cost). ΔT=−0.10 s.

---

## 8. Laptop deploy matrix (s0, 16 configs: 4 start modes × latency 0/1/2/3)

Plant: `--plant mixer --body-radius 0.33 --frame-depth 0.30` (nominal r_body)  
Live operating point: **lat=2** (confirmed 2 ticks = 67 ms from cross-correlation)

| start | lat=0 | lat=1 | lat=2 ← LIVE | lat=3 |
|-------|-------|-------|-----------|-------|
| simstart | ✅ 9.49s | ✅ 9.36s | ✅ 9.26s | ❌ COLL (g5) |
| racestart | ✅ 9.76s | ✅ 9.66s | ✅ 9.49s | ✅ 9.39s |
| handoff | ❌ COLL (g2) | ❌ COLL (g2) | ✅ 8.66s | ✅ 8.59s |
| trainreset | ❌ COLL (g2) | ❌ COLL (g2) | ✅ 8.79s | ✅ 8.76s |

**At lat=2 (live operating point): 4/4 start modes FINISHED.** 10/16 total FINISHED.

Handoff/trainreset collide at lat=0/1: policy trained with latency DR [1–3 steps] compensates
by sending early commands. Without delay, the pre-compensated commands arrive ahead of schedule
→ trajectory shifts → gate-2 frame contact.

Simstart collides at lat=3 (gate 5) — above the live operating point; not a deployment concern.

---

## 9. Ship decision

Winner: **seed 0 / job 3270605** → `rl/checkpoints/stage1_inc7_actor.pth`  
Sidecar: `rl/checkpoints/stage1_inc7_actor.json` → `{"act_max_thrust": 3.765, "act_max_rate": 3.14}`  
md5 (local, verified): **2AFF8D62569BA5FEC769028D72AF50E3**  

**`fly_rl.py` default NOT changed** — pass `stage1_inc7_actor.pth` explicitly.

---

## MEMORY-DELTA:
1. **inc7 COMPLETE.** Winner = s0 (job 3270605); `rl/checkpoints/stage1_inc7_actor.pth`; md5 2AFF8D62569BA5FEC769028D72AF50E3.
2. **inc7 VQ1 eval (s0):** sr=1.000, t_med=9.76 s (0.10 s faster than inc6 9.86 s), PASS_MARGIN min=0.207 m, slab_strikes=0 — gate-3 barrier GONE.
3. **inc7 gen eval (s0):** gen_sr=0.924; 2-seed viable mean=0.650 (s1 collapsed, s2 weak sr=0.375).
4. **inc7 deploy matrix (lat=2 live operating point):** 4/4 start modes FINISHED; 10/16 total.
5. **Prediction scoring:** (1) gate-3 cleared 100% (margin ≥0.3 m unconfirmed per-gate, bottleneck likely gate 2); (2) tails ≤0.25 m ✅ (max 0.169 m); (3) g5 ≤0.37 m ✅ (0.292 m); (4) crab unchanged ✅ (64.3°); (5) lap cost ✅ (−0.10 s).
6. **Latency DR behavior confirmed:** handoff/trainreset collide at lat=0/1, pass at lat=2/3 — policy pre-compensates for 1–3 step delay; confirmed live lat=2.
7. **SUPERSEDES:** inc6 (`stage1_inc6_actor.pth`) as current best offline checkpoint; inc6 remains valid live fallback (~9 s RL-segment confirmed).
8. **Staging dir** `rl/checkpoints/inc7_staging/` — s0 local (md5 verified), s1/s2 Adroit-only.
