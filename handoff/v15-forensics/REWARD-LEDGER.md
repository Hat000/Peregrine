# REWARD-LEDGER FORENSICS — v1 arm A (`v1A_s0`) — why runaway + pitch-down + yaw hunt is the RATIONAL policy

Worktree `wt-fix` @ `7bc5305`, branch `vtrackAr5-recovery-reward`. Files: `rl/peregrine_racing_ego.py`,
`rl/ego_reward.py`, `rl/gate_visibility.py`, `rl/ego_estimator.py`, `rl/vq2_ego_curriculum.py`,
`rl/peregrine_vq2_ego.sbatch`, `handoff/rl-commander-2026-07-16/launch_v1.sh`, `docs/.../DESIGN.md`.
Code is ground truth; premise corrections flagged in §CODE-WINS.

## Effective config for arm A (base sbatch `+env.` < stage `dual_gate_fullstack_floor_pef` < EXTRA `++`; last/`++` wins)

| knob | effective | source (loser→winner) | note |
|---|---|---|---|
| `dt` / `gamma` / `max_time` | **0.0333 s** / **0.9975** / 60 s | fly_rl.py:108, stage `_raw` | 30 Hz; horizon 400 steps=**13.3 s**; max 1800 steps |
| `course_n_gates` | **8** | stage 2 → EXTRA 8 | the 8-gate fine-tune |
| `reward_refined_b` | True | default | uses `compute_ego_reward` |
| `rw_progress` | **2.0** /m | _COMMON | segment-arc, see below |
| `rw_progress_to_center` | **False** | _COMMON True → **stage False** | progress = along-track ARC (perp drift earns 0); homing comes from corridor+centering |
| `vmax_mps` (progress clip) | 39.0 | dataclass | clip band = 39·dt = **1.30 m/step**; never binds ≤13 m/s |
| `rw_parabola_crossing` | **True** | stage | REPLACES passage; drops frame-clip/miss terminal |
| `cross_center`/`cross_zero_m`/`cross_neg_cap` | 20 / **0.75** / 100 | stage(4.0)→EXTRA 0.75 | peak +20 centered, 0 at aperture, floor −100 |
| `rw_clip_terminal` | **0.0 (OFF)** | default | ⇒ a frame strike pays ~0..−100 parabola only, keeps banked |
| `rw_passage` | 5.0 (**INACTIVE**) | _COMMON | parabola on ⇒ passage zeroed |
| `rw_centering` / max | 0.4 / 6.0 m | stage | −0.4·clamp(perp,0,6) dense lateral penalty |
| `rw_corridor` | 4.0 | stage | PBRS contouring (telescoping perp potential) |
| `rw_perception` / exp | **0.02** / 4 | stage | +0.02·exp(−δ⁴); **capped ≤ rw_time by __post_init__** |
| `rw_perception_next` | **0.0 (OFF)** | default | no next-gate pointing reward |
| `rw_yaw_dither` | **0.125**, anneal start **1.0** (from birth), hold 0.3 | EXTRA | −0.125·(Δyaw_cmd)² on post-clamp ch3 |
| `rw_v_cap` | **0.0 (OFF)** | EXTRA explicit | **no speed opposition** (confirmed) |
| `rw_roll_recover`/`rw_att_pitch`/`rw_att_roll`/`rw_gate_vhold`/`rw_align`/`rw_altitude_hold`/`rw_exit_align` | **0.0** | default | all OFF in arm A (roll_recover = arm R only) |
| `rw_rate`/`rw_dact` | 1e-3 / 1e-3 | _COMMON | negligible |
| `rw_tilt` free/cap | 4.0, 60°/70° | dataclass | −4·relu(cos60−R33)²; 0 inside 60° cone |
| `rw_time` | 0.02/step (=0.6/s) | dataclass | flat, speed-independent |
| `rw_finish`/`rw_finish_time` | 20 / **0.25**/s | dataclass, EXTRA(1.0→0.25) | (20+0.25·t_left)·finished |
| `terminal_base`/`_miss`/`_oob` | 100 / 100 / **200** | _COMMON | miss INACTIVE under parabola |
| `terminal_progress_scaled` | True | _COMMON | contact forfeits banked (lethal-masked, below) |
| `ego_yaw_cmd_clamp_rad_s` | **0.7** | stage 0.35 → EXTRA **0.7** | applied at top of step() to ch3 |
| `ego_blur_gate` | **False (OFF)** | stage True → **EXTRA false** | ⇒ perfect-shutter; **yaw sweep is NOT blur-penalized** |
| `ego_spin_*` abort | 3.5 rad/s·0.4 s OR 1.5 rev/4 s; anneal from 2.6× | stage + EXTRA | fatal, all-axis; realized clamp 0.7 ≪ 3.5 → never bites racing |
| `ego_vision_cadence` | True, 30 Hz·p=0.35 = **10.5 Hz valid fix** | EXTRA | else ego-propagate + conf decay (stale horizon 0.5 s) |
| `ego_obs_coast` | **False** | default | obs slot HARD-MASKED when gate not geometrically detectable this step |
| `ego_obs_v2` | **False** → **21-dim** | default | slot1 = next gate only |
| `faithful_rate` / `n_substeps` | True / 5 | EXTRA / stage | expansive plant + 150 Hz substeps |

---

## Q1 — REWARD LEDGER (per-step magnitude, two regimes)

Regime A = **3 m/s closing, level, gate centered**; Regime B = **10 m/s closing, nose-down, gate drifting to top of frame**.
Camera = body-forward **+20° up**, tail-first π-flip, FOV 90°H×58.7°V (fx=fy=320); `cos_view = tz/range`.

| term | formula (file:line) | coeff | Regime A (3 m/s, centered) | Regime B (10 m/s, nose-down) |
|---|---|---|---|---|
| **progress** | `2·clip(Δs_arc, ±1.30)` ego_reward.py:581-593, called :1180 | 2.0 | Δs=0.10 → **+0.200** | Δs=0.333 → **+0.667** (clip inactive) |
| time | `−rw_time` :1273 | 0.02 | **−0.020** | **−0.020** (flat) |
| perception | `0.02·exp(−δ_cam⁴)` :1022-1034 | 0.02 | δ≈0 → **+0.020** | gate ~top edge δ→ large → **≈0.000** |
| centering | `−0.4·clamp(perp,0,6)` :633-642 | 0.4 | perp≈0 → **≈0** | perp≈1 m → **−0.4** |
| corridor (PBRS) | `4·clip(perp_prev−perp,±1.30)` :685-700 | 4.0 | ~0 online | drifting off-line → **−0.x to −1.3** |
| free-cone tilt | `−4·relu(cos60−R33)²` :890-900 | 4.0 | inside 60° → **0** | ~0 unless >60° total tilt |
| yaw-dither | `−0.125·(Δyaw_cmd)²` :923-938 | 0.125 | steady → ~0; **rail-flip Δ=1.4 → −0.245** | same: **−0.245/flip** |
| rate+dact | `−1e-3·‖ω‖ −1e-3·Δa²` :906-920 | 1e-3 | **≈−0.005** | **≈−0.01** |
| **v_cap** | OFF | 0 | **0** | **0 (no speed opposition — confirmed)** |
| parabola (event) | `clamp(20·(1−(e/0.75)²),−100,20)` :738-769 | — | fires only at crossing: centered **+20** | wide/frame **−15..−100** |
| finish (event) | `(20+0.25·t_left)·finished` :880-884 | — | last gate only | + faster ⇒ more t_left |

**Progress is LINEAR in closing speed** (0.20→0.667→0.867/step at 3/10/13 m/s) and **the 39 m/s clip never
binds** at any real speed. BUT it is a **telescoping potential** (segment-arc `s`, ego_reward.py:535-553): the
undiscounted sum over a segment = `2·seg_len` **independent of speed** (15 m seg = +30 whether slow or fast).
So progress is *not* a raw "reward∝speed" term — the speed incentive is second-order (discount + time + finish;
see Q2). **Perception geometry:** nose-DOWN rotates the +20° optical axis down; a level/above gate drifts out
the **top** of the 58.7° V-FOV → `cos_view`↓ → r_perc→0 AND detectability lost (DESIGN: −50° body pitch → cam
−30° → floor). Nose-UP swings it out the **bottom**. **Critical: r_perc max = 0.02/step is CAPPED at rw_time
by `__post_init__` (ego_reward.py:461), i.e. ≤4% of progress at 8 m/s (0.53).** Perception is structurally too
weak to hold the nose up against the speed/progress gradient — this is the pitch-down mechanism (§Q3).
**yaw-dither** is amplitude-blind on *sustained* yaw (Δ≈0) and taxes only the flip transient (−0.245 at the 0.7
rail); a high-amplitude *oscillation* at ~5 flips/s costs only ~5·0.245·dt ≈ **−0.04/step avg** — trivially
outweighed by a +20 gate. With **`ego_blur_gate=false`**, sweeping the camera has **no detectability cost**.

---

## Q2 — SPEED ARGMAX ARITHMETIC (γ=0.9975, dt=0.0333, horizon 400 steps ≈ 13.3 s)

**There is no term that decreases with speed** (`rw_v_cap=0`; time penalty is flat −0.02/step). Every speed
incentive is a completion-time effect, and **only crash risk opposes speed**:

1. **Per-step magnitude** — progress/step scales with speed (0.20→0.87 at 3→13 m/s); the myopic gradient is
   "faster = bigger reward now."
2. **Discount pull-forward (dominant)** — shaving 1.0 s (30 steps) multiplies ALL downstream reward by
   γ⁻³⁰=**1.075** (+7.5%). Downstream return ~150–350 ⇒ **+11.7 to +27.3** per second shaved.
3. **Time penalty + finish** — +0.6/s (time) + 0.25/s (finish bonus) ⇒ **+0.85/s**.

**Telescoping check** (15 m segment): discounted progress 25.0 (3 m/s) → 28.4 (10 m/s) — only +13% within the
segment, but the fast drone banks it **105 steps sooner** and pulls the whole rest of the course forward (lever 2).

**Crash terminal** (the only opposition), discounted by delay to crash:

| failure | terminal | keeps banked? | disc. cost (delay 1 s / banked=120) |
|---|---|---|---|
| **floor dive / ceiling / spin** (lethal) | 100 **+ banked forfeit** | no | ~**204** |
| **OOB** (lateral/ceiling arena) | 200 | yes | ~**185** |
| **frame strike** (parabola path) | parabola only (~−15..−100) | **yes** | ~**−15..−93** |
| **wide forward miss** | parabola (≤−100 cap) | **yes** | ~**−93** |

**Break-even:** riding faster is reward-positive whenever incremental crash-prob per 1 s shaved is below
`gain / disc_terminal`. With gain ≈ 12–27 and disc terminal ≈ 150–220: **p\* ≈ 5–13%.** So the policy will push
speed until the *marginal* crash probability per increment exceeds ~5–13% — i.e. **ride to the controllability
edge and leave zero margin.** That edge is **plant-dependent**: the training (faithful_rate) plant is
controllable to ~12 m/s so the policy learns "ride to ~12"; at deploy the expansive plant loses authority
earlier, but the trained policy still commands the ride-to-12 behavior → crash at ~1 gate. **The anti-sprint
guard (`terminal_progress_scaled`, tested for terminal dominance over banked progress) defends only against
sprint-INTO-a-gate-clip; it does nothing about between-gate cruise speed, which triggers no terminal.** Verdict:
**velocity runaway is the reward argmax** — speed is unpriced, discount-favored, and bounded only by a
finite, discounted, plant-dependent crash penalty.

---

## Q3 — THE BRAKE (nose-up against velocity, gate dropping toward frame bottom)

Braking pays **nothing** and costs on three fronts — but note it is not *permanently* reward-negative because
progress telescopes; the costs are delay/loss, and no term *pays* to decelerate:

1. **Progress collapses / reverses:** `r_prog = 2·clip(Δs_arc)`. Decelerating shrinks the per-step along-track
   advance (0.53/step at 8 m/s → 0.13 at 2 m/s); braking hard enough to back up along-segment makes Δs<0 →
   **negative** progress (ego_reward.py:589 "can be negative").
2. **Time + discount keep bleeding:** −0.02/step accrues over the extra steps; every downstream reward is pushed
   later (γ<1) and finish t_left shrinks. Slowing 1 s costs ~0.85 + 7.5%·downstream (the mirror of Q2).
3. **Perception + observability degrade:** pitching nose-up rotates the +20° camera further up → a level/near
   gate drifts out the **bottom** of the 58.7° V-FOV → `cos_view`↓ (r_perc→0) AND the gate goes geometrically
   undetectable → `_current_detectable` false → **obs slot0 hard-masked to zeros** (obs_coast=False,
   peregrine_racing_ego.py:387-388) → the policy flies the brake **blind**, raising miss/clip risk.

**Verdict: the "punished three ways" claim holds directionally** — progress↓/negative, time+discount bleed,
perception+obs loss — and crucially **no term rewards deceleration while `rw_v_cap=0` makes over-speed free.**
The asymmetry (speed free, brake all-cost) is exactly what ratchets speed up gate-over-gate.

---

## Q4 — OBSERVABILITY (21-dim, ego_actor_obs, peregrine_racing_ego.py:404-411)

```
obs[0:3]  = est.velocity (BODY, estimator/IMU-primary)   ← VELOCITY IS OBSERVED
obs[3:5]  = roll/pitch (leveler)      obs[5:8] = body rates (gyro)     obs[8] = last collective
obs[9:11] = coarse sector of CURRENT target (horiz,vert ∈ {-1,0,1})
obs[11:16]= slot0 CURRENT gate: rel_pos(3)+conf(1)+visible_area(1)
obs[16:21]= slot1 NEXT   gate: rel_pos(3)+conf(1)+visible_area(1)
```
- **Velocity IS in the obs** (obs[0:3], body-frame estimator velocity; `‖v_body‖=‖v_world‖`). **Acceleration is
  NOT.** No dedicated speed/closing-rate scalar — the policy must form `‖v‖` itself. So it *can* represent "too
  fast," but **nothing prices it** (v_cap=0), so it has no reason to.
- **slot1 when the next gate has never been seen = ZEROS (masked).** A slot is masked (rel/conf/area=0) unless
  the gate is geometrically detectable this step (obs_coast=False). It is **NOT** filled from the coarse-map
  prior. Only obs[9:11] (CURRENT sector) carries any map info in the 21-dim obs.
- **obs_v2 (`ego_obs_v2`, default OFF for arm A) appends obs[21:23] = the NEXT gate's coarse SECTOR**
  `sector[clamp(tg+1)]` (horiz,vert), masked past the last gate — **NOT velocity** (peregrine_racing_ego.py:412-420).

---

## Q5 — SCANNING VALUE / why yaw oscillates, and why 8-gate amplifies it

**Reward benefit of pointing at the NEXT gate: NONE** (`rw_perception_next=0`). **Info benefit: some but weak** —
slot1 is masked-zero until the next gate is geometrically detectable, so acquiring it populates slot1 for the
upcoming approach; but with no next-gate reward and no next-gate prior in the 21-dim obs, this is a minor scan driver.

**The dominant yaw driver is CURRENT-gate re-acquire, and it is a limit cycle:**
- Yaw is controlled (ch3) and **yaw RATE is observed (obs[5:8]) but ABSOLUTE HEADING is NOT** (egocentric,
  position-free). The policy can only null the *observed* gate bearing (slot0 body rel_pos).
- The only yaw carrot is `rw_perception` (center the gate → cos_view↑) + keeping the gate detectable to populate
  slot0. That bearing signal is **noisy** (anisotropic σ lat 0.10 / vert 0.28 / depth 0.85 m) and refreshed
  only at **10.5 Hz with decay + a re-acquisition snap** (stale horizon 0.5 s; ego_estimator.py:137,726-730),
  and **masked entirely near the gate (blackout ~4.7 m) and when it leaves the FOV.**
- A delayed/noisy bearing-nulling controller with a **loose ±0.7 clamp** and **no blur cost**
  (`ego_blur_gate=false`) and **no absolute reference** overshoots → rails → re-acquires the opposite sign →
  rails back = the **±0.64 rad/s, 4–7 flips/s** wire signature. The dither penalty (−0.245/flip ≈ −0.04/step
  avg) is far too small vs a +20 gate to damp it.
- **8-gate vs 2-gate:** 8 gates = **7 handoffs + course turns (g2→g3 needs ~50–61° bank)**, each forcing an
  aggressive off-axis re-point and re-acquire; the 2-gate champion (vpeffs0, quiet 0.151/2.2) has one handoff on
  a near-straight leg. The 8-gate fine-tune trains the hard-yaw re-point behavior in — matching the memory note
  "8-gate ADDED the yaw hunt."
- **Deploy amplifies:** the expansive `faithful_rate` plant realizes MORE rotation per unit command than the
  training small-signal gain, so a trained near-rail yaw command reads as larger realized oscillation on the wire;
  the real detector cadence/miss also drives more masking → more re-acquire.

---

## Q6 — PRESCRIPTIONS (ranked; mechanism killed / risk / rule-conflict)

**Ranked recommendation:** (ii) ≻ (i) ≻ (iv) ≻ (v) ≻ (iii, reframe) ≻ (vi, decline).

1. **(ii) Overspeed episode-abort BY CONSTRUCTION** — *kills:* velocity runaway (Q2). Fatal at a speed well
   above racing (e.g. ‖v‖>~13–14 m/s sustained), collision-class exactly like the spin abort (route through the
   `lethal` fold, peregrine_racing_ego.py:1540, so it pays terminal_base+forfeit and can't be a free exit).
   *Why best:* it is the **no-spin precedent** (guarantee-by-construction, not incentive) the owner already
   blessed; it caps the actual failure variable (top speed) and forces braking skill into the curriculum.
   *Q4 interaction:* speed IS observable (obs[0:3]) so the policy CAN learn to respect it — unlike absolute-Z
   which sank R0. *Risk:* threshold must sit clearly above the racing band or it stalls; the obs velocity is
   estimator/IMU-drift, so set the abort on GT ‖v‖ (legal in terminations) with generous margin. *Rules:* pitch
   free ✓, not a deploy band-aid ✓, doesn't fence a course-used axis ✓.

2. **(i) SATURATE progress at v\*** — clip the per-step arc credit at `v*·dt` (no reward above v*, **no penalty**)
   — *kills:* the per-step speed-magnitude gradient (Q1/Q2 lever 1) with **zero stall risk** (unlike the rejected
   v_cap hinge, which was a penalty). *Honest limit:* it does **NOT** remove the discount/time/finish
   pull-forward (Q2 lever 2–3), which is intrinsic to any time-discounted race — so **(i) alone only reduces
   runaway; pair with (ii).** *Rules:* clean (no penalty, pitch free). Cheap to add (`clip(Δs, v*·dt)`).

3. **(iv) Strengthen yaw Δcmd / add duty shaping** — *kills:* the oscillation (Q5). The jerk form ALREADY exists
   (`rw_yaw_dither`) and is directive-safe (a sustained turn has Δ≈0 → preserves the yaw→gate→altitude
   coupling); it is just **too weak at 0.125**. First lever: **raise rw_yaw_dither** (0.125→0.4–1.0, the
   nodither-calibrated range) and/or add a **rate-of-sign-change (duty) penalty**, NOT a magnitude penalty.
   *Risk:* over-taxing sustained gate-tracking yaw would break the hard coupling rule — the squared-jerk/duty
   forms avoid this by construction. *Also tighten the deploy yaw clamp* toward the trained 0.35 rather than 0.7
   (a deploy setting, not training).

4. **(v) Fill slot1 from the coarse-map prior when unseen** (or enable `ego_obs_v2` so obs[21:23] gives the
   next-gate sector) — *kills:* the next-gate acquisition scan value (Q5). *Risk:* low; but this addresses only
   the minor next-gate component, not the dominant current-gate re-acquire limit cycle. Modest.

5. **(iii) "Put KF velocity in obs"** — **NO-OP as stated: velocity is ALREADY obs[0:3]** (see Q4). obs_v2 adds
   the next-gate *sector*, not velocity. The real gap is that observable velocity is **unpriced** — so the fix is
   (i)/(ii), not adding an obs channel. If the concern is that the *estimator* velocity is too drifty to be a
   reliable "am I too fast" cue, that is an estimator-fidelity issue, orthogonal to the ledger.

6. **(vi) Acceleration cap (Fengyou's Q)** — **decline.** Training-side: an accel penalty is an **effort/energy
   penalty** (violates the no-energy-penalty directive) AND doesn't cap the failure variable (you reach high
   *speed* with low *accel*); accel is **not in the obs**, so pricing it is OOD/stall-prone like v_cap.
   Deploy-side: an accel clamp is a **deploy band-aid** (violates no-deploy-bandaids) and fences the plant
   response. **Prefer the overspeed abort (ii) + progress saturation (i)** — they cap the real variable (top
   speed), on an observable quantity, without a penalty or a band-aid.

---

## §CODE-WINS — premise corrections (code beats the brief)

- **"KILL ON CONTACT = large negative":** partly overturned for arm A. Under the parabola regime a **gate FRAME
  STRIKE terminates but pays only the bounded parabola (~−15..−100) and KEEPS banked progress** — it does NOT
  pay terminal_base and does NOT forfeit (compute_ego_reward :1274-1290 charges only `lethal`+oob;
  `rw_clip_terminal=0`). Only floor/ceiling/spin (lethal, 100+forfeit) and OOB (200) are the expensive
  terminals. This makes riding-speed-into-a-clip *cheaper* than the brief assumes.
- **Progress is NOT a raw speed reward:** `rw_progress_to_center=False` (stage) ⇒ progress is a **telescoping
  segment-arc potential**, so per-meter credit and per-segment total are speed-independent; the runaway comes
  from discount+time+finish+per-step-magnitude, not from an unbounded speed term. The 39 m/s clip never binds.
- **obs_v2's 2 extra dims = next-gate coarse SECTOR, not velocity.** Velocity is already obs[0:3].
- **`ego_blur_gate=false` in arm A** (EXTRA overrides the stage's True) ⇒ the anti-spin motion-blur is OFF, so
  yaw sweeping incurs no detectability penalty — reinforcing the oscillation.
- **Perception can't be the pitch lever it's billed as:** `__post_init__` caps `rw_perception ≤ rw_time` (0.02),
  ~4% of progress at cruise — structurally too weak to hold the nose up.
- **Vision cadence:** at dt=30 Hz the 30 Hz frame clock fires ~every tick; net valid-fix rate = 30·0.35 ≈
  **10.5 Hz** (the "40 Hz / 3-of-4 ticks" in the code comment is stale DiffAero wording; the 10.5 Hz result is
  unchanged).
