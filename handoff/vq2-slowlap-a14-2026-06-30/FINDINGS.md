# VQ2 yaw-steer-sign probe — ATTEMPT 14 (measurement, NO fix) — 2026-06-30

**One-line outcome:** The two decision paths **DISAGREE** → escalating the raw table per your instruction.
Eyeball says FLIP (gate RIGHT, nose turns LEFT/away); the estimator-independent cross-check says DON'T-flip
(`wire_yaw` and `raw_gyro_yaw` opposite-signed). The reconciling tell: **`yaw_est` increases toward `yaw_des`
while the drone physically turns the other way → the ESTIMATOR yaw is inverted vs reality, not necessarily
`body_rate_sign[2]`.** Recommend: do NOT auto-flip the wire knob; the inversion looks like estimator/vision-yaw.

**Branch flown:** `claude/loving-galileo-92f020` @ `b4a62dd` (logs `yaw_des_rad` + `raw_gyro_yaw`, instrumentation
only). `--gate-seeker --deploy-profile vq2_case_c --label vq2_slow_seeker_a14`. Loop **11.8 Hz / CHOKED**, 55
ticks; CRASH gates=0. (First attempt crashed in 1.6s with the gate dead-ahead — no signal; this re-fly reached
the gate-RIGHT geometry. Both runs auto-restarted via the GO key from the post-crash waiting state.)
`[seeker-diag] cmds=42 pursuit=27 none=15 (valid_empty=8 continuity=7) bridged=15 (100%) held_legacy=0`.

## THE RAW TABLE — gate clearly to the RIGHT (`yaw_des > +10°`), t4.0–4.5 s
```
 t(s) yaw_des  wire_yaw(br2)  raw_gyro_yaw  yaw_est  roll  pitch  z(up=-)  thr
 4.0   +14.9°    -0.896        -0.009        +2°    +12°   +0°    -2.4    0.60
 4.1   +10.x°    -0.419        +0.231        +3°    + 7°   -0°    -2.3    0.60
 4.2   +18.0°    -0.282        +0.902        +8°    + 4°   -2°    -2.3    0.05
 4.3   +23.7°    -0.748        +0.256        +9°    + 5°   -3°    -2.6    0.05
 4.4   +27.5°    -1.030        +0.244       +10°    + 5°   -3°    -2.5    0.05
 4.5   +33.4°    -1.019        +0.694       +14°    + 2°   -2°    -2.4    0.60
 4.5   +42.5°    -1.009        +0.880       +18°    + 0°   -1°    -2.3    0.60
```
(`raw_gyro_yaw` lags the wire by ~1–2 ticks: at t4.0 the wire just stepped negative and the gyro hasn't
responded (−0.009); from t4.2 on, the sustained negative wire yields a **positive** raw gyro = opposite sign.)

## The two paths
- **PRIMARY (eyeball, pilot):** gate is clearly to the RIGHT (`yaw_des` +15°→+42°), and the nose **physically
  turns LEFT — away** (*"towards the end the drone was actively turning yaw left, like it was turning away"*).
  Per the tree: `yaw_des>0 AND nose LEFT` → **wire-yaw inverted → FLIP body_rate_sign[2]**.
- **CROSS-CHECK (estimator-independent):** `sign(raw_gyro_yaw) != sign(wire_yaw)` on 5/6 ticks (wire −, gyro +)
  → per the tree → **wire-yaw CORRECT → do NOT flip, look elsewhere**.
- **THEY DISAGREE.** Per your note ("if they disagree, send me the raw table"), here it is.

## Reconciling tell (my read — your call to confirm)
`yaw_est` rises +2°→+18°, i.e. TOWARD the positive `yaw_des` — the **estimator believes it is tracking the
gate correctly** while the drone physically turns the OPPOSITE way (left, away). That points to the
**estimator's yaw being inverted vs physical (yaw_est+ = physical LEFT)**, which is consistent BOTH with the
cross-check (the wire→physical sign is fine → don't flip `body_rate_sign[2]`) AND with the eyeball (it turns
away because the seeker's whole yaw frame — `nav.yaw` / vision-yaw / gate-bearing — is yaw-inverted after the
full gyro `(-1,-1,-1)` negation that also flipped yaw). So the one-knob fix is likely **the estimator/vision
yaw sign, NOT `body_rate_sign[2]`.** Flagging rather than asserting — multiple conventions interact; your resolve.

## CONFOUND (pilot, important): the drone never commits nose-down pursuit
*"there was little pitch-down across any of our flights ... it's just sitting on the ground being chaotic."*
Data agrees: whole-flight pitch mean **−3°**, only 16/55 ticks below −5° nose-down; thrust bang-bangs 0.05↔0.60.
The yaw read is taken from a drone that is NOT in clean airborne pursuit (post-handoff chaos). Root tie-in:
A13/A14 `pursuit` ≈ valid-pose ticks only, and `valid_empty` dominates the gaps — pose-starved, so the
proven nose-down forward feedforward (`vq2-forward-pitch-branch-a`) almost never runs. **Even with yaw fixed,
the drone won't thread gates until it (a) survives the egress→pursuit handoff without the nose-up whip + (b)
gets poses often enough to sustain nose-down pursuit.**

## Recommendation
1. **Resolve the yaw knob from the table above.** My lean: the inversion is in the estimator/vision yaw, not
   the wire `body_rate_sign[2]` (cross-check says wire is fine; eyeball + `yaw_est`-tracks-`yaw_des` say the
   frame is inverted). A wire flip would likely be wrong.
2. The yaw fix is necessary but **not sufficient** — the dominant blocker remains pose-starvation +
   handoff-whip keeping the drone out of sustained nose-down pursuit (Leads from A13).

## Artifacts
- `nav_estimate.jsonl` (now incl. `yaw_des_rad`, `raw_gyro_yaw`), `seeker_diag.txt`. Recording on ShadowPC:
  `data/runs/20260630_232739_vq2_slow_seeker_a14_f1/onboard.mp4` (3×). Flight stack untouched; handoff dir only.
