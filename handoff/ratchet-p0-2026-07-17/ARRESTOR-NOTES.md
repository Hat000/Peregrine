# ARRESTOR build notes — `--ego-arrestor` post-gate stare-brake takeover (policy-flight re-target)

2026-07-18 · branch `ratchet-arrestor-2026-07-18` · worktree `wt-arrest`. Re-targets the
`docs/replay-ratchet-arrestor-DESIGN.md` machine (originally tape-replay takeover) to **POLICY
flights**, per the ratchet-P0.3 evidence (`REPORT4.md`): N=5 settled-launch champion-ckpt flights
banked {2,2,1,2,2}, gate-0 reproduces to 58 ms, 4/5 died too-fast/too-high entering gate 2, 0/5
contact-free. The arrestor converts "passed 2 gates then crashed" into "passed 2 gates, arrested
clean, continued" by taking over after a banked gate, bleeding speed while keeping the next gate in
view, and re-spawning the stateless policy into its training distribution.

## What / where

- **`rl/fly_rl.py`**
  - `_arrest_fmt` helper (`:987`) + `class ArrestCommand(NamedTuple)` (`:992`) + `class EgoArrestor`
    (`:1003-1240`) — inserted right after `EgoFloorClamp` (the flight-phase helper cluster). Pure,
    sim-free, injectable-clock state machine mirroring the `EgoTakeoffAssist`/`EgoFloorClamp`
    precedent.
  - CLI flags — `rl/fly_rl.py:3425-3451` (in the ego argparse block, before `--yaw-scale`).
  - Instantiation + banner — `rl/fly_rl.py:2303-2321` (right after `floor_clamp`).
  - Loop integration — `rl/fly_rl.py:2635-2683` (the `arr = None` branch, immediately after
    `obs = builder.update(...)` and replacing the flat policy_step+assist+floor block with an
    `if arr is not None: <arrest> else: <policy>` fork). Single send site unchanged
    (`rl/fly_rl.py:2684-2689`).
  - Forensics — append-only `arrest_phase` (`off`|`policy`|`arrest`) + `arrest_id` at the END of the
    `ego_obs.jsonl` record dict (`rl/fly_rl.py:2735-2736`). No existing field reordered/removed.
- **`tests/test_ego_arrestor.py`** — 16 unit tests (new).
- The `--sysid-replay` path (P0.1 replay code) is **untouched** — the arrestor lives only in the
  policy path, after the sysid `continue`.

## Flags + defaults

| flag | default | meaning |
|---|---|---|
| `--ego-arrestor` | **OFF** | master enable. OFF ⇒ `step()` returns None every tick ⇒ policy path byte-identical. |
| `--ego-arrest-after-gates` | `"1"` | comma list of gate indices whose PASS engages (each fires once). `"1"` = engage the tick `active_gate_index` advances 1→2 (just banked gate 1, before the gate-2 descend wall). `""` = no gate trigger. |
| `--ego-arrest-speed-hi` | `0.0` (off) | safety trigger: engage whenever estimated horizontal KF speed > this; re-arms after each handback with a ~2 s refractory. |
| `--ego-arrest-max-s` | `4.0` | hard arrest timeout → fail-open handback. |

Internal tunables (constructor defaults, not CLI, to keep the diff small — edit in-code if a flight
demands it): `handback_speed=2.0`, `handback_dwell_s=0.3`, `refractory_s=2.0`, `pitch_rate_cap=0.8`,
`pitch_level_rad=-0.05`, `pitch_hb_lo/hi=-0.45/0.0`, `roll_hb=0.17`, `k_yaw=1.5`, `k_brake=0.1`,
`k_z=0.15`, `collective_lo/hi_g=0.7/1.4`, `gate_lost_s=1.0`, `rel_up_drift_m=2.0`. Yaw clamp is taken
from the active `--ego-yaw-clamp` (falls back to 0.7 if that's 0).

## State machine (bullet diagram)

```
POLICY ──(gate p in after_gates just passed, airborne)──▶ ARREST ──(handback)──▶ POLICY
   │     ──(speed > speed_hi, airborne, past refractory)──▶   │
   │                                                          ├──(ABORT: rel_up drift >2 m)──▶ POLICY
   └────────────── never engages while ground_active ────────┤   (ABORT: gate lost >1 s)  ──▶ POLICY
                                                              └──(ABORT: arrest >max_s)   ──▶ POLICY
   every transition to POLICY arms a refractory_s window against the speed trigger.
   HANDBACK / all ABORTS fail-OPEN: step() returns None, the policy runs THAT tick (stateless MLP,
   nothing to reset). ENGAGE + every ABORT/HANDBACK prints a one-line transition record.
```

ARREST command (all ESTIMATOR-only, emitted in FINAL FRD — bypasses policy_step's virtual-flip/clamps):
- **YAW** `wz = clip(-K_yaw · bearing, ±yaw_clamp)`, `bearing = atan2(rel_left, rel_fwd)` from the
  builder's HELD slot0 `rel_flu` (TRUE body FLU `[fwd,left,up]`, survives detection gaps). Gate LEFT
  (bearing>0) ⇒ nose LEFT ⇒ `wz<0` (standard FRD, agrees with the design doc). `wz=0` until the next
  gate is acquired.
- **PITCH** `wy = clip(K_brake · speed, 0, pitch_rate_cap)` NOSE-UP to bleed speed, HARD-STOPPED
  (`wy=0`) once `obs[4] ≥ pitch_level_rad` (~level) — the +20° mount means a level body already puts
  the camera +20° up; pitching further walks the gate out the BOTTOM. Level-out brake, not reverse
  thrust.
- **COLLECTIVE** `g = hover + K_z·(rel_up − rel_up_entry)`, clamped `[0.7,1.4] g` → `rel_up` hold vs
  the gate. `rel_up_entry` is latched LAZILY at the first re-acquisition after engage (the just-banked
  gate's slot was reset, so there is no altitude ref until the NEXT gate is seen; hold hover meanwhile).
- **ROLL** `0` (hold wings).

HANDBACK when `speed < handback_speed` AND gate-in-view (det-proxy) AND ~level (`obs[4] ∈
[pitch_hb_lo, pitch_hb_hi]`, `|obs[3]| < roll_hb`), held for `handback_dwell_s`.

## Recommended first-flight recipe delta

Take REPORT4 §recipe verbatim and **add `--ego-arrestor --ego-arrest-after-gates 1`** (everything
else identical — TRT M-engine detector, `--ego-assist-thrust 1.3`, champion log-echo map, yaw-clamp
0.7, det-hold 0.2, etc.). Full:

```
PYTHONPATH=src python rl/fly_rl.py --endpoint udp:127.0.0.1:14550 --ego-ckpt ckpts/vpeffs0_actor.pth \
  --seeker-detector yolo --seeker-weights <M-engine> --rate 40 --ego-rate-scale 1.2 --virtual-flip \
  --ego-slot1 --ego-sector-mode map --ego-coarse-map <champion-log-echo-map> --ego-pitch-clamp 1.0 \
  --ego-yaw-clamp 0.7 --ego-det-hold 0.2 --ego-stale-horizon 0.5 --ego-assist-thrust 1.3 \
  --ego-arrestor --ego-arrest-after-gates 1 \
  --max-seconds 120 --wait-seconds 1800 --flights 1 --label ratchet_arrest_g1_rN
```

Optionally add `--ego-arrest-speed-hi 8` as a belt-and-suspenders safety brake (champion rode
6.4→15.2 m/s; 8 fires only when clearly hot). Launch ritual unchanged (`fly_rl` first → "Waiting
PASSIVELY" banner → GO; validity = zero gate contact).

## What to watch in the logs (`ego_obs.jsonl` + stdout)

1. **`[ego-arrest] ENGAGE #1 (after gate 1) gi=2 speed=… m/s`** must print right after the gate-1
   pass, and NOT during takeoff (the ground guard). If it never prints, gate-1 never banked (check
   the opening) or `--ego-arrestor` didn't take.
2. **Yaw-stare sign (THE one flight-unknown).** Watch `rate_frd[2]` vs the slot0 bearing (`rel_flu`
   in the record): the gate should re-CENTER, not walk out of frame. **1-BIT FLY-CHECK: if the stare
   drives the gate OUT of view, the yaw sign is flipped — negate `k_yaw` in `EgoArrestor`.** The
   fail-open (gate-lost abort at 1 s) protects against a wrong sign, but it defeats the purpose.
3. **Handback vs abort.** Success = `HANDBACK (slow+in-view+level) dur≈1–2s entry=…→exit<2 m/s`,
   then the policy continues past gate 2. Frequent `ABORT (gate lost …)` = the next gate (gate 2 is a
   DESCEND leg / below) isn't being reacquired while braking — the level-out pitch may point the
   camera too high; try lowering `pitch_level_rad` (keep the nose a touch more down) or arrest after a
   different gate. `ABORT (timeout 4.0s)` = the brake isn't reaching <2 m/s in the inter-gate gap
   (level-out decel is limited by design); expect this at gate-4/5 speeds — ratchet from the low
   gates first.
4. **`arrest_phase`/`arrest_id`** in the record segment the flight into policy vs arrest spans (during
   `arrest`, `actor_mean`/`act_raw` are `[]` and `rate_frd`/`collective` are the stare-brake command).
5. **Altitude hold.** `normed_thrust` (obs[8]) should track `hover ± K_z·drift` within `[0.7,1.4] g`;
   a rel_up-drift abort means the vertical estimate moved >2 m (PnP range error under blur) — a
   fail-open, not a crash.

## Tests

`tests/test_ego_arrestor.py` (16 tests, all green): OFF⇒never-commands (the byte-identical guarantee),
gate trigger (engage/ignore-unlisted/never-on-pad/fire-once), speed trigger + refractory, handback
(all-three + dwell + dwell-reset), all three aborts, yaw sign+clamp, pitch nose-up+cap+level-stop,
collective hold+`[0.7,1.4]` clamp, roll=0, yaw=0-when-no-bearing, output shape. Ran green with the 14
existing `fly_rl`-importing suites (153 passed / 23 skipped = gitignored-ckpt tests) + `--help` parse
smoke (flags register; defaults OFF/`"1"`/`0.0`/`4.0`).

## Deliberate scope calls

- **Secondary `--sysid-arrest-at-end` SKIPPED** (design §D "skip if any doubt"): wiring the arrestor
  into the `--sysid-replay` tape-end would disturb the P0.1 replay code (a hard DO-NOT) for a
  non-load-bearing slot. The primary (policy-flight takeover) is the load-bearing piece; kept the
  diff small + the replay path byte-identical.
- **Yaw sign** defaulted to standard-FRD `-K_yaw·bearing` (agrees with the design doc AND the
  invariance of the yaw axis under the virtual π-flip); flagged as the single first-flight eyeball
  check per the codebase's established 1-BIT FLY-CHECK pattern.
- **`green_gate.py` sentinel** (1091) will need bumping by +16 at merge — a new test file was added;
  that's the RL commander's pre-merge step, not done here.
