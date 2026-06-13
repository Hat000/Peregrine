# Autonomy-Readiness Audit — Submitted Stack, Unattended Judged Run

**Project:** Peregrine (Anduril AI Grand Prix). **Date:** 2026-06-13. **Mode:** ultracode (multi-agent Workflow).
**Question audited:** *will the Python stack we SUBMIT run fully autonomously and survive a judged, unattended,
8-minute-capped run without a DQ, a hang, or a fail-to-finish?* (spec §7: any human interaction during a timed
run = immediate DQ; gate contact = INVALID run).

**Scope:** the SHIPPED autonomous deploy path — `rl/fly_rl.py` (RL standing-start) + its supporting modules
(`src/racer/{finish_hold,race_outcome,mavlink_client,contracts,frames}.py`) and `scripts/fly_vq1.py` (the CTBR
entry, used as a contrast oracle). **NOT** in scope: our ShadowPC test-rig (FlightSim login, Win32 foreground,
GUI restart, fullscreen-minimize) — classified separately in §5.

**Method:** read + analysis + **offline scratch failure-injection** only (no live sim, no sockets, no GUI, no
source edits). ~80 scratch probes under `scratch/` exercise the *real* shipped functions (parser, guards, obs
pipeline, finish logic) with crafted/degraded inputs. All recommended fixes below are **NOT applied** — this is
an audit. Findings were adversarially verified (refute-by-default), then a completeness critic looped until dry.

> **Coverage note (honesty):** a transient server-side rate-limit knocked out 7 of 8 category finders on the
> first workflow pass; the critic loop + the surviving numerical finder still produced findings R1–R5. A second
> **gap-fill** workflow then ran dedicated finders for the four uncovered categories (8-min-cap,
> bad-start/recovery, determinism, degraded-input) plus a re-check of auto-arm, producing the net-new findings
> D1, D2, AR1, N1, n3. All eight categories are now covered. Two refuted-but-forward-looking items (post-run
> disarm confirmation; spin-margin for the envelope-relaxation retrain) are carried in §6/§8.

---

## 1. Executive summary & readiness verdict

**The inc7 policy is proven to fly (live-confirmed, VQ1 passed). The shipped LAUNCH PATH is NOT submission-safe
under bare defaults.** Three things would bite a judged run *today*:

1. **A DQ-class wire behavior is ON by default.** With default flags, `fly_rl.py` auto-reset transmits a real
   **`MAV_CMD 31000` (SIM_RESET)** onto the judged link whenever the organizer's start telemetry diverges from
   our ShadowPC model (stale countdown, lagging pose, or drone not exactly at origin). A client-issued sim
   perturbation during a timed run is the textbook §7 DQ. **(Rank 1.)**
2. **The default invocation is dead-on-arrival.** `--bridge` defaults **True** and loads a gitignored map
   (`data/runs/track_map_...json`) absent from any clean checkout → uncaught `FileNotFoundError` **after arm** →
   process dies **with the drone left ARMED** (the `[safety]` disarm sits after the loop, not in a `finally`,
   and the only handlers catch `KeyboardInterrupt`, never sent on an unattended run). The default `--checkpoint`
   also still points at the **retired inc4** actor. **(Rank 2.)**
3. **Any uncaught exception leaves the drone armed.** Because the force-disarm is not in a `finally` and the
   handlers are `KeyboardInterrupt`-only, *every* crash path (missing map, a degenerate-quat `ValueError`, a
   pre-arm recorder `OSError`) ends with the vehicle armed and the last command latched.

**Top three fixes before submission (none require a retrain):**

- **F-A (config, zero code):** ship a **committed launch wrapper** pinning
  `--checkpoint <repo>/rl/checkpoints/stage1_inc7_actor.pth --no-bridge --no-auto-reset --no-debug-obs`.
  This single artifact closes **R1, R2, R3** and the wrong-checkpoint trap at once. (Better still, flip those
  *defaults* so a bare `python rl/fly_rl.py` is also safe.)
- **F-B (one code change, highest leverage):** move the `[safety]` `disarm(force=True)` into a `finally` that
  wraps the **entire** loop + bridge + finish-hold, and broaden the per-flight/outer handlers from
  `except KeyboardInterrupt` to `except Exception`. Converts every crash into a *disarmed* exit. Closes the
  "dies armed" half of **R2, R3, R4, n3**.
- **F-C (one small struct + gate):** add a **dedicated ODOMETRY recv-timestamp** to `DroneState` and gate both
  command emission and the spin guard on its per-field age (~0.1 s). Closes **D1 and D2** — the most important
  net-new findings — which today are invisible to *every* existing guard.

**Crucial fix interaction:** F-A's `--no-auto-reset` is exactly the config that turns **N1** (a late-joined race)
into a *silent* NO_GO (never armed). The wrapper must therefore also widen the fresh-GO acceptance gate (§3, N1).

**Readiness verdict:** ⚠️ **NOT ready to submit as-is.** With the config wrapper (F-A) the DQ and the
default-crash are removed immediately; with F-B + F-C + the late-join gate the residual fail-to-finish surface is
closed. None of this needs a model change. The dominant residual risks (D1/D2, N1) hinge on two organizer
unknowns (per-message-type telemetry reliability; race-start-vs-process-launch ordering) — see §7 — but the
recommended fixes are cheap insurance regardless.

---

## 2. Prioritized DQ / HANG / FAIL-TO-FINISH register

Severity order: **DQ > HANG > FAIL_TO_FINISH > DEGRADED_TIME.** "Closed by wrapper?" = whether the config-only
launch wrapper (F-A) alone neutralizes it (no code change). "Conf" = confidence the mechanism is real.

| # | Severity | Finding | Trigger likelihood | Closed by wrapper (F-A)? | Conf |
|---|----------|---------|--------------------|--------------------------|------|
| R1 | **DQ** | Auto-reset transmits `MAV_CMD 31000` on any start-telemetry divergence | Fires on any model mismatch; **ON by default** | ✅ `--no-auto-reset` | med |
| R2 | FAIL | `--bridge` default loads absent gitignored map → uncaught `FileNotFoundError` after arm, left ARMED | **Certain** on bare-default invocation | ✅ `--no-bridge` | high |
| n3 | FAIL | Pre-arm `Recorder.start()` `OSError` on read-only/full CWD escapes `KeyboardInterrupt`-only handler → dies pre-arm | If eval CWD/`data/runs` not writable | ❌ needs F-B+code | high |
| D1 | FAIL | ODOMETRY-only drop (LPN/IMU survive) → policy flies open-loop on **frozen stale attitude/rate**; all guards blind | If per-msg-type loss possible on eval host | ❌ needs F-C | high |
| D2 | FAIL | Spin guard false-`SPIN_ABORT`s on a **frozen stale** `\|w\|≥6` during an ODOMETRY drop → mid-air disarm | Same outage + rate latched high at drop | ❌ needs F-C | high |
| N1 | FAIL | Late-joined race (`to_go ≤ −2000 ms` at first sample) → **silent NO_GO**, never armed, under `--no-auto-reset` | If harness can present a running race at attach | ❌ **created by F-A** | med |
| AR1 | FAIL | Single `arm()`, never re-sent; recoverable `TEMPORARILY_REJECTED`/slow armed-bit → terminal `ARM_REFUSED` | If first arm ever transiently rejected | ❌ needs code | med |
| R4 | FAIL | Zero-norm/NaN ODOMETRY quat → `scipy ValueError` in `build_obs` escapes RL loop → dies ARMED, last cmd latched | Latent (no malformed quat in 17+ live runs) | ❌ needs F-B+F-C | med |
| R3 | FAIL | NaN-component quat in bridge phase → `scipy ValueError` escapes `run_bridge` while ARMED | Latent; **bridge path only** | ✅ `--no-bridge` (unreachable) | med |
| R5 | FAIL | Non-finite (NaN/inf) telemetry → NaN `body_rate`+`thrust` straight to the wire (no finite guard) | Latent; rare wire/sim anomaly | ❌ needs F-C/wire guard | med |

### Detailed entries

**R1 — [DQ] Auto-reset transmits `MAV_CMD 31000` (SIM_RESET) onto the judged link.**
- *Trigger:* shipped defaults give `auto_reset = not args.no_auto_reset = True` ([fly_rl.py:611](rl/fly_rl.py:611), [:919](rl/fly_rl.py:919)). In `wait_fresh_go` the reset-suppression `next_reset = max(next_reset, now+reset_after)` ([fly_rl.py:567](rl/fly_rl.py:567)) runs **only** inside `if rs and rs['started'] and live → elif fresh`. Three divergences from our ShadowPC start model skip suppression and, after `--reset-after` (8 s), fire `client.send_sim_reset()` repeatedly: (C) countdown elapsed >2 s ago → `to_go ≤ −2000` → `fresh=False`; (D) `position_ned`/ODOMETRY lags RACE_STATUS so `live=False`; (F) genuinely fresh+elapsed GO but drone not within 5 m of origin (takes the outer `if`, not the `elif`).
- *Current behavior (scratch `ar2_1_wait_fresh_go.py`):* nominal fresh GO at origin → 0 resets. Stale-GO → 8 resets fired; pos-None → 8; valid GO but drone 50 m off → 7. `send_sim_reset()` is a real `COMMAND_LONG(MAV_CMD 31000)` ([mavlink_client.py:489](src/racer/mavlink_client.py:489)) on the **same UDP wire** as arm/control — not a GUI quirk.
- *Code:* [fly_rl.py:545](rl/fly_rl.py:545),[:554-586](rl/fly_rl.py:554); contrast [fly_vq1.py:110-165](scripts/fly_vq1.py:110) `_wait_for_race` **never** auto-sends a reset.
- *Fix (NOT applied):* the submission must **never emit a sim-control command**. Hard-disable auto-reset (default `auto_reset=False`; gate `send_sim_reset()`+`kick_sim_from_home()` behind an explicit `--dev-auto-reset`), adopting fly_vq1's passive wait-for-GO posture. Also move the suppression push above the `to_go`/`pos_off` checks so a fresh-but-not-yet-elapsed or non-origin countdown can never leak a reset.

**R2 — [FAIL] `--bridge` default True loads the gitignored default map → crash after arm, drone left ARMED.**
- *Trigger:* bare `python rl/fly_rl.py`. `--bridge` defaults True ([fly_rl.py:902](rl/fly_rl.py:902)); `fly_once` waits GO, **arms** ([fly_rl.py:620](rl/fly_rl.py:620)), then `run_bridge` → `build_bridge(args.map)` → `load_track_map('data/runs/track_map_20260602_114630.json')`. `data/runs` is gitignored → absent in a clean checkout/package → `Path(path).read_text()` ([navigator.py:163](src/racer/navigator.py:163)) raises `FileNotFoundError`.
- *Current behavior (scratch `m1_bridge_map_missing.py`, `probe_bridge_map_missing.py`):* the raise is **outside** the per-flight try (which catches only `KeyboardInterrupt`, [fly_rl.py:1011](rl/fly_rl.py:1011),[:1045](rl/fly_rl.py:1045)); the `[safety]` disarm ([fly_rl.py:858](rl/fly_rl.py:858)) is past the escape point → **skipped**. Process dies with a traceback, drone ARMED in ACRO receiving no setpoints. The standing-start (`--no-bridge`) path is map-free (uses hard-coded `_GATE_POS_ZUP`).
- *Fix:* flip `--bridge` default False (standing start is the deployment target and is map-free); and/or pre-validate `args.map` existence at load time (before arm); and pin the wrapper. Also F-B (finally-disarm).

**n3 — [FAIL] Pre-arm `Recorder.start()` `OSError` escapes the `KeyboardInterrupt`-only handler → dies pre-arm.**
- *Trigger:* eval host CWD (or `data/runs`) is read-only / permission-denied / full. `Recorder.start()` runs unconditionally before any arm and does `mkdir(parents=True)` + three `open()` against the **relative** path `data/runs/<stamp>` at the organizer's CWD ([fly_rl.py:996-998](rl/fly_rl.py:996), [recording.py:92-102](src/racer/recording.py:92)).
- *Current behavior (scratch `n3_recorder_prearm.py`, `n3_readonly_acl.py`):* `start()` raises `PermissionError`/`FileExistsError` with `_started=False`; it's **outside** the inner try ([fly_rl.py:1011](rl/fly_rl.py:1011)) and the OSError escapes the `KeyboardInterrupt`-only outer handler ([fly_rl.py:1045](rl/fly_rl.py:1045)); arm is never reached. Clean fail-to-finish (drone never armed). `fly_vq1.py:370-372` has the same pre-arm `start()` outside its try.
- *Fix:* make recording non-fatal (`try/except OSError → rec=None`) and CWD-independent (resolve the session dir against an absolute, configurable, writable root with a `tempfile` fallback); broaden the handler (F-B).

**D1 — [FAIL] ODOMETRY-only drop while LPN survives → open-loop flight on a frozen stale attitude/rate.**
- *Trigger:* mid-race the 75 Hz ODOMETRY stream stalls while 97 Hz LOCAL_POSITION_NED and 120 Hz HIGHRES_IMU keep arriving (selective per-msg-type loss / render hitch). ODOMETRY is the **sole** source of the attitude quat + angular rate; LPN carries pos/vel; HIGHRES_IMU drives `sim_time_ns`. **No NaN involved** (distinct from R5).
- *Current behavior (scratch `d1_odo_drop_real_parser.py`, run through the REAL `_handle`):* `dataclasses.replace()` on an LPN update ([mavlink_client.py:237-245](src/racer/mavlink_client.py:237)) refreshes pos/vel but **preserves the last ODOMETRY quat+rate verbatim**. The None-guard ([fly_rl.py:760](rl/fly_rl.py:760)) tests only is-None → never trips. `sim_time_ns` keeps advancing (HIGHRES_IMU) so the 1.5 s stall guard never fires. The only staleness signal — `odo_age_ms` in the debug log — uses the **shared** `recv_monotonic_ns`, which LPN refreshes, so it reads ~0 ms even when the quat is seconds stale. Measured: after a continued roll the policy was fed an attitude **~85° wrong** (frozen 35° bank vs true 310°), then commands body-rates open-loop until the 120 s cap (TIMEOUT) or a gate/env CONTACT (INVALID).
- *Root cause:* `DroneState.recv_monotonic_ns` ([contracts.py:71](src/racer/contracts.py:71)) is a **single shared** arrival stamp bumped by HIGHRES_IMU, LPN, and ODOMETRY alike ([mavlink_client.py:232](src/racer/mavlink_client.py:232),[:243](src/racer/mavlink_client.py:243),[:258](src/racer/mavlink_client.py:258)) — there is no per-field freshness.
- *Fix (F-C):* add `odo_recv_ns` to `DroneState`; gate command emission on `(monotonic_ns − odo_recv_ns) > ~0.15 s` → hold/hover for a bounded recovery window, then abort with a new `ODO_STALE` state + disarm.

**D2 — [FAIL] Spin guard false-`SPIN_ABORT`s on a frozen stale `|w|≥6` during an ODOMETRY drop → mid-air disarm.**
- *Trigger:* ODOMETRY stops while HIGHRES_IMU keeps `sim_time_ns` advancing, with the last `|angular_rate_body|` latched ≥ `spin_rate_abort` (6.0) at the drop (e.g. a post-gate-flare transient near the rate rail); gap > `spin_time_abort` (2.0 s). High-rate dual of D1, same root cause.
- *Current behavior (scratch `d2_spin_guard_stale_rate.py`):* the spin guard reads `s.angular_rate_body` ([fly_rl.py:747](rl/fly_rl.py:747)) **before** the None/partial-state guard ([fly_rl.py:760](rl/fly_rl.py:760)); the rate is frozen (ODOMETRY-only field) and `angular_rate_body` defaults to **zeros, never None** ([contracts.py:81](src/racer/contracts.py:81)) so the None clause is dead for it. A recoverable 2.5 s ODOMETRY blip → `SPIN_ABORT` at ~2.37 s → skips finish-hold → unconditional force-disarm mid-air. A 1.5 s blip correctly does not fire; a full IMU+ODO stall lets the 1.5 s stall guard win first (bounded).
- *Fix (F-C):* gate the spin guard on `odo_recv_ns` age (only fire on a fresh rate), and **move the partial-state/staleness gate above the spin guard** (ordering bug).

**N1 — [FAIL] Late-joined race → silent NO_GO under `--no-auto-reset` (created by the R1 fix).**
- *Trigger:* the process's first parsed RACE_STATUS shows a race already running >2000 ms past GO in sim-boot clock (`to_go = race_start_boot_ms − sim_boot_ms ≤ −2000`). Arises if the organizer starts the race before our process pumps its first RACE_STATUS, or we attach into a running race.
- *Current behavior (scratch `n1_late_join.py`, `test_wait_fresh_go_late_join.py`):* `fresh = race_start_boot_ms ≥ 0 AND to_go > −2000.0` ([fly_rl.py:556](rl/fly_rl.py:556)) — a **strict, irreversible** band; once `to_go ≤ −2000` the race is never accepted. Under `--no-auto-reset` (`next_reset = now + 1e18`), the loop spins to the 180 s deadline and returns False → `NO_GO` → batch break → with `--flights 1`, exits **never armed**, silently (race fully visible, `started=True`, throughout). Same gate in `fly_vq1.py:135`. **This is exactly the config R1's fix mandates.**
- *Fix:* for the `--no-auto-reset` deployment config, accept the **first-seen STARTED race at origin** (`pos_off ≤ 5 m`) with `active_gate_index == 0` and not finished, instead of relying solely on the `to_go > −2000` band; keep `to_go > 0` as a "wait, never control before GO". Log `to_go`+`started` on NO_GO. **Gated on the organizer launch-ordering answer (§7).**

**AR1 — [FAIL] Single `arm()`, never re-sent; recoverable rejection ends the one-shot run.**
- *Trigger:* the autopilot returns `MAV_RESULT_TEMPORARILY_REJECTED(1)` (momentary pre-arm: sensor settle, mode-not-ready, GO-edge timing — clears on a re-send) OR ACKs ACCEPTED but the `SAFETY_ARMED` bit doesn't flip within 5 s.
- *Current behavior (scratch `arm_recovery_probe.py`, `r3m4_arm_no_retry.py`):* `arm()` is sent once ([fly_rl.py:620](rl/fly_rl.py:620)); the ACK result code is **only printed** ([fly_rl.py:623](rl/fly_rl.py:623)), never branched; the sole gate is `wait_armed(True, 5.0)`. For a rejection the bit cannot flip (no second arm in flight) → timeout → `ARM_REFUSED` → `main()` breaks the batch ([fly_rl.py:1036](rl/fly_rl.py:1036)) → with `--flights 1`, exits 0 having flown nothing. Clean fail-to-finish (no errant wire command), **not** a DQ. `fly_vq1.py:441-447` is structurally identical.
- *Fix:* bounded arm re-send (≤3 attempts, short backoff, re-check `wait_armed` each cycle; branch on `ack['result']`; escalate to `arm(force=True)` — the 21196 bypass is already wired at [mavlink_client.py:480](src/racer/mavlink_client.py:480) — as a last attempt) before declaring `ARM_REFUSED`. Keep the budget well inside the 8-min cap.

**R4 — [FAIL] Zero-norm/NaN ODOMETRY quat → `scipy ValueError` in `build_obs` escapes the loop → dies ARMED.**
- *Trigger:* the sim emits an all-zero or NaN-component ODOMETRY quat mid-race. `_handle` stores it as a populated (non-None) array → the None-guard passes → the array reaches the rotation decode.
- *Current behavior (scratch `num2_quat_degenerate.py`, `num2_escape_path.py`):* scipy `Rotation.from_quat` **raises** on both a zero-norm quat and any NaN component. All-zero: absorbed by `_handle`'s `<1e-12` guards but then `build_obs` ([fly_rl.py:212](rl/fly_rl.py:212)) calls `_Rot.from_quat` **with no guard** and raises. NaN-component: the `<1e-12` guard is **NaN-blind** (`NaN < 1e-12` is False — independently confirmed at [frames.py:68](src/racer/frames.py:68),[:92](src/racer/frames.py:92),[:136](src/racer/frames.py:136),[:162](src/racer/frames.py:162)), so it raises one tick earlier inside `pump()→_handle`. The `ValueError` escapes `fly_once` (only `dbg_f.close` runs), bypasses both `KeyboardInterrupt` handlers, and the `[safety]` disarm never runs → drone left **ARMED with the last RL command latched** — the exact "minutes of uncontrolled spinning" the S17 guard was built to prevent. *(Trigger unconfirmed in 17+ live runs; the mechanism is certain.)*
- *Fix:* F-B (finally-disarm) + F-C (finite/zero-norm gate before `build_obs`); optionally make the `frames` guards NaN-safe (`not np.all(np.isfinite(q)) or (q@q)<1e-12`) and have `build_obs` use the already-guarded `frames.R_world_from_odo_quat_wxyz`.

**R3 — [FAIL] NaN-component quat in the bridge phase → `scipy ValueError` escapes `run_bridge` while ARMED.**
- Same NaN-blind-guard mechanism as R4, reached inside `run_bridge`'s `pump()`/`nav.update`/`mission.step` ([fly_rl.py:632-640](rl/fly_rl.py:632)), none wrapped; the only disarm-try there is the non-HANDOFF *return* branch, never an exception. **Bridge path only → made unreachable by `--no-bridge` (F-A).** Fix otherwise = F-B + NaN-safe `frames` guards.

**R5 — [FAIL] Non-finite telemetry propagates straight to the wire `body_rate`+`thrust` (no finite guard).**
- *Trigger:* a genuine NaN/inf in a CRC-valid ODOMETRY/LPN field (an inf quat specifically does **not** raise — it flows through; cf. R4 for the raising case). The RL path uses raw given telemetry (no KF, no division), so the realistic trigger is an already-non-finite field; the finite-overflow route needs `|state| ~ 1e21 m` (scratch `num1_finite_overflow_threshold.py`), unreachable in-range.
- *Current behavior (scratch `num1_nan_propagation.py`):* the None-guard does not trip on a populated NaN; `build_obs` → NaN obs → `policy_step` → `rate_frd=[nan,nan,nan]`, `collective=nan`. The only clips are inert (`--max-rate` defaults 0 OFF; `np.clip(NaN,·)=NaN`; the collective clip tames a huge *finite* thrust but not NaN). `send_command` casts `float()` with **no finite check** ([mavlink_client.py:422-433](src/racer/mavlink_client.py:422)) → NaN on the wire. The spin guard still fires (~2 s) on the resulting tumble, so the run terminates rather than riding the cap — but it is INVALID. `fly_vq1.py:545` has the finite abort the RL loop lacks.
- *Fix:* finite-reject the telemetry tick (F-C), plus a defense-in-depth wire guard before `send_command` (`if not all-finite → zeros + hover`). **A clamp alone is insufficient (`np.clip(NaN)=NaN`); the `isfinite` reject is load-bearing.**

---

## 3. Cross-cutting fix themes

Most findings collapse onto **four** changes:

- **F-A — Committed launch wrapper (config only, no code):** pin
  `--checkpoint <repo>/rl/checkpoints/stage1_inc7_actor.pth --no-bridge --no-auto-reset --no-debug-obs`.
  → closes **R1, R2, R3** + the inc4-default trap. Better: flip those *defaults* so a bare invocation is safe too.
- **F-B — `finally`-disarm + broaden handlers:** move `[safety]` `disarm(force=True)` into a `finally` wrapping
  the whole loop+bridge+finish-hold; change `except KeyboardInterrupt` → `except Exception` (both the per-flight
  [fly_rl.py:1011](rl/fly_rl.py:1011) and outer [fly_rl.py:1045](rl/fly_rl.py:1045)). Mirror `fly_vq1.py:628-634`.
  → closes the "dies ARMED" half of **R2, R3, R4, n3**. *Single highest-leverage code fix.*
- **F-C — Per-field ODOMETRY recv-age gate + telemetry finite gate:** add `odo_recv_ns` to `DroneState`; gate
  command emission **and** the spin guard on its age (~0.1–0.15 s); add an `isfinite`/zero-norm reject of the
  telemetry tick after the None-guard ([fly_rl.py:763](rl/fly_rl.py:763)), mirroring `fly_vq1.py:545`.
  → closes **D1, D2, R4, R5**. *Single struct change closes the whole degraded-input family.*
- **F-D — Late-join + arm-retry robustness:** widen the fresh-GO acceptance for `--no-auto-reset` (N1) and add a
  bounded arm re-send + force-arm last resort (AR1). → closes **N1, AR1**.

---

## 4. Independently confirmed load-bearing claim

The R3/R4/R5 mechanism rests on the `frames.py` quaternion guards being **NaN-blind**. Verified first-hand
([frames.py:57-165](src/racer/frames.py:57)): all four helpers guard with `(q@q) < 1e-12`; for a NaN-component
quat `q@q` is `NaN` and `NaN < 1e-12` is `False`, so the guard does **not** fire and scipy `from_quat` raises.
The docstrings promise "returns zeros rather than raising for a degenerate quaternion" — true for **zero-norm**,
**false for NaN**. `build_obs` ([fly_rl.py:212](rl/fly_rl.py:212)) calls `scipy.from_quat` with **no guard at all**.

---

## 5. Shipped-stack vs test-harness separation

| Item | Class | Note |
|------|-------|------|
| `send_sim_reset()` → `MAV_CMD 31000` from `wait_fresh_go` | **SHIPPED_RISK** | A real wire `COMMAND_LONG`, **not** a GUI action. ON by default. The rank-1 DQ. Must be excluded from the submitted entrypoint. |
| FileNotFoundError on gitignored default map | **SHIPPED_RISK** | Manifests on any clean checkout (`data/runs` gitignored). With `--bridge` default True → R2. A packaging gap, not a rig artifact. |
| Default `--checkpoint` = retired inc4 | **SHIPPED_RISK** | Ships the wrong (aero-blind) actor unless inc7 is passed. inc7 `.pth` **and** its `.json` sidecar are both git-tracked and co-located (the "missing-sidecar overdrive" candidate was **refuted** for the shipped artifact). |
| `Recorder.start()` pre-arm on a relative `data/runs` path | **SHIPPED_RISK** | n3 — a CWD-dependent pre-arm crash on any host with a non-writable CWD. |
| `--debug-obs` default True (per-tick `jsonl` write) | **SHIPPED_RISK** | Low severity, but the only per-tick blocking file I/O in the loop. Disable in the package. |
| No ODOMETRY per-field freshness (`recv_monotonic_ns` shared) | **SHIPPED_RISK** | Root cause of D1/D2; manifests on any host with per-msg-type loss. |
| `connect()` binds 14550/5600; `arm()` pre-heartbeat target_system | SHIPPED, **refuted** | `target_system` resolves 0→1 on the first vehicle HEARTBEAT (pumped throughout `wait_fresh_go` before the single `arm()`); armed gating reads the realized `SAFETY_ARMED` bit. Cleared, not a risk. |
| `kick_sim_from_home` / `full_sim_reset` Win32 escalation (`SetForegroundWindow`/`keybd_event`) | **TEST_RIG_QUIRK** | Drives the FlightSim GUI on our ShadowPC. No window to find on an eval host → no-op/benign. **But** the 31000 reset *precedes* the Win32 escalation in `wait_fresh_go`, so disabling only the Win32 part does **not** close R1 — auto-reset itself must be off. `full_sim_reset()` ships and is reachable with `--flights N>1`. |
| `--full-reset` default True (between-flights ESC→HOME) | TEST_RIG_QUIRK | Multi-flight dev tooling; never runs at `--flights 1` (the submission shape). |
| FlightSim login, 31000 GUI-restart-from-HOME, fullscreen-minimize, idle states | TEST_RIG_QUIRK | Entirely our ShadowPC operating environment; the organizer owns sim launch. Unreachable on an eval host. |
| Spawn-artefact (gate-3-crash 0-tick respawn cycle) | TEST_RIG_QUIRK | Requires a predecessor hard-collision full-reset; a cold `--flights 1` judged run has no predecessor → precondition absent. |
| Collision-FIFO slice miss (`n_coll0 ≥ 200`) / between-flight 31000 | TEST_RIG_QUIRK | Both require `--flights N>1`; at default `--flights 1`, `n_coll0=0` → whole-list slice → hard hit never missed. |

---

## 6. Refuted candidates (do not re-litigate)

Each was traced and, where useful, settled with a scratch test:

- **Missing inc7 sidecar → thrust overdrive** — REFUTED. Both `.pth` and `.json` are git-tracked + co-located
  in `rl/checkpoints/`; `.gitignore` matches `*.pt`/`data/runs`, not `.pth`/`rl/checkpoints`. (Mechanism real if
  separated; kept as a P2 fail-closed hardening — `num3_sidecar_overdrive.py`.)
- **`sim_finish_confirmed` false-finish** (`active_gate_index ≥ n_gates`; `finish_ns == 0`) — REFUTED. The sim's
  demonstrated values never reach the sentinels; the signal only gates a **bounded** post-loop recording-drain
  (`drain_until_finish` cap always wins) — `test_num6_stale_finish_blastradius.py`, `r3m2_*`, `r3m1_*`.
- **Stale `finished=True` at loop entry** — REFUTED (`cc_stale_finished_at_loopentry.py`): the freshness gate +
  SIM_RESET guard prevent a prior-race finish from latching the new run.
- **HIGHRES_IMU clock dependency** (sim_time_ns / stall guard) — REFUTED; HIGHRES_IMU is spec-mandated
  (`verify_m2_highres_imu.py`).
- **Single-shot arm reliability / stale-armed second flight** — REFUTED (the armed decision reads the realized
  `SAFETY_ARMED` bit). *Note:* the **transient-rejection** sub-case is a distinct, real finding = AR1.
- **torch cold-start racing the 2 s GO window** — REFUTED (the freshness check is sim-clock, not wall-clock).
- **Policy nondeterminism** — REFUTED (`test_policy_determinism.py`): `actor.eval()` + `tanh(mean)`, no sampling,
  bitwise-reproducible given the same obs. Wall-clock pacing varies *which sample* a tick sees (the benign
  bimodal lap time) but both modes finish clean — not a validity risk.
- **Offline grader epoch re-baseline** — REFUTED; `race_outcome.py` is not in the judged path.
- **Spin guard NaN-blind** — REFUTED (`num4_nan_guards.py`): `SPIN_ABORT` fires at ~2 s even under sustained NaN
  rate. (Teleport guard *is* NaN-blind but harmless — the finite-safe `reset_counter` primary fires on any reset.)
- **`send_command` missing wire clamp as an independent mode** — REFUTED (`test_num5_wire_clamp.py`): for finite
  inputs the policy `tanh` rail provably bounds `|rate| < 3.14`; the only non-finite path is R5's.
- **8-min-cap / unbounded-wait** — REFUTED as a hang: every wait is hard-bounded (`wait_fresh_go` 180 s deadline,
  RL loop `max_seconds` + 1.5 s sim-stall, `run_bridge` deadline+stall, `drain_until_finish` `max_hold_s`,
  `wait_armed`/`wait_command_ack` deadlines, inner pacing loops sleep+pump) — `timeout_sweep.py`,
  `t3_wallclock_cap.py`. The single-attempt cap-waste is a *strategy* point (Q5), not a code hang.
- **D3 IMU-gap self-heal** and **D4 nominal epoch skew** (<13.3 ms, flown clean) — REFUTED
  (`d3_selfheal_backlog.py`, `d4_nominal_epoch_skew.py`).

**Forward-looking (refuted-as-shipped, kept as guard-rails):**
- **n2 — final disarm confirmation discarded** ([fly_rl.py:861](rl/fly_rl.py:861) ignores `wait_armed(False)`):
  a dropped disarm datagram could leave the vehicle armed *after* the run is already decided. **Not** a DQ/finish
  risk (post-run cleanup), but make it confirmation-driven + bounded-retry (P2).
- **BSR3 — spin-margin razor-thin for the envelope-relaxation retrain.** Replay over 23 inc6 live traces shows the
  longest healthy contiguous `|w|≥6` run is 1.97–1.99 s vs the 2.0 s window — ~10–30 ms margin, **0/23** false-fires
  on shipped inc7. But scaling the busiest trace ≥1.5× (a relaxed-envelope proxy) **does** trip a false
  `SPIN_ABORT`. **Mandate a realized-`|w|` trace gate on any new checkpoint** (esp. the queued `rw_tilt 96→48`
  retrain) and/or widen `spin_rate_abort`→~9–10 rad/s + `spin_time_abort`→3.0 s for the relaxed policy
  (`bsr3_spin_margin_live.py`, `bsr3_closest_call.py`).

---

## 7. Sharpened organizer questions (gate autonomy readiness)

The two highest-impact unknowns determine whether N1 and D1/D2 are *latent* or *live*. Email
`info@theaigrandprix.com`:

1. **Race-start vs process-launch ordering (gates N1).** When our process attaches, is the GO guaranteed to occur
   *after* we're connected and pumping, or can the race already be running (>2 s past GO in sim-clock) at our
   first RACE_STATUS? *Assume-if-unanswered:* a running race may be presented → ship the conditioned freshness gate.
2. **Per-message-type telemetry reliability (gates D1/D2).** Are ODOMETRY, LOCAL_POSITION_NED, and HIGHRES_IMU
   delivered with the same reliability, or can ODOMETRY lag/drop independently (rendered less often / dropped under
   load) while LPN/IMU continue? *Assume-if-unanswered:* selective loss possible → ship the ODOMETRY recv-age gate.
3. **Submission interface / entrypoint contract.** Does the organizer invoke a fixed command line, an entrypoint
   function, or a container CMD? Can we ship a committed wrapper pinning the safe flags, used verbatim?
   *Assume-if-unanswered:* a bare-default run is possible → harden the *defaults*, not just the flags.
4. **§7 sim-control prohibition (gates R1).** Is sending any sim-control command (`MAV_CMD 31000`) during a timed
   run a DQ? *Assume-if-unanswered:* yes → auto-reset hard-off, passive wait-for-GO.
5. **Eval re-runs / determinism budget.** Single scored attempt or unlimited best-of-N? *Assume-if-unanswered:*
   single attempt (worst case) → every fail-to-finish mode is must-fix; F-B/F-C/F-D become must-have, not polish.
6. **State carried between attempts + start vehicle state.** Fresh process per attempt or kept alive? Vehicle
   guaranteed disarmed at origin (≤5 m) at run start? *Assume-if-unanswered:* fresh process, disarmed-at-origin;
   ship `--flights 1`, auto-reset off, a 1–2 tick origin-settle grace.
7. **Arm acceptance policy (gates AR1).** Can the first arm be `TEMPORARILY_REJECTED` / hold `SAFETY_ARMED` off
   >5 s during normal startup? Does any condition need `force=True`? *Assume-if-unanswered:* transient rejection
   possible → ship the bounded arm re-send + force-arm last resort.
8. **Eval hardware / VQ1 deadline.** What GPU/CPU (does `torch.load` + the warmup forward run on CPU?), and the
   firm deadline + any per-run startup budget before GO? *Assume-if-unanswered:* modest/possibly-CPU-only host,
   near-term deadline → keep the actor CPU-safe, disable `--debug-obs`, prioritize the no-retrain config fixes.

---

## 8. Submission autonomy-hardening checklist

**P0 — would DQ / brick a judged run (do before any submission):**
- [ ] **Ship a committed launch wrapper** pinning `--checkpoint .../stage1_inc7_actor.pth --no-bridge
  --no-auto-reset --no-debug-obs`; verify the inc7 `.pth` + `.json` ship. *(F-A; closes R1, R2, R3 + inc4 trap.)*
- [ ] **Hard-disable auto-reset** in the submitted path (`auto_reset=False` default; gate `send_sim_reset()` +
  `kick_sim_from_home()` behind `--dev-auto-reset`); never emit a sim-control command on the judged link. *(R1.)*
- [ ] **Flip `--bridge` default False** (standing start is map-free) and/or pre-validate `args.map` before arm. *(R2.)*
- [ ] **Move the force-disarm into a `finally`** wrapping the whole loop+bridge+finish-hold; broaden handlers to
  `except Exception`. *(F-B; closes the dies-armed half of R2, R3, R4, n3.)*
- [ ] **Add a dedicated `odo_recv_ns`** to `DroneState`; gate command emission + the spin guard on its ~0.1–0.15 s
  age; move the partial-state/staleness gate **above** the spin guard. *(F-C; closes D1, D2.)*

**P1 — fail-to-finish robustness (do if the eval is single-attempt):**
- [ ] **Telemetry finite/zero-norm gate** after the None-guard (`isfinite` of pos/vel/quat/rate; `q·q > 1e-12`):
  skip the tick / safe-hold, abort after N consecutive bad ticks. Mirror `fly_vq1.py:545`. *(R4, R5.)*
- [ ] **Make recording non-fatal + CWD-independent** (`try/except OSError → rec=None`; absolute writable root +
  `tempfile` fallback). Apply to `fly_rl.py` and `fly_vq1.py`. *(n3.)*
- [ ] **Condition/widen the fresh-GO gate** for `--no-auto-reset`: accept the first-seen STARTED race at origin /
  gate 0 / not finished; log `to_go`+`started` on NO_GO. *(N1 — the R1 fix's blind spot.)*
- [ ] **Bounded arm re-send** (≤3 attempts, backoff, branch on `ack['result']`, escalate to `force=True`). *(AR1.)*
- [ ] **Mandate a realized-`|w|` trace gate** on any new checkpoint (esp. the `rw_tilt 96→48` retrain); widen the
  spin thresholds for a relaxed envelope. *(BSR3 forward-looking.)*

**P2 — polish / defense-in-depth:**
- [ ] **NaN-safe `frames` guards** (`not isfinite(q) or (q·q)<1e-12`) + `build_obs` via `R_world_from_odo_quat_wxyz`. *(R3, R4.)*
- [ ] **Wire-level finite reject** before `send_command` (`if not all-finite → zeros + hover`). *(R5.)*
- [ ] **Confirmation-driven final disarm** + bounded retry (capture `wait_armed(False)`; loud STILL-ARMED log). *(n2.)*
- [ ] **Disable `--debug-obs`** in the package (removes per-tick blocking file I/O). 

---

## 9. Scratch failure-injection index

~80 offline probes under `handoff/ultracode-autonomy-readiness-2026-06-13/scratch/` (no sockets / no live sim;
exercise the real shipped functions). Key ones by finding:
- **R1:** `ar2_1_wait_fresh_go.py` · **R2:** `m1_bridge_map_missing.py`, `probe_bridge_map_missing.py` ·
  **n3:** `n3_recorder_prearm.py`, `n3_readonly_acl.py` · **D1:** `d1_odo_drop_real_parser.py` ·
  **D2:** `d2_spin_guard_stale_rate.py`, `probe_spin_falsetrigger.py` · **N1:** `n1_late_join.py`,
  `test_wait_fresh_go_late_join.py` · **AR1:** `arm_recovery_probe.py`, `r3m4_arm_no_retry.py` ·
  **R3/R4:** `num2_quat_degenerate.py`, `num2_escape_path.py`, `ar2_3_bridge_escape.py` ·
  **R5:** `num1_nan_propagation.py`, `num1_finite_overflow_threshold.py`, `num4_nan_guards.py` ·
  refutations: `test_num5_wire_clamp.py`, `test_num6_stale_finish_blastradius.py`, `num3_sidecar_overdrive.py`,
  `test_policy_determinism.py`, `timeout_sweep.py`, `d3_selfheal_backlog.py`, `d4_nominal_epoch_skew.py`,
  `bsr3_spin_margin_live.py`.

---

## 10. Residual gaps (gated on organizer answers)

- **N1** reachability hinges on race-start-vs-launch ordering (Q1). **D1/D2** reachability hinges on per-msg-type
  loss (Q2). Both fixes are cheap insurance regardless of the answers.
- We could **not** test against the *actual* organizer submission interface (unpublished — Q3). If the harness
  invokes our code differently than `python rl/fly_rl.py` (e.g. a callback or container CMD), the default-flag
  findings (R1, R2) may or may not apply verbatim — which is precisely why hardening the **defaults**, not just
  the flags, is recommended.

*Audit complete. All fixes are recommendations only; nothing in the shipped tree was modified.*
