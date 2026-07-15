# Brief: multi-rate vision (~30 Hz) / control (40–60 Hz) for the next ego policy
_ShadowPC → RL-commander, 2026-07-15. Re: training the new policy with a slow vision input and a faster control output without the two conflicting._

## TL;DR
1. **The multi-rate staleness contract you want ALREADY EXISTS and is train↔deploy pinned. Reuse it — don't reinvent.** Staleness is already an obs; IMU-interpolation between vision updates is already the estimator's ego-propagation.
2. **Do NOT use a threaded async detector.** Ours failed (post-mortem §4). The multi-rate is just a single control loop that detects-on-new-frame and IMU-propagates between — which is what the deploy stack already does.
3. **Train vision at the DEPLOY cadence**, not fresh-every-tick. Video arrives ~30 Hz, but *valid gate detections* land at ~7–15 Hz and go **dark on final approach** (loss-onset ~4.7 m). Match that.
4. **Nits:** recorded stream is **30.0 Hz, not 28.8** (reconcile). Control = **60 Hz (2× stream)** gives clean 0/1-frame staleness; **40** is fine because age is already observed.

## 1. What already exists — reuse it (`src/racer/ego_obs.py`)
The deploy obs builder is a faithful, element-pinned port of the training obs (`rl/peregrine_racing_ego.py::ego_actor_obs` + `rl/ego_estimator.py`), verified by `tests/test_ego_deploy_obs.py`. The multi-rate machinery is **in the 21-dim contract**:

| idx | field | note |
|---|---|---|
| 0:3 | velocity (body FLU) | IMU-primary |
| 3:5 | roll, pitch | gravity-leveled |
| 5:8 | body_rates | |
| 8 | last_collective | |
| 9:11 | coarse_sector (h,v) | static per-gate turn prior |
| 11:14 | slot0 rel_pos (active gate) | masked→0 past horizon |
| **14** | **slot0 confidence** | **= clamp(1 − age/stale_horizon, 0, 1) — the "how fresh is my vision" scalar** |
| 15 | slot0 visible_area | |
| 16:19 | slot1 rel_pos (next gate) | |
| **19** | **slot1 confidence** | staleness for the next gate |
| 20 | slot1 visible_area | |

- **Staleness is already an observation** (`obs[14]`, `obs[19]`). The policy already knows how stale its vision is and can weight it against IMU. Keep using it.
- **IMU-interpolation between vision updates already exists**: between accepted fixes the held `rel_pos` is **ego-propagated** — rotate by `−body_rate·dt`, translate by `−v_body·dt` (`ego_estimator.py:447–459`), identical in training and deploy.
- **Blackout cliff matched**: `det_proxy = (age < det_hold_s)`; the slot masks to zeros past the horizon (champion trained coast-OFF → zeros at the crossing).
- Config the last deploy flights used: `stale_horizon 0.6`, `det_hold 0.3` (training defaults are 0.5 / 0.2 — **pick one and match both sides**).

➜ "the policy interpolates with IMU between vision updates, weighted by staleness" **is already the contract.** The new training just has to keep feeding it the same way.

## 2. The rate that actually matters (and the gotcha)
Two different rates — don't conflate them:
- **Video / detector-attempt:** ~30 Hz (one detector attempt per frame). *(Recorded 30.0 Hz over 424 flights — not 28.8; reconcile before baking a number in.)*
- **Valid gate detections:** **~7–15 Hz** — the detector only fires when the gate is geometrically detectable, and it **loses the gate close-in** (loss-onset ~4.7 m median). So the effective vision refresh the policy sees is 7–15 Hz and **drops to zero on final approach** — exactly the blind-approach yaw-hunt you localized.

Training currently recomputes geometric detectability every 33 ms (30 Hz tick); the `det_hold`/staleness machinery bridges the 30 Hz-tick vs 7–15 Hz-arrival gap, but it's an approximation. **For the new run:** model the detector as *attempting* at the deploy frame rate (~30 Hz) with the SAME geometric detectability drop-out (valid detections at ~7–15 Hz, dark near the gate), and let the existing confidence-ramp + ego-propagation carry the gaps. **Do NOT train with vision available every control tick** — trained-fresh / deployed-stale is the exact train/deploy gap that sank our deploy async (§4.3).

## 3. Architecture: single loop, no threads
```
each control tick (40–60 Hz):
    integrate IMU                         # every tick, cheap
    if new frame arrived (frame_id changed):
        run detection                     # ≤30 Hz, valid ~7–15 Hz
    else:
        hold last rel_pos, ego-propagate it, decay confidence
```
Deterministic, no race, one GPU consumer. It's already how the **sync** deploy stack behaves (`detect_cached` fires once per unique `frame_id`; the builder ego-propagates between; `--video-dedup-fastpath` drops the sim's duplicate-frame flood). Duplicate frames are a non-issue — a repeated frame is a cache hit, never a re-inference. Max *new* vision = the stream rate, as expected.

## 4. Async post-mortem — why the threaded version bit us (don't repeat it)
`--video-async-detect` ran the detector on a worker thread and had the loop consume the "already-detected" frame. It failed three ways:
1. **Implementation:** `detect_cached` is a **single-entry** cache (last `frame_id` only). The worker raced ahead and overwrote the slot, so the loop's frame usually **missed** the cache and re-ran inference anyway — the decouple never decoupled (`infer_ms` stayed ~12 ms *in the loop*).
2. **Contention:** it added a second concurrent GPU consumer on the one shared card → worsened the real tail (a ~56 ms GPU sync in `nav`, **not** the `detect()` call) → p99 88→95 ms. (+ Python GIL: the worker isn't truly parallel anyway.)
3. **OOD:** the deployed policy was trained *synchronous*; injected variable staleness → out-of-distribution → **lower gates on every model**.

And the loop tail it chased **doesn't limit gates anyway** (corr(work_p99, gates) ≈ 0). Net: threaded async is the wrong tool; the single-loop multi-rate in §3 avoids all three failure modes.

## 5. Control rate: 40 vs 60
- **40 Hz** works — age is observed, so irregular staleness is in-distribution. But 40/30 = 1.33 → fresh-frame ticks fall in an irregular beat and vision age swings 0–33 ms tick-to-tick.
- **60 Hz = exactly 2× the 30 Hz stream** → every-other-tick fresh, staleness always 0 or 1 frame, zero phase drift. Cleaner/deterministic. Since you said "40 or more," worth considering.

## 6. Checklist for the new training run
- [ ] Reuse the existing obs contract — confidence=staleness ramp + ego-propagation + `det_hold` masking (`ego_obs.py` / `ego_estimator.py:447–459`, pinned by `test_ego_deploy_obs.py`).
- [ ] Train vision at the DEPLOY cadence: ~30 Hz attempts, geometric detectability drop-out (→ ~7–15 Hz valid, **dark near the gate**) — NOT fresh-every-tick.
- [ ] Single control loop, detect-on-new-frame, IMU every tick. No worker thread.
- [ ] Confirm the real stream rate (**30.0 recorded, not 28.8**); pick control = 40 (age-observed) or 60 (clean 2×).
- [ ] Match `stale_horizon` / `det_hold` between train and deploy (deploy flew 0.6 / 0.3).
