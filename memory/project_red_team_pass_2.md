---
name: project-red-team-pass-2
description: "2026-05-30 external red-team pass on the autonomy stack — what was real (fixed), what was deferred-and-noted, what was overstated/wrong (don't re-litigate)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 39507722-36f2-4148-b379-08315f70e12b
---

A 2nd third-party red-team report (received 2026-05-29/30, the eve of the official sim drop) was triaged against the ACTUAL code (not memory). Verdict + actions:

**FIXED — Tier A correctness/robustness (branch `red-team-tier-a`, 154 tests green = 145 + 9 new; NOT yet merged to main):**
- `gate_pose._estimate_p3p`: return `None` when no P3P solution is cheirality-valid (was `valid = in_front or cands` → fed behind-camera poses to the KF at gate transit). The single sharpest catch.
- `gate_pose._estimate_ippe`: additive sub-pixel floor on the ambiguity ratio (`AMBIGUITY_EPS_PX=0.1`). The old `else float("inf")` sentinel bypassed the temporal prior exactly when a frontal/low-noise gate was MOST ambiguous (both IPPE reproj errors ~0). Mostly bit synthetic / estimator-oracle paths; rarely fires in noisy real flight.
- `gate_pose._reproj_rms`: guard the perspective divide (Z≤0 → `+inf`) — defence-in-depth (the None fix already prevents selecting such a pose).
- `controller._accel_to_attitude`: two geometric-law singularity guards — (a) antipode tilt-clamp bypass (straight-down thrust no longer passes an inverted command through; recovers upright via an arbitrary horizontal tilt axis), (b) `b3 ∥ x_c` zero-cross → heading left-normal fallback (no NaN quaternion). Both were LATENT (the 45° tilt clamp guards them today) but cheap.
- `state_estimator.update`: `np.linalg.inv(S)` → `np.linalg.solve` (the Joseph-form update was already present, so NO blow-up — this is numerical hygiene, not the claimed catastrophe).
- `jpeg_receiver.frames`: `select.select()` instead of a `time.sleep(0.001)` busy-poll — the sleep hit Windows' ~15 ms timer granularity, adding frame latency/jitter on the Azure VM. Real perception-latency win.

**FOLDED — Tier B gate-transit soft-coast (`localization.apply_gate_pose_update`):** a 3-corner (`n_corners < 4`) fix gets its covariance inflated (`P3P_FIX_COV_INFLATION=9.0`) so it nudges rather than snaps the KF at transit. Pairs with the P3P-None fix.

**DEFERRED but NOTED (build on first-contact EVIDENCE, not the report's say-so) — recorded in the `state_estimator` module docstring:**
- ESKF bias-STATE (estimate attitude bias vs only inflating Q). Trigger: first-contact data shows the sim's given attitude is biased under high-G. The report is right that Q-inflation masks but does not remove a *systematic* drift — we already concede this; the call is not to rebuild the hardest subsystem on an unmeasured hypothesis the night before the sim drops. Add an attitude-bias observability check (given-attitude vs gravity/vision residual) to the first-contact toolkit so DATA decides.
- Delayed-vision ring buffer (rewind to a fix's capture sim-time, re-propagate buffered IMU). Trigger: VQ2 speed (~0.6–1 m offset at 20 m/s; ~cm at VQ1 ⇒ not a VQ1 blocker). Prereq: reconcile the VIDEO capture clock with HIGHRES_IMU (TIMESYNC) — note the report's "tag with recv_monotonic_ns" is wrong; you rewind to CAPTURE time, not arrival.
- Hard coast (drop vision entirely within X m of a gate) + innovation / Mahalanobis gating of wrong-gate "teleport" fixes (master-plan NEG-3) — belong in the not-yet-built navigator loop + mapper.
- Loop-gating discipline (Finding 3 was INVALID as a TypeError — `DroneState` arrays default to `_vec(3)` zeros, no crash — but the underlying "don't step the KF before the first real ATTITUDE/HIGHRES_IMU, derive dt from sim stamps + clamp it" is sound; honour it when wiring `Mission.run`'s navigator).

**INSTRUMENTED to settle the deferred decisions with DATA (2026-05-30, same branch `red-team-tier-a`, +5 tests = 159):** filled the first-contact gaps so tomorrow's session decides ESKF/ring-buffer empirically, not by the report's say-so — `scripts/clock_probe.py` (NEW, read-only: video↔IMU `sim_time` offset mean/jitter → delayed-vision go/no-go; + datagram-size-vs-MTU + frames-lost-to-missing-chunks → the MTU verdict) · attitude-bias check folded into `session_lifecycle.py` via new `firstcontact.{gravity_tilt, attitude_gravity_residual, sample_attitude_bias}` (given-vs-gravity tilt at rest → ESKF go/no-go) · `JpegUdpReceiver.metrics` (`ReceiverMetrics`) · `docs/first_contact.md` (ordered runbook + observation→action decision table). The existing toolkit already covered R1/R2/arming/clock-monotonicity/easy-mode/plant-sysID; these are the deltas.

**OVERSTATED / INVALID — don't re-litigate:**
- `mavlink_client` `replace()` "race condition": INVALID. `dataclasses.replace` + a single atomic ref-rebind under the GIL yields a consistent immutable snapshot; cross-field time-skew is by-design (the clock TODO), not a torn read.
- "KF will blow up from `inv()`": overstated — Joseph form already keeps P symmetric PD.
- "Dropped-IMU silently warps the state": the KF takes `dt` as a param + exact constant-accel discretisation; just derive `dt` from sim stamps + clamp in the (unbuilt) loop.
- IPPE corner-order CW/CCW fragility: corner order is locked + round-trip tested; the real residual (detector order on REAL frames + frames.py sign trap) is covered by `projection_check` / first-contact validation.
- MAVLink `time_boot_ms` desync: PX4/ArduPilot ignore it on setpoint streams; trivially send 0; verify at first contact.
- UDP MTU fragmentation: contradicted by the spec's app-level chunked protocol (§4.6 — sender keeps chunks < MTU); just measure chunk sizes + add packet-loss metrics at first contact.
- "Easy-mode position control WILL be sluggish" / "rewrite I/O in C/separate process": speculative / over-engineering at 30 Hz — keep the time-boxed `control_mode_probe` test; don't pre-invest.
- Offline obstacle triangulation "delusion": re-derives our OWN logged caveats (bbox-center slides, biased relative pose smears) — it's contingent + unbuilt anyway.

Meta: the report's cited line numbers drifted from the real files (it was reading an approximated copy), so every claim was verified by CONTENT. Hit rate ≈ same as the 2026-05-29 pass: a few real, several already-known, a few wrong — exactly why adversarial input gets verified, not accepted wholesale. Related: [[project-master-plan]], [[feedback-walking-skeleton-no-vq1-crutches]].
