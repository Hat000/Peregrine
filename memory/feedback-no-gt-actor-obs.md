---
name: feedback-no-gt-actor-obs
description: Fengyou hard directive (2026-07-10) — NEVER train the actor on ground truth; actor obs must be estimator-faithful end-to-end (run the deploy estimators on simulated sensors in training)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c864e5bb-f53f-4b54-b78a-a2216bb2c906
---

**Fengyou (2026-07-10, verbatim intent):** "WHY ARE WE GIVING OUR TRAINING ANY GT AT ALL — we should never train with GT. However we estimate which way down is in the real simulator, we train with that."

Context: the ego stack estimator-emulated GATE perception but left attitude (obs[3:5]), body rates (obs[5:8]) and velocity (obs[0:3]) on perfect sim truth, while the wire feeds ESKF-leveler attitude + KF velocity. The A2/20 Hz systematic crash suspect is exactly this train/wire gap (leveler corruption under 3.7 g + 9 rad/s — thrust-dominated accel loses the gravity reference). See [[ego-deploy-contract-2026-07-09]] §A2.

**The rule:**
- ACTOR observations: NO ground truth, ever. Every channel comes from the same estimation pipeline that flies — run the DEPLOY estimator algorithms (ESKF leveler, velocity KF) batched inside training on SIMULATED SENSORS (IMU specific-force + gyro with noise/bias).
- GT is permitted ONLY where it never deploys: the privileged CRITIC (asym-critic doctrine stands), the REWARD, and TERMINATIONS.

**Why algorithm-parity beats an error model:** there is NO GT on the VQ2 wire (banked: "NO GT on the wire ever"), so a hand-built "leveler error model" could never be calibrated — invented constants again. Running the same estimator needs no wire GT; the policy experiences the estimator's true failure modes by construction. The only free parameters (IMU noise/bias) are measurable from REAL wire data already owned: the A1 pad recordings (armed, static, full race clock of HIGHRES_IMU) = sensor characterization dataset.

**How to apply:** estimator-faithful-obs package = own design/build workflow on chaum after the despin package lands (same _percept stage family; parity tests element-exact vs deploy code, like the ego adapter audit). Deploy adapter needs ZERO changes (it already feeds estimated state — training was the liar; this is parity-RESTORING, obs layout/semantics unchanged from deploy's view). Acceptance test: training leveler reproduces the attitude-error magnitude measured by the 20 Hz flight leveler-consistency test. GT-attitude runs remain legal ONLY as disposable diagnostics whose ckpts never fly. Related: [[feedback-no-spin-hard-requirement]].
