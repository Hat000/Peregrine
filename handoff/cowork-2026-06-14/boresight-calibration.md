# MEMO — Peregrine: Monocular Boresight Calibration + Online Attitude/IMU-Bias Estimation

**To:** Fengyou · **Re:** Removing the ε_vert ≈ 0.56° (~3.1 px) vertical boresight bias · feeds path **P1**
**Effort:** MAX · deep-research · **Date:** 2026-06-14
**Setup of record:** forward camera, fixed 20° up-tilt, pinhole 640×360, fx=fy=320, no distortion; gate-relative PnP fix fused with IMU in a Kalman filter; usable vision fixes on ~7% of frames; attitude-bias budget ≤0.6°.

---

## TL;DR — recommended approach (escape hatch: the literature converges, so this leads)

**Fix it offline as a static extrinsic correction; do not try to estimate it online.** For your specific error — a *constant, known-sign, sub-degree pitch* offset traced to a sim-render-vs-PnP-decode convention mismatch, with the ability to stage a static head-on shot at a known gate — the offline path is unambiguously better than carrying it as a filter state. Concretely:

1. **Diagnose & confirm causation (decisive, ~1 hr):** capture the known gate **statically at ≥3 ranges** (e.g. 5/10/22/30 m), camera **leveled by the accelerometer gravity vector**, and plot apparent vertical offset vs range. A **constant ~3.1 px pixel residual** (metric offset growing *linearly* with range) confirms an **angular pitch boresight**; a *constant metric* offset (pixels shrinking ∝1/R) would instead indicate a **map/registration error**. This single test separates the two root causes in pitfall #3.
2. **Estimate ε** by reprojection-residual minimization with camera pitch as the only free extrinsic DOF (or read it off a horizon/vanishing point: pitch ≈ Δv/fy). Sub-degree (~0.1°) is trivial here.
3. **Remove it structurally:** because the cause is a decode-vs-render convention mismatch, **align the PnP decoder's extrinsic rotation to the renderer's** — bake **−ε** into the camera↔body/IMU extrinsic. One-time constant, zero runtime cost; this drives the ~0.2 m gate-4 term to ≈0, recovering the budget.
4. **Do NOT also add a residual pitch attitude-bias state** for the same DOF — that double-counts (see §3). Keep your existing gyro/accel-bias states (different physics). Optionally carry a *tightly-priored* camera-pitch extrinsic-error state as a **watchdog only**, to flag crash-induced shifts — not as a competing corrector.

Why not "just let the online filter learn it": with vision on only ~7% of frames and no guaranteed rotational excitation, a residual attitude/extrinsic-bias state converges slowly, is confounded with the IMU biases and with the very gate offset it should correct, and risks filter over-confidence. The racing stacks you cited treat the camera–IMU/body–camera extrinsic as an **offline-calibrated constant** in the published racing-correction filter (Agilicious uses **Kalibr**; AlphaPilot fixes body↔camera as a constant and its EKF estimates only a VIO-misalignment transform + gate map) — **none estimate the camera–IMU rotation online in the racing loop** (any online extrinsic work is delegated to the VIO front-end, e.g. SVO/RealSense). That is the convergent best practice for your case.

---

## 1. Static boresight / extrinsic calibration — methods menu

| Method | What it estimates | Data / excitation required | Observability condition | Best for / accuracy | Tradeoff for Peregrine |
|---|---|---|---|---|---|
| **A. Camera–IMU extrinsic (Kalibr)** | Full T_cam_imu (rotation **+** translation), camera–IMU time offset, optionally IMU intrinsics; continuous-time B-spline batch ML | ~1–2 min hand-held motion **exciting all 3 rotation axes + translational acceleration** in front of an Aprilgrid; Allan-variance IMU noise model first | Rotation extrinsic observable only with **rotation about ≥2 non-parallel axes** (Mirzaei & Roumeliotis 2008); gravity anchors roll/pitch | Standing camera–IMU calibration. Rotation accuracy **sub-degree, ~0.1–0.5° practical**, degrading under FPV vibration | Gives the *whole* rotation, so it *would* absorb ε — but its own noise (~0.1–0.5°) is comparable to the 0.56° you're chasing, and it **cannot tell you whether the offset is a render/decode convention bug or a physical mount error**. Use as the baseline extrinsic, not the isolating fix. |
| **B. Single-known-target pitch cal (PnP / vanishing-point)** | Camera **pitch** (one extrinsic DOF) from a known-geometry gate | Static captures of the known gate **at multiple ranges**, camera **leveled via gravity**; centered + off-center shots | A **single head-on view at a single range cannot** separate pitch from a vertical target/camera translation or a target-position error (planar-pose flip / bas-relief ambiguity). **Multiple ranges break it.** | Directly targets the suspect DOF; **sub-degree easily**. Vanishing-point/horizon read is roll/translation-insensitive (cleanest pitch isolation) | **Cheapest, most on-target.** Quality limited by how well you know the gate and how level the mount is. **This is your primary diagnostic + estimator.** |
| **C. Hand-eye / boresight-from-mapping geometry** | Fixed sensor↔body rotation, decoupled from lever-arm and datum/registration offset | **Reciprocal (opposing) and perpendicular** look directions at **multiple ranges/altitudes** | Same as hand-eye AX=XB: needs **≥2 non-parallel rotation/look axes**; pure translation leaves rotation/along-axis terms unobservable | Gold standard for **separating a rotation boresight from a registration offset**; airborne self-cal reaches **10⁻³° (arc-seconds)** | Requires staging diverse geometry, but the **reciprocal-heading sign-flip** is the rigorous answer to "is this a boresight or a map offset?" |

**Which observations best isolate a pure pitch boresight from a map/registration error?** Two complementary signatures, both of which your static rig can capture:

- **Range-scaling (primary, verified arithmetic below).** A pure *angular* bias produces a **range-independent pixel residual** and a **metric offset that grows linearly with range**; a fixed *metric* map/registration offset produces a **range-independent metric offset** and a **pixel residual that shrinks as fy·d/R**. Plot apparent vertical offset vs range and read the slope:

  | Range | Angular bias 0.56° → metric / pixel | Fixed 0.215 m map offset → metric / pixel |
  |---:|---|---|
  | 5 m | 0.049 m / **3.13 px** (const) | 0.215 m / 13.76 px |
  | 10 m | 0.098 m / **3.13 px** | 0.215 m / 6.88 px |
  | 22 m | 0.215 m / **3.13 px** | 0.215 m / 3.13 px |
  | 30 m | 0.293 m / **3.13 px** | 0.215 m / 2.29 px |
  | 50 m | 0.489 m / **3.13 px** | 0.215 m / 1.38 px |

  *(The two are indistinguishable only at the single range 22 m where they're tuned to coincide — exactly why a single fixed-range shot is ambiguous and you must vary range. Verified: atan(3.1/320)=0.555°; 320·tan(0.56°)=3.13 px; 22·tan(0.56°)=0.215 m. External cross-check: 0.1°→17.5 cm at 100 m.)*

- **Reciprocal-heading sign-flip (confirmatory).** A pitch boresight produces an along-track offset that **flips sign when you reverse the viewing direction** (so it doubles in the opposing-pass difference and scales with range/height), whereas a map/datum offset and a vertical lever-arm are **heading- and (for z) range-invariant and cancel** in the difference. This is the airborne-mapping decoupling principle (GeoCue/Skaloud) and the hand-eye "≥2 non-parallel axes" requirement in another guise.

**Practical staging notes:** level the camera using the 3-axis accelerometer gravity vector (gives absolute roll/pitch at rest), and **explicitly model any target roll/pitch** — aligning a physical gate to gravity better than ~0.1° is hard, and target tilt aliases directly into estimated camera pitch. Use `cv::SOLVEPNP_IPPE_SQUARE`/`solvePnPGeneric` for the coplanar gate so you get **both** planar-pose solutions and can inspect the reprojection-error ratio rather than silently taking the wrong flip.

---

## 2. Online attitude / IMU-bias estimation as a filter state

**Formulations (error-state / sliding-window):**

- **ESKF (Solà 2017)** — canonical error-state = {δθ, δv, δp, δb_g, δb_a}, optionally augmented with gravity, camera–IMU extrinsic, and time offset. This is the natural home for your gyro/accel-bias states.
- **MSCKF / OpenVINS (Geneva et al.)** — sliding-window EKF (FEJ) that can **online-calibrate camera intrinsics, camera–IMU extrinsic, and time offset**. In sim these converge "rapidly from poor guesses" (~10–60 s) under good excitation; with a bad initial guess and calibration *off*, NEES/ATE blow up — i.e. you must either calibrate well offline or excite well online.
- **VINS-Mono (Qin, Li, Shen 2018)** — optimization-based; performs **online spatial + temporal** camera–IMU calibration and explicitly only needs a *rough* initial extrinsic, refining it online.
- **Racing stacks (the most relevant precedent):** AlphaPilot's EKF estimates a **VIO-frame misalignment transform + a global gate map** (gate positions/headings) via gate-corner reprojection error, with body↔camera a **known constant**; Swift uses a **constant-velocity Kalman filter** on drone pose corrected by IPPE gate fixes, with the sim-to-real perception offset handled by an **offline-learned residual (GP) observation model**; Agilicious calibrates camera–IMU with **Kalibr offline**. **None estimate the camera–IMU rotation online in the racing filter.**

**When are these biases/attitude offsets observable?**

- Under sufficient excitation a monocular VINS has **exactly 4 unobservable directions: 3D global position + global yaw** (the nullspace grows under the degenerate motions below). Critically, **roll and pitch ARE observable** because the accelerometer senses gravity — so a residual *pitch* boresight is, in principle, observable (good news for a watchdog state).
- **But** the camera–IMU **rotation extrinsic** becomes observable only under **rotation about ≥2 non-parallel axes**; **accelerometer bias** needs genuine rotational excitation / **non-constant acceleration** (it's conflated with gravity under constant acceleration). Gyro bias + rotation-extrinsic converge **fastest**; accel bias + translation-extrinsic **slowest**.
- **Degenerate motions** that kill observability of biases/extrinsics/scale: **constant acceleration** (accel bias/scale/gravity conflated), **no rotation** (rotation extrinsic unobservable), **planar/hover**, **constant velocity** ("VINS on Wheels"). Over-parameterizing the calibration also stalls rotation convergence.

**Convergence speed:** with good multi-axis excitation, order **~10–25 s** to converge extrinsics/biases (OpenVINS sim ~10–60 s; EqVIO/online-init ~25 s, sharpest uncertainty drop ~10 s). These are order-of-magnitude, dataset-specific, and **assume dense visual updates**.

**Failure modes when fixes are sparse (your ~7%):**

- Updates are rare and inter-update excitation is uncontrolled, so a residual pitch/extrinsic-bias state **converges slowly or stalls**, and is **confounded** with the gyro/accel biases and with the gate-relative offset it's meant to correct.
- Between fixes the filter **coasts open-loop on IMU**: gyro bias drift integrates into a **growing attitude error (ramp)**, accel bias into position drift. A *constant* boresight is not what these states should absorb — but a poorly-anchored extrinsic/bias state **will** absorb part of it, masking the true cause.
- Standard EKF linearization **gains spurious information along the unobservable directions** → **over-confidence / inconsistency / divergence**; FEJ, OC-VINS, or robocentric (R-VIO) formulations are the standard mitigations and should be in place regardless.

**Bottom line for §2:** online estimation is the right tool for the *time-varying* IMU biases (keep them), and roll/pitch are observable in principle — but at 7% fix density it is the **wrong tool for a constant, already-characterized boresight**. Use it to monitor, not to correct.

---

## 3. Pitfalls — telling the three errors apart, and double-counting

**Distinguishing signatures.** The constant extrinsic boresight, a per-axis IMU bias, and a map-registration offset have *different* fingerprints — which is what lets your static rig isolate them:

| Error source | Time signature | Range / look-direction signature | Survives a static, motionless head-on shot? | Lives in |
|---|---|---|---|---|
| **Extrinsic pitch boresight** (render-vs-decode) | **Constant** (no drift) | Angular: **pixel residual constant w/ range, metric ∝ range**; **flips sign** with viewing direction | **Yes** | camera↔body/IMU rotation |
| **Gyro bias** (per-axis) | **Ramp** between fixes (sawtooth after each correction) | Not range-structured; integrates over time | **No** (manifests only through motion/integration) | IMU state |
| **Accel bias** | Slow; tilts gravity-referenced roll/pitch, drifts velocity/position | Conflated with gravity under constant acceleration | **No** | IMU state |
| **Map-registration / gate-map offset** | **Constant** | Metric offset **constant w/ range; pixels shrink ∝1/R**; **heading-invariant** | **Yes** | world / map frame |

Four clean discriminators fall out: **(1) static vs moving** — only the boresight and the map offset survive a motionless head-on shot; IMU biases need motion. **(2) range-scaling** — boresight metric grows with range, map offset is metric-constant. **(3) time** — IMU biases drift/ramp; boresight and map offset are constant. **(4) reciprocal heading** — boresight flips sign; map offset doesn't. Your multi-range, gravity-leveled, reciprocal-look capture set exercises all four at once.

**Does static-cal + an online residual-bias state double-count? Yes — if both are free to explain the same DOF.** There is a genuine **gauge ambiguity**: a constant pitch offset can be attributed to the camera↔IMU extrinsic *or* to a persistent attitude-bias state, and with sparse fixes the data cannot arbitrate. If you bake **−ε** into the extrinsic **and** leave an unanchored residual pitch-bias state free, the state will drift to re-absorb part of ε, and the two corrections partially cancel/compete — net result is an unstable, under-determined estimate.

**Resolution — pick one home for the constant:**

- **Recommended:** remove ε **offline in the extrinsic**; do **not** carry a free residual *pitch* attitude-bias state for that DOF. Keep the normal gyro/accel-bias states (they model different, time-varying physics and do not double-count).
- **If you want an online check:** carry the camera-pitch extrinsic-error with a **tight prior centered at zero** (post-correction). It then acts as a **monitor/watchdog** for crash- or thermal-induced shifts, not a competing corrector — and converges (if at all) toward zero. Never simultaneously (a) bake in −ε **and** (b) let an unanchored bias state also chase ε.
- This mirrors the racing precedent: the extrinsic is a **calibrated constant**, and residual perception offset is handled as **map states** (AlphaPilot) or an **offline-learned observation residual** (Swift), never as an unconstrained online attitude-bias fighting the calibration.

---

## Recommended approach for Peregrine (consolidated)

1. **Confirm it's an angle, not a map offset** — multi-range static known-gate capture, gravity-leveled; the range-scaling plot above is decisive (and is also your pitfall-#3 separator).
2. **Estimate ε offline** — pitch-only reprojection minimization (use IPPE for the coplanar gate) and/or horizon/VP cross-check.
3. **Bake −ε into the PnP decoder's extrinsic** to match the renderer convention; constant, zero-cost, closes the gate-4 term.
4. **Leave the ESKF gyro/accel-bias states as-is; do not add a residual pitch-bias state** for the same DOF.
5. **Optional watchdog:** tightly-priored camera-pitch extrinsic-error state, monitor-only; **re-run the static cal after crashes**.
6. **Keep a standing Kalibr camera–IMU calibration** as the baseline extrinsic the −ε correction sits on; ensure FEJ/OC-VINS-style consistency handling given your sparse fixes.

**Don't:** rely on the 7%-density online filter to *discover and remove* a constant boresight (slow/unobservable/confounded), or run offline-cal and an unanchored online bias-state on the same DOF (double-counts).

---

## Key references

**Camera–IMU extrinsic & observability**
- Furgale, Rehder, Siegwart, *Unified Temporal and Spatial Calibration for Multi-Sensor Systems*, IROS 2013 — Kalibr's continuous-time batch formulation. https://furgalep.github.io/bib/furgale_iros13.pdf
- Mirzaei & Roumeliotis, *A Kalman Filter-Based Algorithm for IMU–Camera Calibration: Observability Analysis & Performance*, IEEE T-RO 24(5), 2008 — ≥2-axis rotation observability; mm/sub-degree accuracy. https://doi.org/10.1109/TRO.2008.2004486
- Kelly & Sukhatme, *Visual-Inertial Sensor Fusion: Localization, Mapping and Sensor-to-Sensor Self-Calibration*, IJRR 2011. https://journals.sagepub.com/doi/abs/10.1177/0278364910382802
- Kalibr wiki — camera-imu calibration & IMU noise model (incl. "10× inflation for low-cost MEMS"). https://github.com/ethz-asl/kalibr/wiki/camera-imu-calibration · https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model

**Boresight / hand-eye decoupling**
- Skaloud & Lichti, *Rigorous approach to bore-sight self-calibration in airborne laser scanning*, ISPRS J. P&RS 61, 2006. https://www.isprs.org/proceedings/xxxvi/5-c55/papers/skaloud_jan.pdf
- GeoCue/Terrasolid, *Recognizing Misalignment Patterns for Airborne LiDAR Calibration*, 2015 — pitch = along-track offset on opposing lines. https://support.geocue.com/wp-content/uploads/2015/08/AirborneCalibration.pdf
- Tsai & Lenz, *A new technique for fully autonomous 3D robotics hand/eye calibration*, IEEE T-RA 5(3), 1989. https://kmlee.gatech.edu/me6406/handeye.pdf
- Horaud & Dornaika, *Hand-Eye Calibration*, IJRR 1995 (arXiv:2311.12655) — ≥2 non-parallel axes; pure translation leaves rotation unobservable. https://arxiv.org/pdf/2311.12655

**Single-target pitch / PnP ambiguity**
- Collins & Bartoli, *Infinitesimal Plane-Based Pose Estimation (IPPE)*, IJCV 109(3), 2014 — planar-pose two-solution flip & depth→affine degeneracy. https://encov.ip.uca.fr/publications/pubfiles/2014_Collins_etal_IJCV_plane.pdf · https://github.com/tobycollins/IPPE
- OpenCV PnP docs (SOLVEPNP_IPPE_SQUARE, solvePnPGeneric). https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html
- Theers & Singh, *Extrinsic Camera Calibration* (vanishing-point pitch; insensitive to roll/translation). https://thomasfermi.github.io/Algorithms-for-Automated-Driving/CameraCalibration/VanishingPointCameraCalibration.html

**Online estimation & VIO observability**
- Solà, *Quaternion kinematics for the error-state Kalman filter*, arXiv:1711.02508, 2017. https://arxiv.org/abs/1711.02508
- Qin, Li, Shen, *VINS-Mono*, IEEE T-RO 34(4), 2018 — online spatial+temporal calibration. https://arxiv.org/pdf/1708.03852
- Geneva, Eckenhoff, Lee, Yang, Huang, *OpenVINS*, ICRA 2020. https://udel.edu/~ghuang/iros19-vins-workshop/papers/06.pdf
- Yang, Geneva, Zuo, Huang, *Online Self-Calibration for VINS: Models, Analysis and Degeneracy*, IEEE T-RO (arXiv:2201.09170), 2022 — 4 unobservable directions; degenerate motions. https://arxiv.org/pdf/2201.09170
- Huang, *Visual-Inertial Navigation: A Concise Review*, ICRA 2019 (arXiv:1906.02650) — unobservable subspace, FEJ/OC-VINS. https://arxiv.org/pdf/1906.02650
- Wu, Ahmed, Georgiou, Roumeliotis, *VINS on Wheels*, ICRA 2017 — degenerate-motion unobservability. https://mars.cs.umn.edu/papers/KejianWu_VINSonWheels.pdf

**Drone-racing state estimation (precedent)**
- Kaufmann et al., *Beauty and the Beast*, ICRA 2019 — CNN gate pose+covariance fused via EKF. https://arxiv.org/abs/1810.06224
- Foehn et al., *AlphaPilot: Autonomous Drone Racing*, Auton. Robots 2021 — EKF over VIO-misalignment + gate map; constant body↔camera; sparse-gate bridging. https://rpg.ifi.uzh.ch/docs/AURO21_Foehn.pdf
- Kaufmann et al., *Champion-level drone racing (Swift)*, Nature 620, 2023 — IPPE + constant-velocity KF; offline-learned residual observation model. https://www.nature.com/articles/s41586-023-06419-4
- Foehn et al., *Agilicious*, Science Robotics 7(67), 2022 — Kalibr camera–IMU calibration. https://www.science.org/doi/10.1126/scirobotics.abl6259

*Uncertainty flags: Kalibr publishes no single headline rotation-accuracy number — "sub-degree" is well-supported, ~0.1° only on the best axis/clean data, degrading under FPV vibration. Convergence times are order-of-magnitude and dataset-specific. The racing papers do not frame a constant gate offset as an explicit "bias state"; that mapping is inferred from their methods.*

---

## MEMORY-DELTA
- ε_vert ≈ 0.56° ≡ 3.1 px at fy=320 ≡ 0.215 m at 22 m (verified: atan(3.1/320)=0.555°; 22·tan0.56°=0.215 m).
- DECISION: remove ε **offline** by correcting the camera↔IMU extrinsic (decode→render convention); do **not** estimate it online at 7% fix density.
- DIAGNOSTIC: multi-range static gate shot, gravity-leveled → angular bias = **constant ~3.1 px / metric ∝ range**; map offset = **constant metric / pixels ∝1/R**. Single fixed range is ambiguous.
- Reciprocal-heading sign-flip separates boresight (flips) from map/datum offset (invariant); needs ≥2 non-parallel look axes (= hand-eye condition).
- VIO unobservable dirs = global position + yaw; roll/pitch observable via gravity; accel-bias/rotation-extrinsic need ≥2-axis rotation; degenerate under const-vel/accel, no-rotation, planar/hover.
- DOUBLE-COUNT RISK: offline-cal + free online residual-pitch-bias state on the same DOF = gauge ambiguity. Pick one home for the constant; keep gyro/accel-bias states (different physics).
- Optional watchdog: tightly-priored camera-pitch extrinsic-error state, monitor-only; re-run static cal after crashes.
- Precedent: Agilicious/AlphaPilot/Swift calibrate cam–IMU extrinsics offline (Kalibr) as a known constant; residual offset → map states or offline-learned residual model, never an unconstrained online attitude-bias.
- Keep FEJ/OC-VINS consistency handling given sparse fixes (standard EKF over-confidence on unobservable dirs).
- Kalibr = baseline extrinsic, not the isolating fix (its ~0.1–0.5° noise ≈ the bias being chased).
