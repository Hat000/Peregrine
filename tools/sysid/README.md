# tools/sysid — deterministic-plant distillation (SINDy + differentiable-physics ID)

Recover a deterministic flight sim's **known plant parameters** (mass, drag, thrust/torque
gains, rate-loop gains, inertia) from rolled-out `(state, action) -> state_dot` data, to
machine precision. Validated against the **diffaero** `QuadrotorModel` twin (whose parameters
we know exactly), so the method is trusted before we point it at **VQ2's** plant the moment
its sim drops.

Two complementary recoverers:

* `recover_sindy.py` — STLSQ on a **physics-informed candidate library** (one column per
  physical term). Linear-in-parameters, so STLSQ solves it exactly in a noise-free sim.
* `recover_diffid.py` — **differentiable-physics ID**: autograd through a torch mirror of the
  plant, jointly fitting all params. The fallback for terms that resist a clean symbolic
  library, and the tool that exposes parameter **identifiability** (degenerate directions).

`gen_data.py` produces the identification dataset by rolling diffaero out under rich
excitation and logging the EXACT analytic `model.dynamics(X, U)` derivative (no
finite-difference error — the deterministic-sim target noise floor is float32 eps).

## Frame-convention footgun (RESOLVED — do not regress)

diffaero's `RateController` computes the rate error against `R_i2b @ w`, i.e. it rotates the
state angular rate `w` by world->body before differencing with the commanded rate, **while**
the rigid-body Euler update uses the raw state `w`. Any recoverer MUST mirror this transform:
the rate-loop residual is `w_des - R_i2b @ w`, NOT `w_des - w`. Regressing against raw `w`
yields garbage rate gains (K ~= -0.66, R^2 ~= 0.02) on the x/y axes. With the transform
applied, K_x,K_y,K_z recover to ~1e-9 relative error. (z works either way: its Coriolis
lambda is 0 and it is unaffected by the in-plane attitude coupling.)

## What is and isn't identifiable (diffaero, normal-flight envelope)

* **g, drag D/m, rate gains K**: recovered to ~1e-7..1e-9 relative error.
* **m vs D individually**: NOT separable — thrust accel is `R[:,:,2] * action * g` (mass
  cancels) and drag enters only as `D/m`. Only the ratio `D/m` is observable. Anchor `m`
  (e.g. m=1 from the spec) and `D_xy, D_z` then recover exactly.
* **inertia J**: NOT identifiable in normal flight. The controller pre-multiplies the command
  by `J` and the plant divides by `J` (`J^-1 J K err = K err`), and the controller's Coriolis
  *compensation* `act x J act` cancels the plant's rigid `w x J w` — every J-dependent term
  drops out while the compensation norm-clamp (|cross|>100) is inactive, which it is for any
  realistic rate. Rate-loss is flat across J-scale 0.5x..5x. J only becomes observable at
  |w| >~ 70 rad/s (clamp active) — outside any flight envelope.

These are exact structural degeneracies of the sim's control-allocation, NOT method error.
For VQ2: expect the same pattern wherever VQ2 folds J / mass into its controller. Anchor the
unobservable params from the spec; recover the rest to machine precision.

## Run (on Adroit, env `diffaero`)

```bash
source /etc/profile.d/modules.sh && module load anaconda3/2024.10
eval "$(conda shell.bash hook)" && conda activate diffaero
export PYTHONPATH=/scratch/network/fl3689:$PYTHONPATH   # for `import diffaero...`
cd tools/sysid
python gen_data.py            # writes $SYSID_WS/iddata.npz  (default /scratch/network/fl3689/sysid_ws)
python recover_sindy.py       # SINDy/STLSQ recovery + per-coefficient relative error
python recover_diffid.py      # differentiable-physics ID + identifiability report
```

Env knobs: `SYSID_WS` (workspace dir), `N_STEPS` (rollout length), `NSUB` (diff-ID subsample).
