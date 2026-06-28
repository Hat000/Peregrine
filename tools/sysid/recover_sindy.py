"""SINDy / STLSQ recovery of diffaero's QuadrotorModel params on a physics-informed library.

Two linear-in-parameters subsystems, each solved exactly by STLSQ in a noise-free sim:

TRANSLATION (world-frame v_dot):
    v_dot = g * R[:,:,2] * u_thrust  -  g * e_z_world  -  (1/m) R D R^T v
  Library columns are the EXACT physical terms; coeffs recover thrust-gain (=g),
  gravity (=g), drag D_xy, drag D_z. With m known/anchored these recover to ~1e-7.

ROTATION (w_dot) -- FRAME-CORRECT form (see README footgun):
    w_dot = K * (w_des - R_i2b w)  +  J^-1 cross_comp  -  J^-1 (w x J w)
  The controller differences the commanded rate against ``R_i2b @ w`` (the state rate
  rotated world->body), NOT the raw state rate. Regress against ``R_i2b @ w`` and the
  exact extra (Coriolis comp - rigid Coriolis) term -> K_x,K_y,K_z recover to ~1e-9.
  (Regressing against raw ``w`` gives K ~= -0.66, R^2 ~= 0.02 on x/y -- the classic
  frame bug. The blind-quadratic fit at the end demonstrates that failure mode.)

Run inside the diffaero conda env (pysindy 2.1, scipy). See README.md.
"""
from __future__ import annotations

import os

import numpy as np
import pysindy as ps
from scipy.spatial.transform import Rotation

WS = os.environ.get("SYSID_WS", "/scratch/network/fl3689/sysid_ws")
TRUE = {"g": 9.81, "D_xy": 0.6, "D_z": 0.6, "m": 1.0, "J_xy": 0.01, "J_z": 0.02, "K": 1.0}


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else abs(a)


def main():
    d = np.load(os.path.join(WS, "iddata.npz"), allow_pickle=True)
    X, U, Xdot = d["X"], d["U"], d["Xdot"]
    print("loaded", X.shape, "samples")

    q, v, w = X[:, 3:7], X[:, 7:10], X[:, 10:13]
    vdot, wdot = Xdot[:, 7:10], Xdot[:, 10:13]
    qn = np.linalg.norm(q, axis=1)
    keep = (np.linalg.norm(v, axis=1) < 30) & (np.linalg.norm(w, axis=1) < 30) & \
           (np.abs(qn - 1) < 1e-3) & np.isfinite(Xdot).all(1) & np.isfinite(X).all(1)
    X, U, q, v, w, vdot, wdot = (a[keep] for a in (X, U, q, v, w, vdot, wdot))
    print("after envelope filter:", X.shape[0], "samples (%.1f%% kept)" % (100 * keep.mean()))

    # diffaero quats are XYZW (real last); scipy Rotation wants xyzw -> direct.
    R = Rotation.from_quat(q).as_matrix()              # (N,3,3) body->world
    Ri2b = np.transpose(R, (0, 2, 1))
    ez_b = R[:, :, 2]                                  # body z-axis in world
    u_thrust = U[:, 0]
    vb = np.einsum("nji,nj->ni", R, v)                 # R^T v = v in body frame
    N = X.shape[0]

    print("\n" + "=" * 70)
    print("TRANSLATIONAL SUBSYSTEM  (recover g, D_xy, D_z)   m known=1")
    print("=" * 70)
    Y = vdot.reshape(-1)                               # (3N,)
    col_thrust = (ez_b * u_thrust[:, None]).reshape(-1)                     # coeff -> +g
    col_grav = np.tile(np.array([0., 0., 1.]), N)                          # coeff -> -g
    drag_xy = (R[:, :, 0] * vb[:, 0:1] + R[:, :, 1] * vb[:, 1:2]).reshape(-1)  # coeff -> -D_xy
    drag_z = (R[:, :, 2] * vb[:, 2:3]).reshape(-1)                          # coeff -> -D_z
    Phi = np.stack([col_thrust, col_grav, drag_xy, drag_z], axis=1)        # (3N,4)
    opt = ps.STLSQ(threshold=1e-6, alpha=0.0, max_iter=50)
    opt.fit(Phi, Y[:, None])
    c = opt.coef_.ravel()
    print(f"{'term':18s} {'recovered':>14s} {'true':>10s} {'rel_err':>12s}")
    print(f"{'thrust gain = g':18s} {c[0]:14.9f} {TRUE['g']:10.4f} {rel(c[0], TRUE['g']):12.2e}")
    print(f"{'gravity = g':18s} {-c[1]:14.9f} {TRUE['g']:10.4f} {rel(-c[1], TRUE['g']):12.2e}")
    print(f"{'drag D_xy':18s} {-c[2]:14.9f} {TRUE['D_xy']:10.4f} {rel(-c[2], TRUE['D_xy']):12.2e}")
    print(f"{'drag D_z':18s} {-c[3]:14.9f} {TRUE['D_z']:10.4f} {rel(-c[3], TRUE['D_z']):12.2e}")
    resid = Y - Phi @ c
    print("translation fit residual RMS: %.3e  (R^2=%.12f)" % (
        np.sqrt((resid ** 2).mean()), 1 - resid.var() / Y.var()))

    print("\n" + "=" * 70)
    print("ROTATIONAL SUBSYSTEM  (frame-correct: err = w_des - R_i2b w)")
    print("=" * 70)
    Jx, Jy, Jz = TRUE["J_xy"], TRUE["J_xy"], TRUE["J_z"]
    Jvec = np.array([Jx, Jy, Jz])
    w_des = U[:, 1:]
    act = np.einsum("nij,nj->ni", Ri2b, w)            # R_i2b @ w  -- THE frame transform
    Jact = act * Jvec
    cross = np.cross(act, Jact)
    cn = np.linalg.norm(cross, axis=1, keepdims=True)
    cross_cl = cross / np.maximum(cn / 100, 1.0)      # controller norm-clamp (inactive in flight)
    for i, ax in enumerate("xyz"):
        # exact extra term = (J^-1 cross_comp - J^-1 (w x J w))_i ; coeff -> 1.0
        extra = (cross_cl / Jvec - np.cross(w, w * Jvec) / Jvec)[:, i]
        Phi_r = np.stack([(w_des[:, i] - act[:, i]), extra], axis=1)
        o = ps.STLSQ(threshold=1e-9, alpha=0.0, max_iter=80)
        o.fit(Phi_r, wdot[:, i][:, None])
        cr = o.coef_.ravel()
        res = wdot[:, i] - Phi_r @ cr
        print(f"  axis {ax}: K={cr[0]:.9f} (relerr {rel(cr[0], 1.0):.2e})  "
              f"extra_coef={cr[1]:.9f}  residRMS={np.sqrt((res ** 2).mean()):.3e}")

    print("\n  [demonstration] blind quadratic fit against RAW w (the frame bug):")
    idx = {0: (1, 2), 1: (2, 0), 2: (0, 1)}
    for i, ax in enumerate("xyz"):
        j, k = idx[i]
        quad = w[:, j] * w[:, k]
        Phi_b = np.stack([(w_des[:, i] - w[:, i]), quad], axis=1)   # RAW w -> wrong
        o = ps.STLSQ(threshold=1e-9, alpha=0.0, max_iter=80)
        o.fit(Phi_b, wdot[:, i][:, None])
        cb = o.coef_.ravel()
        res = wdot[:, i] - Phi_b @ cb
        print(f"    axis {ax}: K={cb[0]:+.4f}  R^2={1 - res.var() / wdot[:, i].var():.6f}  (raw-w => garbage on x,y)")


if __name__ == "__main__":
    main()
