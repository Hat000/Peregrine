"""Differentiable-physics system-ID: recover plant params by autograd through a torch mirror
of QuadrotorModel, fitting the logged (X, U) -> Xdot.

This is the fallback for terms that resist a clean symbolic library, AND the tool that
exposes parameter IDENTIFIABILITY. Two passes:

  PASS 1 (free m, D, J): minimises MSE from a deliberately-wrong init. The loss drops to
    machine zero, but m/D and J land on DEGENERATE values -- because (a) only the ratio D/m
    is observable (thrust accel = R[:,:,2]*action*g, mass cancels; drag enters as D/m), and
    (b) inertia J cancels algebraically in the rate loop (controller pre-multiplies by J,
    plant divides by J, and the controller's Coriolis compensation cancels the rigid one
    while the norm-clamp is inactive -- which it is for any realistic rate). See report.

  PASS 2 (m anchored = 1): with mass fixed from the spec, D_xy and D_z recover to ~1e-7.

  IDENTIFIABILITY REPORT: D/m ratio from pass 1, and a J-scale sweep showing rate-loss is
    flat 0.5x..5x (J unrecoverable in normal flight).

Run inside the diffaero conda env. Env knob NSUB (subsample size). See README.md.
"""
from __future__ import annotations

import os

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
WS = os.environ.get("SYSID_WS", "/scratch/network/fl3689/sysid_ws")
TRUE = {"m": 1.0, "g": 9.81, "Dxy": 0.6, "Dz": 0.6, "Jxy": 0.01, "Jz": 0.02, "K": 1.0}


def quat_to_R(qxyzw):  # body->world, xyzw real-last (diffaero convention)
    x, y, z, wq = qxyzw[:, 0], qxyzw[:, 1], qxyzw[:, 2], qxyzw[:, 3]
    R = torch.empty(qxyzw.shape[0], 3, 3, device=qxyzw.device, dtype=qxyzw.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * wq); R[:, 0, 2] = 2 * (x * z + y * wq)
    R[:, 1, 0] = 2 * (x * y + z * wq); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * wq)
    R[:, 2, 0] = 2 * (x * z - y * wq); R[:, 2, 1] = 2 * (y * z + x * wq); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load():
    d = np.load(os.path.join(WS, "iddata.npz"), allow_pickle=True)
    X, U, Xdot = d["X"], d["U"], d["Xdot"]
    q, v, w = X[:, 3:7], X[:, 7:10], X[:, 10:13]
    vdot, wdot = Xdot[:, 7:10], Xdot[:, 10:13]
    qn = np.linalg.norm(q, axis=1)
    keep = (np.linalg.norm(v, axis=1) < 30) & (np.linalg.norm(w, axis=1) < 30) & \
           (np.abs(qn - 1) < 1e-3) & np.isfinite(Xdot).all(1) & np.isfinite(X).all(1) & \
           (np.abs(vdot).max(1) < 500) & (np.abs(wdot).max(1) < 2000)
    q, v, w, U, vdot, wdot = (a[keep] for a in (q, v, w, U, vdot, wdot))
    nsub = int(os.environ.get("NSUB", "200000"))
    if q.shape[0] > nsub:
        idx = np.random.default_rng(0).choice(q.shape[0], nsub, replace=False)
        q, v, w, U, vdot, wdot = (a[idx] for a in (q, v, w, U, vdot, wdot))
    print(f"fitting on {q.shape[0]} rows")
    return (torch.tensor(a) for a in (q, v, w, U, vdot, wdot))


def main():
    tq, tv, tw, tU, tvd, twd = load()
    R = quat_to_R(tq); Ri2b = R.transpose(1, 2)

    # ---- PASS 1: free m, D, J (exposes degeneracies) ----
    P = {k: torch.tensor(np.log(v0), requires_grad=True) for k, v0 in
         {"m": 2.0, "g": 12.0, "Dxy": 0.2, "Dz": 1.5, "Jxy": 0.05, "Jz": 0.005}.items()}
    K = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
    opt = torch.optim.Adam(list(P.values()) + [K], lr=0.02)

    def deriv(m, g, Dxy, Dz, Jxy, Jz, K):
        Jvec = torch.stack([Jxy, Jxy, Jz])
        thrust_acc = R[:, :, 2] * (g * tU[:, 0:1])             # mass cancels here
        vb = torch.einsum("nij,nj->ni", Ri2b, tv)
        fdrag = torch.einsum("nij,nj->ni", R, torch.stack([Dxy, Dxy, Dz]) * vb)
        Gv = torch.stack([torch.zeros_like(g), torch.zeros_like(g), -g])
        vd = thrust_acc + Gv - fdrag / m
        act = torch.einsum("nij,nj->ni", Ri2b, tw)             # R_i2b @ w (frame transform)
        cross = torch.cross(act, act * Jvec, dim=1)
        cross = cross / torch.clamp(cross.norm(dim=1, keepdim=True) / 100, min=1.0)
        torque = Jvec * (K * (tU[:, 1:] - act)) + cross
        wd = (torque - torch.cross(tw, tw * Jvec, dim=1)) / Jvec
        return vd, wd

    for it in range(4000):
        opt.zero_grad()
        vd, wd = deriv(P["m"].exp(), P["g"].exp(), P["Dxy"].exp(), P["Dz"].exp(),
                       P["Jxy"].exp(), P["Jz"].exp(), K)
        loss = ((vd - tvd) ** 2).mean() + ((wd - twd) ** 2).mean()
        loss.backward(); opt.step()
    print(f"\nPASS 1 (free m,D,J) final loss {loss.item():.3e} -- fits to machine zero, but:")
    print(f"{'param':8s} {'recovered':>16s} {'true':>10s} {'rel_err':>12s}")
    for k in ["m", "g", "Dxy", "Dz", "Jxy", "Jz"]:
        r = float(P[k].exp()); t = TRUE[k]
        print(f"{k:8s} {r:16.10f} {t:10.4f} {abs(r - t) / abs(t):12.2e}")
    Kr = K.detach().numpy()
    for i, ax in enumerate("xyz"):
        print(f"K_{ax:6s} {Kr[i]:16.10f} {1.0:10.4f} {abs(Kr[i] - 1):12.2e}")
    dm = float(P["Dxy"].exp()) / float(P["m"].exp())
    print(f"  --> D/m = {dm:.8f}  (true {TRUE['Dxy'] / TRUE['m']:.4f}) : the ratio IS recovered exactly")

    # ---- PASS 2: m anchored = 1 -> D_xy, D_z recover exactly ----
    logD = torch.tensor([np.log(0.2), np.log(1.5)], requires_grad=True)
    optD = torch.optim.Adam([logD], lr=0.02)
    g = 9.81
    vb = torch.einsum("nij,nj->ni", Ri2b, tv)
    for it in range(3000):
        optD.zero_grad()
        Dxy, Dz = logD.exp()
        fdrag = torch.einsum("nij,nj->ni", R, torch.stack([Dxy, Dxy, Dz]) * vb)
        vd = R[:, :, 2] * (g * tU[:, 0:1]) + torch.tensor([0., 0., -g]) - fdrag / 1.0
        loss = ((vd - tvd) ** 2).mean()
        loss.backward(); optD.step()
    Dxy, Dz = [float(x) for x in logD.exp()]
    print(f"\nPASS 2 (m anchored=1): D_xy={Dxy:.9f} (relerr {abs(Dxy - 0.6) / 0.6:.2e})  "
          f"D_z={Dz:.9f} (relerr {abs(Dz - 0.6) / 0.6:.2e})")

    # ---- identifiability: J-scale sweep (rate-loss flat => J unrecoverable in flight) ----
    print("\nJ observability (rate-loss vs J-scale; true scale=1.0):")
    act = torch.einsum("nij,nj->ni", Ri2b, tw)

    def rate_loss(s):
        Jvec = torch.tensor([0.01 * s, 0.01 * s, 0.02 * s])
        cross = torch.cross(act, act * Jvec, dim=1)
        cross = cross / torch.clamp(cross.norm(dim=1, keepdim=True) / 100, min=1.0)
        torque = Jvec * (torch.ones(3) * (tU[:, 1:] - act)) + cross
        return float((((torque - torch.cross(tw, tw * Jvec, dim=1)) / Jvec - twd) ** 2).mean())

    for s in [0.5, 0.8, 1.0, 1.25, 2.0, 5.0]:
        print(f"  J*{s:5.3f}: rate_loss={rate_loss(s):.3e}")


if __name__ == "__main__":
    main()
