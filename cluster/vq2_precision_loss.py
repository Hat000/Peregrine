"""LEVER 2 (loss): tighter inner-4 OKS sigma ON CLEAN DATA + an area-UN-normalized keypoint term.

Why: ultralytics' OKS keypoint loss is  (1 - exp(-d / ((2*sigma)^2 * area * 2))).  Two precision
blind-spots for our gate task:
  1. The `area` divisor means a NEAR (large-bbox) gate gets almost no penalty for several-pixel corner
     error -> the "85% of <4 m gates are imprecise" finding. An un-normalized distance term restores a
     gradient that does not vanish with area.
  2. OKS saturates at 1 for large errors (gradient -> 0), so once a corner is "wrong enough" the model
     stops being pushed. The un-normalized term is non-saturating -> keeps pulling corners in.
  3. The 4 INNER corners (idx 0..3) are the ones PnP actually uses; tightening THEIR sigma concentrates
     capacity on the precision-critical opening, away from the off-frame-prone outer corners (4..7).

This is applied ONLY when the trainer is launched with --precision-loss, by monkeypatching
v8PoseLoss so the capacity / baseline runs are completely unaffected. `gradient saturation` (the failure
mode that killed the poisoned-set sigma run) is mitigated by keeping the OKS term AND adding a bounded,
modestly-weighted distance term -- tune --l1-weight / --inner-sigma-scale conservatively.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PrecisionKeypointLoss(nn.Module):
    """Drop-in for ultralytics KeypointLoss: OKS (per-kpt sigma) + area-un-normalized distance term."""

    def __init__(self, sigmas: torch.Tensor, l1_weight: float = 0.05):
        super().__init__()
        self.sigmas = sigmas
        self.l1_weight = float(l1_weight)

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # OKS term (area-normalized, saturating) -- unchanged form, but with per-kpt (tighter inner) sigma.
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        oks = (1 - torch.exp(-e)) * kpt_mask
        # Area-UN-normalized distance term (non-saturating; survives large near-gate area). sqrt(d) is the
        # euclidean corner error in grid units -> a stable L1-like pull. Bounded by l1_weight.
        dist = torch.sqrt(d + 1e-9) * kpt_mask
        per = oks + self.l1_weight * dist
        return (kpt_loss_factor.view(-1, 1) * per).mean()


def apply_precision_loss_patch(inner_sigma_scale: float = 0.5, l1_weight: float = 0.05) -> None:
    """Monkeypatch v8PoseLoss to use PrecisionKeypointLoss with tighter inner-4 sigma.

    inner_sigma_scale < 1 tightens the inner-4 corners (smaller sigma => bigger penalty per pixel).
    Call BEFORE model.train(). Idempotent-ish (re-wraps the original __init__ once).
    """
    import ultralytics.utils.loss as L

    if getattr(L.v8PoseLoss, "_vq2_precision_patched", False):
        return
    orig_init = L.v8PoseLoss.__init__

    def patched_init(self, model, *a, **k):
        orig_init(self, model, *a, **k)
        nkpt = self.kpt_shape[0]
        sig = torch.ones(nkpt, device=self.device) / nkpt
        inner = min(4, nkpt)
        sig[:inner] = sig[:inner] * float(inner_sigma_scale)   # tighter inner-4
        self.keypoint_loss = PrecisionKeypointLoss(sig, l1_weight=l1_weight)

    patched_init._vq2_orig = orig_init
    L.v8PoseLoss.__init__ = patched_init
    L.v8PoseLoss._vq2_precision_patched = True
    print(f"VQ2_PRECISION_LOSS applied: inner_sigma_scale={inner_sigma_scale} l1_weight={l1_weight}")
