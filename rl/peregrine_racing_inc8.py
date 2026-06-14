"""PeregrineRacingInc8 -- the inc8 case-C training env (DiffAero, GPU/cluster-only).

A SUBCLASS of PeregrineRacing (rl/peregrine_racing.py); the pristine diffaero clone is never edited
and the inc7 env is never destructively changed. Registered via ENV_ALIAS["peregrine_racing_inc8"]
by the launcher (rl/peregrine_train_inc8.py).

WHAT IT ADDS over inc7 (all gated behind ``+env.inc8=true``; default OFF == byte-identical inc7):
  * IN-LOOP ESTIMATOR EMULATION (rl/inc8_estimator_emul.py): per step, a batched LinearKF driven by
    the calibrated fix-surrogate runs over all envs -- camera-pointing -> fix density -> KF accuracy
    -> the obs the policy is scored on. The obs[0:6] (pos_g/vel_g) are sourced from the KF estimate
    (the case-C deploy seam); attitude/rates/collective/lookahead stay TRUTH. obs_dim 17 -> 20
    (the FROZEN d5 confidence triple obs[17:20]).
  * REWARD DELTAS (rl/inc8_reward.py) over the FROZEN inc7 contract: R1' arc-length progress along
    the rebuilt corrected-aero line Gamma (replaces R1-to-centre), a FLAT truth-seen GT-estimator-
    error anchor, annealed confidence/staleness shaping, and the configurable COWORK-3 2-axis
    terminal-lock perception reward R5' (arm A default / B flat / C off). Everything else (R2/T1/T2/
    T3/R3/R4 60-deg cone/R5/R6/R7/T4) is untouched.
  * CRITIC 33 -> 36: the asymmetric critic gets the TRUTH gate-relative state (inc7's get_state) +
    the same confidence triple -- the anti-damping mechanism (the critic knows the true margin, so
    needless braking under HIGH confidence is penalised by the advantage; no shaped damping term).
  * BSR3 SPIN-GATE: a sustained-spin termination on the REALIZED body rate (self._w), wide enough
    (spin_rate_abort ~10 rad/s for spin_time_abort ~3 s) that a legitimate ~11 rad/s super-rate
    TRANSIENT does not abort. Gated on spin_rate_abort > 0 (default OFF).

PPO-ONLY / NO_GRAD: the estimator emulation (KF + Bernoulli fix) is non-differentiable and runs under
no_grad; reward is detached (mirrors the parent's ``reward = reward.detach()``). The torch core is
PARITY-GATED against the numpy reference rl/estimator_emul.py (tests/test_inc8_*_torch.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import fields, replace

import numpy as np

try:
    import torch
except Exception:                       # pragma: no cover - torch absent in some tooling contexts
    torch = None

from peregrine_racing import (PeregrineRacing, RewardWeights, compute_reward_terms,  # noqa: E402
                              crossing_events, roll_from_quat_xyzw, slab_frame_hits,
                              tilt_cos_from_quat_xyzw, world_to_gateframe)
import inc8_estimator_emul as IE                                      # noqa: E402
import inc8_reward as R8                                             # noqa: E402
from reference_line_torch import BatchedReferenceLine                # noqa: E402

_REFLINE_JSON = os.path.join(os.path.dirname(__file__), "reference_line_inc8.json")
_FLIP_ZUP_NED = (1.0, -1.0, -1.0)       # involutory Z-up<->NED / FLU<->FRD axis flip


def _inc8_weights_from_cfg(cfg) -> "R8.Inc8RewardWeights":
    """Build Inc8RewardWeights from cfg (``+env.rw_<field>=...``), defaulting to the dataclass."""
    kw = {}
    for f in fields(R8.Inc8RewardWeights):
        kw[f.name] = type(f.default)(getattr(cfg, f"rw_{f.name}", f.default)) \
            if not isinstance(f.default, str) else getattr(cfg, f"rw_{f.name}", f.default)
    return R8.Inc8RewardWeights(**kw)


class PeregrineRacingInc8(PeregrineRacing):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)        # builds the inc7 env (obs_dim 17, gate tensors, rw, ...)
        self._inc8_on = bool(getattr(cfg, "inc8", False))

        # BSR3 spin-gate (active within the inc8 path; default OFF). The inc8 launcher sets
        # spin_rate_abort ~10, spin_time_abort 3.0 (d5 4.1). Gates the REALIZED rate, not the command.
        self.spin_rate_abort = float(getattr(cfg, "spin_rate_abort", 0.0))
        self.spin_time_abort = float(getattr(cfg, "spin_time_abort", 3.0))
        self.rw_spin = float(getattr(cfg, "rw_spin", 25.0))

        if not self._inc8_on:
            return                            # pure inc7 (OFF fallback is byte-identical)

        if self.course_mode != "vq1":
            raise ValueError("inc8 requires course_mode=vq1: R1' progress is measured along the VQ1 "
                             "reference line Gamma (reference_line_inc8.json); a procedural per-env "
                             "course would not match Gamma.")
        if torch is None:                     # pragma: no cover
            raise RuntimeError("inc8 requires torch")

        dev = self.device
        self._inc8_dtype = self.gate_pos.dtype
        self._flip_t = torch.tensor(_FLIP_ZUP_NED, device=dev, dtype=self._inc8_dtype)
        self._r5_arm = str(getattr(cfg, "r5_arm", "A")).upper()
        if self._r5_arm not in ("A", "B", "C"):
            raise ValueError(f"r5_arm must be A|B|C, got {self._r5_arm!r}")
        self._inc8w = _inc8_weights_from_cfg(cfg)
        self._global_step = 0

        # estimator-emulation config (the values that supersede d5 per the prompt / MEMORY NOW).
        self._emul_cfg = IE.EmulConfig(
            sigma_lat_lo=float(getattr(cfg, "emul_sigma_lat_lo", IE.EmulConfig.sigma_lat_lo)),
            sigma_lat_hi=float(getattr(cfg, "emul_sigma_lat_hi", IE.EmulConfig.sigma_lat_hi)),
            bias_mag_lo=float(getattr(cfg, "emul_bias_mag_lo", IE.EmulConfig.bias_mag_lo)),
            bias_mag_hi=float(getattr(cfg, "emul_bias_mag_hi", IE.EmulConfig.bias_mag_hi)),
            inject_bias=bool(getattr(cfg, "emul_inject_bias", IE.EmulConfig.inject_bias)),
        )
        # optional surrogate recalibration from on-disk checkpoints (no scipy: plain json).
        params = IE.TorchSurrogateParams()
        ckpt_dir = getattr(cfg, "fix_surrogate_ckpt_dir", None)
        if ckpt_dir:
            params = self._load_surrogate_params(ckpt_dir, params)

        # NED course geometry for the emulator (VQ1 == shared across envs; take env 0).
        gate_pos_ned = (self.gate_pos[0] * self._flip_t).to(self._inc8_dtype)        # (G,3)
        R_world_gate = IE.ned_gate_frame_torch(self.gate_yaw[0].to(self._inc8_dtype))  # (G,3,3)
        self._emu = IE.BatchedEstimatorEmulator(
            self.n_envs, gate_pos_ned, R_world_gate, config=self._emul_cfg, params=params,
            device=dev, dtype=self._inc8_dtype)
        self._refline = BatchedReferenceLine.load(_REFLINE_JSON, dev, self._inc8_dtype)
        self._spin_clock = torch.zeros(self.n_envs, device=dev, dtype=self._inc8_dtype)

        # obs/critic dims: 17->20, 33->36. Deploy/load gates on the checkpoint sidecar obs-dim.
        self.obs_dim = 20
        # init the emulator at the current (hover/placeholder) truth for all envs; the runner's
        # reset() re-inits at the real spawn before the first real obs.
        self._reset_emulator(self._arange)

    # ---- surrogate recalibration (plain json; no scipy) ---------------------------------------
    @staticmethod
    def _load_surrogate_params(ckpt_dir, base: "IE.TorchSurrogateParams") -> "IE.TorchSurrogateParams":
        """Reload the fitted accept/sigma sub-model checkpoints (mirrors FixSurrogate.from_checkpoints
        for the fields the torch core consumes). Missing files keep the baked defaults."""
        kw = {}
        a = _maybe_json(os.path.join(ckpt_dir, "accept.json"))
        if a:
            bp = a["bandpass_params"]
            kw.update(accept_pmax=bp[0], accept_rlo=bp[1], accept_wlo=bp[2], accept_rhi=bp[3],
                      accept_whi=bp[4], accept_p_out_of_image=a["p_accept_out_of_image"])
        s = _maybe_json(os.path.join(ckpt_dir, "sigma.json"))
        if s:
            kw.update(sigma_lateral_floor=s["lateral"]["floor"], sigma_lateral_a1=s["lateral"]["a1"],
                      sigma_vertical_floor=s["vertical"]["floor"], sigma_vertical_a1=s["vertical"]["a1"],
                      sigma_depth_floor=s["depth"]["floor"], sigma_depth_a1=s["depth"]["a1"])
        return replace(base, **kw)

    # ---- estimator-emulation reset ------------------------------------------------------------
    def _reset_emulator(self, env_idx) -> None:
        """COLD-init the KF at the current truth spawn + draw the per-episode DR for ``env_idx`` +
        reset the spin clock. Reads the post-spawn self._p/_v (Z-up) -> NED."""
        m = int(env_idx.numel())
        if m == 0:
            return
        pos_ned = (self._p[env_idx] * self._flip_t).to(self._inc8_dtype)
        vel_ned = (self._v[env_idx] * self._flip_t).to(self._inc8_dtype)
        sig, bias = IE.BatchedEstimatorEmulator.sample_episode_dr(
            self._emul_cfg, m, self.device, self._inc8_dtype)
        self._emu.reset_idx(env_idx, pos_ned, vel_ned, sig, bias)
        self._spin_clock[env_idx] = 0.0

    def _flipconj(self, R):
        """R_world_body Z-up -> NED: diag(flip) @ R @ diag(flip) (flip involutory). Batched (N,3,3)."""
        f = self._flip_t
        return R * f.view(1, 3, 1) * f.view(1, 1, 3)

    # ---- observation (inc8: KF pose + confidence triple; OFF -> byte-identical inc7) -----------
    def get_observations(self, with_grad=False):
        if not self._inc8_on:
            return super().get_observations(with_grad)
        ar, tg = self._arange, self.target_gates.long()
        nxt = torch.clamp(tg + 1, max=self.n_gates - 1)
        with torch.no_grad():
            R_b2w_zup = IE.quat_xyzw_to_matrix_torch(self._q)        # truth attitude (deploy seam)
            kf_pos = self._emu.kf_pos_zup()
            kf_vel = self._emu.kf_vel_zup()
            triple = self._emu.confidence_channel(tg)
        obs = IE.obs_zup_torch(
            kf_pos, kf_vel, R_b2w_zup, self._w,
            self.gate_pos[ar, tg], self.gate_yaw[ar, tg],
            self.gate_rel_pos[ar, nxt], self.gate_yaw_rel[ar, nxt],
            self.last_action[..., 0], triple=triple, virtual_flip=False)
        finite = torch.isfinite(obs)
        if not bool(finite.all()):
            self._nonfinite_obs += int((~finite).sum())
            obs = torch.where(finite, obs, torch.zeros_like(obs))
        return obs if with_grad else obs.detach()

    # ---- state (asymmetric critic: TRUTH gate-relative 33 + confidence triple = 36) ------------
    def get_state(self, with_grad=False):
        if not self._inc8_on:
            return super().get_state(with_grad)
        base = super().get_state(with_grad=True)                    # 33-dim TRUTH gate-relative
        with torch.no_grad():
            triple = self._emu.confidence_channel(self.target_gates.long())
        state = torch.cat([base, triple], dim=-1)                   # 36-dim
        return state if with_grad else state.detach()

    # ---- step (inc8: estimator-emul + R1'/GT/CS/R5' + BSR3; OFF -> byte-identical inc7) ---------
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        if not self._inc8_on:
            return super().step(action, next_obs_before_reset, next_state_before_reset)
        ar, G = self._arange, self.n_gates
        # capture prev truth (Z-up) BEFORE the dynamics step (for the IMU synthesis)
        prev_pos = self._p.clone()
        prev_vel = self._v.clone()
        prev_q = self._q.clone()
        self.dynamics.step(action)
        curr_pos = self._p
        tg = self.target_gates.long()

        # ---- IN-LOOP estimator emulation (NO_GRAD) + reference-line progress ----
        with torch.no_grad():
            f = self._flip_t
            prev_pos_ned = prev_pos * f
            cur_pos_ned = curr_pos * f
            R_prev_ned = self._flipconj(IE.quat_xyzw_to_matrix_torch(prev_q))
            R_cur_ned = self._flipconj(IE.quat_xyzw_to_matrix_torch(self._q))
            n = self.n_envs
            accept_u = torch.rand(n, device=self.device, dtype=self._inc8_dtype)
            accel_noise = torch.randn(n, 3, device=self.device, dtype=self._inc8_dtype)
            fix_noise = torch.randn(n, 3, device=self.device, dtype=self._inc8_dtype)
            accepted = self._emu.step(
                prev_pos_ned, prev_vel * f, R_prev_ned, cur_pos_ned, self._v * f, R_cur_ned,
                tg, float(self.dt), accept_u, accel_noise, fix_noise)
            s_prev = self._refline.progress(prev_pos_ned)
            s_curr = self._refline.progress(cur_pos_ned)
            geom = self._emu._last_geom
            err_ip = self._emu.gate_frame_error_inplane(tg, cur_pos_ned)
            triple = self._emu.confidence_channel(tg)
            # BSR3 spin gate on the REALIZED body rate (self._w), NOT the command
            self._spin_clock, spin_abort = R8.bsr3_update(
                self._spin_clock, self._w, float(self.dt), self.spin_rate_abort, self.spin_time_abort)

        # ===== crossings/terminations/advance: VERBATIM inc7 (frozen contact contract) =====
        rel_prev = world_to_gateframe(prev_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        rel_curr = world_to_gateframe(curr_pos[:, None, :] - self.gate_pos, self.gate_yaw)
        half_in_eff, half_out_eff = self._contact_bands()
        ev = crossing_events(rel_prev, rel_curr, half_in_eff, half_out_eff)
        slab_hit = (slab_frame_hits(rel_prev, rel_curr, half_in_eff, half_out_eff,
                                    self.frame_depth_m) if self.frame_depth_m > 0.0 else None)
        fwd_t = ev["fwd"][ar, tg]
        gate_passed = fwd_t & ev["pass_ok"][ar, tg]
        gate_miss = fwd_t & ~ev["pass_ok"][ar, tg] & ~ev["in_frame"][ar, tg]
        frame_target = (ev["fwd"] | ev["bwd"])[ar, tg] & ev["in_frame"][ar, tg]
        strike_any = (ev["fwd"] | ev["bwd"]) & ev["in_frame"]
        strike_any[ar, tg] = False
        gate_collision = frame_target | strike_any.any(dim=1)
        if slab_hit is not None:
            slab_t = slab_hit[ar, tg]
            gate_passed = gate_passed & ~slab_t
            gate_miss = gate_miss & ~slab_t
            gate_collision = gate_collision | slab_hit.any(dim=1)
        pass_linf = ev["linf"][ar, tg]

        is_last = tg == (G - 1)
        newly_finished = gate_passed & is_last & ~self.finished
        advance = gate_passed & ~is_last
        self.target_gates[advance] = self.target_gates[advance] + 1
        self.n_passed_gates[gate_passed] += 1
        self.finished |= newly_finished
        tg_new = self.target_gates.long()
        self.target_pos.copy_(self.gate_pos[ar, tg_new])

        oob = ((curr_pos < self.box_min) | (curr_pos > self.box_max)).any(dim=-1)

        # inc8 termination = inc7 set + BSR3 sustained-spin
        terminated = gate_collision | gate_miss | oob | self.finished | spin_abort
        truncated = self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        truncated = truncated & ~terminated
        success = self.finished.clone()

        tilt = torch.arccos(tilt_cos_from_quat_xyzw(self._q).clamp(-1.0, 1.0))
        self._peak_tilt = torch.maximum(self._peak_tilt, tilt)
        self._peak_roll = torch.maximum(self._peak_roll, roll_from_quat_xyzw(self._q).abs())
        speed = torch.linalg.norm(self._v, dim=-1)
        self._speed_sum += speed

        # ===== reward: FROZEN inc7 terms (R1 zeroed) + R1' + GT + CS + R5' - spin penalty =====
        gate_pos_t = self.gate_pos[ar, tg_new]
        prev_d2g = torch.linalg.norm(prev_pos - gate_pos_t, dim=-1)
        curr_d2g = torch.linalg.norm(curr_pos - gate_pos_t, dim=-1)
        time_left_s = (self.max_steps - self.progress).clamp(min=0).float() * self.dt
        a_norm = (action - self._act_lo) / self._act_span
        last_norm = (self.last_action - self._act_lo) / self._act_span
        w_frozen = replace(self.rw, progress=0.0)        # R1-to-centre OFF; R1' replaces it
        reward_frozen, loss_components = compute_reward_terms(
            w_frozen, prev_d2g=prev_d2g, curr_d2g=curr_d2g, gate_passed=gate_passed,
            gate_collision=gate_collision, gate_miss=gate_miss, oob=oob,
            newly_finished=newly_finished, time_left_s=time_left_s, quat_xyzw=self._q,
            omega=self._w, action_norm=a_norm, last_action_norm=last_norm)

        delta_s = s_curr - s_prev
        anneal = R8.confidence_anneal(float(self._global_step), self._inc8w)
        r1p = R8.arc_progress_reward(s_curr, s_prev, self._inc8w.progress)
        gt = R8.gt_estimerr_anchor(err_ip, self._inc8w.estimerr)
        cs = R8.confidence_shaping_reward(triple, self._inc8w.conf_shape, anneal)
        r5 = R8.perception_reward(geom["t_cam"], geom["range"], delta_s, geom["in_image"],
                                  self._r5_arm, self._inc8w)
        spin_pen = self.rw_spin * spin_abort.float()
        reward = reward_frozen + r1p + gt + cs + r5 - spin_pen

        loss = (-reward).detach()
        reward = reward.detach()
        # ---- inc8 diagnostics (the smoke's pointing / fix / estimator-error curves) ----
        in_img = geom["in_image"].float()
        terminal = (geom["range"] <= R8.Inc8RewardWeights().perc_d_lock_m) & (geom["range"] > 0)
        loss_components.update({
            "obs_nonfinite": float(self._nonfinite_obs),
            "inc8_fix_rate": accepted.float().mean().item(),
            "inc8_pointing_rate": in_img.mean().item(),
            "inc8_terminal_pointing": (in_img[terminal].mean().item()
                                       if bool(terminal.any()) else 0.0),
            "inc8_estim_err_inplane_m": err_ip.mean().item(),
            "inc8_c_inplane": triple[:, 0].mean().item(),
            "inc8_age_norm": triple[:, 2].mean().item(),
            "inc8_r1p": r1p.mean().item(),
            "inc8_r5_perc": r5.mean().item(),
            "inc8_gt_anchor": gt.mean().item(),
            "inc8_conf_anneal": anneal,
            "inc8_spin_abort_rate": spin_abort.float().mean().item(),
        })
        self._global_step += 1
        self.last_action.copy_(action.detach())

        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        extra = {
            "truncated": truncated,
            "l": self.progress.clone(),
            "reset": reset,
            "reset_indicies": reset_indices,
            "success": success,
            "loss_components": loss_components,
            "stats_raw": {
                "success_rate": success[reset].float(),
                "survive_rate": truncated[reset].float(),
                "l_episode": ((self.progress.clone() - 1) * self.dt)[reset],
                "n_passed_gates": self.n_passed_gates[reset].float(),
                "collision_rate": gate_collision[reset].float(),
                "miss_rate": gate_miss[reset].float(),
                "oob_rate": oob[reset].float(),
                "spin_abort_rate": spin_abort[reset].float(),
                "peak_tilt_deg": torch.rad2deg(self._peak_tilt)[reset],
                "peak_roll_deg": torch.rad2deg(self._peak_roll)[reset],
                "mean_speed": (self._speed_sum / self.progress.clamp(min=1).float())[reset],
                "finish_time_s": ((self.progress.clone() - 1).float() * self.dt)[newly_finished],
                "pass_offset_m": pass_linf[gate_passed],
                "fix_rate": accepted.float()[reset],
                "pointing_rate": in_img[reset],
                "estim_err_inplane_m": err_ip[reset],
                "slab_collision_rate": (slab_hit.any(dim=1) if slab_hit is not None
                                        else torch.zeros_like(gate_collision))[reset].float(),
                "pass_margin_m": ((half_in_eff.squeeze(-1) if torch.is_tensor(half_in_eff)
                                   else torch.full_like(pass_linf, half_in_eff))
                                  - pass_linf)[gate_passed],
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra

    # ---- reset: spawn (inc7) THEN re-init the emulator at the new truth ------------------------
    def reset_idx(self, env_idx):
        super().reset_idx(env_idx)
        if getattr(self, "_inc8_on", False) and hasattr(self, "_emu"):
            self._reset_emulator(env_idx)


def _maybe_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
