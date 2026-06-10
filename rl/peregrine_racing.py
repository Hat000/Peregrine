"""PeregrineRacing -- DiffAero racing env subclassed onto OUR VQ1 6-gate course (given pose, no vision).

Registered into ``diffaero.env.ENV_ALIAS["peregrine_racing"]`` by the launcher (``peregrine_train_racing``)
so the pristine diffaero clone is never edited (same monkeypatch discipline as the dynamics injection).
Select it with ``env=racing env.name=peregrine_racing`` (keeps racing.yaml's weights/dt/max_time; only
``cfg.name`` routing changes).

WHAT WE CHANGE vs ``diffaero.env.racing.Racing`` (and WHY) -- everything else is inherited:
  * GATES: replace the synthetic flat figure-8 with our 6-gate descending course (``peregrine_course``
    -> ``peregrine_course_diffaero.json``, expressed in DiffAero's Z-up frame). Non-looping: a point-to-
    point race, gate 0 -> gate 5 = FINISH (the parent loops with ``% n_gates``).
  * BOUNDS: the parent's hardcoded |x,y|<5 / z<7 box fits the figure-8; our course spans x in [-159,-23],
    descends 26 m. We replace it with a generous box around the course bbox so only genuinely-lost drones
    truncate.
  * PASS APERTURE: the parent passes/collides on an L1 "radius" 1.5 m; our gate inner opening is a 1.5 m
    SQUARE -> half-width 0.75 m, L-inf (``is_passed``). Crossing the plane outside the square = collision.
  * OBS (+4 vs parent's 13): append body rates (3) + last collective (1) -- our CTBR policy's target I/O
    (the parent omits both). Layout: [pos_g(3), vel_g(3), rpy_g(3), body_rates(3), collective(1),
    next_gate_relpos(3), next_gate_relyaw(1)] = 17.
  * REWARD: inherit the parent's quadrotor reward (dense gate PROGRESS w=10 + COLLISION w=-10 +
    rate-smoothness jerk w=-0.1 -- exactly our target's progress/collision/smoothness) and ADD a discrete
    gate-PASSAGE bonus + a FINISH bonus (the parent has neither for the quadrotor). No reference line
    (the policy learns it); no visibility term (speed > keeping the gate in view). For PPO only ``reward``
    matters -- the parent's ``loss``/``pos_loss``/``oob_loss`` terms are dead code here, untouched.

DR (plant domain randomization) lives in the DYNAMICS adapter (``rl.diffaero_dynamics``), resampled per
env at reset -- not here.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Union

import torch
import pytorch3d.transforms as T
from torch import Tensor

from diffaero.env.racing import Racing, get_gate_rotmat_w2g
from diffaero.utils.math import mvp

_COURSE_JSON = os.path.join(os.path.dirname(__file__), "peregrine_course_diffaero.json")


class PeregrineRacing(Racing):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)  # builds the figure-8; we overwrite the course below

        # --- our course (DiffAero Z-up frame), replacing the synthetic gates ---
        course_path = getattr(cfg, "course_json", None) or _COURSE_JSON
        course = json.loads(open(course_path).read())
        gates = course["gates"]
        self.n_gates = len(gates)
        self.gate_pos = torch.tensor([g["pos_zup"] for g in gates], device=device, dtype=torch.float32)
        self.gate_yaw = torch.tensor([g["yaw"] for g in gates], device=device, dtype=torch.float32)
        self.gate_half_opening_m = float(course.get("inner_opening_m", 1.5)) / 2.0  # 0.75 m

        # recompute the parent's next-gate-relative lookahead for OUR gate count/geometry
        self.gate_rel_pos = torch.zeros(self.n_gates, 3, device=device)
        self.gate_yaw_rel = torch.zeros(self.n_gates, device=device)
        for i in range(self.n_gates):
            self.gate_rel_pos[i] = get_gate_rotmat_w2g(self.gate_yaw[i - 1]) @ (self.gate_pos[i] - self.gate_pos[i - 1])
            yaw_diff = self.gate_yaw[i] - self.gate_yaw[i - 1]
            self.gate_yaw_rel[i] = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

        # non-looping finish bookkeeping
        self.finished = torch.zeros(self.n_envs, dtype=torch.bool, device=device)

        # generous out-of-bounds box around the course bbox (only lost drones truncate)
        margin_xy, margin_z = 15.0, 12.0
        lo = self.gate_pos.amin(dim=0) - torch.tensor([margin_xy, margin_xy, margin_z], device=device)
        hi = self.gate_pos.amax(dim=0) + torch.tensor([margin_xy, margin_xy, margin_z], device=device)
        self.box_min, self.box_max = lo, hi

        # reward shaping bonuses (the parent supplies progress/collision/jerk; we add these)
        self.passage_bonus = float(getattr(cfg, "passage_bonus", 10.0))
        self.finish_bonus = float(getattr(cfg, "finish_bonus", 20.0))

        # STANDING START (S1.3, from the S1.2 live-validation findings 2026-06-10): with this
        # fraction, a reset env spawns at the REAL race start instead of 1 m in front of a random
        # gate: 23.3 m up-course of gate 0, at rest, on the tilted pad. Pose measured from race
        # recordings (first ODOMETRY at GO: NED pos (0,0,+0.02), true rpy (0, -17.8deg, -179.9deg))
        # and mapped through the DEPLOYMENT VIRTUAL FLIP (pi about body z -- rl/fly_rl.py flies the
        # policy in that frame because training is tail-first): zup euler (roll 0, pitch -17.8deg,
        # yaw ~0) = quat XYZW [-0.000135, -0.15471, -0.000862, 0.987959]. Default 0.0 = off.
        self.standing_start_frac = float(getattr(cfg, "standing_start_frac", 0.0))
        self._spawn_pos_zup = torch.tensor([0.0, 0.0, -0.02], device=device)
        self._spawn_quat_xyzw = torch.tensor(
            [-0.000135, -0.15471, -0.000862, 0.987959], device=device)

        # obs = parent's 13 + body rates (3) + collective (1)
        self.obs_dim = 17

    # ---- observation: parent's gate-relative obs + body rates + last collective ----
    def get_observations(self, with_grad=False):
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat_w2g = get_gate_rotmat_w2g(gate_yaw)

        pos_g = mvp(rotmat_w2g, gate_pos - self._p)            # target gate rel pos, gate frame (3)
        vel_g = mvp(rotmat_w2g, self._v)                        # velocity, gate frame (3)
        rotmat_b2w = T.quaternion_to_matrix(self.q.roll(1, dims=-1))
        rotmat_b2g = torch.matmul(rotmat_w2g, rotmat_b2w)
        rpy_g = T.matrix_to_euler_angles(rotmat_b2g, "ZYX")[..., [2, 1, 0]]  # attitude vs gate (3)

        next_gate_idx = torch.clamp(self.target_gates.long() + 1, max=self.n_gates - 1)
        collective = self.last_action[..., 0:1]                # last normed-thrust command (1)

        obs = torch.cat([
            pos_g, vel_g, rpy_g,
            self._w,                                            # body rates (3)
            collective,                                         # collective (1)
            self.gate_rel_pos[next_gate_idx],                  # next gate rel pos (3)
            self.gate_yaw_rel[next_gate_idx].unsqueeze(-1),    # next gate rel yaw (1)
        ], dim=-1)
        return obs if with_grad else obs.detach()

    # ---- pass / collision on OUR 1.5 m square opening (L-inf), plane-crossing on gate +X ----
    def is_passed(self, prev_pos):
        gate_pos = self.gate_pos[self.target_gates]
        gate_yaw = self.gate_yaw[self.target_gates]
        rotmat = get_gate_rotmat_w2g(gate_yaw)
        prev_rel = mvp(rotmat, prev_pos - gate_pos)
        curr_rel = mvp(rotmat, self.p - gate_pos)
        pass_through = (prev_rel[:, 0] < 0) & (curr_rel[:, 0] > 0)
        h = self.gate_half_opening_m
        inside_gate = (curr_rel[:, 1].abs() < h) & (curr_rel[:, 2].abs() < h)
        return pass_through & inside_gate, pass_through & ~inside_gate

    # ---- bounds: course-sized box (replaces the parent's figure-8 box) ----
    def truncated(self):
        oob = (
            (self.p[:, 0] < self.box_min[0]) | (self.p[:, 0] > self.box_max[0]) |
            (self.p[:, 1] < self.box_min[1]) | (self.p[:, 1] > self.box_max[1]) |
            (self.p[:, 2] < self.box_min[2]) | (self.p[:, 2] > self.box_max[2])
        )
        return oob | (self.progress >= self.max_steps)

    # ---- step: parent's loop, but NON-looping with a finish + passage/finish bonuses ----
    def step(self, action, next_obs_before_reset=False, next_state_before_reset=False):
        # type: (Tensor, bool, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor, Dict[str, Union[Dict[str, Tensor], Tensor]]]
        prev_pos = self._p.clone()
        self.dynamics.step(action)

        gate_passed, gate_collision = self.is_passed(prev_pos)
        is_last = self.target_gates == (self.n_gates - 1)
        newly_finished = gate_passed & is_last & ~self.finished
        advance = gate_passed & ~is_last
        self.target_gates[advance] = self.target_gates[advance] + 1   # NO wraparound
        self.n_passed_gates[gate_passed] += 1
        self.finished |= newly_finished
        self.target_pos.copy_(self.gate_pos[self.target_gates])

        terminated = gate_collision | self.finished
        truncated = self.truncated()
        self.progress += 1
        if self.renderer is not None:
            self.renderer.render(self.states_for_render())
            truncated = torch.full_like(truncated, self.renderer.gui_states["reset_all"]) | truncated
        success = self.finished.clone()
        self.last_action.copy_(action.detach())

        reset = terminated | truncated
        reset_indices = reset.nonzero().view(-1)
        loss, reward, loss_components = self.loss_and_reward(action, prev_pos, gate_passed, gate_collision)
        reward = reward + self.passage_bonus * gate_passed.float() + self.finish_bonus * newly_finished.float()

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
            },
        }
        if next_obs_before_reset:
            extra["next_obs_before_reset"] = self.get_observations(with_grad=True)
        if next_state_before_reset:
            extra["next_state_before_reset"] = self.get_state(with_grad=True)
        if reset_indices.numel() > 0:
            self.reset_idx(reset_indices)
        return self.get_observations(), (loss, reward), terminated, extra

    # ---- reset: parent places the drone 1 m in front of a random gate; we also clear `finished`
    # and (optionally) respawn a fraction at the real standing start ----
    def reset_idx(self, env_idx: Tensor):
        super().reset_idx(env_idx)
        self.finished[env_idx] = False
        if self.standing_start_frac > 0.0 and env_idx.numel() > 0:
            pick = env_idx[torch.rand(env_idx.numel(), device=self.device)
                           < self.standing_start_frac]
            if pick.numel() > 0:
                n = pick.numel()
                # pose jitter: +-0.25 m xy, +-0.15 rad axis-angle (robustness around the fixed pad)
                pos = self._spawn_pos_zup.expand(n, 3).clone()
                pos = pos + torch.cat([0.5 * (torch.rand(n, 2, device=self.device) - 0.5),
                                       torch.zeros(n, 1, device=self.device)], dim=-1)
                jit = 0.3 * (torch.rand(n, 3, device=self.device) - 0.5)        # axis-angle
                q_jit = T.axis_angle_to_quaternion(jit)                          # wxyz
                q_spawn = self._spawn_quat_xyzw.expand(n, 4).roll(1, dims=-1)    # xyzw -> wxyz
                q = T.quaternion_multiply(q_spawn, q_jit).roll(-1, dims=-1)      # -> xyzw
                state = torch.zeros(n, self.dynamics.state_dim, device=self.device)
                state[:, 0:3] = pos
                state[:, 3:7] = q
                mask = torch.zeros_like(self.dynamics._state, dtype=torch.bool)
                mask[pick] = True
                full = torch.zeros_like(self.dynamics._state)
                full[pick] = state
                self.dynamics._state = torch.where(mask, full, self.dynamics._state)
                self.target_gates[pick] = 0
                self.init_pos[pick] = pos
                self.target_pos.copy_(self.gate_pos[self.target_gates])
