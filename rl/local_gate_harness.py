"""LOCAL (laptop, CPU-torch) dry run of check_diffaero_gate -- catches torch-mirror bugs before the
Adroit roundtrip. Stubs diffaero's BaseDynamics with a value-faithful minimal base (grad_decay is
value-preserving in the real one -- it only scales GRADIENTS -- so identity is exact for a value
parity check). The REAL gate on Adroit (real BaseDynamics, GPU) remains the authoritative pass.

Run from repo root:  .venv\\Scripts\\python.exe rl\\local_gate_harness.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- stub diffaero.dynamics.base_dynamics.BaseDynamics BEFORE importing the adapter -------------
base_mod = types.ModuleType("diffaero.dynamics.base_dynamics")


class BaseDynamics:                                      # mirror of the attrs the adapter uses
    def __init__(self, cfg, device):
        self.n_agents = int(getattr(cfg, "n_agents", 1))
        self.n_envs = int(getattr(cfg, "n_envs", 1))
        self.dt = float(cfg.dt)
        self.alpha = float(getattr(cfg, "alpha", 1.0))
        self.device = device

    def grad_decay(self, x):                             # value-preserving (real one scales grads)
        return x

    def detach(self):
        self._state = self._state.detach()


base_mod.BaseDynamics = BaseDynamics
pkg = types.ModuleType("diffaero")
dyn_pkg = types.ModuleType("diffaero.dynamics")
pkg.dynamics = dyn_pkg
dyn_pkg.base_dynamics = base_mod
sys.modules.setdefault("diffaero", pkg)
sys.modules.setdefault("diffaero.dynamics", dyn_pkg)
sys.modules["diffaero.dynamics.base_dynamics"] = base_mod

import check_diffaero_gate  # noqa: E402

if __name__ == "__main__":
    check_diffaero_gate.main()
