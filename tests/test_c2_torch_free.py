"""G0 (C2 gauntlet, BLUEPRINT §3.4 G0) -- the estimator chain imports TORCH-FREE.

The whole offline gauntlet (G2/G3) and the deploy estimator run in the torch-free .venv. The C2
estimator chain (RewindKF + gate-relative fix + navigator + the obs-wiring shim) MUST import without
pulling torch -- the obs builder's torch dependency is reached only LAZILY, inside estimator_obs.

Runs in a SUBPROCESS so a torch import by another test in this process cannot mask a regression.
[C2-ESTIMATOR-CHAIN 2026-06-13]
"""
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_SNIPPET = (
    "import sys; sys.path.insert(0, r'{src}');"
    "import racer.kf_rewind, racer.localization, racer.navigator, racer.estimator_obs,"
    " racer.state_estimator, racer.contracts;"
    "assert 'torch' not in sys.modules, 'estimator chain pulled torch at import';"
    "print('C2_TORCH_FREE_OK')"
).format(src=str(_SRC))


def test_estimator_chain_imports_torch_free():
    proc = subprocess.run([sys.executable, "-c", _SNIPPET], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "C2_TORCH_FREE_OK" in proc.stdout
