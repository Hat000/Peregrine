# TOGT time-optimal bound pipeline (laptop-togt-bound-2026-06-10)

Computes the **time-optimal lap bound** for the VQ1 course on our measured plant and the
committed **reference line** (`rl/reference_line_vq1.json`), using
[TOGT-Planner](https://github.com/FSC-Lab/TOGT-Planner) (MIT, FSC Lab) + its
multiple-shooting refinement. Results + caveats:
`handoff/laptop-togt-bound-2026-06-10/WRITEUP.md`.

## Pipeline

```
gen_cases.py          course json + plant params -> per-case TOGT config dirs
   |                  (handoff/laptop-togt-bound-2026-06-10/cases/<case>/)
run_cases.sh  [WSL]   per case:
   |                    1. TOGT phase: planTOGT via the gtest driver (test_peregrine.cpp)
   |                       -> togt_traj.csv (warm start) + togt_wpt.yaml (gate crossings)
   |                    2. refine phase: refine/refine_peregrine.py (CasADi/IPOPT
   |                       multiple shooting, 13-state quad, drag included)
   |                       -> refined_traj.csv  = the bound trajectory
analyze.py            lap @ gate-5 plane crossing, in-plane misses, what-binds -> table
export_reference.py   winning case -> rl/reference_line_vq1.json (NED, schema v1)
../twin_track_reference.py   reality check: track the line through the faithful twin
```

## Environment (one-time, WSL Ubuntu)

```bash
mkdir -p ~/peregrine_togt && cd ~/peregrine_togt
git clone --depth 1 https://github.com/FSC-Lab/TOGT-Planner.git
git clone --depth 1 https://github.com/Tencent/rapidjson.git
# header-only Eigen (no sudo): cmake -S eigen-3.4.0 -B eigen-build
#   -DCMAKE_INSTALL_PREFIX=$HOME/peregrine_togt/local && cmake --install eigen-build
python3 -m venv venv && ./venv/bin/pip install casadi numpy pyyaml scipy
cd TOGT-Planner
# CMake edits (see WRITEUP "build notes"): replace the RapidJSON FetchContent with a
# direct include of ~/peregrine_togt/rapidjson/include (its cmake config breaks
# re-configures), then:
cp <repo>/scripts/togt/test_peregrine.cpp tests/
#   + add test_peregrine.cpp to target_sources in tests/CMakeLists.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/peregrine_togt/local \
      -DRAPIDJSON_BUILD_TESTS=OFF -DRAPIDJSON_BUILD_DOC=OFF -DRAPIDJSON_BUILD_EXAMPLES=OFF
cmake --build build -j12 --target tests
```

## Hard-won environment facts (cost hours; do not rediscover)

- **CRLF kills the C++ YAML parser.** TOGT's hand-rolled loader reads `0.05\r` as garbage
  -> garbage params -> segfault (or clean-looking solve failures). Everything written for
  it must be LF (`gen_cases._write_lf`).
- **The standalone `planners` CLI segfaults before `main()`** in this build env (static-init
  crash) while the tests binary runs the identical flow fine -> the TOGT phase runs as a
  gtest case (`test_peregrine.cpp`, env-var driven).
- **RaceParams needs ABSOLUTE paths**; relative config dirs fail to load (then crash).
- **`RacePlanner::plan()` (two-phase C++ refine) segfaults even on upstream's own inputs** —
  untested upstream code path; stay on `planTOGT()`.
- **`minThr` 0.05 makes the init L-BFGS fail outright** (near-zero-thrust flatness
  singularity); 0.1 N solves. The init is brittle off the nominal parameter point in
  general -> run_cases.sh walks a speedGuess/dynamicConstCheck retry ladder.
- **Driving WSL from Windows:** git-bash MSYS path conversion mangles `/mnt/...` args and
  env-var assignments; long inline `wsl bash -lc '...'` strings get corrupted. Invoke WSL
  from PowerShell and put ALL logic in script FILES (LF), passing only short absolute
  paths. WSL `/tmp` is tmpfs and instances recycle between calls — write logs to
  persistent paths.

## Provenance / licenses

- `refine/{optimization,trajectory}.py` — verbatim copies (MIT, (c) 2024 FSC Lab).
- `refine/quadrotor.py` — copy + the `linear_drag` term (marked `# PEREGRINE`).
- `refine/refine_peregrine.py` — our CLI driver (replaces their `togt_refine.py`).
- `test_peregrine.cpp` — our gtest driver (adapted from their `tests/test_togt.cpp`).
- `traj_planner_peregrine.cpp` — kept for reference; superseded by the gtest driver.
