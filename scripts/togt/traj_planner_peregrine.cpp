// Peregrine TOGT planner CLI — patched from Run-TOGT-Planner/traj_planner/traj_planner_togt.cpp
// (MIT, (c) KafuuChikai; TOGT-Planner MIT (c) 2024 FSC Lab). Changes for the VQ1 time-optimal
// bound (laptop-togt-bound-2026-06-10):
//   1. saveSegments(wpt, piecesPerSegment) instead of saveAllWaypoints(): emits the
//      waypoints/durations/timestamps YAML that scripts/togt/refine/trajectory.py consumes
//      (with planTOGT's 0 corridor midpoints, piecesPerSegment=1 makes the junctions exactly
//      the gate crossings).
//   2. argv[6] = that piecesPerSegment (pass 1 for planTOGT).
// NB: RacePlanner::plan() (the two-phase C++ init+refine) segfaults even on the upstream
// cpc/UZH inputs — it is an untested upstream code path (their tests only exercise planTOGT
// and planAOS), so this CLI stays on the validated planTOGT(); true time-optimality comes
// from the multiple-shooting refine downstream.
//
// Build: overwrite Run-TOGT-Planner/traj_planner/traj_planner_togt.cpp in the TOGT clone with
// this file and `cmake --build build -j` (see scripts/togt/README.md).
#include "drolib/race/race_track.hpp"
#include "drolib/race/race_params.hpp"
#include "drolib/race/race_planner.hpp"
#include <filesystem>
#include <iostream>

using namespace drolib;

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "Usage: " << argv[0]
              << " <config_path> <quad_name> <track_path> <traj_path> <wpt_path> <piecesPerSegment>"
              << std::endl;
    return 1;
  }

  fs::path config_path = argv[1];
  std::string quad_name = argv[2];
  std::string config_name = quad_name + "_setups.yaml";
  std::string track_path = argv[3];
  std::string traj_path = argv[4];
  std::string wpt_path = argv[5];
  const int piecesPerSegment = std::stoi(argv[6]);

  auto raceparams = std::make_shared<RaceParams>(config_path, config_name);
  auto raceplanner = std::make_shared<RacePlanner>(*raceparams);
  auto racetrack = std::make_shared<RaceTrack>(track_path);

  if (!raceplanner->planTOGT(racetrack)) {
    std::cerr << "Failed to plan trajectory." << std::endl;
    return 1;
  }

  TrajExtremum extremum = raceplanner->getExtremum();
  std::cout << extremum << std::endl;

  MincoSnapTrajectory traj = raceplanner->getTrajectory();
  if (!traj.save(traj_path)) {
    std::cerr << "No valid trajectory to save." << std::endl;
    return 1;
  }
  if (!traj.saveSegments(wpt_path, piecesPerSegment)) {
    std::cerr << "Failed to save segment waypoints." << std::endl;
    return 1;
  }

  return 0;
}
