// Peregrine TOGT driver as a gtest case (laptop-togt-bound-2026-06-10).
//
// The standalone `planners` CLI (Run-TOGT-Planner wrapper, and our patched variant) segfaults
// BEFORE main() in this build environment (static-init crash; upstream's own tests binary runs
// the identical planTOGT flow fine), so the driver lives inside the tests binary instead --
// the execution context that demonstrably works. Paths come from environment variables:
//
//   TOGT_CONFIG_DIR  parameter dir containing <quad>_setups.yaml
//   TOGT_QUAD        quad name (e.g. "peregrine")
//   TOGT_TRACK       track yaml
//   TOGT_TRAJ        output trajectory csv
//   TOGT_WPT         output waypoint yaml (saveSegments)
//   TOGT_PIECES      saveSegments piecesPerSegment (1 for planTOGT: junctions = gates)
//
// Install: copy into TOGT-Planner/tests/ and add to target_sources in tests/CMakeLists.txt,
// then `cmake --build build --target tests`. Run:
//   TOGT_CONFIG_DIR=... ./build/tests/tests --gtest_filter=PeregrineTOGT.plan
#include <gtest/gtest.h>
#include "drolib/race/race_track.hpp"
#include "drolib/race/race_params.hpp"
#include "drolib/race/race_planner.hpp"
#include <cstdlib>
#include <filesystem>
#include <string>
using namespace drolib;

static std::string envOr(const char* k, const char* dflt) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string(dflt);
}

TEST(PeregrineTOGT, plan) {
  const std::string config_dir = envOr("TOGT_CONFIG_DIR", "");
  const std::string quad = envOr("TOGT_QUAD", "peregrine");
  const std::string track = envOr("TOGT_TRACK", "");
  const std::string traj = envOr("TOGT_TRAJ", "/tmp/peregrine_togt_traj.csv");
  const std::string wpt = envOr("TOGT_WPT", "/tmp/peregrine_togt_wpt.yaml");
  const int pieces = std::stoi(envOr("TOGT_PIECES", "1"));
  ASSERT_FALSE(config_dir.empty()) << "set TOGT_CONFIG_DIR";
  ASSERT_FALSE(track.empty()) << "set TOGT_TRACK";

  auto raceparams = std::make_shared<RaceParams>(fs::path(config_dir), quad + "_setups.yaml");
  auto raceplanner = std::make_shared<RacePlanner>(*raceparams);
  auto racetrack = std::make_shared<RaceTrack>(fs::path(track));
  ASSERT_TRUE(raceplanner->planTOGT(racetrack));

  TrajExtremum extremum = raceplanner->getExtremum();
  std::cout << extremum << std::endl;

  MincoSnapTrajectory traj_obj = raceplanner->getTrajectory();
  EXPECT_TRUE(traj_obj.save(traj));
  EXPECT_TRUE(traj_obj.saveSegments(wpt, pieces));
}
