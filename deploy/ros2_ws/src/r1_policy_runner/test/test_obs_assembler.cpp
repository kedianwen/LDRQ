// The one piece of deployment logic that can be wrong without anything
// complaining: the observation history layout. Both the correct term-major
// layout and the plausible-but-wrong frame-major one produce a vector of
// exactly 425 floats, so only an explicit test separates them.

#include "r1_policy_runner/obs_assembler.hpp"

#include <gtest/gtest.h>

#include <vector>

using r1_policy_runner::ObsAssembler;
using r1_policy_runner::ObsTermSpec;

namespace
{
// Two terms, widths 2 and 3, three frames of history -- small enough to write
// the expected vector out by hand.
ObsAssembler MakeTiny()
{
  return ObsAssembler({{"a", 2}, {"b", 3}}, 3);
}
}  // namespace

TEST(ObsAssembler, ReportsGeometry)
{
  auto asm_ = MakeTiny();
  EXPECT_EQ(asm_.frame_dim(), 5u);
  EXPECT_EQ(asm_.history(), 3u);
  EXPECT_EQ(asm_.obs_dim(), 15u);
  EXPECT_FALSE(asm_.warm());
}

TEST(ObsAssembler, LayoutIsTermMajorOldestFirst)
{
  auto asm_ = MakeTiny();

  // frame k = {a0,a1, b0,b1,b2} tagged with the frame index in the tens digit.
  for (int k = 1; k <= 3; ++k) {
    const float f[5] = {
      10.f * k + 0, 10.f * k + 1,            // term a
      10.f * k + 2, 10.f * k + 3, 10.f * k + 4  // term b
    };
    asm_.Push(f);
  }
  ASSERT_TRUE(asm_.warm());

  std::vector<float> out(asm_.obs_dim());
  asm_.Fill(out.data());

  const std::vector<float> expected = {
    // term a, frames 1,2,3 (oldest -> newest)
    10, 11, 20, 21, 30, 31,
    // term b, frames 1,2,3
    12, 13, 14, 22, 23, 24, 32, 33, 34,
  };
  EXPECT_EQ(out, expected);

  // Guard against the frame-major mistake explicitly: if the implementation
  // ever emits five consecutive frames, this is what it would produce.
  const std::vector<float> frame_major = {
    10, 11, 12, 13, 14,
    20, 21, 22, 23, 24,
    30, 31, 32, 33, 34,
  };
  EXPECT_NE(out, frame_major);
}

TEST(ObsAssembler, RingDropsOldestFrame)
{
  auto asm_ = MakeTiny();
  for (int k = 1; k <= 4; ++k) {  // one more than the history depth
    const float f[5] = {10.f * k, 10.f * k + 1, 10.f * k + 2, 10.f * k + 3, 10.f * k + 4};
    asm_.Push(f);
  }

  std::vector<float> out(asm_.obs_dim());
  asm_.Fill(out.data());

  // Frame 1 is gone; term a now holds frames 2,3,4.
  EXPECT_FLOAT_EQ(out[0], 20.f);
  EXPECT_FLOAT_EQ(out[4], 40.f);
  // Term b starts at offset 2 * history = 6.
  EXPECT_FLOAT_EQ(out[6], 22.f);
  EXPECT_FLOAT_EQ(out[14], 44.f);
}

TEST(ObsAssembler, NotWarmUntilHistoryFilled)
{
  auto asm_ = MakeTiny();
  const float f[5] = {1, 2, 3, 4, 5};
  asm_.Push(f);
  EXPECT_FALSE(asm_.warm());
  asm_.Push(f);
  EXPECT_FALSE(asm_.warm());
  asm_.Push(f);
  EXPECT_TRUE(asm_.warm());

  asm_.Reset();
  EXPECT_FALSE(asm_.warm());
}

TEST(ObsAssembler, MatchesR1Geometry)
{
  // The real thing: 6 terms, 5 frames, 425 floats.
  ObsAssembler r1(
    {{"base_ang_vel", 3}, {"projected_gravity", 3}, {"velocity_commands", 3},
      {"joint_pos", 26}, {"joint_vel", 26}, {"actions", 24}}, 5);
  EXPECT_EQ(r1.frame_dim(), 85u);
  EXPECT_EQ(r1.obs_dim(), 425u);
}

TEST(ObsAssembler, RejectsDegenerateSpecs)
{
  EXPECT_THROW(ObsAssembler({{"a", 2}}, 0), std::invalid_argument);
  EXPECT_THROW(ObsAssembler({{"a", 0}}, 3), std::invalid_argument);
  EXPECT_THROW(ObsAssembler({}, 3), std::invalid_argument);
}
