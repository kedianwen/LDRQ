// Invariants on the generated joint map.
//
// These would all have passed on a hand-written map that was simply wrong, so
// they are not a substitute for the on-robot measurements in
// deploy/interface/joint_map_r1.md. What they catch is a broken generator or a
// stale header: a duplicate slot means two joints share one motor, a
// mis-sized action map means the policy's output is applied to the wrong
// joints -- both silent at runtime.
#include <gtest/gtest.h>

#include "r1_hw_bridge/joint_map.hpp"

#include <set>

using namespace r1_hw_bridge;

TEST(JointMap, SlotsAreUniqueAndInRange)
{
  std::set<int> seen;
  for (std::size_t j = 0; j < kNumJoints; ++j) {
    EXPECT_GE(kJointSlot[j], 0);
    EXPECT_LT(static_cast<std::size_t>(kJointSlot[j]), kNumSlots);
    EXPECT_TRUE(seen.insert(kJointSlot[j]).second)
      << "slot " << kJointSlot[j] << " used twice (joint " << j << ")";
  }
  EXPECT_EQ(seen.size(), kNumJoints);
}

TEST(JointMap, SlotSetMatchesTheMeasuredLiveSlots)
{
  // probe_lowstate --seconds 5 found exactly these 26 slots alive on the robot;
  // the other nine read zero on every channel with temperature 0 and mode 0.
  const std::set<int> live = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30};
  std::set<int> mapped(kJointSlot.begin(), kJointSlot.end());
  EXPECT_EQ(mapped, live);
}

TEST(JointMap, ActionMapIsInjectiveAndLeavesTwoJointsUndriven)
{
  ASSERT_EQ(kNumActions, 24u);
  std::set<int> seen;
  for (std::size_t a = 0; a < kNumActions; ++a) {
    EXPECT_GE(kActionToArt[a], 0);
    EXPECT_LT(static_cast<std::size_t>(kActionToArt[a]), kNumJoints);
    EXPECT_TRUE(seen.insert(kActionToArt[a]).second);
  }
  EXPECT_EQ(kNumJoints - seen.size(), 2u) << "exactly the two head joints";
}

TEST(JointMap, ActionOrderIsNotTheArticulationOrder)
{
  // The trap this project was warned about: index i in the action vector and
  // index i in joint_pos are different joints. If this ever becomes the
  // identity, someone has regenerated the map from the wrong source.
  bool differs = false;
  for (std::size_t a = 0; a < kNumActions; ++a) {
    if (static_cast<std::size_t>(kActionToArt[a]) != a) {differs = true;}
  }
  EXPECT_TRUE(differs);
}

TEST(JointMap, DefaultsCoverEveryJoint)
{
  EXPECT_EQ(kDefaultPos.size(), kNumJoints);
  // waist_roll (articulation 2) is a policy action but rests at zero; the two
  // hip pitches do not. A table of all zeros would mean the defaults were lost.
  bool any_nonzero = false;
  for (float d : kDefaultPos) {
    if (d != 0.0f) {any_nonzero = true;}
  }
  EXPECT_TRUE(any_nonzero);
}
