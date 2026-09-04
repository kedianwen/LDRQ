// Plain C++ handle on the robot's DDS link. NO ROS types, NO Unitree types.
//
// This header exists because the two include universes cannot be mixed. Unitree
// installs CycloneDDS 0.10.2 under /usr/local/include/dds/, ROS foxy installs
// 0.7.0 under /opt/ros/foxy/include/dds/, and a target that sees both resolves
// each header from whichever prefix comes first -- yielding a compile against a
// blend of the two versions. It fails loudly (`ddsi_sertype does not name a
// type`), but only after the include order has already silently decided which
// half of each library you got.
//
// So: robot_io.cpp is compiled with /usr/local/include and no ROS at all, the
// ROS node is compiled with ROS and no /usr/local/include, and they meet only
// through the POD types below. See CMakeLists.txt, which enforces the split.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace r1_hw_bridge
{

inline constexpr std::size_t kSlots = 35;

// mode_pr: PR = the ankle's pitch/roll joint angles; AB = the two parallel
// actuators behind them. The policy outputs joint angles, so this must be PR.
// Sending AB feeds the ankles a completely different quantity and reports
// nothing.
inline constexpr uint8_t kModePR = 0;

struct MotorReading
{
  float q = 0.0f;
  float dq = 0.0f;
  float tau = 0.0f;
  float temperature = 0.0f;
  int mode = 0;
};

struct ImuReading
{
  std::array<float, 4> quat{};   // (w, x, y, z) -- established on the robot
  std::array<float, 3> gyro{};
  std::array<float, 3> accel{};
};

struct RobotState
{
  std::array<MotorReading, kSlots> motor{};
  ImuReading imu{};
  uint8_t mode_machine = 0;
};

struct MotorCommand
{
  float q = 0.0f;
  float dq = 0.0f;
  float tau = 0.0f;
  float kp = 0.0f;
  float kd = 0.0f;
  uint8_t mode = 0;   // 0 leaves the slot untouched
};

struct RobotIoConfig
{
  int domain = 0;
  std::string iface = "eth10";
  std::string state_topic = "rt/lowstate";
  std::string cmd_topic = "rt/lowcmd";
  bool check_state_crc = true;
  bool enable_output = false;
};

class RobotIo
{
public:
  explicit RobotIo(const RobotIoConfig & cfg);
  ~RobotIo();
  RobotIo(const RobotIo &) = delete;
  RobotIo & operator=(const RobotIo &) = delete;

  // Callback runs on the DDS thread. Frames failing the CRC check never reach it.
  void Start(std::function<void(const RobotState &)> on_state);

  // No-op when enable_output was false: the command is still built by the
  // caller, so the whole chain stays exercisable against a robot that cannot move.
  void Send(const std::array<MotorCommand, kSlots> & cmd, uint8_t mode_machine);

  std::size_t crc_failures() const;
  bool output_enabled() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace r1_hw_bridge
