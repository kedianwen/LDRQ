// The Unitree DDS half of the bridge.
//
// ⚠️ This translation unit must NEVER see a ROS header. It is compiled against
// /usr/local/include, which carries CycloneDDS 0.10.2; ROS foxy ships 0.7.0
// under the same relative paths. See robot_io.hpp and CMakeLists.txt.

#include "r1_hw_bridge/robot_io.hpp"

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <atomic>
#include <cstdio>
#include <type_traits>

namespace r1_hw_bridge
{
namespace
{

using LowState = unitree_hg::msg::dds_::LowState_;
using LowCmd = unitree_hg::msg::dds_::LowCmd_;

// Unitree's CRC. NOT a standard CRC-32: it feeds whole 32-bit words and XORs
// the polynomial on a set data bit rather than into the register's low end. It
// has to match the firmware bit for bit -- a "cleaner" implementation that
// produces different output means every command is silently rejected.
uint32_t Crc32Core(const uint32_t * ptr, uint32_t len)
{
  uint32_t crc = 0xFFFFFFFF;
  const uint32_t polynomial = 0x04C11DB7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t xbit = 1u << 31;
    const uint32_t data = ptr[i];
    for (uint32_t bit = 0; bit < 32; ++bit) {
      crc = (crc & 0x80000000u) ? ((crc << 1) ^ polynomial) : (crc << 1);
      if (data & xbit) {crc ^= polynomial;}
      xbit >>= 1;
    }
  }
  return crc;
}

// The vendor convention: hash the whole struct except its trailing crc word.
template <typename T>
uint32_t StructCrc(const T & msg)
{
  return Crc32Core(reinterpret_cast<const uint32_t *>(&msg),
                   static_cast<uint32_t>((sizeof(T) >> 2) - 1));
}

// The spelling and type of the less-central MotorState fields differ between
// SDK revisions (temperature is a scalar in some, a 2-array in others, absent
// in the oldest). Detect rather than assume: a wrong guess is a compile error
// on the robot, one more round trip over a link we do not have.
template <typename T, typename = void>
struct HasTemperature : std::false_type {};
template <typename T>
struct HasTemperature<T, std::void_t<decltype(std::declval<const T &>().temperature())>>
: std::true_type {};

template <typename V>
float FirstOf(const V & v)
{
  if constexpr (std::is_arithmetic_v<std::decay_t<V>>) {
    return static_cast<float>(v);
  } else {
    return v.empty() ? 0.0f : static_cast<float>(v[0]);
  }
}

template <typename M>
float TemperatureOf(const M & m)
{
  if constexpr (HasTemperature<M>::value) {
    return FirstOf(m.temperature());
  } else {
    return 0.0f;
  }
}

}  // namespace

struct RobotIo::Impl
{
  RobotIoConfig cfg;
  std::function<void(const RobotState &)> on_state;
  unitree::robot::ChannelSubscriberPtr<LowState> sub;
  unitree::robot::ChannelPublisherPtr<LowCmd> pub;
  std::atomic<std::size_t> crc_failures{0};
  RobotState decoded;   // touched only on the DDS thread

  void OnMessage(const void * message)
  {
    const auto & s = *static_cast<const LowState *>(message);
    if (cfg.check_state_crc && s.crc() != StructCrc(s)) {
      // Never silent. If the struct layout ever disagrees with the firmware's,
      // every frame fails and the bridge would otherwise look like a dead
      // network. Rerun with check_state_crc:=false to test that hypothesis.
      const std::size_t n = crc_failures.fetch_add(1);
      if (n % 500 == 0) {
        std::fprintf(stderr,
          "[robot_io] LowState CRC mismatch (%zu so far). If EVERY frame fails, "
          "relaunch with check_state_crc:=false to test that hypothesis.\n", n + 1);
      }
      return;
    }

    for (std::size_t i = 0; i < kSlots && i < s.motor_state().size(); ++i) {
      const auto & m = s.motor_state()[i];
      decoded.motor[i] = {m.q(), m.dq(), m.tau_est(), TemperatureOf(m),
                          static_cast<int>(m.mode())};
    }
    const auto & imu = s.imu_state();
    for (int i = 0; i < 4; ++i) {decoded.imu.quat[i] = imu.quaternion()[i];}
    for (int i = 0; i < 3; ++i) {
      decoded.imu.gyro[i] = imu.gyroscope()[i];
      decoded.imu.accel[i] = imu.accelerometer()[i];
    }
    decoded.mode_machine = s.mode_machine();

    if (on_state) {on_state(decoded);}
  }
};

RobotIo::RobotIo(const RobotIoConfig & cfg)
: impl_(new Impl)
{
  impl_->cfg = cfg;
}

RobotIo::~RobotIo() = default;

void RobotIo::Start(std::function<void(const RobotState &)> on_state)
{
  impl_->on_state = std::move(on_state);
  unitree::robot::ChannelFactory::Instance()->Init(impl_->cfg.domain, impl_->cfg.iface);
  impl_->sub.reset(new unitree::robot::ChannelSubscriber<LowState>(impl_->cfg.state_topic));
  impl_->sub->InitChannel([this](const void * m) {impl_->OnMessage(m);}, 1);

  if (impl_->cfg.enable_output) {
    impl_->pub.reset(new unitree::robot::ChannelPublisher<LowCmd>(impl_->cfg.cmd_topic));
    impl_->pub->InitChannel();
  }
}

void RobotIo::Send(const std::array<MotorCommand, kSlots> & cmd, uint8_t mode_machine)
{
  if (!impl_->pub) {return;}

  // Value-initialised: any slot the caller left at mode 0 stays fully zeroed.
  // Enabling a motor the robot does not have is not something to try blind.
  LowCmd out{};
  out.mode_pr() = kModePR;
  out.mode_machine() = mode_machine;
  for (std::size_t i = 0; i < kSlots && i < out.motor_cmd().size(); ++i) {
    auto & m = out.motor_cmd()[i];
    m.mode() = cmd[i].mode;
    m.q() = cmd[i].q;
    m.dq() = cmd[i].dq;
    m.tau() = cmd[i].tau;
    m.kp() = cmd[i].kp;
    m.kd() = cmd[i].kd;
  }
  out.crc() = StructCrc(out);
  impl_->pub->Write(out);
}

std::size_t RobotIo::crc_failures() const {return impl_->crc_failures.load();}
bool RobotIo::output_enabled() const {return impl_->pub != nullptr;}

}  // namespace r1_hw_bridge
