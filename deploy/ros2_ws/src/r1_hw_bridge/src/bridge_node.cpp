// R1 hardware bridge: robot DDS <-> generic ROS 2 topics.
//
//   rt/lowstate (DDS)  ->  ~/obs            85 floats, one frame per control step
//   ~/joint_target     ->  rt/lowcmd (DDS)  24 targets -> 35 motor slots
//
// The policy node speaks Float32MultiArray and knows nothing of Unitree, which
// is what lets it be unit-tested, replayed from a rosbag and driven from
// simulation with no SDK present.
//
// ⚠️ This translation unit must NEVER include a Unitree header. The SDK ships
// CycloneDDS 0.10.2 under /usr/local/include/dds/, ROS foxy ships 0.7.0 under
// /opt/ros/foxy/include/dds/, and a target that sees both compiles against a
// blend of the two. All robot I/O goes through robot_io.hpp, whose
// implementation is compiled without ROS on the include path.
//
// Two rates, on purpose:
//   * observation/policy at control_rate_hz (50) -- the rate the policy trained at
//   * command at cmd_rate_hz (500) -- what the motor bus expects; the per-joint
//     PD loop itself runs in the motor firmware, so nothing here integrates.
// Targets are re-sent unchanged between policy ticks rather than interpolated:
// in training the target was a 20 ms step, and smoothing it here would be a
// sim-to-real difference introduced by the deployment code.
//
// SAFETY: `enable_output` defaults to FALSE. Nothing is written to rt/lowcmd
// until it is explicitly turned on, so building and running this node cannot
// move the robot by accident.

#include "r1_hw_bridge/joint_map.hpp"
#include "r1_hw_bridge/robot_io.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace r1_hw_bridge
{

using std::chrono::steady_clock;
using std::chrono::duration;

static_assert(kNumSlots == kSlots, "joint_map.hpp and robot_io.hpp disagree on slot count");

enum class State { WaitingState, WaitingPolicy, Running, Degraded };

const char * StateName(State s)
{
  switch (s) {
    case State::WaitingState: return "WAITING_LOWSTATE";
    case State::WaitingPolicy: return "WAITING_POLICY";
    case State::Running: return "RUNNING";
    case State::Degraded: return "DEGRADED";
  }
  return "?";
}

// Isaac Lab's quat_rotate_inverse for q = (w, x, y, z). The component order was
// established on the robot (probe_lowstate: upright reads (0,0,-1) only under
// this ordering, and the accelerometer agrees to 0.04 m/s^2). Do not "fix" it.
inline void QuatRotateInverse(const std::array<float, 4> & q, const float v[3], float out[3])
{
  const float w = q[0];
  const float vec[3] = {q[1], q[2], q[3]};
  const float dot = vec[0] * v[0] + vec[1] * v[1] + vec[2] * v[2];
  const float cross[3] = {vec[1] * v[2] - vec[2] * v[1],
                          vec[2] * v[0] - vec[0] * v[2],
                          vec[0] * v[1] - vec[1] * v[0]};
  for (int i = 0; i < 3; ++i) {
    out[i] = v[i] * (2.0f * w * w - 1.0f) - cross[i] * w * 2.0f + vec[i] * dot * 2.0f;
  }
}

class BridgeNode : public rclcpp::Node
{
public:
  BridgeNode()
  : rclcpp::Node("r1_hw_bridge")
  {
    RobotIoConfig io_cfg;
    io_cfg.iface = declare_parameter<std::string>("iface", "eth10");
    io_cfg.domain = static_cast<int>(declare_parameter<int64_t>("dds_domain", 0));
    io_cfg.state_topic = declare_parameter<std::string>("lowstate_topic", "rt/lowstate");
    io_cfg.cmd_topic = declare_parameter<std::string>("lowcmd_topic", "rt/lowcmd");
    io_cfg.check_state_crc = declare_parameter<bool>("check_state_crc", true);
    io_cfg.enable_output = declare_parameter<bool>("enable_output", false);

    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 50.0);
    cmd_rate_hz_ = declare_parameter<double>("cmd_rate_hz", 500.0);
    // Gains are PER JOINT and come from the training config via kKp/kKd; these
    // two only scale them. The first version of this node had one global kp/kd
    // for all 24 joints, which is not a simplification of six actuator groups
    // spanning kp 20..100 -- it is a different controller. Measured on the robot
    // 2026-09-04: legs with no perceptible damping (their group wants kp 100,
    // kd 2, they got 10 and 1) and a head oscillating at high frequency (its
    // group wants kd 1, it got 3).
    //
    // Scale 1.0 IS the trained controller. Ramp with kp_scale, not with an
    // absolute number, so the endpoint of the ramp is known rather than guessed.
    kp_scale_ = declare_parameter<double>("kp_scale", 1.0);
    kd_scale_ = declare_parameter<double>("kd_scale", 1.0);
    state_timeout_ms_ = declare_parameter<double>("lowstate_timeout_ms", 100.0);
    target_timeout_ms_ = declare_parameter<double>("target_timeout_ms", 60.0);
    min_rate_hz_ = declare_parameter<double>("min_control_rate_hz", 45.0);
    auto_recover_ = declare_parameter<bool>("auto_recover", false);
    cmd_vel_timeout_ms_ = declare_parameter<double>("cmd_vel_timeout_ms", 500.0);

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    obs_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/obs", qos);
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "~/status", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local());
    reset_pub_ = create_publisher<std_msgs::msg::Empty>(
      "~/policy_reset", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    // Raw state, republished for rosbag and `ros2 topic echo`. This is the
    // observability that made unitree_ros2 look attractive; publishing it here
    // gets the same thing without a second DDS stack.
    dbg_jpos_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_pos", qos);
    dbg_jvel_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_vel", qos);
    dbg_imu_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/imu", qos);
    dbg_cmd_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/cmd_debug", qos);

    target_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "joint_target", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnTarget(*m);});
    action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "action", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnAction(*m);});
    cmdvel_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "~/cmd_vel", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnCmdVel(*m);});
    resume_sub_ = create_subscription<std_msgs::msg::Bool>(
      "~/resume", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      [this](std_msgs::msg::Bool::ConstSharedPtr m) {OnResume(*m);});

    frame_.assign(kFrameDim, 0.0f);
    last_action_.assign(kNumActions, 0.0f);
    target_.assign(kNumActions, 0.0f);
    cmd_vel_ = {0.0f, 0.0f, 0.0f};

    io_ = std::make_unique<RobotIo>(io_cfg);
    io_->Start([this](const RobotState & s) {OnRobotState(s);});

    if (io_->output_enabled()) {
      RCLCPP_WARN(get_logger(), "enable_output=true -- THIS NODE WILL COMMAND THE ROBOT");
    } else {
      RCLCPP_INFO(get_logger(),
                  "enable_output=false -- commands computed and published to "
                  "~/cmd_debug only, nothing written to %s", io_cfg.cmd_topic.c_str());
    }

    obs_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / control_rate_hz_), [this]() {OnObsTick();});
    status_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {OnStatusTick();});
    cmd_thread_ = std::thread([this]() {CmdLoop();});

    RCLCPP_INFO(get_logger(),
                "bridge up: iface=%s domain=%d obs %.0f Hz, cmd %.0f Hz, %zu joints",
                io_cfg.iface.c_str(), io_cfg.domain, control_rate_hz_, cmd_rate_hz_,
                kNumJoints);

    // The robot link above and the ROS side below are separate DDS participants.
    // Print the ROS side's isolation settings: joining the factory stack's
    // domain kills discovery with std::bad_alloc, and the symptom ("topic does
    // not appear to be published") points at the publisher rather than at the
    // domain. Having it in the log makes that diagnosis a one-line check.
    const char * dom = std::getenv("ROS_DOMAIN_ID");
    const char * loc = std::getenv("ROS_LOCALHOST_ONLY");
    // Print the gains that will actually be sent. A scale is easy to pass and
    // easy to forget; the numbers below are what reaches the motors.
    RCLCPP_INFO(get_logger(),
                "gains: kp_scale=%.3f kd_scale=%.3f -> hip kp=%.1f kd=%.1f | "
                "ankle kp=%.1f | head kp=%.1f kd=%.1f  (1.0 = as trained)",
                kp_scale_, kd_scale_,
                kKp[0] * kp_scale_, kKd[0] * kd_scale_,
                kKp[16] * kp_scale_, kKp[8] * kp_scale_, kKd[8] * kd_scale_);
    if (kp_scale_ > 1.0 || kd_scale_ > 1.0) {
      RCLCPP_WARN(get_logger(),
                  "kp_scale/kd_scale above 1.0 exceeds the trained gains -- the "
                  "policy has never seen this plant");
    }

    RCLCPP_INFO(get_logger(), "ROS side: ROS_DOMAIN_ID=%s ROS_LOCALHOST_ONLY=%s rmw=%s",
                dom ? dom : "(unset -> 0)", loc ? loc : "(unset -> 0)",
                rmw_get_implementation_identifier());
    if (!dom || !loc || std::string(loc) != "1") {
      RCLCPP_WARN(get_logger(),
        "ROS side is NOT isolated. On this robot that usually means other ROS "
        "processes will die in discovery (std::bad_alloc) and see this node's "
        "topics as unpublished. `source env.sh` sets ROS_DOMAIN_ID=99 and "
        "ROS_LOCALHOST_ONLY=1 -- every terminal needs the same values.");
    }
  }

  ~BridgeNode() override
  {
    running_.store(false);
    if (cmd_thread_.joinable()) {cmd_thread_.join();}
  }

private:
  static constexpr std::size_t kFrameDim = 3 + 3 + 3 + kNumJoints + kNumJoints + kNumActions;

  // The generated header comes from whichever training run dump_interface.py
  // was pointed at. Regenerating it from a run with a different action space
  // would silently change the frame width; the policy node would then reject
  // every frame at runtime. Fail at compile time instead.
  static_assert(kNumJoints == 26, "joint_map.hpp regenerated from a different run");
  static_assert(kNumActions == 24, "joint_map.hpp regenerated from a different run");
  static_assert(kFrameDim == 85, "frame width no longer matches the policy");

  // ---------------------------------------------------------- robot ingress
  void OnRobotState(const RobotState & s)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_ = s;
    have_state_ = true;
    last_state_time_ = steady_clock::now();
  }

  // ------------------------------------------------------------ ROS ingress
  void OnTarget(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != kNumActions) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "joint_target has %zu values, expected %zu",
                            m.data.size(), kNumActions);
      return;
    }
    for (const float v : m.data) {
      if (!std::isfinite(v)) {
        RCLCPP_ERROR(get_logger(), "joint_target contains NaN/inf -- degrading");
        Degrade("non-finite joint_target");
        return;
      }
    }
    const auto now = steady_clock::now();
    RecordLoopLag(now);
    std::lock_guard<std::mutex> lock(target_mutex_);
    std::copy(m.data.begin(), m.data.end(), target_.begin());
    last_target_time_ = now;
    have_target_ = true;
  }

  // Setpoint lag: from publishing an observation to having the setpoint it
  // produced. This is exactly the quantity training randomized -- W04's
  // DelayedJointPositionAction lags the setpoint by {0,1} control steps, i.e.
  // {0, 20} ms at 50 Hz (tasks/r1_flat/mdp.py) -- so it is the number to
  // compare against, and it is directly measurable here rather than inferred
  // by cross-correlating command against motion. The actuator's own response
  // lag is a separate quantity and is NOT what that randomization models.
  //
  // Assumes the policy emits one target per obs frame, true in steady state at
  // 50 Hz. A dropped frame shows up as a sample one control period too long,
  // which is visible as a bimodal spread rather than a quietly wrong median.
  void RecordLoopLag(steady_clock::time_point now)
  {
    const int64_t pub_ns = obs_pub_ns_.load();
    if (pub_ns == 0) {return;}
    const double ms =
      duration<double, std::milli>(now - steady_clock::time_point(steady_clock::duration(pub_ns)))
      .count();
    // Negative or absurd means the pairing is not 1:1 (startup, or a burst);
    // drop rather than poison the statistic.
    if (ms < 0.0 || ms > 1000.0) {return;}
    std::lock_guard<std::mutex> lock(lag_mutex_);
    if (lag_samples_.size() < kLagSamples) {
      lag_samples_.push_back(ms);
    } else {
      lag_samples_[lag_next_] = ms;
      lag_next_ = (lag_next_ + 1) % kLagSamples;
    }
  }

  // p50/p95/max over the window, or all zeros if nothing has been paired yet.
  std::array<double, 3> LoopLag()
  {
    std::vector<double> v;
    {
      std::lock_guard<std::mutex> lock(lag_mutex_);
      v = lag_samples_;
    }
    if (v.empty()) {return {0.0, 0.0, 0.0};}
    std::sort(v.begin(), v.end());
    auto at = [&v](double q) {
        return v[std::min(v.size() - 1,
          static_cast<std::size_t>(q * static_cast<double>(v.size())))];
      };
    return {at(0.50), at(0.95), v.back()};
  }

  void OnAction(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != kNumActions) {return;}
    std::lock_guard<std::mutex> lock(target_mutex_);
    std::copy(m.data.begin(), m.data.end(), last_action_.begin());
  }

  void OnCmdVel(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != 3) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "~/cmd_vel expects 3 floats [vx, vy, wz]");
      return;
    }
    std::lock_guard<std::mutex> lock(target_mutex_);
    cmd_vel_ = {m.data[0], m.data[1], m.data[2]};
    last_cmdvel_time_ = steady_clock::now();
    have_cmdvel_ = true;
  }

  void OnResume(const std_msgs::msg::Bool & m)
  {
    if (!m.data) {return;}
    if (state_.load() != State::Degraded) {
      RCLCPP_INFO(get_logger(), "~/resume ignored: not degraded (%s)",
                  StateName(state_.load()));
      return;
    }
    // Recovery is explicit and always re-warms: the policy's 5-frame history now
    // straddles a gap, which is out of the training distribution. Clearing it and
    // refilling is the only correct restart.
    RCLCPP_WARN(get_logger(), "resuming: clearing policy history, re-warming");
    reset_pub_->publish(std_msgs::msg::Empty());
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      have_target_ = false;
    }
    state_.store(State::WaitingPolicy);
    PublishStatus();
  }

  // ------------------------------------------------------------ observation
  void OnObsTick()
  {
    const auto now = steady_clock::now();
    RobotState s;
    bool ok;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ok = have_state_ &&
        duration<double, std::milli>(now - last_state_time_).count() < state_timeout_ms_;
      s = latest_;
    }
    if (!ok) {
      if (state_.load() != State::Degraded && state_.load() != State::WaitingState) {
        Degrade("lowstate stale");
      }
      return;
    }
    if (state_.load() == State::WaitingState) {
      state_.store(State::WaitingPolicy);
      PublishStatus();
    }

    const float down[3] = {0.0f, 0.0f, -1.0f};
    float grav[3];
    QuatRotateInverse(s.imu.quat, down, grav);

    std::size_t k = 0;
    for (int i = 0; i < 3; ++i) {frame_[k++] = s.imu.gyro[i];}
    for (int i = 0; i < 3; ++i) {frame_[k++] = grav[i];}
    {
      // A teleop that dies must not leave the robot walking on its last command.
      std::lock_guard<std::mutex> lock(target_mutex_);
      const bool cmd_fresh = have_cmdvel_ &&
        duration<double, std::milli>(now - last_cmdvel_time_).count() < cmd_vel_timeout_ms_;
      if (!cmd_fresh && have_cmdvel_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "~/cmd_vel stale -- commanding zero velocity");
      }
      for (int i = 0; i < 3; ++i) {frame_[k++] = cmd_fresh ? cmd_vel_[i] : 0.0f;}
    }
    // joint_pos is RELATIVE to the default pose; joint_vel is absolute. Getting
    // this backwards produces a plausible-looking observation and a policy that
    // has never seen anything like it.
    const std::size_t jpos0 = k;
    for (std::size_t j = 0; j < kNumJoints; ++j) {
      frame_[k++] = s.motor[kJointSlot[j]].q - kDefaultPos[j];
    }
    const std::size_t jvel0 = k;
    for (std::size_t j = 0; j < kNumJoints; ++j) {
      frame_[k++] = s.motor[kJointSlot[j]].dq;
    }
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      for (std::size_t a = 0; a < kNumActions; ++a) {frame_[k++] = last_action_[a];}
    }

    obs_msg_.data.assign(frame_.begin(), frame_.end());
    obs_pub_->publish(obs_msg_);
    // Stamped so OnTarget can measure the loop lag this obs frame incurs. This
    // runs on the SDK's DDS thread while OnTarget runs on the ROS executor, so
    // the handoff is an atomic, not the target_mutex_.
    obs_pub_ns_.store(steady_clock::now().time_since_epoch().count());

    dbg_jpos_pub_->publish(Slice(jpos0, kNumJoints));
    dbg_jvel_pub_->publish(Slice(jvel0, kNumJoints));
    std_msgs::msg::Float32MultiArray imu_msg;
    imu_msg.data = {s.imu.quat[0], s.imu.quat[1], s.imu.quat[2], s.imu.quat[3],
                    s.imu.gyro[0], s.imu.gyro[1], s.imu.gyro[2],
                    grav[0], grav[1], grav[2]};
    dbg_imu_pub_->publish(imu_msg);

    TrackRate(now);
  }

  std_msgs::msg::Float32MultiArray Slice(std::size_t off, std::size_t n) const
  {
    std_msgs::msg::Float32MultiArray m;
    m.data.assign(frame_.begin() + off, frame_.begin() + off + n);
    return m;
  }

  void TrackRate(steady_clock::time_point now)
  {
    // Start the first window at the first frame, not at construction. DDS
    // discovery takes ~100-400 ms, and counting that dead time as missed
    // frames made every single start-up log "observation rate 44.6 Hz below
    // 45.0 Hz (1/3)". A degrade counter that cries wolf on every boot is worse
    // than no counter: it trains you to read past the one that matters.
    if (!rate_primed_) {
      rate_primed_ = true;
      rate_window_ = now;
      obs_count_ = 0;
      return;
    }
    ++obs_count_;
    const double dt = duration<double>(now - rate_window_).count();
    if (dt < 1.0) {return;}
    const double hz = obs_count_ / dt;
    obs_count_ = 0;
    rate_window_ = now;
    obs_hz_.store(hz);
    if (hz < min_rate_hz_ && state_.load() == State::Running) {
      if (++slow_windows_ >= 3) {
        Degrade("control rate below threshold");
      } else {
        RCLCPP_WARN(get_logger(), "observation rate %.1f Hz below %.1f Hz (%d/3)",
                    hz, min_rate_hz_, slow_windows_);
      }
    } else if (hz >= min_rate_hz_) {
      slow_windows_ = 0;
    }
  }

  // ---------------------------------------------------------------- command
  std::string Reason()
  {
    std::lock_guard<std::mutex> lock(reason_mutex_);
    return degrade_reason_;
  }

  void Degrade(const std::string & why)
  {
    // exchange, not check-then-set: the observation timer and the command
    // thread can trip this in the same instant.
    if (state_.exchange(State::Degraded) == State::Degraded) {return;}
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      degrade_reason_ = why;
    }
    RCLCPP_ERROR(get_logger(), "DEGRADED: %s -- holding damping until ~/resume", why.c_str());
    PublishStatus();
  }

  // Runs at cmd_rate_hz on its own thread so a slow inference tick or a busy
  // ROS executor cannot stall the motor bus.
  void CmdLoop()
  {
    const auto period = std::chrono::duration_cast<steady_clock::duration>(
      std::chrono::duration<double>(1.0 / cmd_rate_hz_));
    auto next = steady_clock::now();

    while (running_.load()) {
      next += period;
      const auto now = steady_clock::now();

      bool state_fresh;
      RobotState s;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_fresh = have_state_ &&
          duration<double, std::milli>(now - last_state_time_).count() < state_timeout_ms_;
        s = latest_;
      }

      bool target_fresh = false;
      std::vector<float> target;
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_fresh = have_target_ &&
          duration<double, std::milli>(now - last_target_time_).count() < target_timeout_ms_;
        target = target_;
      }

      if (state_.load() == State::WaitingPolicy && target_fresh) {
        state_.store(State::Running);
        PublishStatus();
      } else if (state_.load() == State::Running && !target_fresh) {
        Degrade("joint_target stale");
      }

      if (state_fresh) {
        SendCommand(s, target, state_.load() == State::Running && target_fresh);
      }

      if (++cmd_window_count_ >= static_cast<uint64_t>(cmd_rate_hz_)) {
        const double dt = duration<double>(now - cmd_window_).count();
        if (dt > 0.0) {cmd_hz_.store(cmd_window_count_ / dt);}
        cmd_window_count_ = 0;
        cmd_window_ = now;
      }

      std::this_thread::sleep_until(next);
      // A long stall must not turn into a burst of catch-up commands.
      if (next < steady_clock::now() - period) {next = steady_clock::now();}
    }
  }

  // `drive` false => damping: zero stiffness, hold nothing, just resist motion.
  // With no R1 client in the SDK (/usr/local/include/unitree/robot has a2, b2,
  // g1, go2, h1 -- no r1) there is no vendor damping/e-stop call to delegate to,
  // so this is the degrade behaviour, implemented here.
  void SendCommand(const RobotState & s, const std::vector<float> & target, bool drive)
  {
    // Slots left at mode 0 are the nine G1 positions R1 has no motor for
    // (14, 20, 21, 27, 28, 31-34). Enabling a motor that is not there is not
    // something to try on a first run.
    std::array<MotorCommand, kSlots> cmd{};

    for (std::size_t j = 0; j < kNumJoints; ++j) {
      auto & m = cmd[kJointSlot[j]];
      m.mode = 1;
      m.q = s.motor[kJointSlot[j]].q;   // undriven: hold where it is
      // Damping only, at the joint's own trained kd. A global damping constant
      // is either negligible on a hip or oscillatory on the head.
      m.kd = static_cast<float>(kKd[j] * kd_scale_);
    }

    if (drive) {
      for (std::size_t a = 0; a < kNumActions; ++a) {
        const std::size_t j = kActionToArt[a];
        auto & m = cmd[kJointSlot[j]];
        m.q = target[a];
        m.kp = static_cast<float>(kKp[j] * kp_scale_);
        m.kd = static_cast<float>(kKd[j] * kd_scale_);
      }
      // The head is not actuated by the policy and was fixed during training,
      // so hold it at the default pose rather than leaving it to flop.
      for (std::size_t j = 0; j < kNumJoints; ++j) {
        if (!IsActuated(j)) {
          auto & m = cmd[kJointSlot[j]];
          m.q = kDefaultPos[j];
          m.kp = static_cast<float>(kKp[j] * kp_scale_);
          m.kd = static_cast<float>(kKd[j] * kd_scale_);
        }
      }
    }

    io_->Send(cmd, s.mode_machine);

    if (++cmd_count_ % static_cast<uint64_t>(cmd_rate_hz_ / 10.0 + 1) == 0) {
      std_msgs::msg::Float32MultiArray dbg;
      dbg.data.reserve(kNumJoints);
      for (std::size_t j = 0; j < kNumJoints; ++j) {
        dbg.data.push_back(cmd[kJointSlot[j]].q);
      }
      dbg_cmd_pub_->publish(dbg);
    }
  }

  static bool IsActuated(std::size_t art)
  {
    for (std::size_t a = 0; a < kNumActions; ++a) {
      if (static_cast<std::size_t>(kActionToArt[a]) == art) {return true;}
    }
    return false;
  }

  // Rates are reported by the node itself rather than left to `ros2 topic hz`:
  // the data topics are best-effort and foxy's CLI subscribes reliable by
  // default, so the CLI shows an empty topic whether or not anything is wrong.
  void OnStatusTick()
  {
    PublishStatus();
    if (++status_ticks_ % 5 != 0) {return;}
    const auto lag = LoopLag();
    const double step_ms = 1000.0 / control_rate_hz_;
    RCLCPP_INFO(get_logger(),
      "%s | obs %.1f Hz (want %.0f) | cmd %.1f Hz (want %.0f) | output=%s | "
      "kp_scale=%.2f | crc_fail=%zu | setpoint lag p50=%.1f p95=%.1f max=%.1f ms "
      "(%.2f steps; trained 0-%.0f)",
      StateName(state_.load()), obs_hz_.load(), control_rate_hz_,
      cmd_hz_.load(), cmd_rate_hz_,
      io_->output_enabled() ? "ON" : "off", kp_scale_, io_->crc_failures(),
      lag[0], lag[1], lag[2], lag[0] / step_ms, step_ms);
    // Training drew the lag from {0, 1} control steps. Past one step the
    // deployed loop is outside the distribution the policy was trained on, and
    // that is a W07 gap-mitigation input, not a crash -- so say it, once every
    // status cycle, rather than degrading on it.
    if (lag[1] > step_ms) {
      RCLCPP_WARN(get_logger(),
        "setpoint lag p95=%.1f ms exceeds the trained range (0-%.0f ms = 0-1 "
        "control steps): the policy is running outside its trained lag "
        "distribution", lag[1], step_ms);
    }
  }

  void PublishStatus()
  {
    std_msgs::msg::String m;
    m.data = std::string(StateName(state_.load())) +
      " output=" + (io_->output_enabled() ? "ON" : "off") +
      " crc_fail=" + std::to_string(io_->crc_failures()) +
      (state_.load() == State::Degraded ? " reason=" + Reason() : "");
    status_pub_->publish(m);
    if (auto_recover_ && state_.load() == State::Degraded) {
      RCLCPP_WARN_ONCE(get_logger(),
                       "auto_recover=true: degraded state will NOT self-clear; "
                       "publish true to ~/resume. The parameter exists only to "
                       "make its absence explicit.");
    }
  }

  // parameters
  double control_rate_hz_ = 50.0, cmd_rate_hz_ = 500.0;
  double kp_scale_ = 1.0, kd_scale_ = 1.0;
  double state_timeout_ms_ = 100.0, target_timeout_ms_ = 60.0, min_rate_hz_ = 45.0;
  double cmd_vel_timeout_ms_ = 500.0;
  bool auto_recover_ = false;

  std::unique_ptr<RobotIo> io_;

  std::mutex state_mutex_;
  RobotState latest_;
  bool have_state_ = false;
  steady_clock::time_point last_state_time_;

  std::mutex target_mutex_;
  std::vector<float> target_, last_action_;
  std::array<float, 3> cmd_vel_{};
  bool have_target_ = false, have_cmdvel_ = false;
  steady_clock::time_point last_target_time_, last_cmdvel_time_;

  std::vector<float> frame_;
  std_msgs::msg::Float32MultiArray obs_msg_;
  // Read and written from both the ROS executor and the command thread.
  std::atomic<State> state_{State::WaitingState};
  std::mutex reason_mutex_;
  std::string degrade_reason_;
  std::atomic<bool> running_{true};
  std::thread cmd_thread_;
  uint64_t cmd_count_ = 0, obs_count_ = 0, cmd_window_count_ = 0, status_ticks_ = 0;
  int slow_windows_ = 0;
  bool rate_primed_ = false;
  static constexpr std::size_t kLagSamples = 512;
  std::atomic<int64_t> obs_pub_ns_{0};
  std::mutex lag_mutex_;
  std::vector<double> lag_samples_;
  std::size_t lag_next_ = 0;
  steady_clock::time_point rate_window_ = steady_clock::now();
  steady_clock::time_point cmd_window_ = steady_clock::now();
  std::atomic<double> obs_hz_{0.0}, cmd_hz_{0.0};

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obs_pub_,
    dbg_jpos_pub_, dbg_jvel_pub_, dbg_imu_pub_, dbg_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
    target_sub_, action_sub_, cmdvel_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr resume_sub_;
  rclcpp::TimerBase::SharedPtr obs_timer_, status_timer_;
};

}  // namespace r1_hw_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<r1_hw_bridge::BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
