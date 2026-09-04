// Identify which of the `unitree_hg` motor slots this R1 actually uses.
//
// The C++ port of tools/probe_lowstate.py. The Python SDK (`unitree_sdk2py`)
// is NOT installed on the robot and installing it needs network + a cyclonedds
// build; the C++ SDK already is installed, so this is the cheap path. It is
// also not throwaway: the W06 bridge has to link libunitree_sdk2 anyway, and
// this is its read side.
//
// READ-ONLY. Subscribes to LowState, never constructs or publishes a LowCmd --
// it cannot command a joint. Safe against a powered robot with the factory
// stack up.
//
//   ./probe_lowstate --iface eth10 --seconds 5     slot survey
//   ./probe_lowstate --iface eth10 --watch         move ONE joint, read its index
//   ./probe_lowstate --iface eth10 --map           guided pass over all 26 joints,
//                                                  writes joint_map.tsv

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using LowState = unitree_hg::msg::dds_::LowState_;

namespace {

std::mutex g_mutex;
LowState g_latest;
std::atomic<bool> g_have{false};
std::atomic<bool> g_stop{false};

void OnLowState(const void * message)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  g_latest = *static_cast<const LowState *>(message);
  g_have.store(true);
}

// The exact spelling and type of the less-central MotorState fields differs
// between SDK revisions (temperature is a scalar in some, a 2-array in others,
// and absent in the oldest). Detect rather than assume: a wrong guess here
// would be a compile error on the robot, one more round-trip over a link we
// do not have.
template <typename T, typename = void>
struct HasTemperature : std::false_type {};
template <typename T>
struct HasTemperature<T, std::void_t<decltype(std::declval<const T &>().temperature())>>
: std::true_type {};

template <typename V>
double FirstOf(const V & v)
{
  if constexpr (std::is_arithmetic_v<std::decay_t<V>>) {
    return static_cast<double>(v);
  } else {
    return v.empty() ? 0.0 : static_cast<double>(v[0]);
  }
}

template <typename M>
double TemperatureOf(const M & m)
{
  if constexpr (HasTemperature<M>::value) {
    return FirstOf(m.temperature());
  } else {
    return 0.0;
  }
}

// One sample, decoupled from the IDL type so the survey can hold a history
// without dragging 35 full MotorState structs around per frame.
struct Slot { double q, dq, tau, temp; int mode; };

std::vector<Slot> Snapshot(const LowState & s)
{
  std::vector<Slot> out;
  out.reserve(s.motor_state().size());
  for (const auto & m : s.motor_state()) {
    out.push_back({m.q(), m.dq(), m.tau_est(), TemperatureOf(m),
                   static_cast<int>(m.mode())});
  }
  return out;
}

// Isaac Lab's quat_rotate_inverse, for q = (w, x, y, z). Same formula the
// training side uses; cross-validated against a rotation-matrix derivation.
void QuatRotateInverse(const double q[4], const double v[3], double out[3])
{
  const double w = q[0];
  const double vec[3] = {q[1], q[2], q[3]};
  const double dot = vec[0] * v[0] + vec[1] * v[1] + vec[2] * v[2];
  const double cross[3] = {vec[1] * v[2] - vec[2] * v[1],
                           vec[2] * v[0] - vec[0] * v[2],
                           vec[0] * v[1] - vec[1] * v[0]};
  for (int i = 0; i < 3; ++i) {
    out[i] = v[i] * (2.0 * w * w - 1.0) - cross[i] * w * 2.0 + vec[i] * dot * 2.0;
  }
}

void ReportImu(const LowState & s)
{
  const auto & imu = s.imu_state();
  const auto & qa = imu.quaternion();
  const auto & g = imu.gyroscope();
  const auto & a = imu.accelerometer();

  std::printf("\n=== IMU ===\n");
  std::printf("  quaternion    %+.4f %+.4f %+.4f %+.4f\n", qa[0], qa[1], qa[2], qa[3]);
  std::printf("  gyroscope     %+.4f %+.4f %+.4f   <- policy obs `base_ang_vel`\n",
              g[0], g[1], g[2]);
  std::printf("  accelerometer %+.4f %+.4f %+.4f\n", a[0], a[1], a[2]);

  // Which ordering is it? Upright, projected gravity in the base frame must be
  // close to (0, 0, -1). Both orderings produce a unit vector and neither
  // errors, so only the robot can answer this.
  const double down[3] = {0.0, 0.0, -1.0};
  const double wxyz[4] = {qa[0], qa[1], qa[2], qa[3]};
  const double xyzw[4] = {qa[3], qa[0], qa[1], qa[2]};
  double r1[3], r2[3];
  QuatRotateInverse(wxyz, down, r1);
  QuatRotateInverse(xyzw, down, r2);
  std::printf("\n  projected gravity, assuming the array is:\n");
  std::printf("    (w,x,y,z) -> %+.4f %+.4f %+.4f\n", r1[0], r1[1], r1[2]);
  std::printf("    (x,y,z,w) -> %+.4f %+.4f %+.4f\n", r2[0], r2[1], r2[2]);
  std::printf("  With the robot UPRIGHT the correct ordering reads about (0, 0, -1).\n");
  std::printf("  That ordering is what the bridge must use. Do not guess it.\n");
}

void Survey(const std::vector<std::vector<Slot>> & samples)
{
  const std::size_t n = samples.front().size();
  std::printf("\n=== 槽位普查（%zu 帧, %zu 槽）===\n", samples.size(), n);
  std::printf("%4s %9s %9s %9s %6s %5s %9s  判定\n",
              "idx", "q", "dq", "tau", "temp", "mode", "q范围");

  std::vector<std::size_t> live;
  for (std::size_t i = 0; i < n; ++i) {
    double lo = samples.front()[i].q, hi = lo;
    bool nonzero = false;
    for (const auto & f : samples) {
      lo = std::min(lo, f[i].q);
      hi = std::max(hi, f[i].q);
      nonzero |= std::fabs(f[i].q) > 1e-9 || std::fabs(f[i].dq) > 1e-9 ||
                 std::fabs(f[i].tau) > 1e-9;
    }
    const Slot & last = samples.back()[i];
    const bool alive = nonzero || last.temp > 0.0 || last.mode != 0;
    if (alive) live.push_back(i);

    std::printf("%4zu %+9.4f %+9.4f %+9.4f %6.0f %5d %9.5f  %s\n",
                i, last.q, last.dq, last.tau, last.temp, last.mode, hi - lo,
                alive ? "LIVE" : "-");
  }

  std::printf("\n判定为 LIVE 的槽位（%zu 个）: ", live.size());
  for (std::size_t i : live) std::printf("%zu ", i);
  std::printf("\n我们的策略是 24 维动作 / 26 维关节观测。\n");
  if (live.size() == 26) {
    std::printf("  -> 26 个，与关节观测维度吻合。用 --watch 逐个确认语义。\n");
  } else if (live.size() == 24) {
    std::printf("  -> 24 个，与动作维度吻合（头部两关节可能不在 LowState 里）。\n");
  } else {
    std::printf("  -> %zu 个，与 24/26 都对不上。别急着写映射，先把这个搞清楚。\n",
                live.size());
  }
  std::printf("\n温度为 0、mode 为 0、读数恒为零的槽位是 G1 布局里 R1 没用上的保留位。\n");
}

void Watch()
{
  std::printf("\n=== 实时监视 ===\n");
  std::printf("用手慢慢掰动**一个**关节，读出跳动的那个索引。Ctrl-C 结束。\n\n");

  std::vector<Slot> base;
  while (!g_stop.load()) {
    std::vector<Slot> cur;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      cur = Snapshot(g_latest);
    }
    if (base.empty()) {
      base = cur;
      std::printf("已记录静止基准。现在开始掰。\n\n");
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    std::vector<std::pair<double, std::size_t>> moved;
    for (std::size_t i = 0; i < cur.size(); ++i) {
      moved.emplace_back(std::fabs(cur[i].q - base[i].q), i);
    }
    std::sort(moved.rbegin(), moved.rend());
    if (moved.front().first > 0.01) {
      std::string line;
      char buf[96];
      for (int k = 0; k < 5 && moved[k].first > 0.01; ++k) {
        const std::size_t i = moved[k].second;
        std::snprintf(buf, sizeof(buf), "idx %2zu: d%+.3f rad, dq %+.3f", i,
                      cur[i].q - base[i].q, cur[i].dq);
        if (!line.empty()) line += " | ";
        line += buf;
      }
      std::printf("\r%-110s", line.c_str());
      std::fflush(stdout);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::printf("\n结束。\n");
}


// ---------------------------------------------------------------- guided map

// One row of joints.tsv, generated by gen_joints.py from policy_interface.json.
struct JointRow { int art; std::string name; int act; };

std::vector<JointRow> LoadJoints(const std::string & path)
{
  std::vector<JointRow> rows;
  std::ifstream f(path);
  if (!f) return rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    JointRow r;
    if (ss >> r.art >> r.name >> r.act) rows.push_back(r);
  }
  return rows;
}

std::vector<Slot> Latest()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  return Snapshot(g_latest);
}

// The baseline is only meaningful if the robot is actually still. A robot whose
// factory controller is holding position twitches continuously, and a baseline
// taken mid-twitch makes every subsequent delta suspect.
bool WaitQuiet(std::vector<Slot> & base, double max_dq, int need, double timeout_s)
{
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<long>(timeout_s * 1000));
  int calm = 0;
  while (!g_stop.load() && std::chrono::steady_clock::now() < deadline) {
    const auto cur = Latest();
    double worst = 0.0;
    for (const auto & sl : cur) worst = std::max(worst, std::fabs(sl.dq));
    calm = (worst < max_dq) ? calm + 1 : 0;
    if (calm >= need) { base = cur; return true; }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  base = Latest();
  return false;
}

struct Hit { int slot = -1; double delta = 0, peak_dq = 0, runner = 0; int runner_slot = -1; };

// Wait until one slot has moved past `thresh` and stayed there. Sustained,
// because a single frame past the threshold can be a dropout or a bump of the
// frame rather than the joint.
Hit DetectMove(const std::vector<Slot> & base, double thresh, int need, double timeout_s)
{
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(static_cast<long>(timeout_s * 1000));
  Hit best;
  int sustained = 0;
  int last_winner = -1;
  double peak_dq = 0.0;

  while (!g_stop.load() && std::chrono::steady_clock::now() < deadline) {
    const auto cur = Latest();
    std::vector<std::pair<double, int>> d;
    for (std::size_t i = 0; i < cur.size(); ++i) {
      d.emplace_back(std::fabs(cur[i].q - base[i].q), static_cast<int>(i));
      peak_dq = std::max(peak_dq, std::fabs(cur[i].dq));
    }
    std::sort(d.rbegin(), d.rend());

    if (d[0].first > thresh) {
      sustained = (d[0].second == last_winner) ? sustained + 1 : 0;
      last_winner = d[0].second;
      if (sustained >= need) {
        best.slot = d[0].second;
        best.delta = cur[best.slot].q - base[best.slot].q;
        best.peak_dq = peak_dq;
        best.runner = d[1].first;
        best.runner_slot = d[1].second;
        return best;
      }
      std::printf("\r  检测到 idx %2d  d%+.3f rad  (保持中 %d/%d)      ",
                  d[0].second, cur[d[0].second].q - base[d[0].second].q,
                  sustained, need);
      std::fflush(stdout);
    } else {
      sustained = 0;
      last_winner = -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return best;
}

int MapMode(const std::string & joints_path, const std::string & out_path,
            double thresh)
{
  const auto rows = LoadJoints(joints_path);
  if (rows.empty()) {
    std::printf("[fail] 读不到 %s（用 gen_joints.py 生成）\n", joints_path.c_str());
    return 1;
  }

  std::printf("\n=== 引导式映射（%zu 个关节）===\n", rows.size());
  std::printf("对每个关节：回车开始 -> 慢慢掰开约 0.1-0.3 rad 并保持 -> 自动判定\n");
  std::printf("  s = 跳过此关节    q = 结束并写出已测部分\n");
  std::printf("⚠️  别掰到限位。移动幅度够大就行，不是越大越好。\n\n");

  std::ofstream out(out_path);
  out << "# probe_lowstate --map, measured on the robot.\n";
  out << "slot\tart_idx\tjoint_name\taction_idx\tdelta_rad\tpeak_dq\tnote\n";

  std::map<int, std::string> taken;
  std::size_t done = 0;

  for (std::size_t k = 0; k < rows.size() && !g_stop.load(); ++k) {
    const auto & r = rows[k];
    std::printf("\n[%zu/%zu] %s  (art %d, action %d)\n", k + 1, rows.size(),
                r.name.c_str(), r.art, r.act);
    std::printf("  回车开始 / s 跳过 / q 结束: ");
    std::fflush(stdout);

    std::string cmd;
    if (!std::getline(std::cin, cmd)) break;
    if (cmd == "q") break;
    if (cmd == "s") { std::printf("  跳过。\n"); continue; }

    std::vector<Slot> base;
    if (!WaitQuiet(base, 0.05, 10, 5.0)) {
      std::printf("  ⚠️ 机器人没完全静止，基准可能带噪声。继续。\n");
    }
    std::printf("  基准已记录。现在掰动 %s 并保持...\n", r.name.c_str());

    const Hit h = DetectMove(base, thresh, 8, 30.0);
    std::printf("\r%-70s\r", "");
    if (h.slot < 0) {
      std::printf("  超时，没检测到明显movement。跳过（可稍后重测）。\n");
      continue;
    }

    // Mechanical coupling is real: moving a knee drags the ankle a little.
    // A winner that barely beats the runner-up is not an identification.
    std::string note;
    if (h.runner > 0.3 * std::fabs(h.delta)) {
      note = "AMBIGUOUS";
      std::printf("  ⚠️ idx %d (d%+.3f) 与 idx %d (d%.3f) 幅度接近 —— 判定不可信。\n",
                  h.slot, h.delta, h.runner_slot, h.runner);
      std::printf("     幅度小一点、只动这一个关节，重测这条。\n");
    }
    if (taken.count(h.slot)) {
      note += note.empty() ? "DUPLICATE" : ",DUPLICATE";
      std::printf("  ⚠️ 槽位 %d 已经分配给 %s —— 两条必有一错。\n",
                  h.slot, taken[h.slot].c_str());
    } else {
      taken[h.slot] = r.name;
    }

    std::printf("  ✓ slot %2d  <-  %s   d%+.3f rad  peak dq %.3f  %s\n",
                h.slot, r.name.c_str(), h.delta, h.peak_dq,
                h.delta > 0 ? "(掰的方向 = q 正向)" : "(掰的方向 = q 负向)");
    out << h.slot << "\t" << r.art << "\t" << r.name << "\t" << r.act << "\t"
        << h.delta << "\t" << h.peak_dq << "\t" << note << "\n";
    out.flush();
    ++done;
  }

  std::printf("\n=== 完成 %zu/%zu，已写入 %s ===\n", done, rows.size(),
              out_path.c_str());
  if (done < rows.size()) {
    std::printf("还差 %zu 个。映射表不完整就不能写 bridge —— 缺的那几个"
                "会被静默地映射到错误的槽位。\n", rows.size() - done);
  }
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string iface = "eth10";
  std::string topic = "rt/lowstate";
  int domain = 0;
  double seconds = 5.0;
  bool watch = false;
  bool map_mode = false;
  double thresh = 0.05;
  std::string joints_path = "joints.tsv";
  std::string out_path = "joint_map.tsv";

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--iface") iface = next();
    else if (a == "--topic") topic = next();
    else if (a == "--domain") domain = std::atoi(next());
    else if (a == "--seconds") seconds = std::atof(next());
    else if (a == "--watch") watch = true;
    else if (a == "--map") map_mode = true;
    else if (a == "--joints") joints_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--thresh") thresh = std::atof(next());
    else { std::printf("unknown arg: %s\n", a.c_str()); return 2; }
  }

  std::signal(SIGINT, [](int) { g_stop.store(true); });

  std::printf("======================================================================\n");
  std::printf("只读探针：仅订阅 LowState，不发布任何指令，无法驱动关节。\n");
  std::printf("======================================================================\n");
  std::printf("iface=%s domain=%d topic=%s\n", iface.c_str(), domain, topic.c_str());

  unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
  unitree::robot::ChannelSubscriberPtr<LowState> sub(
    new unitree::robot::ChannelSubscriber<LowState>(topic));
  sub->InitChannel(OnLowState, 1);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!g_have.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!g_have.load()) {
    std::printf("\n[fail] 5 秒内没收到 %s。\n", topic.c_str());
    std::printf("  - 确认出厂运控在跑: ps aux | grep master_service\n");
    std::printf("  - 网口对不对: ip -brief addr   (阶段 2 实测 eth10)\n");
    std::printf("  - 换个话题名再试: --topic rt/lowstate_hg / lowstate\n");
    return 1;
  }

  {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::printf("\n已收到 %s。motor_state 槽位数: %zu\n", topic.c_str(),
                g_latest.motor_state().size());
    ReportImu(g_latest);
  }

  if (map_mode) return MapMode(joints_path, out_path, thresh);
  if (watch) { Watch(); return 0; }

  std::vector<std::vector<Slot>> samples;
  const auto end = std::chrono::steady_clock::now() +
                   std::chrono::milliseconds(static_cast<long>(seconds * 1000));
  while (std::chrono::steady_clock::now() < end && !g_stop.load()) {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      samples.push_back(Snapshot(g_latest));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (samples.empty()) { std::printf("[fail] 没采到样本\n"); return 1; }

  Survey(samples);
  std::printf("\n下一步: 加 --watch 逐个关节确认语义，把输出发回来写映射表。\n");
  return 0;
}
