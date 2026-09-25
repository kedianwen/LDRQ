// Is anything else writing rt/lowcmd?
//
// This answers the question the preflight's process-name check only guesses at.
// Developer mode does NOT stop /unitree/module/master_service/master_service --
// measured on the robot 2026-09-25, it keeps running with a live pid after the
// handheld switch. So "is the factory process running" is the wrong question and
// always answers yes. The question that matters is whether anything is PUBLISHING
// on the topic we publish on, because two writers at 500 Hz overwriting each
// other is the actual defect -- and its symptoms (torso sway, joint grinding,
// high-frequency tremor, stance not held) all read as a sim2real gap. W06 lost a
// session to exactly that.
//
// READ-ONLY. It subscribes and counts. It has no publisher and cannot move the
// robot; that is the whole reason it is safe to run before the stack comes up.
//
//   probe_lowcmd --seconds 5
//
// Run it with OUR STACK DOWN. With the bridge running it will of course see
// traffic -- ours -- and that tells you nothing. The tool checks for a running
// bridge and refuses rather than reporting a result that would be misread.

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>

#include <dirent.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

using LowCmd = unitree_hg::msg::dds_::LowCmd_;

namespace
{
std::atomic<bool> g_stop{false};
std::atomic<unsigned long> g_count{0};
std::atomic<unsigned long> g_first_ns{0};
std::atomic<unsigned long> g_last_ns{0};

unsigned long NowNs()
{
  return static_cast<unsigned long>(
    std::chrono::steady_clock::now().time_since_epoch().count());
}

// The SDK hands the handler a `const void *` to the decoded message. We only
// need the ARRIVAL, not the contents: any traffic at all is the answer, and
// copying a command frame at 500 Hz would add cost to a measurement. So the
// pointer is deliberately unused.
void OnLowCmd(const void *)
{
  const unsigned long t = NowNs();
  unsigned long expected = 0;
  g_first_ns.compare_exchange_strong(expected, t);
  g_last_ns.store(t);
  g_count.fetch_add(1);
}

/// Is one of our own nodes up? If so the measurement is meaningless, because the
/// traffic it would see is ours.
///
/// Scans /proc rather than shelling out to pgrep. `system("pgrep -f X")` cannot
/// work here: the /bin/sh it spawns carries X in its own argv, so pgrep matches
/// that shell and the answer is always "found", whatever is or is not running.
/// Verified on the dev box -- it returned 0 with no such process anywhere. A
/// check that always fires would make this tool always refuse.
///
/// /proc/<pid>/cmdline is NUL-separated, so the scan joins the arguments with
/// spaces before searching: the node name can appear in argv[0] as a full path
/// or as a later argument, and matching only comm would miss it anyway (comm is
/// truncated to 15 characters, which is why `pgrep r1_hw_bridge_node` finds
/// nothing while it runs).
bool OurStackRunning(std::string * who)
{
  static const char * kOurs[] = {"r1_hw_bridge_node", "r1_policy_node"};
  const pid_t self = getpid();
  const pid_t parent = getppid();

  DIR * d = opendir("/proc");
  if (!d) {return false;}          // cannot tell; do not block on it
  bool found = false;
  while (dirent * e = readdir(d)) {
    const int pid = std::atoi(e->d_name);
    if (pid <= 0 || pid == self || pid == parent) {continue;}
    std::string path = "/proc/" + std::string(e->d_name) + "/cmdline";
    std::ifstream in(path, std::ios::binary);
    if (!in) {continue;}
    std::string raw((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    for (char & c : raw) {if (c == '\0') {c = ' ';}}
    for (const char * pat : kOurs) {
      if (raw.find(pat) != std::string::npos) {
        *who = std::string(pat) + " (pid " + e->d_name + ")";
        found = true;
        break;
      }
    }
    if (found) {break;}
  }
  closedir(d);
  return found;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string iface = "eth10";
  std::string topic = "rt/lowcmd";
  int domain = 0;
  double seconds = 5.0;
  bool force = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--iface") {iface = next();} else if (a == "--topic") {
      topic = next();
    } else if (a == "--domain") {domain = std::atoi(next());} else if (a == "--seconds") {
      seconds = std::atof(next());
    } else if (a == "--force") {
      force = true;
    } else {
      std::printf(
        "usage: %s [--seconds 5] [--iface eth10] [--domain 0] [--topic rt/lowcmd]\n"
        "       [--force]  measure even with our own stack running (the result is\n"
        "                  then our own traffic and answers nothing)\n", argv[0]);
      return 2;
    }
  }

  std::signal(SIGINT, [](int) {g_stop.store(true);});

  std::printf("======================================================================\n");
  std::printf("READ-ONLY. Subscribes to %s and counts. No publisher; cannot\n", topic.c_str());
  std::printf("move the robot.\n");
  std::printf("======================================================================\n");

  std::string who;
  if (OurStackRunning(&who)) {
    std::printf("\n[refusing] %s is running, so any traffic on %s is OURS.\n",
                who.c_str(), topic.c_str());
    std::printf("Stop our stack first (pkill -f r1_hw_bridge_node; pkill -f "
                "r1_policy_node), then re-run.\n");
    if (!force) {return 2;}
    std::printf("--force given; continuing, but the number below is not evidence "
                "of a second writer.\n");
  }

  std::printf("iface=%s domain=%d topic=%s  listening %.1fs...\n",
              iface.c_str(), domain, topic.c_str(), seconds);

  unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
  unitree::robot::ChannelSubscriberPtr<LowCmd> sub(
    new unitree::robot::ChannelSubscriber<LowCmd>(topic));
  sub->InitChannel(OnLowCmd, 1);

  const auto end = std::chrono::steady_clock::now() +
    std::chrono::milliseconds(static_cast<long>(seconds * 1000));
  while (std::chrono::steady_clock::now() < end && !g_stop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  const unsigned long n = g_count.load();
  std::printf("\nmessages in %.1fs: %lu\n", seconds, n);

  if (n == 0) {
    std::printf("\nVERDICT: nothing is writing %s.\n", topic.c_str());
    std::printf("No second writer. Safe to start the stack.\n");
    std::printf("\n(One caveat this cannot rule out: a writer that only starts\n");
    std::printf("when it sees commands appear. If the first seconds of the stack\n");
    std::printf("show sway or grinding, suspect that and re-check with the stack up\n");
    std::printf("using --force, comparing the rate against our own 500 Hz.)\n");
    return 0;
  }

  const unsigned long f = g_first_ns.load(), l = g_last_ns.load();
  const double span = (l > f) ? static_cast<double>(l - f) / 1e9 : 0.0;
  if (span > 0.0) {
    std::printf("rate: %.1f Hz over the %.2fs it was actually publishing\n",
                static_cast<double>(n - 1) / span, span);
  }
  std::printf("\nVERDICT: SOMETHING ELSE IS WRITING %s.\n", topic.c_str());
  std::printf("Do NOT start the stack. Two writers at 500 Hz overwrite each other\n");
  std::printf("and the symptoms -- torso sway, joint grinding, tremor, stance not\n");
  std::printf("held -- all look like a sim2real gap rather than a contention bug.\n");
  std::printf("\nDeveloper mode does not stop master_service from RUNNING (it keeps a\n");
  std::printf("live pid), so a running process is expected. Traffic is not. If this\n");
  std::printf("fires while the handheld says developer mode, the mode did not take.\n");
  return 1;
}
