// Numerical parity between the trained PyTorch policy and the TensorRT engine,
// plus a latency profile of the engine on this host.
//
// The fixture is produced by tools/make_fixture.py, which runs the exported
// TorchScript policy on N observation vectors and records both sides. Passing
// this is what lets us claim the deployed engine *is* the trained policy --
// without it, "the node runs at 50Hz" says nothing about what it computes.
//
//   r1_parity_check --plan policy.plan --fixture parity_fixture.bin [--tol 1e-3]
//
// Two modes, and they answer different questions:
//
//   default            engine vs the TRAINED POLICY. A gate: pass or fail.
//   --baseline <plan>  engine vs ANOTHER ENGINE, same inputs. Not a gate --
//                      INT8 is supposed to differ from FP16. What the W08
//                      acceptance wants from this mode is the magnitude and the
//                      STRUCTURE of the difference, so pair it with --per-dim.
//
// Latency is reported two ways when --duty-hz is given, because on a Jetson they
// differ by ~8x: a compact loop keeps the GPU in a high clock state, while the
// real control loop calls infer() once per 20 ms and the GPU falls back to a low
// power state between calls. Only the paced number describes the robot.

#include "r1_policy_runner/trt_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

// Layout: "R1FX" | u32 version | u32 n | u32 in_dim | u32 out_dim
//         | f32 inputs[n*in_dim] | f32 outputs[n*out_dim]
struct Fixture
{
  uint32_t n = 0, in_dim = 0, out_dim = 0;
  std::vector<float> inputs, outputs;
};

bool LoadFixture(const std::string & path, Fixture * fx, std::string * err)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {*err = "cannot open " + path; return false;}

  char magic[4];
  uint32_t version = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  if (std::string(magic, 4) != "R1FX") {*err = "bad magic in " + path; return false;}
  if (version != 1) {*err = "unsupported fixture version"; return false;}

  in.read(reinterpret_cast<char *>(&fx->n), 4);
  in.read(reinterpret_cast<char *>(&fx->in_dim), 4);
  in.read(reinterpret_cast<char *>(&fx->out_dim), 4);
  if (!in) {*err = "truncated header"; return false;}

  fx->inputs.resize(static_cast<std::size_t>(fx->n) * fx->in_dim);
  fx->outputs.resize(static_cast<std::size_t>(fx->n) * fx->out_dim);
  in.read(reinterpret_cast<char *>(fx->inputs.data()),
    static_cast<std::streamsize>(fx->inputs.size() * sizeof(float)));
  in.read(reinterpret_cast<char *>(fx->outputs.data()),
    static_cast<std::streamsize>(fx->outputs.size() * sizeof(float)));
  if (!in) {*err = "truncated payload"; return false;}
  return true;
}

double Percentile(const std::vector<double> & sorted, double p)
{
  if (sorted.empty()) {return 0.0;}
  const auto idx = static_cast<std::size_t>(p * (sorted.size() - 1));
  return sorted[idx];
}

/// action index -> joint name, from tools/probe_cpp/joints.tsv
/// ("art_idx \t joint_name \t action_idx", action_idx -1 = not policy-actuated).
///
/// Optional on purpose. Without it the per-dim table is still correct, just
/// indexed by number; hard-coding the mapping here would put a second copy of
/// the joint order in the tree, and a stale second copy is worse than no copy.
std::map<int, std::string> LoadActionNames(const std::string & path)
{
  std::map<int, std::string> out;
  std::ifstream in(path);
  if (!in) {
    std::cerr << "[warn] --names " << path << " unreadable; per-dim table will be "
              << "indexed by number only\n";
    return out;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::istringstream ls(line);
    std::string art, name, act;
    if (!std::getline(ls, art, '\t')) {continue;}
    if (!std::getline(ls, name, '\t')) {continue;}
    if (!std::getline(ls, act, '\t')) {continue;}
    try {
      const int a = std::stoi(act);
      if (a >= 0) {out[a] = name;}
    } catch (const std::exception &) {continue;}
  }
  return out;
}

/// Runs every fixture input through @p policy and returns the outputs.
bool RunAll(
  r1_policy_runner::TrtPolicy * policy, const Fixture & fx,
  std::vector<float> * out, std::string * err)
{
  out->resize(static_cast<std::size_t>(fx.n) * fx.out_dim);
  for (uint32_t s = 0; s < fx.n; ++s) {
    if (!policy->Infer(
        fx.inputs.data() + static_cast<std::size_t>(s) * fx.in_dim,
        out->data() + static_cast<std::size_t>(s) * fx.out_dim))
    {
      *err = "inference failed on sample " + std::to_string(s);
      return false;
    }
  }
  return true;
}

/// Times infer(), either back-to-back or paced at @p duty_hz.
///
/// The pacing sleep is deliberately outside the timed region: what we want is
/// the cost of one inference given that the GPU has been idle for ~20 ms, not
/// the cost of the sleep.
std::vector<double> TimeInfer(
  r1_policy_runner::TrtPolicy * policy, const Fixture & fx, int iters, double duty_hz)
{
  std::vector<float> scratch(fx.out_dim);
  std::vector<double> us;
  us.reserve(static_cast<std::size_t>(iters));
  const auto period = std::chrono::duration<double>(duty_hz > 0.0 ? 1.0 / duty_hz : 0.0);
  auto next = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (duty_hz > 0.0) {
      next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
      std::this_thread::sleep_until(next);
    }
    const float * obs = fx.inputs.data() +
      static_cast<std::size_t>(i % fx.n) * fx.in_dim;
    const auto t0 = std::chrono::steady_clock::now();
    policy->Infer(obs, scratch.data());
    us.push_back(
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(us.begin(), us.end());
  return us;
}

void PrintLatency(const char * label, const std::vector<double> & us, int iters)
{
  std::cout << std::fixed << std::setprecision(1)
            << label << " (us over " << iters << "): "
            << "p50=" << Percentile(us, 0.50)
            << "  p95=" << Percentile(us, 0.95)
            << "  p99=" << Percentile(us, 0.99)
            << "  max=" << (us.empty() ? 0.0 : us.back()) << "\n";
}

void Usage(const char * argv0)
{
  std::cerr
    << "usage: " << argv0 << " --plan <p.plan> --fixture <f.bin> [--tol 1e-3]\n"
    << "         [--baseline <b.plan>] [--per-dim] [--names joints.tsv]\n"
    << "         [--latency-iters 2000] [--duty-hz 50]\n\n"
    << "  --baseline <b.plan>  compare against another ENGINE's outputs on the same\n"
    << "                       inputs instead of the fixture's PyTorch reference.\n"
    << "                       Engine-vs-engine has no pass line and always exits 0:\n"
    << "                       INT8 is MEANT to differ from FP16. Use it with\n"
    << "                       --per-dim; the hard no-regression gate is the\n"
    << "                       closed-loop metric in sim, not this number.\n"
    << "  --per-dim            per-action-dimension error table. The question it\n"
    << "                       answers is whether quantisation error is spread over\n"
    << "                       all 24 actions or piled onto a few joints -- those\n"
    << "                       have very different consequences on a biped.\n"
    << "  --names <tsv>        tools/probe_cpp/joints.tsv, to label the table with\n"
    << "                       joint names instead of bare action indices.\n"
    << "  --duty-hz <hz>       ALSO time inference paced at this rate. On a Jetson\n"
    << "                       the compact-loop number understates the real control\n"
    << "                       loop by ~8x, because the GPU drops to a low power\n"
    << "                       state between 20 ms-apart calls. Pass 50 to get the\n"
    << "                       number that describes the robot.\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string plan, fixture_path, baseline_plan, names_path;
  double tol = 1e-3;
  int latency_iters = 2000;
  double duty_hz = 0.0;
  bool per_dim = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
        if (i + 1 >= argc) {std::cerr << "missing value for " << a << "\n"; std::exit(2);}
        return argv[++i];
      };
    if (a == "--plan") {plan = next();} else if (a == "--fixture") {
      fixture_path = next();
    } else if (a == "--tol") {tol = std::stod(next());} else if (a == "--latency-iters") {
      latency_iters = std::stoi(next());
    } else if (a == "--baseline") {
      baseline_plan = next();
    } else if (a == "--names") {
      names_path = next();
    } else if (a == "--duty-hz") {
      duty_hz = std::stod(next());
    } else if (a == "--per-dim") {
      per_dim = true;
    } else if (a == "-h" || a == "--help") {Usage(argv[0]); return 0;} else {
      Usage(argv[0]);
      return 2;
    }
  }
  if (plan.empty() || fixture_path.empty()) {
    std::cerr << "error: --plan and --fixture are required\n";
    return 2;
  }

  Fixture fx;
  std::string err;
  if (!LoadFixture(fixture_path, &fx, &err)) {
    std::cerr << "fixture: " << err << "\n";
    return 1;
  }

  std::unique_ptr<r1_policy_runner::TrtPolicy> policy;
  try {
    policy = std::make_unique<r1_policy_runner::TrtPolicy>(plan);
  } catch (const std::exception & e) {
    std::cerr << "engine: " << e.what() << "\n";
    return 1;
  }

  if (static_cast<uint32_t>(policy->input_dim()) != fx.in_dim ||
    static_cast<uint32_t>(policy->output_dim()) != fx.out_dim)
  {
    std::cerr << "shape mismatch: engine " << policy->input_dim() << "->"
              << policy->output_dim() << ", fixture " << fx.in_dim << "->"
              << fx.out_dim << "\n";
    return 1;
  }

  std::cout << "engine " << policy->input_dim() << " -> " << policy->output_dim()
            << " | fixture " << fx.n << " samples\n"
            << "plan " << r1_policy_runner::PlanFingerprint(plan) << "\n";

  // -- pick the reference ----------------------------------------------------
  // Either the fixture's PyTorch outputs (a gate) or another engine's outputs on
  // the same inputs (a comparison). Both land in `ref` so the rest is common.
  const float * ref = fx.outputs.data();
  std::vector<float> baseline_out;
  const bool vs_engine = !baseline_plan.empty();
  if (vs_engine) {
    // Built and torn down before the engine under test is timed, so the two do
    // not share the GPU during the latency section.
    try {
      r1_policy_runner::TrtPolicy base(baseline_plan);
      if (base.input_dim() != policy->input_dim() ||
        base.output_dim() != policy->output_dim())
      {
        std::cerr << "baseline shape differs from the engine under test\n";
        return 1;
      }
      std::cout << "baseline " << r1_policy_runner::PlanFingerprint(baseline_plan) << "\n";
      if (!RunAll(&base, fx, &baseline_out, &err)) {
        std::cerr << "baseline: " << err << "\n";
        return 1;
      }
    } catch (const std::exception & e) {
      std::cerr << "baseline engine: " << e.what() << "\n";
      return 1;
    }
    ref = baseline_out.data();
    std::cout
      << "\nMODE: engine vs engine. The reference is the BASELINE PLAN, not the\n"
      << "trained policy, so this is not a parity gate and there is no pass line.\n"
      << "A non-zero difference is the expected result of quantisation. What is\n"
      << "being reported is its size and its shape.\n";
  }

  // -- error statistics ------------------------------------------------------
  std::vector<float> got(fx.out_dim);
  double max_abs = 0.0, sum_abs = 0.0;
  double max_rel = 0.0;
  uint32_t worst_sample = 0, worst_elem = 0;
  std::vector<double> dim_max(fx.out_dim, 0.0), dim_sum(fx.out_dim, 0.0);

  for (uint32_t s = 0; s < fx.n; ++s) {
    if (!policy->Infer(fx.inputs.data() + static_cast<std::size_t>(s) * fx.in_dim, got.data())) {
      std::cerr << "inference failed on sample " << s << "\n";
      return 1;
    }
    if (s == 0) {
      // Printed unconditionally: when parity fails, the first thing worth
      // knowing is whether the engine is slightly off or computing something
      // else entirely, and an aggregate max cannot tell those apart.
      std::cout << std::fixed << std::setprecision(5) << "  sample0 trt =";
      for (uint32_t j = 0; j < std::min<uint32_t>(6, fx.out_dim); ++j) {std::cout << " " << got[j];}
      std::cout << "\n  sample0 ref =";
      for (uint32_t j = 0; j < std::min<uint32_t>(6, fx.out_dim); ++j) {
        std::cout << " " << ref[j];
      }
      std::cout << "\n";
    }
    for (uint32_t j = 0; j < fx.out_dim; ++j) {
      const double r = ref[static_cast<std::size_t>(s) * fx.out_dim + j];
      const double d = std::fabs(got[j] - r);
      sum_abs += d;
      dim_sum[j] += d;
      if (d > dim_max[j]) {dim_max[j] = d;}
      if (d > max_abs) {max_abs = d; worst_sample = s; worst_elem = j;}
      const double denom = std::max(1e-6, std::fabs(r));
      max_rel = std::max(max_rel, d / denom);
    }
  }
  const double mean_abs = sum_abs / (static_cast<double>(fx.n) * fx.out_dim);

  std::cout << std::scientific << std::setprecision(3)
            << (vs_engine ? "difference: max_abs=" : "parity: max_abs=") << max_abs
            << "  mean_abs=" << mean_abs
            << "  max_rel=" << max_rel
            << "  (worst at sample " << worst_sample << ", action[" << worst_elem << "])\n";

  // -- per-dimension structure ----------------------------------------------
  if (per_dim) {
    const auto names = names_path.empty() ?
      std::map<int, std::string>{} : LoadActionNames(names_path);
    // Sorted by mean error: a biped tolerates a wrist error and does not
    // tolerate the same error on an ankle, so which dimensions carry the error
    // matters more than the aggregate does.
    std::vector<uint32_t> order(fx.out_dim);
    for (uint32_t j = 0; j < fx.out_dim; ++j) {order[j] = j;}
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return dim_sum[a] > dim_sum[b];
      });

    std::cout << "\n=== per-action error (sorted by mean, worst first) ===\n";
    std::cout << "  " << std::left << std::setw(6) << "act"
              << std::setw(30) << "joint" << std::right
              << std::setw(12) << "mean_abs" << std::setw(12) << "max_abs"
              << std::setw(9) << "share\n";
    for (const uint32_t j : order) {
      const double mean_j = dim_sum[j] / fx.n;
      const auto it = names.find(static_cast<int>(j));
      std::cout << "  " << std::left << std::setw(6) << j
                << std::setw(30) << (it == names.end() ? "-" : it->second) << std::right
                << std::scientific << std::setprecision(3)
                << std::setw(12) << mean_j << std::setw(12) << dim_max[j]
                << std::fixed << std::setprecision(1)
                << std::setw(8) << (sum_abs > 0.0 ? 100.0 * dim_sum[j] / sum_abs : 0.0) << "%\n";
    }
    // Concentration, stated rather than left to the reader's eye: a uniform
    // spread over 24 actions is 4.2% each, so a top-3 share near 12.5% means
    // "uniform" and a top-3 share near 60% means "three joints own the error".
    double top3 = 0.0;
    for (std::size_t k = 0; k < std::min<std::size_t>(3, order.size()); ++k) {
      top3 += dim_sum[order[k]];
    }
    const double top3_pct = sum_abs > 0.0 ? 100.0 * top3 / sum_abs : 0.0;
    std::cout << std::fixed << std::setprecision(1)
              << "  top-3 share " << top3_pct << "%  (uniform over "
              << fx.out_dim << " actions would be "
              << 300.0 / fx.out_dim << "%)\n";
  }

  // -- latency ---------------------------------------------------------------
  std::cout << "\n";
  const auto compact = TimeInfer(policy.get(), fx, latency_iters, 0.0);
  PrintLatency("latency compact-loop", compact, latency_iters);
  if (duty_hz > 0.0) {
    // Fewer iterations: at 50 Hz, 2000 of them would be 40 seconds.
    const int paced_iters = std::min(latency_iters, static_cast<int>(duty_hz * 10.0));
    std::cout << std::fixed << std::setprecision(0)
              << "timing " << paced_iters << " more paced at " << duty_hz
              << " Hz (~" << (paced_iters / duty_hz) << "s)...\n";
    const auto paced = TimeInfer(policy.get(), fx, paced_iters, duty_hz);
    PrintLatency("latency @duty", paced, paced_iters);
    const double c50 = Percentile(compact, 0.50), p50 = Percentile(paced, 0.50);
    if (c50 > 0.0) {
      std::cout << std::fixed << std::setprecision(1)
                << "  paced/compact p50 ratio " << (p50 / c50)
                << "x -- report the PACED number; the control loop calls infer()\n"
                << "  once per cycle, which is the paced case, not the compact one.\n";
    }
  }

  // -- verdict ---------------------------------------------------------------
  if (vs_engine) {
    std::cout
      << "\nCOMPARISON COMPLETE (no pass line -- see MODE above). The gate this\n"
      << "feeds is the closed-loop one: same seed, same command sequence, INT8 vs\n"
      << "the baseline, and tracking error / air-time symmetry / torque headroom\n"
      << "must not regress.\n";
    return 0;
  }

  const bool pass = max_abs <= tol;
  std::cout << (pass ? "PARITY PASS" : "PARITY FAIL")
            << std::scientific << std::setprecision(1)
            << " (tolerance " << tol << ")\n";
  if (!pass) {
    std::cout
      << "\nThis certifies one file, not the ONNX. If max_abs is around 1e-2 the\n"
      << "likely cause is TF32: TensorRT permits it by default, it rounds GEMM\n"
      << "inputs to a 10-bit mantissa, and whether a TF32 kernel wins is decided\n"
      << "by build-time timing -- so the same command can pass once and fail next\n"
      << "time. Rebuild with a build of r1_build_engine that clears kTF32.\n"
      << "\nIf this is an INT8 plan: an INT8 engine is NOT expected to pass a 1e-3\n"
      << "gate against the trained policy, and forcing the tolerance up until it\n"
      << "does would certify nothing. Compare it against the FP16 engine with\n"
      << "--baseline instead, and gate on closed-loop behaviour.\n";
  }
  return pass ? 0 : 1;
}
