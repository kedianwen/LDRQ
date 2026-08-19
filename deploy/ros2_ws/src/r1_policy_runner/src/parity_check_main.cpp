// Numerical parity between the trained PyTorch policy and the TensorRT engine,
// plus a latency profile of the engine on this host.
//
// The fixture is produced by tools/make_fixture.py, which runs the exported
// TorchScript policy on N observation vectors and records both sides. Passing
// this is what lets us claim the deployed engine *is* the trained policy --
// without it, "the node runs at 50Hz" says nothing about what it computes.
//
//   r1_parity_check --plan policy.plan --fixture parity_fixture.bin [--tol 1e-3]

#include "r1_policy_runner/trt_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
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

double Percentile(std::vector<double> sorted, double p)
{
  if (sorted.empty()) {return 0.0;}
  const auto idx = static_cast<std::size_t>(p * (sorted.size() - 1));
  return sorted[idx];
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string plan, fixture_path;
  double tol = 1e-3;
  int latency_iters = 2000;

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
    } else {
      std::cerr << "usage: " << argv[0]
                << " --plan <p.plan> --fixture <f.bin> [--tol 1e-3] [--latency-iters 2000]\n";
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
            << " | fixture " << fx.n << " samples\n";

  // -- numerical parity ------------------------------------------------------
  std::vector<float> got(fx.out_dim);
  double max_abs = 0.0, sum_abs = 0.0;
  double max_rel = 0.0;
  uint32_t worst_sample = 0, worst_elem = 0;

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
        std::cout << " " << fx.outputs[j];
      }
      std::cout << "\n";
    }
    for (uint32_t j = 0; j < fx.out_dim; ++j) {
      const double ref = fx.outputs[static_cast<std::size_t>(s) * fx.out_dim + j];
      const double d = std::fabs(got[j] - ref);
      sum_abs += d;
      if (d > max_abs) {max_abs = d; worst_sample = s; worst_elem = j;}
      const double denom = std::max(1e-6, std::fabs(ref));
      max_rel = std::max(max_rel, d / denom);
    }
  }
  const double mean_abs = sum_abs / (static_cast<double>(fx.n) * fx.out_dim);

  std::cout << std::scientific << std::setprecision(3)
            << "parity: max_abs=" << max_abs
            << "  mean_abs=" << mean_abs
            << "  max_rel=" << max_rel
            << "  (worst at sample " << worst_sample << ", action[" << worst_elem << "])\n";

  // -- latency ---------------------------------------------------------------
  std::vector<double> us;
  us.reserve(static_cast<std::size_t>(latency_iters));
  for (int i = 0; i < latency_iters; ++i) {
    const float * obs = fx.inputs.data() +
      static_cast<std::size_t>(i % fx.n) * fx.in_dim;
    const auto t0 = std::chrono::steady_clock::now();
    policy->Infer(obs, got.data());
    us.push_back(
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(us.begin(), us.end());
  std::cout << std::fixed << std::setprecision(1)
            << "latency (us over " << latency_iters << "): "
            << "p50=" << Percentile(us, 0.50)
            << "  p95=" << Percentile(us, 0.95)
            << "  p99=" << Percentile(us, 0.99)
            << "  max=" << us.back() << "\n";

  const bool pass = max_abs <= tol;
  std::cout << (pass ? "PARITY PASS" : "PARITY FAIL")
            << std::scientific << std::setprecision(1)
            << " (tolerance " << tol << ")\n";
  return pass ? 0 : 1;
}
