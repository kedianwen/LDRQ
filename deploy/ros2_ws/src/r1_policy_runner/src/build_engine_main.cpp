// onnx -> serialised TensorRT engine.
//
// Run this ON THE MACHINE THAT WILL EXECUTE THE POLICY. A plan file encodes the
// TensorRT version, the GPU's compute architecture and kernel tactics selected
// by timing real kernels on that device; it is not portable. The artefact that
// moves from the dev box to the robot is the ONNX, not the plan.
//
//   r1_build_engine --onnx policy.onnx --plan policy.plan [--fp16] [--workspace 256]

#include "r1_policy_runner/trt_policy.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace
{
void Usage(const char * argv0)
{
  std::cerr
    << "usage: " << argv0 << " --onnx <in.onnx> --plan <out.plan> "
    << "[--fp16] [--workspace <MB>]\n\n"
    << "  --fp16       enable FP16 kernels (roughly halves latency on Orin;\n"
    << "               verify numerics with r1_parity_check afterwards)\n"
    << "  --workspace  scratch ceiling in MB for tactic selection (default 256)\n";
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string onnx, plan;
  bool fp16 = false;
  std::size_t workspace_mb = 256;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char * what) -> std::string {
        if (i + 1 >= argc) {
          std::cerr << "error: " << what << " needs a value\n";
          std::exit(2);
        }
        return argv[++i];
      };
    if (a == "--onnx") {onnx = next("--onnx");} else if (a == "--plan") {
      plan = next("--plan");
    } else if (a == "--fp16") {fp16 = true;} else if (a == "--workspace") {
      workspace_mb = std::stoul(next("--workspace"));
    } else if (a == "-h" || a == "--help") {Usage(argv[0]); return 0;} else {
      std::cerr << "error: unknown argument '" << a << "'\n";
      Usage(argv[0]);
      return 2;
    }
  }
  if (onnx.empty() || plan.empty()) {Usage(argv[0]); return 2;}

  std::cout << "building " << plan << " from " << onnx
            << "  (precision=" << (fp16 ? "fp16" : "fp32")
            << ", workspace=" << workspace_mb << "MB)\n";

  const auto t0 = std::chrono::steady_clock::now();
  std::string err;
  if (!r1_policy_runner::BuildEngineFromOnnx(onnx, plan, fp16, workspace_mb, &err)) {
    std::cerr << "engine build failed: " << err << "\n";
    return 1;
  }
  const auto secs =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  // Load it straight back. A plan that builds but does not deserialise is a
  // failure worth catching here rather than at robot start-up.
  try {
    r1_policy_runner::TrtPolicy policy(plan);
    std::cout << "ok in " << secs << "s -- input '" << policy.info().input_name
              << "' [" << policy.input_dim() << "]  output '"
              << policy.info().output_name << "' [" << policy.output_dim() << "]\n";
  } catch (const std::exception & e) {
    std::cerr << "engine written but failed to load back: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
