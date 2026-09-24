// onnx -> serialised TensorRT engine.
//
// Run this ON THE MACHINE THAT WILL EXECUTE THE POLICY. A plan file encodes the
// TensorRT version, the GPU's compute architecture and kernel tactics selected
// by timing real kernels on that device; it is not portable. The artefact that
// moves from the dev box to the robot is the ONNX, not the plan.
//
//   r1_build_engine --onnx policy.onnx --plan policy.plan [--fp16] [--workspace 256]
//   r1_build_engine --onnx policy.onnx --plan policy_int8.plan
//                   --int8 --calib calib_obs.bin --calib-cache calib.table
//
// Precision note: TensorRT permits TF32 by default, which rounds GEMM inputs to
// a 10-bit mantissa and is chosen or not by build-time kernel timing. We clear
// it, so --fp32 (the default) is reproducible fp32. --tf32 opts back in.
//
// And the measured caveat that applies to every low-precision flag here
// (Orin NX, 2026-09-24): this policy is 73k parameters at batch 1, so a step
// costs ~150 kFLOP against ~900 us of wall time -- the loop is kernel-launch
// bound and the arithmetic is noise. Enabling FP16 changed neither the numerics
// (1.717e-05, FP32's order) nor the latency, because the builder timed FP32
// kernels as faster and kept them. Expect the same of INT8. Its value here is
// as a controlled, physically real perturbation for probing robustness, not as
// a speed-up; build it to measure behaviour, and report the latency honestly.

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
    << "[--fp16] [--int8 --calib <f>] [--tf32] [--workspace <MB>]\n\n"
    << "  --fp16       permit FP16 kernels. 'Permit', not 'require': measured on\n"
    << "               the Orin it changed neither numerics nor latency, because\n"
    << "               TensorRT timed FP32 kernels as faster. Verify with\n"
    << "               r1_parity_check rather than assuming.\n"
    << "  --int8       permit INT8 kernels. Requires --calib.\n"
    << "  --calib <f>  calibration observations: R1CB from\n"
    << "               tools/record_calib_obs.py (recorded off the robot -- the\n"
    << "               only defensible input) or R1FX, the parity fixture, whose\n"
    << "               inputs are synthetic and make this a build smoke test only.\n"
    << "  --calib-cache <f>  where to cache the calibration table. If it EXISTS it\n"
    << "               is reused and calibration is skipped, so delete it whenever\n"
    << "               --calib changes. The build says which path it took.\n"
    << "  --tf32       allow TF32 (TensorRT's default, which we otherwise clear).\n"
    << "               TF32 rounds GEMM inputs to a 10-bit mantissa, so an 'fp32'\n"
    << "               engine drifts ~1e-2 AND varies between rebuilds. Do not use\n"
    << "               this for an engine that has to pass a parity gate.\n"
    << "  --workspace  scratch ceiling in MB for tactic selection (default 256)\n";
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string onnx, plan;
  r1_policy_runner::BuildOptions opt;

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
    } else if (a == "--fp16") {opt.fp16 = true;} else if (a == "--int8") {
      opt.int8 = true;
    } else if (a == "--calib") {
      opt.calib_data = next("--calib");
    } else if (a == "--calib-cache") {
      opt.calib_cache = next("--calib-cache");
    } else if (a == "--tf32") {
      opt.allow_tf32 = true;
    } else if (a == "--workspace") {
      opt.workspace_mb = std::stoul(next("--workspace"));
    } else if (a == "-h" || a == "--help") {Usage(argv[0]); return 0;} else {
      std::cerr << "error: unknown argument '" << a << "'\n";
      Usage(argv[0]);
      return 2;
    }
  }
  if (onnx.empty() || plan.empty()) {Usage(argv[0]); return 2;}

  // Spelled out rather than abbreviated: this string is what ends up pasted into
  // the commissioning log as the record of what was built.
  std::string precision = opt.int8 ? "int8" : (opt.fp16 ? "fp16" : "fp32");
  if (opt.int8 && opt.fp16) {precision = "int8+fp16";}
  precision += opt.allow_tf32 ? " +tf32 (NOT reproducible)" : " (tf32 cleared)";
  std::cout << "building " << plan << " from " << onnx
            << "  (precision=" << precision
            << ", workspace=" << opt.workspace_mb << "MB)\n";

  const auto t0 = std::chrono::steady_clock::now();
  std::string err;
  if (!r1_policy_runner::BuildEngineFromOnnx(onnx, plan, opt, &err))
  {
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
              << policy.info().output_name << "' [" << policy.output_dim() << "]\n"
              << "plan " << r1_policy_runner::PlanFingerprint(plan)
              << "  <- parity-check THIS file, and run THIS file\n";
    if (opt.int8) {
      std::cout
        << "\nINT8 next steps -- this plan is NOT accepted by having built:\n"
        << "  1. r1_parity_check --plan " << plan << " --baseline <fp16.plan> \\\n"
        << "       --fixture <parity_fixture.bin> --per-dim\n"
        << "     (engine-vs-engine. There is no pass line here; the deliverable is\n"
        << "      the magnitude and the STRUCTURE of the error -- spread evenly, or\n"
        << "      piled onto a few joints?)\n"
        << "  2. closed-loop metrics in sim against the FP16 baseline. That is the\n"
        << "     only hard 'no regression' gate.\n"
        << "  3. hanging on the robot, as a SAFETY check. Keep FP16 as the fallback.\n";
    }
  } catch (const std::exception & e) {
    std::cerr << "engine written but failed to load back: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
