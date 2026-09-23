// TensorRT execution of the R1 locomotion policy.
//
// The policy is a 4-layer MLP (425 -> 128 -> 128 -> 128 -> 24, ELU), so this
// wrapper is deliberately narrow: one input tensor, one output tensor, batch 1,
// synchronous. What it does care about is the control loop's real-time budget --
// every buffer is allocated once in the constructor, and infer() performs no
// allocation, no exception-throwing lookup and no host-side synchronisation
// beyond the single stream sync it cannot avoid.

#ifndef R1_POLICY_RUNNER__TRT_POLICY_HPP_
#define R1_POLICY_RUNNER__TRT_POLICY_HPP_

#include <cuda_runtime_api.h>

#include <NvInfer.h>

#include <cstdint>
#include <memory>
#include <string>

namespace r1_policy_runner
{

/// Routes TensorRT's internal logging into a caller-supplied sink.
class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(Severity threshold = Severity::kWARNING)
  : threshold_(threshold) {}

  void log(Severity severity, const char * msg) noexcept override;

private:
  Severity threshold_;
};

/// Shape and naming of the engine's I/O, resolved from the engine itself
/// rather than hard-coded, so a re-exported policy with different dimensions
/// fails loudly at load time instead of silently reading past a buffer.
struct EngineInfo
{
  std::string input_name;
  std::string output_name;
  int64_t input_dim = 0;
  int64_t output_dim = 0;
  bool fp16 = false;
};

/// Builds a serialised engine from an ONNX file.
///
/// Engines are tied to the TensorRT version, GPU architecture and driver they
/// were built on -- a plan produced on this dev box will NOT load on the
/// robot's Orin NX. Ship the ONNX and run this on the target.
///
/// @param onnx_path   input ONNX model
/// @param plan_path   output serialised engine
/// @param fp16        enable FP16 kernels (falls back to FP32 where unsupported)
/// @param allow_tf32  leave TensorRT's default TF32 permission in place. TF32 is
///        ENABLED BY DEFAULT by TensorRT and rounds GEMM inputs to a 10-bit
///        mantissa -- FP16-level precision under an "fp32" label. Worse, the
///        flag only *allows* TF32: whether a TF32 kernel actually wins is
///        decided by timing kernels at build time, so two builds of the same
///        ONNX on the same device can differ numerically by ~1e-2. Pass false
///        (the caller default) for anything a parity gate must certify.
/// @param workspace_mb scratch memory ceiling for tactic selection
/// @param error       populated on failure
/// @return true on success
bool BuildEngineFromOnnx(
  const std::string & onnx_path,
  const std::string & plan_path,
  bool fp16,
  bool allow_tf32,
  std::size_t workspace_mb,
  std::string * error);

/// Identifies a plan file by content: "<bytes>B fnv1a=0x...".
///
/// Tactic selection is remeasured on every build, so "parity passed" certifies
/// one specific file, not the ONNX it came from. Print this at build, at parity
/// check and at node start-up so the engine that passed can be shown to be the
/// engine that ran. Returns "unreadable" if the file cannot be opened.
std::string PlanFingerprint(const std::string & plan_path);

/// Loads a serialised engine and runs it, one observation at a time.
class TrtPolicy
{
public:
  /// @throws std::runtime_error if the plan cannot be read, deserialised, or
  ///         does not have exactly one input and one output.
  explicit TrtPolicy(const std::string & plan_path, int device = 0);
  ~TrtPolicy();

  TrtPolicy(const TrtPolicy &) = delete;
  TrtPolicy & operator=(const TrtPolicy &) = delete;

  /// Runs one forward pass.
  /// @param obs    input_dim() floats, caller-owned
  /// @param action output_dim() floats, caller-owned
  /// @return false if the CUDA pipeline reported an error; @p action untouched
  bool Infer(const float * obs, float * action);

  const EngineInfo & info() const noexcept {return info_;}
  int64_t input_dim() const noexcept {return info_.input_dim;}
  int64_t output_dim() const noexcept {return info_.output_dim;}

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  EngineInfo info_;
};

}  // namespace r1_policy_runner

#endif  // R1_POLICY_RUNNER__TRT_POLICY_HPP_
