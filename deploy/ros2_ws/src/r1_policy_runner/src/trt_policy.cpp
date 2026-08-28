#include "r1_policy_runner/trt_policy.hpp"

#include <NvOnnxParser.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace r1_policy_runner
{
namespace
{

// TensorRT 10 removed the binding-index API in favour of named tensors, and
// replaced enqueueV2 with enqueueV3. JetPack 6.0 still carries TensorRT 8.6,
// so both paths stay compiled-in until the R1's JetPack version is confirmed.
#if NV_TENSORRT_MAJOR >= 10
#define R1_TRT10 1
#else
#define R1_TRT10 0
#endif

/// TensorRT objects are reference-counted in 8.x (destroy()) and plain
/// `delete`-able in 10.x. Only 10.x is exercised here; the 8.x branch relies on
/// the deprecated-but-present virtual destructor.
struct TrtDeleter
{
  template<typename T>
  void operator()(T * p) const noexcept {delete p;}
};

template<typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

int64_t volumeOf(const nvinfer1::Dims & d)
{
  int64_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) {
    // A dynamic axis reports -1. The policy is exported with a fixed batch of
    // 1, so treating that as an error is correct rather than restrictive.
    if (d.d[i] < 0) {return -1;}
    v *= d.d[i];
  }
  return v;
}

std::string cudaErr(const char * what, cudaError_t e)
{
  std::ostringstream os;
  os << what << ": " << cudaGetErrorString(e);
  return os.str();
}

}  // namespace

void TrtLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity > threshold_) {return;}
  const char * tag = "INFO";
  switch (severity) {
    case Severity::kINTERNAL_ERROR: tag = "TRT INTERNAL"; break;
    case Severity::kERROR:          tag = "TRT ERROR";    break;
    case Severity::kWARNING:        tag = "TRT WARN";     break;
    case Severity::kINFO:           tag = "TRT INFO";     break;
    case Severity::kVERBOSE:        tag = "TRT VERBOSE";  break;
  }
  std::fprintf(stderr, "[%s] %s\n", tag, msg);
}

// ---------------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------------

bool BuildEngineFromOnnx(
  const std::string & onnx_path,
  const std::string & plan_path,
  bool fp16,
  std::size_t workspace_mb,
  std::string * error)
{
  auto fail = [error](const std::string & m) {
      if (error) {*error = m;}
      return false;
    };

  TrtLogger logger{nvinfer1::ILogger::Severity::kWARNING};

  TrtPtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
  if (!builder) {return fail("createInferBuilder failed");}

#if R1_TRT10
  TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0)};
#else
  const auto flags =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtPtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flags)};
#endif
  if (!network) {return fail("createNetworkV2 failed");}

  TrtPtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, logger)};
  if (!parser) {return fail("createParser failed");}

  if (!parser->parseFromFile(
      onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    std::ostringstream os;
    os << "failed to parse " << onnx_path;
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      os << "\n  " << parser->getError(i)->desc();
    }
    return fail(os.str());
  }

  TrtPtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
  if (!config) {return fail("createBuilderConfig failed");}

#if R1_TRT10
  config->setMemoryPoolLimit(
    nvinfer1::MemoryPoolType::kWORKSPACE, workspace_mb * 1024ULL * 1024ULL);
#else
  config->setMaxWorkspaceSize(workspace_mb * 1024ULL * 1024ULL);
#endif

  if (fp16) {
    if (!builder->platformHasFastFp16()) {
      std::fprintf(stderr, "[warn] platform has no fast FP16; building FP32 instead\n");
    } else {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

#if R1_TRT10
  TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
#else
  TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
#endif
  if (!plan) {return fail("buildSerializedNetwork returned null");}

  std::ofstream out(plan_path, std::ios::binary);
  if (!out) {return fail("cannot open " + plan_path + " for writing");}
  out.write(static_cast<const char *>(plan->data()), static_cast<std::streamsize>(plan->size()));
  if (!out) {return fail("short write to " + plan_path);}

  return true;
}

// ---------------------------------------------------------------------------
// Engine execution
// ---------------------------------------------------------------------------

struct TrtPolicy::Impl
{
  TrtLogger logger{nvinfer1::ILogger::Severity::kWARNING};
  TrtPtr<nvinfer1::IRuntime> runtime;
  TrtPtr<nvinfer1::ICudaEngine> engine;
  TrtPtr<nvinfer1::IExecutionContext> context;

  cudaStream_t stream = nullptr;
  void * d_in = nullptr;
  void * d_out = nullptr;
  float * h_in = nullptr;   // pinned: lets the H2D copy overlap and avoids a staging copy
  float * h_out = nullptr;  // pinned
  std::size_t in_bytes = 0;
  std::size_t out_bytes = 0;
#if !R1_TRT10
  // TensorRT 8's enqueueV2 takes a bindings array indexed by binding index, so
  // the indices must be carried, not assumed. In practice the ONNX parser adds
  // inputs first and index 0 is the input -- but a silently-swapped pair would
  // feed the network its own uninitialised output buffer and still return
  // success, which is precisely the failure mode this project has already been
  // bitten by once (TensorRT 10.3 computing wrong answers without erroring).
  int in_idx = -1;
  int out_idx = -1;
  std::vector<void *> bindings;
#endif

  ~Impl()
  {
    if (h_in) {cudaFreeHost(h_in);}
    if (h_out) {cudaFreeHost(h_out);}
    if (d_in) {cudaFree(d_in);}
    if (d_out) {cudaFree(d_out);}
    if (stream) {cudaStreamDestroy(stream);}
  }
};

TrtPolicy::TrtPolicy(const std::string & plan_path, int device)
: impl_(std::make_unique<Impl>())
{
  if (cudaError_t e = cudaSetDevice(device); e != cudaSuccess) {
    throw std::runtime_error(cudaErr("cudaSetDevice", e));
  }

  std::ifstream in(plan_path, std::ios::binary | std::ios::ate);
  if (!in) {throw std::runtime_error("cannot open engine " + plan_path);}
  const auto size = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<char> blob(size);
  if (!in.read(blob.data(), static_cast<std::streamsize>(size))) {
    throw std::runtime_error("short read on engine " + plan_path);
  }

  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
  if (!impl_->runtime) {throw std::runtime_error("createInferRuntime failed");}

  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(blob.data(), size));
  if (!impl_->engine) {
    // Overwhelmingly this means the plan was built for a different TensorRT
    // version, GPU architecture or driver than the one loading it.
    throw std::runtime_error(
            "deserializeCudaEngine failed for " + plan_path +
            " -- an engine is only valid on the machine that built it "
            "(TensorRT version + GPU arch + driver must match)");
  }

  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {throw std::runtime_error("createExecutionContext failed");}

  // -- resolve I/O from the engine ------------------------------------------
#if R1_TRT10
  int n_in = 0, n_out = 0;
  for (int i = 0; i < impl_->engine->getNbIOTensors(); ++i) {
    const char * name = impl_->engine->getIOTensorName(i);
    const auto mode = impl_->engine->getTensorIOMode(name);
    const auto dims = impl_->engine->getTensorShape(name);
    const int64_t vol = volumeOf(dims);
    if (vol < 0) {
      throw std::runtime_error(
              std::string("tensor '") + name +
              "' has a dynamic shape; this runner expects a fixed batch-1 export");
    }
    if (impl_->engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(std::string("tensor '") + name + "' is not FP32 at the boundary");
    }
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      info_.input_name = name; info_.input_dim = vol; ++n_in;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      info_.output_name = name; info_.output_dim = vol; ++n_out;
    }
  }
  if (n_in != 1 || n_out != 1) {
    throw std::runtime_error("expected exactly 1 input and 1 output tensor");
  }
#else
  const int n_bindings = impl_->engine->getNbBindings();
  impl_->bindings.assign(static_cast<std::size_t>(n_bindings), nullptr);

  int n_in = 0, n_out = 0;
  for (int i = 0; i < n_bindings; ++i) {
    const int64_t vol = volumeOf(impl_->engine->getBindingDimensions(i));
    if (vol < 0) {throw std::runtime_error("dynamic shapes are not supported by this runner");}
    if (impl_->engine->getBindingDataType(i) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(
              std::string("binding '") + impl_->engine->getBindingName(i) +
              "' is not FP32 at the boundary");
    }
    if (impl_->engine->bindingIsInput(i)) {
      info_.input_name = impl_->engine->getBindingName(i);
      info_.input_dim = vol; impl_->in_idx = i; ++n_in;
    } else {
      info_.output_name = impl_->engine->getBindingName(i);
      info_.output_dim = vol; impl_->out_idx = i; ++n_out;
    }
  }
  if (n_in != 1 || n_out != 1) {
    throw std::runtime_error("expected exactly 1 input and 1 output binding");
  }
#endif

  // -- allocate once ---------------------------------------------------------
  impl_->in_bytes = static_cast<std::size_t>(info_.input_dim) * sizeof(float);
  impl_->out_bytes = static_cast<std::size_t>(info_.output_dim) * sizeof(float);

  if (cudaError_t e = cudaStreamCreate(&impl_->stream); e != cudaSuccess) {
    throw std::runtime_error(cudaErr("cudaStreamCreate", e));
  }
  if (cudaError_t e = cudaMalloc(&impl_->d_in, impl_->in_bytes); e != cudaSuccess) {
    throw std::runtime_error(cudaErr("cudaMalloc(input)", e));
  }
  if (cudaError_t e = cudaMalloc(&impl_->d_out, impl_->out_bytes); e != cudaSuccess) {
    throw std::runtime_error(cudaErr("cudaMalloc(output)", e));
  }
  if (cudaError_t e = cudaHostAlloc(
      reinterpret_cast<void **>(&impl_->h_in), impl_->in_bytes, cudaHostAllocDefault);
    e != cudaSuccess)
  {
    throw std::runtime_error(cudaErr("cudaHostAlloc(input)", e));
  }
  if (cudaError_t e = cudaHostAlloc(
      reinterpret_cast<void **>(&impl_->h_out), impl_->out_bytes, cudaHostAllocDefault);
    e != cudaSuccess)
  {
    throw std::runtime_error(cudaErr("cudaHostAlloc(output)", e));
  }

#if R1_TRT10
  // Bind once. enqueueV3 then reuses these addresses every cycle.
  if (!impl_->context->setTensorAddress(info_.input_name.c_str(), impl_->d_in) ||
    !impl_->context->setTensorAddress(info_.output_name.c_str(), impl_->d_out))
  {
    throw std::runtime_error("setTensorAddress failed");
  }
#else
  // Same "bind once" contract as the TRT10 path, by index rather than by name.
  impl_->bindings[static_cast<std::size_t>(impl_->in_idx)] = impl_->d_in;
  impl_->bindings[static_cast<std::size_t>(impl_->out_idx)] = impl_->d_out;
#endif

  // A first pass on load: the initial enqueue pays for lazy kernel-module load
  // and cuBLAS handle creation, which would otherwise land on the first real
  // control cycle as a multi-millisecond outlier.
  std::memset(impl_->h_in, 0, impl_->in_bytes);
  std::vector<float> warm(static_cast<std::size_t>(info_.output_dim));
  for (int i = 0; i < 8; ++i) {
    Infer(impl_->h_in, warm.data());
  }
}

TrtPolicy::~TrtPolicy() = default;

bool TrtPolicy::Infer(const float * obs, float * action)
{
  std::memcpy(impl_->h_in, obs, impl_->in_bytes);

  if (cudaMemcpyAsync(
      impl_->d_in, impl_->h_in, impl_->in_bytes,
      cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess)
  {
    return false;
  }

#if R1_TRT10
  if (!impl_->context->enqueueV3(impl_->stream)) {return false;}
#else
  if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr)) {return false;}
#endif

  if (cudaMemcpyAsync(
      impl_->h_out, impl_->d_out, impl_->out_bytes,
      cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess)
  {
    return false;
  }
  if (cudaStreamSynchronize(impl_->stream) != cudaSuccess) {return false;}

  std::memcpy(action, impl_->h_out, impl_->out_bytes);
  return true;
}

}  // namespace r1_policy_runner
