#include "r1_policy_runner/trt_policy.hpp"

#include <NvOnnxParser.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
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

/// Reads observation vectors out of an R1CB or R1FX file.
///
/// R1CB: "R1CB" | u32 version=1 | u32 n | u32 dim | f32 data[n*dim]
/// R1FX: "R1FX" | u32 version=1 | u32 n | u32 in_dim | u32 out_dim
///                | f32 inputs[n*in_dim] | f32 outputs[n*out_dim]
///
/// Both carry the inputs first, so one reader covers them; the R1FX reference
/// outputs are simply not read. @p synthetic reports which one it was, because
/// that distinction is the whole difference between a calibration set and a
/// decoration.
bool LoadObsFile(
  const std::string & path, std::vector<float> * data, uint32_t * n, uint32_t * dim,
  bool * synthetic, std::string * err)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {*err = "cannot open " + path; return false;}

  char magic[4] = {0, 0, 0, 0};
  uint32_t version = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char *>(&version), 4);
  const std::string tag(magic, 4);
  if (tag != "R1CB" && tag != "R1FX") {
    *err = path + ": magic is '" + tag + "', expected R1CB (record_calib_obs.py) or R1FX (fixture)";
    return false;
  }
  if (version != 1) {*err = path + ": unsupported version"; return false;}
  *synthetic = (tag == "R1FX");

  in.read(reinterpret_cast<char *>(n), 4);
  in.read(reinterpret_cast<char *>(dim), 4);
  if (*synthetic) {
    uint32_t out_dim = 0;
    in.read(reinterpret_cast<char *>(&out_dim), 4);
  }
  if (!in) {*err = path + ": truncated header"; return false;}
  if (*n == 0 || *dim == 0) {*err = path + ": empty"; return false;}

  data->resize(static_cast<std::size_t>(*n) * *dim);
  in.read(reinterpret_cast<char *>(data->data()),
    static_cast<std::streamsize>(data->size() * sizeof(float)));
  if (!in) {*err = path + ": truncated payload"; return false;}
  return true;
}

/// What BuildEngineFromOnnx needs from a calibrator, independent of which
/// TensorRT calibration algorithm it derives from.
struct CalibratorHandle
{
  virtual ~CalibratorHandle() = default;
  virtual nvinfer1::IInt8Calibrator * trt() = 0;
  virtual bool ok() const = 0;
  virtual bool cache_was_reused() const = 0;
};

/// Feeds recorded observations to TensorRT's INT8 calibration pass.
///
/// Templated on the algorithm because the two available ones make OPPOSITE
/// trades and this policy has a reason to care about both:
///
///   Entropy2  minimises information loss over the whole histogram, so a single
///             outlier frame does not cost resolution everywhere else. A 60 s
///             robot recording will contain outlier frames.
///   MinMax    pins the scale to the most extreme activation seen, so nothing
///             clips. For a locomotion policy the tails are where recovery
///             lives: clipping the activation that corresponds to "being
///             shoved" degrades exactly the state that matters most, and that
///             is a failure mode no aggregate error number surfaces.
///
/// Neither is obviously right, both cost ~20 s to build, so build both and
/// compare per-dimension. For a project that is treating INT8 as a controlled
/// perturbation source rather than a speed-up, having two perturbation variants
/// is worth more than picking one and defending it.
///
/// Batch is 1 because the engine is batch 1. That makes calibration slow in
/// wall-clock terms and exactly representative in distribution terms, which is
/// the trade worth making for a few thousand samples.
template<typename Algo>
class ObsCalibratorT : public Algo, public CalibratorHandle
{
public:
  ObsCalibratorT(std::vector<float> data, uint32_t n, uint32_t dim, std::string cache_path)
  : data_(std::move(data)), n_(n), dim_(dim), cache_path_(std::move(cache_path))
  {
    bytes_ = static_cast<std::size_t>(dim_) * sizeof(float);
    if (cudaMalloc(&d_buf_, bytes_) != cudaSuccess) {d_buf_ = nullptr;}
    if (!cache_path_.empty()) {
      std::ifstream in(cache_path_, std::ios::binary);
      if (in) {
        cache_.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
      }
    }
  }

  ~ObsCalibratorT() override
  {
    if (d_buf_) {cudaFree(d_buf_);}
  }

  nvinfer1::IInt8Calibrator * trt() override {return this;}
  bool ok() const override {return d_buf_ != nullptr;}
  bool cache_was_reused() const override {return reused_;}

  int32_t getBatchSize() const noexcept override {return 1;}

  bool getBatch(void * bindings[], char const * names[], int32_t nbBindings) noexcept override
  {
    (void)names;
    if (cursor_ >= n_ || nbBindings < 1 || !d_buf_) {return false;}
    const float * src = data_.data() + static_cast<std::size_t>(cursor_) * dim_;
    if (cudaMemcpy(d_buf_, src, bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {return false;}
    bindings[0] = d_buf_;
    ++cursor_;
    return true;
  }

  void const * readCalibrationCache(std::size_t & length) noexcept override
  {
    if (cache_.empty()) {length = 0; return nullptr;}
    // TensorRT calls this before the first getBatch. A hit here means the whole
    // calibration pass is skipped, which is the single most misleading thing
    // this class can do silently, so record it for the caller to print.
    reused_ = true;
    length = cache_.size();
    return cache_.data();
  }

  void writeCalibrationCache(void const * ptr, std::size_t length) noexcept override
  {
    if (cache_path_.empty()) {return;}
    std::ofstream out(cache_path_, std::ios::binary);
    if (!out) {return;}
    out.write(static_cast<const char *>(ptr), static_cast<std::streamsize>(length));
  }

private:
  std::vector<float> data_;
  uint32_t n_ = 0, dim_ = 0, cursor_ = 0;
  std::size_t bytes_ = 0;
  void * d_buf_ = nullptr;
  std::string cache_path_;
  std::vector<char> cache_;
  bool reused_ = false;
};

using EntropyObsCalibrator = ObsCalibratorT<nvinfer1::IInt8EntropyCalibrator2>;
using MinMaxObsCalibrator = ObsCalibratorT<nvinfer1::IInt8MinMaxCalibrator>;

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

std::string PlanFingerprint(const std::string & plan_path)
{
  std::ifstream in(plan_path, std::ios::binary);
  if (!in) {return "unreadable";}
  std::uint64_t h = 14695981039346656037ULL;     // FNV-1a 64 offset basis
  std::uint64_t n = 0;
  char buf[64 * 1024];
  while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
    const std::streamsize got = in.gcount();
    n += static_cast<std::uint64_t>(got);
    for (std::streamsize i = 0; i < got; ++i) {
      h ^= static_cast<unsigned char>(buf[i]);
      h *= 1099511628211ULL;
    }
  }
  std::ostringstream os;
  os << n << "B fnv1a=0x" << std::hex << std::setw(16) << std::setfill('0') << h;
  return os.str();
}

bool BuildEngineFromOnnx(
  const std::string & onnx_path,
  const std::string & plan_path,
  const BuildOptions & opt,
  std::string * error)
{
  auto fail = [error](const std::string & m) {
      if (error) {*error = m;}
      return false;
    };

  // kINFO is where TensorRT prints the per-tensor dynamic ranges the calibrator
  // produced. That is the only way to see WHICH layers the quantisation is tight
  // on -- and the W08 plan specifically wants a look at the layers where the
  // FP16 build reported subnormal weights.
  TrtLogger logger{opt.verbose ? nvinfer1::ILogger::Severity::kINFO
    : nvinfer1::ILogger::Severity::kWARNING};

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
      onnx_path.c_str(), static_cast<int>(opt.verbose ?
      nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING)))
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
    nvinfer1::MemoryPoolType::kWORKSPACE, opt.workspace_mb * 1024ULL * 1024ULL);
#else
  config->setMaxWorkspaceSize(opt.workspace_mb * 1024ULL * 1024ULL);
#endif

  if (opt.fp16) {
    if (!builder->platformHasFastFp16()) {
      std::fprintf(stderr, "[warn] platform has no fast FP16; building FP32 instead\n");
    } else {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  // -- INT8 ------------------------------------------------------------------
  // Held in scope until buildSerializedNetwork returns: TensorRT calls back into
  // the calibrator during the build, not before it.
  std::unique_ptr<CalibratorHandle> calibrator;
  if (opt.int8) {
    if (!builder->platformHasFastInt8()) {
      return fail("platform reports no fast INT8; refusing to build an INT8 plan here");
    }
    if (opt.calib_data.empty()) {
      // Without a calibrator TensorRT still produces an INT8 engine, using
      // whatever dynamic ranges it can infer. It loads and runs. Its activation
      // scales are then not a property of the deployment distribution, and the
      // acceptance report would be measuring something it does not name.
      return fail(
        "--int8 needs --calib <R1CB|R1FX>. An uncalibrated INT8 engine builds "
        "and runs, which is exactly why it must not be allowed: its scales come "
        "from nowhere. Record real observations with tools/record_calib_obs.py.");
    }

    std::vector<float> calib_data;
    uint32_t n = 0, dim = 0;
    bool synthetic = false;
    std::string load_err;
    if (!LoadObsFile(opt.calib_data, &calib_data, &n, &dim, &synthetic, &load_err)) {
      return fail("calibration data: " + load_err);
    }

    const int64_t net_in = volumeOf(network->getInput(0)->getDimensions());
    if (net_in > 0 && static_cast<int64_t>(dim) != net_in) {
      std::ostringstream os;
      os << "calibration data is " << dim << "-dim but the network takes " << net_in;
      return fail(os.str());
    }

    std::cout << "int8: " << n << " calibration samples x " << dim << " from "
              << opt.calib_data << (synthetic ? "  [SYNTHETIC]" : "  [recorded]") << "\n";
    if (synthetic) {
      // Not an error: a synthetic calibration is the right way to smoke-test
      // this path on a dev box. It is only wrong to report its numbers.
      std::cout
        << "       ^ R1FX inputs are Gaussian noise, not robot observations. INT8\n"
        << "         quantises ACTIVATION ranges, and noise drives activations the\n"
        << "         policy never reaches -- so this engine is a build smoke test,\n"
        << "         NOT the subject of the INT8 acceptance report.\n";
    }
    if (n < 256) {
      std::cout << "       ^ only " << n << " samples; entropy calibration wants a few\n"
                << "         hundred at minimum (>= 1000 preferred = 20 s at 50 Hz)\n";
    }

    if (opt.calib_minmax) {
      calibrator = std::make_unique<MinMaxObsCalibrator>(
        std::move(calib_data), n, dim, opt.calib_cache);
    } else {
      calibrator = std::make_unique<EntropyObsCalibrator>(
        std::move(calib_data), n, dim, opt.calib_cache);
    }
    if (!calibrator->ok()) {return fail("cudaMalloc for the calibration buffer failed");}
    std::cout << "int8: algorithm = "
              << (opt.calib_minmax ? "minmax (nothing clips; one outlier frame "
      "costs resolution everywhere)" : "entropy2 (robust to outliers; the tails "
      "may clip)") << "\n";

    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator->trt());
    if (!opt.calib_cache.empty()) {
      std::cout << "int8: calibration cache " << opt.calib_cache << "\n";
    }
  }

  // TensorRT enables kTF32 by default. TF32 rounds GEMM inputs to a 10-bit
  // mantissa -- the same mantissa width as FP16 -- so a nominally FP32 engine
  // can be off by ~1e-2, and because the flag merely *permits* TF32 the choice
  // is made by build-time kernel timing and is not stable across rebuilds.
  // Clear it unless explicitly asked for, so "fp32" means fp32.
  if (!opt.allow_tf32) {
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
  }

#if R1_TRT10
  TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
#else
  TrtPtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
#endif
  if (!plan) {return fail("buildSerializedNetwork returned null");}

  if (calibrator && calibrator->cache_was_reused()) {
    // The build read an existing table and never looked at calib_data. Silent,
    // this makes a fresh calibration set look like it changed nothing.
    std::cout
      << "int8: REUSED the existing calibration cache -- the calibration pass did\n"
      << "      NOT run and " << opt.calib_data << " was not used. Delete\n"
      << "      " << opt.calib_cache << " to recalibrate.\n";
  }

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
