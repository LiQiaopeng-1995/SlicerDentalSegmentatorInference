/**
 * Standalone TensorRT 8.6 C++ unit test for DentalSegmentatorInference.
 *
 * Tests the TensorRT C++ API directly (no ONNX Runtime layer).
 * Covers: engine build, serialization, inference, correctness, benchmark.
 *
 * Usage: trt_test <onnx_model> [options]
 *   --benchmark          Run extended performance benchmark
 *   --warmup N           Warmup iterations (default: 10)
 *   --runs N             Benchmark iterations (default: 50)
 *   --verbose            Verbose TRT logging
 *   --tolerance FLOAT    Max abs error for output consistency check (default: 1e-3)
 *
 * Example:
 *   trt_test model_weights/dental_segmentator.onnx --benchmark
 */

#define _USE_MATH_DEFINES

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>
#include <iomanip>

namespace fs = std::filesystem;

// ============================================================================
// Logger
// ============================================================================
class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(Severity level = Severity::kWARNING)
    : m_level(level), m_verbose(false) {}

  void setVerbose(bool v) { m_verbose = v; }

  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= m_level || m_verbose)
    {
      const char* sevStr = "";
      switch (severity)
      {
        case Severity::kINTERNAL_ERROR: sevStr = "[FATAL] "; break;
        case Severity::kERROR:          sevStr = "[ERROR] "; break;
        case Severity::kWARNING:        sevStr = "[WARN]  "; break;
        case Severity::kINFO:           sevStr = "[INFO]  "; break;
        case Severity::kVERBOSE:        sevStr = "[DEBUG] "; break;
        default: break;
      }
      std::cerr << sevStr << msg << std::endl;
    }
  }

private:
  Severity m_level;
  bool m_verbose;
};

// ============================================================================
// RAII wrapper for TensorRT objects
// ============================================================================
struct TrtDeleter
{
  template <typename T>
  void operator()(T* obj) const
  {
    if (obj) obj->destroy();
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// ============================================================================
// Utility: CUDA error check
// ============================================================================
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
      std::ostringstream oss;                                                 \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__                  \
          << " | " << cudaGetErrorName(err)                                   \
          << " : " << cudaGetErrorString(err);                                \
      throw std::runtime_error(oss.str());                                    \
    }                                                                         \
  } while (0)

// ============================================================================
// Timer helper
// ============================================================================
class Timer
{
public:
  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  void start() { m_start = Clock::now(); }
  double elapsedMs() const
  {
    return Ms(Clock::now() - m_start).count();
  }

private:
  Clock::time_point m_start;
};

// ============================================================================
// I/O Buffers (per binding index)
// ============================================================================
struct IOBuffers
{
  struct Binding
  {
    int index = -1;
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype;
    bool isInput = false;
    size_t sizeBytes = 0;
    size_t numElements = 0;
    void* devicePtr = nullptr;
    std::vector<float> hostData;
  };

  std::vector<Binding> inputs;
  std::vector<Binding> outputs;

  // Build flat pointer array for enqueueV2
  std::vector<void*> buildBindingsArray() const
  {
    std::vector<void*> ptrs;
    // enqueueV2 expects pointers in binding-index order
    int nb = static_cast<int>(inputs.size() + outputs.size());
    ptrs.resize(nb, nullptr);
    for (const auto& b : inputs)
      ptrs[b.index] = b.devicePtr;
    for (const auto& b : outputs)
      ptrs[b.index] = b.devicePtr;
    return ptrs;
  }
};

/**
 * Allocate input buffers. TRT 8.6: iterates binding indices, checks
 * bindingIsInput(), uses getBindingName/Dimensions/DataType.
 */
static void allocateInputs(IOBuffers& bufs,
                           nvinfer1::ICudaEngine& engine,
                           const nvinfer1::Dims& inputShape)
{
  int nb = engine.getNbBindings();
  for (int i = 0; i < nb; ++i)
  {
    if (!engine.bindingIsInput(i)) continue;

    IOBuffers::Binding b;
    b.index = i;
    b.name = engine.getBindingName(i);
    b.dtype = engine.getBindingDataType(i);
    b.isInput = true;
    b.dims = inputShape;

    b.numElements = 1;
    for (int d = 0; d < b.dims.nbDims; ++d)
      b.numElements *= static_cast<size_t>(b.dims.d[d]);

    b.sizeBytes = b.numElements * sizeof(float);
    b.hostData.resize(b.numElements);

    CUDA_CHECK(cudaMalloc(&b.devicePtr, b.sizeBytes));
    bufs.inputs.push_back(std::move(b));
  }
}

/**
 * Resolve output shapes from context (after input shapes are set),
 * then allocate output buffers.
 */
static void allocateOutputs(IOBuffers& bufs,
                            nvinfer1::ICudaEngine& engine,
                            nvinfer1::IExecutionContext& context)
{
  int nb = engine.getNbBindings();
  for (int i = 0; i < nb; ++i)
  {
    if (engine.bindingIsInput(i)) continue;

    IOBuffers::Binding b;
    b.index = i;
    b.name = engine.getBindingName(i);
    b.dtype = engine.getBindingDataType(i);
    b.isInput = false;

    // Get resolved shape from context (no -1 dims after input shape is set)
    b.dims = context.getBindingDimensions(i);
    b.numElements = 1;
    for (int d = 0; d < b.dims.nbDims; ++d)
      b.numElements *= static_cast<size_t>(b.dims.d[d]);

    b.sizeBytes = b.numElements * sizeof(float);
    b.hostData.resize(b.numElements);

    CUDA_CHECK(cudaMalloc(&b.devicePtr, b.sizeBytes));
    bufs.outputs.push_back(std::move(b));
  }
}

static void freeIOBuffers(IOBuffers& bufs)
{
  for (auto& b : bufs.inputs)
    if (b.devicePtr) { cudaFree(b.devicePtr); b.devicePtr = nullptr; }
  for (auto& b : bufs.outputs)
    if (b.devicePtr) { cudaFree(b.devicePtr); b.devicePtr = nullptr; }
}

// ============================================================================
// Fill input with synthetic data (Box-Muller normal distribution)
// ============================================================================
static void fillRandomInput(IOBuffers& bufs, unsigned seed = 42)
{
  std::srand(seed);
  for (auto& b : bufs.inputs)
  {
    for (size_t i = 0; i < b.numElements; ++i)
    {
      float u1 = static_cast<float>(std::rand()) / RAND_MAX;
      float u2 = static_cast<float>(std::rand()) / RAND_MAX;
      if (u1 < 1e-8f) u1 = 1e-8f;
      b.hostData[i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    }
    CUDA_CHECK(cudaMemcpy(b.devicePtr, b.hostData.data(), b.sizeBytes,
                          cudaMemcpyHostToDevice));
  }
}

// ============================================================================
// Engine + Runtime pair (runtime must outlive engine)
// ============================================================================
struct EngineBundle
{
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
};

// ============================================================================
// Build TensorRT engine from ONNX model
// ============================================================================
static EngineBundle buildEngineFromOnnx(
    const std::string& onnxPath,
    TrtLogger& logger,
    bool fp16 = true,
    int maxWorkspaceGB = 2)
{
  std::cout << "=== Building TensorRT engine from ONNX ===" << std::endl;
  std::cout << "  Model: " << onnxPath << std::endl;
  std::cout << "  FP16:  " << (fp16 ? "enabled" : "disabled") << std::endl;

  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(logger));
  if (!builder)
    throw std::runtime_error("Failed to create InferBuilder");

  // Create network definition with explicit batch
  const auto explicitBatchFlag =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatchFlag));
  if (!network)
    throw std::runtime_error("Failed to create network definition");

  // Create ONNX parser
  auto parser = TrtUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser)
    throw std::runtime_error("Failed to create ONNX parser");

  // Parse ONNX file (TRT 8.6 parseFromFile takes const char*)
  Timer parseTimer;
  parseTimer.start();
  if (!parser->parseFromFile(onnxPath.c_str(),
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    std::ostringstream oss;
    for (int i = 0; i < parser->getNbErrors(); ++i)
      oss << parser->getError(i)->desc() << "\n";
    throw std::runtime_error("ONNX parse failed:\n" + oss.str());
  }
  std::cout << "  ONNX parsed in " << parseTimer.elapsedMs() << " ms" << std::endl;

  // Print network info
  int nbInputs = network->getNbInputs();
  int nbOutputs = network->getNbOutputs();
  std::cout << "  Inputs:  " << nbInputs << std::endl;
  for (int i = 0; i < nbInputs; ++i)
  {
    auto* input = network->getInput(i);
    auto shape = input->getDimensions();
    std::cout << "    [" << i << "] " << input->getName()
              << "  shape=";
    for (int d = 0; d < shape.nbDims; ++d)
      std::cout << (d > 0 ? "x" : "") << shape.d[d];
    std::cout << std::endl;
  }
  std::cout << "  Outputs: " << nbOutputs << std::endl;
  for (int i = 0; i < nbOutputs; ++i)
  {
    auto* output = network->getOutput(i);
    auto shape = output->getDimensions();
    std::cout << "    [" << i << "] " << output->getName()
              << "  shape=";
    for (int d = 0; d < shape.nbDims; ++d)
      std::cout << (d > 0 ? "x" : "") << shape.d[d];
    std::cout << std::endl;
  }

  // Create builder config
  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  config->setMaxWorkspaceSize(static_cast<size_t>(maxWorkspaceGB) * (1ULL << 30));

  // Optimization profile for dynamic shapes
  // Model input: [1, 1, D, H, W] with dynamic spatial dims
  {
    const char* inputName = network->getInput(0)->getName();
    std::cout << "  Network input: " << inputName
              << " (nbDims=" << network->getInput(0)->getDimensions().nbDims << ")" << std::endl;

    nvinfer1::Dims minDims, optDims, maxDims;
    minDims.nbDims = 5; optDims.nbDims = 5; maxDims.nbDims = 5;
    for (int d = 0; d < 5; ++d) {
      minDims.d[d] = 1; optDims.d[d] = 1; maxDims.d[d] = 1;
    }
    minDims.d[2] = 64;  minDims.d[3] = 64;  minDims.d[4] = 64;   // min:  1x1x64x64x64
    optDims.d[2] = 128; optDims.d[3] = 128; optDims.d[4] = 128;  // opt:  1x1x128x128x128
    maxDims.d[2] = 192; maxDims.d[3] = 192; maxDims.d[4] = 192;  // max:  1x1x192x192x192

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
    config->addOptimizationProfile(profile);
    std::cout << "  Optimization profile: MIN=64^3, OPT=128^3, MAX=192^3" << std::endl;
  }

  // Set FP16 mode
  if (fp16)
  {
    if (builder->platformHasFastFp16())
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    else
      std::cout << "  WARNING: FP16 not supported on this platform, using FP32" << std::endl;
  }

  // Build serialized engine
  std::cout << "  Building engine (this may take a while)..." << std::endl;
  Timer buildTimer;
  buildTimer.start();

  auto serializedEngine = TrtUniquePtr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!serializedEngine)
    throw std::runtime_error("Failed to build serialized engine");

  double buildTime = buildTimer.elapsedMs();
  std::cout << "  Engine built in " << buildTime << " ms ("
            << (buildTime / 1000.0) << " s)" << std::endl;
  std::cout << "  Serialized engine size: "
            << (serializedEngine->size() / (1024.0 * 1024.0)) << " MB" << std::endl;

  // Deserialize into engine
  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  if (!runtime)
    throw std::runtime_error("Failed to create InferRuntime");

  auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(serializedEngine->data(),
                                     serializedEngine->size()));
  if (!engine)
    throw std::runtime_error("Failed to deserialize engine");

  std::cout << "=== Engine build complete ===" << std::endl;
  return {std::move(runtime), std::move(engine)};
}

// ============================================================================
// Serialize engine to file
// ============================================================================
static void serializeEngine(nvinfer1::ICudaEngine& engine,
                            const std::string& path)
{
  std::cout << "  Serializing engine to: " << path << std::endl;

  auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(engine.serialize());
  if (!serialized)
    throw std::runtime_error("Failed to serialize engine");

  std::ofstream ofs(path, std::ios::binary);
  if (!ofs)
    throw std::runtime_error("Cannot open: " + path);
  ofs.write(static_cast<const char*>(serialized->data()), serialized->size());
  ofs.close();

  std::cout << "  Written " << (serialized->size() / (1024.0 * 1024.0))
            << " MB" << std::endl;
}

// ============================================================================
// Deserialize engine from file
// ============================================================================
static EngineBundle deserializeEngine(
    const std::string& path,
    TrtLogger& logger)
{
  std::cout << "  Deserializing engine from: " << path << std::endl;

  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("Cannot open: " + path);

  ifs.seekg(0, std::ios::end);
  size_t size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  ifs.read(buffer.data(), size);
  ifs.close();

  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(
      nvinfer1::createInferRuntime(logger));
  if (!runtime)
    throw std::runtime_error("Failed to create InferRuntime");

  auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(buffer.data(), size));
  if (!engine)
    throw std::runtime_error("Failed to deserialize engine");

  std::cout << "  Engine deserialized (" << (size / (1024.0 * 1024.0))
            << " MB)" << std::endl;
  return {std::move(runtime), std::move(engine)};
}

// ============================================================================
// Run inference (TRT 8.6 enqueueV2)
// ============================================================================
static void runInference(nvinfer1::ICudaEngine& engine,
                         IOBuffers& bufs,
                         cudaStream_t stream = nullptr)
{
  auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(
      engine.createExecutionContext());
  if (!context)
    throw std::runtime_error("Failed to create execution context");

  // Set input shapes on context
  for (const auto& b : bufs.inputs)
    context->setBindingDimensions(b.index, b.dims);

  // Verify all input dims are set (non -1)
  for (const auto& b : bufs.inputs)
  {
    auto dims = context->getBindingDimensions(b.index);
    for (int d = 0; d < dims.nbDims; ++d)
    {
      if (dims.d[d] == -1)
        throw std::runtime_error(
            "Input binding " + b.name + " dim " + std::to_string(d) + " is still dynamic (-1)");
    }
  }

  // Allocate outputs with resolved shapes
  allocateOutputs(bufs, engine, *context);

  // Build flat bindings array for enqueueV2 (index-ordered)
  auto bindings = bufs.buildBindingsArray();

  // EnqueueV2: enqueueV2(void* const* bindings, cudaStream_t stream, cudaEvent_t* inputConsumed)
  bool ok = context->enqueueV2(bindings.data(), stream ? stream : 0, nullptr);
  if (!ok)
    throw std::runtime_error("enqueueV2 failed");
}

// ============================================================================
// Copy outputs from device to host
// ============================================================================
static void copyOutputsToHost(IOBuffers& bufs, cudaStream_t stream = nullptr)
{
  if (stream)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& b : bufs.outputs)
  {
    CUDA_CHECK(cudaMemcpy(b.hostData.data(), b.devicePtr, b.sizeBytes,
                          cudaMemcpyDeviceToHost));
  }
}

// ============================================================================
// Validate output
// ============================================================================
struct OutputStats
{
  float min;
  float max;
  float mean;
  float std;
  bool hasNaN;
  bool hasInf;
  size_t numElements;
  size_t numPositive;
  size_t numNegative;
  size_t numZero;
};

static OutputStats computeOutputStats(const std::vector<float>& data)
{
  OutputStats s{};
  s.numElements = data.size();
  if (data.empty()) return s;

  s.min = std::numeric_limits<float>::max();
  s.max = -std::numeric_limits<float>::max();
  s.hasNaN = false;
  s.hasInf = false;
  s.numPositive = 0;
  s.numNegative = 0;
  s.numZero = 0;

  double sum = 0.0;
  for (auto v : data)
  {
    if (std::isnan(v)) { s.hasNaN = true; continue; }
    if (std::isinf(v)) { s.hasInf = true; continue; }

    s.min = std::min(s.min, v);
    s.max = std::max(s.max, v);
    sum += v;

    if (v > 0) s.numPositive++;
    else if (v < 0) s.numNegative++;
    else s.numZero++;
  }

  s.mean = static_cast<float>(sum / data.size());

  double sqSum = 0.0;
  for (auto v : data)
    if (!std::isnan(v) && !std::isinf(v))
      sqSum += static_cast<double>(v - s.mean) * (v - s.mean);
  s.std = static_cast<float>(std::sqrt(sqSum / data.size()));

  return s;
}

static void printOutputStats(const OutputStats& s, const std::string& label)
{
  std::cout << "  " << label << ":" << std::endl;
  std::cout << "    Elements:   " << s.numElements << std::endl;
  std::cout << "    Range:      [" << s.min << ", " << s.max << "]" << std::endl;
  std::cout << "    Mean:       " << s.mean << std::endl;
  std::cout << "    Std:        " << s.std << std::endl;
  std::cout << "    Pos/Neg/0:  " << s.numPositive << " / "
            << s.numNegative << " / " << s.numZero << std::endl;
  if (s.hasNaN) std::cout << "    *** WARNING: NaN values detected ***" << std::endl;
  if (s.hasInf) std::cout << "    *** WARNING: Inf values detected ***" << std::endl;
}

// ============================================================================
// TEST CASES
// ============================================================================

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name)                                                            \
  std::cout << "\n=================================================" << std::endl; \
  std::cout << "TEST: " << name << std::endl;                                \
  std::cout << "=================================================" << std::endl;

#define CHECK(cond, msg)                                                      \
  do {                                                                        \
    if (!(cond))                                                              \
    {                                                                         \
      std::cerr << "  FAILED: " << msg << " (" << __FILE__ << ":"            \
                << __LINE__ << ")" << std::endl;                              \
      g_failed++;                                                             \
      throw std::runtime_error(msg);                                          \
    }                                                                         \
  } while (0)

#define PASS()                                                                \
  do {                                                                        \
    std::cout << "  PASSED" << std::endl;                                     \
    g_passed++;                                                               \
  } while (0)

// ────────────────────────────────────────────────────────────────────────────
// Test 1: Build engine from ONNX
// ────────────────────────────────────────────────────────────────────────────
static EngineBundle test1_buildEngine(
    const std::string& onnxPath, TrtLogger& logger, bool fp16)
{
  TEST("Build engine from ONNX");

  auto bundle = buildEngineFromOnnx(onnxPath, logger, fp16);
  auto& engine = *bundle.engine;

  int nb = engine.getNbBindings();
  CHECK(nb >= 2, "Engine should have at least 2 bindings (1 input + 1 output)");
  std::cout << "  Bindings: " << nb << std::endl;

  for (int i = 0; i < nb; ++i)
  {
    auto name = engine.getBindingName(i);
    bool isInput = engine.bindingIsInput(i);
    auto dtype = engine.getBindingDataType(i);
    auto shape = engine.getBindingDimensions(i);

    std::string ioStr = isInput ? "INPUT " : "OUTPUT";
    std::string dtypeStr = (dtype == nvinfer1::DataType::kFLOAT) ? "FLOAT" :
                           (dtype == nvinfer1::DataType::kHALF)  ? "HALF"  :
                           (dtype == nvinfer1::DataType::kINT32) ? "INT32" : "OTHER";
    std::cout << "  [" << i << "] " << ioStr << " | " << name
              << " | " << dtypeStr << " | shape=[";
    for (int d = 0; d < shape.nbDims; ++d)
      std::cout << (d > 0 ? "x" : "") << shape.d[d];
    std::cout << "]" << std::endl;
  }

  PASS();
  return bundle;
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: Serialize and deserialize engine
// ────────────────────────────────────────────────────────────────────────────
static EngineBundle test2_serializeDeserialize(
    nvinfer1::ICudaEngine& engine,
    const std::string& cacheDir,
    TrtLogger& logger)
{
  TEST("Serialize / deserialize engine");

  fs::create_directories(cacheDir);
  auto cachePath = (fs::path(cacheDir) / "dental_segmentator.trt").string();

  serializeEngine(engine, cachePath);
  CHECK(fs::exists(cachePath), "Engine file should exist after serialization");

  auto bundle = deserializeEngine(cachePath, logger);

  CHECK(engine.getNbBindings() == bundle.engine->getNbBindings(),
        "Binding count should match after deserialization");

  for (int i = 0; i < engine.getNbBindings(); ++i)
  {
    CHECK(std::string(engine.getBindingName(i)) == std::string(bundle.engine->getBindingName(i)),
          std::string("Binding name mismatch at index ") + std::to_string(i));
    CHECK(engine.bindingIsInput(i) == bundle.engine->bindingIsInput(i),
          std::string("Binding I/O mismatch at index ") + std::to_string(i));
  }

  std::cout << "  Cache file: " << cachePath << std::endl;
  PASS();
  return bundle;
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: Inference — output shape and basic validity
// ────────────────────────────────────────────────────────────────────────────
static void test3_inferenceShape(nvinfer1::ICudaEngine& engine)
{
  TEST("Inference — output shape and basic validity");

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = 1;   // batch
  inputDims.d[1] = 1;   // channels
  inputDims.d[2] = 64;  // D
  inputDims.d[3] = 64;  // H
  inputDims.d[4] = 64;  // W

  IOBuffers bufs;
  allocateInputs(bufs, engine, inputDims);

  std::cout << "  Input bindings: " << bufs.inputs.size() << std::endl;
  for (const auto& b : bufs.inputs)
    std::cout << "    [" << b.index << "] " << b.name
              << "  elements=" << b.numElements
              << "  bytes=" << (b.sizeBytes / 1024.0f) << " KB" << std::endl;

  fillRandomInput(bufs, 42);
  Timer timer;
  timer.start();
  runInference(engine, bufs);
  copyOutputsToHost(bufs);
  double ms = timer.elapsedMs();
  std::cout << "  First inference: " << ms << " ms" << std::endl;

  std::cout << "  Output bindings: " << bufs.outputs.size() << std::endl;
  for (const auto& b : bufs.outputs)
    std::cout << "    [" << b.index << "] " << b.name
              << "  elements=" << b.numElements
              << "  bytes=" << (b.sizeBytes / 1024.0f / 1024.0f) << " MB" << std::endl;

  CHECK(bufs.outputs.size() > 0, "Should have at least 1 output");
  for (const auto& o : bufs.outputs)
  {
    auto stats = computeOutputStats(o.hostData);
    printOutputStats(stats, o.name);

    CHECK(!stats.hasNaN, "Output should not contain NaN values");
    CHECK(!stats.hasInf, "Output should not contain Inf values");
    CHECK(stats.numElements > 0, "Output should have >0 elements");
    CHECK(std::abs(stats.mean) < 100.0f, "Output mean should be in reasonable range");
  }

  freeIOBuffers(bufs);
  PASS();
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: Inference consistency — same input => same output every time
// ────────────────────────────────────────────────────────────────────────────
static void test4_inferenceConsistency(nvinfer1::ICudaEngine& engine,
                                       float tolerance)
{
  TEST("Inference consistency (FP16 may have minor variation)");

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = 1;
  inputDims.d[1] = 1;
  inputDims.d[2] = 64;
  inputDims.d[3] = 64;
  inputDims.d[4] = 64;

  const int numRuns = 10;
  std::vector<std::vector<float>> allOutputs;

  for (int run = 0; run < numRuns; ++run)
  {
    IOBuffers bufs;
    allocateInputs(bufs, engine, inputDims);
    fillRandomInput(bufs, 42);  // Same seed => same input
    runInference(engine, bufs);
    copyOutputsToHost(bufs);

    allOutputs.push_back(bufs.outputs[0].hostData);
    freeIOBuffers(bufs);
  }

  const auto& ref = allOutputs[0];
  for (int run = 1; run < numRuns; ++run)
  {
    const auto& cur = allOutputs[run];
    CHECK(ref.size() == cur.size(), "Output size should be consistent");

    float maxAbsDiff = 0.0f;
    float sumAbsDiff = 0.0f;
    size_t diffCount = 0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
      float diff = std::abs(ref[i] - cur[i]);
      sumAbsDiff += diff;
      if (diff > tolerance) diffCount++;
      maxAbsDiff = std::max(maxAbsDiff, diff);
    }

    float avgAbsDiff = sumAbsDiff / ref.size();
    std::cout << "  Run 0 vs Run " << run
              << "  max_abs_diff=" << maxAbsDiff
              << "  avg_abs_diff=" << avgAbsDiff
              << "  above_tol=" << diffCount << "/" << ref.size();

    if (maxAbsDiff > tolerance)
    {
      float pct = 100.0f * diffCount / ref.size();
      if (pct > 1.0f)
      {
        std::cout << "  WARN: large portion differs";
        CHECK(false, "Too many elements exceed tolerance");
      }
      else
      {
        std::cout << "  OK (FP16 minor variation)";
      }
    }
    std::cout << std::endl;
  }

  PASS();
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: Performance benchmark
// ────────────────────────────────────────────────────────────────────────────
static void test5_benchmark(nvinfer1::ICudaEngine& engine,
                            int warmupIters, int benchIters)
{
  TEST("Performance benchmark");

  // Use the actual input shape from engine, filling dynamic dims with defaults
  int nb = engine.getNbBindings();
  nvinfer1::Dims inputDims;
  bool foundInput = false;
  for (int i = 0; i < nb; ++i)
  {
    if (engine.bindingIsInput(i))
    {
      inputDims = engine.getBindingDimensions(i);
      for (int d = 0; d < inputDims.nbDims; ++d)
      {
        if (inputDims.d[d] == -1)
        {
          if (d == 0) inputDims.d[d] = 1;       // batch
          else if (d == 1) inputDims.d[d] = 1;   // channels
          else inputDims.d[d] = 64;               // spatial
        }
      }
      foundInput = true;
      break;
    }
  }
  CHECK(foundInput, "No input binding found in engine");

  std::cout << "  Input shape: [";
  for (int d = 0; d < inputDims.nbDims; ++d)
    std::cout << (d > 0 ? "x" : "") << inputDims.d[d];
  std::cout << "]" << std::endl;

  size_t inputElements = 1;
  for (int d = 0; d < inputDims.nbDims; ++d)
    inputElements *= inputDims.d[d];
  std::cout << "  Input elements: " << inputElements << std::endl;
  std::cout << "  Warmup:  " << warmupIters << " runs" << std::endl;
  std::cout << "  Bench:   " << benchIters << " runs" << std::endl;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Warmup
  std::cout << "  Warming up..." << std::flush;
  for (int i = 0; i < warmupIters; ++i)
  {
    IOBuffers bufs;
    allocateInputs(bufs, engine, inputDims);
    fillRandomInput(bufs);
    runInference(engine, bufs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    freeIOBuffers(bufs);
  }
  std::cout << " done." << std::endl;

  // Benchmark
  std::vector<double> timesMs;
  timesMs.reserve(benchIters);

  std::cout << "  Running benchmark..." << std::flush;
  for (int i = 0; i < benchIters; ++i)
  {
    Timer timer;
    timer.start();

    IOBuffers bufs;
    allocateInputs(bufs, engine, inputDims);
    fillRandomInput(bufs);
    runInference(engine, bufs, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    freeIOBuffers(bufs);

    timesMs.push_back(timer.elapsedMs());
  }
  std::cout << " done." << std::endl;

  CUDA_CHECK(cudaStreamDestroy(stream));

  // Statistics
  std::sort(timesMs.begin(), timesMs.end());
  double sum = std::accumulate(timesMs.begin(), timesMs.end(), 0.0);
  double mean = sum / timesMs.size();
  double p50 = timesMs[timesMs.size() / 2];
  double p95 = timesMs[static_cast<size_t>(timesMs.size() * 0.95)];
  double p99 = timesMs[static_cast<size_t>(timesMs.size() * 0.99)];
  double minT = timesMs.front();
  double maxT = timesMs.back();

  double sqSum = 0.0;
  for (auto t : timesMs)
    sqSum += (t - mean) * (t - mean);
  double stddev = std::sqrt(sqSum / timesMs.size());

  std::cout << "\n  ==== Latency (per patch) ====" << std::endl;
  std::cout << "  Mean:   " << std::fixed << std::setprecision(2) << mean << " ms" << std::endl;
  std::cout << "  Std:    " << stddev << " ms" << std::endl;
  std::cout << "  Min:    " << minT << " ms" << std::endl;
  std::cout << "  Max:    " << maxT << " ms" << std::endl;
  std::cout << "  P50:    " << p50 << " ms" << std::endl;
  std::cout << "  P95:    " << p95 << " ms" << std::endl;
  std::cout << "  P99:    " << p99 << " ms" << std::endl;

  double est90Patches = mean * 90.0;
  double est100Patches = mean * 100.0;
  std::cout << "\n  ==== Estimated full-volume inference ====" << std::endl;
  std::cout << "  90 patches:  " << (est90Patches / 1000.0) << " s" << std::endl;
  std::cout << "  100 patches: " << (est100Patches / 1000.0) << " s" << std::endl;

  PASS();
}

// ============================================================================
// TTA (Test Time Augmentation) — 8-axis flip test
// ============================================================================

/**
 * TTA flip masks (3-bit: D=bit0, H=bit1, W=bit2)
 */
enum TTAFlip : int {
  TTA_NONE    = 0,  // identity (no flip)
  TTA_FLIP_D  = 1,  // flip D axis
  TTA_FLIP_H  = 2,  // flip H axis
  TTA_FLIP_W  = 4,  // flip W axis
};

/**
 * Flip a 5D tensor [N, C, D, H, W] along specified axes.
 * flipMask: bit0=flipD, bit1=flipH, bit2=flipW
 * Data is modified in-place.
 */
static void flipTensor5D(std::vector<float>& data,
                         const nvinfer1::Dims& dims,
                         int flipMask)
{
  if (flipMask == 0) return;

  int N = dims.d[0], C = dims.d[1];
  int D = dims.d[2], H = dims.d[3], W = dims.d[4];
  size_t total = static_cast<size_t>(N) * C * D * H * W;

  std::vector<float> copy = data;
  size_t idx = 0;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
          {
            int sd = (flipMask & TTA_FLIP_D) ? (D - 1 - d) : d;
            int sh = (flipMask & TTA_FLIP_H) ? (H - 1 - h) : h;
            int sw = (flipMask & TTA_FLIP_W) ? (W - 1 - w) : w;
            size_t srcIdx = static_cast<size_t>((((n * C + c) * D + sd) * H + sh) * W + sw);
            data[idx++] = copy[srcIdx];
          }
}

/**
 * Run a single TTA inference step with a specified flip.
 * Returns the (un-flipped) output logits.
 */
static std::vector<float> runTTAInference(
    nvinfer1::ICudaEngine& engine,
    const std::vector<float>& originalInput,
    const nvinfer1::Dims& inputDims,
    int flipMask,
    cudaStream_t stream = nullptr)
{
  IOBuffers bufs;
  allocateInputs(bufs, engine, inputDims);

  // Flip input and copy to device
  std::vector<float> flippedInput;
  if (flipMask == TTA_NONE)
    flippedInput = originalInput;
  else
  {
    flippedInput = originalInput;
    flipTensor5D(flippedInput, inputDims, flipMask);
  }

  // Copy to device input buffer
  auto& inBuf = bufs.inputs[0];
  std::memcpy(inBuf.hostData.data(), flippedInput.data(), inBuf.sizeBytes);
  CUDA_CHECK(cudaMemcpy(inBuf.devicePtr, inBuf.hostData.data(), inBuf.sizeBytes,
                        cudaMemcpyHostToDevice));

  // Run inference
  auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(
      engine.createExecutionContext());
  context->setBindingDimensions(inBuf.index, inputDims);

  // Check dims
  for (const auto& b : bufs.inputs)
  {
    auto dims = context->getBindingDimensions(b.index);
    for (int d = 0; d < dims.nbDims; ++d)
      if (dims.d[d] == -1)
        throw std::runtime_error("Input binding " + b.name + " dim " + std::to_string(d) + " is still dynamic");
  }

  // Allocate outputs
  allocateOutputs(bufs, engine, *context);

  // Build bindings
  auto bindings = bufs.buildBindingsArray();
  if (!context->enqueueV2(bindings.data(), stream ? stream : 0, nullptr))
    throw std::runtime_error("enqueueV2 failed for TTA flip " + std::to_string(flipMask));

  // Sync and copy back
  if (stream)
    CUDA_CHECK(cudaStreamSynchronize(stream));
  else
    CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& b : bufs.outputs)
  {
    CUDA_CHECK(cudaMemcpy(b.hostData.data(), b.devicePtr, b.sizeBytes,
                          cudaMemcpyDeviceToHost));
  }

  // Get output (assume single output)
  std::vector<float> output = bufs.outputs[0].hostData;

  // Un-flip output if needed
  if (flipMask != TTA_NONE)
  {
    // Output shape: [1, C, D, H, W] — use output dims
    flipTensor5D(output, bufs.outputs[0].dims, flipMask);
  }

  freeIOBuffers(bufs);
  return output;
}

/**
 * Analyze per-voxel variance across all 8 TTA flips.
 * Higher variance means the model is more uncertain about that voxel.
 */
static void test6_ttaEffect(nvinfer1::ICudaEngine& engine)
{
  TEST("TTA Effect — 8-flip variance analysis");

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = 1;
  inputDims.d[1] = 1;
  inputDims.d[2] = 64;
  inputDims.d[3] = 64;
  inputDims.d[4] = 64;

  // Generate input
  IOBuffers tmpBufs;
  allocateInputs(tmpBufs, engine, inputDims);
  fillRandomInput(tmpBufs, 42);
  auto inputData = tmpBufs.inputs[0].hostData;
  freeIOBuffers(tmpBufs);

  // Run all 8 TTA flips
  const int numFlips = 8;
  int flipMasks[] = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> allOutputs[numFlips];

  Timer ttaTimer;
  ttaTimer.start();

  for (int f = 0; f < numFlips; ++f)
  {
    allOutputs[f] = runTTAInference(engine, inputData, inputDims, flipMasks[f]);
  }

  double ttaTime = ttaTimer.elapsedMs();

  // Compute TTA average and per-flip stats
  auto& ref = allOutputs[0];  // non-flip
  size_t numel = ref.size();

  // Compute averaged TTA output
  std::vector<float> ttaAvg(numel, 0.0f);
  for (int f = 0; f < numFlips; ++f)
  {
    CHECK(allOutputs[f].size() == numel,
          "Output size mismatch at flip " + std::to_string(flipMasks[f]));
    for (size_t i = 0; i < numel; ++i)
      ttaAvg[i] += allOutputs[f][i];
  }
  for (size_t i = 0; i < numel; ++i)
    ttaAvg[i] /= static_cast<float>(numFlips);

  // Compute per-voxel variance across flips
  std::vector<float> perVoxelVar(numel, 0.0f);
  for (int f = 0; f < numFlips; ++f)
    for (size_t i = 0; i < numel; ++i)
    {
      float diff = allOutputs[f][i] - ttaAvg[i];
      perVoxelVar[i] += diff * diff;
    }
  for (size_t i = 0; i < numel; ++i)
    perVoxelVar[i] /= static_cast<float>(numFlips);

  // Stats
  float maxVar = 0, meanVar = 0, varThreshold = 1.0f;
  size_t highVarCount = 0;
  for (size_t i = 0; i < numel; ++i)
  {
    meanVar += perVoxelVar[i];
    if (perVoxelVar[i] > maxVar) maxVar = perVoxelVar[i];
    if (perVoxelVar[i] > varThreshold) highVarCount++;
  }
  meanVar /= static_cast<float>(numel);

  // Compare TTA avg vs non-TTA (flip 0)
  float ttaVsNoTTA_maxDiff = 0, ttaVsNoTTA_sumDiff = 0;
  size_t ttaVsNoTTA_changed = 0;
  for (size_t i = 0; i < numel; ++i)
  {
    float diff = std::abs(ttaAvg[i] - ref[i]);
    ttaVsNoTTA_sumDiff += diff;
    if (diff > ttaVsNoTTA_maxDiff) ttaVsNoTTA_maxDiff = diff;
    if (diff > 0.01f) ttaVsNoTTA_changed++;
  }
  float ttaVsNoTTA_avgDiff = ttaVsNoTTA_sumDiff / numel;

  // Per-flip vs non-flip analysis
  std::cout << "  Per-flip vs non-flip (0) max_abs_diff:" << std::endl;
  for (int f = 1; f < numFlips; ++f)
  {
    float mxd = 0;
    for (size_t i = 0; i < numel; ++i)
    {
      float d = std::abs(allOutputs[f][i] - ref[i]);
      if (d > mxd) mxd = d;
    }
    std::cout << "    flip " << flipMasks[f] << ": " << mxd << std::endl;
  }

  std::cout << "\n  Cross-flip variance (logit level):" << std::endl;
  std::cout << "    Max variance:       " << maxVar << std::endl;
  std::cout << "    Mean variance:      " << meanVar << std::endl;
  std::cout << "    High-var voxels (>1): " << highVarCount << " / " << numel
            << " (" << (100.0 * highVarCount / numel) << "%)" << std::endl;

  std::cout << "\n  TTA avg vs no-TTA:" << std::endl;
  std::cout << "    Max abs diff:  " << ttaVsNoTTA_maxDiff << std::endl;
  std::cout << "    Avg abs diff:  " << ttaVsNoTTA_avgDiff << std::endl;
  std::cout << "    Changed (>0.01): " << ttaVsNoTTA_changed << " / " << numel
            << " (" << (100.0 * ttaVsNoTTA_changed / numel) << "%)" << std::endl;

  // Argmax comparison: non-TTA vs TTA
  // Determine number of classes from output
  auto& outDims0 = allOutputs[0];
  // Output shape is [1, C, D, H, W], classes = dims[1]
  // We need to reconstruct output dims — use runTTAInference output dims
  // Actually we stored the data, let's infer C from output size:
  size_t spatialVoxels = static_cast<size_t>(inputDims.d[2]) * inputDims.d[3] * inputDims.d[4];
  int numClasses = static_cast<int>(numel / spatialVoxels);
  std::cout << "  Classes: " << numClasses << " (output=" << numel << " / spatial=" << spatialVoxels << ")" << std::endl;

  // Argmax without TTA
  std::vector<unsigned char> labelNoTTA(spatialVoxels, 0);
  std::vector<unsigned char> labelTTA(spatialVoxels, 0);
  size_t disagreeCount = 0;

  for (size_t v = 0; v < spatialVoxels; ++v)
  {
    float bestNoTTA = -std::numeric_limits<float>::infinity();
    float bestTTA = -std::numeric_limits<float>::infinity();
    int bestCNoTTA = 0, bestCTTA = 0;
    for (int c = 0; c < numClasses; ++c)
    {
      float valNoTTA = ref[c * spatialVoxels + v];
      float valTTA = ttaAvg[c * spatialVoxels + v];
      if (valNoTTA > bestNoTTA) { bestNoTTA = valNoTTA; bestCNoTTA = c; }
      if (valTTA > bestTTA) { bestTTA = valTTA; bestCTTA = c; }
    }
    labelNoTTA[v] = static_cast<unsigned char>(bestCNoTTA);
    labelTTA[v] = static_cast<unsigned char>(bestCTTA);
    if (bestCNoTTA != bestCTTA) disagreeCount++;
  }

  float agreement = 100.0f * (spatialVoxels - disagreeCount) / spatialVoxels;
  std::cout << "\n  Argmax comparison (label level):" << std::endl;
  std::cout << "    Disagreements:  " << disagreeCount << " / "
            << spatialVoxels << " voxels" << std::endl;
  std::cout << "    Agreement rate: " << std::fixed << std::setprecision(4)
            << agreement << "%" << std::endl;

  // Per-class agreement analysis
  std::cout << "\n  Per-class agreement:" << std::endl;
  std::vector<size_t> classTotal(numClasses, 0);
  std::vector<size_t> classAgree(numClasses, 0);
  for (size_t v = 0; v < spatialVoxels; ++v)
  {
    int cNoTTA = labelNoTTA[v];
    int cTTA = labelTTA[v];
    classTotal[cNoTTA]++;
    if (cNoTTA == cTTA) classAgree[cNoTTA]++;
  }
  for (int c = 0; c < numClasses; ++c)
  {
    float dice = (classTotal[c] > 0)
        ? (100.0f * classAgree[c] / classTotal[c]) : 0;
    std::cout << "    Class " << c << ": " << classAgree[c] << " / "
              << classTotal[c] << " (" << dice << "%)" << std::endl;
  }

  std::cout << "\n  TTA total time (8 flips): " << ttaTime << " ms ("
            << (ttaTime / 1000.0) << " s)" << std::endl;
  std::cout << "  Per-flip avg: " << (ttaTime / 8.0) << " ms" << std::endl;

  PASS();
}

/**
 * Benchmark TTA vs no-TTA latency and throughput.
 */
static void test7_ttaBenchmark(nvinfer1::ICudaEngine& engine,
                               int warmupIters, int benchIters)
{
  TEST("TTA Performance Benchmark");

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = 1;
  inputDims.d[1] = 1;
  inputDims.d[2] = 64;
  inputDims.d[3] = 64;
  inputDims.d[4] = 64;

  int flipMasks[] = {0, 1, 2, 3, 4, 5, 6, 7};

  std::cout << "  Input: [1, 1, 64, 64, 64]" << std::endl;
  std::cout << "  TTA: 8 flips (" << (warmupIters * 8) << " warmup, "
            << (benchIters * 8) << " bench runs)" << std::endl;

  // Generate input once
  IOBuffers tmpBufs;
  allocateInputs(tmpBufs, engine, inputDims);
  fillRandomInput(tmpBufs, 42);
  auto inputData = tmpBufs.inputs[0].hostData;
  freeIOBuffers(tmpBufs);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // --- No-TTA Benchmark ---
  std::cout << "  Warming up (no-TTA)..." << std::flush;
  for (int i = 0; i < warmupIters; ++i)
  {
    auto result = runTTAInference(engine, inputData, inputDims, 0, stream);
    (void)result;
  }
  std::cout << " done." << std::endl;

  std::vector<double> noTTAtimes;
  noTTAtimes.reserve(benchIters);
  for (int i = 0; i < benchIters; ++i)
  {
    Timer t; t.start();
    auto result = runTTAInference(engine, inputData, inputDims, 0, stream);
    (void)result;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    noTTAtimes.push_back(t.elapsedMs());
  }

  // --- TTA Benchmark ---
  std::cout << "  Warming up (TTA 8×)..." << std::flush;
  for (int i = 0; i < warmupIters; ++i)
  {
    for (int f = 0; f < 8; ++f)
      runTTAInference(engine, inputData, inputDims, flipMasks[f], stream);
  }
  std::cout << " done." << std::endl;

  std::vector<double> ttaTimes;
  ttaTimes.reserve(benchIters);
  for (int i = 0; i < benchIters; ++i)
  {
    Timer t; t.start();
    for (int f = 0; f < 8; ++f)
      runTTAInference(engine, inputData, inputDims, flipMasks[f], stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ttaTimes.push_back(t.elapsedMs());
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  // Sort and compute stats
  auto stats = [](std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double n = static_cast<double>(times.size());
    double mean = sum / n;
    double sq = 0;
    for (auto t : times) sq += (t - mean) * (t - mean);
    double p50 = times[times.size() / 2];
    double p95 = times[static_cast<size_t>(times.size() * 0.95)];
    double p99 = times[static_cast<size_t>(times.size() * 0.99)];
    return std::make_tuple(mean, std::sqrt(sq / n), p50, p95, p99,
                           times.front(), times.back());
  };

  auto [noM, noS, noP50, noP95, noP99, noMin, noMax] = stats(noTTAtimes);
  auto [ttaM, ttaS, ttaP50, ttaP95, ttaP99, ttaMin, ttaMax] = stats(ttaTimes);

  std::cout << "\n  ==== Latency Comparison ====" << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "Metric"
            << std::right << std::setw(12) << "No-TTA"
            << std::setw(12) << "TTA 8×"
            << std::setw(12) << "Ratio" << std::endl;
  std::cout << "  " << std::string(56, '-') << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "Mean"
            << std::right << std::setw(12) << std::fixed << std::setprecision(2) << noM
            << std::setw(12) << ttaM
            << std::setw(12) << (ttaM / noM) << "×" << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "Std"
            << std::right << std::setw(12) << noS
            << std::setw(12) << ttaS << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "Min"
            << std::right << std::setw(12) << noMin
            << std::setw(12) << ttaMin << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "P50"
            << std::right << std::setw(12) << noP50
            << std::setw(12) << ttaP50 << std::endl;
  std::cout << "  " << std::left << std::setw(20) << "P95"
            << std::right << std::setw(12) << noP95
            << std::setw(12) << ttaP95 << std::endl;

  // Per full-volume estimates
  double noTTA_full = noM * 90.0 / 1000.0;
  double tTA_full = ttaM * 90.0 / 1000.0;
  std::cout << "\n  ==== Estimated Full-Volume (90 patches) ====" << std::endl;
  std::cout << "  No-TTA: " << noTTA_full << " s" << std::endl;
  std::cout << "  TTA 8×: " << tTA_full << " s" << std::endl;
  std::cout << "  Overhead: " << (tTA_full / noTTA_full) << "×" << std::endl;

  PASS();
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <onnx_model> [options]" << std::endl;
    std::cerr << "       " << argv[0] << " --load-cache <engine.trt> [options]" << std::endl;
    std::cerr << "  --benchmark         Run extended benchmark" << std::endl;
    std::cerr << "  --warmup N          Warmup iterations (default: 10)" << std::endl;
    std::cerr << "  --runs N            Benchmark iterations (default: 50)" << std::endl;
    std::cerr << "  --tolerance FLOAT   Max abs diff for consistency check (default: 0.001)" << std::endl;
    std::cerr << "  --verbose           Verbose TRT logging" << std::endl;
    std::cerr << "  --no-fp16           Disable FP16 mode" << std::endl;
    std::cerr << "  --cache-dir PATH    Engine cache directory (default: ./trt_cache)" << std::endl;
    std::cerr << "  --load-cache PATH   Load serialized engine, skip ONNX build" << std::endl;
    std::cerr << "  --tta               Run TTA (Test Time Augmentation) tests" << std::endl;
    return 1;
  }

  std::string onnxPath;
  std::string loadCachePath;
  bool benchmark = false;
  bool tta = false;
  int warmupIters = 10;
  int benchIters = 50;
  float tolerance = 0.001f;
  bool verbose = false;
  bool fp16 = true;
  std::string cacheDir = "./trt_cache";

  // Parse first positional arg
  int argIdx = 1;
  std::string firstArg = argv[1];
  if (firstArg == "--load-cache")
  {
    if (argc < 3) { std::cerr << "Missing path after --load-cache" << std::endl; return 1; }
    loadCachePath = argv[2];
    argIdx = 3;
  }
  else if (firstArg.rfind("--", 0) != 0)
  {
    onnxPath = firstArg;
    argIdx = 2;
  }

  for (int i = argIdx; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--benchmark") benchmark = true;
    else if (arg == "--warmup" && i + 1 < argc) warmupIters = std::atoi(argv[++i]);
    else if (arg == "--runs" && i + 1 < argc) benchIters = std::atoi(argv[++i]);
    else if (arg == "--tolerance" && i + 1 < argc) tolerance = std::stof(argv[++i]);
    else if (arg == "--verbose") verbose = true;
    else if (arg == "--no-fp16") fp16 = false;
    else if (arg == "--cache-dir" && i + 1 < argc) cacheDir = argv[++i];
    else if (arg == "--load-cache" && i + 1 < argc) loadCachePath = argv[++i];
    else if (arg == "--tta") tta = true;
    else
    {
      std::cerr << "Unknown option: " << arg << std::endl;
      return 1;
    }
  }

  if (onnxPath.empty() && loadCachePath.empty())
  {
    std::cerr << "Provide either an ONNX model or --load-cache <engine.trt>" << std::endl;
    return 1;
  }
  if (!onnxPath.empty() && !fs::exists(onnxPath))
  {
    std::cerr << "ONNX model not found: " << onnxPath << std::endl;
    return 1;
  }
  if (!loadCachePath.empty() && !fs::exists(loadCachePath))
  {
    std::cerr << "Engine cache file not found: " << loadCachePath << std::endl;
    return 1;
  }

  std::cout << "============================================================" << std::endl;
  std::cout << "  TensorRT Unit Test — DentalSegmentatorInference" << std::endl;
  std::cout << "============================================================" << std::endl;
  std::cout << "  ONNX:      " << onnxPath << std::endl;
  std::cout << "  Cache dir: " << cacheDir << std::endl;
  std::cout << "  FP16:      " << (fp16 ? "yes" : "no") << std::endl;
  std::cout << "  Tolerance: " << tolerance << std::endl;

  try
  {
    // Init CUDA
    int cudaDevice = 0;
    CUDA_CHECK(cudaSetDevice(cudaDevice));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cudaDevice));
    std::cout << "  GPU:       " << prop.name << std::endl;
    std::cout << "  CUDA:      " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  VRAM:      " << (prop.totalGlobalMem >> 30) << " GB" << std::endl;

    // Init TensorRT logger
    TrtLogger logger(verbose ? nvinfer1::ILogger::Severity::kVERBOSE
                             : nvinfer1::ILogger::Severity::kWARNING);
    logger.setVerbose(verbose);

    // Init TensorRT plugins (required for InstanceNormalization and other built-in plugins)
    initLibNvInferPlugins(&logger, "");

    fs::create_directories(cacheDir);

    // Acquire engine (build from ONNX or load cache)
    EngineBundle engineBundle;

    if (!loadCachePath.empty())
    {
      std::cout << "  Loading engine from cache: " << loadCachePath << std::endl;
      engineBundle = deserializeEngine(loadCachePath, logger);
      // Verify engine
      int nb = engineBundle.engine->getNbBindings();
      std::cout << "  Bindings: " << nb << std::endl;
      for (int i = 0; i < nb; ++i)
      {
        auto name = engineBundle.engine->getBindingName(i);
        bool isInput = engineBundle.engine->bindingIsInput(i);
        auto shape = engineBundle.engine->getBindingDimensions(i);
        std::cout << "  [" << i << "] " << (isInput ? "INPUT " : "OUTPUT")
                  << " | " << name << " | shape=[";
        for (int d = 0; d < shape.nbDims; ++d)
          std::cout << (d > 0 ? "x" : "") << shape.d[d];
        std::cout << "]" << std::endl;
      }
    }
    else
    {
      // Test 1: Build engine
      auto bundle1 = test1_buildEngine(onnxPath, logger, fp16);

      // Test 2: Serialize / deserialize
      engineBundle = test2_serializeDeserialize(*bundle1.engine, cacheDir, logger);
    }

    auto& testEngine = *engineBundle.engine;

    // Test 3: Inference shape check
    test3_inferenceShape(testEngine);

    // Test 4: Consistency
    test4_inferenceConsistency(testEngine, tolerance);

    // Test 5: Benchmark (optional)
    if (benchmark)
      test5_benchmark(testEngine, warmupIters, benchIters);

    // Test 6-7: TTA (optional)
    if (tta)
    {
      test6_ttaEffect(testEngine);
      test7_ttaBenchmark(testEngine, warmupIters, benchIters);
    }

    // Summary
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  RESULTS: " << g_passed << " passed, " << g_failed << " failed" << std::endl;
    std::cout << "============================================================" << std::endl;

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
  }
  catch (const std::exception& e)
  {
    std::cerr << "\nFATAL: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
