/**
 * TRT 8.6 + TTA helpers (aligned with test_trt.cxx).
 * Included by trt_tta_helpers.cxx only; do not include from headers used by multiple TUs
 * to avoid ODR issues — keep single .cxx.
 */
#pragma once

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(nvinfer1::ILogger::Severity level = nvinfer1::ILogger::Severity::kWARNING)
    : m_level(level), m_verbose(false) {}
  void setVerbose(bool v) { m_verbose = v; }
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= m_level || m_verbose)
      std::cerr << msg << std::endl;
  }
private:
  Severity m_level;
  bool m_verbose;
};

struct TrtDeleter
{
  template <typename T>
  void operator()(T* obj) const { if (obj) obj->destroy(); }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

#define CUDA_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::ostringstream oss;                                                \
      oss << "CUDA error " << cudaGetErrorName(err) << " " << cudaGetErrorString(err); \
      throw std::runtime_error(oss.str());                                   \
    }                                                                         \
  } while (0)

struct IOBuffers
{
  struct Binding
  {
    int index = -1;
    std::string name;
    nvinfer1::Dims dims;
    nvinfer1::DataType dtype{};
    size_t sizeBytes = 0;
    size_t numElements = 0;
    void* devicePtr = nullptr;
    std::vector<float> hostData;
  };
  std::vector<Binding> inputs;
  std::vector<Binding> outputs;
  std::vector<void*> buildBindingsArray() const
  {
    int nb = static_cast<int>(inputs.size() + outputs.size());
    std::vector<void*> ptrs(nb, nullptr);
    for (const auto& b : inputs)  ptrs[b.index] = b.devicePtr;
    for (const auto& b : outputs) ptrs[b.index] = b.devicePtr;
    return ptrs;
  }
};

struct EngineBundle
{
  TrtUniquePtr<nvinfer1::IRuntime> runtime;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine;
};

class Timer
{
public:
  using Clock = std::chrono::high_resolution_clock;
  void start() { m_t = Clock::now(); }
  double elapsedMs() const
  {
    return std::chrono::duration<double, std::milli>(Clock::now() - m_t).count();
  }
private:
  Clock::time_point m_t;
};

void allocateInputs(IOBuffers& bufs, nvinfer1::ICudaEngine& engine, const nvinfer1::Dims& inputShape);
void allocateOutputs(IOBuffers& bufs, nvinfer1::ICudaEngine& engine, nvinfer1::IExecutionContext& context);
void freeIOBuffers(IOBuffers& bufs);

EngineBundle buildEngineFromOnnx(const std::string& onnxPath, TrtLogger& logger, bool fp16, int maxWorkspaceGB);
void serializeEngine(nvinfer1::ICudaEngine& engine, const std::string& path);
EngineBundle deserializeEngine(const std::string& path, TrtLogger& logger);

/** flipMask: bit0=D, bit1=H, bit2=W; data [N,C,D,H,W] row-major */
void flipTensor5D(std::vector<float>& data, const nvinfer1::Dims& dims, int flipMask);

/**
 * TTA 单种翻转：翻转输入 → 推理 → 同 mask 反翻转输出 logits（与 test_trt.cxx 一致）。
 */
std::vector<float> runTTAInference(
    nvinfer1::ICudaEngine& engine,
    const std::vector<float>& inputNCHW,
    const nvinfer1::Dims& inputDims,
    int flipMask,
    cudaStream_t stream = nullptr);
