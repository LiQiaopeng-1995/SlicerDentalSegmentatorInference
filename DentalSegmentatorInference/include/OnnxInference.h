#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

/**
 * ONNX Runtime inference session wrapper.
 * Supports TensorRT EP (FP16) with engine caching, CUDA EP fallback, and CPU.
 */
class OnnxInference
{
public:
  /**
   * @param modelPath  Path to the .onnx model file
   * @param device     "cuda" or "cpu"
   * @param cacheDir   Directory for TensorRT engine cache (only used with cuda)
   */
  OnnxInference(const std::string& modelPath,
                const std::string& device,
                const std::string& cacheDir);

  /**
   * Run inference on a single patch.
   * @param inputData  Pointer to float data [1, 1, D, H, W]
   * @param inputShape Shape vector {1, 1, D, H, W}
   * @return Output logits as flat float vector, shape [1, C, D, H, W]
   */
  std::vector<float> runPatch(const float* inputData,
                              const std::vector<int64_t>& inputShape);

  /** Get output shape from last inference run */
  std::vector<int64_t> getOutputShape() const { return m_lastOutputShape; }

private:
  Ort::Env m_env;
  Ort::Session m_session{nullptr};
  Ort::AllocatorWithDefaultOptions m_allocator;
  std::vector<int64_t> m_lastOutputShape;
};
