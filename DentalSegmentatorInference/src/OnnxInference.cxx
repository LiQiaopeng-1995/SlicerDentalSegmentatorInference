#include "OnnxInference.h"

#include <iostream>
#include <stdexcept>
#include <numeric>

OnnxInference::OnnxInference(const std::string& modelPath,
                             const std::string& device,
                             const std::string& cacheDir)
  : m_env(ORT_LOGGING_LEVEL_WARNING, "DentalSegmentatorInference")
{
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(0);  // Let ORT decide
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  if (device == "cuda")
  {
    // Try TensorRT EP first (FP16 + engine caching)
    try
    {
      OrtTensorRTProviderOptions trtOpts{};
      trtOpts.device_id = 0;
      trtOpts.trt_fp16_enable = 1;
      trtOpts.trt_engine_cache_enable = 1;
      trtOpts.trt_engine_cache_path = cacheDir.c_str();
      trtOpts.trt_max_workspace_size = static_cast<size_t>(2) << 30;  // 2 GB
      opts.AppendExecutionProvider_TensorRT(trtOpts);
      std::cout << "TensorRT EP configured (FP16, cache: " << cacheDir << ")" << std::endl;
    }
    catch (const Ort::Exception& e)
    {
      std::cout << "TensorRT EP not available: " << e.what() << std::endl;
    }

    // CUDA EP as fallback
    try
    {
      OrtCUDAProviderOptions cudaOpts{};
      cudaOpts.device_id = 0;
      opts.AppendExecutionProvider_CUDA(cudaOpts);
      std::cout << "CUDA EP configured." << std::endl;
    }
    catch (const Ort::Exception& e)
    {
      std::cout << "CUDA EP not available: " << e.what()
                << ". Falling back to CPU." << std::endl;
    }
  }

  std::cout << "Loading ONNX model: " << modelPath << std::endl;
#ifdef _WIN32
  // Windows requires wide string path
  std::wstring wpath(modelPath.begin(), modelPath.end());
  m_session = Ort::Session(m_env, wpath.c_str(), opts);
#else
  m_session = Ort::Session(m_env, modelPath.c_str(), opts);
#endif
  std::cout << "Model loaded successfully." << std::endl;
}

std::vector<float> OnnxInference::runPatch(const float* inputData,
                                           const std::vector<int64_t>& inputShape)
{
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  size_t inputSize = 1;
  for (auto s : inputShape)
    inputSize *= static_cast<size_t>(s);

  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memoryInfo, const_cast<float*>(inputData), inputSize,
    inputShape.data(), inputShape.size());

  // Get input/output names
  auto inputName = m_session.GetInputNameAllocated(0, m_allocator);
  auto outputName = m_session.GetOutputNameAllocated(0, m_allocator);
  const char* inputNames[] = { inputName.get() };
  const char* outputNames[] = { outputName.get() };

  auto outputs = m_session.Run(Ort::RunOptions{nullptr},
                               inputNames, &inputTensor, 1,
                               outputNames, 1);

  // Extract output
  auto& outputTensor = outputs[0];
  auto typeInfo = outputTensor.GetTensorTypeAndShapeInfo();
  m_lastOutputShape = typeInfo.GetShape();

  size_t outputSize = 1;
  for (auto s : m_lastOutputShape)
    outputSize *= static_cast<size_t>(s);

  const float* outputData = outputTensor.GetTensorData<float>();
  return std::vector<float>(outputData, outputData + outputSize);
}
