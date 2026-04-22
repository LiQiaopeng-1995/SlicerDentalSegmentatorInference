#include "DentalSegmentatorInferenceCLP.h"

#include "PlansParser.h"
#include "Preprocessor.h"
#include "OnnxInference.h"
#include "SlidingWindow.h"
#include "Postprocessor.h"

#include <itkImageRegionConstIterator.h>

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static std::string findOnnxModel(const std::string& modelDir)
{
  // Search for .onnx file in modelDir
  for (const auto& entry : fs::directory_iterator(modelDir))
  {
    if (entry.path().extension() == ".onnx")
      return entry.path().string();
  }
  // Also check parent directory
  auto parentDir = fs::path(modelDir).parent_path();
  for (const auto& entry : fs::directory_iterator(parentDir))
  {
    if (entry.path().extension() == ".onnx")
      return entry.path().string();
  }
  throw std::runtime_error("No .onnx model file found in: " + modelDir);
}

int main(int argc, char* argv[])
{
  PARSE_ARGS;

  try
  {
    std::cout << "=== DentalSegmentatorInference (C++ ONNX Runtime) ===" << std::endl;
    std::cout << "Input:  " << inputVolume << std::endl;
    std::cout << "Output: " << outputVolume << std::endl;
    std::cout << "Model:  " << modelDir << std::endl;
    std::cout << "Device: " << device << std::endl;

    // 1. Parse plans.json
    std::cout << "\n--- Parsing plans.json ---" << std::endl;
    auto config = parsePlans(modelDir);
    std::cout << "Patch size: [" << config.patchSize[0] << ", "
              << config.patchSize[1] << ", " << config.patchSize[2] << "]" << std::endl;
    std::cout << "Target spacing: [" << config.targetSpacing[0] << ", "
              << config.targetSpacing[1] << ", " << config.targetSpacing[2] << "]" << std::endl;
    std::cout << "Num classes: " << config.numClasses << std::endl;

    // 2. Preprocess
    std::cout << "\n--- Preprocessing ---" << std::endl;
    auto prepResult = preprocess(inputVolume, config);

    // Get volume data as contiguous float array
    auto image = prepResult.image;
    auto size = image->GetLargestPossibleRegion().GetSize();
    std::array<int, 3> volumeShape = {
      static_cast<int>(size[0]),
      static_cast<int>(size[1]),
      static_cast<int>(size[2])
    };
    const float* volumeData = image->GetBufferPointer();

    // 3. Initialize ONNX Runtime and run sliding window inference
    std::cout << "\n--- Inference ---" << std::endl;
    std::string onnxPath = findOnnxModel(modelDir);
    std::string cacheDir = (fs::path(modelDir) / "trt_cache").string();
    fs::create_directories(cacheDir);

    OnnxInference inference(onnxPath, device, cacheDir);
    SlidingWindow slidingWindow(inference, config.patchSize, config.numClasses, 0.75f);
    auto logits = slidingWindow.run(volumeData, volumeShape);

    // 4. Postprocess
    std::cout << "\n--- Postprocessing ---" << std::endl;
    postprocess(logits, volumeShape, config.numClasses, prepResult, config, outputVolume);

    std::cout << "\n=== Done ===" << std::endl;
    return EXIT_SUCCESS;
  }
  catch (const std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
