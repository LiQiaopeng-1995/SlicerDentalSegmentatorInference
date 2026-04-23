/**
 * Standalone test for DentalSegmentatorInference C++ pipeline.
 * No Slicer/SEM dependency.
 *
 * Usage: dental_test <input.nii.gz> <output.nii.gz> <modelDir> [device]
 */

#include "PlansParser.h"
#include "Preprocessor.h"
#include "OnnxInference.h"
#include "SlidingWindow.h"
#include "Postprocessor.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <chrono>

namespace fs = std::filesystem;

static std::string findOnnxModel(const std::string& modelDir)
{
  for (const auto& entry : fs::directory_iterator(modelDir))
  {
    if (entry.path().extension() == ".onnx")
      return entry.path().string();
  }
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
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0]
              << " <input.nii.gz> <output.nii.gz> <modelDir> [device=cuda] [step_size=0.5]"
              << std::endl;
    return 1;
  }

  std::string inputVolume = argv[1];
  std::string outputVolume = argv[2];
  std::string modelDir = argv[3];
  std::string device = (argc > 4) ? argv[4] : "cuda";
  float stepSize = (argc > 5) ? std::stof(argv[5]) : 0.5f;

  try
  {
    using Clock = std::chrono::high_resolution_clock;

    std::cout << "=== DentalSegmentatorInference C++ Test ===" << std::endl;
    std::cout << "Input:     " << inputVolume << std::endl;
    std::cout << "Output:    " << outputVolume << std::endl;
    std::cout << "ModelDir:  " << modelDir << std::endl;
    std::cout << "Device:    " << device << std::endl;
    std::cout << "Step size: " << stepSize << std::endl;

    auto t_total_start = Clock::now();

    // 1. Parse plans.json
    std::cout << "\n[1/4] Parsing plans.json..." << std::endl;
    auto config = parsePlans(modelDir);
    std::cout << "  Patch size: [" << config.patchSize[0] << ", "
              << config.patchSize[1] << ", " << config.patchSize[2] << "]" << std::endl;
    std::cout << "  Target spacing: [" << config.targetSpacing[0] << ", "
              << config.targetSpacing[1] << ", " << config.targetSpacing[2] << "]" << std::endl;
    std::cout << "  Num classes: " << config.numClasses << std::endl;

    // 2. Preprocess
    std::cout << "\n[2/4] Preprocessing..." << std::endl;
    auto t_pre_start = Clock::now();
    auto prepResult = preprocess(inputVolume, config);
    auto t_pre_end = Clock::now();

    auto image = prepResult.image;
    auto size = image->GetLargestPossibleRegion().GetSize();
    // ITK size is (x,y,z) but sliding window expects (D=z, H=y, W=x)
    std::array<int, 3> volumeShape = {
      static_cast<int>(size[2]),  // D = z (slowest in memory)
      static_cast<int>(size[1]),  // H = y
      static_cast<int>(size[0])   // W = x (fastest in memory)
    };
    const float* volumeData = image->GetBufferPointer();
    double preTime = std::chrono::duration<double>(t_pre_end - t_pre_start).count();
    std::cout << "  Shape: [" << volumeShape[0] << ", " << volumeShape[1]
              << ", " << volumeShape[2] << "]" << std::endl;
    std::cout << "  Preprocess time: " << preTime << "s" << std::endl;

    // 3. Inference
    std::cout << "\n[3/4] Inference (sliding window)..." << std::endl;
    std::string onnxPath = findOnnxModel(modelDir);
    std::cout << "  ONNX model: " << onnxPath << std::endl;
    std::string cacheDir = (fs::path(modelDir) / "trt_cache").string();
    fs::create_directories(cacheDir);

    auto t_inf_start = Clock::now();
    OnnxInference inference(onnxPath, device, cacheDir);
    SlidingWindow slidingWindow(inference, config.patchSize, config.numClasses, stepSize);
    auto logits = slidingWindow.run(volumeData, volumeShape);
    auto t_inf_end = Clock::now();
    double infTime = std::chrono::duration<double>(t_inf_end - t_inf_start).count();
    std::cout << "  Inference time: " << infTime << "s" << std::endl;

    // 4. Postprocess
    std::cout << "\n[4/4] Postprocessing..." << std::endl;
    auto t_post_start = Clock::now();
    postprocess(logits, volumeShape, config.numClasses, prepResult, config, outputVolume);
    auto t_post_end = Clock::now();
    double postTime = std::chrono::duration<double>(t_post_end - t_post_start).count();
    std::cout << "  Postprocess time: " << postTime << "s" << std::endl;

    auto t_total_end = Clock::now();
    double totalTime = std::chrono::duration<double>(t_total_end - t_total_start).count();

    // Summary
    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "  Preprocess:  " << preTime << "s" << std::endl;
    std::cout << "  Inference:   " << infTime << "s" << std::endl;
    std::cout << "  Postprocess: " << postTime << "s" << std::endl;
    std::cout << "  TOTAL:       " << totalTime << "s" << std::endl;

    return EXIT_SUCCESS;
  }
  catch (const std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}
