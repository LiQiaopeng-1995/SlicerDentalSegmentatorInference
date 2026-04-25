/**
 * trt_tta_full — Full volume: ITK pre/post + TensorRT sliding window + optional TTA 8×
 * (mirrors test_trt.cxx TTA; compares fairly to PyTorch TTA8× baseline on same NIfTI).
 *
 * Usage:
 *   trt_tta_full <input.nii.gz> <modelDir> <output.nii.gz> [options]
 * Options:
 *   --tta              TTA 8× per patch (default: on)
 *   --no-tta           single forward per patch
 *   --fp16             TRT FP16 (default: on)
 *   --no-fp16
 *   --step FLOAT       sliding window step ratio (default: 0.5)
 *   --load-cache PATH  .trt engine (skip ONNX build)
 *   --onnx PATH        override ONNX path (else search modelDir)
 *   --cache-out PATH   after ONNX build, save .trt here (optional)
 */

#include "trt_tta_helpers.hxx"
#include "PlansParser.h"
#include "Preprocessor.h"
#include "Postprocessor.h"

#include <itkImageFileWriter.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- Gaussian (same as SlidingWindow.cxx) ----
static std::vector<float> computeGaussianMap(const std::array<int, 3>& patchSize)
{
  size_t total = static_cast<size_t>(patchSize[0]) * patchSize[1] * patchSize[2];
  std::vector<float> gaussian(total);
  std::array<std::vector<float>, 3> g1d;
  for (int d = 0; d < 3; ++d)
  {
    int n = patchSize[d];
    float sigma = static_cast<float>(n) / 8.0f;
    float center = (n - 1.0f) / 2.0f;
    g1d[d].resize(n);
    for (int i = 0; i < n; ++i)
    {
      float diff = static_cast<float>(i) - center;
      g1d[d][i] = std::exp(-0.5f * diff * diff / (sigma * sigma));
    }
  }
  size_t idx = 0;
  for (int z = 0; z < patchSize[0]; ++z)
    for (int y = 0; y < patchSize[1]; ++y)
      for (int x = 0; x < patchSize[2]; ++x)
        gaussian[idx++] = g1d[0][z] * g1d[1][y] * g1d[2][x];
  float maxVal = *std::max_element(gaussian.begin(), gaussian.end());
  float minVal = maxVal * 1e-4f;
  for (auto& v : gaussian) v = std::max(v, minVal);
  return gaussian;
}

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
  throw std::runtime_error("No .onnx in: " + modelDir);
}

static std::vector<float> runSlidingTrtTta(
    nvinfer1::ICudaEngine& engine,
    const float* volumeData,
    const std::array<int, 3>& volumeShape,
    const std::array<int, 3>& patchSize,
    int numClasses,
    float stepSize,
    bool tta8)
{
  int D = volumeShape[0], H = volumeShape[1], W = volumeShape[2];
  size_t spatialSize = static_cast<size_t>(D) * H * W;
  size_t totalLogits = static_cast<size_t>(numClasses) * spatialSize;
  std::vector<float> aggregated(totalLogits, 0.0f);
  std::vector<float> weightSum(spatialSize, 0.0f);

  auto gaussian = computeGaussianMap(patchSize);
  size_t patchVoxels = static_cast<size_t>(patchSize[0]) * patchSize[1] * patchSize[2];

  auto computeStarts = [](int volSize, int ps, float stepRatio) {
    std::vector<int> starts;
    if (volSize <= ps)
    {
      starts.push_back(0);
      return starts;
    }
    int step = static_cast<int>(std::round(ps * stepRatio));
    if (step < 1) step = 1;
    for (int s = 0; s + ps <= volSize; s += step) starts.push_back(s);
    if (starts.back() + ps < volSize) starts.push_back(volSize - ps);
    return starts;
  };

  auto startsZ = computeStarts(D, patchSize[0], stepSize);
  auto startsY = computeStarts(H, patchSize[1], stepSize);
  auto startsX = computeStarts(W, patchSize[2], stepSize);

  int totalPatches = static_cast<int>(startsZ.size() * startsY.size() * startsX.size());
  std::cout << "Sliding window: " << totalPatches << " patches, tta8=" << (tta8 ? "yes" : "no")
            << ", step=" << stepSize << std::endl;

  nvinfer1::Dims inputDims;
  inputDims.nbDims = 5;
  inputDims.d[0] = 1;
  inputDims.d[1] = 1;
  inputDims.d[2] = patchSize[0];
  inputDims.d[3] = patchSize[1];
  inputDims.d[4] = patchSize[2];

  std::vector<float> input5d(1 * 1 * patchVoxels);
  std::vector<float> patchBuf(patchVoxels);
  int nFlip = tta8 ? 8 : 1;

  int patchIdx = 0;
  for (int sz : startsZ)
  {
    for (int sy : startsY)
    {
      for (int sx : startsX)
      {
        size_t bi = 0;
        for (int z = sz; z < sz + patchSize[0]; ++z)
          for (int y = sy; y < sy + patchSize[1]; ++y)
            for (int x = sx; x < sx + patchSize[2]; ++x)
              patchBuf[bi++] = volumeData[static_cast<size_t>(z) * H * W + y * W + x];

        for (size_t j = 0; j < patchVoxels; ++j)
          input5d[j] = patchBuf[j];

        std::vector<float> acc(numClasses * patchVoxels, 0.0f);
        for (int f = 0; f < nFlip; ++f)
        {
          std::vector<float> out = runTTAInference(engine, input5d, inputDims, tta8 ? f : 0, nullptr);
          if (out.size() != acc.size())
            throw std::runtime_error("TRT output size mismatch");
          for (size_t i = 0; i < acc.size(); ++i) acc[i] += out[i];
        }
        for (float& v : acc) v /= static_cast<float>(nFlip);

        for (int c = 0; c < numClasses; ++c)
        {
          size_t classOffset = static_cast<size_t>(c) * patchVoxels;
          size_t globalClassOffset = static_cast<size_t>(c) * spatialSize;
          size_t gi = 0;
          for (int z = sz; z < sz + patchSize[0]; ++z)
            for (int y = sy; y < sy + patchSize[1]; ++y)
              for (int x = sx; x < sx + patchSize[2]; ++x)
              {
                size_t globalIdx = static_cast<size_t>(z) * H * W + y * W + x;
                float w = gaussian[gi];
                aggregated[globalClassOffset + globalIdx] += acc[classOffset + gi] * w;
                if (c == 0) weightSum[globalIdx] += w;
                ++gi;
              }
        }

        ++patchIdx;
        if (patchIdx % 5 == 0 || patchIdx == totalPatches)
          std::cout << "  Patch " << patchIdx << "/" << totalPatches << std::endl;
      }
    }
  }

  for (int c = 0; c < numClasses; ++c)
  {
    size_t offset = static_cast<size_t>(c) * spatialSize;
    for (size_t i = 0; i < spatialSize; ++i)
      if (weightSum[i] > 0.0f) aggregated[offset + i] /= weightSum[i];
  }
  return aggregated;
}

static void printUsage(const char* exe)
{
  std::cerr
      << "Usage: " << exe
      << " <input.nii.gz> <modelDir> <output.nii.gz> [options]\n"
      << "  --tta / --no-tta   TTA 8x per patch (default: --tta)\n"
      << "  --fp16 / --no-fp16\n"
      << "  --step FLOAT (default 0.5)\n"
      << "  --load-cache PATH.trt\n"
      << "  --onnx PATH\n"
      << "  --cache-out PATH.trt  (save engine after build)\n"
      << "  --dump-preprocessed PATH.nii.gz  (debug: write preprocessed volume)\n"
      << "  --dump-prepost-label PATH.nii.gz  (debug: argmax before postprocess)\n";
}

static void writePrepostArgmax(const std::vector<float>& logits,
                               const std::array<int, 3>& shape,
                               int numClasses,
                               const std::string& path)
{
  const int D = shape[0], H = shape[1], W = shape[2];
  const size_t spatialSize = static_cast<size_t>(D) * H * W;

  auto label = LabelImageType::New();
  LabelImageType::RegionType region;
  LabelImageType::IndexType start;
  start.Fill(0);
  LabelImageType::SizeType size;
  size[0] = W;
  size[1] = H;
  size[2] = D;
  region.SetIndex(start);
  region.SetSize(size);
  label->SetRegions(region);
  label->Allocate(true);

  for (int z = 0; z < D; ++z)
  {
    for (int y = 0; y < H; ++y)
    {
      for (int x = 0; x < W; ++x)
      {
        const size_t i = static_cast<size_t>(z) * H * W + y * W + x;
        float best = -std::numeric_limits<float>::infinity();
        unsigned char bestClass = 0;
        for (int c = 0; c < numClasses; ++c)
        {
          const float v = logits[static_cast<size_t>(c) * spatialSize + i];
          if (v > best)
          {
            best = v;
            bestClass = static_cast<unsigned char>(c);
          }
        }
        LabelImageType::IndexType idx;
        idx[0] = x;
        idx[1] = y;
        idx[2] = z;
        label->SetPixel(idx, bestClass);
      }
    }
  }

  using LabelWriter = itk::ImageFileWriter<LabelImageType>;
  auto writer = LabelWriter::New();
  writer->SetFileName(path);
  writer->SetInput(label);
  writer->SetUseCompression(true);
  writer->Update();
}

int main(int argc, char* argv[])
{
  if (argc < 4)
  {
    printUsage(argv[0]);
    return 1;
  }

  std::string inputPath = argv[1];
  std::string modelDir = argv[2];
  std::string outputPath = argv[3];
  bool useTta = true;
  bool fp16 = true;
  float step = 0.5f;
  std::string loadCache;
  std::string onnxOverride;
  std::string cacheOut;
  std::string dumpPreprocessed;
  std::string dumpPrepostLabel;

  for (int i = 4; i < argc; ++i)
  {
    std::string a = argv[i];
    if (a == "--tta") useTta = true;
    else if (a == "--no-tta") useTta = false;
    else if (a == "--fp16") fp16 = true;
    else if (a == "--no-fp16") fp16 = false;
    else if (a == "--step" && i + 1 < argc) step = std::stof(argv[++i]);
    else if (a == "--load-cache" && i + 1 < argc) loadCache = argv[++i];
    else if (a == "--onnx" && i + 1 < argc) onnxOverride = argv[++i];
    else if (a == "--cache-out" && i + 1 < argc) cacheOut = argv[++i];
    else if (a == "--dump-preprocessed" && i + 1 < argc) dumpPreprocessed = argv[++i];
    else if (a == "--dump-prepost-label" && i + 1 < argc) dumpPrepostLabel = argv[++i];
    else
    {
      std::cerr << "Unknown: " << a << std::endl;
      return 1;
    }
  }

  TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
  try
  {
    CUDA_CHECK(cudaSetDevice(0));
    initLibNvInferPlugins(&logger, "");

    std::cout << "=== trt_tta_full ===" << std::endl;
    std::cout << "  Input:  " << inputPath << std::endl;
    std::cout << "  Output: " << outputPath << std::endl;
    std::cout << "  modelDir: " << modelDir << std::endl;
    std::cout << "  TTA8: " << (useTta ? "yes" : "no") << "  FP16: " << (fp16 ? "yes" : "no")
              << "  step: " << step << std::endl;

    auto config = parsePlans(modelDir);
    std::array<int, 3> patchSize = { config.patchSize[0], config.patchSize[1], config.patchSize[2] };
    int numClasses = config.numClasses;

    std::cout << "\n[1/3] Preprocess..." << std::endl;
    auto prep = preprocess(inputPath, config);
    auto image = prep.image;
    if (!dumpPreprocessed.empty())
    {
      using PreprocessWriter = itk::ImageFileWriter<FloatImageType>;
      auto pw = PreprocessWriter::New();
      pw->SetFileName(dumpPreprocessed);
      pw->SetInput(image);
      pw->SetUseCompression(true);
      pw->Update();
      std::cout << "Dumped preprocessed volume: " << dumpPreprocessed << std::endl;
    }
    auto size = image->GetLargestPossibleRegion().GetSize();
    std::array<int, 3> volumeShape = { static_cast<int>(size[2]), static_cast<int>(size[1]),
                                      static_cast<int>(size[0]) };
    const float* volPtr = image->GetBufferPointer();
    std::cout << "  shape [D,H,W] = [" << volumeShape[0] << ", " << volumeShape[1] << ", "
              << volumeShape[2] << "]" << std::endl;

    std::cout << "\n[2/3] TensorRT " << (loadCache.empty() ? "(build from ONNX)" : "(load cache)") << std::endl;

    EngineBundle bundle;
    if (!loadCache.empty()) bundle = deserializeEngine(loadCache, logger);
    else
    {
      std::string onnxPath = onnxOverride.empty() ? findOnnxModel(modelDir) : onnxOverride;
      std::cout << "  ONNX: " << onnxPath << std::endl;
      bundle = buildEngineFromOnnx(onnxPath, logger, fp16, 2);
      if (!cacheOut.empty()) serializeEngine(*bundle.engine, cacheOut);
    }

    std::cout << "\n[2/3] Inference..." << std::endl;
    auto logits = runSlidingTrtTta(*bundle.engine, volPtr, volumeShape, patchSize, numClasses, step, useTta);
    if (!dumpPrepostLabel.empty())
    {
      writePrepostArgmax(logits, volumeShape, numClasses, dumpPrepostLabel);
      std::cout << "Dumped pre-postprocess argmax label: " << dumpPrepostLabel << std::endl;
    }

    std::cout << "\n[3/3] Postprocess..." << std::endl;
    postprocess(logits, volumeShape, numClasses, prep, config, outputPath);
    std::cout << "Done. Wrote: " << outputPath << std::endl;
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
