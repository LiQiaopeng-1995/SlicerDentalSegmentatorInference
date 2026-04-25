#include "Preprocessor.h"

#include <itkImageFileReader.h>
#include <itkCastImageFilter.h>
#include <itkPermuteAxesImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkConstantPadImageFilter.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using ReaderType = itk::ImageFileReader<itk::Image<short, 3>>;
using ShortImageType = itk::Image<short, 3>;
using CastFilterType = itk::CastImageFilter<ShortImageType, FloatImageType>;
using PermuteType = itk::PermuteAxesImageFilter<FloatImageType>;
using ExtractType = itk::ExtractImageFilter<FloatImageType, FloatImageType>;
using PadType = itk::ConstantPadImageFilter<FloatImageType, FloatImageType>;

// ──────────────────────── helpers ────────────────────────

/** Find bounding box of non-zero voxels */
static void computeNonzeroBBox(FloatImageType::Pointer image,
                               FloatImageType::IndexType& bboxStart,
                               FloatImageType::SizeType& bboxSize)
{
  auto region = image->GetLargestPossibleRegion();
  auto size = region.GetSize();

  // Initialize to extreme values
  FloatImageType::IndexType minIdx, maxIdx;
  for (int d = 0; d < 3; ++d)
  {
    minIdx[d] = static_cast<long>(size[d]);
    maxIdx[d] = 0;
  }

  itk::ImageRegionConstIterator<FloatImageType> it(image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    if (it.Get() > -500.0f)
    {
      auto idx = it.GetIndex();
      for (int d = 0; d < 3; ++d)
      {
        minIdx[d] = std::min(minIdx[d], idx[d]);
        maxIdx[d] = std::max(maxIdx[d], idx[d]);
      }
    }
  }

  for (int d = 0; d < 3; ++d)
  {
    bboxStart[d] = minIdx[d];
    bboxSize[d] = static_cast<unsigned long>(maxIdx[d] - minIdx[d] + 1);
  }
}

// ──────────────────────── main pipeline ────────────────────────

PreprocessResult preprocess(const std::string& inputPath, const PlansConfig& config)
{
  PreprocessResult result{};

  // 1. Read volume
  std::cout << "Reading input volume..." << std::endl;
  auto reader = ReaderType::New();
  reader->SetFileName(inputPath);
  reader->Update();

  // Cast to float
  auto caster = CastFilterType::New();
  caster->SetInput(reader->GetOutput());
  caster->Update();
  auto floatImage = caster->GetOutput();

  // Store original properties
  result.originalSize = floatImage->GetLargestPossibleRegion().GetSize();
  result.originalSpacing = floatImage->GetSpacing();
  result.originalOrigin = floatImage->GetOrigin();
  result.originalDirection = floatImage->GetDirection();

  // 2. Transpose forward (manual implementation for robustness)
  std::cout << "Transposing axes..." << std::endl;
  FloatImageType::Pointer transposed;
  {
    auto inputSize = floatImage->GetLargestPossibleRegion().GetSize();
    auto inputSpacing = floatImage->GetSpacing();

    const auto& fwd = config.transposeForward;

    FloatImageType::SizeType newSize;
    FloatImageType::SpacingType newSpacing;
    for (int d = 0; d < 3; ++d)
    {
      newSize[d] = inputSize[fwd[d]];
      newSpacing[d] = inputSpacing[fwd[d]];
    }

    transposed = FloatImageType::New();
    FloatImageType::RegionType newRegion;
    newRegion.SetSize(newSize);
    transposed->SetRegions(newRegion);
    transposed->SetSpacing(newSpacing);
    transposed->Allocate();

    // Copy data with axes permutation
    const float* src = floatImage->GetBufferPointer();
    float* dst = transposed->GetBufferPointer();

    // Input strides (ITK uses x-fastest order)
    long sx = 1;
    long sy = static_cast<long>(inputSize[0]);
    long sz = static_cast<long>(inputSize[0]) * static_cast<long>(inputSize[1]);
    long strides[3] = {sx, sy, sz};

    // Output strides
    long osx = 1;
    long osy = static_cast<long>(newSize[0]);
    long osz = static_cast<long>(newSize[0]) * static_cast<long>(newSize[1]);

    for (long z = 0; z < static_cast<long>(newSize[2]); ++z)
    {
      for (long y = 0; y < static_cast<long>(newSize[1]); ++y)
      {
        for (long x = 0; x < static_cast<long>(newSize[0]); ++x)
        {
          // Output[x,y,z] = Input[coords], where coords are inverse-permuted
          long srcCoords[3];
          srcCoords[fwd[0]] = x;
          srcCoords[fwd[1]] = y;
          srcCoords[fwd[2]] = z;

          long srcIdx = srcCoords[0] * strides[0] + srcCoords[1] * strides[1] + srcCoords[2] * strides[2];
          long dstIdx = x * osx + y * osy + z * osz;
          dst[dstIdx] = src[srcIdx];
        }
      }
    }

    std::cout << "  Transposed size: " << newSize << ", spacing: " << newSpacing << std::endl;
  }

  // 3. Crop to non-zero bounding box (manual implementation)
  std::cout << "Cropping to non-zero region..." << std::endl;
  FloatImageType::IndexType bboxStart;
  FloatImageType::SizeType bboxSize;
  computeNonzeroBBox(transposed, bboxStart, bboxSize);
  result.cropStart = bboxStart;
  result.croppedSize = bboxSize;
  std::cout << "  BBox start: " << bboxStart << ", size: " << bboxSize << std::endl;

  FloatImageType::Pointer cropped;
  {
    auto tSize = transposed->GetLargestPossibleRegion().GetSize();
    auto tSpacing = transposed->GetSpacing();

    cropped = FloatImageType::New();
    FloatImageType::RegionType cRegion;
    cRegion.SetSize(bboxSize);
    cropped->SetRegions(cRegion);
    cropped->SetSpacing(tSpacing);
    cropped->Allocate();

    const float* tData = transposed->GetBufferPointer();
    float* cData = cropped->GetBufferPointer();

    long tSx = 1, tSy = static_cast<long>(tSize[0]),
         tSz = static_cast<long>(tSize[0]) * static_cast<long>(tSize[1]);
    long cSx = 1, cSy = static_cast<long>(bboxSize[0]),
         cSz = static_cast<long>(bboxSize[0]) * static_cast<long>(bboxSize[1]);

    for (long z = 0; z < static_cast<long>(bboxSize[2]); ++z)
      for (long y = 0; y < static_cast<long>(bboxSize[1]); ++y)
        for (long x = 0; x < static_cast<long>(bboxSize[0]); ++x)
        {
          long si = (x + bboxStart[0]) * tSx + (y + bboxStart[1]) * tSy + (z + bboxStart[2]) * tSz;
          long di = x * cSx + y * cSy + z * cSz;
          cData[di] = tData[si];
        }
  }

  // 4. CT Normalization: clip to [percentile_00_5, percentile_99_5], then z-score
  std::cout << "Normalizing..." << std::endl;
  {
    float clipLow = static_cast<float>(config.percentile005);
    float clipHigh = static_cast<float>(config.percentile995);
    float mean = static_cast<float>(config.mean);
    float sd = static_cast<float>(config.std);
    if (sd < 1e-8f) sd = 1.0f;

    itk::ImageRegionIterator<FloatImageType> it(cropped, cropped->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      float v = it.Get();
      v = std::clamp(v, clipLow, clipHigh);
      v = (v - mean) / sd;
      it.Set(v);
    }
  }

  // Store size before resampling (needed for logit resampling in postprocessing)
  result.sizeBeforeResampling = cropped->GetLargestPossibleRegion().GetSize();

  // 5. Resample to target spacing with the same array-coordinate rule as
  // PyTorch interpolate(..., align_corners=False). ITK's physical-coordinate
  // resampler is close but not identical, and this preprocessing path is used
  // to compare against a PyTorch reference.
  std::cout << "Resampling to target spacing..." << std::endl;
  {
    auto croppedSpacing = cropped->GetSpacing();
    auto croppedSize = cropped->GetLargestPossibleRegion().GetSize();

    FloatImageType::SpacingType targetSpacing;
    FloatImageType::SizeType targetSize;
    for (int d = 0; d < 3; ++d)
    {
      targetSpacing[d] = config.targetSpacing[d];
      targetSize[d] = static_cast<unsigned long>(
        std::round(croppedSize[d] * croppedSpacing[d] / targetSpacing[d]));
      if (targetSize[d] < 1) targetSize[d] = 1;
    }

    const int inW = static_cast<int>(croppedSize[0]);
    const int inH = static_cast<int>(croppedSize[1]);
    const int inD = static_cast<int>(croppedSize[2]);
    const int outW = static_cast<int>(targetSize[0]);
    const int outH = static_cast<int>(targetSize[1]);
    const int outD = static_cast<int>(targetSize[2]);
    const float* src = cropped->GetBufferPointer();

    auto at = [&](int z, int y, int x) {
      return src[static_cast<size_t>(z) * inH * inW + static_cast<size_t>(y) * inW + x];
    };
    auto clampCoord = [](double v, int n) {
      if (n <= 1) return 0.0;
      return std::clamp(v, 0.0, static_cast<double>(n - 1));
    };
    auto sample = [&](double z, double y, double x) {
      z = clampCoord(z, inD);
      y = clampCoord(y, inH);
      x = clampCoord(x, inW);
      const int z0 = static_cast<int>(std::floor(z));
      const int y0 = static_cast<int>(std::floor(y));
      const int x0 = static_cast<int>(std::floor(x));
      const int z1 = std::min(z0 + 1, inD - 1);
      const int y1 = std::min(y0 + 1, inH - 1);
      const int x1 = std::min(x0 + 1, inW - 1);
      const float tz = static_cast<float>(z - z0);
      const float ty = static_cast<float>(y - y0);
      const float tx = static_cast<float>(x - x0);

      const float c00 = at(z0, y0, x0) * (1.0f - tx) + at(z0, y0, x1) * tx;
      const float c10 = at(z0, y1, x0) * (1.0f - tx) + at(z0, y1, x1) * tx;
      const float c01 = at(z1, y0, x0) * (1.0f - tx) + at(z1, y0, x1) * tx;
      const float c11 = at(z1, y1, x0) * (1.0f - tx) + at(z1, y1, x1) * tx;
      const float c0 = c00 * (1.0f - ty) + c10 * ty;
      const float c1 = c01 * (1.0f - ty) + c11 * ty;
      return c0 * (1.0f - tz) + c1 * tz;
    };

    auto resampled = FloatImageType::New();
    FloatImageType::RegionType outRegion;
    outRegion.SetSize(targetSize);
    resampled->SetRegions(outRegion);
    resampled->SetSpacing(targetSpacing);
    resampled->SetOrigin(cropped->GetOrigin());
    resampled->SetDirection(cropped->GetDirection());
    resampled->Allocate();
    float* dst = resampled->GetBufferPointer();

    for (int z = 0; z < outD; ++z)
    {
      const double srcZ = (static_cast<double>(z) + 0.5) * static_cast<double>(inD)
                          / static_cast<double>(outD) - 0.5;
      for (int y = 0; y < outH; ++y)
      {
        const double srcY = (static_cast<double>(y) + 0.5) * static_cast<double>(inH)
                            / static_cast<double>(outH) - 0.5;
        for (int x = 0; x < outW; ++x)
        {
          const double srcX = (static_cast<double>(x) + 0.5) * static_cast<double>(inW)
                              / static_cast<double>(outW) - 0.5;
          dst[static_cast<size_t>(z) * outH * outW + static_cast<size_t>(y) * outW + x] =
            sample(srcZ, srcY, srcX);
        }
      }
    }

    cropped = resampled;
    cropped->DisconnectPipeline();
  }

  // 6. Pad to ensure each dimension >= patch_size
  std::cout << "Padding to patch size..." << std::endl;
  {
    auto currentSize = cropped->GetLargestPossibleRegion().GetSize();
    FloatImageType::SizeType padLower, padUpper;
    padLower.Fill(0);
    padUpper.Fill(0);

    for (int d = 0; d < 3; ++d)
    {
      int needed = config.patchSize[d] - static_cast<int>(currentSize[d]);
      if (needed > 0)
      {
        int before = needed / 2;
        int after = needed - before;
        padLower[d] = static_cast<unsigned long>(before);
        padUpper[d] = static_cast<unsigned long>(after);
      }
      result.padBefore[d] = static_cast<int>(padLower[d]);
      result.padAfter[d] = static_cast<int>(padUpper[d]);
    }

    auto padFilter = PadType::New();
    padFilter->SetInput(cropped);
    padFilter->SetPadLowerBound(padLower);
    padFilter->SetPadUpperBound(padUpper);
    padFilter->SetConstant(0.0f);
    padFilter->Update();

    result.image = padFilter->GetOutput();
    result.image->DisconnectPipeline();
  }

  std::cout << "Preprocessing complete. Output size: "
            << result.image->GetLargestPossibleRegion().GetSize() << std::endl;

  return result;
}
