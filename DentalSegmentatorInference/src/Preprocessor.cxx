#include "Preprocessor.h"

#include <itkImageFileReader.h>
#include <itkCastImageFilter.h>
#include <itkPermuteAxesImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkResampleImageFilter.h>
#include <itkBSplineInterpolateImageFunction.h>
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
using ResampleType = itk::ResampleImageFilter<FloatImageType, FloatImageType>;
using BSplineInterpType = itk::BSplineInterpolateImageFunction<FloatImageType, double, float>;
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
    if (std::abs(it.Get()) > 1e-8f)
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

  // 5. Resample to target spacing (BSpline order 3)
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

    auto interpolator = BSplineInterpType::New();
    interpolator->SetSplineOrder(3);

    auto resample = ResampleType::New();
    resample->SetInput(cropped);
    resample->SetInterpolator(interpolator);
    resample->SetOutputSpacing(targetSpacing);
    resample->SetSize(targetSize);
    resample->SetOutputOrigin(cropped->GetOrigin());
    resample->SetOutputDirection(cropped->GetDirection());
    resample->SetDefaultPixelValue(0.0f);
    resample->Update();

    cropped = resample->GetOutput();
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
