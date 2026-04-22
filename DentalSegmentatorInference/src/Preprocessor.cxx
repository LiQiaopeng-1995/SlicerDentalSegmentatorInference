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

  // 2. Transpose forward
  std::cout << "Transposing axes..." << std::endl;
  auto permute = PermuteType::New();
  itk::FixedArray<unsigned int, 3> order;
  for (int i = 0; i < 3; ++i)
    order[i] = static_cast<unsigned int>(config.transposeForward[i]);
  permute->SetOrder(order);
  permute->SetInput(floatImage);
  permute->Update();
  auto transposed = permute->GetOutput();
  transposed->DisconnectPipeline();

  // 3. Crop to non-zero bounding box
  std::cout << "Cropping to non-zero region..." << std::endl;
  FloatImageType::IndexType bboxStart;
  FloatImageType::SizeType bboxSize;
  computeNonzeroBBox(transposed, bboxStart, bboxSize);
  result.cropStart = bboxStart;
  result.croppedSize = bboxSize;

  FloatImageType::RegionType cropRegion(bboxStart, bboxSize);
  auto extract = ExtractType::New();
  extract->SetExtractionRegion(cropRegion);
  extract->SetInput(transposed);
  extract->SetDirectionCollapseToSubmatrix();
  extract->Update();
  auto cropped = extract->GetOutput();
  cropped->DisconnectPipeline();

  // Reset origin to 0 after cropping for simpler resampling
  FloatImageType::PointType zeroOrigin;
  zeroOrigin.Fill(0.0);
  cropped->SetOrigin(zeroOrigin);

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
