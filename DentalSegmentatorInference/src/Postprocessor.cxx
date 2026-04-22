#include "Postprocessor.h"

#include <itkImageFileWriter.h>
#include <itkResampleImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPermuteAxesImageFilter.h>
#include <itkImageRegionIterator.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using LabelPermuteType = itk::PermuteAxesImageFilter<LabelImageType>;
using LinearInterpType = itk::LinearInterpolateImageFunction<FloatImageType, double>;
using FloatResampleType = itk::ResampleImageFilter<FloatImageType, FloatImageType>;
using WriterType = itk::ImageFileWriter<LabelImageType>;

void postprocess(const std::vector<float>& logits,
                 const std::array<int, 3>& logitsShape,
                 int numClasses,
                 const PreprocessResult& prepResult,
                 const PlansConfig& config,
                 const std::string& outputPath)
{
  int D = logitsShape[0], H = logitsShape[1], W = logitsShape[2];
  size_t spatialSize = static_cast<size_t>(D) * H * W;

  // 1. Remove padding: extract the unpadded region from logits
  int unpaddedD = D - prepResult.padBefore[0] - prepResult.padAfter[0];
  int unpaddedH = H - prepResult.padBefore[1] - prepResult.padAfter[1];
  int unpaddedW = W - prepResult.padBefore[2] - prepResult.padAfter[2];

  std::cout << "Postprocessing: unpadding to [" << unpaddedD << ", " << unpaddedH << ", " << unpaddedW << "]" << std::endl;

  // 2. For each class, create ITK image of logits (unpadded), then resample to pre-resampling size
  auto targetSize = prepResult.sizeBeforeResampling;
  std::cout << "Resampling logits to pre-resampling size: " << targetSize << std::endl;

  // We'll do argmax after resampling individual class logit maps
  // Create label image at the resampled-back resolution
  auto labelImage = LabelImageType::New();
  {
    LabelImageType::RegionType labelRegion;
    LabelImageType::IndexType labelStart;
    labelStart.Fill(0);
    labelRegion.SetIndex(labelStart);
    labelRegion.SetSize(targetSize);
    labelImage->SetRegions(labelRegion);
    labelImage->Allocate(true);  // zero-initialized
  }

  // Process one class at a time to save memory
  // For each voxel, track the max logit and corresponding class
  size_t targetVoxels = static_cast<size_t>(targetSize[0]) * targetSize[1] * targetSize[2];
  std::vector<float> maxLogit(targetVoxels, -std::numeric_limits<float>::infinity());

  for (int c = 0; c < numClasses; ++c)
  {
    // Create float image from unpadded logits for this class
    auto classImage = FloatImageType::New();
    FloatImageType::RegionType classRegion;
    FloatImageType::IndexType classStart;
    classStart.Fill(0);
    FloatImageType::SizeType classSize;
    classSize[0] = unpaddedD; classSize[1] = unpaddedH; classSize[2] = unpaddedW;
    classRegion.SetIndex(classStart);
    classRegion.SetSize(classSize);
    classImage->SetRegions(classRegion);

    // Set spacing to the target spacing used during preprocessing resampling
    FloatImageType::SpacingType classSpacing;
    for (int d = 0; d < 3; ++d)
      classSpacing[d] = config.targetSpacing[d];
    classImage->SetSpacing(classSpacing);
    classImage->Allocate();

    // Copy unpadded logits
    size_t classOffset = static_cast<size_t>(c) * spatialSize;
    itk::ImageRegionIterator<FloatImageType> it(classImage, classRegion);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      auto idx = it.GetIndex();
      int z = idx[0] + prepResult.padBefore[0];
      int y = idx[1] + prepResult.padBefore[1];
      int x = idx[2] + prepResult.padBefore[2];
      size_t srcIdx = classOffset + static_cast<size_t>(z) * H * W + y * W + x;
      it.Set(logits[srcIdx]);
    }

    // Resample this class logits to pre-resampling size
    FloatImageType::SpacingType origSpacing;
    // Compute what spacing the pre-resampling image had
    // It was the spacing after transpose + crop (before resample to target)
    // We stored sizeBeforeResampling, and the classImage has target spacing
    // Original spacing = target_spacing * unpadded_size / sizeBeforeResampling
    for (int d = 0; d < 3; ++d)
    {
      if (targetSize[d] > 0)
        origSpacing[d] = config.targetSpacing[d] * static_cast<double>(unpaddedD > 0 ? classSize[d] : 1) / targetSize[d];
      else
        origSpacing[d] = config.targetSpacing[d];
    }

    auto interpolator = LinearInterpType::New();
    auto resample = FloatResampleType::New();
    resample->SetInput(classImage);
    resample->SetInterpolator(interpolator);
    resample->SetSize(targetSize);
    resample->SetOutputSpacing(origSpacing);
    resample->SetOutputOrigin(classImage->GetOrigin());
    resample->SetOutputDirection(classImage->GetDirection());
    resample->SetDefaultPixelValue(0.0f);
    resample->Update();
    auto resampledClass = resample->GetOutput();

    // Update argmax
    itk::ImageRegionConstIterator<FloatImageType> rit(resampledClass, resampledClass->GetLargestPossibleRegion());
    itk::ImageRegionIterator<LabelImageType> lit(labelImage, labelImage->GetLargestPossibleRegion());
    size_t vi = 0;
    for (rit.GoToBegin(), lit.GoToBegin(); !rit.IsAtEnd(); ++rit, ++lit, ++vi)
    {
      float val = rit.Get();
      if (val > maxLogit[vi])
      {
        maxLogit[vi] = val;
        lit.Set(static_cast<unsigned char>(c));
      }
    }
  }

  std::cout << "Argmax complete." << std::endl;

  // 3. Uncrop: create full-size label image and paste the segmentation back
  std::cout << "Uncropping..." << std::endl;
  // The "full size" here is the transposed image size before cropping
  // We need originalSize permuted by transposeForward
  FloatImageType::SizeType transposedFullSize;
  for (int d = 0; d < 3; ++d)
    transposedFullSize[d] = prepResult.originalSize[config.transposeForward[d]];

  auto fullLabel = LabelImageType::New();
  {
    LabelImageType::RegionType fullRegion;
    LabelImageType::IndexType fullStart;
    fullStart.Fill(0);
    fullRegion.SetIndex(fullStart);
    fullRegion.SetSize(transposedFullSize);
    fullLabel->SetRegions(fullRegion);
    fullLabel->Allocate(true);  // zero = background
  }

  // Paste labelImage into fullLabel at cropStart position
  itk::ImageRegionConstIterator<LabelImageType> srcIt(labelImage, labelImage->GetLargestPossibleRegion());
  for (srcIt.GoToBegin(); !srcIt.IsAtEnd(); ++srcIt)
  {
    auto srcIdx = srcIt.GetIndex();
    LabelImageType::IndexType dstIdx;
    for (int d = 0; d < 3; ++d)
      dstIdx[d] = srcIdx[d] + prepResult.cropStart[d];

    // Bounds check
    bool inBounds = true;
    for (int d = 0; d < 3; ++d)
    {
      if (dstIdx[d] < 0 || static_cast<unsigned long>(dstIdx[d]) >= transposedFullSize[d])
      {
        inBounds = false;
        break;
      }
    }
    if (inBounds)
      fullLabel->SetPixel(dstIdx, srcIt.Get());
  }

  // 4. Transpose backward
  std::cout << "Transposing backward..." << std::endl;
  auto permute = LabelPermuteType::New();
  itk::FixedArray<unsigned int, 3> backOrder;
  for (int i = 0; i < 3; ++i)
    backOrder[i] = static_cast<unsigned int>(config.transposeBackward[i]);
  permute->SetOrder(backOrder);
  permute->SetInput(fullLabel);
  permute->Update();
  auto finalLabel = permute->GetOutput();

  // Restore original spacing/origin/direction
  finalLabel->SetSpacing(prepResult.originalSpacing);
  finalLabel->SetOrigin(prepResult.originalOrigin);
  finalLabel->SetDirection(prepResult.originalDirection);

  // 5. Write output
  std::cout << "Writing output: " << outputPath << std::endl;
  auto writer = WriterType::New();
  writer->SetFileName(outputPath);
  writer->SetInput(finalLabel);
  writer->SetUseCompression(true);
  writer->Update();

  std::cout << "Postprocessing complete." << std::endl;
}
