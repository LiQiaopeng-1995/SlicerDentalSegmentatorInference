#include "Postprocessor.h"

#include <itkImageFileWriter.h>
#include <itkPermuteAxesImageFilter.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using LabelPermuteType = itk::PermuteAxesImageFilter<LabelImageType>;
using WriterType = itk::ImageFileWriter<LabelImageType>;

static float sampleLogitLinearBorder(const std::vector<float>& logits,
                                     size_t classOffset,
                                     int fullD,
                                     int fullH,
                                     int fullW,
                                     const std::array<int, 3>& padBefore,
                                     int unpaddedD,
                                     int unpaddedH,
                                     int unpaddedW,
                                     double d,
                                     double h,
                                     double w)
{
  auto clampCoord = [](double v, int n) {
    if (n <= 1) return 0.0;
    return std::clamp(v, 0.0, static_cast<double>(n - 1));
  };

  d = clampCoord(d, unpaddedD);
  h = clampCoord(h, unpaddedH);
  w = clampCoord(w, unpaddedW);

  const int d0 = static_cast<int>(std::floor(d));
  const int h0 = static_cast<int>(std::floor(h));
  const int w0 = static_cast<int>(std::floor(w));
  const int d1 = std::min(d0 + 1, unpaddedD - 1);
  const int h1 = std::min(h0 + 1, unpaddedH - 1);
  const int w1 = std::min(w0 + 1, unpaddedW - 1);

  const float td = static_cast<float>(d - d0);
  const float th = static_cast<float>(h - h0);
  const float tw = static_cast<float>(w - w0);

  auto at = [&](int ld, int lh, int lw) {
    const int gd = ld + padBefore[0];
    const int gh = lh + padBefore[1];
    const int gw = lw + padBefore[2];
    return logits[classOffset + static_cast<size_t>(gd) * fullH * fullW
                  + static_cast<size_t>(gh) * fullW + gw];
  };

  const float c00 = at(d0, h0, w0) * (1.0f - tw) + at(d0, h0, w1) * tw;
  const float c10 = at(d0, h1, w0) * (1.0f - tw) + at(d0, h1, w1) * tw;
  const float c01 = at(d1, h0, w0) * (1.0f - tw) + at(d1, h0, w1) * tw;
  const float c11 = at(d1, h1, w0) * (1.0f - tw) + at(d1, h1, w1) * tw;
  const float c0 = c00 * (1.0f - th) + c10 * th;
  const float c1 = c01 * (1.0f - th) + c11 * th;
  return c0 * (1.0f - td) + c1 * td;
}

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

  // 2. Resample class logits back to pre-resampling size with PyTorch
  // interpolate(align_corners=False), then argmax.
  auto targetSize = prepResult.sizeBeforeResampling;
  std::cout << "Resampling logits to pre-resampling size: " << targetSize << std::endl;

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

  const int targetD = static_cast<int>(targetSize[2]);
  const int targetH = static_cast<int>(targetSize[1]);
  const int targetW = static_cast<int>(targetSize[0]);
  for (int z = 0; z < targetD; ++z)
  {
    const double srcD = (static_cast<double>(z) + 0.5) * static_cast<double>(unpaddedD)
                        / static_cast<double>(targetD) - 0.5;
    for (int y = 0; y < targetH; ++y)
    {
      const double srcH = (static_cast<double>(y) + 0.5) * static_cast<double>(unpaddedH)
                          / static_cast<double>(targetH) - 0.5;
      for (int x = 0; x < targetW; ++x)
      {
        const double srcW = (static_cast<double>(x) + 0.5) * static_cast<double>(unpaddedW)
                            / static_cast<double>(targetW) - 0.5;
        float best = -std::numeric_limits<float>::infinity();
        unsigned char bestClass = 0;
        for (int c = 0; c < numClasses; ++c)
        {
          const size_t classOffset = static_cast<size_t>(c) * spatialSize;
          const float val = sampleLogitLinearBorder(logits, classOffset, D, H, W, prepResult.padBefore,
                                                    unpaddedD, unpaddedH, unpaddedW, srcD, srcH, srcW);
          if (val > best)
          {
            best = val;
            bestClass = static_cast<unsigned char>(c);
          }
        }

        LabelImageType::IndexType idx;
        idx[0] = x;
        idx[1] = y;
        idx[2] = z;
        labelImage->SetPixel(idx, bestClass);
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
