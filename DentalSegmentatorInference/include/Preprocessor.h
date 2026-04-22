#pragma once

#include "PlansParser.h"

#include <itkImage.h>
#include <array>

using FloatImageType = itk::Image<float, 3>;

/**
 * Holds the result of preprocessing, including transformed image
 * and metadata needed to reverse the transformations in postprocessing.
 */
struct PreprocessResult
{
  FloatImageType::Pointer image;  // Preprocessed float image ready for inference

  // Original image properties (before any transforms) for postprocessing
  FloatImageType::SizeType    originalSize;
  FloatImageType::SpacingType originalSpacing;
  FloatImageType::PointType   originalOrigin;
  FloatImageType::DirectionType originalDirection;

  // Crop bounding box (in transposed space, before resampling)
  FloatImageType::IndexType  cropStart;
  FloatImageType::SizeType   croppedSize;

  // Size after crop but before resampling (needed for logit resampling)
  FloatImageType::SizeType   sizeBeforeResampling;

  // Padding applied
  std::array<int, 3> padBefore;
  std::array<int, 3> padAfter;
};

/**
 * Run nnU-Net-equivalent preprocessing on an input volume.
 *
 * Steps: read → transpose → crop to nonzero → CT normalize → resample → pad
 */
PreprocessResult preprocess(const std::string& inputPath, const PlansConfig& config);
