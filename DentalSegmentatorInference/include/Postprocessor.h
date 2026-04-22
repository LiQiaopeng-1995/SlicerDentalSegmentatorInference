#pragma once

#include "Preprocessor.h"
#include "PlansParser.h"

#include <itkImage.h>
#include <string>
#include <vector>

using LabelImageType = itk::Image<unsigned char, 3>;

/**
 * Post-process inference logits back to a label map in original image space.
 *
 * Steps: resample logits → argmax → unpad → uncrop → transpose backward → write
 */
void postprocess(const std::vector<float>& logits,
                 const std::array<int, 3>& logitsShape,  // [D, H, W] of padded/resampled volume
                 int numClasses,
                 const PreprocessResult& prepResult,
                 const PlansConfig& config,
                 const std::string& outputPath);
