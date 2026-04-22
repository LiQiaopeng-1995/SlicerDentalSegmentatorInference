#pragma once

#include <array>
#include <string>
#include <vector>

/**
 * Parses nnU-Net plans.json to extract preprocessing and inference parameters.
 */
struct PlansConfig
{
  // Transpose axes order (e.g., [0,1,2] for no transpose)
  std::array<int, 3> transposeForward;
  std::array<int, 3> transposeBackward;

  // CT normalization parameters
  double percentile005;
  double percentile995;
  double mean;
  double std;

  // Target spacing after resampling
  std::array<double, 3> targetSpacing;

  // Patch size for sliding window
  std::array<int, 3> patchSize;

  // Number of output classes (including background)
  int numClasses;

  // Number of input channels
  int numInputChannels;
};

/**
 * Parse the plans.json file from an nnU-Net model directory.
 * Looks for plans.json in modelDir and common parent directories.
 */
PlansConfig parsePlans(const std::string& modelDir);
