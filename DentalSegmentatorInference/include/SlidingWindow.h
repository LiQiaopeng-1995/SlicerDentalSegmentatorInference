#pragma once

#include "OnnxInference.h"
#include "PlansParser.h"

#include <vector>
#include <array>

/**
 * Sliding window inference with Gaussian importance weighting.
 *
 * Splits the preprocessed volume into overlapping patches, runs inference
 * on each, and aggregates results using Gaussian-weighted blending.
 */
class SlidingWindow
{
public:
  /**
   * @param inference   OnnxInference session (must be initialized)
   * @param patchSize   Patch size [D, H, W]
   * @param numClasses  Number of output classes
   * @param stepSize    Overlap ratio (0.75 = 75% of patch size step)
   */
  SlidingWindow(OnnxInference& inference,
                const std::array<int, 3>& patchSize,
                int numClasses,
                float stepSize = 0.75f);

  /**
   * Run sliding window inference over the full volume.
   * @param volumeData  Pointer to float volume data [D, H, W]
   * @param volumeShape Volume dimensions {D, H, W}
   * @return Aggregated logits as flat vector, shape [C, D, H, W]
   */
  std::vector<float> run(const float* volumeData,
                         const std::array<int, 3>& volumeShape);

private:
  /** Compute 3D Gaussian importance map for the patch size */
  std::vector<float> computeGaussianMap() const;

  OnnxInference& m_inference;
  std::array<int, 3> m_patchSize;
  int m_numClasses;
  float m_stepSize;
};
