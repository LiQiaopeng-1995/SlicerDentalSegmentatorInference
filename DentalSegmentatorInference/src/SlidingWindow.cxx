#include "SlidingWindow.h"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SlidingWindow::SlidingWindow(OnnxInference& inference,
                             const std::array<int, 3>& patchSize,
                             int numClasses,
                             float stepSize)
  : m_inference(inference)
  , m_patchSize(patchSize)
  , m_numClasses(numClasses)
  , m_stepSize(stepSize)
{
}

std::vector<float> SlidingWindow::computeGaussianMap() const
{
  size_t total = static_cast<size_t>(m_patchSize[0]) * m_patchSize[1] * m_patchSize[2];
  std::vector<float> gaussian(total);

  // Compute 1D Gaussians for each axis, then outer product
  std::array<std::vector<float>, 3> g1d;
  for (int d = 0; d < 3; ++d)
  {
    int n = m_patchSize[d];
    float sigma = static_cast<float>(n) / 8.0f;
    float center = (n - 1.0f) / 2.0f;
    g1d[d].resize(n);
    for (int i = 0; i < n; ++i)
    {
      float diff = static_cast<float>(i) - center;
      g1d[d][i] = std::exp(-0.5f * diff * diff / (sigma * sigma));
    }
  }

  // 3D outer product
  size_t idx = 0;
  for (int z = 0; z < m_patchSize[0]; ++z)
    for (int y = 0; y < m_patchSize[1]; ++y)
      for (int x = 0; x < m_patchSize[2]; ++x)
        gaussian[idx++] = g1d[0][z] * g1d[1][y] * g1d[2][x];

  // Clamp minimum to avoid division issues
  float maxVal = *std::max_element(gaussian.begin(), gaussian.end());
  float minVal = maxVal * 1e-4f;
  for (auto& v : gaussian)
    v = std::max(v, minVal);

  return gaussian;
}

std::vector<float> SlidingWindow::run(const float* volumeData,
                                      const std::array<int, 3>& volumeShape)
{
  int D = volumeShape[0], H = volumeShape[1], W = volumeShape[2];
  size_t spatialSize = static_cast<size_t>(D) * H * W;

  // Allocate output buffers: [C, D, H, W]
  size_t totalLogits = static_cast<size_t>(m_numClasses) * spatialSize;
  std::vector<float> aggregated(totalLogits, 0.0f);
  std::vector<float> weightSum(spatialSize, 0.0f);

  auto gaussian = computeGaussianMap();
  size_t patchVoxels = static_cast<size_t>(m_patchSize[0]) * m_patchSize[1] * m_patchSize[2];

  // Compute tile positions for each axis
  auto computeStarts = [](int volSize, int patchSize, float stepSize) -> std::vector<int> {
    std::vector<int> starts;
    if (volSize <= patchSize)
    {
      starts.push_back(0);
      return starts;
    }
    int step = static_cast<int>(std::round(patchSize * stepSize));
    if (step < 1) step = 1;
    for (int s = 0; s + patchSize <= volSize; s += step)
      starts.push_back(s);
    // Ensure last patch covers the end
    if (starts.back() + patchSize < volSize)
      starts.push_back(volSize - patchSize);
    return starts;
  };

  auto startsZ = computeStarts(D, m_patchSize[0], m_stepSize);
  auto startsY = computeStarts(H, m_patchSize[1], m_stepSize);
  auto startsX = computeStarts(W, m_patchSize[2], m_stepSize);

  int totalPatches = static_cast<int>(startsZ.size() * startsY.size() * startsX.size());
  std::cout << "Sliding window: " << totalPatches << " patches ("
            << startsZ.size() << "x" << startsY.size() << "x" << startsX.size()
            << "), step_size=" << m_stepSize << std::endl;

  int patchIdx = 0;
  std::vector<float> patchBuf(patchVoxels);

  for (int sz : startsZ)
  {
    for (int sy : startsY)
    {
      for (int sx : startsX)
      {
        // Extract patch from volume
        size_t bufIdx = 0;
        for (int z = sz; z < sz + m_patchSize[0]; ++z)
          for (int y = sy; y < sy + m_patchSize[1]; ++y)
            for (int x = sx; x < sx + m_patchSize[2]; ++x)
              patchBuf[bufIdx++] = volumeData[static_cast<size_t>(z) * H * W + y * W + x];

        // Run inference: input shape [1, 1, D, H, W]
        std::vector<int64_t> inputShape = {
          1, 1,
          static_cast<int64_t>(m_patchSize[0]),
          static_cast<int64_t>(m_patchSize[1]),
          static_cast<int64_t>(m_patchSize[2])
        };
        auto logits = m_inference.runPatch(patchBuf.data(), inputShape);

        // logits shape: [1, C, pD, pH, pW] — accumulate with Gaussian weight
        for (int c = 0; c < m_numClasses; ++c)
        {
          size_t classOffset = static_cast<size_t>(c) * patchVoxels;
          size_t globalClassOffset = static_cast<size_t>(c) * spatialSize;
          size_t gi = 0;
          for (int z = sz; z < sz + m_patchSize[0]; ++z)
          {
            for (int y = sy; y < sy + m_patchSize[1]; ++y)
            {
              for (int x = sx; x < sx + m_patchSize[2]; ++x)
              {
                size_t globalIdx = static_cast<size_t>(z) * H * W + y * W + x;
                float w = gaussian[gi];
                aggregated[globalClassOffset + globalIdx] += logits[classOffset + gi] * w;
                if (c == 0)
                  weightSum[globalIdx] += w;
                ++gi;
              }
            }
          }
        }

        ++patchIdx;
        if (patchIdx % 5 == 0 || patchIdx == totalPatches)
          std::cout << "  Patch " << patchIdx << "/" << totalPatches << std::endl;
      }
    }
  }

  // Normalize by accumulated weights
  for (int c = 0; c < m_numClasses; ++c)
  {
    size_t offset = static_cast<size_t>(c) * spatialSize;
    for (size_t i = 0; i < spatialSize; ++i)
    {
      if (weightSum[i] > 0.0f)
        aggregated[offset + i] /= weightSum[i];
    }
  }

  std::cout << "Sliding window inference complete." << std::endl;
  return aggregated;
}
