#include "PlansParser.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;
using json = nlohmann::json;

static fs::path findPlansJson(const std::string& modelDir)
{
  // Search in modelDir, then parent, then grandparent.
  // Must run at least one iteration: single-segment relative paths (e.g. "models")
  // have has_parent_path() == false on MSVC, and the old "for (; has_parent_path();)"
  // loop would skip the modelDir check entirely and always throw.
  fs::path dir = fs::path(modelDir);
  for (int level = 0; level < 3; ++level)
  {
    auto candidate = dir / "plans.json";
    if (fs::exists(candidate))
      return candidate;
    // Also check nnUNetPlans.json (nnU-Net v2 naming)
    candidate = dir / "nnUNetPlans.json";
    if (fs::exists(candidate))
      return candidate;
    if (!dir.has_parent_path())
      break;
    dir = dir.parent_path();
  }
  throw std::runtime_error("plans.json not found in or near: " + modelDir);
}

PlansConfig parsePlans(const std::string& modelDir)
{
  auto plansPath = findPlansJson(modelDir);

  std::ifstream ifs(plansPath);
  if (!ifs.is_open())
    throw std::runtime_error("Cannot open: " + plansPath.string());

  json plans = json::parse(ifs);
  PlansConfig cfg{};

  // Transpose axes — convert from numpy (Z,Y,X) convention to ITK (X,Y,Z) dims.
  // numpy axis 0=Z→ITK 2, 1=Y→ITK 1, 2=X→ITK 0;  result[ITK_new] = ITK_old.
  auto transpFwd = plans.value("transpose_forward", json::array({0, 1, 2}));
  auto transpBwd = plans.value("transpose_backward", json::array({0, 1, 2}));
  const int np2itk[3] = {2, 1, 0};  // numpy axis → ITK dimension
  for (int i = 0; i < 3; ++i)
  {
    int npNew = i;
    int npOldFwd = transpFwd[i].get<int>();
    int npOldBwd = transpBwd[i].get<int>();
    cfg.transposeForward[np2itk[npNew]] = np2itk[npOldFwd];
    cfg.transposeBackward[np2itk[npNew]] = np2itk[npOldBwd];
  }

  // Get 3d_fullres configuration
  auto& configs = plans.at("configurations");
  json fullresConfig;
  if (configs.contains("3d_fullres"))
    fullresConfig = configs["3d_fullres"];
  else
    throw std::runtime_error("No 3d_fullres configuration in plans.json");

  // Patch size
  auto patchSizeArr = fullresConfig.at("patch_size");
  for (int i = 0; i < 3; ++i)
    cfg.patchSize[i] = patchSizeArr[i].get<int>();

  // Target spacing
  auto spacingArr = fullresConfig.at("spacing");
  for (int i = 0; i < 3; ++i)
    cfg.targetSpacing[i] = spacingArr[i].get<double>();

  // Normalization parameters from foreground_intensity_properties_per_channel
  // nnU-Net v2 stores per-channel normalization under dataset_properties or plans
  json normParams;
  if (plans.contains("foreground_intensity_properties_per_channel"))
  {
    normParams = plans["foreground_intensity_properties_per_channel"]["0"];
  }
  else if (plans.contains("dataset_properties")
           && plans["dataset_properties"].contains("foreground_intensity_properties_per_channel"))
  {
    normParams = plans["dataset_properties"]["foreground_intensity_properties_per_channel"]["0"];
  }

  if (!normParams.is_null())
  {
    cfg.percentile005 = normParams.value("percentile_00_5", -1024.0);
    cfg.percentile995 = normParams.value("percentile_99_5", 3071.0);
    cfg.mean = normParams.value("mean", 0.0);
    cfg.std = normParams.value("std", 1.0);
  }
  else
  {
    // Defaults for CT
    cfg.percentile005 = -1024.0;
    cfg.percentile995 = 3071.0;
    cfg.mean = 0.0;
    cfg.std = 1.0;
  }

  // Number of classes
  if (plans.contains("num_segmentation_heads"))
    cfg.numClasses = plans["num_segmentation_heads"].get<int>();
  else if (fullresConfig.contains("num_segmentation_heads"))
    cfg.numClasses = fullresConfig["num_segmentation_heads"].get<int>();
  else
    cfg.numClasses = 6;  // DentalSegmentator default: bg + 5 classes

  cfg.numInputChannels = 1;  // Single CT channel

  return cfg;
}
