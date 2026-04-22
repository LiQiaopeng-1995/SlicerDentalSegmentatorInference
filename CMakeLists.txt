cmake_minimum_required(VERSION 3.16.3)

project(DentalSegmentatorInference)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/gaudot/SlicerDentalSegmentator")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "SlicerDentalSegmentator contributors")
set(EXTENSION_DESCRIPTION "C++ ONNX Runtime inference backend for DentalSegmentator. Provides GPU-accelerated dental CT/CBCT segmentation via TensorRT/CUDA.")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# SuperBuild (downloads ONNX Runtime)
option(DentalSegmentatorInference_SUPERBUILD "Build as SuperBuild to fetch ONNX Runtime" ON)
if(DentalSegmentatorInference_SUPERBUILD)
  include("${CMAKE_CURRENT_SOURCE_DIR}/SuperBuild.cmake")
  return()
endif()

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DentalSegmentatorInference)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
