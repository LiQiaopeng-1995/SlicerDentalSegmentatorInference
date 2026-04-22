cmake_minimum_required(VERSION 3.16.3)
project(DentalSegmentatorInferenceCLI)

find_package(SlicerExecutionModel REQUIRED)
include(${SlicerExecutionModel_USE_FILE})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# ONNX Runtime — provided by SuperBuild via ONNXRUNTIME_ROOT_DIR
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
  message(FATAL_ERROR "ONNXRUNTIME_ROOT_DIR not set. Build via SuperBuild or set it manually.")
endif()
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
if(WIN32)
  set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/onnxruntime.lib")
else()
  set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")
endif()

# nlohmann/json (header-only, fetch if not found)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

set(MODULE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(MODULE_SRCS
  src/PlansParser.cxx
  src/Preprocessor.cxx
  src/OnnxInference.cxx
  src/SlidingWindow.cxx
  src/Postprocessor.cxx
)

set(MODULE_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
  nlohmann_json::nlohmann_json
)

SEMMacroBuildCLI(
  NAME DentalSegmentatorInference
  TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
  INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
  ADDITIONAL_SRCS ${MODULE_SRCS}
)

target_include_directories(DentalSegmentatorInference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
target_compile_features(DentalSegmentatorInference PRIVATE cxx_std_17)

# Copy ONNX Runtime shared libraries next to executable
if(WIN32)
  file(GLOB _ort_dlls "${ONNXRUNTIME_ROOT_DIR}/lib/*.dll")
  foreach(dll ${_ort_dlls})
    add_custom_command(TARGET DentalSegmentatorInference POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:DentalSegmentatorInference>")
  endforeach()
endif()
