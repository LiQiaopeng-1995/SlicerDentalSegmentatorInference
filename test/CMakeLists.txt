cmake_minimum_required(VERSION 3.16)
project(DentalSegmentatorTest CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# ONNX Runtime
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR)
  message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime-gpu")
endif()
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_DIR}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT_DIR}/lib/libonnxruntime.so")

# nlohmann/json
include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# Source files from the main project
set(PROJECT_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../DentalSegmentatorInference")

set(SRCS
  test_main.cxx
  ${PROJECT_SRC_DIR}/src/PlansParser.cxx
  ${PROJECT_SRC_DIR}/src/Preprocessor.cxx
  ${PROJECT_SRC_DIR}/src/OnnxInference.cxx
  ${PROJECT_SRC_DIR}/src/SlidingWindow.cxx
  ${PROJECT_SRC_DIR}/src/Postprocessor.cxx
)

add_executable(dental_test ${SRCS})
target_include_directories(dental_test PRIVATE
  ${PROJECT_SRC_DIR}/include
  ${ONNXRUNTIME_INCLUDE_DIRS}
)
target_link_libraries(dental_test PRIVATE
  ${ITK_LIBRARIES}
  ${ONNXRUNTIME_LIBRARIES}
  nlohmann_json::nlohmann_json
)
