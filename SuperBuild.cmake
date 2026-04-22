#-----------------------------------------------------------------------------
# SuperBuild script: download ONNX Runtime, then configure inner build
#-----------------------------------------------------------------------------

include(ExternalProject)

set(ONNXRUNTIME_VERSION "1.17.1")

if(WIN32)
  set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-win-x64-gpu-${ONNXRUNTIME_VERSION}.zip")
elseif(UNIX AND NOT APPLE)
  set(_ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
else()
  message(FATAL_ERROR "ONNX Runtime GPU builds are only provided for Windows and Linux x64.")
endif()

set(ONNXRUNTIME_ROOT_DIR "${CMAKE_BINARY_DIR}/onnxruntime-install")

ExternalProject_Add(onnxruntime
  URL ${_ort_url}
  SOURCE_DIR ${ONNXRUNTIME_ROOT_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

#-----------------------------------------------------------------------------
# Inner build (the actual extension), runs after onnxruntime is downloaded
#-----------------------------------------------------------------------------

set(proj DentalSegmentatorInference_inner)
ExternalProject_Add(${proj}
  DOWNLOAD_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build
  CMAKE_CACHE_ARGS
    -DSlicer_DIR:PATH=${Slicer_DIR}
    -DONNXRUNTIME_ROOT_DIR:PATH=${ONNXRUNTIME_ROOT_DIR}
    -DDentalSegmentatorInference_SUPERBUILD:BOOL=OFF
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_STANDARD:STRING=17
  DEPENDS onnxruntime
  INSTALL_COMMAND ""
)
