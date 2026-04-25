#include "trt_tta_helpers.hxx"

#include <cstring>

// --- allocateInputs / allocateOutputs / freeIOBuffers (from test_trt.cxx) ---

void allocateInputs(IOBuffers& bufs, nvinfer1::ICudaEngine& engine, const nvinfer1::Dims& inputShape)
{
  int nb = engine.getNbBindings();
  for (int i = 0; i < nb; ++i)
  {
    if (!engine.bindingIsInput(i)) continue;
    IOBuffers::Binding b;
    b.index = i;
    b.name = engine.getBindingName(i);
    b.dtype = engine.getBindingDataType(i);
    b.dims = inputShape;
    b.numElements = 1;
    for (int d = 0; d < b.dims.nbDims; ++d)
      b.numElements *= static_cast<size_t>(b.dims.d[d]);
    b.sizeBytes = b.numElements * sizeof(float);
    b.hostData.resize(b.numElements);
    CUDA_CHECK(cudaMalloc(&b.devicePtr, b.sizeBytes));
    bufs.inputs.push_back(std::move(b));
  }
}

void allocateOutputs(IOBuffers& bufs, nvinfer1::ICudaEngine& engine, nvinfer1::IExecutionContext& context)
{
  int nb = engine.getNbBindings();
  for (int i = 0; i < nb; ++i)
  {
    if (engine.bindingIsInput(i)) continue;
    IOBuffers::Binding b;
    b.index = i;
    b.name = engine.getBindingName(i);
    b.dtype = engine.getBindingDataType(i);
    b.dims = context.getBindingDimensions(i);
    b.numElements = 1;
    for (int d = 0; d < b.dims.nbDims; ++d)
      b.numElements *= static_cast<size_t>(b.dims.d[d]);
    b.sizeBytes = b.numElements * sizeof(float);
    b.hostData.resize(b.numElements);
    CUDA_CHECK(cudaMalloc(&b.devicePtr, b.sizeBytes));
    bufs.outputs.push_back(std::move(b));
  }
}

void freeIOBuffers(IOBuffers& bufs)
{
  for (auto& b : bufs.inputs)
    if (b.devicePtr) { cudaFree(b.devicePtr); b.devicePtr = nullptr; }
  for (auto& b : bufs.outputs)
    if (b.devicePtr) { cudaFree(b.devicePtr); b.devicePtr = nullptr; }
}

// --- TTA flip: bits 0,1,2 = D,H,W ---

void flipTensor5D(std::vector<float>& data, const nvinfer1::Dims& dims, int flipMask)
{
  if (flipMask == 0) return;
  int N = dims.d[0], C = dims.d[1];
  int D = dims.d[2], H = dims.d[3], W = dims.d[4];
  std::vector<float> copy = data;
  size_t idx = 0;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int d = 0; d < D; ++d)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
          {
            int sd = (flipMask & 1) ? (D - 1 - d) : d;
            int sh = (flipMask & 2) ? (H - 1 - h) : h;
            int sw = (flipMask & 4) ? (W - 1 - w) : w;
            size_t srcIdx = static_cast<size_t>((((n * C + c) * D + sd) * H + sh) * W + sw);
            data[idx++] = copy[srcIdx];
          }
}

std::vector<float> runTTAInference(
    nvinfer1::ICudaEngine& engine,
    const std::vector<float>& originalInput,
    const nvinfer1::Dims& inputDims,
    int flipMask,
    cudaStream_t stream)
{
  IOBuffers bufs;
  allocateInputs(bufs, engine, inputDims);

  std::vector<float> flippedInput = originalInput;
  if (flipMask != 0)
    flipTensor5D(flippedInput, inputDims, flipMask);

  auto& inBuf = bufs.inputs[0];
  std::memcpy(inBuf.hostData.data(), flippedInput.data(), inBuf.sizeBytes);
  CUDA_CHECK(cudaMemcpy(inBuf.devicePtr, inBuf.hostData.data(), inBuf.sizeBytes, cudaMemcpyHostToDevice));

  auto context = TrtUniquePtr<nvinfer1::IExecutionContext>(engine.createExecutionContext());
  if (!context) throw std::runtime_error("createExecutionContext failed");
  context->setBindingDimensions(inBuf.index, inputDims);

  for (const auto& b : bufs.inputs)
  {
    auto dims = context->getBindingDimensions(b.index);
    for (int d = 0; d < dims.nbDims; ++d)
      if (dims.d[d] == -1)
        throw std::runtime_error("dynamic dim still -1 on " + b.name);
  }

  allocateOutputs(bufs, engine, *context);
  auto bindings = bufs.buildBindingsArray();
  if (!context->enqueueV2(bindings.data(), stream ? stream : 0, nullptr))
    throw std::runtime_error("enqueueV2 failed");

  if (stream) CUDA_CHECK(cudaStreamSynchronize(stream));
  else        CUDA_CHECK(cudaDeviceSynchronize());

  for (auto& b : bufs.outputs)
    CUDA_CHECK(cudaMemcpy(b.hostData.data(), b.devicePtr, b.sizeBytes, cudaMemcpyDeviceToHost));

  std::vector<float> output = bufs.outputs[0].hostData;
  if (flipMask != 0)
    flipTensor5D(output, bufs.outputs[0].dims, flipMask);

  freeIOBuffers(bufs);
  return output;
}

// --- build / (de)serialize ---

EngineBundle buildEngineFromOnnx(
    const std::string& onnxPath,
    TrtLogger& logger,
    bool fp16,
    int maxWorkspaceGB)
{
  std::cout << "=== Building TensorRT engine from ONNX ===" << std::endl;
  std::cout << "  Model: " << onnxPath << std::endl;

  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) throw std::runtime_error("createInferBuilder");

  const auto explicitBatchFlag =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatchFlag));
  if (!network) throw std::runtime_error("createNetworkV2");

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
  if (!parser) throw std::runtime_error("createParser");

  if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    std::ostringstream oss;
    for (int i = 0; i < parser->getNbErrors(); ++i)
      oss << parser->getError(i)->desc() << "\n";
    throw std::runtime_error("ONNX parse failed:\n" + oss.str());
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMaxWorkspaceSize(static_cast<size_t>(maxWorkspaceGB) * (1ULL << 30));

  {
    const char* inputName = network->getInput(0)->getName();
    nvinfer1::Dims minDims, optDims, maxDims;
    minDims.nbDims = 5; optDims.nbDims = 5; maxDims.nbDims = 5;
    for (int d = 0; d < 5; ++d) { minDims.d[d] = 1; optDims.d[d] = 1; maxDims.d[d] = 1; }
    minDims.d[2] = 64;  minDims.d[3] = 64;  minDims.d[4] = 64;
    optDims.d[2] = 128; optDims.d[3] = 128; optDims.d[4] = 128;
    maxDims.d[2] = 192; maxDims.d[3] = 192; maxDims.d[4] = 192;
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, minDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, optDims);
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, maxDims);
    config->addOptimizationProfile(profile);
  }

  if (fp16 && builder->platformHasFastFp16())
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

  auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!serialized) throw std::runtime_error("buildSerializedNetwork");

  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) throw std::runtime_error("createInferRuntime");
  auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
  if (!engine) throw std::runtime_error("deserializeCudaEngine");
  return { std::move(runtime), std::move(engine) };
}

void serializeEngine(nvinfer1::ICudaEngine& engine, const std::string& path)
{
  auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(engine.serialize());
  if (!serialized) throw std::runtime_error("serialize");
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) throw std::runtime_error("open " + path);
  ofs.write(static_cast<const char*>(serialized->data()), serialized->size());
}

EngineBundle deserializeEngine(const std::string& path, TrtLogger& logger)
{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("open " + path);
  ifs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0, std::ios::beg);
  std::vector<char> buf(size);
  ifs.read(buf.data(), size);
  auto runtime = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!runtime) throw std::runtime_error("createInferRuntime");
  auto engine = TrtUniquePtr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(buf.data(), size));
  if (!engine) throw std::runtime_error("deserializeCudaEngine");
  return { std::move(runtime), std::move(engine) };
}
