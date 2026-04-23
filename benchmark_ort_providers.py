#!/usr/bin/env python
"""
Benchmark ONNX Runtime inference with CUDA and TensorRT EP.
"""
import time, os, json
import numpy as np

# Set CUDA 12 lib path for onnxruntime-gpu
nvidia_base = "/home/lqp/miniconda3/lib/python3.10/site-packages/nvidia"
cuda12_paths = [
    f"{nvidia_base}/cublas/lib",
    f"{nvidia_base}/cuda_runtime/lib",
    f"{nvidia_base}/cudnn/lib",
    f"{nvidia_base}/cufft/lib",
    f"{nvidia_base}/cusparse/lib",
    f"{nvidia_base}/cusolver/lib",
    f"{nvidia_base}/nvjitlink/lib",
]
os.environ["LD_LIBRARY_PATH"] = ":".join(cuda12_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import onnxruntime as ort


def benchmark_provider(model_path, patch_size, provider_name, provider_opts=None,
                       num_warmup=5, num_runs=20):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if provider_opts:
        providers = [(provider_name, provider_opts)]
    else:
        providers = [provider_name]
    # Always add CPU fallback
    providers.append("CPUExecutionProvider")

    try:
        sess = ort.InferenceSession(model_path, opts, providers=providers)
    except Exception as e:
        print(f"  FAILED to create session: {e}")
        return None

    active = sess.get_providers()
    print(f"  Active providers: {active}")
    if provider_name.replace("ExecutionProvider", "EP") not in str(active) and \
       provider_name not in str(active):
        print(f"  WARNING: {provider_name} not active!")

    D, H, W = patch_size
    dummy = np.random.randn(1, 1, D, H, W).astype(np.float32)
    input_name = sess.get_inputs()[0].name

    # Warmup
    print(f"  Warming up ({num_warmup})...")
    for _ in range(num_warmup):
        sess.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    print(f"  Mean: {times.mean()*1000:.1f}ms | Std: {times.std()*1000:.1f}ms | "
          f"Min: {times.min()*1000:.1f}ms | Max: {times.max()*1000:.1f}ms")
    return times.mean()


def main():
    model_path = "model_weights/dental_segmentator.onnx"
    plans_path = "model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json"

    with open(plans_path) as f:
        plans = json.load(f)
    patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

    print("=" * 60)
    print("ONNX Runtime Provider Benchmark")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Patch: {patch_size}")
    print(f"Available: {ort.get_available_providers()}")

    results = {}

    # 1. CPU
    print(f"\n--- CPUExecutionProvider ---")
    t = benchmark_provider(model_path, patch_size, "CPUExecutionProvider",
                           num_warmup=2, num_runs=3)
    if t: results["CPU"] = t

    # 2. CUDA
    print(f"\n--- CUDAExecutionProvider ---")
    t = benchmark_provider(model_path, patch_size, "CUDAExecutionProvider",
                           {"device_id": "0"})
    if t: results["CUDA"] = t

    # 3. TensorRT FP32
    print(f"\n--- TensorrtExecutionProvider (FP32) ---")
    cache_dir = "/tmp/trt_cache_fp32"
    os.makedirs(cache_dir, exist_ok=True)
    t = benchmark_provider(model_path, patch_size, "TensorrtExecutionProvider", {
        "device_id": "0",
        "trt_fp16_enable": "0",
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": cache_dir,
    }, num_warmup=3, num_runs=10)
    if t: results["TRT_FP32"] = t

    # 4. TensorRT FP16
    print(f"\n--- TensorrtExecutionProvider (FP16) ---")
    cache_dir = "/tmp/trt_cache_fp16"
    os.makedirs(cache_dir, exist_ok=True)
    t = benchmark_provider(model_path, patch_size, "TensorrtExecutionProvider", {
        "device_id": "0",
        "trt_fp16_enable": "1",
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": cache_dir,
    }, num_warmup=3, num_runs=10)
    if t: results["TRT_FP16"] = t

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (single patch)")
    print(f"{'='*60}")
    for name, t in results.items():
        tiles_90 = t * 90
        tiles_100 = t * 100
        print(f"  {name:<12}: {t*1000:>7.1f} ms/patch | "
              f"90 tiles: {tiles_90:.1f}s | 100 tiles: {tiles_100:.1f}s")


if __name__ == "__main__":
    main()
