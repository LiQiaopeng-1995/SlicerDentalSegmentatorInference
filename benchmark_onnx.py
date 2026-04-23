#!/usr/bin/env python
"""
Benchmark ONNX model inference speed.
Tests single-patch and sliding-window inference on synthetic data.
"""
import argparse
import time
import json
import numpy as np
import onnxruntime as ort


def create_session(model_path, device="cuda"):
    """Create ONNX Runtime session with GPU providers."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = []
    if device == "cuda":
        # Try TensorRT first, then CUDA fallback
        trt_opts = {
            "trt_fp16_enable": "1",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": "/tmp/trt_cache",
        }
        providers.append(("TensorrtExecutionProvider", trt_opts))
        providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
    providers.append(("CPUExecutionProvider", {}))

    session = ort.InferenceSession(model_path, opts, providers=providers)
    active = session.get_providers()
    print(f"Active providers: {active}")
    return session


def benchmark_single_patch(session, patch_size, num_warmup=3, num_runs=10):
    """Benchmark single patch inference."""
    D, H, W = patch_size
    dummy = np.random.randn(1, 1, D, H, W).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    print(f"\nSingle patch [{D}x{H}x{W}] — warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Benchmark
    times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {times[-1]*1000:.1f} ms")

    times = np.array(times)
    print(f"  Mean: {times.mean()*1000:.1f} ms | Std: {times.std()*1000:.1f} ms | "
          f"Min: {times.min()*1000:.1f} ms | Max: {times.max()*1000:.1f} ms")
    return times.mean()


def make_gaussian_importance_map(patch_size, sigma_scale=1.0/8):
    """Create 3D Gaussian importance map for sliding window."""
    maps = []
    for s in patch_size:
        sigma = s * sigma_scale
        coords = np.arange(s, dtype=np.float32) - (s - 1) / 2.0
        g = np.exp(-0.5 * (coords / sigma) ** 2)
        maps.append(g)
    gauss = maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]
    gauss /= gauss.max()
    gauss = np.clip(gauss, 1e-8, None)
    return gauss


def benchmark_sliding_window(session, volume_shape, patch_size, num_classes,
                              step_size=0.75, num_runs=3):
    """Benchmark full sliding-window inference on a synthetic volume."""
    D, H, W = volume_shape
    pD, pH, pW = patch_size
    input_name = session.get_inputs()[0].name

    # Compute tile positions
    def get_steps(vol_s, patch_s, step_s):
        step = max(1, int(patch_s * step_s))
        positions = list(range(0, max(vol_s - patch_s + 1, 1), step))
        if positions[-1] + patch_s < vol_s:
            positions.append(vol_s - patch_s)
        return positions

    pos_d = get_steps(D, pD, step_size)
    pos_h = get_steps(H, pH, step_size)
    pos_w = get_steps(W, pW, step_size)
    total_tiles = len(pos_d) * len(pos_h) * len(pos_w)

    print(f"\nSliding window: volume [{D}x{H}x{W}], patch [{pD}x{pH}x{pW}], "
          f"step={step_size}")
    print(f"  Tile positions: D={len(pos_d)}, H={len(pos_h)}, W={len(pos_w)}, "
          f"total={total_tiles} tiles")

    gaussian = make_gaussian_importance_map(patch_size)

    for run_idx in range(num_runs):
        volume = np.random.randn(D, H, W).astype(np.float32)
        aggregated = np.zeros((num_classes, D, H, W), dtype=np.float32)
        weight_map = np.zeros((D, H, W), dtype=np.float32)

        t0 = time.perf_counter()
        tile_idx = 0
        for d in pos_d:
            for h in pos_h:
                for w in pos_w:
                    patch = volume[d:d+pD, h:h+pH, w:w+pW]
                    inp = patch[np.newaxis, np.newaxis, ...]  # [1,1,D,H,W]
                    out = session.run(None, {input_name: inp})[0]  # [1,C,D,H,W]
                    logits = out[0]  # [C,D,H,W]
                    for c in range(num_classes):
                        aggregated[c, d:d+pD, h:h+pH, w:w+pW] += logits[c] * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian
                    tile_idx += 1

        # Normalize
        weight_map = np.clip(weight_map, 1e-8, None)
        for c in range(num_classes):
            aggregated[c] /= weight_map

        # Argmax
        seg = np.argmax(aggregated, axis=0).astype(np.uint8)
        t1 = time.perf_counter()

        print(f"  Run {run_idx+1}: {t1-t0:.2f}s ({total_tiles} tiles, "
              f"{(t1-t0)/total_tiles*1000:.1f} ms/tile)")

    return total_tiles


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX dental segmentation model")
    parser.add_argument("--model", default="model_weights/dental_segmentator.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--plans", default=None,
                        help="Path to plans.json (auto-detected if not specified)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--volume-size", type=int, nargs=3, default=None,
                        help="Synthetic volume size D H W (default: typical CBCT size)")
    args = parser.parse_args()

    # Load plans if available
    plans_paths = [
        args.plans,
        "model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json",
        "model_weights/plans.json",
    ]
    patch_size = [128, 160, 112]  # default
    num_classes = 6
    target_spacing = [0.449, 0.312, 0.449]

    for p in plans_paths:
        if p and __import__("os").path.exists(p):
            with open(p) as f:
                plans = json.load(f)
            cfg_key = "3d_fullres"
            if "configurations" in plans and cfg_key in plans["configurations"]:
                cfg = plans["configurations"][cfg_key]
                patch_size = cfg.get("patch_size", patch_size)
                target_spacing = cfg.get("spacing", target_spacing)
            if "num_segmentation_heads" in plans:
                num_classes = plans["num_segmentation_heads"]
            print(f"Plans loaded from: {p}")
            print(f"  Patch size: {patch_size}, Classes: {num_classes}")
            print(f"  Target spacing: {target_spacing}")
            break

    # Typical CBCT volume size after resampling to target spacing
    # Original ~400x400x400 at 0.4mm → after resample similar size
    if args.volume_size:
        vol_size = args.volume_size
    else:
        # Typical resampled CBCT: ~350x500x350 (after crop + resample)
        vol_size = [350, 500, 350]

    print(f"\n{'='*60}")
    print(f"ONNX Dental Segmentator Benchmark")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"GPU: ", end="")
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
        print(r.stdout.strip())
    except:
        print("N/A")

    session = create_session(args.model, args.device)

    # Single patch benchmark
    avg_time = benchmark_single_patch(session, patch_size)

    # Sliding window benchmark
    print(f"\nTest volume size: {vol_size}")
    benchmark_sliding_window(session, vol_size, patch_size, num_classes)

    # Also test with smaller volume (quick scan)
    small_vol = [200, 300, 200]
    print(f"\nSmall volume size: {small_vol}")
    benchmark_sliding_window(session, small_vol, patch_size, num_classes)


if __name__ == "__main__":
    main()
