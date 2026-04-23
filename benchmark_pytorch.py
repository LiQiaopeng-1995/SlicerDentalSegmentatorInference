#!/usr/bin/env python
"""
Benchmark dental segmentation model inference speed using PyTorch (CUDA 13).
Tests single-patch and sliding-window inference.
"""
import argparse, time, json, os
import numpy as np
import torch
import torch.nn.functional as F


def load_model(model_folder, fold=0):
    """Load nnU-Net model from checkpoint."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,))
    model = predictor.network
    model.eval()
    model.cuda()
    return model


def load_onnx_as_pytorch(onnx_path):
    """Load ONNX model and run via onnxruntime CPU (fallback benchmark)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess


def make_gaussian(patch_size, sigma_scale=1.0/8):
    maps = []
    for s in patch_size:
        sigma = s * sigma_scale
        coords = torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        maps.append(g)
    gauss = maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]
    gauss /= gauss.max()
    gauss = torch.clamp(gauss, min=1e-8)
    return gauss


def benchmark_single_patch(model, patch_size, num_warmup=5, num_runs=20):
    D, H, W = patch_size
    dummy = torch.randn(1, 1, D, H, W, device='cuda')

    print(f"\n--- Single Patch [{D}x{H}x{W}] ---")
    print(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(num_warmup):
            model(dummy)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    print(f"  Mean: {times.mean()*1000:.1f} ms | Std: {times.std()*1000:.1f} ms | "
          f"Min: {times.min()*1000:.1f} ms | Max: {times.max()*1000:.1f} ms")
    return times.mean()


def benchmark_sliding_window(model, volume_shape, patch_size, num_classes,
                              step_size=0.75, use_fp16=True):
    D, H, W = volume_shape
    pD, pH, pW = patch_size

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

    print(f"\n--- Sliding Window: volume [{D}x{H}x{W}], patch [{pD}x{pH}x{pW}], "
          f"step={step_size} ---")
    print(f"  Tiles: D={len(pos_d)} x H={len(pos_h)} x W={len(pos_w)} = {total_tiles}")

    gaussian = make_gaussian(patch_size).cuda()
    volume = torch.randn(D, H, W, device='cuda')
    aggregated = torch.zeros(num_classes, D, H, W, device='cuda')
    weight_map = torch.zeros(D, H, W, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        for d in pos_d:
            for h in pos_h:
                for w in pos_w:
                    patch = volume[d:d+pD, h:h+pH, w:w+pW].unsqueeze(0).unsqueeze(0)
                    logits = model(patch)[0]  # [C,D,H,W]
                    logits_f32 = logits.float()
                    aggregated[:, d:d+pD, h:h+pH, w:w+pW] += logits_f32 * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    weight_map = torch.clamp(weight_map, min=1e-8)
    aggregated /= weight_map
    seg = torch.argmax(aggregated, dim=0).byte()

    t2 = time.perf_counter()
    infer_time = t1 - t0
    total_time = t2 - t0

    print(f"  Inference: {infer_time:.2f}s ({infer_time/total_tiles*1000:.1f} ms/tile)")
    print(f"  + Argmax:  {total_time:.2f}s")
    print(f"  FP16: {use_fp16}")
    return infer_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",
        default="model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres")
    parser.add_argument("--volume-size", type=int, nargs=3, default=None)
    args = parser.parse_args()

    # Load plans
    plans_path = os.path.join(args.model_folder, "plans.json")
    with open(plans_path) as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)
    spacing = cfg["spacing"]

    print("="*60)
    print("Dental Segmentator Benchmark (PyTorch, CUDA 13)")
    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Patch size: {patch_size}, Classes: {num_classes}")
    print(f"Target spacing: {[f'{s:.3f}' for s in spacing]}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.model_folder)
    print("Model loaded.")

    # Single patch benchmark
    benchmark_single_patch(model, patch_size)

    # FP32 single patch
    print("\n--- Single Patch (FP32, no autocast) ---")
    D, H, W = patch_size
    dummy = torch.randn(1, 1, D, H, W, device='cuda')
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    times = np.array(times)
    print(f"  Mean: {times.mean()*1000:.1f} ms | Min: {times.min()*1000:.1f} ms")

    # Sliding window - typical CBCT
    vol_size = args.volume_size or [350, 500, 350]
    print(f"\n{'='*60}")
    print(f"Full sliding window benchmark")
    print(f"Volume: {vol_size}")
    print(f"{'='*60}")

    benchmark_sliding_window(model, vol_size, patch_size, num_classes, use_fp16=True)
    benchmark_sliding_window(model, vol_size, patch_size, num_classes, use_fp16=False)

    # Smaller volume
    small = [200, 300, 200]
    print(f"\nSmall volume: {small}")
    benchmark_sliding_window(model, small, patch_size, num_classes, use_fp16=True)


if __name__ == "__main__":
    main()
