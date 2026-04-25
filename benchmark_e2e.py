#!/usr/bin/env python
"""
End-to-end benchmark: preprocessing + sliding window inference + postprocessing.
Uses a synthetic volume that mimics real CBCT geometry.
"""
import argparse, time, json, os
import sys
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk

_e2e_root = str(Path(__file__).resolve().parent)
if _e2e_root not in sys.path:
    sys.path.insert(0, _e2e_root)
from sitk_path_io import read_sitk_image_safe


def create_synthetic_cbct(output_path, size=(400, 400, 400), spacing=(0.4, 0.4, 0.4)):
    """Create a synthetic CBCT-like volume with realistic size/spacing."""
    print(f"Creating synthetic CBCT: size={size}, spacing={spacing}")
    arr = np.random.randint(-200, 3000, size=size, dtype=np.int16)
    # Add some structure: a dense sphere (bone-like)
    cx, cy, cz = [s//2 for s in size]
    Z, Y, X = np.ogrid[:size[0], :size[1], :size[2]]
    r = min(size) // 3
    mask = (Z - cz)**2 + (Y - cy)**2 + (X - cx)**2 < r**2
    arr[mask] = np.random.randint(1000, 3000, size=mask.sum(), dtype=np.int16)

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin((0, 0, 0))
    sitk.WriteImage(img, output_path)
    print(f"Saved to {output_path}, file size: {os.path.getsize(output_path)/1024/1024:.1f} MB")
    return output_path


def nnunet_preprocess(image_path, plans):
    """Replicate nnU-Net preprocessing in Python."""
    t0 = time.perf_counter()

    cfg = plans["configurations"]["3d_fullres"]
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    target_spacing = np.array(cfg["spacing"])
    patch_size = cfg["patch_size"]

    # 1. Read
    img = read_sitk_image_safe(image_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    original_spacing = np.array(img.GetSpacing())[::-1]  # sitk xyz -> numpy zyx
    print(f"  Read: shape={arr.shape}, spacing={original_spacing}")

    # 2. Transpose
    arr = np.transpose(arr, transpose_forward)
    transposed_spacing = original_spacing[list(transpose_forward)]
    print(f"  Transpose {transpose_forward}: shape={arr.shape}")

    # 3. Crop to nonzero
    nonzero = np.argwhere(arr > -500)
    if len(nonzero) > 0:
        bbox_min = nonzero.min(axis=0)
        bbox_max = nonzero.max(axis=0) + 1
        arr_cropped = arr[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
    else:
        arr_cropped = arr
        bbox_min = np.zeros(3, dtype=int)
        bbox_max = np.array(arr.shape)
    shape_after_crop = arr_cropped.shape
    print(f"  Crop: shape={shape_after_crop}")

    # 4. CT Normalization
    fg_props = plans.get("foreground_intensity_properties_per_channel",
                         plans.get("foreground_intensity_properties_by_modality", {}))
    if "0" in fg_props:
        props = fg_props["0"]
        clip_low = props["percentile_00_5"]
        clip_high = props["percentile_99_5"]
        mean = props["mean"]
        std = props["std"]
    else:
        clip_low, clip_high, mean, std = -110.0, 3067.0, 1273.7, 558.5

    arr_cropped = np.clip(arr_cropped, clip_low, clip_high)
    arr_cropped = (arr_cropped - mean) / (std + 1e-8)
    print(f"  Normalize: clip=[{clip_low}, {clip_high}], mean={mean:.1f}, std={std:.1f}")

    # 5. Resample to target spacing
    current_shape = np.array(arr_cropped.shape)
    shape_ratio = transposed_spacing / target_spacing
    new_shape = np.round(current_shape * shape_ratio).astype(int)

    arr_tensor = torch.from_numpy(arr_cropped).float().unsqueeze(0).unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        arr_tensor, size=tuple(new_shape.tolist()), mode='trilinear', align_corners=False
    )
    arr_resampled = resampled.squeeze().numpy()
    shape_before_pad = arr_resampled.shape
    print(f"  Resample: {current_shape} -> {new_shape} (target spacing {target_spacing})")

    # 6. Pad to patch_size
    pad_before = []
    pad_after = []
    for i in range(3):
        if arr_resampled.shape[i] < patch_size[i]:
            diff = patch_size[i] - arr_resampled.shape[i]
            pad_before.append(diff // 2)
            pad_after.append(diff - diff // 2)
        else:
            pad_before.append(0)
            pad_after.append(0)

    if any(p > 0 for p in pad_before + pad_after):
        arr_resampled = np.pad(arr_resampled,
            [(pad_before[i], pad_after[i]) for i in range(3)],
            mode='constant', constant_values=0)

    t1 = time.perf_counter()
    print(f"  Pad: final shape={arr_resampled.shape}")
    print(f"  Preprocessing time: {t1-t0:.3f}s")

    return arr_resampled, {
        "original_shape": arr.shape,
        "crop_bbox_min": bbox_min,
        "crop_bbox_max": bbox_max,
        "shape_after_crop": shape_after_crop,
        "shape_before_pad": shape_before_pad,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "transpose_forward": transpose_forward,
        "transpose_backward": plans.get("transpose_backward", [0, 1, 2]),
        "preprocess_time": t1 - t0,
    }


def make_gaussian(patch_size, sigma_scale=1.0/8):
    maps = []
    for s in patch_size:
        sigma = s * sigma_scale
        coords = torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        maps.append(g)
    gauss = maps[0][:, None, None] * maps[1][None, :, None] * maps[2][None, None, :]
    gauss /= gauss.max()
    return torch.clamp(gauss, min=1e-8)


def sliding_window_inference(model, volume, patch_size, num_classes, step_size=0.75):
    """Full sliding window with Gaussian weighting."""
    D, H, W = volume.shape
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

    print(f"  Tiles: {len(pos_d)}x{len(pos_h)}x{len(pos_w)} = {total_tiles}")

    vol_tensor = torch.from_numpy(volume).float().cuda()
    gaussian = make_gaussian(patch_size).cuda()
    aggregated = torch.zeros(num_classes, D, H, W, device='cuda')
    weight_map = torch.zeros(D, H, W, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for d in pos_d:
            for h in pos_h:
                for w in pos_w:
                    patch = vol_tensor[d:d+pD, h:h+pH, w:w+pW].unsqueeze(0).unsqueeze(0)
                    logits = model(patch)[0].float()
                    aggregated[:, d:d+pD, h:h+pH, w:w+pW] += logits * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"  Inference time: {t1-t0:.3f}s ({(t1-t0)/total_tiles*1000:.1f} ms/tile)")
    return aggregated, weight_map, t1 - t0


def postprocess(aggregated, weight_map, meta, original_shape_full):
    """Postprocess: normalize, argmax, unpad, uncrop, transpose back."""
    t0 = time.perf_counter()

    # Normalize
    weight_map = torch.clamp(weight_map, min=1e-8)
    num_classes = aggregated.shape[0]
    for c in range(num_classes):
        aggregated[c] /= weight_map

    # Unpad
    pb = meta["pad_before"]
    pa = meta["pad_after"]
    sbp = meta["shape_before_pad"]
    if any(p > 0 for p in pb + pa):
        aggregated = aggregated[
            :,
            pb[0]:pb[0]+sbp[0],
            pb[1]:pb[1]+sbp[1],
            pb[2]:pb[2]+sbp[2]
        ]

    # Resample logits back to crop shape
    target_shape = meta["shape_after_crop"]
    aggregated_5d = aggregated.unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        aggregated_5d, size=target_shape, mode='trilinear', align_corners=False
    )

    # Argmax
    seg = torch.argmax(resampled[0], dim=0).byte().cpu().numpy()

    # Uncrop
    full_seg = np.zeros(meta["original_shape"], dtype=np.uint8)
    bmin = meta["crop_bbox_min"]
    bmax = meta["crop_bbox_max"]
    full_seg[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] = seg

    # Transpose backward
    full_seg = np.transpose(full_seg, meta["transpose_backward"])

    t1 = time.perf_counter()
    print(f"  Postprocess time: {t1-t0:.3f}s")
    return full_seg, t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",
        default="model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres")
    parser.add_argument("--input", default=None, help="Input NIfTI file (optional, creates synthetic if not given)")
    parser.add_argument("--size", type=int, nargs=3, default=[400, 400, 400])
    parser.add_argument("--spacing", type=float, nargs=3, default=[0.4, 0.4, 0.4])
    args = parser.parse_args()

    # Plans
    plans_path = os.path.join(args.model_folder, "plans.json")
    with open(plans_path) as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)

    print("=" * 60)
    print("End-to-End Dental Segmentator Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Patch: {patch_size}, Classes: {num_classes}")

    # Input volume
    if args.input and os.path.exists(args.input):
        input_path = args.input
        print(f"\nUsing: {input_path}")
    else:
        input_path = "/tmp/synthetic_cbct.nii.gz"
        create_synthetic_cbct(input_path, tuple(args.size), tuple(args.spacing))

    # Load model
    print("\nLoading model...")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=(0,))
    model = predictor.network
    model.eval().cuda()

    # Warmup
    print("Warmup...")
    dummy = torch.randn(1, 1, *patch_size, device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(3):
            model(dummy)
    torch.cuda.synchronize()

    # === End-to-End ===
    print("\n" + "=" * 60)
    print("Running end-to-end pipeline")
    print("=" * 60)

    total_t0 = time.perf_counter()

    # 1. Preprocess
    print("\n[1/3] Preprocessing...")
    volume, meta = nnunet_preprocess(input_path, plans)

    # 2. Inference
    print(f"\n[2/3] Sliding window inference (FP16)...")
    aggregated, weight_map, infer_time = sliding_window_inference(
        model, volume, patch_size, num_classes)

    # 3. Postprocess
    print(f"\n[3/3] Postprocessing...")
    seg, post_time = postprocess(aggregated, weight_map, meta,
                                  tuple(args.size))

    total_t1 = time.perf_counter()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Preprocessing:  {meta['preprocess_time']:.2f}s")
    print(f"  Inference:      {infer_time:.2f}s")
    print(f"  Postprocessing: {post_time:.2f}s")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:          {total_t1-total_t0:.2f}s")
    print(f"\n  Segmentation shape: {seg.shape}")
    print(f"  Unique labels: {np.unique(seg)}")


if __name__ == "__main__":
    main()
