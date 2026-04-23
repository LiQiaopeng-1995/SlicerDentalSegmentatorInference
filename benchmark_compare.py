#!/usr/bin/env python
"""
Benchmark with original nnU-Net default settings:
- step_size=0.5
- TTA mirroring (8x)
- FP32
"""
import time, json, os
import numpy as np
import torch
import SimpleITK as sitk


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


def nnunet_preprocess(image_path, plans):
    t0 = time.perf_counter()
    cfg = plans["configurations"]["3d_fullres"]
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    target_spacing = np.array(cfg["spacing"])
    patch_size = cfg["patch_size"]

    img = sitk.ReadImage(image_path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    original_spacing = np.array(img.GetSpacing())[::-1]

    arr = np.transpose(arr, transpose_forward)
    transposed_spacing = original_spacing[list(transpose_forward)]

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

    fg_props = plans.get("foreground_intensity_properties_per_channel",
                         plans.get("foreground_intensity_properties_by_modality", {}))
    if "0" in fg_props:
        props = fg_props["0"]
        clip_low, clip_high = props["percentile_00_5"], props["percentile_99_5"]
        mean, std = props["mean"], props["std"]
    else:
        clip_low, clip_high, mean, std = -110.0, 3067.0, 1273.7, 558.5

    arr_cropped = np.clip(arr_cropped, clip_low, clip_high)
    arr_cropped = (arr_cropped - mean) / (std + 1e-8)

    current_shape = np.array(arr_cropped.shape)
    shape_ratio = transposed_spacing / target_spacing
    new_shape = np.round(current_shape * shape_ratio).astype(int)
    arr_tensor = torch.from_numpy(arr_cropped).float().unsqueeze(0).unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        arr_tensor, size=tuple(new_shape.tolist()), mode='trilinear', align_corners=False)
    arr_resampled = resampled.squeeze().numpy()
    shape_before_pad = arr_resampled.shape

    pad_before, pad_after = [], []
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
            [(pad_before[i], pad_after[i]) for i in range(3)], mode='constant')

    t1 = time.perf_counter()
    print(f"  Preprocess: {t1-t0:.2f}s, shape {arr_resampled.shape}")
    return arr_resampled, {"preprocess_time": t1-t0, "shape_after_crop": shape_after_crop,
                           "shape_before_pad": shape_before_pad, "pad_before": pad_before,
                           "pad_after": pad_after, "original_shape": arr.shape,
                           "crop_bbox_min": bbox_min, "crop_bbox_max": bbox_max,
                           "transpose_backward": plans.get("transpose_backward", [0,1,2])}


def sliding_window_mirror(model, volume, patch_size, num_classes,
                           step_size, use_mirror, use_fp16):
    """Sliding window with optional mirroring TTA."""
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

    # Mirror axes combinations for TTA
    if use_mirror == "lr_batch":
        mode = "lr_batch"
        mirror_axes = [()]  # handled specially below
    elif use_mirror == "lr_only":
        mode = "lr_sequential"
        mirror_axes = [(), (2,)]
    elif use_mirror:
        mode = "sequential"
        mirror_axes = [(), (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)]
    else:
        mode = "sequential"
        mirror_axes = [()]

    n_passes = total_tiles * (2 if mode == "lr_batch" else len(mirror_axes))
    print(f"  Tiles: {len(pos_d)}x{len(pos_h)}x{len(pos_w)} = {total_tiles}")
    print(f"  Mode: {mode} (TTA={'ON' if use_mirror else 'OFF'})")
    print(f"  FP16: {use_fp16}")
    print(f"  step_size: {step_size}")
    print(f"  Total forward passes: {n_passes}")

    vol_tensor = torch.from_numpy(volume).float().cuda()
    gaussian = make_gaussian(patch_size).cuda()
    aggregated = torch.zeros(num_classes, D, H, W, device='cuda')
    weight_map = torch.zeros(D, H, W, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for tile_i, d in enumerate(pos_d):
            for h in pos_h:
                for w in pos_w:
                    patch = vol_tensor[d:d+pD, h:h+pH, w:w+pW].unsqueeze(0).unsqueeze(0)

                    if mode == "lr_batch":
                        # Batch=2: original + LR-flipped, single forward pass
                        flipped = torch.flip(patch, [4])  # flip W axis (dim 4)
                        batch_inp = torch.cat([patch, flipped], dim=0)  # [2,1,D,H,W]

                        if use_fp16:
                            with torch.amp.autocast('cuda'):
                                batch_out = model(batch_inp)  # [2,C,D,H,W]
                        else:
                            batch_out = model(batch_inp)

                        batch_out = batch_out.float()
                        out_orig = batch_out[0:1]  # [1,C,D,H,W]
                        out_flip = torch.flip(batch_out[1:2], [4])  # flip back
                        tile_logits = (out_orig + out_flip) / 2.0
                        logits = tile_logits[0]
                    else:
                        # Sequential mirror
                        tile_logits = torch.zeros(1, num_classes, pD, pH, pW, device='cuda')
                        for axes in mirror_axes:
                            inp = patch.clone()
                            for ax in axes:
                                inp = torch.flip(inp, [ax + 2])

                            if use_fp16:
                                with torch.amp.autocast('cuda'):
                                    out = model(inp)
                            else:
                                out = model(inp)

                            out = out.float()
                            for ax in axes:
                                out = torch.flip(out, [ax + 2])
                            tile_logits += out

                        tile_logits /= len(mirror_axes)
                        logits = tile_logits[0]

                    aggregated[:, d:d+pD, h:h+pH, w:w+pW] += logits * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  Inference time: {t1-t0:.2f}s ({(t1-t0)/n_passes*1000:.1f} ms/pass)")
    return aggregated, weight_map, t1 - t0


def postprocess(aggregated, weight_map, meta):
    t0 = time.perf_counter()
    weight_map = torch.clamp(weight_map, min=1e-8)
    for c in range(aggregated.shape[0]):
        aggregated[c] /= weight_map

    pb, sbp = meta["pad_before"], meta["shape_before_pad"]
    if any(p > 0 for p in pb):
        aggregated = aggregated[:, pb[0]:pb[0]+sbp[0], pb[1]:pb[1]+sbp[1], pb[2]:pb[2]+sbp[2]]

    resampled = torch.nn.functional.interpolate(
        aggregated.unsqueeze(0), size=meta["shape_after_crop"], mode='trilinear', align_corners=False)
    seg = torch.argmax(resampled[0], dim=0).byte().cpu().numpy()

    full_seg = np.zeros(meta["original_shape"], dtype=np.uint8)
    bmin, bmax = meta["crop_bbox_min"], meta["crop_bbox_max"]
    full_seg[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] = seg
    full_seg = np.transpose(full_seg, meta["transpose_backward"])

    t1 = time.perf_counter()
    print(f"  Postprocess: {t1-t0:.2f}s")
    return full_seg, t1 - t0


def main():
    model_folder = "model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres"
    input_path = "test_data/PostDentalSurgery.nii.gz"

    with open(os.path.join(model_folder, "plans.json")) as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)

    print("=" * 70)
    print("Comparison: Original nnU-Net config vs Optimized config")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {input_path}")
    print(f"Patch: {patch_size}, Classes: {num_classes}")

    # Load model
    print("\nLoading model...")
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))
    model = predictor.network
    model.eval().cuda()

    # Warmup
    dummy = torch.randn(1, 1, *patch_size, device='cuda')
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    torch.cuda.synchronize()

    configs = [
        {
            "name": "Original nnU-Net (step=0.5, TTA 8x, FP16)",
            "step_size": 0.5,
            "use_mirror": True,
            "use_fp16": True,
        },
        {
            "name": "No TTA (step=0.5, FP16)",
            "step_size": 0.5,
            "use_mirror": False,
            "use_fp16": True,
        },
        {
            "name": "LR mirror only (step=0.5, FP16)",
            "step_size": 0.5,
            "use_mirror": "lr_only",
            "use_fp16": True,
        },
        {
            "name": "Optimized (step=0.75, no TTA, FP16)",
            "step_size": 0.75,
            "use_mirror": False,
            "use_fp16": True,
        },
        {
            "name": "Optimized + LR mirror (step=0.75, FP16)",
            "step_size": 0.75,
            "use_mirror": "lr_only",
            "use_fp16": True,
        },
        {
            "name": "Optimized + LR batch (step=0.75, FP16)",
            "step_size": 0.75,
            "use_mirror": "lr_batch",
            "use_fp16": True,
        },
    ]

    # Preprocess once
    print("\nPreprocessing...")
    volume, meta = nnunet_preprocess(input_path, plans)

    results = []
    for cfg_run in configs:
        print(f"\n{'='*70}")
        print(f"Config: {cfg_run['name']}")
        print(f"{'='*70}")

        total_t0 = time.perf_counter()
        aggregated, weight_map, infer_time = sliding_window_mirror(
            model, volume, patch_size, num_classes,
            cfg_run["step_size"], cfg_run["use_mirror"], cfg_run["use_fp16"])
        seg, post_time = postprocess(aggregated, weight_map, meta)
        total_t1 = time.perf_counter()

        total = total_t1 - total_t0
        results.append((cfg_run["name"], infer_time, post_time, total))
        print(f"\n  Inference: {infer_time:.2f}s | Post: {post_time:.2f}s | Total: {total:.2f}s")
        print(f"  Labels: {np.unique(seg)}")

        # Free GPU memory
        del aggregated, weight_map
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY (+ preprocess {meta['preprocess_time']:.2f}s for all)")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'Inference':>10} {'Post':>8} {'Total*':>10}")
    print("-" * 75)
    for name, infer, post, total in results:
        full = meta["preprocess_time"] + total
        print(f"{name:<45} {infer:>9.2f}s {post:>7.2f}s {full:>9.2f}s")

    print(f"\n* Total includes preprocess ({meta['preprocess_time']:.2f}s)")
    print(f"  Note: Original nnU-Net also has ~15-30s subprocess cold start (not included)")


if __name__ == "__main__":
    main()
