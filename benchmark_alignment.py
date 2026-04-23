#!/usr/bin/env python
"""
Compare inference results between:
1. Original nnU-Net (official nnUNetPredictor API)
2. Our optimized Python pipeline
3. (C++ results loaded from file for comparison)

All use step_size=0.5, no TTA, FP16.
Saves segmentation results as NIfTI for cross-comparison.
"""
import argparse, time, json, os, shutil, tempfile
import numpy as np
import torch
import SimpleITK as sitk


# ============================================================
# Method 1: Official nnU-Net Predictor
# ============================================================
def run_official_nnunet(model_folder, input_nifti, output_nifti):
    """Run official nnUNetPredictor with step_size=0.5, no mirror, fold 0."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print("\n[Method 1] Official nnU-Net Predictor")
    print("=" * 60)

    t0 = time.perf_counter()

    predictor = nnUNetPredictor(
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
    )
    predictor.initialize_from_trained_model_folder(
        model_folder, use_folds=(0,)
    )
    # Override step_size
    predictor.configuration_manager.configuration['spacing'] = \
        predictor.configuration_manager.configuration.get('spacing',
            predictor.plans_manager.plans['configurations']['3d_fullres']['spacing'])

    t_init = time.perf_counter()
    print(f"  Init time: {t_init - t0:.2f}s")

    # Prepare input: nnU-Net expects a specific folder structure
    tmpdir = tempfile.mkdtemp(prefix="nnunet_input_")
    tmpout = tempfile.mkdtemp(prefix="nnunet_output_")
    try:
        # nnU-Net expects: <case_id>_0000.nii.gz
        input_case = os.path.join(tmpdir, "case_0000.nii.gz")
        shutil.copy2(input_nifti, input_case)

        t_pred_start = time.perf_counter()
        predictor.predict_from_files(
            list_of_lists_or_source_folder=tmpdir,
            output_folder_or_list_of_truncated_output_files=tmpout,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        t_pred_end = time.perf_counter()
        print(f"  Prediction time: {t_pred_end - t_pred_start:.2f}s")

        # Copy result
        result_file = os.path.join(tmpout, "case.nii.gz")
        if os.path.exists(result_file):
            shutil.copy2(result_file, output_nifti)
            seg = sitk.GetArrayFromImage(sitk.ReadImage(output_nifti))
            print(f"  Output shape: {seg.shape}, labels: {np.unique(seg)}")
        else:
            print(f"  ERROR: result not found at {result_file}")
            print(f"  Files in output: {os.listdir(tmpout)}")
    finally:
        shutil.rmtree(tmpdir)
        shutil.rmtree(tmpout)

    t_total = time.perf_counter()
    print(f"  Total time: {t_total - t0:.2f}s")
    return t_total - t0


# ============================================================
# Method 2: Our optimized Python pipeline
# ============================================================
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


def run_our_python(model_folder, input_nifti, output_nifti, step_size=0.5):
    """Our Python pipeline: preprocess + sliding window + postprocess."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    print(f"\n[Method 2] Our Python Pipeline (step={step_size})")
    print("=" * 60)

    # Load plans
    plans_path = os.path.join(model_folder, "plans.json")
    with open(plans_path) as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)
    target_spacing = np.array(cfg["spacing"])
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    transpose_backward = plans.get("transpose_backward", [0, 1, 2])

    # Load model
    t0 = time.perf_counter()
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))
    model = predictor.network
    model.eval().cuda()

    # Warmup
    dummy = torch.randn(1, 1, *patch_size, device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(2):
            model(dummy)
    torch.cuda.synchronize()
    t_init = time.perf_counter()
    print(f"  Init+warmup: {t_init - t0:.2f}s")

    # --- Preprocess ---
    t_pre0 = time.perf_counter()

    img = sitk.ReadImage(input_nifti)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    original_spacing = np.array(img.GetSpacing())[::-1]  # xyz -> zyx
    original_direction = img.GetDirection()
    original_origin = img.GetOrigin()
    original_size = img.GetSize()
    print(f"  Read: shape={arr.shape}, spacing={original_spacing}")

    # Transpose
    arr = np.transpose(arr, transpose_forward)
    transposed_spacing = original_spacing[list(transpose_forward)]
    print(f"  Transpose {transpose_forward}: shape={arr.shape}")

    # Crop to nonzero
    nonzero = np.argwhere(arr != 0)
    if len(nonzero) > 0:
        bbox_min = nonzero.min(axis=0)
        bbox_max = nonzero.max(axis=0) + 1
    else:
        bbox_min = np.zeros(3, dtype=int)
        bbox_max = np.array(arr.shape)
    arr_cropped = arr[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]
    shape_after_crop = arr_cropped.shape
    original_transposed_shape = arr.shape
    print(f"  Crop: {arr.shape} -> {shape_after_crop}")

    # CT Normalization
    fg_props = plans.get("foreground_intensity_properties_per_channel",
                         plans.get("foreground_intensity_properties_by_modality", {}))
    if "0" in fg_props:
        props = fg_props["0"]
        clip_low, clip_high = props["percentile_00_5"], props["percentile_99_5"]
        mean_val, std_val = props["mean"], props["std"]
    else:
        clip_low, clip_high, mean_val, std_val = -110.0, 3067.0, 1273.7, 558.5

    arr_cropped = np.clip(arr_cropped, clip_low, clip_high)
    arr_cropped = (arr_cropped - mean_val) / (std_val + 1e-8)

    # Resample
    current_shape = np.array(arr_cropped.shape)
    shape_ratio = transposed_spacing / target_spacing
    new_shape = np.round(current_shape * shape_ratio).astype(int)
    arr_t = torch.from_numpy(arr_cropped).float().unsqueeze(0).unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        arr_t, size=tuple(new_shape.tolist()), mode='trilinear', align_corners=False)
    arr_resampled = resampled.squeeze().numpy()
    shape_before_pad = arr_resampled.shape
    print(f"  Resample: {current_shape} -> {new_shape}")

    # Pad
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
    print(f"  Pad: final shape={arr_resampled.shape}")

    t_pre1 = time.perf_counter()
    print(f"  Preprocess time: {t_pre1 - t_pre0:.2f}s")

    # --- Sliding window inference ---
    t_inf0 = time.perf_counter()
    D, H, W = arr_resampled.shape
    pD, pH, pW = patch_size

    def get_steps(vol_s, patch_s, step_s):
        step_len = max(1, int(patch_s * step_s))
        positions = list(range(0, max(vol_s - patch_s + 1, 1), step_len))
        if positions[-1] + patch_s < vol_s:
            positions.append(vol_s - patch_s)
        return positions

    pos_d = get_steps(D, pD, step_size)
    pos_h = get_steps(H, pH, step_size)
    pos_w = get_steps(W, pW, step_size)
    total_tiles = len(pos_d) * len(pos_h) * len(pos_w)
    print(f"  Tiles: {len(pos_d)}x{len(pos_h)}x{len(pos_w)} = {total_tiles}")

    vol_tensor = torch.from_numpy(arr_resampled).float().cuda()
    gaussian = make_gaussian(patch_size).cuda()
    aggregated = torch.zeros(num_classes, D, H, W, device='cuda')
    weight_map = torch.zeros(D, H, W, device='cuda')

    torch.cuda.synchronize()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for d in pos_d:
            for h in pos_h:
                for w in pos_w:
                    patch = vol_tensor[d:d+pD, h:h+pH, w:w+pW].unsqueeze(0).unsqueeze(0)
                    logits = model(patch)[0].float()
                    aggregated[:, d:d+pD, h:h+pH, w:w+pW] += logits * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian
    torch.cuda.synchronize()

    t_inf1 = time.perf_counter()
    print(f"  Inference time: {t_inf1 - t_inf0:.2f}s")

    # --- Postprocess ---
    t_post0 = time.perf_counter()

    weight_map = torch.clamp(weight_map, min=1e-8)
    for c in range(num_classes):
        aggregated[c] /= weight_map

    # Unpad
    pb, sbp = pad_before, shape_before_pad
    if any(p > 0 for p in pb):
        aggregated = aggregated[:, pb[0]:pb[0]+sbp[0], pb[1]:pb[1]+sbp[1], pb[2]:pb[2]+sbp[2]]

    # Argmax at low resolution first (much cheaper than resampling all logits)
    seg_lowres = torch.argmax(aggregated, dim=0).byte()  # [D,H,W]

    # Resample label map back to crop shape using nearest neighbor
    seg_lowres_5d = seg_lowres.float().unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    resampled_seg = torch.nn.functional.interpolate(
        seg_lowres_5d, size=shape_after_crop, mode='nearest')
    seg = resampled_seg[0, 0].byte().cpu().numpy()

    # Uncrop
    full_seg = np.zeros(original_transposed_shape, dtype=np.uint8)
    full_seg[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]] = seg

    # Transpose backward
    full_seg = np.transpose(full_seg, transpose_backward)

    t_post1 = time.perf_counter()
    print(f"  Postprocess time: {t_post1 - t_post0:.2f}s")

    # Save
    seg_img = sitk.GetImageFromArray(full_seg)
    seg_img.SetSpacing(img.GetSpacing())
    seg_img.SetOrigin(img.GetOrigin())
    seg_img.SetDirection(img.GetDirection())
    sitk.WriteImage(seg_img, output_nifti)
    print(f"  Output shape: {full_seg.shape}, labels: {np.unique(full_seg)}")
    print(f"  Saved: {output_nifti}")

    total_time = t_post1 - t_pre0
    print(f"\n  Preprocess:  {t_pre1-t_pre0:.2f}s")
    print(f"  Inference:   {t_inf1-t_inf0:.2f}s")
    print(f"  Postprocess: {t_post1-t_post0:.2f}s")
    print(f"  TOTAL:       {total_time:.2f}s")
    return total_time


# ============================================================
# Compare results
# ============================================================
def compare_segmentations(files, names):
    """Compare segmentation results pairwise."""
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")

    segs = {}
    for name, path in zip(names, files):
        if os.path.exists(path):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            segs[name] = arr
            print(f"\n  {name}: shape={arr.shape}, labels={np.unique(arr)}")
            for label in sorted(np.unique(arr)):
                count = (arr == label).sum()
                print(f"    Label {label}: {count} voxels ({count/arr.size*100:.2f}%)")
        else:
            print(f"\n  {name}: FILE NOT FOUND ({path})")

    # Pairwise comparison
    seg_names = list(segs.keys())
    for i in range(len(seg_names)):
        for j in range(i+1, len(seg_names)):
            n1, n2 = seg_names[i], seg_names[j]
            s1, s2 = segs[n1], segs[n2]
            if s1.shape != s2.shape:
                print(f"\n  {n1} vs {n2}: SHAPE MISMATCH {s1.shape} vs {s2.shape}")
                continue

            match = (s1 == s2).sum()
            total = s1.size
            print(f"\n  {n1} vs {n2}:")
            print(f"    Voxel agreement: {match}/{total} ({match/total*100:.4f}%)")
            print(f"    Disagreements:   {total-match} ({(total-match)/total*100:.4f}%)")

            # Per-label Dice
            all_labels = sorted(set(np.unique(s1)) | set(np.unique(s2)))
            for label in all_labels:
                if label == 0:
                    continue
                m1 = (s1 == label)
                m2 = (s2 == label)
                intersection = (m1 & m2).sum()
                dice = 2 * intersection / (m1.sum() + m2.sum() + 1e-8)
                print(f"    Label {label} Dice: {dice:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder",
        default="model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres")
    parser.add_argument("--input", default="test_data/real_cbct.nii.gz")
    parser.add_argument("--cpp_result", default=None,
        help="Path to C++ inference result NIfTI for comparison")
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--skip_official", action="store_true")
    args = parser.parse_args()

    outdir = "test_data/results"
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Inference Alignment Test")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model_folder}")
    print(f"Step size: {args.step}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    official_out = os.path.join(outdir, "seg_official_nnunet.nii.gz")
    our_out = os.path.join(outdir, "seg_our_python.nii.gz")

    # Method 1: Official nnU-Net
    if not args.skip_official:
        t1 = run_official_nnunet(args.model_folder, args.input, official_out)
    else:
        print("\n[Method 1] Skipped (--skip_official)")
        t1 = None

    # Method 2: Our Python
    t2 = run_our_python(args.model_folder, args.input, our_out, step_size=args.step)

    # Summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    if t1:
        print(f"  Official nnU-Net:  {t1:.2f}s")
    print(f"  Our Python:        {t2:.2f}s")

    # Compare
    files = []
    names = []
    if os.path.exists(official_out):
        files.append(official_out)
        names.append("Official")
    if os.path.exists(our_out):
        files.append(our_out)
        names.append("Ours")
    if args.cpp_result and os.path.exists(args.cpp_result):
        files.append(args.cpp_result)
        names.append("C++")

    if len(files) >= 2:
        compare_segmentations(files, names)


if __name__ == "__main__":
    main()
