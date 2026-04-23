#!/usr/bin/env python
"""
Compare mirror vs no-mirror segmentation on real CBCT data.
Produces 4 results:
  1. Official nnU-Net, no mirror
  2. Official nnU-Net, LR mirror
  3. Our Python, no mirror
  4. Our Python, LR mirror
Then visualize + compute Dice.
"""
import time, json, os, shutil, tempfile
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

MODEL_FOLDER = "model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres"
INPUT = "test_data/real_cbct.nii.gz"
OUTDIR = "test_data/results"
STEP = 0.5

LABEL_NAMES = {1:'Maxilla', 2:'Mandible', 3:'Upper Teeth', 4:'Lower Teeth', 5:'Canal'}
COLORS = {0:[0,0,0,0], 1:[1,0,0,0.5], 2:[0,0,1,0.5], 3:[1,1,0,0.5], 4:[0,1,0,0.5], 5:[1,0,1,0.5]}


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


def get_steps(vol_s, patch_s, step_s):
    step_len = max(1, int(patch_s * step_s))
    positions = list(range(0, max(vol_s - patch_s + 1, 1), step_len))
    if positions[-1] + patch_s < vol_s:
        positions.append(vol_s - patch_s)
    return positions


def run_our_python(model_folder, input_nifti, output_nifti, step_size=0.5, use_lr_mirror=False):
    """Our Python pipeline with optional LR mirror TTA."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    tag = "LR mirror" if use_lr_mirror else "no mirror"
    print(f"\n[Our Python, {tag}]")
    print("=" * 60)

    plans_path = os.path.join(model_folder, "plans.json")
    with open(plans_path) as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)
    target_spacing = np.array(cfg["spacing"])
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    transpose_backward = plans.get("transpose_backward", [0, 1, 2])

    t0 = time.perf_counter()
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))
    model = predictor.network
    model.eval().cuda()

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
    original_spacing = np.array(img.GetSpacing())[::-1]

    arr = np.transpose(arr, transpose_forward)
    transposed_spacing = original_spacing[list(transpose_forward)]

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

    current_shape = np.array(arr_cropped.shape)
    shape_ratio = transposed_spacing / target_spacing
    new_shape = np.round(current_shape * shape_ratio).astype(int)
    arr_t = torch.from_numpy(arr_cropped).float().unsqueeze(0).unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        arr_t, size=tuple(new_shape.tolist()), mode='trilinear', align_corners=False)
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

    t_pre1 = time.perf_counter()
    print(f"  Preprocess: {t_pre1 - t_pre0:.2f}s, shape={arr_resampled.shape}")

    # --- Sliding window ---
    t_inf0 = time.perf_counter()
    D, H, W = arr_resampled.shape
    pD, pH, pW = patch_size

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
    tile_count = 0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for d in pos_d:
            for h in pos_h:
                for w in pos_w:
                    patch = vol_tensor[d:d+pD, h:h+pH, w:w+pW].unsqueeze(0).unsqueeze(0)

                    # Forward pass
                    logits = model(patch)[0].float()

                    if use_lr_mirror:
                        # LR flip = flip along last dim (W axis = dim 4 in [C,D,H,W])
                        patch_flip = torch.flip(patch, [4])
                        logits_flip = model(patch_flip)[0].float()
                        logits_flip = torch.flip(logits_flip, [3])  # flip back
                        logits = (logits + logits_flip) / 2.0

                    aggregated[:, d:d+pD, h:h+pH, w:w+pW] += logits * gaussian
                    weight_map[d:d+pD, h:h+pH, w:w+pW] += gaussian
                    tile_count += 1
                    if tile_count % 20 == 0:
                        print(f"    Tile {tile_count}/{total_tiles}")
    torch.cuda.synchronize()

    t_inf1 = time.perf_counter()
    print(f"  Inference: {t_inf1 - t_inf0:.2f}s")

    # --- Postprocess ---
    t_post0 = time.perf_counter()
    weight_map = torch.clamp(weight_map, min=1e-8)
    for c in range(num_classes):
        aggregated[c] /= weight_map

    pb, sbp = pad_before, shape_before_pad
    if any(p > 0 for p in pb):
        aggregated = aggregated[:, pb[0]:pb[0]+sbp[0], pb[1]:pb[1]+sbp[1], pb[2]:pb[2]+sbp[2]]

    seg_lowres = torch.argmax(aggregated, dim=0).byte()
    seg_lowres_5d = seg_lowres.float().unsqueeze(0).unsqueeze(0)
    resampled_seg = torch.nn.functional.interpolate(
        seg_lowres_5d, size=shape_after_crop, mode='nearest')
    seg = resampled_seg[0, 0].byte().cpu().numpy()

    full_seg = np.zeros(original_transposed_shape, dtype=np.uint8)
    full_seg[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]] = seg
    full_seg = np.transpose(full_seg, transpose_backward)

    t_post1 = time.perf_counter()

    seg_img = sitk.GetImageFromArray(full_seg)
    seg_img.SetSpacing(img.GetSpacing())
    seg_img.SetOrigin(img.GetOrigin())
    seg_img.SetDirection(img.GetDirection())
    sitk.WriteImage(seg_img, output_nifti)

    total_time = t_post1 - t_pre0
    print(f"  Postprocess: {t_post1 - t_post0:.2f}s")
    print(f"  TOTAL: {total_time:.2f}s")
    print(f"  Labels: {np.unique(full_seg)}")
    return total_time


def run_official_nnunet(model_folder, input_nifti, output_nifti, use_mirror=False):
    """Official nnU-Net with or without mirroring."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    tag = "mirror" if use_mirror else "no mirror"
    print(f"\n[Official nnU-Net, {tag}]")
    print("=" * 60)

    t0 = time.perf_counter()
    predictor = nnUNetPredictor(
        use_mirroring=use_mirror,
        perform_everything_on_device=True,
        device=torch.device('cuda'),
    )
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,))

    if use_mirror:
        # Override to only use LR mirror (axis 2 = left-right in nnU-Net 3D)
        predictor.allowed_mirroring_axes = (2,)

    tmpdir = tempfile.mkdtemp(prefix="nnunet_input_")
    tmpout = tempfile.mkdtemp(prefix="nnunet_output_")
    try:
        input_case = os.path.join(tmpdir, "case_0000.nii.gz")
        shutil.copy2(input_nifti, input_case)

        t_pred = time.perf_counter()
        predictor.predict_from_files(
            list_of_lists_or_source_folder=tmpdir,
            output_folder_or_list_of_truncated_output_files=tmpout,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        t_done = time.perf_counter()
        print(f"  Prediction: {t_done - t_pred:.2f}s")

        result_file = os.path.join(tmpout, "case.nii.gz")
        if os.path.exists(result_file):
            shutil.copy2(result_file, output_nifti)
            seg = sitk.GetArrayFromImage(sitk.ReadImage(output_nifti))
            print(f"  Shape: {seg.shape}, labels: {np.unique(seg)}")
        else:
            print(f"  ERROR: not found. Files: {os.listdir(tmpout)}")
    finally:
        shutil.rmtree(tmpdir)
        shutil.rmtree(tmpout)

    total = time.perf_counter() - t0
    print(f"  Total: {total:.2f}s")
    return total


def dice(a, b, label):
    m1, m2 = (a == label), (b == label)
    inter = (m1 & m2).sum()
    return 2 * inter / (m1.sum() + m2.sum() + 1e-8)


def overlay(ct_s, seg_s):
    ct_n = (ct_s - ct_s.min()) / (ct_s.max() - ct_s.min() + 1e-8)
    rgb = np.stack([ct_n]*3, -1)
    for l in range(1, 6):
        m = seg_s == l
        if m.any():
            c = COLORS[l]
            for ch in range(3):
                rgb[..., ch] = np.where(m, rgb[..., ch]*(1-c[3]) + c[ch]*c[3], rgb[..., ch])
    return np.clip(rgb, 0, 1)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    configs = [
        ("Official_NoMirror", lambda: run_official_nnunet(MODEL_FOLDER, INPUT,
            os.path.join(OUTDIR, "seg_official_nomirror.nii.gz"), use_mirror=False)),
        ("Official_LRMirror", lambda: run_official_nnunet(MODEL_FOLDER, INPUT,
            os.path.join(OUTDIR, "seg_official_lrmirror.nii.gz"), use_mirror=True)),
        ("Ours_NoMirror", lambda: run_our_python(MODEL_FOLDER, INPUT,
            os.path.join(OUTDIR, "seg_ours_nomirror.nii.gz"), step_size=STEP, use_lr_mirror=False)),
        ("Ours_LRMirror", lambda: run_our_python(MODEL_FOLDER, INPUT,
            os.path.join(OUTDIR, "seg_ours_lrmirror.nii.gz"), step_size=STEP, use_lr_mirror=True)),
    ]

    times = {}
    # Check if official no mirror already exists
    off_nm_path = os.path.join(OUTDIR, "seg_official_nomirror.nii.gz")
    if not os.path.exists(off_nm_path) and os.path.exists(os.path.join(OUTDIR, "seg_official_nnunet.nii.gz")):
        shutil.copy2(os.path.join(OUTDIR, "seg_official_nnunet.nii.gz"), off_nm_path)
        print("Reusing existing official no-mirror result")
        configs[0] = ("Official_NoMirror", None)

    for name, fn in configs:
        if fn is not None:
            times[name] = fn()
        else:
            times[name] = "reused"

    # === Load all results & compare ===
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(INPUT))
    result_files = {
        "Official_NoMirror": os.path.join(OUTDIR, "seg_official_nomirror.nii.gz"),
        "Official_LRMirror": os.path.join(OUTDIR, "seg_official_lrmirror.nii.gz"),
        "Ours_NoMirror": os.path.join(OUTDIR, "seg_ours_nomirror.nii.gz"),
        "Ours_LRMirror": os.path.join(OUTDIR, "seg_ours_lrmirror.nii.gz"),
    }

    segs = {}
    for name, path in result_files.items():
        if os.path.exists(path):
            segs[name] = sitk.GetArrayFromImage(sitk.ReadImage(path))

    # Pairwise Dice
    names = list(segs.keys())
    print(f"\n{'Pair':<45} {'Agreement':>10} {'Dice L1':>8} {'Dice L2':>8} {'Dice L3':>8} {'Dice L4':>8} {'Dice L5':>8}")
    print("-" * 110)
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = segs[n1], segs[n2]
            if s1.shape != s2.shape:
                continue
            match = (s1 == s2).sum() / s1.size * 100
            dices = [dice(s1, s2, l) for l in range(1, 6)]
            print(f"{n1+' vs '+n2:<45} {match:>9.4f}% {dices[0]:>8.4f} {dices[1]:>8.4f} {dices[2]:>8.4f} {dices[3]:>8.4f} {dices[4]:>8.4f}")

    # Timing summary
    print(f"\n{'Method':<25} {'Time':>10}")
    print("-" * 40)
    for name, t in times.items():
        if isinstance(t, str):
            print(f"{name:<25} {t:>10}")
        else:
            print(f"{name:<25} {t:>9.2f}s")

    # === Visualization ===
    D = ct_arr.shape[0]
    slice_indices = [D//4, D//2, 3*D//4]
    n_cols = len(segs) + 1

    fig, axes = plt.subplots(len(slice_indices), n_cols, figsize=(4.5*n_cols, 4.5*len(slice_indices)))

    for row, si in enumerate(slice_indices):
        ct_s = ct_arr[si]
        ct_n = (ct_s - ct_s.min()) / (ct_s.max() - ct_s.min() + 1e-8)
        axes[row, 0].imshow(ct_n, cmap='gray')
        axes[row, 0].set_title(f"CT z={si}", fontsize=10)
        axes[row, 0].axis('off')

        for col, (name, seg_arr) in enumerate(segs.items(), 1):
            axes[row, col].imshow(overlay(ct_s, seg_arr[si]))
            axes[row, col].set_title(f"{name}\nz={si}", fontsize=9)
            axes[row, col].axis('off')

    legend_elements = [Patch(facecolor=COLORS[l][:3], alpha=0.5, label=LABEL_NAMES[l]) for l in range(1, 6)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    vis_path = os.path.join(OUTDIR, "mirror_comparison.png")
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {vis_path}")

    # Difference maps: mirror vs no-mirror for each pipeline
    fig2, axes2 = plt.subplots(2, len(slice_indices), figsize=(5*len(slice_indices), 10))
    for pidx, (base, mirror) in enumerate([("Official_NoMirror", "Official_LRMirror"),
                                             ("Ours_NoMirror", "Ours_LRMirror")]):
        if base in segs and mirror in segs:
            for col, si in enumerate(slice_indices):
                diff = (segs[base][si] != segs[mirror][si]).astype(float)
                axes2[pidx, col].imshow(diff, cmap='hot', vmin=0, vmax=1)
                total_diff = (segs[base] != segs[mirror]).sum()
                axes2[pidx, col].set_title(f"{base.split('_')[0]} diff z={si}\n({total_diff} total)", fontsize=9)
                axes2[pidx, col].axis('off')
    plt.tight_layout()
    diff_path = os.path.join(OUTDIR, "mirror_diff.png")
    fig2.savefig(diff_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {diff_path}")


if __name__ == "__main__":
    main()
