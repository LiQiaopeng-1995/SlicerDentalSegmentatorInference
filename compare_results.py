#!/usr/bin/env python
"""
Compare C++ vs Python segmentation results:
1. Visual comparison of segmentation overlays (multiple slices)
2. Voxel agreement & per-label Dice
3. Preprocessed tensor comparison (if available)
"""
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os, sys

LABEL_NAMES = {0: "Background", 1: "Maxilla", 2: "Mandible", 3: "Upper Teeth", 4: "Lower Teeth", 5: "Canal"}
LABEL_COLORS = {0: [0,0,0,0], 1: [1,0,0,0.5], 2: [0,0,1,0.5], 3: [1,1,0,0.5], 4: [0,1,0,0.5], 5: [1,0,1,0.5]}

def load_seg(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img

def dice(a, b, label):
    m1 = (a == label)
    m2 = (b == label)
    inter = (m1 & m2).sum()
    return 2 * inter / (m1.sum() + m2.sum() + 1e-8)

def overlay_seg_on_ct(ct_slice, seg_slice, num_labels=6):
    """Create RGBA overlay of segmentation on CT."""
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    rgb = np.stack([ct_norm]*3, axis=-1)
    rgba = np.concatenate([rgb, np.ones_like(ct_norm)[..., None]], axis=-1)
    for label in range(1, num_labels):
        mask = seg_slice == label
        if mask.any():
            c = LABEL_COLORS[label]
            for ch in range(3):
                rgba[..., ch] = np.where(mask, rgba[..., ch] * (1-c[3]) + c[ch] * c[3], rgba[..., ch])
    return np.clip(rgba[..., :3], 0, 1)

def main():
    outdir = "test_data/results"
    ct_path = "test_data/real_cbct.nii.gz"

    cpp_path = os.path.join(outdir, "seg_cpp.nii.gz")
    py_path = os.path.join(outdir, "seg_our_python.nii.gz")
    official_path = os.path.join(outdir, "seg_official_nnunet.nii.gz")

    # Load CT
    ct_img = sitk.ReadImage(ct_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)
    print(f"CT shape: {ct_arr.shape}")

    segs = {}
    for name, path in [("C++", cpp_path), ("Python", py_path), ("Official", official_path)]:
        if os.path.exists(path):
            arr, _ = load_seg(path)
            segs[name] = arr
            print(f"{name}: shape={arr.shape}, labels={np.unique(arr)}")

    if len(segs) < 2:
        print("Need at least 2 segmentations to compare")
        return

    # === Metrics ===
    print("\n" + "="*60)
    print("METRICS")
    print("="*60)
    names = list(segs.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = segs[n1], segs[n2]
            if s1.shape != s2.shape:
                print(f"\n{n1} vs {n2}: SHAPE MISMATCH {s1.shape} vs {s2.shape}")
                continue
            match = (s1 == s2).sum()
            total = s1.size
            print(f"\n{n1} vs {n2}:")
            print(f"  Voxel agreement: {match}/{total} ({match/total*100:.4f}%)")
            all_labels = sorted(set(np.unique(s1)) | set(np.unique(s2)))
            for label in all_labels:
                if label == 0: continue
                d = dice(s1, s2, label)
                print(f"  {LABEL_NAMES.get(label, f'Label {label}')} Dice: {d:.6f}")

    # === Visual comparison ===
    print("\nGenerating visual comparison...")
    # Pick slices at 25%, 50%, 75% through axial
    D = ct_arr.shape[0]
    slice_indices = [D//4, D//2, 3*D//4]

    n_segs = len(segs)
    fig, axes = plt.subplots(len(slice_indices), n_segs + 1, figsize=(4*(n_segs+1), 4*len(slice_indices)))
    if len(slice_indices) == 1:
        axes = axes[np.newaxis, :]

    for row, si in enumerate(slice_indices):
        # CT only
        ct_slice = ct_arr[si]
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        axes[row, 0].imshow(ct_norm, cmap='gray')
        axes[row, 0].set_title(f"CT (z={si})")
        axes[row, 0].axis('off')

        for col, (name, seg_arr) in enumerate(segs.items(), 1):
            if seg_arr.shape == ct_arr.shape:
                overlay = overlay_seg_on_ct(ct_slice, seg_arr[si])
                axes[row, col].imshow(overlay)
            else:
                axes[row, col].text(0.5, 0.5, "Shape mismatch", ha='center', va='center')
            axes[row, col].set_title(f"{name} (z={si})")
            axes[row, col].axis('off')

    plt.tight_layout()
    vis_path = os.path.join(outdir, "comparison_visual.png")
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {vis_path}")

    # === Difference map (C++ vs Python) ===
    if "C++" in segs and "Python" in segs and segs["C++"].shape == segs["Python"].shape:
        fig2, axes2 = plt.subplots(1, len(slice_indices), figsize=(5*len(slice_indices), 5))
        if len(slice_indices) == 1:
            axes2 = [axes2]
        for col, si in enumerate(slice_indices):
            diff = (segs["C++"][si] != segs["Python"][si]).astype(np.float32)
            axes2[col].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes2[col].set_title(f"Diff C++ vs Py (z={si})\n{diff.sum():.0f} diff voxels")
            axes2[col].axis('off')
        plt.tight_layout()
        diff_path = os.path.join(outdir, "comparison_diff.png")
        fig2.savefig(diff_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {diff_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()
