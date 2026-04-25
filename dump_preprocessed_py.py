#!/usr/bin/env python3
"""Dump preprocessed volume from Python pipeline for comparison with C++ output."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_compare import nnunet_preprocess
from sitk_path_io import read_sitk_image_safe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input NIfTI (.nii.gz)")
    p.add_argument("--model-folder", required=True,
                   help="nnUNetTrainer__nnUNetPlans__3d_fullres dir")
    p.add_argument("--out", required=True, help="Output .nii.gz for preprocessed volume")
    args = p.parse_args()

    plans_path = os.path.join(args.model_folder, "plans.json")
    with open(plans_path, encoding="utf-8") as f:
        plans = json.load(f)

    volume, meta = nnunet_preprocess(args.input, plans)

    # volume is numpy array in (D, H, W) order after transpose/crop/normalize/resample/pad
    print(f"Preprocessed shape (numpy ZYX-like): {volume.shape}")
    print(f"  dtype: {volume.dtype}")
    print(f"  min: {volume.min():.4f}, max: {volume.max():.4f}")
    print(f"  mean: {volume.mean():.4f}, std: {volume.std():.4f}")
    print(f"  meta: {json.dumps({k: str(v) if not isinstance(v, (int, float, list)) else v for k, v in meta.items()}, indent=2)}")

    # Write as NIfTI — sitk expects (Z, Y, X) which matches our (D, H, W)
    img = sitk.GetImageFromArray(volume.astype(np.float32))
    # Use dummy metadata — this is just for voxel comparison
    write_path = os.path.abspath(args.out)
    ref = read_sitk_image_safe(args.input)
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    from sitk_path_io import write_nifti_sitk_safe
    write_nifti_sitk_safe(img, write_path, True)

    # Also print first/last few voxels for quick sanity check
    flat = volume.ravel()
    print(f"\nFirst 10 voxels: {flat[:10]}")
    print(f"Last 10 voxels:  {flat[-10:]}")

    # Non-zero stats
    nonzero = volume[volume != 0]
    print(f"Non-zero voxels: {len(nonzero)} / {volume.size} ({100*len(nonzero)/volume.size:.1f}%)")

    print(f"\nWrote: {write_path}")


if __name__ == "__main__":
    main()
