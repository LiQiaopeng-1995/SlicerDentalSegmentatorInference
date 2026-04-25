#!/usr/bin/env python3
"""Diagnose PyTorch trilinear vs SimpleITK linear resampling on the same crop."""
from __future__ import annotations

import json

import numpy as np
import SimpleITK as sitk
import torch

from sitk_path_io import read_sitk_image_safe


def _summary(name: str, out: np.ndarray, ref: np.ndarray) -> None:
    diff = np.abs(out - ref)
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\n{name}")
    print(f"  stats: min={out.min():.6f} max={out.max():.6f} mean={out.mean():.6f} std={out.std():.6f}")
    print(
        "  diff: "
        f"max={diff.max():.6f} mean={diff.mean():.6f} "
        f"p95={np.percentile(diff, 95):.6f} p99={np.percentile(diff, 99):.6f}"
    )
    for threshold in (1e-6, 1e-3, 1e-2, 0.1, 1.0):
        n = int((diff > threshold).sum())
        print(f"  diff > {threshold:g}: {n} ({100.0 * n / diff.size:.4f}%)")
    print(f"  max at {idx}: out={out[idx]:.6f}, torch={ref[idx]:.6f}")


def _sitk_resample(
    crop: np.ndarray,
    in_spacing_zyx: np.ndarray,
    target_spacing_zyx: np.ndarray,
    new_shape_zyx: np.ndarray,
    *,
    origin_shift: bool,
    default_value: float,
) -> np.ndarray:
    image = sitk.GetImageFromArray(crop.astype(np.float32))
    image.SetSpacing(tuple(in_spacing_zyx[::-1].tolist()))

    out_spacing_xyz = tuple(target_spacing_zyx[::-1].tolist())
    out_size_xyz = [int(new_shape_zyx[2]), int(new_shape_zyx[1]), int(new_shape_zyx[0])]
    origin_xyz = list(image.GetOrigin())
    if origin_shift:
        in_spacing_xyz = image.GetSpacing()
        origin_xyz = [
            origin_xyz[i] + 0.5 * (out_spacing_xyz[i] - in_spacing_xyz[i])
            for i in range(3)
        ]

    resampled = sitk.Resample(
        image,
        out_size_xyz,
        sitk.Transform(),
        sitk.sitkLinear,
        origin_xyz,
        out_spacing_xyz,
        image.GetDirection(),
        default_value,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(resampled)


def main() -> None:
    with open("models/plans.json", encoding="utf-8") as f:
        plans = json.load(f)

    cfg = plans["configurations"]["3d_fullres"]
    transpose_forward = plans.get("transpose_forward", [0, 1, 2])
    target_spacing = np.asarray(cfg["spacing"], dtype=np.float64)

    image = read_sitk_image_safe("data/_dicom_for_baseline.nii.gz")
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)[::-1]

    arr = np.transpose(arr, transpose_forward)
    transposed_spacing = original_spacing[list(transpose_forward)]

    nonzero = np.argwhere(arr > -500)
    bbox_min = nonzero.min(axis=0)
    bbox_max = nonzero.max(axis=0) + 1
    crop = arr[
        bbox_min[0] : bbox_max[0],
        bbox_min[1] : bbox_max[1],
        bbox_min[2] : bbox_max[2],
    ].copy()

    props = plans["foreground_intensity_properties_per_channel"]["0"]
    crop = np.clip(crop, props["percentile_00_5"], props["percentile_99_5"])
    crop = (crop - props["mean"]) / (props["std"] + 1e-8)

    new_shape = np.round(np.asarray(crop.shape) * transposed_spacing / target_spacing).astype(int)
    print(f"crop shape: {crop.shape}")
    print(f"new shape:  {tuple(new_shape)}")
    print(f"spacing:    {transposed_spacing}")
    print(f"target:     {target_spacing}")

    torch_resampled = (
        torch.nn.functional.interpolate(
            torch.from_numpy(crop).float()[None, None],
            size=tuple(new_shape.tolist()),
            mode="trilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )
    print(
        f"\ntorch reference: min={torch_resampled.min():.6f} "
        f"max={torch_resampled.max():.6f} mean={torch_resampled.mean():.6f} "
        f"std={torch_resampled.std():.6f}"
    )

    cases = (
        ("sitk_default_origin_default0", False, 0.0),
        ("sitk_shift_origin_default0", True, 0.0),
        ("sitk_default_origin_border_air", False, float(crop.min())),
        ("sitk_shift_origin_border_air", True, float(crop.min())),
    )
    for name, origin_shift, default_value in cases:
        out = _sitk_resample(
            crop,
            transposed_spacing,
            target_spacing,
            new_shape,
            origin_shift=origin_shift,
            default_value=default_value,
        )
        _summary(name, out, torch_resampled)


if __name__ == "__main__":
    main()
