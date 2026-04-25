#!/usr/bin/env python3
"""Run Python preprocessing + ONNX CUDA inference + Python postprocess without TRT EP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import SimpleITK as sitk
import torch

from benchmark_compare import nnunet_preprocess, postprocess
from run_dicom_e2e import sliding_window_onnx
from sitk_path_io import read_sitk_image_safe, write_nifti_sitk_safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--plans", default="models/plans.json")
    parser.add_argument("--onnx", default="models/dental_segmentator.onnx")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.plans, encoding="utf-8") as f:
        plans = json.load(f)
    patch_size = list(plans["configurations"]["3d_fullres"]["patch_size"])
    num_classes = int(plans.get("num_segmentation_heads", 6))

    volume, meta = nnunet_preprocess(args.input, plans)

    providers = []
    if torch.cuda.is_available():
        providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
    providers.append(("CPUExecutionProvider", {}))
    session = ort.InferenceSession(args.onnx, providers=providers)
    print("Active providers:", session.get_providers())

    aggregated, weight, _ = sliding_window_onnx(
        session,
        volume,
        patch_size,
        num_classes,
        step_size=0.5,
    )
    seg, _ = postprocess(aggregated, weight, meta)

    ref = read_sitk_image_safe(args.input)
    out = sitk.GetImageFromArray(np.asarray(seg, dtype=np.uint8))
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    out_path = str(Path(args.out).resolve())
    write_nifti_sitk_safe(out, out_path, True)
    print(f"unique labels: {np.unique(seg)}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
