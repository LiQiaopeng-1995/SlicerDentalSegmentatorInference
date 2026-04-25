#!/usr/bin/env python3
"""Dump argmax label before final postprocessing from a preprocessed volume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from benchmark_compare import sliding_window_mirror
from run_dicom_e2e import create_onnx_session, sliding_window_onnx
from sitk_path_io import write_nifti_sitk_safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed", required=True, help="Preprocessed float NIfTI")
    parser.add_argument("--plans", default="models/plans.json")
    parser.add_argument("--out", required=True, help="Output label NIfTI before postprocess")
    parser.add_argument("--backend", choices=("onnx", "pytorch"), default="onnx")
    parser.add_argument("--onnx", default="models/dental_segmentator.onnx")
    parser.add_argument("--no-trt", action="store_true", help="Use CUDA/CPU ONNX Runtime without TensorRT EP")
    parser.add_argument(
        "--model-folder",
        default="model_weights/Dataset111_453CT/nnUNetTrainer__nnUNetPlans__3d_fullres",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    with open(args.plans, encoding="utf-8") as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = list(cfg["patch_size"])
    num_classes = int(plans.get("num_segmentation_heads", 6))

    volume = sitk.GetArrayFromImage(sitk.ReadImage(args.preprocessed)).astype(np.float32)
    print(f"preprocessed shape: {volume.shape}")

    if args.backend == "onnx":
        if args.tta:
            raise SystemExit("--tta is only implemented for --backend pytorch in this diagnostic")
        if args.no_trt:
            import onnxruntime as ort

            providers = []
            if torch.cuda.is_available():
                providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
            providers.append(("CPUExecutionProvider", {}))
            session = ort.InferenceSession(args.onnx, providers=providers)
            print("Active providers:", session.get_providers())
        else:
            session = create_onnx_session(args.onnx, "cuda", None)
        aggregated, weight, _ = sliding_window_onnx(
            session,
            volume,
            patch_size,
            num_classes,
            step_size=0.5,
        )
    else:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        predictor = nnUNetPredictor(use_mirroring=False)
        predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=(0,))
        model = predictor.network
        model.eval().cuda()
        aggregated, weight, _ = sliding_window_mirror(
            model,
            volume,
            patch_size,
            num_classes,
            0.5,
            bool(args.tta),
            use_fp16=bool(args.fp16),
        )

    weight = torch.clamp(weight, min=1e-8)
    for c in range(aggregated.shape[0]):
        aggregated[c] /= weight
    label = torch.argmax(aggregated, dim=0).byte().cpu().numpy()

    out = sitk.GetImageFromArray(label.astype(np.uint8))
    out_path = str(Path(args.out).resolve())
    write_nifti_sitk_safe(out, out_path, True)
    print(f"unique labels: {np.unique(label)}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
