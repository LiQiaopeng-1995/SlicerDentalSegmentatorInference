#!/usr/bin/env python3
"""
One-time script to export the nnU-Net DentalSegmentator model to ONNX format.

Usage:
    python export_onnx.py --model_folder /path/to/nnUNet/model --output dental_segmentator.onnx

Requirements:
    - torch, onnx, onnxruntime-gpu, nnunetv2
    - The trained nnU-Net model checkpoint

After export, verify correctness by comparing ONNX Runtime output with PyTorch output.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_plans(model_folder: Path) -> dict:
    plans_path = model_folder / "plans.json"
    if not plans_path.exists():
        # Try parent paths common in nnU-Net folder structure
        for parent in [model_folder.parent, model_folder.parent.parent]:
            candidate = parent / "plans.json"
            if candidate.exists():
                plans_path = candidate
                break
    with open(plans_path) as f:
        return json.load(f)


def get_patch_size(plans: dict) -> list:
    """Extract patch_size from plans.json."""
    # nnU-Net v2 stores config under configurations -> 3d_fullres
    config = plans.get("configurations", {}).get("3d_fullres", {})
    patch_size = config.get("patch_size")
    if patch_size is None:
        raise ValueError("Could not find patch_size in plans.json")
    return patch_size


def export_model(model_folder: str, output_path: str, folds: tuple = (0,), verify: bool = True):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    model_folder = Path(model_folder)
    plans = load_plans(model_folder)
    patch_size = get_patch_size(plans)
    print(f"Patch size from plans.json: {patch_size}")

    # Initialize predictor and load model
    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(str(model_folder), use_folds=folds)
    model = predictor.network
    model.eval()
    model.cuda()

    # Create dummy input matching patch_size: [batch=1, channels=1, D, H, W]
    dummy = torch.randn(1, 1, *patch_size, device="cuda")

    # Export to ONNX
    print(f"Exporting to {output_path} ...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "D", 3: "H", 4: "W"},
            "output": {2: "D", 3: "H", 4: "W"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print("Export complete.")

    if verify:
        verify_export(model, dummy, output_path)


def verify_export(model: torch.nn.Module, dummy: torch.Tensor, onnx_path: str):
    """Compare PyTorch output with ONNX Runtime output."""
    import onnxruntime as ort

    print("Verifying export correctness ...")

    # PyTorch reference
    with torch.no_grad():
        pt_output = model(dummy).cpu().numpy()

    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ort_output = sess.run(None, {"input": dummy.cpu().numpy()})[0]

    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("PASSED: ONNX export matches PyTorch output.")
    else:
        print(f"WARNING: Max diff {max_diff:.6e} exceeds 1e-4 threshold. Check model carefully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export nnU-Net model to ONNX")
    parser.add_argument("--model_folder", required=True, help="Path to nnU-Net trained model folder")
    parser.add_argument("--output", default="dental_segmentator.onnx", help="Output ONNX file path")
    parser.add_argument("--folds", default="0", help="Comma-separated fold indices")
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    args = parser.parse_args()

    folds = tuple(int(f) for f in args.folds.split(","))
    export_model(args.model_folder, args.output, folds=folds, verify=not args.no_verify)
