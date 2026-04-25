#!/usr/bin/env python3
"""
Read a DICOM series (folder), run the same pre/post as benchmark_e2e, then
sliding-window inference in Python: ONNX Runtime (default) or PyTorch (nnU-Net).

Default uses models/dental_segmentator.onnx + models/plans.json (no .pth needed).

Example:
  python run_dicom_e2e.py --dicom data/20260421_165639 --out data/seg_out.nii.gz
  python run_dicom_e2e.py --backend pytorch --model-folder /path/to/nnUNetTrainer__...__3d_fullres
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

# Reuse pipeline from the repo
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_e2e import (  # noqa: E402
    nnunet_preprocess,
    postprocess,
    sliding_window_inference,
)

from benchmark_onnx import (  # noqa: E402
    make_gaussian_importance_map,
)
from sitk_path_io import read_sitk_image_safe, write_nifti_sitk_safe  # noqa: E402


def dicom_dir_to_nifti(dicom_dir: str, nifti_path: str) -> None:
    """Read one DICOM series from *dicom_dir* and write a compressed NIfTI."""
    dicom_dir = os.path.abspath(dicom_dir)
    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"Not a directory: {dicom_dir}")
    print(f"Reading DICOM from: {dicom_dir}")
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError("No DICOM series found. Check folder path and file set.")
    if len(series_ids) > 1:
        print(f"  Warning: {len(series_ids)} series; using first: {series_ids[0]!r}")
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    if not file_names:
        raise RuntimeError("Empty DICOM file list")
    print(f"  Slices: {len(file_names)}")
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()
    write_nifti_sitk_safe(image, nifti_path, True)
    print(f"Wrote: {nifti_path}")


def create_onnx_session(
    model_path: str, device: str, trt_cache: str | None
):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list = []
    if device == "cuda":
        if trt_cache is None:
            trt_cache = os.path.join(tempfile.gettempdir(), "dental_ort_trt_cache")
        os.makedirs(trt_cache, exist_ok=True)
        trt_opts = {
            "trt_fp16_enable": "1",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": trt_cache,
        }
        providers.append(("TensorrtExecutionProvider", trt_opts))
        providers.append(("CUDAExecutionProvider", {"device_id": "0"}))
    providers.append(("CPUExecutionProvider", {}))
    print(f"Loading ONNX: {model_path}")
    session = ort.InferenceSession(model_path, opts, providers=providers)
    print("Active providers:", session.get_providers())
    return session


def sliding_window_onnx(
    session,
    volume: np.ndarray,
    patch_size: list[int],
    num_classes: int,
    step_size: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Accumulate logits in NumPy; ORT can still use GPU EPs. Convert to torch for postprocess()."""
    D, H, W = volume.shape
    pD, pH, pW = patch_size
    vol = np.asarray(volume, dtype=np.float32)

    def get_steps(vol_s: int, patch_s: int, step_s: float) -> list[int]:
        step = max(1, int(patch_s * step_s))
        positions = list(range(0, max(vol_s - patch_s + 1, 1), step))
        if positions[-1] + patch_s < vol_s:
            positions.append(vol_s - patch_s)
        return positions

    pos_d = get_steps(D, pD, step_size)
    pos_h = get_steps(H, pH, step_size)
    pos_w = get_steps(W, pW, step_size)
    total = len(pos_d) * len(pos_h) * len(pos_w)
    print(f"  Tile grid: {len(pos_d)}x{len(pos_h)}x{len(pos_w)} = {total} tiles")

    gauss = make_gaussian_importance_map(patch_size)
    input_name = session.get_inputs()[0].name
    aggregated = np.zeros((num_classes, D, H, W), dtype=np.float32)
    weight = np.zeros((D, H, W), dtype=np.float32)

    t0 = time.perf_counter()
    for d in pos_d:
        for h in pos_h:
            for w in pos_w:
                pat = vol[d : d + pD, h : h + pH, w : w + pW]
                x = pat[np.newaxis, np.newaxis, ...]  # [1,1,D,H,W]
                out = session.run(None, {input_name: x})[0]  # [1,C,d,h,w]
                logits = out[0].astype(np.float32)
                for c in range(num_classes):
                    aggregated[c, d : d + pD, h : h + pH, w : w + pW] += logits[c] * gauss
                weight[d : d + pD, h : h + pH, w : w + pW] += gauss
    t1 = time.perf_counter()
    print(f"  ONNX inference: {t1 - t0:.2f}s ({(t1 - t0) / max(total, 1) * 1000:.1f} ms/tile)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        a_t = torch.from_numpy(aggregated).cuda()
        w_t = torch.from_numpy(weight).cuda()
    else:
        a_t = torch.from_numpy(aggregated)
        w_t = torch.from_numpy(weight)
    return a_t, w_t, t1 - t0


def load_pytorch_model(model_folder: str, fold: int = 0):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,))
    m = predictor.network
    m.eval()
    m.cuda()
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="DICOM / NIfTI E2E dental segmentation (Python)")
    p.add_argument(
        "--dicom",
        default=None,
        help="DICOM series folder. If omitted and --input is not set, uses data/20260421_165639",
    )
    p.add_argument(
        "--input",
        default=None,
        help="NIfTI input (optional if --dicom is set)",
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / "data" / "segmentation_e2e.nii.gz"),
        help="Output label map (NIfTI)",
    )
    p.add_argument(
        "--backend",
        choices=("onnx", "pytorch"),
        default="onnx",
        help="onnx: models/dental_segmentator.onnx; pytorch: nnU-Net checkpoint folder",
    )
    p.add_argument(
        "--onnx",
        default=str(_ROOT / "models" / "dental_segmentator.onnx"),
        help="ONNX file (onnx backend)",
    )
    p.add_argument(
        "--plans",
        default=None,
        help="plans.json; default: alongside PyTorch model or models/plans.json",
    )
    p.add_argument(
        "--model-folder",
        default=str(_ROOT / "model_weights" / "Dataset111_453CT" / "nnUNetTrainer__nnUNetPlans__3d_fullres"),
        help="nnU-Net 3d_fullres folder (pytorch backend)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--trt-cache",
        default=None,
        help="TensorRT engine cache directory (onnx, CUDA)",
    )
    p.add_argument(
        "--keep-nifti",
        default=None,
        help="If set, save converted DICOM to this path (e.g. data/cbct.nii.gz)",
    )
    args = p.parse_args()

    dicom_dir = args.dicom
    if dicom_dir is None and not args.input:
        dicom_dir = str(_ROOT / "data" / "20260421_165639")
    if args.dicom and args.input:
        print("Use either --dicom or --input, not both. Preferring --dicom.")
    if dicom_dir and os.path.isdir(dicom_dir):
        nii = args.keep_nifti or str(_ROOT / "data" / "_dicom_converted.nii.gz")
        os.makedirs(os.path.dirname(nii) or ".", exist_ok=True)
        dicom_dir_to_nifti(dicom_dir, nii)
        input_path = nii
    else:
        if not args.input or not os.path.exists(args.input):
            print(
                "Set --dicom to a DICOM series folder, or --input to a NIfTI file. "
                f"Default folder not found: {_ROOT / 'data' / '20260421_165639'}"
            )
            raise SystemExit(1)
        input_path = args.input

    if args.plans:
        plans_path = args.plans
    elif args.backend == "onnx":
        plans_path = str(_ROOT / "models" / "plans.json")
    else:
        plans_path = os.path.join(args.model_folder, "plans.json")
    if not os.path.exists(plans_path):
        print(f"Missing plans.json: {plans_path}")
        raise SystemExit(1)
    with open(plans_path, encoding="utf-8") as f:
        plans = json.load(f)
    patch_size = plans["configurations"]["3d_fullres"]["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)
    if args.backend == "pytorch" and not os.path.exists(os.path.join(args.model_folder, "fold_0")):
        print(
            "PyTorch backend requires nnU-Net fold folder (e.g. fold_0/checkpoint_final.pth). "
            f"Not found under: {args.model_folder}\n"
            "Use --backend onnx, or set --model-folder to your trained model."
        )
        raise SystemExit(1)
    if args.backend == "onnx" and not os.path.exists(args.onnx):
        print(f"Missing ONNX: {args.onnx}")
        raise SystemExit(1)

    print("=" * 60)
    print("Preprocessing (nnU-Net style)")
    print("=" * 60)
    vol, meta = nnunet_preprocess(input_path, plans)

    print("=" * 60)
    print("Inference")
    print("=" * 60)
    t_inf0 = time.perf_counter()
    if args.backend == "pytorch":
        model = load_pytorch_model(args.model_folder)
        dummy = torch.randn(1, 1, *patch_size, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for _ in range(1):
                model(dummy)
        torch.cuda.synchronize()
        agg, wmap, _ = sliding_window_inference(model, vol, patch_size, num_classes, step_size=0.5)
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU for ONNX (slow).")
            dev = "cpu"
        else:
            dev = args.device
        sess = create_onnx_session(args.onnx, dev, args.trt_cache)
        if dev == "cpu" and "CUDAExecutionProvider" in sess.get_providers():
            # Should not happen if we only passed CPU; leave as is
            pass
        agg, wmap, _ = sliding_window_onnx(
            sess, vol, list(patch_size), int(num_classes), step_size=0.5
        )
    t_inf1 = time.perf_counter()
    print(f"Total forward block (includes setup): {t_inf1 - t_inf0:.2f}s")

    print("=" * 60)
    print("Postprocess")
    print("=" * 60)
    # Fourth argument is unused in postprocess (meta carries shapes)
    seg, _ = postprocess(agg, wmap, meta, (0, 0, 0))

    ref = read_sitk_image_safe(input_path)
    out_arr = np.asarray(seg, dtype=np.uint8)
    if out_arr.shape != sitk.GetArrayFromImage(ref).shape:
        print(
            f"Warning: seg shape {out_arr.shape} vs ref {sitk.GetArrayFromImage(ref).shape} — check orientation"
        )
    oimg = sitk.GetImageFromArray(out_arr)
    oimg.SetSpacing(ref.GetSpacing())
    oimg.SetOrigin(ref.GetOrigin())
    oimg.SetDirection(ref.GetDirection())
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_nifti_sitk_safe(oimg, out_path, True)
    print("=" * 60)
    print("Done")
    print(f"  Output: {out_path}")
    print(f"  Unique labels: {np.unique(out_arr)}")


if __name__ == "__main__":
    main()
