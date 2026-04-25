#!/usr/bin/env python3
"""
在本地 PyTorch/nnU-Net 上跑与 benchmark_compare 中 TTA 8× 同构的滑窗：step_size=0.5、8 次镜像 TTA。

默认 **FP32**（全精度）；需要更快可加 `--fp16`（与旧 benchmark 里 “TTA 8x + FP16” 一致）。

需已有 nnU-Net 3d_fullres 权重目录（含 fold_0/checkpoint_final.pth 与 plans.json）。

示例：
  python run_baseline_tta8x.py --dicom data/20260421_165639 --model-folder .../nnUNetTrainer__nnUNetPlans__3d_fullres --out data/baseline_tta8x.nii.gz
  python run_baseline_tta8x.py ... --fp16
  python run_baseline_tta8x.py ... --no-ply
默认同目录生成 `*_preview.png`、`preview_legend.txt` 与 `*_colored.ply`（需 matplotlib、scikit-image；PLY 较大较慢）；`--no-preview` / `--no-ply` 可分别关闭。
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

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark_compare import (  # noqa: E402
    nnunet_preprocess,
    postprocess,
    sliding_window_mirror,
)
from run_dicom_e2e import dicom_dir_to_nifti  # noqa: E402
from sitk_path_io import read_sitk_image_safe, write_nifti_sitk_safe  # noqa: E402


def _assert_torch_sane() -> None:
    """正常 torch 有 cuda 子模块且 __file__ 指向 site-packages。损坏安装常出现无 cuda 或 __file__ 为 None。"""
    tfile = getattr(torch, "__file__", None)
    if not hasattr(torch, "cuda"):
        print(
            "错误：import 的 `torch` 不是完整的 PyTorch（无 `torch.cuda`）。\n"
            f"  torch.__file__ = {tfile!r}（为 None 时常表示包残缺或存在无效目录名如 ~orch）\n"
            "  1) 确认当前工作目录下没有名为 torch.py 的文件。\n"
            "  2) 在 3dslice 中彻底重装：\n"
            "     pip uninstall -y torch torchvision torchaudio\n"
            "     打开 C:\\\\Users\\\\<你>\\\\miniconda3\\\\envs\\\\3dslice\\\\Lib\\\\site-packages \n"
            "     手动删除所有 torch*、~orch* 残留文件夹后：\n"
            "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
            "  3) 验证：python -c \"import torch; print(torch.__file__); print(torch.cuda.is_available())\""
        )
        raise SystemExit(1)
    if not torch.cuda.is_available():
        print(
            "错误：当前 PyTorch 未启用 GPU（`torch.cuda.is_available()` 为 False）。\n"
            "在 3dslice 中执行（与系统 nvcc 小版本可不一致）：\n"
            "  pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
            "  或: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        )
        raise SystemExit(1)


def _check_model_dir(path: str) -> None:
    plans = os.path.join(path, "plans.json")
    fold0 = os.path.join(path, "fold_0")
    if not os.path.isfile(plans):
        raise FileNotFoundError(f"缺少 plans.json: {path}")
    if not os.path.isdir(fold0):
        raise FileNotFoundError(
            f"缺少 fold_0 目录（需 checkpoint）: {path}\n"
            "请从 DentalSegmentator 权重/训练目录放置 nnU-Net 模型。"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="TTA 8× 基准分割（nnU-Net PyTorch）")
    p.add_argument("--dicom", default=None, help="DICOM 目录；与 --input 二选一")
    p.add_argument("--input", default=None, help="已转好的 NIfTI")
    p.add_argument(
        "--model-folder",
        required=True,
        help="nnUNetTrainer__nnUNetPlans__3d_fullres 目录",
    )
    p.add_argument(
        "--out",
        default=str(_ROOT / "data" / "baseline_tta8x.nii.gz"),
        help="输出标签 NIfTI",
    )
    p.add_argument(
        "--keep-nifti",
        default=None,
        help="若从 DICOM 转来，可指定中间体数据路径",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="使用 autocast FP16 推理（更快；默认全 FP32）",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="不输出 PNG 三视图预览与图例 txt（默认会输出，便于快速目视检查）",
    )
    p.add_argument(
        "--no-ply",
        action="store_true",
        help="不导出彩色 PLY 网格（默认会写出 *_colored.ply，需 scikit-image，较慢）",
    )
    args = p.parse_args()
    use_fp16 = args.fp16

    # nnU-Net 导入时会检查环境变量；本地仅加载权重推理时设空目录即可消除告警
    _stub = os.path.join(tempfile.gettempdir(), "nnunet_path_stub")
    for _key, _sub in (
        ("nnUNet_raw", "raw"),
        ("nnUNet_preprocessed", "preprocessed"),
        ("nnUNet_results", "results"),
    ):
        _d = os.path.join(_stub, _sub)
        os.makedirs(_d, exist_ok=True)
        os.environ.setdefault(_key, _d)

    _check_model_dir(args.model_folder)
    _assert_torch_sane()

    dicom = args.dicom
    if dicom is None and not args.input:
        dicom = str(_ROOT / "data" / "20260421_165639")
    if dicom and os.path.isdir(dicom):
        nii = args.keep_nifti or str(_ROOT / "data" / "_dicom_for_baseline.nii.gz")
        dicom_dir_to_nifti(dicom, nii)
        input_path = nii
    elif args.input and os.path.isfile(args.input):
        input_path = args.input
    else:
        print("需要 --dicom 目录 或 已有 --input NIfTI。默认 data/20260421_165639 不存在时也会失败。")
        raise SystemExit(2)

    plans_path = os.path.join(args.model_folder, "plans.json")
    with open(plans_path, encoding="utf-8") as f:
        plans = json.load(f)
    cfg = plans["configurations"]["3d_fullres"]
    patch_size = cfg["patch_size"]
    num_classes = plans.get("num_segmentation_heads", 6)

    print("=" * 60)
    print("Baseline: step=0.5, TTA 8x mirror, " + ("FP16" if use_fp16 else "FP32"))
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {input_path}")
    print(f"Model: {args.model_folder}")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(use_mirroring=False)
    predictor.initialize_from_trained_model_folder(args.model_folder, use_folds=(0,))
    model = predictor.network
    model.eval().cuda()

    dummy = torch.randn(1, 1, *patch_size, device="cuda")
    with torch.no_grad():
        for _ in range(2):
            if use_fp16:
                with torch.amp.autocast("cuda"):
                    model(dummy)
            else:
                model(dummy)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    volume, meta = nnunet_preprocess(input_path, plans)
    print(f"Preprocess wall: {time.perf_counter() - t0:.2f}s")

    agg, wmap, t_inf = sliding_window_mirror(
        model,
        volume,
        patch_size,
        num_classes,
        0.5,
        True,  # 8x TTA
        use_fp16=use_fp16,
    )
    seg, t_post = postprocess(agg, wmap, meta)
    del agg, wmap
    torch.cuda.empty_cache()

    ref = read_sitk_image_safe(input_path)
    out_arr = np.asarray(seg, dtype=np.uint8)
    o = sitk.GetImageFromArray(out_arr)
    o.SetSpacing(ref.GetSpacing())
    o.SetOrigin(ref.GetOrigin())
    o.SetDirection(ref.GetDirection())
    out_path = os.path.abspath(args.out)
    write_nifti_sitk_safe(o, out_path, True)

    out_dir = os.path.dirname(out_path) or "."
    if out_path.endswith(".nii.gz"):
        vis_stem = out_path[: -len(".nii.gz")]
    elif out_path.endswith(".nii"):
        vis_stem = os.path.splitext(out_path)[0]
    else:
        vis_stem = out_path

    if not args.no_preview:
        try:
            from seg_vis_export import export_legend_readme, export_montage_png  # noqa: E402
        except ImportError as e:
            print(f"  预览图跳过（import 失败）: {e}")
        else:
            try:
                preview_png = vis_stem + "_preview.png"
                export_montage_png(out_path, preview_png)
                export_legend_readme(out_dir)
            except Exception as e:
                print(f"  预览图未生成: {e}")

    if not args.no_ply:
        try:
            from export_colored_ply import export_colored_ply  # noqa: E402
        except ImportError as e:
            print(f"  PLY 跳过: {e}")
        else:
            try:
                ply_path = vis_stem + "_colored.ply"
                export_colored_ply(out_path, ply_path, decimate_ratio=0.3)
            except Exception as e:
                print(f"  PLY 未生成: {e}")

    print("=" * 60)
    print("完成")
    print(f"  输出: {out_path}")
    print(f"  标签: {np.unique(out_arr)}")
    print(f"  推理(滑窗+8镜像): {t_inf:.2f}s, 后处理: {t_post:.2f}s")
    if not args.no_preview and os.path.isfile(vis_stem + "_preview.png"):
        print(f"  预览: {vis_stem}_preview.png")
    if not args.no_ply and os.path.isfile(vis_stem + "_colored.ply"):
        print(f"  PLY:  {vis_stem}_colored.ply")


if __name__ == "__main__":
    main()
