#!/usr/bin/env python3
"""
将 **PyTorch baseline**（`run_baseline_tta8x.py` 产出的 TTA 8× 标签体）与 **ONNX Runtime**
（`run_dicom_e2e.py` 的 `--backend onnx`，优先 TensorRT FP16 + CUDA）产出的标签体做对比。

重要说明
--------
- **baseline**：TTA 8× 镜像、默认 FP32（或 `--fp16`）。
- **ONNX**：当前与 C++ 扩展一致，**无 TTA**、仅滑窗 + 高斯融合；**与 baseline 不会逐体素一致**是预期的，
  差异来自「TTA/非 TTA」+ 数值精度。本脚本量化为 Dice / 体素一致率，便于观察 TensorRT 路径与「全精度 TTA 参考」差多少。

用法示例
--------
1. 对**同一**输入 NIfTI（与 baseline 推理用的一致，例如 DICOM 转出的 `_dicom_for_baseline.nii.gz`）跑 ONNX：

   python run_dicom_e2e.py --input data/_dicom_for_baseline.nii.gz --out data/seg_onnx_trt.nii.gz

2. 对比（需两文件形状一致）：

   python compare_baseline_to_onnx.py --baseline data/baseline_tta8x.nii.gz --onnx data/seg_onnx_trt.nii.gz

3. 可选输出表格到文本：

   python compare_baseline_to_onnx.py -b data/baseline_tta8x.nii.gz -o data/seg_onnx_trt.nii.gz --out-report data/trt_vs_baseline.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
from sitk_path_io import read_sitk_image_safe  # noqa: E402

LABEL_NAMES = {
    0: "Background",
    1: "Maxilla",
    2: "Mandible",
    3: "Upper Teeth",
    4: "Lower Teeth",
    5: "Mandibular Canal",
}


def _load_labels(path: str) -> np.ndarray:
    img = read_sitk_image_safe(path)
    return sitk.GetArrayFromImage(img).astype(np.int32)


def dice(a: np.ndarray, b: np.ndarray, label: int) -> float:
    m1 = a == label
    m2 = b == label
    s1, s2 = m1.sum(), m2.sum()
    if s1 == 0 and s2 == 0:
        return 1.0
    if s1 == 0 or s2 == 0:
        return 0.0
    inter = (m1 & m2).sum()
    return float(2.0 * inter / (s1 + s2 + 1e-8))


def main() -> None:
    p = argparse.ArgumentParser(
        description="比较 baseline TTA8x 与 ONNX(TRT) 分割标签体",
    )
    p.add_argument(
        "-b",
        "--baseline",
        required=True,
        help="run_baseline_tta8x 输出的 NIfTI（参考）",
    )
    p.add_argument(
        "-o",
        "--onnx",
        required=True,
        dest="onnx_path",
        help="run_dicom_e2e --backend onnx 输出的 NIfTI",
    )
    p.add_argument(
        "--out-report",
        default=None,
        help="将打印内容保存为 UTF-8 文本",
    )
    args = p.parse_args()

    if not os.path.isfile(args.baseline):
        print(f"找不到 baseline: {args.baseline}", file=sys.stderr)
        raise SystemExit(1)
    if not os.path.isfile(args.onnx_path):
        print(f"找不到 onnx 结果: {args.onnx_path}", file=sys.stderr)
        raise SystemExit(1)

    a = _load_labels(args.baseline)
    b = _load_labels(args.onnx_path)
    if a.shape != b.shape:
        print(
            f"形状不一致: baseline {a.shape} vs onnx {b.shape}，"
            "请确认二者来自同一体数据与同预处理链。",
            file=sys.stderr,
        )
        raise SystemExit(1)

    match = (a == b).sum()
    total = a.size
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Baseline (PyTorch TTA 8×) vs ONNX Runtime（通常 TensorRT FP16 + 无 TTA）")
    lines.append("=" * 60)
    lines.append(f"  Reference:  {os.path.abspath(args.baseline)}")
    lines.append(f"  Candidate:  {os.path.abspath(args.onnx_path)}")
    lines.append(f"  Shape:      {tuple(a.shape)} voxels = {total}")
    lines.append(f"  体素一致率: {100.0 * match / total:.4f}%  ({match} / {total})")
    lines.append(f"  体素差异数:  {int(total - match)}  ({100.0 * (total - match) / total:.4f}%)")
    lines.append("")

    lines.append("  Per-class Dice (reference vs candidate):")
    dices: dict[str, float] = {}
    for lab in range(0, 6):
        d = dice(a, b, lab)
        dices[str(lab)] = d
        name = LABEL_NAMES.get(lab, str(lab))
        lines.append(f"    {lab} {name}: {d:.6f}")
    dforeground = [dice(a, b, i) for i in range(1, 6)]
    lines.append(
        f"  Foreground mean Dice (1–5): {float(np.mean(dforeground)):.6f}"
    )

    payload = {
        "baseline_path": os.path.abspath(args.baseline),
        "onnx_path": os.path.abspath(args.onnx_path),
        "shape": list(a.shape),
        "voxel_agreement": match / float(total),
        "voxel_disagreement": 1.0 - match / float(total),
        "dice_per_class": dices,
        "dice_mean_foreground_1_5": float(np.mean(dforeground)),
    }
    lines.append("")
    lines.append("  JSON (machine-readable):")
    js = json.dumps(payload, ensure_ascii=False, indent=2)
    lines.extend("  " + ln for ln in js.splitlines())

    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out_report:
        rp = os.path.abspath(args.out_report)
        d = os.path.dirname(rp)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写入: {rp}")


if __name__ == "__main__":
    main()
