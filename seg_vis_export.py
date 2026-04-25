#!/usr/bin/env python
"""
从分割 NIfTI 生成分类着色预览，便于不打开 Slicer 时快速人工查看。
正交三视图（约中心层面）：轴状面 / 冠状面 / 矢状面；标签颜色与 `export_colored_ply` 一致。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
from sitk_path_io import read_sitk_image_safe  # noqa: E402

# 与 export_colored_ply.LABELS 一致；0 为背景
LABEL_COLORS = {
    0: (0, 0, 0),
    1: (220, 60, 60),
    2: (60, 80, 220),
    3: (240, 220, 40),
    4: (40, 200, 60),
    5: (220, 40, 220),
}


def _label_slice_to_rgb(slice2d: np.ndarray) -> np.ndarray:
    h, w = slice2d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lab, rgb in LABEL_COLORS.items():
        out[slice2d == lab] = rgb
    return out


def export_montage_png(seg_nifti_path: str, out_png_path: str, dpi: int = 120) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)) or ".", exist_ok=True)
    img = read_sitk_image_safe(seg_nifti_path)
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D label map, got shape {arr.shape}")
    d0, d1, d2 = arr.shape
    # 以带标签体素质心附近取中层面更直观；无前景则取几何中心
    fg = np.argwhere(arr > 0)
    if len(fg) > 0:
        c = fg.mean(axis=0).astype(int)
        iz, iy, ix = np.clip(c, 0, np.array([d0, d1, d2]) - 1)
    else:
        iz, iy, ix = d0 // 2, d1 // 2, d2 // 2

    axial = _label_slice_to_rgb(arr[iz, :, :])
    coronal = _label_slice_to_rgb(arr[:, iy, :])
    sagittal = _label_slice_to_rgb(arr[:, :, ix])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, title in (
        (axes[0], axial, f"轴状面 z={iz}"),
        (axes[1], coronal, f"冠状面 y={iy}"),
        (axes[2], sagittal, f"矢状面 x={ix}"),
    ):
        ax.imshow(np.flipud(im), origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(os.path.basename(seg_nifti_path), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  预览图: {out_png_path}")


def export_legend_readme(out_dir: str) -> None:
    """在输出目录放一小段图例说明，方便对照颜色与解剖名。"""
    p = os.path.join(out_dir, "preview_legend.txt")
    lines = [
        "标签颜色（与 NIfTI / PLY 一致）",
        "0: 背景",
        "1: Maxilla 上颌骨",
        "2: Mandible 下颌骨",
        "3: UpperTeeth 上牙",
        "4: LowerTeeth 下牙",
        "5: MandibularCanal 下颌管",
    ]
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  图例: {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("seg_nifti", help="标签 .nii.gz")
    ap.add_argument("-o", "--out", default=None, help="输出 PNG 路径，默认同目录 *_preview.png")
    args = ap.parse_args()
    p0 = args.seg_nifti
    if p0.endswith(".nii.gz"):
        base = p0[: -len(".nii.gz")]
    else:
        base = os.path.splitext(p0)[0]
    out = args.out or (base + "_preview.png")
    export_montage_png(args.seg_nifti, out)
    export_legend_readme(os.path.dirname(os.path.abspath(out)) or ".")
