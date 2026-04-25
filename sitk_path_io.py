"""
SimpleITK/ITK 在 Windows 上常无法对含非 ASCII 字符的路径直读/直写 NIfTI。
经临时文件（%TEMP% 下 ASCII 路径）中转。
"""
from __future__ import annotations

import os
import shutil
import tempfile

import SimpleITK as sitk


def _path_is_strict_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def read_sitk_image_safe(path: str) -> sitk.Image:
    """对非 ASCII 路径：先 copy 到临时 .nii.gz 再读。"""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if _path_is_strict_ascii(path):
        return sitk.ReadImage(path)
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"ds_sitr_{os.getpid()}_{abs(hash(path)) & 0xFFFFFFFF:08x}.nii.gz",
    )
    shutil.copy2(path, tmp)
    try:
        return sitk.ReadImage(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def write_nifti_sitk_safe(image: sitk.Image, dest_path: str, use_compression: bool = True) -> None:
    """先写临时 .nii.gz 再 copy 到目标。"""
    dest_path = os.path.abspath(dest_path)
    parent = os.path.dirname(dest_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(
        tempfile.gettempdir(),
        f"ds_sitk_{os.getpid()}_{abs(hash(dest_path)) & 0xFFFFFFFF:08x}.nii.gz",
    )
    try:
        sitk.WriteImage(image, tmp, use_compression)
        shutil.copy2(tmp, dest_path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
