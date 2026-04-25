#!/usr/bin/env python3
"""Compare C++ and Python preprocessed volumes voxel-by-voxel."""
import numpy as np
import SimpleITK as sitk

# C++ preprocessed
cpp_img = sitk.ReadImage("data/preprocessed_cpp.nii.gz")
cpp = sitk.GetArrayFromImage(cpp_img).astype(np.float32)

# Python preprocessed
py_img = sitk.ReadImage("data/preprocessed_py.nii.gz")
py = sitk.GetArrayFromImage(py_img).astype(np.float32)

print(f"C++ ITK size: {cpp_img.GetSize()}  numpy shape: {cpp.shape}")
print(f"PY  ITK size: {py_img.GetSize()}   numpy shape: {py.shape}")
print(f"C++ min={cpp.min():.4f}  max={cpp.max():.4f}  mean={cpp.mean():.4f}  std={cpp.std():.4f}")
print(f"PY  min={py.min():.4f}  max={py.max():.4f}  mean={py.mean():.4f}  std={py.std():.4f}")

if cpp.shape == py.shape:
    diff = np.abs(cpp - py)
    print(f"\n|diff| max={diff.max():.6f}  mean={diff.mean():.6f}")
    print(f"Exact matches (<1e-6): {np.sum(diff < 1e-6)} / {diff.size} ({100*np.sum(diff<1e-6)/diff.size:.2f}%)")
    print(f"Diff > 1e-3: {np.sum(diff > 1e-3)}")
    print(f"Diff > 0.1:  {np.sum(diff > 0.1)}")
    print(f"Diff > 1.0:  {np.sum(diff > 1.0)}")
    print(f"\nFirst 10 values: C++ {cpp.ravel()[:10]}")
    print(f"First 10 values: PY  {py.ravel()[:10]}")
else:
    print(f"\nSHAPE MISMATCH: C++ {cpp.shape} vs PY {py.shape}")
    for perm in [(0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        if cpp.shape == tuple(py.shape[i] for i in perm):
            print(f"C++ shape == PY shape permuted by {perm}")
