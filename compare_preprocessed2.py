#!/usr/bin/env python3
"""Compare C++ linear-interp preprocessed vs Python preprocessed."""
import numpy as np
import SimpleITK as sitk

cpp = sitk.GetArrayFromImage(sitk.ReadImage("data/preprocessed_cpp_linear.nii.gz")).astype(np.float32)
py = sitk.GetArrayFromImage(sitk.ReadImage("data/preprocessed_py.nii.gz")).astype(np.float32)

print(f"C++ shape: {cpp.shape}  min={cpp.min():.6f}  max={cpp.max():.6f}  mean={cpp.mean():.6f}  std={cpp.std():.6f}")
print(f"PY  shape: {py.shape}   min={py.min():.6f}  max={py.max():.6f}  mean={py.mean():.6f}  std={py.std():.6f}")

if cpp.shape == py.shape:
    diff = np.abs(cpp - py)
    print(f"\n|diff| max={diff.max():.6f}  mean={diff.mean():.6f}")
    print(f"Exact matches (<1e-6): {np.sum(diff < 1e-6)} / {diff.size} ({100*np.sum(diff<1e-6)/diff.size:.2f}%)")
    print(f"Diff > 1e-3: {np.sum(diff > 1e-3)}")
    print(f"Diff > 0.1:  {np.sum(diff > 0.1)}")

    # Check where the big differences are
    if diff.max() > 1e-3:
        bad_idx = np.argmax(diff)
        bad_pos = np.unravel_index(bad_idx, cpp.shape)
        print(f"\nMax diff at {bad_pos}: C++={cpp[bad_pos]:.6f}  PY={py[bad_pos]:.6f}")
else:
    print(f"\nSHAPE MISMATCH!")
