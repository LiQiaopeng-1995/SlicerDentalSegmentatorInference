#!/usr/bin/env python3
"""Compare two label NIfTI volumes with agreement and per-class Dice."""
from __future__ import annotations

import argparse

import numpy as np
import SimpleITK as sitk


def dice(a: np.ndarray, b: np.ndarray, label: int) -> float:
    ma = a == label
    mb = b == label
    na = int(ma.sum())
    nb = int(mb.sum())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(2 * int((ma & mb).sum()) / (na + nb))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--classes", type=int, default=6)
    args = parser.parse_args()

    ref = sitk.GetArrayFromImage(sitk.ReadImage(args.reference)).astype(np.int16)
    cand = sitk.GetArrayFromImage(sitk.ReadImage(args.candidate)).astype(np.int16)
    print(f"reference shape: {ref.shape}")
    print(f"candidate shape: {cand.shape}")
    if ref.shape != cand.shape:
      raise SystemExit("shape mismatch")

    match = ref == cand
    print(f"agreement: {100.0 * match.mean():.6f}% ({int(match.sum())} / {ref.size})")
    print("reference unique:", dict(zip(*np.unique(ref, return_counts=True))))
    print("candidate unique:", dict(zip(*np.unique(cand, return_counts=True))))
    for label in range(args.classes):
        print(f"dice {label}: {dice(ref, cand, label):.6f}")


if __name__ == "__main__":
    main()
