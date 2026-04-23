#!/usr/bin/env python
"""
Export segmentation NIfTI results to STL files using marching cubes.
Each label is exported as a separate STL file.
"""
import numpy as np
import SimpleITK as sitk
from skimage import measure
import os, struct

LABEL_NAMES = {1: "Maxilla", 2: "Mandible", 3: "UpperTeeth", 4: "LowerTeeth", 5: "MandibularCanal"}


def write_stl_binary(filename, vertices, faces):
    """Write binary STL file."""
    with open(filename, 'wb') as f:
        f.write(b'\0' * 80)  # header
        f.write(struct.pack('<I', len(faces)))
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            n = np.cross(e1, e2)
            norm = np.linalg.norm(n)
            if norm > 0:
                n /= norm
            f.write(struct.pack('<3f', *n))
            f.write(struct.pack('<3f', *v0))
            f.write(struct.pack('<3f', *v1))
            f.write(struct.pack('<3f', *v2))
            f.write(struct.pack('<H', 0))


def export_stl(seg_path, output_dir, tag):
    """Export all labels from a segmentation NIfTI to STL files."""
    print(f"\n{'='*50}")
    print(f"Exporting: {tag}")
    print(f"  Source: {seg_path}")

    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = np.array(img.GetSpacing())[::-1]  # xyz -> zyx

    os.makedirs(output_dir, exist_ok=True)
    exported = []

    for label, name in LABEL_NAMES.items():
        mask = (arr == label).astype(np.uint8)
        voxel_count = mask.sum()
        if voxel_count == 0:
            print(f"  {name} (label {label}): no voxels, skipping")
            continue

        print(f"  {name} (label {label}): {voxel_count} voxels...", end=" ", flush=True)

        try:
            verts, faces, normals, values = measure.marching_cubes(
                mask, level=0.5, spacing=tuple(spacing))
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        out_path = os.path.join(output_dir, f"{tag}_{name}.stl")
        write_stl_binary(out_path, verts, faces)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"{len(verts)} verts, {len(faces)} faces, {size_mb:.1f}MB")
        exported.append(out_path)

    return exported


def main():
    result_dir = "test_data/results"
    stl_dir = "test_data/stl"

    seg_files = {
        "Official_NoMirror": os.path.join(result_dir, "seg_official_nomirror.nii.gz"),
        "Official_LRMirror": os.path.join(result_dir, "seg_official_lrmirror.nii.gz"),
        "Ours_NoMirror": os.path.join(result_dir, "seg_ours_nomirror.nii.gz"),
        "Ours_LRMirror": os.path.join(result_dir, "seg_ours_lrmirror.nii.gz"),
    }

    all_exported = []
    for tag, path in seg_files.items():
        if os.path.exists(path):
            exported = export_stl(path, os.path.join(stl_dir, tag), tag)
            all_exported.extend(exported)
        else:
            print(f"\n{tag}: file not found ({path})")

    print(f"\n{'='*50}")
    print(f"Total: {len(all_exported)} STL files exported")
    for f in all_exported:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
