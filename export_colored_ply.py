#!/usr/bin/env python
"""
Export segmentation to a single colored PLY mesh.
Each anatomical structure gets a distinct color.
"""
import numpy as np
import SimpleITK as sitk
from skimage import measure
import os, sys

LABELS = {
    1: ("Maxilla",          (220, 60,  60)),   # red
    2: ("Mandible",         (60,  80, 220)),   # blue
    3: ("UpperTeeth",       (240, 220, 40)),   # yellow
    4: ("LowerTeeth",       (40,  200, 60)),   # green
    5: ("MandibularCanal",  (220, 40, 220)),   # magenta
}


def write_ply_binary(filename, vertices, faces, colors):
    """Write binary PLY with vertex colors."""
    nv, nf = len(vertices), len(faces)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {nv}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {nf}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(filename, 'wb') as f:
        f.write(header.encode('ascii'))
        # vertices + colors
        for i in range(nv):
            f.write(np.array(vertices[i], dtype=np.float32).tobytes())
            f.write(np.array(colors[i], dtype=np.uint8).tobytes())
        # faces
        for face in faces:
            f.write(np.uint8(3).tobytes())
            f.write(np.array(face, dtype=np.int32).tobytes())


def export_colored_ply(seg_path, output_path, decimate_ratio=0.3):
    """Export all labels from segmentation as a single colored PLY."""
    print(f"Loading: {seg_path}")
    img = sitk.ReadImage(seg_path)
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = np.array(img.GetSpacing())[::-1]  # xyz -> zyx
    origin = np.array(img.GetOrigin())[::-1]

    all_verts = []
    all_faces = []
    all_colors = []
    vert_offset = 0

    for label, (name, color) in LABELS.items():
        mask = (arr == label).astype(np.uint8)
        if mask.sum() == 0:
            print(f"  {name}: skipped (no voxels)")
            continue

        print(f"  {name}: {mask.sum()} voxels...", end=" ", flush=True)

        try:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=tuple(spacing))
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Optional: simple decimation by subsampling faces
        if decimate_ratio < 1.0:
            n_keep = max(100, int(len(faces) * decimate_ratio))
            if n_keep < len(faces):
                idx = np.linspace(0, len(faces)-1, n_keep, dtype=int)
                faces = faces[idx]
                # Remap vertices to only used ones
                used = np.unique(faces.ravel())
                remap = np.full(len(verts), -1, dtype=int)
                remap[used] = np.arange(len(used))
                verts = verts[used]
                faces = remap[faces]

        # Assign color to all vertices
        vert_colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))

        # Offset faces
        faces_offset = faces + vert_offset
        vert_offset += len(verts)

        all_verts.append(verts)
        all_faces.append(faces_offset)
        all_colors.append(vert_colors)

        print(f"{len(verts)} verts, {len(faces)} faces")

    # Merge
    all_verts = np.vstack(all_verts).astype(np.float32)
    all_faces = np.vstack(all_faces).astype(np.int32)
    all_colors = np.vstack(all_colors).astype(np.uint8)

    print(f"\n  Total: {len(all_verts)} verts, {len(all_faces)} faces")
    write_ply_binary(output_path, all_verts, all_faces, all_colors)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {output_path} ({size_mb:.1f}MB)")


def main():
    result_dir = "test_data/results"
    stl_dir = "test_data/stl"
    os.makedirs(stl_dir, exist_ok=True)

    seg_files = {
        "Official_NoMirror": os.path.join(result_dir, "seg_official_nomirror.nii.gz"),
        "Official_LRMirror": os.path.join(result_dir, "seg_official_lrmirror.nii.gz"),
        "Ours_NoMirror": os.path.join(result_dir, "seg_ours_nomirror.nii.gz"),
        "Ours_LRMirror": os.path.join(result_dir, "seg_ours_lrmirror.nii.gz"),
    }

    for tag, path in seg_files.items():
        if os.path.exists(path):
            out = os.path.join(stl_dir, f"{tag}_colored.ply")
            print(f"\n{'='*50}")
            export_colored_ply(path, out, decimate_ratio=0.3)

    print(f"\n{'='*50}")
    print("Done! Open with MeshLab or 3D Slicer.")


if __name__ == "__main__":
    main()
