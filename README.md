# SlicerDentalSegmentatorInference

C++ ONNX Runtime inference backend for [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator).

Replaces the Python nnU-Net subprocess inference (~60-120s) with GPU-accelerated C++ inference (~7-9s on RTX 4060) using ONNX Runtime with TensorRT/CUDA execution providers.

## Build Instructions

### Prerequisites

- **3D Slicer source build** (required to compile C++ extensions)
- Visual Studio 2022 (Windows) or GCC ≥ 9 (Linux)
- CMake ≥ 3.16
- NVIDIA GPU + CUDA Toolkit (for GPU inference)

### Step 1: Build 3D Slicer from source

If you only have the Slicer installer, you need to build Slicer from source first. See: https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html

```bash
# Brief summary (Windows):
git clone https://github.com/Slicer/Slicer.git
mkdir Slicer-build && cd Slicer-build
cmake -G "Visual Studio 17 2022" -A x64 ../Slicer
cmake --build . --config Release  # This takes 1-2 hours
```

### Step 2: Build this extension

```bash
git clone https://github.com/LiQiaopeng-1995/SlicerDentalSegmentatorInference.git
mkdir SlicerDentalSegmentatorInference-build && cd SlicerDentalSegmentatorInference-build

cmake -G "Visual Studio 17 2022" -A x64 \
  -DSlicer_DIR:PATH=/path/to/Slicer-build/Slicer-build \
  ../SlicerDentalSegmentatorInference

cmake --build . --config Release
```

The SuperBuild will automatically download ONNX Runtime GPU.

### Step 3: Load in Slicer

1. Open 3D Slicer (the one you built from source)
2. Edit → Application Settings → Modules
3. Add the build output directory to "Additional module paths":
   - `SlicerDentalSegmentatorInference-build/DentalSegmentatorInference_inner-build/DentalSegmentatorInference/Release/` (Windows)
4. Restart Slicer

### Step 4: Prepare the ONNX model

Use `export_onnx.py` to convert the nnU-Net checkpoint:

```bash
python export_onnx.py \
  --model_folder /path/to/nnUNet/model/Dataset_dental/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --output /path/to/DentalSegmentator/Resources/ML/dental_segmentator.onnx
```

Place the `.onnx` file in the model directory alongside `plans.json`.

## Architecture

```
Slicer (GUI)
  → DentalSegmentator (Python UI extension, existing)
    → DentalSegmentatorInference (this C++ CLI module)
      → ITK: read volume
      → Preprocess: transpose → crop → normalize → resample → pad
      → ONNX Runtime (TensorRT FP16): sliding window inference
      → Postprocess: resample logits → argmax → uncrop → transpose
      → ITK: write segmentation labelmap
```

## Performance (RTX 4060)

| Stage | Time |
|---|---|
| Preprocessing (ITK) | ~1-2s |
| Sliding window inference (TRT FP16) | ~4-5s |
| Postprocessing | ~1-2s |
| **Total** | **~7-9s** |
