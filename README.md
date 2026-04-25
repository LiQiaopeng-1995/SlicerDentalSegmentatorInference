# SlicerDentalSegmentatorInference

用于 [SlicerDentalSegmentator](https://github.com/gaudot/SlicerDentalSegmentator) 的 C++ ONNX Runtime 推理后端。

用 ONNX Runtime 配合 TensorRT/CUDA 执行提供方做 GPU 加速的 C++ 推理，替代原先基于 Python nnU-Net 子进程推理（约 60–120 秒）；在 RTX 4060 上约 7–9 秒。

## 构建说明

### 前置要求

- **3D Slicer 源码构建**（编译 C++ 扩展所必需）
- Visual Studio 2022（Windows）或 GCC ≥ 9（Linux）
- CMake ≥ 3.16
- NVIDIA GPU + CUDA 工具包（用于 GPU 推理）

### 步骤 1：从源码构建 3D Slicer

如果只有 Slicer 安装包，需要先从源码构建 Slicer。参见：https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/index.html

```bash
# 简要说明（Windows）：
git clone https://github.com/Slicer/Slicer.git
mkdir Slicer-build && cd Slicer-build
cmake -G "Visual Studio 17 2022" -A x64 ../Slicer
cmake --build . --config Release  # 约需 1–2 小时
```

### 步骤 2：构建本扩展

```bash
git clone https://github.com/LiQiaopeng-1995/SlicerDentalSegmentatorInference.git
mkdir SlicerDentalSegmentatorInference-build && cd SlicerDentalSegmentatorInference-build

cmake -G "Visual Studio 17 2022" -A x64 \
  -DSlicer_DIR:PATH=/path/to/Slicer-build/Slicer-build \
  ../SlicerDentalSegmentatorInference

cmake --build . --config Release
```

SuperBuild 会自动下载带 GPU 的 ONNX Runtime。

### 步骤 3：在 Slicer 中加载

1. 打开 3D Slicer（须为你从源码构建的版本）
2. 编辑 → 应用程序设置 → 模块
3. 在「附加模块路径」中加入构建输出目录，例如：
   - `SlicerDentalSegmentatorInference-build/DentalSegmentatorInference_inner-build/DentalSegmentatorInference/Release/`（Windows）
4. 重启 Slicer

### 步骤 4：准备 ONNX 模型

使用 `export_onnx.py` 将 nnU-Net 检查点导出为 ONNX：

```bash
python export_onnx.py \
  --model_folder /path/to/nnUNet/model/Dataset_dental/nnUNetTrainer__nnUNetPlans__3d_fullres \
  --output /path/to/DentalSegmentator/Resources/ML/dental_segmentator.onnx
```

将 `.onnx` 放在模型目录中，与 `plans.json` 放在同一处。

## 架构

```
Slicer (GUI)
  → DentalSegmentator（Python UI 扩展，已有）
    → DentalSegmentatorInference（本 C++ CLI 模块）
      → ITK：读入体数据
      → 预处理：转置 → 裁剪 → 归一化 → 重采样 → 填充
      → ONNX Runtime（TensorRT FP16）：滑窗推理
      → 后处理：对 logits 重采样 → argmax → 去填充 → 转置
      → ITK：写出分割标签图
```

## 性能（RTX 4060）

| 阶段 | 时间 |
|---|---|
| 预处理（ITK） | 约 1–2s |
| 滑窗推理（TRT FP16） | 约 4–5s |
| 后处理 | 约 1–2s |
| **合计** | **约 7–9s** |
