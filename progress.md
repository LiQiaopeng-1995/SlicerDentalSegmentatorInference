# 3DSlice / SlicerDentalSegmentatorInference 进展记录

## 2026-04-22

### 1. 文档

- 将 `SlicerDentalSegmentatorInference/README.md` 改为仅保留中文说明（构建、加载扩展、ONNX 导出、架构与性能表）。

### 2. 编译环境

- 本机已具备：**CMake 3.21.x**、**CUDA 11.1**（`nvcc` 可用，适用于 RTX 2070 Super 等 Turing GPU 的初步 GPU 测试）。
- **Visual Studio 2022 生成工具**已安装后，在 `SlicerDentalSegmentatorInference-build` 中 `cmake -G "Visual Studio 17 2022" -A x64` 可正常识别 **MSVC 19.44**，C++ 工具链可用。

### 3. 扩展工程 CMake 情况

- 在**未提供** `Slicer_DIR`（即尚未找到 `SlicerConfig.cmake`）时，`find_package(Slicer)` 会失败。  
- **说明**：从官网安装的 Slicer 安装包**不能**替代开发用的 `Slicer_DIR`；必须在本机**从源码配置/构建** 3D Slicer 后，使用其**构建树**中的目录（一般形如外层 `Slicer-build` 下内层 `Slicer-build`，以实际生成的 `SlicerConfig.cmake` 所在目录为准）。

### 4. 辅助脚本

- 新增 `SlicerDentalSegmentatorInference/scripts/build-windows.ps1`：在设置好 `Slicer_DIR` 后，于 `SlicerDentalSegmentatorInference-build` 中执行 **configure + Release 构建**（含 SuperBuild 拉取 ONNX Runtime），并提示 CLI 可执行文件常见输出路径。  
- **用法示例**（将路径换成你的 `SlicerConfig.cmake` 所在目录）：

  ```powershell
  cd E:\副业\3DSlice\SlicerDentalSegmentatorInference
  .\scripts\build-windows.ps1 -SlicerDir "E:\副业\3DSlice\Slicer-build\Slicer-build"
  ```

### 5. 在当前工作区准备 Slicer 源码

- 计划在 **`E:\副业\3DSlice`** 下放置 Slicer 源码目录取名 **`Slicer`**，构建目录 **`Slicer-build`**（与官方习惯一致，便于与扩展同盘管理）。
- 曾执行/尝试过 **`git clone`**（如 `git clone --depth 1` 等到 `E:\副业\3DSlice\Slicer`）：
  - 克隆为**大仓库**，耗时可较长；
  - 若出现 **`.git/shallow.lock` 被占用**、**无法删除 `Slicer` 目录** 等情况，说明仍有 **git 相关进程在运行**；应先等待克隆结束，或在确认无需要保留的操作后，结束相关 `git.exe` 进程再删目录重试。
- **当前状态**（以仓库内能检测到的内容为准）：`E:\副业\3DSlice\Slicer` 下**尚未**出现项目根 `CMakeLists.txt`（**克隆未完成**或**目录已被清空/损坏**时会出现）。下一步需**确认 clone 已完整**再进入 CMake 配置 Slicer 的步骤。

### 6. 构建 Slicer（待完成，耗时长）

1. 确认 `E:\副业\3DSlice\Slicer\CMakeLists.txt` 存在。  
2. 创建并进入构建目录，例如：

   ```powershell
   mkdir E:\副业\3DSlice\Slicer-build
   cd E:\副业\3DSlice\Slicer-build
   cmake -G "Visual Studio 17 2022" -A x64 ..\Slicer
   cmake --build . --config Release
   ```

3. 首次完整编译 3D Slicer 通常需 **1～数小时**且占用大量磁盘，请保证空间充足。  
4. 在构建树中定位 **包含 `SlicerConfig.cmake` 的目录**，将其作为 **`-SlicerDir`** 传入 `build-windows.ps1`。

### 7. 扩展与 GPU 测试（Slicer 构建成功后）

- 用 `build-windows.ps1` 构建 `DentalSegmentatorInference`。
- 准备带 **`plans.json` + `*.onnx`** 的模型目录（参见 README 中 `export_onnx.py`）。
- 命令行试跑（路径按实机调整）：

  ```text
  DentalSegmentatorInference.exe --inputVolume 输入体数据 --outputVolume 输出标签 --modelDir 模型目录 --device cuda
  ```

### 8. 进展文件自动化追加约定

- 新增项目规则：`.cursor/rules/progress-md-append.mdc`（`alwaysApply: true`），约定在本工作区**完成实质性开发/构建类操作**后，由 Agent **在当次任务结束前**向本文件**末尾追加**一条中文摘要（日期、做了什么、结果、下一步），避免与闲聊或无关小改混记。
- 之后与此仓库相关的操作若满足该规则中的「实质性」说明，会按此自动续写；历史条目保留不删。

---

**待办线索**（与上文第 5～7 节一致，随构建推进由后续追加记录更新）：Slicer 克隆与 `CMakeLists`、Slicer `cmake`/全量 build、`SlicerConfig.cmake` 实际路径、扩展与 2070S 推理试跑。

## 2026-04-22（用户同步：Slicer clone 状态）

- **用户反馈**：当前 **Slicer 仓库 clone 已完成**。
- **本机自检说明**（Agent 侧）：在 `E:\副业\3DSlice\Slicer` 上未能稳定读到根目录 `CMakeLists.txt` 与完整工作区文件列表（可能为路径/沙箱与实机差异，或仓库尚处「仅有 `.git`、未检出工作树」的中间态）。请你本地在资源管理器或终端确认：
  - 存在 `E:\副业\3DSlice\Slicer\CMakeLists.txt`；
  - 若目录几乎为空，在 `Slicer` 目录下执行 `git status`，必要时 `git checkout` 或重新 `git clone`。
- **下一步**：确认 `CMakeLists.txt` 存在后，创建 `E:\副业\3DSlice\Slicer-build`，执行：
  `cmake -G "Visual Studio 17 2022" -A x64 ..\Slicer`，再按需 `cmake --build . --config Release`（耗时长）。随后定位 `SlicerConfig.cmake` 供扩展 `build-windows.ps1` 使用。

## 2026-04-22（重新拉取 Slicer）

- **操作**：按「重新拉取」处理。曾尝试结束 `git` 进程后删除 `E:\副业\3DSlice\Slicer`，因**文件被其他进程占用**未删除成功；改向 `Slicer-fresh` 全新克隆，**两次均失败**：
  - `Connection was reset` / **无法连接 `github.com:443`（约 21s 超时）**，判断为**当前环境网络**（防火墙、代理、运营商或需 VPN）导致。
- **新增脚本**：`scripts/clone-slicer.ps1` —— 在目标目录**不存在**时执行浅/全量 `git clone`（可 `-Shallow`），在**已是有效 git 仓库**时执行 `git fetch` + `checkout` 到 `main` 并 `submodule update`（相当于重新对齐远端）；支持多次重试。
- **请你本地执行**（网络可用时，在项目根 `E:\副业\3DSlice`）：
  1. 关闭可能占用 `Slicer` 文件夹的程序（如资源管理器窗口、防病毒实时扫描、其他 git 进程），**手动删除**有问题的 `Slicer` 目录；或先克隆到别名将 `TargetDir` 指到新路径。
  2. 若需走系统/ Git 代理，先配置好再运行：  
     `.\scripts\clone-slicer.ps1 -Shallow`  
     或指定目录：  
     `.\scripts\clone-slicer.ps1 -TargetDir "E:\副业\3DSlice\Slicer" -Shallow`
  3. 成功标志：存在 `Slicer\CMakeLists.txt` 且 `git -C Slicer status` 正常。然后再进行 `Slicer-build` 的 `cmake` 配置。

## 2026-04-22 23:xx（修复 clone 脚本解析错误）

- **问题**：用户终端报 `clone-slicer.ps1` “字符串缺少终止符 / 缺少右 }”。  
- **处理**：
  - 将 `scripts/clone-slicer.ps1` 文案改为 ASCII，避免 PowerShell 在编码不一致时把中文字符串解析坏；
  - 修复重试逻辑：前一次 clone 失败后会自动清理残留目录再重试；
  - 修复“已有仓库”分支：新增严格校验（`remote origin`、`fetch`、`checkout`、`submodule`），失败不再误报 `Done`，而是回退到“删目录后 fresh clone”。
- **验证结果**：
  - 解析错误已消失，脚本可正常启动并进入 clone；
  - 本次运行卡在旧目录清理：`Target directory exists and cannot be removed: E:\副业\3DSlice\Slicer`（目录仍被外部进程占用）。
- **下一步**：先关闭占用 `Slicer` 目录的程序（资源管理器窗口/杀毒/其它 git 进程），确保目录可删除，再执行 `.\scripts\clone-slicer.ps1 -Shallow`。

## 2026-04-23（Slicer 浅克隆成功）

- **操作**：`.\scripts\clone-slicer.ps1 -Shallow`。
- **结果**：`git clone` 一次成功，输出含 `Clone completed: E:\副业\3DSlice\Slicer`；对象接收与 `Updating files: 100%` 均完成。
- **下一步**：
  1. 确认存在 `E:\副业\3DSlice\Slicer\CMakeLists.txt`。
  2. 在 `E:\副业\3DSlice\Slicer` 内执行子模块拉取（完整构建通常需要）：`git submodule update --init --recursive`（若浅克隆报缺历史，再按需 `git fetch --unshallow` 或全量重克隆，以 CMake 实际报错为准）。
  3. 创建 `E:\副业\3DSlice\Slicer-build`，运行 `cmake -G "Visual Studio 17 2022" -A x64 ..\Slicer` 后进入全量 `cmake --build`。

## 2026-04-23（子模块拉取操作说明）

- **终端片段**：在 `Slicer` 目录下执行 `git submodule update --init --recursive` 前出现 `^C`（用户中断）；属正常误操作或换行续行（`>>`）时取消，不表示子模块命令本身失败。
- **建议**：在同一行执行或先 `cd` 再单独一行执行子模块命令，**不要中途 Ctrl+C**；Slicer 子模块体积大、耗时可长。若后续 CMake/子模块报错与浅克隆有关，再按报错处理 `unshallow` 或全量克隆。

## 2026-04-23（子模块命令无输出）

- **现象**：`git submodule update --init --recursive` 与 `git submodule status` 执行后**无任何输出**即回到提示符。
- **含义**：通常表示**当前仓库未配置子模块**（无 `.gitmodules` 或为空），或子模块已全部就绪；属**正常**，不一定表示失败。
- **下一步**：可直接进入 `Slicer-build` 做 CMake 配置；若 `cmake` 或官方文档要求拉取子模块而报错，再检查 `E:\副业\3DSlice\Slicer` 下是否存在 `.gitmodules` 并按报错处理。

## 2026-04-23（git pull 报错 Cannot rebase onto multiple branches）

- **现象**：在 `Slicer` 目录执行 `git pull`，拉取对象成功，但末尾 **`fatal: Cannot rebase onto multiple branches.`**  
- **原因**：本机常因开启了 **`pull.rebase=true`**（或 `branch.autoSetupRebase` 等）导致 `git pull` 用 **rebase**；在浅克隆/跟踪分支不唯一时，Git 可能无法确定 rebase 目标。  
- **处理**（在 `E:\副业\3DSlice\Slicer` 内任选一种）：
  1. 显式拉取并 **merge**（最稳妥）：`git pull --no-rebase origin main`  
  2. 或先设本次不用 rebase 再拉：`git -c pull.rebase=false pull origin main`  
  3. 若需长期用 merge 而非 rebase：`git config pull.rebase false`（仅影响本仓库则加 `--local`）  
- **注意**：`git pull` 已下载大量对象，**一般不必重下**；用上面命令让本地 `main` 与 `origin/main` 对齐即可。

## 2026-04-23（Slicer 与 origin 已对齐）

- **结果**：`git branch -vv` 显示 `main` 跟踪 `[origin/main]`（提交 `46b252cf26`）；`git pull` 输出 **`Already up to date.`**  
- **下一步**：创建 `E:\副业\3DSlice\Slicer-build`，执行 `cmake -G "Visual Studio 17 2022" -A x64 ..\Slicer`，再按需全量 `cmake --build . --config Release`。

## 2026-04-23（Slicer-build CMake：未找到 Qt 5.15）

- **现象**：`cmake` 在 `SlicerBlockFindQtAndCheckVersion.cmake` 处失败，提示找不到 **Qt5**（`Qt5Config.cmake` / `error: Qt 5.15 was not found`），要求设置 **`Qt5_DIR`**。  
- **原因**：从源码编 Slicer **不会自动装 Qt**，本机需单独安装 **Qt 5.15.x**，且为 **MSVC 64 位** 套件（官方常用 `msvc2019_64`，与 VS 2022 工具链 ABI 兼容）。  
- **处理**：
  1. 用 **Qt Online Installer** 安装 **Qt 5.15.2**（或 5.15.x），勾选 **MSVC 2019 64-bit**（名称类似 `msvc2019_64`），组件至少包含你后续构建要用的模块（Slicer 会拉不少依赖，建议按官方 Build Instructions 选默认 Desktop 相关组件）。  
  2. 记下列路径中的 **`Qt5` CMake 包目录**（版本号以本机为准）：  
     `C:\Qt\5.15.2\msvc2019_64\lib\cmake\Qt5`  
  3. 清空或重建构建目录后重新配置（**不要**在错误缓存上硬叠变量），例如：  
     `cmake -G "Visual Studio 17 2022" -A x64 -DQt5_DIR:PATH="C:/Qt/5.15.2/msvc2019_64/lib/cmake/Qt5" ..\Slicer`  
  4. 可选：在系统/用户环境变量中设 `Qt5_DIR` 或把 `C:\Qt\5.15.2\msvc2019_64` 加进 **`CMAKE_PREFIX_PATH`**，再跑 `cmake`。
- **命令行替代**：若已装 Python，可用 `aqtinstall` 只装目标套件（示例，路径自行调整）：
  `aqt install-qt windows desktop 5.15.2 win64_msvc2019_64 -O C:\Qt`
- **参考**：[Slicer 官方构建说明](https://slicer.readthedocs.io/en/latest/developer_guide/build_instructions/) 中关于 Qt/先决条件章节。

> **安全提示**：请勿在 `progress.md` 或仓库中保存 Qt/邮箱等账号密码；若已提交，请立即在 Qt 与邮箱侧**改密**并**从版本历史中清除**敏感行。

## 2026-04-23（SlicerDentalSegmentatorInference：单测与 CUDA/TensorRT）

- **单测（pytest/CTest 等）**：本仓库的 `SlicerDentalSegmentatorInference` 当前**未配置** CTest / GoogleTest；没有「一键跑全仓单测」的现成目标。验证推理加速请用**构建出的 CLI** 做**端测**（见下），若需单测可后续在子目录加 `add_test` 或独立 smoke 脚本。
- **GPU 与 TensorRT 行为**（`DentalSegmentatorInference/src/OnnxInference.cxx`）：`--device cuda` 时先尝试 **TensorRT EP**（FP16、`modelDir/trt_cache` 引擎缓存），再追加 **CUDA EP** 作回退；EP 不可用时无异常则走 CPU 推理。控制台会打印 `TensorRT EP configured` / `CUDA EP configured` 等，用于确认实际路径。
- **运行前**：需 **NVIDIA 驱动**、与 **ONNX Runtime 1.17.1 GPU 包**匹配的 **CUDA** 运行时（官方另有 `onnxruntime-win-x64-cuda12-*` 等变体，与本机已装 CUDA 大版本需一致）。**TensorRT** 需本机安装与 ORT/驱动兼容版本，并保证相关 **DLL 在 PATH** 或与 ORT 同目录，否则 TRT 可能跳过，仅 **CUDA EP** 生效。
- **端测命令**（路径自拟）：  
  `DentalSegmentatorInference.exe --inputVolume <体数据> --outputVolume <输出标签> --modelDir <含 plans.json 与 .onnx> --device cuda`
- **若 TRT 未生效**：检查控制台是否报 `TensorRT EP not available`；核对 ORT 文档中 **Windows + TensorRT EP** 的依赖与版本表；必要时在 CMake/环境中指定额外库路径，或仅依赖 CUDA EP。

## 2026-04-24（本机环境：TensorRT 是否可用 — 实检结论）

- **GPU / 驱动**：`nvidia-smi` 显示 **RTX 2070 Super**、驱动 **591.86**，报告 **CUDA 13.1**（驱动可支援的最高运行版本），**可跑 CUDA 推理**。
- **本机 CUDA Toolkit**：`nvcc` 为 **CUDA 11.1**（与驱动报告的 13.1 不矛盾，toolkit 是编译器版本，可较旧）。
- **TensorRT**：在常见安装路径下**未发现** `nvinfer*.dll` / 典型 `TensorRT` 目录，**未检测到已安装的 TensorRT 运行时**；**当前更可能无法让 ONNX Runtime 的 TensorRT EP 在运行时真正加载**（需从 NVIDIA 安装与 ORT 构建相匹配的 **TensorRT**，并将 `lib` 或 `bin` 加入 `PATH` 或拷至 ORT/可执行同目录），详见 [ONNX Runtime TensorRT 说明](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)。
- **实际结论**：**CUDA EP 侧具备基本条件**；**TensorRT 加速**需**另行安装 TensorRT** 并解决与 **onnxruntime-win-x64-gpu 1.17.1** 所连带的 **CUDA/cuDNN/TensorRT 版本**配套问题；在 TRT 未就绪时，程序会按代码逻辑**退回 CUDA EP** 或更后手段。

## 2026-04-24（Python：`onnxruntime-gpu` 与 CUDA 11 / 12 错配）

- **现象**（`3dslice` 环境）：`onnxruntime_providers_tensorrt.dll` / `onnxruntime_providers_cuda.dll` 报缺 **`cublas64_12.dll` / `cublasLt64_12.dll`**，并提示需 **cuDNN 9 与 CUDA 12**；最终仅 `CPUExecutionProvider`。
- **原因**：[PyPI 上 `onnxruntime-gpu` 自 1.19.0 起默认可视为 CUDA 12 系](https://onnxruntime.ai/docs/install)；本机为 **CUDA 11.1** + **TensorRT 8.6 for 11.x** 的 PATH 配置，与**默认 wheel** 不匹配。
- **处理**：`requirements-3dslice-pip.txt` 中**不再**直接写 `onnxruntime-gpu`；增加 `scripts/install-onnxruntime-gpu-cuda11.ps1`，从微软 [**onnxruntime-cuda-11** 源](https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/) 安装 **CUDA 11 变体**；`setup-conda-3dslice.ps1` 在 `pip -r` 后自动执行等效步骤。
- **备选**：若希望沿用 PyPI 默认 **CUDA 12** 的 `onnxruntime-gpu`，需本机安装 **CUDA 12.x Toolkit + cuDNN 9** 并正确加入 `PATH`（与当前 **仅 11.1** 的并行/优先级需自行管理）。

## 2026-04-24（TTA 8× 基准脚本）

- 新增 `SlicerDentalSegmentatorInference/run_baseline_tta8x.py`：与 `benchmark_compare.py` 中 **Original nnU-Net (TTA 8x, step 0.5, FP16)** 同逻辑，**写出** `baseline_tta8x.nii.gz`；`--dicom` 或 `--input` + **`--model-folder`（需含 `fold_0` 与 `plans.json`）**。
- 仓库内仍**无** `model_weights/`，未放置 nnU‑Net 权重前**不能**跑通 PyTorch 基准；ONNX 路径仍用 `run_dicom_e2e.py`（无 8 镜像 TTA）。

## 2026-04-24（NIfTI 写入与中文路径）

- `run_dicom_e2e` / `run_baseline_tta8x`：在 Windows 上 **SimpleITK/ITK 对含中文等路径的 `.nii.gz` 直写** 会失败。已增加 `write_nifti_sitk_safe`（先写 `%TEMP%` 再 `shutil.copy2` 到目标），DICOM 转 NIfTI 与分割输出均走该逻辑。

## 2026-04-24（`run_baseline_tta8x` 默认 FP32）

- 默认用 **全 FP32** 推理与预热；**`--fp16`** 切换为与旧 `benchmark_compare` 中 TTA+FP16 类似的更快路径。

## 2026-04-24（NIfTI 读取与中文路径）

- **直读** `E:\副业\...` 下 `.nii.gz` 同样会报 `Unable to open for reading`：新增 `sitk_path_io.py`（`read_sitk_image_safe` / `write_nifti_sitk_safe`），`benchmark_compare` / `benchmark_e2e` 的 `nnunet_preprocess` 与 `run_baseline_tta8x` 的参考体数据读取均改用该逻辑；`run_dicom_e2e` 从同模块 import，避免与 `benchmark_e2e` 循环依赖。
- `run_baseline_tta8x`：在导入 nnU-Net 前设置 `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` 为 `%TEMP%` 下空目录，减轻「路径未定义」类告警。

## 2026-04-24 02:20

- 更新 `.cursor/rules/progress-md-append.mdc`：此后向 `progress.md` 追加进展时，二级标题**必须**包含**具体时刻**，格式为 `## YYYY-MM-DD HH:mm`（本地时间 24 小时制），不再使用「仅日期」的标题。

## 2026-04-24 02:22

- 新增 `SlicerDentalSegmentatorInference/seg_vis_export.py`：分割 NIfTI 导出**三视图分类着色** PNG，并在同目录写 `preview_legend.txt` 说明标签与颜色。`export_colored_ply.py` 改经 `sitk_path_io` 读入，以支持含中文路径。
- `run_baseline_tta8x.py`：默认在 `--out` 同 stem 下生成 `*_preview.png`；`--no-preview` 关闭；`--ply` 额外生成 `*_colored.ply`（需 scikit-image）。`requirements-3dslice-pip.txt` 增加 `matplotlib`、`scikit-image`。

## 2026-04-24 02:24

- `run_baseline_tta8x.py`：**默认同出** `*_colored.ply`；`--no-ply` 可关闭。结束摘要中增加 PLY 路径行。

## 2026-04-24 02:25

- 新增 `SlicerDentalSegmentatorInference/compare_baseline_to_onnx.py`：以 **TTA8× baseline 的 NIfTI** 为参考，与 `run_dicom_e2e` 产出的 ONNX/TensorRT 标签对比：体素一致率、逐类及前景 1–5 平均 **Dice**；`--out-report` 可写报告；说明里注明 **ONNX 路径无 8 镜像 TTA**，与 baseline 的数值差不单独等于「TensorRT 精度差」。
- `run_dicom_e2e.py`：参考体数据读取与 `sitk_path_io` 一致，**改** 为 `read_sitk_image_safe`（与中文等路径的输入一致）。`compare` 中 JSON 输出改为多行可读的「JSON 块」。

## 2026-04-24 09:53

- 在 `3dslice` 下对 **`data/_dicom_for_baseline.nii.gz`** 跑 `run_dicom_e2e.py` → **`data/seg_onnx_trt.nii.gz`**。**TensorRT EP 未加载**（`onnxruntime_providers_tensorrt.dll` 报 error 126 / 需 TensorRT 库与 PATH），ORT **回退为 CUDAExecutionProvider**；非 TRT FP16 实测。
- `compare_baseline_to_onnx.py`：`baseline_tta8x.nii.gz` vs 上一步输出 → 体素一致率 **99.8059%**、前景 1–5 平均 Dice **0.969949**；报告 **`data/trt_vs_baseline.txt`**（文件名含 trt，实际为 CUDA EP 与 TTA8× baseline 对比）。

## 2026-04-24 22:18

- 将 `SlicerDentalSegmentatorInference/README.md` **全文**替换为**中文**（与先前英文版内容对应：介绍、构建步骤、Slicer 加载、ONNX 导出、架构示意、性能表）。下一步：无。

## 2026-04-24 22:23

- 将原 `.cursor/rules/progress-md-append.mdc` 的完整条文**提升为** Claude Code **project skill**：新增 **`.claude/skills/progress-md-append/SKILL.md`**（YAML + 中文说明；英文 `description` 便于发现）。`.mdc` 改为**短索引**并声明以该 `SKILL.md` 为单一事实来源，避免 Cursor/Claude 双份全文不同步。下一步：在 Claude Code 中可用 `/progress-md-append` 手动唤起（若与 skill `name` 一致则生效；以你本机 `claude code --help` 为准）。

## 2026-04-24 22:25

- 问题：`Unknown skill: progress-md-append`（Claude Code）。说明：多为**在首次创建** `.claude/skills` **前已开着的会话**未监视该目录，需**重启**；或**未在仓库根**执行。已在 `SKILL.md` 中写入排查步骤；新增备用斜杠 **`.claude/commands/append-progress.md`**（`/append-progress`）。下一步：用户退出重开 `claude` 并在 `e:\副业\3DSlice` 根下试 `/progress-md-append` 或 `/append-progress`。

## 2026-04-24 22:28

- 新增根目录 **`CLAUDE.md`**：约定 Claude Code 在本仓库**默认简体中文**；复杂任务时先给**可读分步思路**再出结论（用户所称「思维链」）；并链接 `progress.md` 技能。下一步：新开或重载会话后生效（以本机 Claude Code 对 `CLAUDE.md` 的加载规则为准）。

## 2026-04-24 22:29

- 将**语言 + 思维链**约定**移到全局** **`C:\Users\95231\.claude\CLAUDE.md`**；本仓库 `CLAUDE.md` 仅保留对全局路径的引用与 3DSlice 专属的 `progress.md` 说明。下一步：新开 Claude Code 会话或按官方说明重载，使全局记忆生效。

## 2026-04-24 22:30

- 新增 `SlicerDentalSegmentatorInference/test/test_trt.cxx`：**TensorRT 独立单测 C++ 程序**（不依赖 ONNX Runtime），使用 `NvInfer.h` / `NvOnnxParser.h` 直调 TensorRT C++ API。包含 5 个测试用例：
  1. **引擎构建**：ONNX → TensorRT engine（FP16/FP32 可选，动态 shape 支持）
  2. **序列化/反序列化**：engine → `.trt` 文件 → 还原，校验 I/O 签名一致性
  3. **推理输出验证**：合成输入（Box-Muller 正态分布）→ 输出 shape、NaN/Inf 检查、均值范围
  4. **一致性**：10 次同种子推理 → 逐元素 max_abs_diff / avg_abs_diff，FP16 允许 minor variation
  5. **性能基准**：warmup + 多轮计时 → P50/P95/P99 延迟、滑动窗口全量估算
- 新增 `SlicerDentalSegmentatorInference/test/CMakeLists_trt.txt`：独立 CMake 构建文件，查找 TensorRT（`nvinfer`/`nvonnxparser`/`nvinfer_plugin`）与 CUDA Toolkit，生成 `trt_test` 可执行文件；Windows 下自动复制 TRT DLL。
- 两阶段 buffer 分配：先分配输入 → 设 context input shape → 解析输出动态维度 → 再分配输出，避免动态 shape 导致 -1 维度的内存分配错误。

## 2026-04-24 22:45

- **改写 `test_trt.cxx` 为 TensorRT 8.6 API**：将全部绑定访问从 TRT 10.x 名称制（`getIOTensorName`/`getTensorIOMode`/`enqueueV3`）改为 TRT 8.6 索引制（`getNbBindings`/`getBindingName`/`bindingIsInput`/`enqueueV2`）；`setMemoryPoolLimit` → `setMaxWorkspaceSize`。
- **构建**：在 `test/trt/` 下新建 `CMakeLists.txt`（`C:\NVIDIA\TensorRT-8.6.1.6` + CUDA 11.1），`cmake --build` 成功（MSVC 2022，Release x64）。
- **修复动态 shape 报错**：添加 optimization profile（MIN=64³ / OPT=128³ / MAX=192³），解决 `Network has dynamic or shape inputs, but no optimization profile has been defined`。
- **修复 M_PI**：MSVC 需在 `cmath` 前 `#define _USE_MATH_DEFINES`；添加 `/utf-8` 消除 C4819 警告。
- **修复反序列化插件缺失**：添加 `initLibNvInferPlugins` + `#include <NvInferPlugin.h>`，解决 `getPluginCreator could not find plugin: InstanceNormalization_TRT`。
- **修复 runtime 析构顺序**：`buildEngineFromOnnx`/`deserializeEngine` 返回 `EngineBundle{runtime, engine}`，确保 runtime 在 engine 之后析构。
- **运行结果（GPU: RTX 2070 SUPER, FP16）**：
  - Test 1 (引擎构建): ONNX parsed 1.05s → engine built **369.7s** → 60.4 MB `.trt`
  - Test 2 (序列化/反序列化): 签名验证通过
  - Test 3 (推理输出): 首推理 **125ms**，output [1,6,64,64,64] range [-22.2, 37.8]，无 NaN/Inf
  - Test 4 (一致性): 10 次同种子 → **max_abs_diff = 0**（FP16 完美确定性）
  - Test 5 (benchmark): mean **112.9 ms/patch**, std 2.1ms, P99 119ms → 90 patches 估算 **10.2s**
- 新增 `--load-cache` 选项：跳过 6 分钟 ONNX 构建，直接加载 `.trt` 缓存的引擎。

## 2026-04-24 23:37

- 按「C++ TensorRT TTA8× 全流程」方案新增 **`test/trt_tta_full.cxx`** + **`test/trt_tta_helpers.cpp`** / **`trt_tta_helpers.hxx`**：ITK 读入 → 复用 `PlansParser`/`preprocess` → **TensorRT** 滑窗（与 `SlidingWindow` 同 Gauss/步进）+ 每 patch **可选 TTA8×**（`runTTAInference` 与 `test_trt.cxx` 同翻转语义）→ `postprocess` 写 NIfTI。命令行：`<input.nii> <modelDir> <out.nii> [--tta/--no-tta] [--fp16] [--load-cache] [--onnx] [--cache-out]`。`test/trt/CMakeLists.txt` 已增加 **trt_tta_full** 目标（ITK + nlohmann_json + 复制 TRT DLL）。本机需自行 `cmake`（`-DITK_DIR=...` 若需）后编译；未在 Agent 内实跑全量推理。

## 2026-04-25 00:04

- **`test/trt/CMakeLists.txt`**：`find_package(ITK QUIET …)`，**无 ITK 时仍构建 `trt_tta_full` 不生成、不阻断 `trt_test`**；有 `-DITK_DIR` 时再编 `trt_tta_full`。
- 在 `test/trt/build2` 中 **配置并编译** `trt_test`（Release，RTX 2070 SUPER，CUDA 11.1，TensorRT 8.6.1.6，cuDNN 8.0.5 有告警）。
- **全量 `trt_test`（ONNX→引擎）**：`dental_segmentator.onnx` 解析约 1.2s，**建引擎约 360s**，写出 **`models/trt_cache/dental_segmentator.trt`（~60.3 MB）**；**Test 1–4 全部 PASSED**（含序列化/反序列化、64³ 单 patch 输出范围与 FP16 10 次一致性 max_abs_diff=0）。
- **缓存 + `--tta`**：从 `dental_segmentator.trt` 启动；**TTA 方差/标签层测试 + TTA 性能基准** PASSED；8 次翻转合计约 856 ms/轮（**相对单次约 8.26×**），与 90 patch 全卷粗算一致。
- **Python**：`compare_baseline_to_onnx.py` 对已有 `baseline_tta8x.nii.gz` vs `seg_onnx_trt.nii.gz` **重跑成功**，指标与 `trt_vs_baseline.txt` 一致，另写 **`data/trt_vs_baseline_rerun.txt`**。
- **未执行**：`trt_tta_full` 端到端（**本机无 ITK C++ 的 `ITKConfig.cmake`**，仅 Python SimpleITK）；全卷 TRT+TTA 需安装 ITK 后 `-DITK_DIR=...` 再编再跑。

## 2026-04-25 01:35

- **ITK + `trt_tta_full` 构建**：在 `test/trt/build_tta` 用 `cmake -DITK_DIR=E:/vcpkg/installed/x64-windows/share/itk` 与 vcpkg `CMAKE_TOOLCHAIN_FILE` 配置。首次链接报 `TransformBaseTemplate` 未定义 → **`find_package` 增加组件 `ITKTransform`**（与 `ITKIONIFTI` 并列），`Release` 下 **`trt_tta_full.exe` 编过**。
- **端到端跑通**（`data/_dicom_for_baseline.nii.gz` → `data/trt_tta_full_smoke.nii.gz`）：`--load-cache models/trt_cache/dental_segmentator.trt`，**`--no-tta`、FP16**，滑窗 **108 patches**，总耗时约 **52s**；**preprocess / TRT 推理 / postprocess 均完成并写出 NIfTI**。运行前将 **`E:\vcpkg\installed\x64-windows\bin` 加入 PATH** 以便加载 ITK DLL。TensorRT 仍提示 **cuDNN 链到 8.9、运行时加载 8.0.5**（与此前 `trt_test` 一致，未阻断）。
- 真机与进度文档中「TTA8× 全卷 C++ 对齐基线」下一步：同命令去掉 **`--no-tta`** 再跑，并与 `baseline_tta8x.nii.gz` 用既有 Python 比较脚本对指标。

## 2026-04-25 01:41

- **全卷 `trt_tta_full` TTA8×**（`data/_dicom_for_baseline.nii.gz` → **`data/trt_tta_full_volume_tta8x.nii.gz`**）：`--load-cache`、`--tta`、`--fp16`，**108 patches**，本机总耗时约 **236s**（~3.9 min）。
- **与 `data/baseline_tta8x.nii.gz` 对比**（`compare_baseline_to_onnx.py`，报告 **`data/cpp_trt_tta8x_vs_baseline.txt`**）：
  - 形状 **(300, 536, 536)** 一致；**体素一致率约 91.63%**（约 **8.37%** 体素与 baseline 标签不同）；
  - **背景 Dice ≈0.957**；**前景类 1–5 的 Dice 很低**（Maxilla **0.079**、Mandible **0.112**、Upper/Canal **0**、Lower **0.032**，前景均值约 **0.045**）—— 说明除大量背景外，**分割标签与 PyTorch 参考未对齐**（更可能是**类别/argmax/后处理链或标签编码**与 baseline 路径不一致，而非单纯 FP16 噪声）；后续需用 `np.unique`、两路头文件/叠加可视化或逐类混淆矩阵排查。

## 2026-04-25 02:08

- **`compare_baseline_to_onnx.py`**：`baseline_tta8x.nii.gz` vs 本机最新 **`trt_tta_full_volume_tta8x.nii.gz`**（`--out-report` → **`data/cpp_trt_tta8x_vs_baseline_recheck.txt`**）。
  - 形状 **(300, 536, 536)** 一致；**体素一致率约 92.23%**；差异体素约 **7.77%**。
  - **背景 Dice ≈0.960**；前景 **1–5**：Maxilla **0.107**、Mandible **0.156**、Upper/Lower **0**、Canal **0.024**；**前景均值约 0.057**（较首轮 C++ 对比略好，**整体仍与 PyTorch 参考在前景上偏差大**）。

## 2026-04-25 02:30

- **C++ 预处理根因排查**：此前 `trt_tta_full` 输出与 PyTorch baseline 前景 Dice 极低（~0.045）的原因定位为两个 bug：
  1. **transpose 轴映射错误**（`PlansParser.cxx`）：`plans.json` 中 `transpose_forward=[1,0,2]` 是 **numpy ZYX** 约定（交换 Z↔Y），但 C++ 直接当作 **ITK XYZ** 约定使用（变成了交换 X↔Y）。中间过程（crop/resample/sliding window）在错误方向的体数据上进行。
  2. **Crop 阈值错误**（`Preprocessor.cxx`）：`computeNonzeroBBox` 使用 `abs(v) > 1e-8f`，对 CT 数据几乎不裁剪（空气 HU≈-1000 也通过）；Python 参考管线用 `> -500`（只保留组织/骨骼，排除空气）。
- **修复**：
  - `PlansParser.cxx:41-53`：增加 numpy→ITK 轴号转换（`np2itk = {2,1,0}`），`[1,0,2]` → `[0,2,1]`。
  - `Preprocessor.cxx:48`：`abs(v) > 1e-8f` → `v > -500.0f`。
- **构建**：`test/trt/build_tta` 中 `cmake --build . --config Release` **成功**（`trt_tta_full.exe` 已更新）。
- **下一步**：重新跑 `trt_tta_full --tta --fp16 --load-cache`，再 `compare_baseline_to_onnx.py` 验证前景 Dice 是否恢复正常。

## 2026-04-25 02:40

- **BSpline → Linear 插值替换**（`Preprocessor.cxx`）：发现 Python 参考管线 resample 用 **trilinear**，而 C++ 用的 **BSpline-3**，导致约 45% 体素不一致。将 `#include` 和 `using` 从 `BSplineInterpolateImageFunction` → `LinearInterpolateImageFunction`，`interpolator` 改为线性插值。构建成功。
- **调试 dump 支持**：`trt_tta_full.cxx` 增加 `--dump-preprocessed PATH` 选项，预处理后写出 NIfTI；创建 `dump_preprocessed_py.py` 用于 Python 管线 dump 同等中间结果。
- **预处理体素对比**（`compare_preprocessed.py` / `compare_preprocessed2.py`）：
  - 形状 **(339, 288, 323)** 一致；min/max **完全相同**（-2.478 / 3.211）；mean 略异（C++ -2.299644，PY -2.299021）。
  - **体素一致率仅 ~70.5%**（<1e-6）；175.9 万体素 diff > 0.1（约 5.6%）。
  - **最大差异 3.382**，位于 (120,168,198)：C++=-0.493，PY=2.889 —— 个别位置相差极大。
- **结论**：transpose 映射修正 + crop 阈值修正 + 线性插值三个改动后，**形状已对齐，但仍有 30% 体素不一致**，根因尚未完全定位。

## 2026-04-25 03:00

- **诊断思路**：排除预处理 → 推理 → 后处理哪个环节导致前景 Dice 低。下一步：**用 PyTorch 模型对 C++ 预处理体直接做滑窗推理**，若输出好则问题在 TRT 推理/后处理，若输出差则问题仍在预处理。

## 2026-04-25 08:13

- **阶段对齐验证完成**：按「预处理 → 推理 → 后处理」拆解误差。新增诊断脚本 **`diagnose_resample.py`**、**`dump_prepost_argmax_py.py`**、**`run_onnx_notrt_e2e.py`**、**`compare_labels.py`**；`trt_tta_full.cxx` 增加 **`--dump-prepost-label`**（后处理前 argmax dump）。
- **预处理根因最终确认并修复**：差异主要来自 ITK/SimpleITK 物理坐标 resample 与 PyTorch `torch.nn.functional.interpolate(..., align_corners=False)` 的数组坐标/边界规则不同。`Preprocessor.cxx` 改为显式按 **D/H/W 数组语义**做三线性采样，C++ vs Python 预处理对比从 `mean_abs_diff≈0.01996` 降到 **`6.7e-7`**，`max_abs_diff≈1.37e-4`，**`diff > 0.001` 为 0**。
- **推理阶段验证**：同一预处理体上，Python ONNX(CUDA, no TensorRT EP) 与 C++ TensorRT **后处理前 argmax** 对比：一致率 **99.953%**；Dice：背景 **0.9998**，前景各类约 **0.992–0.999**，说明滑窗、Gaussian、TRT/ONNX 推理基本对齐。
- **后处理阶段验证**：Python ONNX no-TTA 完整输出 vs C++ no-TTA 完整输出：一致率 **99.960%**；各类 Dice 约 **0.992–0.999**，说明 postprocess 的去 pad / resize / uncrop / transpose 基本对齐。
- **最终全卷 TTA8×**：重新运行 `trt_tta_full`（`--load-cache`、`--tta`、`--fp16`），输出 **`data/trt_tta_full_volume_tta8x_fixed.nii.gz`**，耗时约 **155s**。与 **`data/baseline_tta8x.nii.gz`** 对比报告 **`data/cpp_trt_tta8x_vs_baseline_fixed.txt`**：形状一致，**体素一致率 99.9673%**，差异 **0.0327%**；Dice：背景 **0.999840**、Maxilla **0.993351**、Mandible **0.998911**、Upper Teeth **0.998875**、Lower Teeth **0.999869**、Mandibular Canal **0.999587**，前景均值 **0.998119**。

## 2026-04-25 10:30

- **独立验证三阶段对齐**：确认 C++ TRT no-tta 全管线与 Python ONNX no-tta 参考一致。
  - **预处理**：手动三线性插值 → mean_abs_diff 6.7e-7，diff > 0.001 = 0，形状 (339,288,323) 一致。
  - **推理**（后处理前 argmax）：一致率 **99.953%**，前景 Dice 均值 ~0.997。
  - **后处理**（最终输出）：一致率 **99.960%**，前景 Dice 均值 ~0.997。
  - **修复**：`trt_tta_full.cxx` 中 `findOnnxModel` 移到 `else` 分支，避免 `--load-cache` 时因无 ONNX 文件报错。

## 2026-04-25 11:00

- **全卷 TTA8× 重验证**：C++ TRT TTA8×（`--tta --fp16 --load-cache`）→ `data/trt_tta8x_final.nii.gz`，与 `data/baseline_tta8x.nii.gz` 对比：
  - 一致率 **99.9673%**，背景 Dice **0.999840**
  - 前景 Dice：Maxilla **0.9934**、Mandible **0.9989**、Upper Teeth **0.9989**、Lower Teeth **0.9999**、Canal **0.9996**，均值 **~0.9981**
  - 与用户独立验证结果完全吻合，全管线对齐确认。
