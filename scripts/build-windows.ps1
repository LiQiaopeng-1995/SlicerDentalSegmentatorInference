# 在已安装 Visual Studio 2022/生成工具 的前提下，配置并构建本扩展（含 SuperBuild 下载 ONNX Runtime）。
# 必须先完成 3D Slicer 的源码配置/编译，并得到包含 SlicerConfig.cmake 的目录，例如：
#   Slicer-build/Slicer-build
#
# 用法:
#   .\scripts\build-windows.ps1 -SlicerDir "D:\Slicer-build\Slicer-build"
#
param(
  [Parameter(Mandatory = $true, HelpMessage = "指向包含 SlicerConfig.cmake 的 Slicer 构建目录")]
  [string] $SlicerDir,

  [string] $SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,

  [string] $BuildDir = (Join-Path (Split-Path $SourceDir -Parent) "SlicerDentalSegmentatorInference-build")
)

$ErrorActionPreference = "Stop"

$cfg = Join-Path $SlicerDir "SlicerConfig.cmake"
if (-not (Test-Path -LiteralPath $cfg)) {
  Write-Error "Slicer_DIR 无效：未找到 $cfg 。请使用从源码配置/构建 Slicer 后产生的目录（非安装包安装路径）。"
  exit 1
}

if (-not (Test-Path -LiteralPath $SourceDir)) {
  Write-Error "未找到源码目录: $SourceDir"
  exit 1
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir
try {
  & cmake -G "Visual Studio 17 2022" -A x64 "-DSlicer_DIR:PATH=$SlicerDir" $SourceDir
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  & cmake --build . --config Release
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  Write-Host ""
  Write-Host "构建完成。CLI 可执行文件通常在:" -ForegroundColor Green
  Write-Host "  $BuildDir\DentalSegmentatorInference_inner-build\DentalSegmentatorInference\Release\DentalSegmentatorInference.exe"
}
finally {
  Pop-Location
}
