---
name: yolo-defect-dev
description: Recorded yolo_defect environment paths, toolchain versions, entry commands, and known local environment pitfalls.
---

# 环境小抄

记录整理于 2026-09-11，来自本机配置与已有运行记录。路径和版本可能变化；这份小抄提供线索，实际配置与命令结果反映当前环境。

## Windows

仓库：`D:\01_Base\CodingSpace\yolo_defect`。本机覆盖配置：`cpp_infer/.stage1.local.psd1`（Git 忽略）。

| 配置项 / 环境变量 | 已记录位置 |
| --- | --- |
| `OrtRoot` / `ONNXRUNTIME_ROOT` | `D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| `OpenCvDir` / `OpenCV_DIR` | `D:\01_Base\Tools\opencv\build\x64\vc16\lib` |
| `OpenCvBin` / `YOLO_DEFECT_OPENCV_BIN` | `D:\01_Base\Tools\opencv\build\x64\vc16\bin` |
| `PythonExe` / `YOLO_DEFECT_PYTHON` | `C:\Users\Everbreath\.conda\envs\TestBase\python.exe` |
| `GTestSource` / `YOLO_DEFECT_GTEST_SOURCE` | 本次可用源码 `cpp_infer/build/googletest`（复用 WSL 的 1.14.0）；local config 中旧 TEMP 路径已失效 |

已记录版本：MSVC 19.50、VS 附带 CMake/CTest 4.1.1、NMake、OpenCV C++ 4.8.0、ORT 1.19.2；Python 3.9.25、NumPy 2.0.2、OpenCV Python 4.13.0。PTQ 环境另需 ONNX，已有记录为 1.19.1。

`cpp_infer\tools\stage1.cmd` 自动通过 vswhere 进入 x64 MSVC 环境；`YOLO_DEFECT_VSDEVCMD` 可指定入口。例如：

```powershell
$exampleGTestSource = Join-Path $PWD 'cpp_infer/build/googletest'
$exampleBuildDir = Join-Path $env:TEMP 'yolo_defect_stage1_docs_refactor'
cpp_infer\tools\stage1.cmd build -GTestSource $exampleGTestSource -BuildDir $exampleBuildDir
cpp_infer\tools\stage1.cmd detect data\images\val\crazing_241.jpg -GTestSource $exampleGTestSource -BuildDir $exampleBuildDir
```

默认构建目录：`$env:TEMP\yolo_defect_stage1_manual_release`。依赖配置优先级：命令参数、local config、环境变量、默认值。

## WSL2 / Linux x86_64

已记录环境：Ubuntu 24.04、GCC 13.3、CMake 3.28.3、Ninja 1.11.1；OpenCV C++ 4.6.0 位于 `/usr`，GoogleTest 源码位于 `/usr/src/googletest`。

```bash
cd /mnt/d/01_Base/CodingSpace/yolo_defect
export ONNXRUNTIME_ROOT="$HOME/.local/opt/onnxruntime-linux-x64-1.19.2"
export YOLO_DEFECT_PYTHON="$HOME/.venvs/yolo-defect-gate-a/bin/python"
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest
bash cpp_infer/tools/stage1.sh build
```

已记录 Python 环境：3.12.3、ORT 1.19.2、OpenCV Python 4.10.0、NumPy 2.0.2。默认构建目录 `/tmp/yolo_defect_stage1_linux_release`；`YOLO_DEFECT_BUILD_DIR` 和 `YOLO_DEFECT_RUN_DIR` 可指定构建及结果目录。

## Linux AArch64 / QEMU

| 内容 | 已记录位置或版本 |
| --- | --- |
| 交叉编译器 / QEMU | `aarch64-linux-gnu-g++` 13.3 / `qemu-aarch64` 8.2.2 |
| 依赖根目录 | `$HOME/.local/opt/yolo-defect-aarch64` |
| ORT 1.19.2 | 依赖根目录下 `onnxruntime-linux-aarch64-1.19.2` |
| OpenCV 4.6.0 私有 sysroot | 依赖根目录下 `ubuntu-noble-opencv-4.6.0` |
| Target loader | `/usr/aarch64-linux-gnu` |
| CMake toolchain | `cpp_infer/cmake/toolchains/linux-aarch64-gnu.cmake` |
| 构建目录 | `/tmp/yolo_defect_stage2_aarch64_core`、`/tmp/yolo_defect_stage2_aarch64_full` |

入口为 `bash cpp_infer/tools/stage2_aarch64.sh <action>`；`YOLO_DEFECT_AARCH64_CONFIG` 可选择 RuntimeConfig。路径变量包括 `YOLO_DEFECT_AARCH64_DEPS_ROOT`、`YOLO_DEFECT_AARCH64_ORT_ROOT`、`YOLO_DEFECT_AARCH64_SYSROOT`、`YOLO_DEFECT_AARCH64_LOADER_PREFIX`；完整选项在脚本帮助中。

## 已知环境问题

- TEMP 和 `/tmp` 中的构建、结果或 GoogleTest 源码可能被清理。
- PowerShell 调用 WSL 时，嵌套 Bash 的 `$变量` 可能被宿主提前展开；非交互式 sudo 可能没有密码输入终端。
- Ubuntu amd64 与 arm64 使用不同 apt 镜像；直接安装 ARM64 OpenCV 开发包可能与 host 开发包冲突。现有 bootstrap 用下载与解包生成私有 sysroot，安装示例见 [C++ 技术手册](../../../cpp_infer/README.md)。
