# 大阶段一开工前置准备记录

> - 日期：2026-07-16
> - 基线提交：`80049e096552ff89f656636ce120b38d3ef8cbb7`
> - 当前分支：`deploy-cpp`
> - 范围：开发终端、ORT C++ SDK、现有 3/3 CTest、GTest 依赖方案、模型 provenance/license 预审
> - 边界：本记录不是 `S1-01` 实现；未修改 `cpp_infer/`，未建立 Runtime/artifact contract，未接入 ORT Session 或 GTest target

## 1. Readiness 结论

| 检查项 | 结论 | 证据 |
|---|---|---|
| Git 基线 | 通过 | 检查前工作区干净，本地分支显示与 `origin/deploy-cpp` 对齐 |
| Visual Studio x64 开发环境 | 通过 | `VsDevCmd.bat -arch=amd64 -host_arch=amd64` 成功；`cl`、`nmake`、`cmake`、`ctest` 均可发现 |
| ONNX Runtime C++ SDK | 通过 | Windows x64 CPU SDK `1.19.2` 的 header、import library、runtime DLL 全部存在 |
| 现有 C++ baseline | 通过 | 全新 `%TEMP%` Release/NMake 构建成功，3/3 CTest 通过 |
| GTest 方案 | 已冻结，尚未接入 | 固定 v1.17.0 完整提交、HTTPS archive 和 SHA-256；S1-01 才写入 CMake |
| 模型来源整理 | 已开始并形成证据链 | 训练配置、实验记录、导出命令、ONNX metadata、Git 时间线及所有者确认均已记录；exact `best.pt` 未在当前副本或 Git 历史中找到，仍需保留为限制 |
| 许可证整理 | 已开始，仍有发布门禁 | 模型 metadata/Ultralytics 官方口径为 AGPL-3.0；NEU 官方页未发现明确再分发许可证文本 |

## 2. 开发终端与本机工具链

本机没有可搜索到的 “Developer PowerShell for VS 2026” 快捷方式，但完整开发环境脚本存在：

```text
D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat
```

从普通 PowerShell 启动等价 x64 环境：

```powershell
cmd.exe /k '"D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64'
```

进入 Developer Command Prompt 后，输入 `powershell` 可得到继承 MSVC 环境的 PowerShell。实测工具为：

```text
Visual Studio Developer Command Prompt  v18.1.1
MSVC compiler                           19.50.35721.0
cl.exe                                  ...\MSVC\14.50.35717\bin\Hostx64\x64\cl.exe
nmake.exe                               ...\MSVC\14.50.35717\bin\Hostx64\x64\nmake.exe
cmake.exe                               Visual Studio bundled CMake 4.1.1-msvc1
ctest.exe                               Visual Studio bundled CTest 4.1.1-msvc1
OpenCV C++                              4.8.0 x64 vc16
```

终端内验证命令：

```powershell
where.exe cl
where.exe nmake
where.exe cmake
where.exe ctest
cmake --version
ctest --version
```

## 3. ONNX Runtime C++ SDK

固定依赖：官方 Windows x64 CPU SDK `1.19.2`，仓库外路径：

```text
D:\01_Base\Tools\onnxruntime-win-x64-1.19.2
```

实测结果：

| 文件 | 存在 | 大小 |
|---|---:|---:|
| `include/onnxruntime_cxx_api.h` | 是 | 111,467 bytes |
| `lib/onnxruntime.lib` | 是 | 2,124 bytes |
| `lib/onnxruntime.dll` | 是 | 11,234,848 bytes |
| `VERSION_NUMBER` | 是 | `1.19.2` |

人工复核命令：

```powershell
$env:ONNXRUNTIME_ROOT = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'

Test-Path "$env:ONNXRUNTIME_ROOT\include\onnxruntime_cxx_api.h"
Test-Path "$env:ONNXRUNTIME_ROOT\lib\onnxruntime.lib"
Test-Path "$env:ONNXRUNTIME_ROOT\lib\onnxruntime.dll"
```

三项均已实测为 `True`。S1-01/S1-02 的 CMake 必须通过 `ONNXRUNTIME_ROOT` 或等价 cache 参数接入，不能写死这台机器的绝对路径。

## 4. 现有 baseline clean build

最终成功证据目录：

```text
C:\Users\EVERBR~1\AppData\Local\Temp\yolo_defect_pre_s1_clean_20260716_214433
```

环境：MSVC `19.50.35721.0`、NMake、Release、OpenCV `4.8.0`。配置、编译和测试全部成功：

```text
1/3 yolo_defect_cpp_help        Passed
2/3 yolo_defect_cpp_config      Passed
3/3 yolo_defect_cpp_preprocess  Passed

100% tests passed, 0 tests failed out of 3
Total Test time (real) = 0.28 sec
```

人工复现时，在继承 VS 环境的 PowerShell 中执行：

```powershell
$env:PATH = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
Set-Location 'D:\01_Base\CodingSpace\yolo_defect'

$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_pre_s1_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))

cmake -S cpp_infer -B $BuildDir `
  -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DCMAKE_BUILD_TYPE=Release

cmake --build $BuildDir
ctest --test-dir $BuildDir --output-on-failure
```

### 调试记录

第一次自动编排命令把 `set "PATH=...;%PATH%"` 放在与 `VsDevCmd.bat` 相同的 CMD 命令行中。CMD 在执行 VS 初始化前提前展开了 `%PATH%`，随后覆盖了 VS 刚加入的 CMake 路径，导致构建成功后报 `ctest is not recognized`。这不是源码或测试失败。

修正方式是在启动 `VsDevCmd.bat` 前，由父 PowerShell 先把 OpenCV DLL 目录加入 `PATH`。随后重新创建了上面的 clean build，配置、编译和 3/3 CTest 一次通过。

## 5. GTest FetchContent 冻结方案

方案只在本记录中冻结，S1-01 才真正接入：

- 版本：GoogleTest `v1.17.0`（当前最新稳定 release）
- 完整提交：`52eb8108c5bdec04579160ae17225d66034bd723`
- C++ 要求：C++17，与本项目一致
- 来源：[GoogleTest v1.17.0 release](https://github.com/google/googletest/releases/tag/v1.17.0)
- 官方方式：[GoogleTest CMake Quickstart](https://google.github.io/googletest/quickstart-cmake.html)
- 下载方式：commit archive URL；不依赖本机 `git clone`
- 实测 archive 大小：1,133,982 bytes
- 实测 SHA-256：`9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`

计划中的 CMake 片段：

```cmake
include(FetchContent)

FetchContent_Declare(
  googletest
  URL
    https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
  URL_HASH
    SHA256=9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# Prevent GoogleTest from replacing the parent MSVC runtime-library choice.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
```

联网预检发现 `git ls-remote` 在本机连接 GitHub 时被 reset，但同一官方 commit archive 已通过 HTTPS 成功下载并完成 SHA-256 核验，因此选择 URL 方案。若未来构建环境完全离线，可人工解压同一 archive，并在正常 CMake configure 命令中追加：

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='D:\01_Base\Tools\googletest-1.17.0'
```

## 6. 当前 ONNX artifact 的已验证身份

| 字段 | 已验证值 |
|---|---|
| 路径 | `models/best.onnx` |
| 大小 | 12,336,935 bytes |
| SHA-256 | `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` |
| Git blob | `a1777f837e6a0268e30a17eebd6d46b4ab56b7a1` |
| 首次进入 Git | `bdf04ad98e1a41ea4bb0afd569b642f7b80cf643`，2026-03-30 16:56 +08:00 |
| producer | PyTorch `2.0.0` |
| exporter metadata | Ultralytics `8.4.24` |
| metadata 时间 | `2026-03-30T14:17:48.264011`（metadata 未记录时区） |
| opset | 17 |
| 输入 | `images` float32 `[1,3,800,800]` |
| 输出 | `output0` float32 `[1,10,13125]` |
| 类别 | `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches` |
| 导出参数 | batch 1、half false、dynamic false、simplify true、nms false |
| metadata license | `AGPL-3.0 License (https://ultralytics.com/license)` |

当前 Conda 环境也能确认：Python `3.9.25`、Ultralytics `8.4.24`、PyTorch `2.0.0`、ONNX `1.19.1`、ONNX Runtime `1.19.2`。这能复核 metadata 和 Python ORT baseline，但 exact `best.pt` 缺失意味着不能重新导出同一 artifact。

## 7. 训练与导出 provenance 证据链

2026-07-16，项目所有者已人工确认两项事实：当前 `models/best.onnx` 是本人在本项目中导出的，并且来源 checkpoint 是 `runs/detect/final_train_2/weights/best.pt`。因此 lineage 状态可记为 `owner_confirmed`。下面的仓库证据也与该确认一致：

1. `configs/final_train_2.yaml` 记录 `yolov8n.pt`、NEU-DET、100 epochs、800 输入、SGD、`lr0=0.01`、mosaic 1.0、mixup 0.0。
2. `docs/archive/experiment_log.md` 记录该 run 的 best checkpoint：mAP@0.5 `0.743`、mAP@50-95 `0.388`。
3. 该配置与结果于 2026-03-26 提交到 `0b18290518d1dec938175f91d6f3cf3720f2750a`。
4. ONNX metadata 的导出时间为 2026-03-30 14:17:48。
5. README 于 2026-03-30 14:48 的提交 `d1f8f1b4ae34f3199a4f6adb159c40c338966ae0` 明确记录：

   ```powershell
   python scripts/export_onnx.py `
     --weights runs/detect/final_train_2/weights/best.pt `
     --imgsz 800
   ```

6. 当前 ONNX 于当日 16:56 提交，尺寸和 metadata 与上述导出路径、输入尺寸、类别相符。
7. `scripts/export_onnx.py` 使用 `YOLO(weights).export(format='onnx', imgsz=..., simplify=True)`，默认移动到 `models/best.onnx`。

但当前项目副本无法独立重放这一导出：

```text
runs/detect/final_train_2/weights/best.pt -> 当前不存在
```

进一步核对结果：

- 当前工作区（含被忽略文件）没有找到任何 `.pt`。
- Git 全部分支和历史中没有找到曾跟踪的 `.pt`。
- `.gitignore` 明确排除了 `runs/`、`models/*.pt`、`yolov8*.pt` 和 `yolo*.pt`，因此 checkpoint 当初未进入 Git 是符合仓库规则的。

这只证明 exact checkpoint 不在当前项目副本或 Git 历史中，不证明它已永久丢失；它仍可能存在于其他硬盘、备份或旧机器。当前准确表述是：**lineage 已由项目所有者确认，但 checkpoint 目前不可用，尚未用 exact checkpoint 重新导出并做哈希复核。** `best.onnx` 本身足以继续本地 C++ Runtime；缺少 `.pt` 只限制重新导出和新的 PyTorch/Python ORT/C++ 三方直接比较。

## 8. 许可证与分发检查点

- 仓库作者编写的源代码目前声明 MIT；该声明不能自动覆盖模型权重或数据集。
- ONNX metadata 明确标注 AGPL-3.0。
- [Ultralytics 官方许可页](https://www.ultralytics.com/license)当前说明，其 YOLO 训练模型默认位于 AGPL-3.0 框架，闭源/商业场景需要适用的商业许可。这里仅记录上游官方口径，不构成法律意见。
- [东北大学 NEU 官方数据页](https://faculty.neu.edu.cn/songkc/en/zdylm/263265)提供 NEU-DET 下载、数据描述和引用要求，但本次审计未在该页面找到明确的 SPDX/Creative Commons/再分发许可证文本。
- 从工程推进角度，当前许可不确定性不阻塞本地技术验证，但它不等于已取得任何使用或分发授权。公开分发模型/数据、商业化或声称整个交付物统一为 MIT 之前，必须单独复核模型与数据的许可义务。
- 项目所有者已明确当前用途是个人学习。个人学习和本地验证本身不构成必须购买 Ultralytics Enterprise 的开工条件；商业许可只在未来采用闭源/专有商业路线等适用场景时重新评估。
- 项目所有者选择 **A：继续在公开仓库分发 `models/best.onnx` 和 NEU-DET 数据**。非商业不等于无分发义务，因此模型的 AGPL 声明、仓库 MIT 声明和数据集未明确的再分发条款仍须分开标注，并在正式发布口径冻结前复核；本次不擅自修改 `LICENSE`。

## 9. 用户确认结果与剩余人工项

| 项目 | 2026-07-16 状态 | 工程含义 |
|---|---|---|
| 当前 ONNX 是否由项目所有者本人导出 | 已确认：是 | provenance 可标记为 `owner_confirmed` |
| 是否来自 `final_train_2/weights/best.pt` | 已确认：是 | artifact contract 可记录该源路径，但不能伪造 checkpoint hash |
| exact `best.pt` 是否仍在其他硬盘、备份或旧机器 | 尚未确认；当前工作区和 Git 历史均没有 | 不阻塞 C++ Runtime；若找到，可再做哈希、重导出和三方比较 |
| 当前是否需要 Ultralytics Enterprise | 不需要作为个人学习/本地验证的开工条件 | 未来若转闭源或专有商业用途，再单独评估适用许可 |
| 是否继续公开分发模型和数据 | 已选择 A：继续公开分发 | 发布前仍需明确遵守模型许可，并核实 NEU-DET 再分发依据 |

寻找备份中的 exact `best.pt` 是可选加分项，不是 S1-01 前置条件。若找到，应先保留原文件、记录绝对路径并计算 SHA-256，不要覆盖当前 `models/best.onnx`；未找到时，后续一致性验收仍按同一 ONNX artifact 做 Python ORT/C++ ORT 严格比较。
