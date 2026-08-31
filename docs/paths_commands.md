# 路径、工具链与环境诊断

本文是本项目**当前工具链版本、可配置依赖入口、构建命令和环境踩坑的唯一操作事实源**。它不记录用户名或个人绝对路径；机器位置通过 ignored local config、环境变量、`$HOME`/`%USERPROFILE%` 或占位符表达。`AGENTS.md` 不复制这些细节；只有任务需要查找工具链路径、依赖版本或位置、环境入口命令或已知环境踩坑时，仓库 Skill `yolo-defect-dev` 才按需读取本文的相关部分。任务仅仅涉及构建、运行、测试、benchmark 或 profiling，不构成加载该 Skill 的理由。历史 closure 中的版本和路径是当时证据快照，不替代本文。

## 1. 先看这里

- 日常从仓库根目录的普通 PowerShell 或 CMD 调用 `cpp_infer\tools\stage1.cmd`；它会发现 Visual Studio、调用 x64 `VsDevCmd.bat`，再启动无 profile 的 PowerShell 调度器。
- 普通 PowerShell 直接运行 `ctest`、`cmake`、`cl` 或 `nmake` 找不到命令，通常表示当前 shell 没有继承 `VsDevCmd.bat` 注入的 `PATH/INCLUDE/LIB`，不能据此判断源码或测试失败。先运行 `stage1.cmd doctor`。
- `call` 是 CMD builtin，不是 PowerShell command。PowerShell 查找可执行文件时使用 `where.exe`，不要误用 `where` alias。
- PowerShell 不会因 native command 返回非零而自动停止；手工运行底层命令后立即检查 `$LASTEXITCODE`。
- Python `onnxruntime` wheel 不能替代 C++ SDK；C++ 构建需要 headers、import library 和匹配的 runtime DLL。
- 机器路径写入 Git 忽略的 `cpp_infer/.stage1.local.psd1` 或进程环境变量，不写入源码、CMake、tracked defaults 或 README。

## 2. Windows 统一入口

查看帮助和检查环境不会运行完整测试：

```powershell
cpp_infer\tools\stage1.cmd help
cpp_infer\tools\stage1.cmd doctor
```

常见开发动作：

```powershell
cpp_infer\tools\stage1.cmd build
cpp_infer\tools\stage1.cmd test
cpp_infer\tools\stage1.cmd detect "data\images\val\crazing_241.jpg"
cpp_infer\tools\stage1.cmd batch "data\images\val" -Workers 4 -QueueCapacity 8
cpp_infer\tools\stage1.cmd batch cpp_infer\tests\fixtures\s2_03_consistency_manifest.txt -Workers 2
cpp_infer\tools\stage1.cmd batch-compare
cpp_infer\tools\stage1.cmd batch-compare -Config cpp_infer\configs\int8_u8s8_config.txt
cpp_infer\tools\stage1.cmd profile
cpp_infer\tools\stage1.cmd profile -Config cpp_infer\configs\int8_u8s8_config.txt -ProfileRuns 10
```

| Action | 行为 |
|---|---|
| `help` | 显示参数，不检查依赖、不构建 |
| `doctor` | 只读检查 x64 MSVC、CMake/CTest、ORT SDK、OpenCV、Python、GTest policy 和有效配置 |
| `build` | 增量 Release 构建；没有构建树时先 configure |
| `clean-build` | 只在受保护 TEMP 边界内清理并重新构建 |
| `test` | 构建当前源码并运行完整 CTest；日常局部改动优先运行相关 target/test |
| `detect <image> [output-directory]` | 复用 `DetectorPipeline` 运行单图，可选 `-Config` 与 `-Overwrite` |
| `batch <input> [output-directory]` | 输入是目录时递归确定性发现，输入是文件时按 UTF-8 path-list manifest 解析；默认 workers=1、queue=`2*workers`，可选 `-Config`、`-Workers`、`-QueueCapacity`、`-OutputImages` 与 `-Overwrite` |
| `batch-compare [-Config <path>]` | 固定 361 图、JSON-only、queue=8，以两个独立 Release 进程比较 workers=1/4；默认 FP32，可选 Round 2 U8S8 config；先要求逐图 detection JSON 字节与语义一致，再报告 throughput/PWS 差异，不设 speedup 门槛 |
| `demo` | 固定样本的 JSON/PNG smoke |
| `consistency` | 固定 manifest 的 Python ORT/C++ ORT 比较 |
| `benchmark` | 先运行 workflow 定义的 correctness 前置，再采集分段 latency/throughput/memory |
| `profile [image]` | 独立 profiling session；可选 config、prefix 和 run count，不进入正式 benchmark |
| `all` | clean build、完整 CTest、Demo、consistency、正式 benchmark 和冻结 30 图 batch 验收；用于单元收口而非每次编辑 |

`batch` 的 manifest 与目录由输入路径实际类型区分；wrapper 最终调用 production CLI 的显式 `--input-dir` 或 `--manifest`。`batch-compare` 的 fresh evidence 位于当前 TEMP build 的 `stage1_evidence/<run-id>/`。`profile` trace 含插桩开销，只用于 operator/node 诊断。`all` 是完整收口入口，不是普通改动的默认命令。

## 3. 当前已验证 Windows 环境

当前本机证据对应 Windows x86_64、C++17、MSVC/NMake、Release：

| 依赖 | 当前版本或位置 |
|---|---|
| Visual Studio C++ | MSVC 19.50.35721.0；`vswhere.exe` 自动发现 `VsDevCmd.bat`，也可用 `YOLO_DEFECT_VSDEVCMD` 覆盖 |
| CMake / CTest | 4.1.1-msvc1，Visual Studio 附带并由 `VsDevCmd.bat` 加入 PATH |
| ONNX Runtime C++ SDK | 1.19.2；实际目录由 ignored local config 的 `OrtRoot` 或 `ONNXRUNTIME_ROOT` 提供 |
| OpenCV C++ | 4.8.0；CMake/DLL 目录由 `OpenCvDir`、`OpenCvBin` 或对应环境变量提供 |
| Python Runtime reference | Python 3.9.25、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2；解释器由 local config 的 `PythonExe` 或 `YOLO_DEFECT_PYTHON` 提供 |
| Python PTQ tooling | ONNX 1.19.1、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2；解释器使用独立环境并通过命令参数或环境变量选择 |
| GoogleTest | v1.17.0；实际 source 目录由本机忽略配置提供，依赖版本由 CMake 固定 |

运行 `doctor` 查看本次进程实际解析到的路径，不要仅依赖本表。

## 4. 配置层次与优先级

机器依赖路径：

```text
显式命令参数 -> cpp_infer/.stage1.local.psd1 -> 进程环境变量 -> portable fallback
```

| 依赖 | 参数 | local key | 环境变量 |
|---|---|---|---|
| ORT SDK | `-OrtRoot` | `OrtRoot` | `ONNXRUNTIME_ROOT` |
| OpenCV CMake | `-OpenCvDir` | `OpenCvDir` | `OpenCV_DIR` |
| OpenCV DLL | `-OpenCvBin` | `OpenCvBin` | `YOLO_DEFECT_OPENCV_BIN` |
| Python | `-PythonExe` | `PythonExe` | `YOLO_DEFECT_PYTHON` |
| GoogleTest source | `-GTestSource` | `GTestSource` | `YOLO_DEFECT_GTEST_SOURCE` |

`VsDevCmd.bat` 按 `YOLO_DEFECT_VSDEVCMD`、`vswhere.exe` 自动发现解析。缺少本地 GoogleTest source 时，只有显式 `-AllowGTestDownload` 才允许 CMake 联网获取固定版本。

不同配置的相对路径基准不能混用：

- workflow/local 文件相对各自声明文件解析；
- `RuntimeConfig` 和 artifact 相对各自配置文件解析；
- CLI 图片和输出路径相对调用者当前工作目录解析；
- `ModelMetadata` 是 ORT 对真实模型的观察，不是配置来源。

## 5. 核心项目路径

| 路径 | 职责 |
|---|---|
| [`../cpp_infer/`](../cpp_infer/) | C++ Runtime、CLI、CMake、tests 和 tools |
| [`../cpp_infer/tools/stage1.cmd`](../cpp_infer/tools/stage1.cmd) | Windows 统一入口 |
| [`../cpp_infer/tools/stage1.defaults.psd1`](../cpp_infer/tools/stage1.defaults.psd1) | tracked、机器无关的 workflow 默认值 |
| [`../cpp_infer/tools/stage1.local.example.psd1`](../cpp_infer/tools/stage1.local.example.psd1) | `cpp_infer/.stage1.local.psd1` 模板 |
| [`../cpp_infer/configs/default_config.txt`](../cpp_infer/configs/default_config.txt) | 默认 FP32 RuntimeConfig |
| [`../cpp_infer/configs/int8_config.txt`](../cpp_infer/configs/int8_config.txt) | S2-01 Round 1 S8S8 RuntimeConfig（历史对照） |
| [`../cpp_infer/configs/int8_u8s8_config.txt`](../cpp_infer/configs/int8_u8s8_config.txt) | S2-01 Round 2 U8S8 RuntimeConfig |
| [`../cpp_infer/artifacts/yolov8_neu_det.artifact.txt`](../cpp_infer/artifacts/yolov8_neu_det.artifact.txt) | FP32 模型语义契约 |
| [`../cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt`](../cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt) | S2-01 Round 1 S8S8 模型语义契约（历史对照）；派生 ONNX 本体受 `.gitignore` 管理 |
| [`../cpp_infer/artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt`](../cpp_infer/artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt) | S2-01 Round 2 U8S8 模型语义契约 |
| [`../models/best.onnx`](../models/best.onnx) | 当前 FP32 ONNX |
| `../models/best.int8.qdq.u8s8.onnx` | 正式 Round 2 U8S8 本地 artifact；model id `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`，SHA-256 `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`；模型本体受 `.gitignore` 管理 |
| [`../cpp_infer/results/s2_03/int8_integration/`](../cpp_infer/results/s2_03/int8_integration/) | S2-01/S2-02/S2-03 正式 U8S8 融合证据根 |
| [`../data/images/val/crazing_241.jpg`](../data/images/val/crazing_241.jpg) | 固定单图 Demo/benchmark 样本 |

`CMakeCache.txt`、Makefiles、objects、binaries 和 staged DLL 都是可丢弃构建状态，不应手工维护或提交。

## 6. 仅用于诊断的底层构建链

只有需要审计 wrapper 或 CMake build graph 时才手工展开。先在 CMD 中初始化 x64 环境：

```bat
set "YOLO_DEFECT_VSDEVCMD=<path-to-VsDevCmd.bat>"
call "%YOLO_DEFECT_VSDEVCMD%" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
powershell.exe -NoProfile
```

随后在同一个 PowerShell 进程链中运行：

```powershell
$RepoRoot = (Resolve-Path '.').Path
$LocalSettings = Import-PowerShellDataFile `
  "$RepoRoot\cpp_infer\.stage1.local.psd1"
$BuildDir = Join-Path ([IO.Path]::GetTempPath()) `
  ('yolo_defect_stage1_manual_audit_' + [guid]::NewGuid().ToString('N'))

$env:ONNXRUNTIME_ROOT = $LocalSettings.OrtRoot
$env:PATH = "$($LocalSettings.OpenCvBin);$($LocalSettings.OrtRoot)\lib;$env:PATH"

cmake -S "$RepoRoot\cpp_infer" -B $BuildDir -G 'NMake Makefiles' `
  "-DOpenCV_DIR=$($LocalSettings.OpenCvDir)" `
  "-DONNXRUNTIME_ROOT=$($LocalSettings.OrtRoot)" `
  "-DPython3_EXECUTABLE=$($LocalSettings.PythonExe)" `
  "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$($LocalSettings.GTestSource)" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'Release build failed.' }

ctest --test-dir $BuildDir --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'CTest failed.' }
```

这段命令只解释 configure/build/test 关系，不替代 `stage1.cmd` 的参数检查、DLL staging 和产品动作。

## 7. TEMP、输出与故障分层

- 默认构建目录是 `%TEMP%\yolo_defect_stage1_manual_release`；自定义 `-BuildDir` 必须满足 `stage1.ps1` 的受保护 TEMP 边界检查。
- `demo`、`consistency`、`benchmark`、`batch-compare`、`profile` 和 `all` 默认写入构建目录下的新 evidence 子目录；没有显式输出目录的 `detect`/`batch` 默认写入 Git 忽略的 manual result 目录。
- 不使用 `cpp_infer/build` 中的旧二进制判断当前源码；需要复现时使用当前 TEMP build。
- 环境失败：命令、SDK、package、DLL、架构或 provider 不可用；先运行 `doctor`。
- configure/build 失败：CMake discovery、编译或链接错误；查看第一条决定性错误。
- test 失败：编译已成功，但某个行为或集成预期不满足；运行相关 case，不先扩大到完整 gate。
- product runtime 失败：沿 config/artifact/metadata、preprocess、`Session::Run`、postprocess、output 边界定位。
- benchmark/profile/batch-compare 结果只适用于记录的机器与协议；PWS/RSS 是进程高水位，不是模型独占内存。Windows PWS 与 Linux RSS 只能各自在同平台内比较。
- 工作树可能已有未提交修改；构建或清理前先检查 `git status`，不要覆盖或删除无关文件。

## 8. S2-02 Gate A：WSL2/Linux x86_64 环境与入口

### 8.1 当前已验证环境

Gate A 当前只代表 **WSL2/Linux x86_64**，不是原生 Linux 实机、AArch64 板卡或 QEMU：

| 依赖 | 当前版本或位置 |
|---|---|
| 系统 | WSL2 Ubuntu `24.04.4 LTS`，x86_64，kernel `6.18.33.2-microsoft-standard-WSL2` |
| GCC / G++ | `13.3.0` |
| CMake / CTest | `3.28.3` |
| Ninja | `1.11.1` |
| pkg-config | `1.8.1` |
| OpenCV C++ | `4.6.0`；Ubuntu `/usr` 下的 distro headers/libraries，由 `pkg-config opencv4` 与 CMake package 解析 |
| ONNX Runtime C++ SDK | `1.19.2`：默认 `$HOME/.local/opt/onnxruntime-linux-x64-1.19.2`，可由 `ONNXRUNTIME_ROOT` 覆盖 |
| Python reference | Python `3.12.3`：默认 `$HOME/.venvs/yolo-defect-gate-a/bin/python`；ORT `1.19.2`、OpenCV Python `4.10.0`、NumPy `2.0.2`，实际 provider 含 `CPUExecutionProvider` |
| GoogleTest | Ubuntu distro source `/usr/src/googletest`，版本 `1.14.0` |

Linux C++ SDK 来自 Microsoft 官方发布包
[`onnxruntime-linux-x64-1.19.2.tgz`](https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-1.19.2.tgz)。Python wheel 仍不能替代 C++ SDK 的 headers 与 `libonnxruntime.so`。

### 8.2 环境准备与重新进入

Ubuntu 侧安装过的最小 apt 包为：

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config libopencv-dev \
  python3-venv python3-pip libgtest-dev gdb
```

Python reference 环境的固定依赖为：

```bash
python3 -m venv "$HOME/.venvs/yolo-defect-gate-a"
"$HOME/.venvs/yolo-defect-gate-a/bin/python" -m pip install \
  onnxruntime==1.19.2 opencv-python-headless==4.10.0.84 numpy==2.0.2
```

每次新 WSL shell 从仓库根目录进入时设置：

```bash
# 从仓库根目录执行
export ONNXRUNTIME_ROOT="$HOME/.local/opt/onnxruntime-linux-x64-1.19.2"
export YOLO_DEFECT_PYTHON="$HOME/.venvs/yolo-defect-gate-a/bin/python"
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest
bash cpp_infer/tools/stage1.sh doctor
```

`stage1.sh` 会在通过 SDK 检查后为当前进程补充 `LD_LIBRARY_PATH`；Linux build 还把该 SDK 的 `lib` 写入 build RPATH。需要直接运行其他自编译程序时，可显式设置：

```bash
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

### 8.3 Linux 统一入口

查看帮助和只读检查：

```bash
bash cpp_infer/tools/stage1.sh help
bash cpp_infer/tools/stage1.sh doctor
```

| Action | 行为 |
|---|---|
| `help` | 显示参数，不构建 |
| `doctor` | 只读检查 WSL/Linux x86_64、工具链、ORT ELF、OpenCV、Python 与 GTest source |
| `build` | 没有构建树时 clean configure，否则增量 Ninja Release 构建；随后检查 ELF/`ldd` |
| `clean-build` | 只清理受保护的 `/tmp/.../yolo_defect_stage1_*` 构建目录，再完整构建 |
| `test` | 构建当前源码并运行完整 CTest |
| `detect <image> [output-dir]` | 复用 `DetectorPipeline` 运行任意单图；可选 `--config` 与 `--overwrite` |
| `batch <input> [output-dir]` | 输入是目录或 UTF-8 path-list manifest；默认 workers=1、queue=`2*workers`，可选 `--config`、`--workers`、`--queue-capacity`、`--output-images` 与 `--overwrite` |
| `batch-compare [--config <path>]` | 固定 361 图、JSON-only、queue=8，以两个独立 Release 进程比较 workers=1/4；默认 FP32，可选 Round 2 U8S8 config；逐图结果完全一致后报告 throughput/peak-RSS 差异，不设 speedup 门槛 |
| `demo` | 验证固定 `crazing_241` 的 JSON/PNG 与 3 个 detections |
| `consistency` | 固定 30 图 Python ORT/C++ ORT 比较 |
| `benchmark` | 先跑 consistency，再运行分段 benchmark；默认 warmup `10`、repeat `100`，可覆盖 |
| `all` | clean build、完整 CTest、Demo、consistency、默认 benchmark 与冻结 30 图 batch 验收；只用于完整 gate |

常见动作：

```bash
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh batch data/images/val --workers 4 --queue-capacity 8
bash cpp_infer/tools/stage1.sh batch cpp_infer/tests/fixtures/s2_03_consistency_manifest.txt --workers 2
bash cpp_infer/tools/stage1.sh batch-compare
bash cpp_infer/tools/stage1.sh batch-compare --config cpp_infer/configs/int8_u8s8_config.txt
bash cpp_infer/tools/stage1.sh demo
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 2
```

默认 run JSON 写到受保护 build tree 下的临时目录。需要把某次 Gate 结果直接留在仓库时，显式选择一个尚无 `demo`、`consistency`、`benchmark`、`batch`、`batch_workers_1`、`batch_workers_4` 或 `batch_comparison.json` 的专用目录；脚本会拒绝覆盖旧 run。下面先保留 S2-02 的历史示例，再给出 S2-03 正式 comparison 的当前入口；复跑必须换新目录名：

```bash
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_02/linux_x86_64"
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 2
unset YOLO_DEFECT_RUN_DIR
```

```bash
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_03/linux_x86_64/performance_rerun_20260830_01"
bash cpp_infer/tools/stage1.sh batch-compare
unset YOLO_DEFECT_RUN_DIR
```

正式 U8S8 比较复跑同样要使用 fresh 目录：

```bash
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_03/int8_integration/linux_x86_64/rerun_YYYYMMDD_HHMMSS"
bash cpp_infer/tools/stage1.sh batch-compare \
  --config cpp_infer/configs/int8_u8s8_config.txt
unset YOLO_DEFECT_RUN_DIR
```

### 8.4 无 ORT/OpenCV 的 core-only smoke

`YOLO_DEFECT_CORE_ONLY=ON` 只构建 dependency-free `project_core` 与 portability smoke，跳过 OpenCV、ORT、Python 和 GoogleTest discovery。它用于验证纯 C++17 的 YOLO decode、class-agnostic NMS 与坐标恢复边界，也为 Gate B 提供最小交叉编译入口；它不等于完整 Runtime inference。

```bash
cmake -S cpp_infer -B /tmp/yolo_defect_s2_02_core_only -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DYOLO_DEFECT_CORE_ONLY=ON
cmake --build /tmp/yolo_defect_s2_02_core_only --parallel
ctest --test-dir /tmp/yolo_defect_s2_02_core_only --output-on-failure
ldd /tmp/yolo_defect_s2_02_core_only/bin/yolo_defect_project_core_smoke
```

当前 Linux 结果是 `1/1` smoke 通过，`ldd` 不含 ONNX Runtime 或 OpenCV 依赖。

### 8.5 WSL 操作注意事项

- 从非交互式 `wsl.exe ... bash -lc ...` 调用 `sudo` 时，密码提示可能没有可用终端而失败或挂起。安装 apt 包应在交互式 WSL shell 中运行，或先在该 shell 执行 `sudo -v`；不要把密码写入脚本。
- 从 PowerShell 拼接 `wsl.exe ... bash -lc ...` 时，Bash 的 `$name`/`${name}` 可能先被宿主 shell 展开。复杂命令优先进入 WSL shell 后运行仓库脚本，或直接传固定 WSL 路径；不要依赖多层 shell 中未验证的变量引用。
- 默认构建和 fresh result JSON 位于 `/tmp/yolo_defect_stage1_linux_release`。WSL 会话重启、系统清理或手工删除后它们可能消失；`/tmp` 不是长期结果库。收口运行应设置 `YOLO_DEFECT_RUN_DIR`，直接写入仓库内命名目录。
- Linux benchmark 的 peak RSS 与 Windows Peak Working Set 是不同 OS 指标，不能直接当成同一测量口径比较。
- Gate A 的实测环境是 WSL2；AArch64 cross compile 与 QEMU 属于 Gate B，不能由本节命令或结果推导。Gate B 当前完成状态与命令见下方第 9 节。

## 9. S2-02 Gate B：Linux x86_64 host → Linux AArch64 target

### 9.1 实测工具链与依赖位置

| 项目 | Gate B 实测值 / 默认位置 |
|---|---|
| Host / target | WSL2 Ubuntu `24.04.4 LTS` x86_64 / Linux AArch64 |
| Cross compiler | `aarch64-linux-gnu-g++` `13.3.0`，target triple `aarch64-linux-gnu` |
| Target binutils | `2.42` |
| QEMU user-mode | `qemu-aarch64` `8.2.2` |
| Target loader prefix | `/usr/aarch64-linux-gnu` |
| CMake toolchain | `cpp_infer/cmake/toolchains/linux-aarch64-gnu.cmake` |
| ARM64 ORT `1.19.2` | `$HOME/.local/opt/yolo-defect-aarch64/onnxruntime-linux-aarch64-1.19.2` |
| ARM64 OpenCV `4.6.0` private sysroot | `$HOME/.local/opt/yolo-defect-aarch64/ubuntu-noble-opencv-4.6.0` |
| Core / full build | `/tmp/yolo_defect_stage2_aarch64_core` / `/tmp/yolo_defect_stage2_aarch64_full` |
| Deployment layout | full build 下的 `deploy/bin`（CLI）与 `deploy/lib`（ORT）；Ubuntu target libraries 保留在 private sysroot |

官方 ARM64 ORT 下载入口为 [onnxruntime-linux-aarch64-1.19.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz)。OpenCV 使用 Ubuntu Noble 的 ARM64 预编译包，没有从源码构建。

### 9.2 一次性 host 准备

先安装 host 上运行的交叉工具和 QEMU：

```bash
sudo apt update
sudo apt install -y \
  gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
  libc6-dev-arm64-cross binutils-aarch64-linux-gnu qemu-user
```

Ubuntu 的 amd64 archive 与 ARM64 ports 使用不同镜像。启用 multiarch 时，给现有 `/etc/apt/sources.list.d/ubuntu.sources` 的每个 deb822 stanza 增加 `Architectures: amd64`，再增加：

```text
# /etc/apt/sources.list.d/ubuntu-ports-arm64.sources
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble-security
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

然后执行：

```bash
sudo dpkg --add-architecture arm64
sudo apt update
bash cpp_infer/tools/bootstrap_aarch64_deps.sh fetch
```

bootstrap 只执行 `apt-get download` 和 `dpkg-deb -x`，不会安装 ARM64 OpenCV 包。原因是直接安装 `libopencv-dev:arm64` 或 `libopencv-imgcodecs*:arm64` 会要求移除现有 amd64 OpenCV dev chain；私有 sysroot 保持 host/target 隔离。

### 9.3 Gate B 统一入口

```bash
bash cpp_infer/tools/stage2_aarch64.sh help
bash cpp_infer/tools/stage2_aarch64.sh doctor
bash cpp_infer/tools/stage2_aarch64.sh clean-build
bash cpp_infer/tools/stage2_aarch64.sh inspect
bash cpp_infer/tools/stage2_aarch64.sh smoke
bash cpp_infer/tools/stage2_aarch64.sh infer
bash cpp_infer/tools/stage2_aarch64.sh batch
bash cpp_infer/tools/stage2_aarch64.sh all
```

默认使用 FP32 `configs/default_config.txt`。使用正式 U8S8 时：

```bash
export YOLO_DEFECT_AARCH64_CONFIG="$PWD/cpp_infer/configs/int8_u8s8_config.txt"
bash cpp_infer/tools/stage2_aarch64.sh all
unset YOLO_DEFECT_AARCH64_CONFIG
```

| Action | 行为 |
|---|---|
| `doctor` | 检查 x86_64 host tools、AArch64 compiler/loader、ARM64 ORT/OpenCV ELF |
| `build` / `clean-build` | Ninja Release 交叉编译 project-core tree 和完整 Runtime/CLI tree；clean 仅允许两个固定 `/tmp` 边界 |
| `inspect` | `file/readelf` 检查 core smoke、Runtime object、CLI、ORT；用 ARM64 loader 列出动态依赖并逐个拒绝 x86_64 library |
| `smoke` | QEMU 实际运行 project-core decode/NMS/坐标恢复、CLI startup/help、config/artifact 和两条错误路径 |
| `infer` | QEMU 下使用选定 RuntimeConfig 运行固定图片 → ARM64 OpenCV/ORT CPU → Detection JSON，并按该 config 的 model id/SHA 调用 validator |
| `batch` | QEMU 下使用选定 RuntimeConfig 验收 2 图目录 workers=1、同集合 manifest workers=2/queue=1、两入口逐图 JSON 一致，以及损坏 JPEG 精确 `2 succeeded + 1 failed`/退出码 2；严格校验三份 `BatchSummary` |
| `all` | doctor → clean-build → inspect → smoke → infer → batch；不包含 benchmark |

可覆盖的 machine-local 路径和运行选择都使用环境变量，不写死到源码：`YOLO_DEFECT_AARCH64_CONFIG`、`YOLO_DEFECT_AARCH64_DEPS_ROOT`、`YOLO_DEFECT_AARCH64_ORT_ROOT`、`YOLO_DEFECT_AARCH64_SYSROOT`、`YOLO_DEFECT_AARCH64_DEB_CACHE_ROOT`、`YOLO_DEFECT_AARCH64_LOADER_PREFIX`、`YOLO_DEFECT_AARCH64_CORE_BUILD_DIR`、`YOLO_DEFECT_AARCH64_FULL_BUILD_DIR`、`YOLO_DEFECT_AARCH64_RESULT_DIR`、`YOLO_DEFECT_AARCH64_BATCH_RESULT_DIR` 与 `YOLO_DEFECT_AARCH64_BATCH_RUN_ID`。`YOLO_DEFECT_AARCH64_CONFIG` 省略时继续使用默认 FP32；batch run id 必须是一个安全 path component，且目标 run root 必须不存在；脚本用 fresh-root 规则防止旧输出误满足验收。

两个默认 build tree 位于 `/tmp`。如果从 PowerShell 分多次启动 `wsl.exe`，WSL 发行版可能在两次命令之间停止并清空 `/tmp`；需要拆开运行 action 时，应留在同一个交互式 WSL shell 中。完整复现优先单次运行 `stage2_aarch64.sh all`。

### 9.4 本次实测结果与边界

- S2-02 最终收口在提交 `436ab4b` 上重新运行三平台关键门：Linux x86_64 build/test/demo/consistency、AArch64 clean cross-build/inspect/QEMU/full inference、Windows clean build/test/demo 均通过；完整教学记录见 [`details/s2_02_closure.md`](details/s2_02_closure.md)。
- core smoke、Runtime object、production CLI、ORT 均为 AArch64 ELF；CLI interpreter 为 `/lib/ld-linux-aarch64.so.1`。
- ARM64 loader 实际解析 138 个 target `.so`，`not found = 0`，逐个 `file/readelf` 均为 AArch64。
- QEMU 实际通过 startup、`--help`、config/artifact、两条 negative contract 和 decode/NMS/坐标恢复 synthetic smoke。
- 固定图完整 ARM64 ORT CPU 推理实际执行，得到 3 个 detections，JSON validator 通过。
- 原生 WSL2/Linux x86_64 clean Release 与 `119/119` CTest 回归通过。
- 没有执行 QEMU benchmark、功耗测试、AArch64 全量 GTest/CTest、Docker multi-arch 或真实板卡；QEMU 不能写成 ARM 板卡性能证据。

完整解释与状态表见 [`details/s2_02_gate_b_closure.md`](details/s2_02_gate_b_closure.md)。

## 10. S2-03：目录/Manifest 有界并发与正式证据

### 10.1 Production CLI 契约

Windows 与 Linux 的同一个 `yolo_defect_cpp` CLI 使用：

```text
--config <config> --batch
  (--input-dir <directory> | --manifest <path-list>)
  --output-dir <directory> --batch-summary <file>
  [--workers <1..64>] [--queue-capacity <1..4096>]
  [--output-images] [--overwrite]
```

默认 `workers=1`、`queue_capacity=2*workers`；任务少于 requested workers
时 effective workers 是 `min(requested, discovered)`，两者都会写入
summary。目录递归发现普通 `.bmp/.jpeg/.jpg/.png/.tif/.tiff/.webp`，不
跟随 symlink，并按 UTF-8 generic relative path 排序。UTF-8 manifest
允许 BOM、LF/CRLF、空行和首个非空字符为 `#` 的注释；有效行相对
manifest 所在目录解析并保留声明顺序。manifest 的绝对路径、缺失或
不支持图片、重复 canonical 输入，以及空目录/遍历错误都在 session
和逐图处理开始前失败。

每个成功任务必写 `items/<六位序号>.detections.json`，其内容继续使用
现有单图 detection schema；`--output-images` 再写对应 PNG。summary
路径必须显式给出。summary 已存在且未传 `--overwrite` 时整次调用在
启动前失败；逐图目标已存在时按 writer 语义只让该任务失败，其他
任务继续。输出计划不能覆盖 source image、config、artifact、model
或 manifest；目录输入还禁止 output root 位于 input root 内部。

公开状态和进程退出码是：

| `BatchSummary.status` | 退出码 | 含义 |
|---|---:|---|
| `succeeded` | 0 | 全部任务成功 |
| `partial_failure` | 2 | 有逐图失败、无 cancelled |
| `cancelled` | 130 | cooperative signal/stop 已被观测；可以有 0 个或多个未开始任务 |
| `fatal` | 1 | session/线程等基础设施失败 |

计数始终满足 `discovered = succeeded + failed + cancelled` 与
`started = succeeded + failed`。SIGINT/SIGTERM（Windows 还覆盖 console
break）只在 handler 中设置安全标志；普通线程发起 stop，拒绝新任务、
取消排队/未开始任务，允许正在执行的同步 ORT 调用结束并 join 全部
worker。`BatchSummary.cooperative_stop_requested` 独立于逐图计数，
避免“所有任务已经开始”时把中断误报为成功。队列只持有 task index；图片、tensor 与 ORT output 只存在于
当前 worker，且每个 worker 独占一个 `DetectorPipeline/Ort::Session`。

### 10.2 Windows x86_64 与 WSL2/Linux x86_64 正式比较

两平台都以 FP32 `CPUExecutionProvider`、Release、ORT sequential、
intra/inter-op `1/1`、361 张 `data/images/val`、JSON-only、queue=8 分别
启动 workers=1 与 workers=4 独立进程。比较工具先确认 361/361 个任务
顺序一致、逐图 detection JSON 字节与语义完全一致，再报告性能；没有
“并发必须更快”的门槛。

| 平台 / 指标 | workers=1 | workers=4 | 变化 |
|---|---:|---:|---:|
| Windows processing wall | 57,433.263 ms | 20,219.646 ms | -64.79% |
| Windows throughput | 6.285556 img/s | 17.853923 img/s | 2.840468x |
| Windows Peak Working Set | 151.805 MiB | 505.086 MiB | +353.281 MiB |
| WSL2/Linux processing wall | 44,492.069 ms | 17,907.115 ms | -59.75% |
| WSL2/Linux throughput | 8.113806 img/s | 20.159584 img/s | 2.484603x |
| WSL2/Linux peak RSS | 205.766 MiB | 588.227 MiB | +382.461 MiB |

四次运行的 queue peak depth 都是 8，从未超过 capacity；workers=1/4 的
producer wait count 分别是 Windows `353/352`、WSL2/Linux `353/349`，因此
backpressure 是实际触发并被记录的行为，不只是配置声明。

WSL2 正式两次运行把同一份 361 图、config、artifact 和 model 复制到同一个
WSL 原生 ext4 临时工作区，并保持 JSON-only 输出策略一致；完成后只把 summary、
逐图 JSON 与 comparison 复制回仓库。它仍然只是 WSL2/Linux 同平台证据，不能与
Windows 数字横向比较。

Windows 记录位于
[`../cpp_infer/results/s2_03/windows_x86_64/`](../cpp_infer/results/s2_03/windows_x86_64/)，
其中 `comparison.json` 绑定 workers=1/4 summary 与 361 项一致性。WSL2
记录位于
[`../cpp_infer/results/s2_03/linux_x86_64/performance/`](../cpp_infer/results/s2_03/linux_x86_64/performance/)，
对应文件是 `batch_workers_1/batch_summary.json`、
`batch_workers_4/batch_summary.json` 与 `batch_comparison.json`。

S2-03 原 FP32 clean Release 收口在 Windows x86_64 与 WSL2/Linux x86_64 均为
`156/156` CTest 通过；当前 U8S8 融合收口已更新为两平台 `157/157`。Windows 中两个需要本地创建 symlink/reparse 的
GTest case 因账号权限显示 skip，对应 path-safety case 已在 Linux
执行。Linux 同轮还检查 11 个 ELF 的 `ldd` 无 `not found`。这些是当前
S2-03 收口事实；上文 S2-02 的 `119/119` 是历史里程碑，不回写。

复现入口：

```powershell
cpp_infer\tools\stage1.cmd all
cpp_infer\tools\stage1.cmd batch-compare
```

```bash
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_03/linux_x86_64/performance_rerun_20260830_01"
bash cpp_infer/tools/stage1.sh all
unset YOLO_DEFECT_RUN_DIR

export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_03/linux_x86_64/comparison_rerun_20260830_01"
bash cpp_infer/tools/stage1.sh batch-compare
unset YOLO_DEFECT_RUN_DIR
```

`all` 做完整回归和冻结 30 图 batch 验收，但不会自动运行 361 图
workers=1/4 正式比较；后者必须单独调用 `batch-compare`。Windows PWS 与
Linux RSS 的平台语义不同，只能各自在本平台比较，不能横向比较两行。
复跑示例中的 `20260830_01` 是 run-id 占位，目标存在时必须换新值。

### 10.3 Linux AArch64/QEMU 功能证据

```bash
export YOLO_DEFECT_AARCH64_RESULT_DIR="$PWD/cpp_infer/results/s2_03/linux_aarch64_qemu/regression_rerun_20260830_01"
export YOLO_DEFECT_AARCH64_BATCH_RESULT_DIR="$PWD/cpp_infer/results/s2_03/linux_aarch64_qemu/batch"
export YOLO_DEFECT_AARCH64_BATCH_RUN_ID="rerun_20260830_01"
bash cpp_infer/tools/stage2_aarch64.sh all
unset YOLO_DEFECT_AARCH64_RESULT_DIR
unset YOLO_DEFECT_AARCH64_BATCH_RESULT_DIR
unset YOLO_DEFECT_AARCH64_BATCH_RUN_ID
```

复跑同样要把 `20260830_01` 换成未使用的 run id；batch action 会拒绝
已存在的 run root。

当前结果证明完整 Runtime/CLI 交叉构建、AArch64 ELF/loader 边界、既有
固定单图 3 detections，以及目录/manifest/部分失败 batch 行为。正式
batch 记录见
[`../cpp_infer/results/s2_03/linux_aarch64_qemu/final_20260830_r2/`](../cpp_infer/results/s2_03/linux_aarch64_qemu/final_20260830_r2/)；
目录 2/2 成功、manifest 2/2 成功且两入口逐图 JSON 完全一致，损坏
JPEG 精确 2 成功 + 1 失败、exit 2。三份 summary 同时记录编译 target
`aarch64`、运行 kernel `x86_64`、execution context
`qemu_user_mode_on_x86_64_host` 与 `memory.publishable=false`。

QEMU user-mode 不是开发板、原生 ARM 或部署性能环境。本项目不发布、
比较或解释该运行中的 latency、throughput、RSS、worker speedup、功耗、
温度或稳定性数字；AArch64 S2-03 结论仅限构建与功能可移植性。

## 11. S2-01/S2-02/S2-03 正式 U8S8 融合收口

本轮冻结对象是 `configs/int8_u8s8_config.txt` 选中的
`yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`，SHA-256 为
`9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`。
U8S8 模型外部 I/O 仍为 float32，因此产品 Runtime、`DetectorPipeline`、
后处理、`BatchRunner`、有界队列和 `BatchSummary` 均直接复用。默认
config 仍为 FP32。

实测结果：

- Windows x86_64 最终 Release 为 `157/157` CTest；两个依赖
  symlink/reparse 权限的 GTest case 在当前账号下 skip，对应 Linux
  用例已执行通过；U8S8 固定图实际得到 3 detections。
- WSL2/Linux x86_64 全量 `157/157` CTest 通过，U8S8 固定图为
  3 detections；30 图 manifest 在 workers=2/queue=4 下 `30/30`成功，
  queue peak=4、producer waits=25。
- 361 图 U8S8 CPU、JSON-only、queue=8 比较中，worker=1 为
  `4.591151 img/s` / peak RSS `192.933594 MiB` / waits=353，worker=4 为
  `15.903088 img/s` / `556.882812 MiB` / waits=350，吞吐比
  `3.463857`，361 份逐图 JSON 字节与语义完全一致。本轮直接
  在仓库所在的 `/mnt/d` DrvFs 运行，数字只用于此次 WSL2/Linux
  同协议内 workers=1/4 比较，不与旧 WSL 原生 ext4 FP32 数字直接对比。
- AArch64 clean cross-build、ELF 和 loader 检查通过。QEMU user-mode
  下 U8S8 单图为 3 detections；目录 workers=1 和 manifest workers=2
  各 `2/2`，两入口逐图 JSON 字节/语义一致；损坏 JPEG 精确得到
  `2 成功 + 1 失败`、exit 2，partial-failure queue=1 且 waits=1。
  三份 summary 的 `memory.publishable=false`；其 latency、throughput 和 RSS
  字段不得当作性能或原生 ARM 证据。
- 可选 config 改动后，默认 FP32 Linux Demo 和 AArch64/QEMU `all`
  回归通过。

S2-01 advisory 结果继续如实保留：agreement precision
`0.938462 < 0.95`，mAP50 drop `0.010356 > 0.01`；本轮不宣称
strict 质量门全部通过。

证据根为
[`../cpp_infer/results/s2_03/int8_integration/`](../cpp_infer/results/s2_03/int8_integration/)：

- `windows_x86_64/final_20260831_single/`：Windows U8S8 单图 JSON/PNG；
- `linux_x86_64/final_20260831/`：Linux U8S8 单图、30 图 manifest、
  workers=1/4 summaries 与 comparison；
- `linux_aarch64_qemu/final_20260831_single/`：AArch64 ELF/loader/smoke 与
  U8S8 单图；
- `linux_aarch64_qemu/final_20260831/`：U8S8 目录、manifest、partial-failure
  summaries 与 acceptance log；
- `linux_x86_64/fp32_regression_20260831/` 与
  `linux_aarch64_qemu/fp32_regression_20260831*`：默认 FP32 回归。
