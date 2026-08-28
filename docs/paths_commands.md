# 路径、工具链与环境诊断

本文是本项目**当前工具链、机器路径、构建入口和环境踩坑的唯一操作事实源**。`AGENTS.md` 不复制这些细节；只有任务需要查找工具链路径、依赖版本或位置、环境入口命令或已知环境踩坑时，仓库 Skill `yolo-defect-dev` 才按需读取本文的相关部分。任务仅仅涉及构建、运行、测试、benchmark 或 profiling，不构成加载该 Skill 的理由。历史 closure 中的版本和路径是当时证据快照，不替代本文。

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
| `demo` | 固定样本的 JSON/PNG smoke |
| `consistency` | 固定 manifest 的 Python ORT/C++ ORT 比较 |
| `benchmark` | 先运行 workflow 定义的 correctness 前置，再采集分段 latency/throughput/memory |
| `profile [image]` | 独立 profiling session；可选 config、prefix 和 run count，不进入正式 benchmark |
| `all` | clean build、完整 CTest、Demo、consistency 和正式 benchmark；用于单元收口而非每次编辑 |

`profile` trace 含插桩开销，只用于 operator/node 诊断。`all` 是完整收口入口，不是普通改动的默认命令。

## 3. 当前已验证 Windows 环境

当前本机证据对应 Windows x86_64、C++17、MSVC/NMake、Release：

| 依赖 | 当前版本或位置 |
|---|---|
| Visual Studio C++ | MSVC 19.50.35721.0；`vswhere.exe` 自动发现；当前 `VsDevCmd.bat`：`D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat` |
| CMake / CTest | 4.1.1-msvc1，Visual Studio 附带并由 `VsDevCmd.bat` 加入 PATH |
| ONNX Runtime C++ SDK | 1.19.2：`D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| OpenCV C++ | 4.8.0：`D:\01_Base\Tools\opencv\build\x64\vc16\lib`，DLL 位于同级 `bin` |
| Python Runtime reference | `C:\Users\Everbreath\.conda\envs\TestBase\python.exe`；Python 3.9.25、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2；供 `stage1.cmd` 的 consistency/profile 前置检查 |
| Python PTQ tooling | `C:\Users\Everbreath\.conda\envs\yolo_defect\python.exe`；ONNX 1.19.1、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2；正式运行 `quantize_s2_01.py` 和 S2-01 Python 工具 |
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
| [`../data/images/val/crazing_241.jpg`](../data/images/val/crazing_241.jpg) | 固定单图 Demo/benchmark 样本 |

`CMakeCache.txt`、Makefiles、objects、binaries 和 staged DLL 都是可丢弃构建状态，不应手工维护或提交。

## 6. 仅用于诊断的底层构建链

只有需要审计 wrapper 或 CMake build graph 时才手工展开。先在 CMD 中初始化 x64 环境：

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
powershell.exe -NoProfile
```

随后在同一个 PowerShell 进程链中运行：

```powershell
$RepoRoot = 'D:\01_Base\CodingSpace\yolo_defect'
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
- `demo`、`consistency`、`benchmark`、`profile` 和 `all` 默认写入构建目录下的新 evidence 子目录；`detect` 默认写入 Git 忽略的 manual result 目录。
- 不使用 `cpp_infer/build` 中的旧二进制判断当前源码；需要复现时使用当前 TEMP build。
- 环境失败：命令、SDK、package、DLL、架构或 provider 不可用；先运行 `doctor`。
- configure/build 失败：CMake discovery、编译或链接错误；查看第一条决定性错误。
- test 失败：编译已成功，但某个行为或集成预期不满足；运行相关 case，不先扩大到完整 gate。
- product runtime 失败：沿 config/artifact/metadata、preprocess、`Session::Run`、postprocess、output 边界定位。
- benchmark/profile 结果只适用于记录的机器与协议；PWS 是进程高水位，不是模型独占内存。
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
| ONNX Runtime C++ SDK | `1.19.2`：`/home/everbreath/.local/opt/onnxruntime-linux-x64-1.19.2` |
| Python reference | Python `3.12.3`：`/home/everbreath/.venvs/yolo-defect-gate-a/bin/python`；ORT `1.19.2`、OpenCV Python `4.10.0`、NumPy `2.0.2`，实际 provider 含 `CPUExecutionProvider` |
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
python3 -m venv /home/everbreath/.venvs/yolo-defect-gate-a
/home/everbreath/.venvs/yolo-defect-gate-a/bin/python -m pip install \
  onnxruntime==1.19.2 opencv-python-headless==4.10.0.84 numpy==2.0.2
```

每次新 WSL shell 从仓库根目录进入时设置：

```bash
cd /mnt/d/01_Base/CodingSpace/yolo_defect
export ONNXRUNTIME_ROOT=/home/everbreath/.local/opt/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON=/home/everbreath/.venvs/yolo-defect-gate-a/bin/python
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
| `demo` | 验证固定 `crazing_241` 的 JSON/PNG 与 3 个 detections |
| `consistency` | 固定 30 图 Python ORT/C++ ORT 比较 |
| `benchmark` | 先跑 consistency，再运行分段 benchmark；默认 warmup `10`、repeat `100`，可覆盖 |
| `all` | clean build、完整 CTest、Demo、consistency 与默认 benchmark；只用于完整 gate |

常见动作：

```bash
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh demo
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 2
```

默认 run JSON 写到受保护 build tree 下的临时目录。需要把某次 Gate 结果直接留在仓库时，显式选择一个尚无 `demo`、`consistency`、`benchmark` 子目录的专用目录；脚本会拒绝覆盖旧 run，仍只调用现有 Demo、consistency 和 benchmark 逻辑，不组装新的 evidence schema。下面是本次首次收口时使用的目录，复跑请换一个新目录名：

```bash
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_02/linux_x86_64"
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 2
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
- Gate A 的实测环境是 WSL2；AArch64 cross compile 与 QEMU 属于 Gate B，不能由本节命令或结果推导为已完成。

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
bash cpp_infer/tools/stage2_aarch64.sh all
```

| Action | 行为 |
|---|---|
| `doctor` | 检查 x86_64 host tools、AArch64 compiler/loader、ARM64 ORT/OpenCV ELF |
| `build` / `clean-build` | Ninja Release 交叉编译 project-core tree 和完整 Runtime/CLI tree；clean 仅允许两个固定 `/tmp` 边界 |
| `inspect` | `file/readelf` 检查 core smoke、Runtime object、CLI、ORT；用 ARM64 loader 列出动态依赖并逐个拒绝 x86_64 library |
| `smoke` | QEMU 实际运行 project-core decode/NMS/坐标恢复、CLI startup/help、config/artifact 和两条错误路径 |
| `infer` | QEMU 下运行固定图片 → ARM64 OpenCV/ORT CPU → Detection JSON，并调用既有 validator |
| `all` | doctor → clean-build → inspect → smoke → infer；不包含 benchmark |

可覆盖的 machine-local 路径都使用环境变量，不写死到源码：`YOLO_DEFECT_AARCH64_DEPS_ROOT`、`YOLO_DEFECT_AARCH64_ORT_ROOT`、`YOLO_DEFECT_AARCH64_SYSROOT`、`YOLO_DEFECT_AARCH64_DEB_CACHE_ROOT`、`YOLO_DEFECT_AARCH64_LOADER_PREFIX`、两个 build dir 与 result dir。

两个默认 build tree 位于 `/tmp`。如果从 PowerShell 分多次启动 `wsl.exe`，WSL 发行版可能在两次命令之间停止并清空 `/tmp`；需要拆开运行 action 时，应留在同一个交互式 WSL shell 中。完整复现优先单次运行 `stage2_aarch64.sh all`。

### 9.4 本次实测结果与边界

- core smoke、Runtime object、production CLI、ORT 均为 AArch64 ELF；CLI interpreter 为 `/lib/ld-linux-aarch64.so.1`。
- ARM64 loader 实际解析 138 个 target `.so`，`not found = 0`，逐个 `file/readelf` 均为 AArch64。
- QEMU 实际通过 startup、`--help`、config/artifact、两条 negative contract 和 decode/NMS/坐标恢复 synthetic smoke。
- 固定图完整 ARM64 ORT CPU 推理实际执行，得到 3 个 detections，JSON validator 通过。
- 原生 WSL2/Linux x86_64 clean Release 与 `119/119` CTest 回归通过。
- 没有执行 QEMU benchmark、功耗测试、AArch64 全量 GTest/CTest、Docker multi-arch 或真实板卡；QEMU 不能写成 ARM 板卡性能证据。

完整解释与状态表见 [`details/s2_02_gate_b_closure.md`](details/s2_02_gate_b_closure.md)。
