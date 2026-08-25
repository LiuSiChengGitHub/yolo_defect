# 路径、环境与复现命令

本文集中保存工业视觉边缘 AI Runtime 与 C++ 工程化系统的核心路径、Windows 环境和大阶段一复现入口。项目状态与路线以根目录双语 README 和 [`Proj1_S2.md`](./Proj1_S2.md) 为准；大阶段一的结果口径见 [`details/stage1_closure.md`](./details/stage1_closure.md)。

## 1. 核心路径

| 路径 | 作用 |
|---|---|
| [`../cpp_infer/`](../cpp_infer/) | C++17 Runtime、CLI、CMake、测试、工具和已跟踪证据 |
| [`../cpp_infer/CMakeLists.txt`](../cpp_infer/CMakeLists.txt) | Runtime library、CLI、GTest/CTest 与依赖链接关系 |
| [`../cpp_infer/configs/default_config.txt`](../cpp_infer/configs/default_config.txt) | 默认 RuntimeConfig：artifact、score/NMS 阈值和 provider |
| [`../cpp_infer/artifacts/yolov8_neu_det.artifact.txt`](../cpp_infer/artifacts/yolov8_neu_det.artifact.txt) | 模型身份、SHA-256、tensor、类别和前后处理契约 |
| [`../models/best.onnx`](../models/best.onnx) | 当前 YOLOv8/NEU-DET 基线模型，12,336,935 bytes |
| [`../data/images/val/crazing_241.jpg`](../data/images/val/crazing_241.jpg) | 固定 Demo 与正式单图 benchmark 样本 |
| [`../cpp_infer/tests/fixtures/consistency_manifest.json`](../cpp_infer/tests/fixtures/consistency_manifest.json) | 六类各 5 张、共 30 张的一致性固定 manifest |
| [`../cpp_infer/results/demo/`](../cpp_infer/results/demo/) | 已跟踪的固定 Demo JSON/PNG |
| [`../cpp_infer/results/consistency/`](../cpp_infer/results/consistency/) | 已跟踪的一致性逐图与汇总 JSON |
| [`../cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json`](../cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json) | 已跟踪的正式 C++ Release CPU benchmark 基线 |
| [`../cpp_infer/tools/stage1.cmd`](../cpp_infer/tools/stage1.cmd) | Windows 统一手工入口；负责发现 VS、继承 x64 环境并调用 PowerShell 调度器 |
| [`../cpp_infer/tools/stage1.defaults.psd1`](../cpp_infer/tools/stage1.defaults.psd1) | 已跟踪、与机器无关的 workflow 默认值 |
| [`../cpp_infer/tools/stage1.local.example.psd1`](../cpp_infer/tools/stage1.local.example.psd1) | 本机依赖配置模板；复制为 Git 忽略的 `cpp_infer/.stage1.local.psd1` |

路径解析规则不能混用：

- workflow/local、RuntimeConfig 和 artifact 中的相对路径，分别相对声明它们的文件解析。
- CLI 的图片和输出路径相对调用者当前工作目录解析。
- `ModelMetadata` 是 ORT 对真实模型的观察和校验证据，不是另一份配置。
- `CMakeCache.txt`、Makefiles、objects、binaries 和 staged DLL 是可丢弃构建状态，不应手工维护或提交。

## 2. 已验证的 Windows 环境

仓库内正式证据对应 Windows `10.0.26200`、x86_64、MSVC `19.50.35721.0`、Release、C++17。当前机器已验证的关键依赖为：

| 依赖 | 版本或路径 |
|---|---|
| Visual Studio C++ | `vswhere.exe` 自动发现；当前 `VsDevCmd.bat` 为 `D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat` |
| ONNX Runtime C++ SDK | `1.19.2`，`D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| OpenCV C++ | `4.8.0`，CMake package 为 `D:\01_Base\Tools\opencv\build\x64\vc16\lib`，DLL 为同级 `bin` |
| Python reference | `C:\Users\Everbreath\.conda\envs\TestBase\python.exe`；Python 3.9.25、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2 |
| GoogleTest | v1.17.0 commit `52eb8108c5bdec04579160ae17225d66034bd723`；archive SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84` |

GoogleTest 的有效源码目录由 Git 忽略的 `.stage1.local.psd1` 指定；该路径可能位于临时目录，不应写进 CMake 或提交到仓库。运行 `doctor` 查看本次真正解析到的路径。

## 3. 配置分层与依赖优先级

机器依赖路径的统一优先级是：

```text
显式命令参数 -> cpp_infer/.stage1.local.psd1 -> 进程环境变量 -> portable fallback
```

| 依赖 | 参数 | local key | 环境变量 | 当前 portable fallback |
|---|---|---|---|---|
| ORT SDK | `-OrtRoot` | `OrtRoot` | `ONNXRUNTIME_ROOT` | `D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| OpenCV CMake | `-OpenCvDir` | `OpenCvDir` | `OpenCV_DIR` | `D:\01_Base\Tools\opencv\build\x64\vc16\lib` |
| OpenCV DLL | `-OpenCvBin` | `OpenCvBin` | `YOLO_DEFECT_OPENCV_BIN` | 从已解析的 `OpenCvDir` 推导同级 `bin` |
| Python | `-PythonExe` | `PythonExe` | `YOLO_DEFECT_PYTHON` | 当前用户目录下 `.conda\envs\TestBase\python.exe` |
| GoogleTest source | `-GTestSource` | `GTestSource` | `YOLO_DEFECT_GTEST_SOURCE` | 无；必须提供已验证源码，或显式允许下载 |

`VsDevCmd.bat` 单独按 `YOLO_DEFECT_VSDEVCMD` -> `vswhere.exe` 自动发现解析。`detect` 的 RuntimeConfig/输出默认值按“显式参数 -> local 默认值 -> tracked workflow 默认值”解析。缺少本地 GoogleTest 源码时，只有显式传入 `-AllowGTestDownload` 才允许 CMake 使用已固定 URL 和 SHA-256 的 FetchContent。

## 4. `stage1.cmd` 十个动作

从普通 PowerShell 或 CMD 在仓库根目录运行；无参数等价于 `help`，不会触发构建。

| 动作 | 行为 |
|---|---|
| `help` | 显示帮助；不要求 Visual Studio 或 SDK |
| `doctor` | 只读检查 x64 MSVC、CMake/CTest、ORT C++ SDK、OpenCV、Python、GoogleTest policy 与解析后的默认值 |
| `build` | 已有构建树时增量构建；没有时先 configure |
| `clean-build` | 只在受保护的 TEMP 边界内删除并重建 NMake Release 构建树 |
| `test` | 构建当前源码并运行完整 CTest |
| `detect <image> [output-directory]` | 任意单图经过现有 Pipeline 输出 JSON/PNG；可用 `-Config` 和显式 `-Overwrite` |
| `demo` | 构建并验证固定样本的 3-detection JSON/PNG Demo |
| `consistency` | 构建并运行固定 30 图 Python ORT/C++ ORT 比较 |
| `benchmark` | 先重新运行 consistency，再按指定 warmup/repeat 运行 benchmark |
| `all` | clean configure/build -> 完整 CTest -> Demo -> consistency -> 正式 benchmark |

常用入口：

```powershell
cpp_infer\tools\stage1.cmd help
cpp_infer\tools\stage1.cmd doctor
cpp_infer\tools\stage1.cmd all
cpp_infer\tools\stage1.cmd detect "data\images\val\crazing_241.jpg"
```

`all` 固定使用 tracked workflow 中的 warmup `10`、repeat `100`；探索性非正式参数只应传给 `benchmark -Warmup <n> -Repeat <n>`。

## 5. 最小手工复现链

推荐的最短正式复现只有两步：

```powershell
cpp_infer\tools\stage1.cmd doctor
cpp_infer\tools\stage1.cmd all
```

如果需要审计 wrapper 背后的构建关系，先在 CMD 中初始化同一进程链里的 x64 开发环境：

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
powershell.exe -NoProfile
```

随后在该 PowerShell 中使用本机忽略配置进行最小 configure/build/CTest。每个 native command 后都必须立即检查退出码：

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

ctest --test-dir $BuildDir -N
if ($LASTEXITCODE -ne 0) { throw 'CTest inventory failed.' }

ctest --test-dir $BuildDir --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Complete CTest failed.' }
```

这段底层命令只复现 configure/build/完整测试。正式 Demo、一致性、benchmark 的顺序、fresh-output 校验和严格 JSON validator 仍以 `stage1.cmd all` 为权威，避免手工长命令漏掉正确性前置门。

## 6. TEMP 与证据边界

- 默认正式构建目录是 `%TEMP%\yolo_defect_stage1_manual_release`。
- 自定义 `-BuildDir` 必须位于真实 TEMP 根目录下，叶目录必须以 `yolo_defect_stage1_` 开头，且不能是 junction/symlink；`clean-build` 只会递归清理通过该检查的目标。
- `demo`、`consistency`、`benchmark` 和 `all` 每次都在 `<build>\stage1_evidence\<timestamp>_<GUID>` 下创建新证据，不覆盖仓库内已有 JSON/PNG。
- `detect` 默认输出到 Git 忽略的 `cpp_infer/results/manual/` 新目录；显式输出已存在时默认拒绝覆盖。
- 仓库内 `cpp_infer/results/{demo,consistency,benchmark}` 是已收口基线。TEMP 结果只有在本次命令、解析器、validator 和预期文件全部成功后才是一次有效复现，但仍不是自动提交的新基线。

## 7. 关键避坑

- 禁止把忽略目录 `cpp_infer/build` 中的旧二进制当作证据；始终使用 fresh out-of-tree TEMP build。
- Python `onnxruntime` wheel 不包含本项目所需的 C++ headers、import library 和 runtime DLL，不能替代官方 ORT C++ SDK。
- Windows 环境变量只从父进程传给子进程。另开 PowerShell 不会继承先前终端的 `VsDevCmd.bat`；日常直接调用 `stage1.cmd`。
- `call` 是 CMD builtin，不是 PowerShell command；PowerShell 查找可执行文件时使用 `where.exe`，不要误用 `where` alias。
- PowerShell 对 native command 的非零退出码不会自动停止；低层复现时必须紧跟 `$LASTEXITCODE` 检查。
- 不要手改 TEMP 下的 CMakeCache/Makefiles，也不要把个人绝对依赖路径写入 CMake、源码、tracked defaults 或模板。
- 不允许静默下载 GoogleTest；先配置并验证 v1.17.0 source，或明确传入 `-AllowGTestDownload`。
- benchmark 必须以同次 consistency 通过为前提。不得用旧 summary 造成假阳性，也不得为了通过而放宽冻结容差。
- 当前 C++ benchmark 与历史 Python 指标协议不同；不能据此无条件声称谁更快。
- 工作树可能包含用户未提交修改；清理或复现前先检查 `git status`，不要覆盖、恢复或删除无关变更。
