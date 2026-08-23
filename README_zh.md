# 工业视觉 AI 推理 Runtime — 钢材表面缺陷检测

[English](README.md)

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![OpenCV](https://img.shields.io/badge/OpenCV-C%2B%2B-green)
![CMake](https://img.shields.io/badge/CMake-enabled-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

V2 定位：本仓库正在从“YOLOv8 缺陷检测 demo”升级为“工业视觉 AI 推理 Runtime 与 C++ 工程化系统”。

YOLOv8 和 NEU-DET 是模型与数据集载体。秋招主线不是“我训练了一个检测模型”，而是“我把视觉模型通过 C++ / ONNX Runtime C++ / OpenCV / CMake / GTest / benchmark 变成可部署、可测试、可评测、可解释的工程化 Runtime”。

当前 V1 资产仍然保留价值：训练、ONNX 导出、历史 PyTorch-vs-ONNX **count-only** 抽查、Python ONNX Runtime 推理、FastAPI、Docker 和 benchmark 脚本。那次 50 图抽查没有验证类别与框坐标容差，不能称为本次严格一致性。V2 在这些资产之上通过 `cpp_infer/` 推进，而不是重写旧代码。

当前 V2 状态：**S1-08 L1 已验收；S1-09 自动门 PASS；用户 L2 待完成；大阶段一尚未完成；大阶段二未开始。** S1-09 没有新增产品功能，而是在全新临时 Release build 中依次复现 configure/build、完整 CTest 106/106（19.91 秒）、固定单图 Demo、六类 30 图 Python ORT/C++ ORT 一致性和 warmup 10/repeat 100 benchmark，并检查 JSON、PNG、核心非零错误和合法空检测。只有用户独立完成 2/5 分钟讲解、追问/排错以及一次“核心行为 + 对应 GTest”的可回滚练习后，才能宣布大阶段一完成；目录 batch、并发、INT8 和其他扩展均未开始。

项目入口刻意集中在本 README 和 `README_zh.md`。`docs/PLAN.md` 是最新规划源，`AGENTS.md` 将其固化为仓库级协作准则，任务、状态和变更证据维护在中英文 README 中。只有当执行细节过长、会降低总入口可读性时，才拆入 `docs/`。

![推理演示](docs/assets/demo_inference_result.gif)

## 项目1 Runtime 总入口蓝图

### 1. 项目定位和顶层设计

本仓库是**项目1：工业视觉边缘 AI Runtime 与 C++ 工程化系统**。它的核心价值不是在这个仓库里重新训练模型，而是把工业缺陷检测模型 artifact 变成可运行、可测试、可评测、可复盘、可面试讲清楚的 C++ Runtime。

项目1只通过明确的 artifact 门禁接收两类模型来源：

- **YOLOv8 + NEU-DET：** 稳定的 P0 Runtime baseline。它输出结构简单，仓库已有训练、ONNX 导出、Python 推理、FastAPI、Docker 和 benchmark 证据，适合先打通 C++ 部署主链路。
- **`paper_detect` D010 / D-FINE-S + DeepPCB：** 后续研究侧 artifact 来源。D010 是 Template-Counterfactual Defect Denoising：它保持 D003 inference path，新增训练期 erase/replay 样本，不引入 D009 feature-pyramid injection。`paper_detect` 负责训练、验证、消融、official test、result card 和定性图；只有稳定导出 ONNX、形成部署契约、真实接入 Runtime 并通过一致性验证后，本仓库才能把 D010 写成 Runtime 结果。

顶层设计原则：**训练侧和研究侧产物进入项目1；项目1负责部署链路、Runtime 行为、测试、benchmark 证据和推理事件输出。**

权威 P0 顶层设计不只是“调用一次推理”：

- **工程契约：** C++17/CMake 多 target、头源分离、依赖边界清晰，并校验模型、输入、类别、阈值、前处理和后处理的 Runtime/artifact schema。
- **推理链路：** OpenCV letterbox/RGB/normalize/NCHW；ONNX Runtime C++ 采用 RAII，并检查 name/shape/dtype/provider；随后完成模型族对应的 decode、过滤、NMS 和坐标还原。
- **可观察输出：** 固定样本 detection JSON、可视化结果和可复现命令。
- **正确性证据：** 声明容差，并比较 Python/ONNX/C++ 的检测数量、类别、置信度和坐标误差。
- **性能证据：** warmup/repeat，preprocess/infer/postprocess/end-to-end P50/P95、throughput、环境元数据，以及可行时的峰值内存/RSS。
- **工程证据：** GTest、非法输入与故障注入、INT8 PTQ 对比、限制说明、原始证据路径和 README 复现命令。

大阶段一建立第一次可投递纵切；大阶段二补齐完整 P0 证据和 INT8 加固。后续 P1 扩展必须受门禁约束：批处理/并发按面试价值选择；TensorRT/Jetson/ARM 必须有真实硬件；Qt 与 gRPC/Triton 只有在多个高优先 JD 反复要求时才做；D010 必须先有稳定 artifact。

### 2. 解决的问题

项目1解决的是“有一个检测模型”和“能把模型作为工程软件部署、测试、评测、讲清楚”之间的断层：

- 把图片和模型 artifact 变成可复现的 C++ 推理链路。
- 将 preprocess、inference、postprocess、NMS、benchmark、输出写入拆成可观察模块。
- 记录命令、样例输出、失败原因和取舍，让项目能服务秋招复盘与面试追问。
- 在项目1 P0 稳定后保留可选的 `inference_event` 桥接到项目2；它不属于大阶段一验收项。

### 3. 总体架构链路

计划完整链路：

```text
model artifact
-> artifact contract / model card
-> artifact schema + RuntimeConfig validation
-> OpenCV image read
-> letterbox preprocess / RGB / float32 / NCHW tensor
-> ONNX Runtime C++ session
-> input/output name / shape / dtype / provider checks
-> postprocess / score filter / NMS / coordinate restore
-> detection JSON
-> visualization
-> fixed-sample Python / ONNX / C++ consistency
-> benchmark report
-> INT8 PTQ comparison
-> tests / failure injection / README evidence
-> optional real-device deployment and Project 2 inference_event bridge
```

当前 S1-09 fresh reproduction 已重新验证的链路：

```text
cpp_infer/configs/default_config.txt
-> RuntimeConfig
-> 相对 config 文件解析 artifact_spec_path
-> ModelArtifactSpec + TensorSpec
-> 相对 artifact 文件解析 model_path
-> RuntimeContract 跨字段校验
-> OnnxRunner PImpl / Ort::Env / SessionOptions / Session RAII
-> 显式注册 CPUExecutionProvider
-> 读取实际 ORT ModelMetadata
-> 校验 provider/count/name/shape/dtype/class channel
-> OpenCV preprocess -> 连续 float32 NCHW vector
-> 严格校验输入 shape/元素数/finite values
-> 借用 CPU Ort::Value -> 同步 Session::Run
-> 校验输出 count/shape/元素数/finite values
-> 把 ORT 输出复制进自有 InferenceOutput
-> 有界 raw-output 摘要
-> 纯函数校验 [1,4+C,N] BCN raw output
-> 无独立 objectness 的 class argmax + float32 strict confidence filter
-> xywh -> xyxy -> stable class-agnostic input-space NMS
-> letterbox 坐标逆变换 -> 原图边界 clip
-> SingleImageDetectionResult
-> 固定字段顺序、UTF-8 安全 escaping、locale 无关的 detection JSON v1
-> 确定性 OpenCV 颜色/标签 -> 无 GUI 可视化文件
-> 默认拒绝覆盖、显式 --overwrite、父目录创建与输入路径保护
-> 四个 GTest target：postprocess 25 + preprocess 7 + output 7 + benchmark 8
-> synthetic metadata/contract 与专用 integration/negative CTest
-> integration 与可行动故障注入 gate
-> 106 项完整 CTest quality gate
-> 固定六类 x 每类 5 张 manifest（索引 241/255/270/285/300）
-> Python 复现同一 contract/preprocess/strict threshold/class-agnostic NMS/坐标语义
-> Python 显式 CPUExecutionProvider + C++ 显式 CPUExecutionProvider
-> 按 class_id 分组、最大 IoU 与确定性 tie-break 匹配，不依赖输出顺序
-> 30/30 图片、62/62 detections 通过预声明门槛
-> per_image.json + summary.json 机器可读正确性证据
-> consistency 2/2 正确性前置门通过后才运行 benchmark
-> Release-only、batch=1、CPU、sequential、intra/inter-op 1/1
-> warmup 10 次不采样 + repeat 100 次正式采样
-> image decode / preprocess / Session::Run / postprocess 分段计时
-> pipeline 与 end-to-end 壁钟计时及 throughput
-> Windows 进程生命周期 Peak Working Set
-> 稳定 benchmark JSON + Python 严格 schema/protocol validator
-> benchmark 14/14、完整 CTest 106/106
-> S1-09 自动门 PASS，用户 L2 待完成
-> 尚无目录 batch、并发或 INT8
```

### 4. 核心模块职责

| 模块 | 职责 | 当前状态 |
|------|------|----------|
| `RuntimeConfig` / `ModelArtifactSpec` / `TensorSpec` / `RuntimeContract` | 分离运行策略与模型身份/I/O/算法语义，按声明文件解析相对路径，并用严格 schema 给出可行动错误。 | S1-01 已验证 |
| `ImagePreprocessor` | 用 OpenCV 读图、letterbox、BGR->RGB、normalize，并输出 NCHW float tensor 与逆变换元数据；文件入口和 `const cv::Mat&` 入口复用同一实现，Mat 边界只接受非空 `CV_8UC3`。 | S1-04 已用横图、竖图、奇数 padding、非正方形模型输入和已知像素精确验证 |
| `OnnxRunner` | 通过 RAII/PImpl 管理 ORT 资源；校验借用的连续 float32 输入 vector，创建 CPU `Ort::Value`，同步运行，校验 raw output，并在 ORT ownership 结束前复制。 | S1-03 raw inference 已验证 |
| `ModelMetadata` | 表示实际 ORT 版本/provider 与 tensor count/name/shape/dtype，再通过可 synthetic 测试的纯校验函数对照 `RuntimeContract`。 | S1-02 已验证 |
| `InferenceOutput` | 独立持有返回的 raw tensor shape 和 float values，不依赖局部 ORT output value 或 Runner 生命周期，并作为纯后处理入口。 | S1-03 ownership 已验证；S1-04 synthetic decode 已验证 |
| `Detection` / `Postprocessor` | 校验并解析 `[1,4+C,N]` BCN output；取最大类别分数、执行 float32 域严格 `confidence > threshold`、`xywh -> xyxy`、IoU、稳定 class-agnostic NMS，再做 letterbox 坐标还原和 clip。 | S1-04 已用 24 项纯 synthetic GTest 验证；不需要 ORT session 或真实模型 |
| `SingleImageDetectionResult` / `DetectorPipeline` | 用自持有的结果对象记录模型、图片、session provider、阈值和 detections，并把已验证模块编排成严格单图纵切；`main.cpp` 只处理 CLI 与调用。 | S1-05 已完成固定单图端到端编排 |
| `ResultWriter` / `Visualizer` | 严格校验输出结果，安全 escape UTF-8 JSON 字符串，使用稳定字段/数值格式，执行目录/覆盖/保护路径策略，并用固定颜色和标签输出无 GUI OpenCV 可视化。 | S1-05 已生成可解析 JSON v1 与可读 PNG，L1 已验收；S1-06 增加不可创建父路径故障 gate |
| `ConsistencyManifest` / `compare_consistency.py` | 冻结六类 x 每类五张验证图及图片 SHA-256，从同一 Runtime/artifact 契约建立 Python ORT CPU reference，调用 C++ 单图 CLI，并按类别与最大 IoU 确定性匹配；输出逐图和汇总 JSON，失败时保留可行动诊断。 | S1-07 已验证并完成 L1 验收；S1-09 fresh reproduction 再次得到 30/30 图片、62/62 detections 通过冻结门槛，两份 JSON 均可解析 |
| `BenchmarkResult` / `BenchmarkRunner` | 固定 Release、batch=1、CPU provider 和线程策略；session 只初始化一次，在完整 warmup 后采集六段 steady-clock 延迟，计算 nearest-rank P50/P95 与 throughput，并取得可披露作用域的内存证据。`OnnxRunner` 在内部只围绕 `Session::Run` 计时，pipeline 另含 tensor 构造、校验和输出复制等真实边界成本。 | S1-08 固定图 warmup 10/repeat 100 已完成；六段 mean/P50/P95、两项 throughput 与 Windows Peak Working Set 已记录 |
| `BenchmarkWriter` | 严格校验 benchmark result，使用 classic locale、finite number 和 UTF-8 JSON escaping 输出稳定 schema；相对路径按 CWD 解析，创建父目录，并执行默认拒绝覆盖、普通文件限定和 config/artifact/model/image 保护。 | S1-08 仓库内历史 JSON 与 S1-09 临时 fresh JSON 分开记录；fresh JSON 通过 `json.tool` 与严格 validator，旧文件不能造成假阳性 |
| `ArtifactRegistry` / `ModelCard` | 记录 artifact 来源、模型族、数据集、指标、配置、后处理类型、runtime 状态和路径。 | YOLO baseline 声明已建立；D010 继续受门禁约束 |
| `Tests` | 四个 GTest target 分别承担 postprocess、`cv::Mat` preprocess、output 和 benchmark statistics；metadata/inference 使用独立测试 executable，另有 Python validator、CLI negative 与真实 integration CTest。CLI 故障同时断言非零退出和可行动诊断。 | S1-09 全新 Release：CTest 列出 106 项并完整通过 106/106（19.91 秒）；四个直接故障均 exit 1 且 actionable；两项合法 empty-detection 测试通过 |

### 5. 快速启动

Windows 的权威入口现在是统一任务脚本，不再要求手工复制 CMD、PowerShell 和 CMake 长命令。在仓库根目录的普通 PowerShell 或 CMD 中运行；不带参数时会安全显示 `help`，不会意外启动耗时构建：

```powershell
.\cpp_infer\tools\stage1.cmd help
```

| 动作 | 精确职责 |
|---|---|
| `help` | 不依赖 Visual Studio 或 SDK，直接显示命令说明 |
| `doctor` | 只读核验 x64 MSVC、CMake/CTest、完整 ORT C++ SDK、OpenCV、Python CPU ORT、GTest 策略和解析后的工作流默认值 |
| `build` | 增量构建；构建树不存在时先自动 configure |
| `clean-build` | 只删除通过边界校验的 Stage-One TEMP 目录，再执行带测试的 NMake Release configure/build |
| `test` | 构建当前源码并运行完整 106 项 CTest |
| `detect` | 任意单图执行前处理 -> ORT -> 后处理 -> JSON/PNG；它不是目录批处理 |
| `demo` | 构建并验证固定的 3 detection 样本 |
| `consistency` | 构建并执行固定 30 图、六类别 Python ORT/C++ ORT 一致性 |
| `benchmark` | 构建、重新执行 consistency，再按配置或本次覆盖的 warmup/repeat 测速 |
| `all` | clean build -> 完整 CTest -> 固定 Demo -> 30 图一致性 -> 正式 10/100 benchmark |

最常用命令现在是：

```powershell
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd clean-build
.\cpp_infer\tools\stage1.cmd test
.\cpp_infer\tools\stage1.cmd detect "D:\images\sample.jpg"
.\cpp_infer\tools\stage1.cmd detect "D:\images\sample.jpg" "D:\outputs"
.\cpp_infer\tools\stage1.cmd demo
.\cpp_infer\tools\stage1.cmd consistency
.\cpp_infer\tools\stage1.cmd benchmark
.\cpp_infer\tools\stage1.cmd all
```

只给图片时，`detect` 会在被 Git 忽略的 `cpp_infer/results/manual/` 下创建全新目录；再给第二个位置参数时，会在该目录生成 `<stem>.detections.json` 和 `<stem>.visualized.png`。已有输出默认受保护，只有显式传入 `-Overwrite` 才允许覆盖。命令会打印解析后的 Runtime config、输入、输出、actual provider、检测数量和文件路径；核心仍调用已有 `DetectorPipeline`，脚本没有复制推理逻辑。

环境和配置按职责明确分开：

| 项目 | 类型与职责 |
|---|---|
| `stage1.cmd` / `stage1.ps1` | 任务入口和调度器，不是模型配置 |
| `vswhere` -> x64 `VsDevCmd.bat` -> `PowerShell -NoProfile` | 查找 Visual Studio、设置临时编译环境变量，并由干净子终端继承 |
| Git 忽略的 `.stage1.local.psd1` | 本机 ORT/OpenCV/Python/GTest 路径和可选 detect 默认值 |
| 仓库跟踪的 `stage1.defaults.psd1` | 与机器无关的 build/Demo/detect/consistency/benchmark 工作流默认值 |
| `CMakeLists.txt` | 构建关系：Runtime library、CLI、tests 和依赖链接 |
| RuntimeConfig -> ArtifactSpec | 声明运行策略 -> 模型身份、tensor 和前后处理契约 |
| `ModelMetadata` | ORT 从真实 ONNX 观察到的实际信息；它是校验证据，不是配置文件 |
| `CMakeCache.txt` / Makefiles | 自动生成的构建状态；用 `clean-build` 重建，不手工修改 |

构建关系是 `CMake -> NMake -> cl/link -> library/CLI/test executables`；测试关系是 `CTest -> GTest executables 以及 CLI/Python/CMake tests`。`Release` 是构建模式，不是工具。命令行图片/输出相对调用者 CWD 解析，workflow/local/Runtime/artifact 路径相对各自声明文件解析。依赖路径优先级为命令参数 -> local 文件 -> 环境变量 -> portable fallback，detect 默认值优先级为命令参数 -> local 默认值 -> tracked workflow 默认值。

`clean-build` 只重建受保护的 `%TEMP%\yolo_defect_stage1_manual_release`；其他动作会先构建当前源码，缺少构建树时自动 configure。`benchmark` 必定先重跑 consistency，`all` 使用仓库正式协议。正式证据进入全新的临时 GUID 目录；日常 `detect` 输出只是便捷结果。Python wheel 不会被当作 ORT C++ SDK，GTest 也不会在缺少显式 `-AllowGTestDownload` 时下载。

下面的展开命令继续作为理解脚本内部行为的底层审计参考，不再是推荐的日常手动入口。

当前 S1-09 clean Release 收口路径。顺序固定为 configure/build -> 完整 CTest -> 固定 Demo -> 30 图一致性 -> benchmark；每一步都立即检查 `$LASTEXITCODE`。所有新证据写到新的 `%TEMP%` build 下，验证器不会误读仓库里已有的历史 JSON：

```powershell
# 先在 CMD 运行 VsDevCmd.bat，再启动：
# powershell.exe -NoProfile -NoExit
# -NoProfile 防止本机 Conda profile 覆盖 VS 的 PATH。
$Repo = 'D:\01_Base\CodingSpace\yolo_defect'
$OrtRoot = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$OpenCvDir = 'D:\01_Base\Tools\opencv\build\x64\vc16\lib'
$OpenCvBin = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin'
$CMakeBin = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'

Set-Location $Repo
$env:ONNXRUNTIME_ROOT = $OrtRoot
$env:PATH = $CMakeBin + ';' + $OpenCvBin + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_09_' + [guid]::NewGuid().ToString('N'))
$EvidenceDir = Join-Path $BuildDir 's1_09_closure'

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$OpenCvDir" `
  -DONNXRUNTIME_ROOT="$OrtRoot" `
  -DPython3_EXECUTABLE="$PythonExe" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
if ($LASTEXITCODE -ne 0) { throw 'S1-09 configure 失败。' }

cmake --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'S1-09 Release build 失败。' }

ctest --test-dir $BuildDir -N
if ($LASTEXITCODE -ne 0) { throw 'CTest 枚举失败。' }

ctest --test-dir $BuildDir --output-on-failure
if ($LASTEXITCODE -ne 0) { throw '完整 CTest 失败；禁止继续收口。' }

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$DemoDir = Join-Path $EvidenceDir 'demo'
$JsonPath = Join-Path $DemoDir 'crazing_241.json'
$VisualizationPath = Join-Path $DemoDir 'crazing_241.png'
$DetectionValidator = (Resolve-Path `
  'cpp_infer\tests\assert_detection_json.py').Path

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --output-json $JsonPath `
  --output-image $VisualizationPath
if ($LASTEXITCODE -ne 0) { throw '固定单图 Demo 失败。' }
if (!(Test-Path -LiteralPath $JsonPath -PathType Leaf) -or
    !(Test-Path -LiteralPath $VisualizationPath -PathType Leaf)) {
  throw 'Demo 没有生成本次临时 JSON/PNG。'
}

& $PythonExe -m json.tool $JsonPath > $null
if ($LASTEXITCODE -ne 0) { throw '本次 Demo JSON 解析失败。' }
& $PythonExe $DetectionValidator $JsonPath --expected-image $Image
if ($LASTEXITCODE -ne 0) { throw '本次 Demo JSON 契约校验失败。' }
& "$BuildDir\bin\yolo_defect_image_probe.exe" $VisualizationPath
if ($LASTEXITCODE -ne 0) { throw '本次 Demo PNG 无法由 OpenCV 回读。' }
Get-Item $JsonPath, $VisualizationPath
Get-FileHash $JsonPath, $VisualizationPath -Algorithm SHA256

$Manifest = (Resolve-Path `
  'cpp_infer\tests\fixtures\consistency_manifest.json').Path
$ConsistencyDir = Join-Path $EvidenceDir 'consistency'
if (Test-Path -LiteralPath $ConsistencyDir) {
  throw '本次 consistency 输出目录不应预先存在。'
}
& $PythonExe 'cpp_infer\tools\compare_consistency.py' `
  --manifest $Manifest `
  --cpp-cli "$BuildDir\bin\yolo_defect_cpp.exe" `
  --cpp-opencv-version 4.8.0 `
  --output-dir $ConsistencyDir
if ($LASTEXITCODE -ne 0) {
  throw 'S1-07 consistency 失败；禁止继续发布 benchmark。'
}
if (!(Test-Path -LiteralPath "$ConsistencyDir\per_image.json" -PathType Leaf) -or
    !(Test-Path -LiteralPath "$ConsistencyDir\summary.json" -PathType Leaf)) {
  throw '本次 consistency 没有生成两份 JSON。'
}
& $PythonExe -m json.tool `
  "$ConsistencyDir\per_image.json" > $null
if ($LASTEXITCODE -ne 0) { throw '本次 per_image.json 解析失败。' }
& $PythonExe -m json.tool `
  "$ConsistencyDir\summary.json" > $null
if ($LASTEXITCODE -ne 0) { throw '本次 summary.json 解析失败。' }

ctest --test-dir $BuildDir -L consistency --output-on-failure
if ($LASTEXITCODE -ne 0) {
  throw 'S1-07 consistency 失败；禁止继续发布 benchmark。'
}

$BenchmarkJson = Join-Path $EvidenceDir `
  'benchmark\yolov8_neu_det_cpu_release.json'
if (Test-Path -LiteralPath $BenchmarkJson) {
  throw '本次 benchmark JSON 不应预先存在。'
}
& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --benchmark `
  --warmup 10 `
  --repeat 100 `
  --benchmark-json $BenchmarkJson
if ($LASTEXITCODE -ne 0) { throw 'S1-09 benchmark 失败。' }
if (!(Test-Path -LiteralPath $BenchmarkJson -PathType Leaf)) {
  throw '本次 benchmark 没有生成新 JSON。'
}

& $PythonExe -m json.tool $BenchmarkJson > $null
if ($LASTEXITCODE -ne 0) { throw '本次 benchmark JSON 解析失败。' }
& $PythonExe 'cpp_infer\tests\assert_benchmark_json.py' `
  $BenchmarkJson `
  --expected-image $Image `
  --expected-warmup 10 `
  --expected-repeat 100
if ($LASTEXITCODE -ne 0) { throw '本次 benchmark 严格校验失败。' }
Get-Item $BenchmarkJson
Get-FileHash $BenchmarkJson -Algorithm SHA256

ctest --test-dir $BuildDir -L benchmark --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Benchmark CTest 失败。' }

ctest --test-dir $BuildDir `
  -R 'yolo_defect_cpp_(inspect_missing_model|damaged_image|output_unwritable_parent|benchmark_invalid_repeat)' `
  --output-on-failure
if ($LASTEXITCODE -ne 0) { throw '核心故障注入 gate 失败。' }

ctest --test-dir $BuildDir `
  -R 'postprocess.PostprocessEmptyTest.ValidTensorWithNoScoreAboveThresholdIsEmpty|output.ResultWriterJsonTest.EmptyDetectionsSerializeAsAnEmptyArray' `
  --output-on-failure
if ($LASTEXITCODE -ne 0) { throw '合法 empty-detection gate 失败。' }
```

`BUILD_TESTING=ON` 时，CMake 才会获取固定的 GoogleTest v1.17.0 commit archive，并校验 SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`。完全离线时，应先对同一 archive 执行 `Get-FileHash -Algorithm SHA256`，确认 hash 后解压，再在 configure 命令追加：

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='<path-to-verified-googletest-source>'
```

source-directory override 会绕过 FetchContent 的下载/hash 步骤，因此不能跳过解压前的手工 hash 复核。CMake 中不含个人 GTest 绝对路径。OpenCV 的 DLL 仍通过上面的 `PATH` 提供给 CLI 和 build-time GTest discovery。

S1-09 configure 继续用显式 `Python3_EXECUTABLE` 检查 NumPy、OpenCV Python、ONNX Runtime 1.19.2 和 `CPUExecutionProvider`，缺失时给出错误并停止；CMake、consistency 工具和 benchmark validator 都不会静默安装 Python 包。正式 benchmark 必须使用 Release build；Debug 或其他 build type 会被 Runtime 拒绝，不能发布成性能证据。

同一路径再次运行时默认以非零退出拒绝覆盖；只有明确希望替换已有普通输出文件时，才在完整 CLI 后追加 `--overwrite`。S1-09 Quick Start 使用新的 GUID build 和临时 evidence 目录，刻意不覆盖仓库内 S1-05/S1-07/S1-08 历史证据。下方旧 Python/YOLO 快速开始仍保留，用于复现 V1 baseline；上面的 C++ 命令是 V2 部署主入口。

请按上面命令使用全新的 out-of-tree build。2026-07-15 已确认被忽略的 `cpp_infer/build` 可执行文件仍是旧 P1-01 产物，会拒绝新 `--config/--image` 参数，不能作为当前源码证据。

### 6. Demo 输入输出

当前 demo 输入：

```text
config: cpp_infer/configs/default_config.txt
artifact: cpp_infer/artifacts/yolov8_neu_det.artifact.txt
image:  data/images/val/crazing_241.jpg
```

S1-05 首次生成、S1-09 fresh reproduction 再次得到的固定单图 Demo 输出：

```text
detection_json: cpp_infer/results/demo/crazing_241.detections.json
visualization:  cpp_infer/results/demo/crazing_241.visualized.png
detections:     3
classes:        crazing, crazing, crazing
actual_provider: CPUExecutionProvider
JSON bytes:     1164
PNG bytes:      39306
```

三个 detection 按稳定 confidence 降序为：

```text
0: class=crazing, confidence=0.445792824, bbox=[0, 53.6803322, 176.90683, 146.240784]
1: class=crazing, confidence=0.417582601, bbox=[21.2503815, 118.812775, 188.814178, 194.868408]
2: class=crazing, confidence=0.308511496, bbox=[22.7723389, 2.68823242, 192.409409, 86.2025604]
```

产物复核值：

```text
JSON SHA-256: E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049
PNG SHA-256:  3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346
```

JSON v1 固定包含 `schema_version`、`model`、`image`、`runtime` 和 `detections`；每个 detection 包含 `class_id`、`class_name`、`confidence` 和 `[x1,y1,x2,y2]` 的 `bbox_xyxy`。无检测是合法的 `"detections": []`，不会写成 `null` 或省略字段。字符串经过 UTF-8 校验和 JSON escaping，数值使用 classic locale 与稳定精度。

CLI 相对图片/输出路径以当前 working directory 为基准；config/artifact 内部相对路径仍分别以声明文件为基准。缺失父目录会递归创建。已有输出默认拒绝并非零退出，只有 `--overwrite` 才允许替换普通输出文件；目录、符号/特殊文件、JSON/图片同径，以及 config、artifact、model、源图片等保护路径始终拒绝。

S1-03 的 `--raw-output-summary` 和 S1-04 的纯算法测试仍作为回归入口保留；当前可展示纵切已经到达 detection JSON、可视化、严格一致性和 C++ benchmark JSON。`inference_event`、目录 batch、并发和 INT8 仍是后续能力，不能写成当前产物。

### 7. 测试命令

当前聚焦测试与完整 CTest quality gate：

```powershell
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir -L unit --output-on-failure
ctest --test-dir $BuildDir -L negative --output-on-failure
ctest --test-dir $BuildDir -L integration --output-on-failure
ctest --test-dir $BuildDir -L consistency --output-on-failure
ctest --test-dir $BuildDir -L benchmark --output-on-failure
ctest --test-dir $BuildDir -L quality_gate --output-on-failure
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir --output-on-failure
```

S1-09 全新临时 Release build 的 fresh reproduction 结果：

```text
consistency 前置门:     2/2 passed
benchmark label:      14/14 passed
CTest -N:                106 tests
完整 CTest:           106/106 passed，19.91 s
```

四个 GTest target 当前分别为 postprocess 25、preprocess 7、output 7、benchmark 8，共 47 项 discovered GTest；metadata 与 inference 是独立测试 executable，不应误写成 GTest target。S1-07 的 manifest/matching 与 30 图真实 integration 是 benchmark 前置正确性门；低 repeat 真实模型 smoke 与严格 Python validator 再共同覆盖 Release/provider/线程、六段统计、Windows memory、稳定 JSON、SHA、披露项和安全覆盖行为。CTest 标签存在重叠，不能把 consistency、benchmark、unit、negative、integration 等数量相加得到 106。

#### S1-06 故障排查

- **Schema：** 从声明文件、行号/字段、expected、actual 和 action 开始；`artifact_spec_path` 相对 Runtime config，`model_path` 相对 artifact 声明文件，不相对进程 working directory。
- **缺模型：** 查看 artifact 中的 `model_path` 和错误打印的规范化路径。
- **Metadata mismatch：** 先运行 `--inspect-model`，再对照实际 name/shape/dtype/provider 与声明；synthetic mismatch 只测试纯 validator，不需要在仓库放入多个大型坏模型。
- **损坏图片：** 区分“路径不存在”和“文件存在但 OpenCV decode 得到空图”，然后用已知正常图片复测。
- **CLI/输出：** 先看 `--help`，检查缺值、重复/冲突参数、父路径类型与权限、保护输入以及显式覆盖策略。

S1-09 全新构建上的四个直接故障证据分别是缺失模型、损坏图片、“普通文件被当成输出父目录”，以及 benchmark `repeat=0`；四条命令均返回退出码 1，且信息包含失败对象/路径、expected、actual 和修改建议。合法 `[1,4+C,N]` tensor 如果没有分数严格大于阈值，应成功得到 `detections: []`；`PostprocessEmptyTest.ValidTensorWithNoScoreAboveThresholdIsEmpty` 与 `ResultWriterJsonTest.EmptyDetectionsSerializeAsAnEmptyArray` 两项测试均通过。错误 rank/channel/count 或 non-finite raw output 必须报错，不能伪装成空检测。

#### S1-07 一致性门槛、证据边界与排错

S1-07 在第一次正式比较前冻结门槛：逐图 detection 数量和匹配后的 `class_id` 必须完全一致，confidence 绝对误差 `<= 1e-4`，四个 bbox 坐标的绝对误差都 `<= 1e-2` 像素，matching IoU `>= 0.999`。匹配先按类别分组，再按最大 IoU 和 canonical value tie-break 确定性配对，因此不依赖 Python/C++ JSON 的 detection 顺序。任何失败都应沿 `image decode/preprocess -> raw output -> strict threshold/NMS -> coordinate restore -> matching` 分层定位，不能为了全绿直接放宽门槛。

实际调试中，Python 3.9 的 `Path.write_text()` 不支持 `newline` 参数；工具改用 `path.open("w", encoding="utf-8", newline="\n")` 写稳定 JSON。该问题属于证据文件写入兼容性，不是模型数值不一致，预声明容差没有调整。

当前证据只证明同一个 ONNX artifact 在 Python ORT 与 C++ ORT 两套实现中的检测结果一致，不是数据集精度评估。匹配的 `best.pt` 当前不可用，所以没有重新完成 PyTorch/Python ORT/C++ 三方实测；session provider 也不是逐节点放置证据。30 张 manifest 图片全部为 200x200，缩放到 800x800 时没有 padding，因此本次跨语言比较没有覆盖非方图 letterbox；这部分只由 S1-06 synthetic 测试覆盖。30 张图片也都至少产生一个检测，没有提供跨语言 empty-detection 样本。

#### S1-08 Benchmark 协议、边界与排错

正式性能证据固定为 Release、batch=1、`CPUExecutionProvider`、sequential、intra-op 1、inter-op 1、graph optimization all、`crazing_241.jpg`、warmup 10 和 repeat 100。使用 `std::chrono::steady_clock`；P50/P95 采用 empirical nearest-rank ceiling。`image_decode` 只包含重复 `imread`，`preprocess` 从已解码 `cv::Mat` 到 NCHW tensor，`session_run` 只围绕 `Ort::Session::Run`，`postprocess` 从 raw output 到 detections；`pipeline` 是 preprocess+完整 runner 边界+postprocess 的壁钟时间，`end_to_end` 再包含 image decode。因此 pipeline 会额外包含输入校验/Ort tensor 构造和输出校验/复制，不应被强行解释成三个独立均值简单相加。

Session/model 初始化、初始路径/文件大小检查、统计计算、Peak Working Set 查询、benchmark JSON 序列化/写盘和可视化不进入 100 次重复计时；可视化根本不执行。Windows Peak Working Set 在写 JSON 前查询，但它是整个进程生命周期峰值，包含 config/session 初始化、warmup、正式迭代、保留的样本向量、统计和 harness 状态，不是单次推理或某个阶段的增量内存。

排错先确认 Release build、固定 CPU actual provider 和 1/1 线程策略，再检查 warmup/repeat 是否为合法整数、每次 detection count 是否稳定、六组 sample 是否 finite、throughput 是否等于 `1000/mean_ms`，最后检查 JSON 输出是否覆盖保护输入或指向目录/符号链接/特殊文件。`--benchmark` 与 detection JSON/PNG、`--inspect-model`、`--raw-output-summary` 互斥；错误必须非零退出并指出 object、expected、actual 和 action。

### 8. 关键数据与产物结果

四类证据必须按实验问题和协议分栏，不能把“都涉及 ONNX”当作同一种结论：

| 证据栏 | 证明什么 | 明确不证明什么 |
|--------|----------|----------------|
| 历史 PT/ONNX count-only 抽查 | 排序后前 50 张 `crazing` 图片的检测数量为 50/50、总数 146 vs 146 | 没有类别/框坐标容差；不是六类严格一致性；匹配 `.pt` 当前也无法重新复跑 |
| 历史 Python ORT benchmark | 历史 Python 脚本在其 100 图、5 次预热协议下记录 CPU 24.4 FPS、GPU 72.1 FPS | 不是当前 C++ Runtime 性能，不能与单图 10/100 C++ 协议无条件比较 |
| 当前 Python ORT/C++ ORT 严格证据 | 同一 ONNX、相同 contract、显式 CPU provider 下，六类 30 图的数量、类别、confidence、bbox 与 matching IoU 实现一致 | 不是 mAP/模型精度评估，也不是缺失 `.pt` 情况下的三方比较 |
| 当前 C++ ORT Release benchmark | 固定机器、模型、单图、batch=1、CPU 单线程策略下的六段延迟、throughput 与进程 Peak Working Set | 不是跨设备结论、冷盘延迟、并发吞吐、逐节点 provider 放置或单阶段独占内存 |

| 项 | 当前记录 |
|----|----------|
| P0 数据集 | NEU-DET 钢材表面缺陷，1,800 张图，6 类，200x200 像素 |
| P0 模型 | YOLOv8n baseline 与调参版本 |
| 当前最佳 YOLO 结果 | `final_train_2`，mAP@0.5 = 0.743，mAP@50-95 = 0.388 |
| 历史 ONNX/PyTorch 对齐 | 50/50 检测数量一致、总检测数 146 vs 146；但排序后的子集全是 `crazing`，且只记录数量/置信度摘要，没有类别/框坐标容差 |
| Baseline ONNX artifact 预检 | 已跟踪 `models/best.onnx`，12,336,935 bytes，opset 17，SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`；metadata 为 `nms=False` |
| 模型 lineage 状态 | 项目所有者确认当前 ONNX 是本人从 `runs/detect/final_train_2/weights/best.pt` 导出的；该 `.pt` 不在当前工作区或 Git 历史中，因此 lineage 已由所有者确认，但目前不能重新导出复核 |
| Baseline ONNX I/O 预检 | Python ORT 1.19.2 确认输入 `images` = float32 `[1,3,800,800]`；输出 `output0` = float32 `[1,10,13125]` |
| 历史 Python ORT benchmark | ONNX CPU 24.4 FPS，ONNX GPU 72.1 FPS（RTX 3060）；**不是 C++ Runtime 性能** |
| 当前 C++ Runtime 状态 | S1-09 使用全新临时 NMake/Release build 复现 MSVC 19.50.35721.0、C++17、C++ OpenCV 4.8.0、C++ ORT 1.19.2 与显式 CPU session；完整 CTest 106/106 在 19.91 秒内通过，随后 Demo、一致性和 benchmark 全部通过。自动门 PASS，但用户 L2 尚未完成，所以大阶段一仍未宣布完成；大阶段二未开始 |
| C++ ORT 实际 metadata | 已加载 `models/best.onnx`；实际 EP inventory 为 `[AzureExecutionProvider,CPUExecutionProvider]`；session 显式注册 `CPUExecutionProvider`；输入 `images` tensor float32 `[1,3,800,800]`；输出 `output0` tensor float32 `[1,10,13125]`；contract 通过 |
| S1-02 依赖/session 边界 | CMake 只通过 `ONNXRUNTIME_ROOT` 消费官方 ORT C++ SDK 1.19.2，校验版本/C/C++/CPU-provider headers/import library/DLL 并复制匹配 DLL。Session 策略为 sequential、intra-op 1、inter-op 1（sequential 下不使用）、graph optimization all |
| S1-03 raw-output 证据 | 固定 `crazing_241.jpg`：输入 float32 `[1,3,800,800]`，1,920,000 个 finite 值，范围 `[0.278431386,1]`；自有输出 float32 `[1,10,13125]`，131,250 个 finite 值，范围 `[0,795.04126]`。这些值只证明 finite raw execution，不证明 decoded detection 正确性或 benchmark 性能 |
| S1-04 postprocess 证据 | 纯 synthetic `[1,4+C,N]` tensor/box 精确验证：无 objectness、最大类别分数、float32 `confidence > threshold`、稳定 class-agnostic input-space NMS、IoU 边界、空候选、letterbox 逆变换和原图 clip；不使用真实模型作为算法正确性的唯一证据 |
| S1-04 GTest 依赖 | 官方 GoogleTest v1.17.0，commit `52eb8108c5bdec04579160ae17225d66034bd723`，archive SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`；`BUILD_TESTING=ON` 才获取，离线 source override 前必须先复核 archive hash |
| S1-05 固定 Demo 证据 | `crazing_241.jpg` 产生 3 个稳定降序的 `crazing` detections；JSON v1 可由 Python 标准库解析，PNG 可由 OpenCV 回读为 200x200 `CV_8UC3`。`cpp_infer/results/demo/crazing_241.detections.json` 为 1,164 bytes、SHA-256 `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`；`crazing_241.visualized.png` 为 39,306 bytes、SHA-256 `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346` |
| S1-06 quality gate 证据 | 全新 `%TEMP%` Release build 列出 90 项测试：unit 51、integration 3、negative 32、contract 19、metadata 16、preprocess 9、postprocess 25、output 18，全部通过。缺模型、损坏图片和不可创建的输出父路径均以 CLI exit 1 失败，并包含 object/path、expected、actual、action；没有增加大型 ONNX fixture |
| S1-07 consistency manifest | `cpp_infer/tests/fixtures/consistency_manifest.json` 固定 NEU-DET 六类，每类使用验证集索引 241、255、270、285、300，共 30 张；记录 class id/name、声明文件相对路径和图片 SHA-256，并冻结 provider、matching 策略与门槛 |
| S1-07 Python/C++ 一致性证据 | `cpp_infer/tools/compare_consistency.py` 使用 Python 3.9.25、ORT 1.19.2、OpenCV 4.13.0、NumPy 2.0.2，并显式只创建 `CPUExecutionProvider` session；C++ 侧为 ORT 1.19.2、OpenCV 4.8.0、显式 CPU session。30/30 图片、62/62 detections 匹配通过；最大 confidence 绝对误差 `8.049977111568296e-07`，最大 bbox 坐标绝对误差 `9.135351561440075e-05` 像素，最小 matching IoU `0.999998927116394`。机器可读证据为 `cpp_infer/results/consistency/per_image.json` 与 `summary.json` |
| S1-08 历史环境与固定协议 | Windows 10.0.26200，`DESKTOP-6OGK71C`，AMD64 Family 25 Model 117、16 个逻辑 CPU；MSVC 19.50.35721.0 Release C++17，OpenCV 4.8.0，ORT 1.19.2。requested/actual provider 为 `cpu`/`CPUExecutionProvider`，sequential、intra/inter-op 1/1、graph optimization all。模型 12,336,935 bytes、输入 `[1,3,800,800]`；样本 `crazing_241.jpg` 为 23,845 bytes、200x200x3，batch/sample=1，score=0.25、NMS=0.45、class-agnostic，warmup=10、repeat=100，稳定得到 3 个框 |
| S1-08 历史 C++ Release latency | mean / P50 / P95（ms）：image decode `0.991129 / 0.9649 / 1.3517`；preprocess `8.244569 / 7.5514 / 12.1265`；仅 `Session::Run` `165.555859 / 164.8985 / 186.2136`；postprocess `0.424115 / 0.4251 / 0.5636`；pipeline `175.560944 / 175.1058 / 195.1376`；end-to-end `176.553060 / 176.1357 / 196.6128` |
| S1-08 历史 throughput 与内存 | pipeline `5.696028 images/s`，end-to-end `5.664020 images/s`；Windows Peak Working Set `160,133,120 bytes`（`152.714844 MiB`）。仓库内历史证据：`cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json`，不得被 S1-09 临时输出静默冒充 |
| S1-09 fresh Demo | 同一固定图重新得到 3 个 detections；JSON 1,164 bytes、SHA-256 `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`；PNG 39,306 bytes、SHA-256 `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`；OpenCV probe 为 200x200 `CV_8UC3` |
| S1-09 fresh consistency | 六类各 5 张，30/30 图片、62/62 matches；最大 confidence 误差 `8.049977111568296e-07`，最大 bbox 误差 `9.135351561440075e-05 px`，最小 matching IoU `0.999998927116394`；本次临时 `per_image.json` 和 `summary.json` 均可解析 |
| S1-09 fresh C++ Release latency | mean / P50 / P95（ms）：image decode `0.816168 / 0.8182 / 0.9251`；preprocess `5.453755 / 5.4547 / 6.2128`；仅 `Session::Run` `134.419309 / 137.5882 / 142.5549`；postprocess `0.345302 / 0.3438 / 0.4424`；pipeline `141.265814 / 144.4673 / 149.8395`；end-to-end `142.082777 / 145.3222 / 150.7653` |
| S1-09 fresh throughput 与内存 | pipeline `7.078853 images/s`，end-to-end `7.038151 images/s`；Windows Peak Working Set `152.578125 MiB`。本次临时 benchmark JSON 为 5,453 bytes、SHA-256 `F32C0DF3157897264F9BD2B9AE3F3DB7B240A3B641494E8D3E7C346FF64E9C6F` |
| S1-09 fresh 故障/空结果 | 缺模型、损坏图片、不可创建输出和 benchmark `repeat=0` 四个直接故障均 exit 1 且 actionable；合法无候选 postprocess 与空数组 JSON 两项测试均通过 |
| Artifact 许可证检查点 | Artifact 声明原样保留 ONNX metadata 文本 `AGPL-3.0 License (https://ultralytics.com/license)`。源码仍是 MIT；所有者选择继续公开分发 ONNX 与 NEU-DET，因此模型义务和数据集未明确的再分发条款仍是独立发布门禁 |
| 后续研究侧 artifact | `paper_detect` D010 方法，位于 D-FINE-S/DeepPCB 研究线；不把它写成新的 Runtime 架构 |
| 外部 D010 研究证据 | Formal-validation AP50-95 = 0.847057；official-test AP50-95 = 0.830385；这些不是项目1 Runtime 结果 |
| D010 关系与消融 | D003 是 ancestor/消融锚点；D010 相对 D003 的 formal 与 official-test 6 类 delta 全部为正；D010A erase-only 和 D010B replay-only 都高于 D003、低于完整 D010 |
| D010 接入门禁 | 稳定 ONNX + result/model card + 部署契约 + 真实 Runtime adapter + 一致性验证；不得阻塞 YOLO P0 收口 |

待补 artifact 路径：

```text
artifacts/paper_detect_d010/result_card.md        # placeholder
artifacts/paper_detect_d010/model_artifact.yaml   # placeholder
artifacts/paper_detect_d010/metrics_table.csv     # placeholder
artifacts/paper_detect_d010/qualitative/          # placeholder
```

C++ 汇总证据已经覆盖机器/OS/编译器/构建类型、模型/输入/固定样本、正确性门槛、六段 mean/P50/P95、pipeline/end-to-end throughput、Peak Working Set、限制、证据路径和复现命令；S1-09 自动门已经完成 fresh reproduction，但用户 L2 门尚未完成，不能提前宣布大阶段一结束，也不能把单机单图结果扩写成跨设备结论。模型许可证仍是必须解决的 provenance 风险，不能因为它不阻塞本地 C++ 实现就隐藏。

当前限制必须按证据理解：JSON 的 `model.declared_sha256` 是从已验证 artifact 声明复制的值，单图 detection CLI 不会每次重新计算模型 hash；S1-08 严格 validator 会在正式证据检查中重算固定模型/样本 hash。`actual_provider` 证明显式 CPU EP 注册和 session 创建成功，是 session 级证据而不是逐节点 profiling。JSON 与 PNG 是两个文件，不构成跨文件事务，也不声称跨进程原子写入。S1-07 是同一 ONNX 的 Python ORT/C++ ORT 固定 30 图实现一致性，不是模型精度评估或缺失 `best.pt` 情况下的三方复跑。

S1-08 与 S1-09 两次 C++ benchmark 都只是单台 Windows CPU 机器、一个 200x200 图片、batch=1 的 warm-cache baseline：重复 `imread` 主要经过已预热的操作系统文件缓存，不代表冷盘延迟；没有锁定 CPU affinity、提升进程优先级或保证系统空闲，因此即使协议相同，两次运行也可能因系统状态得到不同数值。Peak Working Set 是进程生命周期峰值，不是按阶段或单次推理增量。历史 Python ORT CPU/GPU `24.4/72.1 FPS` 使用不同语言、硬件、样本与计时协议，只能分栏保留，不能据此无条件判断 C++ 更快或更慢。CLI 仍只支持单图，并未开始目录 batch、并发、INT8 或 `inference_event`。

### 9. 关键设计取舍

- **Runtime 优先，训练其次：** 保留旧训练资产，但 V2 主线不是继续包装训练。
- **先 YOLO baseline，再 D010 adapter：** YOLO/ONNX 是最快完成 C++ preprocess、infer、postprocess、JSON、benchmark、测试的稳定路线。
- **先过 artifact 门禁，再声称 D010 成果：** 外部 D010 研究指标可以作为来源证据，但 C++ D-FINE 结果必须有稳定导出、契约、adapter 和一致性证据。
- **简单 C++ 工程优先：** C++17、CMake、OpenCV、ONNX Runtime C++、GTest、benchmark 输出已经足够匹配秋招目标。
- **随稳定边界逐步加测试：** S1-01 测声明，S1-02 测 session metadata，S1-03 测一次真实 raw inference 和 Run 前拒绝；S1-04 用 31 项 synthetic GTest 独立证明 postprocess 与 `cv::Mat` preprocess 逻辑，而不是依赖一个真实模型 smoke 推断算法正确。
- **显式 tensor ownership：** CPU input `Ort::Value` 只在同步 Run 期间借用 preprocess vector；局部 ORT values 销毁前，把输出复制进不含 ORT 类型的 `InferenceOutput`。
- **冻结 YOLOv8 baseline 语义：** raw layout 是 `[1,4+C,N]` BCN，无独立 objectness；每个候选取最大类别分数，在 float32 域执行严格 `confidence > score_threshold`；先在模型输入空间做稳定 class-agnostic NMS，再减 padding、除 scale 并裁剪到原图边界。同 confidence 保留原候选顺序，同类别分数选择较小 class id。
- **测试入口不污染产品入口：** postprocess、preprocess、output 和 benchmark 四个 GTest target 都链接 `yolo_defect::runtime`，不编译或复用 `main.cpp`；metadata/inference 另有专用 CTest executable。纯 postprocess 不创建 ORT session，`cv::Mat` 测试显式承担 OpenCV 测试依赖。
- **薄 CLI、厚 Runtime：** `main.cpp` 只解析参数和编排；`DetectorPipeline`、结果校验、JSON serialization、输出路径策略和无 GUI 可视化仍位于 `yolo_defect_runtime`，便于测试与后续复用。
- **稳定机器输出：** JSON schema v1 固定字段顺序、类型和 `detections: []` 空结果；所有字符串先验证 UTF-8 并 escape，finite 数值用 classic locale 与稳定精度。新增/改变字段必须升级 schema，而不能悄悄破坏下游。
- **安全输出策略：** CLI 输出路径相对当前 working directory 解析；父目录自动创建，已有文件默认拒绝，显式 `--overwrite` 才允许替换普通文件，输入 config/artifact/model/image 与目录、符号/特殊文件始终受到保护。
- **一致性不依赖 detection 顺序：** Python 与 C++ detections 先按 `class_id` 分组，再按最大 IoU 与 canonical value tie-break 一对一匹配；数量/类别完全一致后，分别检查 confidence、四坐标和 matching IoU。容差在首次运行前写入 manifest，不因失败结果临时放宽。
- **先正确再测性能：** 正式 benchmark 前必须先通过冻结的 S1-07 consistency gate；正确性失败时不能用性能数字掩盖，也不能先优化再发布结果。
- **分段与总体时间分开：** `Session::Run` 在 `OnnxRunner` 内部精确计时；pipeline/end-to-end 另用壁钟覆盖真实模块边界。Session 初始化、JSON 写盘和绘图不混入 repeat，但都在 JSON 中明确披露；P50/P95 使用固定 nearest-rank 定义，避免只报告一次或平均值。
- **内存证据不伪装精度：** Windows 记录真实 Peak Working Set；不支持的平台必须写 `unsupported` 而不是 0。该指标是进程生命周期峰值，不是单次 inference 增量。
- **扩展受条件约束：** INT8 PTQ 属于 P0 证据加固；TensorRT/Jetson/ARM 是后续真实硬件扩展；Qt 与 gRPC/Triton 受 JD 需求门禁约束。
- **失败也要记录：** INT8、D-FINE 或符合条件的真实设备尝试即使失败，也要记录命令、错误、原因和回退路径，但不能把“尝试过”升级成“已有成果”。

### 10. 任务队列

最新路线以 `docs/PLAN.md` 为准；大阶段摘要未展开之处，由顶层设计补齐：

| 大阶段 | 目标时间 / 门禁 | 项目1阶段出口 |
|--------|-----------------|---------------|
| 已完成 baseline 与工程骨架 | 截至 2026-07-12 | 训练/导出/Python 资产，加上 C++17/CMake/CTest、类型化配置和真实图片 OpenCV 预处理；不声称已有 C++ 推理 |
| **一：可投递闭环（当前）** | 2026-07-13 至 2026-07-27 | 固定 config/image/model 命令完成 ORT、decode/filter/NMS/坐标还原、JSON/可视化；固定样本 Python ORT/C++ 一致性；分段 P50/P95；核心错误和自动化测试；5 分钟讲解及一次“核心行为+测试”修改 |
| 二：证据加固 | 2026-07-28 至 2026-08-10 | 补齐 P0 测试/故障矩阵、可复现性能与内存证据、FP32-vs-INT8 PTQ 对比、最终结果表、简历 bullet、面试题和专项 mock；QAT 仅在有依据时启动 |
| 三：P1 扩展 | P0 稳定后，按条件启动 | 按面试价值做 batch/worker/backpressure；有真实硬件才做 TensorRT/Jetson/ARM；多个高优先 JD 反复要求才做 Qt 或 gRPC/Triton；D010 必须先过 artifact 门禁 |
| 四：冻结与面试优先 | 2026-08-25 起 | 冻结 P0 功能；只允许正确性/demo/复现修复、测试与证据、JD 小补丁、面试反馈调整，以及不破坏 P0 的 P1 推进 |

大阶段一逐步执行的长版方案见 [`docs/STAGE1_EXECUTION_PLAN.md`](docs/STAGE1_EXECUTION_PLAN.md)。这是必要的长细节文档；README 仍然负责状态与证据主线。

### 11. 版本变化与进度记录

当前状态：历史项目1任务 P1-00 到 P1-03，以及大阶段一小阶段 **S1-01 至 S1-08** 均已实现、验证并完成相应 L1；S1-09 没有新增产品功能，自动 clean reproduction 门已经 PASS。用户 L2 讲解、追问/排错和行为+GTest 练习尚未完成，因此大阶段一仍未完成，大阶段二尚未开始。

2026-07-16 的大阶段一前置准备也已完成：ORT C++ 1.19.2 SDK 已存在并核验，VS x64 工具链可发现，全新 `%TEMP%` Release/NMake 构建通过 3/3 CTest，未来 GTest 依赖已固定版本。项目所有者也已确认当前 ONNX 是本人从 `final_train_2` best checkpoint 导出的；该 checkpoint 不在当前工作区或 Git 历史中。可复现命令、证据、GTest hash、模型 lineage 审计和未解决的公开分发许可检查点见 [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md)。这些准备没有开始 S1-01，也没有改变 Runtime 行为。

S1-09 的下一步不是继续写产品功能，而是由用户完成下面的 L2 门；完成后才能把大阶段一标记为完成。大阶段二继续承担 INT8 PTQ、FP32/INT8 正确性/精度/性能比较和完整证据加固，当前没有开始。`S1-*` 表示“大阶段一内部小阶段”，避免把旧项目1历史 ID `P1-*` 与顶层设计中的 P1 扩展类别混淆。

时间线式 V2 入口记录维护在下方“路线图”部分，每完成一个小阶段必须更新。

### 12. 从项目起点到现在的教学式记录

| 阶段 | 做了什么 | 目的 | 实现方式 / 证据 | 问题与排查经验 |
|------|----------|------|-----------------|----------------|
| P1-00 | 冻结 V2 定位，保护旧资产，建立 `cpp_infer/` 入口。 | 防止项目在训练 demo 和 Runtime 工程之间跑偏。 | README/README_zh/AGENTS 与 C++ 工作区骨架。 | README 要作为主线入口，不要把任务拆成很多碎片文档。 |
| P1-01 | 新增最小 C++17/CMake 可执行文件和 CTest help smoke。 | 证明仓库可以构建 C++ Runtime target。 | `yolo_defect_cpp --help` 和 CTest smoke。 | Visual Studio 多配置构建需要 `ctest -C Debug`。 |
| P1-02 | 新增无第三方依赖 ConfigLoader 和 `--config` CLI。 | 在接图像和模型前，先让 Runtime 行为配置化。 | 解析输入尺寸、类别、阈值、backend 并打印稳定摘要。 | CLI 参数错误成为第一类可用 smoke-test 失败信号。 |
| P1-03 | 新增 OpenCV 读图和 YOLO 风格 preprocess。 | 把真实图片转换成模型可吃的 tensor 格式。 | 打印 `original_size`、`scale`、`padding`、`BGR->RGB`、`[0,1]`、`NCHW`、`1x3x800x800`；CTest 3/3 通过。 | OpenCV Windows pack 需要 `OpenCV_DIR=...\x64\vc16\lib`，运行时还要把 `...\x64\vc16\bin` 放进 `PATH`。 |
| S1-01 | 新增严格 `RuntimeConfig + ModelArtifactSpec`、tensor/枚举校验、按声明文件解析的相对路径，以及 Runtime library/CLI targets。 | 在创建任何 ORT session 前，把模型与 Runtime 假设变成可执行、可测试契约。 | Clean Release 构建、稳定 contract/preprocess 摘要、双 working-directory 路径证明、SHA 复核、ORT SDK gate/DLL 复制，以及 15/15 CTest。 | 必须区分“声明的 hash/配置的 provider”和实际 metadata/provider；CLI 负例同时断言非零退出与错误文本；GTest 等到 S1-04。 |
| S1-02 | 新增 `OnnxRunner` RAII/PImpl、自有 `ModelMetadata`、纯 actual-vs-declared 校验和 `--inspect-model`。 | 在 tensor 接线和算法之前，隔离依赖/session/model contract 错误。 | 真实 ORT 1.19.2 CPU session 加载 `best.onnx`；actual 单输入/单输出 name、float32 shape 和 class channel 通过；真实/synthetic 负例及 29/29 CTest 通过。 | `GetAvailableProviders()` 是 inventory，不是 session assignment；必须分别记录 configured、available 和显式注册的 session provider。使用 profile-free PowerShell 避免 Conda 覆盖 VS 工具链 PATH。 |
| S1-03 | 新增零拷贝 CPU input tensor 接线、同步 `OnnxRunner::run()`、自有 `InferenceOutput` 和 `--raw-output-summary`。 | 在后处理算法前，单独隔离 tensor shape/生命周期与真实模型运行问题。 | 固定图得到 finite `[1,10,13125]` / 131,250-value raw output；错误的 1,919,999-value 输入在 ORT tensor/Run 前失败；31/31 CTest 通过。 | user-buffer `Ort::Value` 不拥有输入 vector，必须让它稳定存活到同步 Run 返回；ORT 只在 Value 存活期间拥有输出，所以返回前必须复制。 |
| S1-04 | 新增 `Detection`/`BoundingBox`、YOLOv8 output 校验/decode、严格阈值、IoU、稳定 class-agnostic NMS、letterbox 还原/clip，以及 `cv::Mat` preprocess 入口和 GTest。 | 在接 CLI/JSON 前，用无 ORT、无真实模型依赖的纯函数边界固定后处理语义。 | 24/24 postprocess GTest、7/7 preprocessor GTest、62/62 完整 CTest；覆盖 float32 阈值边界、空候选、横竖图、奇数 padding 和非正方形输入。 | double 配置阈值必须先转成 float32 再与 float output 比较；同分稳定顺序必须显式定义；NMS 必须发生在坐标还原和 clip 之前。 |
| S1-05 | 新增 `SingleImageDetectionResult`、`DetectorPipeline`、`ResultWriter`、JSON v1、确定性 OpenCV 可视化，以及 `--output-json`、`--output-image`、`--overwrite`。 | 第一次形成可展示、机器可读且保持模块边界的单图片 C++ 纵切。 | 固定图得到 3 个 `crazing` detections；JSON 1,164 bytes、PNG 39,306 bytes，Python/OpenCV 回读通过；result-writer 6/6、output 16/16、完整 CTest 78/78。 | 声明 SHA 不等于逐次重算；provider 是 session 级证据；默认拒绝覆盖并保护输入；JSON/PNG 双文件不是事务，任意 Unicode 路径尚未穷尽验证。 |
| S1-06 | 将 Runtime/artifact、preprocess、metadata、postprocess、output、integration 和核心故障扩展成统一带标签的工程质量 gate。 | 在一致性阶段前，分开证明纯算法正确、真实纵切可运行以及错误可诊断。 | 四个不同像素精确 NCHW、横竖图奇数 padding、synthetic metadata、完整空结果、schema/坏图/不可创建输出故障；clean Release CTest 90/90，5.53 秒。 | 坏 metadata 应用 synthetic struct，不需要多个大模型；合法空 detection 与 malformed/异常 raw output 不同；错误要从 object/path 明确指向 expected、actual 和 action。 |
| S1-07 | 新增固定六类 30 图 manifest、Python ORT CPU reference、确定性 class/IoU matching，以及逐图/汇总 JSON 证据。 | 用类别、confidence、坐标和 IoU 数值证明同一 ONNX 的 Python/C++ 实现一致，取代只比较检测数量的弱证据。 | 索引 241/255/270/285/300 各类 5 张；30/30 图片、62/62 detections 通过；最大 confidence 误差 `8.049977111568296e-07`、最大坐标误差 `9.135351561440075e-05` px、最小 IoU `0.999998927116394`；consistency CTest 2/2、完整 92/92。 | Detection 不能按数组顺序 zip；要按 class 和最大 IoU 确定性匹配。Python 3.9 的 `Path.write_text` 不支持 `newline`，改用 `open(..., newline='\n')`；没有调整门槛。该证据不是精度评估或缺失 `.pt` 下的三方复跑。 |
| S1-08 | 新增 Release-only `BenchmarkRunner`、精确 `Session::Run` 计时、`BenchmarkResult/Writer`、`--benchmark/--warmup/--repeat/--benchmark-json`、严格 JSON validator 与 Windows Peak Working Set。 | 在正确性门通过后，为当前 C++ Runtime 建立独立、可复现、可解释的性能与内存基线。 | 固定图 warmup 10/repeat 100；六段 mean/P50/P95、pipeline `5.696028 images/s`、end-to-end `5.664020 images/s`、Peak Working Set `152.714844 MiB`；先 consistency 2/2，再 benchmark 14/14，完整 CTest 106/106。 | `Session::Run` 与 pipeline 边界不能混写；初始化/写盘/绘图要排除并披露。重复 imread 是 warm-cache；Peak Working Set 是进程生命周期峰值。旧 Python ORT 指标协议不同，不能做无条件快慢结论。 |
| S1-09 | 不新增产品功能；用全新临时 Release build 顺序重跑完整 CTest、固定 Demo、六类 30 图一致性、10/100 benchmark、JSON/PNG 回读、故障和合法空结果，并系统对齐 README。 | 把“某次开发时通过”提升为可从干净环境复现的阶段出口，同时建立用户 L2 面试门。 | CTest 106/106（19.91 秒）；Demo 3 框及固定 JSON/PNG hash；一致性 30/30、62/62；fresh pipeline/end-to-end `7.078853/7.038151 images/s`、Peak Working Set `152.578125 MiB`；四个直接故障 exit 1、两个 empty 测试通过。 | 不能用仓库旧 JSON 造成假阳性；外部命令后必须立即检查 exit code。自动门通过不等于大阶段完成，用户还要完成讲解、追问/排错和一次可回滚的行为+GTest 练习。 |
| PLAN-20260715 | 按最新顶层设计校准仓库规则和双语总入口，并建立大阶段一长版方案。 | 保留已验证 baseline，同时避免大阶段摘要过短而遗漏契约、正确性、测试、故障和证据要求。 | `docs/PLAN.md` -> `AGENTS.md` 准则 -> README 阶段/状态摘要 -> `docs/STAGE1_EXECUTION_PLAN.md` 单步方案。 | 历史 Python 指标、外部 D010 指标和未来 C++ 结果必须明确分开。 |

### 13. 大阶段一 L2 面试收口材料

#### 2 分钟自然口述提纲

我把一个已有的 YOLOv8/NEU-DET ONNX 模型，改造成了面向边缘部署的 C++17 单图推理 Runtime。程序先用 RuntimeConfig 保存阈值和 provider 等运行策略，用 ModelArtifactSpec 保存模型路径、SHA、输入输出、类别和前后处理语义；加载时严格检查 schema，并把声明值与 ORT 从真实模型读到的 ModelMetadata 交叉验证。单图链路是 OpenCV 解码，执行 letterbox、BGR 转 RGB、归一化和 NCHW；OnnxRunner 用 RAII 管理 ORT session，校验输入后借用 vector 创建 CPU Ort::Value，同步 Run，再把输出复制到自持有的 InferenceOutput；后处理按 `[1,4+C,N]` 做类别 argmax、严格 `confidence > threshold`、稳定 class-agnostic NMS、坐标还原和裁剪，最后输出稳定 JSON 与无 GUI PNG。

正确性方面，我用 synthetic GTest/CTest 覆盖契约、已知像素、横竖图奇数 padding、metadata、decode、IoU/NMS、空结果和错误注入；再用六类各 5 张固定图比较 Python ORT 与 C++ ORT，30/30 图片、62/62 detections 在预声明容差内匹配。性能方面，在正确性门通过后，用 Release、batch=1、CPU 单线程、固定单图、warmup 10/repeat 100 测六段 mean/P50/P95，S1-09 fresh end-to-end 平均 142.082777 ms、约 7.038151 images/s，进程 Peak Working Set 152.578125 MiB。它证明的是当前 ONNX 的 C++ 工程闭环和实现一致性，不是模型精度、跨设备性能、INT8 或真实硬件部署。

#### 5 分钟口述结构

1. **定位与问题（约 30 秒）：** 不是重新包装训练脚本，而是解决“有 ONNX 模型”和“有可配置、可测试、可复现的 C++ 部署软件”之间的断层。
2. **工程与契约（约 40 秒）：** 讲 Runtime library/薄 CLI、RuntimeConfig、ModelArtifactSpec、ModelMetadata，以及为什么声明文件相对路径和 actual-vs-declared 校验能防止加载错模型。
3. **前处理（约 40 秒）：** 讲 `image path -> cv::Mat -> letterbox -> RGB -> float32/255 -> NCHW vector`，以及 scale/padding 如何服务坐标逆变换。
4. **ORT 与生命周期（约 45 秒）：** 讲 PImpl/RAII、显式 CPU session、metadata gate、输入 Ort::Value 借用 vector 到同步 Run 返回，以及输出为什么必须复制。
5. **后处理与输出（约 45 秒）：** 讲无独立 objectness 的 BCN decode、类别 argmax、严格阈值、IoU、稳定 class-agnostic NMS、先 NMS 后 restore/clip，以及 JSON/PNG 的稳定和安全输出策略。
6. **测试和故障（约 40 秒）：** 区分 pure synthetic unit、真实模型 integration 和 CLI negative；举 schema、metadata、坏图、不可写输出及合法空 detection。
7. **正确性和性能证据（约 35 秒）：** 讲 30 图按 class/最大 IoU 匹配及冻结容差，再讲六段 benchmark 边界、warmup、P50/P95、throughput 和 Peak Working Set。
8. **限制和后续（约 25 秒）：** 说明不是 mAP、不是 `.pt` 三方复跑、provider 不是逐节点证据、benchmark 是单机 warm-cache；大阶段二继续 INT8 PTQ 与 FP32/INT8 证据加固。

#### 面试追问与口述要点

1. **为什么 RuntimeConfig 与 ModelArtifactSpec 要拆开？** 前者是可调运行策略，后者是模型固有身份和 tensor 契约；调阈值不应被误认为模型 artifact 改变。
2. **为什么还需要 ModelMetadata？** ArtifactSpec 是声明，ModelMetadata 是 ORT 从实际 ONNX 读取的值；交叉校验能在 Run 前发现加载错模型或 name/shape/dtype 不匹配。
3. **为什么相对路径不依赖 current working directory？** config 到 artifact、artifact 到 model 都相对各自声明文件，确保从仓库根、build 目录或其他目录启动仍定位同一资源。
4. **为什么前处理保存 scale 和四边 padding？** 模型框位于 letterbox 空间，必须减 padding 再除 scale 才能还原原图坐标；奇数 padding 还要求明确左右/上下分配。
5. **为什么输入 Ort::Value 可以借用 vector，输出却要复制？** 同步 Run 返回前输入 vector 地址有效；局部 ORT output 析构后其数据指针会失效，所以返回前复制到自持有 vector。
6. **输出 10 个 channel 是什么？** 4 个 xywh 参数加 6 类分数；当前导出没有独立 objectness，也不额外 sigmoid。
7. **为什么 NMS 使用 stable sort？** 分数相同时保留原候选顺序，形成确定性 tie-break，使 JSON、测试和重复 benchmark 可稳定复现。
8. **一致性为什么不能按数组下标比较？** 两端检测顺序可能不同；工具先按 class_id 分组，再按最大 IoU 和 canonical value tie-break 一对一匹配同一目标。
9. **一致性通过为什么不等于模型准确率高？** 它证明同一 ONNX 的两套实现一致；mAP 需要与人工标注比较，是不同评估问题。
10. **为什么 benchmark 前必须先过 consistency？** 性能只说明快慢，不能证明结果正确；正确性失败时发布延迟没有意义。
11. **`Session::Run`、pipeline 和 end-to-end 有什么区别？** Run 只测 ORT；pipeline 还包含前处理、tensor/输出校验复制和后处理；end-to-end 再包含图片解码。
12. **actual provider 与 Peak Working Set 的证据边界是什么？** actual provider 是 session 级 CPU EP 证据，不是逐节点 profiling；Peak Working Set 是整个进程生命周期峰值，不是单次推理独占内存。
13. **为什么历史 Python benchmark 不能和 C++ 数字直接比？** 语言、provider、图片集合、预热次数和计时边界不同；只能分栏保留，不能无条件声称谁更快。
14. **MIT、ONNX AGPL metadata 和数据集许可为什么分开？** 它们分别描述源码、模型 artifact 和数据集分发，不能在没有来源证据时互相覆盖。

#### 至少三个错误排查案例

1. **`provider expected [cpu], actual cuda`：** 这是 schema 加载失败，session 尚未创建，更不表示 GPU 被实际使用。先定位声明文件/行号/字段；当前 v1 只支持 CPU，恢复 `cpu`，未来扩展 CUDA 必须同步 SDK、session 注册、actual provider 和测试。
2. **声明输入 `[1,3,800,800]`，实际 metadata `[1,3,640,640]`：** 先用 `--inspect-model` 核对 model path、SHA、name、shape 和 dtype，判断是否误换 ONNX。不能只修改 artifact 掩盖错误；应使用匹配契约或重新导出并重新做一致性。
3. **文件存在但 OpenCV 报 damaged/empty image：** “路径存在”不等于“codec 能解码”。用 image probe/`imread` 检查文件内容和格式，再确认 preprocess 输入必须是非空 `CV_8UC3`。
4. **输出父路径是普通文件：** 程序无法在普通文件下创建目录。检查输出路径每一层并更换目录；该错误应 exit 1，不能误报成模型推理失败。
5. **consistency 失败后 benchmark 被阻止：** 先区分 count/class、unmatched box、confidence、bbox 或 IoU 失败，再依次核对 manifest/config/artifact/model SHA、provider/版本、前处理、strict threshold、NMS 和坐标还原；禁止为全绿无依据放宽容差。

#### 简历 bullet 候选

- 基于 C++17、CMake、OpenCV 与 ONNX Runtime 实现 YOLOv8/NEU-DET 单图工业缺陷 Runtime，完成严格 config/artifact schema、ORT RAII session、letterbox/NCHW、YOLO decode/NMS/坐标还原、稳定 JSON/可视化，并以 GTest/CTest 覆盖核心算法与故障注入。
- 建立可复现正确性与性能证据：六类 30 图 Python ORT/C++ ORT 共 62 个 detection 的数量/类别完全一致，最大 confidence 误差 `8.05e-7`、最大坐标误差 `9.14e-5 px`；记录 Release CPU 单图六段 P50/P95、约 `7.04 images/s` end-to-end 与 `152.58 MiB` 进程峰值内存。

第二条必须保留“同一 ONNX 实现一致性”和“固定单图 CPU 协议”的语境，不能改写成 mAP、通用 FPS 或真实设备部署结果。

#### 核心代码练习清单

- `cpp_infer/src/image_preprocessor.cpp:57` 与 `:108`：RGB float HWC 展平到 NCHW、letterbox/normalize；对应 `cpp_infer/tests/preprocessor_mat_test.cpp:47`、`:82`、`:108`。
- `cpp_infer/src/onnx_runner.cpp:338`、`:402`、`:419`：PImpl/RAII、CPU Ort::Value、同步 Run、输出 ownership 和纯 Run 计时；输入/输出校验位于 `:209`、`:245`。
- `cpp_infer/src/postprocessor.cpp:269`、`:280`、`:311`、`:360`、`:405`、`:450`：xywh/xyxy、IoU、BCN decode、NMS、restore/clip 和完整顺序；对应 `cpp_infer/tests/postprocessor_test.cpp:173`、`:305`、`:328`、`:355`、`:393`、`:411`。
- `cpp_infer/src/model_metadata.cpp:149`：actual-vs-declared metadata 总校验入口。
- `cpp_infer/tools/compare_consistency.py:619`、`:801`、`:922`、`:1435`：Python reference preprocess/postprocess、确定性 matching 和 30 图编排。
- `cpp_infer/src/benchmark_result.cpp:66` 与 `cpp_infer/src/benchmark_runner.cpp:327`、`:371`、`:520`：mean/P50/P95、六段计时、重复结果稳定性和 warmup/repeat；对应 `cpp_infer/tests/benchmark_test.cpp:31`。
- 需要读懂但不优先手写：`cpp_infer/src/key_value_parser.cpp:76`、`:281`，`cpp_infer/src/detector_pipeline.cpp:135`，`cpp_infer/src/result_writer.cpp:764`、`:856`，以及薄 CLI `cpp_infer/src/main.cpp:631`。

#### 用户必须完成的 L2“核心行为 + 对应 GTest”练习

练习目标是在**一次性练习分支**中，临时把 score filter 从严格 `confidence > threshold` 改成 inclusive `>=`，完成 RED -> GREEN -> 恢复。必须先为当前 S1-09 状态建立用户确认的 checkpoint；工作区有未提交改动时不要直接做恢复操作，练习分支绝不能合并回产品分支。

1. **RED：** 先修改 `cpp_infer/tests/postprocessor_test.cpp:173` 的 synthetic 阈值用例，使 `0.25` 在 threshold=0.25 时也应保留；不改产品代码。先运行 `cmake --build $BuildDir --target yolo_defect_postprocess_tests` 重新编译测试，再运行 `yolo_defect_postprocess_tests.exe --gtest_filter='YoloDecodeTest.*Threshold*'`，应看到测试失败。
2. **GREEN：** 在练习分支临时修改 `cpp_infer/src/postprocessor.cpp:344` 为 inclusive 边界，并同步处理 `postprocessor_test.cpp:192` 与 `:456` 受影响的 exact-threshold 用例；重新构建同一测试 target 后，聚焦 GTest 应以退出码 0 通过。
3. **口述：** 说明为什么边界会改变 detection count、为什么可能破坏 Python/C++ consistency，以及产品协议变化为何要同步 Python reference、manifest 语义和 README。
4. **恢复：** 切回 S1-09 checkpoint，不合并练习分支；确认产品仍是严格 `>`，重新构建测试 target，原测试仍证明等于阈值会被拒绝，再运行聚焦 GTest 和完整 CTest。

只有用户能独立完成 2/5 分钟讲解、至少 10 个追问、至少 3 个错误案例，以及上述 RED/GREEN/恢复练习，并且恢复后完整自动门仍通过，才能把“大阶段一尚未完成”改成“大阶段一完成”。

## 项目亮点

- **Python ORT/C++ ORT 严格一致性已验证** — 固定六类 30 图、62/62 detections 在 count/class exact、confidence `<=1e-4`、bbox `<=1e-2 px`、IoU `>=0.999` 门槛下全部通过
- **C++ Release 性能与内存证据已 fresh reproduction** — S1-09 固定 CPU/线程/模型/单图 10/100 协议下，pipeline/end-to-end 为 **7.078853/7.038151 images/s**，Peak Working Set 为 **152.578125 MiB**；S1-08 旧数值另作历史记录，不做无条件快慢结论
- **C++ 单图纵切与质量 gate 已验证** — 固定样本完成 contract -> OpenCV -> ORT CPU -> YOLO postprocess -> JSON/PNG；3 个 `crazing` detections，S1-06 的 90 项 schema/算法/integration/故障测试继续保留
- **当前最佳实验结果** — 当前最佳模型 `final_train_2` 达到 **mAP@0.5 = 0.743**
- **历史 PyTorch / ONNX 数量抽查** — **50/50** 数量一致、总检测数 **146 vs 146**；但排序样本全是 `crazing`，没有证明类别/框坐标容差
- **历史 V1 Python Benchmark** — PyTorch CPU **8.43 FPS**；PyTorch GPU（RTX 3060）**110.8 FPS**；Python ORT CPU **24.4 FPS**；Python ORT GPU **72.1 FPS**，均为 100 张计时图片（5 张预热），不是 C++ 结果
- **Docker 已验证** — `python:3.9-slim` 镜像已成功跑通 `/health` 和 `/detect`
- **克隆即用** — 数据集（28MB）已包含在仓库内，无需额外下载

## 关键指标

| 指标 | 当前结果 |
|------|----------|
| 最佳模型 | `final_train_2` |
| mAP@0.5 | **0.743** |
| mAP@50-95 | **0.388** |
| 历史 PT/ONNX 检测数量一致率 | 全为 `crazing` 的 **50 / 50**（**100%**，只比较数量） |
| 当前 Python ORT/C++ ORT 固定样本一致性 | **30/30 图片、62/62 detections**；最大 confidence 误差 `8.049977111568296e-07`，最大 bbox 误差 `9.135351561440075e-05 px`，最小 IoU `0.999998927116394` |
| 平均检测框数差值 | **0.000** |
| 历史 PyTorch CPU 基准测试 | **8.43 FPS** / **118.66 ms** 每张 |
| 历史 PyTorch GPU 基准测试（RTX 3060） | **110.8 FPS** / **9.0 ms** 每张 |
| 历史 Python ORT CPU 基准测试 | **24.4 FPS** / **40.9 ms** 每张 |
| 历史 Python ORT GPU 基准测试（RTX 3060） | **72.1 FPS** / **13.9 ms** 每张 |
| S1-09 fresh C++ ORT CPU pipeline | **7.078853 images/s** / **141.265814 ms mean**，P50/P95 **144.4673/149.8395 ms**，固定单图 10/100 协议 |
| S1-09 fresh C++ ORT CPU end-to-end | **7.038151 images/s** / **142.082777 ms mean**，P50/P95 **145.3222/150.7653 ms** |
| S1-09 fresh C++ 进程 Peak Working Set | **152.578125 MiB**，进程生命周期作用域 |
| 历史模型大小记录（`best.pt` / 当前 `best.onnx`） | ~6.0 MiB / ~11.8 MiB；当前工作区和 Git 历史中未找到匹配的 `.pt` |

历史 Python 行与当前 C++ 行的样本、实现、硬件/provider 和计时边界不同，不应直接排序或据此声称 C++ 比 Python 更快/更慢。

## V1 Python Baseline 快速开始

```bash
# 克隆（数据集已包含，约 28MB）
git clone https://github.com/LiuSiChengGitHub/yolo_defect.git
cd yolo_defect

# 安装依赖
conda env create -f environment.yml
conda activate yolo_defect

# 数据准备（VOC XML -> YOLO TXT）
python scripts/prepare_data.py

# 训练
python scripts/train.py

# 从默认训练输出导出 ONNX
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt

# 用真实验证集样例做推理
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg
```

## 数据集

### NEU-DET：东北大学钢材表面缺陷数据库

**来源：** [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

NEU-DET 数据集由东北大学宋克臣教授团队发布，包含 1,800 张热轧钢带表面灰度图像，是工业缺陷检测领域最常用的公开基准数据集之一。涵盖 6 类典型表面缺陷：

| 类别 ID | 英文名 | 中文名 | 描述 | 检测难度 |
|---------|--------|--------|------|----------|
| 0 | crazing | 龟裂 | 表面细密裂纹网络 | 高（纹理细密，与背景区分度低） |
| 1 | inclusion | 夹杂 | 钢材内嵌入的异物 | 中 |
| 2 | patches | 斑块 | 不规则变色区域 | 中 |
| 3 | pitted_surface | 麻面 | 表面分布的小凹坑 | 中 |
| 4 | rolled-in_scale | 压入氧化铁皮 | 轧制过程中压入表面的氧化皮 | 中 |
| 5 | scratches | 划痕 | 机械接触产生的线性痕迹 | 低（线性特征明显） |

### 数据统计

- **论文 / 官方描述中的总数：** 1,800（每类 300 张）
- **`data/NEU-DET/` 中实际可读取的 JPG 数：** 1,800
- **图片尺寸：** 200 x 200 像素
- **格式：** JPG（标注中标记为 depth=1 灰度，但实际可以作为 3 通道 BGR 读取）
- **`data/images/` 中已生成的 YOLO 图片划分：** 训练集 1,439 张，验证集 361 张
- **数据集已包含在 `data/NEU-DET/` 目录中**

### 原始数据目录结构

```
data/NEU-DET/
├── train/                         # 训练集（可读取图片 1,439 张）
│   ├── annotations/               # VOC XML 标注（扁平目录，所有类混在一起）
│   │   ├── crazing_1.xml          #   文件名格式：{类名}_{编号}.xml
│   │   ├── inclusion_1.xml
│   │   ├── rolled-in_scale_1.xml  #   注意：类名含连字符！
│   │   └── ...
│   └── images/                    # JPG 图片（按类名分子目录）
│       ├── crazing/               #   文件名格式：{类名}/{类名}_{编号}.jpg
│       │   ├── crazing_1.jpg
│       │   └── ...
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/                    # 验证集（XML 361 个，可读取图片 361 张），结构同 train
    ├── annotations/
    └── images/
```

> **注意设计上的不对称：** annotations 是扁平目录（所有类的 XML 混在一起），而 images 按类名分子目录。这种不对称在数据准备脚本中需要特殊处理。

### 标注格式说明

原始数据使用 **VOC XML** 格式（源自 Pascal VOC 目标检测挑战赛）。每张图对应一个 XML 文件：

```xml
<annotation>
    <size>
        <width>200</width>            <!-- 图片宽度 -->
        <height>200</height>           <!-- 图片高度 -->
        <depth>1</depth>               <!-- 通道数（灰度=1） -->
    </size>
    <object>                           <!-- 一个标注目标（可以有多个 object） -->
        <name>crazing</name>           <!-- 类别名称 -->
        <bndbox>
            <xmin>2</xmin>             <!-- 左上角 x 坐标（像素） -->
            <ymin>2</ymin>             <!-- 左上角 y 坐标（像素） -->
            <xmax>193</xmax>           <!-- 右下角 x 坐标（像素） -->
            <ymax>194</ymax>           <!-- 右下角 y 坐标（像素） -->
        </bndbox>
    </object>
    <!-- 一张图可能有多个 <object>，如 rolled-in_scale 常有 2-3 个 bbox -->
</annotation>
```

> **格式说明：** VOC 格式用绝对像素坐标的角点表示 (xmin, ymin, xmax, ymax)，而 YOLO 格式用归一化的中心点+宽高 (cx, cy, w, h)。这是从标注文件转换到训练输入时最核心的差异。

## 数据准备

### 转换说明

`prepare_data.py` 将 VOC XML 标注转换为 YOLO TXT 格式（Ultralytics YOLOv8 要求的输入格式）。

**VOC 格式**（绝对像素坐标，角点表示）：
```
xmin, ymin, xmax, ymax  →  例如: 2, 2, 193, 194
```

**YOLO 格式**（归一化中心坐标）：
```
class_id cx cy w h  →  例如: 0 0.487500 0.490000 0.955000 0.960000
```

**归一化转换公式：**
```
cx = (xmin + xmax) / 2 / image_width    # 中心点 x，归一化到 0-1
cy = (ymin + ymax) / 2 / image_height   # 中心点 y，归一化到 0-1
w  = (xmax - xmin) / image_width        # 宽度，归一化到 0-1
h  = (ymax - ymin) / image_height       # 高度，归一化到 0-1
```

> **为什么要归一化？** 归一化后坐标与图片分辨率无关。训练时 YOLO 会把 200x200 的原图 resize 到 640x640，归一化坐标会自动适配，不用手动调整标签值。

### 类别映射

| 类别名称 | 类别 ID | 说明 |
|----------|---------|------|
| crazing | 0 | 顺序固定，与 data.yaml 中的 names 对应 |
| inclusion | 1 | |
| patches | 2 | |
| pitted_surface | 3 | |
| rolled-in_scale | 4 | 注意：名字含连字符，不能用下划线分割提取类名 |
| scratches | 5 | |

### 运行

```bash
python scripts/prepare_data.py
# 或指定自定义路径：
python scripts/prepare_data.py --data-root data/NEU-DET --output-dir data
```

### 输出目录结构

```
data/
├── images/
│   ├── train/          # 扁平目录，所有训练图片（从子目录复制过来）
│   └── val/            # 扁平目录，所有验证图片
├── labels/
│   ├── train/          # YOLO TXT 标签（每张图一个 .txt，与图片同名）
│   └── val/
└── data.yaml           # YOLO 数据集配置文件
```

> **YOLO 对目录的要求：** `images/` 和 `labels/` 必须平级，且文件一一对应（`crazing_1.jpg` ↔ `crazing_1.txt`）。所以必须把按类名分的图片"拍平"到一个目录里。

### 踩坑注意事项

- 数据集**已经预划分**好训练集/验证集，不需要也不应该自己做随机划分
- `rolled-in_scale` 类名包含连字符 `-`，如果用 `filename.split('_')[0]` 提取类名会得到 `rolled-in`（错误！）。正确做法是用已知类名列表做前缀匹配，按长度从长到短排序确保最长匹配优先
- 图片必须从按类名分的子目录复制到扁平输出目录（YOLO 格式的硬性要求）
- 如果你手动更新了原始数据集，请重新执行 `prepare_data.py`，让 `data/images/` 和 `data/labels/` 与 `data/NEU-DET/` 保持一致

## 数据分析

对转换后的数据集运行 `data_analysis.py` 可得出以下结论：6 个类别整体上仍然分布均衡，无需过采样或类别加权。所有图片均为 200×200 px。每张图的 bbox 数量在 1 至 9 个之间（均值 2.33），目标密度适中。Bbox 尺寸差异极大——从 8×9 px 的细长划痕到近 199×199 px 的大面积裂纹——是一个多尺度检测的挑战性场景。YOLOv8 的 anchor-free 设计无需手动设置 anchor，天然适合处理这种宽泛的尺寸分布。分析图表已保存至 `docs/assets/`。

```bash
python scripts/data_analysis.py
```

## 训练

### 运行训练

```bash
# 方式一：通过 YAML 配置文件（推荐，实验可追溯）
python scripts/train.py --config configs/train_config.yaml

# 方式二：通过 Ultralytics CLI（快速实验）
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 超参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolov8n.pt` | 模型变体。`n`=nano（最快），`s`/`m`/`l`/`x` 依次增大 |
| `data` | `data/data.yaml` | 数据集配置文件，定义路径和类名 |
| `epochs` | 50 | 总训练轮数。太少欠拟合，太多过拟合 |
| `imgsz` | 640 | 输入图片尺寸。原图 200x200 会被 resize 到 640x640 |
| `batch` | 16 | 批大小。更大 = 梯度更稳定，但需要更多显存 |
| `lr0` | 0.01 | 初始学习率。训练过程中会按 schedule 自动衰减 |
| `optimizer` | `auto` | 优化器。auto 会根据模型自动选择 SGD 或 AdamW |
| `mosaic` | 1.0 | Mosaic 数据增强概率。4 张图拼成 1 张 |
| `mixup` | 0.0 | Mixup 数据增强概率。两张图按比例混合 |
| `device` | 0 | CUDA 设备编号。`cpu` 则用 CPU 训练 |
| `workers` | 8 | 数据加载的工作进程数 |

### 训练流程详解

1. **加载预训练权重（迁移学习）**
   - YOLOv8n 使用在 COCO 数据集（80 类、33 万张图）上预训练的权重
   - 骨干网络（Backbone）已经学会了通用的特征提取能力（边缘、纹理、形状等）
   - 我们只需要在 NEU-DET 上微调（Fine-tune），让模型学习钢材缺陷的特定特征

2. **数据增强（在线增强，不额外占磁盘）**
   - **Mosaic**：把 4 张图拼成一张，每张占一个象限。好处是一次看到更多目标，提升小目标检测
   - **Mixup**：两张图按随机比例混合叠加，起正则化效果
   - **随机翻转**：水平/垂直翻转，增加数据多样性
   - **HSV 调整**：随机调整色相、饱和度、亮度，增强对光照变化的鲁棒性
   - **尺度抖动**：随机缩放输入图片，让模型适应不同大小的目标

3. **多尺度训练**
   - 训练时随机调整输入尺寸（如 480-640-800），让模型在不同分辨率下都能检测
   - 推理时固定为设定的 imgsz

4. **自动保存检查点**
   - `best.pt`：验证集上 mAP 最高的那个 epoch 的权重（用于最终评估和部署）
   - `last.pt`：最后一个 epoch 的权重（用于断点续训）
   - 保存路径：`runs/detect/train/weights/`

## 实验结果

### 实验对比

| 实验 | 模型 | imgsz | lr0 | epochs | mAP@0.5 | mAP@50-95 | 训练时间 | 备注 |
|------|------|-------|-----|--------|---------|-----------|----------|------|
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | 9.4 分钟 | 默认配置，已超过 0.70 目标 |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.733 | 0.391 | 7.2 分钟 | 更快，但 hardest classes 明显下降 |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.742 | 0.385 | 13.4 分钟 | `optimizer=auto` 家族里最好的图片尺寸结果 |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | 9.0 分钟 | 固定 SGD 后最好的严格指标结果 |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | 13.6 分钟 | `mixup=0.1` 没有带来提升 |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.740 | 0.387 | 13.3 分钟 | 去样本混合增强对照组 |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | 26.1 分钟 | 单纯拉长训练并没有变更优 |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | 25.9 分钟 | 手动组合最优候选后的当前最佳 `mAP@0.5` |

\* 本次训练中 `optimizer=auto` 自动选择了 `AdamW(lr=0.001)`，所以 `lr0=0.01` 不是实际生效学习率。

### 当前模型候选

- **`final_train_2`**：如果主打 `mAP@0.5`，这是当前最适合作为部署主模型的 checkpoint
- **`exp3_lr01`**：如果想强调更严格的 `mAP@50-95` 和更干净的 lr 对照设计，它仍然很重要
- **`final_train`**：它证明了一个关键点，单纯增加 epoch 并不会自动得到更好的最终模型

### 各类 AP（当前最佳：`final_train_2`）

| 类别 | AP@0.5 | Precision | Recall |
|------|--------|-----------|--------|
| patches | 0.920 | 0.856 | 0.850 |
| inclusion | 0.827 | 0.773 | 0.742 |
| pitted_surface | 0.807 | 0.821 | 0.701 |
| scratches | 0.803 | 0.602 | 0.843 |
| rolled-in_scale | 0.553 | 0.507 | 0.462 |
| crazing | 0.550 | 0.513 | 0.543 |

### 对比结论

- `imgsz=800` 对整体 `mAP@0.5` 方向是有帮助的，但它本身并没有解决 `crazing`
- 固定 SGD 的学习率对比说明，在当前项目里 `lr0=0.01` 明显优于 `0.001`
- `mixup=0.1` 不适合这个依赖细纹理的工业缺陷任务，去掉样本混合增强更稳
- 手动组合出的 `final_train_2` 成为了当前 `mAP@0.5` 最好的模型，并把 `crazing` 提升到了 `0.550`
- 最实用的经验不是"训练更久一定更好"，而是"要先把参数组合设计对，再给更长训练预算"

### 训练曲线

![Training Results](docs/assets/results_final_train_2.png)

### PR 曲线（当前最佳）

![PR Curve](docs/assets/PR_curve_final_train_2.png)

### 混淆矩阵（当前最佳）

![Confusion Matrix](docs/assets/confusion_matrix_final_train_2.png)

### 预测样例

![Validation Predictions](docs/assets/val_pred_sample_final_train_2.jpg)

## ONNX 部署

### 为什么选择 ONNX？

ONNX（Open Neural Network Exchange）是微软和 Facebook 联合推出的开放神经网络格式：

- **跨平台** — 无需安装 PyTorch，Windows/Linux/macOS/边缘设备均可运行
- **框架无关** — 推理时不依赖训练框架，部署环境只需要轻量的 ONNX Runtime
- **性能优化** — ONNX Runtime 提供硬件加速（CUDA, TensorRT, DirectML），推理速度通常优于原生 PyTorch
- **体积更小** — 不用打包整个 PyTorch 运行时，部署镜像更小

> **部署取舍：** 不直接使用 PyTorch 进行交付，主要是因为 ONNX Runtime 体积更小、依赖更轻，也更适合跨平台部署和边缘场景。

### 导出命令

```bash
# Quick Start 默认导出路径：来自 `scripts/train.py` 的默认训练输出
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
# 输出: models/best.onnx
```

如果希望复现 README 中展示的最佳指标，请改为导出最佳实验 checkpoint：

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800
```

### 推理命令

```bash
# 单张推理
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg

# 批量推理（整个目录）
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

当前导出的 ONNX 模型使用 `imgsz=800`，因此模型输入是 `[1, 3, 800, 800]`，原始输出张量是 `[1, 10, 13125]`（`4 个框参数 + 6 个类别分数`，覆盖全部候选位置）。

### 性能对比

| 检查项 | 数值 | 证据来源 |
|--------|------|----------|
| 历史最佳 PyTorch 验证结果 | **mAP@0.5 = 0.7433**，**mAP@50-95 = 0.3880** | `docs/archive/experiment_log.md` |
| 历史 PyTorch CPU 基准测试 | **8.43 FPS**，**118.66 ms/张**，共 **100** 张计时图片 | `results/pytorch_benchmark_100.json` |
| 历史 PyTorch GPU 基准测试（RTX 3060） | **110.8 FPS**，**9.0 ms/张**，共 **100** 张计时图片 | `results/pytorch_benchmark_gpu.json` |
| 历史 Python ORT CPU 基准测试 | **24.4 FPS**，**40.9 ms/张**，共 **100** 张计时图片 | `results/onnx_benchmark_cpu.json` |
| 历史 Python ORT GPU 基准测试（RTX 3060） | **72.1 FPS**，**13.9 ms/张**，共 **100** 张计时图片 | `results/onnx_benchmark_gpu.json` |
| S1-08 历史 C++ ORT CPU Release baseline | pipeline **5.696028 images/s**，end-to-end **5.664020 images/s**；单张固定图 warmup 10/repeat 100 | `cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json` |
| S1-09 fresh C++ ORT CPU Release reproduction | pipeline **7.078853 images/s**，end-to-end **7.038151 images/s**；单张固定图 warmup 10/repeat 100 | 本次全新 `%TEMP%` build 的 `s1_09_closure/benchmark/yolov8_neu_det_cpu_release.json` |
| 历史 PT / ONNX 检测数量一致率 | 全为 `crazing` 的 **50 / 50**（**100%**，只比较数量） | `results/pt_onnx_compare/compare_50_summary.json` |
| 历史 PT / ONNX 总检测数 | **146 vs 146** | `results/pt_onnx_compare/compare_50_summary.json` |
| 历史平均绝对数量差 | **0.000** | `results/pt_onnx_compare/compare_50_summary.json` |
| 模型大小记录 | 历史 `best.pt = 6,286,072 bytes`；当前已跟踪 `best.onnx = 12,336,935 bytes`；当前工作区和 Git 历史中未找到匹配 `.pt` | artifact/证据审计 |

历史 PyTorch/Python ORT 行与 C++ Runtime 行来自不同协议：历史行使用 100 张计时图片与 5 次预热，S1-08/S1-09 C++ 使用一张固定图片、warmup 10/repeat 100，并分开记录 decode/preprocess/Run/postprocess/pipeline/end-to-end。历史 Python、S1-08 C++ 历史基线和 S1-09 fresh reproduction 只能分别保留，不能无条件比较快慢；S1-08 与 S1-09 即使协议相同，也没有锁定整个机器状态。

### YOLODetector 类（`src/detector.py`）

`src/detector.py` 封装了完整的 ONNX 推理流程，三步 API 设计：

1. **`preprocess(image)`** — 图片预处理
   - BGR → RGB（OpenCV 读的是 BGR，模型期望 RGB）
   - **Letterbox resize**：等比缩放后用灰色（114,114,114）填充至 `800×800`，与 Ultralytics 训练预处理完全对齐
   - 像素值归一化到 0-1（除以 255）
   - HWC → CHW（维度重排，PyTorch/ONNX 的标准）
   - 添加 batch 维度（3维→4维）

2. **`predict(image)`** — 模型推理 + 后处理
   - ONNX Runtime 前向推理
   - 解析输出张量（当前项目 `imgsz=800` 时输出形状 `[1, 10, 13125]`）
   - 置信度过滤（默认 > 0.25）
   - **NMS（非极大值抑制）**：同一目标可能被多个框检测到，NMS 只保留最优框

3. **`draw(image, detections, class_names)`** — 结果可视化
   - 画边界框 + 类名 + 置信度分数

> **NMS 核心流程：**
> 1. 按置信度从高到低排序所有检测框
> 2. 取最高分的框，与其余框逐一计算 IoU（交并比）
> 3. IoU > 阈值的框被抑制（认为检测的是同一个目标）
> 4. 重复直到处理完所有框
>
> 本项目在 `detector.py` 中手动实现了 NMS（不依赖 torchvision）。

该类的设计目的是**复用**：`scripts/inference_onnx.py` 和 FastAPI 服务（`api/app.py`）都直接 `from src.detector import YOLODetector`，推理逻辑只写一份。

另外，`scripts/debug_detector.py` 用于手动展开预处理与 ONNX 前向过程，并打印 5 个关键 shape，适合排查预处理问题。

### FastAPI API 使用

项目现在已经包含一个最小可用的 FastAPI 服务，入口是 `api/app.py`，目前提供两个接口：

- `GET /health`：健康检查，用来确认服务是否启动、模型是否加载成功
- `POST /detect`：上传单张图片，返回检测结果 JSON

启动服务：

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

健康检查示例：

```bash
curl http://127.0.0.1:8000/health
```

示例响应：

```json
{
  "status": "ok",
  "model": "best.onnx",
  "request_stats": {
    "total_requests": 0,
    "avg_response_time_ms": 0.0
  }
}
```

检测请求示例：

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/images/val/crazing_241.jpg"
```

示例响应：

```json
{
  "filename": "crazing_241.jpg",
  "count": 3,
  "image_size": {
    "width": 200,
    "height": 200
  },
  "model": "best.onnx",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45,
  "inference_time_ms": 20.57,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crazing",
      "confidence": 0.4457,
      "bbox": [0.0, 53.68, 176.91, 146.23]
    }
  ]
}
```

使用说明：

- 上传字段名必须是 `file`
- 返回结果是 JSON，不是画框后的图片
- `inference_time_ms` 是服务端模型推理时间，并发场景下客户端总等待时间通常会更长
- 可以用 `scripts/benchmark_api.py` 对 `POST /detect` 做简单压测，统计平均响应时间和 QPS

本地实测结果：

- `GET /health` 已返回 `200 OK`，且 `status=ok`
- `POST /detect` 对 `data/images/val/crazing_241.jpg` 返回 `count=3`
- `scripts/benchmark_api.py` 已提供本地并发压测入口，但原始 benchmark 日志暂未随仓库提交，因此这里不再写 QPS 数字

### Docker 容器化部署

项目当前已经提供最小可用的 Docker 部署方案：

- 基础镜像：`python:3.9-slim`
- 依赖文件：`requirements-api.txt`
- 复制内容：`src/`、`api/`、`models/`
- 暴露端口：`8000`

构建与启动：

```bash
docker build -t yolo-defect-api .
docker run --rm -p 8000:8000 yolo-defect-api
```

快速验证：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \
  -F file=@data/images/val/crazing_241.jpg
```

当前 Docker 本地实测：

- `GET /health` 返回 `status=ok`、`model=best.onnx`
- `POST /detect` 对 `crazing_241.jpg` 返回 `count=3`
- 目前只保留接口级验证结论；Docker benchmark 原始日志暂未随仓库提交

## 与论文仓库的协作方式

本仓库后续与 `paper_detect` 协作，而不是把两个仓库做成重复项目。

| 仓库 | 定位 | 主要职责 |
|---|---|---|
| `paper_detect` | 论文研究主仓库 | 数据划分、baseline、方法改进、消融实验、正式评估、正式 ONNX 导出、PyTorch/ONNX 一致性、Python ORT benchmark、论文图表和表格 |
| `yolo_defect` | 简历作品集与部署工程主仓库 | 稳定 demo、ONNX/Python/C++ 推理、OpenCV 预处理、CMake、GTest、部署 benchmark、FastAPI/Docker、README 和面试展示 |

一句话：`paper_detect` 负责证明“模型为什么更好”，`yolo_defect` 负责证明“模型怎么稳定跑起来”。

两个仓库通过 artifact、benchmark 原始日志、commit/tag、环境说明和结果表格协作，不复制整份代码互相污染。

### Artifact 协议

artifact 不是单独一个模型文件，而是能解释、复现、验证这个模型的一组证据文件：

```text
artifacts/2026-06-20_method_v1/
├── best.pt
├── best.onnx
├── train_config.yaml
├── export_config.yaml
├── input_spec.json
├── class_map.json
├── metrics.json
├── per_class_ap.csv
├── complexity.json
├── compare_pt_onnx.json
├── latency_python_ort.json
└── result_card.md
```

正式训练、正式评估和论文正式 ONNX 导出由 `paper_detect` 负责。本仓库接收完整 artifact，重点消费其中的 `best.onnx`、`input_spec.json`、`class_map.json` 和 `result_card.md`，然后做部署侧验证和 benchmark。

如果本仓库重新导出 ONNX，只能作为作品集复现或 sanity check，不作为论文主结果来源。

### Benchmark 回流

本仓库负责真实运行部署工程实验，并输出 benchmark 原始日志。后续这些日志可以回流到 `paper_detect`，用于生成论文部署表格：

```text
results/cpp_benchmark/
├── 2026-06-20_method_v1_cpp_ort_cpu.json
├── 2026-06-20_method_v1_cpp_ort_gpu.json
└── 2026-06-20_method_v1_consistency.json
```

每份 benchmark 日志必须记录：

- 使用哪个 artifact / 模型；
- 使用哪个 `yolo_defect` commit 或 tag；
- 运行命令；
- 硬件和操作系统；
- ONNX Runtime 版本和 Execution Provider；
- 是否包含 preprocess 和 postprocess/NMS；
- warmup、repeat、mean latency、P50/P90/P99 latency 和 FPS。

只要实验真实运行、日志可追溯、论文中如实描述环境和统计口径，这就是合理的跨仓库实验组织方式。

### Smoke Test

smoke test 是最低成本的冒烟测试，只证明链路能跑通，不证明最终指标最优。

必须保留四类 smoke test：

- train smoke test：在 `paper_detect` 跑一个短训练；
- export smoke test：在 `paper_detect` 用临时权重导出 ONNX；
- Python ORT smoke test：加载 ONNX 并打印输出 shape；
- C++ ORT smoke test：在 `yolo_defect` 编译、加载 ONNX、打印输入输出 shape 和基础 latency。

当前仓库版本约定：

- `v0.1-intern0`：C++ 部署前的稳定实习/简历作品集快照；
- `deploy-cpp`：后续开发 C++ ONNX Runtime、OpenCV、CMake、GTest 和 benchmark 的分支。

## 项目结构

```
yolo_defect/
├── Dockerfile                    # FastAPI 部署镜像
├── AGENTS.md                     # 从最新 PLAN 固化的协作与推进准则
├── README.md                     # 项目说明（英文版）
├── README_zh.md                  # 项目说明（中文版）
├── LICENSE                       # MIT 开源协议
├── requirements-api.txt          # Docker / API 运行时最小依赖
├── requirements.txt              # pip 依赖列表
├── environment.yml               # Conda 环境配置（含 PyTorch + CUDA）
├── .gitignore                    # Git 忽略规则
├── data/
│   ├── data.yaml                 # YOLO 数据集配置（prepare_data.py 自动生成）
│   └── NEU-DET/                  # 原始数据集（28MB，提交到 git）
│       ├── train/                #   训练集 (~240张/类)
│       └── validation/           #   验证集 (~60张/类)
├── scripts/                      # 一次性脚本（命令行运行）
│   ├── prepare_data.py           #   VOC XML → YOLO TXT 格式转换
│   ├── data_analysis.py          #   数据集统计与可视化
│   ├── train.py                  #   训练入口（读取 YAML 配置）
│   ├── evaluate.py               #   模型评估 + PR 曲线 + 混淆矩阵
│   ├── export_onnx.py            #   ONNX 模型导出
│   ├── debug_detector.py         #   中间值打印 / ONNX 输出观察
│   ├── compare_pt_onnx.py        #   PyTorch vs ONNX 50张近似对比
│   ├── benchmark_pytorch.py      #   PyTorch 100张 CPU/GPU FPS 测试
│   ├── benchmark_onnx.py         #   ONNX 100张 CPU/GPU FPS 测试
│   ├── benchmark_api.py          #   POST /detect 并发压测脚本
│   ├── analyze_failures.py       #   误检 / 漏检案例分析
│   ├── select_representative_examples.py # README 代表样本筛选
│   └── inference_onnx.py         #   ONNX 推理（单张 + 批量）
├── src/                          # 可复用模块
│   ├── __init__.py
│   └── detector.py               #   YOLODetector 类（ONNX 推理，FastAPI 复用）
├── api/                          # FastAPI 服务
│   └── app.py                    #   `GET /health` + `POST /detect`
├── cpp_infer/                    # V2 C++ Runtime 工作区
│   ├── CMakeLists.txt            #   Runtime/CLI、固定 GTest FetchContent、GTest/CTest targets
│   ├── README.md                 #   C++ 契约、依赖、命令与证据
│   ├── artifacts/                #   ModelArtifactSpec 声明
│   │   └── yolov8_neu_det.artifact.txt
│   ├── configs/default_config.txt#   RuntimeConfig 策略与 artifact 路径
│   ├── include/yolo_defect_cpp/  #   契约、preprocess/runner/postprocess/benchmark 公共接口
│   │   ├── detection_result.h    #   自有单图输出数据契约
│   │   ├── detector_pipeline.h   #   单图纵切编排边界
│   │   ├── result_writer.h       #   detection JSON/图片输出请求与返回契约
│   │   ├── benchmark_result.h    #   latency/environment/memory 证据结构与统计
│   │   ├── benchmark_runner.h    #   固定 Release benchmark 请求与执行边界
│   │   └── benchmark_writer.h    #   benchmark JSON 与安全输出接口
│   ├── src/                      #   parser、预处理、ORT run、postprocess、benchmark 与薄 CLI
│   │   ├── detector_pipeline.cpp #   contract -> preprocess -> Run -> postprocess
│   │   ├── result_writer.cpp     #   detection JSON、路径策略和 OpenCV 绘图
│   │   ├── benchmark_runner.cpp  #   六段计时、环境与 Peak Working Set 采集
│   │   └── benchmark_writer.cpp  #   benchmark JSON schema、escaping 与安全写盘
│   ├── tools/
│   │   └── compare_consistency.py#   Python ORT reference、确定性 matching 与证据输出
│   ├── tests/                    #   synthetic GTest/CTest、故障、一致性与 benchmark validator
│   │   └── fixtures/consistency_manifest.json # 六类 x 5 固定图片及 SHA-256
│   └── results/
│       ├── demo/                 #   固定 S1-05 JSON/PNG 证据
│       ├── consistency/          #   S1-07 per-image 与 summary JSON
│       └── benchmark/            #   S1-08 C++ Release latency/memory JSON
├── configs/
│   ├── train_config.yaml         # baseline 训练超参数配置
│   └── exp*.yaml                 # 各组实验配置（imgsz / lr / augment / final）
├── models/
│   └── best.onnx                 # 已跟踪的 YOLOv8/NEU-DET P0 artifact
├── docs/
│   ├── PLAN.md                   # 最新顶层设计、推进规则和大阶段
│   ├── STAGE1_EXECUTION_PLAN.md  # 当前大阶段一动态小阶段方案
│   ├── archive/                  # 历史路线与实验文档
│   └── assets/                   # PR 曲线、Demo GIF、分析图表
└── runs/                         # YOLO 训练输出（gitignored）
```

### 设计原则

- **`scripts/`**：一次性脚本，用 argparse 接收参数，从命令行运行。每个脚本独立，做一件事。
- **`src/`**：可复用模块。`detector.py` 同时被推理脚本和 FastAPI 服务 import，避免代码重复。
- **`cpp_infer/`**：V2 C++ 部署工作区，现已承载 Runtime library/薄 CLI、严格 Runtime/artifact 契约、OpenCV 预处理、ORT RAII session、实际 metadata 校验、安全 input tensor、自有 raw output、纯 YOLOv8 postprocess、单图 `DetectorPipeline`、稳定 JSON v1、无 GUI 可视化、S1-06 quality gate、S1-07 固定六类 Python ORT/C++ ORT consistency、S1-08 benchmark/Peak Working Set，以及 S1-09 fresh reproduction；当前完整 CTest 106/106（19.91 秒）。
- **`configs/`**：超参数与代码分离。调参时改配置文件，不用改代码。用 git diff 可以对比两次实验的参数差异。

## 技术栈

| 工具 | 用途 | 版本 |
|------|------|------|
| Python | 编程语言 | 3.9.25 |
| C++ | V2 Runtime 主语言 | C++17 |
| MSVC | 已验证 x64 C++ 编译器 | 19.50.35721.0 |
| PyTorch | 深度学习框架 | 2.0.0 |
| Ultralytics | YOLOv8 训练和推理 | 本机与 artifact metadata 均为 8.4.24 |
| ONNX | 开放神经网络格式 | Python 包 1.19.1；artifact opset 17 |
| ONNX Runtime | Python baseline、Python consistency reference，加 C++ RAII session、metadata/raw inference、单图 pipeline 与精确 Run 计时 | Python 1.19.2 与官方 Windows x64 CPU C++ SDK 1.19.2；S1-07/S1-08 显式使用 `CPUExecutionProvider`；provider 仍只作 session 级证据 |
| OpenCV | Python consistency preprocess、C++ `CV_8UC3` preprocess 与无 GUI detection 可视化/回读 | Python 4.13.0；Windows C++ 4.8.0 x64 vc16；版本差异已写入一致性 summary，冻结门槛没有调整 |
| NumPy | Python consistency 的 float32 NCHW、BCN decode 与数值计算 | 2.0.2 |
| CMake | 当前 C++ 构建系统与 CTest 入口 | 4.1.1-msvc1；Runtime/CLI、GTest、Python detection/consistency/benchmark validator 与 OpenCV image probe 边界清晰 |
| GTest / CTest | 四个 GTest target：postprocess 25、preprocess 7、output 7、benchmark 8；另有 synthetic metadata/inference executable、Python matching/JSON validator、CLI negative 与真实 integration | GTest 官方 v1.17.0 archive，固定 commit `52eb8108c5bdec04579160ae17225d66034bd723` 和 SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`；S1-09 全新 Release 完整 CTest 106/106（19.91 秒）；离线覆盖只使用标准 `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` 和另行校验的源码 |
| Matplotlib | 可视化绘图 | (via ultralytics) |
| FastAPI | REST API 服务 | latest |
| Conda | 环境管理 | — |

## 关键设计决策

### 模型选择

- **YOLOv8 vs YOLOv5：** YOLOv8 是最新一代，架构改进包括 C2f 模块（替代 C3）、Anchor-Free 检测头（不需要预定义锚框）、解耦头（分类和回归分开处理）。同等大小下 YOLOv8 精度更高。
- **为什么 nano (n) 版本：** NEU-DET 只有 1,800 张图，数据集很小。用更大的模型（s/m/l）容易过拟合，且推理速度慢。nano 版本仅 3.2M 参数，在边缘设备上也能实时运行。
- **灵活升级：** 如果 nano 精度不够，改一行配置就能换成 s 或 m，无需改代码。

### 数据集收录

NEU-DET 数据集只有 28MB（远小于 GitHub 的 100MB 单文件限制）。放在仓库里意味着：
- `git clone` 后立刻可以跑，不需要手动下载、注册账号、解压
- 保证完全可复现——每个人用的是完全相同的数据
- 对外验证成本低——几分钟内就能验证结果

### 配置管理

- **可追溯**：每次实验的配置是一个文件，可以 git commit 保存
- **可对比**：用 `diff exp1.yaml exp2.yaml` 直接看两次实验改了什么
- **可复现**：`python train.py --config exp1.yaml` 就能精确重现实验

### 推理模块

- **关注点分离**：推理逻辑不依赖 ultralytics 或 PyTorch，只依赖 ONNX Runtime
- **代码复用**：推理脚本和 FastAPI 服务共用同一份推理代码
- **可测试性**：可以对 detector 类单独写单元测试，不用启动整个训练框架

## 路线图

### V1 Baseline 已完成

- [x] 基线训练与实验记录
- [x] 超参数调优（imgsz / lr / augment 对比）
- [x] 坏样本分析（误检/漏检案例）
- [x] ONNX 导出与 CPU 推理验证
- [x] ONNX 精度对齐（PyTorch vs ONNX）
- [x] FastAPI 服务化（`POST /detect` 上传图片返回 JSON）
- [x] Docker 容器化部署
- [x] Demo GIF 推理演示

### V2 项目1任务队列

V2 队列以 `docs/PLAN.md` 为准。进入每个大阶段前，Codex 先读取当前仓库并生成该大阶段的小阶段方案；每次只执行一个小阶段，随后暂停验收，并根据真实结果复核剩余计划。`docs/STAGE1_EXECUTION_PLAN.md` 是当前必要的长版方案，README 仍是任务、状态和证据总入口。

| ID | 状态 | 任务 | 范围 | 验收标准 |
|----|------|------|------|----------|
| P1-00 | 已完成 | README / AGENTS / C++ 工作区入口 | 冻结 V2 定位、Codex 边界、任务队列和 `cpp_infer/` 骨架 | README/README_zh 说明 YOLO/NEU-DET 是载体、C++ Runtime 是核心；`AGENTS.md` 保护旧资产；`cpp_infer/` 存在但不实现完整推理 |
| P1-01 | 已在 VS Developer Command Prompt 验证 | CMake 骨架 | 建立最小 CMake 工程和可执行目标 | `cpp_infer` 已有最小 C++17 CMake target、可执行目标和 CTest smoke test。Visual Studio 2026 Developer Command Prompt 中 configure/build/run 通过；Visual Studio 多配置构建需要 `ctest -C Debug` |
| P1-02 | 已通过 NMake CTest smoke 验证 | ConfigLoader | 读取 `input_width`、`input_height`、`class_names`、`score_threshold`、`nms_threshold`、`backend` | `cpp_infer/configs/default_config.txt` 会被解析为类型化 `RuntimeConfig`；`yolo_defect_cpp --config ...` 会打印稳定配置摘要；CTest 覆盖 config smoke 路径，但仍不接入 OpenCV、ONNX Runtime、GTest、预处理、后处理、NMS 或 benchmark |
| P1-03 | 已通过 OpenCV CTest smoke 验证 | OpenCV preprocess | 读图、打印 shape/channels、letterbox、BGR 转 RGB、normalize、HWC 转 CHW | `--config ... --image ...` 会读取真实验证图片，并打印原图尺寸、目标输入尺寸、缩放比例、padding、颜色转换、归一化、NCHW tensor shape 和 tensor 元素数量 |
| S1-01 | **已验证，L1 已验收** | Baseline 契约与工程边界 | 严格 Runtime/artifact schema、按声明文件解析路径、Runtime library/CLI targets、可配置 ORT SDK 边界和 CTest 正反例；GTest 继续延后 | Clean Release library/CLI 构建、稳定摘要、路径无关证明、SHA 复核、可行动错误和 15/15 CTest；没有 session/inference |
| S1-02 | **已验证，L1 已验收** | ORT session 与 metadata 校验 | RAII/PImpl session、显式 CPU EP、实际 version/provider/count/name/shape/dtype/class contract 检查与 synthetic validator | `models/best.onnx` 加载成功；实际 float32 `[1,3,800,800] -> [1,10,13125]` metadata 通过；真实/synthetic 负例和 29/29 CTest 通过；没有 `Session::Run` |
| S1-03 | **已验证，L1 已验收** | Tensor 接线与 raw inference | 借用 preprocess vector 构造 CPU ORT tensor，同步运行，校验并复制 raw output 到独立存储 | 固定图得到 finite、自有 `[1,10,13125]` / 131,250-value 输出；错误长度在 Run 前失败；31/31 CTest 通过；没有 decode |
| S1-04 | **已验证，L1 已验收** | YOLO decode/filter/NMS/坐标还原 | 纯函数校验/decode、float32 strict threshold、稳定 class-agnostic input-space NMS、restore/clip；增加 `cv::Mat` preprocess 测试边界和固定 GTest | postprocess 24/24、preprocess 7/7、完整 CTest 62/62；不依赖真实模型证明纯算法 |
| S1-05 | **已验证，L1 已验收** | 端到端 CLI、JSON 与可视化 | `DetectorPipeline` 编排单图纵切，输出稳定 JSON v1 与确定性无 GUI 可视化，并实施父目录/覆盖/保护路径策略 | 固定图 3 个 `crazing`；JSON/Python 与 PNG/OpenCV 回读通过；result-writer 6/6、output 16/16、完整 CTest 78/78；空 detection 是合法 `[]` |
| S1-06 | **已验证，L1 已验收** | 自动化与失败路径 gate | 为严格 Runtime/artifact schema、精确 preprocess layout、synthetic metadata、postprocess、output、integration 和核心故障建立带标签的 GTest/CTest gate | Clean Release 90/90、5.53 秒；unit 51、integration 3、negative 32；缺模型、坏图和不可创建输出均非零并给出可行动错误 |
| S1-07 | **已验证，L1 已验收** | 固定样本 Python ORT/C++ 一致性 | 在相同 artifact/config、显式 CPU provider 与后处理语义下，比较仓库内冻结的六类 30 图 manifest；按 class 与最大 IoU 确定性匹配并输出逐图/汇总 JSON | 30/30 图片、62/62 detections 通过 count/class exact、confidence `<=1e-4`、bbox `<=1e-2 px`、IoU `>=0.999`；consistency 2/2，完整 CTest 92/92；不声称缺失 `.pt` 下的三方复跑 |
| S1-08 | **已验证，L1 已验收** | 可复现 Release benchmark | 用 warmup/repeat 测 image decode、preprocess、仅 `Session::Run`、postprocess、pipeline、end-to-end，并记录环境、throughput 与内存 | 历史固定图 warmup 10/repeat 100；pipeline/end-to-end `5.696028/5.664020 images/s`；Peak Working Set `152.714844 MiB`；先 consistency 2/2，再 benchmark 14/14，完整 CTest 106/106 |
| S1-09 | **自动门 PASS，用户 L2 待完成** | 大阶段一收口 | 不新增产品功能；用全新临时 Release build 重跑完整 gate、Demo、30 图一致性和 benchmark，对齐双语文档，并安排 L2 口述/排错/行为+GTest 练习 | CTest 106/106（19.91 秒）；Demo/JSON/PNG、一致性、benchmark、四个直接故障与两个合法空结果均通过。用户练习完成前大阶段一仍未完成 |

只有用户 L2 门也通过后才宣布大阶段一完成并进入大阶段二；大阶段二固定边界是 P0 证据加固和至少一次 INT8 PTQ，以及 FP32/INT8 正确性、精度、性能与内存对比（QAT 仅在 PTQ 退化明显且时间允许时启动）。大阶段三再按条件选择 P1 扩展。TensorRT 不是无条件任务，项目2 `inference_event` 也是可选桥接，不属于大阶段一验收项。

### P1-01 CMake 骨架命令

P1-01 只建立 C++17/CMake 入口。它刻意不包含 OpenCV、ONNX Runtime、GTest、预处理、后处理或 NMS。

```powershell
# Configure
cmake -S cpp_infer -B cpp_infer\build

# Build
cmake --build cpp_infer\build

# Run：Visual Studio 多配置生成器通常在这里输出可执行文件
.\cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help

# Run：单配置生成器通常在这里输出可执行文件
.\cpp_infer\build\bin\yolo_defect_cpp.exe --help

# Smoke test：Visual Studio 多配置生成器需要指定配置
ctest --test-dir cpp_infer\build -C Debug --output-on-failure
```

2026-06-05 本机验证：在 Visual Studio 2026 Developer Command Prompt 中 configure 和 build 已通过。`ctest --test-dir cpp_infer\build --output-on-failure` 失败原因是 Visual Studio 属于多配置生成器，需要指定配置名。`ctest --test-dir cpp_infer\build -C Debug --output-on-failure` 已通过，`cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help` 能打印 P1-01 skeleton help 文本。

### P1-02 ConfigLoader 命令

P1-02 新增无第三方依赖的 `key = value` 配置解析器和 `--config` CLI 路径。它仍然刻意不接入 OpenCV、ONNX Runtime、GTest、预处理、后处理、NMS 或 benchmark。

```cmd
:: 在 Visual Studio 2026 Developer Command Prompt 中运行
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_02
cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles"
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

预期配置摘要字段：

- `input_width: 800`
- `input_height: 800`
- `class_count: 6`
- `class_names: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`
- `score_threshold: 0.25`
- `nms_threshold: 0.45`
- `backend: cpu`

2026-06-10 本机验证：在 Visual Studio 2026 Developer Command Prompt 中使用 `%TEMP%` 下的 NMake build tree 完成 configure/build/run/CTest。config smoke test 先在 P1-01 skeleton 上因 `Unknown argument: --config` 失败，随后在 ConfigLoader 实现后通过。这是历史证据；当前 schema 与依赖边界请使用上方 S1-01 Quick Start。

### P1-03 OpenCV Preprocess 命令

P1-03 新增 OpenCV 读图和 YOLO 风格 letterbox 预处理。它仍然刻意不接入 ONNX Runtime、推理、后处理、NMS、benchmark 或 GTest。

```cmd
:: 在 Visual Studio 2026 Developer Command Prompt 中运行
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_03
set PATH=D:\01_Base\Tools\opencv\build\x64\vc16\bin;%PATH%

cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles" -DOpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --image data\images\val\crazing_241.jpg

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

预期 preprocess 摘要字段：

- `original_size: 200x200`
- `channels: 3`
- `input_size: 800x800`
- `resized_size: 800x800`
- `scale: 4.000000`
- `padding: left=0, top=0, right=0, bottom=0`
- `color: BGR->RGB`
- `normalization: float32 [0, 1]`
- `layout: NCHW`
- `tensor_shape: 1x3x800x800`
- `tensor_elements: 1920000`

2026-06-13 本机验证：P1-03 smoke test 先在 P1-02 CLI 上因 `--config expects exactly one config file path.` 失败；加入 OpenCV 和 `ImagePreprocessor` 后，configure/build/run/CTest 通过。本机 OpenCV Windows pack 需要使用 `OpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib`；只指向顶层 `D:\01_Base\Tools\opencv\build` 不足以完成 NMake 构建。

### S1-01 契约与构建边界命令

S1-01 使用当前双文件 schema 和全新 Release/NMake 目录。它校验仓库外 ORT C++ SDK 边界，但不会创建 session 或运行推理。

```powershell
$env:ONNXRUNTIME_ROOT = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;' + `
  'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP 'yolo_defect_s1_01'

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --image `
  (Resolve-Path 'data\images\val\crazing_241.jpg').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config `
  (Resolve-Path 'cpp_infer\tests\fixtures\runtime\invalid_provider.txt').Path

ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
(Get-FileHash models\best.onnx -Algorithm SHA256).Hash
```

2026-07-18 本机验证：MSVC 19.50.35721.0/OpenCV 4.8.0 构建出 `yolo_defect_runtime.lib` 和 `yolo_defect_cpp.exe`，把固定的 1.19.2 `onnxruntime.dll` 复制到 CLI 目录，15/15 CTest 在 0.73 秒内通过；preprocess 输出保持不变；`provider = cuda` 以退出码 1 和 expected/actual/action 错误失败；两个 working directory 得到相同 artifact/model 绝对路径；声明 SHA-256 也已重新计算对齐。这些是契约/构建结果，不是 ORT session 或 inference 结果。

### S1-02 ORT Session 与 Metadata 检查命令

S1-02 加载真实 ONNX 并校验实际 metadata，刻意停在构造 input tensor 和调用 `Session::Run` 之前。

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
powershell.exe -NoProfile -NoExit
```

```powershell
$env:ONNXRUNTIME_ROOT = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_02_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --inspect-model
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

2026-07-26 本机验证：ORT runtime 1.19.2 报告可用 provider 为 `[AzureExecutionProvider,CPUExecutionProvider]`；`OnnxRunner` 显式注册 `CPUExecutionProvider` 并成功创建 session。实际输入是 `images` tensor float32 `[1,3,800,800]`；实际输出是 `output0` tensor float32 `[1,10,13125]`；metadata contract 校验通过。29 项 CTest gate 全部通过，其中包括真实的输入尺寸/类别数声明不匹配，以及 synthetic count/name/shape/dtype/provider 失败路径。没有构造 input tensor，也没有执行 inference。

### S1-03 输入 Tensor 与 Raw Inference 命令

S1-03 把现有 preprocess vector 接入一次同步 ORT Run，并把通过校验的 raw output 复制到项目自有存储。它刻意停在 decode、分数过滤、NMS、JSON、可视化和 benchmark 之前。

```powershell
$env:ONNXRUNTIME_ROOT = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_03_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --image $Image `
  --raw-output-summary
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

2026-07-30 本机验证：固定图片生成输入 float32 `[1,3,800,800]`，1,920,000/1,920,000 个值 finite；生成自有 raw output float32 `[1,10,13125]`，131,250/131,250 个值 finite，输出范围 `[0,795.04126]`。错误的 1,919,999-value 路径在构造 `Ort::Value`/Run 前失败，31/31 CTest 通过。这是 raw execution 证据，不是 decoded detection 正确性或性能证据。

### S1-04 YOLOv8 Postprocess 与 GTest 命令

S1-04 不改变 CLI 输出，而是把 `InferenceOutput -> Detection` 后处理实现成无需 ORT session、无需真实模型的纯函数，并通过 synthetic tensor/box/image 固定算法语义。

```powershell
$env:ONNXRUNTIME_ROOT = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;' + `
  'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_04_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir

ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

完全离线 configure 可以在先校验并解压固定 archive 后追加：

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='D:\path\to\verified\googletest-source'
```

固定来源为 GoogleTest v1.17.0 commit `52eb8108c5bdec04579160ae17225d66034bd723`，archive SHA-256 为 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`。source override 不会替你重新校验 archive，必须先执行 `Get-FileHash`。2026-08-15 验证结果为 postprocess 24/24、`cv::Mat` preprocess 7/7、新 GTest 共 31/31、完整 CTest 62/62。S1-04 没有新增 JSON、绘图、benchmark 或一致性工具。

### S1-05 单图 CLI、JSON 与可视化命令

S1-05 用 `DetectorPipeline` 串起已经验证的 contract、preprocess、ORT run 与 postprocess，再由 Runtime library 内的 writer 输出 JSON v1 和无 GUI 可视化；`main.cpp` 只保留 CLI 参数与编排。以下命令重新生成仓库内固定证据，因为文件已存在，所以明确使用 `--overwrite`：

```powershell
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$JsonPath = Join-Path (Get-Location) `
  'cpp_infer\results\demo\crazing_241.detections.json'
$VisualizationPath = Join-Path (Get-Location) `
  'cpp_infer\results\demo\crazing_241.visualized.png'
$PythonExe = (Get-Command python.exe -ErrorAction Stop).Source

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --output-json $JsonPath `
  --output-image $VisualizationPath `
  --overwrite

& $PythonExe -m json.tool $JsonPath
& "$BuildDir\bin\yolo_defect_image_probe.exe" $VisualizationPath
Get-Item $JsonPath, $VisualizationPath
Get-FileHash $JsonPath -Algorithm SHA256
Get-FileHash $VisualizationPath -Algorithm SHA256

ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir `
  -R yolo_defect_cpp_single_image_outputs -V
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

2026-08-16 clean Release 验证：固定 200x200 `CV_8UC3` 图片经 `[1,3,800,800] -> [1,10,13125]` CPU session 得到 3 个 `crazing` detections；Python 标准库 JSON parser 和 OpenCV 200x200 `CV_8UC3` 回读通过。JSON 为 1,164 bytes，SHA-256 `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`；PNG 为 39,306 bytes，SHA-256 `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`。result-writer JSON GTest 6/6、output 聚焦 CTest 16/16、完整 CTest 78/78。

默认不加 `--overwrite` 时，已有输出会以非零退出和可行动错误拒绝；父目录缺失会自动创建，但目录、符号/特殊文件、同一 JSON/图片目标和保护输入路径不能被覆盖。JSON/PNG 是两个文件而非跨文件事务，CLI 也未证明跨进程原子写入；这些边界不能因固定 Demo 成功而省略。

### S1-07 Python ORT/C++ ORT 一致性命令

S1-07 不修改训练或历史 Python 推理资产，而是在 `cpp_infer/` 内建立独立 reference/comparison 工具。Manifest 固定每类验证集索引 241、255、270、285、300，并记录每张图的 SHA-256；Python 与 C++ 都从同一 RuntimeConfig/ModelArtifactSpec 取得模型、输入、类别、阈值和 NMS 语义。Python 解释器刻意写成 TestBase 的绝对路径，避免任意 Conda/base 环境悄悄改变 ORT、OpenCV、NumPy 或 provider；`--cpp-opencv-version 4.8.0` 记录独立核验的 C++ 构建依赖，不会假装它与 Python OpenCV 4.13.0 相同。下面命令会重新生成仓库内机器可读证据：

```powershell
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'
$Manifest = (Resolve-Path `
  'cpp_infer\tests\fixtures\consistency_manifest.json').Path
$ConsistencyDir = (Resolve-Path `
  'cpp_infer\results\consistency').Path

& $PythonExe 'cpp_infer\tools\compare_consistency.py' `
  --manifest $Manifest `
  --cpp-cli "$BuildDir\bin\yolo_defect_cpp.exe" `
  --cpp-opencv-version 4.8.0 `
  --output-dir $ConsistencyDir

& $PythonExe -m json.tool `
  "$ConsistencyDir\per_image.json" > $null
& $PythonExe -m json.tool `
  "$ConsistencyDir\summary.json" > $null
ctest --test-dir $BuildDir -L consistency --output-on-failure
ctest --test-dir $BuildDir --output-on-failure
```

2026-08-22 clean Release 验证：Python 显式 session provider 为 `CPUExecutionProvider`；30/30 图片、62/62 Python/C++ detections 匹配通过。最大 confidence 绝对误差为 `8.049977111568296e-07`，最大 bbox 坐标绝对误差为 `9.135351561440075e-05` 像素，最小 matching IoU 为 `0.999998927116394`。聚焦 consistency CTest 2/2 在 12.58 秒内通过，完整 CTest 92/92 在 17.28 秒内通过。该结果是同一 ONNX 的双实现正确性证据，不是模型精度或新的 PyTorch/ONNX/C++ 三方实验。

### S1-08 C++ Release Benchmark 与内存命令

S1-08 只在 S1-07 正确性门通过后发布性能证据。以下是已有 clean Release build 上重新生成正式 JSON 的最简命令；仓库文件已经存在，因此明确追加 `--overwrite`：

```powershell
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$BenchmarkJson = Join-Path (Get-Location) `
  'cpp_infer\results\benchmark\yolov8_neu_det_cpu_release.json'

ctest --test-dir $BuildDir -L consistency --output-on-failure
if ($LASTEXITCODE -ne 0) {
  throw 'S1-07 consistency 失败；禁止继续发布 benchmark。'
}

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config --image $Image `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json $BenchmarkJson --overwrite

& $PythonExe -m json.tool $BenchmarkJson > $null
& $PythonExe 'cpp_infer\tests\assert_benchmark_json.py' `
  $BenchmarkJson --expected-image $Image `
  --expected-warmup 10 --expected-repeat 100

ctest --test-dir $BuildDir -L benchmark --output-on-failure
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

正式环境为 Windows 10.0.26200、`DESKTOP-6OGK71C`、AMD64 Family 25 Model 117、16 个逻辑 CPU、MSVC 19.50.35721.0 Release C++17、OpenCV 4.8.0 和 ORT 1.19.2。Session 实际使用 `CPUExecutionProvider`，策略为 sequential、intra/inter-op 1/1、graph optimization all。模型为 12,336,935 bytes、输入 `[1,3,800,800]`；固定图片为 23,845 bytes、200x200x3，batch/sample=1，score 0.25、NMS 0.45、class-agnostic，100 次均得到 3 个 detections。

| 段 | Mean ms | P50 ms | P95 ms |
|----|---------|--------|--------|
| Image decode | 0.991129 | 0.9649 | 1.3517 |
| Preprocess (`cv::Mat -> tensor`) | 8.244569 | 7.5514 | 12.1265 |
| `Session::Run` | 165.555859 | 164.8985 | 186.2136 |
| Postprocess | 0.424115 | 0.4251 | 0.5636 |
| Pipeline | 175.560944 | 175.1058 | 195.1376 |
| End-to-end | 176.553060 | 176.1357 | 196.6128 |

Pipeline throughput 为 `5.696028 images/s`，end-to-end throughput 为 `5.664020 images/s`；Windows Peak Working Set 为 `160,133,120 bytes`（`152.714844 MiB`）。正式顺序先通过 consistency 2/2，再通过 benchmark 14/14；`ctest -N` 列出 106 项，完整 CTest 106/106 在 18.44 秒内通过。

Session/model 初始化、初始路径和大小检查、统计计算、Peak Working Set 查询、JSON 序列化/写盘以及绘图均不进入重复计时，绘图没有执行。结果是 warm-cache、单图、单机 CPU baseline；没有 CPU affinity/priority/idle lock，Peak Working Set 是进程生命周期峰值，actual provider 不是逐节点 profiling。历史 Python ORT 24.4/72.1 FPS 协议不同，不能与该 C++ 结果做无条件优劣比较。

### S1-09 大阶段一自动收口复现

S1-09 不修改产品语义，而是在新的 `%TEMP%` NMake/Release build 中，按 configure/build -> 完整 CTest -> Demo -> consistency -> benchmark 顺序重新执行整个大阶段一出口。Quick Start 将所有 S1-09 输出写入 `$BuildDir\s1_09_closure`，每个外部程序返回后立即检查 exit code，并要求目标 JSON 在本次运行前不存在，因此不会因仓库中的旧 S1-05/S1-07/S1-08 证据而假通过。

2026-08-22 fresh reproduction：完整 CTest 106/106 在 19.91 秒内通过。固定 Demo 仍得到 3 个 detections；JSON 为 1,164 bytes、SHA-256 `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`，PNG 为 39,306 bytes、SHA-256 `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`，OpenCV probe 为 200x200 `CV_8UC3`。六类各 5 张的一致性再次通过 30/30 图片和 62/62 matches；最大 confidence 误差 `8.049977111568296e-07`、最大 bbox 误差 `9.135351561440075e-05 px`、最小 matching IoU `0.999998927116394`，本次 `per_image.json` 与 `summary.json` 均通过标准 JSON parser。

| S1-09 fresh 段 | Mean ms | P50 ms | P95 ms |
|-----------------|---------|--------|--------|
| Image decode | 0.816168 | 0.8182 | 0.9251 |
| Preprocess (`cv::Mat -> tensor`) | 5.453755 | 5.4547 | 6.2128 |
| `Session::Run` | 134.419309 | 137.5882 | 142.5549 |
| Postprocess | 0.345302 | 0.3438 | 0.4424 |
| Pipeline | 141.265814 | 144.4673 | 149.8395 |
| End-to-end | 142.082777 | 145.3222 | 150.7653 |

Fresh pipeline/end-to-end throughput 为 `7.078853/7.038151 images/s`，Windows Peak Working Set 为 `152.578125 MiB`。本次临时 benchmark JSON 为 5,453 bytes、SHA-256 `F32C0DF3157897264F9BD2B9AE3F3DB7B240A3B641494E8D3E7C346FF64E9C6F`。这次数字与上面的 S1-08 历史数字使用相同代码协议，但系统状态没有被完全锁定，因此两次证据都保留，不把差异无条件归因于代码优化。

直接执行的缺模型、损坏图片、不可创建输出和 benchmark `repeat=0` 四个故障均 exit 1，并包含可行动信息；合法无候选 postprocess 与 empty-array JSON 两项测试通过。当前状态仍是：**S1-08 L1 已验收；S1-09 自动门 PASS；用户 L2 待完成；大阶段一尚未完成；大阶段二未开始。**

### V2 入口记录

| 日期 | 变更 | 目的 |
|------|------|------|
| 2026-06-04 | 建立 P1-00 V2 入口：README 定位、Codex 边界文件、`cpp_infer/` 骨架 | 在深入 C++ 实现前，先让项目能被解释为工业视觉 AI Runtime 工程项目 |
| 2026-06-05 | 在 Visual Studio 2026 Developer Command Prompt 中验证 P1-01 CMake 骨架 | 确认 configure/build/run/CTest smoke test；记录 Visual Studio 多配置构建需要 `ctest -C Debug` |
| 2026-06-10 | 新增 P1-02 ConfigLoader 和 `--config` smoke 路径 | 引入类型化、无第三方依赖的 runtime 配置解析器，并在进入 OpenCV 预处理前记录 build/run/CTest 证据 |
| 2026-06-13 | 新增 P1-03 OpenCV 读图和 letterbox preprocess smoke 路径 | 在接入 ONNX Runtime 前，确认真实图片预处理输出，包括原图 shape、RGB 转换、归一化、NCHW 布局、scale、padding 和 tensor shape |
| 2026-06-29 | 按当时路线校准项目1主线；该路线现归档为 `docs/archive/路线0628.md` | 记录顶层设计、D010/paper_detect artifact 接入路线、README 必备栏目、阶段队列占位和教学式进度日志，防止后续偏离 C++ Runtime 主线 |
| 2026-07-15 | 将现行路线源替换为 `docs/PLAN.md`，更新 AGENTS 与双语总入口，并新增 `docs/STAGE1_EXECUTION_PLAN.md` | 采用最新九部分教学闭环、权威 P0/P1 边界、artifact 门禁和四个大阶段；确认当前方向未偏，并动态规划 S1-01 至 S1-09 |
| 2026-07-16 | 未开始 S1-01，先完成大阶段一开工前置准备 | 核验 VS x64 终端与 ORT C++ SDK，全新 clean build 通过 3/3 CTest，冻结 SHA-256 固定的 GTest v1.17.0 FetchContent 方案，并在 `docs/PRE_STAGE1_READINESS.md` 记录所有者确认的模型 lineage 与公开分发许可检查点 |
| 2026-07-18 | 完成 S1-01 Runtime/artifact 契约与工程边界 | 新增严格双文件 schema、model/tensor/枚举校验、按声明文件解析的路径、Runtime library/CLI targets、可配置 ORT SDK 校验/DLL 复制和 15 项 CTest 证据；原样保留 AGPL metadata 风险，并停在 ORT session/inference 之前 |
| 2026-07-26 | 完成 S1-02 ORT session 与实际 metadata 校验 | 新增 RAII/PImpl `OnnxRunner`、自持有的 `ModelMetadata`、显式 CPU EP/session 策略、`--inspect-model`、实际值对声明值校验和 29 项真实/synthetic CTest gate；停在构造 input tensor 和 `Session::Run` 之前 |
| 2026-07-30 | 完成 S1-03 input tensor 与自有 raw output 边界 | 新增零拷贝借用 CPU input、同步 `Session::Run`、overflow/shape/count/finite 校验、复制式 `InferenceOutput`、有限 CLI 摘要、错误长度 Run 前失败和 31 项 CTest gate；停在 decode/NMS 之前 |
| 2026-08-15 | 完成 S1-04 纯 YOLOv8 postprocess 与 synthetic GTest 边界 | 新增 `Detection`、BCN decode、float32 strict threshold、稳定 class-agnostic input-space NMS、letterbox restore/clip、`CV_8UC3 cv::Mat` preprocess 入口和固定 GTest 依赖；24+7 项 GTest、62/62 CTest 通过，停在 CLI detection/JSON/可视化之前 |
| 2026-08-16 | 完成 S1-05 固定单图 CLI、JSON v1 与无 GUI 可视化 | 新增自有 detection result、薄 `DetectorPipeline`、安全 JSON/路径/覆盖契约与确定性 OpenCV 绘图；固定图生成 3 个 `crazing` detections、1,164-byte JSON 和 39,306-byte PNG，Python/OpenCV 回读及 78/78 CTest 通过；停在 batch、一致性与 benchmark 之前 |
| 2026-08-22 | 完成 S1-06 自动化质量 gate 与核心故障注入 | 扩展 schema、精确 NCHW、synthetic metadata、完整 postprocess 空结果、输出路径和 CLI 故障证据；全新 Release build 的 90 项命名测试全部通过，缺模型/坏图/不可创建输出均给出可行动非零失败；停在 S1-07 一致性之前 |
| 2026-08-22 | 完成 S1-07 固定六类 Python ORT/C++ ORT 一致性证据 | 冻结仓库内 6x5 manifest 与图片 SHA-256，新增显式 CPU Python reference、顺序无关的 class/最大-IoU matching、逐图/汇总 JSON 和 CTest；30/30 图片、62/62 detections 通过冻结门槛，完整 CTest 92/92；停在 S1-08 benchmark 之前 |
| 2026-08-22 | 完成 S1-08 可复现 C++ Release benchmark 与内存基线 | 在 S1-07 consistency 通过后，新增六段 steady-clock timing、算术 mean 与 nearest-rank P50/P95、pipeline/end-to-end throughput、Windows Peak Working Set、稳定 benchmark JSON 和严格 validator；正式 warmup 10/repeat 100 结果与 106/106 CTest 通过，停在 S1-09 之前 |
| 2026-08-22 | 完成 S1-09 自动收口复现，等待用户 L2 | 不新增产品功能；用全新临时 Release build 依次复现 106/106 CTest、固定 Demo、30 图一致性、10/100 benchmark、四个直接故障和两个合法空结果，并对齐双语总入口。自动门 PASS；用户完成 2/5 分钟讲解、追问/排错及可回滚行为+GTest 练习前，大阶段一仍未完成 |
| 2026-08-23 | 在一次性 L2 练习分支统一 Windows 构建、推理与证据命令 | 将 `stage1.cmd`/`stage1.ps1` 扩展为十个可发现动作（`help/doctor/build/clean-build/test/detect/demo/consistency/benchmark/all`），增加严格的仓库 workflow 默认配置和 Git 忽略的 local 覆盖，并把任意单图检测缩短为来源加可选输出目录。help/doctor、跨 CWD 与特殊字符 detect、可行动负例和最终 fresh `all` 均通过；最终一次通过 106/106 CTest（20.48 秒）、3 框 Demo、30/30 与 62/62 一致性及正式 10/100 benchmark validator。产品推理语义不变，用户 L2 仍待完成 |

## 许可证

仓库作者编写的源代码采用 MIT 许可证，详见 [LICENSE](LICENSE)。该声明不能自动覆盖 `models/best.onnx` 或 NEU-DET 数据集。

当前 ONNX metadata 标注 `AGPL-3.0`。[Ultralytics 官方许可说明](https://www.ultralytics.com/license)称 Ultralytics 训练模型默认使用 AGPL-3.0，除非取得适用的商业许可。[东北大学 NEU 官方数据页](https://faculty.neu.edu.cn/songkc/en/zdylm/263265)提供下载和引用说明，但本次审计没有在该页面找到明确的再分发许可证。这些是 provenance/分发检查点，不是法律结论；详见 [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md)。

当前声明用途是个人学习，因此本地开发不需要把 Enterprise 许可作为前置条件。项目所有者选择 A——继续公开分发 ONNX 与 NEU-DET；非商业意图并不会自动免除分发义务，所以在冻结发布口径前仍须保留模型许可声明并核实数据集的再分发依据。

NEU-DET 数据集由东北大学提供，学术引用请参考：

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.
