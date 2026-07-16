# 大阶段一执行方案：项目 1 可投递闭环

> - 规划来源：`docs/PLAN.md`
> - 生成日期：2026-07-15
> - 当前状态：方案已生成，下一步为 `S1-01`，尚未开始本大阶段的新增实现
> - 前置准备：2026-07-16 已完成开发终端、ORT C++ SDK、clean 3/3 CTest、GTest 固定版本方案与 provenance/license 预审，详见 `docs/PRE_STAGE1_READINESS.md`；这不代表 S1-01 已开始
> - 适用范围：`cpp_infer/`、`README.md`、`README_zh.md`，以及本方案明确允许的 `cpp_infer/` 内测试/工具/证据文件

## 1. 使用方式

这是进入“大阶段一”前，基于当前仓库动态生成的小阶段方案。它不是允许连续自动开发的静态任务清单。

执行规则：

1. 每次只执行一个 `S1-*` 小阶段。
2. 每个小阶段完成代码、测试、双语 README、证据和教学闭环后立即停止。
3. 用户完成 L1 理解与验收后，才进入下一个小阶段。
4. 每次进入下一步前，必须根据刚得到的真实模型输出、依赖、失败和测试结果复核本方案；必要时调整尚未执行的步骤，但不能降低大阶段出口。
5. `docs/PLAN.md` 的项目顶层设计高于本方案；如果本方案与顶层设计冲突，以顶层设计为准并补齐本方案。

为避免概念混淆，本方案使用 `S1-01`～`S1-09` 表示“大阶段一内部小阶段”。README 中已有的 `P1-00`～`P1-03` 是历史“项目 1”任务 ID；`PLAN.md` 中的 P1 则表示 P0 之后的工程扩展，两者不是同一套编号。

## 2. 当前项目是否跑偏

结论：**没有跑偏，可以沿最新顶层设计继续推进。**

当前代码正好停在大阶段一的合理起点：

| 当前事实 | 已有证据 | 与新路线的关系 | 下一缺口 |
|----------|----------|----------------|----------|
| C++17/CMake/CTest 入口 | 全新临时目录用 MSVC 19.50.35721.0/OpenCV 4.8.0 构建，3/3 CTest 通过 | 符合已完成工程骨架 | 目前仍是单 executable，需要拆 Runtime library / CLI / tests target |
| 类型化配置 | `RuntimeConfig` 已解析输入尺寸、类别、阈值和 backend | 符合 config baseline | 缺 model/artifact path、模型族、tensor contract、前后处理类型和枚举校验 |
| OpenCV 预处理 | 真实图片完成 letterbox、BGR→RGB、`[0,1]`、NCHW；手工非正方形 probe `1260×1144 -> 800×726`、top/bottom padding 37 | 符合 P0 preprocess | NEU-DET 原图仍是 200×200，非正方形精确 padding/坐标逆变换必须用 synthetic test 证明 |
| P0 模型 artifact | 已跟踪 `models/best.onnx`，12,336,935 bytes，opset 17，SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` | 可直接作为稳定 YOLO baseline | 尚无 C++ ORT session、raw inference 和模型契约检查；metadata 许可证为 `AGPL-3.0`，发布前需复核与仓库 MIT 的兼容性 |
| 模型 I/O 预检 | Python ORT 1.19.2/CPU 实测：`images` float32 `[1,3,800,800]` → `output0` float32 `[1,10,13125]` | 为 C++ contract 提供真实 expected values | C++ 必须自己读取并验证 actual metadata，不能只相信文档 |
| 历史 PT/ONNX 对比 | 50/50 检测数量一致、146 vs 146 | 只能作为历史弱证据 | 这 50 张全是 `crazing`，只比较数量/置信度摘要；需新增六类、类别与框误差证据 |
| 历史 Python ORT 性能 | CPU 24.4 FPS、GPU 72.1 FPS | 可作为 Python baseline 背景 | 不能当作 C++ Runtime 数据；需全新 Release C++ benchmark |

额外现实约束：

- 2026-07-15 审计时尚未找到可供 CMake/C++ 使用的 ONNX Runtime SDK；2026-07-16 前置准备已在仓库外确认官方 Windows x64 CPU SDK 1.19.2，并验证 header、import library 和 runtime DLL。S1-01 仍须通过可配置 `ONNXRUNTIME_ROOT` 接入并记录 DLL 策略，不能硬编码个人路径。
- 模型 metadata 报告 `AGPL-3.0`，而仓库源码使用 MIT。项目所有者已声明当前用途是个人学习并选择继续公开分发模型与 NEU-DET 数据；个人学习不要求把 Enterprise 许可作为开工条件，但公开分发的模型/数据许可义务仍须分别核对，不能默认模型继承仓库 MIT。
- 项目所有者已确认当前 ONNX 是本人从 `runs/detect/final_train_2/weights/best.pt` 导出的；但当前工作区和 Git 历史都没有该 `.pt`。因此本阶段能直接重跑的是“同一 ONNX artifact 的 Python ORT 与 C++ ORT 严格一致性”；历史 PT/ONNX 结果作为另一段已有证据。除非后续合法恢复匹配 checkpoint，否则不能声称本次重新完成了 PyTorch/Python ORT/C++ 三方直接实跑。
- `cpp_infer/build/` 中可能残留旧 P1-01 构建产物，不能作为当前源码证据。所有验收都使用 `%TEMP%` 下的全新 out-of-tree Release build。

## 3. 顶层设计与大阶段边界对齐

### 大阶段一必须完成

大阶段一不是只做一个 ORT smoke。为了满足阶段出口并补足顶层设计中被摘要省略的内容，本阶段必须交付：

```text
baseline Runtime + artifact contract
-> CMake multi-target and explicit dependencies
-> ORT C++ RAII session and metadata checks
-> preprocess tensor -> real raw inference
-> YOLO decode / strict score filter / NMS / coordinate restore
-> fixed single-image CLI
-> detection JSON + visualization
-> core GTest / CTest and actionable errors
-> six-class Python ORT / C++ ORT consistency
-> reproducible Release benchmark + baseline Peak Working Set
-> README evidence + L2 interview acceptance
```

### 留在大阶段二

- INT8 PTQ 的实际量化、正确性/精度/性能/模型大小比较。
- 只有 PTQ 精度退化明显且时间允许时才考虑 QAT。
- 在大阶段一核心测试基础上扩充更系统的回归、故障注入、样本与性能/内存证据。
- 最终 P0 结果加固、专项 mock 和投递版本冻结。
- D010 只在稳定 ONNX、artifact contract、真实 Runtime adapter 和一致性都具备时进入；不能阻塞 YOLO P0。

### 不进入大阶段一

- 目录批处理、有界队列、worker、backpressure、clean shutdown。
- TensorRT、Jetson、ARM；这些需要真实硬件。
- Qt、gRPC、Triton；这些需要多个高优先 JD 反复要求。
- D-FINE/D010 适配、`inference_event`、服务化、GUI。
- 重写受保护的 Python 训练、评估、FastAPI 或 Docker 资产。

## 4. 总体拆分与关键路径

| ID | 小阶段 | 预计投入 | 前置 | 主要出口 |
|----|--------|---------:|------|----------|
| S1-01 | Baseline 契约、工程边界与依赖准备 | 4–6h | P1-03 | 可校验 contract、多 target、ORT/GTest 接入路径 |
| S1-02 | ORT session 与模型 metadata 校验 | 5–7h | S1-01 | RAII session、真实 provider/name/shape/dtype 证据 |
| S1-03 | 输入 tensor 与真实 raw inference | 3–5h | S1-02 | 固定图片完成 `Session::Run`，raw output 可持有可校验 |
| S1-04 | YOLO decode/filter/NMS/坐标还原 | 6–8h | S1-03 | 纯函数后处理与 synthetic GTest |
| S1-05 | 完整 CLI、Detection JSON 与可视化 | 5–6h | S1-04 | 固定命令生成机器可读与视觉输出 |
| S1-06 | 自动化测试与核心故障注入 | 6–8h | S1-05 | 主链路 gate 与可行动错误矩阵 |
| S1-07 | 固定六类 Python ORT/C++ 一致性 | 5–6h | S1-06 | 预声明容差下的逐图与汇总证据 |
| S1-08 | 分段 Release benchmark 与内存基线 | 5–7h | S1-07 | P50/P95、throughput、Peak Working Set、环境信息 |
| S1-09 | 大阶段自动门、README 与 L2 收口 | 3–5h | S1-08 | 全量复现、讲解/追问/故障/修改练习、简历 bullet |

预计总投入约 42–58 小时。按 2026-07-16 至 2026-07-27 的既定工作日/周末节奏约有 56 小时，目标有机会完成，但 ORT/GTest 依赖准备或一致性排错一旦超预期就几乎没有缓冲。出现延期时不得通过删除正确性、测试或证据门槛来“按时完成”。

关键路径：

```text
S1-01 dependency + contract
-> S1-02 session
-> S1-03 raw inference
-> S1-04 postprocess
-> S1-05 vertical slice
-> S1-06 quality gate
-> S1-07 correctness gate
-> S1-08 performance gate
-> S1-09 large-stage acceptance
```

## 5. 每个小阶段的共同完成定义

任何 `S1-*` 只有同时满足下列条件才可标记完成：

1. 只推进当前小阶段，没有偷跑下一步。
2. 相关 configure/build/run/test 命令实际成功；失败则保持未完成并记录限制。
3. 新增行为有正向和必要的负向证据，错误信息包含失败对象与排查方向。
4. `README.md`、`README_zh.md` 的状态、命令、测试、限制和 V2 日志同步。
5. 产物路径、关键文件/类/函数、输入输出和真实数字已记录。
6. Codex 按 `AGENTS.md` 最新九部分格式完成教学级闭环，给出精确代码行候选。
7. 用户完成当前小阶段 L1 验收后，才允许进入下一步。

## 6. S1-01：Baseline 契约、工程边界与依赖准备

### 目标与设计理由

在加载模型之前，把“Runtime 如何消费模型”从散落在代码中的隐含假设变成可读、可校验、可复用的 contract，并建立后续 ORT、GTest 都能复用的 library 边界。

建议将两个概念分开：

- `RuntimeConfig`：artifact spec 路径、score/NMS threshold、provider 等运行策略。
- `ModelArtifactSpec`：model id/family/path、input/output name/shape/dtype/layout、class names、preprocess/postprocess type、NMS mode 等模型固有契约。

P0 可以继续使用无外部解析依赖的 `key = value` 格式；逻辑上必须是两个类型化 schema。相对路径分别相对“声明它的配置文件”解析，不能依赖调用命令时的 working directory。

### 实施内容

- 扩展/重构配置代码，至少覆盖：
  - `schema_version`
  - `model_id`, `model_family`, `model_path`, `model_sha256`, `opset`
  - `source/provenance`, `artifact_license`
  - `input_name`, `input_width`, `input_height`, `input_channels`, `input_layout`, `input_dtype`
  - `output_name`, `output_layout`, `output_dtype`
  - `class_names`
  - `preprocess_type`, `postprocess_type`, `nms_mode`
  - `score_threshold`, `nms_threshold`, `provider`
- 为缺 key、未知 key、重复 key、非法数字、非法枚举、空类别、错误路径提供明确错误。
- 将 CMake 从单 executable 调整为至少：
  - `yolo_defect_runtime` library
  - `yolo_defect_cpp` CLI executable
  - 后续 `yolo_defect_cpp_tests` 的清晰链接入口
- 盘点并记录 OpenCV、ONNX Runtime C++ SDK、GTest、MSVC/CMake 的来源、版本与路径。
- CMake 用 `ONNXRUNTIME_ROOT` 或等价可配置入口，不写死个人绝对路径；Windows runtime DLL 行为必须有明确方案。
- 前置准备已验证仓库外 ORT C++ SDK 1.19.2 与 GTest v1.17.0 固定 archive 方案；本步负责把这些依赖变成可配置、可复现的 CMake 接入与记录，不再重复下载，也不得拿 Python wheel DLL 冒充 SDK。

### 预期文件变化

```text
cpp_infer/
├── CMakeLists.txt
├── artifacts/yolov8_neu_det.artifact.txt
├── configs/default_config.txt
├── include/yolo_defect_cpp/
│   ├── artifact_spec.h
│   └── config_loader.h
├── src/
│   ├── artifact_spec.cpp
│   └── config_loader.cpp
└── README.md
README.md
README_zh.md
```

具体文件名可在实现时按现有风格微调，但类型边界和职责不能丢失。

### 输入、输出与非目标

- 输入：当前 `default_config.txt`、`models/best.onnx` 路径、已知 Python ORT metadata。
- 输出：类型化 Runtime/artifact contract、multi-target 构建边界、可复现依赖方案。
- 本步不加载模型、不执行 `Session::Run`、不实现后处理。

### 验收命令

以下命令在 Visual Studio Developer PowerShell/Command Prompt 对应环境中执行，实际依赖参数以本步最终记录为准：

```powershell
$BuildDir = Join-Path $env:TEMP 'yolo_defect_s1_01'
cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir
& "$BuildDir\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt
ctest --test-dir $BuildDir --output-on-failure
```

### 验收标准与证据

- Runtime/Artifact 两类契约及其职责可以清楚解释。
- CLI 从任意 working directory 都能解析同一个 model artifact path。
- 合法配置打印完整 contract 摘要；至少一个缺字段和一个非法枚举用例清晰失败。
- Runtime library 与 CLI target 都完成 Release 构建；现有 smoke 不回退。
- ORT/GTest 的来源、版本、路径、CMake 参数和 DLL 策略明确；依赖未准备好则本步不得标记完成。
- README 双语记录真实状态，不声称已加载模型。

### L1 理解与排错重点

- Runtime config 与模型内部 metadata 有什么区别，为什么两者都要校验？
- 为什么 preprocess/postprocess 类型属于 artifact contract，而不仅是代码细节？
- 为什么 library target 比把所有实现塞进 `main.cpp` 更利于测试和后续模型适配？
- 配置在仓库根目录和其他 working directory 下结果不同，应先检查哪一层路径语义？

### 本步实施 Prompt

```text
认真阅读 AGENTS.md、docs/PLAN.md、README.md、README_zh.md 和 docs/STAGE1_EXECUTION_PLAN.md 的 S1-01。先检查 dirty worktree，完整保留用户已有改动。

本次只完成 S1-01：为 YOLOv8/NEU-DET baseline 建立可执行的 RuntimeConfig + ModelArtifactSpec 契约。契约至少覆盖 schema version、model id/family/path/SHA-256/opset、source/provenance、artifact license、input/output name/shape/dtype/layout、class names、score/NMS threshold、provider、preprocess_type、postprocess_type 和 nms_mode；相对路径必须相对声明它的文件解析。缺失字段、未知/重复字段、非法阈值、非法枚举和空类别必须给出可行动错误。记录 ONNX metadata 的 AGPL-3.0 与仓库 MIT 之间需要进一步复核的分发风险，但不要在没有来源证据时擅自改许可证。

把 CMake 从单 executable 调整为 yolo_defect_runtime library + yolo_defect_cpp CLI，并为后续 GTest 保留清晰链接边界。复核 `docs/PRE_STAGE1_READINESS.md` 已验证的 OpenCV、ONNX Runtime C++ SDK、GTest、MSVC/CMake 来源、版本和路径；CMake 不得硬编码个人 ORT 绝对路径。通过可配置 `ONNXRUNTIME_ROOT` 消费现有 SDK，并按已冻结的 GTest commit archive + SHA-256 方案接入；不能把 Python wheel DLL 当成完整 SDK。

严格限制修改范围为 cpp_infer/、README.md、README_zh.md；不要修改受保护的 Python/训练/API/Docker/results 资产。本步不创建 ORT session、不运行推理。运行 clean Release configure/build、现有 CTest 和配置正反例；同步双语 README 的状态、命令、验收和教学记录。按 AGENTS.md 最新九部分格式完成闭环后停止，不要开始 S1-02。
```

## 7. S1-02：ORT Session 与模型 Metadata 校验

### 目标与设计理由

建立只负责“加载模型、管理 ORT 生命周期、读取 actual metadata、验证 contract”的边界。此时还不运行真实推理，以便把依赖/session 错误和 tensor/算法错误分开排查。

### 实施内容

- 新增 `OnnxRunner`、`ModelMetadata` 或等价职责类型。
- 使用 ONNX Runtime C++ RAII 对象管理 `Env`、`SessionOptions`、`Session`、allocator 和 names。
- SessionOptions 固定并记录 P0 CPU provider/线程策略；读取并打印实际 provider，而不是只打印配置字符串。
- `--inspect-model` 输出 ORT 版本、provider、input/output count、name、shape、dtype。
- 校验：单输入/单输出、float32、NCHW 3 通道、配置 H/W、输出 `[1,4+C,N]`、输出 channel 与类别数一致。
- 缺模型、provider 不可用、shape/dtype/name/class count 不匹配时，同时报告 expected、actual 和排查方向。
- Windows 构建/运行对 ORT DLL 的发现或复制行为可复现。

### 输入、输出与非目标

- 输入：S1-01 contract、`models/best.onnx`、ORT C++ SDK。
- 输出：实际模型 metadata 和契约验证结果。
- 本步不构造 input tensor、不调用 `Session::Run`、不 decode。

### 验收命令

```powershell
$BuildDir = Join-Path $env:TEMP 'yolo_defect_s1_02'
$env:PATH = "$env:ONNXRUNTIME_ROOT\lib;D:\01_Base\Tools\opencv\build\x64\vc16\bin;$env:PATH"
cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir
& "$BuildDir\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --inspect-model
ctest --test-dir $BuildDir --output-on-failure
```

### 验收标准与证据

- 当前 artifact 由 C++ 实际加载。
- 输出并验证：input `images` float32 `[1,3,800,800]`；output `output0` float32 `[1,10,13125]`。
- 输出 ORT 版本和 session 实际 provider，P0 CPU provider 确实生效。
- 模型不存在和 synthetic metadata/class mismatch 测试明确失败。
- session/resource 不依赖手工释放；README 不声称已运行 inference。

### L1 理解与排错重点

- RAII 解决了什么资源生命周期问题？
- 静态 shape 与动态 shape 的校验策略有什么不同？
- 为什么 `output_channels - 4` 必须等于 class count？
- “CPU 写在 config 中”和“session 实际使用 CPU provider”为什么不是同一件事？

### 本步实施 Prompt

```text
阅读 AGENTS.md、docs/PLAN.md、双语 README、docs/STAGE1_EXECUTION_PLAN.md 的 S1-02，并先核对 S1-01 的代码、测试和依赖验收证据。

本次只实现 ONNX Runtime C++ session 与模型 contract 校验。新增职责清晰的 OnnxRunner/ModelMetadata，使用 ORT C++ RAII API；CMake 通过 S1-01 确认的 ONNXRUNTIME_ROOT 或 package 接入，不硬编码个人目录，并处理 Windows runtime DLL。增加 --inspect-model：加载 artifact 中的模型，打印 ORT 版本、实际 provider、input/output count、name、shape 和 dtype。

校验单输入/单输出、float32、NCHW 三通道、配置输入尺寸、YOLOv8 输出 [1,4+class_count,N]；错误必须报告失败对象、expected、actual 和建议排查方向。补模型缺失、metadata shape/dtype/class mismatch 的可测试路径。

只修改 cpp_infer/、README.md、README_zh.md。本步不要构造 input tensor、不要调用 Session::Run、不要实现后处理。运行 clean Release build、inspect-model 和 CTest，记录真实输出，按 AGENTS.md 九部分闭环后停止，不要开始 S1-03。
```

## 8. S1-03：输入 Tensor 与真实 Raw Inference

### 目标与设计理由

只打通 `PreprocessResult -> Ort::Value -> Session::Run -> owned raw output`，把 tensor shape、内存生命周期和真实模型运行单独验证，避免与 postprocess 混在一起调试。

### 实施内容

- `OnnxRunner::run` 或等价接口接收预处理后的 float32 NCHW tensor。
- 校验 tensor 元素数、shape、连续性和 finite values。
- 正确构造 CPU `Ort::Value`；底层 vector 生命周期必须覆盖 `Session::Run`。
- 将 raw output 在离开 ORT value 生命周期前复制到拥有自身存储的 `InferenceOutput`，或使用同等安全的 ownership。
- 校验 output count、shape、元素数、NaN/Inf。
- 增加 `--raw-output-summary`：只打印 shape、元素数、min/max/少量摘要，不打印全部 131,250 个值。
- 增加固定真实模型 integration smoke，以及错误 tensor 长度单元测试。

### 输入、输出与非目标

- 输入：固定 config、`crazing_241.jpg`、已验证 session。
- 输出：拥有独立生命周期的 `[1,10,13125]` float raw output。
- 本步不 decode、不做 NMS、不写 JSON/图片。

### 验收标准与证据

- 固定图完成 `image -> preprocess -> Session::Run -> raw output`。
- 输出 shape/元素数正确，数值全部 finite，摘要稳定可读。
- 错误 tensor size 在进入 ORT 前被拒绝。
- CTest 既保留旧 smoke，又增加真实 raw inference smoke。

### L1 理解与排错重点

- `Ort::Value` 是否拥有传入 vector 的内存？为什么生命周期容易出错？
- `[1,10,13125]` 的 10 和 13125 分别表示什么？
- 出现 NaN/Inf 时，应该按 preprocess、input contract、model 还是 ORT provider 的顺序如何排查？

### 本步实施 Prompt

```text
基于已验收的 S1-02，本次只完成 S1-03：把 PreprocessResult 的 float32 NCHW tensor 安全传入 ONNX Runtime，并获得拥有独立生命周期的 raw output。

实现 OnnxRunner::run 或等价接口；严格校验输入 shape、元素数和 finite values，正确构造 CPU Ort::Value，确保底层 vector 在 Session::Run 期间有效。输出离开 ORT Value 生命周期前必须复制或用明确 ownership 安全持有。校验输出 count/shape/元素数/NaN/Inf。新增 --raw-output-summary，固定图片只打印 input/output shape、元素数和有限数值摘要，不能输出整个 tensor。增加真实模型 CTest smoke 和错误 tensor 长度测试。

不要实现 decode、NMS、JSON、可视化或 benchmark。修改范围限于 cpp_infer/ 和双语 README。运行 clean Release build、固定图 raw inference 与 CTest，记录真实结果，按 AGENTS.md 九部分闭环后停止，不要开始 S1-04。
```

## 9. S1-04：YOLO Decode、Filter、NMS 与坐标还原

### 目标与设计理由

把最可能被面试追问的算法从 session/CLI 中拆成纯函数，用 synthetic tensor 精确验证，而不是用“真实图片看起来差不多”代替正确性。

### 冻结的 Baseline 语义

为与当前 Python ORT 参考实现对齐，P0 contract 明确采用：

```text
output layout: [1, 4 + C, N]
box encoding: cx, cy, w, h
class score: max over C; no separate objectness
filter: confidence > threshold (strictly greater)
NMS: class-agnostic
NMS space: model-input coordinates
restore: subtract letterbox padding, divide by scale, then clip to source bounds
```

class-aware 与 class-agnostic NMS 的取舍必须写入 README。后续如果产品语义要改，必须同步 Python reference/contract/tests；不能在一致性阶段悄悄改变。

### 实施内容

- 定义稳定 `Detection`：`class_id`, `class_name`, `confidence`, `bbox_xyxy`。
- 拆分纯函数：
  - output layout/size validation
  - `[1,4+C,N]` 索引与 decode
  - class argmax 与 strict score filter
  - `xywh -> xyxy`
  - IoU
  - stable score ordering 与 NMS
  - letterbox inverse transform 与 clip
- 同分数必须有可复现 tie-break；空候选合法返回空数组。
- 正式接入 GTest，并用 synthetic tensors/boxes 覆盖核心逻辑。
- 为 `ImagePreprocessor` 提供可直接接收 `cv::Mat` 的可测试边界，使非正方形/已知像素测试不依赖外部图片文件。

### 关键测试

- `[1,10,N]` synthetic raw output decode 得到精确 class/confidence/box。
- `confidence == threshold` 被过滤，`>` 才保留。
- 高重叠、低重叠、同分数、空候选 NMS。
- `xywh -> xyxy`、IoU、source-bound clip。
- 横图、竖图、奇数 padding、非正方形 model input 的 letterbox 与坐标反算。

### 输入、输出与非目标

- 输入：S1-03 raw layout contract 和 synthetic tensors。
- 输出：`std::vector<Detection>` 与可独立测试的核心函数。
- 本步不写 JSON、不绘图、不 benchmark。

### 验收标准与证据

- 核心算法无需真实模型即可用精确预期验证。
- 固定真实图片能从 raw output 得到合理 detections，但视觉正确性不是唯一证据。
- 聚焦 GTest 和完整 CTest 都通过。
- NMS 与阈值语义在 contract、代码、测试和 README 中一致。

### L1 理解与代码练习重点

- 能手写 `xywh -> xyxy`、IoU、greedy NMS、letterbox 坐标逆变换。
- 能解释 `[1,10,13125]` 的内存索引。
- 能解释 class-aware/class-agnostic NMS，以及为什么本 baseline 暂用后者。
- 能说明阈值 `>` 与 `>=` 对边界框的影响和对应测试。

### 本步实施 Prompt

```text
阅读当前 src/detector.py 作为只读参考，不要修改它。本次只完成 S1-04：实现可独立测试的 YOLOv8 raw-output postprocess。

新增 Detection 数据结构以及 output 校验、[1,4+C,N] decode、strict confidence filter、xywh->xyxy、IoU、stable class-agnostic NMS、letterbox 坐标还原和 clip 的纯函数。必须复现当前 baseline contract：无独立 objectness、取最大类别分数、confidence > threshold、在模型输入空间 NMS 后再还原原图；同分数定义稳定 tie-break，空候选返回空列表。为 ImagePreprocessor 增加可直接接收 cv::Mat 的测试边界。

接入 GTest，并用 synthetic tensor/box/image 覆盖 decode、阈值边界、IoU/NMS、空输出、横竖图、奇数 padding、非正方形输入、坐标还原与裁剪。不要依赖真实模型来证明纯逻辑，也不要实现 JSON、绘图、benchmark 或一致性工具。运行 clean build、聚焦 GTest 和完整 CTest；同步双语 README，按 AGENTS.md 九部分闭环后停止，不要开始 S1-05。
```

## 10. S1-05：完整 CLI、Detection JSON 与可视化

### 目标与设计理由

第一次形成可以演示的单图片纵切，同时保持 `main.cpp` 只是参数解析和 orchestration，核心逻辑仍位于 Runtime library。

### 固定命令目标

```powershell
& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config cpp_infer\configs\default_config.txt `
  --image data\images\val\crazing_241.jpg `
  --output-json cpp_infer\results\demo\crazing_241.json `
  --output-image cpp_infer\results\demo\crazing_241.jpg
```

### 实施内容

- 用薄 `DetectorPipeline` 或等价 orchestration 连接 config/artifact、preprocess、ORT、postprocess。
- JSON 至少包含：
  - `schema_version`
  - `model_id`, `model_path` 或稳定 artifact id
  - image path、original/input size
  - actual provider
  - score/NMS threshold 与 NMS mode
  - detections
- 每个 detection 包含 `class_id`, `class_name`, `confidence`, `bbox_xyxy`。
- 无检测时输出合法空数组；浮点精度和字段顺序尽量稳定。
- 使用 OpenCV 生成确定性颜色、label、confidence 的可视化文件，不调用 GUI。
- 明确输出目录创建、已有文件覆盖策略、参数依赖和失败退出码。
- 使用标准库安全 JSON serializer 或等价最小方案；字符串必须正确 escape，并用 Python 标准库验证。

### 输入、输出与非目标

- 输入：一个 config 和一张图片。
- 输出：一个 detection JSON 和一张可视化图片。
- 本步只支持 batch=1 单图，不做目录批处理、worker、服务化、benchmark 或 `inference_event`。

### 验收命令与标准

```powershell
python -m json.tool cpp_infer\results\demo\crazing_241.json
Get-Item cpp_infer\results\demo\crazing_241.jpg
```

- 固定命令从全新 Release build 成功运行。
- JSON 可解析、schema/字段/类别/坐标合法；图片可由 OpenCV 重新读取。
- 端到端 CTest smoke 验证文件产生与非零错误路径。
- README Quick Start 与 Demo 输入/输出从“占位”更新为真实证据。

### L1 理解与排错重点

- Pipeline orchestration 与算法模块为什么要分开？
- JSON schema 如何保证下游和未来 Project 2 不被随意字段变化破坏？
- JSON 正确但框画错，应分别检查 Detection 坐标语义、clip 还是 Visualizer？

### 本步实施 Prompt

```text
本次只推进 S1-05：将已验证的 config/artifact、OpenCV preprocess、OnnxRunner 和 postprocess 串成稳定的单图 CLI 纵切。

增加 --output-json 与 --output-image。JSON 必须稳定且可被 Python json 模块解析，包含 schema_version、model/artifact id、image metadata、actual provider、阈值/NMS mode 和 detections；每个 detection 有 class_id、class_name、confidence 和 bbox_xyxy，无检测时输出空数组。用 OpenCV 生成不依赖 GUI 的可视化文件，采用确定性颜色和标签。main.cpp 只负责 CLI 与编排，核心逻辑保留在 runtime library。明确输出目录创建、覆盖和参数错误行为，并安全处理 JSON 字符串 escaping。

不要新增目录批处理、并发、服务、inference_event、INT8 或 benchmark。运行固定样本完整命令，用 python -m json.tool 验证 JSON，并用 OpenCV/文件检查确认图像可读。同步 README/README_zh 的 Quick Start、Demo、任务状态和教学日志，按 AGENTS.md 九部分闭环后停止，不要开始 S1-06。
```

## 11. S1-06：自动化测试与核心故障注入

### 目标与设计理由

把已经存在的稳定边界转成可重复的工程质量 gate。大阶段一至少覆盖主链路和核心错误；大阶段二再扩展样本、平台和回归深度。

### 测试矩阵

| 模块 | 最低覆盖 |
|------|----------|
| Runtime/artifact schema | 合法、缺 key、未知/重复 key、阈值越界、非法 enum、路径解析 |
| Preprocess | 横图、竖图、奇数 padding、非正方形输入、已知 BGR 像素到 RGB/NCHW 数值 |
| Model metadata contract | name/shape/dtype/class count/provider mismatch |
| Postprocess | layout/decode、threshold、IoU/NMS、tie-break、empty output、coordinate restore/clip |
| Integration | 固定模型 + 固定图片 + JSON/visualization smoke |
| CLI/故障 | 模型不存在、损坏图片、参数缺失/冲突、不可写输出路径 |

metadata 错误优先通过纯 `ModelMetadata` validator 注入，不为了测试 shape/dtype 而制造多个大型坏 ONNX。损坏图片可以在测试临时目录生成少量无效 bytes。真实模型只用于少量 integration smoke。

### 错误质量标准

失败信息尽量包含：

```text
failing object or path
expected contract
actual value/state
likely corrective action
non-zero CLI exit code
```

空模型输出本身可合法得到空 detections；“输出 shape 不合法”和“合法 shape 但无框过阈值”必须区分。

### 验收命令与标准

```powershell
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

- 测试名能反映模块与行为，不只追求数量。
- 所有 GTest/CTest 在 clean Release build 通过。
- 至少手动运行一个缺模型和一个损坏图片 CLI，均非零退出且信息可行动。
- 测试失败时输出足够定位到 config/preprocess/session/postprocess/output 层。

### L1 理解与排错重点

- 单元测试、integration smoke、故障注入分别证明什么？
- 为什么 metadata validator 用 synthetic struct 比准备多个坏模型更可维护？
- 如何区分“模型正常但无缺陷”和“模型输出异常为空”？

### 本步实施 Prompt

```text
本次只完成 S1-06 工程质量 gate。扩展 GTest/CTest，覆盖 Runtime/artifact schema、横竖图与奇数 padding letterbox、已知像素 BGR->RGB/normalize/NCHW、模型 metadata contract、YOLO decode、阈值、IoU/NMS、坐标反算/裁剪和合法 empty detections。

故障注入至少覆盖模型不存在、shape/dtype/name/class mismatch、损坏图片、CLI 参数错误和不可写输出。优先把 metadata validation、preprocess(cv::Mat) 和 postprocess 设计为纯函数，用 synthetic 输入测试，不要为错误注入引入多个大型 ONNX fixture。真实模型只用于少量 integration smoke。CLI 错误必须非零退出，并尽量说明失败对象、expected/actual 和排查方向。

不做模糊测试、并发压力、INT8 或大阶段二的跨平台矩阵。运行 clean Release 完整 CTest 并列出测试名；同步双语 README 的测试命令、错误排查和当前证据。按 AGENTS.md 九部分闭环后停止，不要开始 S1-07。
```

## 12. S1-07：固定六类 Python ORT/C++ 一致性

### 目标与设计理由

证明同一个 ONNX artifact、同一个 CPU provider、同一前后处理 contract 在 Python 与 C++ 中得到等价检测结果。不能再用“检测框数量一样”替代类别与坐标证据。

### 固定样本与工具边界

- 在 `cpp_infer/tests/fixtures/` 或等价位置提交 manifest，覆盖 NEU-DET 6 类，每类固定 5 张，共 30 张。
- 比较工具放在 `cpp_infer/tools/`，可以只读复用 `src/detector.py`，但不得修改受保护的 `src/`、`scripts/` 或根 `results/`。
- Python 显式使用 `CPUExecutionProvider`；C++ 使用相同 CPU provider、模型、输入尺寸、strict threshold、class-agnostic NMS 和坐标语义。
- Detection 匹配不依赖 JSON 数组顺序：先按 class，后按最大 IoU 做确定性匹配。

### 预声明初始门槛

```text
per-image detection count: exact match
matched class_id: exact match
absolute confidence error: <= 1e-4
absolute bbox coordinate error: <= 1e-2 pixel
matched box IoU: >= 0.999
```

这些是开始运行前声明的严格初始门槛。若因可解释的跨语言/OpenCV 数值差异需要调整，必须先定位误差来源、保留失败证据、给出数据分布和理由，并同步 README；禁止为了“全绿”直接放宽。

### 输出证据

建议生成：

```text
cpp_infer/results/consistency/
├── manifest.txt
├── per_image.json
└── summary.json
```

Summary 至少包含 artifact/config hash 或稳定标识、provider、样本/类别数、阈值/NMS mode、通过数、最大/均值 confidence/coordinate error、最小 matching IoU、失败详情。

README 必须分开表述：

- 历史 PT/ONNX：50 张全为 `crazing`，只做 count/confidence summary。
- 本阶段 Python ORT/C++ ORT：30 张六类，比较 count/class/confidence/box。
- 当前缺匹配 `best.pt`，所以本次没有直接重跑三方，不得写成新完成的 PyTorch/ONNX/C++ 三方实测。

### 验收标准

- 30 张 manifest 可重复、顺序固定、六类均有 5 张。
- 比较命令返回 0 且所有预声明门槛通过；否则任务保持未完成并有逐图诊断。
- 至少人工抽查一个有框样本和一个空框/低置信样本（若 manifest 中存在）。
- 完整 CTest 不回退。

### L1 理解与排错重点

- 为什么不能只按输出顺序 zip detections？
- 一致性失败时如何按 preprocess → raw output → threshold/NMS → coordinate restore 分层定位？
- 历史 PT/ONNX 证据和当前 Python ORT/C++ 证据如何共同构成链路，又为什么不能说成同一次三方实验？

### 本步实施 Prompt

```text
本次只完成 S1-07 正确性证据。不要修改受保护的 src/、scripts/、runs/、models/ 或根 results/。

在 cpp_infer/ 下建立固定、提交到仓库的 consistency manifest，覆盖 6 类各 5 张验证图。实现 Python ORT vs C++ ORT comparison 工具；Python 必须显式使用 CPUExecutionProvider，并使用与 C++ 完全一致的 artifact/config、输入尺寸、confidence > threshold、class-agnostic NMS 和坐标语义。Detection 按类别和最大 IoU 确定性匹配，不依赖输出顺序。

在运行前固定初始门槛：逐图数量和 class_id 完全一致、confidence 绝对误差 <=1e-4、bbox 坐标绝对误差 <=1e-2 像素、matching IoU >=0.999。失败时输出逐图诊断，禁止无依据放宽容差。输出机器可读 per-image 与 summary JSON 到 cpp_infer/results/consistency/。

当前缺少匹配 best.pt，README 必须区分历史 PT/ONNX count 证据和本次 Python ORT/C++ ORT 严格证据，不能声称本次直接重跑三方。运行完整 comparison 和相关测试；同步双语 README 后按 AGENTS.md 九部分闭环并停止，不要开始 S1-08。
```

## 13. S1-08：分段 Release Benchmark 与内存基线

### 目标与设计理由

在 S1-07 正确性通过之后，建立第一份属于 C++ Runtime 的可复现性能记录。先正确再测性能，且必须把历史 Python ORT 数字与新 C++ 数字分开。

### Benchmark 协议

- 只使用 clean Release build、batch=1、CPU provider、固定线程策略。
- 固定 `crazing_241.jpg` 作为首份基线，记录 sample count=1；大阶段二可扩展到多样本。
- 建议 warmup=10、repeat=100。
- Session/model 初始化不进入重复计时。
- 明确定义：
  - `image_decode_ms`：`imread`。
  - `preprocess_ms`：已加载 `cv::Mat -> tensor`。
  - `infer_ms`：仅 `Session::Run`。
  - `postprocess_ms`：raw output -> detections。
  - `pipeline_ms`：preprocess + infer + postprocess。
  - `end_to_end_ms`：image decode + pipeline；JSON 写盘和绘图不包含，并在结果中声明。
- 每段输出 mean/P50/P95；pipeline 与 end-to-end 输出 throughput。
- Windows 用 `GetProcessMemoryInfo` 或等价方法记录 Peak Working Set；其他平台记录 peak RSS，无法支持时写明确 `unsupported`，不能填 0 伪装成功。

### 结果元数据

Benchmark JSON 至少记录：

```text
timestamp and command
machine / OS
compiler / build type
OpenCV / ONNX Runtime versions
requested and actual provider
thread settings
model id/path/size and input shape
image/sample count
score/NMS thresholds and NMS mode
warmup/repeat
detection count
each latency statistic
throughput
Peak Working Set / peak RSS
timing exclusions and limitations
```

### 固定命令目标

```powershell
& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config cpp_infer\configs\default_config.txt `
  --image data\images\val\crazing_241.jpg `
  --benchmark `
  --warmup 10 `
  --repeat 100 `
  --benchmark-json cpp_infer\results\benchmark\yolov8_neu_det_cpu_release.json

python -m json.tool cpp_infer\results\benchmark\yolov8_neu_det_cpu_release.json
```

### 验收标准与证据

- S1-07 一致性未通过时不得发布 benchmark。
- clean Release 命令实际运行，JSON 可解析且包含上述元数据。
- 所有分段、pipeline、end-to-end 有 mean/P50/P95；throughput 和 Peak Working Set 合法。
- README 的结果表明确区分 Python ORT historical baseline 与 C++ Runtime current baseline，并说明协议不可直接等价比较时的限制。
- 不根据一次单样本结果做夸大结论。

### L1 理解与排错重点

- warmup、repeat、P50、P95 为什么比单次/平均值更可信？
- session 初始化为什么不放入每次 inference；为什么又要在文档中披露？
- pipeline 与 end-to-end 的边界是什么，磁盘 JSON/绘图为什么排除？
- Peak Working Set 与当前工作集有什么区别？

### 本步实施 Prompt

```text
本次只完成 S1-08 可复现性能证据。先确认 S1-07 正确性门槛已通过；正确性不通过时不得先优化或发布 benchmark。

实现 Release benchmark mode，支持 --warmup、--repeat 和 --benchmark-json。固定 batch=1、CPU actual provider、线程策略、模型和单张 baseline 样本。分别测量 image decode、preprocess(cv::Mat->tensor)、Session::Run、postprocess、pipeline(preprocess+infer+postprocess) 和 end-to-end(image decode+pipeline)；Session 初始化、JSON 写盘和绘图不进入重复计时并必须披露。每段输出 mean/P50/P95，pipeline/end-to-end 输出 throughput。

结果 JSON 记录机器/OS、编译器/build type、OpenCV/ORT 版本、requested/actual provider、线程、模型标识/大小/input、样本数、阈值/NMS mode、warmup/repeat、检测数、统计值、计时排除项和限制。Windows 记录 Peak Working Set；不支持的平台明确 unsupported。不要把旧 Python ORT 24.4/72.1 FPS 写成 C++ 结果，也不要对不同协议数字做无条件优劣比较。

运行一次 clean Release warmup=10/repeat=100 benchmark，验证 JSON，更新 README 双语结果表和限制。按 AGENTS.md 九部分闭环后停止，不要开始 S1-09。
```

## 14. S1-09：大阶段自动门、README 与 L2 收口

### 目标与设计理由

本步不新增产品功能，只证明大阶段一可以 clean reproduce、可以演示、可以接受工程和面试追问，并把下一阶段边界冻结。

### 自动化出口

- 全新 `%TEMP%` Release configure/build 成功。
- 完整 GTest/CTest 全通过。
- 固定 CLI 生成可解析 JSON 和可读可视化。
- 30 张六类一致性全部通过预声明门槛。
- Benchmark JSON 有分段/pipeline/end-to-end P50/P95、throughput、环境和 Peak Working Set。
- 核心故障返回清晰非零错误；合法 empty detections 不被误判为系统错误。
- `README.md`、`README_zh.md`、`cpp_infer/README.md` 的命令、数字、路径、状态和限制一致。

### L2 理解出口

用户需要完成：

1. 两分钟压缩讲解和五分钟完整讲解。
2. 至少 10 个由 Codex 给出的追问。
3. 至少 3 个错误案例及分层排查顺序。
4. 指出关键文件、类、函数、输入输出和证据路径。
5. 在 AI 指导下 clean build、测试、Demo、一致性和 benchmark。
6. 独立完成一次“核心行为 + 对应测试”的修改练习。推荐：把 confidence 边界从 `>` 临时改为 `>=`，更新边界 GTest、运行测试、解释影响，最后根据 contract 决定恢复还是正式变更。
7. 形成 1～2 条简历 bullet 和本阶段笔记。
8. 识别代码练习候选：letterbox/NCHW、YOLO output indexing/decode、IoU/NMS、坐标反算、percentile 统计、ORT tensor lifetime。

### README/证据核对

- 项目定位、顶层设计、问题、架构与模块职责。
- Quick Start、Demo 输入输出、测试/错误注入/一致性/benchmark 命令。
- 真实结果表、环境、容差、限制和原始路径。
- 历史 Python benchmark、本次 C++ benchmark、历史 PT/ONNX count 证据、本次 Python ORT/C++ 严格证据分别表述。
- 任务状态、版本记录、教学日志和下一大阶段边界。

### 大阶段关闭条件

自动门完成但用户 L2 理解门尚未完成时，大阶段一仍不能标记最终完成。Codex 应提交自动门证据并等待用户回答/实操，不得自行假定用户已经掌握。

### 本步实施 Prompt

```text
本次只做大阶段一 S1-09 收口，不新增产品功能。使用全新临时 Release build，依次执行 configure/build、完整 CTest、固定样本端到端 Demo、30 张六类一致性和 warmup/repeat benchmark；核对所有 JSON 可解析、可视化可读、核心错误注入返回非零且信息可行动。

系统检查 README.md、README_zh.md 和 cpp_infer/README.md：项目定位、主链路、模块职责、Quick Start、Demo 输入输出、测试命令、真实结果表、环境/容差/限制、任务状态、版本记录和教学日志必须与代码一致；英文和中文事实对齐。历史 Python benchmark、本次 C++ benchmark、历史 PT/ONNX count 证据和本次 Python ORT/C++ 严格证据必须分栏，不能混写。

提供用户的 2 分钟与 5 分钟讲解提纲、至少 10 个追问、至少 3 个错误排查案例、1–2 条简历 bullet，以及值得手写的核心代码清单。安排一次由用户完成的“核心行为 + 对应 GTest”修改验收；不要擅自把临时练习语义留在产品中。自动门和用户 L2 门都完成后才能宣布大阶段一完成。按 AGENTS.md 九部分格式收口，明确大阶段二继续承担的 INT8/证据加固内容，然后停止。
```

## 15. 大阶段一最终验收清单

只有以下全部满足，才能进入大阶段二：

- [ ] 固定命令完成 `config + artifact + image -> preprocess -> ORT -> decode/filter/NMS/坐标还原 -> JSON/可视化`。
- [ ] Runtime/artifact schema 可校验，相对路径稳定，模型 metadata 不匹配会阻止推理。
- [ ] CMake 为多 target，核心 Runtime 可被 CLI 和 tests 复用，依赖入口清楚。
- [ ] ORT session 使用 RAII，实际 provider/name/shape/dtype 有证据。
- [ ] YOLO output indexing、strict threshold、class-agnostic NMS、坐标反算语义固定且有精确 GTest。
- [ ] 配置、非正方形 letterbox、颜色/布局、NMS、坐标还原、非法输入等测试可运行。
- [ ] 模型不存在、contract mismatch、类别错误、损坏图片、空输出有明确且可区分的行为。
- [ ] 固定 30 张、六类覆盖的 Python ORT/C++ ORT 一致性通过预声明容差。
- [ ] Release benchmark 有分段/pipeline/end-to-end P50/P95、throughput、环境和 Peak Working Set。
- [ ] README 双语能复现命令、测试和真实结果，没有把历史 Python 或外部研究数字冒充 C++。
- [ ] 用户能完成 5 分钟讲解、10 个追问、3 个错误排查和一次核心行为+测试修改。
- [ ] 已形成 1～2 条简历 bullet、阶段笔记和代码练习候选清单。

## 16. 停损与调整规则

- **依赖阻塞：** ORT/GTest C++ 依赖未准备好时，停在 S1-01；记录 exact missing component 和可复现方案，取得授权后继续。不得跳过 SDK 校验直接写无法构建的代码。
- **模型契约不符：** C++ actual metadata 与预检值不符时，先检查 artifact/hash/path，再调整 contract；不得为通过而关闭校验。
- **后处理不一致：** 先冻结并比较 raw output，再查 layout/index、strict threshold、NMS mode/tie-break 和坐标还原；不得先放宽一致性容差。
- **正确性未过：** S1-07 未过时禁止发布 S1-08 性能数字。
- **性能异常：** 先检查 Debug/Release、provider、线程、warmup、计时边界和 DLL，再做优化；不得用历史 Python 数字掩盖问题。
- **时间不足：** 可以顺延日期，不能删除固定 Demo、正确性、测试、错误、benchmark 或 README/L2 出口。
- **范围膨胀：** 目录并发、TensorRT、Qt、gRPC/Triton、D010、INT8、服务化全部回到其所属后续阶段。

## 17. 第一条执行指令

下一次项目实现只使用本文件 **S1-01 的实施 Prompt**。S1-01 验收前，不进入 ONNX Runtime session 实现。
