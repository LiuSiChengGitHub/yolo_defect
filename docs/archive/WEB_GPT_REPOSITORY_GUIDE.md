# Web GPT Repository Guide

这是一份面向能够直接读取 GitHub 仓库的网页版 GPT 或外部代码审阅者的快速上下文。它不是新的项目规划源，也不替代 README；它只负责说明应该从哪个分支、按什么顺序阅读，以及哪些事实不能混写。

## 可直接发给网页版 GPT 的提示词

```text
请读取 GitHub 仓库 LiuSiChengGitHub/yolo_defect 的 deploy-cpp 分支，不要默认使用 main。

先阅读 docs/WEB_GPT_REPOSITORY_GUIDE.md，再依次阅读 AGENTS.md、docs/PLAN.md、README_zh.md（或 README.md）和 cpp_infer/README.md。回答前请核对当前分支和文件内容，不要依据文件名、历史日志或计划推断功能已经完成。

这是一个从 Python YOLOv8/NEU-DET baseline 演进而来的 C++17 工业视觉推理 Runtime。请把以下三类证据严格分开：历史 Python baseline、当前 YOLOv8 C++ Runtime、尚未接入 Runtime 的研究侧 D010。分析 C++ 部分时，优先阅读 cpp_infer/CMakeLists.txt、两层契约文件、public headers、对应 src 和 tests；不要递归遍历 data/、runs/、docs/assets/。

如果我要求分析或修改，请先用文件路径和代码证据复述当前状态、相关模块、输入输出、约束及非目标；不确定的地方明确标为待核对，不要擅自扩大范围、改许可证、降低一致性容差或把计划项写成已完成功能。
```

## 1. 首先确认分支与事实来源

- GitHub 仓库：`https://github.com/LiuSiChengGitHub/yolo_defect`
- 当前 C++ 主开发分支：`deploy-cpp`，不要误读较旧的默认 `main`。
- 本说明起草时的产品代码 HEAD 是 `46ca844`；提交本说明会产生后续 commit，因此以后应以所读分支的实际 HEAD 和文件内容为准。
- 顶层定位、范围和阶段边界以 `docs/PLAN.md` 为准，协作与保护规则以 `AGENTS.md` 为准；当前实现状态则以双语 README 的最新状态段、实际代码、测试及机器可读结果交叉判断。
- `docs/PLAN.md` 和其他历史记录包含带日期的旧基线快照，其中“C++ ORT/后处理尚未完成”等文字只描述当时状态，不能覆盖后来 S1-01～S1-09 的代码和证据。
- `docs/STAGE1_EXECUTION_PLAN.md` 是大阶段一的长执行记录，不是当前顶层路线的替代品。

当前提交态仍写明：S1-01 至 S1-08 已实现并通过 L1，S1-09 自动门通过，但用户 L2 在 README 中仍为 `PENDING`，因此大阶段一尚未正式宣布完成，大阶段二尚未开始。不要仅根据自动测试通过推断用户理解验收已经完成。

## 2. 用一句话理解项目

项目把已有的 YOLOv8/NEU-DET ONNX 模型变成一套可配置、可校验、可测试、可比较和可复现的 C++17 工业视觉推理 Runtime，而不只是再次训练一个检测模型。

当前已打通的产品链路是：

```text
RuntimeConfig + ModelArtifactSpec
-> 实际 ONNX ModelMetadata 校验
-> OpenCV 图片解码与 letterbox/RGB/normalize/NCHW 前处理
-> ONNX Runtime C++ CPU Session::Run
-> 自有生命周期的 raw output
-> YOLOv8 BCN decode / strict score filter / stable class-agnostic NMS
-> letterbox 坐标反算与原图边界裁剪
-> SingleImageDetectionResult
-> 稳定 JSON 和无 GUI 可视化 PNG
```

## 3. 仓库很大时的推荐阅读顺序

第一轮只读以下文件即可建立正确心智模型：

1. `AGENTS.md`：范围、证据标准、阶段边界和禁止事项。
2. `docs/PLAN.md`：项目定位、P0/P1 顶层设计和大阶段出口。
3. `README_zh.md` 或 `README.md`：完整项目故事、当前状态、结果和限制。
4. `cpp_infer/README.md`：C++ Runtime 的技术细节与可复现命令。
5. `cpp_infer/CMakeLists.txt`：静态库、CLI、测试 target 和依赖边界。
6. `cpp_infer/configs/default_config.txt` 与 `cpp_infer/artifacts/yolov8_neu_det.artifact.txt`：两层运行契约。
7. `cpp_infer/include/yolo_defect_cpp/`：稳定的 public API 和核心数据结构。
8. 只在具体问题需要时，再读对应的 `cpp_infer/src/`、`cpp_infer/tests/` 或机器可读结果。

不要一开始递归阅读这些大目录：`data/`、`runs/`、`docs/assets/`、根 `results/`。它们主要是数据集、训练历史、图片和旧证据，会挤占上下文并容易混淆主线。

## 4. 两个需要分开的系统

### 工程操作层

```text
stage1.cmd
-> vswhere 定位 Visual Studio
-> x64 VsDevCmd.bat 临时设置 PATH/INCLUDE/LIB
-> PowerShell -NoProfile 运行 stage1.ps1
-> CMake 根据 CMakeLists.txt 生成 NMake 构建规则
-> NMake 调用 cl/link 生成 Runtime library、CLI 和测试程序
-> CTest 调度 GTest、CLI、CMake 和 Python 测试
```

- `cpp_infer/tools/stage1.cmd`：普通 CMD/PowerShell 都可调用的统一入口。
- `cpp_infer/tools/stage1.ps1`：工作流调度和错误检查。
- `cpp_infer/tools/stage1.defaults.psd1`：提交到 Git 的机器无关默认值。
- `cpp_infer/.stage1.local.psd1`：被 Git 忽略的本机 SDK/Python/GTest 路径；模板是 `stage1.local.example.psd1`。
- `Release` 是构建模式，不是工具。
- 构建目录内的 `CMakeCache.txt`、Makefiles、对象文件、EXE 和 staged DLL 是可丢弃的生成状态，不是人工维护的配置。

统一入口支持：`help`、`doctor`、`build`、`clean-build`、`test`、`detect`、`demo`、`consistency`、`benchmark` 和 `all`。最小单图命令是：

```powershell
.\cpp_infer\tools\stage1.cmd detect "D:\images\sample.jpg"
```

`stage1.cmd detect` 会自动补齐 JSON/PNG 参数，因此走完整推理链路；直接运行产品 CLI 时，单独的 `--config --image` 只是 preprocess smoke，只有请求 `--output-json` 或 `--output-image` 才进入完整 `DetectorPipeline`。`all` 固定执行 clean Release build → 完整 CTest → Demo → consistency → benchmark。

### 模型运行层

| 对象 | 来源 | 职责 |
|---|---|---|
| `RuntimeConfig` | `cpp_infer/configs/default_config.txt` | 运行策略：artifact 选择、score/NMS 阈值、requested provider |
| `ModelArtifactSpec` | `cpp_infer/artifacts/yolov8_neu_det.artifact.txt` | 模型身份、路径、SHA、opset、来源/许可证、I/O contract、类别与前后处理语义 |
| `ModelMetadata` | ORT 创建 Session 后从实际 ONNX 读取 | 实际 input/output count、name、shape、dtype；它是运行时观测，不是配置文件 |
| `stage1.defaults.psd1` | 工作流配置 | build、Demo、consistency、benchmark 等操作默认值，不描述模型 tensor |
| `.stage1.local.psd1` | 本机私有配置 | ORT/OpenCV/Python/GTest 的本机路径，不应提交 |
| `CMakeLists.txt` | 构建声明 | target、源码、编译选项和依赖关系，不是推理参数配置 |

路径规则也分层：RuntimeConfig 中的 artifact 相对 RuntimeConfig 文件解析；artifact 中的模型路径相对 artifact 文件解析；workflow 默认路径相对其 `.psd1` 文件解析；CLI 图片和输出参数相对调用时工作目录解析。

## 5. C++ 模块与关键文件

`cpp_infer/CMakeLists.txt` 定义两个产品 target：

- `yolo_defect_runtime`：静态库，承载可以被 CLI 和 tests 复用的核心能力。
- `yolo_defect_cpp`：薄 CLI，只解析参数、选择模式并编排 Runtime；不要把 `main.cpp` 当作核心算法模块。

按链路查代码：

| 模块 | Public header | 主要实现 | 作用 |
|---|---|---|---|
| 配置与 artifact | `config_loader.h`, `artifact_spec.h` | `config_loader.cpp`, `artifact_spec.cpp`, `key_value_parser.cpp` | 严格 schema、枚举、阈值、shape、类别和路径校验 |
| 前处理 | `image_preprocessor.h` | `image_preprocessor.cpp`, `image_decoder.cpp` | `image path/cv::Mat -> PreprocessResult` |
| ORT 与 metadata | `onnx_runner.h`, `model_metadata.h` | `onnx_runner.cpp`, `model_metadata.cpp` | RAII Session、actual metadata 校验、安全输入借用与自有 raw output |
| 后处理 | `postprocessor.h` | `postprocessor.cpp` | decode、strict threshold、IoU、stable NMS、坐标还原/clip 的纯逻辑 |
| 单图纵切 | `detector_pipeline.h`, `detection_result.h` | `detector_pipeline.cpp` | 串起前处理、推理和后处理并形成自包含结果 |
| JSON/PNG | `result_writer.h` | `result_writer.cpp` | 稳定 JSON、escaping、确定性可视化和安全输出路径 |
| Benchmark | `benchmark_runner.h`, `benchmark_result.h`, `benchmark_writer.h` | 对应 `.cpp` | 六段计时、统计、throughput、环境与 Peak Working Set |
| CLI | 无独立 public API | `main.cpp` | `--inspect-model`、raw summary、单图输出、benchmark 等模式编排 |

## 6. 冻结的 YOLOv8 baseline contract

- 模型：`models/best.onnx`，12,336,935 bytes，opset 17。
- SHA-256：`7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`。
- 输入：`images`, float32, NCHW, `[1,3,800,800]`。
- 输出：`output0`, float32, BCN, `[1,10,13125]`。
- 类别顺序：`crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`。
- `10 = 4 + 6`：四个 `xywh` 参数加六个类别分数；没有独立 objectness，不额外执行 sigmoid。
- score filter 是严格的 `confidence > threshold`；NMS 是 stable、class-agnostic，并在模型输入坐标空间执行，之后才反算到原图并 clip。
- 当前 requested/actual provider 为 CPU；session 级 provider 证据不等于逐节点 placement profiling。

## 7. 当前可引用的工程证据

- 完整 Release CTest：106/106 通过；覆盖 synthetic unit、contract、metadata、preprocess、postprocess、output、negative 和少量真实模型 integration。
- 这里的 106 是 CTest 测试用例总数，并不等于 106 个 GTest；其中还包含 synthetic C++ executable、CLI/CMake wrapper、Python validator 和真实模型 smoke。`unit`、`negative`、`integration` 等 label 会重叠，不能把各 label 数量相加。
- 固定 Demo：`data/images/val/crazing_241.jpg`，输出 3 个 detection；提交的 JSON/PNG 位于 `cpp_infer/results/demo/`。
- 严格一致性：固定 manifest 覆盖六类各五张，共 30 张；30/30 图片通过，Python/C++ 各 62 个 detection，62/62 匹配。
- 一致性误差：最大 confidence 绝对误差 `8.049977111568296e-07`，最大 bbox 坐标误差 `9.135351561440075e-05 px`，最小匹配 IoU `0.999998927116394`。
- 提交的一致性证据：`cpp_infer/results/consistency/per_image.json` 和 `summary.json`。
- 提交的 S1-08 C++ Release 10/100 baseline：pipeline mean `175.560944 ms`、end-to-end mean `176.553060 ms`、对应 `5.696028/5.664020 img/s`，Peak Working Set `152.714844 MiB`。
- S1-09 另一次相同协议的临时复现约为 end-to-end `142.082777 ms`、`7.038151 img/s`、Peak Working Set `152.578125 MiB`；临时证据不在 Git 中，也不代表做过性能优化，不能与提交的 S1-08 结果平均或任选较快值冒充唯一结果。

读取结果时优先查看：

```text
cpp_infer/results/demo/
cpp_infer/results/consistency/
cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json
cpp_infer/tests/fixtures/consistency_manifest.json
```

## 8. 三条证据线绝对不能混写

1. **历史 V1 Python baseline**：训练、导出、FastAPI/Docker、旧 benchmark 和较弱的 50 图 PT/ONNX count-only 检查。历史 Python ORT `24.4/72.1 FPS` 不是 C++ 性能。
2. **当前 YOLOv8 C++ Runtime**：`cpp_infer/` 内的 contract、OpenCV、ORT、后处理、JSON/PNG、106 项测试、一致性和 C++ benchmark。
3. **研究侧 D010/D-FINE-S + DeepPCB**：目前是另一个研究 artifact 来源；未完成稳定 ONNX、artifact card、Runtime 接入和一致性门之前，不能写成当前 C++ Runtime 成果。

## 9. 当前限制与禁止夸大的结论

- 产品 CLI 当前只支持 batch=1 的单图；没有目录 batch API、并发 worker、服务化、INT8 或真实 Jetson/TensorRT/ARM 结果。
- 30 图一致性证明同一 ONNX 在 Python ORT 与 C++ ORT 的实现一致性，不证明模型 mAP、全数据集质量或所有平台一致。
- benchmark 只代表一台 Windows CPU、一个固定样本、Release、单线程策略和 warm-cache 协议；Peak Working Set 是进程生命周期峰值，不是单次推理或模型独占内存。
- matching `best.pt` 当前不可用，不能声称新跑过 PyTorch/Python ORT/C++ 三方严格比较。
- artifact loader 校验 `model_sha256` 的声明格式；实际文件哈希由一致性/benchmark 验证工具重新计算。声明值与重新计算值相等是证据，不能把“字段存在”当成“文件已重算”。
- 仓库源码是 MIT；ONNX metadata 声明 AGPL-3.0；NEU-DET 再分发依据仍需单独复核。不要在缺少来源证据时改许可证或声称三者相同。
- 大阶段二仍承担 INT8 PTQ、FP32/INT8 正确性/精度/性能/模型大小比较和进一步证据加固；计划项不是已交付功能。

## 10. GPT 在回答或修改前应遵守的方式

- 先确认分支，再引用实际文件；不要只看 GitHub 默认 `main`。
- 先说明是在“工程操作层”还是“模型运行层”定位问题。
- 回答当前状态时，把 implemented、tested、historical、planned 四种状态分开。
- 分析失败时优先沿边界排查：环境 → configure/build → contract → actual metadata → preprocess → ORT Run → postprocess → output → evidence validator。
- 修改时保持 `main.cpp` 薄、测试链接 `yolo_defect_runtime`，模型特定 decode 留在清晰边界内。
- 不修改受保护的旧 Python/训练/API/Docker/data/model/result 资产，除非用户明确授权具体任务。
- 不静默下载依赖，不把 Python wheel DLL 当完整 ORT C++ SDK，不硬编码个人 SDK 路径。
- 不为让测试通过而无依据放宽 consistency tolerance、改变 strict `>`、NMS mode 或坐标顺序。
- 给结论时附关键文件路径；需要精确实现细节时再读取对应 `.h/.cpp/test`，不要一次性吞入整个仓库。
