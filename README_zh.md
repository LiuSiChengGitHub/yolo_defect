# 工业视觉边缘 AI Runtime 与 C++ 工程化系统

[English](README.md)

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CMake](https://img.shields.io/badge/CMake-enabled-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19.2-orange)
![GTest](https://img.shields.io/badge/GTest-1.17.0-red)

本仓库将工业视觉模型产物转化为可配置、可运行、可测试、可比较、
可复现的 C++ 推理软件。项目面向秋招，重点展示现代 C++、Linux
可移植性、测试与调试、性能分析和模型推理工程，而不是再做一层检测器训练封装。

YOLOv8 和 NEU-DET 是首个 Runtime 实现所使用的稳定模型与数据集载体。
本仓库的价值在于围绕该产物建立工程闭环：可执行契约、C++ 推理、确定性输出、
正确性门禁、Benchmark 证据，以及受控推进 Linux、并发、量化和 TensorRT 的路径。

> **状态 — 2026-08-25：** 大阶段一自动工程门与用户负责的 L2 验收均已完成。
> 大阶段二文档前置准备已经收口；**S2-01 尚未开始**。目前已交付的结论仍仅限于
> 下文说明的 Windows x86_64 ONNX Runtime CPU 单图链路。

![固定推理 Demo](docs/assets/demo_inference_result.gif)

## 1. 项目解决的问题

仅有一个模型文件并不能构成可部署的软件产品。Runtime 还必须回答：

- 实际执行的是哪一套模型、张量、预处理、后处理与阈值契约？
- 同一张图片能否产生确定、可检查的 JSON 与可视化结果？
- 独立实现的 Python 与 C++ 能否在声明的数值门限内保持一致？
- 失败是否可定位，并且性能数字是否只在通过正确性门禁后发布？
- 同一套核心逻辑以后能否跨操作系统、架构、工作负载和推理后端迁移，
  而不复制产品语义？

因此，项目希望形成的秋招叙事是：

> 我把一个已有工业视觉产物转化为有证据支撑的 C++ Runtime，
> 再将其强化为跨平台推理与系统工程项目。

## 2. 架构

### 已验证的大阶段一 Runtime

```text
RuntimeConfig + ModelArtifactSpec
                |
                v
      actual ONNX ModelMetadata
                |
                v
OpenCV decode -> letterbox -> RGB -> float32 NCHW
                |
                v
      ONNX Runtime C++ CPU Session::Run
                |
                v
          owned raw output
                |
                v
YOLO decode -> score filter -> stable NMS -> coordinate restore
                |
                v
 DetectionResult -> schema-v1 JSON + headless visualization
                |
                v
 GTest/CTest + Python/C++ consistency + Release benchmark
```

### 大阶段二目标

```text
FP32 ONNX
  -> static INT8 PTQ + ORT operator/node profiling
  -> Windows and Linux x86_64 shared-source Runtime
  -> AArch64 cross-build + QEMU portability smoke
  -> directory/manifest + bounded queue + workers
  -> Linux x86_64 + RTX 4060 TensorRT path
  -> full evidence, resume variants, interview closure, recruiting freeze
```

产品后端可以扩展，但检测契约保持稳定：

```text
contract + metadata -> model-specific preprocess
                    -> ORT CPU FP32/INT8 or TensorRT
                    -> owned inference output
                    -> the same decode/filter/NMS/restore semantics
```

核心架构原则：

- Runtime 库负责可复用行为，CLI 保持轻量。
- Windows、Linux 和 AArch64 共用同一份业务源码。
- 平台差异仅保留在依赖发现、动态库、内存/信号适配器和工作流脚本中。
- 后续多图 Worker 复用现有单图 `DetectorPipeline`，不复制预处理、推理或后处理。
- 正式 Benchmark 前必须先通过正确性门禁。
- 后端抽象只保留真实后端所需的最小边界。
- 每项结果都记录命令、契约、产物身份、样本、环境、原始证据和限制。

## 3. 大阶段一最终记录

大阶段一已经作为一个完整交付闭环完成。下表仅保留后续仍需携带的事实，
不再复原过去的内部任务拆分。

| 证据 | 最终记录 |
|---|---|
| 模型产物 | `models/best.onnx`，12,336,935 bytes，opset 17，SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` |
| 实际张量契约 | 输入 `images` float32 `[1,3,800,800]`；输出 `output0` float32 `[1,10,13125]`；显式使用 `CPUExecutionProvider` |
| 已验证构建环境 | Windows 10.0.26200、x86_64、MSVC 19.50.35721.0、Release C++17、OpenCV 4.8.0、ONNX Runtime C++ 1.19.2 |
| 固定 Demo | `crazing_241.jpg` 产生三个 `crazing` 检测；已提交的 JSON 可解析，PNG 可由 OpenCV 读取 |
| 自动工程门 | 全新仓库外 Release 构建；106/106 个 CTest/GTest/CLI/Python/negative/integration 用例通过 |
| Python ORT/C++ ORT 一致性 | 冻结的六类别 manifest，每类五张图：30/30 张图片及 62/62 个检测均通过检测数量与类别精确匹配 |
| 最大一致性误差 | 置信度 `8.049977111568296e-07`；bbox 坐标 `9.135351561440075e-05 px`；最小匹配 IoU `0.999998927116394` |
| 正式已跟踪 CPU Benchmark | 固定图片、batch 1、warmup 10/repeat 100：端到端 mean/P50/P95 为 `176.553060/176.1357/196.6128 ms`，`5.664020 img/s` |
| 内存 | Windows 进程生命周期 Peak Working Set `152.714844 MiB` |
| 失败行为 | 模型缺失、图片损坏、输出父路径非法和 Benchmark repeat 非法时以非零码失败，并给出对象/路径、预期、实际值与修正动作；合法空检测仍是有效结果 |
| 用户验收 | 自动工程门，以及 L2 讲解/排错/修改验收均已完成 |

仓库内主要证据：

- [Runtime 配置](cpp_infer/configs/default_config.txt)
- [YOLOv8 产物契约](cpp_infer/artifacts/yolov8_neu_det.artifact.txt)
- [冻结的一致性 manifest](cpp_infer/tests/fixtures/consistency_manifest.json)
- [Demo 输出](cpp_infer/results/demo/)
- [一致性汇总](cpp_infer/results/consistency/summary.json)与
  [逐图证据](cpp_infer/results/consistency/per_image.json)
- [正式已跟踪 Benchmark](cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json)

包含独立临时收口复现的合并详情，见
[大阶段一收口详情](docs/details/stage1_closure.md)。

## 4. 快速开始

### 依赖要求

已验证的 Windows 工作流使用 x64 MSVC 环境、OpenCV C++ 4.8.0、
完整的 ONNX Runtime C++ 1.19.2 SDK、用于一致性验证的兼容 Python 环境，
以及固定的 GoogleTest 源码策略。机器路径必须写入已忽略的
`cpp_infer/.stage1.local.psd1` 或环境变量，绝不能写入已跟踪的 CMake 或源码。

便携发现无法找到依赖时，复制并填写可选本地模板：

```powershell
Copy-Item .\cpp_infer\tools\stage1.local.example.psd1 .\cpp_infer\.stage1.local.psd1
```

### 标准命令

在仓库根目录的普通 PowerShell 或 CMD 中运行：

```powershell
# 查看工作流，不启动构建。
.\cpp_infer\tools\stage1.cmd help

# 只读验证工具链与依赖。
.\cpp_infer\tools\stage1.cmd doctor

# 全新 Release 构建 -> 106 项测试 -> Demo -> 一致性 -> Benchmark。
.\cpp_infer\tools\stage1.cmd all

# 运行任意单张图片；可选第二个参数指定输出目录。
.\cpp_infer\tools\stage1.cmd detect "D:\images\sample.jpg" "D:\outputs"
```

`detect` 仍是复用 `DetectorPipeline` 的单图便捷入口，并不是目录批处理。
完整 action 矩阵、底层命令、环境路径和安全临时构建规则见
[路径与命令](docs/paths_commands.md)。

## 5. 核心模块

| 边界 | 职责 |
|---|---|
| Runtime/artifact/metadata 契约 | 分离可调 Runtime 策略、声明的产物语义和 ORT 实际观察到的张量/provider 事实；在推理前拒绝不匹配 |
| `ImagePreprocessor` | 解码或接收 `CV_8UC3` 图片；执行 letterbox、BGR-to-RGB、归一化，生成连续 NCHW 数据，并保留逆变换元数据 |
| `OnnxRunner` / `InferenceOutput` | 通过 RAII/PImpl 管理 ORT 资源，验证输入/输出，同步执行，并把输出复制到生命周期独立于 ORT 的存储中 |
| 后处理/NMS | 验证 YOLO BCN 输出，选择类别分数，执行严格过滤与稳定的类别无关 NMS，再恢复并裁剪源图坐标 |
| `DetectorPipeline` 与 writers | 编排单张图片并输出自持有结果、稳定 JSON 和确定性的无 GUI 可视化，同时强制安全输出路径 |
| 证据工具链 | 用合成 fixture 测试纯逻辑接缝，谨慎测试真实垂直链，比较 Python/C++ 检测，并发布结构化 Benchmark/内存证据 |

## 6. 大阶段二计划

大阶段二划分为五个完整交付单元。每个单元先冻结最小 SPEC，再实现一个可运行能力，
产出测试和机器可读证据，更新三个项目入口文档，然后停止并等待 L1 验收。

| 单元 | 交付内容 | 诚实边界 | 状态 |
|---|---|---|---|
| S2-01 | Static INT8 PTQ、FP32/INT8 正确性/任务质量/性能对比、ORT operator/node profiling | 不做 QAT、D010 量化，也不把 profiler 数据冒充 Benchmark | **下一步；尚未开始** |
| S2-02 | Linux x86_64 原生链路、共享源码可移植性、AArch64 交叉构建与 QEMU smoke | WSL2 不是开发板；QEMU 不产出性能结论 | 计划中 |
| S2-03 | 目录/manifest 发现、有界队列、workers、背压、失败计数、干净退出、吞吐对比 | 并发单图任务不等于真正的 ONNX batch | 计划中 |
| S2-04 | 一条真实 Linux x86_64 + RTX 4060 TensorRT 执行路径、FP16 正确性与性能 | 仅为本地 GPU/边缘节点证据；不是 Jetson 或嵌入式部署 | 计划中 |
| S2-05 | 适用的完整门禁、结果矩阵、失败案例、三套简历叙事、面试材料、recruiting freeze | 不增加新技术栈 | 计划中 |

### 平台矩阵

| 平台/后端 | 能证明什么 | 当前状态 |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | 当前产品链、正确性、测试、分段 Benchmark、Peak Working Set | 已验证 |
| Windows/Linux x86_64 + ORT CPU INT8 | 在冻结协议下验证量化质量、大小、延迟、内存和 profiling | 计划在 S2-01/S2-02 完成 |
| WSL2 Ubuntu 24.04 x86_64 + ORT CPU | Linux 构建/加载/Runtime 可移植性、一致性、Benchmark、peak RSS | 计划在 S2-02 完成 |
| Linux AArch64 under QEMU | 仅证明交叉编译和可移植性正确性 | 计划在 S2-02 完成；不发布性能结论 |
| Linux x86_64 + RTX 4060 + TensorRT | 真实本地 TensorRT 执行、FP16 正确性/性能、GPU 内存 | 计划在 S2-04 完成；不是 Jetson |

当前简历无需等待大阶段二即可使用。已完成的 S2 单元可以滚动更新简历；
尚未完成的目标绝不能写成已交付结果。

## 7. 证据边界与当前限制

- 30 图结果证明的是同一 ONNX 产物下的实现一致性；它不是检测器 mAP，
  也不是新的 PyTorch/ONNX/C++ 三方对比。
- 匹配的 `.pt` checkpoint 当前不存在。现有 ONNX 的来源由项目所有者确认，
  但目前无法在本工作区重新导出。
- 正式 Benchmark 仅使用一张 `200x200` 图片、batch 1、暖文件缓存、
  一台 Windows CPU 主机、串行 ORT 执行，并且没有锁定 CPU affinity/priority。
- Peak Working Set 是进程生命周期的高水位，不是模型专属或单次推理内存。
- `CPUExecutionProvider` 是 session 级执行证据。逐节点归属需要计划中的 ORT profiling。
- 当前 Runtime 仅支持 CPU 单图。INT8、Linux、AArch64/QEMU、有界并发和 TensorRT
  在各自门禁通过前都仍是计划能力。
- 历史 Python ORT `24.4/72.1 FPS` 使用不同实现、provider、硬件、样本和计时边界；
  只能作为背景，不能与 C++ 结果排名比较。
- 源码、模型与数据集许可证是彼此独立的检查点。MIT 源码许可证不会自动为已分发的
  ONNX 或 NEU-DET 重新授权。

## 8. 历史资产、冻结扩展与参考资料

仓库仍保留最初的 Python 训练、评估、Python ONNX Runtime、FastAPI 和 Docker 资产。
它们是受保护的历史 baseline 材料，不是 V2 产品主线。其原有长篇教程保存在 S2 前的
README 归档中，不在这里重复。

`paper_detect` D010 仍是研究侧产物来源。只有在具备稳定 ONNX 导出、result/artifact card、
部署契约、真实 adapter 集成和一致性验证后，它才能进入本 Runtime。
研究指标不是 C++ Runtime 结果。

D010 集成、Qt、本地 LLM、Agent 工作流和真实 ARM/Jetson 设备保持冻结，
除非秋招证据或具体职位描述能够证明有必要重新开放。

权威资料与操作参考：

- [秋招路线](docs/路线0712-new.md)
- [大阶段二顶层设计](docs/Proj1_S2.md)
- [大阶段一合并收口](docs/details/stage1_closure.md)
- [路径、命令与环境](docs/paths_commands.md)
- [C++ Runtime 技术参考](cpp_infer/README.md)

根目录中英文 README 始终是公开项目入口。详细文档用于支撑主线叙事，
而不是取代它。

## 许可证

仓库源码以 [MIT License](LICENSE) 发布。已跟踪 ONNX 记录了 Ultralytics
AGPL-3.0 元数据；在发布或再分发前，仍需单独核验 NEU-DET 的再分发条款。
当前来源边界见产物契约与大阶段一收口记录。
