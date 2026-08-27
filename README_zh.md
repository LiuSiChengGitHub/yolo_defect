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

> **状态 — 2026-08-28：** 大阶段一与用户负责的 L2 已完成；S2-01 的 Windows
> CPU INT8/PTQ/profiling 实现与证据已按记录的 advisory 练习口径完成。
> **S2-02 Gate A 的 WSL2/Linux x86_64 Native 已完成：** 共享源码 Release Runtime、
> 测试、固定单图产品链、Python/C++ 一致性、短 Benchmark/peak RSS 与 ELF 依赖检查
> 均通过，随后 Windows 回归保持全绿。当前停止并等待用户 L1/后续方向；Gate B
> （AArch64 交叉构建与 QEMU）尚未开始，因此不声称 S2-02 整个单元已经完成。

![固定推理 Demo](docs/assets/demo_inference_result.gif)

## 1. 项目解决的问题

仅有一个模型文件并不能构成可部署的软件产品。Runtime 还必须回答：

- 实际执行的是哪一套模型、张量、预处理、后处理与阈值契约？
- 同一张图片能否产生确定、可检查的 JSON 与可视化结果？
- 独立实现的 Python 与 C++ 能否在声明的数值门限内保持一致？
- 失败是否可定位，并且性能数字是否绑定显式、机器可读的正确性策略，
  而不是悄悄脱离质量结果？
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
  -> [Windows CPU 已交付] static INT8 PTQ + ORT operator/node profiling
  -> [Gate A 已交付] Windows and Linux x86_64 shared-source Runtime
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
- 发布 Benchmark 前，在相同 artifact/config 下完成代表性正确性 smoke；S2-01
  额外保留原始产品差异与任务质量结果作为非阻断诊断。
- 后端抽象只保留真实后端所需的最小边界。
- 性能结论记录解释它所需的 artifact、样本、运行条件和限制；普通输出不要求
  新建 evidence bundle 或 hash 台账。

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

在仓库根目录的普通 PowerShell 或 CMD 中调用 Windows task runner；它会
自动发现并初始化所需的 Visual Studio 环境：

```powershell
.\cpp_infer\tools\stage1.cmd help
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
```

在 WSL2/Linux x86_64 中选择固定版本的 Linux SDK，并使用 Bash 入口：

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON=/path/to/python
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest

bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark
bash cpp_infer/tools/stage1.sh all
```

完整 action 矩阵、当前依赖路径、本机配置优先级、底层 CMake/CTest 审计命令
和环境故障诊断只保存在[路径、工具链与环境诊断](docs/paths_commands.md)。
Gate A 的准确机器快照、证据与解释见
[S2-02 Gate A 收口](docs/details/s2_02_gate_a_closure.md)。

## 5. 核心模块

| 边界 | 职责 |
|---|---|
| Runtime/artifact/metadata 契约 | 分离可调 Runtime 策略、声明的产物语义和 ORT 实际观察到的张量/provider 事实；在推理前拒绝不匹配 |
| `ImagePreprocessor` | 解码或接收 `CV_8UC3` 图片；执行 letterbox、BGR-to-RGB、归一化，生成连续 NCHW 数据，并保留逆变换元数据 |
| `OnnxRunner` / `InferenceOutput` | 通过 RAII/PImpl 管理 ORT 资源，验证输入/输出，同步执行，并把输出复制到生命周期独立于 ORT 的存储中 |
| Static PTQ 工具链 | 冻结校准输入与量化配置，按声明的 activation/weight 类型执行 Conv-only QDQ PTQ，检查选中/量化/失败节点，验证 actual metadata，并生成派生产物卡 |
| `ProfileRunner` 与 profile 汇总器 | 创建隔离的 profiling session，保留 ORT raw trace，按 node/operator/provider 汇总耗时与调用次数，并把 trace 耗时排除在正式 Benchmark 外 |
| 后处理/NMS | 验证 YOLO BCN 输出，选择类别分数，执行严格过滤与稳定的类别无关 NMS，再恢复并裁剪源图坐标 |
| `DetectorPipeline` 与 writers | 编排单张图片并输出自持有结果、稳定 JSON 和确定性的无 GUI 可视化，同时强制安全输出路径 |
| 跨平台构建/平台层 | 保持 Runtime、预处理、后处理和 Pipeline 共用同一源码；CMake 选择 Windows `.lib`/`.dll` staging 或 Linux `libonnxruntime.so` 加 build RPATH，薄 `platform_info` 适配器分别报告 Windows Peak Working Set 与 Linux `getrusage` peak RSS |
| `project_core` 可移植性 smoke | 抽出仅依赖标准库的 YOLO decode/NMS/坐标恢复行为，为未来 Gate B 提供 core smoke，但不冒充已经完成 AArch64 ORT/OpenCV 全链推理 |
| 验证工具链 | 用聚焦 fixture 测试有意义的逻辑接缝，谨慎运行真实垂直链，在相关时比较 Python/C++ 检测并记录限定范围的 Benchmark/内存结果 |

## 6. 大阶段二计划

大阶段二划分为五个完整交付单元。每个单元先定义最小 SPEC，再实现一个可运行能力，
做与改动相称的验证，只记录解释该能力所需的结果，更新三个项目入口文档，然后停止并等待 L1 验收。

| 单元 | 交付内容 | 诚实边界 | 状态 |
|---|---|---|---|
| S2-01 | Static INT8 PTQ、FP32/INT8 正确性/任务质量/性能对比、ORT operator/node profiling | Windows CPU 个人练习收口；产品/质量结果为 advisory；不做 QAT，也不把 profiler 冒充 Benchmark | **实现/证据完成；等待 L1** |
| S2-02 | Linux x86_64 原生链路、共享源码可移植性、AArch64 交叉构建与 QEMU smoke | WSL2 不是开发板；QEMU 不产出性能结论 | **Gate A 已完成；等待 L1/方向；Gate B 尚未开始** |
| S2-03 | 目录/manifest 发现、有界队列、workers、背压、失败计数、干净退出、吞吐对比 | 并发单图任务不等于真正的 ONNX batch | 计划中 |
| S2-04 | 一条真实 Linux x86_64 + RTX 4060 TensorRT 执行路径、FP16 正确性与性能 | 仅为本地 GPU/边缘节点证据；不是 Jetson 或嵌入式部署 | 计划中 |
| S2-05 | 适用的完整门禁、结果矩阵、失败案例、三套简历叙事、面试材料、recruiting freeze | 不增加新技术栈 | 计划中 |

### S2-01 Windows CPU 记录

最终本地产物使用 ONNX Runtime 1.19.2 static PTQ，配置为 `QDQ`、U8S8、
MinMax 校准、per-channel 权重，并将源图全部 64 个 `Conv` 纳入量化目标。
它相对 Round 1 S8S8 协议只改变 activation 类型。外部契约仍为 float32
`images [1,3,800,800] -> output0 [1,10,13125]`；
INT8 是图内部表示，不是应用侧整数 I/O 契约。

| 证据 | FP32 | INT8 / 结果 |
|---|---:|---:|
| 模型文件 | 12,336,935 bytes | 3,544,494 bytes；**缩小 71.269%** |
| Python/C++ ORT 合法性 | 通过 | 通过；输出有限且 actual metadata 一致 |
| 当前 Windows 回归 | 118/118 CTest 通过 | FP32/INT8 profile workflow smoke 通过 |
| 361 图任务质量 | mAP50 `0.710815`；mAP50-95 `0.345786` | `0.700459` / `0.342379`；delta `-0.010356/-0.003407` |
| 30 图产品差异 | 62 个检测 | 65 个检测、61 个匹配；原始 aggregate gate 为 `false` |
| Session 初始化 | `61.986 ms` | `94.858 ms`；一次性 setup 更慢 |
| `Session::Run` mean/P50/P95 | `155.106/155.124/169.639 ms` | `95.040/95.570/110.768 ms`；**mean 快 38.726%** |
| Pipeline mean/P50/P95 | `163.477/163.221/182.008 ms` | `103.872/104.042/121.654 ms`；**mean 快 36.461%** |
| Pipeline 吞吐 | `6.117 img/s` | `9.627 img/s`；**提升 57.383%** |
| Peak Working Set | `150.980 MiB` | `148.832 MiB`；仅是进程高水位的小幅变化 |

Round 1 虽在静态 QDQ 文件中量化了全部 64 个 Conv，但 ORT 优化后的 S8S8
执行图仍有 57 个 float `Conv`，只有 7 个 `QLinearConv`，每次运行还包含
120 个 Q 和 317 个 DQ，因此 `Session::Run` 反而慢 37.16%。Round 2 只将
activation 从 `QInt8` 改成 `QUInt8`；10-run trace 出现 640 次
`QLinearConv` 且没有普通 `Conv`，即每次 64 个整数卷积。`QLinearConv` 现在占
诊断 kernel-event 时间 35.18%，DQ、Resize、Mul、Concat、Q 和 Sigmoid 则成为
下一批热点；全部优化后节点均位于 `CPUExecutionProvider`。

这形成了本单元最重要的学习结论：ONNX 文件中存在 QDQ 不等于实际得到整数
kernel 加速，必须同时检查优化执行图，并用关闭 profiler 的正式 Benchmark 验证。
Profile event 总时长包含插桩开销，不能替代 `Session::Run`。

`models/best.int8.qdq.u8s8.onnx` 当前存在，并已由记录中的 Python/C++ 运行实际加载；
但派生 ONNX 继续遵守项目的模型许可证边界而被 Git 忽略。新 clone 应从冻结 protocol
重建同 SHA 二进制；Git 交付的是绑定 SHA 的 contract、card、工具和机器证据，
而不是暗示仓库正在分发该模型。

S2-01 主要证据：

- [Round 2 PTQ 协议](cpp_infer/protocols/s2_01_ptq_protocol_r2_u8s8.json)与
  [U8S8 产物契约](cpp_infer/artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt)
- [量化产物卡](cpp_infer/results/s2_01/round2/u8s8/quantization_report.json)
- [未改写的正确性/质量结果](cpp_infer/results/s2_01/round2/correctness_u8s8.json)
- [FP32/U8S8 Benchmark 比较](cpp_infer/results/s2_01/round2/benchmark/comparison_u8s8.json)
- [FP32 profile 摘要](cpp_infer/results/s2_01/round2/profile/fp32_summary.json)与
  [U8S8 profile 摘要](cpp_infer/results/s2_01/round2/profile/int8_u8s8_summary.json)
- [Round 2 收口、失败分析与复现详情](docs/details/s2_01_round2_closure.md)
- [Round 1 S8S8 历史收口](docs/details/s2_01_closure.md)

### S2-02 Gate A Linux x86_64 记录

Gate A 保持现有 Runtime、预处理、后处理和 `DetectorPipeline` 为唯一业务主链。
CMake 现在按平台选择 Windows `.lib`/`.dll` 契约或带 build RPATH 的 Linux
`libonnxruntime.so`；薄 `platform_info` 适配器选择 Windows Peak Working Set 或
Linux `getrusage` peak RSS。仅依赖标准库的 `project_core` smoke 覆盖 YOLO decode、
类别无关 NMS 与坐标恢复，为 Gate B 做准备。

| Gate A 证据 | 实测结果 |
|---|---|
| Linux clean Release | WSL2/Linux x86_64、Ninja，119/119 CTest 通过 |
| 固定产品链 | `crazing_241.jpg`，3 个检测，JSON 合法且 PNG 可读 |
| Python/C++ 一致性 | 30/30 张图片、62/62 个匹配检测通过冻结门限 |
| 短性能 smoke | 一次 warmup-1/repeat-2 样本：端到端 mean `135.896991 ms`、`7.358515 img/s`、peak RSS `196.570312 MiB`；持久化收口的 1/2 复跑为 `151.273896 ms`、`6.610526 img/s`、`196.757812 MiB`，显示波动很大 |
| 动态加载 | 用 `ldd` 检查 9 个已构建 ELF 可执行文件；无依赖报告 `not found`，ORT 经配置的 Linux SDK/RPATH 解析 |
| Windows 回归 | Release/NMake 119/119 CTest 通过 |

已提交的固定单图产物位于
[`cpp_infer/results/s2_02/linux_x86_64/`](cpp_infer/results/s2_02/linux_x86_64/)。
命令、机器快照与完整证据解释见[路径、工具链与环境诊断](docs/paths_commands.md)和
[S2-02 Gate A 收口](docs/details/s2_02_gate_a_closure.md)。

### 平台矩阵

| 平台/后端 | 能证明什么 | 当前状态 |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | 当前产品链、正确性、测试、分段 Benchmark、Peak Working Set | 已验证 |
| Windows x86_64 + ORT CPU INT8 | Static PTQ 产物、Runtime 合法性、大小/质量/性能比较、逐节点 profiling | S2-01 已按 advisory 练习口径验证 |
| WSL2/Linux x86_64 + ORT CPU INT8 | 潜在的共享源码 Linux INT8 路径 | Gate A 未单独实测；不作 Linux INT8 对比结论 |
| WSL2 Ubuntu 24.04 x86_64 + ORT CPU FP32 | Linux 构建/加载/Runtime 可移植性、一致性、短 Benchmark、peak RSS | **S2-02 Gate A 已验证；等待 L1/方向** |
| Linux AArch64 under QEMU | 仅证明交叉编译和可移植性正确性 | **Gate B 尚未开始**；不发布性能结论 |
| Linux x86_64 + RTX 4060 + TensorRT | 真实本地 TensorRT 执行、FP16 正确性/性能、GPU 内存 | 计划在 S2-04 完成；不是 Jetson |

当前简历无需等待大阶段二即可使用。已完成的 S2 单元可以滚动更新简历；
尚未完成的目标绝不能写成已交付结果。

## 7. 证据边界与当前限制

- 30 图结果证明的是同一 ONNX 产物下的实现一致性；它不是检测器 mAP，
  也不是新的 PyTorch/ONNX/C++ 三方对比。
- 匹配的 `.pt` checkpoint 当前不存在。现有 ONNX 的来源由项目所有者确认，
  但目前无法在本工作区重新导出。
- 正式已跟踪 Benchmark 仅使用一张 `200x200` 图片、batch 1、暖文件缓存、
  一台 Windows CPU 主机、串行 ORT 执行，并且没有锁定 CPU affinity/priority。
  Gate A 的 Linux warmup-1/repeat-2 样本波动明显，只是功能性性能 smoke，
  不是正式结果，也不能用作跨操作系统速度比较。
- Windows Peak Working Set 与 Linux `getrusage` peak RSS 都是进程生命周期高水位，
  但平台语义不同；二者都不是模型专属或单次推理内存，数值也不能直接比较。
- S2-01 ORT trace 已证明优化后节点由 `CPUExecutionProvider` 执行，但 trace
  时长含 profiler 开销，也不能说明 kernel 内部最终选择了哪条 CPU 指令。
- 当前 Runtime 仍为 CPU 单图；Windows INT8 与 S2-02 Gate A 的 WSL2/Linux
  x86_64 FP32 路径已交付。Gate B/AArch64/QEMU 尚未开始；有界并发和 TensorRT
  也仍在计划中。
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
- [S2-01 INT8/PTQ 与 profiling 收口](docs/details/s2_01_closure.md)
- [S2-02 Gate A Linux x86_64 收口](docs/details/s2_02_gate_a_closure.md)
- [路径、命令与环境](docs/paths_commands.md)
- [C++ Runtime 技术参考](cpp_infer/README.md)

根目录中英文 README 始终是公开项目入口。详细文档用于支撑主线叙事，
而不是取代它。

## 许可证

仓库源码以 [MIT License](LICENSE) 发布。已跟踪 ONNX 记录了 Ultralytics
AGPL-3.0 元数据；在发布或再分发前，仍需单独核验 NEU-DET 的再分发条款。
当前来源边界见产物契约与大阶段一收口记录。
