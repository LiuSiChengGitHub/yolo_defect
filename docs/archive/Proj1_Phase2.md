# 大阶段二执行方案：P0 收口与关键 P1 加速扩展

> - 规划来源：`路线0712`、`Note_Proj1.md`、`Project1_Phase1.md`、`web.md`、`docs/PLAN.md` 与当前仓库实际状态
> - 生成日期：2026-08-23
> - 项目仓库：`D:\01_Base\CodingSpace\yolo_defect`
> - 当前开发分支：`deploy-cpp`
> - 当前基线 HEAD：`46ca8448614981704bc223fa1c62d5d8a3a4fc1b`
> - 前置状态：大阶段一自动工程门已经通过，用户 L2 已完成；仓库三份 README 仍需把旧的 `L2 PENDING / Stage One incomplete` 状态同步为真实状态
> - 本阶段定位：在秋招已经启动的前提下，用较少但更完整的功能单元完成 P0 剩余项，并优先补齐 Linux、C++ 产品化、多模型和本地生成式 AI 证据

---

## 1. 使用方式

这是一份进入“大阶段二”前，基于当前代码、环境调查和 D010 artifact 交接动态生成的执行方案。

它规定的是：

- 大阶段二必须解决的工程问题；
- 六个完整功能单元的边界；
- 每个单元的输入、输出、非目标、证据和停损条件；
- 哪些目标是硬目标，哪些目标受硬件或 artifact 门禁约束；
- 用户在每个单元完成后需要达到的 L1，以及大阶段结束时的 L2。

它不提前锁死：

- 具体类名、函数签名和目录树；
- 每个源文件应如何拆分；
- 线程池采用哪一种现成或自研实现；
- Linux 依赖最终采用系统包、官方压缩包还是其他可复现方式；
- D010 adapter 的最终类型名；
- CUDA/TensorRT 的安装细节；
- 本地语言模型的具体型号。

这些实现细节由 Codex 在每个 `S2-*` 开始时，根据当时的源码、构建结果、依赖和测试动态决定。

执行规则：

1. 一次只推进一个 `S2-*` 完整单元。
2. 每个单元先写或更新该功能的 SPEC，再实现代码、测试、证据和 README。
3. 每个单元完成后立即停止，由用户完成 L1；通过后才进入下一单元。
4. 不再像大阶段一那样把一个纵切拆成大量微阶段；每个 `S2-*` 应交付一个用户可感知、可运行、可测试的完整能力。
5. Codex 可以调整尚未执行单元的内部实现，但不能降低本方案的大阶段出口。
6. 出现真实面试、笔试或高优先级投递时，项目必须在当前稳定 checkpoint 暂停，面试优先。
7. 不等待大阶段二全部完成后再投递。每完成一个能形成新 bullet 的单元，就滚动更新简历和口述材料。

---

## 2. 当前事实与范围判断

### 2.1 大阶段一已经完成什么

当前 YOLOv8/NEU-DET C++ Runtime 已经具备：

```text
RuntimeConfig + ModelArtifactSpec
-> 实际 ONNX ModelMetadata 校验
-> OpenCV letterbox / RGB / normalize / NCHW
-> ONNX Runtime C++ CPU Session::Run
-> 自有生命周期 raw output
-> YOLOv8 BCN decode / strict score filter / stable class-agnostic NMS
-> letterbox 坐标反算与 clip
-> Detection JSON / 无 GUI 可视化
-> 六类 30 图 Python ORT / C++ ORT 一致性
-> Release 分段 benchmark / throughput / Peak Working Set
-> 106 项 CTest/GTest/CLI/Python/negative/integration gate
-> 统一 Windows stage1.cmd 工作流
```

大阶段一的用户 L2 已由用户确认完成，因此大阶段二不再设置补验收小阶段。S2-01 只需要把 README 中的旧状态同步为真实状态。

### 2.2 P0 完成矩阵

| P0 项 | 当前状态 | 大阶段二动作 |
|---|---|---|
| C++17/CMake 多 target | 已完成 | 保持 Windows 回归，并改造成 Linux 可构建 |
| Config 与 artifact schema | 已完成 | 为 INT8 和多输入/多模型扩展，不破坏 v1 |
| OpenCV 前处理 | YOLO 已完成 | D010 增加 direct-resize、paired input/prior adapter |
| ORT C++ session / metadata / RAII | CPU 单输入单输出已完成 | 扩展量化 artifact、多输入多输出和可选 provider |
| YOLO decode/filter/NMS/坐标还原 | 已完成 | 保持冻结语义和回归 |
| JSON 与可视化 | 已完成 | 批处理、多模型继续复用统一结果对象 |
| Python/ONNX/C++ 一致性 | YOLO 已完成 | INT8、Linux和第二模型分别建立新证据 |
| Benchmark / memory | Windows 单图 CPU 已完成 | 增加 INT8、Linux、多图吞吐和并发证据 |
| INT8 PTQ / QAT | **未完成** | S2-01 硬目标；QAT 仅条件启动 |
| GTest / invalid inputs | 已有较完整矩阵 | 新行为随实现同步补测试，不重复追求测试数量 |
| Profiling | **未完成** | S2-01 增加逐算子/节点 profile 与摘要 |
| 故障注入 | 已完成核心矩阵 | Linux、并发、D010、LLM 新增对应失败场景 |
| README / 环境 / 结果 | 主线已完成，状态仍过期 | S2-01 同步 S1 完成状态；S2-06最终收口 |

结论：大阶段二不应重新做一次“大阶段一测试加固”。P0 的明确剩余硬缺口是：

```text
YOLO FP32 -> INT8 PTQ
+ 正确性/任务精度/性能/模型大小比较
+ ORT operator-level profiling
+ 最终 P0 结果表与 README 同步
```

### 2.3 环境 readiness

| 能力 | 当前判定 | 对本阶段的影响 |
|---|---|---|
| Windows C++ CPU | READY | S2-01 可直接开始；所有阶段必须保留 Windows 回归 |
| WSL2 Ubuntu 24.04 x86_64 | PARTIAL | 可作为 Linux 主载体，但需要准备 GCC/CMake/OpenCV/ORT 等工具链 |
| Docker | BLOCKED | 不纳入 Linux 硬目标，不允许因此阻塞 S2-02 |
| CUDA / ORT CUDA EP | PARTIAL/BLOCKED | 不作为 S2-01～S2-04 前置；只在 S2-05 timebox 内处理 |
| TensorRT | BLOCKED | 不能作为大阶段二无条件出口 |
| 真实 ARM64 / Jetson | UNKNOWN/BLOCKED | 无设备时不得写 Jetson 性能、温度或功耗结果 |
| D010 handoff | PARTIAL | 有 checkpoint、ONNX、contract、runner、manifest；仍需关闭 Runtime product gate |
| 本地小语言模型 | 硬件可行、软件未准备 | 适合在 S2-06 做窄范围 PoC，不做大模型系统 |

### 2.4 D010 当前可用事实

D010 当前已有本地交接产物：

```text
checkpoint:
  d010_best_stg2.pth
  165,234,418 bytes
  SHA-256 e168f75d...

ONNX:
  d010_best_stg2_opset16.onnx
  41,790,142 bytes
  SHA-256 baf73771...

inputs:
  tested_rgb        float32 [N,3,640,640]
  edge_prior        float32 [N,1,640,640]
  orig_target_sizes int64   [N,2]

outputs:
  labels int64   [N,300]
  boxes  float32 [N,300,4] xyxy，原图像素
  scores float32 [N,300]

semantics:
  direct resize 640
  no letterbox
  no mean/std
  no NMS
  score >= 0.35
  class order = open, short, mousebite, spur, copper, pin_hole
```

现有 thresholded 结果已达到：10/10 数量一致、10/10 类别一致、最大 confidence 误差约 `4.17e-6`、最大坐标误差约 `9.16e-5 px`。

仍未关闭的门：

- 未过滤 top-300 中一个低分近并列项发生排序变化，原交接整体仍标记 `failed_consistency`；
- C++ 侧 tested/template 解码、prior 生成和输入 tensor 还没有 golden test；
- Pillow 版本导致一张 JPEG 的 tested tensor 不同，Runtime 需要建立自己的 canonical decoder reference；
- 轻量 handoff 当前没有形成远程分支可复现的正式交接状态；
- checkpoint provenance 是证据绑定，不是 checkpoint 内嵌 commit 的密码学绑定；
- DeepPCB research-only 与源码/模型许可证口径仍需明确限制。

因此 D010 是本阶段的首选第二模型，但不能在门禁关闭前被写成已完成 Runtime 成果。

---

## 3. 大阶段二范围冻结

### 3.1 硬目标

本阶段硬目标为：

1. **完成 P0 剩余项：**YOLO INT8 PTQ、FP32/INT8 对比和 ORT Profiling。
2. **Linux x86_64：**同一源码在 WSL2 Ubuntu 完成 clean build、测试、Demo、一致性、benchmark 和 peak RSS。
3. **C++ 产品/系统软件：**目录或 manifest 批处理、有界队列、worker、backpressure、clean shutdown、单线程/并发吞吐比较。
4. **第二模型 artifact：**至少接入一个 DETR 家族 artifact；D010 为首选，D-FINE-S 为第一 fallback，官方 RT-DETR 为第二 fallback。
5. **本地小语言模型部署 PoC：**将 Detection JSON 转为结构化质检报告，对同一小模型的 4-bit/8-bit 版本做资源与性能比较。
6. **测试与证据贯穿：**新功能同步建立 unit/integration/negative/consistency/benchmark，不单独堆测试数量。
7. **最终收口：**README、结果矩阵、失败案例、简历 bullet、口述与专项 mock。

### 3.2 条件目标

以下能力只有前置条件满足时进入：

- Windows/Linux 桌面 GPU 的 ORT CUDA 或 TensorRT backend；
- 借用、租用或远程访问真实 Jetson/ARM64 设备；
- Jetson 上的延迟、内存、温度、功耗或 power mode 结果；
- D010 INT8；
- 原生 TensorRT backend；
- 最小受控 Agent workflow。

条件目标不满足时，不阻塞大阶段二硬目标收口，也不得用模拟结果代替真实设备证据。

### 3.3 本阶段明确不做

- 为当前项目专门购买高价 Jetson；
- 把 WSL、QEMU 或桌面 GPU 描述成 Jetson 实机；
- 为了一个设备 bullet 进入 Yocto、BSP、驱动、内核或刷机深坑；
- 默认启动 QAT；
- 训练或重跑 D010 formal/official test；
- 完整 Qt 客户端；
- RAG、向量数据库、LangChain/LangGraph 后端系统；
- 通用聊天机器人；
- gRPC/Triton 服务化；
- Docker 作为 Linux 硬依赖；
- 同时维护多个大语言模型或大量量化版本；
- 因追求“功能多”破坏已经冻结的 YOLO P0 正确性。

### 3.4 术语与简历口径

必须严格区分：

```text
WSL2 Ubuntu x86_64
!= 嵌入式 Linux 实机
!= ARM64 实机
!= Jetson

桌面 RTX + TensorRT
!= Jetson 部署

可交叉编译
!= 已在目标设备运行

D010 论文精度
!= D010 C++ Runtime 结果

Detection JSON -> LLM 报告
!= Agent

固定目录多图任务
!= ONNX 真正 batch N>1
```

---

## 4. 目标架构

大阶段二完成后的目标链路为：

```text
                         +-----------------------------+
                         | Model Artifact Contract     |
                         | model / tensors / semantics |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         | Model-specific Input Adapter|
                         | YOLO: letterbox              |
                         | D010: pair + prior + resize  |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         | Inference Backend            |
                         | ORT CPU                      |
                         | optional CUDA / TensorRT     |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         | Model-specific Output Adapter|
                         | YOLO decode + NMS            |
                         | D010 top-300 + threshold     |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         | Unified Detection Result     |
                         | JSON / visualization         |
                         +-------+--------------+--------+
                                 |              |
                +----------------v--+       +---v----------------+
                | Batch Scheduler   |       | Local LLM Reporter |
                | bounded queue     |       | report JSON        |
                | workers/shutdown  |       | Q4/Q8 benchmark    |
                +-------------------+       +--------------------+
```

跨平台目标：

```text
same source
-> Windows x86_64 Release CPU
-> WSL2 Ubuntu x86_64 Release CPU
-> optional desktop CUDA/TensorRT
-> optional real ARM64/Jetson
```

架构原则：

- 继续保持 Runtime library 与薄 CLI 分离；
- 模型特定语义位于明确 adapter 边界，不在 `main.cpp` 用 if/else 临时拼接；
- batch/concurrency 调度复用单图核心能力，不复制 preprocess/infer/postprocess；
- JSON/可视化与本地 LLM 都消费统一结果契约；
- GPU/设备 backend 不改变检测语义，先过正确性门再测性能；
- 平台差异集中在依赖发现、内存采集、信号处理和 provider 装配层。

---

## 5. 总体拆分与关键路径

| ID | 完整功能单元 | 预计投入 | 类型 | 主要出口 |
|---|---|---:|---|---|
| S2-01 | P0 收口：INT8 PTQ + Profiling | 8–12h | 硬目标 | FP32/INT8 正确性、精度、性能、大小和逐算子证据 |
| S2-02 | Linux x86_64 跨平台 Runtime | 8–12h | 硬目标 | 同源码 Linux build/test/demo/consistency/benchmark/RSS |
| S2-03 | C++ 目录批处理与并发系统 | 10–14h | 硬目标 | 有界队列、worker、backpressure、shutdown、吞吐比较 |
| S2-04 | 多模型 Runtime 与 D010 接入 | 10–16h | 硬目标，artifact gate | 第二 DETR artifact、统一结果、跨实现一致性 |
| S2-05 | 真实端侧 / TensorRT 部署梯子 | 4–8h preflight；有设备时 10–16h | 条件目标 | desktop accelerator 或真实设备证据；无前置则按门禁跳过 |
| S2-06 | 本地 LLM 质检报告与大阶段收口 | 8–12h | 硬目标 | Q4/Q8 本地推理、结构化报告、总结果、简历和 L2 |

硬目标预计约 44–66 小时。按既定工作日/周末节奏，应控制在约两周左右完成主体，最多用第三周处理 D010 排错或条件设备验证。

关键路径：

```text
S2-01 freeze P0
-> S2-02 cross-platform foundation
-> S2-03 product/system software
-> S2-04 multi-artifact
-> S2-06 local GenAI + final acceptance

S2-05 hardware gate
  可在 S2-02 后调查，
  只有依赖/设备 READY 时插入，
  不允许阻塞关键路径。
```

滚动简历节点：

- S2-01 后：可增加 INT8/PTQ 与 operator profiling 证据；
- S2-02 后：可增加 Windows/Linux 同源码复现；
- S2-03 后：可增加 C++ bounded queue、backpressure 和并发吞吐；
- S2-04 后：可增加论文 artifact 到多模型 C++ Runtime 的交接链；
- S2-05 有实机后：才增加 Jetson/ARM/TensorRT 设备 bullet；
- S2-06 后：可增加本地量化 LLM 结构化质检报告。

---

## 6. 每个小阶段的共同完成定义

任一 `S2-*` 只有同时满足以下条件才可标记完成：

1. 先检查 `git status`、当前分支和已有用户改动，不覆盖 dirty worktree。
2. 当前功能有清晰 SPEC：目标、范围、用户流程、功能要求、边界、技术约束、验收标准。
3. 只推进当前单元，不偷跑后续模块。
4. 新增行为至少具有正向、关键边界和必要负向测试。
5. 真实命令在全新 Release 构建或明确的阶段环境中运行；不能只提交代码不执行。
6. 性能数据必须以前置正确性门通过为条件。
7. 结果记录模型、样本、环境、命令、版本、hash、容差和限制。
8. Windows 现有 YOLO gate 不得回退；从 S2-02 开始还要保留 Linux gate。
9. README 中 implemented、tested、conditional、blocked、historical 必须分开。
10. Codex 按项目要求输出教学闭环：做了什么、模块/文件、人工流程、入口与伪代码、运行测试排错、验收问题、代码练习、trade-off、README 更新。
11. 用户完成 L1 后才进入下一单元。
12. 大文件、模型和临时构建不无控制复制；D 盘空间、WSL 内存和 Git 分发边界必须受到保护。

建议每个阶段开始前建立可回滚 checkpoint。具体采用 commit、临时分支或 patch，由用户与 Codex根据当前 dirty worktree 决定，不允许擅自 reset/clean。

---

## 7. S2-01：P0 收口——INT8 PTQ 与 ORT Profiling

### 7.1 目标与招聘价值

把当前“正确且可测的 FP32 CPU Runtime”升级为“完成量化验证并能解释性能瓶颈的 Runtime”。

本阶段解决两个问题：

1. INT8 是否真的减少模型大小、保持任务质量并改善当前 CPU 性能；
2. 性能时间主要消耗在哪些算子/节点，而不是只知道 `Session::Run` 总耗时。

这是大阶段二唯一的 P0 功能收口阶段。完成后，YOLO P0 feature scope 冻结。

### 7.2 范围

#### A. 状态同步

- 将三份 README 中的大阶段一用户 L2 状态同步为“已完成”；
- 记录 S2-01 正式开始；
- 不重新执行用户 L2。

#### B. 冻结 PTQ 协议

在第一次量化前冻结：

- FP32 baseline artifact 和 SHA；
- calibration 样本 manifest 与每个文件 SHA；
- 任务质量评估数据、类别和标签协议；
- 正确性比较规则；
- benchmark 协议；
- 可接受的质量退化停损门。

门槛必须在看到 INT8 正式结果前写入 SPEC。不得根据结果倒推门槛。

#### C. INT8 PTQ

优先使用适合 CNN 的 static PTQ，并从 ONNX Runtime 推荐的 QDQ/S8S8 路线开始。实施时由 Codex根据模型算子、CPU kernel 支持和调试结果决定是否尝试其他格式或数据类型。

至少产出：

- 一个可加载的 INT8 ONNX；
- 独立 artifact contract/card；
- model size 与 SHA；
- 实际 input/output metadata；
- calibration protocol；
- 量化失败或未量化节点记录。

#### D. 三层验证

1. **Runtime 正确性：**INT8 能由 C++ Runtime 加载、运行、输出合法结果；
2. **检测结果差异：**固定 manifest 上比较 FP32/INT8 count、class、confidence、box 和 matching IoU；
3. **任务精度：**在冻结的带标签数据上比较任务级指标，优先完整 val 或具有明确覆盖的冻结子集；不得只用“输出看起来差不多”代替精度。

#### E. 同协议性能与内存

同一机器、Release、provider、线程、样本、warmup/repeat 下比较：

- model file size；
- session initialization；
- preprocess；
- `Session::Run`；
- postprocess；
- pipeline/end-to-end P50/P95；
- throughput；
- Peak Working Set。

INT8 可能没有加速甚至变慢。真实结果优先，不以“必须更快”为验收条件。

#### F. ORT Profiling

FP32 与 INT8 分别生成独立 profile trace，并输出可读摘要：

- top operators / nodes；
- 累计耗时占比；
- 调用次数；
- provider/placement 证据；
- 与分段 benchmark 的关系；
- profiling 自身开销和不可直接当正式 benchmark 的限制。

Profiling run 与正式 benchmark 必须分开。

### 7.3 非目标

- D010 INT8；
- GPU INT8；
- TensorRT INT8；
- 默认启动 QAT；
- 为追求数字而大规模改网络结构；
- 用单次延迟代替正式协议；
- 把 profile trace 的带开销时间当正式 benchmark。

### 7.4 QAT 停损规则

只有以下条件同时成立时，才允许把 QAT列为后续条件任务：

- PTQ 已按至少一种合理配置完成；
- 任务质量退化超过事先冻结的门；
- 已使用量化调试或节点分析定位主要问题；
- 剩余秋招时间允许；
- 用户明确同意回到训练侧。

否则记录“PTQ 结果与限制”，不启动 QAT。

### 7.5 验收标准

- [ ] 大阶段一状态已同步，旧 L2 `PENDING` 不再存在。
- [ ] calibration manifest 在正式量化前冻结并带 SHA。
- [ ] 至少一个 INT8 artifact 可由 Python ORT 和 C++ ORT 实际加载运行。
- [ ] INT8 contract、actual metadata、模型 SHA 和大小一致。
- [ ] FP32/INT8 固定样本检测差异有机器可读结果。
- [ ] FP32/INT8 任务级精度使用同一协议比较。
- [ ] FP32/INT8 性能、吞吐与内存使用同一协议比较。
- [ ] FP32 和 INT8 各有 profile JSON 与 top-op 摘要。
- [ ] 现有 YOLO correctness gate 与全部测试不回退。
- [ ] README 如实写出“更快、相当或更慢”的实际结论。

### 7.6 L1 理解重点

用户应能回答：

1. static 与 dynamic quantization 的区别；
2. calibration data 为什么必须冻结；
3. QDQ 与 QOperator 的基本区别；
4. 为什么模型变小不等于延迟一定下降；
5. 检测一致性与 mAP/任务精度分别证明什么；
6. 为什么 profiling 与 benchmark 必须分开；
7. operator top time 如何指导优化，而不能直接等于根因；
8. 什么情况下才值得 QAT。

### 7.7 代码练习候选

- CalibrationDataReader / manifest 迭代；
- FP32/INT8 detection matcher；
- profile JSON 聚合 top operators；
- percentile 和同协议 benchmark validator；
- quantized artifact metadata 校验。

### 7.8 本步 Codex 执行指令

```text
阅读 AGENTS.md、docs/PLAN.md、双语 README、cpp_infer/README.md、Proj1_Phase2.md 的 S2-01，以及现有 consistency/benchmark/contract/test 实现。先保护 dirty worktree，并把用户已完成大阶段一 L2 的真实状态同步到三份 README。

本次只完成 YOLO P0 的 INT8 PTQ 与 ORT Profiling。先写 SPEC 并在第一次正式量化前冻结 calibration、任务精度、正确性和 benchmark 协议。优先按 ONNX Runtime 官方建议尝试 CNN static PTQ；生成独立 INT8 artifact contract，验证实际 metadata、hash、模型大小、Python/C++ 可运行性、固定样本检测差异、带标签任务精度、同协议性能和内存。FP32/INT8 分别生成 ORT profile trace 和 top operator 摘要；profile 不得冒充正式 benchmark。

QAT 只有在 PTQ 明显越过预声明质量门、定位原因且用户明确同意时才进入后续条件项。本步不做 D010 INT8、CUDA、TensorRT、Linux 或并发。新行为补齐测试，旧 YOLO gate 不回退，更新 README 和机器可读结果后停止，等待 L1。
```

---

## 8. S2-02：Linux x86_64 跨平台 Runtime

### 8.1 目标与招聘价值

让同一套 C++ Runtime 从“Windows 项目”升级为“Windows/Linux 可复现的跨平台推理软件”。

本阶段使用 WSL2 Ubuntu 24.04 x86_64 作为 Linux 主载体。它证明 Linux 工具链、依赖、动态库、文件系统、信号、RSS 和构建脚本的可移植性，但不声称嵌入式 ARM 或 Jetson。

### 8.2 当前前置缺口

WSL 已存在，但当前缺少：

- GCC/G++；
- CMake/CTest；
- Ninja；
- pkg-config；
- OpenCV C++；
- Linux ORT C++ SDK；
- 可复现 Linux Python consistency 环境。

Docker daemon 当前不可用，因此 Linux 路线不依赖 Docker。

### 8.3 范围

#### A. Linux 工具链与依赖

在用户授权后准备最小必要依赖，并记录：

- Ubuntu/kernel/architecture；
- compiler/CMake/Ninja；
- OpenCV C++；
- ORT C++ 与 Python ORT；
- GTest 来源；
- locale、动态库和搜索路径；
- 安装命令与版本。

依赖版本应尽量与 Windows baseline 对齐；无法完全一致时明确记录差异，不暗中修改 consistency 门。

#### B. 平台解耦

使 CMake 和源码不再假定：

- `onnxruntime.lib` / `onnxruntime.dll`；
- Windows x64 是唯一平台；
- `Psapi` 是唯一内存实现；
- `.cmd/.ps1` 是唯一操作入口；
- 错误信息只描述 Windows SDK。

平台差异集中在清晰边界，不复制 Runtime 主链路。

#### C. Linux 工作流

提供一个 Linux 可用的统一入口或薄脚本，至少覆盖：

```text
doctor
build / clean-build
test
detect / demo
consistency
benchmark
all
```

命令名称可按当前工具结构调整，但语义应与 Windows workflow 对齐。

#### D. Linux 证据

使用 Linux clean Release build 完成：

- runtime/CLI/tests build；
- 聚焦和完整 CTest；
- fixed Demo JSON/PNG；
- Python ORT/C++ ORT consistency；
- benchmark；
- peak RSS；
- environment/result JSON。

#### E. Windows 回归

Linux 改造完成后，重新运行 Windows 关键 gate，证明没有为了 Linux 破坏 Windows。

### 8.4 非目标

- Jetson；
- ARM64；
- QEMU；
- cross-compilation；
- CUDA/TensorRT；
- Docker 镜像；
- systemd 服务；
- 摄像头；
- Linux GUI。

### 8.5 关键设计约束

- Windows 和 Linux 共用核心 C++ 源码；
- 不在源码中硬编码个人绝对路径；
- 构建产物不提交；
- Linux repo 工作目录和 Windows工作目录的关系必须明确，避免两个 Git worktree 同时写同一文件；
- WSL 中的工程文件位置由 Codex根据性能和协作方式选择，但必须可复现；
- OpenCV 版本差异导致 consistency 变化时，先定位 decoder/preprocess/raw output，再决定协议版本，禁止直接放宽容差；
- Linux peak RSS 必须有真实实现，不能填 0。

### 8.6 验收标准

- [ ] WSL2 Linux dependency doctor 可以独立运行。
- [ ] 同一 commit 在 Windows 和 Linux clean Release build 成功。
- [ ] Linux Runtime/CLI/tests 使用真实 Linux ORT/OpenCV，而不是调用 Windows exe。
- [ ] Linux 聚焦测试和完整 gate 通过；平台差异测试有明确标签/理由。
- [ ] Linux fixed Demo 可解析、可回读。
- [ ] Linux Python/C++ consistency 通过冻结或版本化协议。
- [ ] Linux benchmark 有 P50/P95、throughput、peak RSS 和环境元数据。
- [ ] Windows 关键 gate 重新通过。
- [ ] README 明确写“WSL/Linux x86_64”，不写“嵌入式/Jetson”。

### 8.7 L1 理解重点

1. Windows `.lib/.dll` 与 Linux `.so`、RPATH/loader 的差异；
2. CMake 如何按平台发现和链接依赖；
3. WSL 为什么适合 Linux开发验证，又为什么不是端侧实机；
4. build-time、link-time、run-time 错误如何区分；
5. Peak Working Set 与 peak RSS 的口径；
6. 跨平台 consistency 失败的分层排查；
7. 为什么不能只“在 Linux 能编译”就称部署完成。

### 8.8 代码练习候选

- 跨平台 `#ifdef` 边界与平台适配层；
- Linux `/proc` 或 `getrusage` peak RSS 采集；
- CMake imported target / find logic；
- Bash workflow 的退出码和产物检查；
- path/locale/UTF-8 差异。

### 8.9 本步 Codex 执行指令

```text
基于已冻结的 S2-01 P0，本次只完成 Linux x86_64 跨平台 Runtime。使用现有 WSL2 Ubuntu 24.04 作为主载体；Docker 当前不可用，不得作为前置。先写 Linux SPEC 和依赖清单，经用户授权后只安装最小工具链，并记录版本和命令。

重构 CMake、依赖发现、ORT runtime staging、内存采集和工作流，使同一源码在 Windows MSVC/NMake 与 Linux GCC或Clang/Ninja上 clean Release 构建。核心 Runtime 不复制两套。提供 Linux doctor/build/test/demo/consistency/benchmark/all 等价入口，完成真实 Linux CTest、Demo、Python/C++ consistency、benchmark 和 peak RSS；随后重新跑 Windows 关键回归。

本步不做 ARM64、Jetson、QEMU、CUDA、TensorRT、Docker、并发或 D010。任何跨平台数值差异先定位，禁止为全绿无依据放宽容差。更新 README 的平台矩阵和限制后停止，等待 L1。
```

---

## 9. S2-03：C++ 目录批处理、有界队列与并发系统

### 9.1 目标与招聘价值

把单图 CLI 升级为一个可作为产品/系统软件讲解的多图任务：

```text
目录或 manifest
-> deterministic task discovery
-> bounded queue
-> workers
-> session reuse strategy
-> result writing
-> backpressure
-> partial failure accounting
-> clean shutdown
-> throughput comparison
```

重点不是“给单图命令套一个 for 循环”，而是任务生命周期、数据所有权、并发、资源上限、失败传播和关闭语义。

### 9.2 用户流程

最小用户流程应支持：

```text
输入目录或 manifest
+ Runtime config
+ 输出目录
+ worker count
+ queue capacity
+ 输出格式/覆盖策略
-> 执行任务
-> 每图结果
-> batch summary
-> 退出码
```

具体 CLI 形式由 Codex设计，但必须可脚本化、可重复和可测试。

### 9.3 范围

#### A. 任务发现

- 目录或 manifest 输入至少支持一种为正式入口，另一种可选；
- 文件扩展名白名单；
- 确定性排序；
- 空目录、重复项、非普通文件和输出目录递归问题有明确行为；
- 输出路径与输入图片一一映射且避免冲突。

#### B. 有界队列与 backpressure

- queue capacity 可配置且有上限；
- producer 在队列满时阻塞、等待或按 SPEC 定义处理；
- 不允许无限缓存所有图像 tensor/结果；
- close/stop 后不接受新任务；
- wakeup 条件和 ownership 清楚。

#### C. worker 与 session 策略

Codex需要基于 ORT 线程安全、内存、吞吐和实现复杂度，选择并记录：

- 每 worker 一个 Pipeline/Session；
- 多 worker 共享 Session；
- 或其他经过验证的方案。

不能凭感觉声称某方案更快，必须用 correctness、memory 和 benchmark 支撑。

#### D. clean shutdown

至少覆盖：

- 正常完成；
- producer 失败；
- 某张图片失败；
- output writer 失败；
- 用户中断/stop request；
- worker exception；
- 队列关闭时仍有待处理任务。

Windows/Linux 的信号或 console event 差异可在平台层处理。

#### E. 失败模型

需要冻结：

- 单图失败是否继续其他任务；
- batch 最终退出码；
- summary 中成功、失败、跳过和取消数量；
- 失败文件和错误层级；
- 是否保留已成功结果；
- 重跑/overwrite 规则。

#### F. 性能证据

在固定 manifest、相同输出模式和相同模型下比较：

- worker=1；
- 合理的多个 worker 配置；
- queue capacity；
- images/s；
- per-image latency/P50/P95；
- Peak Working Set/peak RSS；
- CPU 利用与线程策略摘要；
- correctness 是否与单图基线一致。

绘图/写盘是否进入性能协议要明确，不允许混淆 compute throughput 与 full-job throughput。

### 9.4 非目标

- ONNX 真正 batch N>1；
- 摄像头视频流；
- 网络服务；
- 分布式任务；
- 无锁队列；
- 工作窃取；
- 动态负载均衡复杂框架；
- GPU stream 并发；
- D010 特殊并发优化。

### 9.5 测试矩阵

至少覆盖：

| 模块 | 关键行为 |
|---|---|
| 队列 | capacity、满队列 backpressure、close、stop、空队列 wakeup |
| 任务发现 | 稳定排序、扩展名、空目录、重复/非法路径 |
| worker | 正常消费、异常传播、无死锁、全部 join |
| 输出 | 路径映射、冲突、部分失败、summary |
| 回归 | batch worker=1 与逐图单图结果一致 |
| 并发 | 多 worker 无数据竞争、无输出覆盖、重复运行稳定 |
| shutdown | 正常、故障和用户取消均能有限时间退出 |
| benchmark | 协议字段和吞吐公式正确 |

测试应优先使用 synthetic tasks 验证队列/关闭语义，真实模型只承担小规模 integration 和正式吞吐证据。

### 9.6 验收标准

- [ ] 一个命令可处理固定目录或 manifest，并生成 per-image outputs + batch summary。
- [ ] worker=1 与现有单图产品链得到等价检测结果。
- [ ] 队列容量真实限制内存中的待处理任务。
- [ ] queue full 时 backpressure 行为有测试和可解释证据。
- [ ] 正常、单图错误、writer 错误、stop/中断均无死锁并 clean shutdown。
- [ ] 多 worker 输出路径无冲突，失败可定位到具体任务。
- [ ] Windows/Linux 都能构建和运行批处理 gate。
- [ ] 单线程/并发吞吐、P50/P95 和内存同协议比较。
- [ ] 没有把“并发多张单图”写成“模型 batch inference”。

### 9.7 L1 理解重点

1. 为什么必须有 bounded queue；
2. backpressure 解决什么问题；
3. condition variable 的 predicate、虚假唤醒和 close 语义；
4. producer/consumer 的数据 ownership；
5. session 共享与 per-worker session 的 trade-off；
6. clean shutdown 为什么比 `std::terminate` 或强制退出复杂；
7. 单图失败继续执行时如何定义最终状态；
8. 为什么 worker 增加后吞吐不一定线性增长；
9. ORT intra-op 线程与应用 worker 数可能如何过度订阅 CPU。

### 9.8 代码练习候选

- `BoundedBlockingQueue<T>` 最小实现；
- close/stop-aware `push()` / `pop()`；
- RAII thread join；
- batch summary state aggregation；
- deterministic output path mapping；
- worker exception capture/rethrow。

### 9.9 本步 Codex 执行指令

```text
基于已完成的 Windows/Linux 单图 Runtime，本次只完成 C++ 产品化多图单元。先写 SPEC，明确目录/manifest 输入、确定性顺序、输出映射、单图失败策略、最终退出码、queue capacity、worker count、backpressure 和 clean shutdown。

实现目录或 manifest -> bounded queue -> worker -> 现有 DetectorPipeline -> per-image outputs -> batch summary 的完整链路。不得复制 preprocess/ORT/postprocess。根据 ORT 线程安全、内存和实测选择 session sharing 或 per-worker session，并记录 ADR/trade-off。建立纯队列/关闭单测、batch integration、故障注入和 worker=1 vs 单图 correctness gate；在 Windows/Linux 使用固定 manifest 比较单线程和并发 throughput、P50/P95 与内存。

本步不做 true ONNX batch、视频流、服务化、GPU stream、无锁队列或 D010 特殊优化。完成 README、机器可读 summary、benchmark 和教学闭环后停止，等待 L1。
```

---

## 10. S2-04：多模型 Runtime 与 D010 Artifact 接入

### 10.1 目标与招聘价值

将当前“契约化的单 YOLO Runtime”升级为“能够通过 artifact contract 接入不同模型家族的 Runtime”。

优先完成这条个人辨识度最高的链路：

```text
paper_detect D010 research checkpoint
-> ONNX + handoff contract
-> yolo_defect model adapter
-> C++ ORT multi-input inference
-> thresholded detection consistency
-> JSON/visualization/batch/benchmark
```

### 10.2 模型选择门

选择顺序：

1. **D010：首选。**已有本地 ONNX/contract/runner/manifest，优先关闭产品门；
2. **D-FINE-S：第一 fallback。**若 D010 被 provenance、导出或 prior contract 长期阻塞，接入其 tested-only基线；
3. **官方 RT-DETR：第二 fallback。**只有 D-FINE-S 也无法形成稳定 artifact 时使用。

本阶段硬出口是“至少一个 DETR 家族第二 artifact 真正进入 C++ Runtime”，不是“只把抽象接口写好”。

### 10.3 D010 前置 Artifact Gate

在修改 Runtime 主架构前，先关闭或明确处理：

#### A. 交接耐久性

- 轻量 handoff、contract、manifest、runner 必须进入可追踪状态，或在 `yolo_defect` 中建立带 SHA 的只读交接记录；
- 大 ONNX/checkpoint 继续 Git 忽略，但路径、SHA 和获取方式明确；
- 不依赖只有当前 Codex 会话知道的临时文件。

#### B. Canonical decoder 与输入 tensor

D010 Runtime 使用 OpenCV/C++，因此应建立 Runtime 自己的 canonical reference：

```text
tested image + template image
-> canonical decode
-> direct resize 640
-> RGB / L [0,1]
-> edge prior generation or frozen prior load
-> tested_rgb / edge_prior / orig_target_sizes
```

不能把冻结环境的某一 Pillow 版本当作 C++ 永久真值。应保留原 research handoff 证据，并建立版本化的 Runtime manifest/golden tensor 证据。

#### C. Prior 协议

优先选择可解释、可测试和可移植的方案：

- 加载冻结 stored prior；或
- C++ 精确复现 Sobel → absolute edge difference → p99.5 → JPEG quality 95 协议。

最终选择由数据可获得性、复现性和 C++ golden test 决定。不得静默用近似 prior。

#### D. 产品语义一致性

D010 产品输出是：

```text
labels / boxes / scores top-300
-> score >= 0.35
-> no NMS
-> unified Detection list
```

低于业务阈值的 top-300 近并列顺序差异，不应自动等同于最终产品 detection 失败；但也不能声称 raw top-300 严格一致。

本阶段应在 SPEC 中明确：

- 产品 contract 是否要求 raw top-300 顺序；
- thresholded detections 如何 order-independent matching；
- raw discrepancy 如何作为已知限制保留；
- 不通过放宽 confidence/box 容差掩盖真正差异。

#### E. Provenance 与许可证

- D010 仅作为个人研究/学习 Runtime artifact；
- checkpoint 与源码链为 evidence-bound，不描述成 cryptographically bound；
- DeepPCB research-only 和模型/源码许可歧义未解决前，不公开分发大模型文件；
- README 将源码、模型和数据许可分别说明。

### 10.4 Runtime 范围

#### A. Artifact contract 扩展

支持表达：

- 多输入；
- 不同 dtype；
- 动态 batch / 固定空间尺寸；
- direct resize 与 letterbox；
- paired image / prior；
- 多输出；
- no-NMS top-k postprocess；
- 类别和阈值语义。

可通过 schema version、variant 或 adapter type 扩展，不能破坏既有 YOLO artifact 的可复现行为。

#### B. Runner 扩展

- 根据 actual metadata 绑定多个 name/dtype/shape；
- 支持 float32 与 int64 输入；
- 返回多个自有生命周期 output；
- 保持 RAII、finite/shape/count 校验；
- 不在 Runner 中硬编码 D010 图片逻辑。

#### C. Model-specific adapters

至少形成：

```text
YOLO adapter:
  single image
  letterbox
  one float input
  raw BCN output
  threshold + NMS + restore

D010/DETR adapter:
  tested/template pair
  direct resize + prior
  three inputs
  labels/boxes/scores outputs
  score >= threshold
  no NMS
```

两者输出同一个稳定 Detection Result contract。

#### D. 产品接入

第二模型应复用：

- JSON schema 的共同部分；
- visualization；
- single-image/pair CLI；
- batch framework；
- benchmark result框架；
- failure reporting。

模型特定字段需要 schema version或明确扩展，不能污染旧 YOLO 证据。

### 10.5 D010 验收协议

初始产品级门：

- 输入 pair 与 prior contract 完整；
- fixed manifest 至少覆盖交接中的 10 组 pair；
- C++ 与 canonical Python reference 使用同一 OpenCV/Runtime preprocess 协议；
- thresholded detection count exact；
- class exact；
- confidence absolute error `<= 1e-4`；
- bbox coordinate absolute error `<= 1e-2 px`；
- order-independent matching；
- input tensor/golden values 有可复现证据；
- raw top-300 reorder 单独记录，不冒充 passed。

如实际差异超出门槛，必须沿以下链路定位：

```text
pair mapping
-> JPEG decode
-> direct resize
-> prior
-> input tensor name/dtype/shape
-> ORT raw outputs
-> score threshold >=
-> output matching/order
-> JSON
```

### 10.6 非目标

- D010 erase/replay 训练逻辑；
- 重新训练 D010；
- D010 INT8；
- 通用支持任意 ONNX 模型；
- 自动推断未知 preprocess/postprocess；
- 将 D010 的论文指标写成 C++ benchmark；
- 为 raw top-300 全排序一致牺牲秋招时间，除非它被证明属于产品可观察语义。

### 10.7 验收标准

- [ ] 至少一个 DETR 家族第二 artifact 真正由 C++ Runtime 加载、推理和输出 Detection。
- [ ] D010 优先；若使用 fallback，阻塞证据和选择理由明确。
- [ ] 第二 artifact 的 contract、actual metadata、hash、license/provenance 和限制完整。
- [ ] 多输入/多输出 Runner 有 synthetic 与真实 integration 测试。
- [ ] 模型特定 preprocess/postprocess 位于 adapter 边界，`main.cpp` 仍薄。
- [ ] YOLO 所有冻结语义和 Windows/Linux gate 不回退。
- [ ] 第二模型 fixed manifest 产品级 consistency 通过预声明门。
- [ ] JSON/visualization/batch 至少有一个完整 Demo。
- [ ] 第二模型有基本 benchmark 和内存记录，但不与不同模型任务精度做无条件快慢结论。
- [ ] raw top-300 已知差异、decoder/provenance/license 限制在 README 中准确表达。

### 10.8 L1 理解重点

1. 为什么 artifact contract 需要表达模型语义而不仅是路径；
2. 多输入 `Ort::Value` 的 name/dtype/shape/lifetime；
3. YOLO 与 DETR/D-FINE 输出语义差异；
4. D010 为什么推理时不实现 erase/replay；
5. direct resize 与 letterbox 为什么不能复用；
6. 为什么 D010 无 NMS；
7. raw top-k 排序差异与产品 detection 一致性的边界；
8. adapter/strategy/registry 的 trade-off；
9. 为什么研究 artifact handoff 是跨仓库 API contract。

### 10.9 代码练习候选

- heterogeneous tensor binding；
- multi-output ownership；
- paired-image manifest parser；
- Sobel/prior 生成最小实现；
- direct resize tensor layout；
- DETR output threshold/filter；
- model adapter dispatch。

### 10.10 本步 Codex 执行指令

```text
本次只完成第二模型 artifact 与多模型 Runtime。先读取 paper_detect 的 d010_runtime_handoff、contract、manifest、reference runner 和大文件 hash，建立 D010 artifact gate；不得根据 README 论文数字推断 Runtime 已准备好。

先关闭交接耐久性、canonical OpenCV decoder、tested/template pair、prior golden test、产品级 thresholded consistency 和 provenance/license 限制。明确 raw top-300 低分近并列重排是否属于产品 contract；必须保留 raw failed evidence，不能为通过而无依据放宽数值门。

若 D010 在预设 timebox 内可关闭门，则扩展 artifact schema、multi-input/multi-output OnnxRunner 和 model-specific preprocess/postprocess adapter，接入 D010；若被真实阻塞，按 D-FINE-S -> 官方 RT-DETR 顺序选择稳定 fallback，并记录理由。最终至少一个 DETR 家族模型必须真实进入 C++ Runtime，输出统一 Detection Result，并复用 JSON/可视化/batch/benchmark。YOLO Windows/Linux/INT8 gate 不得回退。

本步不训练模型、不实现 D010 erase/replay、不做 D010 INT8、TensorRT 或 Qt。完成 fixed manifest consistency、Demo、测试、benchmark、README 和教学闭环后停止，等待 L1。
```

---

## 11. S2-05：条件阶段——真实端侧与 TensorRT 部署梯子

### 11.1 阶段定位

本阶段受真实依赖和设备门禁控制，不属于大阶段二硬关闭条件。

它的目标不是为了“计划完整”强行安装复杂工具链，而是在前四个单元稳定后，检查是否具备一条成本可控、可以形成真实证据的端侧路径。

### 11.2 执行门

进入本阶段前至少满足一种：

- Windows/Linux TensorRT 10、CUDA、cuDNN、ORT GPU C++ SDK 已形成兼容环境；
- 可以借用、租用或远程访问真实 Jetson；
- 可以访问明确型号的其他 ARM64/NPU 实机和公开 SDK；
- 用户明确同意为该设备投入时间。

若都不满足，S2-05 状态记为：

```text
NOT_EXECUTED_BY_HARDWARE_GATE
```

并直接进入 S2-06。该状态不是项目失败。

### 11.3 部署梯子

#### Level A：桌面 GPU backend

在 RTX 4060 Laptop 上，优先完成：

```text
same artifact
-> ORT CUDA or ORT TensorRT EP
-> correctness gate
-> FP32/FP16 performance
-> memory/profile
```

它证明 accelerator backend，但不称端侧实机。

#### Level B：真实 Linux ARM64 CPU

若获得普通 ARM64 board：

- native build 或 toolchain/sysroot cross-build；
- CPU Runtime Demo；
- correctness、latency、RSS；
- 部署目录和依赖；
- 不声称 TensorRT/Jetson。

#### Level C：真实 Jetson

若获得 Jetson：

- 记录准确型号、JetPack/Jetson Linux/CUDA/cuDNN/TensorRT；
- 优先复用 ORT TensorRT EP，并注册 CUDA fallback；
- 只有 EP 路线不能满足需求或高优先 JD 明确要求时，才考虑原生 TensorRT；
- 比较 CPU/CUDA/TensorRT 或适用后端；
- 记录 correctness、P50/P95、throughput、memory、power mode、temperature/tegrastats；
- engine/cache 与具体设备、TensorRT 版本绑定，不能跨设备冒用。

### 11.4 实机验收要求

只有真实设备运行成功，才允许形成设备 bullet。至少记录：

```text
device model
OS / architecture / JetPack or SDK
compiler / build type
backend / precision
model / input / sample protocol
correctness tolerance
latency P50/P95
throughput
memory
power mode
温度或功耗（若平台可观测）
limitations
```

### 11.5 非目标

- 购买板卡作为项目完成前置；
- 模拟器性能；
- QEMU 代替真实设备；
- 刷机/BSP/驱动/内核开发；
- 原生 TensorRT 与 ORT TensorRT 同时做两套；
- 设备上训练模型；
- 摄像头/DeepStream/ROS2。

### 11.6 停损规则

- 环境安装和兼容性排错先 timebox；
- 若主要问题是 CUDA/cuDNN/TensorRT 版本矩阵，保留完整阻塞记录，不无限追包；
- 若远程设备不允许上传二进制、读取温度/功耗或稳定复现，只写实际获得的证据；
- 不因 S2-05 未执行而推迟 S2-06 和投递。

### 11.7 L1 理解重点

1. Linux x86_64、ARM64、Jetson 的关系；
2. CUDA EP、TensorRT EP、原生 TensorRT 的差异；
3. 为什么 TensorRT EP 推荐 CUDA fallback；
4. engine/cache 为什么与设备和版本绑定；
5. host-device copy 与 I/O Binding；
6. FP16/INT8 correctness 与性能如何验证；
7. 为什么模拟不能产生温度、功耗和真实延迟证据。

### 11.8 本步 Codex 执行指令

```text
先重新调查 CUDA/cuDNN/TensorRT C++ SDK 与真实 ARM64/Jetson 访问状态。本阶段受硬门控制；如果没有兼容 accelerator 环境也没有真实设备，明确记录 NOT_EXECUTED_BY_HARDWARE_GATE，禁止为了计划全绿伪造模拟部署，直接停止本步。

若桌面 TensorRT/CUDA READY，优先在不改变检测语义的前提下接入一个 accelerator provider，先过同 artifact correctness gate，再测 FP32/FP16 latency、throughput、memory 和 profile；不得称 Jetson。若获得真实 ARM64/Jetson，记录准确设备与软件栈，优先采用 ORT TensorRT EP + CUDA fallback，只有必要时做原生 TensorRT，并测 correctness、P50/P95、throughput、memory、power mode、temperature/tegrastats。

不购买设备、不做 BSP/驱动/刷机/DeepStream/ROS2。任何失败保留版本矩阵和可行动诊断。完成真实证据或门禁记录后停止。
```

---

## 12. S2-06：本地量化 LLM 质检报告与大阶段收口

### 12.1 目标与招聘价值

在不学习完整后端技术栈、不复活项目 2 的前提下，为工业视觉 Runtime 增加一个窄范围生成式 AI 能力：

```text
Detection JSON
-> validated prompt builder
-> local quantized small LLM
-> structured inspection report JSON
-> schema/eval/fallback
```

它证明“本地 C/C++ 语言模型推理 + 量化 + 性能评测 + 与 CV 业务结果集成”，而不是做通用聊天机器人。

### 12.2 产品边界

CV Runtime 仍然是确定性主链：

- 模型检测结果、类别、坐标、置信度由 CV Runtime 产生；
- LLM 不修改 detection；
- LLM 只生成解释性报告；
- 报告必须保留来源 detection 和模型限制；
- LLM 失败时不影响原始 JSON/PNG 交付。

### 12.3 模型与 Runtime 选择

优先使用支持本地 C/C++ 推理和整数权重量化的成熟 Runtime，例如 `llama.cpp`。

模型选择由 Codex根据以下门决定：

- 同一模型家族同时有 4-bit 与 8-bit；
- 适配 8GB VRAM / 16GB RAM；
- 下载和磁盘可控；
- 中文结构化输出可用；
- 许可证可用于个人学习展示；
- 不需要复杂 tokenizer/模型转换手工修复。

建议从小型 instruct 模型开始，不以参数量作为亮点。

### 12.4 功能范围

#### A. 输入

读取并严格验证现有 Detection JSON。至少使用：

- model/artifact；
- image；
- detections；
- class/confidence/bbox；
- Runtime limitations。

非法 JSON、schema mismatch、空 detection 和超长输入有明确行为。

#### B. 固定 Prompt Builder

Prompt 只要求模型输出受控字段，例如：

```text
summary
risk_level
primary_defects
locations
review_recommendations
limitations
```

Prompt 中明确：

- 只能使用输入 JSON 中出现的事实；
- 不得创造不存在的类别、坐标或数量；
- 空检测时不得声称发现缺陷；
- 输出固定 JSON schema；
- 不能给出超出工业质检辅助范围的结论。

#### C. 结构化输出与 fallback

- 标准 JSON parse/schema validation；
- 非法 JSON 可做有限重试或修复，但策略必须有上限；
- 仍失败时使用 deterministic template fallback；
- 保存 raw model output、validation status 和最终 report；
- 不把 fallback 结果冒充 LLM 成功。

#### D. 量化与性能比较

同一模型的 Q4 与 Q8 至少比较：

- model size；
- load time；
- peak RSS；
- GPU offload/VRAM（若实际使用）；
- TTFT；
- TPOT；
- tokens/s；
- schema success rate；
- grounded factual error；
- fixed prompt output差异。

#### E. Eval

建立 10～20 个固定输入，覆盖：

- 空检测；
- 单类别单框；
- 多框同类；
- 多类别；
- 低置信度；
- bbox 位于不同区域；
- 非法/缺字段；
- D010 与 YOLO 不同类别表。

至少检查：

- JSON schema；
- detection 数量不被篡改；
- 不出现输入之外的 defect class；
- risk/recommendation 与规则边界；
- Q4/Q8 的结构化成功率；
- fallback 可用。

### 12.5 与 Agent 的边界

本阶段硬目标不是 Agent。

最多允许一个条件 stretch：

```text
validate_detection
-> generate_report
-> validate_report
-> fallback_or_finish
```

只有真正存在明确状态、allowlisted tools、失败转移和 eval 时，才能称“最小受控 workflow”。没有 tool calling/state machine 时，只称“本地 LLM report pipeline”。

### 12.6 非目标

- RAG；
- 向量数据库；
- LangChain/LangGraph；
- 多 Agent；
- 外部云 API；
- 微调语言模型；
- 通用聊天 UI；
- FastAPI 后端；
- Qt；
- 把 LLM 报告作为检测真值。

### 12.7 大阶段二总收口

S2-06 同时完成：

- Windows/Linux/INT8/batch/multi-model/LLM 总体架构图；
- 结果矩阵；
- 条件设备状态；
- 失败案例与停损记录；
- Quick Start；
- README 双语对齐；
- 1～3 条第二阶段简历 bullet；
- 30 秒、2 分钟、5 分钟口述；
- 至少 15 个问题与连续追问；
- 一次专项 mock；
- code-practice 核心清单；
- P0 feature freeze 与后续只允许的维护范围。

### 12.8 验收标准

#### 本地 LLM

- [ ] 一个固定命令可把 Detection JSON 转换为 report JSON。
- [ ] Q4/Q8 使用同一模型和固定 prompt/eval 协议。
- [ ] model size、load、RSS、TTFT、TPOT、tokens/s 有真实记录。
- [ ] 输出通过 JSON/schema 验证；失败有有限重试与 deterministic fallback。
- [ ] Eval 能检测虚构类别、数量变化和空检测误报。
- [ ] CV Runtime 即使 LLM 不可用仍能正常交付 JSON/PNG。
- [ ] README 准确称为 local LLM report pipeline，不夸大 Agent。

#### 大阶段二

- [ ] S2-01～S2-04 和 S2-06 硬目标完成。
- [ ] S2-05 有真实结果或明确 `NOT_EXECUTED_BY_HARDWARE_GATE`。
- [ ] Windows/Linux current results、历史 Python 结果和研究 D010 指标分栏。
- [ ] 所有简历数字都可追到机器可读证据。
- [ ] 用户完成大阶段二 L2。

### 12.9 L1/L2 理解重点

本阶段 L1：

1. 权重量化 Q4/Q8 与 CV INT8 PTQ 的区别；
2. TTFT、TPOT、tokens/s 的定义；
3. 为什么结构化输出仍可能失败；
4. schema validation、bounded retry 与 fallback；
5. grounded report 如何防止凭空创造检测事实；
6. 为什么这不是 Agent。

大阶段二 L2：

1. 用 5 分钟讲清 FP32/INT8、Profiling、Linux、并发、多模型和本地 LLM 总链路；
2. 回答至少 15 个追问；
3. 解释至少 4 个失败案例：量化退化、Linux依赖、并发死锁/过度订阅、D010 prior/一致性、LLM schema 失败中至少四类；
4. 在 AI 指导下完成一次跨模块修改并补测试；
5. 独立跑通一个 Windows gate、一个 Linux gate、一个 batch job 和一个 report job；
6. 写出 1～3 条事实准确的阶段 bullet；
7. 指出本阶段最值得手写或看懂的核心代码。

### 12.10 代码练习候选

- Detection JSON schema validator；
- prompt builder；
- structured JSON extraction；
- bounded retry/fallback state machine；
- TTFT/TPOT/tokens/s 统计；
- factual grounding rules；
- subprocess/library runner ownership。

### 12.11 本步 Codex 执行指令

```text
本次只完成本地小语言模型质检报告 PoC 与大阶段二收口。先根据 RTX 4060 8GB、16GB RAM、磁盘余量和许可证选择一个小型 instruct 模型，并固定同一模型的 Q4/Q8 版本；优先使用成熟的本地 C/C++ Runtime。不要同时下载多个大模型。

实现 Detection JSON -> strict validator -> fixed prompt builder -> local LLM -> structured report JSON -> schema/eval/fallback。LLM 只能使用输入 detection 事实，不能修改检测结果；空检测、非法 schema、超长输入和非法模型输出有明确行为。对 Q4/Q8 同协议记录模型大小、load time、peak RSS、实际 VRAM/offload、TTFT、TPOT、tokens/s、schema success rate 和 factual errors。最多把 validate/generate/validate/fallback 做成最小受控 workflow；没有 tool calling/state machine 时不得称 Agent。

随后执行大阶段二总 gate，整理 Windows/Linux/INT8/batch/多模型/LLM 结果矩阵、失败案例、README、Quick Start、简历 bullet、30秒/2分钟/5分钟口述、追问和专项 mock。S2-05 无设备时明确 NOT_EXECUTED_BY_HARDWARE_GATE。完成后停止，不继续 Qt、RAG、服务化或更多模型。
```

---

## 13. 大阶段二停损与优先级规则

### 13.1 投递/面试优先

出现以下情况时暂停项目：

- 次日有面试或笔试；
- 收到高匹配 JD，需要定向补充；
- 当前阶段已形成可投递 bullet，继续开发收益低于面试准备；
- 连续环境排错超过 timebox 且没有新证据。

暂停前必须：

- 保持当前 build/test checkpoint 可复现；
- 记录 TODO、失败命令和下一步；
- 不留下半套产品语义或被临时修改的容差。

### 13.2 最低、目标与超量完成

#### 最低完成

```text
S2-01 INT8 + Profiling
+ S2-02 Linux
+ S2-03 batch/concurrency
+ S2-06 final close
```

这已经能够形成：C++/Linux/量化/性能/测试/系统软件方向的强投递版本。

#### 目标完成

```text
最低完成
+ S2-04 D010或第二 DETR artifact
+ S2-06 local LLM reporter
```

这是本方案的默认大阶段二硬目标。

#### 超量完成

```text
目标完成
+ S2-05 real accelerator/device evidence
```

只有真实设备或兼容 TensorRT 环境具备时进入。

### 13.3 具体停损

- **INT8：**至少完成一个合理 static PTQ 和证据；无收益也可以收口，QAT 不默认启动。
- **Linux：**Docker 不可用不阻塞；优先 WSL native toolchain。
- **并发：**先保证 bounded/clean/correct，再比较性能；不追求复杂无锁设计。
- **D010：**产品门在 timebox 内关不掉就启用 D-FINE-S/RT-DETR fallback，不让论文 artifact 阻塞系统主线。
- **TensorRT/Jetson：**依赖或硬件未 READY 就跳过，不花钱强行补一条 bullet。
- **LLM：**只做一个小模型家族 Q4/Q8；资源不足则缩小模型，不扩硬件。
- **Qt/Agent：**不属于本阶段硬目标；只有真实 JD 和剩余时间同时支持时另开阶段。

---

## 14. 结果与证据总矩阵

大阶段二最终至少形成以下分栏：

| 证据线 | 模型/平台 | 正确性 | 性能 | 内存 | 任务质量 | 限制 |
|---|---|---|---|---|---|---|
| YOLO FP32 | Windows CPU | 30图 consistency | 单图/多图 P50/P95 | Peak Working Set | baseline metric | current P0 baseline |
| YOLO INT8 | Windows CPU | FP32/INT8 matching | 同协议 benchmark | Peak Working Set | INT8 metric/delta | quantization coverage |
| YOLO FP32/INT8 profile | Windows CPU | profile run result稳定 | operator top time | profile metadata | 不适用 | profile overhead |
| YOLO Linux | WSL Ubuntu CPU | Linux Python/C++ | Linux P50/P95 | peak RSS | 同 artifact | WSL不是端侧 |
| Batch/Concurrent | Windows/Linux | worker=1/多worker一致 | images/s / P50/P95 | peak memory | 不改变模型精度 | true batch未做 |
| D010/DETR | Windows/Linux CPU | fixed paired manifest | model benchmark | memory | 论文指标单独列 | prior/provenance/license |
| Accelerator/Device | 条件 | same-artifact gate | P50/P95/throughput | device memory | 不重新评估训练 | 无实机则无结果 |
| Local LLM Q4/Q8 | Windows或Linux | schema/grounding eval | TTFT/TPOT/tokens/s | RSS/VRAM | structured success rate | 不是真值/Agent |

每一行都必须能定位到：

```text
command
config/contract
artifact SHA
sample manifest
raw JSON/CSV/profile
README summary
known limitations
```

---

## 15. 大阶段二最终验收清单

### 15.1 工程自动门

- [ ] 大阶段一 L2 状态已同步到 README。
- [ ] YOLO INT8 PTQ artifact、calibration、任务质量和性能证据完成。
- [ ] FP32/INT8 ORT operator profile 和摘要完成。
- [ ] 同一源码在 Windows和 WSL Linux clean Release 构建。
- [ ] Windows/Linux tests、Demo、consistency 和 benchmark 可运行。
- [ ] Linux peak RSS 有真实数值和口径。
- [ ] 目录/manifest batch、有界队列、backpressure、worker 和 clean shutdown 可运行。
- [ ] 单线程/并发 correctness、throughput 和 memory 有同协议比较。
- [ ] 至少一个 DETR 家族第二模型真实接入；D010优先，fallback 有阻塞证据。
- [ ] 第二模型 preprocess/input/output/postprocess contract 与固定 consistency 完成。
- [ ] 本地 LLM Q4/Q8 report pipeline、结构化 Eval 和性能报告完成。
- [ ] S2-05 有真实结果或明确硬件门状态。
- [ ] 全部历史、当前、研究和条件结果没有混写。
- [ ] 所有新功能有相应测试和故障注入。
- [ ] README.md、README_zh.md、cpp_infer/README.md 事实一致。

### 15.2 用户 L2 门

- [ ] 30 秒说明二阶段解决了什么。
- [ ] 2 分钟说明完整架构和数据流。
- [ ] 5 分钟说明 INT8、Linux、并发、多模型、LLM 和设备边界。
- [ ] 回答至少 15 个追问。
- [ ] 解释至少 4 个真实失败/排错案例。
- [ ] 指出关键模块、类、入口、输入输出和证据路径。
- [ ] 在 AI 指导下跑 Windows、Linux、batch、D010/DETR 和 LLM 中至少四条链路。
- [ ] 完成一次“跨模块行为 + 对应测试”的可回滚修改。
- [ ] 形成 1～3 条阶段 bullet 和对应三层追问。
- [ ] 将 bounded queue、量化/profiling、多输入 runner、structured output validator 等加入代码练习候选。

### 15.3 大阶段关闭后的冻结规则

P0 和大阶段二硬功能完成后，只允许：

- demo/correctness/reproduction bug fix；
- 测试、证据、README 和失败案例；
- 基于真实 JD 的小补丁；
- 基于面试反馈的讲解和题库调整；
- 条件设备实测补录。

不再无目标增加模型、框架、UI、服务或 Agent 功能。

---

## 16. 简历 bullet 方向模板

最终数字必须由真实证据填入，以下仅规定叙事结构。

### 16.1 P0 / 量化与性能

```text
基于 ONNX Runtime 对工业缺陷 YOLO artifact 完成 static INT8 PTQ，冻结 calibration 与任务评测协议，对比 FP32/INT8 的模型大小、检测质量、P50/P95、吞吐和进程峰值内存；结合 ORT profiling 定位主要算子耗时，并以 C++/Python correctness gate 约束性能发布。
```

### 16.2 Linux / C++ 系统软件

```text
将 Windows C++17 Runtime 改造成 Windows/Linux 同源码构建，完成 WSL Ubuntu 下的 CMake/ORT/OpenCV/CTest、Python/C++ 一致性、benchmark 与 peak RSS；进一步实现目录批处理、有界队列、worker、backpressure 和 clean shutdown，并比较单线程与并发吞吐及内存开销。
```

### 16.3 多模型 / 论文 artifact

```text
设计模型 artifact 与 preprocess/postprocess adapter 边界，将论文 D010/D-FINE 系多输入 ONNX 接入 C++ Runtime，完成 tested/template/prior 契约、多 tensor ORT 绑定、无 NMS top-k 后处理和固定样本跨实现一致性，同时保持 YOLO 回归与统一 Detection JSON。
```

### 16.4 本地生成式 AI

```text
基于本地 C/C++ LLM Runtime 将检测 JSON 转换为结构化质检报告，对同一小模型 Q4/Q8 版本比较模型大小、RSS、TTFT、TPOT 和 tokens/s，并通过 schema、事实约束、固定 Eval 与 fallback 防止输出虚构缺陷信息。
```

### 16.5 真实设备条件 bullet

只有 S2-05 实机完成后才允许：

```text
在真实 <device> / <JetPack or SDK> 上部署 <backend/precision>，通过同 artifact correctness gate，并记录 P50/P95、throughput、memory、power mode 与温度/功耗；与 x86 CPU baseline 按不同平台协议分别披露。
```

---

## 17. 外部技术依据（截至 2026-08-23）

本方案参考以下官方资料，但实际实现仍以当前仓库、固定版本和实测为准：

1. ONNX Runtime Quantization：static/dynamic PTQ、calibration、QDQ/QOperator、CNN 优先 static PTQ、QAT 条件。  
   <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
2. ONNX Runtime Profiling：生成包含逐算子 latency/threading 的标准 JSON trace。  
   <https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html>
3. ONNX Runtime C++：官方 C++ 构建覆盖 Windows/Linux 等平台。  
   <https://onnxruntime.ai/docs/get-started/with-cpp.html>
4. Microsoft WSL Development Environment：WSL 可作为完整 Linux 开发环境，Visual Studio/VS Code 支持 CMake 跨平台开发。  
   <https://learn.microsoft.com/en-us/windows/wsl/setup/environment>
5. ONNX Runtime TensorRT EP：C/C++ 显式注册 TensorRT，并建议同时注册 CUDA 作为不支持节点的 fallback。  
   <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html>
6. NVIDIA JetPack：Jetson Linux、Ubuntu、CUDA、cuDNN、TensorRT 构成真实 Jetson edge AI 软件栈。  
   <https://docs.nvidia.com/jetson/jetpack/introduction/index.html>
7. llama.cpp：本地 C/C++ LLM/VLM 推理，支持多种整数权重量化和 CPU/GPU backend。  
   <https://github.com/ggml-org/llama.cpp>

---

## 18. 一句话总览

> 大阶段二先用 YOLO 完成 INT8 与逐算子性能证据，再把 Runtime 迁移到 Linux、升级为有界并发批处理系统，并通过 D010 优先的第二 artifact 证明多模型接入；真实 Jetson/TensorRT 仅在设备和环境具备时实测，最后用一个严格受控的本地量化 LLM 报告链补充生成式 AI 能力，并以完整证据、README、简历和面试 L2 收口。
