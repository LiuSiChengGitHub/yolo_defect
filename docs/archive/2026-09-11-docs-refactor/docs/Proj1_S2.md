# 大阶段二执行方案

> - 重写日期：2026-08-24
> - 项目仓库：`<repo-root>`
> - 当前开发分支：`deploy-cpp`
> - 当前基线 HEAD：`46ca8448614981704bc223fa1c62d5d8a3a4fc1b`
> - 前置状态：大阶段一自动工程门与用户 L2 均已完成；三份 README 仍需同步旧的 `L2 PENDING / Stage One incomplete` 状态。
> - 本版定位：以秋招速度和可追问深度为最高优先级，冻结 D010、Qt、LLM、Agent 与真实板卡扩展；先完成 Linux 应用软件与 AI inference/Edge AI 岗位都能使用的共同底座。
> - 更新时间：0824 2322

---

## 1. 本版为什么重写

旧方案把 INT8、Linux、并发、多模型、真实端侧、本地 LLM 同时纳入大阶段二，功能面完整，但在秋招已经开始的时间点存在三个问题：

1. 技术栈过多，容易让项目从“深入理解一个 C++ 软件系统”退化成“快速收集关键词”；
2. D010、Jetson、LLM、Qt 会分别打开新的环境、模型、硬件或面试追问面，阻塞真正高频的 C++、Linux、测试、并发和性能准备；
3. 项目完成度可能继续提高，但用户对已有模块的掌握、简历投递和面试准备反而被挤压。

本版将大阶段二压缩为五个完整单元：

```text
S2-01  INT8 PTQ + ORT Profiling
    ↓
S2-02  Linux x86_64 + AArch64 交叉编译/QEMU
    ↓
S2-03  目录/manifest 批处理 + 有界并发系统
    ↓
S2-04  Linux x86_64 本地 TensorRT 加速
    ↓
S2-05  全量回归、README、V2 简历与面试 L2 收口
```

这五个单元共同回答：

```text
模型能否更省？
→ INT8

性能瓶颈究竟在哪？
→ ORT Profiling

同一源码能否脱离 Windows？
→ Linux x86_64

能否脱离 x86 架构假设？
→ AArch64 cross compile + QEMU

能否处理真实多任务 workload？
→ directory/manifest + bounded queue + workers

任务过载时是否可控？
→ backpressure

异常或退出时是否会死锁？
→ clean shutdown

能否使用实际 accelerator runtime？
→ TensorRT on Linux + RTX 4060

所有能力能否变成可投递、可追问的证据？
→ S2-05 Recruiting Freeze
```

---

## 2. 使用方式与加速规则

### 2.1 执行方式

1. 一次只推进一个 `S2-*` 完整单元；
2. 每个单元先冻结最小 SPEC，再完成代码、测试、机器可读证据和 README；
3. 单元完成后立即停止，由用户完成 L1 理解与实操；
4. 不再拆成大量微阶段；一个单元必须交付一个用户可运行、可测试的完整能力；
5. 具体类名、文件拆分、API 形式和依赖安装方式由 Codex 根据当时源码与环境决定，本方案只冻结职责、行为、证据和边界；
6. 当前 V1 简历立即投递，不等待大阶段二；
7. `S2-02`、`S2-03`、`S2-04` 均允许形成滚动 V2，不等待最终收口。


---

## 3. 当前基线与二阶段缺口

### 3.1 大阶段一已完成

当前 YOLOv8/NEU-DET C++ Runtime 已具备：

```text
RuntimeConfig + ModelArtifactSpec
-> 实际 ONNX ModelMetadata 校验
-> OpenCV letterbox / RGB / normalize / NCHW
-> ONNX Runtime C++ CPU Session::Run
-> 自有生命周期 raw output
-> YOLOv8 BCN decode
-> strict score filter
-> stable class-agnostic NMS
-> letterbox 坐标反算与 clip
-> Detection JSON / 无 GUI 可视化
-> 六类 30 图 Python ORT / C++ ORT 一致性
-> Release 分段 benchmark / throughput / Peak Working Set
-> 106 项 CTest/GTest/CLI/Python/negative/integration gate
-> 统一 Windows stage1.cmd 工作流
```

大阶段一已经证明：

- 单图 C++ 推理纵切真实可运行；
- 核心纯逻辑可独立测试；
- Python ORT/C++ ORT 在冻结协议下结果一致；
- 正确性通过后可以发布分段性能与内存证据；
- 主要错误能以非零退出和可行动诊断暴露。

大阶段二不重复追求测试数量，也不重写主链。

### 3.2 当前 P0 剩余硬缺口

```text
YOLO FP32 ONNX
-> static INT8 PTQ
-> 正确性/任务精度/性能/模型大小比较

Session::Run 总耗时
-> ORT operator/node profiling
-> top operators / nodes / provider / cumulative time
```

### 3.3 当前环境事实

| 能力 | 当前状态 | 本版处理 |
|---|---|---|
| Windows C++ CPU | READY | 所有单元保留回归 |
| WSL2 Ubuntu 24.04 x86_64 | PARTIAL | S2-02 准备最小 Linux 工具链 |
| Docker | BLOCKED | 不作为任何硬前置 |
| RTX 4060 Laptop | READY at hardware level | S2-04 使用 |
| CUDA / cuDNN / ORT GPU | 版本不匹配、PARTIAL/BLOCKED | S2-04 重新建立兼容栈，不沿用错误组合 |
| TensorRT | BLOCKED | S2-04 独立 timebox 解决 |
| 真实 ARM64/Jetson | UNKNOWN/BLOCKED | 不属于大阶段二核心 |
| D010 handoff | PARTIAL | 移出核心，作为后续条件扩展 |
| 本地 LLM / Agent / Qt | 未准备 | 冻结 |

### 3.4 大阶段二核心硬目标

1. 完成 YOLO static INT8 PTQ、FP32/INT8 对比和 ORT Profiling；
2. 同一源码在 Windows 与 Linux x86_64 下构建、测试、Demo、一致性和 benchmark；
3. 在 Linux host 完成 AArch64 交叉编译，并通过 QEMU 做 ARM64 可移植性 smoke；
4. 实现目录或 manifest 多图任务、有界队列、worker、backpressure、clean shutdown 和并发性能比较；
5. 在 Linux x86_64 + RTX 4060 上形成一个真实 TensorRT 执行路径；
6. 以全量回归、README、结果表、故障案例、V2 简历和面试 L2 收口。


---

## 4. 目标架构

### 4.1 产品链路

```text
                         RuntimeConfig
                              +
                     ModelArtifactSpec
                              |
                              v
                    actual ModelMetadata
                              |
                              v
                 Model-specific Preprocess
                  letterbox/RGB/NCHW
                              |
                              v
               +--------------+--------------+
               |                             |
               v                             v
          ORT CPU Backend             TensorRT Path
       FP32 / INT8 ONNX          Linux x86_64 + RTX
               |                             |
               +--------------+--------------+
                              |
                              v
                    Owned InferenceOutput
                              |
                              v
                  YOLO decode/filter/NMS
                              |
                              v
                    DetectionResult/JSON
```

### 4.2 多图系统链路

```text
Directory / Manifest
        |
        v
Deterministic WorkItem discovery
        |
        v
Producer
        |
        v
BoundedBlockingQueue
        |
        +-----------------------------+
        |             |               |
        v             v               v
     Worker 1      Worker 2        Worker N
        |             |               |
        +-------------+---------------+
                      |
                      v
            existing DetectorPipeline
                      |
                      v
             per-image outputs
                      |
                      v
               BatchSummary
```

### 4.3 平台矩阵

```text
Windows x86_64
  -> ORT CPU FP32/INT8
  -> full correctness / benchmark

Linux x86_64
  -> ORT CPU FP32/INT8
  -> full correctness / benchmark / peak RSS

Linux AArch64 under QEMU
  -> cross-built CLI/tests
  -> portability smoke
  -> no performance claim

Linux x86_64 + RTX 4060
  -> TensorRT execution path
  -> correctness / FP16 benchmark / GPU memory
  -> no Jetson claim
```

### 4.4 架构原则

- `yolo_defect_runtime` 继续承载核心能力，CLI 保持薄；
- Windows/Linux/AArch64 共用业务源码；
- 平台差异集中在依赖发现、动态库、内存采集、信号和脚本层；
- 多图系统复用现有单图 Pipeline，不复制 preprocess/infer/postprocess；
- accelerator backend 不改变冻结的检测语义；
- 正确性门始终先于 benchmark；
- 后端抽象以最小必要边界为准，不为了设计模式本身制造复杂继承树；
- 结果、环境、命令、artifact SHA、样本和限制均进入机器可读证据。


---

## 8. S2-01：INT8 PTQ 与 ORT Profiling

### 8.1 目标

把当前 FP32 CPU Runtime 升级为：

```text
可量化
+ 可比较
+ 可解释瓶颈
```

解决两个问题：

1. INT8 是否降低模型大小、维持任务质量并改善当前 CPU 推理；
2. `Session::Run` 内部时间主要消耗在哪些 operator/node。

这是大阶段二唯一的 P0 功能收口单元。

### 8.2 范围

#### A. 状态同步

- 将三份 README 中大阶段一用户 L2 状态同步为已完成；
- 记录 S2-01 正式开始；
- 不重新执行大阶段一 L2。

#### B. 冻结 PTQ 协议

第一次正式量化前冻结：

```text
FP32 source ONNX + SHA
calibration manifest + image SHA
calibration preprocess
任务质量评估集
FP32/INT8 detection matching
benchmark protocol
质量退化停损门
```

当前 matching `.pt` 不在本机不阻塞 PTQ。PTQ 直接以 FP32 ONNX 为输入。

#### C. Static PTQ

优先采用 CNN 常用的 static PTQ，从 ONNX Runtime 的 QDQ/S8S8 路线开始。

至少产出：

- 可加载的 INT8 ONNX；
- 独立 artifact contract/card；
- source SHA、derived SHA、模型大小；
- calibration manifest 和量化配置；
- 实际 input/output metadata；
- 量化节点、未量化节点或失败节点记录。

#### D. 三层正确性

1. **Runtime 合法性：**Python ORT/C++ ORT 均可创建 session、运行、输出有限值；
2. **产品检测差异：**固定 manifest 比较 count、class、confidence、bbox、matching IoU；
3. **任务质量：**在冻结带标签数据上比较任务级指标，不用“看起来差不多”代替。

#### E. 同协议性能

在同一机器、Release、provider、线程、样本、warmup/repeat 下比较：

- model file size；
- session initialization；
- preprocess；
- `Session::Run`；
- postprocess；
- pipeline/end-to-end mean/P50/P95；
- throughput；
- Peak Working Set。

INT8 变慢也是合法结果，不以“必须加速”为通过条件。

#### F. ORT Profiling

FP32 与 INT8 分别生成 profile trace，并输出摘要：

- top operators / nodes；
- 按 `op_type` 聚合耗时；
- 累计耗时占比；
- 调用次数；
- provider/placement；
- 与 segmented benchmark 的对应关系；
- profiling overhead 与限制。

Profiler run 与正式 benchmark 分离。

### 8.3 非目标

- QAT；
- D010 INT8；
- GPU/TensorRT INT8；
- 模型结构重设计；
- 为让数字更好而改变冻结阈值/NMS；
- 将 profile 中带开销的时间写成正式性能。

QAT 只有在 PTQ 明显越过预声明质量门、已定位主要量化问题、matching checkpoint 可恢复且用户明确同意时，才作为秋招后的条件任务。

### 8.4 验收标准

- [ ] 大阶段一状态已同步；
- [ ] calibration manifest 在量化前冻结并带 SHA；
- [ ] INT8 artifact 可由 Python/C++ ORT 实际运行；
- [ ] INT8 contract、actual metadata、SHA、大小一致；
- [ ] FP32/INT8 detection 差异有机器可读结果；
- [ ] FP32/INT8 任务质量同协议比较；
- [ ] FP32/INT8 性能、吞吐、内存同协议比较；
- [ ] FP32/INT8 各有 profile JSON 与 top-op 摘要；
- [ ] 现有 Windows correctness/tests 不回退；
- [ ] README 如实记录更快、相当或更慢。

### 8.5 L1 理解重点

用户应能回答：

1. static 与 dynamic quantization 的区别；
2. calibration data 为什么必须冻结；
3. QDQ/QOperator 的基本区别；
4. 为什么模型变小不等于延迟下降；
5. detection matching 与 mAP 分别证明什么；
6. 为什么 profiling 与 benchmark 必须分开；
7. top operator 时间为什么只是定位线索，不自动等于根因；
8. 为什么当前没有 `.pt` 仍可以做 PTQ；
9. 什么条件下才值得 QAT。

### 8.6 代码练习候选

- CalibrationDataReader / manifest iterator；
- FP32/INT8 detection matcher；
- profile JSON top-op aggregator；
- percentile/benchmark validator；
- quantized artifact metadata 校验。

### 8.7 Codex 执行指令

```text
阅读 AGENTS.md、docs/PLAN.md、双语 README、cpp_infer/README.md、本方案 S2-01，以及现有 consistency/benchmark/contract/tests。保护 dirty worktree，并先把用户已完成大阶段一 L2 的状态同步到三份 README。

本次只完成 YOLO FP32 ONNX 的 static INT8 PTQ 与 ORT Profiling。先写 SPEC，并在首次正式量化前冻结 source artifact、calibration manifest、任务质量、detection matching 和 benchmark 协议。生成独立 INT8 artifact，验证 actual metadata、SHA、模型大小、Python/C++ 可运行性、固定样本 detection 差异、任务质量、同协议性能与内存。FP32/INT8 分别生成 profile trace 和 top operator/node 摘要；profile 不得冒充 benchmark。

本步不做 Linux、CUDA、TensorRT、并发、D010、QAT。旧 YOLO gate 不回退。完成测试、机器可读证据、README 和教学闭环后停止，等待 L1。
```

---

## 9. S2-02：Linux x86_64 与 AArch64/QEMU Portability

### 9.1 目标

在一个完整单元中连续完成：

```text
去 Windows coupling
        ↓
Linux x86_64 native
        ↓
去 x86 coupling
        ↓
AArch64 cross compile + QEMU smoke
```

Linux x86_64 与 ARM64/QEMU 属于同一 portability 单元，但必须按 Gate A → Gate B 顺序执行。

### 9.2 Gate A：Linux x86_64 Native

#### A. 最小工具链

在用户授权后准备：

- GCC/G++ 或 Clang；
- CMake/CTest；
- Ninja；
- pkg-config；
- OpenCV C++；
- Linux ORT C++ SDK；
- Python/ORT consistency 环境；
- GTest 可复现来源；
- 调试工具：`gdb`、`ldd`、`readelf`、`file` 等。

记录版本、安装命令和来源。

Docker daemon 当前不可用，不允许成为前置。

#### B. 平台解耦

源码和 CMake 不再假定：

- `onnxruntime.lib` / `onnxruntime.dll`；
- Windows x64 是唯一平台；
- `Psapi` 是唯一内存实现；
- `.cmd/.ps1` 是唯一入口；
- 路径分隔符、locale、动态库行为只有 Windows 形式。

平台差异集中在适配层，Runtime 主链不复制两套。

#### C. Linux 工作流

提供 Linux 薄入口，至少覆盖：

```text
doctor
build / clean-build
test
detect / demo
consistency
benchmark
all
```

工作流语义与 Windows 对齐，但不要求脚本实现完全相同。

#### D. Linux 全链路证据

使用 clean Release 完成：

- Runtime/CLI/tests build；
- 聚焦与完整 CTest；
- fixed Demo JSON/PNG；
- Python ORT/C++ ORT consistency；
- benchmark；
- peak RSS；
- environment/result JSON。

随后重新跑 Windows 关键 gate，证明跨平台改造没有破坏原基线。

### 9.3 Gate B：AArch64 Cross Compile + QEMU

#### A. Toolchain 与目标依赖

建立可复现的 CMake toolchain/sysroot 方案：

```text
host: Linux x86_64
target: Linux AArch64
compiler: aarch64-linux-gnu-gcc/g++
target libs: ARM64 OpenCV / ORT / system runtime
```

需要明确：

- `CMAKE_SYSTEM_NAME`；
- `CMAKE_SYSTEM_PROCESSOR`；
- target compiler；
- sysroot/target dependency search；
- host tool 与 target library 的区别；
- 构建产物 architecture。

#### B. Cross Build

至少生成：

- ARM64 CLI；
- ARM64 Runtime library；
- 选定的 ARM64 tests/smoke；
- 依赖清单与部署目录。

使用 `file`、`readelf` 等证明产物为 AArch64 ELF。

#### C. QEMU Smoke

QEMU 只做 correctness/portability，不做性能。

硬验收最少覆盖：

```text
--help / CLI startup
config + artifact contract
synthetic preprocess/postprocess tests
路径与错误处理
动态库加载
```

目标验收：

```text
fixed image
-> ARM64 ORT CPU
-> preprocess/inference/postprocess
-> Detection JSON
```

若 ARM64 ORT/OpenCV/sysroot 或 QEMU 计算成本在本单元 timebox 内无法稳定完成完整推理，可以降级为：

```text
cross-build + QEMU core smoke
full ARM64 inference = NOT_EXECUTED_UNDER_EMULATION
```

但必须明确限制，不能写成 ARM64 完整部署。

#### D. 禁止性能结论

QEMU 下不发布：

- latency；
- throughput；
- CPU 利用；
- power；
- 与 Jetson/RK3588 的任何性能推断。

### 9.4 非目标

- 真实 ARM64 板卡；
- Jetson；
- RK3588；
- TensorRT/RKNN；
- Docker multi-arch；
- systemd 服务；
- BSP、bootloader、device tree；
- QEMU 性能优化。

### 9.5 验收标准

#### Linux x86_64

- [ ] WSL/Linux doctor 可独立运行；
- [ ] 同一 commit 在 Windows/Linux clean Release build；
- [ ] Linux 使用真实 `.so`、Linux ORT/OpenCV；
- [ ] Linux tests/Demo/consistency/benchmark/peak RSS 完成；
- [ ] Windows 关键 gate 重新通过；
- [ ] README 准确写 `WSL2 Ubuntu x86_64`。

#### AArch64/QEMU

- [ ] CMake toolchain 可复现；
- [ ] 产物由工具证明为 AArch64 ELF；
- [ ] QEMU 可执行选定 CLI/tests；
- [ ] target dependency/loader 问题有可行动诊断；
- [ ] 完整 inference 若完成，有固定 smoke 证据；
- [ ] 若未完成，状态明确且不夸大；
- [ ] 不发布任何 QEMU 性能数字。

### 9.6 L1 理解重点

1. Linux kernel、Ubuntu、x86_64、AArch64 的层次关系；
2. Windows `.lib/.dll` 与 Linux `.a/.so`；
3. build-time、link-time、load-time、run-time 错误区别；
4. WSL 为什么适合 Linux开发，又为什么不是板端；
5. cross compile 与 native compile 的区别；
6. toolchain、sysroot、host tool、target library；
7. QEMU user/system emulation 的目的；
8. 为什么 QEMU只能证明 portability；
9. Peak Working Set 与 peak RSS；
10. 跨平台 consistency 失败的排查顺序。

### 9.7 代码练习候选

- CMake toolchain file；
- imported target / platform find logic；
- Linux `/proc` 或 `getrusage` peak RSS；
- `#ifdef` 最小平台边界；
- Bash workflow 的退出码与产物校验；
- `file/readelf/ldd` 排错；
- QEMU target rootfs/loader 调用。

### 9.8 Codex 执行指令

```text
基于已冻结的 S2-01，本次只完成 Linux x86_64 与 AArch64/QEMU portability。先做 Gate A：使用 WSL2 Ubuntu 24.04，准备最小 GCC/CMake/Ninja/OpenCV/ORT/GTest/Python 环境；Docker不得成为前置。重构 CMake、依赖发现、动态库、内存采集与工作流，使同一源码在 Windows和Linux clean Release构建。完成 Linux tests、Demo、Python/C++ consistency、benchmark、peak RSS，并回归 Windows。

Gate A稳定后再做 Gate B：建立 aarch64 CMake toolchain/sysroot，交叉编译 Runtime/CLI/选定 tests，使用 file/readelf证明 ARM64产物，并通过 QEMU执行 CLI/core correctness smoke。完整 ARM64 ORT inference若能在 timebox 内稳定完成则增加固定单图证据；否则明确 NOT_EXECUTED_UNDER_EMULATION，不发布性能。

本步不做并发、TensorRT、真实板卡、D010。完成证据、README与教学闭环后停止，等待 L1。
```

---

## 10. S2-03：目录/Manifest 多图有界并发系统

### 10.1 目标

把当前单图 Demo 升级为可作为通用 C++/Linux 系统软件讲解的 workload processor：

```text
Directory / Manifest
-> deterministic discovery
-> bounded queue
-> workers
-> existing DetectorPipeline
-> per-image outputs
-> partial failure accounting
-> clean shutdown
-> throughput comparison
```

重点不是写 `for` 循环，而是：

- task lifecycle；
- data ownership；
- resource limit；
- producer-consumer；
- backpressure；
- error propagation；
- thread shutdown；
- correctness/performance trade-off。

### 10.2 用户流程

最小 CLI 应支持：

```text
Runtime config
+ input directory or manifest
+ output directory
+ worker count
+ queue capacity
+ overwrite/output mode
-> execute
-> per-image result
-> batch summary
-> process exit code
```

正式 evidence 使用冻结 manifest；目录输入作为用户便利入口。

### 10.3 范围

#### A. WorkItem Discovery

- manifest 作为可复现正式入口；
- directory 作为便利入口；
- 扩展名白名单；
- deterministic sorting；
- 空目录、重复项、非普通文件、递归输出目录有明确行为；
- 输入到输出路径映射稳定且无冲突；
- 不在 discovery 阶段预加载所有图像 tensor。

#### B. Bounded Queue

队列必须：

- capacity 可配置且有合理上限；
- full 时 producer 等待，形成 backpressure；
- empty 时 consumer 等待；
- close/stop 后拒绝新任务；
- 使用 condition variable predicate 处理虚假唤醒；
- 所有任务有清晰 ownership；
- 不允许无限积压 WorkItem、tensor 或 result。

#### C. Worker

每个 worker 循环：

```text
wait/pop WorkItem
-> execute existing single-image pipeline
-> write requested outputs
-> record task status
-> continue or stop according to failure policy
```

Codex根据正确性、线程安全、内存和实现成本选择：

- per-worker Pipeline/Session；
- shared Session；
- 或经过验证的其他策略。

首选最简单、最容易证明正确的方案，不提前追求极致吞吐。

#### D. Failure Model

SPEC 必须冻结：

- 单图失败是否继续；
- 成功、失败、跳过、取消的定义；
- 最终退出码；
- 是否保留已成功产物；
- output writer 失败的行为；
- worker exception 如何捕获并传播；
- 重跑/overwrite 规则。

#### E. Clean Shutdown

硬要求覆盖：

- 正常生产完成；
- producer 提前失败；
- worker exception；
- writer failure；
- queue closed 且仍有待处理任务；
- 内部 stop request；
- 所有线程在有限时间内退出并 `join`。

外部 `Ctrl+C/SIGINT` 集成只在不破坏跨平台和 timebox 的前提下加入；第一版必须先有可测试的内部 stop/cancel 机制。

#### F. Correctness

- `worker=1` 与逐图调用原单图产品链结果一致；
- 多 worker 与 `worker=1` 检测语义一致；
- 输出路径无竞争覆盖；
- 重复运行顺序和 summary 稳定；
- 不改变 score/NMS/坐标语义。

#### G. Performance

固定 manifest 下比较：

```text
worker=1
worker=2
worker=4
必要时 worker=8
```

记录：

- total job time；
- compute throughput / full-job throughput；
- per-image mean/P50/P95；
- peak memory；
- worker/queue settings；
- ORT intra-op/inter-op settings；
- correctness status；
- partial failure count。

必须解释：

- worker 增加为什么不线性加速；
- external workers 与 ORT internal threads 的 oversubscription；
- session 数量与内存；
- 写盘/绘图是否进入协议。

### 10.4 非目标

- true ONNX batch `N>1`；
- 摄像头/视频；
- 网络服务；
- 分布式调度；
- 无锁队列；
- work stealing；
- C++20 `jthread/stop_token` 大改；
- GPU stream concurrency；
- D010 特殊批处理；
- 为了 benchmark 取消正确性或输出。

### 10.5 测试矩阵

| 模块 | 核心行为 |
|---|---|
| Queue | capacity、full backpressure、empty wait、close、stop、虚假唤醒 |
| Discovery | deterministic sort、扩展名、空目录、重复/非法路径 |
| Worker | 消费、异常捕获、状态聚合、全部 join |
| Output | 路径映射、冲突、overwrite、部分失败 |
| Correctness | worker=1 vs 单图；multi-worker vs worker=1 |
| Shutdown | 正常、producer failure、worker failure、stop |
| Benchmark | settings、样本、公式、memory、correctness metadata |

队列/关闭语义优先 synthetic test；真实模型只承担小规模 integration 和正式 benchmark。

### 10.6 验收标准

- [ ] 一个命令可处理冻结 manifest 与目录；
- [ ] 输出 per-image JSON/PNG 或指定结果；
- [ ] 生成机器可读 BatchSummary；
- [ ] worker=1 与单图链一致；
- [ ] queue capacity 真实限制待处理任务；
- [ ] full queue backpressure 有可观察测试；
- [ ] 正常、任务错误、writer 错误、stop 均无死锁；
- [ ] 所有线程有限时间退出；
- [ ] Windows/Linux 都能运行核心 gate；
- [ ] 单线程/并发吞吐、P50/P95、内存同协议比较；
- [ ] README 不把并发单图写成 true batch。

### 10.7 L1 理解重点

1. bounded queue 为什么必要；
2. backpressure 在解决什么系统问题；
3. condition variable predicate 与虚假唤醒；
4. producer/consumer 的 ownership；
5. queue close、stop、empty 的状态语义；
6. shared session 与 per-worker session 的 trade-off；
7. 单图失败继续执行时 batch 最终状态如何定义；
8. clean shutdown 为什么不能用强制结束代替；
9. worker 增加为什么吞吐不线性；
10. ORT内部线程与应用 worker 如何造成 oversubscription；
11. throughput 与 per-image latency 的区别。

### 10.8 代码练习候选

- `BoundedBlockingQueue<T>`；
- close/stop-aware `push/pop`；
- condition variable predicate；
- RAII thread join；
- worker exception capture；
- deterministic output path mapping；
- BatchSummary aggregation；
- percentile/throughput 统计。

### 10.9 Codex 执行指令

```text
基于 Windows/Linux 单图 Runtime，本次只完成目录/manifest 多图有界并发系统。先写 SPEC，冻结输入发现、排序、输出映射、单图失败策略、最终退出码、worker count、queue capacity、backpressure、stop和clean shutdown。

实现 Directory/Manifest -> WorkItem -> BoundedQueue -> Workers -> existing DetectorPipeline -> per-image outputs -> BatchSummary。不得复制 preprocess/ORT/postprocess。选择最简单、可证明正确的 session strategy，并记录 correctness/memory/throughput trade-off。先以 synthetic test验证 queue、close、stop和异常，再以真实固定 manifest验证 worker=1 vs 单图、multi-worker correctness，并在 Windows/Linux比较 throughput、P50/P95和peak memory。

不做 true batch、视频、服务、GPU streams、无锁队列或D010。完成 README、机器可读 summary、benchmark和教学闭环后停止，等待 L1。
```

---

## 11. S2-04：Linux x86_64 本地 TensorRT 加速

### 11.1 目标

在不购买板卡的前提下，使用：

```text
Linux x86_64
+ RTX 4060 Laptop
+ CUDA
+ TensorRT
+ current YOLO ONNX
```

形成真实 accelerator inference evidence。

本单元让用户可以投递和讲解：

- AI inference；
- model deployment；
- Edge AI software；
- CUDA/TensorRT related C++ roles。

但口径必须是：

> Linux x86_64 local GPU / edge-node TensorRT

不得写 Jetson、ARM64 GPU 或 embedded device。

### 11.2 执行原则：先最低风险闭环，再决定 native backend

为了加快完成，实施顺序固定为：

```text
环境 preflight
-> trtexec/current ONNX smoke
-> 一个真实 C++ TensorRT execution path
-> correctness
-> FP16 performance
-> optional INT8/native deepening
```

C++ TensorRT path 的选择顺序：

1. **优先 ORT TensorRT EP：**复用现有 `OnnxRunner`、artifact、pre/postprocess 与测试，最快形成产品闭环；
2. 同时注册 CUDA EP 作为 TensorRT 不支持节点的 fallback；
3. 只有以下任一条件成立时才做 native TensorRT backend：
   - 高优先 JD 反复要求 native TensorRT/CUDA API；
   - ORT TensorRT EP 无法满足产品正确性；
   - EP 路线完成后仍有明确时间预算；
   - native backend 可在额外 timebox 内闭环。

无论采用哪条路线，都必须真实执行 TensorRT，不允许只跑 Python 包检测或仅安装 SDK。

### 11.3 环境 Preflight

重新建立兼容版本矩阵：

```text
NVIDIA driver
CUDA user-space/toolkit
cuDNN
TensorRT
ONNX Runtime GPU C++ SDK（若走 EP）
compiler/CMake
Linux/WSL GPU bridge
```

不得：

- 将 `nvidia-smi` 显示的 CUDA driver capability 当作已安装 Toolkit；
- 混用不兼容 CUDA/cuDNN/TensorRT；
- 用 Python wheel DLL/so 冒充完整 C++ SDK；
- 在系统全局无记录升级多个版本。

最低 preflight：

- `nvidia-smi`/GPU 可见；
- CUDA sample 或最小 C++ smoke；
- `trtexec --version`；
- `trtexec` 能解析并运行当前 YOLO ONNX；
- profile/engine/cache 输出路径可控。

若一个安装路径连续两个 timebox 均失败，应切换到干净隔离方案，而不是无限追包。

### 11.4 产品范围

#### A. TensorRT Execution Path

若采用 ORT TensorRT EP：

- provider config/registration；
- CUDA fallback；
- engine cache；
- actual provider/profile evidence；
- 输入/输出契约与 CPU path 一致；
- FP16 选项；
- 错误信息包含版本/provider/cache/action。

若采用 native backend，至少负责：

```text
ONNX parse/build or serialized engine load
execution context
input/output binding
host/device buffer
CUDA stream/synchronization
inference enqueue
owned output copy
engine/cache identity
```

native backend 只服务冻结 YOLO artifact，不扩成通用任意 ONNX engine。

#### B. Precision

硬目标：

- FP32 或 TensorRT default baseline；
- FP16；
- 与 ORT CPU FP32 结果比较。

条件目标：

- TensorRT INT8。

TensorRT INT8 只有在 S2-01 calibration 可复用、精度协议清楚且不会阻塞 FP16主链时进入。

#### C. Correctness

固定 manifest 比较：

- detection count；
- class；
- confidence error；
- bbox error；
- matching IoU；
- repeated-run stability。

不能因为 GPU 浮点差异直接复用 CPU 最严容差；若需新容差，必须先保留失败分布和理由，再版本化协议，禁止为了全绿随意放宽。

#### D. Performance

在同一 Linux x86_64 主机、固定 manifest、固定 output mode 下记录：

- engine/session initialization；
- preprocess；
- host-to-device/device-to-host（若可观测）；
- TensorRT execution；
- postprocess；
- pipeline/end-to-end mean/P50/P95；
- throughput；
- host RSS；
- GPU memory；
- precision；
- engine/cache build/load 状态。

正式 benchmark 前关闭 profiling；profiling/verbose log 单独运行。

#### E. 和并发单元的边界

第一版 S2-04 先使用单 worker/单 inference path，避免将 GPU stream、worker pool 和 TensorRT环境问题同时引入。

S2-03 的多图系统可以在 S2-04 完成后做一个最小集成 smoke，但 GPU stream并发、multi-context调优不属于本阶段。

### 11.5 非目标

- Jetson；
- ARM64 TensorRT cross compile；
- DeepStream；
- TensorRT-LLM；
- CUDA kernel 自定义优化；
- multi-stream GPU并发；
- D010；
- 同时实现 ORT CUDA、TensorRT EP、native TensorRT 三套完整产品后端；
- 把 RTX 性能外推到 Jetson；
- 为 TensorRT 改变冻结后处理语义。

### 11.6 验收标准

- [ ] Linux RTX/CUDA/TensorRT 兼容环境有版本矩阵；
- [ ] `trtexec` 对当前 ONNX 真实成功；
- [ ] C++ 产品链存在一个真实 TensorRT execution path；
- [ ] FP16 实际运行；
- [ ] TensorRT detection 与 ORT CPU 通过预声明 correctness gate；
- [ ] 重复运行稳定；
- [ ] benchmark 有 initialization/P50/P95/throughput/RSS/GPU memory；
- [ ] engine/cache 与模型 SHA、TensorRT版本、GPU环境绑定；
- [ ] 原 Windows/Linux ORT gate 不回退；
- [ ] README 明确写 `Linux x86_64 + RTX 4060`，不写 Jetson；
- [ ] native TensorRT/INT8 若未做，准确列为后续条件项。

### 11.7 L1 理解重点

1. CUDA、cuDNN、TensorRT、ORT TensorRT EP 的关系；
2. driver capability 与 CUDA Toolkit 的区别；
3. TensorRT engine、runtime、execution context；
4. 为什么 engine/cache 与模型、版本、GPU环境绑定；
5. TensorRT EP 与 native TensorRT 的 trade-off；
6. 为什么推荐 CUDA fallback；
7. FP32/FP16/INT8；
8. host/device memory 与 H2D/D2H；
9. warmup、engine build、steady-state benchmark；
10. GPU快了为什么 end-to-end 不一定等比例下降；
11. 为什么桌面 TensorRT 是 Edge AI readiness，不是 Jetson 实机。

### 11.8 代码练习候选

- provider registration/config；
- TensorRT engine cache key；
- host/device buffer ownership；
- CUDA error/RAII wrapper；
- precision config；
- GPU correctness matcher；
- benchmark boundary；
- environment/version validator。

### 11.9 Codex 执行指令

```text
基于已完成的 Linux/并发核心，本次只完成 Linux x86_64 + RTX 4060 的真实 TensorRT acceleration。先重新调查并冻结 driver/CUDA/cuDNN/TensorRT/ORT GPU C++ SDK兼容矩阵，禁止复用已知不兼容的旧组合。先以 CUDA最小C++ smoke和trtexec/current YOLO ONNX验证环境。

为了加快完成，优先在现有 OnnxRunner 中接入 ORT TensorRT EP，并注册 CUDA fallback、engine cache和可行动诊断；如果 EP无法满足正确性或高优先JD明确要求且有剩余timebox，再实现冻结YOLO的最小native TensorRT backend。至少完成FP16真实执行、同artifact correctness、P50/P95、throughput、host RSS和GPU memory；TensorRT INT8仅在不阻塞主链时做。

本步不做Jetson、ARM64 TRT、DeepStream、GPU stream并发、D010、LLM。README必须准确写Linux x86_64 + RTX，不得称板端。完成证据、回归和教学闭环后停止，等待 L1。
```

---

## 12. S2-05：Recruiting Freeze 与大阶段二收口

### 12.1 目标

本单元不增加技术栈，只把 S2-01～S2-04 转化为：

```text
可复现代码
+ 清晰证据
+ 可解释失败案例
+ 可追问项目叙事
+ 多版本简历
```

这是大阶段二是否真正有秋招价值的最终门。

### 12.2 全量工程 Gate

至少执行并记录：

#### Windows x86_64

- clean Release build；
- full CTest；
- FP32/INT8 Demo/consistency/benchmark；
- profile evidence validation；
- batch worker=1/multi-worker gate。

#### Linux x86_64

- clean Release build；
- full/selected CTest；
- Demo/consistency/benchmark/peak RSS；
- batch/concurrency；
- TensorRT correctness/performance。

#### Linux AArch64/QEMU

- cross build；
- architecture validation；
- QEMU core smoke；
- 完整 inference 若已实现则复现；
- 明确无性能结论。

### 12.3 结果矩阵

至少形成：

| 证据线 | 平台/后端 | 正确性 | 性能 | 内存 | 任务质量 | 限制 |
|---|---|---|---|---|---|---|
| YOLO FP32 | Windows ORT CPU | 30图一致性 | P50/P95 | PWS | baseline | current baseline |
| YOLO INT8 | Windows ORT CPU | FP32/INT8 matching | same protocol | PWS | metric/delta | quantization coverage |
| ORT Profile | Windows CPU | stable run | top op/node | trace metadata | N/A | profiler overhead |
| Linux ORT | Linux x86_64 CPU | Python/C++ | P50/P95 | peak RSS | same artifact | WSL不是实机 |
| ARM64 | QEMU | core/full smoke | 不发布 | 不发布 | N/A | emulation only |
| Batch/Concurrent | Windows/Linux | worker consistency | images/s/P50/P95 | peak memory | model unchanged | not true batch |
| TensorRT | Linux x86_64 RTX | ORT CPU vs TRT | FP16 P50/P95 | RSS/GPU memory | same artifact | not Jetson |

每一行都能追溯到：

```text
command
config/contract
artifact SHA
manifest
raw JSON/profile/log
README summary
known limitations
```

### 12.4 README 收口

`README.md`、`README_zh.md`、`cpp_infer/README.md` 必须事实对齐，并包含：

1. 项目定位；
2. 当前总体架构；
3. Windows/Linux/AArch64/TensorRT 平台矩阵；
4. Quick Start；
5. INT8 与 profiling 结果；
6. batch/concurrency 使用方式；
7. correctness 和 benchmark；
8. 真实失败案例；
9. 当前限制；
10. 明确冻结项和未来扩展。

README 不能变成文档索引；主故事必须在 README 中完整可读。

### 12.5 三套投递口径

#### A. 通用 C++ / Linux 软件版

强调：

- C++17/CMake；
- RAII/ownership；
- Windows/Linux；
- AArch64 portability；
- bounded queue/workers；
- testing/debugging/performance。

#### B. QA / SDET 版

强调：

- requirement/acceptance criteria；
- unit/integration/negative/regression；
- failure injection；
- Python/C++ consistency；
- deterministic fixture；
- automated gates；
- bug reproduction/diagnosis。

#### C. AI Inference / Edge AI 版

强调：

- ONNX artifact；
- ORT/TensorRT；
- INT8/FP16；
- profiling；
- correctness before performance；
- Linux x86_64 local edge-node；
- ARM64 portability；
- 不夸大真实板端。

### 12.6 面试材料

必须产出：

- 30 秒项目定位；
- 2 分钟主链讲解；
- 5 分钟完整讲解；
- 至少 15 个主问题；
- 每个主问题至少 2 层追问；
- 至少 5 个真实失败/排错案例；
- 3 类简历 bullet；
- 一次专项 mock；
- code-practice 清单。

建议五个失败案例：

1. INT8 质量退化或没有加速；
2. Linux 动态库/ABI/路径问题；
3. AArch64 sysroot/loader/QEMU 问题；
4. bounded queue/worker shutdown 或 oversubscription；
5. CUDA/cuDNN/TensorRT版本或 provider fallback 问题。

### 12.7 大阶段二 L2

用户应能：

1. 5 分钟讲清完整系统；
2. 回答至少 15 个主问题与连续追问；
3. 解释至少 5 个失败案例；
4. 指出关键模块、入口、数据 ownership 和证据路径；
5. 在 AI 指导下跑通 Windows、Linux、batch、ARM64 smoke、TensorRT；
6. 完成一次跨模块行为修改并补测试；
7. 根据不同 JD 选择三套 bullet；
8. 明确哪些结论是 Windows、Linux、QEMU、RTX 或未来板端证据。

### 12.8 代码练习冻结清单

优先级最高：

```text
RuntimeConfig/Artifact/Metadata 三层契约
ORT RAII 与 input borrow/output ownership
letterbox + HWC->CHW
YOLO raw indexing + IoU/NMS + restore
BoundedBlockingQueue
condition_variable + clean shutdown
profile top-op aggregation
cross-platform peak memory
CMake Linux/AArch64 toolchain
TensorRT provider/buffer lifecycle
```

### 12.9 验收标准

- [ ] S2-01～S2-04 的硬目标有真实结果或准确 blocker；
- [ ] Windows/Linux gates 可复现；
- [ ] AArch64/QEMU口径准确；
- [ ] 并发系统无死锁且性能协议完整；
- [ ] TensorRT实际执行且不冒充Jetson；
- [ ] 所有简历数字可追溯；
- [ ] README 三份事实一致；
- [ ] 三套简历口径完成；
- [ ] 30秒/2分钟/5分钟口述完成；
- [ ] 用户通过 L2；
- [ ] 项目进入 recruiting freeze，不再无目标加模块。

### 12.10 Codex 执行指令

```text
本次只做大阶段二 Recruiting Freeze，不增加模型、UI、设备或框架。使用可复现 checkpoint 依次执行 Windows、Linux、batch/concurrency、AArch64/QEMU和TensorRT的适用 gates；核对所有结果、SHA、manifest、命令、环境、容差和限制。

更新三份 README，使项目定位、架构、Quick Start、INT8、profiling、Linux、AArch64、并发、TensorRT、测试、benchmark、失败案例和限制事实一致。输出通用C++/Linux、QA/SDET、AI inference/Edge AI三种简历口径，准备30秒/2分钟/5分钟讲解、至少15个问题与追问、至少5个排错案例、专项mock和代码练习清单。

完成后进入 recruiting freeze。D010、Qt、LLM、Agent、真实板卡继续冻结；只有真实JD或面试反馈才能解冻。
```

