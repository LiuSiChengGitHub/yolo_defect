# Codex 协作与推进规则

## 1. 项目定位与当前状态

项目中文名为 **工业视觉边缘 AI Runtime 与 C++ 工程化系统**，英文名为 **Industrial Vision Edge AI Runtime and C++ Engineering System**。它服务于 2026 秋招简历和面试，重点是通过真实开发过程理解现代 C++、跨平台构建、推理工程、性能分析、并发和边缘部署，而不是按生产上线标准堆叠防御工程。

当前状态：**大阶段一与用户 L2 已完成；S2-01、S2-02、S2-03 和 S2-04 的实现、相称回归、证据与教学收口均已完成，当前停止并等待用户 L1；S2-05 尚未开始。** S2-01 最终练习产物是全 64 Conv 的 QDQ/U8S8 Round 2 模型；Round 1 QDQ/S8S8 保留为“量化后变慢”的诊断案例。S2-02 已证明同一业务源码可在 Windows/Linux x86_64 运行并交叉编译到 Linux AArch64。S2-03 增加了目录/manifest 确定性发现、有界并发、背压、逐图输出、部分失败统计、协作式退出和机器可读 `BatchSummary`，并完成双 x86_64 平台 361 图比较及 AArch64/QEMU 功能验证。S2-04 在 WSL2/Linux x86_64 + RTX 4060 Laptop 上完成当前 ONNX 的真实 `trtexec`、ORT TensorRT EP placement 诊断，以及最终无 fallback 的 native TensorRT `enqueueV3` 产品路径；ORT EP 因冻结正确性门禁失败只作诊断，最终 E0 engine 采用 DFL Softmax FP16、其余 compute FP32/noTF32 的受约束混合精度并通过互斥 v4 holdout。QEMU 不是开发板，不产生性能、内存或原生 ARM 设备结论。

已验证主链：

```text
RuntimeConfig + ModelArtifactSpec -> actual ModelMetadata
-> OpenCV preprocess -> ORT CPU / ORT TensorRT EP diagnostic / native TensorRT inference
-> YOLO decode/NMS/coordinate restore -> JSON/visualization
-> single-image output 或 deterministic discovery -> bounded queue -> workers
-> per-image output + ordered BatchSummary/consistency/benchmark/profiling
-> Windows x86_64、WSL2/Linux x86_64 与 QEMU/Linux AArch64 共享业务主链
```

## 2. 渐进式披露与权威来源

- 每次新对话只默认读取本文件；用户当前指令优先于本文件，真实源码和实际运行结果决定完成状态。
- 进入某个 S2 单元时，只读取 `docs/Proj1_S2.md` 的总则、目标架构和对应单元；其中旧路径、旧日期和旧优先级只是历史文字。未经用户明确要求不得改写该文档。
- 当任务需要查找工具链路径、依赖版本或位置、环境入口命令或已知环境踩坑时，使用仓库 Skill：`.agents/skills/yolo-defect-dev/SKILL.md`，并按需读取 `docs/paths_commands.md`。不得仅因任务涉及构建、运行、测试、benchmark 或 profiling 就加载该 Skill；它不规定实现或验证方式。本文件不保存机器路径和工具链细节。
- 单元收口、L1/L2、复盘或面试准备时才读取 `docs/learning_closure.md`；日常开发不加载九部分模板。
- 用户说“九部分输出”的时候，读取 `docs/learning_closure.md`，完成文档中的要求。
- README.md、README_zh.md 和 `cpp_infer/README.md` 只在入口事实变化或单元收口时定向读取和同步。
- `docs/路线0712-new.md` 只在核对秋招方向、项目定位或投递策略时读取；archive 和旧规划不作为当前实现依据。

## 3. 大阶段二路线

1. S2-01：INT8 PTQ 与 ORT Profiling——已实现，等待 L1。
2. S2-02：Linux x86_64 与 AArch64/QEMU——Gate A/Gate B、最终三平台回归和教学收口已完成，等待用户 L1。
3. S2-03：目录/Manifest 有界并发——三平台功能回归与双 x86_64 平台正式比较均已完成，等待用户 L1。
4. S2-04：Linux x86_64 + RTX 4060 TensorRT——真实执行、正确性、性能、资源证据与教学收口已完成，等待用户 L1。
5. S2-05：证据、简历/面试材料与 Recruiting Freeze——尚未开始。

一次只推进一个完整单元，不并行实现后续单元。固定闭环仍是最小 SPEC、实现、相称验证、必要结果、入口文档同步，然后停止等待用户 L1；S2-05 做 L2。

## 4. 学习优先与分级验证

- 优先交付可运行能力、清晰架构、关键 trade-off 和可独立讲解的理解；不因想象中的生产风险扩展任务。
- 纯文档改动只做 diff、链接和基本格式检查；不构建 Runtime。
- 局部代码改动构建相关 target、运行相关测试并做一次代表性主路径 smoke。
- 仅在 S2 单元收口、跨平台、并发生命周期、核心 Runtime 契约变化或用户明确要求时运行完整回归。
- benchmark/profiling 只在性能或执行位置属于任务目标时运行，并与功能 smoke 分开。
- SHA 只用于模型、外部二进制依赖或正式发布 artifact 等确需确认二进制身份的场景；不默认计算普通 JSON、图片、trace 或每个中间产物的 SHA。
- 不默认新增重复 schema validator、evidence assembler、字段级防御测试或与目标无关的 gate；一个测试应对应一个有学习或回归价值的行为。
- 实验可以使用小样本和有限机器条件，但标为“实测”的数字必须真实运行；示例或合成数据必须明确标注，不能伪装成 benchmark。
- 未运行的验证必须准确说明；测试数量、安装成功或文档声明本身不等于功能完成。

## 5. 架构不变量

- 保持 C++17、CMake、OpenCV、ONNX Runtime C++、GTest 主技术栈；依赖显式、可配置、可诊断。
- Runtime library 负责配置、artifact、metadata、preprocess、backend、postprocess 和结果；CLI/workflow 只编排参数、文件与退出码。
- Demo、benchmark、batch 和 backend 复用 `DetectorPipeline` 或清晰抽象，不复制 preprocess、decode、NMS 和坐标恢复。
- `RuntimeConfig` 管运行选择，`ModelArtifactSpec` 管模型语义，`ModelMetadata` 表示 ORT 实际观察；三层权威不混用。
- backend 暴露模型无关的 owned I/O 和可行动错误；模型族逻辑留在 pre/postprocess。
- Windows/Linux 共用业务源码，平台差异放在薄适配层、CMake 或 workflow；用 RAII 和明确 ownership 管理 session、buffer、文件、线程与 profile。
- 优化不得改变 score、NMS、类别和坐标语义；错误说明失败位置、期望、实际和纠正动作。

## 6. 平台与事实边界

- Windows x86_64 已验证 FP32/INT8 ORT CPU、Release/NMake、分段 benchmark、Peak Working Set 和逐节点 profiling；S2-04 最终 clean Release 通过 179/179 CTest，两个需要 symlink/reparse privilege 的 GTest case 按平台 skip、无失败。S2-03 正式 361 图 FP32 CPU、JSON-only、queue=8 比较中，worker=1 为 `6.285556 img/s`、PWS `151.804688 MiB`，worker=4 为 `17.853923 img/s`、PWS `505.085938 MiB`，吞吐比 `2.840468`，361 份逐图检测完全一致。
- WSL2/Linux x86_64 已验证共享源码 Release/Ninja、ELF/`ldd`、固定单图与 30 图一致性；S2-04 当前源码 Release 回归通过 179/179 CTest。S2-03 正式 361 图 FP32 CPU、JSON-only、queue=8 比较在 WSL2 原生 ext4 临时工作区运行，worker=1 为 `8.113806 img/s`、peak RSS `205.765625 MiB`，worker=4 为 `20.159584 img/s`、peak RSS `588.226563 MiB`，吞吐比 `2.484603`，361 份逐图检测完全一致。这些只能写作 WSL2/Linux 证据。
- Linux AArch64 已在 WSL2 x86_64 host 上完成 cross-build、AArch64 ELF/target loader 检查，并在 QEMU user-mode 下通过固定单图以及 S2-03 目录 worker=1、manifest worker=2、有限队列、逐图一致性、`2 成功 + 1 失败`和 `BatchSummary` 回归。它只证明构建和功能可移植性，不是 ARM 板卡或原生 ARM 运行证据，也不发布 QEMU 吞吐、延迟或内存结论。
- S2-04 在 WSL2/Linux x86_64 + RTX 4060 Laptop 上验证了 TensorRT 10.4.0/CUDA 12.6 的真实产品执行。最终 E0 plan SHA-256 为 `E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746`，仅 DFL Softmax compute 为 FP16，其余 compute/I/O 为 FP32 且 noTF32；互斥 v4 30 图 CPU/A/B 正确性全部通过，A/B detection JSON 字节一致。正式 A/B pipeline throughput 为 `137.651666/140.554957 img/s`，pipeline P50/P95 为 `6.974481/8.779498 ms` 与 `6.519100/10.489540 ms`，host peak RSS 为 `384.667969/384.371094 MiB`，device-wide GPU memory baseline-to-peak 均为 `155 MiB`。吞吐稳定但尾延迟不稳定；显存不是 PID-specific。该证据只代表本地 WSL2/Linux x86_64 GPU/edge-node，不得写成 native Linux、Jetson、ARM64 GPU 或嵌入式实机。
- 性能结果必须说明机器、provider、线程、样本和限制；PWS、RSS、GPU memory 不直接等同。
- matching `.pt` 不在工作区或 Git 历史中，不得声称重新完成 PyTorch/ONNX/C++ 三方 lineage。

## 7. 工作区、安全与交付

- 开始前检查 `git status`/相关 diff，保留用户和其他任务的未提交修改；禁止 reset、覆盖、顺手格式化或删除无关文件。
- 默认修改范围是 `cpp_infer/`、AGENTS.md、根双语 README 和当前单元明确命名的 docs。legacy Python/API/Docker/数据/模型范围需用户明确授权。
- 不修改仓库外文件或兄弟项目；破坏性操作先确认精确目标和授权。
- 源码 MIT 不意味着模型继承 MIT；模型 metadata 报告 AGPL-3.0，NEU-DET 条款待确认，公开分发前单独核查。
- 单元结束同步 AGENTS.md、README.md、README_zh.md 的状态、命令、结果、限制和下一步；随后停止，不自动启动下一单元。当前下一步是等待用户完成 S2-04 L1，未经指示不启动 S2-05。
