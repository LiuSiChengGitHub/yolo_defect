# Codex 协作与推进规则

## 1. 项目定位与当前状态

项目中文名为 **工业视觉边缘 AI Runtime 与 C++ 工程化系统**，英文名为 **Industrial Vision Edge AI Runtime and C++ Engineering System**。它服务于 2026 秋招简历和面试，重点是通过真实开发过程理解现代 C++、跨平台构建、推理工程、性能分析、并发和边缘部署，而不是按生产上线标准堆叠防御工程。

当前状态：**大阶段一与用户 L2 已完成；S2-01、S2-02 和 S2-03 已完成正式 Round 2 QDQ/U8S8 融合收口，当前停止并等待用户 L1；S2-04 尚未开始。** 正式 INT8 产物为全 64 Conv 的 `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`（SHA-256 `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`）；Round 1 QDQ/S8S8 保留为“量化后变慢”的诊断案例。当前同一 `RuntimeConfig + ModelArtifactSpec -> DetectorPipeline -> BatchRunner` 主链已在 Windows x86_64 完成 INT8 单图，在 WSL2/Linux x86_64 完成 INT8 单图、manifest 和 361 图有界并发比较，并交叉编译到 Linux AArch64 后在 QEMU user-mode 完成 INT8 单图、目录/manifest、逐图一致与部分失败验收。既有 FP32 默认链路仍保留并回归通过。QEMU 不是开发板，不产生性能、内存或原生 ARM 设备结论。

已验证主链：

```text
RuntimeConfig + ModelArtifactSpec -> actual ModelMetadata
-> OpenCV preprocess -> ONNX Runtime C++ inference
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

1. S2-01：INT8 PTQ 与 ORT Profiling——已实现，Round 2 U8S8 已纳入跨平台多图主链，等待 L1。
2. S2-02：Linux x86_64 与 AArch64/QEMU——Gate A/Gate B、三平台回归与 U8S8 交叉编译/QEMU 功能验收已完成，等待用户 L1。
3. S2-03：目录/Manifest 有界并发——FP32 原收口和正式 U8S8 多图融合证据均已完成，等待用户 L1。
4. S2-04：Linux x86_64 + RTX 4060 TensorRT——尚未开始。
5. S2-05：证据、简历/面试材料与 Recruiting Freeze。

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

- Windows x86_64 最终 Release/NMake 通过 `157/157` CTest；其中两个需要 symlink/reparse 权限的 GTest case 在当前账号下显示 skip，对应行为已在 Linux 执行通过。默认 FP32 主链不变，正式 U8S8 单图产品链实际得到 3 个 detections。历史 361 图 FP32 queue=8 比较仍保留：worker=1 `6.285556 img/s` / PWS `151.804688 MiB`，worker=4 `17.853923 img/s` / `505.085938 MiB`，吞吐比 `2.840468`，361 份逐图检测完全一致。
- WSL2/Linux x86_64 最终 Release/Ninja 通过 `157/157` CTest，默认 FP32 Demo 回归通过。正式 U8S8 单图为 3 detections；30 图 manifest 以 workers=2/queue=4 完成 `30/30`，queue peak=4、producer waits=25。本轮 361 图 U8S8 CPU、JSON-only、queue=8 在 `/mnt/d` DrvFs 工作区运行：worker=1 为 `4.591151 img/s` / peak RSS `192.933594 MiB` / waits=353，worker=4 为 `15.903088 img/s` / `556.882812 MiB` / waits=350，吞吐比 `3.463857`，361 份逐图 JSON 字节与语义完全一致。这些数字只用于本次 WSL2/Linux 同协议内比较，不与旧 ext4 FP32 数字直接对比。
- Linux AArch64 已在 WSL2 x86_64 host 上完成 clean cross-build、AArch64 ELF/target loader 检查。QEMU user-mode 下，正式 U8S8 单图得到 3 detections，目录 worker=1 和 manifest worker=2 各 `2/2` 且逐图字节/语义一致，损坏 JPEG 得到精确 `2 成功 + 1 失败`、exit 2，partial-failure queue=1 且 producer waits=1。默认 FP32 `all` 回归也通过。QEMU 只证明构建与功能可移植性；summary 中的性能/内存字段不可发布，不能写成 ARM 板卡或原生 ARM 结论。
- S2-01 的 advisory 质量事实不改写为 strict 全绿：30 图 agreement precision 为 `0.938462 < 0.95`，361 图 mAP50 drop 为 `0.010356 > 0.01`，本次完成的是产品工程链融合与跨平台功能收口。
- S2-04 的 RTX 4060 只代表本地 Linux x86_64 GPU/edge-node，不得写成 Jetson、ARM64 GPU 或嵌入式实机。
- 性能结果必须说明机器、provider、线程、样本和限制；PWS、RSS、GPU memory 不直接等同。
- matching `.pt` 不在工作区或 Git 历史中，不得声称重新完成 PyTorch/ONNX/C++ 三方 lineage。

## 7. 工作区、安全与交付

- 开始前检查 `git status`/相关 diff，保留用户和其他任务的未提交修改；禁止 reset、覆盖、顺手格式化或删除无关文件。
- 默认修改范围是 `cpp_infer/`、AGENTS.md、根双语 README 和当前单元明确命名的 docs。legacy Python/API/Docker/数据/模型范围需用户明确授权。
- 不修改仓库外文件或兄弟项目；破坏性操作先确认精确目标和授权。
- 源码 MIT 不意味着模型继承 MIT；模型 metadata 报告 AGPL-3.0，NEU-DET 条款待确认，公开分发前单独核查。
- 单元结束同步 AGENTS.md、README.md、README_zh.md 的状态、命令、结果、限制和下一步；随后停止，不自动启动下一单元。
