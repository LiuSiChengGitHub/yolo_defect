# Codex 协作与推进规则

## 1. 项目定位与当前状态

项目中文名为 **工业视觉边缘 AI Runtime 与 C++ 工程化系统**，英文名为 **Industrial Vision Edge AI Runtime and C++ Engineering System**。

本项目服务于 2026 秋招简历和面试，重点展示现代 C++、Linux、测试调试、推理工程、性能分析、并发和边缘部署能力。目标是把工业视觉模型产物做成可配置、可运行、可测试、可比较、可复现、可追问的工程系统。

当前状态：**大阶段一自动工程门与用户 L2 均已完成；S2 前置文档已收口；S2-01 尚未开始。**

已经验证的单图主链为：

~~~text
RuntimeConfig + ModelArtifactSpec
-> 实际 ONNX ModelMetadata 校验
-> OpenCV letterbox / RGB / normalize / NCHW
-> ONNX Runtime C++ CPU inference
-> YOLO decode / score filter / NMS / coordinate restore
-> Detection JSON / visualization
-> Python ORT / C++ consistency
-> segmented benchmark / throughput / memory / automated gates
~~~

## 2. 权威来源与默认阅读规则

- 每次新对话先读本文件；它是默认协作入口。
- 不自动预读 README.md、README_zh.md 或 cpp_infer/README.md。仅在用户明确要求查看/更新，或当前单元已进入既定的三文档收口时，定向读取相关段落。
- docs/Proj1_S2.md 是大阶段二的顶层设计。进入某个 S2 单元前，只读取该文档的总则、目标架构和对应单元。
- 该 S2 文档中残留的旧规划路径、旧优先级标签和旧日期均视为历史文字，不具权威性；旧规划文件不再读取或引用。
- 用户当前指令优先于本文件，本文件优先于阶段文档；真实源码、测试和机器证据决定实际完成状态。
- docs/路线0712-new.md 仅在核对秋招方向、项目定位或投递策略时读取。
- 根双语 README 是公开入口且必须事实一致；长命令和细节放入 docs/，README 不写成流水账或目录索引。
- docs/Proj1_S2.md 未经用户明确要求不得改写。

## 3. 大阶段二路线

大阶段二只包含五个完整单元，按顺序滚动交付：

1. **S2-01 — INT8 PTQ 与 ORT Profiling：**完成 static PTQ、FP32/INT8 对比和可分析 profile。
2. **S2-02 — Linux x86_64 与 AArch64/QEMU：**同一源码跨 Windows/Linux，并交叉编译 AArch64 做 QEMU portability smoke。
3. **S2-03 — 目录/Manifest 有界并发：**以 bounded queue、workers、backpressure、异常传播和 clean shutdown 复用单图 Pipeline。
4. **S2-04 — Linux x86_64 + RTX 4060 TensorRT：**形成真实 TensorRT 路径，先完成 correctness 与 FP16。
5. **S2-05 — 证据与 Recruiting Freeze：**完成跨平台回归、结果矩阵、简历、面试材料和用户 L2，随后冻结。

D010、Qt、LLM、Agent、真实板卡和其他新框架不进入当前主线；只有真实岗位需求、面试反馈或用户明确指令才能解冻。

## 4. 单元推进规则

- 一次只推进一个 S2-* 完整单元，不并行实现后续单元，也不重新拆成大量微阶段。
- 固定闭环为：**最小 SPEC → 实现 → 测试 → 机器可读证据 → 同步三份入口文档 → 停止等待用户 L1。**
- SPEC 先冻结目标、输入输出、接口、错误语义、证据协议、非目标和验收；先检查可发现事实。单元必须可运行、可测试、可解释。
- 正确性先于性能。正式 benchmark 必须先通过同次运行的 correctness gate，并固定 artifact、config、manifest、容差、输出模式、warmup/repeat 和机器环境。
- 每个单元完成后同步 AGENTS.md、README.md、README_zh.md 的状态、命令、证据、限制和下一步；三者未同步，单元不算完成。
- cpp_infer/README.md 仅在技术入口或事实变化时同步，不属于每次固定三文档。
- 单元结束后立即停下做 L1；S2-05 做 L2。冻结后仅接受正确性、复现、证据、测试、定向投递或真实面试反馈驱动的修改。
- Codex 负责实现、测试、证据、命令、文档和调试记录；用户负责形成可独立讲解、追问和修改的理解。

## 5. 架构与实现不变量

- 保持 C++17、CMake、OpenCV、ONNX Runtime C++、GTest 主技术栈；依赖必须显式、可配置、可诊断。
- Runtime library 负责配置、artifact、metadata、preprocess、backend、postprocess 和结果；CLI/workflow 只做参数、文件、退出码与证据编排。
- 所有 Demo、batch、benchmark 和新 backend 必须复用现有 DetectorPipeline 或其清晰抽象，不得复制 preprocess、decode、NMS 或坐标恢复逻辑。
- 配置权威分层不混用：RuntimeConfig 管运行选择和阈值，ModelArtifactSpec 管模型语义，ModelMetadata 是运行时观察到的事实。
- backend 只暴露模型无关的 owned I/O 与可行动错误；模型族逻辑留在 pre/postprocess，不把 YOLO 支持写成 D-FINE 通用兼容。
- Windows/Linux 共用业务源码，平台差异放薄适配层、CMake 或 workflow；以 RAII 和明确 ownership 管理 session、buffer、文件、线程和 profile。
- 多图复用单图产品链，首选最简单且可证明正确的 session 策略，不为极致吞吐引入不可解释复杂度。
- 错误包含失败位置、期望、实际和纠正动作，并返回稳定非零退出码；优化不得改变 score、NMS、类别和坐标语义。

## 6. 平台与证据口径

- **Windows x86_64：**当前已验证基线，使用 ORT CPU、Release/NMake、完整 correctness 与 benchmark gate。
- **WSL2/Linux x86_64：**S2-02 建立本地 Linux 证据；必须明确 WSL2，不冒充独立 Linux 设备。
- **Linux AArch64/QEMU：**只证明 cross-build、架构和功能可移植性；QEMU 下不发布延迟、吞吐、温度或真实板端性能。
- **Linux x86_64 + RTX 4060 Laptop：**S2-04 证明本地 GPU/edge-node TensorRT readiness；不得写成 Jetson、ARM64 GPU 或嵌入式实机部署。
- TensorRT 优先用 ORT TensorRT EP 和 CUDA fallback 复用产品链；仅在正确性、岗位要求或 timebox 充分时扩展 native backend。
- profiling 与正式 benchmark 分开运行；开启 profiler、verbose log 或 engine build 的耗时不能混入 steady-state 主结果。
- 证据记录 command、平台、编译器/build、artifact SHA、config、manifest、provider/precision、容差、原始输出和限制。
- 只有当前成功运行且产物存在的结果才是证据；历史、研究、临时和计划结果必须标注，不能改写为当前 Runtime 结果。
- 性能区分 pre/infer/post/pipeline，记录 mean/P50/P95、throughput 和 peak memory；PWS、RSS、GPU memory 不直接等同。

## 7. 核心路径、环境与避坑

- 主工程：cpp_infer/；公共头、实现、测试分别位于 include/、src/、tests/。
- Windows 统一入口：cpp_infer/tools/stage1.cmd；支持 help、doctor、build、clean-build、test、detect、demo、consistency、benchmark、all。无参数只显示 help。
- 核心契约：cpp_infer/configs/default_config.txt 与 cpp_infer/artifacts/yolov8_neu_det.artifact.txt。
- 固定 manifest：cpp_infer/tests/fixtures/consistency_manifest.json；正式结果在 cpp_infer/results/。
- 当前模型：models/best.onnx；运行时必须重新检查真实 metadata 和 SHA，不能只信文档记录。
- 已验证 Windows 依赖：MSVC/NMake/CMake/CTest，OpenCV C++ 4.8.0 x64 vc16，官方 ORT C++ SDK 1.19.2，GTest 1.17.0，以及 TestBase Python 中的 ORT 1.19.2/OpenCV/NumPy。
- 本机 ORT SDK 位于 D:\01_Base\Tools\onnxruntime-win-x64-1.19.2；通过 ONNXRUNTIME_ROOT、local config 或 CMake cache 注入，绝不提交个人绝对路径。
- Python onnxruntime wheel 不能替代 C++ headers、import library 和 runtime DLL；依赖缺失必须显式报错，不能静默安装。
- 路径优先级：显式参数 → Git 忽略的本机配置 → 环境变量 → portable fallback；机器无关默认值归 tracked defaults。
- 不使用 cpp_infer/build 中的旧二进制作证据；正式 Windows 构建使用受保护的 TEMP out-of-tree Release 目录，并在清理前验证绝对路径边界。
- stage1.cmd 负责发现 Visual Studio、调用 x64 VsDevCmd.bat 并启动无 profile PowerShell；手工诊断使用 where.exe。
- 每条 native 命令立即检查退出码和本次产物；旧 JSON 或另一次 correctness 不能替代当前 gate。
- 更完整的路径、命令和依赖规则集中在 docs/paths_commands.md；大阶段一最终证据集中在 docs/details/stage1_closure.md。

## 8. 工作区、安全与受保护范围

- 工作树可能已有用户修改。开始前检查 git status/diff；保留无关修改，禁止 reset、覆盖、顺手格式化或删除用户文件。
- 默认范围是 cpp_infer/、AGENTS.md、双语根 README 和当前单元明确命名的 docs。
- scripts/、src/、api/、Dockerfile、requirements*.txt、environment.yml、configs/、data/、models/、results/、runs/ 属于 legacy Python/训练/API/Docker 或数据产物范围；除非用户明确授权，不改写、不重构。
- 不修改仓库外文件，尤其不动 D:\01_Base\Obsidian 和 D:\01_Base\CodingSpace 下的兄弟项目。
- 纯文档任务不伪造新 Runtime 结果，只做结构、链接、双语事实、路径、陈旧状态和 diff 检查。
- 源码 MIT 不代表模型自动继承 MIT；best.onnx metadata 报告 AGPL-3.0，NEU-DET 条款待确认，公开发布前单独核查。
- D010 仅在稳定 ONNX、artifact/result card、部署契约、Runtime 集成和一致性通过后进入交付结果。
- best.pt 当前不在工作区或 Git 历史中，不得声称已重新完成 PyTorch/Python ORT/C++ 三方复现。

## 9. 验证与完成标准

- C++ 变更必须给出并实际运行相关 configure、build、test、run 命令；未成功运行时保持任务未完成并准确写明 blocker。
- 新功能覆盖正常、边界、无效输入、故障传播和回归；并发额外覆盖 backpressure、close/stop、异常与有限时间 join。
- 文档检查结构、链接、路径、双语事实、陈旧引用、git diff --check 和完整 diff；重要迁移先备份。
- 测试数量、研究指标或安装成功不等于产品链完成；验收必须对应可运行行为与机器可读输出。

## 10. 每个实现单元的九部分收口

除非用户明确要求简短状态答复，每个实现单元最终回复必须依次包含：

1. 本次高层完成了什么，实际状态和未完成项是什么？
2. 新增或修改了哪些模块，为什么这样设计；同时说明关键 trade-off、输入输出和异常语义。
3. 哪些文件变化、目录树如何变化、各文件在链路中承担什么职责？
4. 不使用 Codex 时，准确的人工实现流程是什么？
5. 入口函数、核心类/函数、输入输出、ownership，以及核心逻辑的宏观伪代码是什么？
6. 如何运行、测试、调试、调参、修改和定制；给出实际命令、证据路径和常见失败诊断。
7. 哪些验收问题和连续追问能证明达到面试理解深度？
8. 哪些代码最可能被追问并应进入代码练习；给出文件与当前行号。
9. 三份入口文档是否同步了状态、命令、证据、限制、路径和关键接口？

## 11. 口述答案格式

当用户说“给出口述答案”时，只输出可复制纯文本，使用“题号.回答”，答案间空一行；不复述问题，不加标题、项目符号、代码块、前言或总结，除非用户明确要求。
