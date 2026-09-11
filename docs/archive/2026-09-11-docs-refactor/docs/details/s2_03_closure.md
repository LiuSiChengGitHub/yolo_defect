# S2-03：目录/Manifest 多图有界并发最终教学收口

> 最终收口日期：2026-08-30  
> 当前状态：Windows x86_64、WSL2/Linux x86_64 与 Linux AArch64/QEMU 功能回归完成；Windows x86_64 与 WSL2/Linux x86_64 正确性、吞吐与内存比较完成；等待用户 L1；S2-04 未开始。

## 1. 讲解本步工作

### 1.1 五分钟口述

S2-03 解决的是现有单图 C++ Runtime 如何在不改变检测语义的前提下，变成一个可以处理目录或 manifest 的多图有界并发系统。单图链路已经稳定：配置与 artifact 决定模型语义，`DetectorPipeline` 完成图片解码、OpenCV 预处理、ONNX Runtime 推理、YOLO decode、NMS、坐标恢复和逐图结果写入。因此本单元没有复制推理代码，也没有做 true batch；并发单位始终是一张图片调用一次 batch=1 Pipeline。

入口先把目录或 manifest 变成确定性的 `BatchTask`。目录只递归接受七种支持扩展名的普通文件，不跟随符号链接，并按 UTF-8 generic 相对路径排序；manifest 是 UTF-8 路径清单，支持 BOM、LF/CRLF、空行和注释，每个有效相对路径从 manifest 所在目录解析，保留声明顺序。空输入、遍历失败、绝对 manifest 路径、缺失文件、不支持扩展名和重复 canonical 输入都会在 worker 启动前失败，所以一次运行面对的是冻结且可复现的任务集。

执行层使用 mutex、condition variable 和 deque 实现 `BoundedQueue<size_t>`。队列里只保存任务索引，不保存图片、tensor 或 ORT 输出；capacity 是严格上限。生产者遇到满队列会阻塞，这就是 backpressure，它防止发现速度长期高于推理与写盘速度时内存无限增长。摘要记录 capacity、峰值深度、生产者等待次数和等待时长，所以“有界”不仅是代码声明，也是可观察契约。

每个 worker 在任何逐图任务开始前顺序创建一个独占的 `DetectorPipeline`，也就是一个独立 ORT session。session 使用 sequential execution，intra-op 与 inter-op 都固定为 1/1，让外层图片级并发是唯一可解释的并发变量。如果任意 session 初始化失败，系统生成 fatal 摘要并取消全部任务，不会处理半批图片。创建完成后 worker 等待同一个 start gate，计时开始才统一放行；生产者按发现顺序入队，worker 可以乱序完成，但结果数组按 `sequence_index` 固定，因此逐图文件名和最终 summary 永远保持发现顺序。

失败是按异常的捕获边界分级，不是按异常类型猜测：`executor->run()` 的逐图 try scope 内，包括解码、推理和写入，任何异常只把当前项标成 failed，其他图片继续；session/executor 初始化、线程创建、producer，或逃出 worker 外层循环的异常才进入 fatal 并触发全局 stop。正常 close 会让 worker drain 已入队任务，stop 则拒绝新任务、清除尚未开始的索引、唤醒阻塞生产者与消费者，已经进入同步 ORT 调用的任务允许自然完成，最后 join 全部线程。SIGINT、SIGTERM 和 Windows console break 的 handler 只写 `sig_atomic_t` 标志，普通监控线程再调用 `request_stop()`，避免在异步信号上下文中加锁、写文件或终止 ORT。

每个成功任务固定写 `items/000000.detections.json` 这样的原单图 schema；可选 `--output-images` 再写 PNG。session 初始化前的预检会验证输出根目录与 `items` 目录是真实目录，并检查所有计划路径的冲突、protected paths、summary overwrite 和输入/输出包含关系；逐图 writer 在实际写入时再检查具体目标对象，已存在的 symlink/reparse/special object 会成为该 item 的失败。这样既避免覆盖输入，也避免把输出递归吸回目录输入。Windows 图片读取改成 `filesystem::path + ifstream + cv::imdecode`，从而让 UTF-8 manifest 中的中文文件名走完整真实 CLI 链路，而不是退回窄字符 `imread`。

机器接口是 schema v1 `BatchSummary`。它记录命令、编译目标架构、运行内核架构、执行上下文、ORT/OpenCV、模型/provider、发现协议、输出策略、requested/effective workers、session 初始化、1/1 线程策略、queue/backpressure、计数、时间、吞吐、PWS/RSS、限制和按发现顺序排列的逐图结果。`cooperative_stop_requested` 与 item 计数独立：中断时即使所有任务都已开始、没有任何 item 保持 cancelled，整批仍是 cancelled/130。计数必须满足 `discovered = succeeded + failed + cancelled` 与 `started = succeeded + failed`；状态与退出码是 succeeded/0、partial_failure/2、cancelled/130、fatal/1。吞吐只用成功图片数除以 processing wall time，这段计时包含入队等待、decode、preprocess、ORT、postprocess、逐图写入和 join，排除发现、session 构造和 summary 序列化。

验证分三层。单元测试覆盖发现、manifest、UTF-8、路径安全、bounded queue 的 FIFO/backpressure/close/stop、恰好执行一次、乱序完成有序汇总、逐图失败、初始化 fatal、取消与 join、summary 不变量和 overwrite；真实 ORT 集成覆盖目录与 manifest、单线程与原单图完全一致、单线程与多线程逐图 JSON 一致、中文路径、损坏 JPEG 的精确部分失败，以及子进程中断退出 130。Windows 与 WSL2/Linux x86_64 都完成 clean Release 和 156 个 CTest，361 图 worker=1/4 用独立进程比较，361 份 JSON 在各平台内全部字节与语义一致。Windows 吞吐从 6.285556 提升到 17.853923 img/s，2.840468 倍，PWS 从 151.804688 增至 505.085938 MiB；WSL2/Linux 在同一原生 ext4 工作区从 8.113806 提升到 20.159584 img/s，2.484603 倍，RSS 从 205.765625 增至 588.226563 MiB。性能变快不是通过门槛，内存增加也不是 bug，而是多个独立 session 的明确代价。

Linux AArch64 则完成同一 Runtime/CLI 交叉构建、AArch64 ELF/loader 检查、固定图 3 检测，以及 QEMU user-mode 下目录 worker=1、manifest worker=2、JSON 一致和 2 成功 + 1 失败。摘要同时记录 target=`aarch64`、host kernel=`x86_64`、context=`qemu_user_mode_on_x86_64_host`，并把内存标成 `publishable:false`。它证明三平台都有功能，却不证明真实 ARM 板卡、原生 ARM 性能、功耗或长期稳定性。S2-03 的实现和工程验证至此完成；用户 L1 尚未完成，S2-04 不会自动开始。

### 1.2 教学级完整讲解

#### 路线位置、输入输出与边界

S2-03 位于 S2-02 跨平台之后、S2-04 TensorRT 之前。前置状态是同一 `DetectorPipeline` 已在 Windows、Linux x86_64 与 AArch64/QEMU 完成单图 CPU 推理。问题不再是“能否检测一张图”，而是“面对一个可重复定义的图片集合，如何限制资源、利用 CPU 并发、隔离单图失败、在中断时留下完整机器证据”。输入是已验证的 Runtime config、artifact、模型，以及目录或 UTF-8 manifest；输出是每个成功图片的原 schema Detection JSON、可选 PNG 和一个 BatchSummary。

非目标必须与实现边界一起理解：没有把多张图片拼成 ONNX tensor batch，没有改变 score/NMS/类别/坐标语义，没有视频、服务、GPU 并发、无锁结构，也没有共享同一个 ORT session。这样能够把学习重点放在任务发现、ownership、有界队列、backpressure、线程生命周期和证据协议上，避免同时引入 tensor shape、动态 batch、GPU stream 或服务请求语义。

#### 端到端控制流

1. CLI 解析 batch 模式并拒绝与单图、inspect、raw summary、benchmark、profile 混用；默认 worker=1，默认 queue=`2*workers`。
2. `discover_batch_tasks()` 对目录或 manifest 做完整发现和校验，得到按协议排序的 `BatchTask`。这一阶段不创建 session，也不写输出。
3. `validate_request_and_outputs()` canonicalize 输入和输出规划，检查输出不覆盖任何受保护输入、目录输出不位于目录输入内部、summary overwrite 规则、`items` containment，以及输出根/`items` 目录的 symlink、reparse point 或 special-object 风险；单个逐图目标的对象状态由 writer 在写入时再验证。
4. `BatchRunner` 初始化 ordered `BatchItemResult` 数组，默认全部 cancelled；再顺序创建 effective worker 数量的 executor/Pipeline/session，校验每个 session 的 provider、线程与模型 metadata 一致。
5. 创建只保存 `size_t` 的有界队列和 worker 线程。所有 worker 在 start gate 等待；计时开始后同时放行。
6. 生产者按 task 顺序 `push(index)`。队列满时条件变量阻塞生产者；消费者 `pop()` 后使用自己独占的 Pipeline 执行一张图片并写对应序号文件。
7. worker 完成顺序不决定结果顺序：每个 worker只写自己取得的唯一 index 槽位。`executor->run()` 的内层 task scope 捕获任何逐图异常并写 failed；初始化、producer、thread creation 或逃出 worker 外层 scope 的异常设置 fatal 并 stop queue。
8. 正常生产完毕调用 close 让消费者 drain；中断或 fatal 调用 stop 清空未开始索引。主线程 join worker，计算 processing wall、queue 统计、计数、成功项 latency、throughput 和 peak process memory。
9. 严格校验 summary 不变量，CLI 将它序列化并按最终状态返回 0/2/130/1。

#### 为什么是有界阻塞队列

无界任务队列会把快生产者和慢消费者之间的速度差变成持续增长的内存；若队列持有解码图片或 tensor，增长更严重。这里队列只保存索引，capacity 又是硬上限，因此排队中的重量工作量最多是 `capacity` 个小整数；图片/tensor/ORT output 只属于正在运行的 worker 当前栈/堆对象。需要准确区分的是：确定性发现所需的 task metadata、固定结果槽和最终 summary 仍按图片数呈 `O(N)`，有界队列约束的是排队深度与重量中间数据，而不是把整个批次的所有元数据变成常量空间。满时阻塞生产者就是 backpressure：系统主动把上游速度压到下游可承受水平。mutex/CV 版本比无锁队列更容易证明 FIFO、close、stop 和 wakeup，不需要为本任务引入 ABA、内存序或 reclamation 难题。

#### worker、session 与 ownership

`DetectorPipeline` 内部持有 ONNX runner/session，因此把 Pipeline 在线程间共享会引入线程安全、provider 行为和内部线程池的额外变量。本实现让每个 worker 独占一个 Pipeline/session，任务内的 `cv::Mat`、NCHW tensor、ORT owned output、detections 和写盘请求也只活在该 worker 当前调用。共享状态被缩到 queue、固定长度 result slots、停止标志和少量统计。不同 worker只写不同 index，因此不需要在结果写入周围加一把全局锁。

代价是内存随 session 数明显上升；正式结果正好证实这一点。选择 ORT sequential + intra/inter 1/1，是为了不让“4 个外层 worker × 每个 session 多个 ORT 内线程”造成嵌套过度订阅。当前吞吐提升来自多张独立图片的重叠执行，不是模型图内部或 tensor batch 并行。

#### 确定性与检测语义

并发系统的确定性不等于线程完成顺序固定。这里固定的是输入任务序列、每个序号的源图、输出文件名和汇总排列；worker 调度可以变化。目录使用 relative path 的 UTF-8 generic form 排序，避免依赖 filesystem iterator 顺序或平台分隔符；manifest 本身就是用户声明的顺序。每张图仍调用原 `DetectorPipeline::run()`，所以 preprocess、ORT、decode、NMS 与坐标恢复没有第二份实现。comparison 工具不仅比较 summary 顺序，还逐个比较 Detection JSON 字节和解析后的语义，从证据上关闭“并发偷偷改变结果”的风险。

#### 失败、取消与退出码

这里的严格规则是“捕获 scope”：损坏图片、推理失败或目标逐图文件已存在且未 overwrite，都会在 `executor->run()` 内层 try scope 内被捕获，因此只标记该 item failed 并继续，最终可得到 `partial_failure`/2。session/executor 初始化、线程创建、producer，或逃出 worker 外层循环的异常影响整个执行可信度，属于 fatal/1。用户中断属于 cancelled/130：未开始项保持 cancelled，已开始项完成或失败，计数仍平衡；如果所有项都已开始，`cooperative_stop_requested=true` 仍使状态保持 cancelled，而不是误报 succeeded。

signal handler 不能安全调用 mutex、condition variable、iostream 或复杂 C++ 对象，所以只写异步信号安全的 `sig_atomic_t`。普通 monitor thread每 10 ms 检查标志并调用 `request_stop()`；main 在 join monitor 前后还会做一次最终握手，避免信号已置位但 monitor 仍在 sleep 时遗失中断。stop 不强杀同步 `Ort::Session::Run`，因为强行终止可能破坏 provider/session 状态或留下半写文件；它只停止新工作并等待已开始任务到达清晰边界。

#### 输出安全与 UTF-8

在启动 session 前计算全部 summary/JSON/PNG 目标，并把源图、config、artifact、model、manifest 视为 protected paths。文本相等之外还在存在的文件上检查 filesystem identity；Windows 的 location comparator使用 `CompareStringOrdinal` 的不区分大小写模式。目录输入禁止输出根位于输入树内，否则下次递归可能把生成 PNG 当作输入。输出根和 `items` 目录若是 symlink/reparse point、special object，或 containment 校验后位于预期根之外，会前置拒绝；具体的逐图文件目标则由 `ResultWriter` 在写入时重新检查，已存在的 symlink/reparse/special object 只会让该 item 失败。

OpenCV Windows 的 narrow `imread(path.string())` 不能可靠打开 Unicode 文件名，所以 decoder 改为 `ifstream(filesystem::path)` 读 owned bytes，再 `cv::imdecode(..., IMREAD_COLOR)`。这保留 OpenCV 解码语义，同时让 Windows wide filesystem path 真正贯穿 manifest 到图片读取。单元测试和真实 CLI 都覆盖了 Unicode 文件名。

#### BatchSummary 与性能口径

BatchSummary 不是控制台日志的翻版，而是稳定机器契约。schema validator检查必需字段、有限数值、状态、路径、items 顺序、provider/线程设置和计数恒等式；状态优先级是 fatal error > cooperative stop/未开始项 > item failure > 全成功。target architecture 与 runtime kernel architecture 分开记录，才能表达“AArch64 ELF 在 x86_64 kernel 的 QEMU user-mode 进程里运行”。

processing wall 从 worker 已创建并在 gate 就绪后开始，到所有 worker join 后结束。它包含生产者等待、推理和逐图 JSON 写入，因此回答的是一次批处理执行阶段的实际吞吐；发现和 session 构造单独排除，避免任务集大小或多 session 冷启动掩盖 steady processing。另一方面 peak process memory 是进程生命周期高水位，仍包含发现、session 构造和保留的结果数组，所以它适合比较整个进程的资源代价，却不能被解释成“队列本身用了多少内存”。只有平台内存查询实际成功且运行不是仿真上下文时，`memory.publishable` 才可为 true；查询失败必须是 false 并保留 reason。Windows PWS 与 Linux RSS 来源不同，只做各自平台内部差值。

#### 验证因果链与最终结论

纯并发测试先用可注入 executor把 ORT 成本拿掉，精确制造阻塞、乱序、逐图异常、初始化失败和取消时点；这样可以证明生命周期，而不是靠真实模型偶然触发竞态。真实 CLI 集成再证明这些抽象最终确实复用 ORT/Pipeline、写出兼容 schema，并能在子进程信号下退出。最后 361 图、独立进程、固定 queue 与输出策略给出正确性、吞吐和 peak memory trade-off。

Windows 与 WSL2/Linux 的正式比较都出现显著吞吐提升和更高内存，但测试没有规定并发必须更快，因为具体 CPU、内存带宽、ORT kernel 和文件系统可能让多 session 竞争资源。AArch64/QEMU 只跑小功能集，因为模拟器时间和 RSS不能预测板卡。完成判定是：实现、三平台功能回归、两平台正式比较、机器摘要、工作流和入口文档同步均完成；它不等于用户 L1、真实 ARM 设备验收或 S2-04 GPU 工作完成。

## 2. 新增或修改的模块与设计原因

| 模块 | 输入 | 输出 | 设计、trade-off 与异常语义 |
|---|---|---|---|
| batch contracts | request/task/item/status/environment | `BatchSummary` v1 | 用 typed C++ 结构承载状态；不增加 C++ JSON parser |
| deterministic discovery | directory 或 UTF-8 manifest | ordered `vector<BatchTask>` | 目录排序、manifest 保序；任一声明错误整体前置失败 |
| `BoundedQueue<size_t>` | task index、capacity | FIFO pop、queue metrics | mutex/CV 易证明生命周期；满时阻塞形成 backpressure |
| executor seam | Runtime contract、worker index | 每 worker executor | 生产使用 Pipeline，测试注入 fake 精确控制竞态 |
| `BatchRunner` | frozen tasks、worker/queue/output policy | ordered items + summary | session 前置初始化；内层 task scope 失败隔离，外层 lifecycle scope 异常 stop |
| path safety | planned paths + protected inputs | validated canonical plan | 前置防覆盖/递归污染和目录逃逸；writer-time 拒绝单项异常目标 |
| summary writer/validator | `BatchSummary` | schema v1 JSON | 有限数值、计数和独立 stop-request 状态严格校验；summary overwrite 前置处理 |
| CLI signal adapter | SIGINT/SIGTERM/SIGBREAK | cooperative stop | handler只设标志；普通线程执行锁和对象操作 |
| UTF-8 decoder | `filesystem::path` | owned BGR `cv::Mat` | byte read + imdecode解决 Windows Unicode 路径窄字符问题 |
| compare/validate tools | 两次 summary/items | comparison JSON/失败退出码 | 标准库 Python；正确性先于性能差值，不设 speedup gate |
| stage workflows | action/env paths | build/test/evidence | Windows/Linux正式比较；AArch64只做功能验收 |

## 3. 文件变化与目录职责

```text
cpp_infer/
├── include/yolo_defect_cpp/
│   ├── batch_result.h              # task/item/status/summary contracts
│   ├── batch_runner.h              # BatchRequest、discovery、runner API
│   ├── batch_writer.h              # summary JSON API
│   ├── detector_pipeline.h         # 原 run 不变；只读 metadata/init timing
│   └── image_preprocessor.h        # filesystem::path 输入
├── src/
│   ├── batch_discovery.cpp         # 目录/manifest 确定性发现
│   ├── bounded_queue.h             # FIFO bounded queue 与 backpressure
│   ├── batch_executor.h            # 可注入执行器 seam
│   ├── batch_path_safety.h         # 跨翻译单元 path safety helper
│   ├── batch_runner.cpp            # session/worker/queue/shutdown 编排
│   ├── batch_writer.cpp            # schema 校验与 JSON 序列化
│   ├── image_decoder.cpp           # path-safe byte decode
│   └── main.cpp                    # batch CLI、signal monitor、退出码
├── tests/
│   ├── batch_test.cpp              # 发现、队列、runner、summary 单元测试
│   ├── image_decoder_test.cpp      # Unicode filesystem path
│   ├── assert_batch_cli.py         # 真实 ORT + shutdown 集成
│   ├── test_s2_03_batch_tools.py   # Python tool contracts
│   └── fixtures/s2_03_consistency_manifest.txt
├── tools/
│   ├── validate_batch_summary.py   # 严格机器摘要 validator
│   ├── compare_batch_runs.py       # 正确性/吞吐/内存同平台比较
│   ├── stage1.ps1 / stage1.sh      # batch 与 batch-compare native入口
│   └── stage2_aarch64.sh           # AArch64/QEMU batch acceptance
└── results/s2_03/
    ├── windows_x86_64/             # PWS + 361图 comparison
    ├── linux_x86_64/               # WSL2 RSS + 361图 comparison
    └── linux_aarch64_qemu/         # ELF/loader/功能摘要，性能不可发布
```

根 `AGENTS.md`、`README.md`、`README_zh.md`、`cpp_infer/README.md` 与 `docs/paths_commands.md` 同步当前状态、命令和证据；`.gitignore` 只排除体量大的逐图 `items/` 与临时 `inputs/`，保留 summary、comparison 和关键回归结果。

## 4. 不使用 Codex 时的人工实现流程

1. 先冻结单图 Detection JSON，并建立“每个并发任务必须调用原 `DetectorPipeline::run()`”的不变量。
2. 定义 request/task/item/summary typed contracts、状态和退出码，再写 summary count invariants。
3. 实现目录 discovery：regular file、扩展名白名单、不跟随 symlink、UTF-8 generic relative sort、canonical duplicate。
4. 实现 manifest parser：binary read、UTF-8/BOM/CRLF、ASCII trim、blank/comment、relative resolution、声明顺序和整体验证。
5. 写 `BoundedQueue<size_t>`，先用单元测试证明 capacity、FIFO、producer wait、close drain、stop clear/wakeup/refuse。
6. 抽象 `BatchTaskExecutor`，生产 factory 包装 Pipeline，测试 factory可制造延迟、失败和初始化异常。
7. 在 `BatchRunner` 中先发现和输出预检，再顺序创建所有 session；随后 start gate、生产、消费、close/stop、join、统计。
8. 让每个 result slot 由唯一 task index写入；用固定数组保持发现顺序，而不是按完成顺序 append。
9. 添加 path protection、summary/逐图 overwrite 语义、directory input/output containment 和 symlink/reparse 防护。
10. 添加信号最小 handler和普通 stop monitor；写子进程 smoke验证 exit 130、已开始完成、未开始 cancelled，再用可控 fake executor 验证“全部已开始、逐图 cancelled=0”时仍返回 cancelled/130。
11. 修改 Windows decoder，用 filesystem path读取字节并 imdecode；加 Unicode unit与真实 manifest integration。
12. 写严格 summary validator和两运行 comparison 工具；先检查 item/order/JSON 一致，再计算吞吐与内存差。
13. 扩展 CMake/Threads、Windows/Linux workflow和 AArch64/QEMU acceptance。
14. 先跑局部单元与集成，再跑 Windows/Linux clean full CTest、361图独立进程比较和 AArch64 cross/ELF/loader/QEMU 功能门。
15. 保存小而关键的机器证据，忽略大量逐图输出；同步入口文档并停止等待 L1。

## 5. 入口、核心接口、ownership 与伪代码

主要入口：`src/main.cpp` 的 batch 分支；契约在 `batch_result.h` 与 `batch_runner.h`；发现位于 `batch_discovery.cpp`；队列位于 `bounded_queue.h`；生命周期核心位于 `batch_runner.cpp`；机器 JSON位于 `batch_writer.cpp`；原业务链仍位于 `detector_pipeline.cpp`。

Ownership：CLI 拥有 `RuntimeContract` 与 `BatchRunner`；runner拥有 factory和活动 queue引用；运行期局部容器拥有所有 executor、worker thread、task和result；每个 executor独占一个 Pipeline/session；worker当前调用独占图片/tensor/ORT output/detections；queue只拥有 index；最终 summary按值返回给 CLI 并写盘。

```cpp
contract = load_runtime_contract(config)
request = parse_batch_cli(argv)
runner = BatchRunner(contract)

tasks = discover_batch_tasks(request.input_kind, request.input_path)
paths = validate_all_planned_outputs(tasks, contract, request)
summary.items = cancelled_slots_in_discovery_order(tasks)

for worker in effective_workers:
    executors.push_back(create_pipeline_and_session(contract))
validate_same_metadata(executors)       // any failure => fatal, no item starts

queue = BoundedQueue<size_t>(capacity)
start workers; wait until all ready
start processing_wall_timer

producer:
    for task in tasks:
        if stop or !queue.push(task.index): break   // full => block/backpressure
    stop ? queue.request_stop() : queue.close()

worker[i]:
    try:                                        // outer lifecycle scope
        while index = queue.pop():
            try:                                // inner per-task scope
                summary.items[index] = executors[i].run_and_write(paths[index])
            catch any task-scope exception:
                summary.items[index] = failed(error)
    catch exception escaping the worker loop:
        record_fatal_once(); queue.request_stop()

thread creation or producer exception:
    record_fatal_once(); queue.request_stop()

join_all_workers()
record_cooperative_stop_request_independently_of_item_counts()
derive_counts_status_latency_throughput_memory()
validate_batch_summary(summary)
write_summary(summary)
return status_to_exit_code(summary.status)
```

## 6. 运行、测试、调试、调参与证据

### CLI

```text
yolo_defect_cpp --config <config> --batch \
  (--input-dir <dir> | --manifest <file>) \
  --output-dir <dir> --batch-summary <file> \
  [--workers <1..64>] [--queue-capacity <1..4096>] \
  [--output-images] [--overwrite]
```

默认 worker=1，默认 queue=`2*workers`；任务少时 effective worker=`min(requested, discovered)`。成功、部分失败、取消、fatal 分别返回 0、2、130、1。

推荐从工作流入口运行：Windows 使用 `cpp_infer\tools\stage1.cmd batch ...` 与 `cpp_infer\tools\stage1.cmd batch-compare`；WSL2/Linux 使用 `bash cpp_infer/tools/stage1.sh batch ...` 与 `bash cpp_infer/tools/stage1.sh batch-compare`；AArch64 功能门使用 `bash cpp_infer/tools/stage2_aarch64.sh batch` 或 `all`。机器本地依赖路径与完整参数以 `docs/paths_commands.md` 为准。

调参时一次只改变 worker，正式比较保持 queue、模型、config、provider、输入和输出策略不变。queue太小会增加 producer wait但限制更紧；queue再大也不会提高正在执行的 worker数。worker过多会增加 session内存并可能争用 CPU/内存带宽。部分失败先看 `items[*].error`；fatal看 `fatal_error`；summary未生成通常说明 CLI/discovery/preflight本身失败。中断先看 `cooperative_stop_requested=true` 与 exit 130，再看有多少 item 实际保持 cancelled，不要把“已全部开始”误判为没有收到 stop。

正式证据：

- Windows：`cpp_infer/results/s2_03/windows_x86_64/{verification.md,comparison.json,workers_1/batch_summary.json,workers_4/batch_summary.json}`。
- WSL2/Linux：`cpp_infer/results/s2_03/linux_x86_64/{verification.md,performance/batch_comparison.json,performance/batch_workers_*/batch_summary.json}`。
- AArch64/QEMU：`cpp_infer/results/s2_03/linux_aarch64_qemu/{verification.md,regression/*,final_20260830_r2/*}`。

## 7. 面试验收问题与连续追问

1. 为什么 queue只放 index？如果放 decoded image，capacity仍为 8是否足够解释内存上限？
2. bounded queue怎样区分 close与stop？各自对阻塞 producer/consumer和已入队任务有什么影响？
3. 为什么每 worker独占 session？为什么 ORT内部固定 sequential 1/1？如果共享 session或开启 ORT多线程会怎样？
4. worker乱序完成时，如何保证输出和 summary顺序确定？是否存在两个线程写同一 result slot？
5. 为什么损坏 JPEG 是 partial failure，而 session初始化失败是 fatal？计数与退出码如何推导？
6. 为什么 signal handler不能直接调用 `request_stop()`？已进入 `Session::Run` 的任务为何不强杀？
7. processing wall为什么排除 discovery/session构造，却包含 queue wait/JSON/join？peak memory为何采用不同范围？
8. 如何证明并发没有改变 score、NMS、类别和坐标？字节相等和语义相等各解决什么问题？
9. 目录排序为什么使用 relative generic UTF-8 form？manifest为什么保留声明顺序而不排序？
10. canonical duplicate、Windows大小写、symlink/reparse和filesystem equivalent之间是什么关系？
11. 为什么 Windows Unicode路径要改成 byte read + imdecode？这是否改变颜色通道或解码模式？
12. worker=4为什么吞吐提升却内存大幅增加？什么时候更多 worker会更慢？
13. 为什么 Windows PWS不能直接与Linux RSS比较？为什么QEMU数据必须 `publishable:false`？
14. 这个系统与 true tensor batch有什么本质不同？下一步若做GPU并发会新增哪些ownership和同步问题？

## 8. 应进入代码练习的高概率追问代码

最终练习应覆盖：`BoundedQueue::push/pop/close/request_stop`，目录/manifest discovery，`validate_request_and_outputs`，`BatchRunner::Impl::run` 的 session/start gate/worker/producer/join，`derive_counts_and_status`，`validate_batch_summary` 与 serializer，CLI signal adapter和batch exit mapping，Unicode decoder，以及 Python comparison/validator。当前准确行号以最终工作树的第 8 部分对话输出和 `rg -n` 结果为准，避免文档后续插入造成静态行号漂移。

练习目标不是背代码，而是能在白板上写出状态转换和伪代码，并解释每个共享对象由谁拥有、哪个线程读写、何时唤醒、如何停止、哪些异常局部化、哪些异常升级为全局 fatal。

## 9. 入口文档同步核对

- `AGENTS.md`：S2-03 完成状态、三平台证据等级、156-test基线、性能边界、等待 L1 与 S2-04未开始。
- `README.md`：英文能力入口、CLI/证据、平台覆盖、性能与 QEMU限制。
- `README_zh.md`：与英文 README 同步的中文事实。
- `cpp_infer/README.md`：Runtime使用、batch schema/退出码、工作流、调试和证据路径。
- `docs/paths_commands.md`：Windows/Linux batch与 batch-compare、AArch64/QEMU acceptance和本机依赖入口。

本收口不修改 `docs/Proj1_S2.md`。S2-03 的工程实现与真实验证已经完成；下一动作是用户 L1，而不是自动进入 S2-04。
