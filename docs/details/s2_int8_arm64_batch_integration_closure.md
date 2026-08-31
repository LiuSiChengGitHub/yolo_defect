## S2-01、S2-02、S2-03 融合收口：九部分输出

### 1. 讲解本步工作

#### 五分钟口述

这一步不是重新写一套 INT8 推理，也不是把三个单元简单放在同一份 README 里，而是检查它们是否真的共享同一条产品链。检查结果是：核心 C++ Runtime 其实已经具备融合条件。正式 Round 2 模型是 QDQ/U8S8 ONNX，量化发生在图内部，模型外部输入和输出仍是 float32，所以原来的 OpenCV preprocess、`OnnxRunner`、YOLO decode、NMS 和坐标恢复都可以继续复用。`DetectorPipeline` 不需要知道当前内部 Conv 是 FP32 还是 INT8，只需要通过 `RuntimeConfig -> ModelArtifactSpec` 选到正确模型，并由 ONNX Runtime 创建合法 session。

原先真正割裂的是外围入口和证据：S2-01 的正式 INT8 主要停留在 Windows 量化、profiling 和单图/质量实验；S2-02 的 Linux x86_64、AArch64/QEMU 工作流固定选择 FP32；S2-03 的目录、manifest、并发与 BatchSummary 也只有 FP32 正式证据。甚至单图 JSON 校验器把 FP32 model ID 和 SHA 写死，因此第一次在 Linux 包装脚本里跑 INT8 时，C++ 推理已经成功并输出 3 个检测，随后却被外围校验误判失败。

最小整合方案因此集中在“选择与验证”层，而不是 Runtime 核心：让 Windows/Linux `batch-compare` 接受显式 config；让 AArch64 工作流接受 `YOLO_DEFECT_AARCH64_CONFIG`，默认值仍为 FP32；让单图 JSON 校验从所选 RuntimeConfig 追到 ModelArtifactSpec，再核对 model ID 与 SHA；在 CTest 中复用既有 batch CLI 集成脚本，为正式 U8S8 模型注册一份独立测试，并显式绑定模型身份。多图仍由原 `BatchRunner` 完成，每个 worker 独占一个 `DetectorPipeline` 和 ORT session，所有任务经过有界队列，逐图 JSON、部分失败、退出码和 BatchSummary schema 都没有分叉。

最终形成的实际链路是：正式 U8S8 artifact 和声明文件进入 `load_runtime_contract`；`OnnxRunner` 校验 ORT 实际观察到的 float32 外部 I/O；单图或多图都进入同一个 `DetectorPipeline`；多图输入经过确定性目录发现或 manifest 顺序解析，再由有界队列分发到独占 session 的 workers；成功项写逐图结果，失败项只记录本项错误，线程汇合后生成有序 BatchSummary。相同业务源码先在 Windows 和 WSL2/Linux x86_64 验证，再交叉编译成 AArch64 ELF，在 QEMU user mode 下执行相同 INT8 单图和多图功能链。

Linux x86_64 的正式 INT8 验证中，30 图 manifest 以 workers=2、queue=4 得到 30/30 成功，队列峰值 4、生产者等待 25 次；固定单图与 manifest 对应项 JSON 字节和语义一致。361 图 workers=1 与 workers=4 都是 361/361 成功，361 份逐图 JSON 全部字节与语义一致；吞吐分别为 4.591151 和 15.903088 img/s，比例 3.463857，peak RSS 分别为 192.933594 和 556.882812 MiB。这说明用更多独立 session 换取了本次运行的更高吞吐，也带来了明显内存成本。它是 WSL2/Linux、仓库位于 `/mnt/d` DrvFs 的同协议内比较，不能直接拿去和历史 ext4 FP32 数字做模型速度结论。

AArch64 侧完成了 clean cross-build、ELF machine、loader 和目标动态库检查。QEMU 下 INT8 固定图得到 3 个检测；目录 worker=1 和 manifest worker=2 各 2/2 成功且逐图结果完全一致；损坏 JPEG 场景得到 2 成功、1 失败、退出码 2，queue=1 且出现一次生产者等待。这里证明的是 AArch64 构建、加载和功能可移植性，不证明原生 ARM64 性能、内存、功耗、温度或长期稳定性。默认 FP32 也在 Linux Demo 和 AArch64/QEMU `all` 工作流中重新通过，因此这次融合没有用 INT8 替换或破坏 FP32 主链。

#### 教学级完整讲解

##### 为什么 QDQ/U8S8 可以复用原 Pipeline

QDQ 模型会在 ONNX 图中显式出现 QuantizeLinear/DequantizeLinear。U8S8 表示激活采用 uint8、权重采用 int8，但本项目正式 artifact 的外部张量契约仍是：输入 `float32 [1,3,800,800]`，输出 `float32 [1,10,13125]`。因此：

1. OpenCV 仍把图片 letterbox、BGR 转 RGB、归一化并排成 NCHW float32；
2. `OnnxRunner` 仍用原 owned float buffer 调用 `Ort::Session::Run`；
3. ORT 在图内部选择 INT8 kernel，并在需要处做 Q/DQ；
4. C++ 后处理仍收到同 shape、同 dtype 的 YOLO raw output；
5. score、class-agnostic NMS 和坐标恢复语义不变。

所以正确抽象不是增加 `Int8DetectorPipeline`，而是把“模型实现”留在 artifact/config 和 ORT graph 中。若未来模型外部 I/O 改成 uint8/int8，才需要显式扩展 preprocess/backend tensor 契约。

##### 为什么每 worker 一个 Session

S2-03 做的是 concurrent batch=1，不是真正把 N 张图片拼成一个 batch tensor。每个 worker 长期持有一个 `DetectorPipeline`，而 pipeline 独占 `OnnxRunner/Ort::Session`。这样做的好处是 ownership 清楚，不需要在多个线程之间围绕一个 session 增加业务锁；代价是 worker 数量增加时，session、优化图和工作 buffer 都会增加，因此 peak RSS 上升。本次 workers=4 比 workers=1 多约 381.63 MB peak RSS，正好体现了这个 trade-off。

##### 有界队列和背压如何工作

生产者按确定性任务顺序把 `sequence_index` 推入固定容量队列。队列已满时，`push` 在 condition variable 上等待；worker `pop` 一个任务后唤醒生产者。这样未处理任务数不会随着输入规模无限增长。`queue_peak_depth == capacity` 和 `producer_wait_count > 0` 是背压实际发生的证据：30 图运行是 peak 4、wait 25；361 图单/四 worker 分别 wait 353/350。

并发完成顺序可能变化，但每个结果写回以 `sequence_index` 预先创建的位置，逐图文件名也是 `000000`、`000001` 等稳定序号；最终 BatchSummary 按发现顺序保存 items。因此并发没有改变外部确定性。

##### 部分失败为何不拖垮整批

图片解码、推理或写输出的普通异常在单个 worker 的任务级 try/catch 内转换成 `BatchItemStatus::kFailed`，worker 随后继续取下一项。只有 worker 基础设施、线程创建或 session 初始化失败才升级为 fatal 并请求队列停止。这样损坏 JPEG 可以得到 `partial_failure`、2 成功、1 失败和退出码 2，而不是丢失已成功结果或伪装成全成功。

### 2. 模块、设计、取舍、输入输出和错误

| 模块 | 输入 | 输出 | 这次融合中的职责 |
|---|---|---|---|
| RuntimeConfig | config path、threshold、provider、artifact path | `RuntimeContract.runtime` | 在 FP32 与正式 INT8 间做运行选择 |
| ModelArtifactSpec | model ID、model path、SHA、I/O/前后处理声明 | `RuntimeContract.artifact` | 冻结 U8S8 身份和 float32 外部 I/O 契约 |
| OnnxRunner | contract、NCHW float tensor | owned raw output、actual metadata | 创建 ORT CPU session，校验实际模型元数据，执行图内 INT8 |
| DetectorPipeline | 单图路径、输出请求 | detection result、JSON/PNG | 唯一 preprocess/inference/postprocess 产品链 |
| batch discovery | 目录或 manifest | 有序 `BatchTask[]` | 目录递归排序或保留 manifest 声明顺序 |
| BoundedQueue | sequence index、capacity | FIFO work stream、wait statistics | 限制在途任务并实现背压 |
| BatchRunner | contract、tasks、workers、queue | ordered item results、BatchSummary | 每 worker 独占 pipeline/session，隔离普通单项失败 |
| writers/validators | detection result、BatchSummary | 稳定 JSON、验证结果 | 绑定模型身份、结果和平台证据 |

关键取舍：

- 复用 concurrent batch=1，而不实现 true tensor batch；代码和检测语义稳定，但 session 数随 worker 增长。
- config override 是显式 opt-in，FP32 保持默认；兼容原入口，但正式 INT8 验证必须明确传 config。
- QEMU summary 仍收集运行字段，但 `memory.publishable=false`；保留机器事实，同时禁止错误地当作原生 ARM 性能。
- 性能比较不设置 speedup 硬门；correctness 必须一致，吞吐和内存只描述实际方向。
- 正式模型文件继续由本地提供且被 Git 忽略；tracked artifact 声明 SHA，测试仅在模型存在时注册 INT8 用例。

主要错误边界：

- config/artifact/model 不一致：session 创建前或 metadata 校验阶段失败；
- 不合法目录/manifest：发现阶段失败，不启动 workers；
- worker session 初始化失败：整批 fatal，任务不开始；
- 单图损坏或输出失败：只标记该 item failed，其他项继续；
- cooperative stop：唤醒阻塞 producer/consumer，已运行任务完成，未开始任务 cancelled；
- summary 计数或字段不自洽：序列化前 `validate_batch_summary` 拒绝写出。

### 3. 文件树与职责

```text
models/
└── best.int8.qdq.u8s8.onnx                  # 本地正式 U8S8 模型，Git ignored
cpp_infer/
├── configs/int8_u8s8_config.txt             # 运行时选择 INT8 artifact
├── artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt
│                                               # model ID/SHA/I/O/语义合同
├── src/main.cpp                              # 单图/batch 公共 CLI 入口
├── src/detector_pipeline.cpp                 # 唯一检测产品链
├── src/onnx_runner.cpp                       # ORT session 与 owned I/O
├── src/batch_discovery.cpp                   # 目录/manifest 确定性发现
├── src/bounded_queue.h                       # 有界 FIFO、背压、stop/close
├── src/batch_runner.cpp                      # 每 worker 独占 Pipeline/session
├── src/batch_writer.cpp                      # BatchSummary 校验与序列化
├── tests/assert_detection_json.py            # 从所选 config 派生模型身份
├── tests/assert_batch_cli.py                 # FP32/INT8 共用 batch 行为测试
├── tools/stage1.ps1                          # Windows config-aware batch compare
├── tools/stage1.sh                           # Linux config-aware detect/batch compare
├── tools/stage2_aarch64.sh                   # AArch64 config override + QEMU 验收
├── .gitattributes                            # Bash 脚本固定 LF
└── results/s2_03/int8_integration/            # 本轮 Linux/AArch64/Windows 证据
docs/details/
├── s2_int8_arm64_batch_integration_spec.md   # 最小整合 SPEC
└── s2_int8_arm64_batch_integration_closure.md
                                                # 本九部分收口
```

没有新增第二套 preprocess、decode、NMS 或坐标恢复；核心 Runtime 业务源码没有因平台分叉。

### 4. 手工实现过程

1. 先核对 Git 分支、工作树和三个单元已有证据，确认正式 artifact 的实际 SHA 与声明一致。
2. 沿 CLI 追踪 `RuntimeConfig -> RuntimeContract -> DetectorPipeline/BatchRunner`，确认 QDQ 外部 I/O 是 float32，Runtime 本身已经兼容。
3. 对比已有证据，定位三处外围割裂：INT8 batch CTest 缺失、batch comparison 固定 FP32、AArch64 workflow 固定 FP32。
4. 写最小 SPEC，冻结“不改检测语义、不新建 INT8 pipeline、不做 QEMU 性能”的边界。
5. 在 CMake 中复用原 batch CLI 集成脚本，模型存在时注册正式 INT8 测试，并给 FP32/INT8 两个用例绑定各自 model ID/SHA。
6. 给 Windows/Linux batch comparison 增加 config 参数；默认 config 不变。
7. 给 AArch64 workflow 增加 `YOLO_DEFECT_AARCH64_CONFIG`，doctor 先解析并打印实际选择；smoke 不再硬编码 FP32 model ID。
8. 第一次 Linux INT8 单图已成功推理，但旧检测 JSON validator 因硬编码 FP32 身份失败。将 validator 改为解析所选 RuntimeConfig 和 ModelArtifactSpec，Windows/Linux/AArch64 三个入口都传入选定 config。
9. Windows 编辑 Bash 文件有行尾风险，因此增加 `.gitattributes` 的 `tools/*.sh text eol=lf`，并用 `bash -n` 验证实际文件。
10. 不并行调用共享同一临时构建目录的 `stage1.sh` 包装动作；一次并行尝试触发共享 build tree 竞争后，其输出被判为无效并删除，正式证据全部串行重跑。
11. 依次完成 Windows 全量回归、Linux 全量回归、Linux INT8 单图/30 图/361 图、AArch64 INT8 `all`、默认 FP32 回归。
12. 最后检查机器 JSON 的 model ID/SHA、counts、queue、comparison、execution context 和 publishable 边界，再同步入口文档。

### 5. 入口、核心类、函数、I/O、所有权与伪代码

#### 入口与所有权

- `main` 只加载一次 `RuntimeContract`。batch 分支把 CLI 参数转换为 `BatchRequest`，运行 `BatchRunner`，然后写 `BatchSummary` 和映射退出码。
- 单图 `DetectorPipeline::Impl` 以值拥有 contract，并独占一个 `OnnxRunner`；`OnnxRunner` 内部独占 ORT env/session 所需资源。
- `BatchRunner` 按 effective worker 数创建 executor。默认 executor 每个拥有一个 `DetectorPipeline`，所以 session 不在线程间共享。
- `BoundedQueue<size_t>` 只传 task index，不复制图片张量；任务和结果数组在 BatchRunner 生命周期内稳定存在。
- 每个 worker 只写自己的当前 item。任务 index 只入队一次，因此没有两个 worker 写同一 item。

#### 核心伪代码

```text
contract = load_runtime_contract(selected_config)

if single_image:
    pipeline = DetectorPipeline(contract)
    image = decode_and_letterbox_to_float32_nchw(path)
    raw = pipeline.owned_ort_session.run(image)
    detections = yolov8_decode_nms_restore(raw)
    write_json_and_optional_png(detections)

if batch:
    tasks = deterministic_discovery(directory_or_manifest)
    results = preallocate_cancelled_items_in_task_order(tasks)
    workers = min(requested_workers, tasks.size)
    executors = [DetectorPipeline(contract) for each worker]
    assert all sessions expose identical metadata
    queue = BoundedQueue(queue_capacity)

    start workers together
    producer:
        for task in tasks:
            queue.push(task.sequence_index)  # full => wait/backpressure
        queue.close()                        # normal drain

    worker[i]:
        while index = queue.pop():
            try:
                results[index] = executors[i].run(tasks[index])
                results[index].status = succeeded
            catch ordinary per-image error:
                results[index].status = failed
                results[index].error = actionable message

    join all workers
    copy queue peak/wait statistics
    derive counts, partial/fatal/cancelled status, throughput and memory
    validate and serialize ordered BatchSummary
```

#### QDQ 数据流

```text
JPEG/PNG bytes
  -> OpenCV BGR uint8
  -> letterbox/RGB/normalize
  -> float32 NCHW [1,3,800,800]
  -> ORT QDQ graph: internal uint8 activation + int8 weights/kernels
  -> float32 output [1,10,13125]
  -> YOLO decode -> class-agnostic NMS -> restore/clip
  -> stable detection JSON / optional PNG
```

### 6. 运行、测试、调试、调优命令与证据

#### Windows x86_64

```powershell
cpp_infer\tools\stage1.cmd test
cpp_infer\tools\stage1.cmd detect data\images\val\crazing_241.jpg `
  cpp_infer\results\s2_03\int8_integration\windows_x86_64\single `
  -Config cpp_infer\configs\int8_u8s8_config.txt
cpp_infer\tools\stage1.cmd batch-compare `
  -Config cpp_infer\configs\int8_u8s8_config.txt
```

最终全量 CTest 为 157/157、0 failed；两个需要 Windows symlink/reparse 权限的用例 skipped，Linux 上实际通过。INT8 单图为 3 检测，INT8/FP32 batch CLI 集成都通过。

#### WSL2/Linux x86_64

```bash
bash cpp_infer/tools/stage1.sh test

bash cpp_infer/tools/stage1.sh detect \
  data/images/val/crazing_241.jpg /tmp/int8-single \
  --config cpp_infer/configs/int8_u8s8_config.txt

bash cpp_infer/tools/stage1.sh batch \
  cpp_infer/tests/fixtures/s2_03_consistency_manifest.txt /tmp/int8-manifest \
  --config cpp_infer/configs/int8_u8s8_config.txt \
  --workers 2 --queue-capacity 4

YOLO_DEFECT_RUN_DIR=/tmp/int8-361 \
  bash cpp_infer/tools/stage1.sh batch-compare \
    --config cpp_infer/configs/int8_u8s8_config.txt
```

结果：157/157 CTest；INT8 固定图 3 检测；30 图 30/30 成功；361 图两组都 361/361 成功并逐图完全一致。性能数字见本文件第 1 部分和 `verification_summary.json`。

#### AArch64 cross-build + QEMU

```bash
export YOLO_DEFECT_AARCH64_CONFIG="$PWD/cpp_infer/configs/int8_u8s8_config.txt"
export YOLO_DEFECT_AARCH64_RESULT_DIR=/tmp/aarch64-int8-single
export YOLO_DEFECT_AARCH64_BATCH_RESULT_DIR=/tmp/aarch64-int8-batch
export YOLO_DEFECT_AARCH64_BATCH_RUN_ID=run_01
bash cpp_infer/tools/stage2_aarch64.sh all
```

该命令必须作为一次完整工作流执行，因为 clean build、inspect、smoke、infer 和 batch 共享临时 cross-build tree。结果应看到 selected RuntimeConfig、AArch64 ELF、正确 loader、INT8 model ID/SHA、单图 3 检测、目录/manifest parity 和 partial failure 2+1。

#### 常见失败定位

- 推理成功但 validator 报 FP32 model ID：确认入口传了 `--expected-config`，并检查 config 相对 artifact 路径。
- INT8 CTest 未注册：模型是本地可选 artifact；确认 `models/best.int8.qdq.u8s8.onnx` 存在且为普通文件，SHA 与 artifact 一致后重新 configure。
- `Text file busy` 或共享构建异常：不要并行运行两个会重建同一 `/tmp/yolo_defect_stage1_linux_release` 的 wrapper action。
- Bash 出现奇怪的 command not found：先检查 CRLF/LF 和 `bash -n`，Git checkout 应遵守 `.gitattributes`。
- AArch64 smoke 提示 build 缺失：运行同一次 `stage2_aarch64.sh all`，或在同一 WSL 环境先 build 再执行后续 action。
- QEMU summary 有吞吐/RSS 数字：读取 `memory.publishable=false` 和 execution context；这些字段用于诊断，不可作为原生 ARM 结论发布。
- workers=4 内存明显增加：这是每 worker 独占 session 的设计成本，不应误判为队列泄漏；结合 session count 和 process-lifetime peak RSS 解释。

证据入口：

- `cpp_infer/results/s2_03/int8_integration/verification_summary.json`
- `cpp_infer/results/s2_03/int8_integration/linux_x86_64/final_20260831/`
- `cpp_infer/results/s2_03/int8_integration/linux_aarch64_qemu/final_20260831_single/`
- `cpp_infer/results/s2_03/int8_integration/linux_aarch64_qemu/final_20260831/`

### 7. 面试验收问题与追问

1. **为什么没有写一个 `Int8DetectorPipeline`？**
   - QDQ 把量化封装在 ONNX 图内部，外部 I/O 契约没变。用 config/artifact 选择模型即可，另写 pipeline 会复制前后处理并增加语义漂移风险。
2. **U8S8 的 U8 和 S8 分别是什么？**
   - 激活 uint8、权重 int8；正式模型 64 个 Conv 全量 QDQ。外部输入输出仍是 float32。
3. **为什么多 worker 更快但内存更大？**
   - 每 worker 独占 ORT session，可以并行执行 batch=1；session 与 buffer 也随 worker 增长。本次吞吐约 3.46x，peak RSS 从约 193 MiB 增到约 557 MiB。
4. **如何证明并发没有改变结果？**
   - 两个进程使用相同 Release build、config、模型、输入顺序和 queue=8，比较 361 份逐图 JSON；字节和语义都一致。
5. **如何证明背压真的发生？**
   - queue peak 达到 capacity，producer wait count 非零；不是只写了一个有界容器却从未阻塞。
6. **目录和 manifest 有何确定性差别？**
   - 目录按递归 generic relative path 排序；manifest 保留 UTF-8 声明顺序，并相对 manifest 所在目录解析。
7. **为什么普通坏图不终止整批？**
   - 单项异常在 worker 任务级捕获，写入 item error；基础设施异常才 fatal stop。
8. **QEMU 证明了什么，没证明什么？**
   - 证明 AArch64 ELF、目标依赖、ORT session 和完整功能链能运行；不证明原生硬件性能、RSS、功耗、温度、驱动和长期稳定性。
9. **INT8 质量是不是完全通过？**
   - 不是。工程融合和 Runtime 合法性通过，但原严格门保留 false：agreement precision 0.938462 小于 0.95，mAP50 drop 0.010356 略高于 0.01。个人项目采用 advisory 收口，不能把失败字段改成成功。
10. **为什么 Linux 本轮不能直接与历史 FP32 数字比较？**
    - 本轮运行在 WSL2 `/mnt/d` DrvFs，历史正式 FP32 在 ext4 临时工作区；存储和协议不同，只能比较本轮同协议 worker=1/4。
11. **如果真机 ARM64 更慢怎么办？**
    - 先用相同 config 和样本建立 native baseline，再看 CPU feature、ORT build、线程、内存带宽和温控；不能用 QEMU 数字预判。
12. **何时值得实现 true batch？**
    - 当模型支持动态/固定 N>1、业务有聚合窗口且端到端延迟预算允许时；需要重新设计 preprocess 聚合、输出拆分、动态 shape 和质量/吞吐协议。

### 8. 可能的手写题与当前代码位置

1. **实现有界阻塞队列：**`cpp_infer/src/bounded_queue.h:24`，`push` 在 `:36`、`pop` 在 `:66`、`request_stop` 在 `:94`；重点是满时等待、空时等待、正常 drain 与 stop 唤醒双方。
2. **实现 worker pool 主循环：**`cpp_infer/src/batch_runner.cpp:847` 的 `BatchRunner::Impl::run`；重点是预创建 executor、start gate、task-level exception isolation、join 和统计回填。
3. **解释单图唯一产品链：**`cpp_infer/src/detector_pipeline.cpp:135` 的 `DetectorPipeline::Impl::run`，按 preprocess、ORT、postprocess、writer 顺序手写伪代码。
4. **实现稳定批处理汇总：**`cpp_infer/include/yolo_defect_cpp/batch_result.h:147` 定义 summary，`cpp_infer/src/batch_writer.cpp:383` 校验不变量、`:706` 序列化、`:1022` 写文件。
5. **实现确定性目录/manifest 发现：**`cpp_infer/src/batch_discovery.cpp:507` 的 `discover_batch_tasks`，说明目录排序与 manifest 声明顺序。
6. **解释 CLI 编排与退出码：**`cpp_infer/src/main.cpp:1037` 是入口，batch 分支从 `:1048` 开始，说明 Runtime library 与 workflow/CLI 的边界。
7. **实现 config-aware 验证：**`cpp_infer/tests/assert_detection_json.py:290` 解析声明，`:318` 从 RuntimeConfig 追到 artifact 并得到 model ID/SHA，避免把 FP32 身份写死在 INT8 工作流。
8. **注册可选正式 artifact 集成测试：**`cpp_infer/CMakeLists.txt:713`，解释本地模型存在时注册、缺失时不拖垮 FP32 主 gate 的取舍。

建议手写顺序：先写 `BoundedQueue<T>`，再写一个 fake executor 的 BatchRunner 小题，最后补 counts/status 推导；这比现场手写 ORT API 更能体现并发 ownership 和工程设计。

### 9. 文档同步状态

- `AGENTS.md`：更新当前阶段、实测结果、平台边界和下一步。
- `README.md` / `README_zh.md`：更新项目入口、正式 INT8 多图链、命令和限制。
- `cpp_infer/README.md`：更新 Runtime 使用方式、测试、证据和 config override。
- `docs/paths_commands.md`：记录 Windows/Linux INT8 batch comparison 与 AArch64 config override 命令。
- `docs/details/s2_int8_arm64_batch_integration_spec.md`：保留实现前的最小 SPEC。
- `docs/details/s2_int8_arm64_batch_integration_closure.md`：本九部分教学收口。
- `cpp_infer/results/s2_03/int8_integration/verification_summary.json`：机器可读的融合总摘要。
- `docs/Proj1_S2.md`：按项目约束未改写。

当前状态：S2-01、S2-02、S2-03 的正式 INT8/跨平台/多图工程链已融合收口，停止并等待用户 L1；S2-04 仍未开始。
