# S2-01 INT8 PTQ 与 ORT Profiling 技术收口

> 本文中的环境版本、路径、命令和实验结果是 S2-01 收口时的历史证据快照，不是当前工具链配置指南。当前操作入口与故障诊断统一以 [`../paths_commands.md`](../paths_commands.md) 为准。

> 收口日期：2026-08-25
>
> 平台：Windows x86_64，ONNX Runtime 1.19.2 CPUExecutionProvider
>
> 最终 artifact：v1 全 64 个 Conv 的 static PTQ，QDQ/S8S8
>
> 完成口径：个人练习项目的 advisory correctness policy

## 1. 收口结论与范围覆盖

S2-01 要回答两个工程问题：第一，能否从冻结的 FP32 ONNX 稳定地产出可加载、可追溯、可比较的 INT8 ONNX；第二，`Ort::Session::Run` 内部的时间主要消耗在哪些 operator/node，以及这些内部热点能否解释正式分段 benchmark 的结果。

最终交付固定为 `s2_01_static_ptq_qdq_s8s8_cpu_v1`：用 180 张冻结校准图对源模型全部 64 个 `Conv` 做 ONNX Runtime static PTQ，采用 QDQ、activation/weight S8S8、MinMax、per-channel weight。派生模型经 ONNX checker、Python ORT 和 Release C++ ORT 实际验证，外部 I/O contract 保持不变。模型从 12,336,935 bytes 缩小到 3,545,141 bytes，减少 8,791,794 bytes，即 71.2640%。

用户随后明确将项目定位为简历个人练习，不再以“找到任何能通过严格产品差异门的极小量化子集”为目标。因此，本次只改变完成判定：Python/C++ Runtime 合法性仍是硬要求，30 图产品检测差异和 361 图任务质量仍按量化前冻结的方法、样本、阈值和门值计算并保留真实布尔值，但它们作为 advisory 诊断项，不再阻断 PTQ、benchmark、profile 和练习收口。没有修改冻结门值，也没有把失败结果改写为成功。

这一区分必须保留：

- `cpp_infer/results/s2_01/correctness_quality_v1_failed.json` 的根 `passed=false`，其中产品检测差异 `passed=false`，任务质量 `passed=true`；
- `cpp_infer/results/s2_01/exercise_completion.json` 的 `passed=true` 只表示 advisory 练习交付完整，且同时记录 `strict_acceptance_passed=false`；
- INT8 在当前 CPU 上变慢是正式结果，不是失败，也不能写成“INT8 加速”；
- v2-v11 是范围切换前的 selective-PTQ 探索史，不是最终发布 artifact。

## 2. 冻结协议、数据和 lineage

### 2.1 协议与模型

| 项目 | 冻结值 |
|---|---|
| protocol | `cpp_infer/protocols/s2_01_ptq_protocol.json` |
| protocol id | `s2_01_static_ptq_qdq_s8s8_cpu_v1` |
| protocol canonical-LF SHA-256 | `0EC9A7B1CF5E4F246CF3AC15275EF06D7C67FB6C0CE11C5218391CFACE5B73F2` |
| FP32 source | `models/best.onnx` |
| FP32 SHA-256 | `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` |
| FP32 size | 12,336,935 bytes（11.7654 MiB） |
| INT8 derived | `models/best.int8.qdq.onnx` |
| INT8 SHA-256 | `C0B4EDAF6B26B1495E22B9B504CF677EA9A08D10B051156AD55649F98C0EDE2F` |
| INT8 size | 3,545,141 bytes（3.3809 MiB） |
| INT8 contract | `cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt` |
| INT8 RuntimeConfig | `cpp_infer/configs/int8_config.txt` |

派生 ONNX 受根 `.gitignore` 的 `models/*.onnx` 规则管理，不作为源码提交；公开复现时应由冻结 protocol、源模型和校准 manifest 重新生成。独立 artifact contract 固定 model id、source/provenance、derived SHA、opset、I/O、class names、pre/postprocess 和许可证提示，运行时实际 metadata 必须再校验，不能只相信文本声明。

### 2.2 校准 manifest

校准 manifest 为 `cpp_infer/tests/fixtures/s2_01_calibration_manifest.json`，id 是 `neu_det_train_ptq_calibration_6x30_v1`，canonical-LF SHA-256 是 `6C0735C6E1510F725E1168A3C57E7107259CC1934D32DEB3E619C1BF6712AA9D`。它从 NEU-DET train split 的六个文件名前缀类别各固定选 30 张，共 180 张；规则是每类选择 `i=1+8*k, k=0..29`。类别只用于平衡取样，PTQ 不消费标签。

每条样本都记录原始图片文件 bytes 的 SHA-256，180 条规范化记录的集合 SHA-256 为 `FDEF7FB3B64E222386387438C0B4A32A6BDECF9761E5ED5C60E9A17B7311AE5F`。量化前和量化时都逐图重算 hash；报告记录 `sample_count_expected=180`、`sample_count_hash_verified=180`、`sample_count_consumed=180`。

校准 preprocess 冻结为：

1. OpenCV `IMREAD_COLOR` 解码为 uint8 BGR；
2. 按正数 `floor(value+0.5)` 的舍入规则做 `INTER_LINEAR` letterbox 到 `800x800`，pad value 为 114；
3. BGR 转 RGB；
4. 转 float32 并除以 255；
5. HWC 转 NCHW，输出 `[1,3,800,800]` 的 C-contiguous tensor。

### 2.3 正确性与性能样本

| 用途 | manifest/sample | 冻结信息 |
|---|---|---|
| 产品检测差异 | `cpp_infer/tests/fixtures/consistency_manifest.json` | 30 图，id `neu_det_val_6x5_v1`，canonical-LF SHA-256 `4A10742F373D1A999839996D45BEAD84F3340F3A37C35A18E9EBF534147F1E46` |
| 任务质量 | `cpp_infer/tests/fixtures/s2_01_quality_manifest.json` | 361 图、857 个 GT box，canonical-LF SHA-256 `CED5CE80B119B1446066B18072B2AD1C7BE7A6DA30429B5C01D617F2AA2BCEF8` |
| 正式 benchmark/profile | `data/images/val/crazing_241.jpg` | raw SHA-256 `1D65EF27EAA9BF27608D954DFE57B40E401FC1AED435884400F35E8000BBF98D` |

产品 matching 冻结为逐图、同 class id、按 IoU 降序贪心一对一匹配，pair IoU 最低 0.5，并定义稳定 tie-break 和 percentile 插值。任务质量采用 IoU 0.50:0.05:0.95、COCO 101-point precision envelope，但不包含 COCO area range 或 max-dets 语义；因此文档只称为本项目冻结的 COCO-style 指标。

## 3. Static PTQ 实现与产物

### 3.1 量化配置

本次调用 `onnxruntime.quantization.quantize_static`，关键参数是：

| 参数 | 值 |
|---|---|
| format | `QDQ` |
| activation / weight | `QInt8 / QInt8`（S8S8） |
| op types | `Conv` |
| selected Conv | 64/64 |
| per channel | `true` |
| reduce range | `false` |
| calibration | `MinMax` |
| activation symmetric | `false` |
| weight symmetric | `true` |
| excluded nodes | 空 |
| external data | `false` |
| preprocess | skip optimization/symbolic shape，执行 ONNX shape inference |

量化不是在源文件上原地写入。工具先在临时目录中预处理和量化，完成 hash、metadata、ONNX checker、图结构审计和 Python ORT smoke 后，再原子发布 derived ONNX 与报告；目标已存在时必须显式给 `--overwrite`。

### 3.2 图审计

源图有 64 个目标 `Conv`。派生 QDQ 图仍保留 `Conv` op type，因此不能仅按 op name 判断是否量化；审计器要求每个目标 Conv 同时具备 activation DequantizeLinear、INT8 per-channel weight 及其 DequantizeLinear、输出 QuantizeLinear，并核对 weight scale 元素数等于输出通道数。

最终审计结果为：

- quantized Conv：64；
- intentional unquantized Conv：0；
- failed/unquantized target Conv：0；
- excluded-policy violation：0；
- 派生图节点数：634；
- `QuantizeLinear`：123，`DequantizeLinear`：250；
- initializer：627，其中 INT8 251、INT32 126、FLOAT 250；
- ONNX checker：passed。

“文件中 64 个 Conv 都通过 QDQ 审计”不等于“ORT 优化后 trace 中会出现 64 个 QLinearConv”，两者观察的是不同层次。前者证明 artifact 的静态量化结构，后者反映特定 ORT build、provider 和图优化下的实际执行图。

### 3.3 大小与 I/O contract

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---:|
| 文件 bytes | 12,336,935 | 3,545,141 | -8,791,794 |
| MiB | 11.7654 | 3.3809 | -8.3845 |
| INT8 / FP32 | 1.0000 | 0.28736 | 减少 71.2640% |

Python ONNX 读取、Python ORT session metadata、C++ ORT session metadata 和 artifact contract 对外部 I/O 的观察一致：

| 方向 | name | shape | dtype |
|---|---|---|---|
| input | `images` | `[1,3,800,800]` | float32 |
| output | `output0` | `[1,10,13125]` | float32 |

QDQ 模型的内部权重/activation 使用 INT8 表达，模型边界仍是 float32；C++ 产品链因此可继续复用同一 preprocess、YOLO decode、score filter、NMS 和坐标恢复。

## 4. Runtime 合法性与检测/质量事实

### 4.1 Python/C++ Runtime 合法性

Python ORT 1.19.2 对 FP32 和 INT8 都成功创建仅含 `CPUExecutionProvider` 的 sequential session，使用 intra/inter-op `1/1`，分别实际执行冻结输入；两者输出 shape 都是 `[1,10,13125]`，共 131,250 个元素，且全部有限。量化报告中的 smoke 观测为：

| 模型 | Python session init | 单次 Session::Run | 输出范围 | finite |
|---|---:|---:|---|---|
| FP32 | 44.7104 ms | 144.3946 ms | `[0, 797.76599]` | true |
| INT8 | 143.4923 ms | 195.8928 ms | `[0, 801.35858]` | true |

这些 smoke 时间只证明能运行，不作为正式 benchmark。

Release C++ ORT 1.19.2 也对两份模型成功创建 session、校验相同的 input/output metadata，并在 30 图产品链上实际运行。Python/C++ 产品结果一致性对 FP32 和 INT8 均通过：FP32 30/30 图通过，最大 confidence 误差约 `8.05e-7`、最大 bbox 坐标误差约 `9.14e-5 px`；INT8 30/30 图通过，最大 confidence 误差约 `4.98e-10`、最大 bbox 坐标误差约 `4.99e-7 px`。这证明当前 Python wheel 与官方 C++ SDK 对最终 v1 artifact 的产品输出一致。

### 4.2 30 图产品检测差异：真实失败，advisory 接受

`correctness_quality_v1_failed.json` 的根 `passed=false`，原因是 FP32 与 INT8 的产品差异有三项越过量化前冻结的严格门。结果如下：

| 指标 | 冻结门 | 实测 | 原门结果 |
|---|---:|---:|---|
| FP32 detection total | — | 62 | — |
| INT8 detection total | — | 65 | — |
| matched detections | — | 61 | — |
| FP32 retention | `>=0.95` | 0.983871 | 通过 |
| INT8 agreement precision | `>=0.95` | 0.938462 | **失败** |
| matched IoU mean | `>=0.90` | 0.924997 | 通过 |
| matched IoU P05 | `>=0.75` | 0.833518 | 通过 |
| confidence abs error mean | `<=0.05` | 0.050771 | **失败** |
| confidence abs error P95 | `<=0.10` | 0.173643 | **失败** |

额外诊断中，bbox 坐标绝对误差 mean/P95/max 分别为 1.5999/7.0001/57.3852 px；总 class count 的主要变化是 inclusion `11 -> 13`、patches `9 -> 10`。这些值没有被删除或改写，只是不再阻断个人练习收口。

### 4.3 361 图任务质量：按原门实际通过

| 指标 | FP32 | INT8 | INT8-FP32 | 冻结最大绝对下降 | 原门结果 |
|---|---:|---:|---:|---:|---|
| mAP50 | 0.710815 | 0.707206 | -0.003610 | 0.010 | 通过 |
| mAP50-95 | 0.345786 | 0.342174 | -0.003612 | 0.020 | 通过 |
| 最差 per-class AP50（scratches） | 0.789265 | 0.766219 | -0.023046 | 0.050 | 通过 |

产品差异失败而任务级 AP 通过并不矛盾：30 图差异门直接约束固定 score=0.25 下的 detection agreement、置信度和几何变化；任务质量在 361 图上对完整 precision-recall 排序积分。两者观察维度不同，所以都必须如实保留。

## 5. 同协议正式性能

### 5.1 协议

FP32 与 INT8 使用同一台 Windows x86_64 机器、同一 Release C++17 CLI、MSVC 19.50、OpenCV C++ 4.8.0、ORT 1.19.2 `CPUExecutionProvider`，execution mode 为 sequential，intra/inter-op `1/1`，graph optimization `all`。两者在独立进程中运行同一张 `crazing_241.jpg`，batch=1、score=0.25、class-agnostic NMS=0.45、warmup=10、repeat=100，profiler 明确关闭。

机器记录为 AMD64 Family 25 Model 117、16 logical CPUs、Windows 10.0.26200。没有设置 CPU affinity、提升进程优先级或锁定系统空闲状态。

### 5.2 结果

| 阶段/指标 | FP32 mean | FP32 P50 | FP32 P95 | INT8 mean | INT8 P50 | INT8 P95 |
|---|---:|---:|---:|---:|---:|---:|
| image decode (ms) | 0.8198 | 0.8151 | 0.9983 | 0.8186 | 0.8038 | 0.9336 |
| preprocess (ms) | 5.5692 | 5.4677 | 6.8621 | 5.7713 | 5.6625 | 6.8241 |
| `Session::Run` (ms) | 139.9201 | 141.6768 | 156.4734 | 191.9134 | 190.9285 | 220.7692 |
| postprocess (ms) | 0.3459 | 0.3417 | 0.4458 | 0.3573 | 0.3330 | 0.4228 |
| pipeline (ms) | 146.9272 | 148.7791 | 163.9209 | 199.2283 | 198.4941 | 229.2748 |
| end-to-end (ms) | 147.7478 | 149.5826 | 164.8248 | 200.0477 | 199.3120 | 230.2798 |

| 其他指标 | FP32 | INT8 | 结论 |
|---|---:|---:|---|
| session initialization | 40.3094 ms | 94.9792 ms | INT8 为 2.3563x，较慢 |
| pipeline throughput | 6.8061 img/s | 5.0194 img/s | INT8 下降 26.25% |
| end-to-end throughput | 6.7683 img/s | 4.9988 img/s | INT8 较慢 |
| Peak Working Set | 158,064,640 bytes（150.7422 MiB） | 158,048,256 bytes（150.7266 MiB） | 差 16 KiB，可视为近似相同 |

最终可比较结论是：INT8 文件缩小 71.2640%，但当前 CPU/graph/provider 上 `Session::Run` mean 从 139.9201 ms 增到 191.9134 ms，即慢 37.16%；pipeline mean 慢 35.60%，吞吐下降约 26.25%。pre/postprocess 基本不变，差异明确来自 `Session::Run`。Peak Working Set 是进程生命周期高水位，包含 session 初始化、warmup、测量和 harness，不是模型独占内存，因此不能从 16 KiB 差值推断量化没有减少权重驻留内存。

## 6. ORT Profiling 与瓶颈解释

### 6.1 采集和摘要方法

FP32/INT8 profile 分别在独立进程、独立 profiling-enabled session 中运行。`ProfileRunner` 只 preprocess 一次，然后对同一 owned NCHW tensor 调用 10 次 `Session::Run`，最后调用 `EndProfilingAllocated` 得到 ORT 实际生成的 Chrome trace；postprocess 只对最后一次输出执行一次。profiler run 与正式 benchmark 完全分离。

摘要只统计 `cat=Node`、`ph=X` 且 event name 以 `_kernel_time` 结尾的事件。两份 trace 都观察到恰好 10 个 model-run event，每个优化后 node 都恰好调用 10 次，placement 100% 为 `CPUExecutionProvider`。摘要还用优化后 operator inventory 检查 precision signature，防止 FP32/INT8 trace 被误交换；但 raw ORT trace 本身不嵌入模型文件 SHA，最终绑定依赖 artifact、protocol、trace SHA 和摘要交叉校验。

| 项目 | FP32 | INT8 |
|---|---:|---:|
| raw trace | `fp32_v1_ort_2026-08-25_16-22-00.json` | `int8_v1_ort_2026-08-25_16-22-01.json` |
| trace SHA-256 | `7F6507FD9069A97567C6F9D4D08771015799F0B3AC170B8C0B065B4B62215B62` | `76E8F8A16EB33EE6950CC63C434C50B33C9B60710DCD6B78ED39619770A6FE25` |
| trace size | 2,274,478 bytes | 5,602,548 bytes |
| kernel events | 2,940 | 6,830 |
| unique optimized nodes | 294 | 683 |
| aggregate kernel time / 10 runs | 1,376.274 ms | 1,918.933 ms |
| aggregate kernel time / run | 137.6274 ms | 191.8933 ms |
| provider | CPU 100%（2,940 calls） | CPU 100%（6,830 calls） |

### 6.2 FP32 top operators

| rank | op_type | calls/10 runs | total ms | 占比 | 累计占比 |
|---:|---|---:|---:|---:|---:|
| 1 | Conv | 640 | 933.169 | 67.8040% | 67.8040% |
| 2 | QuickGelu | 560 | 105.216 | 7.6450% | 75.4490% |
| 3 | ReorderInput | 570 | 98.114 | 7.1290% | 82.5780% |
| 4 | ReorderOutput | 630 | 84.957 | 6.1730% | 88.7509% |
| 5 | Softmax | 10 | 64.376 | 4.6776% | 93.4285% |
| 6 | Concat | 170 | 50.089 | 3.6395% | 97.0680% |

FP32 top nodes 均调用 10 次：

| rank | node | op | total ms | 占比 | 累计占比 |
|---:|---|---|---:|---:|---:|
| 1 | `/model.22/dfl/Softmax` | Softmax | 64.376 | 4.6776% | 4.6776% |
| 2 | `/model.22/cv2.0/cv2.0.0/conv/Conv_output_0_nchwc` | Conv | 54.710 | 3.9752% | 8.6528% |
| 3 | `/model.22/cv3.0/cv3.0.0/conv/Conv_output_0_nchwc` | Conv | 52.253 | 3.7967% | 12.4495% |
| 4 | `/model.22/cv2.0/cv2.0.1/conv/Conv_output_0_nchwc` | Conv | 51.666 | 3.7540% | 16.2035% |
| 5 | `/model.22/cv3.0/cv3.0.1/conv/Conv_output_0_nchwc` | Conv | 51.190 | 3.7195% | 19.9230% |

### 6.3 INT8 top operators

| rank | op_type | calls/10 runs | total ms | 占比 | 累计占比 |
|---:|---|---:|---:|---:|---:|
| 1 | Conv | 570 | 1,238.668 | 64.5498% | 64.5498% |
| 2 | DequantizeLinear | 3,170 | 202.452 | 10.5502% | 75.1001% |
| 3 | Mul | 570 | 123.120 | 6.4161% | 81.5161% |
| 4 | QuantizeLinear | 1,200 | 118.573 | 6.1791% | 87.6952% |
| 5 | Sigmoid | 570 | 62.313 | 3.2473% | 90.9425% |
| 6 | Concat | 170 | 54.130 | 2.8208% | 93.7634% |
| 7 | MaxPool | 30 | 43.468 | 2.2652% | 96.0286% |
| 8 | Softmax | 10 | 28.197 | 1.4694% | 97.4980% |

`QLinearConv` 共 70 calls、9.010 ms、0.4695%，在包含 Split/Add 后的全量排序中位列第 11，累计占比到该项为 99.2099%。

INT8 top nodes 均调用 10 次：

| rank | node | op | total ms | 占比 | 累计占比 |
|---:|---|---|---:|---:|---:|
| 1 | `/model.22/cv2.0/cv2.0.1/conv/Conv` | Conv | 65.384 | 3.4073% | 3.4073% |
| 2 | `/model.22/cv3.0/cv3.0.1/conv/Conv` | Conv | 64.811 | 3.3774% | 6.7848% |
| 3 | `/model.22/cv3.0/cv3.0.0/conv/Conv` | Conv | 64.594 | 3.3661% | 10.1509% |
| 4 | `/model.22/cv2.0/cv2.0.0/conv/Conv` | Conv | 63.200 | 3.2935% | 13.4444% |
| 5 | `/model.1/conv/Conv` | Conv | 60.137 | 3.1339% | 16.5783% |

### 6.4 与 segmented benchmark 的对应关系

正式 benchmark 看到 `Session::Run` mean 从 139.9201 ms 增到 191.9134 ms，INT8/FP32 为 1.3716x。profile 中 node kernel aggregate 每 run 从 137.6274 ms 增到 191.8933 ms，INT8/FP32 为 1.3943x。二者方向和量级一致，因此 profile 能解释“慢在 Session::Run 内部”，但不能用 profile kernel sum 替代正式 wall-clock latency。

当前瓶颈解释是：

1. FP32 的主要成本是 Conv，占 trace 67.8040%；
2. INT8 优化图中仍有每 run 57 个 `Conv`，而 `QLinearConv` 只有每 run 7 个；全量 QDQ artifact 并没有在此 ORT CPU 执行图中全部融合成量化卷积 kernel；
3. INT8 的 `DequantizeLinear + QuantizeLinear` 共 321.025 ms/10 runs，占 trace 16.7293%，且调用数达到每 run 317+120；转换边界是显著新增成本；
4. INT8 的 Conv 仍占 64.5498% 且绝对 total 达 1,238.668 ms，比 FP32 Conv 的 933.169 ms 更高；与此同时 FP32 的 QuickGelu/Reorder 形态在 INT8 优化图中拆成不同的 Mul/Sigmoid/QDQ 组合；
5. 因而，在当前 CPU、ORT 1.19.2、单线程 sequential 和该 YOLO 图上，“文件更小”没有转化为“执行更快”，主要原因是量化融合覆盖不足、剩余 Conv 成本与大量 Q/DQ 转换开销。

这仍是基于 trace 的工程推断。`CPUExecutionProvider` placement 不能证明每个 kernel 使用了哪条整数 SIMD 指令；若要证明 ISA，需要进一步使用 ETW、VTune/perf 或反汇编，这不在 S2-01 范围内。

### 6.5 Profiling overhead 与限制

- ORT profiling 会插桩和写 trace，overhead 存在但本次未单独量化；profile duration 只能用于热点排序和相对解释。
- 正式 benchmark 的 `profiling_enabled=false`，且使用不同进程，避免把 trace 写入和 profiling session 初始化混入性能数字。
- `ORT_ENABLE_ALL` 记录的是优化后执行图；融合、重排和 provider-specific node 不必与原始 ONNX node 一一对应。
- node kernel sum 与外层 `Session::Run` scope 不完全相同：调度、allocation、框架工作及潜在 overlap 都可能造成差异。
- raw trace 不嵌入模型 SHA；本次通过 protocol/artifact SHA、precision operator signature、raw trace SHA 和最终 evidence assembler 做外部绑定。

## 7. 模块、职责、输入输出和 ownership

| 文件/模块 | 责任 | 主要输入 | 主要输出/错误语义 |
|---|---|---|---|
| `cpp_infer/protocols/s2_01_ptq_protocol.json` | 单一冻结协议 | source/calibration/correctness/benchmark/profile 声明 | protocol id、SHA 绑定；漂移即拒绝 |
| `cpp_infer/tools/generate_s2_01_manifests.py` | 确定性生成或 `--check` 校准/质量 manifest | 数据目录、图片和标签 | manifest、逐文件 SHA、集合 SHA；缺图/坏标签/漂移非零退出 |
| `cpp_infer/tools/s2_01_protocol.py` | 强类型加载和交叉校验协议 | protocol、模型、manifests、sample | resolved paths 与冻结字段；期望/实际/修复动作式错误 |
| `cpp_infer/tools/quantize_s2_01.py` | calibration reader、static PTQ、图审计、metadata/runtime smoke、原子发布 | 冻结 protocol | INT8 ONNX、`quantization_report.json`；目标冲突需 `--overwrite`，审计失败不发布 |
| `cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt` | INT8 独立 ModelArtifactSpec/card | derived SHA、I/O 和语义 | C++ runtime contract |
| `cpp_infer/configs/int8_config.txt` | 选择 INT8 artifact 和 CPU provider | artifact path、thresholds | RuntimeConfig |
| `cpp_infer/include/yolo_defect_cpp/onnx_runner.h`、`src/onnx_runner.cpp` | owned ORT session/I/O，Session::Run timing，profiling RAII | `RuntimeContract`、owned input vector、可选 profile prefix | owned `InferenceOutput`；`end_profiling()` 返回 ORT 实际 trace path |
| `cpp_infer/include/yolo_defect_cpp/profile_runner.h`、`src/profile_runner.cpp` | 编排一次 preprocess、N 次 Run、EndProfiling、一次 postprocess | `ProfileRequest` + moved-in `RuntimeContract` | `ProfileResult`；PImpl 独占 contract，runner/session 为局部 RAII owner，禁止 copy |
| `cpp_infer/tools/evaluate_s2_01_correctness.py` | Python/C++ Runtime、30 图 matching、361 图 AP | protocol、两份 config/artifact、Release CLI | 机器 JSON；任何原门失败写出真实 `passed=false` 并非零退出 |
| `cpp_infer/tools/compare_s2_01_benchmarks.py` | 验证同机/同 build/provider/thread/sample/warmup/repeat 并比较 | 两份 benchmark、correctness、protocol | `comparison.json`；advisory 需显式 policy，保留 prerequisite false |
| `cpp_infer/tools/summarize_ort_profile.py` | 验证并聚合 ORT trace | raw trace、protocol、artifact、precision | top/all op/node/provider、占比/累计占比、trace hash；provider/run/signature 不符即失败 |
| `cpp_infer/tools/assemble_s2_01_evidence.py` | 最终交叉绑定所有证据与 raw trace | protocol + 7 份派生 JSON + 两份 trace | `exercise_completion.json`；advisory 下 `passed=true` 与 `strict_acceptance_passed=false` 同时存在 |
| `cpp_infer/tools/stage1.cmd`、`stage1.ps1` | Windows 统一依赖发现、Release build/test 与 profile 入口 | action、可选 config/image/prefix/runs | 保护的 out-of-tree binary/trace；校验 CLI 报告、实际 trace 路径、JSON 可解析性和退出码 |

核心 ownership 规则是：`RuntimeContract` 先由 config/artifact/实际 metadata 建立；`ProfileRunner` 通过 PImpl 独占一份 moved-in contract；每次 profile 构造一个局部 `OnnxRunner`，其 PImpl 独占 `Ort::Env`/SessionOptions/Session/allocator 和 profile 状态；输入 tensor 数据由 `PreprocessResult::tensor_nchw` 的 `std::vector<float>` 拥有，`InferenceOutput` 把输出 shape/value 拷贝为 owned C++ 容器；`EndProfilingAllocated` 返回的 ORT 字符串由 allocator-aware RAII 对象管理，再转换成 `std::filesystem::path`。

## 8. 不依赖 Codex 的人工实现流程与宏观伪代码

### 8.1 人工流程

1. 检查工作树、源 ONNX 真实 SHA/大小/metadata，并确定 Python/C++ ORT 版本和 CPU provider。
2. 确定校准、产品差异、任务质量和 benchmark 样本；写逐图 SHA 的 manifest，先以 `--check` 确认确定性和 calibration/quality 无 image-hash overlap。
3. 在第一次量化前写 protocol，冻结 preprocess、QDQ/S8S8 参数、matching、AP、benchmark 和 profile 协议，并记录 canonical-LF SHA。
4. 写 `CalibrationDataReader`，严格按 manifest 顺序逐图验证 SHA、执行与产品链同语义的 preprocess，每次返回一个 contiguous float32 NCHW tensor。
5. 在临时目录预处理 ONNX 并调用 `quantize_static`；完成 ONNX checker、实际 metadata、64 个 Conv 的 QDQ/per-channel graph audit、模型 hash/大小和 Python ORT finite smoke 后原子发布。
6. 新建独立 INT8 artifact spec 和 RuntimeConfig，让现有 C++ `DetectorPipeline` 复用同一 pre/postprocess，而不是复制一套 INT8 产品逻辑。
7. 对 FP32/INT8 运行 Python和 C++ legality、一致性、30 图产品 matching 与 361 图 AP；即使失败也保留完整 JSON，不调门值修结果。
8. 在两个独立 Release CLI 进程中、profiler 关闭时运行固定样本 warmup 10/repeat 100；比较 init、分段 latency、throughput 和 PWS。
9. 再用两个独立 profiling-enabled session 各运行 10 次，只将 trace 用于诊断；摘要聚合 top node/op/provider，并绑定 raw trace SHA。
10. 最终 assembler 重算 protocol/model/manifest/sample/trace SHA，交叉检查所有 evidence lineage；本次显式使用 advisory policy，输出练习完成但 strict acceptance false。

### 8.2 宏观伪代码

```text
protocol = load_and_verify_frozen_protocol()
verify_sha(source_model, calibration_manifest, product_manifest,
           quality_manifest, benchmark_sample)

reader = FrozenCalibrationDataReader(manifest_order)
for sample in calibration_manifest:
    verify_raw_image_sha(sample)
    tensor = letterbox_bgr_to_rgb_float32_nchw(sample.image)
    reader.enqueue_owned_contiguous_tensor(tensor)

staged_model = quantize_static(
    source_model,
    reader,
    format=QDQ,
    activation=QInt8,
    weight=QInt8,
    op_types=[Conv],
    per_channel=true,
    calibrate_method=MinMax)

assert onnx_checker(staged_model)
assert external_io(staged_model) == frozen_contract
audit = inspect_qdq_structure_for_each_source_conv(staged_model)
assert audit.quantized == 64 and audit.failed == 0
assert python_ort_run(staged_model).all_finite
atomically_publish(staged_model, quantization_report)

for precision in [FP32, INT8]:
    assert python_runtime_and_cpp_runtime_are_legal(precision)
correctness = evaluate_30_image_matching_and_361_image_quality()
write_truthful_json(correctness)  # v1 root passed remains false

for precision in [FP32, INT8]:
    spawn_release_process(profiling=false)
    warmup(10)
    record_100_iterations(init, decode, pre, run, post, pipeline, e2e,
                          throughput, peak_working_set)
compare_same_protocol(correctness_policy=advisory)

for precision in [FP32, INT8]:
    tensor = preprocess_once(frozen_sample)
    session = create_session(profiling=true, prefix)
    repeat 10: session.Run(tensor)
    trace = session.EndProfiling()
    summary = aggregate_kernel_events_and_bind_trace_sha(trace)

assemble_all_evidence(correctness_policy=advisory)
assert exercise_completion.passed == true
assert exercise_completion.strict_acceptance_passed == false
```

## 9. 运行、复现和证据命令

以下命令从仓库根执行。`$Py` 应指向安装 onnx/onnxruntime 1.19.2、NumPy 和 OpenCV 的量化环境；`$Cli` 应指向 `stage1.cmd build` 本次生成的 Release `yolo_defect_cpp.exe`，不要硬编码个人绝对路径。正式输出已存在时，建议先写到新的 repro 路径核验；确需替换当前 evidence 时才使用 `--overwrite`。

```powershell
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd test

& $Py cpp_infer/tools/generate_s2_01_manifests.py --check

& $Py cpp_infer/tools/quantize_s2_01.py `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --overwrite

& $Py cpp_infer/tools/evaluate_s2_01_correctness.py `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --fp32-config cpp_infer/configs/default_config.txt `
  --int8-config cpp_infer/configs/int8_config.txt `
  --cpp-cli $Cli `
  --output-json cpp_infer/results/s2_01/correctness_quality_v1_failed.json
```

当前 Release 回归已通过 118/118 个 CTest，无失败。统一入口也分别用默认 FP32 config 和 INT8 config 实际完成 `stage1.cmd profile` smoke，两个进程均 exit 0：

```powershell
.\cpp_infer\tools\stage1.cmd profile -ProfileRuns 1
.\cpp_infer\tools\stage1.cmd profile `
  -Config cpp_infer\configs\int8_config.txt `
  -ProfileRuns 1
```

这两条是统一入口/trace 生成 smoke；下面直接调用 `$Cli` 的命令才是本次固定 prefix、固定 10 runs 的正式 profile evidence 采集方式。

上面的 `evaluate_s2_01_correctness.py` 命令在 v1 上因冻结产品门失败而返回非零，这是预期行为；应检查已经写出的 JSON，而不是改门或丢弃结果。两条 `stage1.cmd profile` smoke 均为 exit 0。

正式 benchmark 必须两个独立进程、profiler 关闭：

```powershell
& $Cli --config cpp_infer/configs/default_config.txt `
  --image data/images/val/crazing_241.jpg `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json cpp_infer/results/s2_01/benchmark/fp32_cpu_release.json `
  --overwrite

& $Cli --config cpp_infer/configs/int8_config.txt `
  --image data/images/val/crazing_241.jpg `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json cpp_infer/results/s2_01/benchmark/int8_cpu_release.json `
  --overwrite

& $Py cpp_infer/tools/compare_s2_01_benchmarks.py `
  --fp32 cpp_infer/results/s2_01/benchmark/fp32_cpu_release.json `
  --int8 cpp_infer/results/s2_01/benchmark/int8_cpu_release.json `
  --correctness cpp_infer/results/s2_01/correctness_quality_v1_failed.json `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --correctness-policy advisory `
  --output cpp_infer/results/s2_01/benchmark/comparison.json `
  --overwrite
```

profile 使用相同图片，但不与 benchmark 混跑：

```powershell
& $Cli --config cpp_infer/configs/default_config.txt `
  --image data/images/val/crazing_241.jpg `
  --profile `
  --profile-prefix cpp_infer/results/s2_01/profile/fp32_v1_ort `
  --profile-runs 10

& $Cli --config cpp_infer/configs/int8_config.txt `
  --image data/images/val/crazing_241.jpg `
  --profile `
  --profile-prefix cpp_infer/results/s2_01/profile/int8_v1_ort `
  --profile-runs 10

& $Py cpp_infer/tools/summarize_ort_profile.py `
  --trace cpp_infer/results/s2_01/profile/fp32_v1_ort_2026-08-25_16-22-00.json `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --artifact cpp_infer/artifacts/yolov8_neu_det.artifact.txt `
  --precision fp32 `
  --output cpp_infer/results/s2_01/profile/fp32_summary.json `
  --overwrite

& $Py cpp_infer/tools/summarize_ort_profile.py `
  --trace cpp_infer/results/s2_01/profile/int8_v1_ort_2026-08-25_16-22-01.json `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --artifact cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt `
  --precision int8 `
  --output cpp_infer/results/s2_01/profile/int8_summary.json `
  --overwrite
```

ORT 会在 prefix 后附带时间戳；重新运行后必须把 `--trace` 改成 CLI 实际返回的新文件名，不能继续引用旧 trace。

最终 advisory evidence assembly：

```powershell
& $Py cpp_infer/tools/assemble_s2_01_evidence.py `
  --protocol cpp_infer/protocols/s2_01_ptq_protocol.json `
  --quant-report cpp_infer/results/s2_01/quantization_report.json `
  --correctness cpp_infer/results/s2_01/correctness_quality_v1_failed.json `
  --fp32-benchmark cpp_infer/results/s2_01/benchmark/fp32_cpu_release.json `
  --int8-benchmark cpp_infer/results/s2_01/benchmark/int8_cpu_release.json `
  --benchmark-comparison cpp_infer/results/s2_01/benchmark/comparison.json `
  --fp32-profile-summary cpp_infer/results/s2_01/profile/fp32_summary.json `
  --int8-profile-summary cpp_infer/results/s2_01/profile/int8_summary.json `
  --correctness-policy advisory `
  --output cpp_infer/results/s2_01/exercise_completion.json
```

## 10. 调试、失败诊断和定制

### 10.1 常见失败

- **protocol/manifest SHA 漂移：**先运行 manifest `--check`，再重算模型和样本 hash。不要在量化后回改 v1；数据或配置有意变化时新建 protocol id/version。
- **Python ORT 能跑、C++ ORT 不能跑：**Python wheel 不能替代 C++ SDK。检查官方 headers/import library/runtime DLL、x64 架构、ORT 版本和 `stage1.cmd doctor` 输出。
- **量化报告显示 Conv 数量不对：**检查 `op_types_to_quantize`、nodes_to_exclude、preprocess 中间图和源节点 identity。QDQ 图仍保留 Conv 名称，必须看 activation/weight/output QDQ 完整结构。
- **量化中途失败却出现半成品：**正式工具应只从 staging 原子发布；不要手工把临时模型复制到 `models/best.int8.qdq.onnx`。
- **correctness CLI 返回 1：**先读输出 JSON。v1 的产品差异 `passed=false` 是已知、必须保留的事实；advisory policy 只能改变阻断性，不能改写这个字段。
- **benchmark 无法比较：**检查是否同 hostname/OS/compiler/Release/provider/thread/sample/threshold/warmup/repeat，确认两份 JSON 的 `profiling_enabled=false`，并确保它们来自独立进程。
- **profile summary 拒绝 trace：**核对实际时间戳文件、precision、artifact、provider、恰好 10 个 model-run、每 node 10 calls 和 raw trace SHA。不要把 INT8 trace 配给 FP32 artifact。
- **INT8 反而更慢：**先看 optimized graph 中 `QLinearConv` 覆盖、Q/DQ calls/time、未融合 Conv 和线程配置，而不是先改 benchmark。当前结果已证明“量化成功”和“CPU 加速”不是同一个命题。
- **PWS 几乎不变：**它是进程生命周期峰值，包含 runtime arena、输入/输出、warmup 和 harness；若要看模型增量内存，需要另设进程基线或更细粒度采样，不能从本次 PWS 直接推出。

### 10.2 调参和后续实验规则

若要练习 alternative calibration、U8S8、reduce-range、不同 per-channel 策略、节点排除、线程数或 graph optimization，必须新建 protocol id、derived filename、artifact、config 和 evidence root，v1 保持不可变。比较时双方仍需使用同一数据、阈值、provider、线程和 benchmark 协议；profile 仍独立运行。

如果目标从个人练习升级为产品发布，则应恢复 `correctness-policy required`，并以新的数据和门值重新冻结协议，而不是把本次 advisory completion 当成严格 acceptance。对当前性能问题，优先实验的方向是：检查 CPU 对 QLinearConv 的实际 kernel 覆盖、尝试适合该硬件的 quantization representation/ORT 版本、减少 Q/DQ 边界、对热点 head Conv 做有依据的选择，并始终用相同正式 benchmark 复测。

## 11. 机器可读证据索引

| 证据 | 路径 | 当前 raw SHA-256 |
|---|---|---|
| frozen protocol | `cpp_infer/protocols/s2_01_ptq_protocol.json` | `0EC9A7B1CF5E4F246CF3AC15275EF06D7C67FB6C0CE11C5218391CFACE5B73F2` |
| quantization card/report | `cpp_infer/results/s2_01/quantization_report.json` | `21BFE1501E171B452A1545129EC4773B2608EBFB15F3F0A124A62E3172D49133` |
| correctness/quality truth | `cpp_infer/results/s2_01/correctness_quality_v1_failed.json` | `73E417D16BDCDEB0B95C1946DA53F552DAF4DF9895CFCF95EB685ADB6A9B9062` |
| FP32 benchmark | `cpp_infer/results/s2_01/benchmark/fp32_cpu_release.json` | `28F345CB5F35F2CA2E1C3A45F84D2F9CC5052961B9B3B38C0079AAA6D516E5F0` |
| INT8 benchmark | `cpp_infer/results/s2_01/benchmark/int8_cpu_release.json` | `0643EF9F6A7EF7985605AE33F0968465EC801204EC8B37268F8C2A28EBF4A0FD` |
| benchmark comparison | `cpp_infer/results/s2_01/benchmark/comparison.json` | `EDC2AF9CA388FB2442D68E72FD03BBE9F6F4905C06017D55664DC33A09ABA8D9` |
| FP32 raw trace | `cpp_infer/results/s2_01/profile/fp32_v1_ort_2026-08-25_16-22-00.json` | `7F6507FD9069A97567C6F9D4D08771015799F0B3AC170B8C0B065B4B62215B62` |
| INT8 raw trace | `cpp_infer/results/s2_01/profile/int8_v1_ort_2026-08-25_16-22-01.json` | `76E8F8A16EB33EE6950CC63C434C50B33C9B60710DCD6B78ED39619770A6FE25` |
| FP32 profile summary | `cpp_infer/results/s2_01/profile/fp32_summary.json` | `0BABB42263BAB9D21A53E92F2971095AC272E62530A42CE6D071CFC072227254` |
| INT8 profile summary | `cpp_infer/results/s2_01/profile/int8_summary.json` | `74A162962D20FAF23018D22F9AFE6DD5AC1319E9928E7F53F27CED01AAC37021` |
| advisory completion | `cpp_infer/results/s2_01/exercise_completion.json` | `F8B7132F2B85FB2CAEBDD57775F1321EA9FB6F1D21CB2EF3BF3B53A68BCFF7D9` |

最终 completion 会重算并绑定 protocol、source/derived model、三个 manifest、benchmark sample、所有输入 evidence 和两份 raw trace 的路径、大小与 SHA。当前输出的关键 truth 是：`passed=true`、`strict_acceptance_passed=false`、`correctness.policy=advisory`、`correctness.reported_passed=false`、`product_detection_difference_passed=false`、`task_quality_passed=true`。

## 12. 探索史、限制与停止点

在用户切换范围之前，曾按原严格产品门探索 v2-v11，包括保留检测头 FP32、不同 backbone/neck 子集、连续前缀和 Entropy calibration 等；其中有产品门失败、跨 Python/C++ 数值一致性失败、candidate screen 失败和 Entropy 内存失败。这些记录用于说明 selective PTQ 并非“量化节点越少越容易过门”，但不再继续搜索，也不参与最终 v1 artifact、benchmark、profile 或 completion 的 lineage。

当前证据边界如下：

- 只证明本机 Windows x86_64 + ORT CPU 1.19.2；不能外推 Linux、AArch64、其他 CPU、CUDA/TensorRT 或真实边缘板卡。
- 正式性能只用一张 200x200 图、batch=1 和 warm OS file cache；没有 affinity/priority/system-idle 控制。
- calibration 只有平衡抽取的 180 张训练图，适合个人 PTQ 练习，不代表生产数据覆盖充分。
- 361 图任务质量按本项目冻结的 COCO-style 实现，不等同官方 COCO evaluator 的全部 area/max-dets 口径。
- profile overhead 未量化，trace 是优化后执行图；CPU provider 不能证明整数 ISA。
- Peak Working Set 不是 per-stage 或模型独占内存。
- matching `.pt` 不在本机，本次按用户允许直接以 FP32 ONNX 为 PTQ source；没有声称重新完成 PyTorch/ONNX 三方 lineage。
- source metadata 声明 Ultralytics AGPL-3.0；派生模型分发仍受源模型和数据集条款约束。

S2-01 在“完成 INT8 static PTQ、生成独立 contract/card、Python/C++ 实际运行、同协议性能比较、FP32/INT8 raw profile 与 top-op/node 解释、机器 evidence 交叉绑定”的练习口径下完成，当前 Windows Release 回归为 118/118。严格产品 acceptance 没有通过，这一事实保持公开。下一步应先等待用户 L1 讲解与追问，不在本单元内启动 S2-02。
