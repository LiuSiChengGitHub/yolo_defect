# S2-01 Round 2：从 S8S8 量化变慢到 U8S8 量化加速

> 收口日期：2026-08-27  
> 平台：Windows x86_64，ONNX Runtime 1.19.2 `CPUExecutionProvider`  
> 最终 artifact：全 64 个 Conv 的 static PTQ，QDQ/U8S8  
> 完成口径：Runtime 合法、模型轻量化、同协议 CPU 加速；产品差异与任务质量保留为 advisory

## 1. 最终结论

S2-01 Round 2 已完成，可以形成一条完整且可验证的学习叙事：

```text
Round 1：QDQ/S8S8 文件量化成功，但 Runtime 变慢
-> 用 optimized graph 与 ORT profile 定位到整数 Conv 覆盖率不足
-> 不再继续减少 Conv，而只把 activation QInt8 改为 QUInt8
-> Round 2：64/64 Conv 均以 QLinearConv 执行
-> 模型缩小 71.269%，Session::Run 加速 38.726%，pipeline 加速 36.461%
```

最终 U8S8 模型为 `models/best.int8.qdq.u8s8.onnx`，大小 3,544,494 bytes，约为 FP32 的 28.73%。正式 C++ Release benchmark 中，`Session::Run` mean 从 155.106 ms 降至 95.040 ms，pipeline mean 从 163.477 ms 降至 103.872 ms，吞吐从 6.117 提升到 9.627 images/s。361 图任务质量仅小幅下降：mAP50 下降约 1.04 个百分点，mAP50-95 下降约 0.34 个百分点。

“实现完成”和“原严格门通过”仍需区分：Python/C++ Runtime 合法性、轻量化、整数 kernel 覆盖和性能目标已经完成；30 图 agreement 与原 mAP50 严格门仍保留 `false`，但按个人练习项目的 advisory 口径不阻断收口，也没有把失败布尔值改写为成功。

## 2. Round 1 为什么会变慢

Round 1 使用的是 QDQ/S8S8：activation 为有符号 `QInt8`，weight 也是 `QInt8`。静态 ONNX 审计看到 64 个 `Conv` 都有量化权重和 Q/DQ 结构，因此“模型文件量化成功”；但这不等于运行时一定执行整数卷积。

ORT 1.19.2 优化后的执行图给出了关键证据：

| Round 1 optimized graph（每次运行） | 数量 |
|---|---:|
| 普通 float `Conv` | 57 |
| `QLinearConv` | 7 |
| `QuantizeLinear` | 120 |
| `DequantizeLinear` | 317 |
| 优化后总节点 | 683 |

也就是说，64 个源 `Conv` 虽然在文件中被 QDQ 标记，真正进入整数卷积执行路径的却只有 7 个；其余 57 个仍支付 float Conv 成本，同时还增加 Q/DQ 转换。FP32 原本可把 `x * sigmoid(x)` 的 SiLU 模式优化成 `QuickGelu`，Round 1 中则出现独立的 `Sigmoid` 和 `Mul`。正式结果因此从 FP32 的 139.920 ms 退化到 S8S8 的 191.913 ms。

进一步检查源图拓扑后发现，64 个 Conv 输出中正好有 57 个存在 fan-out：56 个经 output-DQ 同时进入 `Sigmoid` 与 `Mul`，另 1 个进入其他多个 consumer；仅 7 个是单 consumer。这与“57 个 float Conv + 7 个 QLinearConv”精确对应。结合 ORT 的 x86 signed-to-unsigned QDQ rewrite/selector 实现，可以提出一个强因果假设：S8 activation、该 YOLO SiLU fan-out 图以及 ORT 1.19.2 CPU 优化路径组合在一起，阻断了多数整数 Conv 融合。

这是由 A/B 执行图和 profile 支持的工程推断，不写成“ORT 维护者已确认的 bug”，也不声称已经反汇编证明某条 VNNI 指令实际执行。ONNX Runtime 的量化文档本身也强调：QDQ/S8S8 是常见起点，但实际性能取决于硬件和可用 kernel，Q/DQ 开销或不合适的表示可能使量化模型更慢。[ORT quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) Ultralytics 社区也有 YOLOv8 ONNX INT8 变慢的实际报告；它只能证明“成熟生态不等于每个组合都自动加速”，不能替代本项目自己的根因证据。[Ultralytics issue #4097](https://github.com/ultralytics/ultralytics/issues/4097)

## 3. 改进选择：只改 activation S8 -> U8

Round 2 没有继续做 v12、v13 式的“减少 Conv 直到跑得快”，而是进行单变量因果实验：

| 参数 | Round 1 | Round 2 |
|---|---|---|
| format | QDQ | QDQ |
| activation | QInt8 | **QUInt8** |
| weight | QInt8 | QInt8 |
| calibration | 180 图、MinMax | 相同 |
| Conv target | 全部 64 个 | 相同 |
| weight granularity | per-channel | 相同 |
| reduce range | false | 相同 |
| preprocess / I/O / benchmark | 冻结值 | 相同 |

U8S8 表示让激活范围使用无符号 8 bit、权重继续使用有符号 8 bit。它没有改变外部 I/O：应用仍传入 float32 `images [1,3,800,800]`，仍得到 float32 `output0 [1,10,13125]`。改变的是模型内部量化张量表示，使 CPU EP 可以直接匹配 unsigned-activation 的整数 Conv 路径，不再依赖把 signed activation 图成功改写后才能融合。

ORT 1.19.2 的 Python API 注释也把 CPU 的 asymmetric activation + symmetric weight 作为推荐组合；本项目对应为 `QUInt8` activation 与 `QInt8` weight。[ORT 1.19.2 quantize.py](https://github.com/microsoft/onnxruntime/blob/v1.19.2/onnxruntime/python/tools/quantization/quantize.py#L2210-L2217)

QDQ 被继续保留，因为它仍是 ORT 推荐、易检查且已被现有工具链完整支持的表示。临时 feasibility screen 也验证过 QOperator/U8S8 能跑，但 QDQ/U8S8 已明显超过 FP32，因此按停止条件不再引入第二种正式格式、升级 ORT 或继续搜索更多候选。

## 4. 实现与量化产物

协议 loader 新增 `s2_01_static_ptq_qdq_u8s8_cpu_r2`，量化工具不再硬编码 activation/weight 为 `QInt8`，而是把协议声明安全映射为 ORT `QuantType.QInt8` 或 `QuantType.QUInt8`。Round 1 的 11 个历史 protocol 行为保持不变，新测试验证 Round 2 相对 v1 只有 `activation_type` 一个量化参数发生变化。

正式量化配置为：

```python
quantize_static(
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["Conv"],
    per_channel=True,
    reduce_range=False,
    calibrate_method=CalibrationMethod.MinMax,
)
```

| 产物事实 | 结果 |
|---|---|
| FP32 source | `models/best.onnx`，12,336,935 bytes |
| U8S8 derived | `models/best.int8.qdq.u8s8.onnx`，3,544,494 bytes |
| derived / source | 0.287308 |
| 文件缩减 | **71.269%** |
| source Conv / selected / quantized | 64 / 64 / 64 |
| intentional exclusions / failures | 0 / 0 |
| Python ORT CPU session + finite output | 通过 |
| C++ ORT CPU session + finite output | 通过 |
| actual metadata | float32 `[1,3,800,800]` -> float32 `[1,10,13125]`，与 contract 一致 |

模型 SHA 只用于正式模型 artifact 身份：FP32 为 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`，U8S8 为 `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`。普通 benchmark JSON、trace 和中间优化图不需要在日常工作中重复计算 SHA。

## 5. 正确性与任务质量

三类验证回答不同问题：Runtime legality 证明模型能创建 session、运行且输出有限值；30 图 detection matching 观察同一产品阈值下 FP32/INT8 检测变化；361 图带标签评估才回答任务质量变化。

| 30 图产品差异 | 结果 |
|---|---:|
| FP32 / INT8 detections | 62 / 65 |
| matched detections | 61 |
| FP32 retention | 0.9839 |
| INT8 agreement precision | 0.9385 |
| matched mean IoU / P05 | 0.9243 / 0.8315 |
| confidence abs error mean / P95 | 0.0494 / 0.1603 |

原产品门要求 agreement precision 至少 0.95、confidence P95 不超过 0.10，因此该部分仍为 `passed=false`。

| 361 图任务质量 | FP32 | U8S8 | delta |
|---|---:|---:|---:|
| mAP50 | 0.710815 | 0.700459 | -0.010356 |
| mAP50-95 | 0.345786 | 0.342379 | -0.003407 |
| operating-point precision | 0.675978 | 0.670732 | -0.005246 |
| operating-point recall | 0.705951 | 0.705951 | 0 |

mAP50 原门允许最多下降 0.010000，实测多下降 0.000356，因此机器结果如实保留 `false`；但从个人练习项目的目标看，这是约 1.04 个百分点的有限精度损失，不能成为继续删层或篡改门值的理由。

## 6. 同协议正式性能结果

FP32 与 U8S8 分别由独立 C++ Release 进程运行，均固定 `CPUExecutionProvider`、sequential execution、intra/inter-op `1/1`、同一张 `crazing_241.jpg`、warmup 10、repeat 100，并明确关闭 profiler。

| 指标 | FP32 | U8S8 | 变化 |
|---|---:|---:|---:|
| session initialization | 61.986 ms | 94.858 ms | U8S8 慢 53.03% |
| `Session::Run` mean | 155.106 ms | 95.040 ms | **下降 38.726%，1.632x speedup** |
| `Session::Run` P50 | 155.124 ms | 95.570 ms | 下降 38.39% |
| `Session::Run` P95 | 169.639 ms | 110.768 ms | 下降 34.70% |
| pipeline mean | 163.477 ms | 103.872 ms | **下降 36.461%** |
| pipeline P50 / P95 | 163.221 / 182.008 ms | 104.042 / 121.654 ms | 均改善 |
| pipeline throughput | 6.117 img/s | 9.627 img/s | **提升 57.383%** |
| Peak Working Set | 150.980 MiB | 148.832 MiB | 小降 2.15 MiB，不能视为模型独占内存 |

初始化变慢并不与稳态推理加速矛盾：session 创建还包含模型读取、QDQ 图优化、kernel 选择和内存规划，且每个模型只有一个初始化观测；服务长期运行时主要看多次 `Session::Run`，短命 CLI 则需同时考虑初始化。

## 7. Round 2 ORT Profiling

Profiler 使用各自独立、开启 profiling 的 C++ session，每个模型运行 10 次。trace 只用来解释节点和 placement，不参与上节 benchmark。

| optimized execution graph | FP32 | Round 1 S8S8 | Round 2 U8S8 |
|---|---:|---:|---:|
| unique executed nodes | 294 | 683 | 439 |
| float `Conv` per run | 64 | 57 | **0** |
| `QLinearConv` per run | 0 | 7 | **64** |
| `QuantizeLinear` per run | 0 | 120 | 60 |
| `DequantizeLinear` per run | 0 | 317 | 140 |

Round 2 的核心成功证据不是“ONNX 文件里有 QDQ”，而是 10-run trace 中出现 640 次 `QLinearConv`、0 次普通 `Conv`，即每次运行 64 个整数卷积，且全部 placement 为 `CPUExecutionProvider`。

本次 trace 的 top operator 聚合如下：

| FP32 operator | calls / 10 runs | trace total | 占比 |
|---|---:|---:|---:|
| Conv | 640 | 843.890 ms | 74.75% |
| QuickGelu | 560 | 84.008 ms | 7.44% |
| ReorderInput | 570 | 61.971 ms | 5.49% |
| ReorderOutput | 630 | 48.857 ms | 4.33% |

| U8S8 operator | calls / 10 runs | trace total | 占比 |
|---|---:|---:|---:|
| QLinearConv | 640 | 337.527 ms | 35.18% |
| DequantizeLinear | 1400 | 127.960 ms | 13.34% |
| Resize | 20 | 126.862 ms | 13.22% |
| Mul | 570 | 100.819 ms | 10.51% |
| Concat | 160 | 84.316 ms | 8.79% |
| QuantizeLinear | 600 | 58.404 ms | 6.09% |
| Sigmoid | 570 | 53.054 ms | 5.53% |

这些数据既解释了成功，也指出剩余瓶颈：Conv 主成本显著下降，但 Q/DQ、未融合的 SiLU (`Sigmoid` + `Mul`)、Resize 和 Concat 占比上升。若以后继续优化，应先减少这些边界或在配套升级 ORT 后复测，而不是继续随机删 Conv。

Profile trace 插桩会改变调度和各节点耗时，因此不能用 1128.934 ms 与 959.457 ms 的 trace 总和替代正式 benchmark，也不能据此声称 ORT 内部使用了某条具体 CPU 指令。

## 8. 完成判定、trade-off 与边界

Round 2 可以按个人练习目标收口，原因是：

- 模型由 12.34 MB 降到 3.54 MB，达到轻量化；
- Python/C++ ORT 都真实创建 session、运行并得到有限输出；
- 64 个 Conv 全部进入 `QLinearConv`，profile 给出执行层证据；
- 同协议正式 C++ benchmark 的 Run、pipeline、P50、P95 和吞吐均明显改善；
- 361 图 mAP 的损失有限且被真实记录；
- Round 1 负结果没有被删除，形成“失败—定位—单变量改进—成功”的证据链。

它不等于生产发布验收：单图、batch 1、单机、单线程、无 CPU affinity；初始化只有一个观测；PWS 是进程生命周期高水位；产品 agreement 仍未过原门；matching `.pt` 不在仓库；profile 不证明具体 ISA。当前 CPU 能力探测显示 VNNI 可用，但这只说明硬件/运行环境报告该能力，不能替代 kernel 反汇编。

Ultralytics 当前 ONNX 导出实现同样只选择带权重的 Conv/Gemm/MatMul 进行量化，避免检测头中 box 尺度与 class probability 共用量化尺度导致分数严重舍入；这支持本项目继续采用 Conv-only，而不是为了“更多 INT8 节点”量化所有算子。[Ultralytics ONNX exporter](https://docs.ultralytics.com/reference/utils/export/onnx/)

## 9. 复现入口与证据索引

当前机器路径和环境入口只查 [`../paths_commands.md`](../paths_commands.md)。核心命令为：

```powershell
# 生成并审计正式 U8S8 artifact
& 'C:\Users\Everbreath\.conda\envs\yolo_defect\python.exe' `
  cpp_infer\tools\quantize_s2_01.py `
  --protocol cpp_infer\protocols\s2_01_ptq_protocol_r2_u8s8.json

# 构建、单图运行和独立 profile
cpp_infer\tools\stage1.cmd build
cpp_infer\tools\stage1.cmd detect data\images\val\crazing_241.jpg `
  -Config cpp_infer\configs\int8_u8s8_config.txt
cpp_infer\tools\stage1.cmd profile data\images\val\crazing_241.jpg `
  -Config cpp_infer\configs\int8_u8s8_config.txt `
  -ProfileRuns 10
```

正式证据：

- [Round 2 protocol](../../cpp_infer/protocols/s2_01_ptq_protocol_r2_u8s8.json)
- [U8S8 artifact contract](../../cpp_infer/artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt)
- [U8S8 RuntimeConfig](../../cpp_infer/configs/int8_u8s8_config.txt)
- [Quantization report](../../cpp_infer/results/s2_01/round2/u8s8/quantization_report.json)
- [Correctness and quality result](../../cpp_infer/results/s2_01/round2/correctness_u8s8.json)
- [Formal benchmark comparison](../../cpp_infer/results/s2_01/round2/benchmark/comparison_u8s8.json)
- [FP32 profile summary](../../cpp_infer/results/s2_01/round2/profile/fp32_summary.json)
- [U8S8 profile summary](../../cpp_infer/results/s2_01/round2/profile/int8_u8s8_summary.json)
- [Round 1 historical closure](s2_01_closure.md)

派生 ONNX 受根 `.gitignore` 管理，不随源码提交；raw profile JSON 是诊断证据，正式性能数值只读取 profiler 关闭的 benchmark JSON。
