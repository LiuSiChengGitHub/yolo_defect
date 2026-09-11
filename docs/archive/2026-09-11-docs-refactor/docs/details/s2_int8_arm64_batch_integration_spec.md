# S2-01/S2-02/S2-03 INT8 × AArch64 × 多图整合 SPEC

> 分支：`feature/union_int8-arm64-multi`  
> 审计基线：`33c99d3`  
> 冻结日期：2026-08-31

## 1. 问题与已有边界

三个单元的核心实现已经可以复用，但正式证据仍是分段的：

- S2-01 的正式 Round 2 产物是 QDQ/U8S8 模型，只在 Windows x86_64 的 C++ 单图产品链、正确性、质量、benchmark 和 profiling 中验证；
- S2-02 的 Linux x86_64 与 AArch64/QEMU 单图证据使用 FP32；
- S2-03 的 Windows、Linux x86_64 与 AArch64/QEMU 多图证据也全部使用 FP32；
- Linux `batch` 已允许选择 RuntimeConfig，但正式 `batch-compare` 固定默认 FP32；AArch64 workflow 的单图和 batch 同样固定默认 FP32。

C++ 数据面本身不需要分叉：QDQ/U8S8 的外部 I/O 仍是 float32，因此既有 `RuntimeConfig + ModelArtifactSpec -> DetectorPipeline -> OnnxRunner -> postprocess` 可以直接加载 INT8 ONNX；`BatchRunner` 继续为每个 worker 创建独占的 `DetectorPipeline/Ort::Session`。

## 2. 冻结整合对象

| 对象 | 冻结值 |
|---|---|
| RuntimeConfig | `cpp_infer/configs/int8_u8s8_config.txt` |
| model id | `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2` |
| model | `models/best.int8.qdq.u8s8.onnx` |
| SHA-256 | `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1` |
| input/output | float32 `[1,3,800,800]` -> float32 `[1,10,13125]` |
| provider/session | ORT CPU, sequential, intra/inter-op `1/1`, graph optimization `all` |
| manifest correctness set | `cpp_infer/tests/fixtures/s2_03_consistency_manifest.txt`，30 图 |
| Linux performance set | `data/images/val`，361 图，JSON-only，queue=8，workers=1/4 |

派生 INT8 ONNX 是本机 artifact，按现有仓库策略不提交 Git；tracked artifact contract、RuntimeConfig、协议与证据绑定其 SHA。

## 3. 最小实现

1. 保持 Runtime、`DetectorPipeline`、`OnnxRunner`、后处理和 `BatchRunner` 不变。
2. 复用现有 batch CLI 集成测试，并在正式 U8S8 模型存在时以 INT8 config 再注册一次；同一个测试覆盖单图与 worker=1 等价、目录/manifest、worker=1/2 确定性、部分失败、退出码和 summary，并显式断言正式 INT8 model id/SHA。模型缺失时不让 FP32 主 gate 失败，但 CMake 会明确报告未注册 INT8 测试。
3. Linux x86_64 的 `batch-compare` 增加可选 RuntimeConfig，默认仍是 FP32：
   - 固定图继续通过普通 `detect --config` 走产品 C++ 单图链；
   - 30 图 manifest 继续通过普通 `batch --config` 走 INT8 batch；
   - 361 图目录通过 `batch-compare --config` 分别以 workers=1/4 运行，并复用现有比较器确认逐图结果一致后报告吞吐与 peak RSS。
4. AArch64 workflow 增加 `YOLO_DEFECT_AARCH64_CONFIG` 选择入口，默认仍是 FP32；用正式 INT8 config 执行既有 `all`：
   - clean cross-build 与 ELF/loader 检查；
   - QEMU 下固定单图完整 ORT CPU 推理；
   - 两图目录 workers=1 与等价 manifest workers=2/queue=1；
   - 两入口逐图 JSON 一致；
   - 损坏 JPEG 精确得到 `2 succeeded + 1 failed`、退出码 2；
   - 单图和三份 BatchSummary 均绑定正式 INT8 model id/SHA。
5. 既有默认 action 仍使用 FP32，原 CLI、退出码、队列、背压、部分失败和 summary schema 不变。

## 4. 验证与证据边界

### Linux x86_64（WSL2）

- clean/incremental Release 构建与完整 CTest；
- INT8 单图 3 detections；
- INT8 30 图 manifest 全成功，且第一项与单图产品输出一致；
- INT8 361 图 workers=1/4 全成功、逐图结果一致、queue depth 不超过 8、backpressure 可观察；
- 同一 WSL2/Linux x86_64 环境内比较 throughput 与 peak RSS，不设加速硬门；
- FP32 Demo/关键 gate 与 Windows 回归不退化。

### Linux AArch64 under QEMU user-mode

- 证明交叉编译产物和动态依赖是 AArch64；
- 证明 ARM64 OpenCV + ARM64 ORT CPU 能加载正式 INT8 模型并完成单图/多图功能；
- 证明目录、manifest、有限队列、逐图结果、部分失败和 BatchSummary 在目标机器码下保持行为；
- 不发布或解释 QEMU latency、throughput、RSS、功耗、温度和 worker speedup。

### 仍需真实 ARM64 硬件

- 原生延迟、吞吐、内存、功耗、温度与长时间稳定性；
- 板端 OS/驱动/部署包兼容性；
- 针对真实核心数和内存预算选择 worker/queue；
- 真实设备上的 INT8 相对 FP32 收益。

## 5. 非目标

- 不重新量化、不修改阈值/NMS、不把 advisory 质量结果改写为严格全通过；
- 不实现 true ONNX batch `N>1`、共享 session、无锁队列或 ARM 专用推理代码；
- 不增加 QEMU 性能门、Windows 361 图重跑或新的证据 schema；
- 不开始 S2-04 TensorRT。
