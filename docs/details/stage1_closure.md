# 大阶段一最终收口记录

大阶段一已经完成自动工程门和用户 L2 验收。它交付的是一个可配置、可测试、可复现的 C++17 单图工业缺陷推理 Runtime，而不是新的训练方法。本文合并记录整个阶段，不再按内部小阶段拆分。Windows 复现路径见 [`../paths_commands.md`](../paths_commands.md)，后续方向见 [`../Proj1_S2.md`](../Proj1_S2.md)。

## 1. 已交付链路

```text
RuntimeConfig
-> ModelArtifactSpec
-> ORT-observed ModelMetadata validation
-> OpenCV decode
-> letterbox + BGR/RGB + normalize + HWC->CHW / NCHW tensor
-> ONNX Runtime C++ CPU inference
-> YOLOv8 decode + score filter + stable class-agnostic NMS
-> coordinate restore + clip
-> deterministic detection JSON + visualization PNG
-> Python ORT/C++ ORT consistency evidence
-> segmented C++ benchmark and process-memory evidence
```

核心边界如下：

- [`default_config.txt`](../../cpp_infer/configs/default_config.txt) 保存运行策略：artifact、score threshold、NMS threshold 和 provider。
- [`yolov8_neu_det.artifact.txt`](../../cpp_infer/artifacts/yolov8_neu_det.artifact.txt) 保存模型身份、SHA、I/O tensor、类别与前后处理语义；模型族特定的 YOLO decode 没有伪装成通用后端。
- Runtime library 承担 config/artifact 校验、preprocess、ORT RAII session、postprocess 和结果写入；CLI 只负责参数与编排。
- 输入张量由 OpenCV 图像生成 float32 NCHW；输出复制到自持有内存后再进入后处理，避免泄漏 ORT 生命周期。
- score filter 使用严格 `confidence > threshold`；NMS 顺序稳定，在模型坐标中完成后再恢复和裁剪到原图。
- 一致性和性能是不同证据门：正确性失败时禁止发布 benchmark。

## 2. 环境与模型契约

### 已验证环境

| 项目 | 值 |
|---|---|
| 主机/系统 | `DESKTOP-6OGK71C`，Windows `10.0.26200`，x86_64，16 logical CPUs |
| 编译 | MSVC `19.50.35721.0`，C++17，NMake，Release |
| C++ Runtime | OpenCV `4.8.0`，ONNX Runtime `1.19.2` |
| Python reference | Python `3.9.25`，ORT `1.19.2`，OpenCV `4.13.0`，NumPy `2.0.2` |
| Provider/session | requested `cpu`，actual `CPUExecutionProvider`；sequential，intra/inter-op `1/1`，graph optimization `all` |

### 模型与 Runtime 契约

| 项目 | 值 |
|---|---|
| 模型 | [`models/best.onnx`](../../models/best.onnx)，12,336,935 bytes |
| SHA-256 | `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` |
| 模型 ID/family | `yolov8n_neu_det_final_train_2` / `yolov8` |
| ONNX | opset `17`，metadata `nms=False` |
| 输入 | `images`，float32 `[1,3,800,800]`，NCHW |
| 输出 | `output0`，float32 `[1,10,13125]`，BCN，即 `4 + 6 classes` |
| 类别 | `crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches` |
| preprocess | letterbox、BGR->RGB、`[0,1]` normalize、HWC->CHW |
| postprocess | `yolov8_raw`、score `0.25`、NMS IoU `0.45`、class-agnostic |

加载时仍以真实文件和 ORT metadata 为准，逐项比较实际 name/shape/dtype/provider 与声明契约；文档中的记录不是跳过运行时校验的理由。

## 3. Demo 与 106 项自动工程门

固定样本 [`crazing_241.jpg`](../../data/images/val/crazing_241.jpg) 为 23,845 bytes、`200x200x3`。当前已跟踪 Demo 结果为 3 个 `crazing` detections：

| 产物 | 大小 | SHA-256 |
|---|---:|---|
| [`crazing_241.detections.json`](../../cpp_infer/results/demo/crazing_241.detections.json) | 1,164 bytes | `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049` |
| [`crazing_241.visualized.png`](../../cpp_infer/results/demo/crazing_241.visualized.png) | 39,306 bytes | `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346` |

JSON 经过解析与 schema validator，PNG 经 OpenCV 回读为 `200x200 CV_8UC3`。输出默认拒绝覆盖，并保护 config、artifact、model 和输入图像路径。

完整 CTest inventory 为 `106`，正式收口与 fresh closure reproduction 均为 `106/106` 通过。覆盖重点包括：

- RuntimeConfig/artifact schema、路径、SHA、enum、范围与缺失/重复/未知字段。
- 已知像素的 RGB/normalize/NCHW、横竖非方图、奇数 padding、空图和无效输入。
- ORT input/output name、shape、dtype、class count、provider 和输入元素数校验。
- YOLO output、严格阈值、IoU、稳定 NMS、坐标还原/clip、合法空 detection 与非有限值。
- JSON/PNG、覆盖保护、路径独立性、固定 Demo 和 CLI 负例。
- 30 图一致性以及 benchmark statistics、CLI、JSON schema、结果稳定性和错误条件。

CTest label 之间存在重叠，不能把各 label 数量相加后冒充总测试数。

## 4. 固定 30 图一致性

manifest [`consistency_manifest.json`](../../cpp_infer/tests/fixtures/consistency_manifest.json) 固定六类各 5 张。比较使用同一个 ONNX、config 和 artifact，Python 与 C++ 都显式使用 CPUExecutionProvider；matching 先要求 class exact，再用确定性的最大 IoU 匹配，不能按输出数组顺序直接 zip。

| 门槛/结果 | 冻结要求 | 实际结果 |
|---|---:|---:|
| 图片 | 30/30 | `30/30` |
| Python/C++/matched detections | count exact | `62/62/62` |
| class id | exact | 全部通过 |
| 最大 confidence 绝对误差 | `<= 1e-4` | `8.049977111568296e-07` |
| 最大 bbox 坐标绝对误差 | `<= 1e-2 px` | `9.135351561440075e-05 px` |
| 最小 matching IoU | `>= 0.999` | `0.999998927116394` |

已跟踪证据为 [`per_image.json`](../../cpp_infer/results/consistency/per_image.json) 和 [`summary.json`](../../cpp_infer/results/consistency/summary.json)。这证明固定集合上的 Python ORT/C++ ORT 实现一致性，不是模型精度评估、bitwise equality 或缺失 `.pt` 时的新三方 PyTorch/ONNX/C++ 复跑。

## 5. 仓库内正式 C++ benchmark

正式记录是 [`yolov8_neu_det_cpu_release.json`](../../cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json)。协议固定为 Release、batch 1、单张固定图、CPU sequential 1/1、warmup `10`、repeat `100`；percentile 使用 empirical nearest-rank ceiling。

| Segment | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Image decode | 0.991129 | 0.9649 | 1.3517 |
| Preprocess | 8.244569 | 7.5514 | 12.1265 |
| `Ort::Session::Run` | 165.555859 | 164.8985 | 186.2136 |
| Postprocess | 0.424115 | 0.4251 | 0.5636 |
| Pipeline | 175.560944 | 175.1058 | 195.1376 |
| End to end | **176.553060** | **176.1357** | **196.6128** |

- Pipeline throughput：`5.696028 img/s`。
- End-to-end throughput：`5.664020 img/s`。
- Windows Peak Working Set：`160,133,120 bytes`，即 `152.714844 MiB`。

`Session::Run` 只计同步 ORT 调用；pipeline 还包含 preprocess、tensor 构造/输入校验、输出校验/复制和 postprocess；end-to-end 再包含 `cv::imread`。session 初始化、warmup、JSON/PNG 写入和可视化不计入逐图时延。Peak Working Set 是整个进程生命周期高水位，包含初始化、warmup、采样和 harness，不是当前 RSS、单阶段增量或模型独占内存。

## 6. 临时 closure reproduction

阶段关闭时又在新的 `%TEMP%\yolo_defect_s1_09_<GUID>` Release/NMake 构建和唯一证据目录中完整复跑：clean configure/build、`106/106` CTest、固定 3-detection Demo、30/30 与 62/62 consistency、10/100 benchmark、JSON/PNG 回读、故障注入和合法空结果均通过。性能复跑为：

| Segment | Mean ms | P50 ms | P95 ms |
|---|---:|---:|---:|
| Image decode | 0.816168 | 0.8182 | 0.9251 |
| Preprocess | 5.453755 | 5.4547 | 6.2128 |
| `Ort::Session::Run` | 134.419309 | 137.5882 | 142.5549 |
| Postprocess | 0.345302 | 0.3438 | 0.4424 |
| Pipeline | 141.265814 | 144.4673 | 149.8395 |
| End to end | **142.082777** | **145.3222** | **150.7653** |

- Pipeline/end-to-end throughput：`7.078853 / 7.038151 img/s`。
- Peak Working Set：`159,989,760 bytes`，即 `152.578125 MiB`。
- 临时 benchmark JSON：5,453 bytes，SHA-256 `F32C0DF3157897264F9BD2B9AE3F3DB7B240A3B641494E8D3E7C346FF64E9C6F`。

这些 TEMP 文件**未纳入 Git**，也不替换仓库内正式 JSON。它们只证明同一代码和协议能够 fresh reproduce；两次时延差异没有被归因于代码优化，不能忽略后台负载、warm file cache 和普通 OS scheduling 的影响。根 README 的主性能记录继续采用仓库内正式基线。

## 7. 故障与诊断边界

错误统一尽量给出 failing object/path、expected、actual 和 action。自动门及 closure 重点覆盖：

| 故障 | 稳定边界与处理 |
|---|---|
| 模型缺失/路径错误/SHA 不符 | 在 artifact/model preflight 失败；检查相对 artifact 声明文件解析后的路径与目标模型 |
| input/output shape 或 dtype 错误、class count 不符 | 声明契约与 ORT-observed metadata 交叉校验失败；禁止进入推理 |
| 损坏图片 | 文件存在但 OpenCV decode 返回 empty；与“路径不存在”分开诊断 |
| 输入元素数、raw output shape/元素数或非有限值错误 | 在 runner/postprocess 边界失败；不把 malformed output 当作“无缺陷” |
| 合法零候选 | 成功返回空 detection list，JSON 写出 `"detections": []`；它不是异常 |
| 输出 parent 不可创建、目标受保护或已存在 | 默认拒绝写入/覆盖；选择可写的新目录，只有明确意图时才用 `--overwrite` |
| benchmark repeat 为 0、非 Release、provider/thread/模型/样本漂移 | 拒绝生成或发布正式性能证据；恢复冻结协议后重新先跑 consistency |

closure 中缺模型、损坏图片、不可写输出、`--repeat 0` 四个直接故障均返回 exit code `1` 并包含可行动信息；合法无候选 postprocess 与空数组 JSON 两项检查通过。

## 8. 证据与许可边界

- 一致性只覆盖固定 30 图和同一 ONNX；它不是 mAP、完整数据集精度或跨平台证明。
- benchmark 只覆盖一个 200x200 样本、一个 Windows CPU 主机、batch 1、单线程策略和 warm cache；未锁定 CPU affinity、优先级或空闲系统，也不是 GPU、并发、冷盘、全数据集或跨机器排名。
- `actual_provider` 是显式 CPU EP 注册、session 创建和成功运行的 session-level 证据；没有 ORT profiling 时不能声称逐节点 placement。
- 历史 Python ORT `24.4/72.1 FPS` 使用不同实现、样本/硬件和计时协议，不能冒充当前 C++ 结果或与其无条件排序。
- 当前工作区和 Git 历史中没有匹配的 `best.pt`。artifact 对 ONNX 来源的记录是项目所有者确认，不能写成当前可重新导出的完整 lineage。
- 仓库源码使用 [`MIT`](../../LICENSE)，但 ONNX artifact 原样声明 `AGPL-3.0 License (https://ultralytics.com/license)`；NEU-DET 再分发条款未明确。源码、模型和数据集的义务必须分别核验，公开发布前仍需完成 provenance/redistribution checkpoint。
- 大阶段一没有交付 INT8、ORT per-node profiling、Linux/AArch64 portability、目录并发或 TensorRT；这些能力只能在后续真实实现和证据完成后写入结果。
