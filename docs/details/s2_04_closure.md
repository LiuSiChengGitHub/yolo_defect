# S2-04：WSL2/Linux x86_64 + RTX 4060 Laptop 真实 TensorRT 加速收口

> 收口日期：2026-08-31  
> 完成口径：实现、真实 GPU 执行、冻结正确性门禁、性能/内存证据和 Windows/Linux CPU 回归均完成，停止等待用户 L1；S2-05 未开始。  
> 证据边界：这里只证明 **WSL2/Linux x86_64 本地 GPU/edge-node + NVIDIA GeForce RTX 4060 Laptop GPU**。它不是 Jetson、ARM64 GPU、嵌入式板卡或裸机原生 Linux GPU 证据。

## 1. 讲解本步工作

### 1.1 五分钟口述

S2-04 要解决的问题不是“YOLO 能不能检测”，因为前面的 CPU Runtime、量化、跨平台和多图并发已经完成；这一步要证明的是：同一个 C++ 产品链、同一个 ONNX artifact、同一套 OpenCV 前处理和 YOLO 后处理，能否在 RTX 4060 Laptop 上由 TensorRT 真实执行，并给出正确性、重复性、延迟、吞吐、host RSS 和 GPU memory 证据。

我先按低侵入方案把 ORT TensorRT EP 接入现有 `OnnxRunner`。GPU 专用 build 使用 ORT GPU 1.20.1，provider 顺序固定为 TensorRT、CUDA、CPU，并打开 FP16、engine cache 和 timing cache；默认 Windows/Linux CPU build 仍使用 ORT 1.19.2。这样 `DetectorPipeline` 的输入输出不变，TensorRT 仍接收 FP32 NCHW，输出仍复制成 owned FP32 vector，decode、NMS、坐标恢复和 JSON 都复用旧代码。`trtexec --fp16` 对当前 SHA 为 `7B8A...AF68` 的 ONNX 成功建出 engine，保存后重新加载 100 次也成功；ORT profile 还观察到 10 个 `TensorrtExecutionProvider` kernel event，CUDA 和 CPU fallback 都是 0，所以这条路确实运行了 TensorRT，而不只是配置名写成了 TensorRT。

但是“真实执行”不等于“检测语义正确”。冻结 v1 的严格数值门禁后，ORT EP 在 `inclusion_300` 上失败；再冻结互斥样本的 v2 语义门禁，仍有两张图越过 confidence 或一像素坐标上限。两次 TensorRT 重复结果彼此完全一致，说明它稳定地得到了一组与 CPU 语义门禁不兼容的结果。当前 ORT 1.20.1 又没有后续版本才提供的算子排除选项，不能只把敏感算子踢出 TRT 子图。因此我没有放宽阈值，而是按预案实现最小 load-only native TensorRT backend。

native backend 不重新实现 YOLO。它只加载一个离线生成、SHA 绑定的 TensorRT 10.4 plan，校验 TensorRT/CUDA/SM 8.9、engine SHA、输入输出名称、shape 和 dtype，创建 execution context、非默认 CUDA stream 和持久 device buffers；每次调用依次做 H2D、`setTensorAddress`、`enqueueV3`、D2H 和 stream synchronize，再返回旧的 `InferenceOutput`。`OnnxRunner` 仍是对上层的统一接口，所以产品数据流和检测语义没有分叉。native plan 没有 fallback；如果环境、engine、config 或 metadata 不匹配，会在推理前给出 expected/actual/action 诊断。

精度策略也经过失败驱动。第一版 native v3 只让最终类别 Sigmoid 使用 FP16，其余 FP32/noTF32；它在已冻结的第三组 30 图上仍有一张 bbox 最大偏差 3.271 像素，虽然 A/B 两次结果完全一致，所以 v3 被保留为失败证据。随后只用已经消耗的 v1、v2、v3 共 90 图筛选策略，得到最终 E0 engine：只有 DFL Softmax 是实际 FP16 计算，旁边两层 reformat 触及 Half，其余计算和外部 I/O 都是 FP32，并关闭 TF32。E0 在旧 90 图上全过后，才冻结从未运行过、与前三组互斥的 v4 30 图。

正式 v4 中，CPU FP32 对 native A/B 都是 30/30 通过，64 个匹配 detection 的最大 confidence 差 `1.0044e-5`、最大坐标差 `0.032166 px`、最小 IoU `0.998619`；native A/B 按文件字节完全一致。这样才允许发布性能。最终 engine SHA 为 `E0CBB0...8746`，21,144,012 bytes；`trtexec` reload 100 次为 301.55 q/s，host P50/P95 `3.074/3.536 ms`，GPU compute P50/P95 `2.420/2.883 ms`，但未锁频的 Laptop GPU 有明显波动。

产品 benchmark 使用同一 GPU build、同一张图、batch=1、warmup=10、repeat=100。CPU FP32 pipeline throughput 是 `8.325 img/s`。native warm A/B 初始化为 `684.570/619.423 ms`，session boundary P50 为 `3.877/3.633 ms`，pipeline P50/P95 为 `6.974/8.779 ms` 和 `6.519/10.490 ms`，pipeline throughput 为 `137.652/140.555 img/s`，约为 CPU 的 `16.54x/16.88x`。这只是“整个 native TensorRT engine 相对 ORT CPU”的结果，不能把加速归因于唯一 FP16 Softmax。A/B correctness 稳定、吞吐相差约 2.1%，但 P95 抖动明显，所以不能说 tail latency 稳定。host peak RSS 为约 `384.7/384.4 MiB`；GPU memory 是 `nvidia-smi` 的 device-wide baseline-to-peak `155 MiB`，不是目标 PID 独占显存。

最终判断是：`trtexec`、ORT TRT EP 和 native C++ 产品路径都观察到了真实 TensorRT；ORT EP 的正确性失败被如实保留，最终发布路径是 SHA 绑定、DFL-Softmax-only FP16 的 native plan；同 artifact 冻结 v4 correctness、重复运行、正式性能和原 Windows/Linux ORT 回归已完成。INT8 没有扩展，因为当前 artifact 是 FP32 ONNX，严谨的 INT8 还需要独立 calibration/QDQ 契约，属于“不阻塞主链时再做”的可选项。S2-04 到此停止，等待用户 L1。

### 1.2 教学级完整讲解

#### 路线位置、输入输出和非目标

S2-04 位于 S2-03 有界多图并发之后、S2-05 招聘材料冻结之前。输入是当前 opset 17、`images:float32[1,3,800,800] -> output0:float32[1,10,13125]` 的 YOLOv8 ONNX，以及既有的 `RuntimeConfig + ModelArtifactSpec`。输出仍是原 `DetectionResult` JSON/可视化；新增的是 provider/engine 身份、正确性协议、benchmark、profile 和资源证据。

本单元不改变 score、class-agnostic NMS、类别顺序、letterbox 或坐标恢复；不做 GPU batch、多 stream 并发、服务化、功耗、温度或长稳；不把 WSL2 说成 Jetson/ARM 板；也不声称重建了缺失的 PyTorch lineage。

#### 端到端链路

```text
RuntimeConfig v1/v2 + ModelArtifactSpec
  -> 选择 CPU / ORT TensorRT EP / native TensorRT
  -> 实际 ModelMetadata 与 provider/engine 契约校验
  -> OpenCV decode + letterbox + BGR->RGB + [0,1] + FP32 NCHW
  -> OnnxRunner 统一边界
       CPU: Ort::Session::Run
       ORT TRT EP: TensorRT -> CUDA -> CPU provider chain
       native: H2D -> setTensorAddress -> enqueueV3 -> D2H -> sync
  -> owned FP32 output [1,10,13125]
  -> 原 YOLO decode + class-agnostic NMS + coordinate restore
  -> 原 Detection JSON / benchmark / correctness comparator
```

统一边界很重要：上层只看 owned input/output，不知道底层由 ORT 还是 TensorRT context 执行。这样加速实验不会复制前后处理，也不会制造两套 detection semantics。

#### 为什么先 ORT EP、再 native

ORT TensorRT EP 是成本最低的集成方式。ORT 负责读取 ONNX、划分 TensorRT 支持子图、管理 TensorRT engine 和 fallback，并继续提供 `Session::Run`。它让项目快速获得真实 TensorRT、cache 和 provider profile，但控制粒度有限：敏感节点的精度策略由 EP/Builder 决定，`Session::Run` 还包含传输和同步。

本模型在 ORT EP 上不是运行失败，而是正确性门禁失败。这是更有价值的工程结论：provider 可用、重复结果稳定，但检测语义相对 CPU 漂移。v2 已经使用一像素/IoU 0.90/confidence 0.005 的业务门禁，继续放宽就会让验收追着结果走。ORT 1.20.1 又不能用 `trt_op_types_to_exclude` 精确排除算子，所以转 native backend，显式冻结每层精度，是比改阈值更可解释的选择。

#### native ownership、执行和错误语义

`NativeTensorRtRunner::Impl` 独占 logger、runtime、engine、execution context、CUDA stream、input/output device buffer 和 metadata。构造阶段读取并哈希 plan，检查 config 宣称的 SHA，反序列化 engine，按 name-based TensorRT 10 API 校验 I/O，然后一次分配持久 buffer。析构按 RAII 释放 CUDA 和 TensorRT 资源。

调用方的 input vector 只在同步调用期间被借用；H2D 后 TensorRT 使用 runner-owned device input。D2H 写入函数内新建的 output vector，返回后由 `InferenceOutput` 独占。所有 CUDA/TensorRT failure 都说明失败对象、期望、实际和动作；engine SHA、版本、compute capability、I/O 或 execution policy 不一致时，不允许“尽量跑一下”。

`tensorrt_max_workspace_size_bytes` 对 native v4 只是离线 build provenance/config compatibility 字段，load-only backend 本身不再消费 workspace。engine 也不是可跨 GPU/版本发布的通用 artifact；TensorRT/CUDA/SM 或 builder policy 变化后必须重建。

#### “真实 FP16”到底证明了什么

原始 unconstrained `trtexec --fp16` engine 大量使用 Half，但 ORT EP correctness 失败。最终 E0 engine 采用受约束 mixed precision：只有 `/model.22/dfl/Softmax` 是 Half compute，两个相邻 reformat 触及 Half，其他 compute 与外部 I/O 都是 Float，并禁用 TF32。因此它满足“至少有 FP16 实际执行”，却绝不是全 FP16 网络。

这也决定了性能解释：约 16–17 倍的提升来自 TensorRT 原生 engine、kernel/tactic/fusion、GPU 执行和整体 backend 差异，无法从当前实验中分离出“FP16 单独贡献了多少”。把 speedup 直接归因于 FP16 会超过证据。

#### 正确性协议为什么分四轮

- v1 在运行前冻结 30 图和严格数值门禁，ORT EP 失败 1 图；保留结果，不改写。
- v2 使用与 v1 互斥的 30 图和业务语义门禁，ORT EP 失败 2 图；两次 EP 结果完全一致。
- v3 使用第三组互斥 30 图验证第一版 native precision policy，失败 1 图；同样保留。
- E0 只在已消耗的 v1+v2+v3 共 90 图上选策略。完成后冻结从未推理的 v4 第四组 30 图；v4 不允许失败后再改 engine、样本或阈值。

门禁要求 detection count、class id/name 精确相等，再按同类最大 IoU 贪心匹配；confidence 绝对误差不超过 0.005、任一 bbox 坐标绝对误差不超过 1 像素、IoU 不低于 0.90。CPU-vs-A、CPU-vs-B、A-vs-B 三组都过，才设置 `performance_publication_allowed=true`。这证明 bounded implementation correctness，不等于全数据集 mAP，也不是 PyTorch/ONNX/C++ lineage。

v4 的 pre-freeze 顺序由文件系统 mtime 观察和后补哈希清单记录，不是同时签名或 commit 的密码学证明；正式结果对 config 的解析语义有记录，但精确的“推理当时 config 文件字节”没有 contemporaneous hash。收口中保留这一证据边界，不把它包装成更强的 anti-tuning 证明。

#### 性能和内存怎样读

产品 `session_run` 对 native 表示 H2D、`enqueueV3`、D2H 和 stream synchronize，不是纯 GPU kernel；`pipeline` 再包含 preprocess、output validation/copy 和 postprocess；`end_to_end` 还包含每次 `imread`。`trtexec` 的 GPU compute 是更低层边界，不能与产品 pipeline 混写。

初始化只测一次，没有 P50/P95；它包括 model/engine 读取与 SHA、反序列化、context/stream/buffer 建立。host peak RSS 是进程生命周期高水位。WSL 下目标 PID 没有可用的 per-process GPU memory 行，因此采样工具回退到指定 GPU 的 device-wide `memory.used`；155 MiB 可能包含同时期其他 GPU consumer。

A/B detection 字节一致，pipeline throughput 只差约 2.1%，说明功能与平均吞吐可重复；但 session P95 从 5.329 到 7.468 ms、pipeline P95 从 8.779 到 10.490 ms，未锁频 Laptop GPU 的 tail 不稳定。`trtexec` reload 的 GPU-compute coefficient of variation 也达到 24.18%，所以不发布受控时钟或稳定尾延迟结论。

#### 完成判定和未做项

验收中的 `trtexec`、真实 C++ TensorRT path、实际 FP16、重复 correctness、初始化/P50/P95/throughput/RSS/GPU memory 和原 Windows/Linux ORT gate 均有真实证据。native backend 因 ORT EP correctness 失败而成为必要实现，不再是可选扩展。TensorRT INT8 没有做：当前 FP32 artifact 没有 calibration cache/representative-set/QDQ scale 的冻结契约，临时 INT8 会扩张主链且难以解释，不阻塞本单元完成。

## 2. 新增或修改的模块与设计原因

| 模块 | 设计与输入输出 | 关键 trade-off / 异常语义 |
|---|---|---|
| `RuntimeConfig` v2 | 增加 `cpu`、`tensorrt`、`tensorrt_native`，device、precision、workspace、cache、plan path/SHA | v1 CPU 保持兼容；provider 专属字段混用、负 device、缺 cache/SHA 会在 session 前失败 |
| ORT TRT EP session | 注册 TensorRT → CUDA → CPU，启用 FP16/cache/timing cache | 集成轻、复用 ORT；无法在 ORT 1.20.1 精确排除敏感 op |
| `NativeTensorRtRunner` | load-only plan，持久 stream/buffers，返回 owned FP32 output | 获得逐层精度控制；plan 与 TensorRT/CUDA/SM 强绑定且无 fallback |
| `OnnxRunner` 统一适配 | 对上层继续暴露 metadata、run、timed run | 类名历史上仍叫 OnnxRunner，但 native path 不执行 ONNX parser；provider evidence 是实际权威 |
| metadata/provider contract | 校验 provider chain、engine/cache、policy、I/O 和 runtime identity | 防止“配置写 TRT、实际跑别处”；严格不匹配会拒绝启动 |
| correctness 工具链 | 冻结协议/manifest、产品批量单图运行、三向比较、raw tree provenance | bounded set 不是 mAP；失败证据不可覆盖，v4 才允许发布性能 |
| benchmark wrapper | 验证 provider/model/image/engine/cache，采集 RSS 与 GPU memory | device-wide GPU memory 不是 PID 独占；P50/P95 只对应固定协议 |
| CMake GPU build | 隔离 ORT GPU 1.20.1、TRT 10.4.0.26、CUDA 12.6，设置 RPATH | GPU 依赖不污染原 CPU build；只支持 Linux x86_64 |

BatchRunner 仍明确拒绝非 CPU provider，避免把原 S2-03 “每 worker 独占 ORT session”误当成已验证的 GPU 并发模型。S2-04 的正式性能是单 session、batch=1。

## 3. 文件变化、目录树与职责

```text
cpp_infer/
├─ configs/
│  ├─ tensorrt_fp16_config.txt                 # ORT EP 诊断路径
│  ├─ tensorrt_native_fp16_config.txt          # 已消耗的 native v3 失败策略
│  └─ tensorrt_native_fp16_config_v4.txt       # 当前 E0 发布配置
├─ protocols/
│  ├─ s2_04_tensorrt_fp16_protocol{,_v2}.json  # ORT EP v1/v2
│  └─ s2_04_tensorrt_native_fp16_protocol_v{3,4}.json
├─ src/
│  ├─ native_tensorrt_runner.{h,cpp}           # native plan ownership/execution
│  ├─ onnx_runner.cpp                          # CPU、ORT EP、native 统一适配
│  ├─ config_loader.cpp                        # schema v2 typed config
│  ├─ model_metadata.cpp                       # provider/engine/policy gate
│  └─ benchmark_*.cpp / result_writer.cpp      # provider-aware evidence
├─ tools/
│  ├─ cuda_runtime_smoke.cpp
│  ├─ summarize_s2_04_ort_profile.py
│  ├─ run_s2_04_product_set.py
│  ├─ compare_s2_04_correctness.py
│  └─ run_s2_04_gpu_benchmark.py
├─ tests/
│  ├─ fixtures/s2_04_correctness_manifest_v{2,3,4}.json
│  ├─ test_s2_04_evidence_tools.py
│  └─ test_run_s2_04_product_set.py
└─ results/s2_04/linux_x86_64_rtx4060/
   ├─ trtexec/       # 当前 ONNX unconstrained FP16 build/reload
   ├─ profile/       # ORT EP provider-attributed trace
   ├─ native_engine/ # E0 plan build/layer/reload manifest
   ├─ correctness/   # v1/v2/v3 失败、E0 screen、v4 通过/provenance
   └─ benchmark/     # same-SDK CPU、native warm A/B
```

根 `AGENTS.md`、双语 README、`cpp_infer/README.md` 和 `docs/paths_commands.md` 同步当前状态；本文件承担完整九部分教学闭环。离线 engine 与 cache 在 `.gitignore` 中，不提交 21 MB plan。

## 4. 不使用 Codex 时的人工实现流程

1. 固定事实：记录 GPU、driver、WSL/kernel、CUDA、TensorRT、ORT、OpenCV、compiler；对当前 ONNX 重新计算 SHA、读取 I/O/opset。
2. 先验证 CUDA：`nvidia-smi`、动态库 `ldd`、CUDA C++ malloc/memset/copy/free smoke。
3. 用 `trtexec --fp16` 直接 build/save/reload 当前 ONNX，保留 console、times、profile 和 layer info；这一步只证明 parser/runtime。
4. 扩展 typed RuntimeConfig，并建立隔离的 Linux x86_64 GPU build；保持 CPU build 和 artifact contract 不变。
5. 在 `OnnxRunner` 注册 TRT→CUDA→CPU，加入 cache namespace、RAII options 和 actionable diagnostics；用 ORT profile 证明真实 placement。
6. 在任何正式比较前冻结 v1 protocol/manifest；运行 CPU、TRT A、TRT B。失败就保留，不后改阈值。
7. 用互斥 v2 再验证业务门禁。若 EP 仍失败且当前版本无法控制敏感 op，进入预案，而不是宣称 EP 通过。
8. 实现 load-only native runner：engine SHA、版本/SM、I/O、buffer/stream、enqueueV3、owned output、no fallback。
9. 冻结并运行 v3。若 precision policy 失败，保留 v3；只能在已消耗样本上筛选下一策略。
10. E0 在旧 90 图通过后，冻结从未运行的 v4；运行 CPU/A/B 和三向 comparator。只有 v4 通过才发布 benchmark。
11. 用同一 GPU-enabled binary 做 same-SDK CPU/native、相同图、batch=1、warmup/repeat 相同的 A/B benchmark，分别解释 initialization、session、pipeline、RSS 和 device-wide GPU memory。
12. clean 构建并跑 Windows/Linux CPU 全门禁，再重建 GPU binary 做 inspect/raw/detect smoke；最后做 JSON、链接、diff 和证据审计。

## 5. 入口、核心类/函数、ownership 与宏观伪代码

主要入口是 `main()` 读取 CLI 后调用 `load_runtime_contract()`，再构造 `DetectorPipeline` 或 `BenchmarkRunner`。核心对象关系是：

```text
CLI owns RuntimeContract
DetectorPipeline owns OnnxRunner
OnnxRunner::Impl owns either
  Ort::Env + SessionOptions + Ort::Session
or
  NativeTensorRtRunner::Impl
    owns runtime + engine + context + stream + device buffers
returned InferenceOutput owns host output vector
DetectionResult owns postprocessed detections + provider evidence
```

宏观伪代码：

```cpp
contract = load_runtime_contract(config_path);
pipeline = DetectorPipeline(contract);

image = decode(image_path);
pre = letterbox_rgb_fp32_nchw(image, contract.artifact);

if (provider == cpu || provider == tensorrt_ep) {
  output = ort_session_run(pre.values);
} else {
  verify_engine_sha_runtime_gpu_io_policy();
  cudaMemcpyAsync(H2D);
  context.setTensorAddress(input, output);
  context.enqueueV3(stream);
  cudaMemcpyAsync(D2H);
  cudaStreamSynchronize(stream);
}

detections = decode_nms_restore(output, pre.geometry, artifact);
write_json(detections, actual_provider, provider_evidence);
```

初始化与运行错误都必须落到清晰边界：config parse、artifact/model SHA、provider availability、shared library loader、engine SHA/version/SM、TensorRT I/O、CUDA transfer、enqueue、output finite/shape 或 correctness gate。不能用一个笼统的“GPU failed”吞掉定位信息。

## 6. 运行、测试、调试、调参与证据

### 6.1 环境与 GPU build

以下变量只示意机器本地路径，不写入 tracked config：

```bash
export REPO=/mnt/d/01_Base/CodingSpace/yolo_defect
export ORT_GPU="$HOME/.local/opt/onnxruntime-linux-x64-gpu-1.20.1"
export TRT_ROOT="$HOME/.local/opt/tensorrt-10.4.0.26-cuda12.6"
export CUDA_BUNDLE="$HOME/.local/opt/cuda-12.6.1-cudnn-9.4.0.58"
export CUDA_ROOT="$CUDA_BUNDLE/usr/local/cuda-12.6"
export BUILD="$HOME/.local/state/yolo-defect-s2-04/build_gpu_release"
export LD_LIBRARY_PATH="$ORT_GPU/lib:$TRT_ROOT/usr/lib/x86_64-linux-gnu:$CUDA_ROOT/targets/x86_64-linux/lib:$CUDA_BUNDLE/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S "$REPO/cpp_infer" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DYOLO_DEFECT_REQUIRE_TENSORRT_EP=ON \
  -DYOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND=ON \
  -DONNXRUNTIME_ROOT="$ORT_GPU" -DTENSORRT_ROOT="$TRT_ROOT" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
cmake --build "$BUILD" --parallel 4
```

### 6.2 accepted E0 精确复跑、fresh candidate rebuild 与产品 smoke

```bash
# 精确复跑只加载机器本地保留的 accepted E0，不重建或覆盖它。
test "$(sha256sum <accepted-E0-engine> | awk '{print toupper($1)}')" = \
  E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746
"$TRT_ROOT/usr/src/tensorrt/bin/trtexec" \
  --loadEngine=<accepted-E0-engine> --warmUp=1000 --iterations=100 \
  --duration=0 --profilingVerbosity=detailed --separateProfileRun \
  --exportTimes=reload_times.json --exportProfile=reload_profile.json \
  --exportLayerInfo=reload_layers.json

# 重建始终写 fresh candidate；不得把输出直接指向 accepted E0 cache。
CANDIDATE=<fresh-run-root>/candidate.engine
TIMING_CACHE=<fresh-run-root>/candidate.timing.cache
test ! -e "$CANDIDATE"
"$TRT_ROOT/usr/src/tensorrt/bin/trtexec" \
  --onnx=<SHA-bound-best.onnx> --fp16 --noTF32 \
  --memPoolSize=workspace:2048M --precisionConstraints=obey \
  '--layerPrecisions=*:fp32,/model.22/dfl/Softmax:fp16' \
  --timingCacheFile="$TIMING_CACHE" --saveEngine="$CANDIDATE" \
  --profilingVerbosity=detailed --exportLayerInfo=build_layers.json --skipInference
sha256sum "$CANDIDATE"

"$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/tensorrt_native_fp16_config_v4.txt" \
  --inspect-model
"$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/tensorrt_native_fp16_config_v4.txt" \
  --image "$REPO/data/images/val/crazing_241.jpg" --raw-output-summary
```

accepted E0 plan 受 `.gitignore` 保护且未随仓库分发；fresh clone 无法从
tracked files 推导出逐字节相同的 plan。TensorRT rebuild 也不承诺相同
SHA。新 candidate 只要 SHA 不同，就必须建立新的 engine identity、
versioned config/protocol，并用未消费的新 holdout 重新过正确性门禁；不能
把它放入 v4 config 后沿用既有 v4 结论。完整无占位路径见
[`../paths_commands.md`](../paths_commands.md) 第 11.3 节。

正式 correctness 用以下三次 product-set 与一次 comparator；每个 `$RUN/*` 目标必须预先不存在：

```bash
MODEL_SHA=7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68
ENGINE_SHA=E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746
MANIFEST="$REPO/cpp_infer/tests/fixtures/s2_04_correctness_manifest_v4.json"
RUN="$HOME/.local/state/yolo-defect-s2-04/rerun_v4"

python3 "$REPO/cpp_infer/tools/run_s2_04_product_set.py" --cli "$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/default_config.txt" --manifest "$MANIFEST" \
  --output-dir "$RUN/cpu" --expected-model-sha256 "$MODEL_SHA" \
  --expected-actual-provider CPUExecutionProvider
python3 "$REPO/cpp_infer/tools/run_s2_04_product_set.py" --cli "$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/tensorrt_native_fp16_config_v4.txt" --manifest "$MANIFEST" \
  --output-dir "$RUN/native_a" --expected-model-sha256 "$MODEL_SHA" \
  --expected-actual-provider TensorRTNative --expected-engine-sha256 "$ENGINE_SHA"
python3 "$REPO/cpp_infer/tools/run_s2_04_product_set.py" --cli "$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/tensorrt_native_fp16_config_v4.txt" --manifest "$MANIFEST" \
  --output-dir "$RUN/native_b" --expected-model-sha256 "$MODEL_SHA" \
  --expected-actual-provider TensorRTNative --expected-engine-sha256 "$ENGINE_SHA"
python3 "$REPO/cpp_infer/tools/compare_s2_04_correctness.py" \
  --protocol "$REPO/cpp_infer/protocols/s2_04_tensorrt_native_fp16_protocol_v4.json" \
  --cpu-dir "$RUN/cpu" --tensorrt-run-a-dir "$RUN/native_a" --tensorrt-run-b-dir "$RUN/native_b" \
  --expected-cpu-provider CPUExecutionProvider --expected-tensorrt-provider TensorRTNative \
  --summary-output "$RUN/summary.json" --per-image-output "$RUN/per_image.json"
```

正式 benchmark 用 GPU wrapper 包装产品的 `--benchmark --warmup 10 --repeat 100`：

```bash
CACHE="$REPO/cpp_infer/.cache/tensorrt_native/7b8a37610018a6ae_trt10_4_cuda12_6_sm89_fp16_dfl_softmax_fp32_else_no_tf32"
python3 "$REPO/cpp_infer/tools/run_s2_04_gpu_benchmark.py" --backend-mode native \
  --benchmark-json "$RUN/source_benchmark.json" --output "$RUN/gpu_benchmark.json" \
  --cache-dir "$CACHE" -- \
  "$BUILD/bin/yolo_defect_cpp" \
  --config "$REPO/cpp_infer/configs/tensorrt_native_fp16_config_v4.txt" \
  --image "$REPO/data/images/val/crazing_241.jpg" --benchmark --warmup 10 --repeat 100 \
  --benchmark-json "$RUN/source_benchmark.json"
```

工具会拒绝覆盖 protocol、manifest、model、image、config、engine、cache 或输入结果目录。完整环境、engine 与复跑命令同时记录在 [`../paths_commands.md`](../paths_commands.md) 第 11 节。

### 6.3 CPU 回归与工具测试

```powershell
cpp_infer\tools\stage1.cmd clean-build
cpp_infer\tools\stage1.cmd test
```

```bash
cmake --build <linux-cpu-release-build> --parallel 4
ctest --test-dir <linux-cpu-release-build> --output-on-failure
python3 -m unittest \
  cpp_infer.tests.test_s2_04_evidence_tools \
  cpp_infer.tests.test_run_s2_04_product_set
```

最终 Windows x86_64 和 WSL2/Linux x86_64 均为 179/179 CTest 通过；Windows 两个需要 symlink/reparse privilege 的 case 按平台报告 skipped，不计失败。S2-04 Python evidence/product-set 工具为 35/35 通过。最终 GPU 源码重建后，inspect-model、raw `enqueueV3` 和 3-detection 单图 JSON smoke 均通过。

### 6.4 证据索引

- 当前 ONNX unconstrained FP16 `trtexec`：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/trtexec/`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/trtexec/)
- ORT EP 真实节点执行：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/profile/provider_summary.json`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/profile/provider_summary.json)
- 最终 E0 build/layer/reload：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/native_engine/engine_manifest.json`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/native_engine/engine_manifest.json)
- v1/v2/v3 失败与 v4 通过：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/)
- v4 正式 summary：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/native_fp16_v4_summary.json`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/native_fp16_v4_summary.json)
- E0 旧 90 图 screen：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/native_fp16_v4_tuning_screen.json`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/correctness/native_fp16_v4_tuning_screen.json)
- same-SDK CPU/native A/B：[`../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/benchmark/`](../../cpp_infer/results/s2_04/linux_x86_64_rtx4060/benchmark/)

### 6.5 常见失败诊断

| 现象 | 先看什么 | 动作 |
|---|---|---|
| provider `.so` load failed | `ldd` 是否 `not found`，ORT/TRT/CUDA/cuDNN major 是否匹配 | 修正 isolated `LD_LIBRARY_PATH`，不要混入另一套 CUDA |
| `trtexec` parser/build failed | ONNX SHA、opset、静态 shape、首个 parser error | 先在同一 ONNX 上独立复现，不绕到产品代码 |
| provider 名是 TRT 但怀疑 fallback | ORT trace 的 provider-attributed kernel events | registration 只证明意图；以 profile placement 为准 |
| native plan 拒绝加载 | engine SHA、TRT 10.4.0、CUDA 12.6、SM 8.9、I/O | 恢复精确 plan 或按新环境重建，不能复用不兼容 engine |
| correctness 失败 | failed sample、count/class、confidence、bbox、IoU | 保留失败协议和结果；不要看完结果再放宽 gate |
| A/B P95 波动 | GPU clock/thermal/system load、trtexec CV | 区分 correctness/throughput repeat 与 tail stability |
| Windows 增量构建出现随机 enum/heap 错误 | 是否刚改变 public struct/header 且复用旧 NMake tree | 用受保护的 `stage1.cmd clean-build` 清理 TEMP ABI 残留，再重跑全 gate |
| GPU memory 是 0 或无 PID row | `compute-apps` 是否暴露目标 PID | 明确回退为 device-wide baseline/peak，不伪装进程显存 |

官方语义参考： [ORT TensorRT EP](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)、[ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)、[TensorRT precision control](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)、[TensorRT accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)、[CUDA on WSL](https://docs.nvidia.com/cuda/archive/13.0.2/wsl-user-guide/index.html)。

## 7. 面试验收问题与连续追问

1. **怎样证明不是“配置了 TensorRT、实际跑 CPU”？** 先说明 provider chain，再给 ORT trace 中 10 个 TRT event、0 CUDA/CPU fallback；最终 native 路径还给出 `enqueueV3`、engine SHA 和 no-fallback provider evidence。
2. **为什么 ORT EP 能跑却不能作为最终路径？** v1/v2 两个预冻结正确性协议都失败；重复结果一致只证明稳定，不证明与 CPU 检测语义兼容。
3. **为什么不直接放宽 bbox/confidence 阈值？** v2 已是业务语义门禁；结果出来后改阈值会让 holdout 失效。正确动作是保留失败并改变 backend/precision policy，再用新互斥 holdout。
4. **native backend 为什么仍复用 `OnnxRunner`？** 这是现有推理抽象。内部可委托 ORT session 或 native runner，上层始终得到相同 owned tensor，避免复制前后处理。
5. **什么资源归谁所有？** native Impl 独占 TensorRT runtime/engine/context、stream 和 device buffers；调用方 input 只借用到同步返回；output vector 转移给返回对象。
6. **FP16 到底在哪里？** 只有 DFL Softmax 是 Half compute，两个 reformat 触及 Half；其余 compute/I/O FP32、noTF32。不能称全 FP16。
7. **为什么约 16–17 倍不能说成 FP16 speedup？** 对照同时改变了 ORT CPU 到 native TensorRT/GPU、fusion/tactic/transfer；实验没有隔离唯一 FP16 layer 的贡献。
8. **P50/P95、throughput 为什么同时要报？** P50 代表典型延迟，P95 暴露尾部抖动，throughput 由 mean pipeline latency 得出；A/B 平均吞吐接近但 P95 不稳定。
9. **155 MiB 是模型显存吗？** 不是。它是 device-wide `nvidia-smi memory.used` 的 baseline-to-peak，目标 PID 行不可用，可能含其他 consumer。
10. **engine cache 为什么不能提交后到处运行？** TensorRT plan 绑定 TRT/CUDA、GPU compute capability、builder policy 和 tactic；环境变化必须重建并重新过正确性。
11. **v4 如何防止调参污染？** E0 只看已消耗 90 图；v4 样本、gate、engine SHA 在首个 v4 inference 前冻结。证据顺序基于 filesystem observation，未夸大成同时签名的密码学证明。
12. **为什么不做 INT8？** 当前 FP32 artifact 缺少冻结 calibration/QDQ contract；可选 INT8 的收益不足以合理扩张已经通过的主链。

能连续回答“ORT EP 为什么失败、native 怎么管理资源、precision 怎样冻结、correctness 为什么先于性能、每个时间/内存指标的边界是什么”，才算达到本单元面试理解深度。

## 8. 最可能被追问并应进入代码练习的代码

| 练习点 | 文件与当前行号 | 应能手写/解释的内容 |
|---|---|---|
| typed provider/config gate | `cpp_infer/src/config_loader.cpp:52,141,248` | provider parsing、专属字段、不变量与 actionable error |
| ORT/native 统一分派 | `cpp_infer/src/onnx_runner.cpp:611,620,663,728` | PImpl ownership、native delegation、TRT→CUDA→CPU 注册 |
| native 初始化/identity | `cpp_infer/src/native_tensorrt_runner.cpp:353,541` | engine SHA、runtime/SM/I/O、metadata/provider evidence |
| native run | `cpp_infer/src/native_tensorrt_runner.cpp:639-660` | H2D、tensor address、enqueueV3、D2H、同步和异常安全 |
| metadata gate | `cpp_infer/src/model_metadata.cpp:26,58,119,301` | CPU/EP/native 三类 provider chain 与 E0 execution policy |
| pipeline 复用 | `cpp_infer/src/detector_pipeline.cpp:109` | provider evidence 如何进入不变 DetectionResult |
| benchmark boundary | `cpp_infer/src/benchmark_runner.cpp:169,424,517` | session/pipeline/end-to-end、native timing scope、RSS |
| correctness matcher | `cpp_infer/tools/compare_s2_04_correctness.py:733,803,818,936` | 同类 IoU matching、三向聚合、protocol/source protection |
| GPU evidence wrapper | `cpp_infer/tools/run_s2_04_gpu_benchmark.py:147,654,795` | provider/engine/cache验证、GPU memory fallback、证据组装 |
| CMake isolation | `cpp_infer/CMakeLists.txt:68,225,231,430` | GPU SDK exactness、Linux x86_64 gate、RPATH、CPU build 隔离 |

建议练习顺序：先独立写 config/provider state machine，再写一个 RAII CUDA buffer/stream 小类，再写同步 `enqueueV3`，最后手写 detection matcher 与 nearest-rank percentile。不要先背 API 名；先讲清 ownership、失败边界和为什么 gate 不可后改。

## 9. 入口文档同步状态

- `AGENTS.md`：已把 S2-04 标为实现、双 CPU gate、GPU correctness/performance/证据与教学收口完成，当前停止等待 L1；S2-05 未开始。
- `README.md`：已同步英文项目状态、真实 native TensorRT path、关键结果、限制和证据入口。
- `README_zh.md`：已同步同一中文状态与事实边界。
- `cpp_infer/README.md`：已补充 GPU build/config/产品命令、ORT EP 失败到 native v4 的决策、指标和证据路径。
- `docs/paths_commands.md`：已记录当前隔离 SDK 版本/入口、GPU build、`trtexec`、smoke、正确性与 benchmark 命令，以及 WSL/plan/cache 陷阱。
- `docs/Proj1_S2.md`：按仓库指令保持不改；它是路线规划文档，不作为本次实测结果的写入目标。

收口结论：S2-04 完整单元结束，验收项全部有真实证据；INT8 作为非阻塞可选实验未做。下一动作只能由用户 L1 触发，本次不自动进入 S2-05。
