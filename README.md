# Industrial Vision Edge AI Runtime and C++ Engineering System

[中文版](README_zh.md)

![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![CMake](https://img.shields.io/badge/CMake-enabled-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19.2-orange)
![GTest](https://img.shields.io/badge/GTest-1.17.0-red)

This repository turns an industrial-vision model artifact into configurable,
runnable, testable, comparable, and reproducible C++ inference software. Its
recruiting focus is modern C++, Linux portability, testing and debugging,
performance analysis, and model-inference engineering—not another detector
training wrapper.

YOLOv8 and NEU-DET are the stable model and dataset carriers for the first
Runtime implementation. The repository's value is the engineering chain around
that artifact: executable contracts, C++ inference, deterministic outputs,
correctness gates, benchmark evidence, and a controlled path toward Linux,
concurrency, quantization, and TensorRT.

> **Status — 2026-08-31:** Large Stage One, user-owned L2, and S2-01 through
> S2-04 are complete. S2-04 adds real TensorRT execution on WSL2/Linux x86_64
> with an RTX 4060 Laptop GPU. ORT TensorRT EP execution was proved but failed
> two frozen correctness protocols; the accepted product path is therefore a
> SHA-bound, load-only native TensorRT engine with one real FP16 DFL Softmax
> compute layer and FP32/noTF32 elsewhere. Its untouched v4 30-image holdout
> passed twice, and same-SDK CPU/native latency, throughput, host RSS, and
> device-wide GPU-memory evidence is retained. Final Windows and WSL2/Linux CPU
> gates pass 179/179 tests. Work stops for user L1; S2-05 has not started. This
> is local WSL2 GPU/edge-node evidence, not Jetson, ARM64 GPU, embedded hardware,
> or bare-metal native Linux evidence.

![Fixed inference demo](docs/assets/demo_inference_result.gif)

## 1. What the Project Solves

A model file alone does not provide a deployable software product. The Runtime
must still answer:

- Which model, tensor, preprocessing, postprocessing, and threshold contract is
  actually being executed?
- Can the same image produce deterministic, inspectable JSON and visualization?
- Do independent Python and C++ implementations agree under declared numeric
  gates?
- Are failures actionable, and are performance numbers bound to an explicit,
  machine-readable correctness policy rather than silently detached from quality?
- Can the same core later move across operating systems, architectures,
  workloads, and inference backends without duplicating product semantics?

The intended recruiting story is therefore:

> I converted an existing industrial-vision artifact into an evidence-backed
> C++ Runtime, then hardened it as a cross-platform inference and systems
> engineering project.

## 2. Architecture

### Verified Stage One Runtime

```text
RuntimeConfig + ModelArtifactSpec
                |
                v
      actual ONNX ModelMetadata
                |
                v
OpenCV decode -> letterbox -> RGB -> float32 NCHW
                |
                v
      ONNX Runtime C++ CPU Session::Run
                |
                v
          owned raw output
                |
                v
YOLO decode -> score filter -> stable NMS -> coordinate restore
                |
                v
 DetectionResult -> schema-v1 JSON + headless visualization
                |
                v
 GTest/CTest + Python/C++ consistency + Release benchmark
```

### Stage Two Target

```text
FP32 ONNX
  -> [delivered on Windows CPU] static INT8 PTQ + ORT operator/node profiling
  -> [Gate A delivered] Windows and Linux x86_64 shared-source Runtime
  -> [Gate B delivered] AArch64 cross-build + QEMU functional portability
  -> [S2-03 delivered] directory/manifest + bounded queue + workers
  -> [S2-04 delivered] WSL2/Linux x86_64 + RTX 4060 TensorRT path
  -> full evidence, resume variants, interview closure, recruiting freeze
```

The product backend may expand, but the detection contract stays stable:

```text
contract + metadata -> model-specific preprocess
                    -> ORT CPU FP32/INT8 or TensorRT
                    -> owned inference output
                    -> the same decode/filter/NMS/restore semantics
```

Core architecture rules:

- The Runtime library owns reusable behavior; the CLI stays thin.
- Windows, Linux, and AArch64 use the same business source.
- Platform differences stay in dependency discovery, dynamic libraries,
  memory/signal adapters, and workflow scripts.
- Multi-image workers reuse the existing single-image
  `DetectorPipeline`; they do not copy preprocessing, inference, or
  postprocessing.
- Before publishing a benchmark, run a representative correctness smoke under
  the same artifact/config. S2-01 additionally retains its original product and
  task-quality outcomes as non-blocking diagnostics.
- Backend abstraction remains the smallest boundary required by real backends.
- Performance claims state the artifact, sample, runtime conditions, and
  limitations needed to interpret them; routine outputs do not require a new
  evidence bundle or hash ledger.

## 3. Large Stage One Final Record

Large Stage One is complete as one consolidated deliverable. The table below
contains the facts worth carrying forward; it does not reconstruct its former
internal task sequence.

| Evidence | Final record |
|---|---|
| Model artifact | `models/best.onnx`, 12,336,935 bytes, opset 17, SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68` |
| Actual tensor contract | Input `images` float32 `[1,3,800,800]`; output `output0` float32 `[1,10,13125]`; explicit `CPUExecutionProvider` |
| Verified build environment | Windows 10.0.26200, x86_64, MSVC 19.50.35721.0, Release C++17, OpenCV 4.8.0, ONNX Runtime C++ 1.19.2 |
| Fixed Demo | `crazing_241.jpg` produces three `crazing` detections; committed JSON is parseable and the PNG is OpenCV-readable |
| Automated engineering gate | Fresh out-of-tree Release build; 106/106 CTest/GTest/CLI/Python/negative/integration cases passed |
| Python ORT/C++ ORT consistency | Frozen six-class manifest, five images per class: 30/30 images and 62/62 detections passed exact count/class matching |
| Maximum consistency errors | Confidence `8.049977111568296e-07`; bbox coordinate `9.135351561440075e-05 px`; minimum matching IoU `0.999998927116394` |
| Formal tracked CPU benchmark | Fixed image, batch 1, warmup 10/repeat 100: end-to-end mean/P50/P95 `176.553060/176.1357/196.6128 ms`, `5.664020 img/s` |
| Memory | Windows process-lifetime Peak Working Set `152.714844 MiB` |
| Failure behavior | Missing model, damaged image, invalid output parent, and invalid benchmark repeat fail nonzero with object/path, expected, actual, and corrective action; legal empty detections remain valid |
| User acceptance | Automated gate and L2 explanation/troubleshooting/modification acceptance complete |

Primary repository evidence:

- [Runtime configuration](cpp_infer/configs/default_config.txt)
- [YOLOv8 artifact contract](cpp_infer/artifacts/yolov8_neu_det.artifact.txt)
- [Frozen consistency manifest](cpp_infer/tests/fixtures/consistency_manifest.json)
- [Demo outputs](cpp_infer/results/demo/)
- [Consistency summary](cpp_infer/results/consistency/summary.json) and
  [per-image evidence](cpp_infer/results/consistency/per_image.json)
- [Formal tracked benchmark](cpp_infer/results/benchmark/yolov8_neu_det_cpu_release.json)

The detailed consolidated record, including the separate temporary closure
reproduction, is in
[Stage One closure details](docs/details/stage1_closure.md).

## 4. Quick Start

Run the Windows task runner from an ordinary PowerShell or CMD at the repository
root. It discovers and initializes the required Visual Studio environment:

```powershell
.\cpp_infer\tools\stage1.cmd help
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd batch data\images\val batch-output -Workers 4 -QueueCapacity 8
.\cpp_infer\tools\stage1.cmd batch-compare
```

In WSL2/Linux x86_64, select the pinned Linux SDKs and use the Bash entry:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON=/path/to/python
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest

bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh batch data/images/val batch-output --workers 4 --queue-capacity 8
bash cpp_infer/tools/stage1.sh batch-compare
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark
bash cpp_infer/tools/stage1.sh all
```

For Gate B on the same WSL2 x86_64 host, bootstrap target-only dependencies and
run the AArch64 workflow:

```bash
bash cpp_infer/tools/bootstrap_aarch64_deps.sh fetch
bash cpp_infer/tools/stage2_aarch64.sh doctor
bash cpp_infer/tools/stage2_aarch64.sh all
```

The complete action matrix, current dependency paths, local-configuration
precedence, low-level CMake/CTest audit commands, and environment troubleshooting
live only in [Paths, toolchains, and environment diagnosis](docs/paths_commands.md).
The exact Gate A machine snapshot, evidence, and interpretation are in the
[S2-02 Gate A closure](docs/details/s2_02_gate_a_closure.md); Gate B's host/target
boundary and QEMU evidence are in the
[S2-02 Gate B closure](docs/details/s2_02_gate_b_closure.md). S2-03's design,
three-platform functional evidence, and same-platform performance comparisons
are consolidated in the [S2-03 closure](docs/details/s2_03_closure.md). S2-04's
provider decision, native TensorRT implementation, frozen gates, measurements,
and commands are in the [S2-04 closure](docs/details/s2_04_closure.md).

## 5. Core Modules

| Boundary | Responsibility |
|---|---|
| Runtime/artifact/metadata contracts | Separate adjustable runtime policy, declared artifact semantics, and actual ORT-observed tensor/provider facts; reject mismatches before inference |
| `ImagePreprocessor` | Decode or accept a `CV_8UC3` image; letterbox, BGR-to-RGB, normalize, produce contiguous NCHW data, and retain inverse-transform metadata |
| `OnnxRunner` / `NativeTensorRtRunner` / `InferenceOutput` | Select an ORT session or the SHA-bound native TensorRT plan behind one owned-I/O boundary; validate provider/engine/input/output, run synchronously, and return an implementation-independent host lifetime |
| Static PTQ toolchain | Freeze calibration inputs and quantization configuration, run Conv-only QDQ PTQ with declared activation/weight types, inspect selected/quantized/failed nodes, validate actual metadata, and emit a derived artifact card |
| `ProfileRunner` and profile summarizer | Create an isolated profiling-enabled session, retain the ORT raw trace, aggregate node/operator/provider time and call counts, and keep trace timing outside formal benchmarks |
| Postprocessor/NMS | Validate YOLO BCN output, select class scores, apply strict filtering and stable class-agnostic NMS, then restore and clip source coordinates |
| `DetectorPipeline` and writers | Orchestrate one image and emit owned results, stable JSON, and deterministic GUI-free visualization while enforcing safe output paths |
| `BatchRunner`, `BoundedQueue`, and batch writers | Discover directory/UTF-8 manifest tasks deterministically, apply bounded backpressure, give each worker one batch=1 `DetectorPipeline`/ORT session, preserve discovery-order summaries, and emit per-image results plus `BatchSummary` |
| Cross-platform build/platform layer | Keep shared Runtime/preprocess/postprocess/Pipeline source unchanged while CMake selects Windows `.lib`/`.dll`, native Linux `.so`, or explicit ARM64 target libraries/toolchain; a thin `platform_info` adapter reports Windows Peak Working Set or Linux `getrusage` peak RSS |
| `project_core` portability smoke | Isolate standard-library-only YOLO decode/NMS/coordinate-restore behavior; Gate B cross-compiles and runs it under QEMU before the separately verified full ARM64 OpenCV/ORT path |
| Verification harness | Test meaningful seams with focused fixtures, exercise the real vertical slice sparingly, compare Python/C++ detections when relevant, and record scoped benchmark/memory results |

## 6. Stage Two Plan

Stage Two uses five complete delivery units. Each unit defines a minimum SPEC,
implements one runnable capability, performs proportional verification, records
only the results needed to explain it, updates the three project entry documents,
and then stops for L1 acceptance.

| Unit | Delivery | Honest boundary | Status |
|---|---|---|---|
| S2-01 | Static INT8 PTQ, FP32/INT8 correctness/task-quality/performance comparison, ORT operator/node profiling | Windows CPU exercise closure; product/quality results are advisory; no QAT or profiler-as-benchmark claim | **Implementation/evidence complete; awaiting L1** |
| S2-02 | Linux x86_64 native chain, shared-source portability, AArch64 cross-build and QEMU smoke | WSL2/QEMU is not a board; QEMU produces no performance claim | **Gate A/Gate B implementation and evidence complete; awaiting user L1** |
| S2-03 | Directory/manifest discovery, bounded queue, workers, backpressure, failure accounting, clean shutdown, throughput comparison | Concurrent single-image work is not true ONNX batch; QEMU is functional evidence only | **Implementation/evidence complete; awaiting user L1** |
| S2-04 | One real Linux x86_64 + RTX 4060 TensorRT execution path, FP16 correctness and performance | WSL2 local GPU/edge-node evidence only; constrained mixed FP16/FP32, not Jetson or embedded deployment | **Implementation/evidence/teaching closure complete; awaiting user L1** |
| S2-05 | Applicable full gates, result matrix, failure cases, three resume narratives, interview material, recruiting freeze | Adds no new technology stack | Planned |

### S2-01 Windows CPU Record

The final local artifact uses ONNX Runtime 1.19.2 static PTQ with `QDQ`, U8S8,
MinMax calibration, per-channel weights, and all 64 source `Conv` nodes in the
quantization target. It changes only the activation type from the Round 1 S8S8
protocol; its external contract remains float32
`images [1,3,800,800] -> output0 [1,10,13125]`; INT8 is internal graph
representation, not an integer application I/O contract.

| Evidence | FP32 | INT8 / outcome |
|---|---:|---:|
| Model file | 12,336,935 bytes | 3,544,494 bytes; **71.269% smaller** |
| Python/C++ ORT legality | Passed | Passed; finite outputs and matching actual metadata |
| S2-01 closure Windows regression | 118/118 CTest passed | Historical S2-01 count; S2-02 final regression is 119/119 |
| 361-image task quality | mAP50 `0.710815`; mAP50-95 `0.345786` | `0.700459` / `0.342379`; deltas `-0.010356/-0.003407` |
| 30-image product comparison | 62 detections | 65 detections, 61 matches; original aggregate gate `false` |
| Session initialization | `61.986 ms` | `94.858 ms`; slower one-time setup |
| `Session::Run` mean/P50/P95 | `155.106/155.124/169.639 ms` | `95.040/95.570/110.768 ms`; **38.726% faster mean** |
| Pipeline mean/P50/P95 | `163.477/163.221/182.008 ms` | `103.872/104.042/121.654 ms`; **36.461% faster mean** |
| Pipeline throughput | `6.117 img/s` | `9.627 img/s`; **57.383% higher** |
| Peak Working Set | `150.980 MiB` | `148.832 MiB`; small process-high-water change only |

Round 1 had quantized all 64 Conv nodes in the static QDQ file but ORT's
optimized S8S8 execution graph retained 57 float `Conv` nodes and produced only
7 `QLinearConv` nodes, plus 120 Q and 317 DQ calls per run. That explains its
37.16% `Session::Run` regression. Round 2 changed only activation `QInt8` to
`QUInt8`: its 10-run trace records 640 `QLinearConv` calls and no float `Conv`,
or all 64 integer convolutions per run. `QLinearConv` is now the leading
operator at 35.18% of diagnostic kernel-event time; DQ, Resize, Mul, Concat, Q,
and Sigmoid expose the next optimization opportunities. All optimized nodes
were placed on `CPUExecutionProvider`.

This is the intended learning result: model-file QDQ coverage is not sufficient
performance evidence; the optimized execution graph and an unprofiled benchmark
must confirm integer-kernel coverage. Profile event totals contain
instrumentation overhead and are never substituted for `Session::Run` timing.

`models/best.int8.qdq.u8s8.onnx` exists and was loaded by the recorded Python/C++
runs, but derived ONNX files remain intentionally Git-ignored alongside the
project’s model-license boundary. A fresh clone regenerates the exact binary
from the frozen protocol; Git carries its SHA-bound contract, card, tools, and
machine evidence rather than silently claiming to distribute the model.

Primary S2-01 evidence:

- [Round 2 PTQ protocol](cpp_infer/protocols/s2_01_ptq_protocol_r2_u8s8.json) and
  [U8S8 artifact contract](cpp_infer/artifacts/yolov8_neu_det_int8_qdq_u8s8.artifact.txt)
- [Quantization artifact card](cpp_infer/results/s2_01/round2/u8s8/quantization_report.json)
- [Unmodified correctness/quality result](cpp_infer/results/s2_01/round2/correctness_u8s8.json)
- [FP32/U8S8 benchmark comparison](cpp_infer/results/s2_01/round2/benchmark/comparison_u8s8.json)
- [FP32 profile summary](cpp_infer/results/s2_01/round2/profile/fp32_summary.json) and
  [U8S8 profile summary](cpp_infer/results/s2_01/round2/profile/int8_u8s8_summary.json)
- [Round 2 closure, failure analysis, and reproducibility details](docs/details/s2_01_round2_closure.md)
- [Round 1 S8S8 historical closure](docs/details/s2_01_closure.md)

### S2-02 Gate A Linux x86_64 Record

Gate A keeps the existing Runtime, preprocessing, postprocessing, and
`DetectorPipeline` as the single business path. CMake now selects the pinned
Windows `.lib`/`.dll` contract or Linux `libonnxruntime.so` with build RPATH;
the thin `platform_info` adapter selects Windows Peak Working Set or Linux
`getrusage` peak RSS. A standard-library-only `project_core` smoke covers YOLO
decode, class-agnostic NMS, and coordinate restore and is reused by Gate B.

| Gate A evidence | Recorded result |
|---|---|
| Linux clean Release | Final closure rerun: WSL2/Linux x86_64, Ninja, 119/119 CTest passed |
| Fixed product path | `crazing_241.jpg`, three detections, valid JSON and readable PNG |
| Python/C++ consistency | 30/30 images and 62/62 matched detections passed the frozen gates |
| Short performance smoke | Earlier Gate A samples used warmup 1 / repeat 2: end-to-end mean `135.896991 ms`, `7.358515 img/s`, peak RSS `196.570312 MiB`; the durable closure rerun with the same warmup/repeat measured `151.273896 ms`, `6.610526 img/s`, `196.757812 MiB`, confirming high variance. Benchmark was not repeated in the final functional closure |
| Dynamic loading | Nine built ELF executables checked with `ldd`; no dependency reported `not found`, and ORT resolved through the configured Linux SDK/RPATH |
| Windows regression | Final closure rerun: Release/NMake 119/119 CTest and fixed Demo passed |

The committed fixed-image outputs are under
[`cpp_infer/results/s2_02/linux_x86_64/`](cpp_infer/results/s2_02/linux_x86_64/).
Commands, the machine snapshot, and the full evidence interpretation are in
[Paths, toolchains, and environment diagnosis](docs/paths_commands.md) and the
[S2-02 Gate A closure](docs/details/s2_02_gate_a_closure.md) and the
[complete S2-02 closure](docs/details/s2_02_closure.md).

### S2-02 Gate B AArch64/QEMU Record

Gate B uses an x86_64 WSL2 host and a Linux AArch64 target. The GNU toolchain
file keeps CMake/Ninja as host tools while importing only ARM64 OpenCV and the
official ARM64 ONNX Runtime SDK from a private target tree. The production
Runtime and CLI use the same sources as Windows and native Linux; only CMake,
dependency staging, and the Bash workflow know about cross execution.

| Gate B evidence | Recorded result |
|---|---|
| Cross-build | Final closure rerun: Ninja Release generated AArch64 `project_core`, full Runtime archive, and production CLI |
| ELF/dependency proof | CLI is AArch64 ELF with `/lib/ld-linux-aarch64.so.1`; ARM64 loader resolved 138 target libraries, zero `not found`, and no x86_64 library |
| QEMU functional smoke | Startup/help, config + artifact, two actionable failures, and real decode/NMS/coordinate restore passed |
| Full emulated inference | Final closure rerun: fixed image ran through ARM64 OpenCV + ORT CPU and existing postprocess; validated JSON contains three detections |
| Native regression | WSL2/Linux x86_64 clean Release, nine `ldd` checks, and 119/119 CTest passed |
| Deliberate exclusions at S2-02 closure | No QEMU benchmark/power result, physical board, Jetson, Docker multi-arch, or then-future S2-03 work; S2-03 is now separately verified below |

Raw outputs are under
[`cpp_infer/results/s2_02/aarch64_qemu/`](cpp_infer/results/s2_02/aarch64_qemu/).
Commands and interpretation are in the
[S2-02 Gate B closure](docs/details/s2_02_gate_b_closure.md) and the
[complete S2-02 closure](docs/details/s2_02_closure.md).

### S2-03 Bounded Multi-Image Record

S2-03 keeps inference at batch=1 and reuses the existing preprocess → ORT →
postprocess → writer chain. A producer discovers tasks deterministically and
pushes only task indices into a bounded FIFO; each worker owns one
`DetectorPipeline`/ORT session. Ordinary image failures remain item-local,
SIGINT/SIGTERM requests cooperative stop, and the ordered `BatchSummary` keeps
counts, backpressure, timing, memory, outputs, errors, and an explicit stop-request
flag machine-readable. An observed stop therefore returns `cancelled`/130 even
when every item had already started and no item remains individually cancelled.

| S2-03 evidence | Recorded result |
|---|---|
| Windows x86_64 correctness | Clean Release 156/156 CTest passed; all 361 per-image detection JSON files are identical between workers=1 and workers=4 |
| Windows x86_64 formal comparison | FP32 CPU, JSON-only, queue=8: workers=1 `6.285556 img/s`, PWS `151.804688 MiB`; workers=4 `17.853923 img/s`, PWS `505.085938 MiB`; throughput ratio `2.840468` |
| WSL2/Linux x86_64 correctness | Clean Release 156/156 CTest passed; all 361 per-image detection JSON files are identical between workers=1 and workers=4 |
| WSL2/Linux x86_64 formal comparison | FP32 CPU, JSON-only, queue=8, WSL2-native ext4 work area: workers=1 `8.113806 img/s`, peak RSS `205.765625 MiB`; workers=4 `20.159584 img/s`, peak RSS `588.226563 MiB`; throughput ratio `2.484603` |
| Linux AArch64/QEMU functionality | Cross-built Runtime/CLI passed directory workers=1, manifest workers=2 with a finite queue, per-image equality, exact `2 succeeded + 1 failed` partial failure, and schema/count/target checks for `BatchSummary` |
| Honest boundary | Windows PWS and Linux RSS are compared only within their own platform. QEMU numbers are not performance or memory evidence and do not represent native ARM hardware |

Machine-readable evidence is under
[`cpp_infer/results/s2_03/windows_x86_64/`](cpp_infer/results/s2_03/windows_x86_64/),
[`cpp_infer/results/s2_03/linux_x86_64/`](cpp_infer/results/s2_03/linux_x86_64/),
and
[`cpp_infer/results/s2_03/linux_aarch64_qemu/`](cpp_infer/results/s2_03/linux_aarch64_qemu/).
The command protocol and interpretation are in
[Paths, commands, and environment](docs/paths_commands.md) and the
[S2-03 closure](docs/details/s2_03_closure.md).

### S2-04 RTX 4060 TensorRT Record

S2-04 keeps `DetectorPipeline`, preprocessing, decode, NMS, coordinate restore,
and output schemas unchanged. The first implementation registered ORT TensorRT
EP → CUDA EP → CPU EP with FP16 and engine/timing caches. `trtexec --fp16`
successfully built and reloaded the current ONNX, and an ORT trace attributed 10
kernel events to `TensorrtExecutionProvider` with zero CUDA/CPU fallback events.
However, the frozen ORT v1 and disjoint v2 detection gates both failed, so the
ORT benchmark files remain diagnostic and are not accepted performance claims.

The accepted C++ path is a minimal load-only native TensorRT backend behind the
existing `OnnxRunner` interface. It verifies the plan SHA, TensorRT/CUDA/SM 8.9
identity and tensor contract, then runs H2D → `enqueueV3` → D2H on an owned
non-default CUDA stream with persistent device buffers and no fallback. The
final E0 plan is 21,144,012 bytes, SHA-256
`E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746`.
Only `/model.22/dfl/Softmax` is FP16 compute; two adjacent reformats touch Half,
while all other compute and external I/O are FP32 with TF32 disabled. It is real
but deliberately constrained mixed precision, not a full-FP16 network.

| Accepted evidence | Recorded result |
|---|---|
| Frozen correctness | Untouched v4 holdout: 30/30 CPU-vs-native A and 30/30 CPU-vs-native B passed; 64 matched detections, max confidence error `1.0044e-5`, max coordinate error `0.032166 px`, min IoU `0.998619`; native A/B output trees are byte-identical |
| Engine reload | 100 timed `trtexec` queries passed: `301.55 q/s`; host P50/P95 `3.07379/3.53577 ms`; GPU-compute P50/P95 `2.41962/2.88257 ms` |
| Same-SDK CPU reference | ORT 1.20.1 CPU, batch=1, warmup=10/repeat=100: pipeline P50/P95 `118.436/133.059 ms`, `8.3247 img/s`, peak RSS `200.121 MiB` |
| Native warm A | Initialization `684.570 ms`; session P50/P95 `3.877/5.329 ms`; pipeline P50/P95 `6.974/8.779 ms`; `137.652 img/s`; peak RSS `384.668 MiB` |
| Native warm B | Initialization `619.423 ms`; session P50/P95 `3.633/7.468 ms`; pipeline P50/P95 `6.519/10.490 ms`; `140.555 img/s`; peak RSS `384.371 MiB` |
| Overall comparison | Native pipeline throughput is `16.5353x/16.8841x` the same-SDK ORT CPU reference. This is overall native TensorRT/GPU acceleration, not an isolated FP16 contribution |
| GPU memory | A/B device-wide `nvidia-smi memory.used` baseline-to-peak is `155 MiB`; PID-specific memory was unavailable, so this is not process- or model-exclusive VRAM |
| Repeatability boundary | Detection outputs are exact and average throughput differs by about 2.1%, but P95 differs materially; unlocked Laptop GPU tail latency is not claimed stable |

The current source also passes 179/179 CTest on Windows x86_64 and
WSL2/Linux x86_64. TensorRT INT8 was not added because the FP32 artifact lacks a
frozen representative calibration/QDQ contract and INT8 was explicitly
non-blocking. Evidence and the nine-part explanation are in the
[S2-04 closure](docs/details/s2_04_closure.md) and
[`cpp_infer/results/s2_04/linux_x86_64_rtx4060/`](cpp_infer/results/s2_04/linux_x86_64_rtx4060/).

### Platform Matrix

| Platform/backend | What it proves | Current status |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | Current product chain, single-image and bounded multi-image correctness, final 179/179 tests, formal same-platform throughput/PWS comparison | **S2-04 regression verified** |
| Windows x86_64 + ORT CPU INT8 | Static PTQ artifact, Runtime legality, size/quality/performance comparison, per-node profiling | Verified in S2-01 under advisory exercise policy |
| WSL2/Linux x86_64 + ORT CPU INT8 | Potential shared-source Linux INT8 path | Not separately exercised in Gate A; no Linux INT8 comparison claim |
| WSL2 Ubuntu 24.04 x86_64 + ORT CPU FP32 | Linux build/load/runtime portability, single-image and bounded multi-image correctness, final 179/179 tests, formal same-platform throughput/RSS comparison | **S2-04 regression verified in WSL2** |
| Linux AArch64 under QEMU | Cross-build plus single-image and bounded multi-image ARM64 ORT CPU functional correctness | **S2-03 verified under emulation**; not a board and no performance/memory claims |
| WSL2/Linux x86_64 + RTX 4060 Laptop + TensorRT | Real local `trtexec`, ORT EP diagnostic placement, accepted native `enqueueV3`, constrained FP16 correctness/performance, host RSS and device-wide GPU memory | **S2-04 verified**; not native Linux, Jetson, ARM64 GPU, or embedded hardware |

The current resume can be used without waiting for Stage Two. Completed S2
units may produce rolling resume updates; unfinished targets must never be
written as delivered results.

## 7. Evidence Boundaries and Current Limits

- The 30-image result proves implementation consistency for the same ONNX
  artifact; it is not detector mAP or a new PyTorch/ONNX/C++ three-way run.
- The matching `.pt` checkpoint is not present. The current ONNX lineage is
  owner-confirmed, not currently re-exportable from this workspace.
- The Stage One formal tracked benchmark is one `200x200` image, batch 1, warm file cache,
  one Windows CPU host, sequential ORT execution, and no CPU-affinity/priority
  lock. Gate A's Linux warmup-1/repeat-2 samples varied materially and are only
  functional performance smokes, not a formal result or a cross-OS comparison.
- S2-03's formal comparison uses the same 361-image directory, FP32 CPU,
  JSON-only outputs, queue=8, and independent workers=1/workers=4 Release
  processes on each x86_64 platform. It checks every detection JSON before
  calculating speedup; Windows and WSL2 results are not compared to each other.
- Windows Peak Working Set and Linux `getrusage` peak RSS are process-lifetime
  high-water metrics with platform-specific semantics; neither is model-only or
  per-inference memory, and their values are not directly comparable.
- S2-01 ORT traces prove optimized-node placement on `CPUExecutionProvider`,
  but trace durations include profiler overhead and do not identify exact CPU
  instructions selected inside a kernel.
- The Runtime now has CPU, diagnostic ORT TensorRT EP, and accepted load-only
  native TensorRT paths. Each call remains batch=1; S2-03's bounded workers are
  still CPU-only and must not be reinterpreted as validated GPU concurrency.
  There is no tensor-level true batch, video, service, multi-stream GPU
  scheduler, or lock-free queue.
- The accepted E0 plan has only one FP16 compute layer. The native/CPU speedup
  is an overall backend/GPU result, not an isolated FP16 gain. Native A/B P95
  differs materially, host RSS is a process high-water mark, and the reported
  155 MiB GPU delta is device-wide rather than PID-specific.
- QEMU does not establish board latency, throughput, memory, power, thermals,
  or deployment stability. S2-04 similarly establishes only local WSL2 x86_64
  GPU/edge-node behavior—not native Linux, Jetson, ARM64 GPU, or embedded use.
- Historical Python ORT `24.4/72.1 FPS` used different implementations,
  providers, hardware, samples, and timing boundaries; it is context only and
  must not be ranked against the C++ result.
- Source, model, and dataset licenses are separate checkpoints. An MIT source
  license does not automatically relicense the distributed ONNX or NEU-DET.

## 8. Historical Assets, Frozen Extensions, and References

The repository still preserves the original Python training, evaluation,
Python ONNX Runtime, FastAPI, and Docker assets. They are protected historical
baseline material, not the main V2 product path. Their former long tutorial is
preserved in the archived pre-S2 README rather than duplicated here.

`paper_detect` D010 remains a research-side artifact source. It may enter this
Runtime only after stable ONNX export, a result/artifact card, a deployment
contract, actual adapter integration, and consistency validation. Research
metrics are not C++ Runtime results.

D010 integration, Qt, local LLM, Agent workflows, and real ARM/Jetson devices
remain frozen unless recruiting evidence or a concrete job description
justifies reopening them.

Authoritative and operational references:

- [Recruiting route](docs/路线0712-new.md)
- [Stage Two top-level design](docs/Proj1_S2.md)
- [Stage One consolidated closure](docs/details/stage1_closure.md)
- [S2-01 INT8/PTQ and profiling closure](docs/details/s2_01_closure.md)
- [S2-02 Gate A Linux x86_64 closure](docs/details/s2_02_gate_a_closure.md)
- [S2-02 Gate B AArch64/QEMU closure](docs/details/s2_02_gate_b_closure.md)
- [S2-02 complete teaching closure](docs/details/s2_02_closure.md)
- [S2-03 bounded multi-image closure](docs/details/s2_03_closure.md)
- [S2-04 RTX 4060 TensorRT closure](docs/details/s2_04_closure.md)
- [Paths, commands, and environment](docs/paths_commands.md)
- [C++ Runtime technical reference](cpp_infer/README.md)

The root bilingual READMEs remain the public project entry. Detailed documents
support that story without replacing it.

## License

Repository source code is released under the [MIT License](LICENSE). The
tracked ONNX records Ultralytics AGPL-3.0 metadata, and NEU-DET redistribution
terms require separate verification before release or redistribution. See the
artifact contract and Stage One closure record for the current provenance
boundary.
