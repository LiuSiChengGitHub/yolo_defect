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

> **Status — 2026-08-31:** Large Stage One and user-owned L2 are complete.
> S2-01, S2-02, and S2-03 now have a consolidated Round 2 QDQ/U8S8 closure:
> the formal artifact `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2` (SHA-256
> `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`)
> runs through the shared C++ single-image and bounded multi-image chain on
> Windows and WSL2/Linux x86_64, and through the cross-built AArch64 chain under
> QEMU user-mode. Final Windows and Linux gates passed 157/157 CTest entries;
> default FP32 Linux and AArch64/QEMU regressions also passed. Work stops for
> user L1; S2-04 has not started. QEMU is not an ARM board, so its performance
> and memory fields are not publishable native-device evidence.

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
  -> [integration delivered] formal U8S8 through Linux + AArch64/QEMU batch
  -> Linux x86_64 + RTX 4060 TensorRT path
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
.\cpp_infer\tools\stage1.cmd batch-compare -Config cpp_infer\configs\int8_u8s8_config.txt
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
bash cpp_infer/tools/stage1.sh batch-compare --config cpp_infer/configs/int8_u8s8_config.txt
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

export YOLO_DEFECT_AARCH64_CONFIG="$PWD/cpp_infer/configs/int8_u8s8_config.txt"
bash cpp_infer/tools/stage2_aarch64.sh all
unset YOLO_DEFECT_AARCH64_CONFIG
```

The complete action matrix, current dependency paths, local-configuration
precedence, low-level CMake/CTest audit commands, and environment troubleshooting
live only in [Paths, toolchains, and environment diagnosis](docs/paths_commands.md).
The exact Gate A machine snapshot, evidence, and interpretation are in the
[S2-02 Gate A closure](docs/details/s2_02_gate_a_closure.md); Gate B's host/target
boundary and QEMU evidence are in the
[S2-02 Gate B closure](docs/details/s2_02_gate_b_closure.md). S2-03's design,
three-platform functional evidence, and same-platform performance comparisons
are consolidated in the [S2-03 closure](docs/details/s2_03_closure.md).

## 5. Core Modules

| Boundary | Responsibility |
|---|---|
| Runtime/artifact/metadata contracts | Separate adjustable runtime policy, declared artifact semantics, and actual ORT-observed tensor/provider facts; reject mismatches before inference |
| `ImagePreprocessor` | Decode or accept a `CV_8UC3` image; letterbox, BGR-to-RGB, normalize, produce contiguous NCHW data, and retain inverse-transform metadata |
| `OnnxRunner` / `InferenceOutput` | Own ORT resources with RAII/PImpl, validate input/output, run synchronously, and copy output into an ORT-independent lifetime |
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
| S2-01 | Static INT8 PTQ, FP32/INT8 correctness/task-quality/performance comparison, ORT operator/node profiling | Product/quality results remain advisory; no QAT or profiler-as-benchmark claim | **Round 2 U8S8 integrated into the cross-platform multi-image chain; awaiting L1** |
| S2-02 | Linux x86_64 native chain, shared-source portability, AArch64 cross-build and QEMU smoke | WSL2/QEMU is not a board; QEMU produces no performance claim | **U8S8 Linux/AArch64 functional integration complete; awaiting user L1** |
| S2-03 | Directory/manifest discovery, bounded queue, workers, backpressure, failure accounting, clean shutdown, throughput comparison | Concurrent single-image work is not true ONNX batch; QEMU is functional evidence only | **FP32 closure and formal U8S8 integration evidence complete; awaiting user L1** |
| S2-04 | One real Linux x86_64 + RTX 4060 TensorRT execution path, FP16 correctness and performance | Local GPU/edge-node evidence only; not Jetson or embedded deployment | Not started |
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

### S2-01/S2-02/S2-03 U8S8 Integration Record

The integration keeps `RuntimeConfig`, `ModelArtifactSpec`, `DetectorPipeline`,
postprocessing, and `BatchRunner` unchanged. The U8S8 graph retains float32
external I/O, so selecting `cpp_infer/configs/int8_u8s8_config.txt` changes the
model artifact while preserving the established single-image and batch
contracts. The default configuration remains FP32.

| Integration evidence | Recorded result |
|---|---|
| Formal identity | `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`; SHA-256 `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1` |
| Windows x86_64 | Final Release gate `157/157`; the two permission-dependent symlink/reparse GTest cases were skipped locally and their Linux counterparts passed; the U8S8 fixed image produced 3 detections |
| WSL2/Linux functional correctness | Final Release gate `157/157`; U8S8 fixed image produced 3 detections; the 30-image manifest at workers=2/queue=4 completed 30/30 with queue peak 4 and 25 producer waits |
| WSL2/Linux 361-image comparison | U8S8 CPU, JSON-only, queue=8: workers=1 `4.591151 img/s`, peak RSS `192.933594 MiB`, 353 waits; workers=4 `15.903088 img/s`, peak RSS `556.882812 MiB`, 350 waits; ratio `3.463857`; all 361 outputs were byte- and semantic-equal |
| Linux AArch64/QEMU | Clean cross-build plus ELF/loader checks passed; U8S8 fixed image produced 3 detections; directory workers=1 and manifest workers=2 each completed 2/2 with identical outputs; corrupt input produced exactly 2 successes + 1 failure, exit 2, with queue=1 and one producer wait |
| FP32 regression | The default FP32 Linux Demo and AArch64/QEMU `all` workflow passed after the selectable-config change |

The new Linux comparison ran from the repository under `/mnt/d` (DrvFs), not
the WSL-native ext4 work area used by the older FP32 comparison. Its throughput
and peak-RSS values are only a within-run WSL2/Linux U8S8 workers=1 versus
workers=4 comparison; they must not be ranked directly against the old ext4
FP32 numbers. QEMU timing and RSS fields remain explicitly non-publishable and
prove only AArch64 functional portability. The S2-01 advisory quality facts also
remain unchanged: agreement precision is `0.938462 < 0.95`, and mAP50 drop is
`0.010356 > 0.01`; the strict quality result is not rewritten as fully green.

Machine-readable integration evidence is under
[`cpp_infer/results/s2_03/int8_integration/`](cpp_infer/results/s2_03/int8_integration/).
The frozen scope is recorded in the
[integration SPEC](docs/details/s2_int8_arm64_batch_integration_spec.md).

### Platform Matrix

| Platform/backend | What it proves | Current status |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | Current product chain, single-image and bounded multi-image correctness, historical same-platform throughput/PWS comparison | **Current 157/157 regression passed** |
| Windows x86_64 + ORT CPU INT8 | Static PTQ, Runtime legality, advisory quality/performance/profile evidence, and fixed-image product inference | **Round 2 U8S8 fixed image verified; included in the 157-test gate** |
| WSL2/Linux x86_64 + ORT CPU INT8 | Shared product chain, fixed image, 30-image manifest, bounded workers/backpressure, and same-run 361-image correctness/throughput/RSS comparison | **Round 2 U8S8 integration verified** |
| WSL2 Ubuntu 24.04 x86_64 + ORT CPU FP32 | Linux build/load/runtime portability, single-image and bounded multi-image correctness, historical same-platform throughput/RSS comparison | **Default FP32 regression passed in the current 157-test closure** |
| Linux AArch64 under QEMU + ORT CPU FP32/U8S8 | Clean cross-build, ELF/loader, single-image and bounded multi-image functional correctness for the selected config | **Both configs verified under emulation**; not a board and no publishable performance/memory claim |
| Linux x86_64 + RTX 4060 + TensorRT | Real local TensorRT execution, FP16 correctness/performance, GPU memory | Planned in S2-04; not Jetson |

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
- The historical S2-03 comparison uses FP32 CPU and the current integration
  comparison uses U8S8 CPU; both fix the 361-image directory, JSON-only output,
  queue=8, and independent workers=1/workers=4 Release processes. Every output
  is checked before the ratio is calculated. The U8S8 run used `/mnt/d` DrvFs,
  whereas the old Linux FP32 pair used WSL-native ext4, so their absolute
  throughput/RSS values are not a direct FP32-versus-INT8 comparison.
- Windows Peak Working Set and Linux `getrusage` peak RSS are process-lifetime
  high-water metrics with platform-specific semantics; neither is model-only or
  per-inference memory, and their values are not directly comparable.
- S2-01 ORT traces prove optimized-node placement on `CPUExecutionProvider`,
  but trace durations include profiler overhead and do not identify exact CPU
  instructions selected inside a kernel.
- The current Runtime is CPU-only and each inference call remains batch=1, but
  S2-03 can process multiple images through bounded independent workers. This is
  not tensor-level true batch, video, a service, GPU concurrency, or a lock-free
  queue. QEMU does not establish board latency, throughput, memory, power,
  thermals, or deployment stability; TensorRT remains planned.
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
- [S2-01/S2-02/S2-03 U8S8 integration SPEC](docs/details/s2_int8_arm64_batch_integration_spec.md)
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
