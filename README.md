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

> **Status — 2026-08-25:** Large Stage One's automated engineering gate and
> user-owned L2 acceptance are complete. The Stage Two documentation preflight
> is closed; **S2-01 implementation and machine evidence are complete and now
> await user L1**. The final locally generated S2-01 artifact is the full
> 64-Conv QDQ/S8S8 model. Product-difference and task-quality results are retained as advisory
> diagnostics under the user-approved exercise scope; strict three-layer
> acceptance is not claimed.

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
  -> Windows and Linux x86_64 shared-source Runtime
  -> AArch64 cross-build + QEMU portability smoke
  -> directory/manifest + bounded queue + workers
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
- Future multi-image workers reuse the existing single-image
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
```

The complete action matrix, current dependency paths, local-configuration
precedence, low-level CMake/CTest audit commands, and environment troubleshooting
live only in [Paths, toolchains, and environment diagnosis](docs/paths_commands.md).

## 5. Core Modules

| Boundary | Responsibility |
|---|---|
| Runtime/artifact/metadata contracts | Separate adjustable runtime policy, declared artifact semantics, and actual ORT-observed tensor/provider facts; reject mismatches before inference |
| `ImagePreprocessor` | Decode or accept a `CV_8UC3` image; letterbox, BGR-to-RGB, normalize, produce contiguous NCHW data, and retain inverse-transform metadata |
| `OnnxRunner` / `InferenceOutput` | Own ORT resources with RAII/PImpl, validate input/output, run synchronously, and copy output into an ORT-independent lifetime |
| Static PTQ toolchain | Freeze calibration inputs and quantization configuration, run Conv-only QDQ/S8S8 PTQ, inspect selected/quantized/failed nodes, validate actual metadata, and emit a derived artifact card |
| `ProfileRunner` and profile summarizer | Create an isolated profiling-enabled session, retain the ORT raw trace, aggregate node/operator/provider time and call counts, and keep trace timing outside formal benchmarks |
| Postprocessor/NMS | Validate YOLO BCN output, select class scores, apply strict filtering and stable class-agnostic NMS, then restore and clip source coordinates |
| `DetectorPipeline` and writers | Orchestrate one image and emit owned results, stable JSON, and deterministic GUI-free visualization while enforcing safe output paths |
| Verification harness | Test meaningful seams with focused fixtures, exercise the real vertical slice sparingly, compare Python/C++ detections when relevant, and record scoped benchmark/memory results |

## 6. Stage Two Plan

Stage Two uses five complete delivery units. Each unit defines a minimum SPEC,
implements one runnable capability, performs proportional verification, records
only the results needed to explain it, updates the three project entry documents,
and then stops for L1 acceptance.

| Unit | Delivery | Honest boundary | Status |
|---|---|---|---|
| S2-01 | Static INT8 PTQ, FP32/INT8 correctness/task-quality/performance comparison, ORT operator/node profiling | Windows CPU exercise closure; product/quality results are advisory; no QAT or profiler-as-benchmark claim | **Implementation/evidence complete; awaiting L1** |
| S2-02 | Linux x86_64 native chain, shared-source portability, AArch64 cross-build and QEMU smoke | WSL2 is not a board; QEMU produces no performance claim | Planned |
| S2-03 | Directory/manifest discovery, bounded queue, workers, backpressure, failure accounting, clean shutdown, throughput comparison | Concurrent single-image work is not true ONNX batch | Planned |
| S2-04 | One real Linux x86_64 + RTX 4060 TensorRT execution path, FP16 correctness and performance | Local GPU/edge-node evidence only; not Jetson or embedded deployment | Planned |
| S2-05 | Applicable full gates, result matrix, failure cases, three resume narratives, interview material, recruiting freeze | Adds no new technology stack | Planned |

### S2-01 Windows CPU Record

The final local artifact uses ONNX Runtime 1.19.2 static PTQ with `QDQ`, S8S8,
MinMax calibration, per-channel weights, and all 64 source `Conv` nodes in the
quantization target. Its external contract remains float32
`images [1,3,800,800] -> output0 [1,10,13125]`; INT8 is internal graph
representation, not an integer application I/O contract.

| Evidence | FP32 | INT8 / outcome |
|---|---:|---:|
| Model file | 12,336,935 bytes | 3,545,141 bytes; **71.264% smaller** |
| Python/C++ ORT legality | Passed | Passed; finite outputs and matching actual metadata |
| Current Windows regression | 118/118 CTest passed | FP32/INT8 profile workflow smokes passed |
| 361-image task quality | mAP50 `0.710815`; mAP50-95 `0.345786` | `0.707206` / `0.342174`; deltas `-0.003610/-0.003612` |
| 30-image product comparison | 62 detections | 65 detections, 61 matches; original aggregate gate `false` |
| Session initialization | `40.309 ms` | `94.979 ms` |
| `Session::Run` mean/P50/P95 | `139.920/141.677/156.473 ms` | `191.913/190.929/220.769 ms`; **37.16% slower mean** |
| Pipeline mean/P50/P95 | `146.927/148.779/163.921 ms` | `199.228/198.494/229.275 ms` |
| Pipeline throughput | `6.806 img/s` | `5.019 img/s` |
| Peak Working Set | `150.742 MiB` | `150.727 MiB`; effectively unchanged at process high-water scope |

The profiler was run in separate 10-call sessions and placed every optimized
node on `CPUExecutionProvider`. FP32 attributed `67.80%` of kernel-event time
to `Conv`. INT8 attributed `64.55%` to remaining `Conv`, `10.55%` to
`DequantizeLinear`, `6.18%` to `QuantizeLinear`, and only `0.47%` to
`QLinearConv`; the optimized graph grew from 294 to 683 executed nodes. This
explains the measured slowdown: file compression succeeded, but this graph/CPU
combination retained expensive convolution work and added many Q/DQ boundaries.
Profile event totals are diagnostic and include instrumentation overhead; they
are never substituted for the unprofiled `Session::Run` benchmark.

`models/best.int8.qdq.onnx` exists and was loaded by the recorded Python/C++
runs, but derived ONNX files remain intentionally Git-ignored alongside the
project's model-license boundary. A fresh clone regenerates the exact binary
from the frozen protocol; Git carries its SHA-bound contract, card, tools, and
machine evidence rather than silently claiming to distribute the model.

Primary S2-01 evidence:

- [Frozen PTQ protocol](cpp_infer/protocols/s2_01_ptq_protocol.json) and
  [INT8 artifact contract](cpp_infer/artifacts/yolov8_neu_det_int8_qdq.artifact.txt)
- [Quantization artifact card](cpp_infer/results/s2_01/quantization_report.json)
- [Unmodified correctness/quality result](cpp_infer/results/s2_01/correctness_quality_v1_failed.json)
- [FP32/INT8 benchmark comparison](cpp_infer/results/s2_01/benchmark/comparison.json)
- [FP32 profile summary](cpp_infer/results/s2_01/profile/fp32_summary.json) and
  [INT8 profile summary](cpp_infer/results/s2_01/profile/int8_summary.json)
- [Advisory exercise-completion record](cpp_infer/results/s2_01/exercise_completion.json)
- [S2-01 closure and reproducibility details](docs/details/s2_01_closure.md)

### Platform Matrix

| Platform/backend | What it proves | Current status |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | Current product chain, correctness, tests, segmented benchmark, Peak Working Set | Verified |
| Windows x86_64 + ORT CPU INT8 | Static PTQ artifact, Runtime legality, size/quality/performance comparison, per-node profiling | Verified in S2-01 under advisory exercise policy |
| WSL2/Linux x86_64 + ORT CPU INT8 | Shared-source Linux load/runtime portability, comparison, and peak RSS | Planned in S2-02 |
| WSL2 Ubuntu 24.04 x86_64 + ORT CPU | Linux build/load/runtime portability, consistency, benchmark, peak RSS | Planned in S2-02 |
| Linux AArch64 under QEMU | Cross-compilation and portability correctness only | Planned in S2-02; no performance claims |
| Linux x86_64 + RTX 4060 + TensorRT | Real local TensorRT execution, FP16 correctness/performance, GPU memory | Planned in S2-04; not Jetson |

The current resume can be used without waiting for Stage Two. Completed S2
units may produce rolling resume updates; unfinished targets must never be
written as delivered results.

## 7. Evidence Boundaries and Current Limits

- The 30-image result proves implementation consistency for the same ONNX
  artifact; it is not detector mAP or a new PyTorch/ONNX/C++ three-way run.
- The matching `.pt` checkpoint is not present. The current ONNX lineage is
  owner-confirmed, not currently re-exportable from this workspace.
- The formal benchmark is one `200x200` image, batch 1, warm file cache, one
  Windows CPU host, sequential ORT execution, and no CPU-affinity/priority lock.
- Peak Working Set is a process-lifetime high-water mark, not model-only or
  per-inference memory.
- S2-01 ORT traces prove optimized-node placement on `CPUExecutionProvider`,
  but trace durations include profiler overhead and do not identify exact CPU
  instructions selected inside a kernel.
- The current Runtime is single-image and CPU-only. Windows INT8 is delivered;
  Linux, AArch64/QEMU, bounded concurrency, and TensorRT remain planned until
  their own units produce evidence.
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
