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
> is closed; **S2-01 has not started**. Current delivered claims remain the
> Windows x86_64 ONNX Runtime CPU single-image chain described below.

![Fixed inference demo](docs/assets/demo_inference_result.gif)

## 1. What the Project Solves

A model file alone does not provide a deployable software product. The Runtime
must still answer:

- Which model, tensor, preprocessing, postprocessing, and threshold contract is
  actually being executed?
- Can the same image produce deterministic, inspectable JSON and visualization?
- Do independent Python and C++ implementations agree under declared numeric
  gates?
- Are failures actionable, and are performance numbers published only after
  correctness passes?
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
  -> static INT8 PTQ + ORT operator/node profiling
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
- Correctness gates run before formal benchmarks.
- Backend abstraction remains the smallest boundary required by real backends.
- Every result records the command, contract, artifact identity, sample,
  environment, raw evidence, and limitation.

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

### Requirements

The verified Windows workflow uses an x64 MSVC environment, OpenCV C++ 4.8.0,
the complete ONNX Runtime C++ 1.19.2 SDK, a compatible Python environment for
consistency validation, and the pinned GoogleTest source policy. Machine paths
belong in the ignored `cpp_infer/.stage1.local.psd1` or environment variables,
never in tracked CMake or source files.

Copy and fill the optional local template when portable discovery does not find
the dependencies:

```powershell
Copy-Item .\cpp_infer\tools\stage1.local.example.psd1 .\cpp_infer\.stage1.local.psd1
```

### Canonical Commands

Run from an ordinary PowerShell or CMD at the repository root:

```powershell
# Discover the workflow without starting a build.
.\cpp_infer\tools\stage1.cmd help

# Read-only toolchain and dependency validation.
.\cpp_infer\tools\stage1.cmd doctor

# Clean Release build -> 106 tests -> Demo -> consistency -> benchmark.
.\cpp_infer\tools\stage1.cmd all

# Run one arbitrary image; an optional second argument selects the output dir.
.\cpp_infer\tools\stage1.cmd detect "D:\images\sample.jpg" "D:\outputs"
```

`detect` remains a single-image convenience entry and reuses
`DetectorPipeline`. It is not directory batch processing. The complete action
matrix, low-level commands, environment paths, and safe temporary-build rules
are documented in [Paths and commands](docs/paths_commands.md).

## 5. Core Modules

| Boundary | Responsibility |
|---|---|
| Runtime/artifact/metadata contracts | Separate adjustable runtime policy, declared artifact semantics, and actual ORT-observed tensor/provider facts; reject mismatches before inference |
| `ImagePreprocessor` | Decode or accept a `CV_8UC3` image; letterbox, BGR-to-RGB, normalize, produce contiguous NCHW data, and retain inverse-transform metadata |
| `OnnxRunner` / `InferenceOutput` | Own ORT resources with RAII/PImpl, validate input/output, run synchronously, and copy output into an ORT-independent lifetime |
| Postprocessor/NMS | Validate YOLO BCN output, select class scores, apply strict filtering and stable class-agnostic NMS, then restore and clip source coordinates |
| `DetectorPipeline` and writers | Orchestrate one image and emit owned results, stable JSON, and deterministic GUI-free visualization while enforcing safe output paths |
| Evidence harness | Test pure seams with synthetic fixtures, test the real vertical slice sparingly, compare Python/C++ detections, and publish structured benchmark/memory evidence |

## 6. Stage Two Plan

Stage Two uses five complete delivery units. Each unit freezes a minimum SPEC,
implements one runnable capability, produces tests and machine-readable
evidence, updates the three project entry documents, and then stops for L1
acceptance.

| Unit | Delivery | Honest boundary | Status |
|---|---|---|---|
| S2-01 | Static INT8 PTQ, FP32/INT8 correctness/task-quality/performance comparison, ORT operator/node profiling | No QAT, D010 quantization, or profiler-as-benchmark claims | **Next; not started** |
| S2-02 | Linux x86_64 native chain, shared-source portability, AArch64 cross-build and QEMU smoke | WSL2 is not a board; QEMU produces no performance claim | Planned |
| S2-03 | Directory/manifest discovery, bounded queue, workers, backpressure, failure accounting, clean shutdown, throughput comparison | Concurrent single-image work is not true ONNX batch | Planned |
| S2-04 | One real Linux x86_64 + RTX 4060 TensorRT execution path, FP16 correctness and performance | Local GPU/edge-node evidence only; not Jetson or embedded deployment | Planned |
| S2-05 | Applicable full gates, result matrix, failure cases, three resume narratives, interview material, recruiting freeze | Adds no new technology stack | Planned |

### Platform Matrix

| Platform/backend | What it proves | Current status |
|---|---|---|
| Windows x86_64 + ORT CPU FP32 | Current product chain, correctness, tests, segmented benchmark, Peak Working Set | Verified |
| Windows/Linux x86_64 + ORT CPU INT8 | Quantization quality, size, latency, memory, and profiling under frozen protocols | Planned in S2-01/S2-02 |
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
- `CPUExecutionProvider` is session-level execution evidence. Per-node
  placement requires the planned ORT profiling work.
- The current Runtime is single-image and CPU-only. INT8, Linux, AArch64/QEMU,
  bounded concurrency, and TensorRT remain planned until their own gates pass.
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
