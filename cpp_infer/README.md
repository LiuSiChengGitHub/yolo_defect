# cpp_infer

This directory is the C++ Runtime workspace for the **Industrial Vision Edge AI Runtime and C++ Engineering System**.

Current status: **the Large-Stage-One automatic engineering gate and user-owned L2 are complete; S2-01 through S2-04 have complete implementations, evidence, and teaching closure.** S2-01/S2-02/S2-03 now share a consolidated Round 2 QDQ/U8S8 product chain across Windows single-image inference, Linux x86_64 bounded multi-image execution, and cross-built Linux AArch64/QEMU functional execution. The formal artifact is `yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`, SHA-256 `9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`; default FP32 regressions remain retained. S2-04 separately proves real TensorRT on WSL2/Linux x86_64 + RTX 4060 Laptop. Its ORT TensorRT EP path remains a real-execution but failed-correctness diagnostic; the accepted product path is a SHA-bound load-only native TensorRT plan with one FP16 DFL Softmax compute layer and FP32/noTF32 elsewhere, and its untouched v4 holdout passed twice. The merged source passes 180/180 CTest on both Windows x86_64 and WSL2/Linux x86_64. Work stops for user L1; S2-05 has not started. QEMU is not a physical ARM board, and S2-04 is local WSL2 GPU/edge-node evidence rather than native Linux, Jetson, ARM64 GPU, or embedded evidence. The root bilingual READMEs remain the project-status and roadmap entry points; this file retains the Runtime's technical details and historical evidence.

| Gate | Status |
|---|---|
| S1-08 L1 | Accepted |
| S1-09 clean reproduction / automatic gate | PASS |
| User Large-Stage-One L2 | Accepted |
| Large Stage One | Complete |
| S2 preparatory documentation closure | Complete |
| S2-01 implementation and evidence | Round 2 U8S8 integrated into the cross-platform multi-image chain; advisory quality facts retained |
| S2-01 user L1 | Awaiting |
| S2-02 Gate A + Gate B implementation/evidence/final closure | Complete, including selected U8S8 on Linux x86_64 and AArch64/QEMU |
| S2-02 user L1 | Awaiting |
| S2-03 bounded multi-image implementation/evidence/final closure | FP32 closure and formal U8S8 integration evidence complete |
| S2-03 user L1 | Awaiting |
| S2-04 TensorRT implementation/evidence/final closure | Complete |
| S2-04 user L1 | Awaiting |

## Current single-image chain

```text
RuntimeConfig + ModelArtifactSpec
-> DetectorPipeline
-> canonical single-image path
-> OpenCV decode + letterbox + BGR/RGB + normalize + NCHW
-> OnnxRunner: ORT CPU / ORT TensorRT EP / native TensorRT enqueueV3
-> owned InferenceOutput [1,10,13125]
-> validate + BCN decode + strict score filter
-> stable class-agnostic model-space NMS
-> inverse letterbox + source-bound clip
-> SingleImageDetectionResult
-> stable JSON v1 + deterministic OpenCV visualization
```

`main.cpp` owns only CLI parsing and orchestration. Contract loading, preprocessing, ORT/TensorRT lifetime, postprocess, result validation, JSON serialization, output-path policy, and visualization remain in `yolo_defect_runtime`.

The S1-07 evidence path is separate from the product chain:

```text
fixed six-class manifest + the same Runtime/artifact declarations
-> Python ORT 1.19.2 CPU reference
-> existing C++ single-image CPU CLI
-> class-first, maximum-IoU deterministic matching
-> fixed tolerance checks
-> per_image.json + summary.json
```

The S1-08 performance path is gated by that correctness evidence:

```text
same frozen config + model + crazing_241.jpg
-> clean Release CPU session, sequential, intra/inter threads 1/1
-> warmup 10 (not sampled)
-> repeat 100 with steady-clock segmented timings
-> arithmetic mean + nearest-rank P50/P95 + batch-1 throughput
-> Windows process-lifetime Peak Working Set
-> benchmark JSON written only after timing is complete
```

The S2-01 path reuses the same product semantics and keeps formal timing and
profiling in separate sessions:

```text
frozen FP32 ONNX + 180-image calibration manifest + declared QDQ INT8 protocol
-> static Conv PTQ -> audited INT8 ONNX + artifact/config/card
-> Python/C++ Runtime legality + advisory product/task-quality comparison
-> two independent unprofiled Release benchmark processes
-> two independent profiling-enabled sessions
-> raw ORT traces -> node/operator/provider summaries
-> cross-bound advisory exercise-completion JSON
```

S2-02 Gate A keeps that same business path and confines platform differences:

```text
shared Runtime + preprocess + postprocess + DetectorPipeline
-> CMake: Windows .lib/.dll staging | Linux libonnxruntime.so + build RPATH
-> platform_info: Windows Peak Working Set | Linux getrusage peak RSS
-> stage1.cmd/PowerShell | stage1.sh/Bash orchestration
-> shared tests + fixed image + consistency + scoped benchmark
```

S2-03 composes the unchanged single-image pipeline at image granularity:

```text
directory or UTF-8 path-list manifest
-> deterministic discovery and preflight output-path protection
-> bounded FIFO queue containing task indices only
-> N workers, each owning one DetectorPipeline / CPU Ort::Session
-> existing batch=1 preprocess -> ORT -> postprocess -> detection writer
-> ordered per-image results + BatchSummary schema v1
```

This is image-level concurrency, not tensor true batch. ORT remains sequential
with intra/inter-op threads `1/1` inside every worker session. BatchRunner keeps
an explicit CPU-only provider guard; S2-04 does not claim GPU worker concurrency.

S2-04 keeps that single-image product boundary while selecting the backend:

```text
RuntimeConfig v2 + one ModelArtifactSpec
-> cpu: explicit ORT CPU Session::Run
 | tensorrt: ORT TensorRT EP -> CUDA EP -> CPU EP (diagnostic only)
 | tensorrt_native: SHA-bound plan -> H2D -> enqueueV3 -> D2H -> sync
-> identical owned FP32 [1,10,13125] output
-> unchanged YOLO decode / NMS / restore / JSON
```

The native path has no fallback and verifies the TensorRT/CUDA/SM, engine SHA,
execution policy, tensor names/shapes/dtypes, and cache namespace before use.

The completed S2-01/S2-02/S2-03 integration selects the model only through the
existing declaration layer:

```text
int8_u8s8_config.txt -> U8S8 ModelArtifactSpec + frozen model SHA
-> unchanged DetectorPipeline / ORT CPU float32 external I/O
-> single-image output or unchanged BatchRunner
-> Linux x86_64 native evidence | AArch64 ELF under QEMU functional evidence
```

## Layout and responsibilities

```text
cpp_infer/
├── CMakeLists.txt
├── artifacts/
│   ├── yolov8_neu_det.artifact.txt
│   ├── yolov8_neu_det_int8_qdq.artifact.txt
│   └── yolov8_neu_det_int8_qdq_u8s8.artifact.txt
├── configs/
│   ├── default_config.txt
│   ├── int8_config.txt
│   ├── int8_u8s8_config.txt
│   ├── tensorrt_fp16_config.txt
│   ├── tensorrt_native_fp16_config.txt
│   └── tensorrt_native_fp16_config_v4.txt
├── include/yolo_defect_cpp/
│   ├── artifact_spec.h
│   ├── batch_result.h
│   ├── batch_runner.h
│   ├── batch_writer.h
│   ├── benchmark_result.h
│   ├── benchmark_runner.h
│   ├── benchmark_writer.h
│   ├── config_loader.h
│   ├── detection_result.h
│   ├── detector_pipeline.h
│   ├── image_preprocessor.h
│   ├── model_metadata.h
│   ├── onnx_runner.h
│   ├── profile_runner.h
│   ├── project_core.h
│   ├── postprocessor.h
│   └── result_writer.h
├── src/
│   ├── artifact_spec.cpp
│   ├── batch_discovery.cpp
│   ├── batch_executor.h
│   ├── batch_path_safety.h
│   ├── batch_runner.cpp
│   ├── batch_writer.cpp
│   ├── bounded_queue.h
│   ├── benchmark_result.cpp
│   ├── benchmark_runner.cpp
│   ├── benchmark_writer.cpp
│   ├── config_loader.cpp
│   ├── detector_pipeline.cpp
│   ├── image_decoder.cpp
│   ├── image_decoder.h
│   ├── image_preprocessor.cpp
│   ├── key_value_parser.cpp
│   ├── key_value_parser.h
│   ├── model_metadata.cpp
│   ├── native_tensorrt_runner.cpp
│   ├── native_tensorrt_runner.h
│   ├── onnx_runner.cpp
│   ├── platform_info.cpp
│   ├── platform_info.h
│   ├── profile_runner.cpp
│   ├── project_core.cpp
│   ├── postprocessor.cpp
│   ├── result_writer.cpp
│   └── main.cpp
├── results/demo/
│   ├── crazing_241.detections.json
│   └── crazing_241.visualized.png
├── results/consistency/
│   ├── per_image.json
│   └── summary.json
├── results/benchmark/
│   └── yolov8_neu_det_cpu_release.json
├── results/s2_01/
│   ├── quantization_report.json
│   ├── correctness_quality_v1_failed.json
│   ├── benchmark/
│   ├── profile/
│   ├── round2/
│   └── exercise_completion.json
├── results/s2_02/linux_x86_64/
│   └── detect/
├── results/s2_02/aarch64_qemu/
│   ├── detect/
│   ├── elf_inspection.txt
│   ├── loader_resolution.txt
│   └── qemu_smoke.txt
├── results/s2_03/
│   ├── windows_x86_64/
│   ├── linux_x86_64/
│   └── linux_aarch64_qemu/
├── results/s2_04/linux_x86_64_rtx4060/
│   ├── trtexec/
│   ├── profile/
│   ├── native_engine/
│   ├── correctness/
│   └── benchmark/
├── cmake/toolchains/
│   └── linux-aarch64-gnu.cmake
├── protocols/
│   ├── s2_01_ptq_protocol.json
│   ├── s2_01_ptq_protocol_r2_u8s8.json
│   ├── s2_04_tensorrt_fp16_protocol.json
│   ├── s2_04_tensorrt_fp16_protocol_v2.json
│   ├── s2_04_tensorrt_native_fp16_protocol_v3.json
│   └── s2_04_tensorrt_native_fp16_protocol_v4.json
├── tools/
│   ├── quantize_s2_01.py
│   ├── evaluate_s2_01_correctness.py
│   ├── compare_s2_01_benchmarks.py
│   ├── summarize_ort_profile.py
│   ├── assemble_s2_01_evidence.py
│   ├── compare_consistency.py
│   ├── compare_batch_runs.py
│   ├── validate_batch_summary.py
│   ├── summarize_s2_04_ort_profile.py
│   ├── run_s2_04_product_set.py
│   ├── compare_s2_04_correctness.py
│   ├── run_s2_04_gpu_benchmark.py
│   ├── bootstrap_aarch64_deps.sh
│   ├── stage2_aarch64.sh
│   ├── stage1.cmd
│   ├── stage1.sh
│   ├── stage1.ps1
│   ├── stage1.defaults.psd1
│   └── stage1.local.example.psd1
└── tests/
    ├── result_writer_test.cpp
    ├── image_file_probe.cpp
    ├── postprocessor_test.cpp
    ├── preprocessor_mat_test.cpp
    ├── inference_runner_test.cpp
    ├── model_metadata_validator_test.cpp
    ├── project_core_smoke.cpp
    ├── test_consistency.py
    ├── benchmark_test.cpp
    ├── batch_test.cpp
    ├── image_decoder_test.cpp
    ├── assert_batch_cli.py
    ├── test_s2_03_batch_tools.py
    ├── test_s2_04_evidence_tools.py
    ├── test_run_s2_04_product_set.py
    ├── assert_benchmark.cmake
    ├── assert_benchmark_json.py
    ├── assert_detection_json.py
    ├── assert_single_image_outputs.cmake
    ├── assert_cli_arguments_failure.cmake
    ├── assert_*.cmake
    └── fixtures/
        ├── consistency_manifest.json
        └── s2_03_consistency_manifest.txt
```

- `RuntimeConfig` owns runtime policy: score/NMS thresholds, configured provider, and the artifact declaration path.
- `ModelArtifactSpec` owns model identity, declared SHA-256, provenance/license, tensor I/O, classes, and preprocess/postprocess/NMS semantics.
- `project_core` contains standard-library-only YOLO raw-output validation, decode, class-agnostic NMS, and coordinate restore. The normal Runtime links it; Gate B cross-compiles and runs its smoke under QEMU before separately exercising the full ARM64 OpenCV/ORT Runtime.
- `ImagePreprocessor` exposes file-path and `const cv::Mat&` entries. Both use the same letterbox/RGB/normalize/NCHW implementation.
- `OnnxRunner` owns an ORT session or delegates to `NativeTensorRtRunner` through RAII/PImpl. `run()` borrows the preprocess vector only for the synchronous backend call, then returns copied host storage independent of ORT/TensorRT lifetimes.
- `OnnxRunnerOptions::profile_file_prefix` enables ORT profiling before session construction; the runner finalizes it exactly once through `EndProfilingAllocated` and validates the returned trace path.
- `ProfileRunner` prepares one image once, creates a dedicated profiling session, executes the declared number of `Session::Run` calls, finalizes the raw trace before postprocess, and returns owned run/model/provider/output metadata. It cannot be mixed with benchmark mode.
- `Postprocessor` performs validated BCN decode, class argmax, strict score filtering, `xywh -> xyxy`, IoU, stable class-agnostic NMS, inverse letterbox, and clipping.
- `SingleImageDetectionResult` is a self-contained result object holding model, image, Runtime, provider, and detection evidence needed by output writers.
- `DetectorPipeline` connects the established modules for exactly one image and protects config, artifact, model, and source-image paths from output overwrite.
- `BatchRunner` discovers a directory or UTF-8 path-list manifest before worker startup, pre-creates one `DetectorPipeline`/ORT session per effective worker, and composes the unchanged single-image `run()` call through a bounded FIFO queue of task indices.
- The internal `BoundedQueue` provides mutex/condition-variable backpressure, peak-depth and producer-wait evidence, normal close-and-drain, and cooperative stop. `BatchRunner::request_stop()` rejects new work, cancels queued/unstarted tasks, lets an active synchronous ORT call finish, joins every worker before returning, and records the request independently of the number of unstarted items.
- `BatchSummary` and `BatchWriter` define and serialize machine-readable schema v1, including ordered per-image status, an explicit `cooperative_stop_requested` flag, counts, worker/session policy, queue/backpressure, timing, throughput, environment architecture, and platform memory disclosure. Exit codes map `succeeded/partial_failure/cancelled/fatal` to `0/2/130/1`.
- `ResultWriter` validates the result, escapes JSON strings, uses locale-independent stable numeric formatting, creates parent directories, enforces overwrite policy, and renders deterministic labels/colors without a GUI.
- The internal `image_decoder` seam opens the native `std::filesystem::path` as bytes and calls `cv::imdecode(IMREAD_COLOR)`, preserving Unicode Windows paths and the same normalized `CV_8UC3` contract used by normal preprocessing; its measured decode interval includes the encoded-file read.
- `BenchmarkResult` defines the Release benchmark protocol, six latency segments, environment/runtime/model/sample metadata, throughput, memory evidence, timing exclusions, and limitations. Its pure statistics helper uses empirical nearest-rank ceiling for P50/P95 and derives batch-1 throughput from mean latency.
- `BenchmarkRunner` creates and validates the CPU session before repeated timing, performs untimed warmup, rejects result drift between iterations, measures decode/preprocess/`Session::Run`/postprocess/pipeline/end-to-end with `std::chrono::steady_clock`, and queries the thin platform adapter after the timed iterations.
- `platform_info` owns the narrow operating-system boundary for timestamps, host/compiler metadata, and process peak memory: `GetProcessMemoryInfo` Peak Working Set on Windows and `getrusage(RUSAGE_SELF).ru_maxrss` peak RSS on Linux.
- CMake models ORT as one imported target but validates/stages `onnxruntime.lib` plus `onnxruntime.dll` on Windows, links native Linux `libonnxruntime.so`, and imports only ARM64 ORT/OpenCV from the explicit private sysroot during cross-build. The Runtime/preprocess/postprocess/Pipeline source is not forked per operating system or CPU architecture.
- `BenchmarkWriter` serializes finite schema-v1 JSON with the classic locale, creates output parents, refuses overwrite by default, and protects config, artifact, model, and source-image paths. JSON serialization and filesystem writing happen after the measured loop.
- `consistency_manifest.json` freezes six artifact classes, five validation images per class, declaration-relative paths, image SHA-256 values, and the acceptance requirements before comparison runs.
- `compare_consistency.py` strictly reloads the same Runtime/artifact declarations, verifies file hashes and model metadata, runs Python ORT with only `CPUExecutionProvider`, invokes the existing C++ CLI, performs order-independent matching, and atomically replaces each machine-readable evidence file.
- `test_consistency.py` exercises manifest validation, frozen-threshold rejection, strict score/NMS semantics, class mismatch handling, and order-independent maximum-IoU matching without running the 30-image experiment.
- `benchmark_test.cpp` fixes the mean and nearest-rank percentile definitions, single-sample behavior, invalid empty/negative/NaN/Inf rejection, and throughput calculation independently of machine speed.
- `assert_benchmark.cmake` runs a short `warmup=1/repeat=2` Release smoke, invokes the strict Python validator, then proves a second run refuses overwrite without changing the first file.
- `assert_benchmark_json.py` rejects duplicate/non-finite JSON, validates the complete Release/CPU/thread/batch/sample/model/input/statistics/memory/disclosure schema, recomputes the referenced model and image SHA-256 values, and checks both throughput values equal `1000 / mean_ms`.
- `quantize_s2_01.py` validates the frozen protocol and every calibration image hash before PTQ, publishes through staged files, audits all selected Conv/QDQ structures, checks actual metadata and finite Python ORT output, and emits the independent artifact card.
- `evaluate_s2_01_correctness.py` runs the same FP32/INT8 product semantics over the frozen product and labeled quality manifests and invokes the Release C++ CLI for legality/consistency evidence. Its original failed product gate remains machine-readable.
- `summarize_ort_profile.py` counts `Session/model_run` and per-node calls, validates CPU placement and an optimized-graph FP32/INT8 signature, and aggregates top nodes/operators with percentages and cumulative percentages. Raw trace time remains diagnostic.
- `assemble_s2_01_evidence.py` defaults to strict correctness. Its explicit `--correctness-policy advisory` mode still requires model/protocol/manifest/runtime/benchmark/trace bindings, preserves every failed quality boolean, and emits `strict_acceptance_passed=false` rather than laundering the result.
- `stage1.sh` is the thin WSL2/Linux x86_64 workflow. It checks the pinned Linux SDK/toolchain, uses a guarded `/tmp` Ninja Release tree, verifies built ELF dependencies with `ldd`, and orchestrates the same product/test/consistency/benchmark/batch behavior as the Windows entry where applicable. `batch-compare --config` selects FP32 or U8S8 while retaining the frozen 361-image/workers/queue/output protocol.
- `bootstrap_aarch64_deps.sh` downloads the official ARM64 ORT SDK and extracts Ubuntu ARM64 OpenCV packages into a private target tree without installing them over the host development packages. `stage2_aarch64.sh` cross-builds, inspects, and functionally executes the AArch64 artifacts under QEMU; `YOLO_DEFECT_AARCH64_CONFIG` selects the RuntimeConfig while the default stays FP32, and its `batch` action is functional acceptance and never a benchmark.
- GTest targets link `yolo_defect::runtime`; they never compile or reuse `main.cpp`.

## S2-01 static PTQ, comparison, and ORT Profiling

Round 1 used the all-64-Conv QDQ/S8S8 v1 protocol. Its model was 71.264%
smaller, but the ORT 1.19.2 optimized graph retained 57 float `Conv` nodes and
formed only 7 `QLinearConv` nodes. Added Q/DQ boundaries and loss of the FP32
SiLU/QuickGelu fusion made formal `Session::Run` 37.16% slower. That result and
the earlier selective-Conv experiments remain intact as diagnosis history.

Round 2 uses
[`s2_01_ptq_protocol_r2_u8s8.json`](protocols/s2_01_ptq_protocol_r2_u8s8.json).
It keeps the same 12,336,935-byte FP32 source, 180-image calibration manifest,
MinMax, per-channel weights, all 64 Conv targets, external metadata, product and
quality manifests, and 10/100 performance protocol. Its only quantization
parameter change from v1 is activation `QInt8 -> QUInt8`; weight remains
`QInt8`, so the internal representation is U8S8 QDQ.

The derived model is `models/best.int8.qdq.u8s8.onnx`, 3,544,494 bytes and
71.269% smaller than FP32. Python and Release C++ ORT both create CPU sessions,
run finite output, and observe float32
`images [1,3,800,800] -> output0 [1,10,13125]`. The graph audit records
64 selected and quantized Conv nodes, no exclusions, and no failed target.

The 30-image comparison has 62 FP32 detections, 65 U8S8 detections, and 61
matches; its original product aggregate remains false. Across 361 labeled
images, mAP50 changes `0.710815 -> 0.700459` and mAP50-95 changes
`0.345786 -> 0.342379`. The original mAP50 gate is missed by 0.000356 beyond
its allowed drop, so the unmodified result remains false and advisory.

Formal unprofiled Release evidence uses independent FP32/U8S8 processes,
`CPUExecutionProvider`, sequential execution, intra/inter threads `1/1`, one
fixed image, warmup 10, and repeat 100:

| Metric | FP32 | U8S8 |
|---|---:|---:|
| model size | 12,336,935 B | 3,544,494 B (-71.269%) |
| session initialization | 61.986 ms | 94.858 ms |
| `Session::Run` mean/P50/P95 | 155.106/155.124/169.639 ms | 95.040/95.570/110.768 ms |
| pipeline mean/P50/P95 | 163.477/163.221/182.008 ms | 103.872/104.042/121.654 ms |
| pipeline throughput | 6.117 img/s | 9.627 img/s |
| Peak Working Set | 150.980 MiB | 148.832 MiB |

Mean `Session::Run` is 38.726% lower, pipeline mean is 36.461% lower, and
throughput is 57.383% higher. Initialization remains slower and is reported as
a separate one-time cost.

The separate 10-run U8S8 trace records 640 `QLinearConv` calls and no float
`Conv`: all 64 Conv nodes now execute through the integer operator on every
run. The optimized graph has 439 nodes versus Round 1's 683. QLinearConv takes
35.18% of diagnostic kernel-event time; DQ 13.34%, Resize 13.22%, Mul 10.51%,
Concat 8.79%, Q 6.09%, and Sigmoid 5.53%. This both explains the speedup and
identifies remaining boundary/activation hotspots. Profile timing includes
instrumentation overhead and is not substituted for the unprofiled benchmark.

Machine evidence is under [`results/s2_01/round2/`](results/s2_01/round2/).
The full failure-to-fix narrative, exact evidence links, limitations, and
reproduction entry points are in
[`s2_01_round2_closure.md`](../docs/details/s2_01_round2_closure.md). The
original S8S8 record remains in
[`s2_01_closure.md`](../docs/details/s2_01_closure.md).

## S2-02 Gate A: WSL2/Linux x86_64 Native

Gate A removed the Windows-only build and measurement assumptions without
forking product semantics. `RuntimeConfig`, artifact/metadata validation,
OpenCV preprocessing, `OnnxRunner`, the standard-library postprocess core,
`DetectorPipeline`, JSON, and visualization remain shared business code.
CMake resolves the platform-specific ORT runtime contract, `platform_info`
owns the small OS boundary, and `stage1.sh` supplies Linux orchestration.

The recorded clean Release closure produced:

| Evidence | Result |
|---|---|
| Linux build/test | Final closure rerun: WSL2/Linux x86_64, Ninja Release, 119/119 CTest passed |
| Fixed-image path | `crazing_241.jpg`, three detections, valid JSON and readable PNG |
| Python ORT/C++ ORT | 30/30 images and 62/62 matched detections passed the frozen gates |
| Short benchmark smoke | Earlier Gate A samples used warmup 1 / repeat 2: end-to-end mean `135.896991 ms`, `7.358515 img/s`, peak RSS `196.570312 MiB`; the same-protocol durable rerun measured `151.273896 ms`, `6.610526 img/s`, `196.757812 MiB`, confirming high variance. Benchmark was not repeated in the final functional closure |
| Linux dynamic loading | Nine built ELF executables inspected with `ldd`; no dependency was `not found`, and the CLI resolved `libonnxruntime.so` from the configured SDK/RPATH |
| Windows regression | Final closure rerun: NMake Release, 119/119 CTest and fixed Demo passed |

The fixed-image JSON/PNG are tracked under
[`results/s2_02/linux_x86_64/`](results/s2_02/linux_x86_64/). Exact toolchain
locations and workflow diagnosis remain in
[`paths_commands.md`](../docs/paths_commands.md); the full machine snapshot,
commands, evidence, and interpretation are in
[`s2_02_gate_a_closure.md`](../docs/details/s2_02_gate_a_closure.md) and
[`s2_02_closure.md`](../docs/details/s2_02_closure.md).

This evidence is WSL2/Linux x86_64 only. The warmup-1/repeat-2 samples vary
materially and are functional performance smokes, not a formal benchmark or a
cross-OS comparison.
Linux peak RSS and Windows Peak Working Set have different platform semantics
and are not directly comparable. Gate A itself remains native x86_64 evidence;
the separate Gate B record below owns every AArch64/QEMU claim.

## S2-02 Gate B: Linux AArch64 Cross Compile + QEMU

Gate B keeps CMake/Ninja/cross-G++ as x86_64 host tools and imports target-only
ARM64 OpenCV and official ORT 1.19.2 libraries. A private package-extraction
sysroot avoids replacing the host OpenCV development environment. The same
Runtime and CLI sources produce the AArch64 artifacts.

| Evidence | Result |
|---|---|
| Cross-build | Final closure rerun: Release `project_core`, full Runtime archive, and production CLI generated for AArch64 |
| ELF and loader | CLI Machine is `AArch64`, interpreter is `/lib/ld-linux-aarch64.so.1`; target loader resolved 138 ARM64 libraries with zero `not found` and no x86_64 mix |
| QEMU contracts/core | Startup/help, config + artifact, two negative contracts, and decode → NMS → coordinate restore passed |
| Full inference | Final closure rerun: fixed image ran through ARM64 OpenCV and ARM64 ORT CPU; existing postprocess wrote a validated JSON with three detections |
| Native regression | WSL2/Linux x86_64 clean Release, nine `ldd` checks, and 119/119 CTest passed |

Raw records are under [`results/s2_02/aarch64_qemu/`](results/s2_02/aarch64_qemu/).
The reproducible dependency/toolchain commands and honest status table are in
[`paths_commands.md`](../docs/paths_commands.md) and
[`s2_02_gate_b_closure.md`](../docs/details/s2_02_gate_b_closure.md) and
[`s2_02_closure.md`](../docs/details/s2_02_closure.md).
QEMU is not a physical ARM device. Gate B deliberately records no emulated
latency, throughput, power, or board-performance inference.

## S2-03 bounded multi-image concurrency

S2-03 adds a batch orchestration layer without changing detection semantics.
Every worker owns one `DetectorPipeline` and one CPU ORT session; each task is
still one image and one batch=1 `DetectorPipeline::run()` call. ORT remains in
sequential execution mode with intra/inter-op thread counts `1/1` per session.
There is no true tensor batch, video, service, GPU concurrency, or lock-free
queue.

The direct CLI contract is:

```text
--config <config> --batch
  (--input-dir <directory> | --manifest <path-list>)
  --output-dir <directory> --batch-summary <file>
  [--workers <1..64>] [--queue-capacity <1..4096>]
  [--output-images] [--overwrite]
```

The default is `workers=1` and `queue_capacity=2*workers`; effective workers
are `min(requested_workers, discovered_tasks)`. Directory discovery is
recursive, accepts regular `.bmp/.jpeg/.jpg/.png/.tif/.tiff/.webp` files,
does not follow symlinks, and sorts by the UTF-8 generic relative path. A
manifest is a UTF-8 path list that allows a BOM, LF/CRLF, blank lines, and
comments whose first non-blank character is `#`; entries resolve relative to
the manifest parent and retain declaration order. Absolute, missing,
unsupported, or duplicate-canonical manifest entries fail before work starts.

Every successful image always writes
`items/<six-digit-sequence>.detections.json` with the existing single-image
schema; `--output-images` additionally writes the matching PNG. Per-image
decode/inference/write failures remain isolated and appear in the ordered
summary while other images continue. Existing outputs are refused unless
`--overwrite` is explicit. Preflight also prevents the summary or item outputs
from replacing source images, config, artifact, model, or manifest inputs; a
directory output root may not be inside its input directory.

`BatchSummary` schema v1 records status, original command, build/runtime/model
identity, target and running-kernel architectures, deterministic input policy,
output policy, requested/effective workers, session count and ORT `1/1`
threading, queue capacity/peak/backpressure, balanced counts, wall time,
successful-image throughput, process memory disclosure, an explicit
`cooperative_stop_requested` flag, limitations, and one
ordered item record per discovered task. The public status/exit mapping is:

| Status | Meaning | Exit code |
|---|---|---:|
| `succeeded` | every discovered item succeeded | 0 |
| `partial_failure` | at least one item failed and none was cancelled | 2 |
| `cancelled` | cooperative signal/stop was observed; zero or more items may remain unstarted | 130 |
| `fatal` | infrastructure/session failure | 1 |

The formal throughput numerator is successful images and the denominator is
processing wall time. That interval includes queue waits, image processing,
per-image writes, drain/cancellation, and joins; it excludes discovery,
Runtime/artifact loading, worker session construction, and summary
serialization. The historical S2-03 closure contained `156/156` passing CTest
entries on Windows x86_64 and WSL2/Linux x86_64; the current integration gate is
`157/157` on both platforms. Two Windows tests that require
creating symlinks/reparse points reported GTest skips because the local account
lacked that privilege; the corresponding path-safety tests ran on Linux.

Formal same-platform FP32 CPU, JSON-only, 361-image comparisons used separate
Release processes and a fixed queue capacity of eight. `compare_batch_runs.py`
proved byte and semantic equality for all 361 per-image detection JSON files
before reporting performance:

| Platform / metric | workers=1 | workers=4 | workers=4 change |
|---|---:|---:|---:|
| Windows processing wall | 57.433 s | 20.220 s | -64.79% |
| Windows throughput | 6.286 img/s | 17.854 img/s | 2.840x |
| Windows Peak Working Set | 151.805 MiB | 505.086 MiB | +353.281 MiB |
| WSL2/Linux processing wall | 44.492 s | 17.907 s | -59.75% |
| WSL2/Linux throughput | 8.114 img/s | 20.160 img/s | 2.485x |
| WSL2/Linux peak RSS | 205.766 MiB | 588.227 MiB | +382.461 MiB |

All four runs reached, but never exceeded, queue depth eight. Producer wait
counts were Windows `353/352` and WSL2/Linux `353/349` for workers `1/4`, so
the summaries contain observed backpressure rather than only a configured
capacity.

These are two independent within-platform comparisons, not a Windows-versus-
Linux benchmark. The WSL2 pair used the same copied 361-image/config/model set
and JSON-only outputs in one WSL-native ext4 work area, avoiding `/mnt/d` I/O
from the measured pair. The increased process high-water memory is expected because
workers own independent sessions; no minimum speedup was an acceptance gate.
Machine-readable records are the Windows
[`comparison.json`](results/s2_03/windows_x86_64/comparison.json) plus its
[`workers=1`](results/s2_03/windows_x86_64/workers_1/batch_summary.json) and
[`workers=4`](results/s2_03/windows_x86_64/workers_4/batch_summary.json)
summaries, and the WSL2/Linux
[`batch_comparison.json`](results/s2_03/linux_x86_64/performance/batch_comparison.json)
plus the corresponding
[`workers=1`](results/s2_03/linux_x86_64/performance/batch_workers_1/batch_summary.json)
and
[`workers=4`](results/s2_03/linux_x86_64/performance/batch_workers_4/batch_summary.json)
summaries.

The same production Runtime/CLI was cross-compiled to Linux AArch64 and run
under QEMU user-mode. Acceptance covered a two-image directory at workers=1,
the same ordered images through a manifest at workers=2 and queue=1,
byte/semantic equality between both per-image outputs, and a corrupt-JPEG
manifest with exactly two successes, one failure, status `partial_failure`,
and exit code 2. Each summary records target `aarch64`, running kernel
`x86_64`, execution context `qemu_user_mode_on_x86_64_host`, and
`memory.publishable=false`. The durable acceptance log is
[`qemu_batch_acceptance.txt`](results/s2_03/linux_aarch64_qemu/final_20260830_r2/qemu_batch_acceptance.txt),
with machine summaries for
[`directory workers=1`](results/s2_03/linux_aarch64_qemu/final_20260830_r2/directory_workers1/batch_summary.json),
[`manifest workers=2`](results/s2_03/linux_aarch64_qemu/final_20260830_r2/manifest_workers2/batch_summary.json),
and
[`partial failure`](results/s2_03/linux_aarch64_qemu/final_20260830_r2/partial_failure/batch_summary.json).
No QEMU latency, throughput, RSS, speedup, physical-board, native ARM, power,
thermal, or deployment-performance conclusion is published.

## S2-04 WSL2/Linux x86_64 + RTX 4060 TensorRT

The GPU build is deliberately isolated from the verified ORT 1.19.2 CPU builds:

```text
YOLO_DEFECT_REQUIRE_TENSORRT_EP=ON
YOLO_DEFECT_ENABLE_NATIVE_TENSORRT_BACKEND=ON
ORT GPU 1.20.1 + TensorRT 10.4.0.26 + CUDA runtime 12.6
Linux x86_64 only; exact SDK roots are CMake cache inputs
```

The first product route uses ORT TensorRT EP with a fixed
`TensorrtExecutionProvider -> CUDAExecutionProvider -> CPUExecutionProvider`
chain, FP16, a 2 GiB builder workspace, engine cache, and timing cache. The
current ONNX SHA is
`7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`.
Standalone `trtexec --fp16` built and reloaded it successfully; an ORT profile
contains ten TensorRT-attributed kernel events and no CUDA/CPU fallback event.
That proves real execution, not correctness. Frozen v1 failed one image and the
disjoint semantic v2 failed two; both TensorRT repetitions were internally
identical. Consequently, the retained ORT warm benchmark JSONs are diagnostic,
non-publishable performance observations.

The accepted route is `provider=tensorrt_native`. `NativeTensorRtRunner` is a
load-only TensorRT 10 name-based-I/O backend: it hashes the plan, verifies the
declared SHA/runtime/CUDA/SM 8.9/I/O/execution policy, deserializes one engine,
creates one context and non-default CUDA stream, allocates persistent buffers,
then performs synchronous H2D → `setTensorAddress` → `enqueueV3` → D2H → stream
synchronization. It has no fallback. The public `OnnxRunner` API and all
pre/postprocessing stay unchanged.

The consumed native v3 plan used FP16 only for the final class Sigmoid and
failed one of its frozen 30 images at the one-pixel coordinate gate. The final
E0 plan was selected only against the already consumed v1+v2+v3 90 images, then
tested on a new disjoint v4 set frozen before its first inference. E0 details:

- engine SHA-256:
  `E0CBB0A8A620C1FCF3F8FE215BC716313A3884D2A9CCDE4F3D18B4571ABD8746`;
  size 21,144,012 bytes;
- build policy: `--fp16 --noTF32 --precisionConstraints=obey` with
  `*:fp32,/model.22/dfl/Softmax:fp16`;
- exactly one Half compute layer, `/model.22/dfl/Softmax`; two reformats touch
  Half; all other compute and external I/O are Float;
- v4 CPU-vs-native A/B each passed 30/30; 64 matched detections, max confidence
  error `1.0044e-5`, max coordinate error `0.032166 px`, min IoU `0.998619`;
  native A/B result trees are byte-identical.

The accepted batch=1, warmup=10, repeat=100 product measurements use the same
GPU-enabled binary and `crazing_241.jpg`:

| Metric | Same-SDK ORT CPU FP32 | Native warm A | Native warm B |
|---|---:|---:|---:|
| Initialization | `36.784 ms` | `684.570 ms` | `619.423 ms` |
| Backend/session P50 / P95 | `110.314 / 124.604 ms` | `3.877 / 5.329 ms` | `3.633 / 7.468 ms` |
| Pipeline P50 / P95 | `118.436 / 133.059 ms` | `6.974 / 8.779 ms` | `6.519 / 10.490 ms` |
| Pipeline throughput | `8.3247 img/s` | `137.6517 img/s` | `140.5550 img/s` |
| Process peak RSS | `200.121 MiB` | `384.668 MiB` | `384.371 MiB` |

Native throughput is `16.5353x/16.8841x` the same-SDK CPU reference. This is an
overall native TensorRT/GPU comparison; the experiment cannot isolate a speedup
caused by the sole FP16 Softmax. Detection repeatability is exact and throughput
differs by about 2.1%, but the P95 values are not stable. A/B GPU memory is a
device-wide `nvidia-smi memory.used` baseline-to-peak delta of `155 MiB`; the
target PID exposed no numeric memory row, so it is not process-exclusive VRAM.

The plan is an ignored machine-local derivative, not a portable model artifact.
`tensorrt_max_workspace_size_bytes` is build provenance/config compatibility for
the native path; the load-only runtime does not allocate builder workspace.
BatchRunner remains CPU-only. TensorRT INT8 is not implemented because the FP32
artifact has no frozen calibration/QDQ contract and INT8 was non-blocking.

The S2-04 branch originally passed 179/179 CTest; after merging the formal U8S8
batch integration, the current source passes 180/180 on Windows x86_64 and
WSL2/Linux x86_64. The rebuilt GPU product passed model inspection, raw `enqueueV3`, and one-image
three-detection output. Reproduction commands and exact local dependency entry
points are in [`paths_commands.md`](../docs/paths_commands.md). Durable machine
records are under [`results/s2_04/linux_x86_64_rtx4060/`](results/s2_04/linux_x86_64_rtx4060/),
and the decision history, evidence limits, and nine-part teaching record are in
[`s2_04_closure.md`](../docs/details/s2_04_closure.md).

## S2-01/S2-02/S2-03 formal U8S8 integration closure

The formal model is
`yolov8n_neu_det_s2_01_int8_qdq_u8s8_r2`, SHA-256
`9F2B3356555232B11F403D2D9071146006DDCB19E531DBF0DA727341B1E268B1`,
selected by [`int8_u8s8_config.txt`](configs/int8_u8s8_config.txt). Its external
input/output remain float32, so the integration does not duplicate or change
Runtime, preprocessing, postprocessing, `DetectorPipeline`, `BatchRunner`,
bounded-queue, item-output, partial-failure, or `BatchSummary` behavior.

The current test and workflow layer adds only model selection and evidence
identity checks. When the ignored local U8S8 ONNX exists, CMake registers a
second real batch CLI integration test with the expected model id/SHA; the
existing FP32 test remains unchanged. `batch-compare` accepts an optional
RuntimeConfig on Windows and Linux, and `YOLO_DEFECT_AARCH64_CONFIG` selects the
AArch64/QEMU config while defaulting to FP32.

| Evidence | Result |
|---|---|
| Windows x86_64 | Final Release `157/157` CTest; two permission-dependent symlink/reparse GTest cases skipped locally and their Linux equivalents passed; U8S8 `crazing_241` produced 3 detections |
| WSL2/Linux x86_64 | Final Release `157/157` CTest; U8S8 fixed image produced 3 detections; 30-image manifest workers=2/queue=4 completed 30/30 with queue peak 4 and 25 producer waits |
| Linux 361-image U8S8 | workers=1: `4.591151 img/s`, peak RSS `192.933594 MiB`, 353 waits; workers=4: `15.903088 img/s`, `556.882812 MiB`, 350 waits; throughput ratio `3.463857`; queue peak 8; all 361 detection JSON files byte- and semantic-equal |
| AArch64/QEMU U8S8 | Clean cross-build, AArch64 ELF, and target-loader checks passed; fixed image produced 3 detections; directory workers=1 and manifest workers=2/queue=1 each completed 2/2 with identical outputs; corrupt JPEG produced exact 2 success + 1 failure, exit 2, and one producer wait at queue=1 |
| Default FP32 regression | Linux Demo and AArch64/QEMU `all` passed with no config override |

The U8S8 Linux pair ran directly from the repository under `/mnt/d` DrvFs.
The older FP32 pair ran in a WSL-native ext4 work area, so the new values are
only a same-run U8S8 workers=1/4 comparison and are not a controlled
FP32-versus-INT8 comparison. QEMU summaries record target `aarch64`, running
kernel `x86_64`, context `qemu_user_mode_on_x86_64_host`, and
`memory.publishable=false`; their timing/RSS fields are not reported as
performance evidence.

The engineering closure does not rewrite S2-01's advisory quality result:
agreement precision remains `0.938462 < 0.95`, and mAP50 drop remains
`0.010356 > 0.01`. Machine-readable results are under
[`results/s2_03/int8_integration/`](results/s2_03/int8_integration/), and the
frozen scope is in the
[`S2-01/S2-02/S2-03 integration SPEC`](../docs/details/s2_int8_arm64_batch_integration_spec.md).

## Frozen YOLOv8 baseline semantics

```text
raw layout: [1,4+C,N] in contiguous BCN order
index: values[channel * N + candidate]
box encoding: cx, cy, w, h in model-input coordinates
confidence: maximum of C class scores; no separate objectness or sigmoid
class-score tie: lower class id wins
filter: confidence > score_threshold (strict, float32 domain)
NMS: class-agnostic, in model-input coordinates
NMS suppression: IoU > nms_threshold; equality is kept
equal-confidence tie: preserve original candidate/input order
restore: subtract left/top padding, divide by scale, then source-bound clip
```

The fixed order is:

```text
validate -> decode/argmax/strict filter -> model-space NMS
-> subtract padding -> divide by scale -> source-bound clip
```

Changing class-aware/class-agnostic behavior, threshold equality, or tie ordering requires coordinated contract, Python reference, C++, and test changes.

## S1-05 CLI and output contract

The output flags are:

```text
--output-json <path>    write stable detection JSON
--output-image <path>   write an OpenCV visualization without opening a GUI
--overwrite             explicitly allow replacement of existing regular files
```

At least one output flag is required to enter the S1-05 pipeline; either output may be requested independently, and the fixed demo requests both. Relative CLI image/output paths are resolved from the current working directory. This is different from config/artifact relative paths, which remain relative to their declaring files.

Missing parent directories are created recursively. Existing outputs are rejected by default with a nonzero exit and an actionable `already exists` error. `--overwrite` permits replacement only for regular output files; directories, symbolic/special files, identical JSON/image destinations, and protected input paths remain rejected.

JSON schema v1 has a fixed nested shape:

```text
schema_version
model
  model_id
  declared_sha256
image
  path
  original_size {width, height, channels}
  input_size {width, height}
runtime
  actual_provider
  provider_evidence
  score_threshold
  nms_threshold
  nms_mode
detections[]
  class_id
  class_name
  confidence
  bbox_xyxy [x1, y1, x2, y2]
```

All JSON strings are UTF-8 validated and safely escape quotes, backslashes, standard control characters, embedded NUL, and other bytes below U+0020. Numbers are finite JSON numbers formatted with the classic locale and stable precision. A valid no-detection result is always `"detections": []` rather than `null` or a missing field.

## Development environment and platform task runners

Current MSVC/CMake/CTest, ONNX Runtime, OpenCV, Python, GoogleTest, local-path
precedence, TEMP rules, raw CMake audit commands, and shell pitfalls live only in
the canonical [paths, toolchains, and environment diagnosis](../docs/paths_commands.md).
Historical evidence sections below retain the environment and result snapshots
recorded at their original milestones; they are not current setup instructions.

Run the wrapper from an ordinary PowerShell or CMD at the repository root:

```powershell
.\cpp_infer\tools\stage1.cmd help
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd batch data\images\val -Workers 4 -QueueCapacity 8
.\cpp_infer\tools\stage1.cmd batch-compare
.\cpp_infer\tools\stage1.cmd batch-compare -Config cpp_infer\configs\int8_u8s8_config.txt
```

The wrapper initializes the x64 Visual Studio environment before invoking the
PowerShell workflow. A plain PowerShell that cannot find `ctest`, `cmake`,
`cl`, or `nmake` has usually skipped that environment chain; diagnose it
with `stage1.cmd doctor` before treating it as a source or test failure. Use
`test`, `all`, `benchmark`, or `profile` only when the requested change
and the project's proportional-validation policy call for them.

In WSL2/Linux x86_64, select the documented Linux SDKs and run the Bash entry
from the repository root:

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON=/path/to/python
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest

bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh batch data/images/val --workers 4 --queue-capacity 8
bash cpp_infer/tools/stage1.sh batch-compare
bash cpp_infer/tools/stage1.sh batch-compare --config cpp_infer/configs/int8_u8s8_config.txt
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark
bash cpp_infer/tools/stage1.sh all
```

`doctor` is read-only. `batch-compare` fixes 361 images, workers `1/4`,
queue `8`, CPUExecutionProvider, and JSON-only output in independent processes;
its optional config selects FP32 (the default) or U8S8. For AArch64/QEMU, set
`YOLO_DEFECT_AARCH64_CONFIG="$PWD/cpp_infer/configs/int8_u8s8_config.txt"`
before `stage2_aarch64.sh all` and unset it afterward; omission keeps FP32. `all`
performs a clean build, full CTest, Demo, consistency, the workflow's normal
benchmark, and frozen 30-image batch acceptance, so it is a closure action
rather than the default after every edit. Use the verified versions and
configurable environment entries from [`paths_commands.md`](../docs/paths_commands.md),
not the placeholders above.

## Verified S1-05 evidence

Clean Release verification on 2026-08-16 produced:

```text
result-writer JSON GTest:          6/6 passed
output-focused CTest:            16/16 passed
complete CTest:                  78/78 passed
fixed image:                     crazing_241.jpg, 200x200 CV_8UC3
model input/output:              [1,3,800,800] -> [1,10,13125]
actual session provider:         CPUExecutionProvider
fixed demo detections:           3
fixed demo detected class:       crazing (3/3)
Python standard JSON parser:      passed
OpenCV visualization read-back:  passed, 200x200 CV_8UC3
recorded demo JSON:              1,164 bytes, SHA-256 E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049
recorded demo PNG:               39,306 bytes, SHA-256 3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346
```

The three fixed-demo detections, in stable confidence order, were:

```text
0: class=crazing, confidence=0.445792824, bbox=[0, 53.6803322, 176.90683, 146.240784]
1: class=crazing, confidence=0.417582601, bbox=[21.2503815, 118.812775, 188.814178, 194.868408]
2: class=crazing, confidence=0.308511496, bbox=[22.7723389, 2.68823242, 192.409409, 86.2025604]
```

The 16 output-focused tests consist of:

- six JSON GTests for golden field order, quote/backslash/control-byte escaping, legal empty detections, non-finite detection/runtime rejection, and locale-independent decimal formatting;
- one fixed real-model integration test that creates nested parents, validates JSON with `python -m json.tool` and a strict semantic checker, and reads the visualization through OpenCV;
- nine CLI/output negative tests for missing/duplicate arguments, missing image, mode conflicts, meaningless overwrite, identical destinations, protected input overwrite, and directory targets.

The integration test also proves that a second run without `--overwrite` fails without changing either file, while a later explicit overwrite recreates byte-identical JSON and PNG output. The previous 62 contract, metadata, inference, preprocess, and postprocess tests remain in the complete 78-test gate.

## Verified S1-06 quality gate

A fresh Release/NMake build on 2026-08-22 used MSVC 19.50.35721.0, OpenCV 4.8.0, ONNX Runtime C++ 1.19.2, Python 3.9.25, and the pinned GoogleTest v1.17.0 source. `ctest -N` listed 90 behavior-oriented names and the complete gate passed 90/90 in 5.53 seconds:

```text
unit label:          51/51 passed
integration label:   3/3 passed
negative label:     32/32 passed
contract label:     19/19 passed
metadata label:     16/16 passed
preprocess label:    9/9 passed
postprocess label:  25/25 passed
output label:       18/18 passed
quality_gate:       90/90 passed
```

The gate deliberately separates three kinds of evidence:

- pure/synthetic tests validate strict Runtime/artifact schemas, four distinct BGR pixels through RGB/`[0,1]`/NCHW flattening, landscape/portrait odd padding, synthetic model metadata, BCN decode, strict thresholding, IoU/stable NMS, coordinate restore/clip, legal empty detections, JSON, and output-path rules;
- three positive integration tests inspect the real model, run one owned raw output, and execute the fixed JSON/PNG vertical slice;
- negative tests inject a missing model, Runtime/artifact schema errors, name/shape/input-and-output-dtype/class/provider mismatches, damaged image bytes, invalid tensor length, CLI argument conflicts, and a regular file used as an uncreatable output parent.

The missing-model, damaged-image, and uncreatable-output-parent CLI checks each returned exit code 1. Their diagnostics identify the failing path/object and include expected state, actual state, and an action. The damaged image is generated in the disposable build tree; metadata mismatches use a synthetic `ModelMetadata` value, so no additional large ONNX fixtures are committed.

### S1-06 failure triage

- Schema failure: read the declaration path, line/field, expected/actual, and action. Runtime paths are relative to the Runtime config; model paths are relative to the artifact declaration, never the process working directory.
- Missing model: inspect the artifact's `model_path` and the normalized path printed in the error.
- Metadata mismatch: run `--inspect-model`, then compare the actual name/shape/dtype/provider with the declaration. Synthetic cases test the pure validator; they are not extra malformed models.
- Damaged image: distinguish `path does not exist` from `file exists but OpenCV decoding returned an empty image`, then retry with a known-good image.
- CLI error: run `--help`, then check missing values, duplicate flags, required arguments, and mutually exclusive modes.
- Output failure: verify that the parent is a writable directory, the target is a regular file or absent, protected inputs are not targeted, and overwrite was explicitly requested when appropriate.

A valid `[1,4+C,N]` tensor with no score strictly above the threshold is a successful result with `detections: []`. A wrong rank/channel/element count or non-finite output is instead a contract failure and must not be converted to an empty result.

## Verified S1-07 Python ORT/C++ ORT consistency

S1-07 freezes a reproducible correctness experiment rather than adding a second production inference API. The fixed repository manifest selects validation indices `241`, `255`, `270`, `285`, and `300` for each artifact class:

```text
crazing:          5 images
inclusion:        5 images
patches:          5 images
pitted_surface:   5 images
rolled-in_scale:  5 images
scratches:        5 images
total:           30 images
```

Every entry records `sample_id`, source class id/name, an image path relative to the manifest file, and the image SHA-256. The manifest also points to the same `default_config.txt`; the tool follows that config to the same artifact and model instead of accepting parallel command-line thresholds or class lists. Before inference it validates the manifest's exact six-class/five-image coverage, unique paths and IDs, hashes, frozen requirements, config/artifact fields, actual model SHA-256, and Python ORT metadata.

The two implementations use the same frozen semantics:

```text
provider: Python explicitly requests only CPUExecutionProvider;
          C++ reports CPUExecutionProvider from explicit CPU EP session setup
input:    OpenCV letterbox, pad 114, BGR -> RGB, float32 [0,1], NCHW
decode:   [1,4+C,N], no objectness, no extra sigmoid, lower class id wins ties
filter:   confidence > float32(score_threshold)
NMS:      stable class-agnostic NMS in model-input space, suppress IoU > threshold
restore:  subtract left/top padding, divide by scale, then source-bound clip
```

Detection arrays are not zipped by output position. The comparison first partitions by `class_id`, canonicalizes values for deterministic tie handling, then repeatedly chooses the remaining pair with maximum IoU. Only after matching does it evaluate the requirements that were frozen before the first full run:

```text
per-image detection count:               exact
matched class_id:                         exact
absolute confidence error:               <= 1e-4
maximum absolute bbox coordinate error:  <= 1e-2 pixel
matched box IoU:                          >= 0.999
```

The clean Release comparison on 2026-08-22 passed without changing those requirements:

```text
images:                              30/30 passed
Python/C++ detections:               62/62
matched detections:                  62/62
maximum confidence absolute error:   8.049977111568296e-07
maximum bbox-coordinate error:       9.135351561440075e-05 pixel
minimum matching IoU:                0.999998927116394
consistency-labeled CTest:            2/2 passed in 12.58 seconds
complete CTest:                      92/92 passed in 17.28 seconds
unit label:                          52/52 passed
integration label:                    4/4 passed
negative label:                      32/32 passed
```

The repository-local [`per_image.json`](results/consistency/per_image.json) contains both implementations' detections, class counts, deterministic matches, per-pair errors, unmatched values, and failure messages for every image. The repository-local [`summary.json`](results/consistency/summary.json) records manifest/config/artifact/model hashes, model and tensor identity, provider/session policy, library versions, fixed requirements, aggregate metrics, per-source-class results, and limitations. These paths are evidence locations, not a claim about current Git tracking state. Both documents use finite JSON values and can be read by Python's standard `json` module.

The verified runtime split is explicit rather than hidden: Python 3.9.25 used ONNX Runtime CPU 1.19.2, OpenCV 4.13.0, and NumPy 2.0.2; C++ used the official ONNX Runtime C++ SDK 1.19.2 and OpenCV 4.8.0. The OpenCV version difference was recorded before interpreting results, and the original strict requirements were not relaxed.

### S1-07 failure triage

- Python dependency/provider failure: confirm CMake receives the `PythonExe` resolved by the local workflow configuration, imports ORT/OpenCV/NumPy, and lists `CPUExecutionProvider`. Current machine paths and precedence are documented only in the central toolchain reference. The comparison refuses a Python Session whose provider list is not exactly CPU.
- Manifest failure: use the reported sample/field/path, expected/actual, and action. Do not replace a missing image or update its hash silently; any sample change creates a new evidence protocol.
- Count/class failure: inspect the per-image Python/C++ class counts and unmatched detections before looking at numeric tolerances. Matching never crosses class boundaries.
- Confidence or box failure: debug in order: decoded image and letterbox metadata, normalized NCHW tensor, raw ORT output, float32 strict threshold, stable class-agnostic NMS, then inverse letterbox/clip. Do not widen the gate merely to turn the run green.
- Low matching IoU: inspect all same-class candidate pair IoUs and canonical keys. The matcher uses maximum IoU and does not assume JSON array order.
- C++ CLI failure: reproduce the recorded command for that single image and use its object/path, expected, actual, and action diagnostics.
- Evidence write failure: verify `--output-dir` is writable and neither output target is a directory. Evidence files are written through a temporary sibling and then replaced so an incomplete JSON document is not promoted.

The first evidence run exposed a Python 3.9 compatibility issue: `Path.write_text(..., newline="\n")` is not supported by that interpreter. The writer was changed to `Path.open(..., newline="\n")`, which preserves explicit LF and atomic replacement while supporting Python 3.9. This was an evidence-writer API fix; no model logic, sample, matching rule, or numerical threshold changed.

## Verified S1-08 Release benchmark and memory evidence

S1-08 was run only after the final clean-build consistency label passed 2/2. The benchmark therefore measures the same model, contract, CPU execution policy, and fixed sample that passed S1-07 rather than a separately tuned path.

The repository-local [`yolov8_neu_det_cpu_release.json`](results/benchmark/yolov8_neu_det_cpu_release.json) records this protocol. The path is an evidence location, not a claim about current Git tracking state:

```text
machine:                 DESKTOP-6OGK71C
processor:               AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
logical CPU count:       16
OS:                      Windows 10.0.26200, x86_64
compiler/build:          MSVC 19.50.35721.0, Release, C++17
C++ OpenCV / ORT:        4.8.0 / 1.19.2
requested provider:      cpu
actual session provider: CPUExecutionProvider
session policy:          sequential, intra-op=1, inter-op=1,
                         graph optimization=all
model:                   yolov8n_neu_det_final_train_2,
                         12,336,935 bytes, [1,3,800,800] float32 NCHW
sample:                  crazing_241.jpg, 200x200x3
batch / sample count:    1 / 1
warmup / repeat:         10 / 100
score / NMS threshold:   0.25 / 0.45, class_agnostic
detections per repeat:   3
clock / percentile:      std::chrono::steady_clock /
                         empirical nearest-rank ceiling
```

All latency values are milliseconds. Throughput is batch-1 images per second derived as `1000 / mean_ms` for the matching aggregate segment:

| Segment | Mean | P50 | P95 | Throughput |
|---|---:|---:|---:|---:|
| Image decode (`cv::imread`) | 0.991129 | 0.964900 | 1.351700 | — |
| Preprocess (`cv::Mat -> float32 NCHW`) | 8.244569 | 7.551400 | 12.126500 | — |
| ONNX Runtime `Session::Run` only | 165.555859 | 164.898500 | 186.213600 | — |
| Postprocess (`raw output -> detections`) | 0.424115 | 0.425100 | 0.563600 | — |
| Pipeline (`preprocess + infer + postprocess`) | 175.560944 | 175.105800 | 195.137600 | 5.696028 img/s |
| End to end (`decode + pipeline`) | 176.553060 | 176.135700 | 196.612800 | 5.664020 img/s |

Windows `GetProcessMemoryInfo` reported Peak Working Set `160,133,120` bytes (`152.714844 MiB`). This is the process-lifetime high-water mark including config/session initialization, warmup, measured iterations, retained timing vectors, statistics, and benchmark-harness state; it is queried before JSON serialization/write. It is not current working set, per-stage memory, or incremental model-only memory.

The benchmark deliberately excludes the following from every repeated latency sample:

- RuntimeConfig/ModelArtifactSpec loading and validation;
- `Ort::Env`, session/model initialization, and metadata validation;
- initial path validation and file-size queries;
- statistics calculation and the Peak Working Set query;
- benchmark JSON serialization and filesystem writing;
- visualization/GUI work, which is not executed in benchmark mode.

The implementation still discloses those exclusions rather than hiding them. In particular, session initialization is a real startup cost even though it is not part of steady-state per-image inference. `pipeline` includes preprocess, input validation/tensor construction, `Session::Run`, output validation/copy, and postprocess; `session_run` isolates only the synchronous ORT call. `end_to_end` additionally includes image decode.

The final S1-08 gates passed on 2026-08-22:

```text
consistency label before benchmark:  2/2 passed
benchmark label:                    14/14 passed
  benchmark statistics GTest:        8/8 passed
  short Release JSON integration:    1/1 passed
  benchmark CLI negative cases:      5/5 passed
complete CTest:                     106/106 passed in 18.44 seconds
formal benchmark JSON parsing:       json.tool + strict validator passed
model/sample evidence SHA checks:    passed
```

The short benchmark test uses `warmup=1/repeat=2`; it verifies schema and behavior, not a speed threshold. The recorded formal S1-08 result uses `warmup=10/repeat=100`. Label counts overlap and must not be added to infer the complete test count.

### S1-08 interpretation and failure triage

- A non-Release build, a provider other than actual `CPUExecutionProvider`, a changed thread policy, a changed baseline model/input/sample/hash, or detection drift aborts publication before a result is accepted.
- Invalid empty/negative/NaN/Inf timing samples fail the pure statistics boundary. JSON also rejects non-finite numbers, duplicate fields, inconsistent sample counts, invalid throughput equations, and missing disclosures.
- A second write to an existing benchmark JSON fails by default. The smoke confirms that refusal leaves the original file hash unchanged; use `--overwrite` only for an intentional re-baseline.
- Repeated `cv::imread` benefits from the operating-system file cache, so `image_decode` is warm-cache latency rather than cold-disk latency.
- No CPU affinity, elevated priority, or idle-system lock was applied. Background load can change the distribution, particularly P95.
- This is one fixed 200x200 image on one Windows CPU machine. It is a reproducible baseline, not a full-dataset distribution, cross-machine ranking, GPU result, or universal deployment claim.
- `actual_provider` proves the configured CPU session was created and used for successful runs; without ORT node profiling it does not prove every graph node's placement independently.
- Historical Python ORT `24.4/72.1 FPS` used a different implementation and protocol. It remains historical context and cannot be compared unconditionally with the current C++ `5.696028/5.664020 img/s` pipeline/end-to-end figures.

## S1-09 fresh automatic closure evidence

S1-09 added no product capability. It used a new `%TEMP%` Release/NMake build and a unique disposable evidence subdirectory to prove that the current source can reproduce the whole Large-Stage-One chain without overwriting repository-local evidence. The automated gate passed on 2026-08-22:

The CMake project version remains `0.7.0`: S1-09 is a reproduction, documentation, and interview-acceptance closure over the S1-08 product surface, not a new Runtime feature release.

```text
clean Release configure/build:          passed
CTest inventory:                        106 tests
complete CTest:                         106/106 passed in 19.91 seconds
fixed Demo detections:                  3
Demo JSON parse + strict validator:     passed
Demo PNG OpenCV probe:                  200x200 CV_8UC3, passed
six-class coverage:                     6 classes x 5 images
consistency images:                     30/30 passed
Python/C++ detections and matches:       62/62
maximum confidence absolute error:      8.049977111568296e-07
maximum bbox-coordinate error:          9.135351561440075e-05 pixel
minimum matching IoU:                   0.999998927116394
consistency per-image/summary JSON:      both parseable
formal benchmark protocol:              warmup 10 / repeat 100
benchmark JSON parse + strict validator: passed
legal empty-detections checks:           passed
```

The S1-09 performance reproduction retained the S1-08 protocol rather than changing settings after seeing the result:

| Segment | Mean ms | P50 ms | P95 ms | Throughput |
|---|---:|---:|---:|---:|
| Image decode (`cv::imread`) | 0.816168 | 0.818200 | 0.925100 | — |
| Preprocess (`cv::Mat -> float32 NCHW`) | 5.453755 | 5.454700 | 6.212800 | — |
| ONNX Runtime `Session::Run` only | 134.419309 | 137.588200 | 142.554900 | — |
| Postprocess (`raw output -> detections`) | 0.345302 | 0.343800 | 0.442400 | — |
| Pipeline (`preprocess + infer + postprocess`) | 141.265814 | 144.467300 | 149.839500 | 7.078853 img/s |
| End to end (`decode + pipeline`) | 142.082777 | 145.322200 | 150.765300 | 7.038151 img/s |

Windows Peak Working Set was `159,989,760` bytes (`152.578125 MiB`). As in S1-08, this is the process-lifetime high-water mark, not current RSS, per-stage allocation, or model-only memory. The fresh values are a reproduction sample under the same protocol, not a replacement for the S1-08 record and not evidence of an optimization. Background load, warm file cache, and ordinary OS scheduling can change latency distributions between runs.

The fresh disposable files were independently parsed/read and hashed: Demo JSON `1,164` bytes, SHA-256 `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`; Demo PNG `39,306` bytes, SHA-256 `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`; consistency `per_image.json` `125,870` bytes, SHA-256 `09B4A4E538CF94B0875A4C13FA0681CC14181E7C3180590CF8F8F13C35908E21`; consistency `summary.json` `6,250` bytes, SHA-256 `90D4F43F1F2DD98D33B84C15F5976107B5B1C1428868812248154EF7650EEC17`; benchmark JSON `5,453` bytes, SHA-256 `F32C0DF3157897264F9BD2B9AE3F3DB7B240A3B641494E8D3E7C346FF64E9C6F`. These are S1-09 temporary reproduction records, not a claim that the temporary directory is Git-tracked.

Four direct fault injections each returned exit code `1` and included the failing object/path plus expected state, actual state, and an action:

| Fault | Stable failure boundary | First action |
|---|---|---|
| Missing model | artifact `model_path` does not resolve to a regular ONNX file | Check the path relative to the artifact declaration and verify the intended model/hash. |
| Damaged image | file exists, but OpenCV decode returns an empty image | Distinguish corrupt/unsupported bytes from a missing path, then retry a known-good encoded image. |
| Unwritable output | output parent is a regular file rather than a directory | Choose a writable directory; do not weaken protected-path or overwrite checks. |
| `--repeat 0` | CLI requires an integer in `[1,1000000]` | Use a positive repeat count; the formal baseline remains `100`. |

A valid tensor that yields no score strictly above the threshold remains a successful empty detection list, and the JSON writer emits `"detections": []`. That behavior is different from malformed shape, element-count, dtype, or non-finite raw output, which must fail.

### Evidence lanes: do not mix protocols

| Evidence lane | Protocol and current result | What it can and cannot prove |
|---|---|---|
| Historical PyTorch/ONNX count evidence | First 50 sorted validation files, all from `crazing`; historical result `146` versus `146` detections plus confidence summaries | Weak export-era evidence only. The matching `best.pt` is absent, so this is not a newly rerun three-way PyTorch/Python-ORT/C++ comparison. |
| Historical Python ORT benchmark | Historical CPU/GPU figures `24.4/72.1 FPS` from a different Python implementation and measurement protocol | Context only. It must not be presented as current C++ speed or compared unconditionally with the C++ numbers. |
| Strict Python ORT/C++ ORT correctness | Same ONNX/config/artifact, explicit CPU providers, six classes x five images, deterministic class/maximum-IoU matching; S1-09 reproduced 30/30 images and 62/62 matches within the frozen tolerances | Strong implementation-consistency evidence for this fixed set. It is not model-accuracy evaluation, bitwise equality, or proof for every platform/image. |
| C++ Release performance | S1-08 recorded the original 10/100 baseline; S1-09 independently reproduced the same six timing boundaries, throughput, environment record, and Peak Working Set | Current C++ evidence only for batch 1, one fixed image, one Windows CPU, single-thread policy, and warm cache. The two runs are retained separately rather than averaged or ranked. |

### S1-09 technical teaching log and completed user L2 acceptance

The automatic gate proves reproducibility but does not by itself prove that the user can explain or modify the system. The following teaching and acceptance material was used for the user-owned L2 gate, which is now complete and closes Large Stage One.

Two-minute explanation outline:

1. Position the project as a C++ industrial-vision Runtime, not merely a model-training repository.
2. Walk through contract loading, OpenCV preprocess, ORT session/run, YOLO decode/filter/NMS/restore, then JSON/PNG.
3. Explain that synthetic GTest covers pure boundaries while the real model is reserved for a few integration smokes.
4. Separate S1-07 correctness evidence from S1-08/S1-09 performance evidence.
5. End with current limits and the S2-01 INT8 PTQ/ORT Profiling boundary.

Five-minute explanation outline:

1. Explain `RuntimeConfig`, `ModelArtifactSpec`, relative paths, strict schema errors, and actual metadata validation.
2. Explain letterbox, BGR/RGB, normalize, NCHW, `Ort::Value` borrowing, synchronous `Session::Run`, and owned raw output.
3. Explain BCN indexing, no objectness/sigmoid, strict `confidence > threshold`, stable class-agnostic NMS, inverse letterbox, and clipping.
4. Explain `DetectorPipeline`, stable JSON, deterministic headless visualization, and output safety.
5. Explain the 30-image class-first/maximum-IoU comparison and frozen numeric gates.
6. Explain the six benchmark boundaries, warmup/repeat, mean/P50/P95, throughput, Peak Working Set, and timing exclusions.
7. Close with three failure paths, licensing limitations, absent `best.pt`, and the work explicitly deferred to the five S2 units.

L2 follow-up questions:

1. Why are thresholds Runtime policy while tensor shape and class names belong to the artifact?
2. Why must actual ORT metadata be checked even when the declaration has already passed schema validation?
3. Why does the input `Ort::Value` borrow the preprocess vector while the returned raw output owns its storage?
4. Why is the output indexed as `channel * N + candidate`, and why must channels equal `4 + C`?
5. Why are there no separate objectness multiplication or sigmoid steps for this artifact?
6. Why does equality fail the score filter but survive NMS when IoU equals the NMS threshold?
7. Why is NMS performed in model space before inverse letterbox restoration?
8. Why does consistency matching use class id and maximum IoU instead of JSON array position?
9. What do count/class/confidence/bbox/IoU gates each detect that the others cannot?
10. Why must consistency pass before benchmark publication?
11. What is included in `Session::Run`, pipeline, and end-to-end timing, and what is deliberately excluded?
12. Why can Peak Working Set not be called model-only or single-inference memory?
13. Why can the historical Python FPS not be used to claim a C++ speedup?
14. Which facts are session-provider evidence, and what would require per-node ORT profiling?

The user should also be able to triage at least these three cases in order: missing model (`config -> artifact-relative model path -> file/hash -> metadata`), consistency mismatch (`image/hash -> preprocess tensor -> raw output -> strict threshold -> NMS -> coordinate restore -> matcher`), and abnormal benchmark (`Release/provider/threads -> correctness gate -> warmup/timing boundary -> background load/P95`). The four automatic fault injections above provide concrete examples, not substitutes for the user's explanation.

Interview-ready resume bullets accepted at the L2 closure:

- Built a C++17/OpenCV/ONNX Runtime industrial-defect inference Runtime with strict model contracts, RAII session/tensor ownership, deterministic YOLOv8 postprocess, stable JSON/PNG outputs, and 106 automated GTest/CTest checks.
- Established correctness and performance evidence: Python ORT/C++ ORT matched 62 detections across a six-class 30-image manifest within `1e-4` confidence, `1e-2 px` bbox, and `0.999` IoU gates; added Release warmup/repeat segmented P50/P95, throughput, and Windows Peak Working Set reporting.

Core code-practice candidates:

- [`image_preprocessor.cpp`](src/image_preprocessor.cpp): letterbox, BGR/RGB, normalization, NCHW flattening, and transform metadata.
- [`onnx_runner.cpp`](src/onnx_runner.cpp): RAII/PImpl, tensor validation, borrowed input lifetime, `Session::Run`, and owned output copy.
- [`postprocessor.cpp`](src/postprocessor.cpp): BCN indexing, strict score boundary, IoU/stable NMS, restore, and clipping.
- [`detector_pipeline.cpp`](src/detector_pipeline.cpp): the single-image vertical slice and self-contained result assembly.
- [`compare_consistency.py`](tools/compare_consistency.py): frozen manifest/contract checks and deterministic maximum-IoU matching.
- [`benchmark_result.cpp`](src/benchmark_result.cpp) and [`benchmark_runner.cpp`](src/benchmark_runner.cpp): nearest-rank percentiles, timing boundaries, result-drift checks, throughput, and memory evidence.

Required core-behavior-plus-GTest exercise:

1. After creating a clean checkpoint and a disposable practice branch, run `postprocess.YoloDecodeTest.StrictThresholdKeepsOnlyScoresGreaterThanThreshold` and confirm the current strict-`>` baseline is green.
2. **RED:** change the synthetic expectation in `postprocessor_test.cpp` so equality is retained, but leave product code unchanged. Run `cmake --build $BuildDir --target yolo_defect_postprocess_tests`, then rerun the focused test and require it to fail.
3. **GREEN:** temporarily change the comparison in `postprocessor.cpp` from strict `>` to inclusive `>=`, update the other exact-threshold expectations, rebuild the same target, and require the focused test to pass. Explain how equality changes detections and why a real contract change would also require the Python reference, consistency evidence, and README updates.
4. Because schema-v1 freezes strict `>`, return to the checkpoint without merging the practice branch, rebuild, rerun the original focused test and complete CTest, and verify no temporary exercise diff remains. Do not leave `>=` in the product merely to finish the exercise.

The root bilingual READMEs are the project-status and roadmap entry points and contain the consolidated interview-facing record. This technical README records the same state: **Large Stage One complete; S2-01/S2-02/S2-03 formal Round 2 U8S8 integration complete across Windows single-image, Linux x86_64 bounded multi-image, and AArch64/QEMU functional execution while preserving the advisory-quality record; S2-04 real TensorRT implementation/evidence/teaching closure complete; user L1 pending; S2-05 not started**.

## Current limits

- Normal `actual_provider` remains session-level evidence. S2-01's separate raw ORT traces add optimized-node placement evidence: every profiled node event names `CPUExecutionProvider`; trace time includes instrumentation overhead.
- `model.declared_sha256` is copied from the validated artifact declaration. The ordinary S1-05 detection CLI does not recompute the model hash on every invocation; the S1-07 comparison and strict S1-08 evidence validator independently bind their runs to the actual frozen model SHA-256.
- JSON and visualization are validated before writing, but they are two files rather than one filesystem transaction. A disk-level failure while writing the second file can leave the first file present.
- The C++17 path check and later file open are not a cross-process atomic transaction. This S1-05 CLI is verified for one local invocation, not for competing writers racing on the same destination; an explicit overwrite can also leave a partial output after a disk-level write failure.
- The benchmark and profile modes remain single-image. Batch mode is an image-level worker system that repeatedly reuses the exact batch=1 `DetectorPipeline`; it is not tensor true batch, video, a service, GPU concurrency, a lock-free queue, or `inference_event`.
- Queue capacity bounds only pending task indices. Images, tensors, and ORT outputs live inside the worker currently processing an item, while each worker still owns an independent ORT session; increasing workers therefore trades substantially higher process high-water memory for potential throughput.
- Cooperative signal shutdown does not forcibly interrupt an active synchronous ORT call. It stops new enqueue/dequeue work, cancels queued and otherwise unstarted items, lets active calls finish, writes a balanced summary when possible, and joins all worker threads.
- S1-07 proves Python ORT/C++ ORT implementation consistency for one fixed 30-image set under the frozen contract. It is not an accuracy evaluation or proof for every possible image/platform/library version; it is the required correctness gate that precedes performance publication.
- S1-08 is a warm-cache, single-image, batch-1 result from one Windows CPU machine with no affinity/priority/idle-system controls. It does not establish full-dataset latency, cold-disk behavior, cross-platform performance, or a statistically controlled hardware comparison.
- Peak Working Set is a process-lifetime high-water mark that includes session initialization and benchmark harness state. It is not a per-stage allocation profile, current RSS, or incremental model memory.
- S2-02 Gate A was verified only inside WSL2/Linux x86_64. WSL is a useful Linux development environment, but this result is not physical edge-board or embedded-device evidence.
- Gate A's Linux warmup-1/repeat-2 samples varied materially and are functional performance smokes. They are neither the formal 10/100 protocol nor a Windows-versus-Linux speed comparison.
- Linux `getrusage` peak RSS and Windows Peak Working Set are both process-lifetime high-water measurements, but their platform semantics differ and their numeric values must not be compared directly.
- Gate B/S2-03 were executed with Linux AArch64 binaries under QEMU user-mode, including fixed-image ARM64 ORT CPU inference and directory/manifest bounded-concurrency acceptance. This is functional emulation evidence only: QEMU summary time/RSS is explicitly non-publishable, and no physical-board, native-ARM performance, latency, throughput, power, thermal, driver, or deployment-stability claim follows from it.
- The current U8S8 workers=1/4 comparison ran under `/mnt/d` DrvFs, while the historical Linux FP32 pair ran in WSL-native ext4. Their absolute throughput and peak-RSS values are not a controlled precision comparison; only each same-run workers=1/4 ratio is interpreted.
- S2-04 ran under WSL2/Linux x86_64 on an RTX 4060 Laptop GPU. It is not
  bare-metal native Linux, Jetson, ARM64 GPU, embedded-device, power, thermal,
  or long-duration stability evidence.
- ORT TensorRT EP genuinely executed TensorRT but failed the frozen v1/v2
  correctness gates. Its benchmark JSON `passed` fields mean the benchmark
  harness completed, not that correctness allowed publication; accepted
  product performance is limited to the E0 native path after v4 passed.
- E0 has one FP16 compute layer and FP32 external I/O/remaining compute. The
  native-vs-CPU speedup is an overall backend/GPU observation, not an isolated
  FP16 gain. Native P95 varied materially, and the reported 155 MiB GPU delta
  is device-wide rather than PID-specific.
- Native plans are bound to the recorded TensorRT/CUDA/SM and policy. The cache
  directory is machine-local and the engine bytes are SHA-checked; it must be
  rebuilt and revalidated for a changed runtime/GPU. S2-04 batch/worker GPU
  concurrency and TensorRT INT8 are not implemented.
- The S2-01 advisory completion is not the original strict acceptance: the 30-image product aggregate remains false, and Round 2's mAP50 drop exceeds its original 0.010 limit by 0.000356. The machine record exposes both facts rather than rewriting them.
- ORT profile summaries describe the optimized execution graph. A source Conv represented by QDQ in the ONNX file may appear as `QLinearConv`, `Conv`, and Q/DQ transition nodes after optimization; the trace does not prove exact hardware instructions.
- The matching `best.pt` is absent from the workspace and Git history. Historical PT/ONNX evidence covers 50 sorted `crazing` images and only count/confidence summaries; S1-07 is a separate same-ONNX Python ORT/C++ ORT comparison and must not be described as a newly rerun PyTorch/Python-ORT/C++ three-way experiment.
- Python OpenCV 4.13.0 and C++ OpenCV 4.8.0 are recorded separately. The 30-image evidence passed the original requirements, but this does not establish bitwise equality across arbitrary OpenCV builds.
- UTF-8 strings, JSON control-byte escaping, Windows separator escaping, paths containing spaces, and the fixed local paths are tested. Arbitrary Unicode input/output paths across every Windows locale and filesystem have not been broadly validated.
- The six baseline class labels are ASCII. OpenCV's built-in Hershey text renderer is not evidence of correct arbitrary Unicode label rendering.
- Class-agnostic NMS remains deliberate baseline behavior and can suppress a lower-scoring box from another class when boxes overlap.
- The visualization is deterministic for the pinned OpenCV build; it is evidence output, not an annotation editor or GUI.

Large Stage One, the S2-01/S2-02/S2-03 formal Round 2 U8S8 integration closure,
and S2-04 TensorRT are complete. The shared CPU product chain covers Windows
U8S8 single-image inference, Linux x86_64 U8S8 manifest and bounded-concurrency
evidence, and AArch64/QEMU U8S8 functional acceptance while default FP32
regressions remain green; the separate S2-04 product path covers accepted native
TensorRT execution on the local WSL2/Linux x86_64 RTX 4060. Work stops here for
user L1; S2-05 evidence/resume/interview closure and recruiting freeze has not
started. QEMU remains functional portability evidence only, and the RTX 4060
result remains local WSL2 GPU/edge-node evidence rather than Jetson or embedded
deployment.

## License checkpoint

Repository-authored source remains MIT. The baseline declaration records the ONNX metadata text `AGPL-3.0 License (https://ultralytics.com/license)` without changing it. Public model/data distribution still requires separate review of model obligations and the NEU-DET redistribution basis; this remains a release checkpoint, not a legal conclusion.
