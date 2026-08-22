# Industrial Vision AI Runtime for Steel Surface Defect Detection

[中文版](README_zh.md)

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange?logo=onnx)
![OpenCV](https://img.shields.io/badge/OpenCV-C%2B%2B-green)
![CMake](https://img.shields.io/badge/CMake-enabled-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

V2 positioning: this repository is being upgraded from a YOLOv8 defect-detection demo into an industrial vision AI inference runtime and C++ engineering project.

YOLOv8 and NEU-DET are the model and dataset carriers. The autumn-recruiting story is not "I trained a detector"; it is "I turned a vision model into a deployable runtime with C++ / ONNX Runtime C++ / OpenCV / CMake / GTest / benchmark evidence and deployment-optimization analysis."

Current V1 assets remain valuable: training, ONNX export, PyTorch-vs-ONNX consistency checks, Python ONNX Runtime inference, FastAPI, Docker, and benchmark scripts. V2 builds on these assets through `cpp_infer/` instead of rewriting them.

Current V2 status: **S1-04 is L1 accepted; S1-05 is implemented and verified, awaiting L1 acceptance.** The fixed single-image C++ CLI now reaches contract loading, OpenCV preprocessing, ONNX Runtime CPU inference, deterministic YOLOv8 postprocess, schema-v1 detection JSON, and a headless OpenCV visualization. S1-06 is only the next step after acceptance; batch processing, consistency evidence, and C++ benchmark results do not exist yet.

The project entry is intentionally concentrated in this README and `README_zh.md`. `docs/PLAN.md` is the latest planning source, `AGENTS.md` turns it into repository-wide collaboration rules, and task/status/change evidence stays in the two READMEs. Long execution detail is split into `docs/` only when it would make the entry point harder to use.

![Inference Demo](docs/assets/demo_inference_result.gif)

## Project 1 Runtime Blueprint

### 1. Project Positioning and Top-Level Design

This repository is **Project 1: Industrial Vision Edge AI Runtime and C++ Engineering System**. Its core value is not to retrain a detector inside this repo, but to turn industrial defect-detection model artifacts into a runnable, testable, benchmarkable, and interview-explainable C++ runtime.

Two model sources are admitted through explicit artifact gates:

- **YOLOv8 + NEU-DET:** the stable P0 runtime baseline. It is used to finish the C++ deployment chain quickly because its output format is simple and the existing repo already has training, ONNX export, Python inference, FastAPI, Docker, and benchmark evidence.
- **`paper_detect` D010 / D-FINE-S + DeepPCB:** the later research-side artifact source. D010 is Template-Counterfactual Defect Denoising: it keeps the D003 inference path, adds training-time erase/replay samples, and does not introduce D009 feature-pyramid injection. `paper_detect` owns training, validation, ablation, official test, result cards, and qualitative figures; this repo may report D010 as a Runtime result only after stable ONNX export, a deployment contract, actual Runtime integration, and consistency validation.

The top-level design follows this rule: **training and research artifacts flow into Project 1; Project 1 owns deployment, runtime behavior, testing, benchmark evidence, and runtime event output.**

The authoritative P0 design is broader than “make one inference call”:

- **Engineering contract:** C++17/CMake multi-target structure, header/source separation, explicit dependencies, and validated runtime/artifact schemas for the model, input, classes, thresholds, preprocess, and postprocess.
- **Inference chain:** OpenCV letterbox/RGB/normalize/NCHW, ONNX Runtime C++ with RAII and name/shape/dtype/provider checks, then model-family decode, filtering, NMS, and coordinate restoration.
- **Observable outputs:** fixed-sample detection JSON and visualization with a reproducible command.
- **Correctness evidence:** Python/ONNX/C++ comparison with declared tolerances for detection count, class, confidence, and coordinates.
- **Performance evidence:** warmup/repeat plus preprocess/infer/postprocess/end-to-end P50/P95, throughput, environment metadata, and peak memory/RSS where feasible.
- **Engineering evidence:** GTest, invalid-input and fault-injection paths, INT8 PTQ comparison, limitations, raw evidence paths, and reproducible README commands.

Large stage one builds the first deliverable vertical slice; large stage two completes the full P0 evidence and INT8 hardening. Later P1 extensions are gated: batch/concurrent processing is interview-value driven; TensorRT/Jetson/ARM requires real hardware; Qt and gRPC/Triton require repeated demand from high-priority job descriptions; D010 requires a stable artifact.

### 2. Problem Solved

Project 1 solves the gap between "a detector exists" and "a detector can be deployed and explained as engineering software":

- Convert images and model artifacts into a reproducible C++ inference path.
- Make preprocessing, inference, postprocessing, NMS, benchmark, and output writing observable as separate modules.
- Record commands, sample outputs, failures, and trade-offs so the project can be reproduced during autumn recruiting.
- Preserve an optional future `inference_event` bridge for Project 2 after the Project 1 P0 chain is stable; it is not a large-stage-one acceptance item.

### 3. End-to-End Runtime Link

Planned full chain:

```text
model artifact
-> artifact contract / model card
-> artifact schema + RuntimeConfig validation
-> OpenCV image read
-> letterbox preprocess / RGB / float32 / NCHW tensor
-> ONNX Runtime C++ session
-> input/output name / shape / dtype / provider checks
-> postprocess / score filter / NMS / coordinate restore
-> detection JSON
-> visualization
-> fixed-sample Python / ONNX / C++ consistency
-> benchmark report
-> INT8 PTQ comparison
-> tests / failure injection / README evidence
-> optional real-device deployment and Project 2 inference_event bridge
```

Current verified chain through S1-05:

```text
cpp_infer/configs/default_config.txt
-> RuntimeConfig
-> config-relative artifact_spec_path
-> ModelArtifactSpec + TensorSpec
-> artifact-relative model_path
-> RuntimeContract cross-field validation
-> OnnxRunner PImpl / Ort::Env / SessionOptions / Session RAII
-> explicit CPUExecutionProvider registration
-> actual ORT ModelMetadata inspection
-> provider/count/name/shape/dtype/class-channel validation
-> OpenCV preprocess -> contiguous float32 NCHW vector
-> exact input shape/element/finite-value validation
-> borrowed CPU Ort::Value -> synchronous Session::Run
-> output count/shape/element/finite-value validation
-> copy ORT output into owned InferenceOutput
-> bounded raw-output summary
-> pure YOLOv8 [1,4+C,N] BCN validation and decode
-> maximum class score, no objectness, strict float32 confidence filter
-> xywh-to-xyxy in model-input coordinates
-> stable class-agnostic NMS in model-input coordinates
-> subtract letterbox padding / divide by scale / source-bound clip
-> owned SingleImageDetectionResult
-> stable schema-v1 JSON with safe UTF-8/control-character escaping
-> deterministic OpenCV rectangle/label visualization without a GUI
-> explicit output-parent creation / fail-on-existing / --overwrite policy
-> fixed sample: 3 crazing detections + parseable JSON + readable PNG
-> 31 retained S1-04 GTests + 6 output GTests + integration/negative gates
-> 78-case complete CTest gate
-> no batch, Python/C++ consistency, or C++ benchmark yet
```

### 4. Core Module Responsibilities

| Module | Responsibility | Current Status |
|--------|----------------|----------------|
| `RuntimeConfig` / `ModelArtifactSpec` / `TensorSpec` / `RuntimeContract` | Separate runtime policy from model identity/I/O/algorithm semantics, resolve declaration-relative paths, and enforce a strict schema with actionable errors. | S1-01 verified |
| `ImagePreprocessor` | Read a file or accept a `CV_8UC3` `cv::Mat`, then letterbox, convert BGR->RGB, normalize, and produce an NCHW float tensor plus inverse-transform metadata. Both entry points share one implementation. | S1-04 synthetic landscape/portrait, odd-padding, non-square, color/layout, and invalid-input GTest verified |
| `OnnxRunner` | Own ORT resources through RAII/PImpl; validate a borrowed contiguous float32 input vector, create a CPU `Ort::Value`, run synchronously, validate the raw output, and copy it before ORT ownership ends. | S1-03 raw inference verified |
| `ModelMetadata` | Represent actual ORT version/provider and tensor count/name/shape/dtype facts, then compare them with `RuntimeContract` through a pure synthetic-testable validator. | S1-02 verified |
| `InferenceOutput` | Own the returned raw tensor shape and float values independently of local ORT output values and the Runner lifetime, then provide the ORT-free input boundary for postprocess. | S1-03 verified and consumed by the S1-04 pure postprocessor |
| `PostProcessor` / `NmsProcessor` | Validate/decode YOLOv8 BCN output, apply strict float32 score filtering, `xywh -> xyxy`, IoU, stable class-agnostic model-space NMS, then restore/clip source coordinates. | S1-04 verified and L1 accepted; core code-practice candidate |
| `DetectorPipeline` | Hold a copied `RuntimeContract` plus an RAII `OnnxRunner` behind PImpl, validate one source image, and orchestrate preprocess -> synchronous Run -> postprocess -> output writing without exposing OpenCV or ORT in its public header. | S1-05 fixed single-image vertical slice verified |
| `SingleImageDetectionResult` / `DetectionImageMetadata` | Own the model identity, source/input image metadata, session-provider evidence, thresholds/NMS mode, class contract, and restored detections needed by downstream writers after local inference objects end. | S1-05 self-contained result boundary verified |
| `ResultWriter` / `Visualizer` | Validate result invariants, serialize stable JSON v1 with safe string escaping, create output parents, enforce protected/overwrite rules, and encode deterministic rectangles and labels through OpenCV without a GUI. | S1-05 JSON/Python parse and OpenCV read-back verified |
| `ConsistencyValidator` | Compare fixed Python ORT and C++ results by count, class, confidence, and box tolerance. | S1-07 pending |
| `BenchmarkRunner` | Measure warmup/repeat preprocess, inference, postprocess, end-to-end latency, throughput, and memory metadata. | S1-08 pending |
| `ArtifactRegistry` / `ModelCard` | Record artifact source, model family, dataset, metrics, config, postprocess type, runtime status, and paths. | YOLO baseline declaration established; D010 remains gated |
| `Tests` | Preserve the 31 S1-04 synthetic postprocess/preprocess GTests, add isolated output-schema tests, and keep one fixed real-model CLI integration plus focused argument/output failures. Test targets link the Runtime boundary rather than `main.cpp`; OpenCV is explicit only where test source needs it. | S1-05: output GTest 6/6, `output` label 16/16, complete CTest 78/78 |

### 5. Quick Start

Current S1-05 clean Release build, complete single-image command, and test path:

```powershell
# Example verified tools root. From CMD first run VsDevCmd.bat under this
# root, then start:
# powershell.exe -NoProfile -NoExit
# -NoProfile prevents the local Conda profile from replacing the VS PATH.
$ToolsRoot = 'D:\01_Base\Tools'
$PythonExe = (Get-Command python.exe -ErrorAction Stop).Source
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = (Join-Path $ToolsRoot 'VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin') + ';' + `
  (Join-Path $ToolsRoot 'opencv\build\x64\vc16\bin') + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_05_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$ToolsRoot\opencv\build\x64\vc16\lib" `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DPython3_EXECUTABLE="$PythonExe" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$DemoDir = Join-Path $BuildDir 'demo outputs'
$OutputJson = Join-Path $DemoDir 'crazing_241.json'
$OutputImage = Join-Path $DemoDir 'crazing_241.png'

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --output-json $OutputJson `
  --output-image $OutputImage

& $PythonExe -m json.tool $OutputJson
& "$BuildDir\bin\yolo_defect_image_probe.exe" $OutputImage
Get-Item $OutputJson, $OutputImage
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

With `BUILD_TESTING=ON`, CMake uses the official GoogleTest v1.17.0 archive at commit `52eb8108c5bdec04579160ae17225d66034bd723` and verifies SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`. An offline clean configure may pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='<verified-google-test-source>'`, but only after independently checking the pinned archive hash before extraction; no personal source path is committed.

The older Python/YOLO quick start remains below for V1 baseline reproduction. The C++ path above is the V2 deployment entry.

Use a fresh out-of-tree build as shown. The ignored `cpp_infer/build` executable was confirmed on 2026-07-15 to be a stale P1-01 artifact and rejects the newer `--config/--image` CLI; it is not current-source evidence.

### 6. Demo Input and Output

Current demo input:

```text
config: cpp_infer/configs/default_config.txt
artifact: cpp_infer/artifacts/yolov8_neu_det.artifact.txt
image:  data/images/val/crazing_241.jpg
```

Current committed S1-05 demo outputs:

```text
detection_json: cpp_infer/results/demo/crazing_241.detections.json
visualization:   cpp_infer/results/demo/crazing_241.visualized.png

fixed result: 3 detections, all class_id=0 / class_name=crazing
JSON: 1,164 bytes; SHA-256 E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049
PNG:  39,306 bytes; SHA-256 3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346
```

The JSON document uses schema version 1 and fixed root objects for `model`, `image`, `runtime`, and `detections`. It records the model id and declared artifact SHA, source/original/input image metadata, session-level provider evidence, score/NMS thresholds and mode, then `class_id`, `class_name`, `confidence`, and `bbox_xyxy` for each detection. A valid no-detection result is represented as `"detections": []`, never `null` or an omitted field.

Output parents are created recursively. Existing regular files fail by default; `--overwrite` is an explicit opt-in. JSON and image paths must differ, directories/symlinks are rejected as file targets, and the source image, Runtime config, artifact declaration, and ONNX model are protected from overwrite. Relative CLI image/output paths use the current working directory; declaration-internal paths keep their declaration-relative rules.

The older S1-03 fixed-image raw-output regression remains available as a diagnostic command and historical evidence:

```text
S1-03 raw output summary
input_shape: [1,3,800,800]
input_elements: 1920000
input_finite_values: 1920000/1920000
input_min: 0.278431386
input_max: 1
output_shape: [1,10,13125]
output_elements: 131250
output_finite_values: 131250/131250
output_min: 0
output_max: 795.04126
session_run: completed
raw_output_ownership: copied_to_InferenceOutput
scope: raw inference only; no decode, NMS, JSON, visualization, or benchmark.
```

The optional preprocess command still reports the verified `200x200 -> 800x800`, `BGR->RGB`, float32 `[0,1]`, NCHW tensor with 1,920,000 elements. No benchmark JSON or `inference_event` is produced by S1-05.

### 7. Test Commands

Current focused GTest and complete CTest gates:

```powershell
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir --output-on-failure
```

Expected current result:

```text
postprocess-focused GTest: 24/24 passed
cv::Mat preprocess GTest:   7/7 passed
retained S1-04 GTest total: 31/31 passed
output-focused GTest:        6/6 passed
output-labeled CTest:       16/16 passed
complete CTest:             78/78 passed
```

The retained 24 postprocess and seven `cv::Mat` preprocess GTests continue to prove the S1-04 pure algorithms independently of a real model. Six output GTests freeze JSON field order, exact schema, empty arrays, quote/backslash/control-byte escaping, finite-number rejection, and locale-independent decimal formatting. The `output` label also includes the fixed real-model CLI run, Python `json.tool` plus semantic JSON validation, OpenCV image read-back, deterministic explicit overwrite, and focused argument/path protection failures. The fixed integration smoke proves the vertical slice and output contract; it does not replace the future S1-07 Python/C++ numerical-consistency evidence.

#### Current S1-05 Limits

- JSON records the artifact's **declared** SHA-256; S1-05 does not recompute the model hash at runtime.
- `actual_provider` means the explicitly registered provider of the successfully created and executed session. It is session-level evidence, not per-node placement from ORT profiling.
- JSON and image data are both prepared before writing, but the two files are not committed as one transaction and cross-process filesystem replacement is not guaranteed atomic.
- The fixed ASCII Windows input path is verified. Arbitrary Unicode input image paths through the existing narrow-string OpenCV `imread` boundary are not yet claimed as supported; output path/JSON UTF-8 handling is separate.
- The current six baseline labels are ASCII. OpenCV's Hershey text renderer is deterministic for them but is not a general Unicode font engine.
- S1-05 is batch-1/single-image only. It adds no directory batch mode, concurrency, service, `inference_event`, Python/C++ consistency result, benchmark, INT8, or performance claim.

### 8. Key Data and Artifact Results

| Item | Current Record |
|------|----------------|
| P0 dataset | NEU-DET steel surface defects, 1,800 images, 6 classes, 200x200 pixels |
| P0 model | YOLOv8n baseline and tuned variants |
| Best current YOLO result | `final_train_2`, mAP@0.5 = 0.743, mAP@50-95 = 0.388 |
| Historical ONNX/PyTorch alignment | 50/50 detection-count matches and 146 vs 146 total detections, but the sorted subset is all `crazing` and records counts/confidence summaries rather than class/box tolerances |
| Baseline ONNX artifact preflight | Tracked `models/best.onnx`, 12,336,935 bytes, opset 17, SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`; metadata says `nms=False` |
| Model lineage status | The project owner confirms the current ONNX was personally exported from `runs/detect/final_train_2/weights/best.pt`; that `.pt` is absent from the workspace and Git history, so the lineage is owner-confirmed but not currently re-exportable |
| Baseline ONNX I/O preflight | Python ORT 1.19.2 confirms input `images` = float32 `[1,3,800,800]`; output `output0` = float32 `[1,10,13125]` |
| Historical Python ORT benchmark | ONNX CPU 24.4 FPS, ONNX GPU 72.1 FPS on RTX 3060; **not C++ Runtime performance** |
| Current C++ Runtime state | S1-05 clean Release/NMake build with MSVC 19.50, OpenCV 4.8.0, and ORT 1.19.2 now completes the fixed single-image C++ vertical slice through JSON and headless visualization; output label 16/16 and complete CTest 78/78 pass. No Python/C++ consistency or C++ performance result exists yet |
| C++ ORT actual metadata | Loaded `models/best.onnx`; available EP inventory `[AzureExecutionProvider,CPUExecutionProvider]`; explicitly registered session EP `CPUExecutionProvider`; input `images` tensor float32 `[1,3,800,800]`; output `output0` tensor float32 `[1,10,13125]`; contract passed |
| S1-02 dependency/session boundary | CMake consumes the official external ORT C++ SDK 1.19.2 only through `ONNXRUNTIME_ROOT`, validates the version/C/C++/CPU-provider headers/import library/DLL and stages the matching DLL. Session policy is sequential, intra-op 1, inter-op 1 (unused by sequential mode), graph optimization all |
| S1-03 raw-output evidence | Fixed `crazing_241.jpg`: input float32 `[1,3,800,800]`, 1,920,000 finite values, range `[0.278431386,1]`; owned output float32 `[1,10,13125]`, 131,250 finite values, range `[0,795.04126]`. These values prove finite raw execution, not decoded detection correctness or benchmark performance |
| S1-04 postprocess evidence | ORT-free synthetic tests verify `[1,4+C,N]` BCN decode, maximum class score without objectness/sigmoid, float32-domain strict `confidence > threshold`, `xywh -> xyxy`, robust IoU, stable class-agnostic NMS before coordinate restore, and letterbox inverse transform/clip. Postprocess GTest 24/24, preprocess GTest 7/7, complete CTest 62/62 |
| S1-05 single-image output evidence | Fixed `crazing_241.jpg` produces 3 `crazing` detections, a Python-parseable 1,164-byte JSON (`E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`) and an OpenCV-readable 39,306-byte PNG (`3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`). Six output GTests, 16 output-labeled CTests, and all 78 tests pass |
| Artifact license checkpoint | The artifact declaration preserves the ONNX metadata text `AGPL-3.0 License (https://ultralytics.com/license)`. Source remains MIT; because the owner chose to keep publicly distributing the ONNX and NEU-DET, model obligations and the dataset's unspecified redistribution terms remain separate release checkpoints |
| Incoming research artifact | `paper_detect` D010 method on the D-FINE-S/DeepPCB research line; not a new Runtime architecture claim |
| External D010 research evidence | Formal-validation AP50-95 = 0.847057; official-test AP50-95 = 0.830385; these are not Project 1 Runtime results |
| D010 relationship and ablation | D003 is the ancestor/ablation anchor; all 6 D010 class deltas over D003 are positive on formal and official test; D010A erase-only and D010B replay-only each beat D003 but trail full D010 |
| D010 integration gate | Stable ONNX + result/model card + deployment contract + real Runtime adapter + consistency validation; it must not block the YOLO P0 closure |

Pending artifact paths:

```text
artifacts/paper_detect_d010/result_card.md        # placeholder
artifacts/paper_detect_d010/model_artifact.yaml   # placeholder
artifacts/paper_detect_d010/metrics_table.csv     # placeholder
artifacts/paper_detect_d010/qualitative/          # placeholder
```

The consolidated C++ result table is still pending. It must eventually record machine/OS/compiler/build type, model/input/sample set, correctness tolerances, segmented and end-to-end P50/P95, throughput, memory/RSS, any extension comparison, failure cases, conclusions, evidence paths, and reproduction commands. The model-license checkpoint is a provenance risk to resolve, not a C++ implementation blocker to hide.

### 9. Key Design Trade-Offs

- **Runtime first, training second:** this repo keeps old training assets but does not make training the V2 main story.
- **YOLO baseline before D010 adapter:** YOLO/ONNX is the quickest stable path to finish C++ preprocess, inference, postprocess, JSON, benchmark, and tests.
- **Artifact gate before D010 claims:** external D010 research metrics may be cited as source evidence, but a C++ D-FINE result requires stable export, contract, adapter, and consistency evidence.
- **Simple C++ over broad framework work:** C++17, CMake, OpenCV, ONNX Runtime C++, GTest, and benchmark output are enough for the interview target.
- **Tests grow with stable seams:** S1-01 tests declarations, S1-02 tests session metadata, S1-03 tests one real raw inference plus rejection before `Run`, and S1-04 uses synthetic tensors/boxes/images to test algorithms independently of model output.
- **Pipeline versus algorithms:** `DetectorPipeline` owns only single-image orchestration; config loading, preprocess, ORT execution, postprocess, serialization, and drawing remain independently testable Runtime seams while `main.cpp` stays a CLI coordinator.
- **Stable and defensive outputs:** JSON v1 has fixed field order, locale-independent finite numbers, escaped UTF-8 strings, and a legal empty `detections` array. Output directories are created, existing files fail by default, `--overwrite` is explicit, and source/config/artifact/model inputs remain protected.
- **Explicit tensor ownership:** the CPU input `Ort::Value` borrows the preprocess vector only for synchronous `Run`, while the output is copied into an ORT-free `InferenceOutput` before local ORT values are destroyed.
- **Frozen YOLOv8 semantics:** `[1,4+C,N]` is decoded without separate objectness or an extra sigmoid; class-score ties choose the lower class id, filtering uses float32-domain strict `>`, and NMS is class-agnostic in model-input space before restore/clip.
- **Deterministic NMS ties:** equal-confidence candidates preserve original input order. This replaces the historical NumPy `argsort()[::-1]` ambiguity with an explicit, tested C++ rule.
- **Conditional extensions:** INT8 PTQ belongs to P0 evidence hardening; TensorRT/Jetson/ARM is a later real-hardware extension, while Qt and gRPC/Triton are job-description gated.
- **Failure records matter:** INT8, D-FINE, or eligible real-device attempts may fail, but commands, errors, root causes, and fallback decisions must be documented without promoting the attempt to a result.

### 10. Task Queue

The latest roadmap follows `docs/PLAN.md`. The top-level design fills any detail omitted by a short stage summary:

| Large Stage | Target / Gate | Project 1 Exit |
|-------------|---------------|----------------|
| Completed baseline and engineering skeleton | Through 2026-07-12 | Training/export/Python assets plus C++17/CMake/CTest, typed config, and real-image OpenCV preprocessing; no C++ inference claim |
| **1. Deliverable loop (current)** | 2026-07-13 to 2026-07-27 | Fixed config/image/model command reaches ORT, decode/filter/NMS/restore, JSON/visualization; fixed-sample Python ORT/C++ consistency; segmented P50/P95; core errors and automated tests; five-minute explanation and one behavior-plus-test modification |
| 2. Evidence hardening | 2026-07-28 to 2026-08-10 | Complete the P0 test/fault matrix, reproducible performance/memory evidence, FP32-vs-INT8 PTQ comparison, final result table, resume bullets, interview set, and focused mock; QAT only if justified |
| 3. P1 extensions | After stable P0; condition gated | Batch/worker/backpressure work by interview value; real TensorRT/Jetson/ARM only with hardware; Qt or gRPC/Triton only with repeated high-priority JD demand; D010 only after its artifact gate |
| 4. Freeze and interview priority | From 2026-08-25 | Freeze P0 features; allow correctness/demo/reproduction fixes, tests/evidence, small JD-specific patches, interview-feedback updates, and non-disruptive P1 work |

Large stage one's detailed, one-step-at-a-time execution plan is in [`docs/STAGE1_EXECUTION_PLAN.md`](docs/STAGE1_EXECUTION_PLAN.md). It is a justified long-form artifact, while this README remains the status and evidence source of truth.

### 11. Version Changes and Progress Records

Current state: historical Project 1 tasks P1-00 through P1-03 and large-stage-one tasks **S1-01 through S1-05** are implemented and verified. S1-04 is L1 accepted. S1-05 adds the fixed single-image CLI, self-contained detection result, stable JSON/visualization output boundary, six output GTests, and a 78-case complete gate; it is awaiting user L1 acceptance.

The 2026-07-16 pre-stage readiness pass is also complete: the ORT C++ 1.19.2 SDK is present and verified, the VS x64 toolchain is discoverable, a new `%TEMP%` Release/NMake build passes 3/3 CTest, and the future GTest dependency is pinned. The owner also confirmed the current ONNX was personally exported from the `final_train_2` best checkpoint; the checkpoint is not in this workspace or Git history. The durable commands, evidence, GTest hash, model-lineage audit, and unresolved public-distribution license checkpoints are in [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md). This preparation did not start S1-01 or change Runtime behavior.

The next implementation step, only after user S1-05 L1 acceptance, is **S1-06: automated main-path and core failure-path hardening**. `S1-*` means “large stage one small stage” and avoids confusing the old Project 1 `P1-*` history with the top-level P1 extension category.

The chronological V2 entry log is kept in the Roadmap section below and must be updated after every small stage.

### 12. Teaching Log From Project Start to Now

| Stage | What Was Done | Purpose | Implementation / Evidence | Issue and Debugging Lesson |
|-------|---------------|---------|----------------------------|----------------------------|
| P1-00 | Froze V2 positioning, protected legacy assets, created `cpp_infer/` entry. | Stop the repo from drifting between training demo and runtime project. | README/README_zh/AGENTS plus C++ workspace skeleton. | Keep README as the main story; avoid scattering tasks into many docs. |
| P1-01 | Added minimal C++17/CMake executable and CTest help smoke. | Prove the repo can build a C++ runtime target. | `yolo_defect_cpp --help` and CTest smoke. | Visual Studio multi-config builds need `ctest -C Debug`. |
| P1-02 | Added no-dependency ConfigLoader and `--config` CLI path. | Make runtime behavior config-driven before adding image/model code. | Parsed input size, class names, thresholds, backend; printed stable summary. | CLI argument errors became the first useful smoke-test failure signal. |
| P1-03 | Added OpenCV image read and YOLO-style preprocess. | Convert a real image into the model-ready tensor format. | `original_size`, `scale`, `padding`, `BGR->RGB`, `[0,1]`, `NCHW`, `1x3x800x800`; CTest 3/3 passed. | OpenCV Windows pack required `OpenCV_DIR=...\x64\vc16\lib` and `PATH=...\x64\vc16\bin`. |
| S1-01 | Added strict `RuntimeConfig + ModelArtifactSpec`, tensor/enumeration validation, declaration-relative paths, and Runtime library/CLI targets. | Make model/runtime assumptions executable and testable before any ORT session. | Clean Release build, stable contract/preprocess summaries, two-working-directory path proof, SHA recheck, ORT SDK gate/DLL staging, and 15/15 CTest. | Keep “declared hash/configured provider” separate from actual metadata/provider; use nonzero+message wrappers for negative CLI tests; GTest waits for S1-04. |
| S1-02 | Added `OnnxRunner` RAII/PImpl, owned `ModelMetadata`, pure actual-vs-declared validation, and `--inspect-model`. | Isolate dependency/session/model-contract failures before tensor wiring and algorithms. | Real ORT 1.19.2 CPU session loaded `best.onnx`; actual 1-in/1-out names, float32 shapes and class channels passed; real/synthetic failures and 29/29 CTest passed. | `GetAvailableProviders()` is inventory, not session assignment; record configured, available, and explicitly registered session provider separately. A profile-free PowerShell avoids Conda replacing the VS toolchain PATH. |
| S1-03 | Added zero-copy CPU input tensor wiring, synchronous `OnnxRunner::run()`, owned `InferenceOutput`, and `--raw-output-summary`. | Isolate tensor shape/lifetime and raw model execution before postprocess algorithms. | Fixed image produced finite `[1,10,13125]` / 131,250-value raw output; invalid 1,919,999-value input failed before ORT tensor/Run; 31/31 CTest passed. | A user-buffer `Ort::Value` does not own the input vector; keep it stable through synchronous Run. ORT owns output only while its Value lives, so copy before return. |
| S1-04 | Added `Detection`/`BoundingBox`, pure YOLOv8 raw-output validation/decode, strict score filtering, IoU, stable class-agnostic NMS, coordinate restore/clip, and a direct `cv::Mat` preprocess boundary. | Prove model-specific algorithms independently of ORT/model variability before connecting user-facing outputs. | Synthetic postprocess GTest 24/24, `cv::Mat` preprocess GTest 7/7, complete CTest 62/62; S1-03 raw regression retained. | Compare float32 scores/IoU in the float32 threshold domain; perform NMS before restore; preserve input order for equal confidence; reject non-`CV_8UC3` Mat input instead of silently applying the wrong normalization. |
| S1-05 | Added the PImpl `DetectorPipeline`, self-contained `SingleImageDetectionResult`, schema-v1 JSON serializer, deterministic OpenCV visualizer, output safety rules, and `--output-json` / `--output-image` / `--overwrite`. | Turn the already-tested seams into the first reproducible, interview-demoable single-image C++ vertical slice while keeping algorithms out of `main.cpp`. | Fixed sample produced 3 `crazing` detections; Python parsed and semantically validated the 1,164-byte JSON; OpenCV read the 39,306-byte PNG; output GTest 6/6, output label 16/16, complete CTest 78/78. | Serialize only owned validated values; escape every JSON string; make overwrite opt-in and protect inputs. `actual_provider` is session-level evidence rather than per-node profiling, and a two-file write is not an atomic transaction. |
| PLAN-20260715 | Aligned repository rules and the bilingual entry points to the latest top-level design; created the long-form large-stage-one plan. | Preserve the verified baseline while preventing the short stage summary from dropping contract, correctness, test, failure, and evidence requirements. | `docs/PLAN.md` -> `AGENTS.md` rules -> README stage/status summary -> `docs/STAGE1_EXECUTION_PLAN.md` one-step plan. | Historical Python metrics, external D010 metrics, and future C++ results must stay explicitly separated. |

## Highlights

- **Best Experimental Result** — Best checkpoint `final_train_2` reaches **mAP@0.5 = 0.743** on NEU-DET
- **Historical PyTorch vs ONNX Count Check** — **50/50** count matches and **146 vs 146** detections, but the sorted sample is all `crazing` and does not prove class/box tolerance
- **Historical V1 Python Benchmarks** — PyTorch CPU **8.43 FPS**; PyTorch GPU (RTX 3060) **110.8 FPS**; Python ORT CPU **24.4 FPS**; Python ORT GPU **72.1 FPS** — all measured on 100 timed images (5 warmup), not C++ results
- **Docker Verified** — `python:3.9-slim` image has been tested with `/health` and `/detect`
- **Clone & Run** — Dataset (28MB) included in the repo, no external downloads needed

## Key Metrics

| Metric | Value |
|--------|-------|
| Best model | `final_train_2` |
| mAP@0.5 | **0.743** |
| mAP@50-95 | **0.388** |
| Historical PT/ONNX same-count ratio | **50 / 50** all-`crazing` images (**100%**, count only) |
| Mean abs count diff | **0.000** |
| Historical PyTorch CPU benchmark | **8.43 FPS** / **118.66 ms** per image |
| Historical PyTorch GPU benchmark (RTX 3060) | **110.8 FPS** / **9.0 ms** per image |
| Historical Python ORT CPU benchmark | **24.4 FPS** / **40.9 ms** per image |
| Historical Python ORT GPU benchmark (RTX 3060) | **72.1 FPS** / **13.9 ms** per image |
| Historical model-size record (`best.pt` / current `best.onnx`) | ~6.0 MiB / ~11.8 MiB; the matching `.pt` was not found in the current workspace or Git history |

## V1 Python Baseline Quick Start

```bash
# Clone (dataset included, ~28MB)
git clone https://github.com/LiuSiChengGitHub/yolo_defect.git
cd yolo_defect

# Install dependencies
conda env create -f environment.yml
conda activate yolo_defect

# Prepare data (VOC XML -> YOLO TXT)
python scripts/prepare_data.py

# Train
python scripts/train.py

# Export ONNX from the default training output
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt

# Inference on a real validation image
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg
```

## Dataset

### NEU-DET: Northeastern University Surface Defect Database

**Source:** [NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/)

The NEU-DET dataset contains 1,800 grayscale images of hot-rolled steel strip surfaces, covering 6 types of typical surface defects:

| Class | English | Chinese | Description |
|-------|---------|---------|-------------|
| 0 | crazing | 龟裂 | Network of fine cracks on the surface |
| 1 | inclusion | 夹杂 | Foreign material embedded in the steel |
| 2 | patches | 斑块 | Irregular discolored areas |
| 3 | pitted_surface | 麻面 | Small pits scattered across the surface |
| 4 | rolled-in_scale | 压入氧化铁皮 | Oxide scale pressed into the surface during rolling |
| 5 | scratches | 划痕 | Linear marks from mechanical contact |

### Statistics

- **Dataset paper / official description:** 1,800 images (300 per class)
- **Files bundled in `data/NEU-DET/`:** 1,800 readable JPG images
- **Image size:** 200 x 200 pixels
- **Format:** JPG (grayscale, 1 channel in annotation but readable as 3-channel)
- **Generated YOLO copy in `data/images/`:** 1,439 train + 361 val images

### Directory Structure

The dataset is pre-split and included at `data/NEU-DET/`:

```
data/NEU-DET/
├── train/                         # 1,439 readable images
│   ├── annotations/               # VOC XML (flat directory)
│   │   ├── crazing_1.xml
│   │   ├── inclusion_1.xml
│   │   └── ...
│   └── images/                    # JPG (subdirectories by class)
│       ├── crazing/
│       ├── inclusion/
│       ├── patches/
│       ├── pitted_surface/
│       ├── rolled-in_scale/
│       └── scratches/
└── validation/                    # 361 XMLs, 361 readable images
    ├── annotations/
    └── images/                    # Same structure as train
```

### Annotation Format

VOC XML format with `<bndbox>` containing absolute pixel coordinates:

```xml
<object>
    <name>crazing</name>
    <bndbox>
        <xmin>2</xmin>
        <ymin>2</ymin>
        <xmax>193</xmax>
        <ymax>194</ymax>
    </bndbox>
</object>
```

Each image may contain multiple bounding boxes (multiple defect instances).

## Data Preparation

### What the conversion does

`prepare_data.py` converts the original VOC XML annotations to YOLO TXT format that Ultralytics YOLOv8 expects.

**VOC XML format** (absolute pixel coordinates):
```
xmin, ymin, xmax, ymax  →  e.g., 2, 2, 193, 194
```

**YOLO TXT format** (normalized center coordinates):
```
class_id cx cy w h  →  e.g., 0 0.487500 0.490000 0.955000 0.960000
```

The normalization formula:
- `cx = (xmin + xmax) / 2 / image_width`
- `cy = (ymin + ymax) / 2 / image_height`
- `w = (xmax - xmin) / image_width`
- `h = (ymax - ymin) / image_height`

### Class Mapping

| Class Name | Class ID |
|------------|----------|
| crazing | 0 |
| inclusion | 1 |
| patches | 2 |
| pitted_surface | 3 |
| rolled-in_scale | 4 |
| scratches | 5 |

### Run

```bash
python scripts/prepare_data.py
# or specify custom paths:
python scripts/prepare_data.py --data-root data/NEU-DET --output-dir data
```

### Output Structure

```
data/
├── images/
│   ├── train/          # Flat directory, all training images
│   └── val/            # Flat directory, all validation images
├── labels/
│   ├── train/          # YOLO TXT labels, one per image
│   └── val/
└── data.yaml           # YOLO dataset config
```

### Important Notes

- The dataset is **already split** into train/validation — no random splitting needed
- `rolled-in_scale` contains a hyphen, so the script uses known class name prefix matching (longest match first) instead of naive underscore splitting
- Images are copied from class subdirectories to a flat output directory (YOLO requirement)
- If the raw dataset is updated manually, rerun `prepare_data.py` so `data/images/` and `data/labels/` stay in sync with `data/NEU-DET/`

## Data Analysis

Running `data_analysis.py` on the converted dataset reveals the following characteristics: the dataset is effectively balanced across all 6 classes, so no oversampling or class-weighting is needed. All images are uniformly 200×200 px. Each image contains between 1 and 9 bounding boxes (mean: 2.33), indicating moderate defect density. Bounding box sizes vary dramatically — from as small as 8×9 px (narrow scratches) to nearly 199×199 px (crazing covering the entire image) — making this a challenging multi-scale detection task. The anchor-free design of YOLOv8 handles this wide size range well without manual anchor tuning. Analysis charts are saved in `docs/assets/`.

```bash
python scripts/data_analysis.py
```

## Training

### Run Training

```bash
# Using YAML config (recommended)
python scripts/train.py --config configs/train_config.yaml

# Or directly via Ultralytics CLI
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `yolov8n.pt` | Pre-trained model variant. `n`=nano (fastest), `s`/`m`/`l`/`x` for larger models |
| `data` | `data/data.yaml` | Dataset configuration file with paths and class names |
| `epochs` | 50 | Total training epochs. More epochs = better convergence, but risk of overfitting |
| `imgsz` | 640 | Input image size. Larger = better accuracy, slower training. Images are resized from 200x200 |
| `batch` | 16 | Batch size. Larger = more stable gradients, more GPU memory needed. Use -1 for auto |
| `lr0` | 0.01 | Initial learning rate. The optimizer adjusts this during training via scheduling |
| `optimizer` | `auto` | Optimizer selection. `auto` picks the best based on model and dataset |
| `mosaic` | 1.0 | Mosaic augmentation probability. Combines 4 images into one, improving small object detection |
| `mixup` | 0.0 | Mixup augmentation probability. Blends two images together for regularization |
| `device` | 0 | CUDA device index. Use `cpu` for CPU training |
| `workers` | 8 | Number of dataloader worker processes for data loading |

### Training Process Overview

1. **Pre-trained weights loading** — YOLOv8n is initialized with COCO pre-trained weights, providing a strong feature extraction baseline (transfer learning)
2. **Data augmentation** — Mosaic (4-image composition), mixup, random flip, HSV adjustment, and scale jitter are applied on-the-fly to improve generalization
3. **Multi-scale training** — Images are randomly resized during training to make the model robust to different object scales
4. **Automatic checkpointing** — `best.pt` (highest mAP) and `last.pt` (latest epoch) are saved under `runs/detect/train/weights/`

## Results

### Experiment Comparison

| Experiment | Model | imgsz | lr0 | epochs | mAP@0.5 | mAP@50-95 | Train Time | Notes |
|------------|-------|-------|-----|--------|---------|-----------|------------|-------|
| baseline | yolov8n | 640 | 0.01 | 50 | **0.734** | 0.390 | 9.4 min | Default config, exceeds 0.70 target |
| exp1 | yolov8n | 512 | 0.01* | 50 | 0.733 | 0.391 | 7.2 min | Faster, but hurts hard texture classes |
| exp2 | yolov8n | 800 | 0.01* | 50 | 0.742 | 0.385 | 13.4 min | Best result in the `optimizer=auto` image-size family |
| exp3_lr01 | yolov8n (SGD) | 640 | 0.01 | 50 | 0.736 | **0.395** | 9.0 min | Best `mAP@50-95`, valid fixed-SGD lr baseline |
| exp4 | yolov8n | 800 | 0.01* | 50 | 0.741 | 0.384 | 13.6 min | `mixup=0.1` did not help |
| exp5 | yolov8n | 800 | 0.01* | 50 | 0.740 | 0.387 | 13.3 min | No-mix augmentation control |
| final_train | yolov8n | 800 | 0.01* | 100 | 0.729 | 0.379 | 26.1 min | Longer training alone did not improve the model |
| final_train_2 | yolov8n (SGD) | 800 | 0.01 | 100 | **0.743** | 0.388 | 25.9 min | Manually combined final candidate, current best `mAP@0.5` |

\* `optimizer=auto` selected AdamW(lr=0.001) at runtime, so `lr0=0.01` was not the effective learning rate.

### Current Model Candidates

- **`final_train_2`** is the current deployment candidate if the headline metric is `mAP@0.5`
- **`exp3_lr01`** remains important because it has the best `mAP@50-95` under the cleanest fixed-SGD design
- **`final_train`** shows an important lesson: longer training alone is not enough if the optimizer/parameter family is not the strongest one

### Per-Class AP (Current Best: `final_train_2`)

| Class | AP@0.5 | Precision | Recall |
|-------|--------|-----------|--------|
| patches | 0.920 | 0.856 | 0.850 |
| inclusion | 0.827 | 0.773 | 0.742 |
| pitted_surface | 0.807 | 0.821 | 0.701 |
| scratches | 0.803 | 0.602 | 0.843 |
| rolled-in_scale | 0.553 | 0.507 | 0.462 |
| crazing | 0.550 | 0.513 | 0.543 |

### Comparison Insight

- `imgsz=800` helped the overall `mAP@0.5` direction, but did not solve `crazing` by itself
- Fixed-SGD learning-rate ablation showed that `lr0=0.01` clearly outperformed `0.001` under the same 50-epoch budget
- `mixup=0.1` did not help this industrial fine-texture task, while disabling sample mixing preserved some classes better
- The manually combined `final_train_2` run became the strongest `mAP@0.5` result and improved `crazing` to `0.550`
- Practical conclusion: the best final model came from a **validated cross-experiment combination**, not from longer training alone

### Training Curves

![Training Results](docs/assets/results_final_train_2.png)

### PR Curve

![PR Curve](docs/assets/PR_curve_final_train_2.png)

### Confusion Matrix

![Confusion Matrix](docs/assets/confusion_matrix_final_train_2.png)

### Sample Predictions

![Validation Predictions](docs/assets/val_pred_sample_final_train_2.jpg)

## ONNX Deployment

### Why ONNX?

- **Cross-platform** — Run on Windows, Linux, macOS, edge devices without PyTorch installed
- **Framework-agnostic** — No dependency on the training framework at inference time
- **Performance** — ONNX Runtime provides optimized inference with hardware-specific acceleration (CUDA, TensorRT, DirectML)
- **Smaller footprint** — No need to ship the entire PyTorch runtime in production

### Export

```bash
# Quick Start path: export the checkpoint produced by the default `scripts/train.py` run
python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt
# Output: models/best.onnx
```

If you want to reproduce the best reported metrics in this README, export the best experiment checkpoint instead:

```bash
python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800
```

### Inference

```bash
# Single image
python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg

# Batch (entire directory)
python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/
```

The current ONNX deployment target is exported with `imgsz=800`, so the model input is `[1, 3, 800, 800]` and the raw output tensor is `[1, 10, 13125]` (`4 bbox params + 6 class scores` across all candidate locations).

### Performance Comparison

| Check | Value | Evidence |
|-------|-------|----------|
| Best historical PyTorch validation result | **mAP@0.5 = 0.7433**, **mAP@50-95 = 0.3880** | `docs/archive/experiment_log.md` |
| Historical PyTorch CPU benchmark | **8.43 FPS**, **118.66 ms/image** over **100** timed images | `results/pytorch_benchmark_100.json` |
| Historical PyTorch GPU benchmark (RTX 3060) | **110.8 FPS**, **9.0 ms/image** over **100** timed images | `results/pytorch_benchmark_gpu.json` |
| Historical Python ORT CPU benchmark | **24.4 FPS**, **40.9 ms/image** over **100** timed images | `results/onnx_benchmark_cpu.json` |
| Historical Python ORT GPU benchmark (RTX 3060) | **72.1 FPS**, **13.9 ms/image** over **100** timed images | `results/onnx_benchmark_gpu.json` |
| Historical PT vs ONNX detection-count match | **50 / 50** all-`crazing` images (**100%**, count only) | `results/pt_onnx_compare/compare_50_summary.json` |
| Historical PT vs ONNX total detections | **146 vs 146** | `results/pt_onnx_compare/compare_50_summary.json` |
| Historical mean absolute count difference | **0.000** | `results/pt_onnx_compare/compare_50_summary.json` |
| Model-size record | Historical `best.pt = 6,286,072 bytes`; current tracked `best.onnx = 12,336,935 bytes`; the matching `.pt` was not found in the current workspace or Git history | artifact/evidence audit |

All latency rows above are V1 Python/Python-ORT evidence. The first C++ Runtime latency table will be created in S1-08 and must use a separately documented protocol.

### YOLODetector Class (`src/detector.py`)

The `YOLODetector` class provides a clean 3-step inference API:

1. **`preprocess(image)`** — BGR to RGB, letterbox resize (aspect-ratio preserving with gray padding, matching Ultralytics training preprocessing), normalize to 0-1, HWC to CHW, add batch dimension
2. **`predict(image)`** — Run ONNX inference, parse output tensor, apply confidence filtering and NMS, return detections list
3. **`draw(image, detections, class_names)`** — Draw bounding boxes with class labels and confidence scores

This class is designed to be directly reused by the FastAPI service in `api/`, keeping inference logic in one place.

For debugging, `scripts/debug_detector.py` manually expands the preprocessing and forward path and prints 5 key shapes:
- original image shape
- resized image shape
- CHW tensor shape
- batched input shape
- raw ONNX output shape

### FastAPI API Usage

The project now includes a minimal FastAPI service in `api/app.py` with two endpoints:

- `GET /health` — health check for service and model readiness
- `POST /detect` — upload one image and receive detection results in JSON

Start the API service:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Health check example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model": "best.onnx",
  "request_stats": {
    "total_requests": 0,
    "avg_response_time_ms": 0.0
  }
}
```

Detection request example:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/images/val/crazing_241.jpg"
```

Example response:

```json
{
  "filename": "crazing_241.jpg",
  "count": 3,
  "image_size": {
    "width": 200,
    "height": 200
  },
  "model": "best.onnx",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45,
  "inference_time_ms": 20.57,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crazing",
      "confidence": 0.4457,
      "bbox": [0.0, 53.68, 176.91, 146.23]
    }
  ]
}
```

Notes:

- The upload field name must be `file`
- The API returns JSON results, not visualization images
- `inference_time_ms` is service-side model inference time; client-observed response time can be larger under concurrent load
- `scripts/benchmark_api.py` can be used for a simple concurrency benchmark of `POST /detect`

Current local verification:

- `GET /health` returned `200 OK` with `{"status":"ok","model":"best.onnx"}`
- `POST /detect` on `data/images/val/crazing_241.jpg` returned `count=3`
- `scripts/benchmark_api.py` is included for local concurrency testing, but its raw benchmark log is not committed yet, so throughput numbers are omitted here

### Docker Deployment

A minimal deployment image is now provided via `Dockerfile`:

- base image: `python:3.9-slim`
- runtime deps only: `requirements-api.txt`
- copied into image: `src/`, `api/`, `models/`
- exposed port: `8000`

Build and run:

```bash
docker build -t yolo-defect-api .
docker run --rm -p 8000:8000 yolo-defect-api
```

Quick verification:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/detect \
  -F file=@data/images/val/crazing_241.jpg
```

Current Docker verification:

- `GET /health` returned `status=ok`, `model=best.onnx`
- `POST /detect` on `crazing_241.jpg` returned `count=3`
- Numeric Docker benchmark logs are not committed yet, so only endpoint-level verification is reported here

## Research Repository Coordination

This repository now works together with `paper_detect` instead of duplicating it.

| Repository | Role | Responsibilities |
|------------|------|------------------|
| `paper_detect` | Research and paper repository | Dataset splits, baselines, method changes, ablations, formal evaluation, formal ONNX export, PyTorch/ONNX consistency checks, Python ORT benchmark, paper tables and figures |
| `yolo_defect` | Portfolio and deployment engineering repository | Stable demo, ONNX/Python/C++ inference, OpenCV preprocessing, CMake, GTest, deployment benchmark, FastAPI/Docker, README and interview-facing documentation |

In short, `paper_detect` proves why the model is better; `yolo_defect` proves how the model runs reliably.

The two repositories should exchange artifacts, benchmark logs, commit/tag references, environment notes, and result tables. They should not copy entire codebases back and forth.

### Artifact Contract

An artifact is not just one model file. It is a small evidence package that explains, reproduces, and validates a model:

```text
artifacts/2026-06-20_method_v1/
├── best.pt
├── best.onnx
├── train_config.yaml
├── export_config.yaml
├── input_spec.json
├── class_map.json
├── metrics.json
├── per_class_ap.csv
├── complexity.json
├── compare_pt_onnx.json
├── latency_python_ort.json
└── result_card.md
```

Formal training, evaluation, and paper ONNX export are owned by `paper_detect`. This repository consumes the exported artifact, especially `best.onnx`, `input_spec.json`, `class_map.json`, and `result_card.md`, then runs deployment-oriented validation and benchmarks.

If this repository re-exports ONNX, that export is only for portfolio reproduction or sanity checks. It is not the source of formal paper results.

### Benchmark Feedback

Deployment logs generated here can flow back into `paper_detect` for paper tables, for example:

```text
results/cpp_benchmark/
├── 2026-06-20_method_v1_cpp_ort_cpu.json
├── 2026-06-20_method_v1_cpp_ort_gpu.json
└── 2026-06-20_method_v1_consistency.json
```

Each benchmark log should record:

- the artifact/model used;
- the `yolo_defect` commit or tag;
- the command line;
- hardware and operating system;
- ONNX Runtime version and execution provider;
- whether preprocessing and postprocessing/NMS are included;
- warmup, repeat count, mean latency, P50/P90/P99 latency, and FPS.

As long as the experiment really runs, the logs are traceable, and the paper describes the measurement scope honestly, this is a valid cross-repository experiment organization.

### Smoke Tests

Smoke tests prove that a chain runs without crashing. They are not final accuracy or performance claims.

Required smoke tests:

- train smoke test: one short training run in `paper_detect`;
- export smoke test: one temporary ONNX export in `paper_detect`;
- Python ORT smoke test: load ONNX and print output shapes;
- C++ ORT smoke test: compile, load ONNX, print input/output shapes and basic latency in `yolo_defect`.

Current repository versioning:

- `v0.1-intern0`: stable internship/portfolio baseline before C++ deployment work;
- `deploy-cpp`: branch for C++ ONNX Runtime, OpenCV, CMake, GTest, and benchmark development.

## Project Structure

```
yolo_defect/
├── Dockerfile                    # Docker image for FastAPI deployment
├── AGENTS.md                     # Latest PLAN-derived collaboration and advancement rules
├── README.md                     # This file (English)
├── README_zh.md                  # Chinese version
├── LICENSE                       # MIT License
├── requirements-api.txt          # Minimal runtime dependencies for Docker/API
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment (PyTorch + CUDA)
├── .gitignore                    # Ignore rules
├── data/
│   ├── data.yaml                 # YOLO dataset config (auto-generated)
│   └── NEU-DET/                  # Original dataset (committed to git)
│       ├── train/                #   Training split (~240/class)
│       └── validation/           #   Validation split (~60/class)
├── scripts/
│   ├── prepare_data.py           # VOC XML -> YOLO TXT converter
│   ├── data_analysis.py          # Dataset statistics & visualization
│   ├── train.py                  # Training entry point (reads YAML config)
│   ├── evaluate.py               # Model evaluation + PR curve + confusion matrix
│   ├── export_onnx.py            # ONNX model export
│   ├── debug_detector.py         # Debug script for intermediate shapes / ONNX output
│   ├── compare_pt_onnx.py        # 50-image approximate comparison of PT vs ONNX outputs
│   ├── benchmark_pytorch.py      # PyTorch CPU/GPU FPS benchmark on a fixed image subset
│   ├── benchmark_onnx.py         # ONNX CPU/GPU FPS benchmark on a fixed image subset
│   ├── benchmark_api.py          # Simple concurrent benchmark for POST /detect
│   ├── analyze_failures.py       # Failure-case analysis for false positives/negatives
│   ├── select_representative_examples.py  # Select representative examples for README
│   └── inference_onnx.py         # ONNX inference (single + batch)
├── src/
│   ├── __init__.py
│   └── detector.py               # YOLODetector class (ONNX inference, FastAPI reuse)
├── api/
│   └── app.py                    # FastAPI service (`GET /health`, `POST /detect`)
├── cpp_infer/                    # V2 C++ runtime workspace
│   ├── CMakeLists.txt            # yolo_defect_runtime library + yolo_defect_cpp CLI + CTest
│   ├── README.md                 # C++ contract, dependencies, commands, and evidence
│   ├── artifacts/                # ModelArtifactSpec declarations
│   │   └── yolov8_neu_det.artifact.txt
│   ├── configs/default_config.txt# RuntimeConfig policy and artifact path
│   ├── include/yolo_defect_cpp/  # Public contract/preprocess/runner/postprocess/output value APIs
│   │   ├── detector_pipeline.h   # PImpl single-image Runtime orchestration boundary
│   │   ├── detection_result.h    # Self-contained result and image metadata
│   │   └── result_writer.h       # Stable JSON/visualization output request API
│   ├── src/                      # Parser, preprocess, ORT run, postprocess, pipeline, writers, thin CLI
│   ├── results/demo/             # Verified S1-05 JSON and visualized PNG evidence
│   └── tests/                    # GTest logic/output + CTest contract/session/integration/failure gates
├── configs/
│   ├── train_config.yaml         # Baseline training hyperparameters
│   └── exp*.yaml                 # Experiment configs (imgsz/lr/augment/final runs)
├── models/
│   └── best.onnx                 # Tracked YOLOv8/NEU-DET P0 artifact
├── docs/
│   ├── PLAN.md                   # Latest project design, advancement rules, and large stages
│   ├── STAGE1_EXECUTION_PLAN.md  # Current large-stage-one dynamic small-stage plan
│   ├── archive/                  # Historical route/experiment documents
│   └── assets/                   # PR curves, demo GIFs, plots
└── runs/                         # YOLO training outputs (gitignored)
```

### Design Principles

- **`scripts/`** — One-off scripts for data processing, training, evaluation, export. Run from command line with argparse.
- **`src/`** — Reusable modules. `detector.py` is imported by both `inference_onnx.py` and the FastAPI service.
- **`cpp_infer/`** — V2 C++ deployment workspace. It now owns the Runtime library/CLI boundary, strict Runtime/artifact contract, OpenCV preprocessing, ORT RAII session, actual metadata validation, safe tensor ownership, pure YOLOv8 postprocess, the single-image Pipeline, stable JSON/visualization outputs, GTest, and the 78-case CTest gate; later S1 steps add broader failure hardening, consistency, and benchmark evidence.
- **`configs/`** — Separated hyperparameters. Easy to track experiments by diffing config files.

## Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | Language | 3.9.25 |
| C++ | V2 runtime language | C++17 |
| MSVC | Verified x64 C++ compiler | 19.50.35721.0 |
| PyTorch | Deep learning framework | 2.0.0 |
| Ultralytics | YOLOv8 training & inference | 8.4.24 locally and in artifact metadata |
| ONNX | Model interchange format | Python package 1.19.1; artifact opset 17 |
| ONNX Runtime | Python baseline plus C++ RAII session, metadata validation, raw inference, and S1-05 single-image pipeline execution | Python 1.19.2 and official Windows x64 CPU C++ SDK 1.19.2; input is borrowed only through synchronous `Session::Run`, output is copied before ORT ownership ends, and output records session-level CPU provider evidence |
| OpenCV | Python utilities, verified file/`CV_8UC3 cv::Mat` C++ preprocessing, deterministic headless drawing, image encoding, and output read-back | Windows C++ 4.8.0 x64 vc16 |
| CMake | Active C++ build system and CTest entry | 4.1.1-msvc1 |
| GTest | Synthetic postprocess/preprocess plus stable JSON-output unit tests linked to `yolo_defect_runtime` | Official GitHub v1.17.0 archive; commit `52eb8108c5bdec04579160ae17225d66034bd723` and SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84` pinned; verified 24 postprocess + 7 preprocess + 6 output cases; offline source override requires a separately hash-verified extraction |
| Matplotlib | Visualization & plotting | (via ultralytics) |
| FastAPI | REST API service | latest |
| Conda | Environment management | — |

## Key Design Decisions

### Model Selection

YOLOv8 is the latest generation with improved architecture (C2f modules, anchor-free detection, decoupled head). The `nano` variant is chosen because:
- NEU-DET is a small dataset (1,800 images) — a larger model would overfit
- Edge deployment friendly — fast inference on CPU and mobile devices
- Easy to scale up: if `n` isn't enough, swap to `s`/`m` with one config change

### Dataset Inclusion

The NEU-DET dataset is only 28MB. Including it means:
- `git clone` → immediately runnable, no manual downloads or registration
- Guaranteed reproducibility — the exact same data every time
- Easy to verify — anyone can reproduce the pipeline in minutes

### Config Management

- **Traceability** — Each experiment's config is a file that can be version-controlled and diffed
- **Reproducibility** — Re-run any experiment by pointing to its config
- **Comparison** — Side-by-side parameter comparison across experiments

### Detector Module

- **Separation of concerns** — Inference logic is independent of the training framework
- **FastAPI reuse** — The API service imports `YOLODetector` directly, no code duplication
- **Testing** — The detector can be unit-tested in isolation

## Roadmap

### V1 Baseline Already Done

- [x] Baseline training and experiment tracking
- [x] Hyperparameter tuning (imgsz / lr / augment comparisons)
- [x] Bad sample analysis (misdetections, class confusion)
- [x] ONNX export and CPU inference validation
- [x] ONNX accuracy alignment (PyTorch vs ONNX)
- [x] FastAPI service with file upload endpoint
- [x] Docker containerization for deployment
- [x] Demo GIF for inference walkthrough

### V2 Project 1 Task Queue

The V2 queue follows `docs/PLAN.md`. Before entering each large stage, Codex reads the current repository and creates that large stage's small-stage plan; only one small stage is executed at a time, then work pauses for acceptance and the remaining plan is revalidated. `docs/STAGE1_EXECUTION_PLAN.md` is the current justified long-form plan. README remains the task/status/evidence entry point.

| ID | Status | Task | Scope | Acceptance |
|----|--------|------|-------|------------|
| P1-00 | Done | README / AGENTS / C++ workspace entry | Freeze V2 positioning, Codex boundaries, task queue, and `cpp_infer/` skeleton | README/README_zh explain that YOLO/NEU-DET are carriers and C++ Runtime is the core; `AGENTS.md` protects legacy assets; `cpp_infer/` exists without full inference implementation |
| P1-01 | Verified with VS Developer Command Prompt | CMake skeleton | Add the first minimal CMake project and executable target | `cpp_infer` has a minimal C++17 CMake target, executable target, and CTest smoke test. Configure/build/run pass in the Visual Studio 2026 Developer Command Prompt; Visual Studio multi-config builds require `ctest -C Debug` |
| P1-02 | Verified with NMake CTest smoke | ConfigLoader | Load `input_width`, `input_height`, `class_names`, `score_threshold`, `nms_threshold`, and `backend` | `cpp_infer/configs/default_config.txt` is parsed into a typed `RuntimeConfig`; `yolo_defect_cpp --config ...` prints a stable config summary; CTest covers the config smoke path without OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, NMS, or benchmark wiring |
| P1-03 | Verified with OpenCV CTest smoke | OpenCV preprocess | Read an image, print shape/channels, letterbox, BGR to RGB, normalize, HWC to CHW | `--config ... --image ...` reads a real validation image and prints original shape, target input size, scale, padding, color conversion, normalization, NCHW tensor shape, and tensor element count |
| S1-01 | **Verified; L1 accepted** | Baseline contract and engineering boundary | Strict Runtime/artifact schemas, declaration-relative paths, Runtime library/CLI targets, configurable ORT SDK boundary, and CTest positive/negative paths; GTest remains deferred | Clean Release library/CLI build, stable summaries, path-independence proof, SHA recheck, actionable failures, and 15/15 CTest; no session/inference |
| S1-02 | **Verified; L1 accepted** | ORT session and metadata validation | RAII/PImpl session, explicit CPU EP, actual version/provider/count/name/shape/dtype/class-contract inspection, and synthetic validator | `models/best.onnx` loads; actual float32 `[1,3,800,800] -> [1,10,13125]` metadata passes; real/synthetic negative paths and 29/29 CTest pass; no `Session::Run` |
| S1-03 | **Verified; L1 accepted** | Tensor wiring and raw inference | Borrow the preprocess vector for a CPU ORT tensor, run synchronously, validate and copy raw output into independent storage | Fixed image produces finite owned `[1,10,13125]` / 131,250-value output; invalid length fails before Run; 31/31 CTest pass; no decode |
| S1-04 | **Verified; L1 accepted** | YOLO decode/filter/NMS/coordinate restore | Pure model-specific postprocess functions, direct `CV_8UC3 cv::Mat` preprocess boundary, and synthetic GTest | Float32 strict thresholds, BCN decode, stable class-agnostic input-space NMS, empty output, clipping, odd/non-square inverse letterbox are deterministic; GTest 31/31 and complete CTest 62/62 pass |
| S1-05 | **Implemented and verified; awaiting L1 acceptance** | End-to-end CLI, JSON, and visualization | `DetectorPipeline` orchestrates the single-image vertical slice; `DetectionResult` snapshots owned output metadata; `ResultWriter` emits stable JSON v1 and deterministic headless visualization with explicit file safety rules | Fixed command creates 3 `crazing` detections, Python-parseable JSON and an OpenCV-readable PNG; empty detections remain valid `[]`; output GTest 6/6, output label 16/16, complete CTest 78/78 |
| S1-06 | **Next after S1-05 L1** | Automated and failure-path gate | Expand GTest/CTest across contract, preprocess, metadata, postprocess, integration, and core failures | Tests cover missing model, shape/dtype/class mismatch, damaged image, and empty output with nonzero actionable errors where appropriate |
| S1-07 | Pending | Fixed-sample Python ORT/C++ consistency | Compare a committed six-class manifest under the same CPU provider and postprocess semantics | Count/class match and predeclared confidence/box/IoU tolerances pass or produce per-image diagnostics; no unsupported direct PT rerun claim |
| S1-08 | Pending | Reproducible Release benchmark | Measure decode/preprocess/infer/postprocess/pipeline with warmup/repeat and environment/memory metadata | JSON contains mean/P50/P95, throughput, build/provider/model/sample metadata, and Windows Peak Working Set or an explicit unsupported value |
| S1-09 | Pending | Large-stage-one closure | Clean-build all gates, align documentation/evidence, and complete L2 interview acceptance | Fixed demo/tests/consistency/benchmark pass; user can explain for five minutes, handle follow-ups/failures, and change one behavior plus its test |

Large stage two will be decomposed only when S1-09 is accepted. Its fixed boundary is P0 evidence hardening and INT8 PTQ (QAT only if justified). Large stage three then chooses condition-gated P1 extensions. TensorRT is not an unconditional queue item, and a Project 2 `inference_event` is an optional bridge rather than a large-stage-one acceptance item.

### P1-01 CMake Skeleton Commands

P1-01 only establishes a C++17/CMake entry point. It intentionally does not include OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, or NMS.

```powershell
# Configure
cmake -S cpp_infer -B cpp_infer\build

# Build
cmake --build cpp_infer\build

# Run: Visual Studio multi-config generators usually place the executable here
.\cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help

# Run: single-config generators usually place the executable here
.\cpp_infer\build\bin\yolo_defect_cpp.exe --help

# Smoke test for Visual Studio multi-config generators
ctest --test-dir cpp_infer\build -C Debug --output-on-failure
```

Local verification on 2026-06-05: configure and build passed in the Visual Studio 2026 Developer Command Prompt. `ctest --test-dir cpp_infer\build --output-on-failure` failed because Visual Studio is a multi-config generator and needs a configuration name. `ctest --test-dir cpp_infer\build -C Debug --output-on-failure` passed, and `cpp_infer\build\bin\Debug\yolo_defect_cpp.exe --help` printed the P1-01 skeleton help text.

### P1-02 ConfigLoader Commands

P1-02 adds a no-third-party-dependency `key = value` config parser and a `--config` CLI path. It intentionally does not connect OpenCV, ONNX Runtime, GTest, preprocessing, postprocessing, NMS, or benchmark logic.

```cmd
:: Run from a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_02
cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles"
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Expected config summary fields:

- `input_width: 800`
- `input_height: 800`
- `class_count: 6`
- `class_names: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`
- `score_threshold: 0.25`
- `nms_threshold: 0.45`
- `backend: cpu`

Local verification on 2026-06-10: configure/build/run/CTest passed in a Visual Studio 2026 Developer Command Prompt with the NMake build tree under `%TEMP%`. The config smoke test first failed against the P1-01 skeleton with `Unknown argument: --config`, then passed after the ConfigLoader implementation. This remains historical evidence; use the current S1-01 Quick Start above for the active schema and dependency boundary.

### P1-03 OpenCV Preprocess Commands

P1-03 adds OpenCV image reading and YOLO-style letterbox preprocessing. It intentionally does not connect ONNX Runtime, inference, postprocessing, NMS, benchmark, or GTest.

```cmd
:: Run from a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_03
set PATH=D:\01_Base\Tools\opencv\build\x64\vc16\bin;%PATH%

cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles" -DOpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --image data\images\val\crazing_241.jpg

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Expected preprocess summary fields:

- `original_size: 200x200`
- `channels: 3`
- `input_size: 800x800`
- `resized_size: 800x800`
- `scale: 4.000000`
- `padding: left=0, top=0, right=0, bottom=0`
- `color: BGR->RGB`
- `normalization: float32 [0, 1]`
- `layout: NCHW`
- `tensor_shape: 1x3x800x800`
- `tensor_elements: 1920000`

Local verification on 2026-06-13: the P1-03 smoke test first failed against the P1-02 CLI with `--config expects exactly one config file path.` Configure/build/run/CTest then passed after adding OpenCV and `ImagePreprocessor`. The local OpenCV Windows pack requires `OpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib`; pointing to the top-level `D:\01_Base\Tools\opencv\build` was not sufficient for this NMake build.

### S1-01 Contract and Build Boundary Commands

S1-01 uses the current two-file schema and a fresh Release/NMake tree. It validates the external ORT C++ SDK boundary but does not create a session or run inference.

```powershell
$ToolsRoot = 'D:\01_Base\Tools'
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;' + `
  'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP 'yolo_defect_s1_01'

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --image `
  (Resolve-Path 'data\images\val\crazing_241.jpg').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config `
  (Resolve-Path 'cpp_infer\tests\fixtures\runtime\invalid_provider.txt').Path

ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
(Get-FileHash models\best.onnx -Algorithm SHA256).Hash
```

Local verification on 2026-07-18: MSVC 19.50.35721.0/OpenCV 4.8.0 built `yolo_defect_runtime.lib` and `yolo_defect_cpp.exe`, staged the pinned 1.19.2 `onnxruntime.dll`, passed 15/15 CTest in 0.73 seconds, preserved the preprocess output, rejected `provider = cuda` with exit 1 and an expected/actual/action message, proved identical resolved artifact/model paths from two working directories, and rechecked the declared SHA-256. These are contract/build results, not ORT session or inference results.

### S1-02 ORT Session and Metadata Inspection Commands

S1-02 loads the real ONNX and validates actual metadata. It deliberately stops before input tensor construction and `Session::Run`.

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
powershell.exe -NoProfile -NoExit
```

```powershell
$ToolsRoot = 'D:\01_Base\Tools'
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_02_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --inspect-model
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

Local verification on 2026-07-26: ORT runtime 1.19.2 reported available providers `[AzureExecutionProvider,CPUExecutionProvider]`; `OnnxRunner` explicitly registered `CPUExecutionProvider` and created the session. Actual input was `images` tensor float32 `[1,3,800,800]`; actual output was `output0` tensor float32 `[1,10,13125]`; metadata contract validation passed. The 29-case CTest gate passed, including real input-size/class-count declaration mismatches and synthetic count/name/shape/dtype/provider failures. No input tensor or inference was executed.

### S1-03 Input Tensor and Raw Inference Commands

S1-03 connects the existing preprocess vector to one synchronous ORT run and copies the validated raw output into project-owned storage. It deliberately stops before decode, score filtering, NMS, JSON, visualization, and benchmark work.

```powershell
$ToolsRoot = 'D:\01_Base\Tools'
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_03_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
& "$BuildDir\bin\yolo_defect_cpp.exe" --config $Config --image $Image `
  --raw-output-summary
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

Local verification on 2026-07-30: the fixed image produced input float32 `[1,3,800,800]` with 1,920,000/1,920,000 finite values and owned raw output float32 `[1,10,13125]` with 131,250/131,250 finite values. Output range was `[0,795.04126]`. The invalid 1,919,999-value path failed before `Ort::Value` construction/Run, and 31/31 CTest passed. This is raw-execution evidence, not decoded detection correctness or performance evidence.

### S1-04 Pure YOLOv8 Postprocess and GTest Commands

S1-04 keeps the CLI at the raw-output boundary and validates postprocess independently with synthetic tensors, boxes, and images. The Runtime target now contains the pure postprocessor; GTest executables link `yolo_defect::runtime`, never `main.cpp`, and do not need an ORT session or real model to prove algorithm behavior.

```powershell
$ToolsRoot = 'D:\01_Base\Tools'
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;' + `
  'D:\01_Base\Tools\opencv\build\x64\vc16\bin;' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_04_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR='D:\01_Base\Tools\opencv\build\x64\vc16\lib' `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir

ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

For an offline clean configure, add the following only after verifying that the extracted source came from the pinned v1.17.0 archive whose SHA-256 is `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84`:

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='<verified-google-test-source>'
```

Local verification on 2026-08-15: the 24-case postprocess GTest target and 7-case `cv::Mat` preprocess GTest target both passed, and the S1-04 gate passed 62/62. The tests freeze float32-domain strict score/NMS threshold equality, stable equal-score input order, class-agnostic NMS in model-input coordinates, restore/clip ordering, and `CV_8UC3` preprocessing behavior. This was the pure-algorithm evidence before S1-05 connected the accepted behavior to user-facing outputs.

### S1-05 Single-Image CLI, JSON, and Visualization Commands

S1-05 keeps `main.cpp` as a CLI coordinator and moves the vertical slice into `DetectorPipeline` plus owned detection/output value types. The first run creates parents and refuses existing files; the reproducible demo below passes `--overwrite` explicitly because the verified evidence files are already present.

```powershell
$ToolsRoot = 'D:\01_Base\Tools'
$env:ONNXRUNTIME_ROOT = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$env:PATH = (Join-Path $ToolsRoot 'VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin') + ';' + `
  (Join-Path $ToolsRoot 'opencv\build\x64\vc16\bin') + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_05_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$ToolsRoot\opencv\build\x64\vc16\lib" `
  -DONNXRUNTIME_ROOT="$env:ONNXRUNTIME_ROOT" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir

$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$OutputJson = Join-Path (Get-Location) `
  'cpp_infer\results\demo\crazing_241.detections.json'
$OutputImage = Join-Path (Get-Location) `
  'cpp_infer\results\demo\crazing_241.visualized.png'

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config --image $Image `
  --output-json $OutputJson --output-image $OutputImage --overwrite
python -m json.tool $OutputJson
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir --output-on-failure
```

Local verification on 2026-08-16: the fixed command completed a real ORT CPU Run and emitted three `crazing` detections. Python's standard JSON module accepted the 1,164-byte schema-v1 document, and OpenCV read the 39,306-byte visualization as a `200x200` `CV_8UC3` image. The JSON and PNG SHA-256 values were respectively `E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049` and `3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`. Output GTest passed 6/6, the `output` label passed 16/16, and the clean complete gate passed 78/78. These are single-image functionality and reproducibility results, not consistency or performance evidence.

### V2 Entry Log

| Date | Change | Purpose |
|------|--------|---------|
| 2026-06-04 | Established P1-00 V2 entry: README positioning, Codex boundary file, and `cpp_infer/` skeleton | Make the project explainable as an industrial vision AI Runtime project before deeper C++ implementation starts |
| 2026-06-05 | Verified P1-01 CMake skeleton in the Visual Studio 2026 Developer Command Prompt | Confirmed configure/build/run/CTest smoke test; documented the `ctest -C Debug` requirement for Visual Studio multi-config builds |
| 2026-06-10 | Added P1-02 ConfigLoader and `--config` smoke path | Introduced a typed no-dependency runtime config parser and documented the build/run/CTest evidence before moving toward OpenCV preprocessing |
| 2026-06-13 | Added P1-03 OpenCV read-image and letterbox preprocess smoke path | Confirmed real-image preprocessing output, including original shape, RGB conversion, normalization, NCHW layout, scale, padding, and tensor shape before ONNX Runtime integration |
| 2026-06-29 | Aligned README with the then-current route, now archived as `docs/archive/路线0628.md` | Recorded top-level design, D010/paper_detect artifact path, required README sections, phase queue placeholders, and teaching log so later work stays on the C++ Runtime route |
| 2026-07-15 | Replaced the active route source with `docs/PLAN.md`, updated AGENTS and both entry READMEs, and added `docs/STAGE1_EXECUTION_PLAN.md` | Adopted the latest nine-part teaching closure, authoritative P0/P1 boundaries, artifact gates, four large stages, verified no current direction drift, and dynamically planned S1-01 through S1-09 |
| 2026-07-16 | Completed pre-stage-one readiness without starting S1-01 | Verified the x64 VS terminal and ORT C++ SDK, passed a new clean 3/3 CTest, froze a SHA-256-pinned GTest v1.17.0 FetchContent plan, and recorded the owner-confirmed model lineage plus public-distribution license checkpoints in `docs/PRE_STAGE1_READINESS.md` |
| 2026-07-18 | Completed S1-01 Runtime/artifact contract and engineering boundary | Added strict two-file schemas, model/tensor/enumeration checks, declaration-relative paths, Runtime library/CLI targets, configurable ORT SDK validation/DLL staging, and 15-case CTest evidence; preserved AGPL metadata as a distribution checkpoint and stopped before ORT session/inference |
| 2026-07-26 | Completed S1-02 ORT session and actual metadata validation | Added RAII/PImpl `OnnxRunner`, owned `ModelMetadata`, explicit CPU EP/session policy, `--inspect-model`, actual-vs-declared validation, and a 29-case real/synthetic CTest gate; stopped before input tensor construction and `Session::Run` |
| 2026-07-30 | Completed S1-03 input tensor and owned raw output boundary | Added zero-copy borrowed CPU input, synchronous `Session::Run`, overflow/shape/count/finite checks, copied `InferenceOutput`, bounded CLI summary, invalid-length pre-Run failure, and a 31-case CTest gate; stopped before decode/NMS |
| 2026-08-15 | Completed S1-04 pure YOLOv8 postprocess and `cv::Mat` test boundary | Added validated BCN decode, no-objectness class argmax, float32 strict filtering, `xywh -> xyxy`, robust IoU, stable class-agnostic model-space NMS, inverse letterbox/clip, direct `CV_8UC3` preprocessing, pinned GTest integration, and a 62-case CTest gate; stopped before detection CLI/JSON/visualization |
| 2026-08-16 | Completed S1-05 fixed single-image CLI, JSON, and visualization vertical slice | Added PImpl pipeline orchestration, an owned detection result, stable escaped JSON v1, deterministic headless OpenCV drawing, parent/overwrite/protected-path rules, fixed demo artifacts and hashes, six output GTests, a 16-case output label, and a 78-case clean CTest gate; stopped before S1-06 failure hardening |

## License

Repository-authored source code is licensed under the MIT License — see [LICENSE](LICENSE). This statement does not automatically cover `models/best.onnx` or the NEU-DET dataset.

The tracked ONNX metadata declares `AGPL-3.0`. [Ultralytics' official licensing guidance](https://www.ultralytics.com/license) states that Ultralytics-trained models use AGPL-3.0 by default unless an applicable commercial license is obtained. The [official NEU dataset page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265) provides downloads and citation guidance, but this audit did not find an explicit redistribution license there. These are provenance/distribution checkpoints rather than a legal conclusion; see [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md).

The declared use is personal learning, so an Enterprise license is not a prerequisite for local development. The owner selected option A—continue public distribution of the ONNX and NEU-DET—so noncommercial intent does not remove the need to preserve the model's license notice and verify the dataset's redistribution basis before the release position is frozen.

The NEU-DET dataset is provided by Northeastern University (NEU). Please cite the original paper if you use this dataset in academic work:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.
