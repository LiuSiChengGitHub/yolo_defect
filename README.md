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

Current V1 assets remain valuable: training, ONNX export, the historical 50-image PyTorch/ONNX detection-count check, Python ONNX Runtime inference, FastAPI, Docker, and benchmark scripts. V2 builds on these assets through `cpp_infer/` instead of rewriting them.

Current V2 status: **S1-08 is L1 accepted. The S1-09 automatic gate passes; the user L2 gate is pending. Large Stage One is therefore not yet complete, and Large Stage Two has not started.** A fresh temporary Release build passed 106/106 CTests in 19.91 seconds, reproduced the fixed three-detection JSON/PNG demo, passed the repository-resident six-class 30-image Python ORT/C++ ORT gate, reran the warmup-10/repeat-100 CPU benchmark, and reconfirmed four actionable nonzero CLI failures. S1-09 adds no product behavior; it closes automated reproducibility and prepares the user-owned explanation and modification exercise.

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

Current verified chain through the S1-09 automatic gate:

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
-> four GTest targets: postprocess 25 + preprocess 7 + output 7 + benchmark 8
-> integration and actionable failure-injection gates
-> 106-case complete CTest quality gate
-> repository-resident six-class / 30-image consistency manifest with frozen image hashes
-> Python CPUExecutionProvider reference using the same config/artifact semantics
-> deterministic class-first / maximum-IoU matching independent of output order
-> machine-readable per-image and summary consistency evidence
-> 30/30 images and 62/62 detections pass the predeclared numerical gates
-> correctness gate confirmed again before performance publication
-> Release-only batch-1 CPU benchmark, warmup 10 / repeat 100
-> image decode / preprocess / Session::Run / postprocess timing boundaries
-> pipeline and end-to-end P50/P95 plus throughput
-> Windows process-lifetime Peak Working Set and machine-readable benchmark JSON
-> fresh S1-09 reproduction: 106/106 CTest, Demo, consistency, benchmark, and faults
-> user L2 explanation/modification gate still pending
-> no batch directory processing, concurrency, service, or INT8
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
| `ConsistencyManifest` / `compare_consistency.py` | Freeze six classes x five validation images plus image hashes, run a Python `CPUExecutionProvider` reference under the same Runtime/artifact contract, invoke the C++ CLI, match detections by class and deterministic maximum IoU rather than array order, enforce predeclared thresholds, and emit per-image/summary JSON. | S1-07 L1 accepted: 30/30 images and 62/62 matched detections passed |
| `BenchmarkRunner` / `BenchmarkResult` / `BenchmarkWriter` | Keep one validated CPU session alive across warmup/repeat; time `imread`, `cv::Mat -> tensor`, only `Ort::Session::Run`, postprocess, pipeline, and end-to-end with `steady_clock`; calculate the arithmetic mean plus empirical nearest-rank P50/P95 and throughput; capture Release/provider/thread/model/sample/environment/Peak-Working-Set evidence; then safely write JSON outside the repeated timing. | S1-08 L1 accepted; the S1-09 fresh reproduction passed the same fixed 10/100 protocol without changing benchmark behavior |
| `ArtifactRegistry` / `ModelCard` | Record artifact source, model family, dataset, metrics, config, postprocess type, runtime status, and paths. | YOLO baseline declaration established; D010 remains gated |
| `Tests` | Use synthetic data at pure seams for Runtime/artifact schema, `cv::Mat` preprocess, model metadata, postprocess, JSON, output paths, deterministic consistency matching, and latency statistics; use a short real-model benchmark plus the strict Python validator for the complete benchmark-result schema. Test targets link the Runtime boundary rather than `main.cpp`; CLI faults assert nonzero exit and actionable diagnostics. | S1-09 fresh Release gate: 106/106 in 19.91 seconds; direct missing-model/damaged-image/unwritable-parent/repeat-zero failures returned exit 1 with actionable diagnostics |

### 5. Quick Start

S1-09 uses a new out-of-tree Release build and disposable evidence paths. Initialize the x64 compiler environment in CMD, then start a profile-free PowerShell in the same window:

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
powershell.exe -NoProfile -NoExit
```

Run the following from that PowerShell. The order is deliberate: build, complete CTest, Demo, consistency, benchmark, and direct fault evidence. Every native command is checked immediately so an old JSON file cannot turn a failed run into a false pass.

```powershell
$Repo = 'D:\01_Base\CodingSpace\yolo_defect'
$ToolsRoot = 'D:\01_Base\Tools'
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'
$OrtRoot = Join-Path $ToolsRoot 'onnxruntime-win-x64-1.19.2'
$OpenCvDir = Join-Path $ToolsRoot 'opencv\build\x64\vc16\lib'
$OpenCvBin = Join-Path $ToolsRoot 'opencv\build\x64\vc16\bin'
$CMakeBin = Join-Path $ToolsRoot `
  'VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'

Set-Location $Repo
$env:ONNXRUNTIME_ROOT = $OrtRoot
$env:PATH = $CMakeBin + ';' + $OpenCvBin + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_09_' + [guid]::NewGuid().ToString('N'))
$EvidenceDir = Join-Path $BuildDir 's1_09_evidence'

function Assert-NativeSuccess([string]$Step) {
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed with native exit code $LASTEXITCODE."
  }
}

# 1. Fresh Release configure and build.
cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$OpenCvDir" `
  -DONNXRUNTIME_ROOT="$OrtRoot" `
  -DPython3_EXECUTABLE="$PythonExe" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
Assert-NativeSuccess 'CMake configure'

cmake --build $BuildDir
Assert-NativeSuccess 'Release build'

# 2. Complete automated gate before any manual evidence command.
ctest --test-dir $BuildDir -N
Assert-NativeSuccess 'CTest enumeration'
ctest --test-dir $BuildDir --output-on-failure
Assert-NativeSuccess 'Complete CTest'

# 3. Fixed end-to-end Demo; outputs remain outside the repository.
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$DemoDir = Join-Path $EvidenceDir 'demo'
$OutputJson = Join-Path $DemoDir 'crazing_241.json'
$OutputImage = Join-Path $DemoDir 'crazing_241.png'
$Cli = "$BuildDir\bin\yolo_defect_cpp.exe"
$DetectionValidator = (Resolve-Path `
  'cpp_infer\tests\assert_detection_json.py').Path

& $Cli --config $Config --image $Image `
  --output-json $OutputJson --output-image $OutputImage
Assert-NativeSuccess 'Fixed Demo'
& $PythonExe -m json.tool $OutputJson *> $null
Assert-NativeSuccess 'Demo JSON parse'
& $PythonExe $DetectionValidator $OutputJson --expected-image $Image
Assert-NativeSuccess 'Demo JSON contract validation'
& "$BuildDir\bin\yolo_defect_image_probe.exe" $OutputImage
Assert-NativeSuccess 'Visualization OpenCV probe'
Get-Item $OutputJson, $OutputImage
Get-FileHash $OutputJson, $OutputImage -Algorithm SHA256

# 4. Six classes x five images, using the frozen S1-07 requirements.
$Manifest = (Resolve-Path `
  'cpp_infer\tests\fixtures\consistency_manifest.json').Path
$ConsistencyDir = Join-Path $EvidenceDir 'consistency'
& $PythonExe cpp_infer\tools\compare_consistency.py `
  --manifest $Manifest --cpp-cli $Cli `
  --output-dir $ConsistencyDir --cpp-opencv-version 4.8.0
Assert-NativeSuccess '30-image consistency comparison'
& $PythonExe -m json.tool "$ConsistencyDir\per_image.json" *> $null
Assert-NativeSuccess 'Per-image consistency JSON parse'
& $PythonExe -m json.tool "$ConsistencyDir\summary.json" *> $null
Assert-NativeSuccess 'Consistency summary JSON parse'

# 5. Correctness is green, so run the fixed 10/100 protocol.
$BenchmarkJson = Join-Path $EvidenceDir 'benchmark.json'
& $Cli --config $Config --image $Image `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json $BenchmarkJson
Assert-NativeSuccess 'Formal benchmark'
& $PythonExe -m json.tool $BenchmarkJson *> $null
Assert-NativeSuccess 'Benchmark JSON parse'
& $PythonExe cpp_infer\tests\assert_benchmark_json.py $BenchmarkJson `
  --expected-image $Image --expected-warmup 10 --expected-repeat 100
Assert-NativeSuccess 'Strict benchmark validator'
Get-Item $BenchmarkJson
Get-FileHash $BenchmarkJson -Algorithm SHA256

# 6. Four direct CLI faults must fail with actionable diagnostics.
function Assert-ActionableCliFailure(
    [string]$Name,
    [string[]]$Arguments,
    [string[]]$RequiredText) {
  $Text = & $Cli @Arguments 2>&1 | Out-String
  $ExitCode = $LASTEXITCODE
  if ($ExitCode -ne 1) {
    throw "$Name expected exit 1, actual $ExitCode."
  }
  foreach ($Token in (@('expected', 'actual') + $RequiredText)) {
    if ($Text -notmatch [regex]::Escape($Token)) {
      throw "$Name omitted actionable token '$Token'. Output: $Text"
    }
  }
  if ($Text -notmatch 'action[:=]') {
    throw "$Name omitted an action. Output: $Text"
  }
  Write-Host "$Name`: exit 1 with a failing path/object and expected/actual/action"
}

$MissingModelConfig = (Resolve-Path `
  'cpp_infer\tests\fixtures\runtime\missing_model_artifact.txt').Path
$DamagedImage = Join-Path $BuildDir `
  'test_inputs\s1_06_faults\damaged_image.jpg'
$BlockedParent = Join-Path $BuildDir `
  'test_inputs\s1_06_faults\blocked_output_parent'

Assert-ActionableCliFailure -Name 'missing model' `
  -Arguments @('--config', $MissingModelConfig, '--inspect-model') `
  -RequiredText @('model artifact does not exist')
Assert-ActionableCliFailure -Name 'damaged image' `
  -Arguments @('--config', $Config, '--image', $DamagedImage) `
  -RequiredText @('OpenCV decoding returned an empty image')
Assert-ActionableCliFailure -Name 'unwritable parent' `
  -Arguments @('--config', $Config, '--image', $Image, '--output-json',
    (Join-Path $BlockedParent 'detections.json')) `
  -RequiredText @('output.json_path.parent')
Assert-ActionableCliFailure -Name 'repeat zero' `
  -Arguments @('--config', $Config, '--image', $Image, '--benchmark',
    '--warmup', '1', '--repeat', '0', '--benchmark-json',
    (Join-Path $EvidenceDir 'invalid-repeat.json')) `
  -RequiredText @('object=--repeat')

ctest --test-dir $BuildDir `
  -R '^(postprocess\.PostprocessEmptyTest\.ValidTensorWithNoScoreAboveThresholdIsEmpty|output\.ResultWriterJsonTest\.EmptyDetectionsSerializeAsAnEmptyArray)$' `
  --output-on-failure
Assert-NativeSuccess 'Legal empty-detections tests'
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

The fresh S1-09 temporary-build reproduction produced the same three detections and the same 1,164-byte JSON/39,306-byte PNG hashes. Python parsed the JSON, and the independent OpenCV probe read the PNG as `200x200 CV_8UC3`. This rerun validates reproducibility without promoting temporary build output as a Git-committed artifact.

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
ctest --test-dir $BuildDir -L unit --output-on-failure
ctest --test-dir $BuildDir -L negative --output-on-failure
ctest --test-dir $BuildDir -L integration --output-on-failure
ctest --test-dir $BuildDir -L quality_gate --output-on-failure
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir -L postprocess --output-on-failure
ctest --test-dir $BuildDir -L preprocess --output-on-failure
ctest --test-dir $BuildDir -L consistency --output-on-failure
ctest --test-dir $BuildDir -L benchmark --output-on-failure
ctest --test-dir $BuildDir --output-on-failure
```

Expected current result:

```text
fresh Release configure/build: passed
ctest --show-only / -N:        106 tests listed
complete clean Release gate:   106/106 passed in 19.91 s
fixed Demo:                    3 detections; JSON parsed; PNG probed
30-image consistency:          30/30 images; 62/62 matches; JSON parsed
formal benchmark:              warmup 10/repeat 100; strict JSON passed
direct CLI faults:             4/4 exit 1 with actionable diagnostics
legal empty detections:        2/2 focused tests passed
```

The existing schema, metadata, preprocess, postprocess, output, consistency, and benchmark tests continue to prove their pure and integration boundaries. The short real-model benchmark smoke uses low repeat and checks behavior rather than enforcing a speed threshold. S1-09 separately reruns the formal 10/100 protocol into a new temporary result file, so a prior repository-resident JSON cannot mask a failed command.

#### S1-06 Failure Triage

- **Schema:** start with declaration file, line/field, expected, actual, and action. `artifact_spec_path` is relative to the Runtime config; `model_path` is relative to the artifact declaration, not the process working directory.
- **Missing model:** inspect the artifact `model_path` and its normalized path in the error.
- **Metadata mismatch:** run `--inspect-model`, then compare actual name/shape/dtype/provider with the declaration. Synthetic mismatches test the pure validator and do not require several malformed ONNX models.
- **Damaged image:** distinguish a missing path from an existing file whose OpenCV decode returned an empty image, then retry with a known-good image.
- **CLI/output:** run `--help`; check missing/duplicate/conflicting flags, parent path type and permission, protected inputs, and the explicit overwrite policy.

Fresh S1-09 direct fault evidence produced exit code 1 for a missing model, generated damaged-image bytes, a regular file used as an output parent, and `--repeat 0`. All four diagnostics included object, expected, actual, and action. Two focused tests also reconfirmed that a valid tensor with no score strictly above the threshold and a valid result with no detections both succeed with `detections: []`; an invalid output rank/channel/count or non-finite value remains an error rather than an empty result.

#### S1-07 Consistency Triage

- **Count/class mismatch:** first compare the per-image Python/C++ detections, then verify the same config/artifact, explicit CPU provider, strict `confidence > threshold`, class-agnostic NMS, and class names.
- **Raw numerical drift:** isolate preprocess first, then raw output, threshold/NMS tie behavior, and finally coordinate restoration. Do not relax a gate merely to turn the run green.
- **Order differences:** JSON array order is not correctness evidence. Matching groups by `class_id`, chooses the maximum-IoU remaining pair, and uses a canonical value tie-break.
- **Evidence files:** `per_image.json` contains image-level matches and failures; `summary.json` contains identifiers, providers, thresholds, aggregate errors, pass counts, and failure details.

#### S1-08 Benchmark Triage

- **Correctness first:** rerun the S1-07 consistency label before interpreting or publishing performance. A failed correctness gate blocks benchmark claims.
- **Unexpected latency:** verify a clean `Release` build, `CPUExecutionProvider`, sequential execution, intra/inter-op `1/1`, graph optimization `all`, warmup/repeat, and the staged ORT DLL before considering optimization.
- **Boundary interpretation:** `image_decode` times only `cv::imread`; `preprocess` starts from an already decoded `cv::Mat`; `session_run` times only `Ort::Session::Run`; pipeline includes safe input/tensor/output wrapper work; end-to-end adds decode.
- **Evidence validation:** parse with `python -m json.tool`, then run `cpp_infer/tests/assert_benchmark_json.py` with the expected image, warmup, and repeat. Existing output is rejected unless `--overwrite` is explicit.

#### Current Single-Image, Consistency, and Benchmark Limits

- JSON records the artifact's **declared** SHA-256; S1-05 does not recompute the model hash at runtime.
- `actual_provider` means the explicitly registered provider of the successfully created and executed session. It is session-level evidence, not per-node placement from ORT profiling.
- JSON and image data are both prepared before writing, but the two files are not committed as one transaction and cross-process filesystem replacement is not guaranteed atomic.
- The fixed ASCII Windows input path is verified. Arbitrary Unicode input image paths through the existing narrow-string OpenCV `imread` boundary are not yet claimed as supported; output path/JSON UTF-8 handling is separate.
- The current six baseline labels are ASCII. OpenCV's Hershey text renderer is deterministic for them but is not a general Unicode font engine.
- S1-07 proves implementation consistency for the same ONNX artifact, not detector accuracy against ground truth.
- The matching `best.pt` checkpoint is unavailable, so this is not a newly rerun PyTorch/Python-ORT/C++ three-way experiment. The historical 50-image PT/ONNX count evidence remains separate.
- Python explicitly requests `CPUExecutionProvider`, while C++ records its explicitly registered session provider. Neither is per-node placement evidence from ORT profiling.
- All 30 frozen consistency images are `200x200`, so this cross-language run exercises resize to `800x800` without padding. Non-square/odd-padding behavior is covered synthetically in C++ tests but not yet cross-language on this manifest.
- The 30-image run produced no empty-detection image, so legal `detections: []` remains unit/output-schema evidence rather than this consistency sample's integration evidence.
- The benchmark is one `200x200` image, batch 1, on one Windows CPU host. It is a reproducible baseline, not a dataset-wide or cross-machine performance conclusion.
- Repeated `imread` observes a warmed operating-system file cache rather than cold-disk latency. No CPU affinity, elevated process priority, or idle-system lock was applied, so concurrent load can move the distribution.
- Session/model initialization, config/artifact loading, initial path/file-size checks, statistics, the memory query, benchmark JSON writing, and visualization are excluded from repeated latency. JSON and drawing are not executed inside the benchmark loop.
- Peak Working Set is a process-lifetime peak that includes config/session initialization, warmup, timed iterations, retained samples, statistics, and harness state. It is neither per-stage nor incremental inference memory.
- `actual_provider=CPUExecutionProvider` is session-level creation/execution evidence, not ORT per-node placement profiling.
- Historical Python ORT 24.4/72.1 FPS used a different implementation, sample protocol, and hardware/provider context; it must not be ranked unconditionally against this C++ single-image result.
- The Runtime remains batch-1/single-image only. S1-09 adds no directory batch mode, concurrency, service, `inference_event`, or INT8; its automatic gate passes, but Large Stage One remains open until the user L2 gate passes.

### 8. Key Data and Artifact Results

| Item | Current Record |
|------|----------------|
| P0 dataset | NEU-DET steel surface defects, 1,800 images, 6 classes, 200x200 pixels |
| P0 model | YOLOv8n baseline and tuned variants |
| Best current YOLO result | `final_train_2`, mAP@0.5 = 0.743, mAP@50-95 = 0.388 |
| Baseline ONNX artifact preflight | Tracked `models/best.onnx`, 12,336,935 bytes, opset 17, SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`; metadata says `nms=False` |
| Model lineage status | The project owner confirms the current ONNX was personally exported from `runs/detect/final_train_2/weights/best.pt`; that `.pt` is absent from the workspace and Git history, so the lineage is owner-confirmed but not currently re-exportable |
| Baseline ONNX I/O preflight | Python ORT 1.19.2 confirms input `images` = float32 `[1,3,800,800]`; output `output0` = float32 `[1,10,13125]` |
| Current C++ Runtime state | S1-09 fresh Release/NMake reproduction with MSVC 19.50.35721.0, C++17, OpenCV 4.8.0, and ORT C++ 1.19.2 passed 106/106 CTests in 19.91 seconds, the fixed Demo, 30-image consistency, 10/100 benchmark, four direct faults, and two legal-empty tests. Product semantics are unchanged; user L2 remains pending |
| C++ ORT actual metadata | Loaded `models/best.onnx`; available EP inventory `[AzureExecutionProvider,CPUExecutionProvider]`; explicitly registered session EP `CPUExecutionProvider`; input `images` tensor float32 `[1,3,800,800]`; output `output0` tensor float32 `[1,10,13125]`; contract passed |
| S1-02 dependency/session boundary | CMake consumes the official external ORT C++ SDK 1.19.2 only through `ONNXRUNTIME_ROOT`, validates the version/C/C++/CPU-provider headers/import library/DLL and stages the matching DLL. Session policy is sequential, intra-op 1, inter-op 1 (unused by sequential mode), graph optimization all |
| S1-03 raw-output evidence | Fixed `crazing_241.jpg`: input float32 `[1,3,800,800]`, 1,920,000 finite values, range `[0.278431386,1]`; owned output float32 `[1,10,13125]`, 131,250 finite values, range `[0,795.04126]`. These values prove finite raw execution, not decoded detection correctness or benchmark performance |
| S1-04 postprocess evidence | ORT-free synthetic tests verify `[1,4+C,N]` BCN decode, maximum class score without objectness/sigmoid, float32-domain strict `confidence > threshold`, `xywh -> xyxy`, robust IoU, stable class-agnostic NMS before coordinate restore, and letterbox inverse transform/clip. Postprocess GTest 24/24, preprocess GTest 7/7, complete CTest 62/62 |
| S1-05 single-image output evidence | Fixed `crazing_241.jpg` produces 3 `crazing` detections, a Python-parseable 1,164-byte JSON (`E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049`) and an OpenCV-readable 39,306-byte PNG (`3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346`). Six output GTests, 16 output-labeled CTests, and all 78 tests pass |
| S1-06 quality-gate evidence | Fresh `%TEMP%` Release build lists 90 tests: unit 51, integration 3, negative 32, contract 19, metadata 16, preprocess 9, postprocess 25, and output 18; all 90 pass. Missing model, damaged image, and uncreatable output parent each return CLI exit 1 with object/path, expected, actual, and action diagnostics; no extra large ONNX fixture is used |
| Artifact license checkpoint | The artifact declaration preserves the ONNX metadata text `AGPL-3.0 License (https://ultralytics.com/license)`. Source remains MIT; because the owner chose to keep publicly distributing the ONNX and NEU-DET, model obligations and the dataset's unspecified redistribution terms remain separate release checkpoints |
| Incoming research artifact | `paper_detect` D010 method on the D-FINE-S/DeepPCB research line; not a new Runtime architecture claim |
| External D010 research evidence | Formal-validation AP50-95 = 0.847057; official-test AP50-95 = 0.830385; these are not Project 1 Runtime results |
| D010 relationship and ablation | D003 is the ancestor/ablation anchor; all 6 D010 class deltas over D003 are positive on formal and official test; D010A erase-only and D010B replay-only each beat D003 but trail full D010 |
| D010 integration gate | Stable ONNX + result/model card + deployment contract + real Runtime adapter + consistency validation; it must not block the YOLO P0 closure |

#### Correctness evidence lanes

| Evidence lane | Implementations and sample | Result | Valid claim | Limitation and path |
|---|---|---|---|---|
| Historical PT/ONNX count-only check | Historical PyTorch and ONNX paths; first 50 sorted validation images, all `crazing` | 50/50 images have the same detection count; 146 vs 146 total detections | Weak historical count alignment only | No six-class coverage, class pairing, box tolerance, or current `.pt` rerun; [`compare_50_summary.json`](results/pt_onnx_compare/compare_50_summary.json) |
| Current strict Python ORT/C++ ORT evidence | Same frozen ONNX/config/artifact; six classes x five images; Python and C++ explicit CPU sessions; class-first maximum-IoU matching | S1-09 fresh reproduction: 30/30 images and 62/62 matches; max confidence error `8.049977111568296e-07`, max bbox error `9.135351561440075e-05` px, min IoU `0.999998927116394`; both JSON files parsed | Implementation consistency under the predeclared exact-count/class, `1e-4`, `1e-2` px, and `0.999` gates | Not mAP, not every image/platform, and not a new PyTorch three-way run; repository-resident [`per_image.json`](cpp_infer/results/consistency/per_image.json) and [`summary.json`](cpp_infer/results/consistency/summary.json) plus fresh temporary outputs |

#### Performance evidence lanes

| Evidence lane | Implementations and protocol | Result | Valid claim | Limitation and path |
|---|---|---|---|---|
| Historical V1 Python benchmarks | PyTorch/Python ORT; 5 warmup plus 100 timed images under their historical CPU/GPU protocols | PyTorch CPU `8.43 FPS`, PyTorch GPU `110.8 FPS`; Python ORT CPU `24.4 FPS`, Python ORT GPU `72.1 FPS` | Historical Python-side context only | Different implementation, sample/hardware/provider, and timing boundaries; see `results/*benchmark*.json` |
| Current C++ Release benchmark | Fixed `crazing_241.jpg`, batch 1, CPU `CPUExecutionProvider`, sequential intra/inter-op `1/1`, warmup 10/repeat 100 | S1-09 fresh reproduction: pipeline `7.078853 img/s`, end-to-end `7.038151 img/s`, end-to-end mean/P50/P95 `142.082777/145.3222/150.7653 ms`, Peak Working Set `152.578125 MiB` | Reproducible single-image warm-cache C++ baseline on this host and protocol | Not full-dataset, cold-disk, GPU, concurrent, or cross-machine performance; temporary JSON was 5,453 bytes, SHA-256 `F32C0DF3157897264F9BD2B9AE3F3DB7B240A3B641494E8D3E7C346FF64E9C6F` |

Pending artifact paths:

```text
artifacts/paper_detect_d010/result_card.md        # placeholder
artifacts/paper_detect_d010/model_artifact.yaml   # placeholder
artifacts/paper_detect_d010/metrics_table.csv     # placeholder
artifacts/paper_detect_d010/qualitative/          # placeholder
```

Historical S1-08 published latency record, retained unchanged:

| Segment | Boundary | Mean (ms) | P50 (ms) | P95 (ms) |
|---------|----------|----------:|---------:|---------:|
| Image decode | `cv::imread` only | 0.991129 | 0.9649 | 1.3517 |
| Preprocess | decoded `cv::Mat -> float32 NCHW tensor` | 8.244569 | 7.5514 | 12.1265 |
| Session run | only synchronous `Ort::Session::Run` | 165.555859 | 164.8985 | 186.2136 |
| Postprocess | owned raw output -> filtered/NMS/restored detections | 0.424115 | 0.4251 | 0.5636 |
| Pipeline | preprocess + safe inference wrapper + postprocess | 175.560944 | 175.1058 | 195.1376 |
| End-to-end | image decode + pipeline | 176.553060 | 176.1357 | 196.6128 |

Fresh S1-09 reproduction under the same declared protocol:

The rerun used the same `DESKTOP-6OGK71C` Windows 10.0.26200 host, MSVC 19.50.35721.0 Release C++17 build, OpenCV 4.8.0, ORT 1.19.2, requested CPU/actual `CPUExecutionProvider`, sequential intra/inter-op `1/1`, graph optimization `all`, model/input/sample, thresholds, batch 1, and three-detection result as S1-08. No runtime setting was tuned between the two records.

| Segment | Boundary | Mean (ms) | P50 (ms) | P95 (ms) |
|---------|----------|----------:|---------:|---------:|
| Image decode | `cv::imread` only | 0.816168 | 0.8182 | 0.9251 |
| Preprocess | decoded `cv::Mat -> float32 NCHW tensor` | 5.453755 | 5.4547 | 6.2128 |
| Session run | only synchronous `Ort::Session::Run` | 134.419309 | 137.5882 | 142.5549 |
| Postprocess | owned raw output -> filtered/NMS/restored detections | 0.345302 | 0.3438 | 0.4424 |
| Pipeline | preprocess + safe inference wrapper + postprocess | 141.265814 | 144.4673 | 149.8395 |
| End-to-end | image decode + pipeline | 142.082777 | 145.3222 | 150.7653 |

The S1-09 automatic result is a second same-protocol reproduction, not a claim that the implementation was optimized between S1-08 and S1-09. Uncontrolled host load and warmed caches can move latency. The model-license checkpoint remains a provenance/distribution risk rather than a C++ implementation blocker to hide. Large Stage One remains open until the user L2 gate passes.

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
- **Order-independent consistency:** detections are grouped by class and paired by deterministic maximum IoU with a canonical tie-break, so correctness does not depend on Python and C++ serializing detections in the same order.
- **Correctness before performance:** S1-07 must pass before S1-08 numbers are published. The benchmark cannot turn a wrong result into a successful performance claim.
- **Explicit benchmark boundaries:** `imread`, decoded-Mat preprocess, only `Session::Run`, postprocess, pipeline, and end-to-end are timed separately with one `steady_clock` protocol. Session initialization and file outputs stay outside repeated timing and are disclosed rather than silently omitted.
- **Distribution over one-shot latency:** warmup 10/repeat 100 plus mean/P50/P95 exposes normal and tail behavior. Pipeline/end-to-end throughput is derived from matching mean latency; Peak Working Set is reported separately as a process-lifetime memory baseline.
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

Current state: historical Project 1 tasks P1-00 through P1-03 and large-stage-one tasks **S1-01 through S1-08** are implemented, verified, and L1 accepted. The **S1-09 automatic gate passes**, but its user-owned L2 gate is pending. Large Stage One is not complete and Large Stage Two has not started.

The 2026-07-16 pre-stage readiness pass is also complete: the ORT C++ 1.19.2 SDK is present and verified, the VS x64 toolchain is discoverable, a new `%TEMP%` Release/NMake build passes 3/3 CTest, and the future GTest dependency is pinned. The owner also confirmed the current ONNX was personally exported from the `final_train_2` best checkpoint; the checkpoint is not in this workspace or Git history. The durable commands, evidence, GTest hash, model-lineage audit, and unresolved public-distribution license checkpoints are in [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md). This preparation did not start S1-01 or change Runtime behavior.

The next action is not another product feature: the user must complete the S1-09 L2 explanation, follow-up, debugging, and temporary behavior-plus-GTest exercise below. Only then may Large Stage One be marked complete and Large Stage Two be planned. `S1-*` means “large stage one small stage” and avoids confusing the old Project 1 `P1-*` history with the top-level P1 extension category.

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
| S1-06 | Expanded the Runtime/artifact, preprocess, metadata, postprocess, output, integration, and failure matrix into one labeled quality gate. | Separate proof of pure algorithm correctness, real vertical-slice operability, and failure diagnosability before consistency work. | Exact four-pixel NCHW, landscape/portrait odd padding, synthetic metadata, full empty-result path, schema faults, damaged image, and uncreatable output parent; clean Release CTest 90/90 in 5.53 seconds. | Bad metadata needs synthetic structs, not several large models. A legal empty detection list differs from malformed/empty raw output. Diagnostics should always lead from object/path to expected, actual, and action. |
| S1-07 | Added a repository-resident frozen six-class manifest plus an independent Python ORT CPU reference and deterministic Python/C++ detector matcher. | Replace weak count-only evidence with reproducible class, confidence, coordinate, and IoU evidence under frozen gates. | 30/30 images and 62/62 matches passed; max confidence error `8.049977111568296e-07`, max coordinate error `9.135351561440075e-05` px, minimum IoU `0.999998927116394`; consistency CTest 2/2 and complete CTest 92/92. | Python 3.9 `Path.write_text()` has no `newline` argument, so evidence writing uses `open(..., newline='\n')`. This was a serialization-compatibility fix; no correctness tolerance was relaxed. |
| S1-08 | Added a Release-only `BenchmarkRunner`, timed ORT boundary, benchmark result/writer, CLI mode, strict JSON validator, and Windows Peak Working Set capture. | Establish performance evidence only after correctness, while separating decode, preprocess, model execution, postprocess, pipeline, and user-visible end-to-end costs. | On the fixed 10/100 protocol, pipeline mean/P50/P95 is `175.560944/175.1058/195.1376` ms and end-to-end is `176.553060/176.1357/196.6128` ms; throughput is `5.696028/5.664020` images/s and Peak Working Set is `152.714844` MiB. Benchmark 14/14 and complete CTest 106/106 pass. | Time only `Session::Run` for the infer segment, but include safe tensor construction/output-copy overhead in pipeline. Warmed file cache, uncontrolled host load, process-lifetime memory, and session-level provider evidence must be disclosed; unlike protocols must not be ranked directly. |
| S1-09 automatic gate | Added no product behavior; rebuilt and reran the full deliverable chain from a fresh temporary Release directory. | Prove that the stage-one output is reproducible and that documentation/evidence can withstand engineering review before the user L2 gate. | Configure/build passed; CTest 106/106 in 19.91 seconds; deterministic Demo hashes, 30/30 consistency, fresh 10/100 benchmark, four actionable exit-1 faults, and two legal-empty tests passed. | An automated green gate cannot prove the user can explain or modify the system. S1-09 and Large Stage One remain open until the exercise below is completed and reverted cleanly. |
| PLAN-20260715 | Aligned repository rules and the bilingual entry points to the latest top-level design; created the long-form large-stage-one plan. | Preserve the verified baseline while preventing the short stage summary from dropping contract, correctness, test, failure, and evidence requirements. | `docs/PLAN.md` -> `AGENTS.md` rules -> README stage/status summary -> `docs/STAGE1_EXECUTION_PLAN.md` one-step plan. | Historical Python metrics, external D010 metrics, and future C++ results must stay explicitly separated. |

### 13. S1-09 L2 Interview Gate

Automatic status is **PASS**. User L2 status is **PENDING**. The following material is the acceptance kit rather than evidence that the user has already completed it.

#### Two-minute explanation outline

1. Position the project as a C++17 industrial-vision Runtime that consumes an existing YOLOv8/NEU-DET ONNX artifact rather than another training wrapper.
2. Explain the contract boundary: `RuntimeConfig` holds runtime policy, `ModelArtifactSpec` holds declared model facts, and actual ORT `ModelMetadata` must match before inference.
3. Walk the single-image chain: OpenCV decode -> letterbox/RGB/normalize/NCHW -> RAII ORT CPU session -> owned raw output -> BCN decode/strict score filter/stable class-agnostic NMS/coordinate restore -> JSON/PNG.
4. Close with evidence: synthetic GTest/CTest and faults, six-class 30-image Python ORT/C++ ORT consistency, and a correctness-gated Release benchmark. State the limits: no model-accuracy claim, INT8, batch/concurrency, or real-device result.

#### Five-minute explanation outline

1. **0:00-0:30, positioning:** define the Python-prototype-to-deployable-Runtime gap and why model artifacts need executable contracts.
2. **0:30-1:10, structure:** explain the static Runtime library, thin CLI, Runtime/artifact/actual-metadata separation, declaration-relative paths, and actionable errors.
3. **1:10-1:50, preprocessing:** explain `image path -> cv::Mat -> letterbox -> RGB -> float32/255 -> NCHW`, including scale/padding retained for inverse coordinates.
4. **1:50-2:35, ORT and lifetime:** explain PImpl/RAII, explicit CPU EP, borrowed input vector lifetime through synchronous `Run`, and output copying before `Ort::Value` dies.
5. **2:35-3:20, postprocess/output:** explain `[1,4+C,N]`, no objectness/sigmoid, class argmax, strict `>`, stable class-agnostic NMS, restore/clip, stable JSON, and headless visualization.
6. **3:20-4:00, tests/faults:** separate synthetic unit evidence from the few real-model integration smokes and show how missing model, damaged image, bad metadata/tensor, bad CLI, and bad output paths fail.
7. **4:00-4:35, correctness/performance:** explain class-first maximum-IoU matching on six classes x five images, then the six benchmark timing boundaries and Peak Working Set scope.
8. **4:35-5:00, limits/next stage:** distinguish implementation consistency from mAP and session provider from per-node profiling; put INT8 PTQ and broader evidence hardening in Large Stage Two.

#### Follow-up questions and concise answers

| # | Question | Interview answer |
|---:|---|---|
| 1 | Why separate `RuntimeConfig`, `ModelArtifactSpec`, and `ModelMetadata`? | Runtime policy is adjustable, artifact facts are declared and model-specific, and metadata is what ORT actually reads. Cross-checking them catches the wrong model before `Run`. |
| 2 | Why resolve relative paths from the declaring file? | The process CWD changes with the launcher; declaration-relative resolution selects the same artifact/model from the repo, build directory, or another CWD. |
| 3 | Why use a Runtime library plus a thin CLI? | Algorithms and resource management can be reused and tested without compiling `main.cpp`; the CLI only parses arguments and orchestrates. |
| 4 | What does RAII protect in `OnnxRunner`? | `Env`, `SessionOptions`, `Session`, allocators, and names follow object lifetime and unwind safely on exceptions instead of relying on manual release. |
| 5 | Why may the input `Ort::Value` borrow the preprocess vector but the returned output may not expose an ORT pointer? | The vector remains stable until synchronous `Run` returns; local output `Ort::Value` storage would die on return, so values are copied into `InferenceOutput`. |
| 6 | What do 10 and 13,125 mean in `[1,10,13125]`? | Ten is `4 + 6` box/class channels; 13,125 is the candidate count, not the final detection count. |
| 7 | Why is there no objectness multiplication or extra sigmoid? | The frozen export/contract directly exposes class scores in the baseline semantics; adding either would change threshold behavior and break consistency. |
| 8 | Why do strict `>` and stable NMS matter? | Exact-threshold candidates must be rejected by contract, and deterministic equal-score ordering keeps repeat results, JSON, and tests reproducible. |
| 9 | Why not compare Python and C++ detections by array index? | Equivalent detections may be serialized in another order, so matching first groups by class and then chooses deterministic maximum-IoU pairs. |
| 10 | Why must correctness pass before benchmark publication? | Latency only says how fast code ran; a wrong detection result cannot become valid because it is fast. |
| 11 | How do `Session::Run`, pipeline, and end-to-end timing differ? | `Session::Run` isolates ORT; pipeline also includes preprocess, tensor/wrapper checks, output copy, and postprocess; end-to-end additionally includes image decode. |
| 12 | What does `actual_provider=CPUExecutionProvider` prove? | It proves explicit CPU EP session setup and successful runs, but not independent per-node placement without ORT profiling. |
| 13 | Why is Peak Working Set not “model inference memory”? | It is the process-lifetime high-water mark, including session initialization, warmup, samples, statistics, and harness state. |
| 14 | Why discuss MIT source, ONNX AGPL metadata, and NEU-DET redistribution separately? | They concern different assets and obligations; one license cannot be assigned to the others without source evidence. |

#### Error-triage cases

| Failure | Diagnosis order | Correct response |
|---|---|---|
| Config says `provider expected [cpu], actual cuda` | Read file/line/field first; this is schema loading, not evidence that inference used a GPU | Restore `cpu`; future CUDA support would require schema, SDK/CMake, session registration, actual-provider checks, and tests together |
| Inspect reports declared `[1,3,800,800]`, actual `[1,3,640,640]` | Verify normalized model path and SHA, then actual name/shape/dtype | Use the matching artifact/model or legitimately re-export and repeat consistency; do not edit the declaration merely to silence the check |
| Image exists but OpenCV returns an empty `cv::Mat` | Separate path existence from decoder/codec/content validity | Probe or re-encode the image and retain the non-empty `CV_8UC3` check; do not bypass preprocessing validation |
| Output parent is a regular file | Inspect each parent component and the target type/permissions | Choose or create a directory; keep the nonzero output-path failure separate from inference errors |
| Consistency fails and benchmark is blocked | Inspect per-image count/class/unmatched boxes before numeric errors; then trace preprocess -> raw output -> threshold/NMS -> restore -> matcher | Fix the cause under the frozen protocol; do not widen tolerances or publish performance first |

#### Resume bullets

- Built a C++17/CMake/OpenCV/ONNX Runtime single-image industrial-defect Runtime with strict Runtime/artifact schemas, RAII session management, letterbox/NCHW preprocessing, YOLOv8 decode/stable NMS/coordinate restoration, deterministic JSON/PNG output, and a 106-case GTest/CTest gate with actionable fault injection.
- Established correctness and performance evidence for the same frozen ONNX: matched 62/62 detections across a six-class 30-image Python ORT/C++ ORT set with max confidence error `8.05e-7` and max box error `9.14e-5 px`; reproduced a fixed Release CPU 10/100 baseline at `7.038151 img/s` end to end and `152.578125 MiB` process Peak Working Set.

The second bullet must retain the same-ONNX and fixed single-image CPU-protocol qualifiers; it is not an mAP, general FPS, or device-deployment claim.

#### Core code worth handwriting

- Letterbox/RGB/normalize/NCHW: `cpp_infer/src/image_preprocessor.cpp:57` and `:108`; tests at `cpp_infer/tests/preprocessor_mat_test.cpp:47`, `:82`, and `:108`.
- ORT RAII, borrowed input, synchronous run, and owned output: `cpp_infer/src/onnx_runner.cpp:338`, `:402`, and `:419`.
- YOLO box conversion, IoU, BCN decode, stable NMS, and restore/clip: `cpp_infer/src/postprocessor.cpp:269`, `:280`, `:311`, `:360`, `:405`, and `:450`; threshold test at `cpp_infer/tests/postprocessor_test.cpp:173`.
- Actual metadata versus contract: `cpp_infer/src/model_metadata.cpp:149`.
- Python reference preprocessing/postprocess and deterministic matcher: `cpp_infer/tools/compare_consistency.py:619`, `:801`, `:922`, and `:1435`.
- Mean/nearest-rank percentiles and six timing boundaries: `cpp_infer/src/benchmark_result.cpp:66`, `cpp_infer/src/benchmark_runner.cpp:327`, `:371`, and `:520`; tests at `cpp_infer/tests/benchmark_test.cpp:31`.
- Strict parser/path semantics, pipeline orchestration, and stable writers: `cpp_infer/src/key_value_parser.cpp:76`/`:281`, `cpp_infer/src/detector_pipeline.cpp:135`, and `cpp_infer/src/result_writer.cpp:764`/`:856`.

#### User-owned behavior + GTest exercise

Use the score-filter boundary as a disposable RED -> GREEN -> restore exercise. The product contract remains strict `confidence > threshold`; the inclusive behavior must never be merged.

1. First create a clean S1-09 checkpoint. Because the current stage worktree may contain valuable uncommitted work, do not use a broad `git restore` or start the exercise in that dirty tree. Use a disposable practice branch only after the checkpoint exists.
2. **RED:** change the synthetic expectation in `cpp_infer/tests/postprocessor_test.cpp:173` so a score exactly equal to `0.25` is expected to survive. Leave product code unchanged, rebuild the test target, and run:

```powershell
cmake --build $BuildDir --target yolo_defect_postprocess_tests
if ($LASTEXITCODE -ne 0) { throw 'RED test rebuild failed.' }
& "$BuildDir\bin\yolo_defect_postprocess_tests.exe" `
  --gtest_filter='YoloDecodeTest.*Threshold*'
```

The test must fail, demonstrating the current strict `>` contract.

3. **GREEN:** temporarily change `cpp_infer/src/postprocessor.cpp:344` to inclusive `>=` semantics and update the exact-threshold expectations around `postprocessor_test.cpp:192` and `:456`. Rebuild the same target, rerun the focused test, and require exit 0. Explain why detection counts may change and why a real contract change would also require Python reference, consistency evidence, and README updates.
4. **RESTORE:** abandon the disposable branch or manually reverse only the exercise lines back to strict `>`. Rebuild, rerun the original focused test and complete CTest, and inspect `git diff` to prove no inclusive behavior remains on the product branch.

L2 passes only after the user can deliver both outlines, answer at least ten follow-ups, explain at least three failures, identify the core files/evidence, and complete RED -> GREEN -> restore without leaving product changes. Until then the authoritative status remains **S1-09 automatic PASS / user L2 PENDING / Large Stage One NOT complete**.

## Highlights

- **Best Experimental Result** — Best checkpoint `final_train_2` reaches **mAP@0.5 = 0.743** on NEU-DET
- **Historical PyTorch vs ONNX Count Check** — **50/50** count matches and **146 vs 146** detections, but the sorted sample is all `crazing` and does not prove class/box tolerance
- **Current Strict Consistency Evidence** — Six classes x five images, **30/30 images** and **62/62 matched detections** pass the frozen Python ORT/C++ ORT count/class/confidence/box/IoU gates
- **Historical V1 Python Benchmarks** — PyTorch CPU **8.43 FPS**; PyTorch GPU (RTX 3060) **110.8 FPS**; Python ORT CPU **24.4 FPS**; Python ORT GPU **72.1 FPS** — all measured on 100 timed images (5 warmup), not C++ results
- **Current C++ Release Benchmark Reproduction** — fixed batch-1 CPU sample, warmup 10/repeat 100: pipeline **7.078853 images/s**, end-to-end **7.038151 images/s**, end-to-end P50/P95 **145.3222/150.7653 ms**, Peak Working Set **152.578125 MiB**; the older S1-08 run remains separately documented and no optimization claim is made
- **Docker Verified** — `python:3.9-slim` image has been tested with `/health` and `/detect`
- **Dataset Included** — The 28MB NEU-DET copy needs no separate dataset download; the V2 C++ path still requires the documented OpenCV, ORT C++ SDK, compiler, CMake, and test dependencies

## Key Metrics

### Model metrics

| Metric | Value |
|---|---|
| Best model | `final_train_2` |
| mAP@0.5 | **0.743** |
| mAP@50-95 | **0.388** |

### Correctness metrics — evidence kept separate

| Evidence lane | Value |
|---|---|
| Historical PT/ONNX count-only | **50/50** all-`crazing` images, 146/146 detections, mean absolute count difference **0.000**; no class/box tolerance claim |
| Current Python ORT/C++ ORT strict evidence | **30/30 images, 62/62 matches**; max confidence error `8.049977111568296e-07`, max bbox error `9.135351561440075e-05 px`, minimum IoU `0.999998927116394` |

### Performance metrics — evidence kept separate

Historical V1 Python-side protocol:

| Metric | Value |
|---|---|
| Historical PyTorch CPU benchmark | **8.43 FPS** / **118.66 ms** per image |
| Historical PyTorch GPU benchmark (RTX 3060) | **110.8 FPS** / **9.0 ms** per image |
| Historical Python ORT CPU benchmark | **24.4 FPS** / **40.9 ms** per image |
| Historical Python ORT GPU benchmark (RTX 3060) | **72.1 FPS** / **13.9 ms** per image |

Current S1-09 C++ Release fresh-reproduction protocol:

| Metric | Value |
|---|---|
| C++ ORT CPU pipeline | **7.078853 images/s** / **141.265814 ms mean**, fixed one-image 10/100 protocol |
| C++ ORT CPU end-to-end | **7.038151 images/s** / **142.082777 ms mean**, P50/P95 **145.3222/150.7653 ms** |
| C++ process Peak Working Set | **152.578125 MiB**, process-lifetime scope |
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

All latency rows in this historical V1 subsection remain Python/Python-ORT evidence. The current C++ Runtime S1-08 table is documented separately in the V2 section above because its single-image, batch-1, Release CPU protocol is different and must not be compared unconditionally.

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
│   ├── include/yolo_defect_cpp/  # Public contract/preprocess/runner/postprocess/output/benchmark APIs
│   │   ├── detector_pipeline.h   # PImpl single-image Runtime orchestration boundary
│   │   ├── detection_result.h    # Self-contained result and image metadata
│   │   ├── result_writer.h       # Stable JSON/visualization output request API
│   │   ├── benchmark_runner.h    # Release-only warmup/repeat measurement boundary
│   │   ├── benchmark_result.h    # Typed environment/latency/throughput/memory evidence
│   │   └── benchmark_writer.h    # Strict safe machine-readable benchmark JSON output
│   ├── src/                      # Parser, preprocess, ORT, postprocess, pipeline, benchmark, writers, thin CLI
│   ├── results/demo/             # Verified S1-05 JSON and visualized PNG evidence
│   ├── results/consistency/      # S1-07 machine-readable per-image and summary evidence
│   ├── results/benchmark/        # S1-08 machine-readable Release performance evidence
│   ├── tools/compare_consistency.py # Python CPU ORT reference and order-independent comparator
│   └── tests/                    # GTest/CTest, manifests, assert_benchmark_json.py and fault gates
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
- **`cpp_infer/`** — V2 C++ deployment workspace. It now owns the Runtime library/CLI boundary, strict Runtime/artifact contract, OpenCV preprocessing, ORT RAII session, actual metadata validation, safe tensor ownership, pure YOLOv8 postprocess, the single-image Pipeline, stable JSON/visualization outputs, the S1-06 failure gate, repository-resident S1-07 Python/C++ consistency evidence, and repository-resident S1-08 segmented Release benchmark/Peak-Working-Set evidence. The S1-09 automatic reproduction passes; user L2 remains pending.
- **`configs/`** — Separated hyperparameters. Easy to track experiments by diffing config files.

## Tech Stack

| Tool | Purpose | Version |
|------|---------|---------|
| Python | V1 utilities plus S1-07 independent consistency reference | 3.9.25; consistency environment also uses NumPy 2.0.2 |
| C++ | V2 runtime language | C++17 |
| MSVC | Verified x64 C++ compiler | 19.50.35721.0 |
| PyTorch | Deep learning framework | 2.0.0 |
| Ultralytics | YOLOv8 training & inference | 8.4.24 locally and in artifact metadata |
| ONNX | Model interchange format | Python package 1.19.1; artifact opset 17 |
| ONNX Runtime | Python consistency reference plus C++ RAII session, metadata validation, raw inference, and single-image pipeline execution | Python 1.19.2 explicitly requests `CPUExecutionProvider`; official Windows x64 CPU C++ SDK 1.19.2 explicitly registers the CPU session provider. This is session-level, not per-node profiling evidence |
| OpenCV | Python/C++ consistency preprocessing plus verified file/`CV_8UC3 cv::Mat` C++ preprocessing, deterministic drawing, encoding, and read-back | Python 4.13.0; Windows C++ 4.8.0 x64 vc16; the explicit version difference is recorded in S1-07 evidence |
| CMake | Active C++ build system and CTest entry | 4.1.1-msvc1 |
| GTest | Synthetic postprocess/preprocess/output plus benchmark-statistics/result tests linked to `yolo_defect_runtime` | Official GitHub v1.17.0 archive; commit `52eb8108c5bdec04579160ae17225d66034bd723` and SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84` pinned; benchmark label 14/14 and current complete CTest 106/106 pass; offline source override requires a separately hash-verified extraction |
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
- [x] Historical 50-image PyTorch/ONNX detection-count check (all `crazing`; count/confidence summary only)
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
| S1-05 | **Verified; L1 accepted** | End-to-end CLI, JSON, and visualization | `DetectorPipeline` orchestrates the single-image vertical slice; `DetectionResult` snapshots owned output metadata; `ResultWriter` emits stable JSON v1 and deterministic headless visualization with explicit file safety rules | Fixed command creates 3 `crazing` detections, Python-parseable JSON and an OpenCV-readable PNG; empty detections remain valid `[]`; output GTest 6/6, output label 16/16, complete CTest 78/78 |
| S1-06 | **Verified; L1 accepted** | Automated and failure-path gate | Labeled GTest/CTest coverage across strict Runtime/artifact schemas, exact preprocess layout, synthetic metadata, postprocess, outputs, integration, and core faults | Clean Release 90/90 in 5.53 seconds; unit 51, integration 3, negative 32. Missing model, damaged image, and uncreatable output parent return nonzero actionable errors |
| S1-07 | **Verified; L1 accepted** | Fixed-sample Python ORT/C++ consistency | Compare a frozen repository-resident six-class x five-image manifest under the same artifact/config, explicit CPU providers, strict threshold, class-agnostic NMS, and coordinate semantics; pair detections by class and deterministic maximum IoU | 30/30 images and 62/62 matches pass the frozen gates; per-image/summary JSON preserve metrics and diagnostics; consistency 2/2 and complete CTest 92/92 pass; no unsupported direct PT rerun claim |
| S1-08 | **Verified; L1 accepted** | Reproducible Release benchmark | Measure `imread`, decoded-Mat preprocess, only `Session::Run`, postprocess, pipeline, and end-to-end under a fixed CPU/thread/model/sample protocol; record throughput, environment and memory evidence | Formal warmup 10/repeat 100 JSON contains mean/P50/P95 for all six boundaries, pipeline/end-to-end throughput, full protocol metadata, and Windows Peak Working Set; benchmark 14/14 and complete CTest 106/106 pass |
| S1-09 | **Automatic gate PASS; user L2 PENDING** | Large-stage-one closure | Fresh-build all gates, align documentation/evidence, and complete user-owned L2 interview acceptance without adding product behavior | Automated Demo/tests/consistency/benchmark/fault/empty gates pass; the user must still deliver the explanations and complete/revert the behavior-plus-GTest exercise |

Large Stage Two will be decomposed only after the S1-09 user L2 gate passes. Its fixed boundary is broader P0 regression/fault/sample/performance-memory evidence hardening plus an FP32-versus-INT8 PTQ correctness, accuracy, speed, and model-size comparison; QAT starts only if PTQ degradation is material and time permits. Final P0 result consolidation, focused mock interviews, and delivery freeze also remain there. Large Stage Three then chooses condition-gated P1 extensions. TensorRT is not unconditional, and a Project 2 `inference_event` remains optional.

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

### S1-07 Python ORT/C++ ORT Consistency Commands

Use the same clean Release executable built in Quick Start. The Python interpreter is intentionally explicit: using an arbitrary active Conda/base Python could silently change ORT, OpenCV, NumPy, or provider availability. `--cpp-opencv-version` records the independently verified C++ build dependency rather than pretending it equals Python OpenCV.

```powershell
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'
$Manifest = (Resolve-Path 'cpp_infer\tests\fixtures\consistency_manifest.json').Path
$ConsistencyDir = (Resolve-Path 'cpp_infer\results\consistency').Path

& $PythonExe cpp_infer\tools\compare_consistency.py `
  --manifest $Manifest `
  --cpp-cli "$BuildDir\bin\yolo_defect_cpp.exe" `
  --output-dir $ConsistencyDir `
  --cpp-opencv-version 4.8.0
if ($LASTEXITCODE -ne 0) {
  throw 'S1-07 consistency comparison failed.'
}

& $PythonExe -m json.tool "$ConsistencyDir\per_image.json" *> $null
if ($LASTEXITCODE -ne 0) { throw 'per_image.json parse failed.' }
& $PythonExe -m json.tool "$ConsistencyDir\summary.json"
if ($LASTEXITCODE -ne 0) { throw 'summary.json parse failed.' }
ctest --test-dir $BuildDir -L consistency --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Consistency CTest failed.' }
ctest --test-dir $BuildDir --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Complete CTest failed.' }
```

Local verification on 2026-08-22: the frozen repository-resident manifest fixed validation indices 241, 255, 270, 285, and 300 for every artifact class, five images per class and 30 total. Every declared image path and SHA-256 passed before inference. Python explicitly created an ORT 1.19.2 session with `CPUExecutionProvider`; C++ used ORT 1.19.2 with its explicitly registered CPU session provider. Under the same contract's `800x800` input, strict `confidence > threshold`, class-agnostic NMS, and coordinate semantics, order-independent class/maximum-IoU matching paired 62/62 detections across 30/30 images. Maximum confidence error was `8.049977111568296e-07`, maximum bbox coordinate error was `9.135351561440075e-05` pixels, and minimum matching IoU was `0.999998927116394`, passing the unchanged predeclared `1e-4`, `1e-2` pixel, and `0.999` gates. Repository-resident `per_image.json` and `summary.json` preserve the machine-readable evidence; S1-09 generated and parsed new temporary copies. The historical consistency label passed 2/2 in 12.58 seconds and the then-complete gate passed 92/92 in 17.28 seconds.

The first evidence-write attempt exposed a Python 3.9 compatibility issue: `Path.write_text()` does not accept a `newline` argument. The tool now writes with `open(..., newline='\n')` to keep deterministic LF JSON. This changed serialization only; none of the frozen correctness thresholds were relaxed. Because the matching `best.pt` is unavailable, this is a Python ORT/C++ ORT comparison of one ONNX artifact, not a newly rerun three-way PyTorch experiment and not an accuracy evaluation.

### S1-08 Release Benchmark Commands

Use the same clean Release executable only after the S1-07 consistency gate passes. The repository-resident evidence path may already exist in the working tree; the historical command below uses explicit overwrite for an intentional refresh. The S1-09 Quick Start above is safer for acceptance because it writes to a new temporary path.

```powershell
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$BenchmarkJson = (Resolve-Path 'cpp_infer\results\benchmark').Path + `
  '\yolov8_neu_det_cpu_release.json'

ctest --test-dir $BuildDir -L consistency --output-on-failure
if ($LASTEXITCODE -ne 0) {
  throw 'S1-07 consistency failed; benchmark publication is forbidden.'
}
& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config --image $Image `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json $BenchmarkJson --overwrite
if ($LASTEXITCODE -ne 0) { throw 'Formal benchmark failed.' }

& $PythonExe -m json.tool $BenchmarkJson
if ($LASTEXITCODE -ne 0) { throw 'Benchmark JSON parse failed.' }
& $PythonExe cpp_infer\tests\assert_benchmark_json.py $BenchmarkJson `
  --expected-image $Image --expected-warmup 10 --expected-repeat 100
if ($LASTEXITCODE -ne 0) { throw 'Benchmark validator failed.' }
ctest --test-dir $BuildDir -L benchmark --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Benchmark CTest failed.' }
ctest --test-dir $BuildDir -N
if ($LASTEXITCODE -ne 0) { throw 'CTest enumeration failed.' }
ctest --test-dir $BuildDir --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'Complete CTest failed.' }
```

The six latency intervals are `imread`, already-decoded `cv::Mat -> tensor`, only synchronous `Ort::Session::Run`, raw-output postprocess, pipeline, and decode-plus-pipeline end-to-end. Session/model initialization, contract loading, initial path/file-size checks, statistics, memory query, JSON serialization/write, and visualization are not repeated latency. The formal run used `crazing_241.jpg`, batch/sample 1, CPU sequential `1/1` threads, warmup 10/repeat 100, produced three detections, passed the strict JSON validator, and was followed by benchmark 14/14 plus complete CTest 106/106.

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
| 2026-08-22 | Completed S1-06 automated quality gate and core failure injection | Expanded schema, exact NCHW, synthetic metadata, full postprocess-empty, output-path and CLI fault evidence; all 90 named quality-gate tests passed in a clean Release build, and missing model/damaged image/uncreatable output returned actionable nonzero failures; stopped before S1-07 consistency |
| 2026-08-22 | Completed S1-07 fixed six-class Python ORT/C++ ORT consistency evidence | Created a repository-resident SHA-frozen 30-image manifest, independent explicit-CPU Python reference, deterministic class/maximum-IoU matcher, per-image/summary JSON, unchanged numerical gates, 30/30 image and 62/62 detection matches, and a 92-case complete CTest gate; stopped before S1-08 benchmark work |
| 2026-08-22 | Completed S1-08 reproducible Release benchmark and memory evidence | Reconfirmed S1-07 correctness before performance; added six explicit timing boundaries, warmup/repeat statistics, throughput, full Release/CPU/thread/model/sample metadata, strict JSON validation, and Windows Peak Working Set; generated the repository-resident 10/100 result, passed benchmark 14/14 and complete CTest 106/106, and stopped before S1-09 closure |
| 2026-08-22 | Passed the S1-09 automatic large-stage closure gate; user L2 remains pending | Added no product behavior. A fresh temporary Release build passed 106/106 CTests in 19.91 seconds, reproduced byte-identical Demo outputs, passed 30/30 consistency, generated/validated a fresh 10/100 benchmark, and reconfirmed four actionable exit-1 faults plus legal empty detections. The temporary inclusive-threshold practice has not been applied to product code; Large Stage One remains open until the user completes and restores that exercise |

## License

Repository-authored source code is licensed under the MIT License — see [LICENSE](LICENSE). This statement does not automatically cover `models/best.onnx` or the NEU-DET dataset.

The tracked ONNX metadata declares `AGPL-3.0`. [Ultralytics' official licensing guidance](https://www.ultralytics.com/license) states that Ultralytics-trained models use AGPL-3.0 by default unless an applicable commercial license is obtained. The [official NEU dataset page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265) provides downloads and citation guidance, but this audit did not find an explicit redistribution license there. These are provenance/distribution checkpoints rather than a legal conclusion; see [`docs/PRE_STAGE1_READINESS.md`](docs/PRE_STAGE1_READINESS.md).

The declared use is personal learning, so an Enterprise license is not a prerequisite for local development. The owner selected option A—continue public distribution of the ONNX and NEU-DET—so noncommercial intent does not remove the need to preserve the model's license notice and verify the dataset's redistribution basis before the release position is frozen.

The NEU-DET dataset is provided by Northeastern University (NEU). Please cite the original paper if you use this dataset in academic work:

> K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013.
