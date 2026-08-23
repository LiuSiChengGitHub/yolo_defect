# cpp_infer

This directory is the V2 C++ deployment workspace for the steel-surface defect project.

Current status: **S1-08 L1 accepted; S1-09 automatic PASS; user L2 PENDING; Large Stage One not complete; Stage Two not started.** The fixed single-image CLI connects the validated Runtime/artifact contract, OpenCV preprocessing, an ONNX Runtime CPU session, owned raw output, YOLOv8 postprocess, stable detection JSON, and a GUI-free visualization. S1-07 proves Python ORT/C++ ORT correctness on a fixed six-class, 30-image manifest, S1-08 records the Release-only benchmark baseline, and the fresh S1-09 reproduction proves that the complete chain still builds and runs without adding product behavior. Automatic success is not a substitute for the user-owned L2 explanation, troubleshooting, and modification exercise.

| Gate | Status |
|---|---|
| S1-08 L1 | Accepted |
| S1-09 clean reproduction / automatic gate | PASS |
| User Large-Stage-One L2 | PENDING |
| Large Stage One | Not complete until user L2 passes |
| Stage Two | Not started |

## Current single-image chain

```text
RuntimeConfig + ModelArtifactSpec
-> DetectorPipeline
-> canonical single-image path
-> OpenCV decode + letterbox + BGR/RGB + normalize + NCHW
-> OnnxRunner CPU Session::Run
-> owned InferenceOutput [1,10,13125]
-> validate + BCN decode + strict score filter
-> stable class-agnostic model-space NMS
-> inverse letterbox + source-bound clip
-> SingleImageDetectionResult
-> stable JSON v1 + deterministic OpenCV visualization
```

`main.cpp` owns only CLI parsing and orchestration. Contract loading, preprocessing, ORT lifetime, postprocess, result validation, JSON serialization, output-path policy, and visualization remain in `yolo_defect_runtime`.

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

## Layout and responsibilities

```text
cpp_infer/
├── CMakeLists.txt
├── artifacts/yolov8_neu_det.artifact.txt
├── configs/default_config.txt
├── include/yolo_defect_cpp/
│   ├── artifact_spec.h
│   ├── benchmark_result.h
│   ├── benchmark_runner.h
│   ├── benchmark_writer.h
│   ├── config_loader.h
│   ├── detection_result.h
│   ├── detector_pipeline.h
│   ├── image_preprocessor.h
│   ├── model_metadata.h
│   ├── onnx_runner.h
│   ├── postprocessor.h
│   └── result_writer.h
├── src/
│   ├── artifact_spec.cpp
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
│   ├── onnx_runner.cpp
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
├── tools/
│   └── compare_consistency.py
└── tests/
    ├── result_writer_test.cpp
    ├── image_file_probe.cpp
    ├── postprocessor_test.cpp
    ├── preprocessor_mat_test.cpp
    ├── inference_runner_test.cpp
    ├── model_metadata_validator_test.cpp
    ├── test_consistency.py
    ├── benchmark_test.cpp
    ├── assert_benchmark.cmake
    ├── assert_benchmark_json.py
    ├── assert_detection_json.py
    ├── assert_single_image_outputs.cmake
    ├── assert_cli_arguments_failure.cmake
    ├── assert_*.cmake
    └── fixtures/
        └── consistency_manifest.json
```

- `RuntimeConfig` owns runtime policy: score/NMS thresholds, configured provider, and the artifact declaration path.
- `ModelArtifactSpec` owns model identity, declared SHA-256, provenance/license, tensor I/O, classes, and preprocess/postprocess/NMS semantics.
- `ImagePreprocessor` exposes file-path and `const cv::Mat&` entries. Both use the same letterbox/RGB/normalize/NCHW implementation.
- `OnnxRunner` owns ORT resources through RAII/PImpl. `run()` borrows the preprocess vector only for synchronous `Session::Run`, then copies output into ORT-independent storage.
- `Postprocessor` performs validated BCN decode, class argmax, strict score filtering, `xywh -> xyxy`, IoU, stable class-agnostic NMS, inverse letterbox, and clipping.
- `SingleImageDetectionResult` is a self-contained result object holding model, image, Runtime, provider, and detection evidence needed by output writers.
- `DetectorPipeline` connects the established modules for exactly one image and protects config, artifact, model, and source-image paths from output overwrite.
- `ResultWriter` validates the result, escapes JSON strings, uses locale-independent stable numeric formatting, creates parent directories, enforces overwrite policy, and renders deterministic labels/colors without a GUI.
- The internal `image_decoder` seam measures only `cv::imread` while preserving the same normalized `CV_8UC3` image contract used by normal preprocessing.
- `BenchmarkResult` defines the Release benchmark protocol, six latency segments, environment/runtime/model/sample metadata, throughput, memory evidence, timing exclusions, and limitations. Its pure statistics helper uses empirical nearest-rank ceiling for P50/P95 and derives batch-1 throughput from mean latency.
- `BenchmarkRunner` creates and validates the CPU session before repeated timing, performs untimed warmup, rejects result drift between iterations, measures decode/preprocess/`Session::Run`/postprocess/pipeline/end-to-end with `std::chrono::steady_clock`, and queries process memory after the timed iterations.
- `BenchmarkWriter` serializes finite schema-v1 JSON with the classic locale, creates output parents, refuses overwrite by default, and protects config, artifact, model, and source-image paths. JSON serialization and filesystem writing happen after the measured loop.
- `consistency_manifest.json` freezes six artifact classes, five validation images per class, declaration-relative paths, image SHA-256 values, and the acceptance requirements before comparison runs.
- `compare_consistency.py` strictly reloads the same Runtime/artifact declarations, verifies file hashes and model metadata, runs Python ORT with only `CPUExecutionProvider`, invokes the existing C++ CLI, performs order-independent matching, and atomically replaces each machine-readable evidence file.
- `test_consistency.py` exercises manifest validation, frozen-threshold rejection, strict score/NMS semantics, class mismatch handling, and order-independent maximum-IoU matching without running the 30-image experiment.
- `benchmark_test.cpp` fixes the mean and nearest-rank percentile definitions, single-sample behavior, invalid empty/negative/NaN/Inf rejection, and throughput calculation independently of machine speed.
- `assert_benchmark.cmake` runs a short `warmup=1/repeat=2` Release smoke, invokes the strict Python validator, then proves a second run refuses overwrite without changing the first file.
- `assert_benchmark_json.py` rejects duplicate/non-finite JSON, validates the complete Release/CPU/thread/batch/sample/model/input/statistics/memory/disclosure schema, recomputes the referenced model and image SHA-256 values, and checks both throughput values equal `1000 / mean_ms`.
- GTest targets link `yolo_defect::runtime`; they never compile or reuse `main.cpp`.

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

## Dependency boundary

| Dependency | Verified S1-09 reproduction input |
|---|---|
| Compiler | MSVC 19.50.35721.0, x64 |
| CMake / CTest | 4.1.1-msvc1, Visual Studio bundled runtime |
| C++ OpenCV | 4.8.0 x64 vc16 at `D:\01_Base\Tools\opencv` |
| ONNX Runtime C++ SDK | Official Windows x64 CPU SDK 1.19.2 at `D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| Python comparison environment | `C:\Users\Everbreath\.conda\envs\TestBase\python.exe`: Python 3.9.25, ONNX Runtime CPU 1.19.2, OpenCV 4.13.0, NumPy 2.0.2 |
| GoogleTest | Official v1.17.0 commit `52eb8108c5bdec04579160ae17225d66034bd723`; archive SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84` |

CMake fetches GoogleTest only with `BUILD_TESTING=ON`, verifies the pinned archive hash, disables installation/GMock, and keeps the parent MSVC runtime choice. A fully offline configure can point at a separately verified extracted source:

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='<path-to-verified-googletest-source>'
```

The source override bypasses archive download/hash checking during that configure, so verify the pinned archive before extraction. No personal GTest path is committed. ORT is located only through `ONNXRUNTIME_ROOT`, and its matching DLL is staged beside ORT-using executables. With `BUILD_TESTING=ON`, configure also checks that the selected Python can import the pinned ORT version plus OpenCV/NumPy and exposes `CPUExecutionProvider`; a different or incomplete Python environment fails with an actionable dependency error.

## S1-09 clean Release reproduction and acceptance

From CMD, initialize the x64 MSVC environment and start a profile-free PowerShell:

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -NoExit
```

Use the verified TestBase interpreter rather than the Windows Store `python.exe` alias. The GoogleTest source override must point to a separately hash-verified extraction of the pinned archive; it avoids an implicit network fetch. The build and every generated S1-09 artifact use one unique disposable directory, so these reproduction commands do not overwrite repository-local demo, consistency, or benchmark evidence.

PowerShell does not automatically stop after every native executable returns nonzero. The helper below must be called immediately after each `cmake`, `ctest`, CLI, or Python command:

```powershell
$Repo = 'D:\01_Base\CodingSpace\yolo_defect'
$OrtRoot = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$OpenCvDir = 'D:\01_Base\Tools\opencv\build\x64\vc16\lib'
$OpenCvBin = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin'
$CMakeBin = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
$PythonExe = 'C:\Users\Everbreath\.conda\envs\TestBase\python.exe'
$GTestSource = '<path-to-verified-googletest-source>'
$ErrorActionPreference = 'Stop'

function Assert-NativeSuccess {
  param([string]$Step)
  if ($global:LASTEXITCODE -ne 0) {
    throw "$Step failed with exit code $global:LASTEXITCODE."
  }
}

Set-Location $Repo
$env:ONNXRUNTIME_ROOT = $OrtRoot
$env:PATH = $CMakeBin + ';' + $OpenCvBin + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_09_' + [guid]::NewGuid().ToString('N'))
$EvidenceDir = Join-Path $BuildDir 's1_09_evidence'

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$OpenCvDir" `
  -DONNXRUNTIME_ROOT="$OrtRoot" `
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$GTestSource" `
  -DPython3_EXECUTABLE="$PythonExe" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
Assert-NativeSuccess 'S1-09 configure'

cmake --build $BuildDir
Assert-NativeSuccess 'S1-09 Release build'

ctest --test-dir $BuildDir -N
Assert-NativeSuccess 'CTest inventory'

ctest --test-dir $BuildDir --output-on-failure
Assert-NativeSuccess 'complete CTest'
```

Do not continue after a failed complete CTest. Once it is green, run the fixed Demo and validate both output formats:

```powershell
$Cli = Join-Path $BuildDir 'bin\yolo_defect_cpp.exe'
$ImageProbe = Join-Path $BuildDir 'bin\yolo_defect_image_probe.exe'
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$DemoJson = Join-Path $EvidenceDir 'demo\crazing_241.json'
$DemoImage = Join-Path $EvidenceDir 'demo\crazing_241.png'
$DetectionValidator = (Resolve-Path `
  'cpp_infer\tests\assert_detection_json.py').Path

& $Cli --config $Config --image $Image `
  --output-json $DemoJson --output-image $DemoImage
Assert-NativeSuccess 'fixed single-image Demo'

& $PythonExe -m json.tool $DemoJson > $null
Assert-NativeSuccess 'Demo JSON parse'

& $PythonExe $DetectionValidator $DemoJson --expected-image $Image
Assert-NativeSuccess 'Demo JSON strict validation'

$Demo = Get-Content $DemoJson -Raw | ConvertFrom-Json
if ($Demo.detections.Count -ne 3) {
  throw "Fixed Demo expected 3 detections, actual $($Demo.detections.Count)."
}

& $ImageProbe $DemoImage
Assert-NativeSuccess 'Demo visualization OpenCV probe'
```

Next, run the six-class comparison into the same disposable evidence root. The comparison returns zero only when all 30 images pass the frozen requirements; it deliberately has no CLI option for weakening tolerances:

```powershell
$Manifest = (Resolve-Path `
  'cpp_infer\tests\fixtures\consistency_manifest.json').Path
$ConsistencyTool = (Resolve-Path `
  'cpp_infer\tools\compare_consistency.py').Path
$ConsistencyDir = Join-Path $EvidenceDir 'consistency'
$PerImageJson = Join-Path $ConsistencyDir 'per_image.json'
$SummaryJson = Join-Path $ConsistencyDir 'summary.json'

& $PythonExe $ConsistencyTool `
  --manifest $Manifest `
  --cpp-cli $Cli `
  --output-dir $ConsistencyDir `
  --cpp-opencv-version 4.8.0
Assert-NativeSuccess '30-image Python ORT/C++ ORT consistency'

& $PythonExe -m json.tool $PerImageJson > $null
Assert-NativeSuccess 'consistency per_image.json parse'

& $PythonExe -m json.tool $SummaryJson > $null
Assert-NativeSuccess 'consistency summary.json parse'

$PerImage = Get-Content $PerImageJson -Raw | ConvertFrom-Json
$Summary = Get-Content $SummaryJson -Raw | ConvertFrom-Json
if (-not $Summary.passed -or
    $Summary.result.images_total -ne 30 -or
    $Summary.result.images_passed -ne 30 -or
    $Summary.result.python_detections_total -ne 62 -or
    $Summary.result.cpp_detections_total -ne 62 -or
    $Summary.result.matched_detections_total -ne 62 -or
    $Summary.result.max_confidence_abs_error -gt 1.0e-4 -or
    $Summary.result.max_bbox_coordinate_abs_error_pixels -gt 1.0e-2 -or
    $Summary.result.min_matching_iou -lt 0.999 -or
    $PerImage.images.Count -ne 30 -or
    @($PerImage.images | Where-Object { -not $_.passed }).Count -ne 0 -or
    $Summary.source_class_results.Count -ne 6 -or
    @($Summary.source_class_results | Where-Object {
      $_.images_total -ne 5 -or $_.images_passed -ne 5
    }).Count -ne 0) {
  throw 'Consistency evidence breached the frozen 30-image/six-class gate.'
}
```

Only after that explicit correctness gate passes may the fixed Release benchmark run. Its JSON is parsed and then checked by the strict schema/protocol/hash/statistics validator:

```powershell
$BenchmarkJson = Join-Path $EvidenceDir `
  'benchmark\yolov8_neu_det_cpu_release.json'
$BenchmarkValidator = (Resolve-Path `
  'cpp_infer\tests\assert_benchmark_json.py').Path

& $Cli --config $Config --image $Image `
  --benchmark --warmup 10 --repeat 100 `
  --benchmark-json $BenchmarkJson
Assert-NativeSuccess 'formal warmup=10 repeat=100 benchmark'

& $PythonExe -m json.tool $BenchmarkJson > $null
Assert-NativeSuccess 'benchmark JSON parse'

& $PythonExe $BenchmarkValidator $BenchmarkJson `
  --expected-image $Image `
  --expected-warmup 10 `
  --expected-repeat 100
Assert-NativeSuccess 'benchmark JSON strict validation'
```

Finally, directly inject four stable faults and require both a nonzero exit and actionable text. The damaged image and blocked output parent are generated by configure under the fresh build tree; using a regular file as the output parent is more reproducible than relying on machine-specific filesystem permissions.

```powershell
function Assert-CliFailure {
  param(
    [string]$Name,
    [string[]]$Arguments,
    [string[]]$RequiredText
  )
  $Lines = & $Cli @Arguments 2>&1
  $ExitCode = $LASTEXITCODE
  $Text = $Lines -join [Environment]::NewLine
  if ($ExitCode -eq 0) { throw "$Name unexpectedly exited 0." }
  foreach ($Needle in $RequiredText) {
    if (-not $Text.Contains($Needle)) {
      throw "$Name did not report required text '$Needle'."
    }
  }
  Write-Host "[$Name] exit=$ExitCode"
  Write-Host $Text
}

$MissingModelConfig = (Resolve-Path `
  'cpp_infer\tests\fixtures\runtime\missing_model_artifact.txt').Path
$DamagedImage = Join-Path $BuildDir `
  'test_inputs\s1_06_faults\damaged_image.jpg'
$BlockedParent = Join-Path $BuildDir `
  'test_inputs\s1_06_faults\blocked_output_parent'

Assert-CliFailure -Name 'missing model' `
  -Arguments @('--config', $MissingModelConfig, '--inspect-model') `
  -RequiredText @('model artifact does not exist',
    'expected', 'actual', 'action:')

Assert-CliFailure -Name 'damaged image' `
  -Arguments @('--config', $Config, '--image', $DamagedImage) `
  -RequiredText @('OpenCV decoding returned an empty image',
    'expected', 'actual', 'action:')

Assert-CliFailure -Name 'unwritable output parent' `
  -Arguments @('--config', $Config, '--image', $Image, '--output-json',
    (Join-Path $BlockedParent 'detections.json')) `
  -RequiredText @('output.json_path.parent',
    'expected', 'actual', 'action:')

Assert-CliFailure -Name 'invalid benchmark repeat' `
  -Arguments @('--config', $Config, '--image', $Image, '--benchmark',
    '--repeat', '0', '--benchmark-json',
    (Join-Path $EvidenceDir 'invalid.json')) `
  -RequiredText @('object=--repeat',
    'expected=', 'actual=', 'action=')

ctest --test-dir $BuildDir `
  -R '^(postprocess\.PostprocessEmptyTest\.ValidTensorWithNoScoreAboveThresholdIsEmpty|output\.ResultWriterJsonTest\.EmptyDetectionsSerializeAsAnEmptyArray)$' `
  --output-on-failure
Assert-NativeSuccess 'legal empty-detections checks'
```

`--benchmark`, `--warmup`, `--repeat`, and `--benchmark-json` form a separate CLI mode. Detection JSON, visualization, model inspection, and raw-output summary flags cannot be mixed into it. Warmup/repeat must be bounded integers; repeat must be positive. Relative CLI image/output paths use the current working directory, while config/artifact paths use their declaration files. Output writers protect config, artifact, model, and source-image inputs and refuse overwrite by default.

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

- Python dependency/provider failure: confirm CMake receives `C:\Users\Everbreath\.conda\envs\TestBase\python.exe`, imports ORT/OpenCV/NumPy, and lists `CPUExecutionProvider`. The comparison refuses a Python Session whose provider list is not exactly CPU.
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

### S1-09 technical teaching log and pending user L2 gate

The automatic gate proves reproducibility; it does not prove that the user can explain or modify the system. Large Stage One remains open until the following user-owned L2 gate is completed.

Two-minute explanation outline:

1. Position the project as a C++ industrial-vision Runtime, not merely a model-training repository.
2. Walk through contract loading, OpenCV preprocess, ORT session/run, YOLO decode/filter/NMS/restore, then JSON/PNG.
3. Explain that synthetic GTest covers pure boundaries while the real model is reserved for a few integration smokes.
4. Separate S1-07 correctness evidence from S1-08/S1-09 performance evidence.
5. End with current limits and the Stage Two INT8/evidence-hardening boundary.

Five-minute explanation outline:

1. Explain `RuntimeConfig`, `ModelArtifactSpec`, relative paths, strict schema errors, and actual metadata validation.
2. Explain letterbox, BGR/RGB, normalize, NCHW, `Ort::Value` borrowing, synchronous `Session::Run`, and owned raw output.
3. Explain BCN indexing, no objectness/sigmoid, strict `confidence > threshold`, stable class-agnostic NMS, inverse letterbox, and clipping.
4. Explain `DetectorPipeline`, stable JSON, deterministic headless visualization, and output safety.
5. Explain the 30-image class-first/maximum-IoU comparison and frozen numeric gates.
6. Explain the six benchmark boundaries, warmup/repeat, mean/P50/P95, throughput, Peak Working Set, and timing exclusions.
7. Close with three failure paths, licensing limitations, absent `best.pt`, and the work explicitly deferred to Stage Two.

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

Interview-ready resume bullets, to finalize only after L2 acceptance:

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

The root bilingual READMEs contain the full interview script and acceptance worksheet. This technical README records the same gate state: **automatic PASS, user L2 PENDING, Large Stage One not complete**.

## Current limits

- `actual_provider` is session-level evidence from explicit CPU EP registration plus successful session creation. It is not per-node profiling evidence.
- `model.declared_sha256` is copied from the validated artifact declaration. The ordinary S1-05 detection CLI does not recompute the model hash on every invocation; the S1-07 comparison and strict S1-08 evidence validator independently bind their runs to the actual frozen model SHA-256.
- JSON and visualization are validated before writing, but they are two files rather than one filesystem transaction. A disk-level failure while writing the second file can leave the first file present.
- The C++17 path check and later file open are not a cross-process atomic transaction. This S1-05 CLI is verified for one local invocation, not for competing writers racing on the same destination; an explicit overwrite can also leave a partial output after a disk-level write failure.
- The product and benchmark CLI support exactly one image. The S1-07 tool invokes the product interface repeatedly for a fixed experiment; neither path is a directory-batch API, worker/concurrency system, service, `inference_event`, or INT8 path.
- S1-07 proves Python ORT/C++ ORT implementation consistency for one fixed 30-image set under the frozen contract. It is not an accuracy evaluation or proof for every possible image/platform/library version; it is the required correctness gate that precedes performance publication.
- S1-08 is a warm-cache, single-image, batch-1 result from one Windows CPU machine with no affinity/priority/idle-system controls. It does not establish full-dataset latency, cold-disk behavior, cross-platform performance, or a statistically controlled hardware comparison.
- Peak Working Set is a process-lifetime high-water mark that includes session initialization and benchmark harness state. It is not a per-stage allocation profile, current RSS, or incremental model memory.
- The matching `best.pt` is absent from the workspace and Git history. Historical PT/ONNX evidence covers 50 sorted `crazing` images and only count/confidence summaries; S1-07 is a separate same-ONNX Python ORT/C++ ORT comparison and must not be described as a newly rerun PyTorch/Python-ORT/C++ three-way experiment.
- Python OpenCV 4.13.0 and C++ OpenCV 4.8.0 are recorded separately. The 30-image evidence passed the original requirements, but this does not establish bitwise equality across arbitrary OpenCV builds.
- UTF-8 strings, JSON control-byte escaping, Windows separator escaping, paths containing spaces, and the fixed local paths are tested. Arbitrary Unicode input/output paths across every Windows locale and filesystem have not been broadly validated.
- The six baseline class labels are ASCII. OpenCV's built-in Hershey text renderer is not evidence of correct arbitrary Unicode label rendering.
- Class-agnostic NMS remains deliberate baseline behavior and can suppress a lower-scoring box from another class when boxes overlap.
- The visualization is deterministic for the pinned OpenCV build; it is evidence output, not an annotation editor or GUI.

S1-09's automated gate has passed, but the user L2 gate is still pending, so Large Stage One is not complete. Stage Two has not started. After L2 acceptance, Stage Two remains responsible for INT8 PTQ comparison, broader evidence hardening, expanded regression/failure coverage, and final P0 result consolidation; batch deployment, fuzzing, concurrency stress, and a cross-platform matrix have not begun.

## License checkpoint

Repository-authored source remains MIT. The baseline declaration records the ONNX metadata text `AGPL-3.0 License (https://ultralytics.com/license)` without changing it. Public model/data distribution still requires separate review of model obligations and the NEU-DET redistribution basis; this remains a release checkpoint, not a legal conclusion.
