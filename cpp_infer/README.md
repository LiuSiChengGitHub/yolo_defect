# cpp_infer

This directory is the V2 C++ deployment workspace for the steel-surface defect project.

Current status: **S1-04 L1 is accepted. S1-05 is implemented and verified, awaiting L1 acceptance.** The fixed single-image CLI now connects the validated Runtime/artifact contract, OpenCV preprocessing, an ONNX Runtime CPU session, owned raw output, YOLOv8 postprocess, stable detection JSON, and a GUI-free visualization. S1-06 is the only next implementation stage; batch processing, consistency tooling, and benchmarking have not started.

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

## Layout and responsibilities

```text
cpp_infer/
├── CMakeLists.txt
├── artifacts/yolov8_neu_det.artifact.txt
├── configs/default_config.txt
├── include/yolo_defect_cpp/
│   ├── artifact_spec.h
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
│   ├── config_loader.cpp
│   ├── detector_pipeline.cpp
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
└── tests/
    ├── result_writer_test.cpp
    ├── image_file_probe.cpp
    ├── postprocessor_test.cpp
    ├── preprocessor_mat_test.cpp
    ├── inference_runner_test.cpp
    ├── model_metadata_validator_test.cpp
    ├── assert_detection_json.py
    ├── assert_single_image_outputs.cmake
    ├── assert_cli_arguments_failure.cmake
    ├── assert_*.cmake
    └── fixtures/
```

- `RuntimeConfig` owns runtime policy: score/NMS thresholds, configured provider, and the artifact declaration path.
- `ModelArtifactSpec` owns model identity, declared SHA-256, provenance/license, tensor I/O, classes, and preprocess/postprocess/NMS semantics.
- `ImagePreprocessor` exposes file-path and `const cv::Mat&` entries. Both use the same letterbox/RGB/normalize/NCHW implementation.
- `OnnxRunner` owns ORT resources through RAII/PImpl. `run()` borrows the preprocess vector only for synchronous `Session::Run`, then copies output into ORT-independent storage.
- `Postprocessor` performs validated BCN decode, class argmax, strict score filtering, `xywh -> xyxy`, IoU, stable class-agnostic NMS, inverse letterbox, and clipping.
- `SingleImageDetectionResult` is a self-contained result object holding model, image, Runtime, provider, and detection evidence needed by output writers.
- `DetectorPipeline` connects the established modules for exactly one image and protects config, artifact, model, and source-image paths from output overwrite.
- `ResultWriter` validates the result, escapes JSON strings, uses locale-independent stable numeric formatting, creates parent directories, enforces overwrite policy, and renders deterministic labels/colors without a GUI.
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

| Dependency | Verified S1-05 input |
|---|---|
| Compiler | MSVC 19.50.35721.0, x64 |
| CMake / CTest | 4.1.1-msvc1, Visual Studio bundled runtime |
| OpenCV | 4.8.0 x64 vc16 at `D:\01_Base\Tools\opencv` |
| ONNX Runtime C++ SDK | Official Windows x64 CPU SDK 1.19.2 at `D:\01_Base\Tools\onnxruntime-win-x64-1.19.2` |
| Python test interpreter | Python 3.9 or newer; used only when `BUILD_TESTING=ON` for standard-library JSON verification |
| GoogleTest | Official v1.17.0 commit `52eb8108c5bdec04579160ae17225d66034bd723`; archive SHA-256 `9A56A54AE784394FF664CD55E8F4C9A03B503EBF0CB99576321C78AB3D87CA84` |

CMake fetches GoogleTest only with `BUILD_TESTING=ON`, verifies the pinned archive hash, disables installation/GMock, and keeps the parent MSVC runtime choice. A fully offline configure can point at a separately verified extracted source:

```powershell
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST='<path-to-verified-googletest-source>'
```

The source override bypasses archive download/hash checking during that configure, so verify the pinned archive before extraction. No personal GTest path is committed. ORT is located only through `ONNXRUNTIME_ROOT`, and its matching DLL is staged beside ORT-using executables.

## Clean Release build, demo, and acceptance

From CMD, initialize the x64 MSVC environment and start a profile-free PowerShell:

```bat
call "D:\01_Base\Tools\VisualStudio_Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64
powershell.exe -NoProfile -NoExit
```

Ensure Python 3.9+ is on `PATH`, or replace `$PythonExe` below with its absolute executable path. Then run from the repository root:

```powershell
$OrtRoot = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$OpenCvDir = 'D:\01_Base\Tools\opencv\build\x64\vc16\lib'
$OpenCvBin = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin'
$CMakeBin = 'D:\01_Base\Tools\VisualStudio_Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin'
$PythonExe = (Get-Command python.exe -ErrorAction Stop).Source

$env:ONNXRUNTIME_ROOT = $OrtRoot
$env:PATH = $CMakeBin + ';' + $OpenCvBin + ';' + $env:PATH
$BuildDir = Join-Path $env:TEMP `
  ('yolo_defect_s1_05_' + [guid]::NewGuid().ToString('N'))

cmake -S cpp_infer -B $BuildDir -G 'NMake Makefiles' `
  -DOpenCV_DIR="$OpenCvDir" `
  -DONNXRUNTIME_ROOT="$OrtRoot" `
  -DPython3_EXECUTABLE="$PythonExe" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON
cmake --build $BuildDir
```

Run the fixed complete single-image demo. Outputs remain under the disposable build directory instead of modifying source-controlled inputs:

```powershell
$Config = (Resolve-Path 'cpp_infer\configs\default_config.txt').Path
$Image = (Resolve-Path 'data\images\val\crazing_241.jpg').Path
$DemoDir = Join-Path $BuildDir 'demo outputs'
$JsonPath = Join-Path $DemoDir 'crazing_241.json'
$VisualizationPath = Join-Path $DemoDir 'crazing_241.png'

& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --output-json $JsonPath `
  --output-image $VisualizationPath

& $PythonExe -m json.tool $JsonPath
& "$BuildDir\bin\yolo_defect_image_probe.exe" $VisualizationPath
Get-Item $JsonPath, $VisualizationPath
```

Re-running the same command without `--overwrite` must fail because both files exist. Add the explicit flag only when replacement is intended:

```powershell
& "$BuildDir\bin\yolo_defect_cpp.exe" `
  --config $Config `
  --image $Image `
  --output-json $JsonPath `
  --output-image $VisualizationPath `
  --overwrite
```

Run focused and complete test gates:

```powershell
ctest --test-dir $BuildDir -L output --output-on-failure
ctest --test-dir $BuildDir `
  -R yolo_defect_cpp_single_image_outputs -V
ctest --test-dir $BuildDir -N
ctest --test-dir $BuildDir --output-on-failure
```

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
committed demo JSON:             1,164 bytes, SHA-256 E8445BC92201307430A17B7B51B6CCEFC5A74D2D473617170F50AD921CCF9049
committed demo PNG:              39,306 bytes, SHA-256 3A0C6C57EE977EE02762F05FCDE6928C8AACBD20883596D3622A6225942E2346
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

## Current limits

- `actual_provider` is session-level evidence from explicit CPU EP registration plus successful session creation. It is not per-node profiling evidence.
- `model.declared_sha256` is copied from the validated artifact declaration. S1-05 does not recompute the model file hash during every CLI run.
- JSON and visualization are validated before writing, but they are two files rather than one filesystem transaction. A disk-level failure while writing the second file can leave the first file present.
- The C++17 path check and later file open are not a cross-process atomic transaction. This S1-05 CLI is verified for one local invocation, not for competing writers racing on the same destination; an explicit overwrite can also leave a partial output after a disk-level write failure.
- The CLI supports exactly one image. There is no directory batch mode, worker/concurrency system, service, `inference_event`, INT8 path, benchmark, or memory result.
- The fixed sample proves this C++ vertical slice and stable output contract. It does not yet prove Python/C++ postprocess consistency; that remains a later stage.
- UTF-8 strings, JSON control-byte escaping, Windows separator escaping, paths containing spaces, and the fixed local paths are tested. Arbitrary Unicode input/output paths across every Windows locale and filesystem have not been broadly validated.
- The six baseline class labels are ASCII. OpenCV's built-in Hershey text renderer is not evidence of correct arbitrary Unicode label rendering.
- Class-agnostic NMS remains deliberate baseline behavior and can suppress a lower-scoring box from another class when boxes overlap.
- The visualization is deterministic for the pinned OpenCV build; it is evidence output, not an annotation editor or GUI.

S1-06, the automated main-path and core failure-injection gate, is the only next stage after S1-05 L1 acceptance. Do not infer that S1-06, consistency, benchmark, or batch deployment has already begun.

## License checkpoint

Repository-authored source remains MIT. The baseline declaration records the ONNX metadata text `AGPL-3.0 License (https://ultralytics.com/license)` without changing it. Public model/data distribution still requires separate review of model obligations and the NEU-DET redistribution basis; this remains a release checkpoint, not a legal conclusion.
