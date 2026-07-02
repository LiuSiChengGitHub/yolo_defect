# cpp_infer

This directory is the V2 C++ deployment workspace for the steel surface defect project.

Current status: P1-03 OpenCV preprocess is ready. The target prints a runtime banner, `--help`, a `--config <path>` summary, or a `--config <path> --image <path>` preprocess summary. It still does not use ONNX Runtime, GTest, inference, postprocessing, NMS, or benchmark code.

Planned layout:

```text
cpp_infer/
├── CMakeLists.txt
├── configs/default_config.txt
├── include/yolo_defect_cpp/
├── src/
└── tests/
```

## Build Commands

```cmd
:: Run from the repository root in a Visual Studio 2026 Developer Command Prompt.
set BUILD_DIR=%TEMP%\yolo_defect_cpp_p1_03
set PATH=D:\01_Base\Tools\opencv\build\x64\vc16\bin;%PATH%

cmake -S cpp_infer -B "%BUILD_DIR%" -G "NMake Makefiles" -DOpenCV_DIR=D:\01_Base\Tools\opencv\build\x64\vc16\lib
cmake --build "%BUILD_DIR%"

"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --help
"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt
"%BUILD_DIR%\bin\yolo_defect_cpp.exe" --config cpp_infer\configs\default_config.txt --image data\images\val\crazing_241.jpg

ctest --test-dir "%BUILD_DIR%" --output-on-failure
```

Local verification on 2026-06-13 passed in the Visual Studio 2026 Developer Command Prompt with the NMake build tree under `%TEMP%`. The preprocess command printed `original_size: 200x200`, `input_size: 800x800`, `color: BGR->RGB`, `normalization: float32 [0, 1]`, `layout: NCHW`, `tensor_shape: 1x3x800x800`, and `tensor_elements: 1920000`.
