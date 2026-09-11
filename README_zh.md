# 工业视觉边缘 AI Runtime 与 C++ 工程化系统

[English](README.md)

基于 C++17、CMake、OpenCV 和 ONNX Runtime 的工业表面缺陷推理系统。项目将模型契约、图像处理、推理、有界并发和性能分析组合为可复用的 Runtime library。

当前实现支持 FP32 与 QDQ/U8S8 INT8 模型、单图检测、目录/manifest 处理、Python/C++ 一致性比较、benchmark 和 ORT profiling。主链可在 Windows x86_64、WSL2/Linux x86_64 运行；同一源码还可交叉编译为 Linux AArch64，并通过 QEMU user-mode 执行。QEMU 执行用于展示仿真环境下的功能可移植性。

![推理演示](docs/assets/demo_inference_result.gif)

## 架构设计

```text
RuntimeConfig + ModelArtifactSpec -> actual ModelMetadata
                                      |
Image -> OpenCV preprocess -> ONNX Runtime CPU -> owned output tensors
                                      |
                       YOLO decode -> NMS -> coordinate restore
                                      |
                      DetectionResult -> JSON / visualization
                                      |
Directory / manifest -> bounded queue -> worker-owned DetectorPipeline
                                      |
                        ordered per-image results + BatchSummary
```

`RuntimeConfig` 选择模型声明、provider 和阈值；`ModelArtifactSpec` 描述模型及其张量、处理语义；`ModelMetadata` 记录实际加载模型的信息。`DetectorPipeline` 组合预处理、推理和后处理；`BatchRunner` 将图像分发给各自拥有 pipeline 和 ORT session 的 worker。

Runtime library 承载可复用行为，CLI 负责参数与文件编排；平台适配包括依赖发现、动态库加载、内存测量和信号处理。Windows 与 Linux 共享检测实现。

## 功能模块

| 位置 | 用途 |
|---|---|
| [`cpp_infer/include`](cpp_infer/include/) / [`cpp_infer/src`](cpp_infer/src/) | C++ 契约、Runtime、批处理与 CLI |
| [`cpp_infer/configs`](cpp_infer/configs/) / [`cpp_infer/artifacts`](cpp_infer/artifacts/) | 运行选项与模型声明 |
| [`cpp_infer/tools`](cpp_infer/tools/) | 构建运行入口、一致性、量化与结果分析 |
| [`cpp_infer/tests`](cpp_infer/tests/) | C++、Python 测试，CLI 检查与输入 fixture |
| [`cpp_infer/protocols`](cpp_infer/protocols/) | 量化及比较实验参数 |
| [`models`](models/) / `data/` | 模型产物与本地图像、标签 |
| [`scripts`](scripts/) / [`src`](src/) / [`api`](api/) | Python 训练、评估、推理与 API 工具 |

随项目提供的 YOLOv8 模型检测 NEU-DET 的六类缺陷：裂纹、夹杂、斑块、麻点、轧入氧化皮和划痕。FP32 与 INT8 通过配置文件切换，面向 Runtime 的输入输出均为相同的 float32 契约。

## 快速开始

按照 [C++ 技术手册](cpp_infer/README.md) 配置依赖后，在仓库根目录运行以下命令。

### Windows

复制环境示例到 Git 忽略的本地配置文件，填入自己的 SDK 和 Python 路径：

```powershell
Copy-Item cpp_infer/tools/stage1.local.example.psd1 cpp_infer/.stage1.local.psd1
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd detect data\images\val\crazing_241.jpg
.\cpp_infer\tools\stage1.cmd batch data\images\val -Workers 4 -QueueCapacity 8
```

`stage1.cmd` 自动初始化 Visual Studio x64 环境。`detect` 默认输出 JSON 和 PNG；`batch` 输出逐图 JSON 与汇总，增加 `-OutputImages` 可输出图片。

### Linux / WSL2

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON=/path/to/python
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest
bash cpp_infer/tools/stage1.sh build
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg
bash cpp_infer/tools/stage1.sh batch data/images/val --workers 4 --queue-capacity 8
```

`detect`、`batch`、`batch-compare` 可通过 Windows 的 `-Config cpp_infer\configs\int8_u8s8_config.txt` 或 Linux 的 `--config cpp_infer/configs/int8_u8s8_config.txt` 选择 INT8。默认配置选择 FP32。

## 工具使用流程

| 动作 | 行为 |
|---|---|
| `help` / `doctor` | 显示用法 / 检查构建环境 |
| `build` / `clean-build` | 增量构建 / 重建构建目录 |
| `test` | 构建并运行 CTest |
| `detect` / `demo` | 处理指定图片 / 运行固定演示 |
| `batch` | 通过有界 worker 处理目录或 UTF-8 路径列表 manifest |
| `batch-compare` | 固定图集、queue=8，对比 workers=1 与 workers=4 |
| `consistency` | 比较 Python ORT 与 C++ ORT 检测结果 |
| `benchmark` | 独立运行 benchmark，默认 warmup=10、repeat=100 |
| `all` | 增量构建、完整 CTest、benchmark 和 batch |

Windows 另外提供 `profile` 动作。两个平台都可通过 C++ CLI 的 `--profile` 获取带插桩 ORT session 的算子、节点和 provider 计时；benchmark 使用独立的无插桩 session。

AArch64 交叉编译与 QEMU 执行使用 `bootstrap_aarch64_deps.sh` 和 `stage2_aarch64.sh`，安装与命令见[技术手册](cpp_infer/README.md#aarch64-交叉编译与-qemu)。

## 文档入口

- [C++ 技术手册](cpp_infer/README.md)：依赖、配置、CLI、输出格式、计时与分析工具。
- [Runtime 配置](cpp_infer/configs/default_config.txt)与[模型声明](cpp_infer/artifacts/yolov8_neu_det.artifact.txt)：可执行的配置示例。
- [历史归档](docs/archive/)：旧文档与实验过程记录。
