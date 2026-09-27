# C++ Runtime 技术手册

[项目介绍](../README_zh.md) · [English overview](../README.md)

`cpp_infer` 提供 `yolo_defect_runtime` library 和 `yolo_defect_cpp` CLI。单图、批处理、benchmark 与 profiling 共用配置、模型声明及图像处理组件；Windows、Linux x86_64 与交叉编译的 Linux AArch64 使用同一业务源码。

另有可选的 [Qt 6 Widgets 单图检测客户端](apps/qt/README.md)，支持后台检测、原图与标注图预览、结果表格及文件输出。构建开关 `YOLO_DEFECT_BUILD_QT` 默认关闭，使用原 CLI 时无须安装 Qt。

## 构建与依赖

主依赖为 C++17 编译器、CMake 3.16+、OpenCV 4.x 和 ONNX Runtime C++ SDK 1.19.2。测试使用 GoogleTest 和可导入 `cv2`、`numpy`、`onnxruntime==1.19.2` 的 Python；量化工具另外使用 `onnx`。Python 包与 C++ SDK 是独立依赖。

### Windows

安装 Visual Studio C++ 构建工具、OpenCV C++ 和官方 Windows x64 ORT SDK。环境配置示例为 [`tools/stage1.local.example.psd1`](tools/stage1.local.example.psd1)，复制为 `cpp_infer/.stage1.local.psd1` 后填写 `OrtRoot`、`OpenCvDir`、`OpenCvBin`、`PythonExe`、`GTestSource`。该本地文件由 Git 忽略。

```powershell
Copy-Item cpp_infer/tools/stage1.local.example.psd1 cpp_infer/.stage1.local.psd1
.\cpp_infer\tools\stage1.cmd doctor
.\cpp_infer\tools\stage1.cmd build
.\cpp_infer\tools\stage1.cmd test
```

命令从仓库根目录运行。`stage1.cmd` 初始化 x64 Visual Studio 环境，再调用 PowerShell 入口；构建使用 Release/NMake。机器无关默认值位于 [`stage1.defaults.psd1`](tools/stage1.defaults.psd1)，命令参数可覆盖本地配置，例如 `-BuildDir`、`-OrtRoot`、`-PythonExe`、`-Config`。

### Linux / WSL2

Ubuntu 上的基础依赖安装示例：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config \
  libopencv-dev libgtest-dev python3-venv curl file binutils
python3 -m venv .venv
.venv/bin/python -m pip install numpy opencv-python onnxruntime==1.19.2 onnx
```

解压官方 Linux x64 ORT 1.19.2 SDK 后设置入口：

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.19.2
export YOLO_DEFECT_PYTHON="$PWD/.venv/bin/python"
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest
bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh build
bash cpp_infer/tools/stage1.sh test
```

Linux wrapper 使用 Release/Ninja，`YOLO_DEFECT_BUILD_DIR` 可选择 `/tmp` 下名称以 `yolo_defect_stage1_` 开头的构建目录；`YOLO_DEFECT_RUN_DIR` 可选择新的结果目录。该目录约束来自脚本对 `clean-build` 的实现。动态库通过 build RPATH 和 ORT 库路径加载。

不使用 wrapper 时也可直接构建：

```bash
cmake -S cpp_infer -B build/cpp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" \
  -DPython3_EXECUTABLE="$YOLO_DEFECT_PYTHON" \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$YOLO_DEFECT_GTEST_SOURCE"
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

`OpenCV_DIR` 可指定 OpenCV CMake package 目录；`BUILD_TESTING=OFF` 关闭测试构建，`YOLO_DEFECT_CORE_ONLY=ON` 只构建不依赖 OpenCV/ORT 的核心逻辑 smoke。CLI 位于构建目录的 `bin/`。

## 配置与模型契约

| 层次 | 内容 |
|---|---|
| `RuntimeConfig` | artifact 路径、score/NMS 阈值、provider |
| `ModelArtifactSpec` | 模型标识及路径、声明 SHA、opset、张量与类别、前后处理语义 |
| `ModelMetadata` | ORT 实际观察到的输入输出名称、形状、类型与 session provider |

[`configs/default_config.txt`](configs/default_config.txt) 示例：

```ini
schema_version = 1
artifact_spec_path = ../artifacts/yolov8_neu_det.artifact.txt
score_threshold = 0.25
nms_threshold = 0.45
provider = cpu
```

声明文件使用 `key = value` 格式，支持空行和 `#` 注释。config 中的 `artifact_spec_path` 相对 config 文件解析；artifact 中的 `model_path` 相对 artifact 文件解析。CLI 的 config、image 和输出相对路径则从调用者工作目录解析。batch manifest 内的图片路径相对 manifest 所在目录解析。

| 配置 | 模型 |
|---|---|
| [`default_config.txt`](configs/default_config.txt) | FP32 `models/best.onnx` |
| [`int8_u8s8_config.txt`](configs/int8_u8s8_config.txt) | QDQ/U8S8 `models/best.int8.qdq.u8s8.onnx` |
| [`int8_config.txt`](configs/int8_config.txt) | QDQ/S8S8 `models/best.int8.qdq.onnx` |

三个配置使用 CPU provider。当前模型的外部输入是 `images`、float32、NCHW `[1,3,800,800]`；输出是 `output0`、float32、BCN `[1,10,13125]`。INT8 模型内部采用量化算子，外部 I/O 保持 float32。六个类别依次为 `crazing`、`inclusion`、`patches`、`pitted_surface`、`rolled-in_scale`、`scratches`。

## 图像处理与 Runtime

`DetectorPipeline` 组合 OpenCV 解码和预处理、`OnnxRunner`、YOLO 后处理与结果对象。`OnnxRunner` 持有 ORT session 并返回独立拥有内存的输出张量。每个 session 使用 CPU、sequential execution、intra/inter-op threads=1/1。

预处理将图片解码为三通道 BGR，按 `min(input_width/source_width, input_height/source_height)` 等比例缩放。缩放尺寸取整后用值 114 居中补边；奇数 padding 的多一个像素落在右侧或底部。随后 BGR 转 RGB、除以 255，排列为连续 float32 NCHW。

后处理按以下语义执行：

1. 读取连续 BCN 输出，候选框为模型坐标中的 `cx, cy, w, h`。
2. 取六类得分的最大值作为 confidence；没有额外 objectness 或 sigmoid，相同类别得分取较小 class id。
3. 保留 `confidence > score_threshold` 的候选。
4. 在模型坐标执行 class-agnostic NMS；`IoU > nms_threshold` 时抑制，相同 confidence 保持原候选顺序。
5. 减去左/上 padding、除以缩放系数，裁剪到原图范围，输出 `xyxy` 坐标。

`include/yolo_defect_cpp/` 提供公共接口，`src/` 包含实现；`main.cpp` 负责 CLI 参数和运行模式编排，结果序列化分别由 detection、benchmark 与 batch writer 完成。

## 单图与诊断 CLI

以下示例使用 Linux 直接构建得到的 CLI；Windows 对应 `yolo_defect_cpp.exe`。

```bash
build/cpp/bin/yolo_defect_cpp \
  --config cpp_infer/configs/default_config.txt \
  --image data/images/val/crazing_241.jpg \
  --output-json outputs/sample.json --output-image outputs/sample.png
```

单图检测可独立选择 `--output-json` 或 `--output-image`。输出目录自动创建，已有文件可通过 `--overwrite` 替换；输出不能覆盖输入或指向目录、符号链接等特殊目标。图片输出通过 OpenCV 写入，无需 GUI。

其他入口：

| 参数组合 | 行为 |
|---|---|
| `--help` | 显示完整 CLI 用法 |
| `--config <file>` | 加载配置和 artifact 声明 |
| `--config <file> --image <file>` | 执行图像预处理并输出摘要 |
| `--config <file> --inspect-model` | 创建 ORT session，检查实际模型 metadata |
| `--config <file> --image <file> --raw-output-summary` | 运行推理并输出张量摘要 |

Detection JSON 的 `schema_version` 为 1：

| 字段 | 内容 |
|---|---|
| `model` | `model_id`、`declared_sha256` |
| `image` | 路径、`original_size`、`input_size` |
| `runtime` | 实际 provider、provider 信息、score/NMS 阈值、NMS 模式 |
| `detections[]` | `class_id`、`class_name`、`confidence`、`bbox_xyxy` |

JSON 使用 UTF-8、固定字段顺序和稳定数字格式；无检测时为 `detections: []`。`declared_sha256` 来自模型声明，字段本身不表示 C++ 推理过程重新计算了模型摘要。单图及诊断命令成功返回 0，错误返回 1。

## 目录、Manifest 与有界并发

```bash
build/cpp/bin/yolo_defect_cpp \
  --config cpp_infer/configs/int8_u8s8_config.txt --batch \
  --input-dir data/images/val \
  --output-dir outputs/batch --batch-summary outputs/batch/summary.json \
  --workers 4 --queue-capacity 8
```

`--input-dir` 与 `--manifest` 二选一。默认 workers=1，queue capacity=2×workers；范围分别为 1..64 和 1..4096。有效 worker 数为请求数量与任务数量的较小值，每个 worker 拥有独立 `DetectorPipeline` 和 ORT session，每次处理一张图片。队列保存任务索引，容量限制产生背压。

目录递归发现普通 `.bmp/.jpeg/.jpg/.png/.tif/.tiff/.webp` 文件，不跟随符号链接，按 UTF-8 相对路径排序。manifest 是 UTF-8 文本路径列表，允许 BOM、LF/CRLF、空行，以及首个非空字符为 `#` 的注释；有效行按声明顺序执行。例如文件放在 `data/` 下时：

```text
# selected.txt
images/val/crazing_241.jpg
images/val/crazing_242.jpg
```

manifest 的绝对路径、缺失图片、不支持的文件、重复 canonical 输入，以及空目录会使输入发现失败。输出目录不能位于输入目录内部，输出路径也不能覆盖 config、artifact、model、manifest 或源图片。

成功项写入 `items/<六位序号>.detections.json`；`--output-images` 增加同序号 PNG，`--overwrite` 允许替换已有普通输出文件。逐图失败记录在汇总中，其他任务继续执行；汇总项顺序与输入任务顺序一致。

`BatchSummary` schema v1 记录模型和运行条件、输入输出策略、workers/session 数、队列容量/峰值/等待、计数、计时、内存、协作停止状态与逐图结果。

| 状态 | 含义 | 退出码 |
|---|---|---:|
| `succeeded` | 全部成功 | 0 |
| `partial_failure` | 至少一项失败，且没有取消项 | 2 |
| `cancelled` | 收到协作停止请求 | 130 |
| `fatal` | 基础设施或 session 失败 | 1 |

## Benchmark、Profiling 与结果分析

```bash
build/cpp/bin/yolo_defect_cpp \
  --config cpp_infer/configs/default_config.txt \
  --image data/images/val/crazing_241.jpg --benchmark \
  --warmup 10 --repeat 100 --benchmark-json outputs/benchmark.json

build/cpp/bin/yolo_defect_cpp \
  --config cpp_infer/configs/int8_u8s8_config.txt \
  --image data/images/val/crazing_241.jpg --profile \
  --profile-prefix outputs/int8_ort --profile-runs 10
```

Benchmark 当前实现使用 Release、CPU、batch=1、score=0.25、NMS=0.45 和 class-agnostic NMS，并检查固定样本 `crazing_241.jpg` 的文件名与大小。warmup/repeat 默认 10/100，可通过参数调整；warmup 不计入统计，repeat 样本给出均值、nearest-rank P50/P95 和吞吐量。session 初始化单独记录，绘图不执行，统计与 JSON 写入位于采样之外。

| 计时字段 | 区间 |
|---|---|
| `image_decode` | OpenCV 图像读取/解码 |
| `preprocess` | 缩放、补边、颜色转换与张量排列 |
| `session_run` | ORT `Session::Run` |
| `postprocess` | decode、过滤、NMS 与坐标恢复 |
| `pipeline` | 预处理开始至后处理结束，包含推理周围的必要处理 |
| `end_to_end` | 图像解码开始至后处理结束 |

Batch throughput 为成功图片数除以 processing wall time。该区间包括排队、逐图处理和写出、队列排空及线程 join，不含输入发现、声明加载、session 构造和 summary 写出。Windows 记录进程 Peak Working Set，Linux 记录 peak RSS；二者是各平台进程内存高水位指标。

Profiling 创建独立的带插桩 session，对预处理后的固定图片运行并调用 `EndProfiling`，CLI 打印实际 trace 路径。trace 适合分析算子、节点与执行 provider；其计时包含插桩开销。

分析与实验工具：

| 工具 | 用途 |
|---|---|
| `compare_consistency.py` | 同一模型的 Python ORT/C++ ORT 检测比较 |
| `quantize_s2_01.py --protocol <json>` | 按协议执行静态 PTQ，生成模型和量化报告 |
| `evaluate_s2_01_correctness.py` | FP32/INT8 产品检测及有标签数据质量比较 |
| `screen_s2_01_product.py` | 小样本量化候选筛选 |
| `summarize_ort_profile.py` | trace 的算子、节点与 provider 汇总 |
| `compare_s2_01_benchmarks.py` | 对应实验协议内的 FP32/INT8 benchmark 比较 |
| `validate_batch_summary.py` / `compare_batch_runs.py` | batch 输出检查和运行结果比较 |

各工具的 `--help` 列出参数。FP32/INT8 benchmark 比较示例：

```bash
python cpp_infer/tools/compare_s2_01_benchmarks.py \
  --protocol cpp_infer/protocols/s2_01_ptq_protocol_r2_u8s8.json \
  --fp32 outputs/fp32.json --int8 outputs/int8.json \
  --output outputs/comparison.json
```

`--correctness <json>` 可附带正确性结果作为背景；缺省或未达标不阻止数值比较。比较工具保留对应实验的参数与平台约定。结果分析读取声明及结果文件，模型或数据集不必留在分析机器上；执行量化或评估的工具则读取它们实际消费的模型和样本。

## 平台脚本入口

Windows 使用 `stage1.cmd`，Linux 使用 `bash cpp_infer/tools/stage1.sh`。两者提供 `help`、`doctor`、`build`、`clean-build`、`test`、`detect`、`demo`、`batch`、`batch-compare`、`consistency`、`benchmark` 和 `all`。

`all` 依次进行增量构建、完整 CTest、固定 warmup=10/repeat=100 的 benchmark 和固定 manifest batch；独立 `benchmark` 可调整次数，不隐式运行一致性比较。单独的 `clean-build` 重建构建目录。`batch-compare` 使用固定 361 图、queue=8、workers=1/4、JSON-only 协议，分别运行后比较逐图结果和性能。

```powershell
.\cpp_infer\tools\stage1.cmd detect data\images\val\crazing_241.jpg -Config cpp_infer\configs\int8_u8s8_config.txt
.\cpp_infer\tools\stage1.cmd batch cpp_infer\tests\fixtures\s2_03_consistency_manifest.txt -Workers 2
.\cpp_infer\tools\stage1.cmd benchmark -Warmup 1 -Repeat 3
.\cpp_infer\tools\stage1.cmd profile -Config cpp_infer\configs\int8_u8s8_config.txt
```

```bash
bash cpp_infer/tools/stage1.sh detect data/images/val/crazing_241.jpg --config cpp_infer/configs/int8_u8s8_config.txt
bash cpp_infer/tools/stage1.sh batch cpp_infer/tests/fixtures/s2_03_consistency_manifest.txt --workers 2
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 3
bash cpp_infer/tools/stage1.sh batch-compare --config cpp_infer/configs/int8_u8s8_config.txt
```

Windows 提供 `profile` wrapper 动作；Linux 的 profiling 使用前述 C++ CLI。机器上的已记录 SDK 位置与环境问题集中在[环境小抄](../.agents/skills/yolo-defect-dev/SKILL.md)。

## AArch64 交叉编译与 QEMU

现有 bootstrap 对应 Ubuntu 24.04 Noble x86_64 host：使用 GNU AArch64 工具链、官方 ARM64 ORT SDK、从 Ubuntu ARM64 包提取的 OpenCV 私有 sysroot，以及 QEMU user-mode。以下是该环境的安装示例。

```bash
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu \
  libc6-dev-arm64-cross binutils-aarch64-linux-gnu qemu-user
```

ARM64 软件包来自 Ubuntu ports。给现有 `/etc/apt/sources.list.d/ubuntu.sources` 的 amd64 deb822 源添加 `Architectures: amd64`，在 `/etc/apt/sources.list.d/ubuntu-ports-arm64.sources` 中配置：

```text
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports
Suites: noble-security
Components: main restricted universe multiverse
Architectures: arm64
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

```bash
sudo dpkg --add-architecture arm64
sudo apt update
bash cpp_infer/tools/bootstrap_aarch64_deps.sh fetch
bash cpp_infer/tools/stage2_aarch64.sh build
bash cpp_infer/tools/stage2_aarch64.sh infer
bash cpp_infer/tools/stage2_aarch64.sh batch
```

bootstrap 通过 `apt-get download` 和 `dpkg-deb -x` 提取 ARM64 OpenCV 包，不将它们安装到 host。交叉编译使用 [`linux-aarch64-gnu.cmake`](cmake/toolchains/linux-aarch64-gnu.cmake)。部署目录包含 `deploy/bin` 与 `deploy/lib`，OpenCV 等 target 库位于私有 sysroot。

| 动作 | 行为 |
|---|---|
| `doctor` | 检查 host 工具与 target 依赖 |
| `build` / `clean-build` | 增量交叉编译 / 重建 core 与完整 Runtime |
| `inspect` | 检查 ELF 架构、loader 与动态依赖 |
| `smoke` | QEMU 下执行核心逻辑与 CLI 诊断 |
| `infer` | QEMU 下执行固定单图检测 |
| `batch` | 目录/manifest、多 worker 与部分失败场景 |
| `all` | doctor、clean-build、inspect、smoke、infer、batch |

`YOLO_DEFECT_AARCH64_CONFIG` 选择 config，省略时为 FP32；例如：

```bash
YOLO_DEFECT_AARCH64_CONFIG="$PWD/cpp_infer/configs/int8_u8s8_config.txt" \
  bash cpp_infer/tools/stage2_aarch64.sh infer
```

依赖、sysroot、构建和结果目录可通过脚本 `help` 及环境小抄列出的 `YOLO_DEFECT_AARCH64_*` 变量设置。QEMU 在 x86_64 host 上模拟 AArch64 程序执行，结果描述构建与功能可移植性；该入口不运行原生 ARM 设备 benchmark。
