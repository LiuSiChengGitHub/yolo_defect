# S2-02：Linux x86_64 与 AArch64/QEMU 最终教学收口

> 最终收口日期：2026-08-28  
> 验证提交：`436ab4b`  
> 当前状态：Gate A、Gate B 与三平台关键回归完成；等待用户 L1；S2-03 未开始。

## 1. 讲解本步工作

### 1.1 五分钟口述

S2-02 解决的是同一套 C++ 推理 Runtime 能不能脱离 Windows，并进一步脱离 x86_64 架构假设的问题。开始时，项目已经能在 Windows x86_64 上完成配置加载、OpenCV 预处理、ONNX Runtime CPU 推理、YOLO decode、NMS、坐标恢复和 JSON/可视化输出，但构建依赖默认是 Windows 的 `.lib/.dll`，内存统计依赖 Windows API，工作流也只有 Windows 入口。我的原则不是复制一套 Linux 或 ARM 业务代码，而是保持 Runtime、`DetectorPipeline`、预处理和后处理只有一份，把差异限制在 CMake、薄平台适配层和脚本。

这个单元分成连续的两道门。Gate A 先处理操作系统可移植性：在 WSL2 Ubuntu x86_64 中使用 GCC、Ninja、Ubuntu OpenCV 和官方 Linux x64 ONNX Runtime SDK，修改 CMake 让 Windows 继续使用 ORT `.lib/.dll`，Linux 使用真实 `libonnxruntime.so` 和 build RPATH；内存统计在 Windows 保留 Peak Working Set，在 Linux 使用 `getrusage` 的 peak RSS。然后增加 Bash 工作流，覆盖 doctor、Release build、CTest、固定图片、Python/C++ consistency 和 benchmark。纯 C++ 的 YOLO decode、类别无关 NMS 与 letterbox 坐标恢复进入 `project_core`，完整 Runtime 直接链接它；这个边界既避免复制算法，也为后续交叉编译提供了不依赖 ORT/OpenCV 的最小 smoke。

Gate A 稳定后，Gate B 再处理 CPU 架构可移植性。host 是 WSL2/Linux x86_64，target 是 Linux AArch64。CMake、Ninja、交叉编译器和 QEMU 都是运行在 host 上的工具；最终链接的 libc、OpenCV 和 ONNX Runtime 必须是 target 的 ARM64 库。项目使用 GNU `aarch64-linux-gnu` toolchain、微软官方 ARM64 ORT SDK，以及 Ubuntu Noble 的 ARM64 OpenCV 预编译包。因为直接安装 ARM64 OpenCV 会冲突现有 amd64 开发包，所以脚本只下载 `.deb` 并解压到私有 target tree。这个目录不是完整 rootfs，因此没有把它伪装成 `CMAKE_SYSROOT`；target libc 和启动对象继续由 GNU cross toolchain 管理。

交叉编译成功还不等于可运行，所以验证分了几层。首先用 `file/readelf` 确认 core smoke、Runtime 对象、CLI 和 ORT 都是 AArch64 ELF，CLI 解释器是 `/lib/ld-linux-aarch64.so.1`。然后不使用 host `ldd`，而是在 QEMU 下调用 ARM64 loader 的 `--list`，确认 138 个目标动态库全部解析、没有 `not found`、没有混入 x86_64 库。接着实际运行 CLI startup、`--help`、配置与 artifact 契约、两条负向错误路径，以及真实复用 decode/NMS/坐标恢复的 synthetic smoke。依赖稳定后又完成固定图片经过 ARM64 OpenCV、ARM64 ORT CPU 和现有后处理输出 Detection JSON，得到 3 个检测。

最终收口在同一提交上重新跑了三条平台链。WSL2/Linux x86_64 clean Release、9 个 ELF 依赖检查、119/119 CTest、固定图 JSON/PNG 和 30 图 consistency 全部通过；consistency 是 30/30 图、62/62 个检测匹配。AArch64 重新交叉构建，ELF、loader、QEMU contracts/core 和完整固定图推理再次通过。Windows 也重新完成 MSVC/NMake clean Release、119/119 CTest 和固定图 JSON/PNG 3 个检测，证明共享 CMake 和业务边界没有破坏原基线。

这个单元最终证明的是构建、加载和功能可移植性。WSL2 是 Linux 开发环境，但不是裸机 Linux 或边缘板；QEMU 是 user-mode 指令模拟，不是真实 ARM 板卡，所以没有发布任何 QEMU latency、throughput、功耗、散热或设备稳定性结论。Gate A 先前的短 benchmark 和 peak RSS 证据仍然保留，但最终复跑没有重复 benchmark，因为本轮目标是功能与跨平台回归，不需要再制造一组高波动数字。S2-02 到这里完成工程与教学收口，下一步仍由用户先做 L1，不自动进入 S2-03。

### 1.2 教学级完整讲解

S2-02 位于 S2-01 的 INT8/PTQ 与 profiling 之后、S2-03 并发系统之前。它的输入不是一个新模型或新业务需求，而是已经冻结的单图 Runtime：`RuntimeConfig + ModelArtifactSpec` 决定运行与模型语义，ORT session 提供实际 `ModelMetadata`，OpenCV 完成图片预处理，ORT 执行推理，项目自己的代码完成 YOLO 后处理和结果写入。输出也没有改变，仍是 Detection JSON 和可选 PNG。非目标包括 Docker multi-arch、真实 ARM 板、Jetson、RK3588、TensorRT、服务化和并发。

完整因果链是：Windows-only 构建和 OS API 会阻止 Linux 原生运行，因此先把平台差异变成显式适配；Linux x86_64 通过后，仍不能证明源码摆脱了 x86_64 ABI 和目标库假设，因此再引入 AArch64 toolchain；AArch64 编译成功仍不能证明 loader 和行为正确，因此再做 ELF、依赖和 QEMU 分层验证；共享构建图被修改后可能回归 Windows，因此最后必须回到 Windows 重跑关键门。

Gate A 的 native compile 指“编译器与产物在同一种 CPU 架构上运行”：GCC 在 x86_64 Linux 上生成 x86_64 ELF。CMake 的 imported ORT target 在 Windows 指向 import library 并由构建后步骤 staging DLL，在 Linux 指向 `libonnxruntime.so`。Linux build RPATH 让 build tree 中的程序在不依赖临时 shell 环境的情况下找到选定 SDK。`platform_info` 把 OS API 隔离：Windows 使用 `GetProcessMemoryInfo` 的 Peak Working Set，Linux 使用 `getrusage(RUSAGE_SELF).ru_maxrss` 的 peak RSS。两者都是进程生命周期高水位，但来源和单位语义不同，不能直接拿数值判断哪个平台更省内存。

`project_core` 是可移植性边界，不是第二条产品链。它拥有标准库数据结构和纯算法：owned raw tensor、letterbox metadata、Detection、YOLO BCN decode、稳定的 class-agnostic NMS 和坐标恢复。正常 `DetectorPipeline` 仍然调用同一份 core；core-only 模式只是跳过 OpenCV、ORT、Python 和 GTest discovery，快速回答“这段项目核心逻辑能否被目标编译器生成并执行”。如果只跑 core smoke，就不能宣称图片预处理或完整推理已经可移植；所以 Gate B 后面还单独跑了完整 ARM64 Runtime。

Cross compile 指编译工具运行架构与目标产物架构不同。这里 host tool 包括 x86_64 CMake、Ninja、`aarch64-linux-gnu-g++`、Python 和 QEMU；target library 包括 ARM64 libc、OpenCV、ORT 和 C++ runtime。toolchain 文件设置 `CMAKE_SYSTEM_NAME=Linux`、`CMAKE_SYSTEM_PROCESSOR=aarch64` 和 GNU cross compiler，并让 CMake 的 program search 留在 host、library/include/package search 面向 target。这样 CMake 不会为了寻找一个可执行工具跑去 ARM64 tree，也不会把 host OpenCV 链进 ARM64 CLI。

sysroot 通常表示目标系统根目录视图，可能包含 target headers、libraries、loader、libc 和启动对象。本项目的私有 OpenCV tree 只是从 Ubuntu ARM64 包解出的用户态依赖集合，并不是完整 rootfs，所以只通过显式变量和 CMake imported target使用它，而没有设置成全局 `CMAKE_SYSROOT`。这是一个重要 trade-off：显式路径稍多，但避免遮蔽交叉编译器已经配置好的 libc/startup objects，也更容易诊断“哪个库来自哪里”。

QEMU user-mode 读取 AArch64 用户态指令并在 x86_64 host 上翻译执行，它复用 host kernel，不模拟整台机器。它适合检查 ELF 能否启动、loader 能否解析 `.so`、配置和算法是否正确；它不等价于 QEMU system emulation，更不等价于真实 ARM SoC。正因为指令翻译、host kernel 和硬件微架构都不同，QEMU 时间不能用于推导板卡 latency、throughput 或功耗。

排错必须先判断失败阶段。configure-time 错误通常是 SDK、header、架构或 CMake discovery；link-time 错误通常是目标库、启动对象或未解析符号；load-time 错误通常是 interpreter、ABI 或 `.so not found`；run-time 错误才进入配置、预处理、`Session::Run`、后处理和输出行为。跨平台 consistency 失败时，应按模型/config/artifact、输入解码、letterbox/RGB/NCHW、raw tensor、阈值/NMS、坐标恢复和匹配协议的顺序定位，而不是先修改容差。

验证选择与风险相称。Gate A 触及共享 CMake、类型位置和平台内存接口，因此需要完整 CTest、产品 Demo、consistency 和 Windows 回归。Gate B 不让 host CTest discovery直接执行 target ELF，而选择显式的 core/contracts/full-inference smoke；同时用 `file/readelf` 和 target loader覆盖行为测试看不到的 ABI/依赖问题。普通 JSON、PNG 和 trace 没有额外计算 SHA；模型与外部二进制身份仍沿用已有必要契约。

最终完成判定是“实现完成 + 三平台关键门复跑 + 当前文档同步完成”。它不是用户 L1 已通过，也不是严格的真实板卡验收。真实板卡性能、驱动、温度、功耗、长稳和部署方式仍然未知；AArch64 全量 GTest/CTest 也没有在 QEMU 下运行。S2-03 的目录/manifest、有界队列和并发完全没有开始。

## 2. 新增或修改的模块与设计原因

| 模块 | 输入 | 输出 | 设计与异常语义 |
|---|---|---|---|
| 统一 CMake 构建图 | 平台、ORT/OpenCV roots、toolchain | native/cross targets | 平台分支只选择依赖和加载方式；错误报告期望架构、实际文件和纠正动作 |
| `platform_info` | 当前进程与 OS | 环境字段、peak memory | 把 Windows/Linux API 限制在薄边界；不混淆 PWS 与 RSS |
| `project_core` | owned raw output、阈值、letterbox metadata | owned detections | 只依赖 C++17；完整 Runtime 和 portability smoke 复用同一算法 |
| `stage1.sh` | action、可配置 SDK/工具路径 | build/test/demo/consistency 退出码 | 先 doctor，再运行产品逻辑；任何命令非零立即失败 |
| AArch64 toolchain | host cross tools、target roots | AArch64 CMake 配置 | program 在 host 找，include/library/package 在 target 找 |
| dependency bootstrap | 官方 ARM64 ORT、Ubuntu ARM64 package indexes | 私有 target dependency tree | 不安装覆盖 host；空 dpkg 状态计算闭包；每次隔离下载目录 |
| `stage2_aarch64.sh` | target deps、action | AArch64 artifacts、静态/动态/功能结论 | 拒绝非 AArch64 ELF、`not found`、x86_64 库和失真的 contract smoke |
| 综合收口文档 | Gate A/B 实现、最终 fresh results | 单一 S2-02 教学入口 | 阶段记录保留历史，总文档负责当前结论和边界 |

主要备选方案与取舍：没有复制 Linux/ARM Runtime；没有源码编译 OpenCV/ORT；没有直接安装冲突的 ARM64 OpenCV；没有用不完整目录冒充完整 sysroot；没有在 QEMU 下跑性能；没有为了数量再新增 schema、evidence assembler 或普通 artifact SHA。

## 3. 文件变化与目录职责

```text
cpp_infer/
├── CMakeLists.txt                         # 三平台统一构建图
├── cmake/toolchains/
│   └── linux-aarch64-gnu.cmake           # AArch64 cross identity/search policy
├── include/yolo_defect_cpp/project_core.h # 纯 C++17 owned types/API
├── src/
│   ├── project_core.cpp                   # decode/NMS/坐标恢复
│   ├── platform_info.cpp                  # PWS/RSS 薄 OS 适配
│   ├── detector_pipeline.cpp              # 唯一产品业务编排
│   ├── image_preprocessor.cpp             # OpenCV preprocess
│   └── onnx_runner.cpp                    # ORT session 与 owned output
├── tests/project_core_smoke.cpp           # 真实项目核心 synthetic smoke
├── tools/
│   ├── stage1.cmd / stage1.ps1            # Windows 入口
│   ├── stage1.sh                           # Linux x86_64 入口
│   ├── bootstrap_aarch64_deps.sh           # ARM64 依赖准备
│   └── stage2_aarch64.sh                   # cross/ELF/QEMU 入口
└── results/s2_02/
    ├── linux_x86_64/                       # Gate A 已提交证据
    └── aarch64_qemu/                       # Gate B 已提交证据

docs/details/
├── s2_02_gate_a_closure.md                # Gate A 阶段快照
├── s2_02_gate_b_closure.md                # Gate B 阶段快照
└── s2_02_closure.md                       # 当前完整单元收口
```

最终审计还移除了仓库根误跟踪的 `CMakeFiles/CMakeSystem.cmake`，并增加根 `/CMakeFiles/` ignore；它是历史 CMake 探测生成物，不是源码。

## 4. 不使用 Codex 时的人工实现流程

1. 冻结既有 RuntimeConfig、artifact、metadata、preprocess、ORT、postprocess 和输出语义。
2. 搜索 `.lib/.dll`、Psapi、Windows-only 路径和 workflow 假设。
3. 把 OS 信息与 peak memory 放入薄 `platform_info`，不把 `#ifdef` 扩散进业务链。
4. 把只依赖标准库的 owned types 与 decode/NMS/restore 提取为 `project_core`，让完整 Runtime 继续调用它。
5. 修改 CMake：Windows 导入 `.lib/.dll`，Linux 导入 `.so`；保持 Runtime sources 唯一。
6. 写 Linux doctor/build/test/demo/consistency workflow，在 WSL2/Linux x86_64 clean Release 运行，并回归 Windows。
7. 安装 host cross compiler、binutils 与 QEMU；获取官方 ARM64 ORT 和 Ubuntu ARM64 OpenCV 预编译包。
8. 用私有 target tree隔离 ARM64 OpenCV，不替换 host amd64开发环境。
9. 写 toolchain，明确 host program 与 target library/include/package search。
10. 先 cross-build core smoke，再 cross-build同一 Runtime/CLI。
11. 用 `file/readelf` 检查 Machine、interpreter、`NEEDED`、RUNPATH；用 QEMU + ARM64 loader检查完整依赖。
12. 运行 QEMU startup/help、配置/artifact、错误契约、core smoke；依赖稳定后运行固定图完整推理。
13. 最后在 Linux x86_64 和 Windows 重新运行关键门，统一文档，停止等待 L1。

## 5. 入口、核心函数、输入输出与 ownership

主要入口：CLI `src/main.cpp:733`，配置契约 `src/config_loader.cpp:103`，产品编排 `src/detector_pipeline.cpp:130-190`，preprocess `src/image_preprocessor.cpp:108-169`，ORT owner/session/run `src/onnx_runner.cpp:416-668`，后处理编排 `src/postprocessor.cpp:76-109`，decode/NMS/restore `src/project_core.cpp:291-428`，输出 `src/result_writer.cpp:856`。

```cpp
contract = load_runtime_contract(config_path);
pipeline = DetectorPipeline(move(contract));

source = decode_and_validate(image_path);
preprocess = preprocess_image(source, artifact);       // owns vector<float>
raw = runner.run(shape, preprocess.tensor_nchw);       // copies ORT output
detections = postprocess_yolov8_raw(raw, contract,
                                     preprocess);       // new owned vector
document = make_detection_result(move(detections));
write_detection_outputs(document, request);
```

`DetectorPipeline` 通过 `unique_ptr<Impl>` 独占实现；`Impl` 以值拥有 contract，并独占一个 move-only `OnnxRunner`。`OnnxRunner` 通过 RAII 拥有 `Ort::Env`、options、session、allocator 和 metadata。输入 tensor只在同步 `Session::Run` 期间借用 `vector<float>` 内存；输出在离开 ORT 前复制成 owned `InferenceOutput`，所以不会把 ORT 内部 buffer 的生命周期泄漏给调用方。

## 6. 运行、测试、调试与定制

Windows：

```bat
cpp_infer\tools\stage1.cmd doctor
cpp_infer\tools\stage1.cmd clean-build
cpp_infer\tools\stage1.cmd test
cpp_infer\tools\stage1.cmd demo
```

WSL2/Linux x86_64：

```bash
export ONNXRUNTIME_ROOT="$HOME/.local/opt/onnxruntime-linux-x64-1.19.2"
export YOLO_DEFECT_PYTHON="$HOME/.venvs/yolo-defect-gate-a/bin/python"
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest

bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh demo
bash cpp_infer/tools/stage1.sh consistency
```

AArch64/QEMU：

```bash
bash cpp_infer/tools/bootstrap_aarch64_deps.sh doctor
bash cpp_infer/tools/stage2_aarch64.sh all
```

最终 fresh 验证结果：

| 平台 | 实际运行 | 结果 |
|---|---|---|
| WSL2/Linux x86_64 | clean Ninja Release、`ldd`、CTest、Demo、consistency | 40 build steps；9 个 ELF 零 `not found`；119/119；3 detections + 200x200 PNG；30/30 图、62/62 detections |
| Linux AArch64 under QEMU | clean cross-build、file/readelf、ARM loader、contracts/core/full inference | Machine AArch64；解释器正确；138 库零缺失/零 x86_64；固定图 3 detections |
| Windows x86_64 | clean MSVC/NMake Release、CTest、Demo | 119/119；固定图 JSON/PNG 3 detections |

Gate A 先前的短 benchmark 为 warmup 1/repeat 2，只证明流程与 peak RSS 采集可运行；两次样本的 peak RSS 分别为 `196.570312 MiB` 和 `196.757812 MiB`，且 latency 波动明显。本次最终复跑没有重跑 benchmark，不把旧数字冒充 fresh 结果，也不进行 Windows/Linux 排名。

常见诊断：`Incomplete SDK` 查 C++ headers/library；cross architecture mismatch 查 ARM64 dependency roots；`crt1.o/libc` 缺失查假 sysroot；`Exec format error` 表示 host直接运行 target ELF；host `ldd` 不适合 ARM64，应使用 QEMU + ARM loader；`.so not found` 查 loader prefix和target library path；consistency失败按 contract→preprocess→raw→postprocess→matching 顺序排查。

证据索引：Gate A 结果在 [`../../cpp_infer/results/s2_02/linux_x86_64/`](../../cpp_infer/results/s2_02/linux_x86_64/)，Gate B 原始记录在 [`../../cpp_infer/results/s2_02/aarch64_qemu/`](../../cpp_infer/results/s2_02/aarch64_qemu/)。普通 JSON、PNG 和 trace 没有新增 SHA。

## 7. 验收问题与连续追问

1. native compile 和 cross compile 的根本区别是什么？
2. host tool 与 target library 为什么必须分开？
3. 为什么 Linux x86_64 通过不能自动证明 AArch64 可用？
4. 为什么 `file/readelf` 通过还不等于程序一定能运行？
5. 为什么不能用 host `ldd` 检查 ARM64 ELF？
6. 私有 OpenCV tree为什么不直接设成完整 `CMAKE_SYSROOT`？
7. `project_core` 是不是复制出来的第二条推理链？它能证明和不能证明什么？
8. `$ORIGIN/../lib` 与 native Linux build RPATH 分别解决什么问题？
9. PWS 与 peak RSS 为什么不能直接比较？
10. QEMU user-mode 能证明什么，为什么不能发布板卡性能？
11. 如何证明 Windows/Linux/AArch64 使用同一业务源码？
12. 为什么跨平台重构后必须回归 Windows？
13. consistency 首图失败时按什么顺序排查，为什么不先放宽容差？
14. 如果 ARM64 full inference 没跑通，简历口径应如何降级？

能够不依赖背诵命令，连续回答这些问题并指出对应代码/证据，才达到本单元面试理解深度。

## 8. 最可能被追问和应进入代码练习的部分

1. `cpp_infer/src/project_core.cpp:291-337`：YOLOv8 BCN decode、strict threshold、`xywh→xyxy`。
2. `cpp_infer/src/project_core.cpp:340-382`：稳定排序的 class-agnostic NMS 与 `IoU > threshold`。
3. `cpp_infer/src/project_core.cpp:385-428`：letterbox 逆变换、去 padding、除 scale、clip。
4. `cpp_infer/tests/project_core_smoke.cpp:22-79`：最小 synthetic core smoke。
5. `cpp_infer/src/image_preprocessor.cpp:108-169`：letterbox、BGR→RGB、归一化、HWC→NCHW。
6. `cpp_infer/src/onnx_runner.cpp:416-668`：ORT RAII、输入借用、输出复制；公开 `run()` 从第 662 行开始。
7. `cpp_infer/src/detector_pipeline.cpp:130-190`：PImpl、move-only owner 与完整控制流。
8. `cpp_infer/src/platform_info.cpp:178-293`：最小 OS 边界、PWS/RSS 单位语义。
9. `cpp_infer/CMakeLists.txt:152-320`：`.lib/.dll`、`.so`、target readelf guard、RPATH。
10. `cpp_infer/cmake/toolchains/linux-aarch64-gnu.cmake:1-68`：最小 cross toolchain。
11. `cpp_infer/tools/stage2_aarch64.sh:193-332`：ELF/loader/QEMU/inference 验证。
12. `cpp_infer/tools/stage1.sh:262-418`：native ELF、Release、CTest、Demo、consistency。

## 9. 文档同步状态

- `AGENTS.md`：S2-02 最终三平台回归完成、等待 L1、S2-03 未开始。
- `README.md` / `README_zh.md`：同步 Gate A/B、最终 fresh 结果、综合收口入口和 QEMU 边界。
- `cpp_infer/README.md`：同步 Runtime 技术视角、命令、证据与限制。
- `docs/paths_commands.md`：保留版本和可配置入口，移除用户名/个人绝对路径。
- Gate A/B 阶段文档：保留各自证据，修正 Gate A 对 Gate B 的历史状态措辞。
- 根 CMake 生成物已移除并加入 ignore；active source/CMake/workflow 未硬编码个人路径或 x86_64 host target library。
- raw result logs 保留当次程序实际输出路径，它们是历史运行证据，不是源码默认值；archive 保留历史快照，不作为当前入口。
- WSL2/Linux x86_64、QEMU/Linux AArch64 与真实 ARM 板卡始终分开表述；没有新增 QEMU 性能、Docker multi-arch、真实板卡或 S2-03 结论。

S2-02 的实现、相称验证、综合文档和教学闭环到此完成。当前停止等待用户 L1。
