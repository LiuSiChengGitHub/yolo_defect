# S2-02 Gate B：AArch64 Cross Compile + QEMU smoke 收口记录

> 本文记录 2026-08-28 Gate B 的实际构建与执行结果。当前依赖位置和复现入口统一以 [`../paths_commands.md`](../paths_commands.md) 为准。

Gate B 已在 WSL2 Ubuntu 24.04.4 LTS 的 Linux x86_64 host 上，以 Linux AArch64 为 target 完成交叉构建、ELF/动态依赖证明、QEMU user-mode 功能 smoke，并进一步跑通了固定图片的 ARM64 ONNX Runtime CPU 完整推理。QEMU 不是 ARM 开发板；本步没有采集、比较或发布任何模拟器性能数字。

## 1. Host、target 与依赖边界

```text
x86_64 CMake/Ninja/aarch64-linux-gnu-g++  （host tools）
                         |
                         v
                 AArch64 ELF 产物
                         |
                         v
ARM64 libc/OpenCV/ONNX Runtime            （target libraries）
                         |
                         v
x86_64 qemu-aarch64 + ARM64 loader        （功能模拟执行）
```

| 项目 | 实测值 |
|---|---|
| Host | WSL2 Ubuntu `24.04.4 LTS`，Linux `x86_64`，kernel `6.18.33.2-microsoft-standard-WSL2` |
| Target | Linux `AArch64`，GNU triple `aarch64-linux-gnu` |
| Cross compiler | GCC/G++ `13.3.0` |
| Binutils | `2.42` |
| CMake / Ninja | `3.28.3` / `1.11.1` |
| QEMU user-mode | `8.2.2` |
| Target loader | `/usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1` |
| Target OpenCV | Ubuntu Noble ARM64 `4.6.0` packages，解压到私有 sysroot，没有安装或替换 host OpenCV |
| Target ONNX Runtime | Microsoft 官方 `onnxruntime-linux-aarch64-1.19.2.tgz` |

Ubuntu 的 `libopencv-dev:arm64` 会与现有 amd64 OpenCV 开发包发生冲突。实际方案没有强行安装它，而是由 [`bootstrap_aarch64_deps.sh`](../../cpp_infer/tools/bootstrap_aarch64_deps.sh) 下载 ARM64 package closure 与三个 component dev 包，再把 150 个 ARM64 `.deb` 解压到 `$HOME` 下的私有 target sysroot。这样 host 工具继续是 x86_64，target headers/libraries 明确来自 ARM64 包，也没有进入 OpenCV 或 ORT 源码编译。

## 2. 构建边界

- [`linux-aarch64-gnu.cmake`](../../cpp_infer/cmake/toolchains/linux-aarch64-gnu.cmake) 声明 target OS/CPU、GNU cross compiler 和 host/target 查找规则；它有意不把私有 OpenCV 目录设置为 `CMAKE_SYSROOT`，因为 Ubuntu cross compiler 自己管理 target libc/startup objects。
- CMake 在 cross-build 时只从显式私有 sysroot 导入 ARM64 OpenCV，并在 configure 阶段用 target `readelf` 拒绝混入 x86_64 OpenCV/ORT。
- 正常 Runtime、CLI、preprocess、ORT backend、postprocess 和 `DetectorPipeline` 仍使用 Windows/Linux x86_64 的同一份源码；没有复制 AArch64 业务实现。
- `project_core` 的独立 tree 只构建纯 C++17 decode/NMS/坐标恢复 smoke；另一个 full tree 则交叉编译完整 `yolo_defect_runtime` 和生产 `yolo_defect_cpp`。
- cross CTest/GTest discovery 没有启用，因为 host CMake 不应在 configure/discovery 阶段直接执行 target ELF；选定的 target smoke 由 QEMU 工作流显式运行。

## 3. 静态与动态证明

`file` 与 `readelf` 的实测结果为：

| 产物 | 证明 |
|---|---|
| `yolo_defect_project_core_smoke` | ELF64、ARM aarch64、解释器 `/lib/ld-linux-aarch64.so.1` |
| `detector_pipeline.cpp.o` | ELF64 relocatable、ARM aarch64；证明完整 Runtime 的共享业务源码确实被交叉编译 |
| `yolo_defect_cpp` | ELF64 PIE、Machine `AArch64`、解释器 `/lib/ld-linux-aarch64.so.1` |
| `libonnxruntime.so.1.19.2` | ELF64 shared object、ARM aarch64 |
| CLI direct `NEEDED` | ARM64 OpenCV core/imgproc/imgcodecs、ORT、C++ runtime、libc 等；RUNPATH 仅为 `$ORIGIN/../lib` |

随后不是用 host `ldd` 猜测，而是在 QEMU 下调用 ARM64 dynamic loader 的 `--list`。它实际解析出 138 个 target shared libraries，`not found = 0`；工作流逐个对解析后的文件执行 `file/readelf`，全部为 AArch64，没有 x86_64 library 混入。

原始记录：

- [ELF inspection](../../cpp_infer/results/s2_02/aarch64_qemu/elf_inspection.txt)
- [ARM64 loader resolution](../../cpp_infer/results/s2_02/aarch64_qemu/loader_resolution.txt)

## 4. QEMU 实际功能结果

| 检查项 | 分类 | 结果 |
|---|---|---|
| CMake toolchain、private sysroot 与统一 workflow | 已完成 | 可从依赖 bootstrap 连续执行到 full inference |
| project-core cross-build | 仅静态证明 | AArch64 library/object/executable 均生成；实际行为另见下一行 |
| Runtime + production CLI cross-build | 仅静态证明 | 同一业务源码生成 AArch64 Runtime archive 和 CLI；实际行为另见后续行 |
| CLI startup / `--help` | 实际运行 | QEMU user-mode exit `0` |
| RuntimeConfig + ModelArtifactSpec | 实际运行 | 从 `/tmp` 调用绝对 config，正确解析相对 artifact/model 路径 |
| 错误路径 | 实际运行 | 非法 NMS threshold、未知 artifact field 均 nonzero 且包含 expected/actual/action 信息 |
| YOLO synthetic core smoke | 实际运行 | 真实复用项目 decode → class-agnostic NMS → letterbox 坐标恢复，exit `0` |
| 固定图片 preprocess | 实际运行 | ARM64 OpenCV 读取 `crazing_241.jpg` 并进入现有 preprocess |
| 固定图片 ARM64 ORT CPU inference | 实际运行 | `CPUExecutionProvider` 完成 `Session::Run` |
| Detection JSON | 实际运行 | 3 个 detections；既有 JSON validator 通过 |
| ARM64 visualization PNG | 未执行 | Gate B 只要求完整推理 JSON；没有为增加证据数量再写一张重复 PNG |
| AArch64 全量 GTest/CTest | 未执行 | target discovery 会跨 host/target 边界；本 gate 选择显式 QEMU contracts/core/full-inference smoke |
| QEMU benchmark / latency / throughput / power | 未执行 | 明确禁止；模拟器不作为性能环境 |
| 真实 ARM 板卡、Jetson、Docker multi-arch | 未执行 | 不在 Gate B 范围；QEMU 结果不冒充板卡部署 |

QEMU smoke 原始输出见 [qemu_smoke.txt](../../cpp_infer/results/s2_02/aarch64_qemu/qemu_smoke.txt)。完整推理 JSON 见 [crazing_241.detections.json](../../cpp_infer/results/s2_02/aarch64_qemu/detect/crazing_241.detections.json)。本次成功执行了 full ARM64 inference，因此不是 `NOT_EXECUTED_UNDER_EMULATION` 降级状态。

## 5. 原生回归与结论

cross-build 改动之后，WSL2/Linux x86_64 重新完成 clean Ninja Release build，9 个原生 ELF 的 `ldd` 均无 `not found`，完整 CTest 为 `119/119 passed`。这说明新增的 target-only CMake 分支没有改变原生 OpenCV/ORT 查找和业务行为。

Gate B 最终证明的是：同一 C++ Runtime 不仅能生成 AArch64 机器码，核心逻辑和完整 ARM64 ORT 单图链路也能在 QEMU user-mode 下正确运行。它不证明真实 ARM 板卡的延迟、吞吐、功耗、散热、驱动或部署稳定性，也没有开始 S2-03。
