# S2-02 Gate A：WSL2/Linux x86_64 Native 收口记录

> 本文记录 2026-08-28 Gate A 收口时的环境与实测快照。当前工具链路径、安装命令和环境入口统一以 [`../paths_commands.md`](../paths_commands.md) 为准。

S2-02 Gate A 已在 WSL2 Ubuntu 24.04.4 LTS x86_64 上完成 clean Release build、完整测试、固定 Demo、Python/C++ consistency、短 benchmark、peak RSS、ELF/动态依赖检查和 Windows 回归。它证明同一业务源码可以在 Windows 与 WSL2/Linux x86_64 上运行。本文保留 Gate A 阶段快照；后续 Gate B 与 S2-02 总收口均已完成，整体结论见 [`s2_02_closure.md`](s2_02_closure.md)。

## 1. 环境快照

| 项目 | Gate A 实测值 |
|---|---|
| 平台 | WSL2 Ubuntu `24.04.4 LTS`，x86_64 |
| Kernel | `6.18.33.2-microsoft-standard-WSL2` |
| 编译器 | GCC/G++ `13.3.0`，C++17 |
| 构建工具 | CMake/CTest `3.28.3`，Ninja `1.11.1`，pkg-config `1.8.1` |
| OpenCV C++ | Ubuntu `/usr` distro package，`4.6.0` |
| ONNX Runtime C++ | 官方 Linux x64 SDK `1.19.2`，默认 `$HOME/.local/opt/onnxruntime-linux-x64-1.19.2` |
| Python reference | Python `3.12.3` venv；ORT `1.19.2`、OpenCV `4.10.0`、NumPy `2.0.2`；`CPUExecutionProvider` |
| GoogleTest | Ubuntu distro source `/usr/src/googletest`，`1.14.0` |

依赖准备使用 `build-essential cmake ninja-build pkg-config libopencv-dev python3-venv python3-pip libgtest-dev gdb`。C++ ORT 使用 Microsoft 官方 `onnxruntime-linux-x64-1.19.2.tgz`，没有把 Python wheel 当成 C++ SDK。

## 2. 交付的跨平台边界

- CMake 在 Windows 导入 `.lib/.dll`，在 Linux 导入官方 `libonnxruntime.so`；Linux executable 通过 build RPATH 解析指定 SDK，不复制 Runtime 主链。
- 平台信息与进程 peak memory 进入薄适配层：Windows 保留 Peak Working Set，Linux 使用自身的 peak RSS 语义。
- Linux [`stage1.sh`](../../cpp_infer/tools/stage1.sh) 提供 `doctor`、`build`、`clean-build`、`test`、`detect`、`demo`、`consistency`、`benchmark` 和 `all`，其 correctness-before-benchmark 语义与 Windows 入口一致。
- 纯 C++17 的 `project_core` 承载 YOLO raw decode、class-agnostic NMS 与 letterbox 坐标恢复；完整 Windows/Linux Runtime 复用它，`YOLO_DEFECT_CORE_ONLY=ON` 则跳过 ORT/OpenCV/Python/GTest，为 Gate B 留出有界 portability smoke。

## 3. Linux clean build、测试与固定 Demo

Ninja Release clean build 完成 `40` 个 build steps。完整 Linux CTest 为：

```text
119/119 passed
final post-hardening run: 218.19 s
initial closure run: 223.20 s
```

两个数字只是完整回归墙钟时间，不用于性能比较；最终一轮还包含收紧后的 CTest inventory 与无 `LD_LIBRARY_PATH` 的 RPATH 检查。

固定输入 `data/images/val/crazing_241.jpg` 经 Linux C++ Runtime 得到 3 个 `crazing` detections；JSON 中原图尺寸为 `200x200`，PNG 经无 GUI 路径生成并回读为 `200x200`：

- [Linux detection JSON](../../cpp_infer/results/s2_02/linux_x86_64/detect/crazing_241.detections.json)
- [Linux visualization PNG](../../cpp_infer/results/s2_02/linux_x86_64/detect/crazing_241.visualized.png)

## 4. Python ORT/C++ ORT consistency

比较继续使用同一个 FP32 ONNX、RuntimeConfig、artifact 和固定 30 图 manifest，Python 与 C++ 均为 CPU EP：

| 指标 | Linux 实测 |
|---|---:|
| 图片 | `30/30` passed |
| Python/C++ matched detections | `62/62` |
| 最大 confidence 绝对误差 | `8.049977111568296e-07` |
| 最大 bbox 坐标绝对误差 | `9.135351561440075e-05 px` |
| 最小 matching IoU | `0.999998927116394` |

这证明固定集合上的 Python ORT/C++ ORT 实现一致性；它不是模型精度评估，也不外推到 AArch64。

原始机器可读结果保留为 [summary.json](../../cpp_infer/results/s2_02/linux_x86_64/consistency/summary.json) 与 [per_image.json](../../cpp_infer/results/s2_02/linux_x86_64/consistency/per_image.json)。

## 5. Linux 短 benchmark 与 peak RSS

本次只为验证 Linux benchmark 与 memory 路径可运行，使用固定单图、batch 1、CPU EP、warmup `1`、repeat `2`。首次收口与持久化收口复跑均为真实执行：

| 指标 | 首次短跑 | 持久化收口短跑 |
|---|---:|---:|
| `Session::Run` mean | `120.807577 ms` | `135.699394 ms` |
| Pipeline mean | `128.968319 ms` | `144.103708 ms` |
| End-to-end mean | `135.896991 ms` | `151.273896 ms` |
| End-to-end throughput | `7.358515 img/s` | `6.610526 img/s` |
| Peak RSS | `206,118,912 bytes` / `196.570312 MiB` | `206,315,520 bytes` / `196.757812 MiB` |

两次短跑本身已有明显波动，两个 measured iterations 更不足以形成稳定分布；这些数字只是工作流 smoke，不作为正式 Linux 性能基线，也不与 Windows 延迟排名。Linux peak RSS 与 Windows Peak Working Set 来自不同 OS 接口和语义，不能把二者的数值差直接解释为内存优化或回退。

[原始 benchmark JSON](../../cpp_infer/results/s2_02/linux_x86_64/benchmark/yolov8_neu_det_cpu_release.json) 同时记录 Linux kernel、x86_64、GNU `13.3.0`、OpenCV `4.6.0`、ORT `1.19.2`、CPU EP、协议、分段结果和 `peak_rss`，因此直接满足 environment/result JSON 留档，不另造 evidence schema。

## 6. ELF、动态依赖与 core-only smoke

full build 的 `bin` 中检查了 `9` 个 ELF executables，所有 `ldd` 结果均无 `not found`。检查时显式移除 `LD_LIBRARY_PATH`，CLI 仍通过 build RPATH 把 `libonnxruntime.so` 解析到固定官方 SDK；OpenCV 动态库解析到 Ubuntu `/usr` system installation。

独立 core-only Release configure/build 的 CTest 为 `1/1` 通过；其 smoke executable 的动态依赖不含 ONNX Runtime 或 OpenCV。该 smoke 真实执行固定 raw output 的 decode、NMS 和坐标恢复，但不创建 ORT session、不读取图片，因而不能冒充完整 inference。

## 7. Windows 回归

跨平台重构后重新运行 Windows x86_64 Release 完整测试：

```text
119/119 passed
final shared-core rerun: 31.36 s
```

这证明新增 Linux/CMake/platform core 边界没有让现有 Windows 自动门回退；它不把 Linux 和 Windows 测试耗时当作性能对比。

## 8. 复现入口与边界

新 WSL shell 先进入仓库并设置依赖：

```bash
# 从仓库根目录执行
export ONNXRUNTIME_ROOT="$HOME/.local/opt/onnxruntime-linux-x64-1.19.2"
export YOLO_DEFECT_PYTHON="$HOME/.venvs/yolo-defect-gate-a/bin/python"
export YOLO_DEFECT_GTEST_SOURCE=/usr/src/googletest
export YOLO_DEFECT_RUN_DIR="$PWD/cpp_infer/results/s2_02/linux_x86_64/rerun_20260828"

bash cpp_infer/tools/stage1.sh doctor
bash cpp_infer/tools/stage1.sh clean-build
bash cpp_infer/tools/stage1.sh test
bash cpp_infer/tools/stage1.sh demo
bash cpp_infer/tools/stage1.sh consistency
bash cpp_infer/tools/stage1.sh benchmark --warmup 1 --repeat 2
```

core-only smoke：

```bash
cmake -S cpp_infer -B /tmp/yolo_defect_s2_02_core_only -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DYOLO_DEFECT_CORE_ONLY=ON
cmake --build /tmp/yolo_defect_s2_02_core_only --parallel
ctest --test-dir /tmp/yolo_defect_s2_02_core_only --output-on-failure
```

操作边界：

- 这是 WSL2/Linux x86_64 结果，不是 bare-metal Linux、ARM64 板卡或 Jetson 结果。
- 非交互式 WSL 命令中的 `sudo` 可能无法取得密码提示；依赖安装应在交互式 shell 中完成或先执行 `sudo -v`。
- `/tmp` 中的 build 与默认 fresh run 结果可能随 WSL 会话或系统清理消失；收口运行通过 `YOLO_DEFECT_RUN_DIR` 把原始 JSON 直接写入上述仓库目录。
- Gate A 本身只产生 WSL2/Linux x86_64 证据；后续 Gate B 已完成 AArch64 cross-build、QEMU correctness smoke 和固定图完整推理，详见 [Gate B 阶段记录](s2_02_gate_b_closure.md)与[S2-02 总收口](s2_02_closure.md)。两者都没有发布 QEMU 性能数字。
