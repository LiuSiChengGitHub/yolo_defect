# Qt 单图检测客户端

第一步实现 Qt 6 Widgets 单图闭环：选择 RuntimeConfig 和图片，在后台加载模型、检测和写出结果，查看检测框、类别与置信度，并打开 JSON/PNG 输出。客户端直接调用 `yolo_defect::runtime`，与 CLI 共用配置、推理和 writer。

## Windows x64 构建与启动

需要 Qt 6.2 或以上的 **MSVC x64** SDK，以及原 Runtime 使用的 MSVC、OpenCV 4 和 ONNX Runtime 1.19.2。Qt 的 MinGW SDK 不适用于这套 MSVC 依赖。

日常入口为 [`tools/qt.cmd`](../../tools/qt.cmd)，可以从普通 PowerShell 或 CMD 调用，构建时自动发现并初始化 x64 MSVC。首次使用时，将 [`qt.local.example.psd1`](../../tools/qt.local.example.psd1) 复制为 `cpp_infer/.qt.local.psd1`，填写正式安装的 `QtRoot`（例如 `D:\SDK\Qt\6.8.3\msvc2022_64`）。

ORT、OpenCV、Python 和 GoogleTest 路径复用现有 `cpp_infer/.stage1.local.psd1`；尚未配置的机器参考 [`stage1.local.example.psd1`](../../tools/stage1.local.example.psd1)。也可在 Qt 本地配置中覆盖这些路径。两个本地配置文件均由 Git 忽略，仓库提交的示例只包含占位路径。

在仓库根目录执行：

```powershell
cpp_infer\tools\qt.cmd help
cpp_infer\tools\qt.cmd build
cpp_infer\tools\qt.cmd run
```

默认使用 `cpp_infer/build/qt-msvc-release/`，与 CLI 构建分开。`configure` 仅配置，`build` 配置并增量构建客户端，`test` 启用并运行 Qt 集成测试，`run` 启动已构建程序并预填默认配置、样图和 `results/qt/`。`run` 无须 MSVC、Python 或 GoogleTest；第一次使用或修改源码后先运行 `build` 或 `test`。

路径优先级为：命令参数 → Qt 本地配置 → Stage-1 本地配置 → 环境变量 → 默认值。配置文件内的相对路径基于该文件所在目录，显式命令参数中的相对路径基于调用目录；默认路径基于脚本位置，因此从其他工作目录调用也可用。`-QtRoot`、`-BuildDir` 等参数及环境变量见 `help`。更换 Qt SDK、编译器或生成器时使用新的构建目录，避免复用不兼容缓存。

脚本只设置当前子进程的 DLL 和插件搜索路径，不修改系统环境变量。ONNX Runtime DLL 由已有 CMake 逻辑复制到 `bin`，Qt 和 OpenCV 从已安装 SDK 加载。这是开发入口，演示目录打包属于计划第三步。

`YOLO_DEFECT_BUILD_QT` 默认 `OFF`；原 CLI 构建命令无须更改。`YOLO_DEFECT_CORE_ONLY=ON` 会在查找 Qt 和其他 Runtime 依赖前返回，即使同时指定 Qt 开关也仅构建 project-core。

## 操作

1. 选择 `cpp_infer/configs/default_config.txt`（FP32）或 `cpp_infer/configs/int8_u8s8_config.txt`（正式 INT8）。模型文件沿用相应 artifact 声明，取得模型的方式见 [C++ Runtime 说明](../../README.md)。
2. 选择图片，例如 `data/images/val/crazing_241.jpg`，再选择输出目录。
3. 开始检测。配置读取、模型构建、推理及写出均在工作线程执行，状态区显示运行、成功或错误；完成后显示本次实际生效的模型和参数。
4. 查看自动适应窗口的原图和标注图，以及类别、置信度和原图坐标表格；打开 JSON 或输出目录查看 Runtime 生成的文件。切换配置后再次运行即可比较 FP32/INT8。

当前仅包含单图功能。输出命名、失败提示以及运行时关闭窗口的收尾由客户端负责，推理结果和写出格式由 Runtime 负责。

每次运行创建独立结果子目录，保存 `detections.json` 和 `detections.png`。修改配置、图片或输出目录会清除上一次结果，实际参数在后台成功读取配置后显示。运行中禁止重复启动；此时关闭窗口会等待本次写出完成后自动关闭，窗口仍能响应。界面展示的任务总耗时包含加载和写出，不替代 Runtime benchmark 数据。

左侧核心参数按键值对齐显示，展开“类别与文件详情”可查看完整类别、模型和配置路径。长路径在未编辑时使用中间省略，聚焦后编辑全文，悬停可查看完整路径；结果保存提示也可悬停查看输出目录。滚动区固定使用与工作台一致的浅色背景，适配 Windows 深色系统主题。

也可以预填输入后启动（仍由用户点击“运行检测”）：

```powershell
cpp_infer\tools\qt.cmd run `
  -Config cpp_infer\configs\int8_u8s8_config.txt `
  -Image data\images\val\crazing_241.jpg `
  -OutputDir results\qt
```

`MainWindow` 负责控件与状态，`DetectionWorker` 通过 `moveToThread` 在后台调用 `load_runtime_contract → DetectorPipeline::run`。跨线程传递配置、检测结果和拥有像素内存的 `QImage` 值；`DetectionTableModel` 只将 Runtime 检测数组映射到表格。Qt 依赖留在 `apps/qt`，原 Runtime 公共接口未改动。批处理、缩放和结果选择联动留给第二步。

## Qt 适配测试

按 [C++ Runtime 测试说明](../../README.md) 配置 Python 和本地 GoogleTest 源码，并安装含 Qt Test 的 SDK；正式 FP32/U8S8 模型也需要到位。测试入口主动启用 `BUILD_TESTING=ON`，构建客户端、Qt 测试和 CLI，然后只运行 Qt 集成测试：

```powershell
cpp_infer\tools\qt.cmd test
cpp_infer\tools\qt.cmd test -Screenshots
```

测试通过 `offscreen` 平台运行，并复用真实配置和输入比较 Qt 适配结果与 CLI。`-Screenshots` 将截图写到构建目录的 `screenshots/`；也可用 `-ScreenshotDir` 指定目录。普通客户端构建不要求测试依赖，`test` 缺少必需依赖或测试目标时会报错。

测试文本日志写入构建目录的 `qt_client_test.txt`。入口会检查测试确实已注册，且 QtTest 汇总为零失败、零跳过；缺少模型导致的跳过不会视为验收成功。脚本自动加载可用的 Windows 中文字体。直接运行测试程序时，可设置 `YOLO_DEFECT_QT_SCREENSHOT_DIR` 保存初始、完成和错误状态的离屏截图，并设置 `YOLO_DEFECT_QT_TEST_FONT=C:\Windows\Fonts\msyh.ttc`（Windows 离屏平台不自动枚举系统字体）。这些仅用于测试，不改变正常客户端的系统字体行为。

Windows 原生渲染检查可直接运行测试程序并设置 `QT_QPA_PLATFORM=windows`、`YOLO_DEFECT_QT_TEST_HIDDEN=1`，在不弹出窗口的情况下保存 Qt 控件渲染图。可选截图还覆盖展开详情和最小窗口布局；本机已检查原生 200% 缩放，正常检测的模型、阈值和结果仍沿用上述集成测试验证。

## 文件组织与样式维护

| 位置 | 用途 | 提交 Git |
|---|---|---|
| `cpp_infer/apps/qt/` | Qt 客户端源码、独立 CMake target 与本说明 | 是 |
| `cpp_infer/tests/qt_client_test.cpp` | Qt 与 CLI 一致性、响应及生命周期测试 | 是 |
| `cpp_infer/tools/qt.cmd`、`qt.ps1`、`qt.local.example.psd1` | 可复用开发入口及机器配置示例 | 是 |
| `cpp_infer/.qt.local.psd1`、`.stage1.local.psd1` | 当前机器的依赖路径 | 否 |
| `cpp_infer/build/qt-msvc-release/` | CMake 缓存、二进制、测试日志与可再生成截图 | 否 |
| `results/qt/` | 每次检测生成的 JSON/PNG | 否 |
| `tmp/qt_plan/` | 本地设计讨论草稿，无运行依赖 | 否 |

开发入口不依赖 `tmp/` 中的脚本、SDK 或构建缓存；Qt SDK 使用仓库外的正式安装目录。展示用截图到第三步再挑选并存入正式文档资源目录，测试截图不直接全部入库。

界面组件已按职责拆分：`MainWindow` 组织布局和任务状态，`ModelInfoPanel` 展示配置，`PathEdit` 处理长路径，`ImageView` 绘制预览，`DetectionTableModel` 映射结果，`DetectionWorker` 适配后台 Runtime。配色和图标修改不需要改推理、批处理调度或结果协议。

当前仍是单套样式：QSS 位于 `main_window.cpp`，画布颜色位于 `image_view.cpp`，尚未实现动态主题切换。后续需要深浅主题或图标时，将 QSS、绘制颜色和 `.qrc` 资源集中到客户端即可；常规美化是局部 UI 工作，大幅改变操作流程或改用 QML 则是另一个范围的开发。

## 第一步验收记录（2026-09-27）

Windows x64 / MSVC 19.50 / Qt 6.8.3 / OpenCV 4.8.0 / ORT 1.19.2，Release 构建通过。

- Qt 集成测试通过，无失败或跳过：FP32 与正式 U8S8 的完整 JSON 分别与同配置、同输入的 CLI 输出相同；表格类别、置信度与坐标对应 JSON。
- QtTest 检查运行期间事件循环仍响应、重复启动被阻止、Unicode/空格路径、损坏图片错误后恢复、同一窗口切换到 U8S8 后重新运行，以及运行中关闭窗口的异步收尾。初始、完成、错误状态已做离屏渲染检查。
- `YOLO_DEFECT_BUILD_QT=OFF` 的原 Runtime 默认构建成功，相关 `contract|output|metadata|project_core` 回归 **54/54** 通过；`CORE_ONLY=ON` 与 Qt 开关同时开启时仍跳过 Qt，core smoke **1/1** 通过。
- 开发入口整理后，从 `docs/` 工作目录调用 `qt.cmd test -Screenshots`，在新的 `cpp_infer/build/qt-msvc-release/` 完成正式 SDK 配置、构建与 QtTest **6 passed / 0 failed / 0 skipped**。清理旧临时 SDK/构建后，`qt.cmd run` 的原生窗口启动、响应、正常关闭及成功退出码均已验证。

当前未改动 Runtime 源码和公共接口。Windows 演示目录打包、交互缩放、批处理及 GitHub 展示材料按原计划留待后两步。
