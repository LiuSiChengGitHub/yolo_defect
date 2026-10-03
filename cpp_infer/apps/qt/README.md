# Qt 检测客户端

Qt 6 Widgets 客户端支持单张图片、图片目录和 manifest 清单检测：选择 RuntimeConfig 与输入，在后台执行检测和写出，查看检测框、类别、置信度、批次汇总及逐图失败原因。客户端直接调用 `yolo_defect::runtime`，与 CLI 共用配置、推理、批处理调度和 writer。

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

不想使用命令行时，双击仓库根目录的 `start_qt_workbench.cmd`：客户端尚未构建时它先调用一次 `qt.cmd build`，随后执行 `qt.cmd run`；启动失败时窗口暂停并保留错误信息。它只在可执行文件缺失时构建，修改源码后仍需手动 `build`。成品包则直接双击包内 `run.cmd`。

路径优先级为：命令参数 → Qt 本地配置 → Stage-1 本地配置 → 环境变量 → 默认值。配置文件内的相对路径基于该文件所在目录，显式命令参数中的相对路径基于调用目录；默认路径基于脚本位置，因此从其他工作目录调用也可用。`-QtRoot`、`-BuildDir` 等参数及环境变量见 `help`。更换 Qt SDK、编译器或生成器时使用新的构建目录，避免复用不兼容缓存。

脚本只设置当前子进程的 DLL 和插件搜索路径，不修改系统环境变量。ONNX Runtime DLL 由已有 CMake 逻辑复制到 `bin`，Qt 和 OpenCV 从已安装 SDK 加载。这是开发入口；独立演示目录使用下述 `package` 入口。

`YOLO_DEFECT_BUILD_QT` 默认 `OFF`；原 CLI 构建命令无须更改。`YOLO_DEFECT_CORE_ONLY=ON` 会在查找 Qt 和其他 Runtime 依赖前返回，即使同时指定 Qt 开关也仅构建 project-core。

## 操作

1. 选择 `cpp_infer/configs/default_config.txt`（FP32）或 `cpp_infer/configs/int8_u8s8_config.txt`（正式 INT8）。模型文件沿用相应 artifact 声明，取得模型的方式见 [C++ Runtime 说明](../../README.md)。
2. 在分段切换中选择输入方式：“单张图片”“图片目录”或“Manifest”（清单），再浏览对应文件或目录。单图示例为 `data/images/val/crazing_241.jpg`；目录示例为 `data/images/val/`。manifest 使用 UTF-8 文本，每行一个图片路径，相对路径以清单文件所在目录为基准，规则与 [Runtime 批处理](../../README.md#目录manifest-与有界并发) 一致。
3. 选择输出目录。目录批处理的输出必须位于输入目录之外；默认的 `results/qt/` 可用于上述数据目录。批处理可设置“并发数量”和“队列容量”，默认分别为 1 和 2，范围分别为 1–64 和 1–4096；两项独立设置。每个 Runtime worker 持有独立模型 session。
4. 点击“运行检测”或“开始批处理”。配置读取、模型构建、推理和写出均在后台执行；成功读取配置后，左侧显示本次实际生效的模型和参数。运行期间显示忙碌状态，批次结束后显示总数、成功、失败及取消计数，不估算实时百分比。
5. 批处理完成后，在“逐图结果”中选择图片，查看原图、标注图及“检测明细”。列表包含输入序号、图片、状态、目标数、耗时和说明；失败或取消项显示原因，其他成功项仍可浏览。目录结果按 Runtime 的相对路径排序，manifest 结果按声明顺序排列，不按完成先后重新排序。
6. 通过“打开 JSON”“打开批次汇总”和“打开结果目录”查看生成文件。切换配置后再次运行即可比较 FP32/INT8。

图像支持滚轮缩放、左键拖动平移、工具栏放大/缩小、适应窗口和 `1:1` 查看；双击恢复适应窗口。工具栏在没有图像时自动禁用。选择检测明细行会在两张图中高亮对应框，点击图中的检测框也会选中并滚动到对应明细行。缩放和高亮只影响当前视图，不修改保存的标注 PNG 或检测坐标。结果区把批次汇总和检出数放在标签页同一行，下方为列表和文件操作栏，将更多高度留给列表，支持按像素平滑滚动。图像区与结果区之间的分隔线可拖动调整；最小窗口仍保留多行表格数据，长失败原因在独立只读区域中滚动查看。

每次运行创建独立结果子目录。单图保存 `detections.json` 和 `detections.png`；批处理保存 `batch_summary.json`，成功项由 Runtime 写入 `items/<六位序号>.detections.json` 和对应的 `.visualized.png`。损坏图片作为逐图失败记录，不中断其他图片；配置、输入发现或输出预检失败则显示整批错误。

运行中禁止重复启动和修改输入。批处理点击“停止”会请求协作停止：未开始的图片取消，已经执行的图片允许完成，随后写出最终汇总并恢复启动按钮。运行中关闭窗口也会发出停止请求，待后台任务和预览线程收尾后自动关闭，期间事件循环仍响应。单图运行中关闭则等待该图片处理和写出完成。界面展示的任务总耗时包含配置加载、session 构建和写出，不替代 Runtime benchmark 或批处理 processing wall time。

修改配置、输入方式、路径或批处理参数会清除上一次显示结果。逐图浏览按需在后台读取已生成的 JSON/PNG 和原图，不再次执行推理；快速切换时只保留最新待加载项，旧请求不会覆盖当前选择。结果文件被移走或损坏时，显示预览错误并清空对应图像和明细。

左侧核心参数按键值对齐显示，展开“类别与文件详情”可查看完整类别、模型和配置路径。长路径在未编辑时使用中间省略，聚焦后编辑全文，悬停可查看完整路径；“打开结果目录”按钮可悬停查看输出目录。界面固定使用深色主题，不随 Windows 浅色/深色系统设置变化；Windows 上标题栏同样为深色。

也可以预填输入后启动（仍由用户点击“运行检测”）：

```powershell
cpp_infer\tools\qt.cmd run `
  -Config cpp_infer\configs\int8_u8s8_config.txt `
  -Image data\images\val\crazing_241.jpg `
  -OutputDir results\qt
```

这个启动入口仍预填单图输入；目录和 manifest 模式在窗口中选择。脚本和 Qt 可执行程序没有增加批处理 CLI 参数。

## 后台适配与生命周期

`DetectionWorker` 在 Qt 后台线程调用 `load_runtime_contract → DetectorPipeline::run`；`BatchWorker` 在同样的线程边界调用 `load_runtime_contract → BatchRunner::run → write_batch_summary_json`。每批新建一个 Runner，图片调度、有限队列、worker/session 复用与线程 join 全部沿用 Runtime，Qt 不增加一套推理调度系统。

`BatchRunner::run()` 同步占用后台对象的事件循环，因此停止由 GUI 线程直接调用共享 `BatchTaskControl::requestStop()`，进而调用线程安全的 `BatchRunner::request_stop()`。控制对象只在发布 Runner 和读取停止状态时短暂持锁，运行过程不持锁；配置读取期间提前点击停止也会保留该请求，在 Runner 发布时补交。任务完成后再释放线程和控制对象，下一批使用新实例。

`PreviewWorker` 只负责后台读取所选成功项的原图、标注图和检测 JSON。客户端最多保留一个正在读取项和一个待读取项，通过 generation 判断返回值是否仍对应当前选择。跨线程数据为配置、结果结构体和拥有像素内存的 `QImage` 值；所有控件、视图变换及表格选择操作留在 GUI 线程。Qt 依赖留在 `apps/qt`，Runtime 源码和公共接口未改动。

## Qt 适配测试

按 [C++ Runtime 测试说明](../../README.md) 配置 Python 和本地 GoogleTest 源码，并安装含 Qt Test 的 SDK；正式 FP32/U8S8 模型也需要到位。测试入口主动启用 `BUILD_TESTING=ON`，构建客户端、Qt 测试和 CLI，然后只运行 Qt 集成测试：

```powershell
cpp_infer\tools\qt.cmd test
cpp_infer\tools\qt.cmd test -Screenshots
```

测试通过 `offscreen` 平台运行，并复用真实配置和输入比较 Qt 适配结果与 CLI。`-Screenshots` 将截图写到构建目录的 `screenshots/`；也可用 `-ScreenshotDir` 指定目录。普通客户端构建不要求测试依赖，`test` 缺少必需依赖或测试目标时会报错。

测试文本日志写入构建目录的 `qt_client_test.txt`。入口会检查测试确实已注册，且 QtTest 汇总为零失败、零跳过；缺少模型导致的跳过不会视为验收成功。脚本自动加载可用的 Windows 中文字体。直接运行测试程序时，可设置 `YOLO_DEFECT_QT_SCREENSHOT_DIR` 保存初始、完成和错误状态的离屏截图，并设置 `YOLO_DEFECT_QT_TEST_FONT=C:\Windows\Fonts\msyh.ttc`（Windows 离屏平台不自动枚举系统字体）。这些仅用于测试，不改变正常客户端的系统字体行为。

Windows 原生渲染检查可直接运行测试程序并设置 `QT_QPA_PLATFORM=windows`、`YOLO_DEFECT_QT_TEST_HIDDEN=1`，在不弹出窗口的情况下保存 Qt 控件渲染图。可选截图还覆盖展开详情和最小窗口布局；本机已检查原生 200% 缩放，正常检测的模型、阈值和结果仍沿用上述集成测试验证。

第二步增加以下验收覆盖：

- 真实目录和 manifest 输入混合有效图片与损坏图片；比较 Qt 与 CLI 的输入顺序、计数、逐图状态、稳定汇总字段及成功项完整检测 JSON。耗时、进程信息和各次独立输出目录等运行相关字段不作逐字相等比较。
- 运行期间事件循环响应、输入锁定和重复启动保护；成功项切换、失败原因、图像与表格内容对应，以及快速选择后显示最后一项。
- 提前停止、实际图片处理期间停止、停止后重新启动，以及批处理中关闭窗口的协作收尾。
- 缩放、`1:1`、适应窗口和检测框与明细的双向选择；可选截图覆盖批处理成功、失败、取消和最小窗口布局。

Windows 现有 CLI 的批处理命令参数记录仍使用本机代码页，中文 argv 可能在汇总的 UTF-8 校验处失败。因此 CLI 对照使用 ASCII（含空格）的入口参数，同时保留中文图片名、子目录和 GUI 输出路径；GUI 中文配置及输入路径另由单图恢复测试覆盖。本轮没有修改 CLI 的参数编码逻辑。

## 文件组织与样式维护

| 位置 | 用途 | 提交 Git |
|---|---|---|
| `cpp_infer/apps/qt/` | Qt 客户端源码、独立 CMake target 与本说明 | 是 |
| `cpp_infer/tests/qt_client_test.cpp` | 单图/批处理与 CLI 一致性、响应、停止、浏览交互及生命周期测试 | 是 |
| `cpp_infer/tools/qt.cmd`、`qt.ps1`、`qt.local.example.psd1` | 可复用开发入口及机器配置示例 | 是 |
| `cpp_infer/tools/qt_package.ps1`、`qt_package/` | 可移动演示目录生成器与启动/验证模板 | 是 |
| `cpp_infer/tools/qt_demo_capture.cpp`、`qt_demo_media.py`、`requirements-qt-media.txt` | 真实界面捕获、媒体编码与可选依赖 | 是 |
| `docs/demo/`、`docs/assets/qt/`、`.github/workflows/qt-demo-pages.yml` | HTML 指南、精选媒体与手动发布配置 | 是 |
| `cpp_infer/.qt.local.psd1`、`.stage1.local.psd1` | 当前机器的依赖路径 | 否 |
| `cpp_infer/build/qt-msvc-release/` | CMake 缓存、二进制、测试日志与可再生成截图 | 否 |
| `dist/yolo-defect-qt/` | 可重新生成的 Windows 成品目录及验证输出 | 否 |
| `docs/me/Qt.md` | 用户教材和 AI 交接补充，个人资料 | 否 |
| `results/qt/` | 每次单图/批处理生成的 JSON/PNG 与批次汇总 | 否 |
| `tmp/qt_plan/` | 本地设计讨论草稿，无运行依赖 | 否 |

开发入口不依赖 `tmp/` 中的脚本、SDK 或构建缓存；Qt SDK 使用仓库外的正式安装目录。正式演示素材保存在 `docs/assets/qt/`，测试截图和原始捕获帧留在构建目录，不直接全部入库。

客户端源码按职责组织：

| 组件 | 职责 |
|---|---|
| `MainWindow` | 控件布局、输入与运行状态、生命周期、逐图和检测项选择联动 |
| `ModelInfoPanel`、`PathEdit` | 生效参数及可展开文件详情、长路径显示与编辑 |
| `SegmentedControl` | 输入方式分段切换，提供与 `QComboBox` 相同的 `currentIndex` 接口 |
| `ImageView` | 图像绘制、缩放和平移、检测框命中与选择高亮 |
| `DetectionTableModel`、`BatchTableModel` | 分别将检测数组、Runtime 批处理逐图结果映射到标准表格 |
| `StatusPillDelegate`、`ConfidenceBarDelegate` | 逐图状态胶囊、置信度条的单元格绘制，不改变模型的显示文本 |
| `theme`、`resources/` | 统一色板、带色板变量的 QSS、SVG 图标与 `.qrc` 资源 |
| `DetectionWorker`、`BatchWorker` | 单图/批处理 Runtime 调用与信号适配 |
| `BatchTaskControl` | 跨 GUI 与工作线程共享的协作停止状态和 Runner 生命周期引用 |
| `PreviewWorker` | 已有输出的按需异步读取，不执行推理 |
| `detection_types.h`、`batch_types.h` | Qt 跨线程请求和返回值 |
| `task_io` | 单图和批处理共用的图像读取、独立结果目录创建 |

配色和图标修改不需要改推理、批处理调度或结果协议。

界面采用暖调暗色风格：标题栏、侧栏面板、双画布查看面板、结果面板和底部状态栏。暖灰底色分为画布、窗口、面板、输入框四级明度，区域靠底色与留白区分，几乎不画描边；陶土橙只用于主按钮、输入焦点和选中行，状态徽标为灰底加彩色小圆点，成功、失败、取消在表格中用低饱和的状态胶囊表示；画布为纯色，不画网格。数值、路径和模型 ID 使用等宽字体。客户端统一使用 Qt Fusion 样式，绘制只取决于色板和样式表，避免 Windows 原生样式在表格单元格上自行绘制而破坏暗色效果；Windows 上通过 DWM 属性把原生标题栏设为与窗口相同的深色（需链接 `dwmapi`，旧系统忽略不支持的属性）。字体使用系统自带的 Segoe UI Variable、Cascadia Mono 与 Microsoft YaHei UI，缺失时依次回退，不随包分发字体文件。

样式集中在客户端内部：`theme.cpp` 定义唯一的深色 `Palette`，`resources/workbench.qss` 用 `@accent`、`@panel` 等色板变量书写，加载时由 `theme::styleSheet()` 替换为实际颜色；画布、检测框高亮、表格状态色和委托绘制直接读取同一份 `Palette`。图标是 `resources/icons/` 下的单色 SVG，经 `qt_client.qrc` 编入客户端静态库，`theme::icon()` 按状态在绘制时着色，因此同一个 SVG 可用于正常、禁用和深色背景，并在高 DPI 下保持清晰；QSS 中的步进箭头和分隔线手柄直接引用资源 URL。SVG 由 Qt 的 `qsvgicon`/`qsvg` 插件读取，`windeployqt` 默认随包收集。

调整配色只需修改 `Palette`；新增图标时放入 `resources/icons/` 并登记到 `.qrc`。当前只有一套深色主题，未实现运行时深浅切换；如需浅色主题，可增加第二套 `Palette` 并重新应用样式表。常规美化是局部 UI 工作，大幅改变操作流程或改用 QML 则是另一个范围的开发。

## 第一步验收记录（2026-09-27）

Windows x64 / MSVC 19.50 / Qt 6.8.3 / OpenCV 4.8.0 / ORT 1.19.2，Release 构建通过。

- Qt 集成测试通过，无失败或跳过：FP32 与正式 U8S8 的完整 JSON 分别与同配置、同输入的 CLI 输出相同；表格类别、置信度与坐标对应 JSON。
- QtTest 检查运行期间事件循环仍响应、重复启动被阻止、Unicode/空格路径、损坏图片错误后恢复、同一窗口切换到 U8S8 后重新运行，以及运行中关闭窗口的异步收尾。初始、完成、错误状态已做离屏渲染检查。
- `YOLO_DEFECT_BUILD_QT=OFF` 的原 Runtime 默认构建成功，相关 `contract|output|metadata|project_core` 回归 **54/54** 通过；`CORE_ONLY=ON` 与 Qt 开关同时开启时仍跳过 Qt，core smoke **1/1** 通过。
- 开发入口整理后，从 `docs/` 工作目录调用 `qt.cmd test -Screenshots`，在新的 `cpp_infer/build/qt-msvc-release/` 完成正式 SDK 配置、构建与 QtTest **6 passed / 0 failed / 0 skipped**。清理旧临时 SDK/构建后，`qt.cmd run` 的原生窗口启动、响应、正常关闭及成功退出码均已验证。

以上为第一步及开发入口整理时的历史验收记录。

## 第二步验收记录（2026-09-27）

沿用正式 Qt 6.8.3 / MSVC x64 Release 构建：

- `qt.cmd test -Screenshots`：QtTest **11 passed / 0 failed / 0 skipped**，包含第一步单图 FP32/U8S8 回归及第二步目录/manifest、损坏图片、停止、关闭与交互检查。日志：`cpp_infer/build/qt-msvc-release/qt_client_test.txt`。
- `QT_QPA_PLATFORM=windows`、`YOLO_DEFECT_QT_TEST_HIDDEN=1`：批处理浏览与停止用例 **7 passed / 0 failed / 0 skipped**，原生屏幕 DPR 为 2。验证真实鼠标点击、滚轮、拖动与最小窗口布局。日志：构建目录的 `qt_native_test.txt`。
- 已检查原生成功、失败、取消和最小窗口截图，保存在构建目录 `native-visual/`；长错误文本可滚动，选框标签限制在可见区域。离屏截图位于 `screenshots/`。
- 结果区紧凑布局调整后，上述离屏与原生验收通过。在 200% 缩放、1280×730 逻辑窗口下，逐图列表可显示 5 行完整数据（表体 152 px、行高 28 px）；980×700 窗口可显示 4 行完整数据。最小窗口的失败状态另经原生截图检查，窗口保持 980×700，错误区与底部操作正常显示。截图：`batch_browse_compact.png`、`batch_browse_minimum.png`、`batch_failure_minimum.png`；补充日志：`qt_layout_test.txt`。
- Runtime 源码、公共接口、调度和输出协议未改动；共享 CMake 仅更新 Qt 开关的说明文字。本轮未重复第一步已完成的无 Qt 平台回归。

## Windows 演示交付与展示素材

演示目录地图、启动与操作顺序集中在 [中文 HTML 演示页](../../../docs/demo/index.html) / [English](../../../docs/demo/index.en.html)，同时展示 Runtime 架构、量化、并发与跨平台证据。克隆仓库后直接在浏览器打开；GitHub 页面默认显示 HTML 源码，在线浏览需要[手动发布 Pages](../../../docs/demo/README.md#发布-github-pages)。

```powershell
# 在已配置 SDK 的开发环境打包，无需 Python / GoogleTest
cpp_infer\tools\qt.cmd package
# 默认 dist/yolo-defect-qt；目标必须是新目录或空目录
cpp_infer\tools\qt.cmd package -PackageDir dist/yolo-defect-qt-new
# 对成品包运行依赖和真实检测检查
powershell -NoProfile -ExecutionPolicy Bypass -File dist/yolo-defect-qt/verify.ps1
```

`package` 构建 Qt 和 CLI Release，用 `windeployqt` 收集 Qt，再收集 OpenCV/ORT 的实际 DLL 依赖及 app-local MSVC CRT。成品包含 `run.cmd`、`verify.ps1`、相对路径配置/模型声明、FP32 模型、可用时的 U8S8 模型、六类各一张样例、manifest、HTML/媒体及原发行版许可材料。模型/样例来源和 INT8 生成方式见 [模型与样例](../../README.md#模型与样例获取)。

打包不覆盖非空目录，不运行 VC 安装器，不修改系统环境；成品和验证输出由 Git 忽略。`verify.ps1` 将 PATH 限制为包目录和 Windows 系统目录，运行 FP32/可选 U8S8 单图、六图 manifest 批处理及真实 Windows QPA 初始化。它是本机依赖隔离检查，不等于干净 Windows 虚拟机验收，也不替代 Qt 集成测试。

### 可重复更新截图与 GIF

```powershell
# 需要 Qt Test、既有测试依赖，以及 Python Pillow
python -m pip install -r cpp_infer/tools/requirements-qt-media.txt
cpp_infer\tools\qt.cmd media
# 已有原始帧时只重新编码
python cpp_infer/tools/qt_demo_media.py --frames cpp_infer/build/qt-msvc-release/demo-frames
python cpp_infer/tools/render_demo.py
```

可选 target `yolo_defect_qt_capture` 通过真实控件与 Runtime 完成六图全成功检测、选择缩放、逐图浏览和恢复总览，使用独立 QSettings；它不属于 CTest，也不进入交付目录。捕获程序逐项检查成功状态，编码器验证六帧清单和成功汇总。`qt_demo_capture.cpp` 保存原始截图和帧时长清单到构建目录，`qt_demo_media.py` 生成 `docs/assets/qt/` 的 PNG/WebP/GIF。11.8 秒 GIF 是实际状态序列，停留时间经过编排，不能用来推断推理性能。

HTML、根 README 和成品包共用稳定素材路径。`qt.cmd media` 在媒体生成后自动重新生成两种语言的 HTML。颜色/字体/图标变化后重新运行 `media`、检查画面、在新目录 `package` 即可；控件对象名或工作流程变化时同步维护捕获程序。两份 README 顶部直接展示 GIF，中间用操作流程文字与截图分隔。可选连续录屏及视频替换方法见 HTML。`docs/me/Qt.md` 为本地学习/AI 交接补充，沿用个人笔记的 Git 忽略规则。

中英文 HTML 由 `docs/demo/page.template.html`、`content.json`、`evidence.json` 和同一生成器维护，不直接修改生成页。工程图由已有正式报告生成，不重新运行性能实验：

```powershell
# 只在图表或正式报告更新时需要 Matplotlib；UI 媒体更新只需要 Pillow
python -m pip install -r cpp_infer/tools/requirements-demo-charts.txt
python cpp_infer/tools/demo_charts.py
python cpp_infer/tools/render_demo.py
python cpp_infer/tools/render_demo.py --check
```

量化图包含模型大小、pipeline 延迟和逐 IoU 阈值 AP；并发图同时展示吞吐收益与内存成本。`evidence.json` 记录来源、测量条件和平台验证范围；AArch64/QEMU 作为功能验证展示。完整目录职责、双语更新与发布步骤见[演示维护说明](../../../docs/demo/README.md)。

`.github/workflows/qt-demo-pages.yml` 仅通过 `workflow_dispatch` 手动发布中英文 HTML、Qt 媒体与工程 SVG；发布前检查双语生成页是否与源文件一致。不会发布整个 `docs/`、模型或私人教材；仓库 Pages 的 Source 需先设为 GitHub Actions。本轮准备配置和素材，不执行远端发布。

## 第三步验收记录（2026-09-27）

| 检查 | 本轮结果与证据 |
|---|---|
| Qt 最终回归 | `qt.cmd test -Screenshots`：QtTest **11 passed / 0 failed / 0 skipped**。日志 `build/qt-msvc-release/qt_client_test.txt`；涵盖 GUI/CLI 一致性与任务生命周期。QtTest 计数包含初始化/清理，不等同 11 个独立业务场景。 |
| 原 Runtime/CLI | 在新目录 `build/qt-disabled-verification/` 显式 `YOLO_DEFECT_BUILD_QT=OFF` 配置、构建，并运行全部 CTest：**155 通过、2 跳过、0 失败**。跳过两项目录符号链接测试，原因是本机缺少创建符号链接权限。日志在该构建的 `Testing/Temporary/LastTest.log`。 |
| 媒体与真实交互 | `qt.cmd media` 成功捕获 6 个真实状态（Windows QPA、DPR 2），生成 PNG、WebP 和 **11.8 秒 GIF**；已检查实际客户端截图。原始帧及清单在 `build/qt-msvc-release/demo-frames/`。 |
| Windows 成品包 | `dist/yolo-defect-qt/` 包含 app-local CRT、Qt/OpenCV/ORT、FP32/U8S8、六张样例与完整 HTML 媒体。`verify.ps1` 在限制 PATH 后完成两模型单图、六图 manifest 与 Qt Windows QPA 启动；结果位于包内 `outputs/verify_*/`。没有在干净虚拟机运行，Qt 启动探针也不代替 GUI 集成测试。 |
| 文档与网页 | 中英文 README、本地链接、HTML 资源及页内锚点、JavaScript 语法检查通过。浏览器工具阻止 `file://`，本轮未进行 HTML 浏览器视觉验收；请本地打开 `docs/demo/index.html` 确认。Pages 工作流已准备，未推送或部署。 |

本轮仅扩展 Qt 的可选展示目标和开发工具，没有修改 Runtime 源码、输出协议或跨平台实现；未重复无关性能实验与 AArch64/QEMU 验证。

后续展示更新（2026-09-27）：重新运行 `qt.cmd media`，六张图片逐项成功，GIF 解码为 6 帧 / 11.8 秒。新增中英文共用模板与正式报告图表，双语生成同步、报告来源、资源/锚点、JavaScript、PowerShell 和 Pages YAML 静态检查通过；两份 README 的 GIF 均直接展开。已有 `dist/yolo-defect-qt/` 的 HTML、媒体和图表已同步，逐文件核对与源码素材一致。本次未改业务 Runtime，也未重复整套推理回归；HTML 尚未做浏览器目视验收，Pages 尚未发布。

## 界面改版验收记录（2026-10-03）

| 检查 | 本轮结果与证据 |
|---|---|
| Qt 回归 | `qt.cmd test`：QtTest **11 passed / 0 failed / 0 skipped**；原生 Windows QPA（DPR 2、隐藏窗口）运行单图、目录批处理、停止与 Unicode 恢复用例 **6 passed / 0 failed / 0 skipped**，含真实点击、滚轮与拖动。980×700 最小窗口仍显示多行列表数据。 |
| 媒体与网页 | `qt.cmd media` 重新捕获 6 个真实状态（六图全部成功）并生成 PNG、WebP 与 GIF；工程图表按新色板由 `demo_charts.py` 重新生成，`--check` 确认证据与已提交报告一致；`render_demo.py --check` 双语同步。两种语言页面已用无头 Edge 渲染检查。 |
| Windows 成品包 | 新目录 `dist/yolo-defect-qt-20261003/`，包含 `qsvgicon` / `qsvg` 插件；包内媒体和 HTML 与 `docs/` 一致。`verify.ps1` 通过：两个模型单图、六图 manifest 批处理及原生 Qt 启动。 |

本轮只改 Qt 界面层、展示页样式与图表配色；Runtime 源码、输出协议和检测结果未改动。测试仅把 `inputMode` 的查找方式改为读取 `currentIndex` 属性，断言不变。
