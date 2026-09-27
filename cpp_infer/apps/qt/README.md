# Qt 单图检测客户端

第一步实现 Qt 6 Widgets 单图闭环：选择 RuntimeConfig 和图片，在后台加载模型、检测和写出结果，查看检测框、类别与置信度，并打开 JSON/PNG 输出。客户端直接调用 `yolo_defect::runtime`，与 CLI 共用配置、推理和 writer。

## Windows x64 构建与启动

需要 Qt 6.2 或以上的 **MSVC x64** SDK，以及原 Runtime 使用的 MSVC、OpenCV 4 和 ONNX Runtime 1.19.2。Qt 的 MinGW SDK 不适用于这套 MSVC 依赖。

在仓库根目录打开 **Developer PowerShell for VS（x64）**，将示例路径改为本机路径后执行：

```powershell
$qtRoot = 'D:\01_Base\Tools\Qt\6.8.3\msvc2022_64'
$ortRoot = 'D:\01_Base\Tools\onnxruntime-win-x64-1.19.2'
$openCvDir = 'D:\01_Base\Tools\opencv\build\x64\vc16\lib'
$openCvBin = 'D:\01_Base\Tools\opencv\build\x64\vc16\bin'
$qtBuildDir = Join-Path $env:TEMP 'yolo_defect_qt_release'

cmake -S cpp_infer -B $qtBuildDir -G 'NMake Makefiles' `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF `
  -DYOLO_DEFECT_BUILD_QT=ON "-DCMAKE_PREFIX_PATH=$qtRoot" `
  "-DONNXRUNTIME_ROOT=$ortRoot" "-DOpenCV_DIR=$openCvDir"
cmake --build $qtBuildDir --target yolo_defect_qt

$env:PATH = "$qtRoot\bin;$openCvBin;$env:PATH"
$env:QT_PLUGIN_PATH = "$qtRoot\plugins"
& "$qtBuildDir\bin\yolo_defect_qt.exe"
```

这是开发环境运行方式；ONNX Runtime DLL 由已有 CMake 逻辑复制到 `bin`。Qt 和 OpenCV 从上述 SDK 路径加载，演示目录打包属于计划第三步。

`YOLO_DEFECT_BUILD_QT` 默认 `OFF`；原 CLI 构建命令无须更改。`YOLO_DEFECT_CORE_ONLY=ON` 会在查找 Qt 和其他 Runtime 依赖前返回，即使同时指定 Qt 开关也仅构建 project-core。

## 操作

1. 选择 `cpp_infer/configs/default_config.txt`（FP32）或 `cpp_infer/configs/int8_u8s8_config.txt`（正式 INT8）。模型文件沿用相应 artifact 声明，取得模型的方式见 [C++ Runtime 说明](../../README.md)。
2. 选择图片，例如 `data/images/val/crazing_241.jpg`，再选择输出目录。
3. 开始检测。配置读取、模型构建、推理及写出均在工作线程执行，状态区显示运行、成功或错误；完成后显示本次实际生效的模型和参数。
4. 查看自动适应窗口的原图和标注图，以及类别、置信度和原图坐标表格；打开 JSON 或输出目录查看 Runtime 生成的文件。切换配置后再次运行即可比较 FP32/INT8。

当前仅包含单图功能。输出命名、失败提示以及运行时关闭窗口的收尾由客户端负责，推理结果和写出格式由 Runtime 负责。

每次运行创建独立结果子目录，保存 `detections.json` 和 `detections.png`。修改配置、图片或输出目录会清除上一次结果，实际参数在后台成功读取配置后显示。运行中禁止重复启动；此时关闭窗口会等待本次写出完成后自动关闭，窗口仍能响应。界面展示的任务总耗时包含加载和写出，不替代 Runtime benchmark 数据。

也可以预填输入后启动（仍由用户点击“运行检测”）：

```powershell
& "$qtBuildDir\bin\yolo_defect_qt.exe" `
  --config "$PWD\cpp_infer\configs\default_config.txt" `
  --image "$PWD\data\images\val\crazing_241.jpg" `
  --output-dir "$PWD\results\qt"
```

`MainWindow` 负责控件与状态，`DetectionWorker` 通过 `moveToThread` 在后台调用 `load_runtime_contract → DetectorPipeline::run`。跨线程传递配置、检测结果和拥有像素内存的 `QImage` 值；`DetectionTableModel` 只将 Runtime 检测数组映射到表格。Qt 依赖留在 `apps/qt`，原 Runtime 公共接口未改动。批处理、缩放和结果选择联动留给第二步。

## Qt 适配测试

在上述构建配置上启用 `BUILD_TESTING=ON`，并按 [C++ Runtime 测试说明](../../README.md) 提供现有 Python 环境及 GoogleTest 源码（`Python3_EXECUTABLE`、`FETCHCONTENT_SOURCE_DIR_GOOGLETEST`）。SDK 含 Qt Test 时会生成 `yolo_defect_qt_tests`：

```powershell
cmake --build $qtBuildDir --target yolo_defect_qt_tests
ctest --test-dir $qtBuildDir -R '^yolo_defect_qt_client$' --output-on-failure
```

测试通过 `offscreen` 平台运行，并复用真实配置和输入比较 Qt 适配结果与 CLI。缺少 Qt Test 时仍能构建客户端，CMake 会提示跳过 Qt 测试。

测试文本日志写入构建目录的 `qt_client_test.txt`。需要检查界面排版时，可设置 `YOLO_DEFECT_QT_SCREENSHOT_DIR` 保存初始、完成和错误状态的离屏截图；Windows 离屏平台不自动枚举系统字体，可同时设置 `YOLO_DEFECT_QT_TEST_FONT=C:\Windows\Fonts\msyh.ttc`。这些仅用于测试，不改变正常客户端的系统字体行为。

## 第一步验收记录（2026-09-27）

Windows x64 / MSVC 19.50 / Qt 6.8.3 / OpenCV 4.8.0 / ORT 1.19.2，Release 构建通过。

- Qt 集成测试通过，无失败或跳过：FP32 与正式 U8S8 的完整 JSON 分别与同配置、同输入的 CLI 输出相同；表格类别、置信度与坐标对应 JSON。
- QtTest 检查运行期间事件循环仍响应、重复启动被阻止、Unicode/空格路径、损坏图片错误后恢复、同一窗口切换到 U8S8 后重新运行，以及运行中关闭窗口的异步收尾。初始、完成、错误状态已做离屏渲染检查。
- `YOLO_DEFECT_BUILD_QT=OFF` 的原 Runtime 默认构建成功，相关 `contract|output|metadata|project_core` 回归 **54/54** 通过；`CORE_ONLY=ON` 与 Qt 开关同时开启时仍跳过 Qt，core smoke **1/1** 通过。

当前未改动 Runtime 源码和公共接口。Windows 演示目录打包、交互缩放、批处理及 GitHub 展示材料按原计划留待后两步。
