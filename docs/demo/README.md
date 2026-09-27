# 演示页维护与发布 / Demo maintenance and publishing

展示入口：[中文](index.html) · [English](index.en.html)。两个页面共享结构、媒体和正式报告数据，支持直接离线打开，也用于 GitHub Pages 和 Windows 演示包。

## 文件职责

| 文件 | 职责 |
|---|---|
| `page.template.html` | 共用页面布局、样式和交互 |
| `content.json` | 成对维护的中英文文案 |
| `evidence.json` | 从现有正式报告提取的指标、来源和验证范围 |
| `index.html`、`index.en.html` | 生成结果，提交到 Git；不要手工分别修改 |
| `../assets/qt/` | 同一份真实客户端截图、六个步骤帧和成功检测 GIF |
| `../assets/engineering/` | 中英文量化与并发 SVG 图表 |
| `../../cpp_infer/tools/render_demo.py` | 使用 Python 标准库生成两页，`--check` 检查同步状态 |
| `../../cpp_infer/tools/demo_charts.py` | 读取既有报告，生成证据 JSON 与工程图 |

页面使用相对媒体地址和内联样式/脚本，无 CDN 或在线字体依赖。移动离线展示时一起复制 `demo/` 和同级的 `assets/`。Windows 打包脚本会自动复制所需 HTML 和媒体；Pages 工作流只暂存公开展示文件。

## 更新 UI 截图和 GIF

在已配置 Qt、GoogleTest 和 Python 的开发环境中，从仓库根目录执行：

```powershell
python -m pip install -r cpp_infer/tools/requirements-qt-media.txt
cpp_infer\tools\qt.cmd media
```

将 Pillow 安装到 `cpp_infer/.qt.local.psd1` 或公共配置中 `PythonExe` 指定的同一个环境。`media` 构建可选捕获程序，完成六图真实检测与控件操作，验证六项全部成功，生成稳定路径的媒体，再自动生成两种语言的 HTML。原始帧、运行输出留在被忽略的构建目录中。

只改配色、字体或图标，通常不需要调整捕获脚本。控件对象名或操作流程变化时，同步调整 `qt_demo_capture.cpp`；步骤文案变化需在 `content.json` 同时更新中英文。根目录两份 README 共用 `workbench.png` 和 `walkthrough.gif`，无需另存一套媒体。

## 更新文案、图表和报告

只改文案或布局，编辑共同模板与双语文案，然后运行：

```powershell
python cpp_infer/tools/render_demo.py
python cpp_infer/tools/render_demo.py --check
```

只有正式性能报告或图表设计变化时，才需要更新工程图：

```powershell
python -m pip install -r cpp_infer/tools/requirements-demo-charts.txt
python cpp_infer/tools/demo_charts.py
python cpp_infer/tools/demo_charts.py --check
python cpp_infer/tools/render_demo.py
python cpp_infer/tools/render_demo.py --check
```

Matplotlib 仅用于生成图表。中文生成环境需 Microsoft YaHei、Noto Sans CJK SC、Source Han Sans SC 或 SimHei 之一；SVG 保存字形后，阅读端无需安装字体。脚本从 `cpp_infer/results/` 已跟踪的正式报告提取数值，不重新跑推理，也不从 GIF 估算耗时。

量化展示同时保留模型大小、耗时和质量结果；当前 U8S8 的严格质量门禁未通过，不将性能收益写成全指标通过。并发比较保持平台内相同协议；AArch64/QEMU 展示交叉构建和功能结果，不当作原生 ARM 硬件性能。

更新后打开两种语言页面，检查语言切换、分步浏览、GIF、复制按钮和窄屏排版。将生成的 HTML、图表和媒体与它们的源文件一起提交。需要更新 Windows 演示包时运行 `qt.cmd package -PackageDir <新的空目录>`。

## 发布 GitHub Pages

1. 将本轮修改提交并推送，合并到默认分支 `main`。手动工作流文件需要存在于默认分支，GitHub 才显示 **Run workflow**。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。本仓库已有工作流，不需要另建模板。
3. 进入 **Actions → Qt demo Pages → Run workflow**，选择 `main`，再次点击 **Run workflow**。
4. 等待 `deploy` 成功，打开下列地址。后续展示素材更新后重复步骤 3 即可；普通构建不会自动发布。

| 语言 | 发布地址 |
|---|---|
| 中文 | <https://LiuSiChengGitHub.github.io/yolo_defect/demo/> |
| English | <https://LiuSiChengGitHub.github.io/yolo_defect/demo/index.en.html> |

上述链接在首次成功部署后生效。README 已分别指向对应语言。可以把中文或英文地址填入 GitHub 仓库 **About → Website**，作为项目展示入口。

若发布前的同步检查失败，运行 `python cpp_infer/tools/render_demo.py`，提交两页及对应源文件后重试。

官方说明：[Pages 发布来源](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) · [手动运行工作流](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)。

## English maintenance quick reference

Edit `page.template.html` and both translations in `content.json`; keep the generated pages in Git. Run `python cpp_infer/tools/render_demo.py`, then `--check`. Both pages share the same successful Qt captures and recorded engineering evidence.

For UI changes, run `cpp_infer\tools\qt.cmd media` in the configured Windows development environment; this refreshes the media and both HTML files. For evidence/chart changes, install `requirements-demo-charts.txt`, run `demo_charts.py`, then `render_demo.py`. Preview both languages before committing. Copy `demo/` together with the adjacent `assets/` for offline use.

To publish, merge the workflow and showcase changes into the default `main` branch, select **Settings → Pages → Source → GitHub Actions**, then run **Actions → Qt demo Pages → Run workflow** on `main`. The English README links directly to `demo/index.en.html`. Subsequent updates use the same manual workflow; no deployment is triggered by a normal desktop build.
