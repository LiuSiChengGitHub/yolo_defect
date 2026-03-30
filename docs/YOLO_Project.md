# YOLO 缺陷检测项目 — 学习笔记 & 进度追踪

> 每次推进项目后更新，记录做了什么、学到了什么、下一步做什么。

---

## 当前进度：Step 8 进行中（FastAPI 已跑通，Docker 本地 build/run/curl 已验证，下一步补 README Docker 使用说明）

| Step | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据准备 | Done (2026-03-23) |
| Step 2 | 基线训练 | Done (2026-03-24) — mAP@0.5=0.734 |
| Step 3 | 调参优化 | Done (2026-03-26) — 9组实验, best mAP@0.5=0.743 (final_train_2) |
| Step 4 | 结果分析 + 实验收尾 | Done (2026-03-26) — 误检案例分析, experiment_log 汇总, best model 选定 |
| Step 5 | ONNX 导出 + 推理验证 | Done (2026-03-28) — ONNX GPU 修复完成, ONNX GPU 69.8 FPS, 50张精度对比 50/50 一致 |
| Step 6 | SAM 集成 | - |
| Step 7 | GitHub 美化 | - |
| Step 8 | FastAPI + Docker | In Progress (2026-03-30) — `api/app.py`、Dockerfile、`requirements-api.txt` 已完成，Docker 本地验证通过 |

---

## Step 1：数据准备（2026-03-23）

### 做了什么

1. **运行 `prepare_data.py`** — VOC XML → YOLO TXT 格式转换
   - 输入：`data/NEU-DET/{train,validation}/`（原始 VOC 格式）
   - 输出：`data/images/{train,val}/` + `data/labels/{train,val}/` + `data/data.yaml`
   - 结果：train 1439 张，val 360 张（`crazing_240.jpg` 缺失，已清理孤立标签）

2. **运行 `data_analysis.py`** — 生成数据分析图表
   - `docs/assets/class_distribution.png` — 类别分布柱状图
   - `docs/assets/bbox_sizes.png` — bbox 宽高散点图
   - `docs/assets/bbox_per_image.png` — 每张图 bbox 数量分布

### 数据集关键数据（面试要能说出来）

| 指标 | 数值 |
|------|------|
| 总图片数 | 1800（每类 300） |
| 图片尺寸 | 200 x 200 px |
| 类别数 | 6（均衡分布，不需要过采样） |
| 每图 bbox 数 | 1-9 个，均值 2.33 |
| Bbox 宽度范围 | 8-199 px（均值 71.4） |
| Bbox 高度范围 | 9-199 px（均值 95.0） |
| 训练/验证比 | 80:20（已预划分，不需要自己分） |

### 面试知识点

**Q: VOC 和 YOLO 标注格式有什么区别？**
- VOC：绝对像素坐标，角点表示 `(xmin, ymin, xmax, ymax)`
- YOLO：归一化中心坐标 `(class_id, cx, cy, w, h)`，值域 0-1
- 转换公式：`cx = (xmin+xmax)/2/width`, `cy = (ymin+ymax)/2/height`, `w = (xmax-xmin)/width`, `h = (ymax-ymin)/height`

**Q: 为什么 YOLO 要用归一化坐标？**
- 归一化后坐标与分辨率无关。训练时 200x200 的原图会 resize 到 640x640，归一化坐标自动适配，不用修改标签。

**Q: 你的数据集有什么特点？**
- 类别均衡（每类 300 张），不需要处理类别不平衡问题
- 图片很小（200x200），训练时上采样到 640x640，可能引入模糊
- bbox 尺寸差异极大：crazing 类 bbox 接近整张图（~190px），scratches 则是细长条（宽 ~17px）
- 每张图平均 2.33 个目标，目标密度适中

**Q: `rolled-in_scale` 类名解析有什么坑？**
- 文件名 `rolled-in_scale_1` 包含连字符和下划线混用
- 不能用 `split('_')[0]` 提取类名（会得到 `rolled-in` 而非 `rolled-in_scale`）
- 正确做法：用已知类名列表做最长前缀匹配

### 手算验证示例

`crazing_1.txt` 内容：`0 0.487500 0.490000 0.955000 0.960000`

原始 XML：xmin=2, ymin=2, xmax=193, ymax=194，图片 200x200
- cx = (2+193)/2/200 = 97.5/200 = 0.4875 ✓
- cy = (2+194)/2/200 = 98/200 = 0.49 ✓
- w = (193-2)/200 = 191/200 = 0.955 ✓
- h = (194-2)/200 = 192/200 = 0.96 ✓

---

## Step 2：基线训练（2026-03-24）

### 做了什么

1. **修复环境依赖冲突** — NumPy 2.0.2 与 PyTorch 2.0.0 编译不兼容
   - 将 numpy 降至 1.26.4，opencv-python 降至 4.9.0
   - 根本原因：PyTorch 2.0 是用 NumPy 1.x ABI 编译的，NumPy 2.x 改了 C API

2. **运行基线训练**
   ```bash
   /d/Base/Tools/Anaconda/Anaconda3/envs/yolo_defect/python.exe scripts/train.py --config configs/train_config.yaml
   ```
   - 配置：YOLOv8n，50 epochs，imgsz=640，batch=16，lr0=0.01，GPU RTX 3060
   - 训练时间：约 9.4 分钟（0.156 hours）
   - 输出目录：`runs/detect/train2/`（train 目录已存在，自动递增）

### 训练结果（best.pt 在验证集上的表现）

| 类别 | 图片数 | 实例数 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|--------|--------|-----------|--------|---------|-----------|
| **all** | 360 | 854 | **0.684** | **0.685** | **0.734** | **0.390** |
| crazing | 60 | 162 | 0.668 | 0.395 | 0.542 | 0.211 |
| inclusion | 71 | 159 | 0.733 | 0.760 | 0.793 | 0.439 |
| patches | 73 | 193 | 0.857 | 0.855 | 0.928 | 0.622 |
| pitted_surface | 60 | 87 | 0.749 | 0.755 | 0.830 | 0.456 |
| rolled-in_scale | 60 | 132 | 0.551 | 0.485 | 0.581 | 0.257 |
| scratches | 60 | 121 | 0.545 | 0.860 | 0.731 | 0.357 |

**结论：mAP@0.5 = 0.734，超过目标 0.70，baseline 即达标。**

图表已保存：`docs/assets/confusion_matrix.png`、`docs/assets/PR_curve.png`、`docs/assets/results.png`

### 我自己做的决策（待填写）

> **Step 2 是基线，不涉及调参决策。Step 3 开始你来决定调什么。**
>
> 用下面的模板填写你对结果的分析（5 分钟空白页测试）：
> - crazing AP 最低（0.542），我认为原因是：___
> - rolled-in_scale AP 也偏低（0.581），我认为原因是：___
> - patches AP 最高（0.928），我认为原因是：___
> - Loss 曲线的收敛情况：___
> - 我打算在 Step 3 调整的参数和原因：___

### 如何读懂训练图表

详细说明已移至：

- [[notes/YOLO_notes#如何读懂训练图表|如何读懂训练图表]]

### 你能调整的超参数（面试常问"你怎么调参"）

权重由反向传播自动更新，但以下超参数控制训练过程：

**训练过程参数：**
```yaml
epochs: 50       # Loss 还在下降，可以改 100-150
lr0: 0.01        # 初始学习率，偏大时收敛不稳定，可试 0.005
lrf: 0.01        # 最终 lr = lr0 * lrf，控制学习率衰减终点
model: yolov8n   # 换 yolov8s（参数量 ×4）可以提升 5-8% mAP
imgsz: 640       # 换 800 对细小纹理有帮助，但显存需求翻倍
```

**数据增强参数（控制训练时看到的数据分布）：**
```yaml
hsv_v: 0.4       # 亮度抖动，帮助模型适应不同光照条件
scale: 0.5       # 尺度缩放（-50%~+50%），让模型对尺寸鲁棒
copy_paste: 0.1  # 实例复制粘贴，可增加难类（crazing）的样本密度
mosaic: 1.0      # 4 图拼接（最后 10 epoch 自动关闭）
```

**Loss 权重参数：**
```yaml
cls: 0.5   # 分类 loss 权重，调大让模型更注重"是否有目标"（针对漏检）
box: 7.5   # 框位置 loss 权重
dfl: 1.5   # 框边界分布 loss 权重
```

**Freeze（冻结层）：** YOLOv8 分 Backbone（层 0-9，特征提取）和 Head（层 10-22，框预测）。`freeze: null` 为全量微调（默认）。对 COCO→工业缺陷的跨域迁移，全量微调效果更好，因为 backbone 需要重新适应工业灰度图。

### 工程化数据记录

```
runs/detect/train2/         ← 整个目录被 .gitignore 忽略
├── weights/best.pt         ← ⭐ 手动备份（网盘）6.3MB
├── weights/last.pt         ← 可选备份
├── results.csv             ← ⭐ 手动备份（实验数据）
├── args.yaml               ← 可选（参数记录）
├── *.png（图表）            ← 已复制到 docs/assets/ 由 git 管理
└── train_batch*.jpg        ← 不重要，可丢弃

docs/assets/                ← git 管理（committed）
├── results.png             ← 训练曲线
├── confusion_matrix.png
├── PR_curve.png
└── val_pred_sample.jpg     ← 验证集推理样本
```

**原则：** runs/ 里的文件不进 git。重要图表手动复制到 docs/assets/ 后通过 git 管理。权重文件通过网盘备份。

### Step 3 改进方向（实验规划）

基于 baseline 结果，按优先级：

1. **增加 Epochs（最简单，预期 +3-5%）** — Loss 未收敛，`epochs: 150`，观察平台化位置
2. **换 yolov8s（预期 +5-8%）** — 参数量 ×4，特征提取能力更强，`model: yolov8s.pt`
3. **加强亮度增强（预期 +2-3%）** — crazing 与背景亮度接近，`hsv_v: 0.6`
4. **调大 cls loss 权重（针对漏检）** — `cls: 1.0`，让模型对"是否有目标"更敏感

每组实验：单独一个 yaml 文件（`configs/exp1.yaml`），保存到独立目录（`runs/detect/exp1/`），**用户写分析结论**。

### 面试知识点

**Q: 什么是迁移学习（Transfer Learning）？为什么用 yolov8n.pt 而不从头训练？**
- yolov8n.pt 是在 COCO 数据集（80 类）上预训练的，已经学会了识别边缘、纹理、轮廓等底层特征
- 从预训练权重出发，模型只需要在顶层"适应"新任务（钢材缺陷），收敛更快，效果更好
- 从头训练需要更多数据和更长时间

**Q: mAP@0.5 和 mAP@50-95 有什么区别？**
- mAP@0.5：IoU 阈值固定为 0.5，预测框与真实框重叠 ≥50% 算检测成功
- mAP@50-95：在 IoU=0.50, 0.55, ..., 0.95 这 10 个阈值下分别算 mAP，取平均值
- mAP@50-95 更严格，反映精准定位能力；工业检测通常关注 mAP@0.5

**Q: 为什么 crazing 的 AP 最低？**
- crazing（裂纹）的视觉特征是细小、弥散的纹理，没有明确的边界框
- bbox 接近整张图大小（~190px），但实际缺陷区域和背景混合，难以精确定位
- Recall 只有 0.395，说明模型漏检严重，而不是框得不准

**Q: 训练时 "Closing dataloader mosaic" 是什么意思？**
- Mosaic 数据增强在最后 10 个 epoch 关闭（close_mosaic=10 默认值）
- 原因：训练后期关闭 Mosaic 让模型适应真实单图推理分布，有助于提升最终 mAP

**Q: 模型保存到 train2 而不是 train，为什么？**
- Ultralytics 检测到 `runs/detect/train/` 已存在，自动递增为 `train2`
- 这是 `exist_ok=False` 的默认行为，防止覆盖已有实验结果

---

## Step 3：调参优化（进行中）

### Experiment 1：`imgsz=512`（2026-03-24）

#### 这次改了什么

- **用户决策：** 先测试更小输入尺寸 `imgsz=512`
- **AI 执行：** 新建 `configs/exp1.yaml`，保持其余参数与 baseline 一致
- 输出目录：`runs/detect/exp1/`

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.659** | **0.665** | **0.730** | **0.393** |
| crazing | 0.585 | 0.235 | 0.476 | 0.215 |
| inclusion | 0.732 | 0.774 | 0.835 | 0.449 |
| patches | 0.798 | 0.860 | 0.909 | 0.582 |
| pitted_surface | 0.770 | 0.759 | 0.821 | 0.456 |
| rolled-in_scale | 0.490 | 0.492 | 0.520 | 0.250 |
| scratches | 0.581 | 0.868 | 0.821 | 0.409 |

#### 和 baseline 对比

- baseline：`imgsz=640`，mAP@0.5 = **0.734**，训练时间约 **9.4 分钟**
- exp1：`imgsz=512`，mAP@0.5 = **0.730**，训练时间约 **7.2 分钟**
- **结论：** 总体 mAP 基本持平，但难类 `crazing` 和 `rolled-in_scale` 明显下降，说明较小输入尺寸会损失细粒度纹理信息

#### 已记录的产物

- `docs/assets/results_exp1.png`
- `docs/assets/confusion_matrix_exp1.png`
- `docs/assets/PR_curve_exp1.png`
- `docs/assets/val_pred_sample_exp1.jpg`

#### 重要工程细节

- 本次训练日志显示 `optimizer=auto` 自动选择了 **AdamW(lr=0.001)**
- 这意味着配置文件里的 `lr0=0.01` **没有真正生效**
- 所以如果后面要做“学习率对比实验”，必须先把 `optimizer` 固定，不能继续用 `auto`

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为 `imgsz=512` 对总体 mAP 影响不大，但对 hardest classes 有负面影响，原因是：___
> - `crazing` 从 0.542 降到 0.476，说明：___
> - `rolled-in_scale` 从 0.581 降到 0.520，说明：___
> - 这组实验支持/不支持我继续往更小输入尺寸方向优化，因为：___

### Experiment 2：`imgsz=800`（2026-03-25）

#### 这次改了什么

- **用户决策：** 测试更大输入尺寸 `imgsz=800`
- **AI 执行：** 新建 `configs/exp2.yaml`，保持其余参数与 baseline / exp1 一致
- 输出目录：`runs/detect/exp2/`

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.698** | **0.684** | **0.741** | **0.384** |
| crazing | 0.683 | 0.327 | 0.476 | 0.187 |
| inclusion | 0.726 | 0.802 | 0.833 | 0.472 |
| patches | 0.820 | 0.855 | 0.921 | 0.600 |
| pitted_surface | 0.864 | 0.690 | 0.807 | 0.423 |
| rolled-in_scale | 0.563 | 0.537 | 0.596 | 0.231 |
| scratches | 0.529 | 0.893 | 0.815 | 0.389 |

#### 和 baseline / exp1 对比

- baseline：`imgsz=640`，mAP@0.5 = **0.734**，训练时间约 **9.4 分钟**
- exp1：`imgsz=512`，mAP@0.5 = **0.730**，训练时间约 **7.2 分钟**
- exp2：`imgsz=800`，mAP@0.5 = **0.741**，训练时间约 **13.4 分钟**

#### 这组实验说明了什么

- 更大输入尺寸确实让 **总体 mAP@0.5** 来到了目前图片尺寸实验里的最高点
- `rolled-in_scale` 从 **0.581 → 0.596**，说明更细输入网格对部分纹理类是有帮助的
- 但最关键的 hardest class `crazing` **没有改善**，仍然停在 **0.476**
- 这说明 `imgsz=800` 带来的收益并不是“全面改善难类”，更像是：
  - 对部分类别有帮助
  - 但对最核心的 `crazing` 瓶颈帮助有限
  - 同时训练时间明显增加

#### 重要工程细节

- 本次训练继续使用 `optimizer=auto`
- 日志显示 Ultralytics 再次自动选择了 **AdamW(lr=0.001)**
- 所以 exp2 和 baseline / exp1 在 **imgsz 维度** 上仍然可以对比，但不要拿它去和固定 SGD 的 lr 实验直接混着解释

#### 已记录的产物

- `docs/assets/results_exp2.png`
- `docs/assets/confusion_matrix_exp2.png`
- `docs/assets/PR_curve_exp2.png`
- `docs/assets/val_pred_sample_exp2.jpg`

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为 `imgsz=800` 带来的主要收益是：___
> - 为什么 `rolled-in_scale` 提升了，但 `crazing` 没有明显改善：___
> - 这组实验支持/不支持我继续走“大输入尺寸”方向，因为：___
> - 如果要进一步验证这个方向，我下一步会：___

### Experiment 3：固定优化器后的 learning-rate 对比（2026-03-25）

#### 这次你要学会的核心步骤

1. **先判断旧对照组是否有效**
   - 上一次我们发现 baseline/exp1 都用了 `optimizer=auto`
   - 所以“baseline 的 `lr0=0.01`”并不是一个真正生效的学习率对照

2. **先补一个公平对照组**
   - 新建 `configs/exp3_lr01.yaml`
   - 固定 `optimizer=SGD`
   - 保持 `imgsz=640`、`epochs=50`、`batch=16`、`mosaic=1.0` 不变

3. **再跑只改一个变量的实验组**
   - 新建 `configs/exp3_lr001.yaml`
   - 唯一改动：`lr0: 0.01 → 0.001`

4. **训练后先看 `args.yaml`，再看指标**
   - 这一步非常关键：先确认实际生效参数真的是 `optimizer=SGD`
   - 然后再比较 `results.png`、`results.csv`、per-class AP 和 hardest classes

#### 这次改了什么

- **用户决策：** 推进 Experiment 3，做 `lr0=0.001` vs `0.01` 的学习率对比
- **AI 执行：**
  - 新建 `configs/exp3_lr01.yaml`（公平对照组）
  - 新建 `configs/exp3_lr001.yaml`（低学习率组）
  - 修复 `scripts/train.py` 的 YAML 编码问题：读取配置时显式使用 `utf-8`

#### 训练结果（best.pt 在验证集上的表现）

| 实验 | Optimizer | lr0 | Precision | Recall | mAP@0.5 | mAP@50-95 | 时间 |
|------|-----------|-----|-----------|--------|---------|-----------|------|
| exp3_lr01 | SGD | 0.01 | 0.638 | 0.691 | **0.736** | **0.395** | 约 9.0 分钟 |
| exp3_lr001 | SGD | 0.001 | 0.646 | 0.674 | 0.711 | 0.387 | 约 10.1 分钟 |

#### hardest classes 对比

| 类别 | exp3_lr01 | exp3_lr001 |
|------|-----------|------------|
| crazing | 0.470 | 0.462 |
| rolled-in_scale | 0.537 | 0.507 |
| scratches | 0.846 | 0.762 |

#### 这组实验说明了什么

- 在 **相同 50 epoch 预算** 下，`lr0=0.001` 没有带来更好的结果
- 更小学习率虽然常常更“稳”，但这次更像是 **步子太小，50 个 epoch 内没学够**
- hardest classes 没有改善，反而继续下降，说明这组设置不适合当前项目作为默认方案
- 这次真正值得记住的不是“0.001 不好”，而是：
  - **做学习率实验前，必须先锁死 optimizer**
  - **看学习率时不能只看总 mAP，还要看 hardest classes**

#### 已记录的产物

- `docs/assets/results_exp3_lr01.png`
- `docs/assets/confusion_matrix_exp3_lr01.png`
- `docs/assets/PR_curve_exp3_lr01.png`
- `docs/assets/val_pred_sample_exp3_lr01.jpg`
- `docs/assets/results_exp3_lr001.png`
- `docs/assets/confusion_matrix_exp3_lr001.png`
- `docs/assets/PR_curve_exp3_lr001.png`
- `docs/assets/val_pred_sample_exp3_lr001.jpg`

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为这次 `lr0=0.001` 没有优于 `0.01`，主要原因是：___
> - 从 `crazing` 和 `rolled-in_scale` 的结果来看，我判断更小学习率：___
> - 这组实验给我的调参方法论启发是：___
> - 如果以后还想继续试 learning rate，我会怎样改实验设计：___

### Experiment 4：增强策略 `mosaic=1.0, mixup=0.1`（2026-03-25）

#### 这次我具体是怎么操作的

1. **先选底座配置**
   - 我没有把“best imgsz”和“best lr”硬拼在一起
   - 而是先从当前已经完成的实验里，选出 **整体表现最好的完整配置 `exp2`**
   - 这样做更稳妥，因为 `exp2` 是一个真实跑过并验证过的完整方案

2. **再只改增强参数**
   - 复制 `exp2` 的配置思路，新建 `configs/exp4.yaml`
   - 保持这些参数不变：
     - `imgsz=800`
     - `epochs=50`
     - `batch=16`
     - `optimizer=auto`
     - `mosaic=1.0`
   - 唯一新增改动：
     - `mixup: 0.0 -> 0.1`

3. **训练后先验收配置有没有生效**
   - 不是直接看 mAP
   - 而是先检查 `runs/detect/exp4/args.yaml`
   - 确认 `mixup: 0.1` 真正写进实际配置

4. **最后再比较结果**
   - 和 `exp2` 做一对一对比
   - 因为这次真正想回答的问题是：
     - “在 `imgsz=800` 的最优底座上，加一点 mixup，会不会更好？”

#### 这次改了什么

- **用户决策：** 推进增强实验，测试 `mosaic=1.0, mixup=0.1`
- **AI 执行：**
  - 新建 `configs/exp4.yaml`
  - 基于 `exp2` 只新增 `mixup=0.1`
  - 训练输出目录：`runs/detect/exp4/`

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.692** | **0.679** | **0.735** | **0.384** |
| crazing | 0.637 | 0.314 | 0.464 | 0.196 |
| inclusion | 0.696 | 0.786 | 0.812 | 0.444 |
| patches | 0.789 | 0.891 | 0.930 | 0.611 |
| pitted_surface | 0.883 | 0.724 | 0.813 | 0.412 |
| rolled-in_scale | 0.608 | 0.485 | 0.577 | 0.241 |
| scratches | 0.535 | 0.875 | 0.815 | 0.402 |

#### 和 exp2 对比

- exp2：`imgsz=800`, `mosaic=1.0`, `mixup=0.0`
  - best mAP@0.5 = **0.742**
- exp4：`imgsz=800`, `mosaic=1.0`, `mixup=0.1`
  - best mAP@0.5 = **0.741**

#### 这组实验说明了什么

- 加入 `mixup=0.1` 后，整体结果基本没有提升
- 最关键的 hardest classes 反而略有下降：
  - `crazing`: **0.476 -> 0.464**
  - `rolled-in_scale`: **0.596 -> 0.577**
- 这说明在工业缺陷这种 **细纹理、局部结构很关键** 的任务里，过强的图像混合增强不一定有利
- 你可以先形成一个直觉：
  - `mixup` 对自然图像分类常常有帮助
  - 但对这种依赖局部纹理的缺陷检测，可能会把缺陷边界和纹理模式“搅得更不自然”

#### 重要工程细节

- 这次依然使用 `optimizer=auto`
- 训练日志显示实际还是 **AdamW(lr=0.001)**
- 所以 exp4 的正确比较对象是 **exp2**
- 这是一组“增强策略对比实验”，不是 lr 实验

#### 已记录的产物

- `docs/assets/results_exp4.png`
- `docs/assets/confusion_matrix_exp4.png`
- `docs/assets/PR_curve_exp4.png`
- `docs/assets/val_pred_sample_exp4.jpg`

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为 `mixup=0.1` 没有带来提升，主要原因是：___
> - 为什么这种增强在工业缺陷检测里可能不如分类任务有效：___
> - 这组实验给我的增强策略启发是：___
> - 如果继续做增强实验，我下一步会试：___

### Experiment 5：关闭增强 `mosaic=0, mixup=0`（2026-03-26）

#### 这次我具体是怎么操作的

1. **先明确对照家族**
   - 这次不是回到最早的 baseline
   - 而是沿用 `exp2 / exp4` 的同一底座：
     - `imgsz=800`
     - `epochs=50`
     - `batch=16`
     - `optimizer=auto`

2. **再把目标增强项全部关掉**
   - 新建 `configs/exp5.yaml`
   - 显式写成：
     - `mosaic: 0.0`
     - `mixup: 0.0`
   - 这样它就成为 `exp2 / exp4` 这条增强实验线里的“去增强对照组”

3. **训练后先验收配置**
   - 先看 `runs/detect/exp5/args.yaml`
   - 确认：
     - `mosaic: 0.0`
     - `mixup: 0.0`
   - 这一步是为了确保“关增强”不是你以为关了，而是真的关了

4. **最后再比较结果**
   - 重点和 `exp2`、`exp4` 比
   - 这次真正想回答的问题是：
     - “在同样的 800 输入底座上，强增强、弱增强、关增强，哪一种更适合工业缺陷检测？”

#### 这次改了什么

- **用户决策：** 推进关闭增强实验，设定 `mosaic=0, mixup=0`
- **AI 执行：**
  - 新建 `configs/exp5.yaml`
  - 保持底座配置不变，只关闭样本混合增强
  - 输出目录：`runs/detect/exp5/`

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.655** | **0.685** | **0.737** | **0.395** |
| crazing | 0.523 | 0.265 | 0.462 | 0.196 |
| inclusion | 0.746 | 0.774 | 0.838 | 0.435 |
| patches | 0.750 | 0.912 | 0.928 | 0.607 |
| pitted_surface | 0.753 | 0.701 | 0.800 | 0.448 |
| rolled-in_scale | 0.523 | 0.566 | 0.522 | 0.246 |
| scratches | 0.633 | 0.893 | 0.874 | 0.436 |

#### 和 exp2 / exp4 对比

- exp2：`mosaic=1.0`, `mixup=0.0`
  - best mAP@0.5 = **0.742**
- exp4：`mosaic=1.0`, `mixup=0.1`
  - best mAP@0.5 = **0.741**
- exp5：`mosaic=0.0`, `mixup=0.0`
  - best mAP@0.5 = **0.740**

#### 这组实验说明了什么

- 关掉样本混合增强以后，整体 mAP 并没有明显崩掉
- 某些类别反而更好：
  - `inclusion`: **0.812 -> 0.838**（相对 exp4）
  - `scratches`: **0.815 -> 0.874**
- 这说明对这个项目来说，保留真实纹理分布可能比强行混图更重要
- 但 `crazing` 依然没有改善，说明 hardest class 的问题并不主要来自增强策略，而更可能来自：
  - 训练轮数不够
  - 纹理特征表达不足
  - 类别本身边界弥散

#### 重要工程细节

- 这次依然使用 `optimizer=auto`
- 训练日志显示实际还是 **AdamW(lr=0.001)**
- 所以 `exp5` 的正确比较对象是 **exp2 / exp4**
- 这是增强策略对照实验，不是优化器或学习率实验

#### 已记录的产物

- `docs/assets/results_exp5.png`
- `docs/assets/confusion_matrix_exp5.png`
- `docs/assets/PR_curve_exp5.png`
- `docs/assets/val_pred_sample_exp5.jpg`

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为关闭增强后结果没有明显变差，说明：___
> - 为什么 `scratches` 和 `inclusion` 可能从去增强中受益：___
> - 为什么 `crazing` 依然没有改善：___
> - 这组实验对我下一步调参方向的启发是：___

### Final Train：最优参数组合 + `epochs=100`（2026-03-26）

#### 这次我是怎么具体操作的
1. **先确定“最终训练”不是随意拼参数**
   - 我没有把每个实验里看起来最好的单项结果机械拼接
   - 而是先选出一条 **方法上自洽、已经完整跑通过的最优配置家族**
   - 这次选的是 `exp2` 这条 `imgsz=800` 路线，因为它在完整实验里拿到了最高的 `mAP@0.5`

2. **再吸收增强实验里的结论**
   - `exp4` 说明 `mixup=0.1` 没带来提升
   - `exp5` 说明完全关闭增强不会崩，但也没有整体超过 `exp2`
   - 所以最终训练采用：
     - `imgsz=800`
     - `optimizer=auto`
     - `mosaic=1.0`
     - `mixup=0.0`

3. **最后只改一个长预算变量**
   - 新建 `configs/final_train.yaml`
   - 只把 `epochs: 50 -> 100`
   - 输出目录：`runs/detect/final_train/`

4. **训练后先验收，再下结论**
   - 先检查 `runs/detect/final_train/args.yaml`
   - 确认 `imgsz=800`、`epochs=100`、`mosaic=1.0`、`mixup=0.0` 真正生效
   - 然后再单独验证 `weights/best.pt`
   - 这是为了避免只看训练过程里某一个高点

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.680** | **0.688** | **0.729** | **0.379** |
| crazing | 0.514 | 0.444 | 0.468 | 0.200 |
| inclusion | 0.727 | 0.786 | 0.813 | 0.457 |
| patches | 0.842 | 0.850 | 0.920 | 0.588 |
| pitted_surface | 0.899 | 0.690 | 0.812 | 0.419 |
| rolled-in_scale | 0.575 | 0.485 | 0.544 | 0.226 |
| scratches | 0.524 | 0.876 | 0.816 | 0.385 |

#### 训练过程里的一个重要现象

- `results.csv` 里训练过程最高的 `mAP@0.5` 出现在 **epoch 87 = 0.736**
- 但最终 `best.pt` 单独复验是 **0.729 / 0.379**
- 这说明：
  - **训练过程里某个 epoch 的单指标最高，不一定就是最终 best.pt**
  - Ultralytics 选 `best.pt` 看的是综合 fitness，更接近 `mAP@50-95` 这类更严格指标

#### 和前面最强实验对比

- exp2：`imgsz=800`, 50 epochs
  - `best.pt` = **0.741 / 0.384**
- exp3_lr01：`imgsz=640`, `optimizer=SGD`, `lr0=0.01`, 50 epochs
  - `best.pt` = **0.736 / 0.395**
- final_train：`imgsz=800`, 100 epochs
  - `best.pt` = **0.729 / 0.379**

#### 这次最终训练说明了什么

- 在当前这条 `imgsz=800 + optimizer=auto` 路线下，**单纯把训练拉长到 100 epochs 并没有带来更好的最终模型**
- hardest classes 依然没有实质突破：
  - `crazing` 相比 exp2 没提升
  - `rolled-in_scale` 反而下降
- 训练时间增加到约 **26.1 分钟**，已经接近 50 epoch `imgsz=800` 实验的两倍
- 所以这次很好的结论不是“100 epoch 更强”，而是：
  - **更长训练预算不等于更优最终 checkpoint**
  - **最终训练也要回到 best.pt 的真实验证结果来判断**

#### 已记录的产物

- `docs/assets/results_final_train.png`
- `docs/assets/confusion_matrix_final_train.png`
- `docs/assets/PR_curve_final_train.png`
- `docs/assets/val_pred_sample_final_train.jpg`

#### 这一步你最该学会的实验方法

- 最终训练不是把所有“看起来最好”的数字硬拼在一起
- 应该先选一个 **完整、自洽、已经验证过的配置家族**
- 然后只增加“最终预算变量”，这里就是 `epochs=100`
- 跑完后一定区分：
  - **训练过程中的峰值**
  - **最终 best.pt 的真实验证结果**

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为 final_train 没有超过 exp2 / exp3_lr01，说明：___
> - 为什么训练过程的最高 mAP@0.5 不等于最终 best.pt：___
> - 这次最终训练对我理解“如何选最终模型”的启发是：___

### Final Train 2：手动组合最优候选参数（2026-03-26）

#### 这次是谁做了什么

- **用户手动执行：**
  - 自己创建最终候选配置
  - 自己在终端运行训练命令
- **AI 接手收尾：**
  - 复验 `best.pt`
  - 归档图表到 `docs/assets/`
  - 更新实验日志、项目笔记和 README

#### 这次配置是怎么确定的

这次不是继续沿用 `final_train` 的 `optimizer=auto` 路线，而是把前面实验里更有说服力的结果重新组合成一个**新的最终候选**：

- `imgsz=800`
  - 来自图片尺寸实验里的最优方向
- `optimizer=SGD`
  - 来自有效 learning-rate 对比实验
- `lr0=0.01`
  - 是固定 SGD 条件下更优的学习率
- `mosaic=1.0`
  - 作为主增强基线保留
- `mixup=0.0`
  - 因为 exp4 证明 `mixup=0.1` 没有带来收益
- `epochs=100`
  - 作为最终收尾训练预算

#### 训练前先验收的关键点

这次一定要先看 `runs/detect/final_train_2/args.yaml`，确认下面这些参数真的生效：

- `optimizer: SGD`
- `lr0: 0.01`
- `imgsz: 800`
- `epochs: 100`
- `mosaic: 1.0`
- `mixup: 0.0`

这一步非常关键，因为它证明这次训练确实是你想验证的“手动组合最优候选”，而不是又被 `optimizer=auto` 改写了。

#### 训练结果（best.pt 在验证集上的表现）

| 类别 | Precision | Recall | mAP@0.5 | mAP@50-95 |
|------|-----------|--------|---------|-----------|
| **all** | **0.679** | **0.690** | **0.743** | **0.388** |
| crazing | 0.513 | 0.543 | 0.550 | 0.202 |
| inclusion | 0.773 | 0.742 | 0.827 | 0.454 |
| patches | 0.856 | 0.850 | 0.920 | 0.598 |
| pitted_surface | 0.821 | 0.701 | 0.807 | 0.439 |
| rolled-in_scale | 0.507 | 0.462 | 0.553 | 0.255 |
| scratches | 0.602 | 0.843 | 0.803 | 0.381 |

#### 训练过程中的关键信号

- `results.csv` 里最佳一轮出现在 **epoch 90**
- 当时训练过程指标是：
  - `mAP@0.5 = 0.743`
  - `mAP@50-95 = 0.388`
- 单独复验 `best.pt` 后结果几乎一致
- 这说明这次训练的最佳 checkpoint 非常稳定，不是偶然抖上去的一帧

#### 和之前最强实验对比

- exp2：`imgsz=800`, `optimizer=auto`, 50 epochs
  - `best.pt` = **0.741 / 0.384**
- exp3_lr01：`imgsz=640`, `optimizer=SGD`, `lr0=0.01`, 50 epochs
  - `best.pt` = **0.736 / 0.395**
- final_train：`imgsz=800`, `optimizer=auto`, 100 epochs
  - `best.pt` = **0.729 / 0.379**
- final_train_2：`imgsz=800`, `optimizer=SGD`, `lr0=0.01`, 100 epochs
  - `best.pt` = **0.743 / 0.388**

#### 这次手动最终训练说明了什么

- 这次 **确实验证了“把当前最优候选参数组合起来”是有价值的**
- `final_train_2` 成为了当前项目里 **mAP@0.5 最高** 的模型
- 最明显的亮点是 hardest class `crazing` 提升到了 **0.550**
- 但也要保持诚实：
  - `rolled-in_scale` 仍然不如 exp2
  - `mAP@50-95` 仍然不如 exp3_lr01
- 所以最合理的工程结论是：
  - 如果主打整体 `mAP@0.5`，`final_train_2` 是当前最佳部署候选
  - 如果主打更严格定位指标，`exp3_lr01` 仍然值得保留做对照

#### 已记录的产物

- `docs/assets/results_final_train_2.png`
- `docs/assets/confusion_matrix_final_train_2.png`
- `docs/assets/PR_curve_final_train_2.png`
- `docs/assets/val_pred_sample_final_train_2.jpg`

#### 这一步你最该学会的实验方法

- “理论最优参数”不是纸面上直接推出来的
- 更好的做法是：
  - 先在各自可比的实验家族里选优
  - 再把跨家族的最优候选组合成一个**新实验**
  - 最后用一次独立训练去验证这个组合到底是不是真的更优

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我认为 `final_train_2` 能超过 `exp2`，说明：___
> - 为什么 `crazing` 提升明显，但 `rolled-in_scale` 没有同步成为最优：___
> - 这次手动最终训练让我对“理论最优参数”和“实证最优参数”的区别理解为：___

### 误检/漏检案例分析（2026-03-25）

#### 这次做了什么

- 基于当前候选最佳模型 `runs/detect/exp3_lr01/weights/best.pt`
- 新增脚本 `scripts/analyze_failures.py`
- 在验证集上逐张推理，按 **同类 + IoU>=0.5** 的规则匹配预测框和真值框
- 自动筛出失败最明显的 10 张图，保存到：
  - `docs/assets/failure_cases_exp3_lr01/`

#### 已保存的案例产物

- `docs/assets/failure_cases_exp3_lr01/case_01_inclusion_287.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_02_crazing_299.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_03_rolled-in_scale_259.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_04_rolled-in_scale_292.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_05_crazing_278.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_06_pitted_surface_279.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_07_crazing_246.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_08_crazing_250.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_09_pitted_surface_280.jpg`
- `docs/assets/failure_cases_exp3_lr01/case_10_scratches_293.jpg`
- 摘要表：`docs/assets/failure_cases_exp3_lr01/failure_summary_exp3_lr01.md`

#### 目前观察到的共性

- `crazing`、`rolled-in_scale`、`scratches` 反复出现在失败案例中，和之前的 per-class AP 结论一致
- 很多失败图不是单纯“没框出来”，而是 **漏检 + 误检同时存在**
- 背景纹理和缺陷纹理相似时，容易出现同类误检
- 同图目标较多时，模型更容易同时出现漏检和重复/偏移检测

#### 你做人为分析时的标准流程

1. **先看失败类型**
   - 橙色框是漏检的 GT（FN）
   - 红色框是没有匹配到 GT 的预测（FP）

2. **再判断错因属于哪一类**
   - 纹理太细 / 对比度太低
   - 边界弥散 / 框定义宽泛
   - 背景纹理相似
   - 同图目标密集，定位和去重更难

3. **最后把现象连接到调参方向**
   - 细纹理缺陷多：考虑 `imgsz↑`、更强模型
   - 明显没学够：考虑 `epochs↑`
   - 漏检多：考虑 `cls` 或针对性增强
   - 误检多：考虑更干净的数据增强或更强特征表达

#### 这次你应该学会的核心方法

- 不要手挑几张“看起来像失败”的图片
- 应该先定义匹配规则，再系统筛选 top-k 失败案例
- 误检案例分析不是只写“模型错了”，而是要把：
  - **视觉现象**
  - **类别特性**
  - **可能的训练改进方向**
  连起来讲清楚

#### 你的分析（待填写）

> 你可以用下面模板写你自己的判断：
> - 我观察到 hardest cases 主要集中在：___
> - 这些图的共同视觉特征是：___
> - 我认为最主要的失败原因是：___
> - 如果继续优化，我优先尝试的方向是：___

### Step 3+4 收尾总结：单次实验的标准流程

每组实验从头到尾经过以下步骤：

1. **确定实验设计** — 明确只改一个变量（或一组有理由的组合），选定对照组
2. **创建配置文件** — `configs/expN.yaml`，基于对照组修改目标参数
3. **运行训练** — `python scripts/train.py --config configs/expN.yaml`
4. **验收配置生效** — 检查 `runs/detect/expN/args.yaml`，确认参数真正生效（尤其 `optimizer=auto` 会覆写 `lr0`）
5. **独立复验 best.pt** — `python scripts/evaluate.py --weights runs/detect/expN/weights/best.pt`，不依赖训练过程峰值
6. **归档图表** — 复制 `results.png`、`confusion_matrix.png`、`PR_curve.png`、`val_pred_sample.jpg` 到 `docs/assets/`
7. **更新实验日志** — 在 `docs/experiment_log.md` 填入 Training Results 行 + Per-Class AP 行
8. **更新项目笔记** — 在 `docs/YOLO_Project.md` 写入实验配置、结果表、对比分析、待填用户分析
9. **更新 README** — 同步 Results 表格和图表引用

#### 实验汇总

| 排名 | 实验 | mAP@0.5 | 关键配置差异 |
|------|------|---------|-------------|
| 1 | **final_train_2** | **0.743** | imgsz=800, SGD, lr0=0.01, epochs=100 |
| 2 | exp2 | 0.742 | imgsz=800, auto(AdamW), epochs=50 |
| 3 | exp4 | 0.741 | + mixup=0.1 (无提升) |
| 4 | exp5 | 0.740 | 关闭 mosaic+mixup |
| 5 | exp3-lr01 | 0.736 | SGD, lr0=0.01 (最优 mAP@50-95=0.395) |
| 6 | baseline | 0.734 | imgsz=640, 默认配置 |
| 7 | exp1 | 0.733 | imgsz=512 (速度快但损精度) |
| 8 | final_train | 0.729 | auto(AdamW) + 100 epochs (单纯加轮不够) |
| 9 | exp3-lr001 | 0.711 | SGD, lr0=0.001 (学习率太小) |

#### Best Model 选定

- **部署首选：`final_train_2`** — mAP@0.5 = 0.743，当前最高
- **对照保留：`exp3_lr01`** — mAP@50-95 = 0.395，最严格定位指标最优
- 权重路径：`runs/detect/final_train_2/weights/best.pt`

#### 核心调参结论（面试 3 句话版）

1. **imgsz=800 > 640 > 512** — 工业缺陷图原尺寸小(200px)，大输入分辨率保留更多纹理细节
2. **SGD + lr0=0.01 > AdamW(auto)** — 固定优化器后学习率对比有效，0.001 在 50 epoch 预算下欠拟合
3. **mixup 不适合本任务** — 工业缺陷依赖局部纹理，图像混合破坏纹理结构

---

---

## Step 5：ONNX 导出 + 推理验证（2026-03-27）

### 做了什么

#### 5.1 ONNX 导出

- 使用 `scripts/export_onnx.py` 将 `final_train_2` 的 best.pt 导出为 ONNX 格式
- 命令：`python scripts/export_onnx.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800`
- 导出参数：`imgsz=800`（与训练时一致）、`simplify=True`（图优化）
- 输出：`models/best.onnx`（11.77 MB）

#### 5.2 推理核心类（src/detector.py）

- 封装了 `YOLODetector` 类，完整的 ONNX 推理流程：
  - `__init__`：加载 ONNX 模型，自动选择 CUDA/CPU 后端
  - `preprocess`：BGR→RGB → Resize(800x800) → 归一化(0-1) → HWC→CHW → 加batch维
  - `predict`：预处理 → ONNX Runtime 推理 → 解析输出 `[1,10,13125]` → 置信度过滤 → 坐标转换(cxcywh→xyxy) → 手写 NMS → 返回检测结果列表
  - `draw`：在图上画框 + 类名 + 置信度
  - `_nms`：**手写 NMS 实现**，不依赖 torchvision
- **设计意图**：不依赖 ultralytics，同一个类被 `inference_onnx.py`（CLI 测试）和后续 `FastAPI`（Web 服务）复用

#### 5.3 批量推理验证（scripts/inference_onnx.py）

- 对验证集全部 360 张图片做 ONNX 推理
- 命令：`python scripts/inference_onnx.py --model models/best.onnx --image-dir data/images/val --output-dir results/`
- **结果：CPU 平均 22.5 FPS**（~44.4 ms/image）
- 带标注的结果图保存在 `results/` 目录

#### 5.4 PyTorch 精度基线测量（2026-03-28）

- 用 `scripts/evaluate.py` 对 `final_train_2` 的 `best.pt` 在验证集上重新跑 `model.val()`
- 命令：`python scripts/evaluate.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800`
- **为什么 `imgsz` 也要写 800：**
  - `final_train_2` 训练时使用的是 `imgsz=800`
  - 我们后面要做的是 **PyTorch vs ONNX 精度对齐**
  - ONNX 导出时同样用了 `imgsz=800`，所以 PyTorch 基线也必须保持一致，避免“不是模型差异，而是输入尺寸差异”
- 结果：
  - `mAP@0.5 = 0.7433`
  - `mAP@50-95 = 0.3880`
  - `Precision = 0.6785`
  - `Recall = 0.6902`
- 逐类 AP@0.5 / AP@50-95：
  - `crazing`: `0.5497 / 0.2020`
  - `inclusion`: `0.8272 / 0.4538`
  - `patches`: `0.9204 / 0.5975`
  - `pitted_surface`: `0.8066 / 0.4390`
  - `rolled-in_scale`: `0.5534 / 0.2553`
  - `scratches`: `0.8029 / 0.3805`
- 评估结果目录：`runs/detect/val5`

#### 5.5 为什么要先测 PyTorch mAP

- **这一步不是重复训练结果，而是建立“部署前基线”**
- 训练日志里的 best metric 说明“训练过程中最好达到过什么水平”
- `model.val()` 重新评估 `best.pt`，说明“现在我要拿去部署和对齐的这个权重，在验证集上实际是多少”
- 后面做 ONNX 对齐时，真正要比较的是：
  - PyTorch `best.pt` 的验证集结果
  - ONNX 模型在同一验证集上的结果
- 只有先把 PyTorch 基线钉住，后面才知道 ONNX 是否真的有精度损失

#### 5.6 中间值打印实验（2026-03-28）

- 新建 `scripts/debug_detector.py`，用于手动展开 `preprocess` 与 ONNX 前向过程
- 成功打印 5 个关键 shape：
  - 原图：`(200, 200, 3)`
  - Resize 后：`(800, 800, 3)`
  - `HWC -> CHW` 后：`(3, 800, 800)`
  - 加 batch 维后：`(1, 3, 800, 800)`
  - ONNX 原始输出：`(1, 10, 13125)`
- 这次实验的目的不是改模型，而是把“原图 -> 模型输入张量 -> 模型原始输出”的链路看清楚
- 运行时 CUDA EP 创建失败，但 ONNX Runtime 已自动回退到 CPU，实验本身有效

#### 5.7 这一步我真正学会了什么

- `detector = YOLODetector(model_path)` 不是“处理图片”，而是“创建检测器对象并在 `__init__` 里加载模型、记录输入信息”
- 中间值打印不是“回头插进已经执行完的流程”，而是把 `preprocess()` 关键步骤手动展开，并在步骤之间加 `print`
- 当前项目导出的 `imgsz=800` ONNX 模型原始输出 shape 是 **`(1, 10, 13125)`**，比背 `8400` 更贴近当前部署版本

#### 5.8 conf / iou 阈值观察实验（2026-03-28）

- 在同一张验证图 `data/images/val/crazing_241.jpg` 上，只改后处理阈值，不改模型权重
- 基线：`conf=0.25, iou=0.45`
  - `3 detections`，`47.2 ms`
- 降低置信度阈值：`conf=0.10, iou=0.45`
  - `8 detections`，`49.0 ms`
- 提高置信度阈值：`conf=0.50, iou=0.45`
  - `0 detections`，`34.2 ms`
- 调整 NMS 阈值：
  - `conf=0.25, iou=0.30` → `3 detections`
  - `conf=0.25, iou=0.60` → `3 detections`

#### 5.9 这组阈值实验说明了什么

- 在 `crazing_241.jpg` 这张图上，**`conf_thresh` 比 `iou_thresh` 敏感得多**
- `conf=0.25 -> 0.10` 时框数从 `3 -> 8`，说明有不少低分候选框原本被过滤掉
- `conf=0.25 -> 0.50` 时框数从 `3 -> 0`，说明这张图上的候选框整体置信度并不高
- `iou=0.30` 和 `iou=0.60` 最终框数都还是 `3`，说明这张图的主要矛盾不是重复框，而是候选框置信度分布
- 这一步的目的不是“找最优阈值”，而是理解 ONNX 推理后处理里 `conf` 控制“先保留哪些框”，`iou` 控制“NMS 去重有多狠”

#### 5.10 近似精度对比（2026-03-28）

- 新建 `scripts/compare_pt_onnx.py`
- 方法：
  - 从 `data/images/val` 按文件名顺序抽取 50 张图
  - PyTorch 端使用 `best.pt` + `imgsz=800` + `conf=0.25` + `iou=0.45`
  - ONNX 端使用 `best.onnx` + `YOLODetector` 同样的阈值
  - 不追求“逐框 IoU 精确匹配”，先做 **工程上足够快的近似对比**：
    - 每张图检测框数量是否一致
    - 所有检测框的置信度分布是否接近
- 命令：`python scripts/compare_pt_onnx.py --num-images 50 --imgsz 800 --device cpu`
- 结果：
  - 50 张图里有 **48 张** 检测框数量完全一致，比例 **96%**
  - 平均绝对框数差 `0.04`
  - 最大框数差 `1`
  - 两边总检测框数都为 **147**
  - 只出现 2 张有差异的图片：
    - `crazing_280.jpg`：PyTorch 1 框，ONNX 2 框
    - `crazing_288.jpg`：PyTorch 3 框，ONNX 2 框
- 置信度分布也非常接近：
  - PyTorch mean / median：`0.4021 / 0.3751`
  - ONNX mean / median：`0.4021 / 0.3748`
- 输出文件：
  - `results/pt_onnx_compare/compare_50_images.csv`
  - `results/pt_onnx_compare/compare_50_summary.json`

#### 5.11 为什么先做“50 张近似对比”

- 因为它比“直接重写一整套 ONNX mAP 评估器”更快，适合先判断 ONNX 是否**大体没跑偏**
- 如果抽样结果已经出现：
  - 检测框数量普遍差很多
  - 置信度分布明显漂移
  - 某些类大量消失
  那就说明 ONNX 部署链路很可能有问题，没必要继续往后做 FastAPI
- 这一步的定位是 **快速 sanity check**
- 它不能替代最终的全量 mAP 对齐，但能先证明：**PyTorch 和 ONNX 在样本级输出上已经非常接近**

#### 5.12 PyTorch CPU FPS 测量（2026-03-28）

- 新建 `scripts/benchmark_pytorch.py`
- 命令：`python scripts/benchmark_pytorch.py --num-images 100 --warmup 5 --imgsz 800 --device cpu`
- **为什么固定 `device=cpu`：**
  - 当前 ONNX 已记录的是 **CPU 22.5 FPS**
  - 如果 PyTorch 用 GPU、ONNX 用 CPU，就不是公平对比
  - 所以这一步故意把后端对齐，做 **同硬件后端** 的速度比较
- **为什么先 warmup 5 张：**
  - 第一次推理通常包含额外初始化开销
  - warmup 后再测 100 张，平均值更稳定
- **为什么先把图片读进内存：**
  - FPS 想测的是模型推理速度，不是磁盘 IO 速度
  - 所以脚本把图片预读到内存，计时只包住 `model.predict()`
- 结果：
  - 平均延迟：`118.7 ms/image`
  - 平均速度：**`8.43 FPS`**
  - 平均每张图检测框数：`2.66`
  - 结果文件：`results/pytorch_benchmark_100.json`
- 与 ONNX CPU 对比：
  - ONNX CPU：`22.5 FPS`
  - PyTorch CPU：`8.43 FPS`
  - **ONNX CPU 大约快 2.67x**

#### 5.13 代表性结果图筛选（2026-03-28）

- 新建 `scripts/select_representative_examples.py`
- 目标：每类自动筛选 `1 张正确案例 + 1 张错误案例`，共 12 张
- 做法：
  - 使用 ONNX `YOLODetector` 在验证集上逐张推理
  - 读取 `data/labels/val/*.txt` 作为 GT
  - 用“同类 + IoU>=0.5”匹配预测框与 GT
  - 正确案例：该类 GT 全部匹配成功，且无明显额外框
  - 错误案例：该类存在 FN，或有明显 FP / 错分
- 输出目录：`docs/assets/representative_examples/`
- 自动生成内容：
  - 12 张单图
  - 1 张总览拼图：`representative_examples_grid.jpg`
  - 1 份摘要：`representative_examples_summary.json`
- 本次自动筛选结果：
  - `crazing`：正确 `crazing_290.jpg`；错误 `crazing_252.jpg`
  - `inclusion`：正确 `inclusion_298.jpg`；错误 `inclusion_282.jpg`
  - `patches`：正确 `patches_245.jpg`；错误 `patches_248.jpg`
  - `pitted_surface`：正确 `pitted_surface_276.jpg`；错误 `pitted_surface_280.jpg`
  - `rolled-in_scale`：正确 `rolled-in_scale_264.jpg`；错误 `rolled-in_scale_259.jpg`
  - `scratches`：正确 `scratches_296.jpg`；错误 `scratches_271.jpg`
- 这一步的价值：
  - 不再靠肉眼在几百张图里硬翻
  - 选图逻辑可复现，后面 README 换图也方便
  - 正确/错误案例都有，便于讲“模型什么时候做得好、什么时候会失败”

#### 5.14 性能对比表格（2026-03-28，含 GPU 修复后数据）

| 格式 | mAP@0.5 | mAP@50-95 | FPS (CPU) | FPS (GPU) | 模型大小 | 备注 |
|------|---------|-----------|-----------|-----------|----------|------|
| PyTorch (.pt) | **0.7433** | **0.3880** | **7.1** | **60.5** | **~6.3 MB** | ultralytics `model.val()` + 100图 benchmark |
| ONNX (.onnx) | **≈0.743** | **≈0.388** | **22.0** | **69.8** | **11.77 MB** | `detector.py` + `onnxruntime`，50图近似对比 **50/50 一致** |

#### 5.15 分析文字

- PyTorch 基线评估结果为 `mAP@0.5 = 0.7433`、`mAP@50-95 = 0.3880`，作为 ONNX 精度对齐的参考标准。
- 修复 ONNX Runtime GPU 后重做 50 张近似对比：**50/50 全部一致**（之前 CPU 模式是 48/50），总检测框数 `148 vs 148`，置信度均值 `0.4011 vs 0.4011`，中位数 `0.3745 vs 0.3746`，精度完全对齐。
- 速度方面，ONNX GPU 达到 `69.8 FPS`，是 PyTorch CPU 的 `9.8x`，也略优于 PyTorch GPU 的 `60.5 FPS`。即使在 CPU 上，ONNX `22.0 FPS` 也是 PyTorch CPU `7.1 FPS` 的 `3.1x`。

#### 5.16 闭卷 3 题总结（2026-03-28）

**1. ONNX 输出 `[1, 10, 13125]`，13125 怎么来的？**
- `13125` 是当前 `imgsz=800` 时 3 个检测尺度上的候选位置总数。
- YOLOv8 这里可以近似理解为 3 个特征层：
  - `100 x 100 = 10000`
  - `50 x 50 = 2500`
  - `25 x 25 = 625`
- 三者相加就是 `13125`。
- 更稳妥的记法是：输出通式为 **`[1, 4+nc, num_predictions]`**，本项目里 `nc=6`，所以第二维是 `10`。

**2. preprocess() 做了哪 5 步？顺序？**
- `BGR -> RGB`
- `resize` 到模型输入尺寸 `800x800`
- 像素值归一化到 `0-1`
- `HWC -> CHW`
- 加 batch 维，变成 `[1, 3, 800, 800]`
- 顺序不能乱，因为推理输入分布必须尽量和训练时保持一致。

**3. NMS 的 4 步流程？**
1. 按置信度从高到低排序所有候选框。
2. 取最高分框加入保留列表。
3. 计算它和剩余框的 IoU，删除 IoU 高于阈值的重复框。
4. 对剩余框重复上述过程，直到处理完。

#### 5.17 Day 1 完成确认（对应任务单）

- [x] 通读 `detector.py`
- [x] 中间值打印实验（`debug_detector.py`）
- [x] 改 `conf/iou` 阈值看效果
- [x] PyTorch mAP 测量（`model.val()`）
- [x] 近似精度对比（50 张图）
- [x] PyTorch FPS 测量（100 张图平均）
- [x] 填写对比表格 + 简短分析
- [x] 挑选 12 张代表性结果图（已脚本化自动筛选）
- [x] 更新 `YOLO_Project.md` Step 5

- **结论：今天已完成 [YOLO 0327-0409.md](D:/Base/CodingSpace/yolo_defect/docs/tasks/YOLO 0327-0409.md) 中的 Day 1：精度对齐 + 理解推理链路。**

#### 5.18 README 同步（2026-03-28 早）

- 已同步更新 `README.md` 的性能对比表，补入 `mAP@50-95`、PyTorch CPU `8.43 FPS`、ONNX CPU `22.5 FPS`
- 已在 README 明确当前近似对比结论：`48/50` 张框数一致，总检测框数同为 `147`
- 已在 README 标记：代表性结果图已自动筛选完成，Day 1（精度对齐 + 理解推理链路）已收口

#### 5.19 ONNX Runtime GPU 修复（2026-03-28）

**问题现象：** `onnxruntime_providers_cuda.dll` LoadLibrary error 126，CUDAExecutionProvider 创建失败，自动回退 CPU。

**根因：** 不是版本不兼容，是 **DLL 搜索路径问题**。ORT 需要的 4 个 CUDA DLL 分散在两个非标准位置，都不在 Windows DLL 搜索路径上：

| DLL | 实际位置 | 说明 |
|-----|----------|------|
| `cudart64_110.dll` | `conda_env/bin/` | CUDA Runtime |
| `cublas64_11.dll` | `conda_env/bin/` | cuBLAS |
| `cufft64_10.dll` | `conda_env/bin/` | cuFFT |
| `cudnn64_8.dll` | `torch/lib/` | cuDNN（随 PyTorch 安装） |

PyTorch 能用 GPU 是因为它有自己的 DLL 加载逻辑，不走 Windows 标准 LoadLibrary。ORT 是纯 C++ DLL，完全依赖系统级搜索。

**修复方式：** 在 `src/detector.py` 中，`import onnxruntime` **之前**，把 `conda_env/bin/` 和 `torch/lib/` 加入 `os.add_dll_directory()` 和 `PATH` 环境变量。

关键点：必须在 import 之前执行，因为 ORT 在 import 阶段就注册 providers。

**修复后速度对比（100 张图 benchmark，RTX 3060）：**

| 后端 | FPS | ms/img | vs PyTorch CPU |
|------|-----|--------|----------------|
| PyTorch CPU | 7.1 | 141.6 | 1x |
| ONNX CPU | 22.0 | 45.4 | 3.1x |
| PyTorch GPU | 60.5 | 16.5 | 8.5x |
| **ONNX GPU** | **69.8** | **14.3** | **9.8x** |

**修复后 50 张近似对比（ONNX GPU）：**
- 一致率：**50/50**（100%，之前 CPU 模式 48/50）
- 总检测框数：PyTorch `148` vs ONNX `148`
- 置信度均值：`0.4011` vs `0.4011`
- 置信度中位数：`0.3745` vs `0.3746`

#### 5.20 README / CLAUDE.md 同步（2026-03-28）

- 性能对比表已补入 GPU FPS 数据
- `CLAUDE.md` 已补入 ONNX Runtime GPU DLL 路径修复说明和 Step 5 完成状态
- 50 张近似对比结果已更新为 GPU 模式下的 50/50 一致

### 关键数据

| 指标 | 数值 |
|------|------|
| 模型来源 | `final_train_2` best.pt (mAP@0.5=0.743) |
| ONNX 模型大小 | 11.77 MB |
| 输入尺寸 | [1, 3, 800, 800] |
| 输出形状 | [1, 10, 13125]（4坐标 + 6类别，imgsz=800 时的候选框总数） |
| PyTorch mAP@0.5 | **0.7433** |
| PyTorch mAP@50-95 | **0.3880** |
| 50张近似对比一致率 | **100%**（50/50 张检测框数量一致，ONNX GPU 模式） |
| PyTorch 置信度均值 / 中位数 | `0.4011 / 0.3745` |
| ONNX 置信度均值 / 中位数 | `0.4011 / 0.3746` |
| PyTorch CPU 推理速度 | **7.1 FPS**（141.6 ms/image） |
| PyTorch GPU 推理速度 | **60.5 FPS**（16.5 ms/image） |
| ONNX CPU 推理速度 | **22.0 FPS**（45.4 ms/image） |
| ONNX GPU 推理速度 | **69.8 FPS**（14.3 ms/image） |
| 代表性结果图 | 已自动筛选 12 张（每类正确+错误各 1） |
| 验证集数量 | 360 张 |
| conf_thresh | 0.25 |
| iou_thresh | 0.45 |

### 面试知识点

**Q: 你项目中为什么要导出 ONNX，不直接用 PyTorch 部署？**
- PyTorch 安装包 >1GB，生产环境或边缘设备太重
- ONNX Runtime 只需 ~50MB，且支持 C++/Java/C# 多语言调用
- ORT 做了算子融合、内存优化，推理速度通常比 PyTorch 快 1.2-2x

**Q: 导出时 imgsz 为什么选 800？**
- 必须和训练时的 imgsz 一致。我的 final_train_2 训练时用的 imgsz=800
- 导出后 ONNX 模型的输入形状是固定的 [1,3,800,800]，推理时预处理也必须 resize 到 800x800
- NEU-DET 原图只有 200x200，大输入分辨率能保留更多纹理细节（调参阶段验证过 800>640>512）

**Q: detector.py 的预处理为什么要和训练完全一致？**
- 模型在训练时"学会了"特定分布的输入（RGB、0-1归一化、CHW 格式）
- 如果推理时用了不同的预处理（比如忘了 BGR→RGB），输入分布完全不同，模型预测就会乱
- 五步必须全做：BGR→RGB → Resize → 归一化 → HWC→CHW → 加 batch 维

**Q: 为什么在做 ONNX 对齐前，还要先跑一次 PyTorch 的 `model.val()`？**
- 因为要先建立一个“当前部署权重的 PyTorch 基线”
- 训练日志里的 best metric 只能说明训练过程中最好达到过什么水平
- 真正做部署对齐时，要比较的是 **同一份 best.pt** 和 **对应导出的 ONNX** 在 **同一验证集、同一输入尺寸** 下的结果
- 这样如果后面 ONNX mAP 掉了，才能确定是导出/预处理/后处理问题，而不是评估条件不一致

**Q: 为什么这次 `model.val()` 要显式写 `--imgsz 800`？**
- 因为 `final_train_2` 训练和 ONNX 导出都基于 `imgsz=800`
- 如果这里偷懒用默认 `640`，那拿到的不是部署前的公平基线
- 面试里可以直接说：**精度对齐前，必须先把输入尺寸、数据划分、权重文件都对齐**

**Q: 为什么还要做“50 张近似对比”，而不是直接说 ONNX 一定没问题？**
- 因为 `model.val()` 只告诉我 PyTorch 基线，不代表 ONNX 部署链路就一定对
- 抽样 50 张图比较检测框数量和置信度分布，是一个很快的 sanity check
- 如果 50 张里大部分图片框数都不一致，或者置信度分布明显漂移，就说明 ONNX 预处理/后处理可能有问题
- 我这次结果是 48/50 张框数一致、总检测框数完全一致、置信度统计几乎重合，所以可以先判断 ONNX 与 PyTorch 非常接近

**Q: 部署速度对比的完整数据？**
- 修复 ONNX Runtime CUDA 后，已补全 4 组对比：
  - PyTorch CPU `7.1 FPS` → ONNX CPU `22.0 FPS`（3.1x 加速，同为 CPU 公平对比）
  - PyTorch GPU `60.5 FPS` → ONNX GPU `69.8 FPS`（1.15x 加速，同为 GPU 公平对比）
- ONNX 在 CPU 和 GPU 上都比 PyTorch 快，验证了 ONNX Runtime 算子融合/内存优化的效果
- 面试时强调：对比要**同后端**才有参考价值

**Q: 为什么 benchmark 要 warmup，还要把图片先读进内存？**
- warmup 是为了排除首次推理的初始化开销
- 预读图片是为了排除磁盘 IO 对 FPS 的干扰
- 这样测出来的才更接近“模型本身的推理速度”

**Q: 你的 NMS 是怎么实现的？（面试手写高频题）**
1. 按置信度从高到低排序
2. 取最高分的框放入 keep 列表
3. 计算它与剩余框的 IoU（交集面积 / 并集面积）
4. IoU > 0.45 的框被抑制（认为是同一目标的重复检测）
5. 重复直到处理完所有框
- IoU = 交集面积 / (面积A + 面积B - 交集面积)

**Q: YOLOv8 的输出张量到底怎么解读？**
- 更稳妥的记法是通式 **`[1, 4+nc, num_predictions]`**
- 在本项目里 `nc=6`，所以第二维是 **10 = 4个框参数 + 6个类别分数**
- **4 个框参数不是两个角点 `(x1, y1, x2, y2)`**，而是 **`(cx, cy, w, h)`**
- 这里的 `(cx, cy, w, h)` 是模型输出尺度上的框参数，后处理时再转成 `xyxy`
- `num_predictions` 是所有特征层候选位置总数；`640` 输入时常见是 `8400`，当前项目 `imgsz=800` 时更应该记成“与输入尺寸相关的候选框总数”
- 后处理：转置为 `[num_predictions, 10]` → 取每行最大类别分数 → 过滤低置信度 → `cxcywh→xyxy` → 映射回原图 → NMS 去重

### 3/28 面试复盘：detector.py 必记纠正点

- **BGR→RGB 不是因为 OpenCV 和 ONNX 不同**，而是因为 OpenCV 读图默认是 BGR，而模型训练时按 RGB 处理；本质是“推理输入分布要和训练一致”
- **输出里的 4 个框参数要记成 `cx, cy, w, h`**，不是角点坐标；角点坐标是后处理转换出来的
- **`10 = 4 + 6`**，表示 4 个框参数 + 6 个类别分数；当前 YOLOv8 这版输出里不是“4个框参数 + objectness + 6类”
- 置信度过滤要放在 NMS 前面，这样可以先删掉大量低质量候选框，减少 IoU 计算量
- NMS 不能只背“删掉 IoU 高的框”，要按完整流程说：**按分数排序 → 保留最高分框 → 计算与其余框的 IoU → 抑制高重叠框 → 重复**
- `detector.py` 单独封装成类的面试答法：**职责清晰、CLI/FastAPI 复用、方便测试和后续维护**

### 3/28 面试复盘：一句话背诵版

- 预处理五步：`BGR→RGB → resize → normalize → HWC→CHW → add batch`
- 预处理要尽量和训练一致，否则就是输入分布漂移，精度会掉
- `predict()` 主链路：预处理 → ONNX Runtime 前向 → 解析输出 → 置信度过滤 → `cxcywh→xyxy` → 映射回原图 → NMS
- `xyxy` 更适合画框、算面积、算 IoU 和做 NMS
- 映射回原图是因为模型输出坐标基于输入尺寸，不是原图尺寸

**Q: ONNX GPU 69.8 FPS 算什么水平？**
- 工业检测场景（产线质检）通常要求 10-30 FPS 即可满足实时性，69.8 FPS 远超要求
- 即使 CPU 22.0 FPS 也已合格，说明不依赖 GPU 也能部署
- 如果需要更快：可以用 TensorRT（GPU 可达 100+ FPS）或 OpenVINO（Intel CPU 优化）

**Q: 为什么 PyTorch GPU 正常，但 ONNX Runtime 无法用 GPU？（实际排查经历）**
- PyTorch 有自己的 DLL 加载逻辑，能自动找到 `conda_env/bin/` 和 `torch/lib/` 里的 CUDA DLL
- ONNX Runtime 的 `onnxruntime_providers_cuda.dll` 是原生 C++ DLL，完全依赖 Windows LoadLibrary 标准搜索
- CUDA DLL（cudart、cublas、cufft）在 `conda_env/bin/`，cuDNN 在 `torch/lib/`，都不在系统 PATH 上
- error 126 = "DLL 的依赖链断了"，不是 DLL 本身不存在
- 修复：在 `import onnxruntime` 之前把这两个目录加入 `os.add_dll_directory()` 和 `PATH`

### 决策记录

- **部署模型选择**：`final_train_2`（用户决策，基于 mAP@0.5=0.743 最优）
- **预处理方式**：简单 resize 而非 letterbox（NEU-DET 原图是正方形 200x200，直接 resize 差异不大）
- **手写 NMS 而非调用 torchvision**：摆脱 PyTorch 依赖 + 面试加分项
- **当前阈值实验结论**：在已观察的 `crazing_241.jpg` 上，优先记住 `conf` 对结果更敏感，`iou` 影响要看是否存在明显重复框
- **PyTorch 基线**：先用 `model.val()` 钉住 `best.pt` 在验证集上的 `0.7433 / 0.3880`，后续 ONNX 对齐都以它为参照
- **近似精度对比结论**：50 张抽样在 GPU 模式下 100% 框数一致，置信度几乎完全相同，ONNX 精度对齐已确认
- **速度对比口径**：已补全 CPU/GPU 四组对比。ONNX GPU 69.8 FPS 最快，ONNX CPU 22.0 FPS 也足够部署
- **ONNX GPU 修复方式**：在 `import onnxruntime` 前把 `conda_env/bin/` 和 `torch/lib/` 加入 DLL 搜索路径，零新包安装
- **代表图选择方式**：改为“脚本基于 GT+预测自动筛”，避免人工随手挑图导致展示样本失真

---

## Step 8：FastAPI 服务化（进行中，2026-03-29）

### 8.1 这次做了什么

1. **新建 `api/app.py`**，完成 FastAPI 最小服务骨架
2. **实现 `GET /health`**，用于检查服务是否启动、模型是否加载成功
3. **实现 `POST /detect`**，接收上传图片，调用 `src/detector.py` 的 `YOLODetector` 做 ONNX 推理并返回 JSON
4. **模型启动时只加载一次**，避免每个请求都重新加载 `best.onnx`
5. **补充基础错误处理、请求日志、响应时间统计**
   - 空文件、坏图片、不支持的文件类型、模型未就绪时能给出明确报错
   - 中间件统一记录 `method/path/status/client/elapsed`
   - 在响应头写入 `X-Response-Time-MS`，并在 `/health` 返回累计请求数与平均响应时间
6. **新增 `scripts/benchmark_api.py` 简单压测脚本**
   - 用 `requests` 并发上传 10 张图片到 `POST /detect`
   - 统计单次响应时间、平均响应时间、总耗时和 QPS
   - 作为 Step 8 “接口能跑”之后的最小性能验证工具
7. **补充 API 使用文档**
   - `README.md` 已加入 `GET /health` 和 `POST /detect` 的请求/响应示例
   - `YOLO_Project.md` 已搭好 API 文档框架，核心理解由用户自己填写

### 8.2 这个任务到底在做什么

这一步不是在“改模型精度”，而是在把**已经训练好、已经验证好的 ONNX 推理能力封装成一个 Web 服务**。

也就是说，之前我们只能：

- 在脚本里 `python scripts/inference_onnx.py ...`

现在我们开始支持：

- 任何会发 HTTP 请求的调用方都可以上传图片
- 服务内部自动完成图片解码、模型推理、结果格式化
- 调用方拿到标准 JSON，不需要知道 YOLODetector 内部细节

**一句话**：这是把“本地脚本推理”升级成“可被系统集成的服务化推理”。

### 8.3 请求流怎么记（面试直接照这个讲）

`POST /detect` 的完整链路：

1. 客户端上传一张图片（multipart/form-data）
2. FastAPI 读取上传文件的二进制内容
3. `cv2.imdecode()` 把字节流转成 OpenCV 图片
4. 调用 `detector.predict(image)`
5. `YOLODetector` 内部完成：预处理 → ONNX Runtime 前向 → 置信度过滤 → `cxcywh→xyxy` → NMS
6. `app.py` 再把结果整理成 JSON：
   - `class_id`
   - `class_name`
   - `confidence`
   - `bbox`
7. 返回给调用方

`GET /health` 的作用更简单：

- 不跑推理，只回答“服务活着吗，模型准备好了吗”

### 8.4 为什么这样设计

- **模型只加载一次**：ONNX 模型加载很慢，如果每次请求都重新建 `YOLODetector`，响应时间会非常差
- **推理逻辑继续放在 `src/detector.py`**：`app.py` 只负责 Web 层，`detector.py` 负责模型层，职责清晰
- **先做最小两个接口**：现在的目标是先跑通“服务化闭环”，不是一上来就做复杂鉴权、数据库、异步队列
- **先补基础可观测性**：即使是最小服务，也要能看见谁在请求、是否报错、响应花了多久

### 8.5 你现在至少要会讲的面试点

**Q: 这一步的出发点是什么？**
- 因为简历项目不能只停留在“我能本地跑脚本”
- 服务化之后，前端、测试脚本、产线系统都可以通过 HTTP 调你的模型
- 这才体现“训练 → 部署 → 服务化”闭环

**Q: 为什么要有 `GET /health`？**
- 运维和部署里最先关心的不是“精度多少”，而是“服务是不是活着、模型有没有加载成功”
- `health` 是最基础的健康检查接口，Docker/K8s/反向代理都会用到这个思路

**Q: 为什么 `app.py` 不直接手写推理逻辑？**
- 因为 `detector.py` 已经在 CLI 推理里验证过了
- 继续复用它可以避免重复代码，也减少 API 层引入新 bug
- 这是典型的关注点分离：Web 层只管请求/响应，模型层只管推理

**Q: FastAPI 和 Flask 相比有什么优势？**
- 类型注解友好，参数校验和接口文档自动生成
- 自带 `/docs`，调试接口很快
- 更适合快速搭建这种“工程展示型”的推理服务

**Q: 这版 API 做了哪些基础工程化处理？**
- 对上传文件做了空文件、坏图片、文件类型的校验
- 用中间件统一记录请求日志
- 给每个响应补了总响应时间，便于后续压测和排障
- `/health` 除了看服务是否存活，也能顺手看累计请求数和平均响应时间
- 单独写了一个 `benchmark_api.py`，可以并发请求 `POST /detect` 并输出平均响应时间和 QPS

### 8.6 当前状态与下一步

- **当前已完成**：`api/app.py` 初版、`POST /detect`、`GET /health`、基础异常处理、请求日志、响应时间统计、简单压测脚本、API 使用文档
- **下一步**：
  - 再进入 Docker 容器化

### 8.7 今日关键数据与结果（2026-03-29）

#### 8.7.1 本地 API 启动结果

- `uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload` 已成功启动
- 服务日志显示：
  - `Loading detector from models\best.onnx`
  - `Detector loaded successfully.`
  - `Application startup complete.`

#### 8.7.2 本地接口测试结果

- `GET /health`
  - 返回：`200 OK`
  - 关键字段：`status=ok`，`model=best.onnx`
- `POST /detect`
  - 测试图片：`data/images/val/crazing_241.jpg`
  - 返回：`200 OK`
  - `count=3`
  - 3 个检测结果的 `class_name` 都是 `crazing`
  - 本次返回里的 `inference_time_ms`：`1947.72 ms`

#### 8.7.3 简单压测结果

压测命令：`scripts/benchmark_api.py`

压测设置：

- 图片数：`10`
- 并发数：`10`
- 接口：`POST /detect`

压测结果：

- 成功请求数：`10/10`
- 失败请求数：`0`
- 平均客户端响应时间：`2333.37 ms`
- 总耗时：`3.06 s`
- 吞吐量：`3.27 QPS`

同一轮压测里的服务端推理时间特征：

- 大多数 `inference_time_ms` 在 `13-23 ms`
- 第一条请求达到 `2198.88 ms`

#### 8.7.4 这组数据怎么解释

- **接口稳定性已经验证通过**：10/10 请求都成功，没有 4xx / 5xx
- **客户端总响应时间不等于纯模型推理时间**：`2333.37 ms` 是端到端等待时间，包含排队和服务调度
- **当前瓶颈更像服务并发处理能力**：大多数服务端纯推理时间只有 `13-23 ms`
- **第一条请求明显更慢**：更像是冷启动 / 本地开发模式 / 并发排队带来的影响，而不是模型每次都要 2 秒

### 8.8 API 使用文档（框架已搭好，核心理解由我自己填写）

#### 8.8.1 服务启动命令

```bash
/d/Base/Tools/Anaconda/Anaconda3/envs/yolo_defect/python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

**我自己的理解（你来填）：**
- 为什么这里用 `uvicorn api.app:app`：uvicorn 是 FastAPI 常用的 ASGI 服务器，api.app:app 表示启动 api/app.py 里的 app 对象
- `--host 127.0.0.1` 的含义：本地访问
- `--port 8000` 的作用：监听端口
- `--reload` 适合什么场景：开发阶段，代码改动后服务自动重启

#### 8.8.2 `GET /health`

**作用：**
- 用来检查服务是否正常启动
- 用来确认 `best.onnx` 是否已经在启动阶段加载成功

**请求示例：**

```bash
curl http://127.0.0.1:8000/health
```

**响应示例：**

```json
{
  "status": "ok",
  "model": "best.onnx",
  "request_stats": {
    "total_requests": 0,
    "avg_response_time_ms": 0.0
  }
}
```

**我自己的理解（你来填）：**
- 为什么 API 服务通常都要先有一个 `/health`：先确认模型成功加载
- `status` / `model` / `request_stats` 各自表示什么：status 表示服务和模型状态，model 表示当前加载的模型文件，request_stats 表示累计请求数和平均响应时间
- 我本地实测时看到了什么返回：返回了 {"status":"ok","model":"best.onnx"...}，说明服务已启动且模型已成功加载。

#### 8.8.3 `POST /detect`

**作用：**
- 接收上传图片
- 调用 `YOLODetector` 做 ONNX 推理
- 返回标准 JSON 检测结果

**请求示例：**

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@data/images/val/crazing_241.jpg"
```

**响应示例：**

```json
{
  "filename": "crazing_241.jpg",
  "count": 3,
  "image_size": {
    "width": 200,
    "height": 200
  },
  "model": "best.onnx",
  "conf_thresh": 0.25,
  "iou_thresh": 0.45,
  "inference_time_ms": 20.57,
  "detections": [
    {
      "class_id": 0,
      "class_name": "crazing",
      "confidence": 0.4457,
      "bbox": [-1.34, 53.68, 176.91, 146.24]
    }
  ]
}
```

**我自己的理解（你来填）：**
- `multipart/form-data` 为什么适合图片上传：因为它适合传文件，客户端可以直接把图片作为表单文件上传给服务端
- 上传字段名为什么必须是 `file`：因为 app.py 里路由参数写的是 file: UploadFile = File(...)，字段名必须和这里一致。
- `count` / `image_size` / `inference_time_ms` / `detections` 分别表示什么：count 是检测框数量，image_size 是原图尺寸，inference_time_ms 是服务端模型推理时间，detections 是每个检测框的详细结果
- `detections` 里每个字段我会怎么向面试官解释：class_id 是类别编号，class_name 是类别名，confidence 是该检测结果的置信度，bbox 是边界框坐标 [x1, y1, x2, y2]

#### 8.8.4 我自己要能口头讲清楚的请求流

```text
上传图片 -> 读取字节流 -> OpenCV 解码 -> detector.predict() -> 格式化 detections -> 返回 JSON
```

**我自己的口语版（你来填，控制在 30-60 秒）：**
> 客户端先上传图片，FastAPI 读取文件字节流，再用 OpenCV 解码成图片，然后调用 detector.predict() 做 ONNX 推理和 NMS，最后把检测结果整理成 JSON 返回

#### 8.8.5 API 测试记录（你来填）

**本地测试命令：**

```bash
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@data/images/val/crazing_241.jpg"
```

**我的测试结果摘要（你来填）：**
- `/health` 返回：status=ok，model=best.onnx
- `/detect` 返回 `count = 3 `
- 这次测试说明了什么：说明 FastAPI 服务已经能正常接收图片、调用 ONNX 检测器，并返回标准 JSON 结果，服务化链路已经跑通。

#### 8.8.6 面试表达模板（你来填自己的版本）

**一句话版：**
> 我把 ONNX 推理封装成了 FastAPI 服务，提供了 /health 和 /detect 两个接口，并且已经本地实测通过。

**1 分钟版：**
> 在完成 ONNX 导出和精度验证之后，我继续做了服务化，把推理能力封装成 FastAPI 接口。GET /health 用来检查服务和模型是否就绪，POST /detect 用来接收上传图片并返回 JSON 检测结果。本地我已经用 curl 实测过，能够成功上传验证集图片并返回类别、置信度和边界框。这一步的意义是把原来的本地脚本推理升级成可集成的服务接口，证明项目具备从训练到部署再到服务化的完整闭环能力。

### 8.9 面试高频问题（今天这一步重点背诵）

**Q: 你今天这步 FastAPI 的核心价值是什么？**
- 不是提升模型精度，而是把已经验证好的 ONNX 推理能力封装成服务
- 这样项目就从“只能本地跑脚本”升级成“可以被其他系统调用的 HTTP 接口”
- 这一步证明了训练 → 部署 → 服务化闭环

**Q: FastAPI 和 Flask 有什么区别？**
- Flask：轻量、经典、自由度高
- FastAPI：类型注解更强、参数校验更方便、自带 `/docs` 自动文档，更适合这种模型接口项目
- 面试时不要说 Flask 不行，而要说 FastAPI 对当前“接口展示 + 文档调试”场景更合适

**Q: 为什么推理逻辑放在 `src/detector.py`，而不是直接写在 `app.py`？**
- `app.py` 管 Web 层：路由、请求、响应、状态码、日志
- `detector.py` 管模型层：预处理、ONNX Runtime 前向、NMS、结果组织
- 这么拆的价值是：可复用、可测试、可维护

**Q: 如果有 100 个并发请求，你的 API 会怎样？**
- 当前这版是单 worker 的本地开发模式，100 并发时会明显排队
- 客户端总响应时间会上升，吞吐量很快碰到瓶颈
- 优化方向不是先怀疑模型前向，而是优先考虑多 worker、正式部署、请求队列

**Q: 今天的压测结果怎么讲，才不会说错？**
- 平均客户端响应时间是 `2333.37 ms`，QPS 是 `3.27`
- 但大多数服务端 `inference_time_ms` 只有 `13-23 ms`
- 这说明“接口端到端等待时间”和“纯模型推理时间”不是一回事
- 当前瓶颈更像服务并发处理能力，而不是模型单次前向本身


### 8.10 Docker 容器化初版（2026-03-30）

#### 8.10.1 这次做了什么

1. **新增 `requirements-api.txt`**
   - 单独为 Docker 部署准备最小依赖集合
   - 只保留 `fastapi`、`uvicorn[standard]`、`python-multipart`、`numpy`、`opencv-python-headless`、`onnxruntime`
   - 明确**不把训练依赖打进镜像**，例如 `ultralytics`、`segment-anything`、`onnxruntime-gpu`
2. **新增根目录 `Dockerfile`**
   - 基础镜像：`python:3.9-slim`
   - 工作目录：`/app`
   - 安装系统库：`libgl1`、`libglib2.0-0`
   - 安装 API 最小 Python 依赖
   - 只复制 `src/`、`api/`、`models/`
   - 暴露 `8000` 端口
   - 用 `uvicorn api.app:app --host 0.0.0.0 --port 8000` 启动服务
3. **把 Docker 设计意图写清楚**
   - 这次目标是 Linux CPU 部署，不是把 Windows 本地开发环境原样搬进去
   - 镜像追求“能部署 API”而不是“同时能训练 YOLO”

#### 8.10.2 为什么要拆出 `requirements-api.txt`

- 现有 `requirements.txt` 是**训练 + 部署混合版**
- 如果 Docker 直接安装它，会把训练、SAM、GPU 方向的包一起装进去
- 这样会带来 3 个问题：
  - 镜像更大
  - 构建更慢
  - 在 `python:3.9-slim` 里更容易出现不必要的兼容性问题

所以这次专门拆出一个“部署专用依赖文件”，只服务于 FastAPI + ONNX Runtime 这条链路。

#### 8.10.3 Dockerfile 每一层在做什么（面试可以按这个顺序讲）

1. `FROM python:3.9-slim`
   - 选择和项目一致的 Python 版本
   - `slim` 比完整镜像更轻，适合展示部署能力
2. `WORKDIR /app`
   - 让后续命令和代码都以 `/app` 为当前目录
   - 这样 `api/app.py` 里按相对位置找 `models/best.onnx` 也能正常工作
3. `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1`
   - 不生成 `.pyc`
   - 日志直接刷到终端，方便看容器日志
4. `RUN apt-get update ... libgl1 libglib2.0-0`
   - `python:3.9-slim` 很干净
   - OpenCV 在 Linux 运行时需要这些系统动态库
5. `COPY requirements-api.txt .`
   - 先只复制依赖文件
   - 这样以后只改 Python 代码时，Docker 可以复用依赖安装缓存
6. `RUN pip install ... -r requirements-api.txt`
   - 只安装 API 推理链路真正需要的包
7. `COPY src/ src/`、`COPY api/ api/`、`COPY models/ models/`
   - 只把运行服务需要的代码和模型放进镜像
   - 不用 `COPY . .`，避免把数据集、文档、训练输出一起打包
8. `EXPOSE 8000`
   - 声明服务监听端口
9. `CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]`
   - 容器里必须监听 `0.0.0.0`
   - 如果只监听 `127.0.0.1`，宿主机端口映射后也访问不到

#### 8.10.4 这一步和本地 Windows GPU 版本有什么不同

- 当前 Docker 目标是 **Linux CPU 部署**
- 所以容器里使用的是 `onnxruntime`，不是 `onnxruntime-gpu`
- `src/detector.py` 里 Windows 的 `_add_cuda_dll_dirs()` 逻辑在 Linux 容器里不会生效，也不是这一步的重点
- 这并不冲突：
  - 本地 Windows 继续保留 GPU 推理能力
  - Docker 先证明“跨环境可部署的 CPU 服务版”已经具备

#### 8.10.5 Docker 本地实测结果（2026-03-30）

- `Dockerfile` 和 `requirements-api.txt` 已完成
- `models/best.onnx` 本地已存在，并已成功复制进镜像
- 已在 Windows + Docker Desktop 环境完成本地验证：

```bash
docker build -t yolo-defect-api .
docker run --rm -p 8000:8000 yolo-defect-api
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@data/images/val/crazing_241.jpg"
```

**实测结果：**

- `docker build -t yolo-defect-api .` 已成功完成
  - 构建总耗时约 `184.7 s`
  - `apt-get` 系统库安装阶段约 `95.2 s`
  - `pip install` Python 依赖阶段约 `67.7 s`
- `docker run --rm -p 8000:8000 yolo-defect-api` 已成功启动容器
- `GET /health` 返回：
  - `status = ok`
  - `model = best.onnx`
- `POST /detect` 测试图片：
  - `data/images/val/crazing_241.jpg`
  - 返回 `count = 3`
  - 3 个检测结果的 `class_name` 都是 `crazing`
  - 本次容器内返回的 `inference_time_ms = 73.99`

**这一步说明了什么：**

- 镜像已经能成功构建，说明 Dockerfile、依赖文件、模型复制路径都没问题
- 容器已经能成功启动，说明 `uvicorn api.app:app` 在镜像内可运行
- `/health` 成功说明服务与模型加载链路正常
- `/detect` 成功说明“上传图片 → OpenCV 解码 → ONNX Runtime 推理 → NMS → JSON 返回”整条容器内服务链路已经打通

**当前剩余工作：**

- README Docker 使用说明已补齐
- 下一步进入 Git Push / 最终仓库收尾

#### 8.10.6 我今天至少要能脱口而出的面试点

- 我没有直接复用训练环境依赖，而是专门拆了一个部署版 `requirements-api.txt`
- Dockerfile 没有 `COPY . .`，因为服务运行只需要 `src`、`api`、`models`
- 选择 `python:3.9-slim` 是为了兼顾版本一致性和镜像体积
- `opencv-python-headless` 更适合容器，因为服务端不需要 GUI
- `CMD` 里必须监听 `0.0.0.0`，这是容器网络访问的基本要求
#### 8.10.7 我现在已经能自己讲清楚的 Dockerfile 理解

- Dockerfile 不能只背命令，要按“环境 → 依赖 → 应用 → 启动”的顺序理解
- 我现在能口头解释它的完整顺序：
  - 先准备 Python 基础环境
  - 再设定工作目录
  - 再通过 `ENV` 优化 Python 在容器里的运行方式
  - 再安装 Linux 系统依赖
  - 再复制部署依赖文件并安装 Python 依赖
  - 再复制 API 所需的代码和模型
  - 再声明监听端口
  - 最后通过 `CMD` 启动 FastAPI 服务
- 我已经理解两个关键“为什么”：
  - 为什么先 `FROM` 再 `WORKDIR`：因为要先有基础环境，再决定默认工作目录
  - 为什么 Python 依赖不直接放进基础镜像：因为基础镜像只提供通用 Python 环境，项目依赖要按当前服务需求单独安装
- 我也已经能区分：
  - `EXPOSE 8000` 是声明端口
  - `CMD ["uvicorn", ...]` 才是容器真正启动时执行的服务命令

#### 8.10.8 README 全量更新（2026-03-30）

- 已把 `docs/assets/demo_inference_result.gif` 接入 README 顶部，作为项目推理演示
- 已在 README 中新增“关键指标 / Key Metrics”总表，统一展示：
  - `mAP@0.5 = 0.743`
  - `mAP@50-95 = 0.388`
  - ONNX `22.0 FPS (CPU)` / `69.8 FPS (GPU)`
  - PyTorch `7.1 FPS (CPU)` / `60.5 FPS (GPU)`
  - API 吞吐量 `3.27 QPS`
- 已把模型体积写成当前本地真实文件数据：
  - `best.pt = 6,286,072 bytes`
  - `best.onnx = 12,336,935 bytes`
- 已在 README 的 FastAPI 部分补入 Docker 用法，包含：
  - `docker build -t yolo-defect-api .`
  - `docker run --rm -p 8000:8000 yolo-defect-api`
  - `/health` 与 `/detect` 的 curl 验证命令
- 已同步 README 中英文两部分，并把 roadmap 里的 Docker / Demo GIF 状态更新为已完成

#### 8.10.9 README 自检与 Quick Start 实跑（2026-03-30）

- 已完成 README 静态自检：
  - 实验结果表保留真实数据
  - PR 曲线 / 混淆矩阵 / Demo GIF 链接全部存在
  - API 示例、Docker 用法、ONNX 性能对比表均已齐全
  - README 中不再残留 `TODO`、`your_image.jpg`、`path/to/image.jpg`、`test.jpg`
- 已把 Quick Start 从“看起来像流程”改成真正的 4 步闭环：
  - `python scripts/prepare_data.py`
  - `python scripts/train.py`
  - `python scripts/export_onnx.py --weights runs/detect/train/weights/best.pt`
  - `python scripts/inference_onnx.py --model models/best.onnx --image data/images/val/crazing_241.jpg`
- 已完成 Quick Start 实跑验证：
  - `prepare_data.py` 跑通，但发现验证集缺少 `crazing_240.jpg`，因此统计结果变为 `1439 train / 361 val`
  - `train.py` 跑通，默认 `runs/detect/train/weights/best.pt` 成功生成
  - 本轮默认训练的最终验证结果为 `mAP@0.5 = 0.734`、`mAP@50-95 = 0.390`
  - `export_onnx.py` 跑通，成功生成 `models/best.onnx`，导出大小约 `11.68 MB`
  - 导出时提示缺少 `onnxslim`，但不影响 ONNX 导出成功
  - `inference_onnx.py` 跑通，对 `crazing_241.jpg` 返回 `1 detection(s)`，结果图成功保存到 `results/`
  - Quick Start 验证结束后，已把 `models/best.onnx` 恢复为 `final_train_2` 的 `imgsz=800` 部署版本，避免影响 README / Docker / API 默认模型
- 这一轮的实际意义：README 里的 Quick Start 现在不只是“语法正确”，而是已经用真实命令验证过整条训练 → 导出 → 推理链路

---

### 下一步

- 继续 **Day 3 最后一项：Git Push / 仓库最终收尾**

---

## 笔记跳转

以下长期知识型内容已整理到学习笔记，项目推进文档只保留跳转：

- [[notes/YOLO_notes#如何读懂训练图表|如何读懂训练图表]]
- [[notes/YOLO_notes#项目文件观|项目文件观]]
- [[notes/YOLO_notes#调参判断方法|调参判断方法]]
- [[notes/YOLO_notes#ONNX 部署|ONNX 部署]]（8.0~8.5：ONNX 概念、计算图、导出原理、ORT、detector 设计、预处理踩坑）
