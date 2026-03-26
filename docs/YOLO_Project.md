# YOLO 缺陷检测项目 — 学习笔记 & 进度追踪

> 每次推进项目后更新，记录做了什么、学到了什么、下一步做什么。

---

## 当前进度：Step 3 已完成（exp1 + exp2 + exp3 + exp4 + exp5 + final_train + final_train_2 已完成）

| Step | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据准备 | Done (2026-03-23) |
| Step 2 | 基线训练 | Done (2026-03-24) — mAP@0.5=0.734 |
| Step 3 | 调参优化 | Done (2026-03-26) — 9组实验, best mAP@0.5=0.743 (final_train_2) |
| Step 4 | 结果分析 + 实验收尾 | Done (2026-03-26) — 误检案例分析, experiment_log 汇总, best model 选定 |
| Step 5 | ONNX 导出 + 推理验证 | Next |
| Step 6 | SAM 集成 | - |
| Step 7 | GitHub 美化 | - |
| Step 8 | FastAPI + Docker | - |

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

### 下一步候选

- 进入 **Step 5：ONNX 导出 + 推理验证**
- 选定要作为部署主模型的 checkpoint：
  - 如果主打 `mAP@0.5`，优先考虑 `final_train_2`
  - 如果想强调更规范的 lr 对照和更高 `mAP@50-95`，保留 `exp3_lr01` 作为重要候选
- 如果以后还想继续做纯调参扩展：
  - 可以把 `epochs=150` 作为可选 `exp6`
  - 但它不再是当前项目推进的最高优先级

---

## 笔记跳转

以下长期知识型内容已整理到学习笔记，项目推进文档只保留跳转：

- [[notes/YOLO_notes#如何读懂训练图表|如何读懂训练图表]]
- [[notes/YOLO_notes#项目文件观|项目文件观]]
- [[notes/YOLO_notes#调参判断方法|调参判断方法]]
