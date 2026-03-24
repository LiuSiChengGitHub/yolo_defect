# YOLO 缺陷检测项目 — 学习笔记 & 进度追踪

> 每次推进项目后更新，记录做了什么、学到了什么、下一步做什么。

---

## 当前进度：Step 2 完成，准备进入 Step 3

| Step | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据准备 | Done (2026-03-23) |
| Step 2 | 基线训练 | Done (2026-03-24) — mAP@0.5=0.734 |
| Step 3 | 调参优化 | Next |
| Step 4 | 结果分析 | - |
| Step 5 | ONNX 导出 + 推理验证 | - |
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

#### results.png（训练过程曲线）

图分两行：上行训练集，下行验证集。左侧 3 列是 Loss，右侧 2 列是指标。

**Loss 三个分量：**
- `box_loss`：预测框位置与真实框的距离误差
- `cls_loss`：类别分类错误（把 crazing 认成 inclusion 之类）
- `dfl_loss`：Distribution Focal Loss，YOLOv8 特有，框边界的分布精度

**健康的训练曲线特征：**
- 训练 loss 和验证 loss 走势一致 → 无过拟合
- 两条线都持续下降 → 还有提升空间
- 验证 loss 先降后升 → 过拟合信号，应早停或加正则化

**本次 baseline 表现：** Loss 在 epoch 50 仍在缓慢下降，说明 50 epoch 不够，有提升空间。

#### confusion_matrix_normalized.png（归一化混淆矩阵）

- **横轴**：真实类别（Ground Truth）；**纵轴**：预测类别（Predicted）
- **对角线**：预测正确（越深越好）
- **最后一行 background**：漏检——模型没检测出来的真实目标

| 真实类别 | 正确率 | 主要问题 |
|---------|-------|---------|
| crazing | 0.48 | 0.52 漏检为 background |
| inclusion | 0.77 | 0.16 漏检 |
| patches | 0.87 | 表现最好 |
| pitted_surface | 0.77 | 极少类间混淆 |
| rolled-in_scale | 0.58 | 0.42 漏检 |
| scratches | 0.88 | 0.31 漏检（但检出的很准）|

**结论：主要问题是漏检（recall 低），不是类间混淆（各类边界清晰）。**

#### PR_curve.png（Precision-Recall 曲线）

- 每条线 = 一个类别，曲线下面积 = AP（该类的平均精度）
- 越靠近右上角越好（高 Precision + 高 Recall 同时成立）
- **crazing 曲线短且低**：Recall 超过 0.4 后 Precision 急剧下降，说明模型只能找到约 40% 的 crazing，不是阈值问题，是特征提取问题

**面试口述：** "PR 曲线中 crazing 的 AP 最低（0.542），曲线形态显示 Recall 上限约 40%，结合混淆矩阵中 52% 的漏检率，判断是特征表达问题——crazing 缺陷边界模糊，与背景纹理相近，需要更强的特征提取能力。"

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

## Step 3：调参优化（待开始）

根据 baseline 结果，由用户分析后决定优化方向（见上方"我自己做的决策"）。
