# YOLO 目标检测 — 从入门到项目落地学习笔记

> **作者背景：** PyTorch 基础（CIFAR-10 分类 + ResNet 迁移学习）→ 目标检测入门
> **目标岗位：** 外企 CV 算法/部署工程师（博世、西门子、KLA 等）
> **配套项目：** YOLOv8 钢材缺陷检测（NEU-DET 数据集）
> **创建日期：** 2026 年 3 月 23 日

---

## 一、目标检测 vs 图像分类：到底变了什么

### 1.1 任务对比

| 维度 | 图像分类（你已经会的） | 目标检测（你要学的） |
|------|----------------------|-------------------|
| 输入 | 一张图片 | 一张图片 |
| 输出 | 一个类别标签 | **N 个检测结果**，每个包含位置 + 类别 + 置信度 |
| 典型输出形状 | `[batch, num_classes]` 如 `[1, 10]` | `[batch, 4+nc, num_candidates]` 如 `[1, 10, 8400]` |
| 回答的问题 | "这张图是什么？" | "这张图里有什么？在哪里？有多确定？" |
| 后处理 | argmax 取最大概率 | 置信度过滤 + NMS 去重 |
| 典型模型 | ResNet, VGG, EfficientNet | YOLO, Faster RCNN, DETR |
| 损失函数 | CrossEntropyLoss | 分类损失 + 回归损失 + 目标置信度损失 |

### 1.2 数据流对比

```
分类：图片 → Backbone → FC Layer → [1, 10] → argmax → "cat"

检测：图片 → Backbone → Neck(FPN) → Head → [1, 10, 8400] → 过滤+NMS → N个框
                                                                     ↓
                                                 每个框 = [x, y, w, h, class, confidence]
```

核心区别：检测多了 Neck（多尺度特征融合）和 Head（输出框坐标），以及后处理步骤。

### 1.3 从 ResNet 到 YOLOv8 的对应关系

| ResNet 分类 | YOLOv8 检测 | 说明 |
|------------|------------|------|
| ResNet Backbone | CSPDarknet Backbone | 都是卷积堆叠提取特征，YOLO 用 C2f 模块替代 ResBlock |
| Global Average Pooling | FPN Neck | 分类把特征压成一个向量；检测保留空间信息，做多尺度融合 |
| FC Layer (512→10) | Decoupled Head | 分类只输出类别概率；检测同时输出坐标+类别（解耦处理） |
| argmax | NMS | 分类直接取最大值；检测要去除重复框 |
| 无 | Anchor-free | YOLOv8 直接回归坐标，不依赖预定义的锚框 |

---

## 二、核心概念速查

### 2.1 Bounding Box（边界框）表示法

**VOC 格式**（Pascal VOC 标注标准）：

```
(xmin, ymin, xmax, ymax) — 绝对像素坐标，左上角 + 右下角
例：(40, 30, 170, 150) 表示左上角在 (40,30)，右下角在 (170,150)
```

**YOLO 格式**（Ultralytics 训练要求的输入格式）：

```
class_id cx cy w h — 归一化中心坐标 + 宽高，值域 [0, 1]
例：0 0.525 0.45 0.65 0.60
```

**转换公式**（VOC → YOLO）：

```python
cx = (xmin + xmax) / 2 / image_width     # 中心点 x，归一化
cy = (ymin + ymax) / 2 / image_height    # 中心点 y，归一化
w  = (xmax - xmin) / image_width         # 宽度，归一化
h  = (ymax - ymin) / image_height        # 高度，归一化
```

**为什么要归一化？** 归一化后坐标与分辨率无关。原图 200×200 训练时 resize 到 640×640，归一化坐标自动适配，不用手动调整标签。

### 2.2 IoU（Intersection over Union，交并比）

IoU 衡量两个框的重叠程度，是检测任务的基础度量。

```
IoU = 交集面积 / 并集面积，值域 [0, 1]
```

**Python 实现**（面试手写题）：

```python
def compute_iou(box_a, box_b):
    """
    计算两个框的 IoU
    Args:
        box_a, box_b: [xmin, ymin, xmax, ymax] 格式
    Returns:
        float: IoU 值
    """
    # 交集区域
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 并集 = A面积 + B面积 - 交集
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0
```

### 2.3 NMS（Non-Maximum Suppression，非极大值抑制）

**为什么需要 NMS？** 模型输出 8400 个候选框，同一个目标往往被多个框检测到。NMS 的作用是去除冗余框，只保留最好的那个。

**4 步流程**：

1. **排序**：按置信度从高到低排序所有检测框
2. **选最优**：取置信度最高的框，加入最终结果
3. **抑制冗余**：计算该框与剩余所有框的 IoU，IoU > 阈值（通常 0.45-0.5）的框被删除
4. **重复**：回到第 2 步，处理剩余的框，直到全部处理完

**Python 实现框架**（面试高频考点，detector.py 中手写实现）：

```python
def nms(boxes, scores, iou_threshold=0.45):
    """
    Args:
        boxes: [[x1,y1,x2,y2], ...] 所有候选框
        scores: [0.95, 0.87, ...] 对应的置信度
        iou_threshold: IoU 阈值，超过则认为是同一目标
    Returns:
        keep: 保留的框的索引列表
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []

    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order
                 if compute_iou(boxes[best], boxes[i]) < iou_threshold]

    return keep
```

**C++ 部署时的选择**：

- `cv::dnn::NMSBoxes()`：OpenCV 内置，简单好用，大多数场景够用
- 手写 NMS：完全可控，可以针对特定场景优化（如 Soft-NMS、类别感知 NMS）
- 面试建议：先说知道 OpenCV 的实现，再展示手写能力

### 2.4 评估指标体系

**TP / FP / FN 在检测任务中的定义**：

| 术语 | 含义 | 判定条件 |
|------|------|---------|
| TP（True Positive） | 正确检测 | 预测框与某个真实框的 IoU ≥ 阈值 |
| FP（False Positive） | 误检 | 预测框找不到匹配的真实框（IoU 都 < 阈值），或重复匹配 |
| FN（False Negative） | 漏检 | 真实框没有被任何预测框匹配到 |

**Precision 和 Recall**：

```
Precision = TP / (TP + FP)  → 检测出来的框里有多少是对的（准不准）
Recall    = TP / (TP + FN)  → 真实目标里有多少被检测到了（漏不漏）
```

**AP 和 mAP**：

- **AP（Average Precision）** = 单个类别的 PR 曲线下面积。PR 曲线横轴是 Recall，纵轴是 Precision
- **mAP（mean AP）** = 所有类别 AP 的平均值
- **mAP@0.5** = IoU 阈值取 0.5 时的 mAP。"mAP@0.5 = 0.70" 意味着在 "框重叠超过 50% 算对" 的标准下，模型的平均检测准确率为 70%
- **mAP@50-95** = IoU 阈值从 0.5 到 0.95 每隔 0.05 取一次，算 10 个 mAP 的平均值。这是更严格的指标，要求框画得非常精准

**面试一句话解释**：mAP@0.5 = 0.70 意味着，如果我们认为"预测框和真实框重叠超过一半就算找对了"，那模型在所有缺陷类别上平均有 70% 的检测准确率。

---

## 三、YOLO 系列演进

### 3.1 发展脉络（了解级，不需要深入每个版本）

| 版本 | 年份 | 关键改进 | 一句话特点 |
|------|------|---------|----------|
| YOLOv1 | 2016 | 开创性地将检测当作回归问题 | 第一个实时检测器，速度快但精度一般 |
| YOLOv3 | 2018 | 多尺度预测（FPN）、Darknet-53 | 平衡了速度和精度，工业界开始广泛使用 |
| YOLOv5 | 2020 | PyTorch 实现、AutoAnchor、丰富的数据增强 | 工程化做得最好，社区最活跃 |
| YOLOv8 | 2023 | **Anchor-free、C2f 模块、解耦头** | 当前主流，Ultralytics 官方维护 |
| YOLOv9 | 2024 | PGI（可编程梯度信息） | 精度更高，但生态不如 v8 成熟 |

**面试回答 "Why YOLOv8 over v5?"**：

- YOLOv8 是 Anchor-free 的，不需要预定义锚框大小，对不同尺度目标更灵活
- C2f 模块比 C3 更高效（更丰富的梯度流）
- 解耦头让分类和定位任务不互相干扰
- Ultralytics 官方维护，API 统一，训练/导出/部署一条龙

### 3.2 一阶段 vs 二阶段检测器

| 维度 | 一阶段（YOLO） | 二阶段（Faster RCNN） |
|------|-------------|-------------------|
| 流程 | 一次前向传播直接出框 | 先生成候选区域(RPN)，再分类+回归 |
| 速度 | 快（实时 30+ FPS） | 慢（通常 5-10 FPS） |
| 精度 | 略低（v8 已经很接近了） | 略高（尤其是小目标） |
| 适用场景 | 实时检测、边缘部署、工业产线 | 精度优先、不要求实时 |
| 代表 | YOLOv8, SSD, RetinaNet | Faster RCNN, Mask RCNN, Cascade RCNN |

**面试一句话**：YOLO 是一阶段检测器，一次前向传播直接输出所有框，速度快适合工业部署；Faster RCNN 是两阶段，先用 RPN 生成候选区域再精细分类，精度更高但速度慢。

---

## 四、YOLOv8 架构详解

### 4.1 整体结构

```
输入图片 (640×640×3)
    ↓
┌─── Backbone (CSPDarknet) ───┐    特征提取，类似 ResNet
│  Conv → C2f → Conv → C2f    │    输出 3 个尺度的特征图：
│  → Conv → C2f → SPPF        │    P3 (80×80), P4 (40×40), P5 (20×20)
└──────────────────────────────┘
    ↓
┌─── Neck (FPN + PAN) ────────┐    多尺度特征融合
│  自顶向下融合 + 自底向上融合   │    让每个尺度都包含大目标和小目标的信息
└──────────────────────────────┘
    ↓
┌─── Head (Decoupled Head) ───┐    解耦头，分类和定位分开处理
│  分类分支 → 6 个类别概率       │
│  回归分支 → 4 个坐标值         │
└──────────────────────────────┘
    ↓
原始输出: [1, 4+6, 8400]  （batch=1, 10个值, 8400个候选框）
    ↓
┌─── 后处理 ──────────────────┐
│  置信度过滤 (>0.25)           │
│  NMS 去重 (IoU>0.45 的删掉)   │
└──────────────────────────────┘
    ↓
最终输出: N 个检测框 [x, y, w, h, class_id, confidence]
```

### 4.2 输出张量 Shape 详解

YOLOv8 的原始输出形状为 `[1, 4+nc, 8400]`，以 NEU-DET 项目（nc=6）为例是 `[1, 10, 8400]`。

**8400 怎么来的？**

```
三个检测尺度的网格数之和：
  P3: 80 × 80 = 6400  (检测小目标)
  P4: 40 × 40 = 1600  (检测中目标)
  P5: 20 × 20 =  400  (检测大目标)
  合计: 6400 + 1600 + 400 = 8400
```

**10 是什么？**

```
  前 4 个: bbox 坐标 (cx, cy, w, h)
  后 6 个: 6 个类别的概率 (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
```

**后处理时怎么解析**（detector.py 核心逻辑）：

```python
# output shape: [1, 10, 8400]
output = output[0]                # [10, 8400]
output = output.T                 # [8400, 10] → 每行是一个候选框

boxes = output[:, :4]             # [8400, 4] → 坐标
class_probs = output[:, 4:]       # [8400, 6] → 类别概率

confidences = class_probs.max(axis=1)  # 每个框的最大类别概率
class_ids = class_probs.argmax(axis=1) # 每个框的预测类别

# 过滤低置信度
mask = confidences > 0.25
boxes, confidences, class_ids = boxes[mask], confidences[mask], class_ids[mask]

# NMS 去重
keep = nms(boxes, confidences, iou_threshold=0.45)
```

### 4.3 三个关键架构特点（面试关键词）

**Anchor-free（无锚框）**：

- YOLOv5 需要预定义一组 Anchor 尺寸（如 10×13, 16×30, 33×23...），模型在 Anchor 基础上做微调
- YOLOv8 直接回归目标的中心偏移和宽高，不依赖预定义尺寸
- 好处：不需要 Anchor 聚类分析，对不同数据集适应性更强

**Decoupled Head（解耦头）**：

- 早期 YOLO 用一个分支同时预测分类+定位（耦合头）
- YOLOv8 将分类和定位拆成两个独立分支（解耦头）
- 好处：分类关注"是什么"，定位关注"在哪里"，两个任务不互相干扰，训练更稳定

**C2f 模块**：

- 替代 YOLOv5 的 C3 模块
- 引入更多跳跃连接（类似 DenseNet 思想），梯度流更丰富
- 不需要深入理解内部结构，知道"比 C3 更高效"即可

### 4.4 YOLOv8 模型尺寸对比

| 变体 | 参数量 | FLOPs | mAP@0.5 (COCO) | 适用场景 |
|------|--------|-------|----------------|---------|
| YOLOv8n (nano) | 3.2M | 8.7G | 37.3 | **边缘部署、资源受限**，本项目首选 |
| YOLOv8s (small) | 11.2M | 28.6G | 44.9 | 精度和速度的平衡 |
| YOLOv8m (medium) | 25.9M | 78.9G | 50.2 | 服务器部署，精度优先 |
| YOLOv8l (large) | 43.7M | 165.2G | 52.9 | 高精度需求 |
| YOLOv8x (xlarge) | 68.2M | 257.8G | 53.9 | 最高精度，不考虑速度 |

本项目选择 **YOLOv8n** 的原因：NEU-DET 只有 1800 张小图，用大模型容易过拟合；nano 版本推理快，符合边缘部署定位；如果精度不够再换 v8s，只需改一行配置。

---

## 五、训练全流程

### 5.1 Ultralytics Python API

```python
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov8n.pt')  # 自动下载 COCO 预训练权重

# 2. 训练
results = model.train(
    data='data/data.yaml',   # 数据集配置
    epochs=50,               # 训练轮数
    imgsz=640,               # 输入尺寸
    batch=16,                # 批大小
    lr0=0.01,                # 初始学习率
    device=0,                # GPU 编号
)

# 3. 评估
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@50-95: {metrics.box.map}")

# 4. 导出 ONNX
model.export(format='onnx', simplify=True)

# 5. 推理
results = model.predict('test.jpg', conf=0.25)
```

### 5.2 Ultralytics CLI（命令行方式）

```bash
# 训练
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16

# 评估
yolo detect val data=data/data.yaml model=runs/detect/train/weights/best.pt

# 预测
yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg

# 导出
yolo export model=runs/detect/train/weights/best.pt format=onnx simplify=True
```

### 5.3 超参数详解

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|---------|
| `model` | `yolov8n.pt` | 模型尺寸，n/s/m/l/x | 小数据集从 n 开始，精度不够换 s |
| `epochs` | 50 | 训练轮数 | 看 loss 是否收敛，通常 50-200 |
| `imgsz` | 640 | 输入图片尺寸 | 更大=更准但更慢，常见调参变量 |
| `batch` | 16 | 批大小 | -1 自动选择最大 batch，显存不够就减小 |
| `lr0` | 0.01 | 初始学习率 | 微调时可降到 0.001 |
| `lrf` | 0.01 | 最终学习率比例 | 最终 lr ≈ lr0 × lrf |
| `optimizer` | `auto` | 优化器 | auto 自动选择 SGD 或 AdamW |
| `mosaic` | 1.0 | Mosaic 增强概率 | 4 图拼 1 图，提升小目标检测 |
| `mixup` | 0.0 | Mixup 增强概率 | 两图混合叠加，正则化效果 |
| `close_mosaic` | 10 | 最后 N 个 epoch 关闭 mosaic | 让模型最后阶段适应正常图片 |
| `patience` | 50 | 早停耐心值 | 连续 N 个 epoch 没提升就停止 |

**学习率 `lr0` 和 `lrf` 的关系**：

- `lr0`：初始学习率，控制"起跑速度"
- `lrf`：最终学习率比例，控制"收尾精细度"，最终 lr ≈ `lr0 * lrf`
- 前期需要较大的步子快速靠近较优区域，后期需要较小的步子做细调

**注意**：`optimizer=auto` 时 Ultralytics 可能自动覆盖 `lr0`（如选 AdamW 时会改为 0.001）。做学习率对比实验时必须先固定 optimizer，否则实验不成立。

### 5.4 data.yaml 配置文件格式

```yaml
# data.yaml — YOLO 数据集配置
path: data               # 数据集根目录
train: images/train       # 训练集图片目录（相对于 path）
val: images/val           # 验证集图片目录

names:
  0: crazing
  1: inclusion
  2: patches
  3: pitted_surface
  4: rolled-in_scale
  5: scratches
```

YOLO 的硬性要求：`images/` 和 `labels/` 必须平级，文件一一对应（`xxx.jpg` ↔ `xxx.txt`）。

### 5.5 数据增强（检测任务专用）

| 增强方式 | 效果 | 对比分类任务 |
|---------|------|------------|
| **Mosaic** | 4 张图拼成 1 张，每张占一个象限 | 分类没有这个。检测用它增加目标密度和上下文信息 |
| **Mixup** | 两张图按随机比例 α 混合叠加 | 类似分类的 Mixup，但标签是框不是类别 |
| **随机翻转** | 水平/垂直翻转 | 跟分类一样 |
| **HSV 调整** | 随机调色相/饱和度/亮度 | 类似 ColorJitter |
| **尺度抖动** | 随机缩放输入尺寸 | 分类用 RandomResizedCrop，检测用全图缩放 |

**Mosaic 的直觉**：训练时把 4 张图拼成 1 张大图，模型一次能看到 4 张图的目标。好处是增加了"小目标在大图中"的场景（因为每张图被缩到 1/4），并且增加了目标周围的上下文信息。

### 5.6 训练输出（runs/ 目录结构）

```
runs/detect/train/
├── weights/
│   ├── best.pt        # 验证集 mAP 最高的权重（用于评估和部署）
│   └── last.pt        # 最后一个 epoch 的权重（用于断点续训）
├── results.csv        # 每个 epoch 的指标记录
├── results.png        # 训练曲线可视化
├── confusion_matrix.png        # 混淆矩阵
├── PR_curve.png                # PR 曲线
├── F1_curve.png                # F1 曲线
├── val_batch0_pred.jpg         # 验证集预测可视化
└── args.yaml                   # 本次训练的完整参数记录
```

**关于 `exp1.yaml` 和 `args.yaml` 的区别**：

- `exp1.yaml` = 你的实验意图（只写主动覆盖的参数）
- `runs/detect/<exp>/args.yaml` = 训练真正生效的完整快照（含 Ultralytics 补全的默认值）

---

## 六、评估与结果分析

### 6.1 必看的 4 张图

| 图表 | 文件 | 看什么 |
|------|------|--------|
| **训练曲线** | `results.png` | loss 是否收敛、是否过拟合（train loss 下降但 val loss 上升） |
| **PR 曲线** | `PR_curve.png` | 每个类别的 AP，曲线越靠右上角越好 |
| **混淆矩阵** | `confusion_matrix.png` | 哪些类别容易互相误判（如 crazing ↔ scratches） |
| **验证集预测** | `val_batch0_pred.jpg` | 直观看检测效果，找误检/漏检案例 |

### 6.2 如何读懂训练曲线（results.png）

图分两行：上行训练集，下行验证集。左侧 3 列是 Loss，右侧 2 列是指标。

**Loss 三个分量**：

- `box_loss`：预测框位置与真实框的距离误差
- `cls_loss`：类别分类错误（把 crazing 认成 inclusion 之类）
- `dfl_loss`：Distribution Focal Loss，YOLOv8 特有，框边界的分布精度

**健康的训练曲线特征**：

| 曲线形态 | 含义 | 建议操作 |
|---------|------|---------|
| train/val loss 都持续下降，最后还没平台 | 还没收敛 | `epochs ↑` |
| train loss 继续降，但 val loss 开始升 | 过拟合 | 更强数据增强 / 减少 epoch / 更小模型 |
| loss 曲线震荡明显 | 学习率过大 | 固定 optimizer，`lr0 ↓` |
| mAP@0.5 还行，但 mAP@50-95 明显偏低 | 能找到但定位不精 | `imgsz ↑`，关注 box/dfl |
| cls_loss 降得慢，漏检多 | 模型对"有没有目标"不敏感 | `cls ↑`，`epochs ↑` |

**本项目 baseline 的实际信号**：Loss 在 epoch 50 仍在缓慢下降，说明 50 epoch 不够，有继续训练空间。

### 6.3 如何读懂混淆矩阵（confusion_matrix_normalized.png）

- **横轴**：真实类别（Ground Truth）
- **纵轴**：预测类别（Predicted）
- **对角线**：预测正确（越深越好）
- **最后一行 background**：漏检，即模型没有检测出来的真实目标

| 真实类别 | 正确率 | 主要问题 |
|---------|-------|---------|
| crazing | 0.48 | 0.52 漏检为 background |
| inclusion | 0.77 | 0.16 漏检 |
| patches | 0.87 | 表现最好 |
| pitted_surface | 0.77 | 极少类间混淆 |
| rolled-in_scale | 0.58 | 0.42 漏检 |
| scratches | 0.88 | 0.31 漏检（但检出的很准）|

**两类混淆矩阵问题的处置方向**：

- **大量落入 background（漏检）** → `imgsz ↑`、`epochs ↑`、`cls ↑`、针对 hardest classes 做增强
- **不同类别之间互相混淆严重** → 更强模型（yolov8s）、有针对性的增强、检查标签质量

**本项目结论**：主要问题是**漏检（recall 低）**，不是类间混淆主导。

### 6.4 如何读懂 PR 曲线（PR_curve.png）

- 每条线 = 一个类别
- 曲线下面积 = AP（该类的平均精度）
- 越靠近右上角越好（高 Precision + 高 Recall 同时成立）

| PR 曲线形态 | 含义 | 建议操作 |
|------------|------|---------|
| 某类曲线整体很低、很短 | 该类特征表达不足 | `imgsz ↑`、更强模型、针对该类增强 |
| Precision 高但 Recall 很低 | 模型太保守，检出来的准但漏得多 | `cls ↑`、`epochs ↑`、`imgsz ↑` |
| Recall 上去后 Precision 掉很快 | 模型能找但不够稳 | 数据增强、查标签质量、更稳学习率 |

**本项目 crazing 的观察**：曲线低且短，Recall 超过 0.4 后 Precision 急剧下降，Recall 上限偏低。说明模型只能找到约 40% 的 crazing，不是简单阈值问题，而是特征提取能力不足。

**面试口述模板**：

> PR 曲线中 crazing 的 AP 最低，曲线形态显示 Recall 上限偏低；结合混淆矩阵中大量漏检进 background，可以判断主要瓶颈是特征表达能力不足，而不是类别间混淆。

### 6.5 per-class AP 分析原则

工业项目不能只看 overall mAP，还要看 hardest classes。

- **overall mAP 接近，但 hardest classes 掉很多** → 通常不能算更优实验
- **overall mAP 没涨多少，但 hardest classes 明显改善** → 在工业项目里很可能是有价值的优化方向

### 6.6 误检案例分析思路

找 10 张误检/漏检图，按以下维度分析：

- 是哪个类漏检了？该类本身是否纹理细密、对比度低（如 crazing）？
- 误检的框预测成了什么类？两个类是否视觉上相似？
- 是否存在标注错误（数据质量问题）？
- 框位置偏了还是类别搞错了？→ 决定优化方向是调增强还是调模型

### 6.7 调参实验记录模板

每组实验只改一个变量，用 YAML 配置文件管理超参，用 git diff 对比两次实验的参数差异。

```markdown
| 实验 | imgsz | lr0 | epochs | mosaic | mAP@0.5 | mAP@50-95 | 训练时间 | 备注 |
|------|-------|-----|--------|--------|---------|-----------|----------|------|
| baseline | 640 | 0.01 | 50 | 1.0 | ? | ? | ? | YOLOv8n 默认 |
| exp1 | 512 | 0.01 | 50 | 1.0 | ? | ? | ? | 缩小输入 |
| exp2 | 800 | 0.01 | 50 | 1.0 | ? | ? | ? | 增大输入 |
```

### 6.8 训练结束后如何记录结果

训练跑完后，不要一上来就只看某一张图，也不要只盯着终端最后一行。建议固定成下面这套顺序：

1. **先看 `args.yaml`**
   - 确认这次实验到底有没有按你以为的方式跑起来
   - 重点核对：
     - `optimizer`
     - `lr0`
     - `imgsz`
     - `epochs`
     - `mosaic`
     - `mixup`

2. **再看 `results.csv` 和 `results.png`**
   - `results.csv` 负责抄数字
   - `results.png` 负责看趋势
   - 常看的数字：
     - `metrics/mAP50(B)`
     - `metrics/mAP50-95(B)`
     - `metrics/precision(B)`
     - `metrics/recall(B)`
   - 常看的趋势：
     - loss 是否还在下降
     - mAP 是否已经平台
     - 曲线是否震荡

3. **如果要写最终结论，再单独验证 `best.pt`**
   - 训练过程里的某一轮高点，不一定就是最终最适合汇报的模型结果
   - 更稳妥的做法是跑：

```bash
python scripts/evaluate.py --weights runs/detect/<exp>/weights/best.pt --imgsz 800 --save-dir temp_eval
```

   - 这个脚本会给你：
     - 最终 `best.pt` 的 `mAP@0.5`
     - `mAP@50-95`
     - `Precision`
     - `Recall`
     - 每个类别的 AP

4. **再看 4 张关键图做分析**
   - `results.png`：看收敛、震荡、过拟合趋势
   - `confusion_matrix_normalized.png`：看漏检是不是大量掉进 background
   - `BoxPR_curve.png`：看 hardest classes 的曲线是不是整体偏低
   - `val_batch0_pred.jpg`：看视觉上是漏检、误检还是框偏

### 6.9 哪些数字写进 log，哪些图只用于分析

**应该写进 `docs/experiment_log.md` 的数字：**

- 实验名
- `imgsz`
- `optimizer`
- `lr0`
- `epochs`
- `mosaic`
- `batch`
- `mAP@0.5`
- `mAP@50-95`
- `Time`
- 每个类别的 `AP@0.5`

这些数字的推荐来源是：

- **优先** `evaluate.py` 的输出（对应最终 `best.pt`）
- **其次** `results.csv`

**主要用于写分析，不直接抄进表格的图：**

- `results.png`
  - 用来判断 loss 是否收敛、mAP 是否平台化
- `confusion_matrix_normalized.png`
  - 用来判断主要问题是漏检还是类间混淆
- `BoxPR_curve.png`
  - 用来判断 hardest classes 的 Precision / Recall 结构
- `val_batch0_pred.jpg`
  - 用来给出视觉层面的误检/漏检解释

一句话记忆：

```text
表格抄数字 → args.yaml / results.csv / evaluate.py
分析写原因 → results.png / 混淆矩阵 / PR 曲线 / 预测图
```

### 6.10 图表归档与收尾动作

每次训练完成后，固定做下面 4 件事：

1. **复制关键图到 `docs/assets/`**

```bash
Copy-Item runs/detect/<exp>/results.png docs/assets/results_<exp>.png
Copy-Item runs/detect/<exp>/confusion_matrix_normalized.png docs/assets/confusion_matrix_<exp>.png
Copy-Item runs/detect/<exp>/BoxPR_curve.png docs/assets/PR_curve_<exp>.png
Copy-Item runs/detect/<exp>/val_batch0_pred.jpg docs/assets/val_pred_sample_<exp>.jpg
```

2. **更新 `docs/experiment_log.md`**
   - Training Results 表新增一行
   - Per-Class AP 表新增一列
   - 新增这次实验的分析小节

3. **更新 `docs/YOLO_Project.md`**
   - 记录：
     - 这次改了什么
     - 结果怎样
     - 你从中学到了什么

4. **决定这次实验的角色**
   - 是新的最优模型？
   - 还是失败但有启发的对照实验？
   - 这一步很重要，因为不是所有实验都应该被当成“最终模型”

---

## 七、调参决策方法

### 7.1 超参数三大类

#### 训练过程参数

```yaml
epochs
imgsz
batch
lr0
lrf
optimizer
model
device
workers
name
project
```

作用：控制训练多久、输入图像大小、学习率如何变化、选什么优化器、使用哪个模型尺寸。

#### 数据增强参数

```yaml
mosaic
mixup
hsv_h
hsv_s
hsv_v
scale
translate
fliplr
flipud
copy_paste
```

作用：改变训练时模型看到的数据分布，增强泛化能力，对 hardest classes 做针对性增强。

#### Loss 权重参数

```yaml
box    # 更关注框的位置回归
cls    # 更关注类别判断 / 是否有目标
dfl    # 更关注边界框分布精度
```

### 7.2 调参决策树

**起点：先判断主要问题是哪一类**

1. **Loss 到最后还在下降吗？**
   - 是 → `epochs ↑`
   - 否 → 进入下一步

2. **主要问题是漏检到 background 吗？**
   - 是 → 优先考虑：`imgsz ↑`、`cls ↑`、`epochs ↑`
   - 否 → 进入下一步

3. **主要问题是类间混淆吗？**
   - 是 → 优先考虑：更强模型（yolov8s）、数据增强、检查标签质量
   - 否 → 进入下一步

4. **曲线震荡明显吗？**
   - 是 → 固定 optimizer 后 `lr0 ↓`
   - 否 → 进入下一步

5. **overall mAP 接近但 hardest classes 下降吗？**
   - 是 → 不接受该实验为更优解
   - 否 → 继续比较速度、显存和部署成本

### 7.3 本项目的优先调参路线

结合 baseline 和 exp1 的结果，目前更合理的优先级是：

1. **`imgsz=800`** — 验证更大输入是否改善细纹理 hardest classes
2. **固定 optimizer 后再做 learning-rate 对比** — 否则 `lr0` 实验不成立
3. **`epochs=150`** — baseline 和 exp1 都显示 50 epoch 可能还没完全收敛
4. **如果漏检问题依旧明显，再考虑 `cls ↑`** — 用于强化"是否有目标"的学习

### 7.4 压缩记忆版

```
loss 没收敛            → epochs ↑
曲线震荡大             → 固定 optimizer，lr0 ↓
hardest classes 漏检重 → imgsz ↑、cls ↑
类间混淆重             → 更强模型、增强、查标签
overall mAP 接近但 hardest classes 掉很多 → 这组实验不算优
```

---

## 八、ONNX 部署

### 8.0 ONNX 前置知识：从零理解

#### 8.0.1 什么是 ONNX

ONNX = Open Neural Network Exchange（开放神经网络交换格式）。

一句话理解：**ONNX 是模型的"通用语言"。** 就像 PDF 让你不用关心文档是用 Word 还是 WPS 写的，ONNX 让推理引擎不用关心模型是用 PyTorch 还是 TensorFlow 训练的。

```
训练阶段                        部署阶段
┌──────────┐    导出     ┌──────────┐    加载     ┌──────────────────┐
│ PyTorch  │ ────────→  │  .onnx   │ ────────→  │ ONNX Runtime     │
│ TensorFlow│            │  文件    │            │ TensorRT         │
│ PaddlePaddle│          │ (通用格式)│            │ OpenVINO         │
└──────────┘             └──────────┘            │ DirectML (Windows)│
  你只用一个框架训练        中间交换文件              └──────────────────┘
                                                  可以选最适合硬件的引擎
```

#### 8.0.2 计算图：理解 ONNX 的核心概念

ONNX 文件存储的不是 Python 代码，而是一个**计算图（Computational Graph）**。

**什么是计算图？**

你在 PyTorch 里写的 `model(x)` 实际上做了一系列数学操作。计算图把这些操作记录成一个有向无环图（DAG）：

```
输入 [1,3,800,800]
       │
    ┌──▼──┐
    │ Conv │  ← 节点（operator），存储了卷积核权重
    └──┬──┘
       │
    ┌──▼──┐
    │ BN  │  ← BatchNorm，存储了 mean/var/gamma/beta
    └──┬──┘
       │
    ┌──▼──┐
    │ SiLU│  ← 激活函数
    └──┬──┘
       │
      ...（几十到几百个节点）
       │
    ┌──▼──┐
    │Concat│ ← 把多个分支的特征拼起来
    └──┬──┘
       │
输出 [1,10,8400]
```

**关键点：**
- 每个节点 = 一个算子（Conv、Relu、Add、Reshape...），ONNX 定义了 ~180 个标准算子
- 每条边 = 一个张量（Tensor），有明确的形状和数据类型
- 权重（weights）直接嵌入在节点里，不需要额外的 `.pt` 文件
- 图结构是**静态的**——没有 if/else、没有 for 循环、没有 Python 逻辑

#### 8.0.3 导出（Export）到底做了什么

当你执行 `model.export(format='onnx')` 时，背后发生的事：

```
Step 1: Tracing（跟踪）
   PyTorch 给模型喂一个假输入（dummy input），
   记录所有经过的算子和张量形状
   → 得到一个"执行轨迹"

Step 2: 转换
   把 PyTorch 算子映射到 ONNX 标准算子
   例: torch.nn.Conv2d → onnx::Conv
   例: torch.nn.BatchNorm2d → onnx::BatchNormalization

Step 3: 优化（simplify=True 时）
   - 常量折叠：把能在导出时算好的东西提前算
   - BN 融合：BatchNorm 的参数合并进 Conv 的权重
     （推理时 BN 不再是独立操作，速度更快）
   - 死节点消除：去掉 training-only 的分支

Step 4: 序列化
   把优化后的计算图 + 权重写入 .onnx 文件（protobuf 格式）
```

**面试关键：** 导出后的 ONNX 模型**不包含任何 Python 代码**。它就是一个数据文件（计算图 + 权重），任何支持 ONNX 格式的推理引擎都能直接加载运行。

#### 8.0.4 ONNX Runtime 是什么

ONNX Runtime（简称 ORT）是微软开源的推理引擎，专门用来运行 .onnx 模型。

**和 PyTorch 的关系：**
- PyTorch = 训练框架（能训练、能推理，但推理不是最优化的）
- ONNX Runtime = 专用推理引擎（不能训练，但推理做了大量优化）

**为什么 ORT 推理更快？**

| 优化手段 | 说明 |
|---------|------|
| 算子融合 | Conv + BN + ReLU 三步合成一步执行 |
| 内存优化 | 重用中间张量的内存，减少 allocation |
| 硬件调度 | 自动选择 CUDA / TensorRT / CPU 最优实现 |
| 无 Python 开销 | 推理过程纯 C++ 执行，没有 GIL 锁 |

**Execution Providers（EP）= 硬件后端：**
```python
# 优先用 GPU，不行就退回 CPU
session = ort.InferenceSession("model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```
常见 EP：`CUDAExecutionProvider`（NVIDIA GPU）、`TensorrtExecutionProvider`（极致 GPU 加速）、`CPUExecutionProvider`（通用 CPU）、`DirectMLExecutionProvider`（Windows GPU）

#### 8.0.5 PyTorch 推理 vs ONNX 推理的数据流对比

```
PyTorch 推理（训练框架直接用）：
  image → Python 预处理 → torch.Tensor → model(x) → Python 后处理 → 结果
                                          ↑
                               PyTorch runtime（含大量训练相关代码）
                               ~1GB 安装包

ONNX 推理（专用引擎）：
  image → Python/C++ 预处理 → numpy array → session.run() → Python/C++ 后处理 → 结果
                                              ↑
                                   ONNX Runtime（纯推理优化）
                                   ~50MB 安装包
```

**注意：** 预处理和后处理**不在** ONNX 模型里。ONNX 只负责 `输入张量 → 输出张量` 这一步。所以你需要自己保证：
- 预处理（resize、归一化、HWC→CHW）和训练时完全一致
- 后处理（置信度过滤、NMS）自己用 numpy/OpenCV 实现

这就是 `src/detector.py` 存在的原因——它封装了预处理和后处理。

#### 8.0.6 ONNX 的输入输出——你的模型具体长什么样

导出后可以用 `session.get_inputs()` / `session.get_outputs()` 查看：

```
输入:
  name: "images"
  shape: [1, 3, 800, 800]     ← batch=1, RGB 3通道, 800x800（和 imgsz 一致）
  type: float32

输出:
  name: "output0"
  shape: [1, 10, 13125]       ← 当前项目 imgsz=800 时的实际输出形状
  type: float32
```

更稳妥的记法是通式 `[1, 4+nc, num_predictions]`。

输出的 `[1, 10, 13125]` 怎么解读：
- `10 = 4 + 6`：前 4 个值是 `(cx, cy, w, h)` 归一化坐标，后 6 个是 6 个类别的置信度
- `13125`：这是当前 `imgsz=800` 时 3 个尺度特征图候选位置总数
  - `100 × 100 = 10000`
  - `50 × 50 = 2500`
  - `25 × 25 = 625`
  - 合计 `13125`
- 如果是常见的 `imgsz=640`，那才是 `80×80 + 40×40 + 20×20 = 8400`
- 后处理就是：对所有候选框做置信度过滤 → 坐标转换 → NMS 去重 → 最终 N 个检测结果

#### 8.0.6b `conf_thresh` 和 `iou_thresh` 到底控制什么

这两个参数都属于**推理后处理**，不是训练超参数。

**1. `conf_thresh`（置信度阈值）**
- 作用位置：先对模型输出的所有候选框做第一轮过滤
- 逻辑：低于这个阈值的框直接丢掉，连 NMS 都进不去
- 直觉：
  - `conf` 低 → 保留更多框 → 误检可能增加，但漏检可能减少
  - `conf` 高 → 保留更少框 → 结果更保守，但漏检可能增加

**2. `iou_thresh`（NMS 阈值）**
- 作用位置：候选框经过置信度过滤后，进入 NMS 去重
- 逻辑：如果两个框 IoU 太高，就认为它们在检测同一个目标，只保留分数更高的那个
- 直觉：
  - `iou` 低 → NMS 更激进 → 重复框更少，但相邻真实目标也可能被误删
  - `iou` 高 → NMS 更宽松 → 更容易保留重叠框，同一目标可能出现多个框

**工程上为什么要改这两个值？**
- 不改模型权重，也能显著改变最终可视化结果
- `conf` 主要影响“要不要报这个框”
- `iou` 主要影响“重复框删得有多狠”
- 所以它们是面试里最常被追问的后处理参数

**项目里的实际观察（`crazing_241.jpg`）**
- `conf=0.25, iou=0.45` → `3 detections`
- `conf=0.10, iou=0.45` → `8 detections`
- `conf=0.50, iou=0.45` → `0 detections`
- `iou=0.30` 和 `iou=0.60` 最终都还是 `3 detections`

这说明：
- 在这张图上，`conf_thresh` 比 `iou_thresh` 更敏感
- 这张图的主要问题是“候选框置信度不高”，不是“重复框太多”

#### 8.0.6c 为什么 ONNX 对齐前要先跑一次 PyTorch `model.val()`

这一步很容易被误解成“训练都结束了，为什么还要再评估一次”。正确理解是：

- **训练时的 best metric**：表示训练过程中，某个 epoch 的验证表现最好
- **重新跑 `model.val()`**：表示我现在手上的 `best.pt`，在当前验证集设置下，真实基线是多少

做 ONNX 对齐时，我们真正要比较的是：
- PyTorch `best.pt` 在验证集上的结果
- 对应导出的 ONNX 模型在同一验证集上的结果

所以先跑 PyTorch `model.val()` 的目的，是把“参考答案”钉住。

**为什么这次必须写 `imgsz=800`**
- `final_train_2` 训练时就是 `imgsz=800`
- ONNX 导出时也用了 `imgsz=800`
- 如果这里评估却偷懒用默认 `640`，那后面 PyTorch vs ONNX 的差异就掺进了“输入尺寸变化”这个变量，不公平

**手动操作最小步骤**
1. 确认权重路径：`runs/detect/final_train_2/weights/best.pt`
2. 确认数据配置：`data/data.yaml`
3. 运行：
   `python scripts/evaluate.py --weights runs/detect/final_train_2/weights/best.pt --imgsz 800`
4. 记下 4 个总指标：
   - `mAP@0.5`
   - `mAP@50-95`
   - `Precision`
   - `Recall`
5. 再看逐类 AP，确认 hardest class 还是不是 `crazing`

**本项目这次实测结果**
- `mAP@0.5 = 0.7433`
- `mAP@50-95 = 0.3880`
- `Precision = 0.6785`
- `Recall = 0.6902`

**面试回答模板**
“在做 ONNX 精度对齐之前，我先用 Ultralytics 的 `model.val()` 对最终部署候选权重 `best.pt` 重新跑了一次验证集，目的是建立 PyTorch 基线。我显式把 `imgsz` 设成 800，因为训练和 ONNX 导出都使用 800，这样后面比较 ONNX 结果时，不会把输入尺寸差异混进来。最终 PyTorch 基线是 mAP@0.5 0.7433，mAP@50-95 0.3880。” 

#### 8.0.6d 为什么还要做“50 张近似精度对比”

在全量 mAP 对齐之前，先做一轮 **sample-level sanity check** 很有工程价值。

**做法**
- 从验证集抽 50 张图
- PyTorch 和 ONNX 用同样的：
  - `imgsz=800`
  - `conf=0.25`
  - `iou=0.45`
- 不去强求“每个框逐一 IoU 匹配”，先看两个更快的指标：
  1. 每张图的检测框数量是否接近
  2. 所有检测框的置信度分布是否接近

**为什么这样做**
- 如果 ONNX 结果已经严重跑偏，那它往往会先表现为：
  - 框数明显变多/变少
  - 某些图完全没框
  - 置信度整体漂移
- 这些信号不用先写完整 mAP 评估器，抽样就能看出来

**本项目这次抽样结果**
- 50 张图里，48 张检测框数量完全一致，比例 **96%**
- 平均绝对框数差 `0.04`
- 最大差值只有 `1`
- 两边总检测框数都为 `147`
- 置信度分布几乎重合：
  - PyTorch mean / median：`0.4021 / 0.3751`
  - ONNX mean / median：`0.4021 / 0.3748`

**结论怎么说**
- 这不能替代最终的 mAP 对齐
- 但已经足够说明：**ONNX 没有明显跑偏，和 PyTorch 的样本级输出非常接近**

**面试回答模板**
“在正式做全量 mAP 对齐前，我先抽了 50 张验证图做近似对比，用检测框数量和置信度分布做 sanity check。结果 48/50 张图片的框数完全一致，PyTorch 和 ONNX 的总检测框数都为 147，置信度均值和中位数也几乎重合。这说明 ONNX 部署链路没有明显跑偏，可以继续做全量精度验证。” 

#### 8.0.6e PyTorch FPS 为什么要测 CPU、为什么要 warmup

**1. 为什么这次测 CPU，不直接测 GPU**
- 当前项目里已经有 ONNX 的 CPU 结果：`22.5 FPS`
- 如果拿 PyTorch GPU 去和 ONNX CPU 比，结论不公平
- 所以这一步固定 `device=cpu`，做 **同后端对比**

**2. 为什么要 warmup**
- 第一次推理通常会有额外初始化开销
- 如果直接把第一张图也算进平均值，FPS 会偏低
- 所以先 warmup 5 张，再正式计时 100 张

**3. 为什么图片要先读进内存**
- 想测的是模型推理速度，不是硬盘读图速度
- 所以 benchmark 里把图片预读到内存，计时只包住 `model.predict()`

**本项目这次结果**
- 命令：
  `python scripts/benchmark_pytorch.py --num-images 100 --warmup 5 --imgsz 800 --device cpu`
- 平均延迟：`118.7 ms/image`
- 平均速度：`8.43 FPS`
- 平均每张图检测框数：`2.66`
- 对比 ONNX CPU：`22.5 / 8.43 ≈ 2.67x`

**面试回答模板**
“我单独写了一个 PyTorch benchmark 脚本，在 CPU 上先 warmup 5 张，再对 100 张图做平均，且把图片预先读进内存，避免磁盘 IO 干扰。最终 PyTorch CPU 是 8.43 FPS，同机同后端下 ONNX CPU 是 22.5 FPS，说明 ONNX Runtime 在部署侧明显更适合这个项目。” 

#### 8.0.7 精度对齐：为什么 ONNX 结果可能有微小差异

导出后的 ONNX 模型和 PyTorch 模型的 mAP 差异通常 **< 0.005**，但不会完全相同：

| 差异来源 | 说明 |
|---------|------|
| 浮点精度 | PyTorch 用 CUDA kernel，ORT 可能用不同实现，浮点误差会累积 |
| BN 融合 | BatchNorm 合并到 Conv 后，计算路径变了 |
| Resize 实现 | OpenCV resize 和 PyTorch F.interpolate 的插值算法可能不完全一致 |
| NMS 实现 | PyTorch 用 torchvision.ops.nms，ONNX 推理用 OpenCV/numpy 实现 |

**工程规则：** 差异 < 0.01 就算对齐成功。如果差 > 0.02，一定是预处理或后处理有 bug。

#### 8.0.8 面试高频题

**Q: 为什么不直接用 PyTorch 部署？**
- 安装包太大（>1GB），服务器/边缘设备不想装完整训练框架
- PyTorch 推理带 Python GIL 和训练时代码路径，不如专用引擎高效
- ONNX Runtime 支持多语言（C++/C#/Java），PyTorch 几乎只能 Python

**Q: ONNX 导出后模型能继续训练吗？**
- 不能。ONNX 是纯推理格式，只有前向计算图，没有梯度信息
- 如果要继续训练（fine-tune），必须回到 PyTorch 的 .pt 文件

**Q: 什么是算子融合（Operator Fusion）？举个例子。**
- Conv → BatchNorm → ReLU 三个独立操作可以合成一个融合算子
- 好处：减少内存读写次数（不用存中间结果）、减少 kernel 启动开销
- BN 融合的数学原理：BN 是线性变换 `y = γ(x-μ)/σ + β`，Conv 也是线性变换，两个线性变换可以合成一个

**Q: ONNX 模型的输入输出形状怎么确定？**
- 导出时由 dummy input 的形状决定
- 可以用 `session.get_inputs()[0].shape` 查看
- 如果要支持不同尺寸输入，导出时需要设置 dynamic axes

**Q: 你的项目里 ONNX 导出后精度有变化吗？怎么验证的？**
- 用**同一组验证集图片**分别跑 PyTorch 推理和 ONNX 推理
- 比较两者的 mAP@0.5 和 mAP@50-95，差异应 < 0.01
- 如果差异大，排查预处理是否一致（颜色空间、归一化、resize 方式）

---

### 8.1 为什么需要 ONNX

| 维度 | PyTorch 直接部署 | ONNX Runtime 部署 |
|------|---------------|-----------------|
| 安装包大小 | >1 GB（含 CUDA 工具包） | ~50 MB |
| 依赖 | 需要完整 PyTorch 环境 | 只需 onnxruntime |
| 跨平台 | 仅 Python | C++、Java、C#、JS 均可调用 |
| 硬件加速 | CUDA | CUDA、TensorRT、DirectML、OpenVINO 等 |
| 推理速度 | 基准 | 通常快 1.2-2x（图优化、算子融合） |
| 边缘设备 | 几乎不可能 | Jetson、树莓派等可直接跑 |

### 8.2 导出与验证流程

```bash
# 1. 导出
yolo export model=runs/detect/train/weights/best.pt format=onnx simplify=True
# 输出 → models/best.onnx

# 2. 用 Netron 查看模型结构（可选，方便理解输入输出）
pip install netron
netron models/best.onnx

# 3. 推理验证
python scripts/inference_onnx.py --model models/best.onnx --image test.jpg

# 4. 精度对齐（PyTorch vs ONNX 的 mAP 差异应 < 0.01）
python scripts/evaluate.py --model models/best.onnx --format onnx
```

### 8.3 推理核心类设计（src/detector.py）

`YOLODetector` 类封装了完整的 ONNX 推理流程，三步 API 设计：

```python
import onnxruntime as ort
import cv2
import numpy as np

class YOLODetector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        self.session = ort.InferenceSession(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # 读取模型输入尺寸
        input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]
        self.input_size = (input_shape[2], input_shape[3])

    def preprocess(self, image):
        """BGR → RGB → Resize → Normalize → HWC→CHW → AddBatch"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0       # 归一化 0-1
        img = img.transpose(2, 0, 1)                # HWC → CHW
        img = np.expand_dims(img, axis=0)           # 添加 batch 维度
        return img

    def predict(self, image):
        """预处理 → ONNX 推理 → 后处理（过滤 + NMS）"""
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        # 解析输出 [1, 4+nc, 8400] → N 个检测框
        return self._postprocess(outputs[0], image.shape)

    def draw(self, image, detections, class_names):
        """在图片上画框 + 类名 + 置信度"""
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = f"{class_names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, label, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image
```

**设计目的**：`scripts/inference_onnx.py` 和 `api/app.py`（FastAPI）都 `from src.detector import YOLODetector`，推理逻辑只写一份。

### 8.4 预处理踩坑清单

| 步骤 | 常见错误 | 正确做法 |
|------|---------|---------|
| 颜色空间 | OpenCV 读的是 BGR，直接送模型 | 必须 BGR → RGB |
| 归一化 | 忘记除以 255 | `img / 255.0`，float32 |
| 维度顺序 | NumPy 默认 HWC (H,W,3) | 必须转成 CHW (3,H,W) |
| Batch 维度 | 3 维张量直接送模型 | 必须 `expand_dims` 加 batch 维 |
| 数据类型 | int8 / float64 | 模型期望 float32 |
| Resize 方式 | 直接 resize 不保持比例 | 可以 letterbox（加灰边保持比例），也可以直接 resize（简单项目够用） |

### 8.5 ONNX 性能对比表格模板

```markdown
| 格式 | mAP@0.5 | FPS (CPU) | FPS (GPU) | 模型大小 |
|------|---------|-----------|-----------|----------|
| PyTorch (.pt) | 0.xx | xx | xx | xx MB |
| ONNX (.onnx) | 0.xx | xx | xx | xx MB |
| 差异 | < 0.01 | +xx% | +xx% | -xx% |
```

---

## 九、数据准备（NEU-DET 项目专用）

### 9.1 数据集概况

| 项目 | 说明 |
|------|------|
| 名称 | NEU-DET（东北大学钢材表面缺陷数据库） |
| 图片总数 | 1,800（每类 300） |
| 图片尺寸 | 200×200 像素，灰度图 |
| 类别 | 6 类：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| 标注格式 | VOC XML（需转换为 YOLO TXT） |
| 划分 | 已预划分：train (~1440) / validation (~360) |
| 数据集大小 | ~28MB，可直接放在 Git 仓库 |

### 9.2 类别映射（顺序固定）

| 类别 | ID | 中文 | 检测难度 | 视觉特征 |
|------|----|------|---------|---------|
| crazing | 0 | 龟裂 | 高 | 细密裂纹网络，与背景区分度低 |
| inclusion | 1 | 夹杂 | 中 | 嵌入的异物颗粒 |
| patches | 2 | 斑块 | 中 | 不规则变色区域 |
| pitted_surface | 3 | 麻面 | 中 | 表面小凹坑 |
| rolled-in_scale | 4 | 压入氧化铁皮 | 中 | 轧制时压入的氧化皮 |
| scratches | 5 | 划痕 | 低 | 线性痕迹，特征明显 |

### 9.3 数据转换踩坑

- **不要自己做 train/val 划分**：NEU-DET 已经预划分好了，直接用
- **`rolled-in_scale` 的连字符陷阱**：文件名用 `_` 分隔类名和编号（如 `rolled-in_scale_1.jpg`），但类名内部有 `-`。用 `split('_')[0]` 提取类名会得到 `rolled-in`（错误）。正确做法：用已知类名列表做最长前缀匹配
- **目录结构不对称**：annotations 是扁平目录（所有 XML 混在一起），images 按类名分子目录。脚本中需要特殊处理
- **YOLO 的扁平目录要求**：输出的 `images/train/` 必须是扁平目录（不能有子目录），需要把按类名分的图片复制到同一层

---

## 十、项目文件观

### 10.1 一次训练真正需要哪些文件？

**最小闭环**：

1. **YOLO 格式数据**：`data/images/train/`、`data/images/val/`、`data/labels/train/`、`data/labels/val/`、`data/data.yaml`
2. **训练配置文件**：`configs/train_config.yaml` 或某个实验配置，如 `configs/exp1.yaml`
3. **训练入口脚本**：`scripts/train.py`
4. **初始模型权重**：`yolov8n.pt`
5. **Python 环境与依赖**：`ultralytics`、`torch`、`opencv`、`yaml` 等

**注意**：`data/NEU-DET/` 是**原始数据源**，只在"数据准备阶段"必须；`runs/detect/` 里的旧权重、旧 CSV、旧图表不是新训练的前置依赖，它们是**历史实验产物**。

### 10.2 训练流水线全景

```
原始数据 (data/NEU-DET/)
    ↓ scripts/prepare_data.py
YOLO 格式数据 (data/images/ + data/labels/ + data/data.yaml)
    ↓ scripts/train.py + configs/*.yaml
训练产物 (runs/detect/<exp>/)
    ├── weights/best.pt
    ├── results.csv / results.png
    ├── confusion_matrix*.png
    └── BoxPR_curve.png
    ↓ scripts/evaluate.py + 手动归档
评估记录 (docs/experiment_log.md) + 展示图表 (docs/assets/)
    ↓ scripts/export_onnx.py
ONNX 模型 (models/best.onnx)
    ↓ src/detector.py + api/app.py
推理服务 (FastAPI + Docker)
```

**最终链路一句话**：原始数据 → YOLO 格式数据 → 训练生成 best.pt → 评估与记录 → 导出 ONNX → 推理/服务化

### 10.3 runs/detect/ 和 docs/assets/ 的区别

| 目录 | 定位 | Git 跟踪 | 文件名 |
|------|------|---------|--------|
| `runs/detect/` | 训练程序自动生成的**实验现场**，内容最完整 | `.gitignore`，不进入 Git | 自动命名 |
| `docs/assets/` | 人工挑出来保存的**展示版成果** | 进入 Git | 整理成 `results_exp1.png` 等清晰形式 |

### 10.4 当前项目目录速查

| 目录/文件 | 作用 |
|----------|------|
| `data/NEU-DET/` | 原始数据集（VOC XML + 原始图片） |
| `data/images/` + `data/labels/` + `data/data.yaml` | YOLO 真正训练时读取的数据 |
| `configs/` | baseline 和各组实验配置 |
| `scripts/` | 各阶段一次性 CLI 脚本入口 |
| `src/` | 可复用模块，目前主要是 ONNX 推理核心类 |
| `runs/detect/` | 每次训练自动生成的实验产物 |
| `docs/assets/` | 复制保存的关键图表，用于 Git 跟踪与展示 |
| `docs/experiment_log.md` | 横向对比各实验结果 |
| `docs/YOLO_Project.md` | 项目推进笔记、学习笔记、用户分析入口 |
| `models/` | 导出的 ONNX 模型 |
| `api/` | FastAPI 服务代码 |

### 10.5 6 层文件观（记忆版）

```
1. 原始输入层  → data/NEU-DET/
2. 训练输入层  → data/images/、data/labels/、data/data.yaml
3. 实验定义层  → configs/*.yaml
4. 执行层      → scripts/*.py
5. 训练产物层  → runs/detect/*
6. 总结展示层  → docs/*
```

如果面试时能把这 6 层讲顺，项目的结构感会非常清楚。



---

## 十一、FastAPI 服务化入门速查

### 11.1 YOLO、ONNX、FastAPI、Docker 如何协同工作

这四个东西不是四套互相独立的技术，而是一条**从训练到部署再到对外提供服务**的完整流水线。

| 组件 | 负责什么 | 本项目里的对应产物 |
|------|---------|----------------|
| **YOLO** | 训练出“会检测缺陷”的模型 | `best.pt` |
| **ONNX** | 把训练好的模型转成更适合部署的通用格式 | `best.onnx` |
| **FastAPI** | 把模型封装成 HTTP 接口，别人不用 import Python 代码也能调用 | `api/app.py` |
| **Docker** | 把代码、依赖、模型和启动方式一起打包，换台机器也能跑 | `Dockerfile` |

**从项目全链路看，它们的关系是：**

```text
NEU-DET 数据集
    ↓
YOLOv8 训练
    ↓
得到 best.pt
    ↓
导出 ONNX
    ↓
得到 best.onnx
    ↓
src/detector.py 用 ONNX Runtime 做推理
    ↓
FastAPI 暴露 /health 和 /detect
    ↓
Docker 打包整个服务
    ↓
其他人通过 HTTP 请求调用检测能力
```

**如果从“一张图片进入系统后发生了什么”来看，工作流是：**

```text
客户端上传图片
    ↓
FastAPI 接收文件
    ↓
OpenCV 解码图片
    ↓
YOLODetector 预处理
    ↓
ONNX Runtime 跑 best.onnx
    ↓
NMS 后处理
    ↓
FastAPI 返回 JSON 检测结果
    ↓
Docker 保证这套流程在别的机器上也能跑
```

**面试一句话**：我先用 YOLO 完成模型训练，再把最优权重导出成 ONNX，用 `detector.py` 封装 ONNX Runtime 推理流程，用 FastAPI 暴露 HTTP 接口，最后用 Docker 打包整套服务，形成训练 → 部署 → 服务化 → 容器化的完整闭环。

### 11.2 FastAPI 到底解决什么问题

在这个项目里，FastAPI 的作用不是“训练模型”，而是把已经写好的推理逻辑封装成一个可以通过 HTTP 调用的服务接口。这样前端、测试脚本、产线系统都不需要直接 import Python 代码，只要发请求就能拿到检测结果。

**一句话理解**：FastAPI 是把 `src/detector.py` 里的推理能力，包装成 `api/app.py` 里的可调用接口。

### 11.3 最小 FastAPI 程序（先看懂入口和路由）

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "YOLO defect API is running"}
```

这几行的含义：

- `FastAPI()`：创建应用对象，后面所有路由都挂在这个对象上
- `@app.get("/")`：声明一个 GET 路由，请求 `/` 时会进入下面这个函数
- `def root()`：真正处理请求的 Python 函数
- `return {...}`：FastAPI 会自动把字典转成 JSON 响应

启动后常见地址：

- `/`：最简单的健康检查入口
- `/docs`：Swagger 文档页，可直接在线调接口
- `/redoc`：另一种风格的接口文档页

**这一节记住什么**：

- `app = FastAPI()` 是入口
- `@app.get(...)` 是路由装饰器
- FastAPI 的基本思路就是“URL -> Python 函数 -> JSON 响应”

### 11.4 Path Parameters（路径参数）：定位你要操作的资源

路径参数就是 URL 路径里变化的那一段，通常表示“你要处理哪一个对象”。

```python
@app.get("/images/{image_name}")
def read_image(image_name: str):
    return {"image_name": image_name}
```

访问：

```text
/images/crazing_241.jpg
```

这里的 `crazing_241.jpg` 会传给函数里的 `image_name`。

如果写成带类型注解的形式：

```python
@app.get("/runs/{run_id}")
def read_run(run_id: int):
    return {"run_id": run_id}
```

FastAPI 会自动做三件事：

- 把字符串转成你声明的类型
- 校验类型是否合法
- 自动把参数信息写进接口文档

**顺序注意**：固定路径要写在参数路径前面。

```python
@app.get("/runs/latest")
def read_latest():
    return {"run": "final_train_2"}

@app.get("/runs/{run_name}")
def read_run(run_name: str):
    return {"run_name": run_name}
```

否则 `/runs/latest` 可能会被误当成 `run_name="latest"`。

**这一节记住什么**：

- `{xxx}` 表示路径参数
- 路径参数更适合表达“是谁、哪个资源、哪条记录”
- 固定路由写前面，通用参数路由写后面

### 11.5 Query Parameters（查询参数）：控制怎么查、怎么返回

查询参数写在 URL 的 `?` 后面，通常表示“怎么查、怎么筛、怎么控制返回结果”。

```python
@app.get("/predict")
def predict(conf: float = 0.25, iou: float = 0.45, show_label: bool = True):
    return {"conf": conf, "iou": iou, "show_label": show_label}
```

访问：

```text
/predict?conf=0.30&iou=0.50&show_label=false
```

这里：

- `conf`：控制置信度阈值
- `iou`：控制 NMS 阈值
- `show_label`：控制是否返回标签信息

为什么它适合做查询参数？因为这些值不是在“指定哪张图”，而是在“控制这次推理怎么执行”。

默认值的含义：

- 有默认值：可不传，不传就走默认设置
- 默认值是 `None`：可选参数
- 没有默认值：必填参数

例如：

```python
@app.get("/search")
def search_images(keyword: str | None = None, limit: int = 10):
    return {"keyword": keyword, "limit": limit}
```

**这一节记住什么**：

- `?` 后面的是查询参数
- 查询参数适合表达筛选、分页、阈值、开关等控制项
- FastAPI 会自动完成类型转换和校验

### 11.6 Path 和 Query 的区别（部署时最容易混的点）

可以用一句话区分：

- **Path Parameters**：表示“你要处理谁”
- **Query Parameters**：表示“你想怎么处理”

对照看最清楚：

```text
/images/crazing_241.jpg
```

这里 `crazing_241.jpg` 是路径参数，因为它在定位资源。

```text
/predict/crazing_241.jpg?conf=0.25&save_vis=true
```

这里：

- `crazing_241.jpg` 是路径参数，表示哪张图
- `conf=0.25`、`save_vis=true` 是查询参数，表示这次推理的控制选项

**面试一句话**：Path 用来定位资源，Query 用来控制查询或返回方式。

### 11.7 一个贴近本项目的综合例子

下面这个例子已经接近后面 `api/app.py` 的思路了，只是为了先理解 FastAPI 的参数解析机制：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/predict/{image_name}")
def predict_image(
    image_name: str,
    conf: float = 0.25,
    iou: float = 0.45,
    save_vis: bool = False
):
    return {
        "image_name": image_name,
        "conf": conf,
        "iou": iou,
        "save_vis": save_vis
    }
```

访问：

```text
/predict/crazing_241.jpg?conf=0.30&save_vis=true
```

这一条请求里：

- `image_name` 是路径参数，表示推理哪张图
- `conf` 和 `iou` 是查询参数，表示阈值怎么设
- `save_vis` 是查询参数，表示要不要保存可视化结果

在真实项目里，函数内部就会继续调用：

```python
detector.predict(...)
```

然后把检测框、类别、置信度整理成 JSON 返回给调用方。

### 11.8 本项目里的最小 API 闭环

对这个项目来说，FastAPI 不是“重新做一套推理”，而是把已经存在的 ONNX 推理能力包装成 HTTP 接口。

最小闭环就是两条路由：

- `GET /health`：检查服务和模型是否就绪
- `POST /detect`：上传图片并返回检测结果 JSON

这样一来，项目就从“只能本地跑脚本”升级成了“可以被前端、测试脚本、产线系统调用的服务”。

**面试一句话**：FastAPI 这一步的价值不在于模型更准，而在于把推理能力服务化，证明项目具备训练→部署→服务化的工程闭环。

### 11.9 FastAPI 和 Flask 怎么区分（面试高频）

**FastAPI 更适合当前这个项目的原因：**

- **类型注解友好**：请求参数和返回结构更清晰
- **自动参数校验**：少写很多手工判断
- **自动文档**：`/docs` 能直接调接口，演示和排错都很快
- **更适合接口型项目**：做模型服务展示时性价比高

**Flask 的特点：**

- 更轻量、更经典
- 自由度高
- 但很多参数校验和接口文档能力需要自己补

**面试回答思路**：不是说 Flask 不行，而是 FastAPI 更适合这种“快速搭建模型接口、还要顺手展示文档和类型定义”的场景。

### 11.10 为什么推理逻辑不直接写在 `app.py`

这是一个典型的**关注点分离**问题：

- `app.py`：负责 Web 层，请求、响应、状态码、日志
- `src/detector.py`：负责模型层，预处理、ONNX Runtime 前向、NMS、结果组织

这么拆的好处：

- CLI 脚本和 FastAPI 服务复用同一个 `YOLODetector`
- 以后改推理逻辑时，不用改 Web 层代码
- 更容易单测、排障和维护

**面试一句话**：`app.py` 管请求，`detector.py` 管推理，这样代码才可复用、可测试、可维护。

### 11.11 如果有 100 个并发请求，你的 API 会怎样

这题不要只背 “GIL” 两个字，要先说清现象和优化方向。

当前这版 API 是**最小可用版本**，本地通常用：

```bash
uvicorn api.app:app --reload
```

这意味着它更偏开发调试，而不是生产高并发部署。

如果突然来 100 个并发请求，典型现象是：

- 客户端总响应时间明显上升
- 请求开始排队
- 吞吐量很快碰到瓶颈

原因不是模型单次推理一定慢，而是：

- 单 worker 开发模式本身不适合高并发
- 推理任务是重计算型工作
- 总响应时间包含了排队、调度、JSON 序列化等开销

优化方向通常是：

- 增加 worker 数
- 去掉 `--reload`
- 用更正式的部署方式
- 必要时做请求队列或异步任务拆分

**面试一句话**：100 并发时，当前单 worker 开发版会排队、延迟上升；优化重点应该放在服务并发处理能力，而不是先怀疑模型前向速度。

### 11.12 今日 FastAPI 实测数据（2026-03-29）

今天在本地拿到的关键结果：

- `/health`：返回 `200 OK`，`status=ok`
- `/detect`：对 `crazing_241.jpg` 返回 `count=3`
- `benchmark_api.py`（10 张图、并发 10）：
  - 成功请求数：`10/10`
  - 平均客户端响应时间：`2333.37 ms`
  - 总耗时：`3.06 s`
  - 吞吐量：`3.27 QPS`

**怎么解释这些数据：**

- 客户端总响应时间 ≠ 纯模型推理时间
- 大多数服务端 `inference_time_ms` 只有 `13-23 ms`
- 第一条请求达到 `2198.88 ms`，更像是冷启动 / 排队 / 本地开发模式的影响

**面试回答方向**：这组数据说明接口稳定性已经通过，但当前瓶颈更像服务并发处理能力，而不是模型单次前向本身。

**这一章压缩记忆版**：

- FastAPI 的本质是把 Python 函数变成 HTTP API
- `app = FastAPI()` 是应用入口
- `@app.get("/xxx")` 定义 GET 路由
- Path 参数负责定位资源
- Query 参数负责控制查询方式和返回形式
- 对这个项目来说，FastAPI 是把 ONNX 推理能力服务化的关键一步
- `app.py` 管请求，`detector.py` 管推理
- 客户端总响应时间和纯模型推理时间要分开看
- 面试里讲 FastAPI，重点是工程闭环和服务化价值，不是背框架语法

---

## 十二、Docker 容器化部署（Day 3 前置知识）

### 12.1 Docker 解决什么问题

你的 YOLO 项目现在能在你电脑上跑，但换一台电脑可能跑不了——Python 版本不对、`onnxruntime` 没装、`fastapi` 版本冲突……Docker 的作用就是**把你的代码 + 所有依赖 + 运行环境打成一个包**，别人拿到这个包直接就能跑，不用管他电脑装了什么。

**类比理解**：

```
没有 Docker：
  你："你先装 Python 3.9，再 pip install 这 8 个包，注意版本号……"
  面试官：环境报错了，算了不看了。

有了 Docker：
  你："docker run --rm -p 8000:8000 yolo-defect-api"
  面试官：跑起来了，curl 一下试试。
```

**面试一句话**：Docker 保证了"我机器上能跑 = 任何机器上都能跑"，消除了环境差异导致的部署问题。

### 12.2 三个核心概念

| 概念 | 类比 | 说明 |
|------|------|------|
| **镜像（Image）** | 安装光盘 | 一个只读的模板，包含 OS + 依赖 + 你的代码。`docker build` 产出的就是镜像 |
| **容器（Container）** | 正在运行的虚拟机 | 从镜像启动的一个实例，是活的进程。`docker run` 产出的就是容器 |
| **Dockerfile** | 安装说明书 | 一个文本文件，一行一行告诉 Docker "怎么构建这个镜像" |

**关系链**：

```
Dockerfile（说明书）
    ↓ docker build
镜像 Image（安装光盘）
    ↓ docker run
容器 Container（正在运行的实例）
```

**面试一句话**：Dockerfile 定义构建步骤，`docker build` 生成镜像，`docker run` 从镜像启动容器。镜像是静态的，容器是动态的。

### 12.3 Dockerfile 逐行解读（本项目实际使用）

```dockerfile
FROM python:3.9-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt
COPY src/ src/
COPY api/ api/
COPY models/ models/
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**逐行解释**（面试要能口头讲清楚每一行）：

| 行 | 指令 | 做了什么 | 类比 |
|----|------|---------|------|
| 1 | `FROM python:3.9-slim` | 选一个干净的 Python 3.9 基础环境，`slim` 版更轻量 | 选一台预装了 Python 的干净电脑 |
| 2 | `WORKDIR /app` | 设置工作目录，后续命令都在 `/app` 下执行 | 先 `cd /app` |
| 3-4 | `ENV ...` | 不生成 `.pyc`，日志直接输出到终端，便于看容器日志 | 提前把运行习惯设好 |
| 5-7 | `RUN apt-get ...` | 安装 OpenCV 在 Linux 运行所需的系统库 | 先补基础运行环境 |
| 8 | `COPY requirements-api.txt .` | 把部署专用依赖清单复制到镜像里 | 先把购物清单递进去 |
| 9-10 | `RUN pip install ...` | 安装 API 推理所需的最小依赖 | 按清单把材料装好 |
| 11-13 | `COPY src/ api/ models/` | 把你的代码和模型复制到镜像里 | 把项目文件搬进去 |
| 14 | `EXPOSE 8000` | 声明容器使用 8000 端口（只是文档标注，不实际开端口） | 门牌号 |
| 15 | `CMD [...]` | 容器启动时执行的命令：用 uvicorn 启动 FastAPI 服务 | 按下电源键后自动运行的程序 |

### 12.4 为什么先 COPY requirements-api.txt 再 COPY 代码？

这是 Docker 面试最常见的追问。答案是**层缓存（Layer Cache）机制**。

```
Dockerfile 每一行都会生成一个"层"（Layer）。
Docker 构建时，如果某一层的输入没变，就直接用缓存，跳过该层。

COPY requirements-api.txt .     ← 依赖清单不常变
RUN pip install ...         ← 这层缓存住了，不用重装（省 2-3 分钟）
COPY src/ src/              ← 你改了代码，只有这层和之后的层重新执行
COPY api/ api/              ← 重新复制（几秒钟）
```

**如果反过来写（先复制代码再装依赖）**：

```
COPY . .                    ← 你改了任何一行代码，这层就失效
RUN pip install ...         ← 缓存失效，每次都要重装（2-3 分钟白等）
```

**面试一句话**：把不常变的层（依赖安装）放前面，常变的层（代码复制）放后面，利用 Docker 层缓存加速重复构建。
### 12.4b 我现在应该怎样理解 Dockerfile 的顺序

不要把 Dockerfile 看成一堆零散命令，而要把它看成一条“从环境到服务”的流水线：

```text
先准备 Python 基础环境
→ 再设定工作目录
→ 再优化 Python 在容器里的运行方式
→ 再安装 Linux 系统依赖
→ 再复制部署依赖文件并安装 Python 依赖
→ 再复制 API 所需的代码和模型
→ 再声明监听端口
→ 最后定义容器启动命令
```

**为什么先 `FROM` 再 `WORKDIR`？**

- `FROM python:3.9-slim` 解决的是“我用什么基础环境作为起点”
- `WORKDIR /app` 解决的是“我后续默认在哪个目录下工作”
- 逻辑上一定是先有这台“装好 Python 的电脑”，再决定默认在哪个文件夹办公

**为什么 Python 依赖不在基础镜像里一起装好？**

- `python:3.9-slim` 是一个通用基础镜像，只负责提供 Linux + Python 3.9
- `fastapi`、`uvicorn`、`onnxruntime`、`opencv-python-headless` 这些都是**当前项目自己的依赖**
- 所以基础镜像解决“能运行 Python”，`pip install` 解决“能运行这个项目”

**我现在可以这样口头概括 Dockerfile 顺序：**

> 先准备 Python 基础环境，然后设定工作目录，并通过环境变量优化 Python 在容器里的运行方式。接着安装 OpenCV 所需的 Linux 系统依赖，再复制部署依赖文件并安装 Python 依赖。之后复制 API 所需的代码和模型，声明服务使用的监听端口，最后通过 uvicorn 启动 FastAPI 服务。

**注意最后不要漏掉 `CMD`：**

- `EXPOSE 8000` 只是声明服务端口
- `CMD ["uvicorn", ...]` 才是容器启动后真正执行的命令

**面试一句话**：Dockerfile 的本质顺序就是“先搭环境，再装依赖，再放应用，最后启动服务”。

### 12.5 端口与端口映射

**端口是什么？** 一台电脑上可以同时跑多个网络服务（网页、数据库、API……），端口号用来区分"这个请求是发给谁的"。你的 FastAPI 监听 8000 端口，意思是"发到 8000 端口的请求由我来处理"。

**为什么需要映射？** 容器是一个隔离环境，有自己的网络。容器内部的 8000 端口，外面默认访问不到。`-p` 参数把宿主机（你的电脑）的端口和容器内的端口连起来。

```
你的电脑（宿主机）              Docker 容器
┌─────────────────┐          ┌──────────────────┐
│                 │          │                  │
│  浏览器/curl    │          │  FastAPI 服务     │
│  访问 localhost:8000  ──────→  监听 8000 端口   │
│                 │   -p     │                  │
│  端口 8000      │  映射     │  端口 8000        │
└─────────────────┘          └──────────────────┘

命令：docker run --rm -p 8000:8000 yolo-defect-api
                  ↑      ↑
              宿主机端口  容器内端口
```

**面试一句话**：`-p 8000:8000` 表示把宿主机的 8000 端口映射到容器内的 8000 端口，这样外部的请求才能到达容器内的服务。

### 12.6 四条命令够用

| 命令 | 作用 | 你什么时候用 |
|------|------|------------|
| `docker build -t yolo-defect-api .` | 根据当前目录的 Dockerfile 构建镜像，命名为 `yolo-defect-api` | 写完 Dockerfile 后第一步 |
| `docker run --rm -p 8000:8000 yolo-defect-api` | 从镜像启动容器，映射端口 | 构建成功后，启动服务 |
| `docker ps` | 查看正在运行的容器（含 Container ID） | 确认容器跑起来了 |
| `docker stop <container_id>` | 停止指定容器 | 测试完毕后关掉 |

**Day 3 的完整操作流程**：

```bash
# 1. 构建镜像（在项目根目录执行，Dockerfile 在这里）
docker build -t yolo-defect-api .

# 2. 启动容器
docker run --rm -p 8000:8000 yolo-defect-api

# 3. 另开一个终端，测试 API
curl http://localhost:8000/health
# 应返回：{"status": "ok", "model": "best.onnx"}

curl -X POST http://localhost:8000/detect \
  -F "file=@data/images/val/crazing_241.jpg"
# 应返回：{"detections": [...], "count": N, "inference_time_ms": XX}

# 4. 测试完毕，停止容器
docker ps                    # 找到 Container ID
docker stop <container_id>   # 停止
```

### 12.7 Docker vs 虚拟机

面试偶尔会追问"Docker 和虚拟机有什么区别"。

| 维度 | Docker 容器 | 虚拟机（VM） |
|------|------------|------------|
| 启动速度 | 秒级 | 分钟级 |
| 体积 | 几十 MB ~ 几百 MB | 几 GB |
| 隔离级别 | 进程级（共享宿主机内核） | 硬件级（独立操作系统） |
| 性能损耗 | 几乎为零 | 10-20% |
| 适用场景 | 应用部署、微服务 | 需要完整 OS 隔离的场景 |

**面试一句话**：Docker 是轻量级的进程隔离，共享宿主机内核所以启动快、开销小；虚拟机是完整的 OS 隔离，更重但隔离更彻底。部署 API 服务用 Docker 就够了。

### 12.8 为什么简历项目要用 Docker

面试问"Docker 部署有什么好处"，回答这三点：

1. **环境一致性**：开发环境、测试环境、生产环境完全相同，不会出现"我这跑得好好的"问题
2. **一键部署**：`docker run` 一行命令启动服务，不需要对方手动配环境
3. **可复现性**：Dockerfile 就是部署文档，任何人看到 Dockerfile 就知道这个项目需要什么依赖、怎么启动

对于你的 YOLO 简历项目，Docker 的核心价值是**证明你具备"训练 → 导出 → 服务化 → 容器化"的完整工程闭环能力**，这是工业外企最看重的。

### 12.9 现在不需要知道的（面试不会问）

| 概念 | 为什么不需要 | 什么时候学 |
|------|------------|----------|
| docker-compose | 多容器编排（如 API + 数据库），你的项目只有一个服务 | 实习期间如果项目需要 |
| 多阶段构建 | 优化镜像体积，生产级技巧 | V2 简历时可加 |
| Docker Volume | 数据持久化，你的模型当前默认打进镜像 | 需要动态更新模型时 |
| Docker Network | 容器间网络通信 | 微服务架构时 |
| Kubernetes (K8s) | 容器集群管理，运维级别 | 工作后 |
| CI/CD 集成 | GitHub Actions + Docker 自动构建 | V2 简历加分项 |


---

## 十三、常见踩坑清单

### 13.1 训练阶段

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| mAP 停在 0.3-0.4 | imgsz 太小（原图 200px 放到 640 会模糊） | 试 512 或保持 640，确认增强策略合理 |
| loss 震荡不收敛 | 学习率太大 | 降到 0.001，或用 AdamW |
| 某个类 AP 特别低 | 该类样本少或特征不明显 | 分析混淆矩阵，针对性增强 |
| 训练 OOM (显存不够) | batch 太大或 imgsz 太大 | 减小 batch，或 `batch=-1` 自动选择 |
| Windows 上 DataLoader 卡死 | workers > 0 在 Windows 上有问题 | 设 `workers=0` |

### 13.2 ONNX 阶段

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ONNX 精度比 PyTorch 低很多 | 导出时 simplify 出问题 | 试 `simplify=False`，对比两者 |
| ONNX Runtime 推理报错 | 输入 shape 或 dtype 不对 | 用 `session.get_inputs()` 确认模型期望的输入 |
| 推理结果全是空 | 后处理代码 bug | 先不做置信度过滤，看原始输出是否有值 |
| C++ 和 Python 结果不一致 | 预处理不一致（BGR/RGB、归一化方式） | 对齐每一步的中间结果 |

---

## 十四、面试高频题集

### 14.1 基础概念（必答）

**Q: YOLO 和 Faster RCNN 最大的区别？**

> YOLO 是一阶段检测器，一次前向传播直接输出所有框，速度快适合实时场景。Faster RCNN 是两阶段，先用 RPN 生成候选区域再精细分类，精度略高但速度慢。工业部署通常选 YOLO。

**Q: IoU 是什么？怎么计算？**

> 两个框的交集面积除以并集面积，衡量框的重叠程度。交集 = 两个框重叠区域面积，并集 = A面积 + B面积 - 交集。IoU=1 表示完全重合，IoU=0 表示无重叠。

**Q: NMS 的作用和流程？**

> 去除对同一目标的重复检测。流程：1) 按置信度排序；2) 取最高分的框；3) 计算它与剩余框的 IoU；4) IoU 超过阈值的框被删除（认为检测的是同一目标）；5) 重复直到处理完。

**Q: mAP@0.5 = 0.70 怎么解释？**

> 在 "预测框和真实框重叠超过 50% 才算对" 的标准下，模型在所有类别上的平均检测准确率是 70%。

**Q: YOLOv8 输出 [1, 10, 8400] 是什么意思？**

> 1 是 batch size；10 = 4 个坐标 + 6 个类别概率；8400 是三个检测尺度的网格数之和 (80² + 40² + 20²)。

### 14.2 项目深入（加分项）

**Q: 为什么选 YOLOv8n？**

> NEU-DET 只有 1800 张图，数据量小，大模型容易过拟合。nano 版本 3.2M 参数，边缘设备也能实时跑。不够的话换 v8s 只需改一行配置。

**Q: ONNX 导出后精度有损失怎么办？**

> 先检查 simplify 是否有问题，试 simplify=False。如果还有差异，逐步对比预处理和后处理的中间结果，定位是算子转换问题还是浮点精度差异。差异 < 0.01 通常可接受。

**Q: 在 C++ 部署中，NMS 怎么实现？**

> 两种方案：1) OpenCV 的 `cv::dnn::NMSBoxes()`，开箱即用；2) 手写 NMS，完全可控。优先用 OpenCV 版本，如果有特殊需求（如 Soft-NMS、类别感知 NMS）再手写。

**Q: 怎么处理类别不平衡？**

> NEU-DET 每类 300 张，相对均衡。如果不均衡，可以：1) 过采样少数类；2) 调整损失函数权重（`cls_pw` 参数）；3) Focal Loss（降低易分类样本的损失权重）。

**Q: 怎么分析误检原因？**

> 看混淆矩阵找易混淆类对，看漏检图片分析视觉特征（如 crazing 纹理太细），看置信度分布判断是模型能力不足还是阈值设置问题。

---

## 十五、学习路线与踩点建议

### 15.1 推荐学习顺序

```
第 0 步（2h）：概念速通
  IoU → NMS → mAP → YOLO 直觉 → YOLOv8 输出形状
  ✅ 你已经完成了

第 1 步（Day 1-2）：数据准备 + Baseline 训练
  跑通 prepare_data.py → 训练第一个模型 → 记录 mAP
  重点：data.yaml 格式、ultralytics CLI

第 2 步（Day 3-4）：调参 + 结果分析
  imgsz / lr / mosaic 对比实验 → PR 曲线 + 混淆矩阵分析
  重点：一次只改一个变量，表格记录

第 3 步（Day 5）：推理脚本 + README
  写 detector.py → 可视化推理结果 → 写 README
  重点：理解输出 tensor 的 shape 和后处理流程

第 4 步（Day 6）：ONNX 导出 + 推理验证
  导出 → ONNX Runtime 推理 → 精度对齐 → FPS 对比
  重点：预处理每一步对齐

第 5 步（Day 7）：SAM 集成 + 周复盘
  YOLO 框 → SAM 精细分割 → 对比可视化
  重点：展示级，简历亮点
```

### 15.2 关键踩点提醒

- **不要纠结理论**：YOLO 的损失函数（CIoU + DFL + BCE）ultralytics 全封装了，面试时知道"分类损失 + 回归损失 + 置信度损失"三部分就够
- **不要手写训练循环**：直接用 `model.train()`，YOLO 不像 CIFAR-10 需要你自己写 epoch 循环
- **一定要做实验对比**：至少 3 组调参实验，面试官看的是"系统性调参思路"而不是"最终数值"
- **先跑通再优化**：baseline 哪怕 mAP 只有 0.5 也先记录，不要卡在调参上
- **ONNX 预处理要精确对齐**：Python 和 C++ 的推理结果差异 99% 来自预处理不一致

---

## 十六、适合的岗位与工作方向

### 16.1 YOLO 技能对应的岗位

| 岗位方向 | 公司类型 | 核心技能要求 | YOLO 项目的价值 |
|---------|---------|------------|--------------|
| CV 算法工程师 | 博世、西门子、Cognex | YOLO/检测 + 数据分析 + 调参 | 直接对口，核心项目 |
| AI 部署工程师 | KLA、应用材料 | ONNX/TensorRT + C++ + 边缘设备 | ONNX 导出 + C++ 推理经验 |
| 工业视觉工程师 | 海克斯康、ABB | 缺陷检测 + 光学 + 产线集成 | 工业场景经验 + 数据闭环思路 |
| MLOps / 模型服务化 | 各类外企 IT | FastAPI + Docker + CI/CD | API 服务化 + 容器化部署 |
| 机器人视觉工程师 | ABB、发那科 | 目标检测 + 位姿估计 + ROS | YOLO 检测 + 部署能力 |

### 16.2 简历中的项目描述模板

```
工业缺陷检测系统（2026.03）
• 基于 YOLOv8 构建钢材表面缺陷检测系统，覆盖 6 类缺陷，mAP@0.5 = 0.XX
• 完成数据分析、模型调优（imgsz/lr/augment 对比实验）、误检案例分析
• 导出 ONNX，ONNX Runtime 推理精度对齐（差异 < 0.01），CPU 推理 XX FPS
• 基于 FastAPI 开发 RESTful 检测 API，Docker 容器化部署
• 技术栈：Python / PyTorch / YOLOv8 / ONNX / FastAPI / Docker
```

---

## 附录 A：ultralytics 常用函数速查

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')           # 加载预训练模型
model = YOLO('runs/.../best.pt')     # 加载训练好的模型

model.train(data=..., epochs=...)     # 训练
model.val()                           # 评估（返回 metrics 对象）
model.predict(source=..., conf=0.25)  # 推理
model.export(format='onnx')           # 导出

# metrics 对象常用属性
metrics.box.map50       # mAP@0.5
metrics.box.map         # mAP@50-95
metrics.box.maps        # 每个类别的 AP 列表
```

## 附录 B：ONNX Runtime 速查

```python
import onnxruntime as ort

# 创建推理 Session
session = ort.InferenceSession('model.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 查看输入输出信息
input_info = session.get_inputs()[0]
print(input_info.name, input_info.shape, input_info.type)
# 例: 'images', [1, 3, 640, 640], 'tensor(float)'

# 推理
outputs = session.run(None, {input_info.name: input_tensor})
# outputs[0].shape → [1, 10, 8400]
```

## 附录 C：关键文件路径速查（项目内）

| 文件 | 作用 |
|------|------|
| `data/data.yaml` | 数据集配置（路径 + 类名） |
| `configs/train_config.yaml` | 训练超参数 |
| `scripts/prepare_data.py` | VOC → YOLO 格式转换 |
| `scripts/train.py` | 训练入口 |
| `scripts/evaluate.py` | 评估 + 可视化 |
| `scripts/export_onnx.py` | ONNX 导出 |
| `scripts/inference_onnx.py` | ONNX 推理 |
| `src/detector.py` | 推理核心类（FastAPI 复用） |
| `docs/experiment_log.md` | 实验记录 |
| `runs/detect/train/weights/best.pt` | 最优权重 |
| `models/best.onnx` | 导出的 ONNX 模型 |

