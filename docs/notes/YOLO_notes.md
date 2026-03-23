
# YOLO 目标检测 — 从入门到项目落地学习笔记

> 作者背景：PyTorch 基础（CIFAR-10 分类 + ResNet 迁移学习）→ 目标检测入门 目标岗位：外企 CV 算法/部署工程师（博世、西门子、KLA 等） 配套项目：YOLOv8 钢材缺陷检测（NEU-DET 数据集） 创建日期：2026年3月23日

---

## 一、目标检测 vs 图像分类：到底变了什么

### 1.1 任务对比

|维度|图像分类（你已经会的）|目标检测（你要学的）|
|---|---|---|
|输入|一张图片|一张图片|
|输出|一个类别标签|**N 个检测结果**，每个包含位置+类别+置信度|
|典型输出形状|`[batch, num_classes]` 如 `[1, 10]`|`[batch, 4+nc, num_candidates]` 如 `[1, 10, 8400]`|
|回答的问题|"这张图是什么？"|"这张图里有什么？在哪里？有多确定？"|
|后处理|argmax 取最大概率|置信度过滤 + NMS 去重|
|典型模型|ResNet, VGG, EfficientNet|YOLO, Faster RCNN, DETR|
|损失函数|CrossEntropyLoss|分类损失 + 回归损失 + 目标置信度损失|

### 1.2 数据流对比

```
分类：图片 → Backbone → FC Layer → [1, 10] → argmax → "cat"

检测：图片 → Backbone → Neck(FPN) → Head → [1, 10, 8400] → 过滤+NMS → N个框
                                                                        ↓
                                                    每个框 = [x, y, w, h, class, confidence]
```

核心区别：检测多了 Neck（多尺度特征融合）和 Head（输出框坐标），以及后处理步骤。

### 1.3 从 ResNet 到 YOLOv8 的对应关系

|ResNet 分类|YOLOv8 检测|说明|
|---|---|---|
|ResNet Backbone|CSPDarknet Backbone|都是卷积堆叠提取特征，YOLO 用 C2f 模块替代 ResBlock|
|Global Average Pooling|FPN Neck|分类把特征压成一个向量；检测保留空间信息，做多尺度融合|
|FC Layer (512→10)|Decoupled Head|分类只输出类别概率；检测同时输出坐标+类别（解耦处理）|
|argmax|NMS|分类直接取最大值；检测要去除重复框|
|无|Anchor-free|YOLOv8 直接回归坐标，不依赖预定义的锚框|

---

## 二、核心概念速查表

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

|术语|含义|判定条件|
|---|---|---|
|TP（True Positive）|正确检测|预测框与某个真实框的 IoU ≥ 阈值|
|FP（False Positive）|误检|预测框找不到匹配的真实框（IoU 都 < 阈值），或重复匹配|
|FN（False Negative）|漏检|真实框没有被任何预测框匹配到|

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

|版本|年份|关键改进|一句话特点|
|---|---|---|---|
|YOLOv1|2016|开创性地将检测当作回归问题|第一个实时检测器，速度快但精度一般|
|YOLOv3|2018|多尺度预测（FPN）、Darknet-53|平衡了速度和精度，工业界开始广泛使用|
|YOLOv5|2020|PyTorch 实现、AutoAnchor、丰富的数据增强|工程化做得最好，社区最活跃|
|YOLOv8|2023|**Anchor-free、C2f 模块、解耦头**|当前主流，Ultralytics 官方维护|
|YOLOv9|2024|PGI（可编程梯度信息）|精度更高，但生态不如 v8 成熟|

**面试回答 "Why YOLOv8 over v5?"**：

- YOLOv8 是 Anchor-free 的，不需要预定义锚框大小，对不同尺度目标更灵活
- C2f 模块比 C3 更高效（更丰富的梯度流）
- 解耦头让分类和定位任务不互相干扰
- Ultralytics 官方维护，API 统一，训练/导出/部署一条龙

### 3.2 一阶段 vs 二阶段检测器

|维度|一阶段（YOLO）|二阶段（Faster RCNN）|
|---|---|---|
|流程|一次前向传播直接出框|先生成候选区域(RPN)，再分类+回归|
|速度|快（实时 30+ FPS）|慢（通常 5-10 FPS）|
|精度|略低（v8 已经很接近了）|略高（尤其是小目标）|
|适用场景|实时检测、边缘部署、工业产线|精度优先、不要求实时|
|代表|YOLOv8, SSD, RetinaNet|Faster RCNN, Mask RCNN, Cascade RCNN|

**面试一句话**：YOLO 是一阶段检测器，一次前向传播直接输出所有框，速度快适合工业部署；Faster RCNN 是二阶段，先用 RPN 生成候选区域再精细分类，精度更高但速度慢。

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

|变体|参数量|FLOPs|mAP@0.5 (COCO)|适用场景|
|---|---|---|---|---|
|YOLOv8n (nano)|3.2M|8.7G|37.3|**边缘部署、资源受限**，本项目首选|
|YOLOv8s (small)|11.2M|28.6G|44.9|精度和速度的平衡|
|YOLOv8m (medium)|25.9M|78.9G|50.2|服务器部署，精度优先|
|YOLOv8l (large)|43.7M|165.2G|52.9|高精度需求|
|YOLOv8x (xlarge)|68.2M|257.8G|53.9|最高精度，不考虑速度|

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

|参数|默认值|说明|调参建议|
|---|---|---|---|
|`model`|`yolov8n.pt`|模型尺寸，n/s/m/l/x|小数据集从 n 开始，精度不够换 s|
|`epochs`|50|训练轮数|看 loss 是否收敛，通常 50-200|
|`imgsz`|640|输入图片尺寸|更大=更准但更慢，常见调参变量|
|`batch`|16|批大小|-1 自动选择最大 batch，显存不够就减小|
|`lr0`|0.01|初始学习率|微调时可降到 0.001|
|`optimizer`|`auto`|优化器|auto 自动选择 SGD 或 AdamW|
|`mosaic`|1.0|Mosaic 增强概率|4 图拼 1 图，提升小目标检测|
|`mixup`|0.0|Mixup 增强概率|两图混合叠加，正则化效果|
|`close_mosaic`|10|最后 N 个 epoch 关闭 mosaic|让模型最后阶段适应正常图片|
|`patience`|50|早停耐心值|连续 N 个 epoch 没提升就停止|

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

|增强方式|效果|对比分类任务|
|---|---|---|
|**Mosaic**|4 张图拼成 1 张，每张占一个象限|分类没有这个。检测用它增加目标密度和上下文信息|
|**Mixup**|两张图按随机比例 α 混合叠加|类似分类的 Mixup，但标签是框不是类别|
|**随机翻转**|水平/垂直翻转|跟分类一样|
|**HSV 调整**|随机调色相/饱和度/亮度|类似 ColorJitter|
|**尺度抖动**|随机缩放输入尺寸|分类用 RandomResizedCrop，检测用全图缩放|

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

---

## 六、评估与结果分析

### 6.1 必看的 4 张图

|图表|文件|看什么|
|---|---|---|
|**训练曲线**|`results.png`|loss 是否收敛、是否过拟合（train loss 下降但 val loss 上升）|
|**PR 曲线**|`PR_curve.png`|每个类别的 AP，曲线越靠右上角越好|
|**混淆矩阵**|`confusion_matrix.png`|哪些类别容易互相误判（如 crazing ↔ scratches）|
|**验证集预测**|`val_batch0_pred.jpg`|直观看检测效果，找误检/漏检案例|

### 6.2 调参实验记录模板

每组实验只改一个变量，用 YAML 配置文件管理超参，用 git diff 对比两次实验的参数差异。

```markdown
| 实验 | imgsz | lr0 | epochs | mosaic | mAP@0.5 | mAP@50-95 | 训练时间 | 备注 |
|------|-------|-----|--------|--------|---------|-----------|----------|------|
| baseline | 640 | 0.01 | 50 | 1.0 | ? | ? | ? | YOLOv8n 默认 |
| exp1 | 512 | 0.01 | 50 | 1.0 | ? | ? | ? | 缩小输入 |
| exp2 | 800 | 0.01 | 50 | 1.0 | ? | ? | ? | 增大输入 |
| exp3 | 640 | 0.001 | 50 | 1.0 | ? | ? | ? | 降低学习率 |
| exp4 | 640 | 0.01 | 50 | 0.0 | ? | ? | ? | 关闭 mosaic |
```

### 6.3 误检案例分析思路

找 10 张误检/漏检图，按以下维度分析：

- 是哪个类漏检了？该类本身是否纹理细密、对比度低（如 crazing）？
- 误检的框预测成了什么类？两个类是否视觉上相似？
- 是否存在标注错误（数据质量问题）？
- 框位置偏了还是类别搞错了？→ 决定优化方向是调增强还是调模型

---

## 七、ONNX 部署

### 7.1 为什么需要 ONNX

|维度|PyTorch 直接部署|ONNX Runtime 部署|
|---|---|---|
|安装包大小|>1 GB（含 CUDA 工具包）|~50 MB|
|依赖|需要完整 PyTorch 环境|只需 onnxruntime|
|跨平台|仅 Python|C++、Java、C#、JS 均可调用|
|硬件加速|CUDA|CUDA、TensorRT、DirectML、OpenVINO 等|
|推理速度|基准|通常快 1.2-2x（图优化、算子融合）|
|边缘设备|几乎不可能|Jetson、树莓派等可直接跑|

### 7.2 导出与验证流程

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

### 7.3 推理核心类设计（src/detector.py）

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

### 7.4 预处理踩坑清单

|步骤|常见错误|正确做法|
|---|---|---|
|颜色空间|OpenCV 读的是 BGR，直接送模型|必须 BGR → RGB|
|归一化|忘记除以 255|`img / 255.0`，float32|
|维度顺序|NumPy 默认 HWC (H,W,3)|必须转成 CHW (3,H,W)|
|Batch 维度|3 维张量直接送模型|必须 `expand_dims` 加 batch 维|
|数据类型|int8 / float64|模型期望 float32|
|Resize 方式|直接 resize 不保持比例|可以 letterbox（加灰边保持比例），也可以直接 resize（简单项目够用）|

### 7.5 ONNX 性能对比表格模板

```markdown
| 格式 | mAP@0.5 | FPS (CPU) | FPS (GPU) | 模型大小 |
|------|---------|-----------|-----------|----------|
| PyTorch (.pt) | 0.xx | xx | xx | xx MB |
| ONNX (.onnx) | 0.xx | xx | xx | xx MB |
| 差异 | < 0.01 | +xx% | +xx% | -xx% |
```

---

## 八、数据准备（NEU-DET 项目专用）

### 8.1 数据集概况

|项目|说明|
|---|---|
|名称|NEU-DET（东北大学钢材表面缺陷数据库）|
|图片总数|1,800（每类 300）|
|图片尺寸|200×200 像素，灰度图|
|类别|6 类：crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches|
|标注格式|VOC XML（需转换为 YOLO TXT）|
|划分|已预划分：train (~1440) / validation (~360)|
|数据集大小|~28MB，可直接放在 Git 仓库|

### 8.2 类别映射（顺序固定）

|类别|ID|中文|检测难度|视觉特征|
|---|---|---|---|---|
|crazing|0|龟裂|高|细密裂纹网络，与背景区分度低|
|inclusion|1|夹杂|中|嵌入的异物颗粒|
|patches|2|斑块|中|不规则变色区域|
|pitted_surface|3|麻面|中|表面小凹坑|
|rolled-in_scale|4|压入氧化铁皮|中|轧制时压入的氧化皮|
|scratches|5|划痕|低|线性痕迹，特征明显|

### 8.3 数据转换踩坑

- **不要自己做 train/val 划分**：NEU-DET 已经预划分好了，直接用
- **`rolled-in_scale` 的连字符陷阱**：文件名用 `_` 分隔类名和编号（如 `rolled-in_scale_1.jpg`），但类名内部有 `-`。用 `split('_')[0]` 提取类名会得到 `rolled-in`（错误）。正确做法：用已知类名列表做最长前缀匹配
- **目录结构不对称**：annotations 是扁平目录（所有 XML 混在一起），images 按类名分子目录。脚本中需要特殊处理
- **YOLO 的扁平目录要求**：输出的 `images/train/` 必须是扁平目录（不能有子目录），需要把按类名分的图片复制到同一层

---

## 九、常见踩坑清单

### 9.1 训练阶段

|问题|原因|解决方案|
|---|---|---|
|mAP 停在 0.3-0.4|imgsz 太小（原图 200px 放到 640 会模糊）|试 512 或保持 640，确认增强策略合理|
|loss 震荡不收敛|学习率太大|降到 0.001，或用 AdamW|
|某个类 AP 特别低|该类样本少或特征不明显|分析混淆矩阵，针对性增强|
|训练 OOM (显存不够)|batch 太大或 imgsz 太大|减小 batch，或 `batch=-1` 自动选择|
|Windows 上 DataLoader 卡死|workers > 0 在 Windows 上有问题|设 `workers=0`|

### 9.2 ONNX 阶段

|问题|原因|解决方案|
|---|---|---|
|ONNX 精度比 PyTorch 低很多|导出时 simplify 出问题|试 `simplify=False`，对比两者|
|ONNX Runtime 推理报错|输入 shape 或 dtype 不对|用 `session.get_inputs()` 确认模型期望的输入|
|推理结果全是空|后处理代码 bug|先不做置信度过滤，看原始输出是否有值|
|C++ 和 Python 结果不一致|预处理不一致（BGR/RGB、归一化方式）|对齐每一步的中间结果|

---

## 十、面试高频题集

### 10.1 基础概念（必答）

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

### 10.2 项目深入（加分项）

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

## 十一、学习路线与踩点建议

### 11.1 推荐学习顺序

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

### 11.2 关键踩点提醒

- **不要纠结理论**：YOLO 的损失函数（CIoU + DFL + BCE）ultralytics 全封装了，面试时知道"分类损失 + 回归损失 + 置信度损失"三部分就够
- **不要手写训练循环**：直接用 `model.train()`，YOLO 不像 CIFAR-10 需要你自己写 epoch 循环
- **一定要做实验对比**：至少 3 组调参实验，面试官看的是"系统性调参思路"而不是"最终数值"
- **先跑通再优化**：baseline 哪怕 mAP 只有 0.5 也先记录，不要卡在调参上
- **ONNX 预处理要精确对齐**：Python 和 C++ 的推理结果差异 99% 来自预处理不一致

---

## 十二、适合的岗位与工作方向

### 12.1 YOLO 技能对应的岗位

|岗位方向|公司类型|核心技能要求|YOLO 项目的价值|
|---|---|---|---|
|CV 算法工程师|博世、西门子、Cognex|YOLO/检测 + 数据分析 + 调参|直接对口，核心项目|
|AI 部署工程师|KLA、应用材料|ONNX/TensorRT + C++ + 边缘设备|ONNX 导出 + C++ 推理经验|
|工业视觉工程师|海克斯康、ABB|缺陷检测 + 光学 + 产线集成|工业场景经验 + 数据闭环思路|
|MLOps / 模型服务化|各类外企 IT|FastAPI + Docker + CI/CD|API 服务化 + 容器化部署|
|机器人视觉工程师|ABB、发那科|目标检测 + 位姿估计 + ROS|YOLO 检测 + 部署能力|

### 12.2 简历中的项目描述模板

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

|文件|作用|
|---|---|
|`data/data.yaml`|数据集配置（路径 + 类名）|
|`configs/train_config.yaml`|训练超参数|
|`scripts/prepare_data.py`|VOC → YOLO 格式转换|
|`scripts/train.py`|训练入口|
|`scripts/evaluate.py`|评估 + 可视化|
|`scripts/export_onnx.py`|ONNX 导出|
|`scripts/inference_onnx.py`|ONNX 推理|
|`src/detector.py`|推理核心类（FastAPI 复用）|
|`docs/experiment_log.md`|实验记录|
|`runs/detect/train/weights/best.pt`|最优权重|
|`models/best.onnx`|导出的 ONNX 模型|