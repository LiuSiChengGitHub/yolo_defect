# YOLO 缺陷检测项目 — 学习笔记 & 进度追踪

> 每次推进项目后更新，记录做了什么、学到了什么、下一步做什么。

---

## 当前进度：Step 1 完成，准备进入 Step 2

| Step | 内容 | 状态 |
|------|------|------|
| Step 1 | 数据准备 | Done (2026-03-23) |
| Step 2 | 基线训练 | Next |
| Step 3 | 调参优化 | - |
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

## Step 2：基线训练（待开始）

下一步：
```bash
/d/Base/Tools/Anaconda/Anaconda3/envs/yolo_defect/python.exe scripts/train.py
```
预期产出：baseline mAP@0.5 数值（预期 0.50-0.65），loss 曲线
