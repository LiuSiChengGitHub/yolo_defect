# TransUNet 面试突击 — 7 天学习方案

> 每天 1 小时 = 3 个 20 分钟块
> 目标：能在 5-10 分钟内清晰、自信地回答面试官对这个项目的所有追问

---

## Day 1：项目概述 — "30 秒讲清楚这个项目"

### 20min 读代码/文档
- 读 `README.md` 全文（重点看 Highlights、Architecture、Results 三节）
- 读 `docs/learning_guide.md` 的 Part 3.1（30 秒项目介绍）
- 打开 `src/model.py:527-566` 的 `get_model()` 函数，理解三个模型变体

### 20min 记关键点

**必须记住的 5 个数字/事实：**
1. Dice 0.788，LGG 数据集，TransUNet-Lite，50 epochs
2. 模型参数量：TransUNet-Lite ~9.44M，UNet ~7.76M
3. 输入 (B, 3, 256, 256) → 输出 (B, 1, 256, 256)
4. 110 个患者，~3929 张切片，70/30 患者级划分
5. 损失函数：Dice + Focal BCE，优化器：AdamW

**面试问答：**
- Q: 这个项目做的是什么？ → "脑部 MRI 肿瘤分割，用自定义的 TransUNet 混合模型，结合 ViT 全局特征和 CNN 局部细节，在 LGG 数据集上达到 Dice 0.788。"
- Q: 为什么做这个项目？ → "我想在简历上展示对 Transformer 在 CV 中应用的深入理解，选择医学影像分割是因为它对精度要求高、对数据处理有特殊考量（比如患者级划分），能体现算法和工程两方面的能力。"

### 20min 模拟面试
1. 请用 30 秒介绍你的 TransUNet 项目。
2. 为什么选择脑部 MRI 分割这个任务？
3. 你的模型效果怎么样？Dice 0.788 代表什么水平？
4. 这个项目跟你的 YOLO 项目有什么互补？
5. 你在这个项目中遇到的最大挑战是什么？

---

## Day 2：TransUNet 架构 — "为什么把 ViT 和 CNN 结合？"

### 20min 读代码
- 读 `src/model.py:408-494` — `TransUNet` 类的 `__init__` 和 `forward`
- 重点理解 forward 里三步：ViT 编码 → CNN 编码 → 解码器融合
- 读 `src/model.py:255-283` — `CNNEncoder`，理解四层下采样和 skip connection
- 读 `src/model.py:286-344` — `Decoder`，理解 fusion + 4x 上采样

### 20min 记关键点

1. **双编码器设计**：ViT 和 CNN 分别处理同一张输入图，ViT 负责全局，CNN 负责局部
   - Q: 为什么需要两个编码器？ → "CNN 的卷积核只看局部，需要很多层才有全局视野。ViT 通过自注意力一层就能看到全局，但对细节不够敏感。两者互补。"

2. **Fusion 机制**：在 bottleneck 处 concatenate → 1x1 conv 降维
   - Q: ViT 和 CNN 特征怎么融合的？ → "在最深层（16x16）将 ViT 特征和 CNN 特征 concatenate，用 1x1 卷积融合降维到 512 通道。"

3. **Skip Connection**：CNN 编码器的四层特征直接传给解码器
   - Q: Skip connection 的作用？ → "编码器压缩过程中丢失的空间细节（边缘、纹理）通过跳跃连接直接传给解码器，避免上采样时细节模糊。"

4. **Lite vs Full**：Lite 减少 embed_dim（256 vs 768）和深度（6 vs 12）
   - Q: 为什么用 Lite 版本？ → "Full 版本约 105M 参数，需要大显存。Lite 只有 9.44M，在 RTX 3060 上训练无压力，且小数据集上 Full 反而容易过拟合。"

### 20min 模拟面试
1. 画出 TransUNet 的架构并解释数据流。
2. ViT 编码器和 CNN 编码器分别做了什么？
3. 如果只用 ViT、不用 CNN 编码器，会怎样？
4. 跳跃连接在你的模型里连接了哪些层？
5. TransUNet-Lite 相比标准版做了什么简化？

---

## Day 3：ViT 核心机制 — "解释 Patch Embedding 和自注意力"

### 20min 读代码
- 读 `src/model.py:21-64` — `PatchEmbedding`，理解卷积做 patch 嵌入 + 位置编码
- 读 `src/model.py:67-103` — `MultiHeadAttention`，理解 QKV、scale、softmax
- 读 `src/model.py:133-152` — `TransformerBlock`，理解 Pre-norm + 残差连接
- 读 `docs/learning_guide.md` 的 1.4-1.6 节

### 20min 记关键点

1. **Patch Embedding = 一个大卷积**
   - `nn.Conv2d(3, 256, kernel_size=16, stride=16)` → 直接把 16x16 的 patch 映射成 256 维向量
   - Q: Patch Embedding 底层是什么操作？ → "本质是一个 stride=patch_size 的卷积，等价于把 patch 展平后做线性映射，但用卷积实现更高效。"

2. **自注意力的计算**
   - Q→K→V→注意力分数→softmax→加权求和
   - 时间复杂度 O(n²·d)，n=256 patches，d=256 embed_dim
   - Q: 自注意力的计算复杂度？ → "O(n² × d)，n 是 patch 数量。256x256 图、patch=16 时 n=256，可以接受。但如果 patch 更小，n 就会暴增。"

3. **多头的意义**
   - 8 个头，每个头 256/8=32 维
   - Q: 多头注意力为什么比单头好？ → "每个头在不同的子空间里学习注意力模式。比如一个头学纹理相似性，另一个头学空间邻近关系。最后 concat 起来信息更丰富。"

4. **Pre-norm vs Post-norm**
   - 本项目用 Pre-norm：先 LayerNorm 再 Attention
   - Q: 为什么用 Pre-norm？ → "Pre-norm 训练更稳定，不容易梯度爆炸，是 ViT 的标准做法。"

5. **位置编码**
   - 可学习参数，shape (1, 256, embed_dim)
   - Q: 位置编码为什么必要？ → "自注意力对输入顺序不敏感（是 permutation-equivariant 的），如果不加位置编码，打乱 patch 顺序结果不变，模型就丢失了空间信息。"

### 20min 模拟面试
1. 解释 Patch Embedding 的过程（从输入到输出 shape）。
2. 自注意力是怎么计算的？Q、K、V 分别是什么？
3. 为什么需要除以 √d_k？（防止点积值过大导致 softmax 饱和）
4. 多头注意力的"多头"指什么？为什么需要？
5. 如果去掉位置编码会怎样？

---

## Day 4：损失函数 + 数据处理 — "为什么用 Dice+BCE？为什么按患者划分？"

### 20min 读代码
- 读 `src/losses.py:16-54` — `DiceLoss`
- 读 `src/losses.py:57-88` — `BinaryFocalLoss`
- 读 `src/losses.py:91-116` — `BCEDiceLoss`
- 读 `src/dataset.py:178-254` — `_split_indices_by_patient()`
- 读 `docs/learning_guide.md` 的 2.3-2.4 节

### 20min 记关键点

1. **Dice Loss 公式**：`1 - (2·交集 + ε) / (预测总和 + 真实总和 + ε)`
   - Q: Dice Loss 直觉是什么？ → "直接优化预测和真实 mask 的重叠度。完美重叠 Dice=1，loss=0。对小目标比 BCE 更敏感，因为分母不包含大量正确分类的背景像素。"

2. **Focal Loss**：`-(1-p_t)^γ · log(p_t)`，γ=2
   - Q: Focal 的 γ 参数什么意思？ → "γ 越大，对容易分类样本的抑制越强。γ=2 时，如果模型对某个像素预测概率 0.95（很有信心），它的损失权重降为 (0.05)²=0.0025，几乎忽略。"

3. **组合损失**：`0.5 × Dice + 0.5 × Focal`
   - Q: 为什么不单独用一个？ → "Dice 关注全局重叠度但梯度不稳定，Focal BCE 提供稳定的像素级梯度。组合后兼顾两者优点。"

4. **Patient-Level Split**
   - 用文件夹名（如 `TCGA_CS_4941_19960909`）作为患者 ID
   - 按患者随机分组，seed=42 保证可复现
   - Q: 为什么不能按切片随机划分？ → "同一患者相邻切片几乎一模一样，随机划分会导致数据泄漏，验证集 Dice 虚高但泛化到新患者时大幅下降。"

### 20min 模拟面试
1. Dice Loss 的公式是什么？它解决什么问题？
2. Focal Loss 跟标准 BCE 有什么区别？
3. 你用的损失函数组合方式是什么？权重怎么分配的？
4. 什么是数据泄漏？你的项目中怎么避免的？
5. 如果数据集只有 10 个患者，patient-level split 还可行吗？（可能不行，验证集只有 3 个患者，方差太大）

---

## Day 5：训练细节 — "优化器、学习率、过拟合"

### 20min 读代码
- 读 `src/train.py:80-134` — `WarmupCosineScheduler`
- 读 `src/train.py:137-182` — `train_one_epoch()`
- 读 `src/train.py:358-583` — `train()` 主函数
- 关注：AMP、scaler、scheduler.step() 的位置

### 20min 记关键点

1. **AdamW vs Adam**
   - Q: 为什么用 AdamW？ → "Adam 的 weight decay 和梯度更新耦合在一起，效果不稳定。AdamW 把 weight decay 从梯度更新中解耦出来，正则化效果更好，是训练 Transformer 的标准选择。"

2. **Warmup + Cosine Schedule**
   - 前 5 个 epoch：LR 从 0.2×base 线性升到 base
   - 之后：Cosine 衰减到 0.01×base
   - Q: 为什么需要 warmup？ → "Transformer 参数初始化后，梯度方向不稳定。如果一开始就用大学习率，可能直接跑飞。Warmup 让模型先用小学习率'热身'，找到合理的参数区间后再加速。"

3. **AMP（自动混合精度）**
   - 用 float16 做前向和反向，用 float32 做参数更新
   - Q: AMP 有什么好处和风险？ → "好处：显存减半，训练加速 30-50%。风险：float16 精度低可能导致梯度下溢，所以用 GradScaler 动态调整 loss 的 scale。"

4. **过拟合防护**
   - Dropout 0.1（TransUNet-Lite）
   - AdamW weight decay 1e-4
   - 数据增强（翻转、旋转、对比度）
   - Q: 你做了哪些防过拟合措施？ → "三方面：模型级（Dropout）、优化器级（weight decay）、数据级（增强）。如果还不够可以加 early stopping。"

5. **Checkpoint Resume 的实现**
   - 保存 model + optimizer + scheduler + history 状态
   - Q: 断点续训需要保存哪些状态？ → "模型权重、优化器状态（含动量）、学习率调度器状态、当前 epoch 和 best_dice。漏存任何一个都会导致续训后行为不一致。"

### 20min 模拟面试
1. 你的学习率调度策略是什么？为什么这样设计？
2. AdamW 和 Adam 有什么区别？
3. AMP 是什么？你在训练中用了吗？
4. 训练中做了哪些防止过拟合的措施？
5. 如果训练到一半机器挂了，你怎么办？（checkpoint resume）

---

## Day 6：部署思考 + 改进方向 — "如果要部署/改进，你会怎么做？"

### 20min 读代码/文档
- 读 `src/inference.py:92-134` — `load_model()` 和 `predict()`
- 读 `src/model.py:501-524` — `TransUNetLite` 的简化配置
- 回顾 `docs/learning_guide.md` 的 2.5 节（提升方向）

### 20min 记关键点

1. **ONNX 导出可行性**
   - Q: 这个模型能导出 ONNX 吗？ → "可以。模型是标准 PyTorch，不含动态图操作。`torch.onnx.export(model, dummy_input, 'model.onnx')` 即可。但需要注意固定输入尺寸。"

2. **提升 Dice 的 Top 3 方向**
   - 预训练权重（最大增益）、更多数据增强（弹性变形）、延长训练
   - Q: 如果给你一周时间提升效果，你先做什么？ → "第一件事加载 ImageNet 预训练的 ViT 权重，预计直接提升 3-5 个点。第二件事加弹性变形增强。"

3. **Lite vs 标准版的取舍**
   - Q: 什么时候该用标准版？ → "数据量大（>5000 张）且有足够显存（>8GB）时用标准版。数据少或显存小就用 Lite，否则大模型反而过拟合。"

4. **模型推理速度**
   - TransUNet-Lite ~9.44M 参数，256x256 输入，GPU 推理通常 <20ms/张
   - Q: 推理速度能满足实时需求吗？ → "医学影像不需要实时（不像自动驾驶），离线分析几十 ms 完全够用。如果要快，可以 ONNX + TensorRT 优化。"

5. **多分类扩展**
   - 输出通道 1 → 4，损失改 CE+Dice，数据要换成 NIfTI 格式
   - Q: 你为什么没做 BraTS 四分类？ → "代码架构已经支持多分类（损失、数据集、推理可视化都有），但没有获取到 BraTS 真实数据完成完整实验。这是项目的诚实边界。"

### 20min 模拟面试
1. 如果面试官给你一个新的医学影像数据集，你怎么迁移这个项目？
2. 你的模型推理延迟大概多少？够不够用？
3. 预训练权重能提升多少？为什么？
4. Lite 和标准版的参数量差多少？各适合什么场景？
5. 这个项目有什么局限性？你怎么改进？

---

## Day 7：综合模拟 — "完整项目深挖演练"

### 20min 回顾
- 快速过一遍速查卡片（见下方）
- 过一遍 STAR 话术（见下方）
- 重读 `docs/learning_guide.md` Part 3

### 20min 全流程模拟（自己口述，计时）
按以下顺序模拟一次完整的面试问答（每题 1 分钟以内）：
1. 30 秒介绍项目
2. 画架构图，解释 ViT+CNN 融合
3. 解释 Patch Embedding 和自注意力
4. 为什么 Dice+Focal？
5. Patient-level split 是什么？
6. 训练策略：warmup+cosine、AdamW、AMP
7. 结果分析：0.788 是什么水平？
8. 和 YOLO 项目怎么互补？

### 20min 极限追问模拟
1. 你说 ViT 看全局，具体全局信息在分割任务中起什么作用？→ "肿瘤区域和远处正常组织之间的对比关系、对称性破坏等，都是全局信息。比如脑部肿瘤往往破坏了左右对称性，ViT 能捕捉这种远距离关系。"
2. 如果数据集从 110 人变成 10 人，你会怎么调整？→ "增加数据增强强度、用更强的正则化、考虑用预训练权重、可能要换成 5-fold 交叉验证来获得更可靠的评估。"
3. Dice 从 0.78 提升到 0.85，你觉得最关键的一步是什么？→ "加载 ImageNet 预训练权重。从头训练的 ViT 在小数据集上很难充分学习，预训练权重提供了好的初始化。"
4. 你的模型有多少 FLOPs？→ "具体数字没算过，但 TransUNet-Lite 约 9.44M 参数，256x256 输入，估计在 10-15 GFLOPs 量级。"
5. 如果面试官说'这个 Dice 太低了'，你怎么回应？→ "作为从头训练、无预训练权重的 baseline，0.788 是合理的。论文里带预训练的 TransUNet 在 Synapse 上也就 ~0.78。我的项目重点不仅是追求最高 Dice，更是展示完整的工程能力和算法理解。"

---

## 速查卡片（面试前 5 分钟看一遍）

```
项目：TransUNet 脑部 MRI 肿瘤分割
结果：Dice 0.788 | LGG 数据集 | 110 患者 ~3929 切片
模型：TransUNet-Lite = ViT(256d, 6层, 8头) + CNN(32ch, 4层) + Decoder
参数：9.44M（Lite）| 105M（Full）| 7.76M（UNet）
输入：(B, 3, 256, 256) → 输出：(B, 1, 256, 256)
损失：0.5×Dice + 0.5×Focal（γ=2）
优化：AdamW(lr=1e-4, wd=1e-4) + Warmup(5ep) + Cosine
数据：Patient-level 70/30 split，防泄漏
增强：随机翻转、90°旋转、对比度
关键设计：双编码器融合 | 患者级划分 | Focal+Dice | Warmup

亮点三连：
1. 自定义实现（不是调包）
2. 患者级划分（医学影像规范）
3. 完整工程闭环（训练→推理→可视化→产物导出）

提升方向 Top3：预训练权重 > 数据增强 > 延长训练

vs YOLO：YOLO=工程落地(检测+部署) | TransUNet=算法深度(分割+Transformer)
```

---

## STAR 话术（1 分钟项目介绍）

### 中文版

**S（情境）**："在研究生阶段，我参与了实验室的脑部 MRI 分析方向的工作，需要在 MRI 图像上自动分割肿瘤区域。"

**T（任务）**："我负责搭建一个基于 Transformer 的分割 baseline，需要在消费级 GPU 上可训练，同时代码要可复现、可展示。"

**A（行动）**："我自定义实现了 TransUNet 混合架构，ViT 编码器负责全局特征、CNN 编码器负责局部细节。为了确保实验可靠性，我实现了患者级数据划分防止泄漏，引入了 Focal+Dice 混合损失处理类别不平衡，加入了 Warmup+Cosine 学习率调度。同时做了完整的工程化：配置导出、训练曲线、checkpoint 元信息、推理汇总。"

**R（结果）**："在 LGG 数据集上达到 Dice 0.788，形成了一个可复现、可对比（内置 U-Net baseline）、可继续扩展的项目框架。"

### English Version

**S**: "During my first year of graduate school, I worked on brain MRI analysis and needed to automatically segment tumor regions from MRI scans."

**T**: "My task was to build a Transformer-based segmentation baseline that could train on a consumer GPU while being reproducible and presentation-ready."

**A**: "I implemented a custom TransUNet hybrid architecture — ViT encoder for global context, CNN encoder for local detail. I introduced patient-level data splitting to prevent leakage, Focal+Dice loss for class imbalance, warmup+cosine scheduling, and comprehensive experiment artifact export."

**R**: "Achieved Dice 0.788 on the LGG dataset, producing a reproducible, comparison-ready project with a built-in U-Net baseline."

---

## 高频面试 Q&A（20 题）

### 项目层面

**1. 项目做了什么？**
脑部 MRI 肿瘤二分类分割。用自定义 TransUNet（ViT + CNN 混合架构），在 LGG 数据集上达到 Dice 0.788。

**2. 为什么选 TransUNet 不用纯 U-Net？**
纯 U-Net 只有局部感受野，要很多层才能看到全局。TransUNet 用 ViT 一层就能建模全局关系，更适合需要理解肿瘤与周围组织空间关系的医学影像。

**3. 为什么不直接用纯 ViT（如 SegFormer）？**
纯 ViT 在 patch 边界会丢失细节，对精确分割不利。CNN 编码器的 skip connection 能补回局部细节。

**4. 项目的工程亮点是什么？**
三个亮点：患者级数据划分防泄漏、Focal+Dice 处理不平衡、完整的实验产物导出（config/history/curves/summary）。

**5. 项目的局限性是什么？**
没有用预训练权重（影响 Dice）、没有完成 BraTS 四分类实验、没有做 ONNX 部署。

### 算法层面

**6. ViT 的自注意力机制怎么工作？**
每个 patch 生成 Q/K/V 三个向量，Q 与所有 K 做点积得到注意力分数，softmax 归一化后加权求和 V，得到融合了全局信息的输出。

**7. 多头注意力为什么比单头好？**
不同的头在不同子空间学不同的注意力模式（纹理、空间、语义），综合起来信息更丰富。

**8. Patch Embedding 本质是什么？**
一个 stride=patch_size 的卷积，等价于把 patch 展平后线性映射。

**9. 位置编码的作用？**
给 Transformer 提供空间位置信息。没有位置编码的话打乱 patch 顺序结果不变，模型丢失空间信息。

**10. Dice Loss 的直觉？**
直接优化预测和真实 mask 的重叠度，对小目标比 BCE 敏感。因为分母只包含前景相关的像素。

**11. 为什么 Dice + Focal 一起用？**
Dice 关注全局重叠度但梯度不稳，Focal 给每个像素稳定梯度并抑制简单样本。互补组合。

**12. Focal Loss 的 γ 参数？**
γ 越大，越抑制容易分类的样本。γ=2 时，置信度 0.95 的样本权重只有正常的 0.25%。

### 数据层面

**13. 什么是 Patient-Level Split？**
按患者而不是按切片划分训练/验证集，确保同一患者所有切片只出现在一个集合中。

**14. 不做 Patient-Level Split 会怎样？**
数据泄漏。同一患者的相邻切片几乎一模一样，验证 Dice 虚高但不代表真实泛化能力。

**15. 数据增强做了什么？**
随机水平/垂直翻转、90° 旋转、对比度抖动。注意增强要同时应用到图像和 mask。

### 训练层面

**16. 为什么用 AdamW？**
相比 Adam，weight decay 与梯度更新解耦，正则化效果更好。是训练 Transformer 的标准选择。

**17. Warmup 的作用？**
Transformer 初始阶段梯度不稳定，warmup 用小学习率"热身"，让参数先进入合理区间再加速训练。

**18. AMP 是什么？**
自动混合精度——前向和反向用 float16 节省显存和加速，参数更新仍用 float32 保证精度。

### 扩展层面

**19. 如果要做 BraTS 四分类，改什么？**
输入改 4 通道（T1/T2/FLAIR/T1ce），输出 4 通道，损失改 CE+Dice，评估按区域分别算 Dice。

**20. Dice 0.788 怎么提升到 0.85+？**
最有效：加载 ImageNet 预训练 ViT 权重（+3-5 点）。其次：弹性变形增强、延长到 100+ epochs、CRF 后处理。
