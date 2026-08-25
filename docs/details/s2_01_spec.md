# S2-01 INT8 PTQ 与 ORT Profiling 最小 SPEC

状态：原始机器协议保持 `FROZEN_BEFORE_FIRST_FORMAL_PTQ`；单元按下述用户范围覆盖以 `EXERCISE_ADVISORY_COMPLETION` 收口。协议文件和原始结果不回写、不伪造。

> **2026-08-25 用户范围覆盖：**本项目定位为简历个人练习，不再以“找到任何能通过严格产品差异门的极小量化子集”为目标。最终交付固定为 v1 全 64 个 Conv 的 QDQ/S8S8 static PTQ，并完成 Python/C++ Runtime 合法性、同协议 FP32/INT8 性能比较和两份 ORT profile。30 图产品差异与 361 图任务质量仍照原冻结方法计算并保留真实布尔值，但从阻断门改为 advisory 诊断项；正式 benchmark 只要求双方 Runtime 合法、模型 lineage 一致且 profiler 关闭。这个覆盖只改变“是否阻断练习收口”，不改变协议、样本、指标、门值或历史结果。v2-v11 是范围切换前的 selective-PTQ 探索记录，不是最终 artifact，也不再继续搜索。

权威机器协议为 `cpp_infer/protocols/s2_01_ptq_protocol.json`，canonical-LF SHA-256 为 `0EC9A7B1CF5E4F246CF3AC15275EF06D7C67FB6C0CE11C5218391CFACE5B73F2`，`protocol_id=s2_01_static_ptq_qdq_s8s8_cpu_v1`。机器协议、manifest 和工具校验结果优先于本文件的自然语言复述。

## 1. 目标与问题

本单元把现有 FP32 CPU 单图产品链扩展为“可量化、可比较、可解释瓶颈”的 S2-01 闭环，回答两个独立问题：

1. 固定输入、质量与性能协议后，static INT8 PTQ 是否减小模型、维持检测质量，以及是否改善本机 CPU Runtime；INT8 变慢仍是合法且必须如实发布的结果。
2. FP32 与 INT8 的 `Ort::Session::Run` 内部主要时间落在哪些优化后 node/operator，以及实际由哪个 provider 执行。

原始 SPEC 要求三层正确性全部通过后才能发布性能；用户范围覆盖后，正式 benchmark 必须引用同次、同 lineage 的完整正确性结果并通过 Python/C++ Runtime 合法性，但产品差异和任务质量布尔值可为 false 且必须原样展示。profiling 使用另一独立进程和 profiling-enabled session，trace 时间绝不替代正式 benchmark。

## 2. 冻结输入与产物

FP32 source 为 `models/best.onnx`：12,336,935 bytes，raw SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`。matching `.pt` 不在工作区不阻塞 PTQ；本单元不声称重新完成 PyTorch/ORT/C++ 三方复现。

量化输入、输出和派生关系：

```text
best.onnx + frozen calibration manifest + frozen PTQ options
-> ORT quant_pre_process intermediate
-> quantize_static QDQ/S8S8, Conv only
-> models/best.int8.qdq.onnx
-> cpp_infer/results/s2_01/quantization_report.json
```

量化 report 同时是派生 artifact card，必须记录 source/derived SHA 与大小、实际 ONNX/ORT metadata、依赖版本、命令、64 个候选 Conv 的量化/未量化/失败审计，以及 Python CPU session/finite-output smoke。INT8 Runtime contract 使用独立 ModelArtifactSpec 和 RuntimeConfig，外部 I/O 仍为 `images float32 [1,3,800,800]` 与 `output0 float32 [1,10,13125]`；INT8 仅是图内部 QDQ 语义。

## 3. 校准协议

校准 manifest 为 `cpp_infer/tests/fixtures/s2_01_calibration_manifest.json`：

- canonical-LF SHA-256：`6C0735C6E1510F725E1168A3C57E7107259CC1934D32DEB3E619C1BF6712AA9D`；
- sample-set SHA-256：`FDEF7FB3B64E222386387438C0B4A32A6BDECF9761E5ED5C60E9A17B7311AE5F`；
- 只取 train，六个 filename source class 各 30 张，共 180 张；每类固定索引 `i=1+8*k, k=0..29`；
- source class 只用于平衡采样，PTQ 不读取 label；校准集与质量集 image SHA 交集必须为 0；
- 每张图片记录 declaration-relative path 与 raw image SHA，sample-set digest 由固定 TAB 行格式、固定顺序和 LF 行尾计算。

校准 preprocess 与 C++ 产品链同语义：OpenCV `IMREAD_COLOR` BGR uint8，`800x800` letterbox，正数 `floor(x+0.5)` resize rounding，`INTER_LINEAR`，pad 114，BGR→RGB，float32 `/255`，NCHW、C contiguous。

## 4. Static PTQ 配置

固定使用 ONNX Runtime 1.19.2 的 static PTQ：

- `quant_pre_process(skip_optimization=True, skip_symbolic_shape=True, skip_onnx_shape=False)`；
- `QuantFormat.QDQ`、activation `QInt8`、weight `QInt8`，即 CPU QDQ/S8S8 起点；
- `op_types_to_quantize=["Conv"]`，源图预期选择 64 个 Conv；
- `per_channel=True`、`reduce_range=False`、`MinMax` calibration；
- `nodes_to_exclude=[]`、`use_external_data_format=False`；
- `ActivationSymmetric=False`、`WeightSymmetric=True`、`CalibTensorRangeSymmetric=False`、`CalibMovingAverage=False`、`ForceQuantizeNoInputCheck=False`、`AddQDQPairToWeight=False`、`DedicatedQDQPair=False`、`QDQKeepRemovableActivations=False`。

首次正式量化只允许使用冻结协议入口。源 SHA、manifest canonical hash、180 张图片 raw hash、环境版本、源 metadata、Conv 数量或输出目标策略任一不符时，工具必须在写派生产物前失败，并输出 object/expected/actual/action。发布使用 staged file + atomic replace；已有目标默认拒绝，只有明确 `--overwrite` 才可替换。

### 4.1 v1 实测结论与 v2 冻结

全 64 个 Conv 的 v1 必须作为真实失败候选保留，不能通过回改协议或调低门值“修复”。v1 在 Runtime 合法性、Python/C++ 一致性和 361 图任务质量门上通过，但 30 图产品门失败：INT8 agreement precision 为 `0.9384615385 < 0.95`，confidence absolute-error mean 为 `0.0507706101 > 0.05`，P95 为 `0.1736433506 > 0.10`。失败原始结果保存为 `cpp_infer/results/s2_01/correctness_quality_v1_failed.json`，raw SHA-256 为 `73E417D16BDCDEB0B95C1946DA53F552DAF4DF9895CFCF95EB685ADB6A9B9062`。

在 v1 失败之后、第二次正式量化之前，冻结独立的 v2 protocol `s2_01_static_ptq_qdq_s8s8_head_fp32_cpu_v2`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v2.json`，canonical-LF SHA-256 为 `D083F182E24290DFBA7864A2C840803DDE82AEC49EAE6405F1C63EB1B2C22068`。v2 保持源模型、180 图校准集、preprocess、QDQ/S8S8、MinMax、per-channel、正确性门、benchmark 和 profiling 协议不变；唯一量化策略变化是显式排除 `/model.22` 检测头的 19 个 Conv（3 个尺度的 box/class 两分支各 3 个 Conv，以及 DFL Conv），其余 45 个 backbone/neck Conv 仍必须形成完整 QDQ。这样做的原因是 v1 已证明任务级 AP 可接受，而直接产品 confidence/额外框发生越界；保留任务头 FP32 是可审计的 selective static PTQ，不是事后修改输出阈值。19 个节点的完整有序身份以 v2 protocol 为权威，量化报告必须分别记录 `quantized=45`、`intentional_unquantized=19`、`failed=0`；任何其他排除集合都需要新 protocol 版本。

v2 量化和 Runtime 合法性通过，但产品门仍真实失败：INT8 agreement precision `0.9230769231`、confidence error mean `0.0518089960`、P95 `0.1743268594`，而任务质量门仍通过。该结果证明主要误差来自上游 activation 量化传播，而非仅由末端检测头权重量化引起；原始结果 `cpp_infer/results/s2_01/correctness_quality_v2.json` 的 raw SHA-256 为 `88B6B478037A43F8C32C7F66A2FA263E30CCDA52F2C38CBDF93F5D1BBD95910C`。

在 v2 失败之后、第三次正式量化之前，冻结只改变一个变量的 v3 protocol `s2_01_static_ptq_qdq_s8s8_head_fp32_entropy_cpu_v3`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v3.json`，canonical-LF SHA-256 为 `4CFAB466786FF02A79F9F43020B5B5C06FA2B495D4FDE3AC560A624B17BD4DF3`。v3 保持 v2 的 45/19 节点选择、S8S8、per-channel、calibration manifest、所有 correctness gates、benchmark 和 profiling 完全不变，只把 activation calibration 从 `MinMax` 改为 `Entropy`。ORT 1.19.2 本地实现的冻结默认值为 `num_bins=128`、`num_quantized_bins=128`，且 `CalibTensorRangeSymmetric=false`；工具必须在报告中显式记录这些有效参数。若 v3 仍失败，不得回改 v3 或放宽产品门，后续尝试必须继续使用新 protocol 版本。

v3 在派生产物发布前失败：ORT 1.19.2 Entropy calibrator 收集直方图时请求 shape `(230400000,)` 的 float32 数组，连续分配约 `879 MiB` 触发 `MemoryError`。因此 v3 没有模型、artifact card 或 correctness 结果，不能作为可运行候选；本单元不通过静默增加内存、减少冻结校准图或改写 v3 来掩盖不可复现性。

在 v3 失败之后、第四次正式量化之前，冻结 v4 protocol `s2_01_static_ptq_qdq_s8s8_backbone_only_cpu_v4`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v4.json`，canonical-LF SHA-256 为 `EE1FE1998DD20404497E24613F449660F9EF91F3CA2DEBB5E2CCDD7801761935`。v4 恢复已可复现的 MinMax，保持校准数据和所有下游协议不变；节点选择作为唯一变量收缩为 model.0–9 的 27 个 backbone Conv 做 QDQ，model.12–22 的 18 个 neck Conv 与 19 个 head Conv（共 37 个）保持 FP32。量化审计预期 `quantized=27`、`intentional_unquantized=37`、`target_unquantized=0`、`failed=0`，37 个排除节点的完整有序列表以 v4 protocol 为权威。

v4 的模型大小下降 `30.0064%`，Runtime、Python/C++ 与任务质量门通过；产品 confidence mean 改善为 `0.0460400447` 并过门，但 INT8 agreement precision 为 `0.9117647059`，confidence P95 为 `0.1745662779`，仍失败。结果保存为 `cpp_infer/results/s2_01/correctness_quality_v4.json`，raw SHA-256 `3CD94D3C1367A8C5A46041CF7B39B6B1362A6FC20B5F0E169428F62275617D38`。该结果说明直接产品问题并非只与量化节点总数单调相关，需要区分早期和后期 backbone block，而不是继续按大范围盲目收缩。

在 v4 失败之后、第五次正式量化之前，冻结 early-backbone v5 protocol `s2_01_static_ptq_qdq_s8s8_early_backbone_cpu_v5`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v5.json`，canonical-LF SHA-256 `C9441D27762F46736A2D8F87A4226C2108834E6C3F7301E5A8AD9C2B61754D30`。v5 保持 MinMax、S8S8、per-channel、校准与全部下游协议不变，只量化 model.0–4 的前 13 个高分辨率 backbone Conv，后 51 个 Conv 保持 FP32；预期审计为 `quantized=13`、`intentional_unquantized=51`、`target_unquantized=0`、`failed=0`。v5 的 30 图产品门先通过非正式 candidate screen 后才允许执行 361 图/C++ 正式正确性，screen 明确不是 acceptance 证据。

v5 candidate screen 失败并停止其正式 361 图评估：文件只缩小 `1.5600%`，30 图 INT8 agreement precision `0.9117647059`，confidence P95 `0.1624116138`；它与 v4 一样产生 68 个 INT8 框，说明 early backbone 是 threshold crossing 的主要来源。screen 保存于 `cpp_infer/results/s2_01/product_screen_v5.json`，raw SHA-256 `9F378AAB97E796CDF692995B5885004A7AE936B0AAEDADD2CAE70570CB98E2DD`，并明确 `formal_acceptance=false`。

在 v5 screen 失败之后、第六次正式量化之前，冻结互补的 late-backbone v6 protocol `s2_01_static_ptq_qdq_s8s8_late_backbone_cpu_v6`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v6.json`，canonical-LF SHA-256 `60600A9BA8C6262C8CC5439F6051F82848EA70DE4BB53223B2EEA0C842E46615`。v6 只量化 model.5–9 的 14 个后半 backbone Conv，early backbone 13 个与 neck/head 37 个（共 50 个）保持 FP32；其他协议不变，预期审计 `quantized=14`、`intentional_unquantized=50`、`target_unquantized=0`、`failed=0`。它同样必须先过非正式 30 图 screen，才进入完整正确性。

v6 文件缩小 `28.3179%`，screen 的全部几何和 confidence 门均明显通过：FP32 retention `1.0`、matched IoU mean/P05 `0.9895824530/0.9772567183`、confidence error mean/P95 `0.0112798305/0.0280603588`；唯一失败是 4 个额外 INT8 框使 agreement precision 为 `62/66=0.9393939394`。screen 保存于 `cpp_infer/results/s2_01/product_screen_v6.json`，raw SHA-256 `03147BC1B5F27C39D388831D46E0DC9948517B7EA396FB034358DCCC2EE6F5C5`。因此继续把 late backbone 对半拆分，定位额外框来自 model.5–6 还是 model.7–9。

在 v6 screen 失败之后、第八次正式量化之前，先冻结更有模型体积收益的 deep-backbone v8 protocol `s2_01_static_ptq_qdq_s8s8_deep_backbone_cpu_v8`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v8.json`，canonical-LF SHA-256 `B5FB4969BD356D31ED62FD6D0E61F22136860E8A82B61AB2F83964F80EC700FD`。v8 只量化 model.7–9 的 7 个 Conv，其余 57 个保持 FP32；所有数据、MinMax/S8S8 参数和门值不变。若 v8 screen 失败，才冻结并运行互补的 model.5–6 v7；版本号保留了该互补实验的语义，未运行的版本不生成伪证据。

v8 candidate screen 通过并成为唯一进入完整正式正确性的候选：模型大小下降 `21.9241%`，FP32 retention `1.0`、INT8 agreement precision `0.9538461538`、matched IoU mean/P05 `0.9924641626/0.9792580217`、confidence error mean/P95 `0.0094499992/0.0216670066`，全部满足预声明门。screen 保存于 `cpp_infer/results/s2_01/product_screen_v8.json`，raw SHA-256 `57B0071FA73E7355D5FB30B5DFA974179532AD8E7331213545FCB09FD73F35DA`，仍不替代 361 图任务质量和 C++ 一致性证据。

v8 的正式 Python 产品门与 361 图任务质量门都通过，但 C++/Python INT8 一致性失败：18/30 图超过冻结实现容差，最大 confidence 差 `0.0058794317`、bbox 坐标差 `0.552002 px`，并有 1 张发生 detection-count threshold crossing；FP32 一致性仍通过。该现象说明同为 ORT 1.19.2 的 Python wheel 与官方 C++ SDK 对孤立 deep-block QDQ 图存在实际 build/kernel 数值差异，因此 v8 不能发布。正式结果保存为 `cpp_infer/results/s2_01/correctness_quality_v8.json`，raw SHA-256 `06406FB3D7AE7FBACF1C2C895360029D2A9AFC3D84FE77736CA60EB8DD623281`。

在 v8 正式失败之后、第七个实际候选量化之前，冻结互补的 mid-backbone v7 protocol `s2_01_static_ptq_qdq_s8s8_mid_backbone_cpu_v7`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v7.json`，canonical-LF SHA-256 `7CEA43AB52C030AFA47DC3C733A0F9D96055BAAD30B0296C93ECE06E9529B4DD`。v7 只量化 model.5–6 的 7 个 Conv，其余 57 个保持 FP32；MinMax/S8S8、校准、产品、质量、benchmark 和 profiling 协议均不变。v7 仍必须先过 30 图 screen，再接受完整 Python/C++ 与 361 图门。

v7 candidate screen 通过：模型大小下降 `6.2654%`，FP32 retention `0.9838709677`、INT8 agreement precision `0.953125`、matched IoU mean/P05 `0.9829999316/0.9649842978`、confidence error mean/P95 `0.0086958374/0.0228808224`，原始 screen `cpp_infer/results/s2_01/product_screen_v7.json` 的 raw SHA-256 为 `D330775C2FFF2161D2713C38DC20A377840EAB90037528F350335387DCE47196`。完整评估中，产品门和 361 图任务质量门继续通过，mAP50 与 mAP50-95 delta 分别为 `+0.0065917565/+0.0027655232`；但 C++/Python INT8 一致性仍只有 12/30 图通过，最大 confidence 差 `0.0030912464`、bbox 坐标差 `0.286789 px`，因此正式结果 `cpp_infer/results/s2_01/correctness_quality_v7.json` 以 `passed=false` 保存，raw SHA-256 为 `B055E7DE22E74A9C419E5540A0463B47CB1C796AA202561ECB9AB933CEAF4B46`。v7 与 v8 的共同失败支持一个新假设：孤立插入在模型中部的 QDQ 子图边界会触发 Python wheel 与官方 C++ SDK 的不同优化/kernel 数值路径。

在 v7 正式失败之后、第九个候选量化之前，冻结连续输入前缀 v9 protocol `s2_01_static_ptq_qdq_s8s8_prefix_model0_2_cpu_v9`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v9.json`，canonical-LF SHA-256 `AB31FEEAE6E46B9D82544028AC7D596DAFB1AA125C90C34A8DC031865DD02D4B`。v9 只量化源图最前面的 model.0–2 共 6 个连续 Conv，后 58 个 Conv 保持 FP32；预期审计为 `quantized=6`、`intentional_unquantized=58`、`target_unquantized=0`、`failed=0`。这一候选不是为了追求更高压缩率，而是用最小连续 QDQ 前缀区分“量化数值误差”和“孤立子图跨 SDK 优化差异”。它保持所有已冻结数据、MinMax/S8S8 参数、门值、benchmark 和 profiling 协议不变，并仍须先过 30 图非正式 screen，才能运行正式 Python/C++ 与 361 图评估。

v9 的 6/58 图审计与 Python Runtime smoke 通过，模型仅缩小 `0.0624223%`；30 图 screen 仍在 INT8 agreement precision `0.9117647059` 和 confidence P95 `0.1662984103` 两项失败，虽然 retention `1.0`、matched IoU mean/P05 `0.9388581255/0.8142753780` 与 confidence mean `0.0441004312` 已过门。screen 保存于 `cpp_infer/results/s2_01/product_screen_v9.json`，raw SHA-256 `59F8E67FF87F1F7C2D70DBDAA90D6FF3DB0AB8CA438725FF272842D9788DC6BA`；按停损规则不运行 v9 的 361 图正式评估。

在 v9 screen 失败之后、第十个候选量化之前，冻结 v10 protocol `s2_01_static_ptq_qdq_s8s8_prefix_model0_1_cpu_v10`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v10.json`，canonical-LF SHA-256 `18C66D0F163EFE6DEF58ED12CC927F5351BBDAF862795BAA0FB037295B9F082C`。v10 只量化输入端 model.0 与 model.1 两个连续 Conv，后 62 个 Conv 保持 FP32；预期审计 `quantized=2`、`intentional_unquantized=62`、`target_unquantized=0`、`failed=0`。除节点范围外，所有数据、算法、门值和后续性能/profile 协议不变；同样先过 30 图 screen 才允许进入正式完整评估。

v10 的 2/62 图审计与 Python Runtime smoke 通过，但 QDQ scale/zero-point/node 开销超过被压缩的两层权重，派生文件比 FP32 大 `4,690 bytes`，即模型大小变化为 `-0.0380159%`。30 图 screen 的 agreement precision `0.9393939394` 与 confidence P95 `0.1375658363` 仍失败；其余 retention `1.0`、matched IoU mean/P05 `0.9571545018/0.8719390422`、confidence mean `0.0332327322` 通过。screen `cpp_infer/results/s2_01/product_screen_v10.json` raw SHA-256 为 `6069244E23C36A7CCC2925775AFA7DFDC475B0AA524E2F20AB9DD18D2568D44A`，因此不运行 v10 的正式 361 图评估。

在 v10 screen 失败之后、第十一个也是本轮预声明搜索的最后一个候选量化之前，冻结 v11 protocol `s2_01_static_ptq_qdq_s8s8_prefix_model0_cpu_v11`，文件为 `cpp_infer/protocols/s2_01_ptq_protocol_v11.json`，canonical-LF SHA-256 `C4E9B351E291791E2A893E8044001821AE1918D1D321A256B9D03E30D5408FB2`。v11 只量化输入端首个 `/model.0/conv/Conv`，后 63 个 Conv 保持 FP32；预期审计 `quantized=1`、`intentional_unquantized=63`、`target_unquantized=0`、`failed=0`。若 v11 仍不过冻结产品门，则当前 QDQ/S8S8/MinMax static PTQ 路线没有可正式发布候选，必须如实收口失败而不是继续把“量化零层”冒充 INT8。

v11 的 1/63 图审计、actual metadata、Python CPU session 与 finite-output smoke 均通过；派生 SHA 为 `11B28A11995D7DCB05881F15586263DA7E4F5B3B0308CDE82526CA5970E2337F`，大小 `12,353,240 bytes`，因 QDQ 开销比 FP32 增大 `16,305 bytes`（size reduction `-0.1321641%`）。30 图 candidate screen 全部过门：retention `0.9838709677`、agreement precision `0.9682539683`、matched IoU mean/P05 `0.9711182919/0.9163097739`、confidence error mean/P95 `0.0199558725/0.0640256405`。screen `cpp_infer/results/s2_01/product_screen_v11.json` raw SHA-256 为 `4C0B26A15B0A2F72C9344BA3A3EB3AD73B4B2B24F121FD8E7F75D5D5FF4C7D2E`，因此 v11 允许进入正式完整评估；screen 本身仍不是 acceptance 证据。

## 5. 三层正确性与停损门

### 5.1 Runtime 合法性

Python ORT 与 Release C++ ORT 必须分别对 FP32/INT8：创建 CPU-only session，验证实际 provider、单输入/单输出 name/type/shape，至少执行冻结产品样本，并确认原始输出和产品 JSON 中数值有限。正式 correctness CLI 强制要求 `--cpp-cli`；缺少 C++ 结果时总 `passed` 必须为 false。

### 5.2 产品检测差异

产品 manifest 为现有 6 类×5 张、共 30 张的 `consistency_manifest.json`，canonical-LF SHA-256 `4A10742F373D1A999839996D45BEAD84F3340F3A37C35A18E9EBF534147F1E46`。两模型都使用严格 `score > 0.25`、class-agnostic NMS `0.45`。

matching 在量化前冻结为：逐图、exact class_id 分组；计算 float32 continuous-xyxy IoU；按 IoU 降序做 greedy one-to-one assignment；并以 `class_id,-confidence,x1,y1,x2,y2` 的双方 detection key 和原始 index 做确定性 tie-break；只有 `matching_iou >= pair_iou_min` 才接受，其余检测保持 unmatched。P05/P95 使用位置 `p*(n-1)` 的相邻 order-statistics 线性插值。机器结果必须保存双方 count/class histogram，并对每个 pair 保存 FP32/INT8 confidence、bbox、坐标误差、IoU、accepted 状态以及 unmatched detection 内容。

冻结门值：pair IoU ≥ 0.50；FP32 retention ≥ 0.95；INT8 agreement precision ≥ 0.95；matched mean IoU ≥ 0.90；matched IoU P05 ≥ 0.75；confidence absolute-error mean ≤ 0.05、P95 ≤ 0.10。

### 5.3 任务质量

质量 manifest 为当前全部 361 张 val 图片与对应 YOLO TXT，共 857 个 GT 框；canonical-LF SHA-256 `CED5CE80B119B1446066B18072B2AD1C7BE7A6DA30429B5C01D617F2AA2BCEF8`，sample-set SHA-256 `F90692D9898C6F92D94BD4CE3B2AD4DF996A864A0AF7FC0DAFCE97B33C780E33`。六类 GT 数为 `165/159/193/87/132/121`。不得读取路径与数量均陈旧的 `data/labels/val.cache`。

两模型都以 score floor `0.001`、strict `>`、class-agnostic NMS `0.45` 预测；按 IoU `0.50:0.05:0.95`、逐类 one-to-one GT matching 和 101 点 precision envelope 计算 AP。该指标明确称为 `COCO-style 101-point` 相对比较：`max_detections_per_image=null`，不实现 COCO area ranges/maxDets，因此不冒充官方 COCO evaluator 或历史 Ultralytics 指标。

原始严格停损门：INT8 相对 FP32 的 mAP50-95 absolute drop ≤ 0.020，mAP50 drop ≤ 0.010，每类 AP50 drop ≤ 0.050。门值和计算结果继续保留；按 2026-08-25 范围覆盖，它们作为 advisory 诊断项，不再阻断个人练习的性能/profile 采集。若将来声称产品级严格 acceptance，则仍必须恢复阻断语义，不能调低门值。

## 6. 同协议性能

正式性能固定在同一 Windows x86_64 机器、同一 Release CLI、CPUExecutionProvider、sequential、intra/inter-op `1/1`、graph optimization `all` 下运行；FP32 和 INT8 必须是两个独立 CLI 进程。固定图片 `crazing_241.jpg`，raw SHA-256 `1D65EF27EAA9BF27608D954DFE57B40E401FC1AED435884400F35E8000BBF98D`，warmup 10、repeat 100、batch 1、score/NMS 与产品协议一致，profiler 必须关闭。

每个模型记录文件大小、一次 `Ort::Session` 构造时间、image decode、preprocess、`Session::Run`、postprocess、pipeline 与 end-to-end mean/P50/P95、pipeline/end-to-end throughput，以及进程生命周期 Peak Working Set。Session 初始化只有单次观察值，不声明 P50/P95；Peak Working Set 不是当前 RSS、阶段增量或模型独占内存。比较器默认 `correctness-policy=required`；本次必须显式选择 `advisory`，仍要求 Python/C++ Runtime 合法、协议 id/hash 和双方 artifact SHA 一致，同时原样记录 correctness prerequisite `passed=false`、`blocking=false`。速度、吞吐或内存方向只做结果，不作为通过门。

## 7. ORT Profiling

FP32/INT8 各在独立 C++ 进程中新建 profiling-enabled CPU session，对 benchmark 同一预处理张量执行 10 次 `Session::Run`，调用 `EndProfilingAllocated` 获取实际 raw trace。profile 与正式 benchmark 分离，`performance_gate=false`。

摘要只聚合 Chrome trace 中 `cat=Node`、`ph=X`、name 以 `_kernel_time` 结尾的事件；每个执行 node 的调用次数必须等于 10。输出 top nodes、top/all operators、provider placement、total/mean ms、占比与累计占比，并记录 raw trace SHA/大小。node kernel 总和只是 profiled `Session::Run` 的诊断组成，不能替代外层 steady-clock wall time；instrumentation overhead 存在但不量化，优化后融合 node 也不必一一对应源 ONNX node。

## 8. 证据、错误与非目标

正式证据根为 `cpp_infer/results/s2_01/`，至少包含 quantization report/card、三层 correctness/quality、FP32/INT8 benchmark、benchmark comparison、两份 raw profile、两份 profile summary 和最终 completion JSON。严格模式仍生成 acceptance；本次 advisory 模式生成 `s2_01_exercise_completion`，同时保存 `passed=true` 与 `strict_acceptance_passed=false`。assembler 必须交叉校验 protocol id/hash、source/derived SHA、manifest hashes、各 evidence type/真实 pass 状态、benchmark/profile artifact 绑定与 raw trace hash；任一缺失或漂移均失败。

错误统一返回非零退出码并包含失败位置、期望、实际与纠正动作。所有 JSON 拒绝 duplicate key、NaN/Infinity 和不匹配 schema；已有输出默认不覆盖。

本单元非目标：dynamic PTQ、QOperator/U8U8 搜索、QAT、TensorRT/GPU、Linux/AArch64、batch/concurrency、重训或 `.pt` 三方复现。原始质量/产品失败必须记录；范围覆盖后不再继续搜索极小 selective 子集。其他量化路线若日后重启，必须作为新 protocol 版本，而不是修改本 v1。S2-01 完成后停止等待用户 L1，不提前实现 S2-02。

## 9. 验收

按用户范围覆盖，S2-01 练习完成条件为：冻结 manifest/protocol 在首次 PTQ 前存在并通过 loader；INT8 ONNX 与独立 contract/card 的 SHA/大小/metadata 一致；Python/C++ 均实际运行 FP32/INT8 且输出有限；30 图产品差异和 361 图任务质量按原协议计算并保留真实结果；两个独立 unprofiled benchmark 可比较；两个 raw trace 与摘要可审计；现有 Windows CTest/correctness 实现不回退；最终机器 completion 明确为 advisory 且 strict acceptance false；AGENTS.md、README.md、README_zh.md 同步真实结果、命令、证据、限制与下一步。若要声称原始严格 acceptance，则产品差异和任务质量仍必须全部通过。
