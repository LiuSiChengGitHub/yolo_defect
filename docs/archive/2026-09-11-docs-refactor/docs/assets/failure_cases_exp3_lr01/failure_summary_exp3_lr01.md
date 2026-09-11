# Failure Case Analysis

以下案例由脚本按“同类 + IoU>=0.5”的规则自动筛选，优先保留漏检/误检最多的验证集图片。

| Case | Image | Failure Type | FN Classes | FP Classes | Initial Hypothesis |
|------|-------|--------------|------------|------------|--------------------|
| case_01 | inclusion_287.jpg | FN x4 + FP x3 | inclusion | inclusion | 局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_02 | crazing_299.jpg | FN x4 + FP x1 | crazing | crazing | 细纹理/细长结构对比度弱，特征不够稳定；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_03 | rolled-in_scale_259.jpg | FN x4 + FP x1 | rolled-in_scale | rolled-in_scale | 局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_04 | rolled-in_scale_292.jpg | FN x4 + FP x1 | rolled-in_scale | rolled-in_scale | 局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_05 | crazing_278.jpg | FN x3 + FP x3 | crazing | crazing | 细纹理/细长结构对比度弱，特征不够稳定；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_06 | pitted_surface_279.jpg | FN x2 + FP x5 | inclusion, pitted_surface | patches, pitted_surface | 缺陷范围大且边界弥散，框的定位目标不够明确；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_07 | crazing_246.jpg | FN x3 + FP x2 | crazing | crazing | 细纹理/细长结构对比度弱，特征不够稳定；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_08 | crazing_250.jpg | FN x3 + FP x2 | crazing | crazing | 细纹理/细长结构对比度弱，特征不够稳定；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_09 | pitted_surface_280.jpg | FN x3 + FP x2 | pitted_surface | patches, pitted_surface | 局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |
| case_10 | scratches_293.jpg | FN x1 + FP x6 | scratches | scratches | 细纹理/细长结构对比度弱，特征不够稳定；局部背景纹理与缺陷模式相似，产生误检；同图目标较多，定位与去重更容易出错 |

## 人工分析建议

1. 先看橙色 FN 框：确认是完全没框出来，还是只框到局部。
2. 再看红色 FP 框：判断是不是背景纹理、边缘反光、相邻缺陷导致的误报。
3. 结合类别特性写原因：纹理细、边界弥散、目标密集、标签宽泛、分辨率不足。
4. 最后给出改进方向：调 imgsz、延长 epochs、换模型、调 cls、做针对性增强。
