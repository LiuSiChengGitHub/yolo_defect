"""
detector.py - YOLODetector class for ONNX inference.

Encapsulates ONNX model loading, preprocessing, inference with NMS,
and result visualization. Designed to be reused by FastAPI service.

Usage:
    from src.detector import YOLODetector

    detector = YOLODetector("models/best.onnx")
    detections = detector.predict(image)
    result = detector.draw(image, detections, class_names)
"""

import cv2
import numpy as np
import onnxruntime as ort


class YOLODetector:
    """ONNX-based YOLOv8 detector for steel surface defect detection.

    为什么要单独封装这个类？
    1. 关注点分离：推理逻辑独立于训练框架，不依赖 ultralytics
    2. 复用性：inference_onnx.py 和 FastAPI 都直接 import 这个类
    3. 可测试性：可以单独对这个类写单元测试

    推理流程：preprocess → ONNX Runtime 推理 → 后处理（NMS）→ 可视化

    Attributes:
        session: ONNX Runtime 推理会话
        input_name: 模型输入张量的名称
        input_shape: 期望的输入形状 (batch, channels, height, width)
        conf_thresh: 置信度阈值，低于此值的检测结果被过滤
        iou_thresh: NMS 的 IoU 阈值，重叠度高于此值的框被抑制
    """

    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        """Initialize detector with ONNX model.

        Args:
            model_path: ONNX 模型文件路径
            conf_thresh: 置信度阈值（默认 0.25，即只保留 >25% 把握的检测）
            iou_thresh: NMS 的 IoU 阈值（默认 0.45，重叠 >45% 的框只保留最高分的）
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # 加载 ONNX 模型
        # providers 列表定义推理后端的优先级：先尝试 GPU（CUDA），不行就用 CPU
        # 面试考点：ONNX Runtime 支持多种 ExecutionProvider（CUDA、TensorRT、DirectML、CPU）
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 获取模型输入信息
        # YOLOv8 的输入通常是 [1, 3, 640, 640]（batch=1, RGB 3通道, 640x640）
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = model_input.shape
        self.input_h = self.input_shape[2]  # 输入高度（如 640）
        self.input_w = self.input_shape[3]  # 输入宽度（如 640）

    def preprocess(self, image):
        """Preprocess image for model input.

        图片预处理流程（必须和训练时一致，否则推理结果会出问题）：

        1. BGR → RGB：OpenCV 默认读取 BGR 格式，但模型训练用 RGB
        2. Resize：缩放到模型期望尺寸（如 640x640）
        3. Normalize：像素值从 0-255 归一化到 0-1（float32）
        4. HWC → CHW：图片维度从 (H, W, C) 转为 (C, H, W)（PyTorch/ONNX 的标准格式）
        5. 添加 batch 维度：从 (C, H, W) 变为 (1, C, H, W)

        面试考点：预处理必须与训练时完全一致，否则模型性能会急剧下降！

        Args:
            image: 输入 BGR 图片（numpy 数组，HWC 格式）

        Returns:
            预处理后的 float32 numpy 数组 [1, 3, H, W]
            缩放因子 (scale_x, scale_y) 用于将检测结果映射回原图尺寸
        """
        orig_h, orig_w = image.shape[:2]

        # Step 1: BGR → RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 2: Resize 到模型输入尺寸
        # 注意：这里用简单的 resize，没有做 letterbox（保持长宽比的填充）
        # YOLOv8 训练时用的是 letterbox，但简单 resize 在正方形图片（200x200）上差异不大
        img = cv2.resize(img, (self.input_w, self.input_h))

        # Step 3: 归一化到 0-1
        img = img.astype(np.float32) / 255.0

        # Step 4: HWC → CHW（把通道维度放到最前面）
        img = np.transpose(img, (2, 0, 1))

        # Step 5: 添加 batch 维度 → [1, 3, H, W]
        img = np.expand_dims(img, axis=0)

        # 计算缩放因子，后续用来把检测框坐标映射回原图
        scale_x = orig_w / self.input_w
        scale_y = orig_h / self.input_h

        return img, (scale_x, scale_y)

    def predict(self, image):
        """Run inference on an image and return detections after NMS.

        完整推理流程：预处理 → ONNX 推理 → 解析输出 → 置信度过滤 → NMS

        Args:
            image: 输入 BGR 图片（numpy 数组）

        Returns:
            检测结果列表，每个元素是字典：
            {"bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int}
        """
        # 预处理
        input_tensor, (scale_x, scale_y) = self.preprocess(image)

        # ONNX Runtime 推理
        # session.run(输出名列表, 输入字典)
        # None 表示获取所有输出
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # ============================================================
        # 解析 YOLOv8 输出
        # ============================================================
        # YOLOv8 输出形状：[1, num_classes+4, num_predictions]
        # - 前 4 个值是 cx, cy, w, h（中心坐标和宽高，相对于输入尺寸）
        # - 后 num_classes 个值是每个类别的置信度分数
        #
        # 转置后变为 [num_predictions, 4+num_classes]，方便按行处理
        output = outputs[0]
        output = np.transpose(output[0])  # [num_predictions, 4+num_classes]

        # 分离坐标和分数
        boxes_xywh = output[:, :4]   # 前 4 列：cx, cy, w, h
        scores = output[:, 4:]        # 后面的列：各类别置信度

        # 取每个预测的最高分类别
        class_ids = np.argmax(scores, axis=1)      # 最高分的类别 ID
        confidences = np.max(scores, axis=1)        # 最高分的置信度值

        # ============================================================
        # 置信度过滤
        # ============================================================
        # 只保留置信度 > conf_thresh 的预测
        # YOLOv8 通常输出 8400 个预测框，大部分置信度很低，需要过滤
        mask = confidences > self.conf_thresh
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # ============================================================
        # 坐标格式转换：中心格式 → 角点格式
        # ============================================================
        # (cx, cy, w, h) → (x1, y1, x2, y2)
        # x1 = cx - w/2, y1 = cy - h/2 （左上角）
        # x2 = cx + w/2, y2 = cy + h/2 （右下角）
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # 将坐标从模型输入尺寸（640x640）映射回原图尺寸（200x200）
        boxes_xyxy[:, 0] *= scale_x
        boxes_xyxy[:, 1] *= scale_y
        boxes_xyxy[:, 2] *= scale_x
        boxes_xyxy[:, 3] *= scale_y

        # ============================================================
        # NMS（非极大值抑制）
        # ============================================================
        # 面试高频考点！
        # 问题：同一个目标可能被多个框检测到，怎么只保留最好的那个？
        # NMS 算法：
        # 1. 按置信度从高到低排序
        # 2. 取最高分的框，与其余所有框计算 IoU
        # 3. IoU > 阈值的框被抑制（删除）
        # 4. 重复步骤 2-3 直到所有框都处理完
        indices = self._nms(boxes_xyxy, confidences, self.iou_thresh)

        # 组装最终检测结果
        detections = []
        for i in indices:
            detections.append({
                "bbox": boxes_xyxy[i].tolist(),
                "confidence": float(confidences[i]),
                "class_id": int(class_ids[i]),
            })

        return detections

    @staticmethod
    def _nms(boxes, scores, iou_thresh):
        """Non-Maximum Suppression (非极大值抑制).

        手动实现 NMS，不依赖 torchvision.ops.nms。
        面试可能要求手写这个算法！

        IoU (Intersection over Union) 计算：
            IoU = 两个框交集面积 / 两个框并集面积
            IoU = 1 表示完全重叠，IoU = 0 表示没有重叠

        Args:
            boxes: [N, 4] 的边界框数组 (x1, y1, x2, y2)
            scores: [N] 的置信度数组
            iou_thresh: IoU 阈值

        Returns:
            保留的框索引列表
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)  # 每个框的面积

        # 按置信度从高到低排序
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            # 取当前最高分的框
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # 计算当前框与其余所有框的 IoU
            # 交集区域的左上角 = 两个框左上角的最大值
            # 交集区域的右下角 = 两个框右下角的最小值
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            # 交集面积（如果没有交集则为 0）
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            # IoU = 交集 / 并集
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 只保留 IoU <= 阈值的框（即与当前框重叠不大的框）
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]  # +1 因为 order[1:] 的索引从 0 开始

        return keep

    @staticmethod
    def draw(image, detections, class_names):
        """Draw bounding boxes with class names and confidence scores.

        在图片上画检测框、类名和置信度。

        Args:
            image: 输入 BGR 图片（会复制，不修改原图）
            detections: predict() 返回的检测结果列表
            class_names: 类名字符串列表

        Returns:
            带标注的图片副本
        """
        img = image.copy()  # 不修改原图
        # 每个类别一种颜色（BGR 格式）
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
        ]

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            cls_id = det["class_id"]

            color = colors[cls_id % len(colors)]
            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            label = f"{cls_name} {conf:.2f}"

            # 画边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 画标签背景（填充矩形，让文字更清晰）
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)  # -1 表示填充
            cv2.putText(img, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img
