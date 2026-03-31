"""ONNX 推理封装：模型加载、预处理、NMS 后处理、可视化。供推理脚本和 FastAPI 复用。"""

import os
import sys


def _add_cuda_dll_dirs():
    # 将 CUDA/cuDNN 的 DLL 目录加入 Windows DLL 搜索路径。

    if sys.platform != "win32":
        return

    # conda env 的 bin/ 目录：含 cudart64_110.dll, cublas64_11.dll, cufft64_10.dll
    env_bin = os.path.join(sys.prefix, "bin")
    # torch/lib/ 目录：含 cudnn64_8.dll
    try:
        torch_lib = os.path.join(
            os.path.dirname(__import__("torch").__file__), "lib"
        )
    except ImportError:
        torch_lib = None

    dirs = [d for d in [env_bin, torch_lib] if d and os.path.isdir(d)]

    for d in dirs:
        os.add_dll_directory(d)

    # 把路径加到 PATH 前面，兼容旧式 DLL 搜索
    path_env = os.environ.get("PATH", "")
    prepend = ";".join(dirs)
    os.environ["PATH"] = prepend + ";" + path_env


# 必须在 import onnxruntime 之前执行
_add_cuda_dll_dirs()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


class YOLODetector:
    # 推理流程：preprocess → ONNX Runtime 推理 → 后处理（NMS）→ 可视化


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
        # 图片预处理流程（BGR → RGB，Letterbox，Normalize，HWC → CHW，添加 batch 维度）：

        orig_h, orig_w = image.shape[:2]

        # Step 1: BGR → RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Step 2: Letterbox —— 保持长宽比缩放，四周填灰（与 Ultralytics 训练预处理一致）
        scale = min(self.input_h / orig_h, self.input_w / orig_w)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 居中填充（整数像素，左/上取 floor，右/下补齐余量）
        pad_left = (self.input_w - new_w) // 2
        pad_top  = (self.input_h - new_h) // 2
        pad_right  = self.input_w - new_w - pad_left
        pad_bottom = self.input_h - new_h - pad_top
        img = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # Step 3: 归一化到 0-1
        img = img.astype(np.float32) / 255.0

        # Step 4: HWC → CHW（把通道维度放到最前面）
        img = np.transpose(img, (2, 0, 1))

        # Step 5: 添加 batch 维度 → [1, 3, H, W]
        img = np.expand_dims(img, axis=0)

        # 返回 letterbox 参数，供坐标逆变换使用
        return img, (scale, pad_left, pad_top, orig_w, orig_h)

    def predict(self, image):

        # 完整推理流程：预处理 → ONNX 推理 → 解析输出 → 置信度过滤 → NMS

        # 预处理
        input_tensor, (scale, pad_left, pad_top, orig_w, orig_h) = self.preprocess(image)

        # ONNX Runtime 推理
        # session.run(输出名列表, 输入字典)
        # None 表示获取所有输出
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 解析 YOLOv8 输出，转置后变为 [num_predictions, 4+num_classes]，方便按行处理
        output = outputs[0]
        output = np.transpose(output[0])  # [num_predictions, 4+num_classes]

        # 分离坐标和分数
        boxes_xywh = output[:, :4]   # 前 4 列：cx, cy, w, h（在模型输入空间，如 640x640）
        scores = output[:, 4:]        # 后面的列：各类别置信度

        # 取每个预测的最高分类别
        class_ids = np.argmax(scores, axis=1)      # 最高分的类别 ID
        confidences = np.max(scores, axis=1)        # 最高分的置信度值

        # 置信度过滤
        mask = confidences > self.conf_thresh
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        # 坐标格式转换：中心格式 → 角点格式（仍在模型输入空间）
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # NMS 在模型输入空间执行（与 Ultralytics 对齐）
        indices = self._nms(boxes_xyxy, confidences, self.iou_thresh)

        # Letterbox 逆变换：先减去 padding 偏移，再除以缩放因子，映射回原图空间
        detections = []
        for i in indices:
            x1 = float(np.clip((boxes_xyxy[i, 0] - pad_left) / scale, 0, orig_w))
            y1 = float(np.clip((boxes_xyxy[i, 1] - pad_top)  / scale, 0, orig_h))
            x2 = float(np.clip((boxes_xyxy[i, 2] - pad_left) / scale, 0, orig_w))
            y2 = float(np.clip((boxes_xyxy[i, 3] - pad_top)  / scale, 0, orig_h))
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidences[i]),
                "class_id": int(class_ids[i]),
            })

        return detections

    @staticmethod
    def _nms(boxes, scores, iou_thresh):

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
