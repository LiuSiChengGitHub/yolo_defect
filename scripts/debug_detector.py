"""调试工具：打印预处理中间值和 ONNX 输出张量形状，用于排查推理问题。"""

import os
import sys
import argparse

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.detector import YOLODetector

def parse_args():
    parser = argparse.ArgumentParser(description="Debug detector preprocessing and ONNX output shapes")
    parser.add_argument("--model", type=str, default="models/best.onnx", help="ONNX model path")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    return parser.parse_args()


def main():
    args = parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    detector = YOLODetector(args.model)

    print(f"1. original image shape: {image.shape}")

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (detector.input_w, detector.input_h))
    print(f"2. resized image shape: {img.shape}")

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    print(f"3. CHW tensor shape: {img.shape}")

    input_tensor = np.expand_dims(img, axis=0)
    print(f"4. batched input shape: {input_tensor.shape}")

    outputs = detector.session.run(None, {detector.input_name: input_tensor})
    print(f"5. raw ONNX output shape: {outputs[0].shape}")



if __name__ == "__main__":
    main()




