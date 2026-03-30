"""FastAPI 服务入口：GET /health 健康检查，POST /detect 上传图片返回检测结果。"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 用脚本所在目录推导项目根目录，避免写死绝对路径。
api_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(api_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.detector import YOLODetector


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

DEFAULT_MODEL_PATH = os.path.join(project_root, "models", "best.onnx")
DEFAULT_CONF_THRESH = 0.25
DEFAULT_IOU_THRESH = 0.45
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/webp",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("yolo_defect.api")


def _create_detector(model_path):
    """创建 ONNX 检测器实例。"""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    return YOLODetector(
        model_path=model_path,
        conf_thresh=DEFAULT_CONF_THRESH,
        iou_thresh=DEFAULT_IOU_THRESH,
    )


def _decode_image(file_bytes):
    """把上传的二进制内容解码成 OpenCV BGR 图片。"""
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("上传文件不是可解析的图片，无法完成解码。")
    return image


def _format_detections(detections):
    """把 detector.predict() 的结果整理成更适合 API 返回的 JSON 结构。"""
    results = []
    for det in detections:
        class_id = det["class_id"]
        class_name = (
            CLASS_NAMES[class_id]
            if 0 <= class_id < len(CLASS_NAMES)
            else f"class_{class_id}"
        )
        bbox = [round(float(v), 2) for v in det["bbox"]]
        results.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(float(det["confidence"]), 4),
                "bbox": bbox,
            }
        )
    return results


@asynccontextmanager
async def lifespan(app_instance):
    """FastAPI 生命周期：服务启动时只加载一次模型。"""
    app_instance.state.model_path = DEFAULT_MODEL_PATH
    app_instance.state.detector = None
    app_instance.state.load_error = None
    app_instance.state.request_count = 0
    app_instance.state.total_response_time_ms = 0.0

    try:
        logger.info(
            "Loading detector from %s",
            os.path.relpath(DEFAULT_MODEL_PATH, project_root),
        )
        app_instance.state.detector = _create_detector(DEFAULT_MODEL_PATH)
        logger.info("Detector loaded successfully.")
    except Exception as exc:
        app_instance.state.load_error = str(exc)
        logger.exception("Failed to load detector during startup.")

    yield


app = FastAPI(
    title="YOLO Defect Detection API",
    description="Minimal FastAPI service for ONNX-based steel defect detection.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_request_and_timing(request: Request, call_next):
    """记录请求日志，并统计总响应时间。"""
    start_time = time.perf_counter()
    client = request.client.host if request.client else "unknown"

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        app.state.request_count += 1
        app.state.total_response_time_ms += elapsed_ms
        logger.exception(
            "request_failed method=%s path=%s client=%s elapsed=%.2fms",
            request.method,
            request.url.path,
            client,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    app.state.request_count += 1
    app.state.total_response_time_ms += elapsed_ms
    response.headers["X-Response-Time-MS"] = f"{elapsed_ms:.2f}"

    logger.info(
        "request method=%s path=%s status=%s client=%s elapsed=%.2fms",
        request.method,
        request.url.path,
        response.status_code,
        client,
        elapsed_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理 422 参数校验错误。"""
    logger.warning(
        "validation_error method=%s path=%s detail=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数校验失败。",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception):
    """处理未捕获异常，避免直接把栈暴露给客户端。"""
    logger.exception(
        "unexpected_error method=%s path=%s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误，请查看服务端日志。",
        },
    )


@app.get("/health")
def health_check():
    """健康检查接口：确认服务活着，模型是否已经准备好。"""
    model_name = os.path.basename(app.state.model_path)
    avg_response_time_ms = (
        app.state.total_response_time_ms / app.state.request_count
        if app.state.request_count > 0
        else 0.0
    )

    if app.state.detector is None:
        return {
            "status": "error",
            "model": model_name,
            "detail": app.state.load_error or "detector not loaded",
            "request_stats": {
                "total_requests": app.state.request_count,
                "avg_response_time_ms": round(avg_response_time_ms, 2),
            },
        }

    return {
        "status": "ok",
        "model": model_name,
        "request_stats": {
            "total_requests": app.state.request_count,
            "avg_response_time_ms": round(avg_response_time_ms, 2),
        },
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """上传图片并返回检测结果。
    1. 接收上传文件
    2. 解码成 OpenCV 图片
    3. 调用 detector.predict()
    4. 整理成 JSON 返回
    """
    if app.state.detector is None:
        raise HTTPException(
            status_code=503,
            detail=app.state.load_error or "模型未加载，暂时无法推理。",
        )

    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            "bad_request path=/detect filename=%s content_type=%s reason=unsupported_type",
            file.filename,
            file.content_type,
        )
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {file.content_type}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        logger.warning(
            "bad_request path=/detect filename=%s reason=empty_file",
            file.filename,
        )
        raise HTTPException(status_code=400, detail="上传文件为空。")

    try:
        image = _decode_image(file_bytes)
    except ValueError as exc:
        logger.warning(
            "bad_request path=/detect filename=%s reason=decode_failed",
            file.filename,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_time = time.perf_counter()
    try:
        detections = app.state.detector.predict(image)
    except Exception as exc:
        logger.exception("Inference failed for file=%s", file.filename)
        raise HTTPException(status_code=500, detail="模型推理失败。") from exc
    inference_time_ms = (time.perf_counter() - start_time) * 1000.0

    formatted = _format_detections(detections)
    height, width = image.shape[:2]

    logger.info(
        "detect file=%s type=%s size=%sx%s count=%s inference=%.1fms",
        file.filename or "uploaded",
        file.content_type or "unknown",
        width,
        height,
        len(formatted),
        inference_time_ms,
    )

    return {
        "filename": file.filename,
        "count": len(formatted),
        "image_size": {
            "width": width,
            "height": height,
        },
        "model": os.path.basename(app.state.model_path),
        "conf_thresh": DEFAULT_CONF_THRESH,
        "iou_thresh": DEFAULT_IOU_THRESH,
        "inference_time_ms": round(inference_time_ms, 2),
        "detections": formatted,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
