"""
benchmark_api.py - Simple concurrent benchmark for FastAPI /detect endpoint.

用途：
1. 并发上传多张图片到 POST /detect
2. 记录每个请求的响应时间
3. 汇总平均响应时间、总耗时和 QPS

示例：
    python scripts/benchmark_api.py
    python scripts/benchmark_api.py --url http://127.0.0.1:8000/detect --num-images 10 --concurrency 10
"""

import argparse
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images", "val")
DEFAULT_URL = "http://127.0.0.1:8000/detect"
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Benchmark FastAPI /detect endpoint with concurrent requests."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Target detect API URL (default: http://127.0.0.1:8000/detect)",
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help="Directory of test images (default: data/images/val)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=10,
        help="Number of images to benchmark (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60)",
    )
    return parser.parse_args()


def collect_image_paths(image_dir, num_images):
    """收集待测图片路径。"""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    image_paths = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(SUPPORTED_SUFFIXES):
            image_paths.append(os.path.join(image_dir, filename))

    if not image_paths:
        raise FileNotFoundError(f"目录中没有可用图片: {image_dir}")

    if num_images <= 0:
        raise ValueError("--num-images 必须大于 0")

    return image_paths[:num_images]


def send_detect_request(url, image_path, timeout):
    """发送一次检测请求并返回结果统计。"""
    filename = os.path.basename(image_path)
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = "application/octet-stream"

    start_time = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            files = {
                "file": (filename, f, mime_type),
            }
            response = requests.post(url, files=files, timeout=timeout)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        response.raise_for_status()
        payload = response.json()

        return {
            "filename": filename,
            "success": True,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "count": payload.get("count"),
            "inference_time_ms": payload.get("inference_time_ms"),
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "filename": filename,
            "success": False,
            "status_code": None,
            "elapsed_ms": elapsed_ms,
            "error": str(exc),
        }


def main():
    args = parse_args()

    image_paths = collect_image_paths(args.image_dir, args.num_images)
    concurrency = max(1, min(args.concurrency, len(image_paths)))

    print(f"Target URL: {args.url}")
    print(f"Image dir : {os.path.abspath(args.image_dir)}")
    print(f"Images    : {len(image_paths)}")
    print(f"Concurrency: {concurrency}")
    print("-" * 60)

    results = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_detect_request, args.url, image_path, args.timeout)
            for image_path in image_paths
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result["success"]:
                print(
                    f"[OK] {result['filename']}: "
                    f"status={result['status_code']}, "
                    f"elapsed={result['elapsed_ms']:.2f} ms, "
                    f"count={result['count']}, "
                    f"inference={result['inference_time_ms']} ms"
                )
            else:
                print(
                    f"[FAIL] {result['filename']}: "
                    f"elapsed={result['elapsed_ms']:.2f} ms, "
                    f"error={result['error']}"
                )

    wall_elapsed = time.perf_counter() - wall_start

    success_results = [r for r in results if r["success"]]
    fail_results = [r for r in results if not r["success"]]

    avg_response_ms = (
        sum(r["elapsed_ms"] for r in success_results) / len(success_results)
        if success_results
        else 0.0
    )
    qps = len(success_results) / wall_elapsed if wall_elapsed > 0 else 0.0

    print("-" * 60)
    print("Summary")
    print(f"Success requests      : {len(success_results)}")
    print(f"Failed requests       : {len(fail_results)}")
    print(f"Average response time : {avg_response_ms:.2f} ms")
    print(f"Total wall time       : {wall_elapsed:.2f} s")
    print(f"QPS                   : {qps:.2f}")

    if fail_results:
        print("\nFailed request details:")
        for item in fail_results:
            print(f"- {item['filename']}: {item['error']}")


if __name__ == "__main__":
    main()
