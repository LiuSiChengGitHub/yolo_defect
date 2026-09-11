# S2-03 WSL2/Linux x86_64 verification

Date: 2026-08-30

## Environment and protocol

- WSL2 Ubuntu 24.04 on Linux x86_64, GCC 13, Ninja Release, OpenCV 4.6.0, ONNX Runtime 1.19.2 CPUExecutionProvider.
- ORT execution is sequential with intra-op/inter-op threads fixed at 1/1 per worker.
- Formal comparison input: all 361 regular images under `data/images/val`, recursive directory discovery, JSON-only output, queue capacity 8.
- Worker=1 and worker=4 were separate processes from the same clean Release build.
- Throughput is `successful_images / processing_wall_time`; peak memory is Linux process peak RSS.

## Functional regression

- Clean Release build completed; 11 built ELF executables were inspected with `ldd`, and all dynamic dependencies resolved.
- CTest: 156/156 passed, including the symlink/path-containment tests available on Linux.
- Fixed image `crazing_241.jpg`: 3 detections and valid JSON/PNG.
- Frozen 30-image Python/C++ consistency: 30/30 images and 62/62 detections matched.
- Frozen 30-image batch smoke with worker=2 and queue=4: 30/30 succeeded at 9.780699 img/s; this smoke is not the formal worker comparison.
- Single-image Release benchmark (100 repeats): pipeline mean 120.423122 ms, 8.304053 img/s, peak RSS 197.179688 MiB.

## Formal 361-image comparison

| Run | Success | Processing wall | Throughput | Peak RSS |
|---|---:|---:|---:|---:|
| worker=1 | 361/361 | 44,492.069286 ms | 8.113806 img/s | 205.765625 MiB |
| worker=4 | 361/361 | 17,907.115330 ms | 20.159584 img/s | 588.226563 MiB |

- Throughput ratio: 2.484603x; absolute delta: +12.045779 img/s.
- Peak RSS delta: +401,039,360 bytes (+382.460938 MiB).
- All 361 ordered per-image Detection JSON files were byte-identical and semantically identical between the two runs.
- Both formal runs used the same copied 361-image/config/artifact/model set and JSON-only outputs in one WSL-native ext4 work area; the resulting evidence was copied back to this directory.
- Concurrency being faster is an observed result, not a pass threshold; the memory increase reflects four independent Pipeline/ORT sessions.

## Machine-readable evidence

- `performance/batch_workers_1/batch_summary.json`
- `performance/batch_workers_4/batch_summary.json`
- `performance/batch_comparison.json`
- `regression/benchmark/yolov8_neu_det_cpu_release.json`
- `regression/consistency/summary.json`
- `regression/batch/batch_summary.json`

The generated `items/` trees are intentionally ignored. This is WSL2/Linux evidence, not a native bare-metal edge host result. Linux RSS and Windows Peak Working Set are different platform metrics and are compared only within their own platform.
