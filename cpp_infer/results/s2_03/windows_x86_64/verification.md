# S2-03 Windows x86_64 verification

Date: 2026-08-30

## Environment and protocol

- Native Windows x86_64, MSVC 19.50.35721.0, Release, OpenCV 4.8.0, ONNX Runtime 1.19.2 CPUExecutionProvider.
- ORT execution is sequential with intra-op/inter-op threads fixed at 1/1 per worker.
- Formal comparison input: all 361 regular images under `data/images/val`, recursive directory discovery, JSON-only output, queue capacity 8.
- Worker=1 and worker=4 were separate processes from the same clean Release build.
- Throughput is `successful_images / processing_wall_time`; peak memory is Windows process Peak Working Set.

## Functional regression

- Clean Release build completed.
- CTest inventory: 156 entries; 156 passed. Inside the GTest binaries, two cases that require creating directory symlinks reported skip on this non-elevated Windows environment; their equivalent cases ran on Linux.
- Fixed image `crazing_241.jpg`: 3 detections and valid JSON/PNG.
- Frozen 30-image Python/C++ consistency: 30/30 images and 62/62 detections matched.
- Real batch integration covered directory and manifest inputs, worker=1/single-image equivalence, worker=1/multi-worker equivalence, a UTF-8 manifest image name, exact `2 succeeded + 1 failed`, overwrite refusal, output PNG, and cooperative interrupt exit 130.

## Formal 361-image comparison

| Run | Success | Processing wall | Throughput | Peak Working Set |
|---|---:|---:|---:|---:|
| worker=1 | 361/361 | 57,433.2627 ms | 6.285556 img/s | 151.804688 MiB |
| worker=4 | 361/361 | 20,219.6458 ms | 17.853923 img/s | 505.085938 MiB |

- Throughput ratio: 2.840468x; absolute delta: +11.568367 img/s.
- Peak Working Set delta: +370,442,240 bytes (+353.281250 MiB).
- All 361 ordered per-image Detection JSON files were byte-identical and semantically identical between the two runs.
- Concurrency being faster is an observed result, not a pass threshold; the memory increase is expected from four independent Pipeline/ORT sessions.

## Machine-readable evidence

- `workers_1/batch_summary.json`
- `workers_4/batch_summary.json`
- `comparison.json`

The generated `items/` trees are intentionally ignored because `comparison.json` records the complete ordered parity result. These numbers are comparable only within this recorded Windows run and must not be compared directly with Linux RSS or QEMU observations.
