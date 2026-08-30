# S2-03 Linux AArch64/QEMU functional verification

Date: 2026-08-30

## Evidence boundary

- Host: WSL2/Linux x86_64. Target binary: Linux AArch64. Execution: QEMU user-mode.
- Cross compiler: GCC 13; target OpenCV 4.6.0; target ONNX Runtime 1.19.2 CPUExecutionProvider.
- `BatchSummary.environment.target_architecture` is `aarch64`, `runtime_kernel_architecture` is the injected real host-kernel `x86_64`, and `execution_context` is `qemu_user_mode_on_x86_64_host`.
- Every QEMU summary marks peak-memory evidence `publishable: false`. No QEMU throughput, latency, RSS, native-board, power, thermal, or deployment-performance conclusion is published.

## Cross-build and functional results

- Clean AArch64 core and complete Runtime/CLI cross-build passed.
- `file`/`readelf` confirmed AArch64 ELF and interpreter `/lib/ld-linux-aarch64.so.1`.
- The target loader resolved 138 target libraries with no missing or x86_64 library.
- QEMU startup/help/config/negative contracts and the project-core smoke passed.
- Fixed image `crazing_241.jpg` completed the AArch64 OpenCV + ORT CPU + existing postprocess path with 3 detections.
- Directory input, worker=1, queue=2: 2/2 succeeded.
- Equivalent manifest input, worker=2, queue=1: 2/2 succeeded.
- The two runs' ordered per-image Detection JSON files were byte-identical and semantically identical.
- Manifest with one corrupt JPEG: exact `2 succeeded + 1 failed`, `status=partial_failure`, exit code 2.
- All three BatchSummary files passed the strict schema, order, count, provider, target architecture, runtime kernel architecture, execution-context, and non-publishable-memory checks.

## Machine-readable evidence

- `regression/elf_inspection.txt`
- `regression/loader_resolution.txt`
- `regression/qemu_smoke.txt`
- `regression/detect/crazing_241.detections.json`
- `final_20260830_r2/qemu_batch_acceptance.txt`
- `final_20260830_r2/directory_workers1/batch_summary.json`
- `final_20260830_r2/manifest_workers2/batch_summary.json`
- `final_20260830_r2/partial_failure/batch_summary.json`

Generated `inputs/` and `items/` are intentionally ignored. This proves build, loader, and bounded-batch functional portability under QEMU user-mode; it does not prove behavior or performance on a physical ARM64 device.
