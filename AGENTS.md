# 工业视觉边缘 AI Runtime 与 C++ 工程化系统

Industrial Vision Edge AI Runtime and C++ Engineering System，服务于秋招简历、面试与实际开发学习。

当前已实现基于 C++17、CMake、OpenCV 和 ONNX Runtime 的 FP32/INT8 推理、单图与目录/manifest 有界并发处理、结果比较、benchmark 和 profiling。业务主链为 `RuntimeConfig + ModelArtifactSpec -> DetectorPipeline -> BatchRunner`，已在 Windows x86_64、WSL2/Linux x86_64 和 Linux AArch64/QEMU 完成功能验证；QEMU 记录的是模拟环境中的功能结果。

- `docs/archive/` 是旧文档与历史记录，除非用户要求，否则无需读取，不作为当前工作规则。
- 仅在用户明确要求“九部分输出”时使用 [简短模板](docs/learning_closure.md)。
- 避免冗余防御工程、重复验证和无必要的 SHA；回复直接说明结果，不反复强调假想风险或边界。
