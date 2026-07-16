# Codex Collaboration Rules

## Source of Truth and Priority

`docs/PLAN.md` is the latest source of truth for project positioning, top-level design, stage boundaries, and teaching-oriented advancement rules. Read it before planning or implementing a large stage. Do not edit it unless the user explicitly asks.

When the top-level design and a short large-stage description appear incomplete or inconsistent, the top-level design wins. Supplement the large-stage execution plan with the missing top-level requirements instead of narrowing the project to the shorter wording.

`README.md` and `README_zh.md` are the unique project entry points and must remain aligned. Long implementation details may live under `docs/` only when putting them in README would make the main story harder to use. The current large-stage-one breakdown is the intentionally long execution artifact `docs/STAGE1_EXECUTION_PLAN.md`.

## Current Goal and Positioning

Project V2 is an **industrial vision edge AI Runtime and C++ engineering system** for 2026 autumn recruiting. The goal is to turn a public industrial-vision model artifact into configurable, runnable, testable, comparable, and reproducible C++ inference software for edge deployment.

The interview-facing chain is:

```text
model artifact
-> model card / artifact contract
-> config and schema validation
-> OpenCV preprocess
-> ONNX Runtime C++ inference
-> decode / score filter / NMS / coordinate restore
-> detection JSON / visualization
-> Python / ONNX / C++ consistency evidence
-> benchmark and memory evidence
-> INT8 PTQ validation
-> optional real-device deployment
-> optional sample inference_event / Project 2 bridge
-> tests, failure records, and README evidence
```

YOLOv8 and NEU-DET are the stable P0 Runtime baseline carriers. The later `paper_detect` D010 artifact uses D-FINE-S on DeepPCB and is a research-side artifact source, not training logic owned by this repository.

The learning direction is AI application engineering, built on modern C++, Linux, testing/debugging, performance analysis, and model-inference engineering, with edge deployment as the main differentiator. Use AI coding to accelerate implementation while preserving interview-level understanding of the architecture and core logic.

## P0 Top-Level Design

The complete P0 is the authority for the first two large stages. It includes:

1. A C++17/CMake multi-target project with header/source separation and explicit dependencies.
2. Runtime config and artifact-schema validation for model path, input size, classes, thresholds, and preprocess/postprocess types.
3. OpenCV preprocessing: letterbox, BGR/RGB conversion, normalization, and HWC-to-CHW layout.
4. An ONNX Runtime C++ session with RAII plus input/output name, shape, dtype, and provider checks.
5. Stable-model decode, score filtering, NMS, and coordinate restoration.
6. Detection JSON and visualization output with a fixed sample and reproducible command.
7. Fixed-sample Python/ONNX/C++ consistency checks covering tolerance, detection count, class, confidence, and coordinate error.
8. A benchmark with warmup/repeat, P50/P95, throughput, segmented timings, and peak memory/RSS when feasible.
9. INT8 PTQ validation; start QAT only if PTQ degradation is material and schedule allows it.
10. GTest coverage for config, non-square letterbox, color/layout, NMS, coordinate restoration, and invalid inputs.
11. Failure injection for missing model, wrong shape/dtype, class mismatch, empty output, and damaged image.
12. README evidence covering positioning, architecture, Quick Start, tests, results, limitations, and reproducible environment.

Large stage one creates the first deliverable vertical slice and the minimum evidence needed to explain it. Large stage two completes the broader P0 evidence matrix, fault injection, reproducible benchmark/memory record, and INT8 comparison. Do not move INT8 or the full hardening matrix into large stage one merely to make the first slice broader; do not defer requirements that large stage one's own exit explicitly needs.

## P1 Extensions and Stop Conditions

After P0 is stable, choose extensions by recruiting value rather than feature count:

- **P1-A, C++ product/system software:** directory batch processing, bounded queues/workers, backpressure, clean shutdown, and single-thread versus concurrent throughput.
- **P1-B, inference deployment:** real TensorRT/Jetson/ARM deployment with correctness, latency, memory, and temperature comparisons. Do not claim or prioritize this without the required hardware.
- **P1-C, EDA/Windows/product UI:** a Qt result viewer only when several high-priority job descriptions require Qt.
- **P1-D, serving/model platform:** gRPC or Triton only when several high-priority job descriptions require it.
- **P1-E, D010 artifact integration:** consume the research artifact only after its export and Runtime contract are stable.

Do not broaden P0 with speculative UI, serving, concurrency, or hardware work before the deliverable chain and evidence are complete.

## Model Artifact Rules

- The existing YOLOv8/NEU-DET artifact is the stable P0 baseline.
- The tracked `models/best.onnx` preflight record is 12,336,935 bytes, opset 17, SHA-256 `7B8A37610018A6AE6CACDFC869590A95BBE31AFB7579C39BE0FFEC537196AF68`, input `images` float32 `[1,3,800,800]`, output `output0` float32 `[1,10,13125]`, and metadata `nms=False`. C++ must still inspect actual metadata at runtime instead of trusting this record.
- The ONNX metadata reports `AGPL-3.0` while repository source is MIT. Treat model provenance and distribution compatibility as an explicit checkpoint before release; do not imply that the model artifact inherits the source-code license without verification.
- `paper_detect` D010 is the proposed research method. D-FINE-S + DeepPCB is the strong baseline context; D003 is the adaptive-prior input-adapter reference/proposed ancestor and ablation anchor.
- D010 keeps the D003 inference path and does not include D009 feature-pyramid injection. Its key addition is training-time template-counterfactual erase/replay sampling.
- Recorded research evidence: D010 formal-validation AP50-95 is `0.847057`; official-test AP50-95 is `0.830385`; all six class deltas over D003 are positive on both evaluations; D010A erase-only and D010B replay-only beat D003 but remain below full D010.
- These are research-side results, not C++ Runtime results. D010 enters this repository's delivered results only after stable ONNX export, an artifact/result card and deployment contract, real Runtime integration, and consistency validation.
- Never describe the historical Python ONNX Runtime `24.4/72.1 FPS` as current C++ Runtime performance.

## Current Stage Roadmap

### Completed Baseline and Engineering Skeleton — through 2026-07-12

The verified reusable base is:

```text
YOLOv8/NEU-DET training and tuning
-> ONNX export
-> 50-image PyTorch/ONNX detection-count check
-> Python ONNX Runtime / FastAPI / Docker / historical benchmark
-> C++17/CMake/CTest entry
-> typed RuntimeConfig
-> real-image OpenCV letterbox / RGB / normalize / NCHW preprocess
```

This proves a baseline only. C++ ONNX Runtime, postprocess, C++ consistency, and C++ benchmark are still pending.

The 2026-07-15 audit established these execution facts:

- A fresh `%TEMP%` NMake build with MSVC 19.50.35721.0 and OpenCV 4.8.0 passes the existing 3/3 CTest smokes. The ignored `cpp_infer/build` executable is a stale P1-01 artifact and must not be used as evidence; use a fresh out-of-tree build.
- No native ONNX Runtime C++ header/import library was found locally. Only Python `onnxruntime-gpu 1.19.2` runtime files exist, so S1-01 must provision and pin a real C++ SDK before session work.
- The historical 50-image PT/ONNX check contains the sorted first 50 files, all from `crazing`, and checks counts/confidence summaries rather than class/box tolerances. Keep it as weak historical evidence and create new six-class Python ORT/C++ evidence in large stage one.
- The matching `best.pt` checkpoint is not currently present. Do not claim a newly rerun direct PT/Python-ORT/C++ three-way comparison unless that exact artifact is legitimately restored and verified.

The 2026-07-16 pre-stage readiness pass resolved the dependency-preparation gap without starting S1-01:

- The official Windows x64 CPU ONNX Runtime C++ SDK 1.19.2 is now present outside the repository at `D:\01_Base\Tools\onnxruntime-win-x64-1.19.2`; `onnxruntime_cxx_api.h`, `onnxruntime.lib`, and `onnxruntime.dll` were all verified. CMake must consume it through `ONNXRUNTIME_ROOT` or an equivalent configurable cache entry, never a committed personal absolute path.
- `VsDevCmd.bat` successfully exposes x64 `cl`, `nmake`, CMake, and CTest. A new `%TEMP%` Release/NMake build with MSVC 19.50.35721.0 and OpenCV 4.8.0 again passes 3/3 CTest smokes.
- The future GTest dependency is frozen to v1.17.0 commit `52eb8108c5bdec04579160ae17225d66034bd723` via a SHA-256-pinned HTTPS archive and `FetchContent`; it is not integrated until S1-01.
- Model provenance and license evidence is recorded in `docs/PRE_STAGE1_READINESS.md`. The project owner confirms that the tracked ONNX was personally exported from `runs/detect/final_train_2/weights/best.pt`; the workspace and Git history contain no `.pt`, so this is owner-confirmed lineage rather than a newly reproducible re-export.
- The declared use is personal learning, so an Enterprise license is not a prerequisite for local implementation. The owner chose to keep publicly distributing `models/best.onnx` and NEU-DET; the model's AGPL-3.0 metadata and the unspecified NEU-DET redistribution terms therefore remain explicit release checkpoints, separate from the MIT source license.
- This readiness record changes no Runtime behavior and does not mark S1-01 as started.

### Large Stage One — Project 1 Deliverable Loop, 2026-07-13 to 2026-07-27

Required exit:

```text
fixed config/image/model command
-> ONNX Runtime C++
-> decode/filter/NMS/coordinate restore
-> detection JSON and visualization
-> fixed-sample Python/ONNX/C++ consistency
-> preprocess/infer/postprocess/end-to-end P50/P95
-> explicit core errors and automated main-path tests
```

The user must also be able to explain the chain for five minutes without AI and independently change one core behavior plus its test. Follow `docs/STAGE1_EXECUTION_PLAN.md`, execute one small stage at a time, and stop for acceptance after each one.

### Large Stage Two — Project 1 Evidence Hardening, 2026-07-28 to 2026-08-10

Complete the full config/preprocess/NMS/coordinate/invalid-input test matrix, failure injection, reproducible benchmark and memory record, and at least one INT8 PTQ comparison. Start QAT only when PTQ degradation and remaining time justify it. D010 may enter only after the artifact gates above. Close with a final result table, resume bullets, interview questions, and a focused mock interview.

### Large Stage Three — Project 1 P1 Extensions

Attempt real TensorRT/Jetson/ARM deployment only with suitable hardware, and integrate D010 only when its stable ONNX artifact and deployment contract exist. Other P1 extensions remain job-description gated.

### Large Stage Four — Freeze, Rolling Applications, Interview Priority, from 2026-08-25

Freeze P0 feature scope. Allow only demo/correctness/reproduction fixes, tests and evidence, result/evaluation/failure-case improvements, small job-specific patches, and adjustments based on real interview feedback. Continue P1 only without destabilizing the frozen P0.

## Advancement Rules

### Time Cadence

- Monday to Friday: three hours after work per day.
- Saturday and Sunday: eight hours per day.

Use the cadence to keep small stages finishable and reviewable; do not trade away acceptance and understanding just to increase feature count.

### Role Split

- Codex advances implementation, documentation, tests, commands, reproducible evidence, and debugging records.
- The user owns understanding of the architecture, chain, modules, data flow, inputs/outputs, tests, failure diagnosis, design choices, and core logic.
- Do not slow project advancement so the user can hand-write every line. Identify the small set of interview-relevant algorithms or logic for a separate code-practice module.
- This is teaching-grade advancement: complete one step, explain the important parts, wait for understanding and acceptance, then continue.

### Large and Small Stages

- Divide the project into large stages with explicit exit criteria.
- On entering a large stage, read the repository and current interview schedule, then create that stage's detailed small-stage plan. Do not statically decompose every future large stage at project start.
- Execute exactly one small stage at a time. Every small stage has explicit scope, inputs/outputs, commands, evidence, and acceptance criteria.
- After each small stage, summarize the actual state, pause for the user's L1 acceptance, and revalidate the remaining plan against what was learned. Adjust future small stages when evidence requires it.
- After each large stage, stop for L2 interview-facing understanding before entering the next large stage.

### README and Documentation

README must tell the complete main story and include:

```text
1. Project positioning, top-level design, and problem solved
2. Architecture/data-flow diagram and module responsibilities
3. Quick Start, demo input/output, and exact test commands
4. Stage records, version changes, task queue, and current status
5. Key metrics, artifacts, results, and evidence paths
6. Core design decisions, details, trade-offs, limits, and failure lessons
7. A teaching-oriented history: what was done, why, how, what failed, and how it was debugged
```

Documentation principles:

```text
README tells the complete main story.
docs stores only details that are too long or task-specific artifacts/results.
README must not become a directory index.
docs must not become an unmaintained fragment warehouse.
```

Every implementation small stage must update both README languages before it is complete. At minimum, align current status, task status, commands, tests, acceptance evidence, limitations, and the V2 entry log.

### Required Small-Stage Closure — Latest Nine-Part Format

After every implemented small stage, the final response must contain these nine parts unless the user explicitly requests a shorter status-only answer:

```text
1. What was done this time at a high level?
2. Which modules were added or changed, why, and why were they designed this way?
3. Which files changed, how did the file tree change, why, and what role does each file play in the chain?
4. What would the exact manual implementation workflow be without Codex?
5. What are the entry functions and core classes/functions, their inputs/outputs, and the macro pseudocode for the core logic?
6. How do you run, test, debug, tune, modify, and customize it?
7. What acceptance questions and follow-up questions prove interview-level understanding?
8. Which exact code sections are most likely to be asked about and should enter the code-practice module? Include file and current line references.
9. Was README/log/progress updated with the important data, paths, files, functions, and classes?
```

This nine-part format replaces the previous thirteen-part closure list.

### Understanding Depth

The only reason to advance the project is interview readiness: the user should be able to explain the chain, architecture, modules, decisions, implementation highlights, and debugging paths.

- **L1, after every small stage:** answer the macro questions; answer stage acceptance questions; explain the current chain/data flow in about 30 seconds; and, with AI guidance, run commands, tests, small changes, and debugging exercises.
- **L2, after every large stage:** answer macro questions plus roughly ten follow-ups; explain several failure cases; give a two-minute project explanation; make larger changes with AI guidance; write one or two resume bullets; keep stage notes; and identify the stage's code-practice candidates.
- **L3, after the first complete version during the August recruiting sprint:** turn README and notes into a teaching-level walkthrough and interview question set; re-read or hand-write the most important logic; explain the complete chain and key files/functions; answer likely follow-ups; describe concrete highlights and failure diagnosis; modify/test/debug across the project; and produce five selectable resume bullets.

Do not over-fragment L1 understanding and block progress, but do not accept explanations so broad that one follow-up exposes a gap.

## Result and Evidence Standard

The final result table must record at least:

```text
machine, OS, compiler, and build type
model, input size, dataset/fixed sample count
correctness standard and numeric tolerance
preprocess/infer/postprocess/end-to-end P50/P95
throughput and memory/RSS
single-thread or extension comparison when implemented
failure cases and conclusions
raw evidence paths and exact reproduction commands
```

Treat generated JSON, visualization, consistency, benchmark, and test results as evidence only after the documented command has run successfully.

## Allowed Scope by Default

- `cpp_infer/`
- `README.md`
- `README_zh.md`
- `AGENTS.md`
- A specific `docs/` artifact only when the user explicitly asks for it or the current large-stage plan names it

Read `docs/PLAN.md` for current planning context. Preserve it as the authored planning source unless the user asks for a change.

## Protected Scope

Do not rewrite or refactor the legacy Python training, evaluation, FastAPI, or Docker assets unless the user explicitly asks for that task:

- `scripts/`
- `src/`
- `api/`
- `Dockerfile`
- `requirements*.txt`
- `environment.yml`
- `configs/`
- `data/`
- `models/`
- `results/`
- `runs/`

Do not modify files outside this repository. In particular:

- Do not edit anything under `D:\01_Base\Obsidian`.
- Do not edit sibling projects under `D:\01_Base\CodingSpace`.

## Implementation Rules

- Do one small Project 1 task at a time, then stop for acceptance.
- Prefer runnable, testable, explainable modules over broad framework work.
- Keep C++ work simple: C++17, CMake, OpenCV, ONNX Runtime C++, GTest, and structured benchmark output.
- Do not introduce unrelated dependencies or replace the technology stack.
- Keep model-family-specific decode behind a clear contract so the YOLO P0 path does not falsely imply D-FINE compatibility.
- Make schema and runtime errors actionable: include the failing field/path, expected contract, actual value, and likely corrective action where practical.
- Never promote a placeholder, planned path, historical Python metric, research-side metric, or failed attempt into a delivered C++ result.
- If a step truly needs no README change, state why and verify that both READMEs already describe the behavior exactly; this should be rare.

## Verification Rules

For documentation-only changes, verify Markdown structure, internal links/paths, bilingual alignment, stale-route references, and `git diff`.

For C++ work, include the exact configure, build, run, and test commands used. Do not claim completion unless the relevant commands have run successfully, or clearly state the limitation and keep the task incomplete.
