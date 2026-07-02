# Codex Collaboration Rules

This repository uses `README.md` and `README_zh.md` as the main project entry points. Keep extra documents rare; add a standalone document only when README would become harder to use. By default, task queue and change-log content belongs in the README files, not in separate `TASKS.md` or `CHANGELOG.md` files.

## Current Goal

Project V2 is an industrial vision edge AI Runtime and C++ engineering project. YOLOv8 and NEU-DET are the P0 runtime baseline carriers. The later `paper_detect` D010 artifact uses an improved D-FINE-S architecture on DeepPCB and should be treated as a research-side artifact source, not as training logic owned by this repository.

The autumn-recruiting value is the deployment chain:

```text
model artifact
-> artifact contract / model card
-> C++ preprocess
-> ONNX Runtime C++ inference
-> postprocess / NMS
-> detection JSON / visualization
-> benchmark
-> INT8 PTQ
-> TensorRT attempt
-> sample inference_event
-> tests and README evidence
```

## Allowed Scope By Default

- `cpp_infer/`
- `README.md`, including task queue and change-log sections
- `README_zh.md`, including task queue and change-log sections
- `AGENTS.md`

Read `docs/路线0628.md` for the latest planning context, especially the Project 1 sections, but do not edit it unless the user explicitly asks.

## Time Cadence

- Monday to Friday: spend 3 hours after work advancing the projects.
- Saturday and Sunday: spend 8 hours per day advancing the projects.

## Role Split

Project advancement is optimized for autumn-recruiting interview readiness, not for the user hand-writing every line of project code.

- Codex advances implementation, documentation, tests, commands, and evidence.
- The user focuses on understanding the chain, architecture, modules, inputs/outputs, tests, failure debugging, and core logic.
- If a project contains core algorithms or logic worth hand-writing or reading deeply, do not slow the project stage for that. Record it as a candidate for the separate "code practice" module.

## Stage Process

- Project advancement is divided into large stages.
- Each large stage is split into small stages dynamically during execution.
- Do not statically write every small stage at the beginning. After each small stage is complete, summarize the current state, then define the next small-stage plan.
- Before starting a large stage, the user may ask a web GPT session to read the GitHub repository and write a detailed small-stage plan for that large stage.
- After each large stage, stop for interview-facing understanding before entering the next large stage.
- Every small stage and large stage needs explicit acceptance criteria.

## Documentation Standard

`README.md` and `README_zh.md` are the unique project entry points. Do not create many standalone docs.

README must include:

```text
1. Project positioning and top-level design
2. Problem solved
3. Overall architecture diagram or text chain
4. Core module responsibilities
5. Quick start
6. Demo input/output
7. Test commands
8. Key data and artifact results
9. Key design trade-offs
10. Task queue
11. Version changes and progress records
12. Teaching-style record from project start to now: what was done, why, how, what failed, and how it was debugged
```

Documentation principle:

```text
README tells the complete main story.
docs only stores overly long details, specific task artifacts, benchmark details, API examples, and interview question sets.
Do not let README become a directory index.
Do not let docs become an unmaintained fragment warehouse.
```

## Understanding Depth

The only purpose of project advancement is interview readiness: the user should be able to explain the chain, architecture, modules, decisions, and follow-up questions.

Use three levels:

- L1 during small-stage advancement.
- L2 when closing a large stage.
- L3 during the August autumn-recruiting sprint after the first full implementation is complete.

At L1, after each small stage/module, the user should be able to:

```text
1. Answer macro questions: what problem the module solves, where it sits in the chain, upstream input, downstream output, core classes/functions/files, why designed this way, how to run/test it, and where to debug failures.
2. Answer Codex-provided acceptance questions in an interview-like style.
3. Explain the current chain, architecture, data flow, or business flow in about 30 seconds.
4. With AI guidance, run commands, tests, small customizations, modifications, debugging, and hands-on experiments.
```

At L2, after each large stage, the user should be able to:

```text
1. Answer macro questions, acceptance questions, and about 10 follow-up questions.
2. Explain several error cases.
3. Explain the current chain, architecture, data flow, or business flow in about 2 minutes.
4. With AI guidance, perform larger customizations, tests, runs, and debugging.
5. Write 1-2 resume bullets.
6. Keep stage notes.
7. Ask Codex to identify core algorithms and logic to move into the code-practice module for reading or minimal hand-writing.
```

At L3, during the autumn-recruiting sprint, the user should be able to:

```text
1. Turn README and stage notes into two review docs: a teaching-level full project walkthrough and an interview question set.
2. Re-read or hand-write the most important core logic.
3. Explain the complete project chain, architecture, data flow, and key files/functions.
4. Answer likely interview questions and follow-ups.
5. Explain concrete implementation highlights.
6. Describe likely error scenarios and debugging paths.
7. Modify, test, run, and debug across the whole project.
8. Produce 5 resume bullets and choose among them for different job descriptions.
```

## Required Small-Stage Closure

After each Codex small-stage implementation, the response must include these 13 parts unless the user explicitly asks for a shorter status-only answer:

```text
1. What was done this time?
2. Why is this module designed this way?
3. Which files changed, and why?
4. Role of each file in the chain.
5. Key entry functions.
6. Core classes/functions.
7. Inputs and outputs.
8. How to run.
9. How to test.
10. How to debug failures.
11. Acceptance questions and follow-up questions.
12. Which core logic should be hand-written or understood deeply.
13. README/log/progress update status, including key data, paths, files, functions, and classes.
```

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

## Task Rules

- Do one small P1 task at a time.
- Update README/README_zh when behavior, commands, scope, or acceptance criteria change.
- Every implementation step must update the README entry point before it is considered complete. At minimum, keep README/README_zh task status, run commands, test commands, acceptance criteria, and V2 entry log aligned with the code change.
- If a step truly does not require a README/README_zh change, explicitly state why in the final response and verify that the existing README still describes the current behavior correctly.
- Prefer runnable, testable, explainable modules over broad framework work.
- Do not introduce unrelated dependencies or replace the technology stack.
- Keep C++ work simple: C++17, CMake, OpenCV, ONNX Runtime C++, GTest, benchmark output.

## Verification Rules

For documentation-only changes, verify by checking the rendered Markdown structure and `git diff`.

For future C++ work, include the exact commands used to build and test `cpp_infer/`. Do not claim a module is complete unless the relevant command has run successfully or the limitation is clearly stated.
