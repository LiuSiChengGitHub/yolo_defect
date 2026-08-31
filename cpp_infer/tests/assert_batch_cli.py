#!/usr/bin/env python3
"""End-to-end assertions for the S2-03 bounded batch CLI."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence


class BatchCliAssertionError(RuntimeError):
    """Raised when the real CLI violates the frozen S2-03 contract."""


@contextlib.contextmanager
def test_workspace(work_root: Path):
    """Use ordinary mkdir so Windows ACLs remain inherited and traversable."""
    path = work_root / f"s2_03_batch_cli_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--expected-model-id")
    parser.add_argument("--expected-model-sha256")
    return parser.parse_args()


def run_cli(arguments: Sequence[str], expected_exit: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(arguments),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != expected_exit:
        raise BatchCliAssertionError(
            "CLI exit code mismatch: "
            f"expected={expected_exit}, actual={completed.returncode}\n"
            f"command={list(arguments)!r}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise BatchCliAssertionError(f"Expected non-empty JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BatchCliAssertionError(f"Invalid UTF-8 JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise BatchCliAssertionError(f"Expected a JSON object at {path}")
    return value


def assert_detection_json(path: Path, expected_source: Path) -> dict[str, Any]:
    document = read_json(path)
    if document.get("schema_version") != 1:
        raise BatchCliAssertionError(f"Detection schema_version is not 1: {path}")
    image = document.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("path"), str):
        raise BatchCliAssertionError(f"Detection image.path is missing: {path}")
    if Path(image["path"]).resolve() != expected_source.resolve():
        raise BatchCliAssertionError(
            f"Detection source mismatch at {path}: "
            f"expected={expected_source.resolve()}, actual={image['path']}"
        )
    if not isinstance(document.get("detections"), list):
        raise BatchCliAssertionError(f"Detection array is missing: {path}")
    return document


def assert_summary(
    path: Path,
    *,
    expected_status: str,
    discovered: int,
    succeeded: int,
    failed: int,
    cancelled: int,
    expected_workers: int,
    expected_sources: Sequence[Path],
    expected_cooperative_stop_requested: bool = False,
    expected_model_id: Optional[str] = None,
    expected_model_sha256: Optional[str] = None,
) -> dict[str, Any]:
    document = read_json(path)
    if document.get("schema_version") != 1:
        raise BatchCliAssertionError(f"Batch schema_version is not 1: {path}")
    if document.get("status") != expected_status:
        raise BatchCliAssertionError(
            f"Batch status mismatch at {path}: "
            f"expected={expected_status!r}, actual={document.get('status')!r}"
        )
    if document.get("cooperative_stop_requested") is not expected_cooperative_stop_requested:
        raise BatchCliAssertionError(
            f"Batch cooperative_stop_requested mismatch at {path}: "
            f"expected={expected_cooperative_stop_requested!r}, "
            f"actual={document.get('cooperative_stop_requested')!r}"
        )

    counts = document.get("counts")
    expected_counts = {
        "discovered": discovered,
        "enqueued": discovered,
        "started": succeeded + failed,
        "succeeded": succeeded,
        "failed": failed,
        "cancelled": cancelled,
    }
    if not isinstance(counts, dict):
        raise BatchCliAssertionError(f"Batch counts object is missing: {path}")
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise BatchCliAssertionError(
                f"Batch counts.{name} mismatch at {path}: "
                f"expected={expected}, actual={counts.get(name)!r}"
            )
    if counts["discovered"] != (
        counts["succeeded"] + counts["failed"] + counts["cancelled"]
    ):
        raise BatchCliAssertionError(f"Batch terminal-count invariant failed: {path}")

    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        raise BatchCliAssertionError(f"Batch runtime object is missing: {path}")
    if runtime.get("requested_workers") != expected_workers:
        raise BatchCliAssertionError(
            f"requested_workers mismatch at {path}: {runtime!r}"
        )
    if runtime.get("effective_workers") != min(expected_workers, discovered):
        raise BatchCliAssertionError(
            f"effective_workers mismatch at {path}: {runtime!r}"
        )

    model = document.get("model")
    if not isinstance(model, dict):
        raise BatchCliAssertionError(f"Batch model object is missing: {path}")
    if expected_model_id is not None and model.get("model_id") != expected_model_id:
        raise BatchCliAssertionError(
            f"Batch model_id mismatch at {path}: "
            f"expected={expected_model_id!r}, actual={model.get('model_id')!r}"
        )
    if (
        expected_model_sha256 is not None
        and model.get("declared_sha256") != expected_model_sha256
    ):
        raise BatchCliAssertionError(
            f"Batch declared_sha256 mismatch at {path}: "
            f"expected={expected_model_sha256!r}, "
            f"actual={model.get('declared_sha256')!r}"
        )

    items = document.get("items")
    if not isinstance(items, list) or len(items) != discovered:
        raise BatchCliAssertionError(f"Batch items are incomplete at {path}")
    if [
        item.get("sequence_index") for item in items if isinstance(item, dict)
    ] != list(range(discovered)):
        raise BatchCliAssertionError(f"Batch items are not discovery-ordered: {path}")
    actual_sources = [
        Path(item["source_path"]).resolve()
        for item in items
        if isinstance(item, dict) and isinstance(item.get("source_path"), str)
    ]
    if actual_sources != [source.resolve() for source in expected_sources]:
        raise BatchCliAssertionError(
            f"Batch source order mismatch at {path}: "
            f"expected={list(expected_sources)!r}, actual={actual_sources!r}"
        )
    return document


def item_json(output_directory: Path, index: int) -> Path:
    return output_directory / "items" / f"{index:06d}.detections.json"


def item_image(output_directory: Path, index: int) -> Path:
    return output_directory / "items" / f"{index:06d}.visualized.png"


def batch_command(
    cli: Path,
    config: Path,
    *,
    input_option: str,
    input_path: Path,
    output_directory: Path,
    summary_path: Path,
    workers: int,
    queue_capacity: int,
    output_images: bool = False,
) -> list[str]:
    command = [
        str(cli),
        "--config",
        str(config),
        "--batch",
        input_option,
        str(input_path),
        "--output-dir",
        str(output_directory),
        "--batch-summary",
        str(summary_path),
        "--workers",
        str(workers),
        "--queue-capacity",
        str(queue_capacity),
    ]
    if output_images:
        command.append("--output-images")
    return command


def assert_same_bytes(first: Path, second: Path, description: str) -> None:
    if first.read_bytes() != second.read_bytes():
        raise BatchCliAssertionError(
            f"{description} differs: first={first}, second={second}"
        )


def run_shutdown_smoke(
    cli: Path, config: Path, source_image: Path, root: Path
) -> None:
    input_directory = root / "shutdown-inputs"
    input_directory.mkdir()
    sources = []
    for index in range(64):
        target = input_directory / f"{index:06d}.jpg"
        shutil.copy2(source_image, target)
        sources.append(target)

    output_directory = root / "shutdown-output"
    summary_path = root / "shutdown-summary.json"
    command = batch_command(
        cli,
        config,
        input_option="--input-dir",
        input_path=input_directory,
        output_directory=output_directory,
        summary_path=summary_path,
        workers=1,
        queue_capacity=1,
    )
    creation_flags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creation_flags,
    )
    first_output = item_json(output_directory, 0)
    deadline = time.monotonic() + 30.0
    while not first_output.is_file() and process.poll() is None:
        if time.monotonic() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            raise BatchCliAssertionError(
                "Shutdown smoke produced no first output within 30 seconds\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        time.sleep(0.01)
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        raise BatchCliAssertionError(
            "Shutdown smoke completed before interruption could be sent\n"
            f"exit={process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    process.send_signal(
        signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT
    )
    try:
        stdout, stderr = process.communicate(timeout=60.0)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise BatchCliAssertionError(
            "Interrupted batch did not cooperatively join within 60 seconds\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error
    if process.returncode != 130:
        raise BatchCliAssertionError(
            "Interrupted batch did not return exit 130: "
            f"actual={process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    summary = read_json(summary_path)
    if summary.get("status") != "cancelled":
        raise BatchCliAssertionError(
            f"Interrupted batch status is not cancelled: {summary.get('status')!r}"
        )
    if summary.get("cooperative_stop_requested") is not True:
        raise BatchCliAssertionError(
            "Interrupted batch did not record cooperative_stop_requested=true"
        )
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise BatchCliAssertionError("Interrupted batch counts are missing")
    required_counts = {
        name: counts.get(name)
        for name in (
            "discovered",
            "enqueued",
            "started",
            "succeeded",
            "failed",
            "cancelled",
        )
    }
    if any(not isinstance(value, int) for value in required_counts.values()):
        raise BatchCliAssertionError(
            f"Interrupted batch counts are not integers: {required_counts!r}"
        )
    if (
        counts["discovered"] != len(sources)
        or counts["succeeded"] <= 0
        or counts["started"] <= 0
        or counts["cancelled"] <= 0
    ):
        raise BatchCliAssertionError(
            "Interrupted batch lacks both completed and cancelled work: "
            f"{counts!r}"
        )
    if counts["started"] != counts["succeeded"] + counts["failed"]:
        raise BatchCliAssertionError(
            f"Interrupted batch started-count invariant failed: {counts!r}"
        )
    if counts["discovered"] != (
        counts["succeeded"] + counts["failed"] + counts["cancelled"]
    ):
        raise BatchCliAssertionError(
            f"Interrupted batch terminal-count invariant failed: {counts!r}"
        )
    if not (
        counts["started"] <= counts["enqueued"] <= counts["discovered"]
    ):
        raise BatchCliAssertionError(
            f"Interrupted batch enqueue-count invariant failed: {counts!r}"
        )

    items = summary.get("items")
    if not isinstance(items, list) or len(items) != len(sources):
        raise BatchCliAssertionError("Interrupted batch items are incomplete")
    status_counts = {"succeeded": 0, "failed": 0, "cancelled": 0}
    for index, (item, source) in enumerate(zip(items, sources)):
        if not isinstance(item, dict):
            raise BatchCliAssertionError(f"Interrupted item {index} is not an object")
        if item.get("sequence_index") != index:
            raise BatchCliAssertionError("Interrupted items lost discovery order")
        if Path(item.get("source_path", "")).resolve() != source.resolve():
            raise BatchCliAssertionError(
                f"Interrupted item {index} source path changed"
            )
        status = item.get("status")
        if status not in status_counts:
            raise BatchCliAssertionError(
                f"Interrupted item {index} has invalid status {status!r}"
            )
        status_counts[status] += 1
        expected_json = item_json(output_directory, index)
        declared_json = item.get("json_output_path")
        error = item.get("error")
        latency_ms = item.get("latency_ms")
        if status == "succeeded":
            if not expected_json.is_file():
                raise BatchCliAssertionError(
                    f"Succeeded interrupted item {index} lacks JSON output"
                )
            if not isinstance(declared_json, str) or (
                Path(declared_json).resolve() != expected_json.resolve()
            ):
                raise BatchCliAssertionError(
                    f"Succeeded interrupted item {index} declares wrong output"
                )
            if error not in (None, ""):
                raise BatchCliAssertionError(
                    f"Succeeded interrupted item {index} has an error"
                )
        else:
            if expected_json.exists() or declared_json is not None:
                raise BatchCliAssertionError(
                    f"{status} interrupted item {index} unexpectedly has output"
                )
            if not isinstance(error, str) or not error:
                raise BatchCliAssertionError(
                    f"{status} interrupted item {index} lacks an error reason"
                )
            if status == "cancelled" and latency_ms != 0:
                raise BatchCliAssertionError(
                    f"Cancelled interrupted item {index} has non-zero latency"
                )
    for status, actual in status_counts.items():
        if actual != counts[status]:
            raise BatchCliAssertionError(
                f"Interrupted item/count mismatch for {status}: "
                f"items={actual}, counts={counts[status]}"
            )
    written_count = len(list((output_directory / "items").glob("*.detections.json")))
    if written_count != counts["succeeded"]:
        raise BatchCliAssertionError(
            "Interrupted batch output count differs from succeeded count: "
            f"outputs={written_count}, counts={counts!r}"
        )


def main() -> int:
    arguments = parse_arguments()
    cli = arguments.cli.resolve(strict=True)
    config = arguments.config.resolve(strict=True)
    source_image = arguments.image.resolve(strict=True)
    work_root = arguments.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    with test_workspace(work_root) as root:
        input_directory = root / "inputs"
        nested_directory = input_directory / "nested"
        nested_directory.mkdir(parents=True)
        first_image = input_directory / "a.jpg"
        second_image = nested_directory / "z.jpg"
        shutil.copy2(source_image, first_image)
        shutil.copy2(source_image, second_image)
        (input_directory / "ignored.txt").write_text(
            "unsupported extension must be filtered\n", encoding="utf-8"
        )

        directory_output = root / "directory-output"
        directory_summary = root / "directory-summary.json"
        directory_command = batch_command(
            cli,
            config,
            input_option="--input-dir",
            input_path=input_directory,
            output_directory=directory_output,
            summary_path=directory_summary,
            workers=1,
            queue_capacity=1,
            output_images=True,
        )
        run_cli(directory_command, 0)
        assert_summary(
            directory_summary,
            expected_status="succeeded",
            discovered=2,
            succeeded=2,
            failed=0,
            cancelled=0,
            expected_workers=1,
            expected_sources=[first_image, second_image],
            expected_model_id=arguments.expected_model_id,
            expected_model_sha256=arguments.expected_model_sha256,
        )
        assert_detection_json(item_json(directory_output, 0), first_image)
        assert_detection_json(item_json(directory_output, 1), second_image)
        for index in range(2):
            visualization = item_image(directory_output, index)
            if not visualization.is_file() or visualization.stat().st_size == 0:
                raise BatchCliAssertionError(
                    f"Expected non-empty batch visualization: {visualization}"
                )

        single_json = root / "single-image.json"
        run_cli(
            [
                str(cli),
                "--config",
                str(config),
                "--image",
                str(first_image),
                "--output-json",
                str(single_json),
            ],
            0,
        )
        assert_same_bytes(
            single_json,
            item_json(directory_output, 0),
            "single-image and workers=1 batch detection JSON",
        )

        manifest = root / "ordered-manifest.txt"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            stream.write(
                "\ufeff# BOM, comments, CRLF, and declaration order are intentional\r\n"
                f"{second_image.relative_to(root).as_posix()}\r\n"
                "\r\n"
                f"{first_image.relative_to(root).as_posix()}\r\n"
            )
        manifest_sources = [second_image, first_image]
        manifest_outputs: list[Path] = []
        for workers in (1, 2):
            output_directory = root / f"manifest-workers-{workers}"
            summary_path = root / f"manifest-workers-{workers}.summary.json"
            run_cli(
                batch_command(
                    cli,
                    config,
                    input_option="--manifest",
                    input_path=manifest,
                    output_directory=output_directory,
                    summary_path=summary_path,
                    workers=workers,
                    queue_capacity=1,
                ),
                0,
            )
            assert_summary(
                summary_path,
                expected_status="succeeded",
                discovered=2,
                succeeded=2,
                failed=0,
                cancelled=0,
                expected_workers=workers,
                expected_sources=manifest_sources,
                expected_model_id=arguments.expected_model_id,
                expected_model_sha256=arguments.expected_model_sha256,
            )
            for index, expected_source in enumerate(manifest_sources):
                assert_detection_json(
                    item_json(output_directory, index), expected_source
                )
            manifest_outputs.append(output_directory)

        for index in range(2):
            assert_same_bytes(
                item_json(manifest_outputs[0], index),
                item_json(manifest_outputs[1], index),
                f"workers=1 and workers=2 item {index}",
            )

        unicode_image = input_directory / "中文表面.jpg"
        shutil.copy2(source_image, unicode_image)
        unicode_manifest = root / "unicode-manifest.txt"
        with unicode_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(unicode_image.relative_to(root).as_posix() + "\n")
        unicode_output = root / "unicode-output"
        unicode_summary = root / "unicode-summary.json"
        run_cli(
            batch_command(
                cli,
                config,
                input_option="--manifest",
                input_path=unicode_manifest,
                output_directory=unicode_output,
                summary_path=unicode_summary,
                workers=1,
                queue_capacity=1,
            ),
            0,
        )
        assert_summary(
            unicode_summary,
            expected_status="succeeded",
            discovered=1,
            succeeded=1,
            failed=0,
            cancelled=0,
            expected_workers=1,
            expected_sources=[unicode_image],
            expected_model_id=arguments.expected_model_id,
            expected_model_sha256=arguments.expected_model_sha256,
        )
        assert_detection_json(item_json(unicode_output, 0), unicode_image)

        corrupt_image = input_directory / "corrupt.jpg"
        corrupt_image.write_bytes(b"S2-03 deliberately invalid JPEG bytes\n")
        partial_manifest = root / "partial-manifest.txt"
        partial_sources = [first_image, corrupt_image, second_image]
        partial_manifest.write_text(
            "\n".join(
                source.relative_to(root).as_posix() for source in partial_sources
            )
            + "\n",
            encoding="utf-8",
        )
        partial_output = root / "partial-output"
        partial_summary = root / "partial-summary.json"
        partial_command = batch_command(
            cli,
            config,
            input_option="--manifest",
            input_path=partial_manifest,
            output_directory=partial_output,
            summary_path=partial_summary,
            workers=2,
            queue_capacity=1,
        )
        run_cli(partial_command, 2)
        partial = assert_summary(
            partial_summary,
            expected_status="partial_failure",
            discovered=3,
            succeeded=2,
            failed=1,
            cancelled=0,
            expected_workers=2,
            expected_sources=partial_sources,
            expected_model_id=arguments.expected_model_id,
            expected_model_sha256=arguments.expected_model_sha256,
        )
        statuses = [item.get("status") for item in partial["items"]]
        if statuses != ["succeeded", "failed", "succeeded"]:
            raise BatchCliAssertionError(
                f"Unexpected partial-failure item statuses: {statuses!r}"
            )
        failure = partial["items"][1]
        if not isinstance(failure.get("error"), str) or not failure["error"]:
            raise BatchCliAssertionError("Failed item lacks an actionable error")
        assert_detection_json(item_json(partial_output, 0), first_image)
        if item_json(partial_output, 1).exists():
            raise BatchCliAssertionError("Corrupt input unexpectedly wrote detection JSON")
        assert_detection_json(item_json(partial_output, 2), second_image)

        summary_bytes = partial_summary.read_bytes()
        refusal = run_cli(partial_command, 1)
        combined_refusal = refusal.stdout + refusal.stderr
        if "already exists" not in combined_refusal:
            raise BatchCliAssertionError(
                "Second batch run did not explain summary overwrite refusal:\n"
                + combined_refusal
            )
        if partial_summary.read_bytes() != summary_bytes:
            raise BatchCliAssertionError(
                "Refused batch run modified the existing BatchSummary"
            )

        run_shutdown_smoke(cli, config, source_image, root)

    print(
        "S2-03 real CLI passed: directory/manifest discovery, workers 1/2 "
        "detection equality, single-image equivalence, UTF-8 manifest image "
        "paths, visualization, "
        "partial failure exit 2, summary invariants, overwrite refusal, and "
        "cooperative interrupt exit 130."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BatchCliAssertionError as error:
        raise SystemExit(f"S2-03 batch CLI assertion failed: {error}") from error
