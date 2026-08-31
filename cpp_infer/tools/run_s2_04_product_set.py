#!/usr/bin/env python3
"""Run the frozen 30-image S2-04 set through the single-image product CLI.

This deliberately uses the existing product path once per sample instead of
introducing a second inference implementation.  Every input is resolved and
SHA-verified before the fresh output directory is created.  Each emitted
``<sample_id>.detections.json`` is then bound back to the expected model,
source image, and actual execution provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, NoReturn, Sequence, Tuple


FROZEN_MANIFEST_ID = "neu_det_val_6x5_v1"
FROZEN_MANIFEST_IDS = {
    FROZEN_MANIFEST_ID,
    "neu_det_val_s2_04_holdout_6x5_v2",
    "neu_det_val_s2_04_native_holdout_6x5_v3",
    "neu_det_val_s2_04_native_holdout_6x5_v4",
}
FROZEN_SAMPLE_COUNT = 30
SHA256_PATTERN = re.compile(r"[0-9A-Fa-f]{64}")
SAMPLE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
CONFIG_PROVIDER_BY_ACTUAL_PROVIDER = {
    "CPUExecutionProvider": "cpu",
    "TensorrtExecutionProvider": "tensorrt",
    "TensorRTNative": "tensorrt_native",
}


class ProductSetError(RuntimeError):
    """An actionable product-set preflight, execution, or evidence failure."""


def fail(object_name: str, expected: str, actual: Any, action: str) -> NoReturn:
    raise ProductSetError(
        "S2-04 product-set run failed: "
        f"object={object_name}; expected={expected}; actual={actual!r}; "
        f"action={action}"
    )


def _reject_duplicate_keys(
    pairs: Iterable[Tuple[str, Any]],
) -> MutableMapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("json", "unique object keys", key, "remove the duplicate key")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> NoReturn:
    fail("json.number", "a finite JSON number", value, "replace NaN or Infinity")


def load_json_object(path: Path, object_name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except ProductSetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(
            object_name,
            "one readable UTF-8 JSON document",
            str(error),
            "fix the path, encoding, or JSON syntax",
        )
    if not isinstance(value, dict):
        fail(
            object_name,
            "a JSON object root",
            type(value).__name__,
            "replace the root value",
        )
    return value


def sha256_file(path: Path, object_name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        fail(
            object_name,
            "a readable regular file",
            str(error),
            "restore the frozen file and its permissions",
        )
    return digest.hexdigest().upper()


def load_key_value_file(path: Path, object_name: str) -> Mapping[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail(
            object_name,
            "one readable UTF-8 key/value declaration",
            str(error),
            "fix the path, encoding, or permissions",
        )
    result: MutableMapping[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(
                f"{object_name}.line[{line_number}]",
                "key = value",
                raw_line,
                "restore the tracked declaration syntax",
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key or not value:
            fail(
                f"{object_name}.line[{line_number}]",
                "a non-empty key and value",
                raw_line,
                "restore the tracked declaration field",
            )
        if key in result:
            fail(
                f"{object_name}.{key}",
                "one declaration",
                "duplicate field",
                "remove the duplicate field",
            )
        result[key] = value
    return result


def require_declaration_field(
    declaration: Mapping[str, str], field: str, object_name: str
) -> str:
    value = declaration.get(field)
    if value is None:
        fail(
            f"{object_name}.{field}",
            "one non-empty declaration",
            None,
            f"restore {field} in the tracked declaration",
        )
    return value


@dataclass(frozen=True)
class ValidatedRuntimeArtifacts:
    model_path: Path
    native_engine_path: Path | None = None
    native_engine_sha256: str | None = None


def validate_loaded_artifact(
    config_path: Path,
    expected_model_sha256: str,
    expected_actual_provider: str,
    expected_engine_sha256: str | None,
) -> ValidatedRuntimeArtifacts:
    expected_config_provider = CONFIG_PROVIDER_BY_ACTUAL_PROVIDER.get(
        expected_actual_provider
    )
    if expected_config_provider is None:
        fail(
            "expected_actual_provider",
            "CPUExecutionProvider, TensorrtExecutionProvider, or TensorRTNative",
            expected_actual_provider,
            "use one frozen product provider name",
        )

    config = load_key_value_file(config_path, "runtime_config")
    actual_config_provider = require_declaration_field(
        config, "provider", "runtime_config"
    )
    if actual_config_provider != expected_config_provider:
        fail(
            "runtime_config.provider",
            expected_config_provider,
            actual_config_provider,
            "pass the RuntimeConfig that selects the expected product provider",
        )
    actual_schema_version = require_declaration_field(
        config, "schema_version", "runtime_config"
    )
    expected_schema_version = (
        "1" if expected_config_provider == "cpu" else "2"
    )
    if actual_schema_version != expected_schema_version:
        fail(
            "runtime_config.schema_version",
            expected_schema_version,
            actual_schema_version,
            "use the frozen provider-specific RuntimeConfig schema",
        )
    if expected_config_provider in {"tensorrt", "tensorrt_native"}:
        actual_device_id = require_declaration_field(
            config, "device_id", "runtime_config"
        )
        if actual_device_id != "0":
            fail(
                "runtime_config.device_id",
                "0",
                actual_device_id,
                "use the frozen RTX 4060 device selection",
            )
        actual_precision = require_declaration_field(
            config, "precision", "runtime_config"
        )
        if actual_precision != "fp16":
            fail(
                "runtime_config.precision",
                "fp16",
                actual_precision,
                "use the frozen TensorRT FP16 acceptance config",
            )
    elif "precision" in config:
        fail(
            "runtime_config.precision",
            "absent for the schema-v1 CPU reference",
            config["precision"],
            "use the frozen CPU FP32 RuntimeConfig",
        )

    raw_artifact_path = require_declaration_field(
        config, "artifact_spec_path", "runtime_config"
    )
    artifact_path = (config_path.parent / raw_artifact_path).resolve()
    if not artifact_path.is_file():
        fail(
            "runtime_config.artifact_spec_path",
            "an existing regular artifact declaration",
            str(artifact_path),
            "restore the config-relative ModelArtifactSpec",
        )
    artifact = load_key_value_file(artifact_path, "artifact_spec")
    declared_sha = require_declaration_field(
        artifact, "model_sha256", "artifact_spec"
    )
    if not SHA256_PATTERN.fullmatch(declared_sha):
        fail(
            "artifact_spec.model_sha256",
            "64 hexadecimal characters",
            declared_sha,
            "restore the frozen model declaration",
        )
    if declared_sha.upper() != expected_model_sha256:
        fail(
            "artifact_spec.model_sha256",
            expected_model_sha256,
            declared_sha.upper(),
            "use the exact frozen artifact declaration",
        )

    raw_model_path = require_declaration_field(
        artifact, "model_path", "artifact_spec"
    )
    model_path = (artifact_path.parent / raw_model_path).resolve()
    if not model_path.is_file():
        fail(
            "artifact_spec.model_path",
            "an existing regular ONNX file",
            str(model_path),
            "restore the artifact-relative frozen model",
        )
    actual_sha = sha256_file(model_path, "artifact_spec.model_path")
    if actual_sha != expected_model_sha256:
        fail(
            "artifact_spec.model_path SHA-256",
            expected_model_sha256,
            actual_sha,
            "restore the exact frozen ONNX bytes before running inference",
        )
    if expected_config_provider != "tensorrt_native":
        if expected_engine_sha256 is not None:
            fail(
                "expected_engine_sha256",
                "absent for CPU and ORT TensorRT EP runs",
                expected_engine_sha256,
                "pass an engine SHA only for TensorRTNative",
            )
        return ValidatedRuntimeArtifacts(model_path=model_path)

    if expected_engine_sha256 is None or not SHA256_PATTERN.fullmatch(
        expected_engine_sha256
    ):
        fail(
            "expected_engine_sha256",
            "64 hexadecimal characters for TensorRTNative",
            expected_engine_sha256,
            "pass the frozen constrained-engine SHA-256",
        )
    expected_engine_sha256 = expected_engine_sha256.upper()
    declared_engine_sha = require_declaration_field(
        config, "tensorrt_engine_sha256", "runtime_config"
    )
    if (
        not SHA256_PATTERN.fullmatch(declared_engine_sha)
        or declared_engine_sha.upper() != expected_engine_sha256
    ):
        fail(
            "runtime_config.tensorrt_engine_sha256",
            expected_engine_sha256,
            declared_engine_sha,
            "use the frozen native config and constrained engine",
        )
    raw_engine_path = require_declaration_field(
        config, "tensorrt_engine_path", "runtime_config"
    )
    engine_path = (config_path.parent / raw_engine_path).resolve()
    if not engine_path.is_file():
        fail(
            "runtime_config.tensorrt_engine_path",
            "an existing frozen TensorRT engine",
            str(engine_path),
            "build and place the constrained engine before inference",
        )
    raw_cache_path = require_declaration_field(
        config, "tensorrt_engine_cache_path", "runtime_config"
    )
    cache_path = (config_path.parent / raw_cache_path).resolve()
    if engine_path.parent != cache_path:
        fail(
            "runtime_config.native_engine_cache_identity",
            str(cache_path),
            str(engine_path.parent),
            "keep the engine directly inside its frozen cache namespace",
        )
    actual_engine_sha = sha256_file(
        engine_path, "runtime_config.tensorrt_engine_path"
    )
    if actual_engine_sha != expected_engine_sha256:
        fail(
            "runtime_config.tensorrt_engine_path SHA-256",
            expected_engine_sha256,
            actual_engine_sha,
            "restore the exact constrained engine bytes",
        )
    return ValidatedRuntimeArtifacts(
        model_path=model_path,
        native_engine_path=engine_path,
        native_engine_sha256=actual_engine_sha,
    )


@dataclass(frozen=True)
class ResolvedSample:
    sample_id: str
    image_path: Path
    image_sha256: str


def resolve_samples(manifest_path: Path) -> Sequence[ResolvedSample]:
    manifest = load_json_object(manifest_path, "manifest")
    if manifest.get("schema_version") != 1:
        fail(
            "manifest.schema_version",
            "1",
            manifest.get("schema_version"),
            "use the tracked frozen consistency manifest",
        )
    if manifest.get("manifest_id") not in FROZEN_MANIFEST_IDS:
        fail(
            "manifest.manifest_id",
            f"one of {sorted(FROZEN_MANIFEST_IDS)}",
            manifest.get("manifest_id"),
            "use one tracked S2-04 frozen manifest",
        )

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != FROZEN_SAMPLE_COUNT:
        fail(
            "manifest.samples",
            f"an array containing exactly {FROZEN_SAMPLE_COUNT} samples",
            type(raw_samples).__name__
            if not isinstance(raw_samples, list)
            else len(raw_samples),
            "restore the frozen six-class, five-image-per-class selection",
        )

    manifest_directory = manifest_path.resolve().parent
    resolved = []
    seen_ids = set()
    seen_paths = set()
    for index, raw_sample in enumerate(raw_samples):
        object_name = f"manifest.samples[{index}]"
        if not isinstance(raw_sample, dict):
            fail(
                object_name,
                "an object",
                type(raw_sample).__name__,
                "restore the sample declaration",
            )
        sample_id = raw_sample.get("sample_id")
        if not isinstance(sample_id, str) or not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            fail(
                f"{object_name}.sample_id",
                "a non-empty filesystem-safe identifier",
                sample_id,
                "restore the frozen sample id",
            )
        if sample_id in seen_ids:
            fail(
                f"{object_name}.sample_id",
                "a unique identifier",
                sample_id,
                "remove the duplicate sample",
            )
        seen_ids.add(sample_id)

        raw_image_path = raw_sample.get("image_path")
        if not isinstance(raw_image_path, str) or not raw_image_path:
            fail(
                f"{object_name}.image_path",
                "a non-empty manifest-relative path",
                raw_image_path,
                "restore the frozen image path",
            )
        image_path = (manifest_directory / raw_image_path).resolve()
        if not image_path.is_file():
            fail(
                f"{object_name}.image_path",
                "an existing regular file",
                str(image_path),
                "restore the NEU-DET validation image at the declared path",
            )
        if image_path in seen_paths:
            fail(
                f"{object_name}.image_path",
                "a unique resolved image path",
                str(image_path),
                "restore the frozen one-row-per-image selection",
            )
        seen_paths.add(image_path)

        declared_sha = raw_sample.get("image_sha256")
        if not isinstance(declared_sha, str) or not SHA256_PATTERN.fullmatch(declared_sha):
            fail(
                f"{object_name}.image_sha256",
                "64 hexadecimal characters",
                declared_sha,
                "restore the frozen image identity",
            )
        actual_sha = sha256_file(image_path, f"{object_name}.image_path")
        if actual_sha != declared_sha.upper():
            fail(
                f"{object_name}.image_sha256",
                declared_sha.upper(),
                actual_sha,
                "use the exact frozen image bytes or version a new manifest",
            )
        resolved.append(ResolvedSample(sample_id, image_path, actual_sha))
    return resolved


def _portable_basename(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rsplit("/", 1)[-1]


def validate_detection_result(
    result_path: Path,
    sample: ResolvedSample,
    expected_model_sha256: str,
    expected_actual_provider: str,
    invocation_directory: Path,
    expected_engine_sha256: str | None,
) -> None:
    result = load_json_object(result_path, f"result[{sample.sample_id}]")

    model = result.get("model")
    if not isinstance(model, dict):
        fail(
            f"result[{sample.sample_id}].model",
            "an object",
            type(model).__name__,
            "fix the product result schema",
        )
    actual_model_sha = model.get("declared_sha256")
    if (
        not isinstance(actual_model_sha, str)
        or actual_model_sha.upper() != expected_model_sha256
    ):
        fail(
            f"result[{sample.sample_id}].model.declared_sha256",
            expected_model_sha256,
            actual_model_sha,
            "run the exact frozen model/config pair",
        )

    image = result.get("image")
    if not isinstance(image, dict):
        fail(
            f"result[{sample.sample_id}].image",
            "an object",
            type(image).__name__,
            "fix the product result schema",
        )
    raw_result_image = image.get("path")
    if not isinstance(raw_result_image, str) or not raw_result_image:
        fail(
            f"result[{sample.sample_id}].image.path",
            "a non-empty path",
            raw_result_image,
            "emit the source image path in the product result",
        )
    actual_basename = _portable_basename(raw_result_image)
    if actual_basename != sample.image_path.name:
        fail(
            f"result[{sample.sample_id}].image.path basename",
            sample.image_path.name,
            actual_basename,
            "run the CLI with the manifest-resolved image",
        )
    result_image_path = Path(raw_result_image).expanduser()
    if not result_image_path.is_absolute():
        result_image_path = invocation_directory / result_image_path
    result_image_path = result_image_path.resolve()
    actual_image_sha = sha256_file(
        result_image_path, f"result[{sample.sample_id}].image.path"
    )
    if actual_image_sha != sample.image_sha256:
        fail(
            f"result[{sample.sample_id}].image SHA-256",
            sample.image_sha256,
            actual_image_sha,
            "run inference on the exact frozen manifest image bytes",
        )

    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        fail(
            f"result[{sample.sample_id}].runtime",
            "an object",
            type(runtime).__name__,
            "fix the product result schema",
        )
    actual_provider = runtime.get("actual_provider")
    if actual_provider != expected_actual_provider:
        fail(
            f"result[{sample.sample_id}].runtime.actual_provider",
            expected_actual_provider,
            actual_provider,
            "fix provider registration/fallback before accepting this run",
        )
    if expected_actual_provider == "TensorRTNative":
        evidence = runtime.get("provider_evidence")
        expected_evidence = (
            "native_tensorrt_enqueue_v3;"
            "precision_policy=fp16_dfl_softmax_fp32_else_no_tf32;"
            f"declared_engine_sha256={expected_engine_sha256};"
            f"actual_engine_sha256={expected_engine_sha256};"
            "tensorrt_runtime=10.4.0;compiled_headers=10.4.0.26;"
            "cuda_runtime=12.6;"
            "compute_capability=8.9;fallback=none"
        )
        if evidence != expected_evidence:
            fail(
                f"result[{sample.sample_id}].runtime.provider_evidence",
                expected_evidence,
                evidence,
                "execute the SHA-verified constrained engine directly",
            )


def _bounded_process_text(value: str, limit: int = 4000) -> str:
    value = value.strip()
    return value if len(value) <= limit else value[-limit:]


def run_product_set(
    cli_path: Path,
    config_path: Path,
    manifest_path: Path,
    output_directory: Path,
    expected_model_sha256: str,
    expected_actual_provider: str,
    expected_engine_sha256: str | None = None,
) -> Sequence[Path]:
    cli_path = cli_path.resolve()
    config_path = config_path.resolve()
    manifest_path = manifest_path.resolve()
    output_directory = output_directory.resolve()

    if not cli_path.is_file():
        fail(
            "cli",
            "an existing product CLI file",
            str(cli_path),
            "build yolo_defect_cpp and pass its explicit path",
        )
    if not config_path.is_file():
        fail(
            "config",
            "an existing RuntimeConfig file",
            str(config_path),
            "pass the config used for this provider run",
        )
    if not manifest_path.is_file():
        fail(
            "manifest",
            "an existing frozen consistency manifest",
            str(manifest_path),
            "pass cpp_infer/tests/fixtures/consistency_manifest.json",
        )
    if not SHA256_PATTERN.fullmatch(expected_model_sha256):
        fail(
            "expected_model_sha256",
            "64 hexadecimal characters",
            expected_model_sha256,
            "pass the frozen current ONNX SHA-256",
        )
    expected_model_sha256 = expected_model_sha256.upper()
    if not expected_actual_provider:
        fail(
            "expected_actual_provider",
            "a non-empty exact product provider name",
            expected_actual_provider,
            "pass CPUExecutionProvider, TensorrtExecutionProvider, or TensorRTNative explicitly",
        )

    # Bind the run to the exact model bytes and provider selected by the
    # supplied RuntimeConfig before any output directory is created.
    validated_artifacts = validate_loaded_artifact(
        config_path,
        expected_model_sha256,
        expected_actual_provider,
        expected_engine_sha256,
    )

    # Resolve and hash every input before creating any run output.
    samples = resolve_samples(manifest_path)
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        fail(
            "output_directory",
            "a path that does not exist",
            str(output_directory),
            "choose a fresh run directory; do not mix or overwrite evidence",
        )
    except OSError as error:
        fail(
            "output_directory",
            "a creatable fresh directory",
            str(error),
            "fix the parent path or permissions",
        )

    invocation_directory = Path.cwd().resolve()
    written = []
    for sample in samples:
        output_path = output_directory / f"{sample.sample_id}.detections.json"
        command = [
            str(cli_path),
            "--config",
            str(config_path),
            "--image",
            str(sample.image_path),
            "--output-json",
            str(output_path),
        ]
        try:
            # Deliberately omit ``env`` so CUDA/TensorRT loader variables and
            # every other caller-supplied environment value are inherited.
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as error:
            fail(
                f"cli[{sample.sample_id}]",
                "a launchable product CLI",
                str(error),
                "check the executable bit, loader dependencies, and CLI path",
            )
        if completed.returncode != 0:
            fail(
                f"cli[{sample.sample_id}].returncode",
                "0",
                {
                    "returncode": completed.returncode,
                    "stdout_tail": _bounded_process_text(completed.stdout),
                    "stderr_tail": _bounded_process_text(completed.stderr),
                },
                "inspect the product diagnostic; partial run outputs were preserved",
            )
        if not output_path.is_file():
            fail(
                f"result[{sample.sample_id}]",
                "the requested detection JSON file",
                str(output_path),
                "inspect CLI stdout/stderr and output-path handling",
            )
        validate_detection_result(
            output_path,
            sample,
            expected_model_sha256,
            expected_actual_provider,
            invocation_directory,
            expected_engine_sha256.upper()
            if expected_engine_sha256 is not None
            else None,
        )
        written.append(output_path)
    if validated_artifacts.native_engine_path is not None:
        final_engine_sha = sha256_file(
            validated_artifacts.native_engine_path,
            "runtime_config.tensorrt_engine_path.after_run",
        )
        if final_engine_sha != validated_artifacts.native_engine_sha256:
            fail(
                "runtime_config.tensorrt_engine_path SHA-256 after run",
                validated_artifacts.native_engine_sha256,
                final_engine_sha,
                "discard the run because engine bytes changed during evidence capture",
            )
    return written


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run and validate the frozen 30-image S2-04 set through an explicit "
            "single-image yolo_defect_cpp CLI/config pair."
        )
    )
    parser.add_argument("--cli", required=True, type=Path, help="Path to yolo_defect_cpp")
    parser.add_argument(
        "--config", required=True, type=Path, help="RuntimeConfig for this run"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Frozen consistency_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh directory for <sample_id>.detections.json files",
    )
    parser.add_argument(
        "--expected-model-sha256",
        required=True,
        help="Exact SHA-256 declared by every accepted product result",
    )
    parser.add_argument(
        "--expected-actual-provider",
        required=True,
        help="Exact product provider name required in every result",
    )
    parser.add_argument(
        "--expected-engine-sha256",
        help="Frozen native engine SHA-256; required only for TensorRTNative",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        written = run_product_set(
            cli_path=arguments.cli,
            config_path=arguments.config,
            manifest_path=arguments.manifest,
            output_directory=arguments.output_dir,
            expected_model_sha256=arguments.expected_model_sha256,
            expected_actual_provider=arguments.expected_actual_provider,
            expected_engine_sha256=arguments.expected_engine_sha256,
        )
    except ProductSetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        "S2-04 product-set run passed: "
        f"samples={len(written)}; provider={arguments.expected_actual_provider}; "
        f"output_directory={arguments.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
