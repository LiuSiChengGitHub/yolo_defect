#!/usr/bin/env python3
"""Run the frozen 30-image product gate for an S2-01 PTQ candidate.

This is a cheap candidate screen, not formal S2-01 correctness evidence.  It
reuses the formal evaluator's contract loading, Python ORT product pipeline,
matching, gates, and per-image records while deliberately skipping the
361-image task metric and C++ consistency run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import evaluate_s2_01_correctness as evaluator  # noqa: E402
import s2_01_protocol as machine_protocol  # noqa: E402


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "s2_01_candidate_product_screen"


def screen(arguments: argparse.Namespace) -> Mapping[str, Any]:
    frozen = machine_protocol.load_s2_01_protocol(arguments.protocol)
    product_manifest = evaluator.load_product_manifest(
        frozen.consistency_manifest_path
    )
    quality_manifest = evaluator.load_frozen_manifest(
        frozen.quality_manifest_path, "quality"
    )
    if product_manifest["manifest_path"] != frozen.consistency_manifest_path:
        evaluator.fail(
            "protocol.product_manifest",
            str(frozen.consistency_manifest_path),
            str(product_manifest["manifest_path"]),
            "restore the product manifest frozen by this protocol",
        )
    if quality_manifest["manifest_path"] != frozen.quality_manifest_path:
        evaluator.fail(
            "protocol.quality_manifest",
            str(frozen.quality_manifest_path),
            str(quality_manifest["manifest_path"]),
            "restore the gate source frozen by this protocol",
        )

    consistency = evaluator.load_consistency_tool()
    consistency.require_dependencies()
    fp32 = consistency.load_contract(arguments.fp32_config)
    int8 = consistency.load_contract(arguments.int8_config)
    evaluator.validate_contract_pair(fp32, int8)
    for contract, artifact, model, name in (
        (fp32, arguments.fp32_artifact, arguments.fp32_model, "fp32"),
        (int8, arguments.int8_artifact, arguments.int8_model, "int8"),
    ):
        evaluator.assert_optional_path(
            contract["artifact_path"], artifact, f"{name}.artifact_path"
        )
        evaluator.assert_optional_path(
            contract["model_path"], model, f"{name}.model_path"
        )
    if fp32["model_path"] != frozen.source_model_path:
        evaluator.fail(
            "protocol.fp32_model",
            str(frozen.source_model_path),
            str(fp32["model_path"]),
            "screen the frozen source model",
        )
    if int8["model_path"] != frozen.output_model_path:
        evaluator.fail(
            "protocol.int8_model",
            str(frozen.output_model_path),
            str(int8["model_path"]),
            "screen the exact candidate produced by this protocol",
        )

    fp32_session = consistency.create_python_session(fp32)
    int8_session = consistency.create_python_session(int8)
    samples = product_manifest["resolved_samples"]
    detections = {
        "fp32": evaluator.run_python_product(
            consistency, fp32, fp32_session, samples
        ),
        "int8": evaluator.run_python_product(
            consistency, int8, int8_session, samples
        ),
    }
    product = evaluator.product_difference(
        detections["fp32"],
        detections["int8"],
        consistency.match_detections,
        quality_manifest["product_matching_gates"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": EVIDENCE_TYPE,
        "passed": bool(product["passed"]),
        "formal_acceptance": False,
        "protocol": {
            "protocol_id": frozen.protocol_id,
            "path": str(frozen.declaration_path),
            "raw_sha256": machine_protocol.sha256_file_raw(
                frozen.declaration_path
            ),
            "canonical_lf_sha256": machine_protocol.sha256_file_canonical_lf(
                frozen.declaration_path
            ),
        },
        "manifests": {
            "product": {
                "manifest_id": product_manifest.get("manifest_id"),
                "path": str(product_manifest["manifest_path"]),
                "raw_sha256": machine_protocol.sha256_file_raw(
                    product_manifest["manifest_path"]
                ),
                "canonical_lf_sha256": product_manifest[
                    "manifest_canonical_lf_sha256"
                ],
                "sample_count": len(samples),
            },
            "quality_gate_source": {
                "manifest_id": quality_manifest["manifest_id"],
                "path": str(quality_manifest["manifest_path"]),
                "canonical_lf_sha256": quality_manifest[
                    "manifest_canonical_lf_sha256"
                ],
            },
        },
        "artifacts": {
            "fp32": evaluator.contract_evidence(fp32),
            "int8": evaluator.contract_evidence(int8),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "onnxruntime_version": consistency.ort.__version__,
            "opencv_version": consistency.cv2.__version__,
            "numpy_version": consistency.np.__version__,
            "provider": consistency.CPU_PROVIDER,
            "execution_mode": "sequential",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "all",
        },
        "product_detection_difference": product,
        "limitations": [
            "Candidate screen only: it is not formal S2-01 acceptance evidence.",
            "The 361-image task-quality metric and Release C++ consistency are intentionally not run here.",
            "A passing candidate must still pass the full formal evaluator before benchmark publication.",
        ],
    }


def write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    resolved = path.resolve()
    temporary = resolved.with_name(resolved.name + ".tmp")
    if resolved.exists() or temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite candidate-screen output: {resolved}"
        )
    serialized = json.dumps(
        document, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        os.replace(temporary, resolved)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the non-formal frozen 30-image S2-01 candidate screen"
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--fp32-config", required=True, type=Path)
    parser.add_argument("--int8-config", required=True, type=Path)
    parser.add_argument("--fp32-artifact", required=True, type=Path)
    parser.add_argument("--int8-artifact", required=True, type=Path)
    parser.add_argument("--fp32-model", required=True, type=Path)
    parser.add_argument("--int8-model", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    output = arguments.output_json.resolve()
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        print(f"refusing to overwrite candidate-screen output: {output}", file=sys.stderr)
        return 1
    try:
        document = screen(arguments)
    except Exception as error:
        document = {
            "schema_version": SCHEMA_VERSION,
            "evidence_type": EVIDENCE_TYPE,
            "passed": False,
            "formal_acceptance": False,
            "setup_error": str(error),
        }
    try:
        write_new_json(output, document)
    except Exception as error:
        print(f"Could not write candidate screen: {error}", file=sys.stderr)
        return 1
    print(f"S2-01 product screen: passed={document.get('passed')}; output={output}")
    if "setup_error" in document:
        print(document["setup_error"], file=sys.stderr)
    return 0 if document.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
