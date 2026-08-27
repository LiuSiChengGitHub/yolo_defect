#!/usr/bin/env python3
"""Create and audit a declared S2-01 static QDQ INT8 artifact.

All heavyweight dependencies are imported only after the dependency-free
protocol loader has verified source/model bytes, the canonical-LF calibration
manifest hash, and every one of the 180 raw calibration image hashes.

The only output override is ``--overwrite``.  Model/report paths and every PTQ
choice come from the machine protocol JSON so a command-line convenience flag
cannot silently change the formal experiment.
"""

from __future__ import annotations

import argparse
import collections
import importlib.metadata
import inspect
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from s2_01_protocol import (
    FROZEN_CALIBRATION_SAMPLE_COUNT,
    FROZEN_SELECTED_CONV_COUNT,
    FrozenS201Protocol,
    S201ProtocolError,
    load_s2_01_protocol,
    sha256_file_canonical_lf,
    sha256_file_raw,
)


REPORT_SCHEMA_VERSION = 1
ENTROPY_HISTOGRAM_DEFAULTS: Mapping[str, int] = {
    "num_bins": 128,
    "num_quantized_bins": 128,
}
EXPECTED_QUANTIZE_STATIC_PARAMETERS = (
    "model_input",
    "model_output",
    "calibration_data_reader",
    "quant_format",
    "op_types_to_quantize",
    "per_channel",
    "reduce_range",
    "activation_type",
    "weight_type",
    "nodes_to_quantize",
    "nodes_to_exclude",
    "use_external_data_format",
    "calibrate_method",
    "extra_options",
)


class S201QuantizationError(RuntimeError):
    """An actionable dependency, quantization, validation, or publish error."""


def fail(object_name: str, expected: str, actual: str, action: str) -> None:
    raise S201QuantizationError(
        "S2-01 PTQ failed: "
        f"object={object_name}; expected={expected}; actual={actual}; "
        f"action={action}"
    )


@dataclass(frozen=True)
class Dependencies:
    cv2: Any
    np: Any
    onnx: Any
    ort: Any
    CalibrationDataReader: Any
    CalibrationMethod: Any
    QuantFormat: Any
    QuantType: Any
    quant_pre_process: Callable[..., None]
    quantize_static: Callable[..., None]
    quantize_static_signature: str
    entropy_calibrater_signature: str


def _distribution_version(candidates: Sequence[str]) -> Mapping[str, str]:
    found: Dict[str, str] = {}
    for candidate in candidates:
        try:
            found[candidate] = importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def load_dependencies(protocol: FrozenS201Protocol) -> Dependencies:
    """Import and pin the exact formal environment after protocol validation."""

    try:
        import cv2
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.calibrate import EntropyCalibrater
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as error:
        fail(
            "python.dependencies",
            "importable cv2, numpy, onnx, onnxruntime, and "
            "onnxruntime.quantization APIs",
            repr(error),
            "run with the pinned S2-01 interpreter; do not install packages "
            "silently from this tool",
        )

    actual_versions = {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
    }
    for key, actual in actual_versions.items():
        expected = protocol.environment[key]
        if actual != expected:
            fail(
                f"python.environment.{key}",
                expected,
                actual,
                "select the exact interpreter declared by the machine protocol",
            )

    provider = protocol.environment["execution_provider"]
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        fail(
            "python.environment.available_providers",
            f"a list containing {provider}",
            repr(available_providers),
            "use an ONNX Runtime 1.19.2 build with CPUExecutionProvider",
        )

    signature = inspect.signature(quantize_static)
    actual_parameters = tuple(signature.parameters)
    if actual_parameters != EXPECTED_QUANTIZE_STATIC_PARAMETERS:
        fail(
            "python.quantize_static.signature",
            repr(EXPECTED_QUANTIZE_STATIC_PARAMETERS),
            repr(actual_parameters),
            "restore the pinned ONNX Runtime quantization API",
        )
    if str(QuantFormat.QDQ) != "QDQ":
        fail(
            "python.QuantFormat.QDQ",
            "QDQ",
            str(QuantFormat.QDQ),
            "restore the pinned quantization enum",
        )
    if (
        QuantType.QInt8.name != "QInt8"
        or QuantType.QUInt8.name != "QUInt8"
        or CalibrationMethod.MinMax.name != "MinMax"
        or CalibrationMethod.Entropy.name != "Entropy"
    ):
        fail(
            "python.quantization.enums",
            "QInt8, QUInt8, MinMax, and Entropy",
            (
                f"{QuantType.QInt8!r}, {QuantType.QUInt8!r}, "
                f"{CalibrationMethod.MinMax!r}, "
                f"{CalibrationMethod.Entropy!r}"
            ),
            "restore the pinned quantization enums",
        )

    entropy_signature = inspect.signature(EntropyCalibrater)
    expected_entropy_defaults: Mapping[str, Any] = {
        "symmetric": False,
        **ENTROPY_HISTOGRAM_DEFAULTS,
    }
    actual_entropy_defaults = {
        parameter_name: (
            entropy_signature.parameters[parameter_name].default
            if parameter_name in entropy_signature.parameters
            else "<missing>"
        )
        for parameter_name in expected_entropy_defaults
    }
    if actual_entropy_defaults != expected_entropy_defaults:
        fail(
            "python.EntropyCalibrater.defaults",
            repr(dict(expected_entropy_defaults)),
            repr(actual_entropy_defaults),
            "restore ONNX Runtime 1.19.2 before Entropy calibration",
        )

    return Dependencies(
        cv2=cv2,
        np=np,
        onnx=onnx,
        ort=ort,
        CalibrationDataReader=CalibrationDataReader,
        CalibrationMethod=CalibrationMethod,
        QuantFormat=QuantFormat,
        QuantType=QuantType,
        quant_pre_process=quant_pre_process,
        quantize_static=quantize_static,
        quantize_static_signature=str(signature),
        entropy_calibrater_signature=str(entropy_signature),
    )


def _resolve_calibration_method(deps: Dependencies, declared_name: str) -> Any:
    methods = {
        "MinMax": deps.CalibrationMethod.MinMax,
        "Entropy": deps.CalibrationMethod.Entropy,
    }
    method = methods.get(declared_name)
    if method is None or getattr(method, "name", None) != declared_name:
        fail(
            "quantization.calibrate_method",
            "a declared MinMax or Entropy enum from the pinned ORT",
            repr(declared_name),
            "restore the protocol or ONNX Runtime 1.19.2 enum mapping",
        )
    return method


def _resolve_quant_type(deps: Dependencies, declared_name: str) -> Any:
    quant_types = {
        "QInt8": deps.QuantType.QInt8,
        "QUInt8": deps.QuantType.QUInt8,
    }
    quant_type = quant_types.get(declared_name)
    if quant_type is None or getattr(quant_type, "name", None) != declared_name:
        fail(
            "quantization.quant_type",
            "a declared QInt8 or QUInt8 enum from the pinned ORT",
            repr(declared_name),
            "restore the protocol or ONNX Runtime quantization enum mapping",
        )
    return quant_type


def _calibration_method_evidence(
    quantization: Mapping[str, Any], deps: Dependencies
) -> Mapping[str, Any]:
    method_name = quantization["calibrate_method"]
    symmetric = quantization["extra_options"][
        "CalibTensorRangeSymmetric"
    ]
    evidence: Dict[str, Any] = {
        "name": method_name,
        "symmetric": symmetric,
        "symmetric_source": (
            "protocol.quantization.extra_options.CalibTensorRangeSymmetric"
        ),
        "entropy_histogram": None,
    }
    if method_name == "Entropy":
        evidence["entropy_histogram"] = {
            **dict(ENTROPY_HISTOGRAM_DEFAULTS),
            "configuration_source": (
                "ONNX Runtime 1.19.2 EntropyCalibrater constructor defaults"
            ),
            "verified_signature": deps.entropy_calibrater_signature,
        }
    return evidence


def load_reference_preprocess() -> Tuple[Callable[..., Tuple[Any, Mapping[str, Any]]], Path]:
    """Reuse the frozen Python/C++ consistency preprocessing implementation."""

    try:
        import compare_consistency
    except ImportError as error:
        fail(
            "calibration.preprocess.implementation",
            "importable sibling tools/compare_consistency.py",
            repr(error),
            "run this script from the checked-out cpp_infer/tools directory",
        )
    compare_consistency.require_dependencies()
    implementation_path = Path(compare_consistency.__file__).resolve(strict=True)
    return compare_consistency.preprocess_image, implementation_path


def ensure_output_targets_available(
    protocol: FrozenS201Protocol, overwrite: bool
) -> None:
    for object_name, path in (
        ("output.model_path", protocol.output_model_path),
        ("output.report_path", protocol.output_report_path),
    ):
        if path.exists() and path.is_dir():
            fail(
                object_name,
                "a regular output file path",
                f"existing directory {path}",
                "choose a file path",
            )
        if path.exists() and not overwrite:
            fail(
                object_name,
                "a path that does not exist when --overwrite is absent",
                f"existing file {path}",
                "inspect the existing evidence, then pass --overwrite only for "
                "an intentional regeneration",
            )
        parent = path.parent
        if parent.exists() and not parent.is_dir():
            fail(
                f"{object_name}.parent",
                "a directory or a path that can be created as a directory",
                f"regular file {parent}",
                "choose a writable output parent",
            )


def _make_calibration_reader(
    deps: Dependencies,
    protocol: FrozenS201Protocol,
    preprocess_image: Callable[..., Tuple[Any, Mapping[str, Any]]],
) -> Any:
    input_name = protocol.model_contract["input_name"]
    input_shape = protocol.model_contract["input_shape"]
    samples = protocol.calibration_samples

    class FrozenCalibrationDataReader(deps.CalibrationDataReader):
        def __init__(self) -> None:
            self._start = 0
            self._end = len(samples)
            self._index = 0
            self.consumed_sample_ids: List[str] = []

        def get_next(self) -> Optional[Mapping[str, Any]]:
            if self._index >= self._end:
                return None
            sample = samples[self._index]
            tensor, _ = preprocess_image(sample.image_path, input_shape)
            self._index += 1
            self.consumed_sample_ids.append(sample.sample_id)
            return {input_name: tensor}

        def __len__(self) -> int:
            return self._end - self._start

        def set_range(self, start_index: int, end_index: int) -> None:
            if (
                type(start_index) is not int
                or type(end_index) is not int
                or start_index < 0
                or end_index < start_index
                or end_index > len(samples)
            ):
                fail(
                    "calibration.reader.range",
                    f"0 <= start <= end <= {len(samples)}",
                    f"start={start_index}, end={end_index}",
                    "use the complete frozen reader without an invalid stride range",
                )
            self._start = start_index
            self._end = end_index
            self._index = start_index
            self.consumed_sample_ids = []

        def rewind(self) -> None:
            self._index = self._start
            self.consumed_sample_ids = []

    return FrozenCalibrationDataReader()


def _tensor_shape(value_info: Any) -> List[Any]:
    tensor_type = value_info.type.tensor_type
    shape: List[Any] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def _onnx_value_info(value_info: Any, onnx: Any) -> Mapping[str, Any]:
    tensor_type = value_info.type.tensor_type
    try:
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    except ValueError:
        dtype = f"UNKNOWN({tensor_type.elem_type})"
    return {
        "name": value_info.name,
        "dtype": dtype,
        "shape": _tensor_shape(value_info),
    }


def _op_counts(model: Any) -> Mapping[str, int]:
    return dict(
        sorted(collections.Counter(node.op_type for node in model.graph.node).items())
    )


def _initializer_dtype_counts(model: Any, onnx: Any) -> Mapping[str, int]:
    counts: collections.Counter = collections.Counter()
    for initializer in model.graph.initializer:
        try:
            name = onnx.TensorProto.DataType.Name(initializer.data_type)
        except ValueError:
            name = f"UNKNOWN({initializer.data_type})"
        counts[name] += 1
    return dict(sorted(counts.items()))


def _onnx_metadata(model: Any, onnx: Any) -> Mapping[str, Any]:
    return {
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opset_imports": [
            {
                "domain": item.domain if item.domain else "ai.onnx",
                "version": int(item.version),
            }
            for item in model.opset_import
        ],
        "inputs": [_onnx_value_info(value, onnx) for value in model.graph.input],
        "outputs": [_onnx_value_info(value, onnx) for value in model.graph.output],
        "value_info_count": len(model.graph.value_info),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "op_counts": _op_counts(model),
        "initializer_dtype_counts": _initializer_dtype_counts(model, onnx),
        "metadata_properties": {
            item.key: item.value for item in model.metadata_props
        },
    }


def _load_and_check_model(path: Path, deps: Dependencies, object_name: str) -> Any:
    try:
        model = deps.onnx.load(str(path), load_external_data=True)
        deps.onnx.checker.check_model(model)
    except Exception as error:
        fail(
            object_name,
            "a loadable ONNX model accepted by onnx.checker",
            f"{type(error).__name__}: {error}",
            "inspect the preceding preprocess/quantization operation and model bytes",
        )
    return model


def _validate_source_conv_nodes(model: Any) -> List[str]:
    names = [node.name for node in model.graph.node if node.op_type == "Conv"]
    if len(names) != FROZEN_SELECTED_CONV_COUNT:
        fail(
            "source.graph.Conv.count",
            str(FROZEN_SELECTED_CONV_COUNT),
            str(len(names)),
            "restore the frozen FP32 source graph",
        )
    if any(not name for name in names):
        fail(
            "source.graph.Conv.names",
            "64 non-empty names",
            "at least one empty name",
            "restore stable node identities before graph auditing",
        )
    if len(set(names)) != len(names):
        fail(
            "source.graph.Conv.names",
            "64 unique names",
            "duplicate Conv node names",
            "restore stable node identities before graph auditing",
        )
    return names


def _tensor_element_count(tensor: Any) -> int:
    count = 1
    for dimension in tensor.dims:
        count *= int(dimension)
    return count


def _audit_qdq_graph(
    source_model: Any,
    derived_model: Any,
    selected_conv_names: Sequence[str],
    excluded_conv_names: Sequence[str],
    deps: Dependencies,
) -> Mapping[str, Any]:
    selected_set = set(selected_conv_names)
    excluded_set = set(excluded_conv_names)
    if len(excluded_set) != len(excluded_conv_names):
        fail(
            "graph_audit.excluded_conv_nodes",
            "unique source Conv identities",
            repr(list(excluded_conv_names)),
            "restore the duplicate-free frozen exclusion list",
        )
    unknown_exclusions = [
        name for name in excluded_conv_names if name not in selected_set
    ]
    if unknown_exclusions:
        fail(
            "graph_audit.excluded_conv_nodes",
            "only source Conv node identities",
            repr(unknown_exclusions),
            "restore the frozen source-addressable exclusion list",
        )
    source_ordered_exclusions = [
        name for name in selected_conv_names if name in excluded_set
    ]
    if source_ordered_exclusions != list(excluded_conv_names):
        fail(
            "graph_audit.excluded_conv_nodes.order",
            repr(source_ordered_exclusions),
            repr(list(excluded_conv_names)),
            "keep exclusions in source graph order",
        )
    target_conv_names = [
        name for name in selected_conv_names if name not in excluded_set
    ]
    derived_by_name = {node.name: node for node in derived_model.graph.node}
    producers = {
        output_name: node
        for node in derived_model.graph.node
        for output_name in node.output
        if output_name
    }
    consumers: Dict[str, List[Any]] = collections.defaultdict(list)
    for node in derived_model.graph.node:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    initializers = {
        initializer.name: initializer
        for initializer in derived_model.graph.initializer
    }
    int8_type = deps.onnx.TensorProto.INT8
    float_type = deps.onnx.TensorProto.FLOAT

    quantized_names: List[str] = []
    failed: List[Mapping[str, Any]] = []
    excluded_policy_violations: List[Mapping[str, Any]] = []
    details: List[Mapping[str, Any]] = []
    for name in selected_conv_names:
        reasons: List[str] = []
        node = derived_by_name.get(name)
        if name in excluded_set:
            activation_dq = None
            weight_dq = None
            direct_weight_initializer = None
            output_quantizers: List[str] = []
            if node is None:
                reasons.append("excluded source Conv name is absent from derived graph")
            elif node.op_type != "Conv":
                reasons.append(
                    f"excluded derived node op_type is {node.op_type!r}, not 'Conv'"
                )
            elif len(node.input) < 2:
                reasons.append(
                    f"excluded derived Conv has only {len(node.input)} inputs"
                )
            else:
                activation_dq = producers.get(node.input[0])
                weight_dq = producers.get(node.input[1])
                if weight_dq is not None and weight_dq.op_type == "DequantizeLinear":
                    reasons.append(
                        "excluded Conv weight is produced by DequantizeLinear"
                    )
                else:
                    direct_weight_initializer = initializers.get(node.input[1])
                    if direct_weight_initializer is None:
                        reasons.append(
                            "excluded Conv weight is not a direct FP32 initializer"
                        )
                    elif direct_weight_initializer.data_type != float_type:
                        reasons.append(
                            "excluded Conv direct weight initializer dtype is "
                            f"{direct_weight_initializer.data_type}, not "
                            "TensorProto.FLOAT"
                        )
                output_quantizers = [
                    consumer.name
                    for output_name in node.output
                    for consumer in consumers.get(output_name, [])
                    if consumer.op_type == "QuantizeLinear"
                ]
            if reasons:
                excluded_policy_violations.append(
                    {"name": name, "reasons": list(reasons)}
                )
            details.append(
                {
                    "name": name,
                    "policy": "intentional_unquantized",
                    "quantized": False,
                    "intentional_unquantized": True,
                    "policy_compliant": not reasons,
                    "activation_dequantize_node": (
                        activation_dq.name
                        if activation_dq is not None
                        and activation_dq.op_type == "DequantizeLinear"
                        else None
                    ),
                    "weight_dequantize_node": (
                        weight_dq.name
                        if weight_dq is not None
                        and weight_dq.op_type == "DequantizeLinear"
                        else None
                    ),
                    "direct_fp32_weight_initializer": (
                        direct_weight_initializer.name
                        if direct_weight_initializer is not None
                        else None
                    ),
                    "output_quantize_nodes": output_quantizers,
                    "reasons": reasons,
                }
            )
            continue
        if node is None:
            reasons.append("source Conv name is absent from derived graph")
            details.append(
                {
                    "name": name,
                    "policy": "quantization_target",
                    "quantized": False,
                    "intentional_unquantized": False,
                    "reasons": reasons,
                }
            )
            failed.append({"name": name, "reasons": reasons})
            continue
        if node.op_type != "Conv":
            reasons.append(f"derived node op_type is {node.op_type!r}, not 'Conv'")
        if len(node.input) < 2:
            reasons.append(f"derived Conv has only {len(node.input)} inputs")

        activation_dq = producers.get(node.input[0]) if node.input else None
        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            reasons.append("activation input is not produced by DequantizeLinear")
        else:
            activation_q = (
                producers.get(activation_dq.input[0])
                if activation_dq.input
                else None
            )
            if activation_q is None or activation_q.op_type != "QuantizeLinear":
                reasons.append(
                    "activation DequantizeLinear input is not produced by QuantizeLinear"
                )

        weight_dq = producers.get(node.input[1]) if len(node.input) >= 2 else None
        quantized_weight = None
        weight_scale = None
        expected_weight_channels = None
        if weight_dq is None or weight_dq.op_type != "DequantizeLinear":
            reasons.append("weight input is not produced by DequantizeLinear")
        elif not weight_dq.input:
            reasons.append("weight DequantizeLinear has no quantized data input")
        else:
            quantized_weight = initializers.get(weight_dq.input[0])
            if quantized_weight is None:
                reasons.append("weight DequantizeLinear data input is not an initializer")
            elif quantized_weight.data_type != int8_type:
                reasons.append(
                    "weight initializer dtype is "
                    f"{quantized_weight.data_type}, not TensorProto.INT8"
                )
            elif not quantized_weight.dims:
                reasons.append("weight initializer has no output-channel dimension")
            else:
                expected_weight_channels = int(quantized_weight.dims[0])
            if len(weight_dq.input) >= 2:
                weight_scale = initializers.get(weight_dq.input[1])
                if weight_scale is None:
                    reasons.append("weight scale is not an initializer")
                elif (
                    expected_weight_channels is not None
                    and _tensor_element_count(weight_scale)
                    != expected_weight_channels
                ):
                    reasons.append(
                        "weight scale element count is "
                        f"{_tensor_element_count(weight_scale)}, expected "
                        f"{expected_weight_channels} output channels"
                    )

        output_quantizers = [
            consumer.name
            for output_name in node.output
            for consumer in consumers.get(output_name, [])
            if consumer.op_type == "QuantizeLinear"
        ]
        if not output_quantizers:
            reasons.append("Conv output is not consumed by QuantizeLinear")

        quantized = not reasons
        if quantized:
            quantized_names.append(name)
        else:
            failed.append({"name": name, "reasons": reasons})
        details.append(
            {
                "name": name,
                "policy": "quantization_target",
                "quantized": quantized,
                "intentional_unquantized": False,
                "activation_dequantize_node": (
                    activation_dq.name
                    if activation_dq is not None
                    and activation_dq.op_type == "DequantizeLinear"
                    else None
                ),
                "weight_dequantize_node": (
                    weight_dq.name
                    if weight_dq is not None
                    and weight_dq.op_type == "DequantizeLinear"
                    else None
                ),
                "quantized_weight_initializer": (
                    quantized_weight.name if quantized_weight is not None else None
                ),
                "per_channel_scale_elements": (
                    _tensor_element_count(weight_scale)
                    if weight_scale is not None
                    else None
                ),
                "weight_output_channels": expected_weight_channels,
                "output_quantize_nodes": output_quantizers,
                "reasons": reasons,
            }
        )

    source_unselected_nodes = [
        {"name": node.name, "op_type": node.op_type}
        for node in source_model.graph.node
        if node.name not in selected_set
    ]
    source_unselected_counts = collections.Counter(
        node["op_type"] for node in source_unselected_nodes
    )
    unquantized_names = [
        name for name in target_conv_names if name not in set(quantized_names)
    ]
    return {
        "selection": {
            "op_types_to_quantize": ["Conv"],
            "expected_selected_count": FROZEN_SELECTED_CONV_COUNT,
            "selected_count": len(selected_conv_names),
            "selected_conv_nodes": list(selected_conv_names),
            "expected_excluded_count": len(excluded_conv_names),
            "excluded_conv_count": len(excluded_conv_names),
            "excluded_conv_nodes": list(excluded_conv_names),
            "expected_target_count": (
                FROZEN_SELECTED_CONV_COUNT - len(excluded_conv_names)
            ),
            "target_conv_count": len(target_conv_names),
            "target_conv_nodes": target_conv_names,
            "source_unselected_nodes": source_unselected_nodes,
            "source_unselected_op_counts": dict(
                sorted(source_unselected_counts.items())
            ),
        },
        "result": {
            "quantized_conv_count": len(quantized_names),
            "quantized_conv_nodes": quantized_names,
            "intentional_unquantized_conv_count": len(excluded_conv_names),
            "intentional_unquantized_conv_nodes": list(excluded_conv_names),
            "unquantized_conv_count": len(unquantized_names),
            "unquantized_conv_nodes": unquantized_names,
            "failed_conv_count": len(failed),
            "failed_conv_nodes": failed,
            "excluded_policy_violation_count": len(
                excluded_policy_violations
            ),
            "excluded_policy_violations": excluded_policy_violations,
        },
        "derived_graph": {
            "node_count": len(derived_model.graph.node),
            "op_counts": _op_counts(derived_model),
            "quantize_linear_count": sum(
                node.op_type == "QuantizeLinear"
                for node in derived_model.graph.node
            ),
            "dequantize_linear_count": sum(
                node.op_type == "DequantizeLinear"
                for node in derived_model.graph.node
            ),
            "initializer_count": len(derived_model.graph.initializer),
            "initializer_dtype_counts": _initializer_dtype_counts(
                derived_model, deps.onnx
            ),
        },
        "conv_details": details,
        "classification_note": (
            "QDQ preserves Conv op_type; a Conv is classified as quantized only "
            "when its activation and INT8 per-channel weight Q/DQ structure and "
            "output QuantizeLinear are present. Protocol-excluded Conv nodes are "
            "classified separately as intentional_unquantized and must retain a "
            "direct FP32 weight initializer."
        ),
    }


def _validate_session_metadata(
    inputs: Sequence[Any], outputs: Sequence[Any], contract: Mapping[str, Any]
) -> None:
    if len(inputs) != 1 or len(outputs) != 1:
        fail(
            "runtime.session.io_count",
            "one input and one output",
            f"inputs={len(inputs)}, outputs={len(outputs)}",
            "restore the frozen single-input/single-output model",
        )
    expected = (
        (
            "input[0]",
            inputs[0],
            contract["input_name"],
            contract["input_shape"],
            "tensor(float)",
        ),
        (
            "output[0]",
            outputs[0],
            contract["output_name"],
            contract["output_shape"],
            "tensor(float)",
        ),
    )
    for object_name, actual, name, shape, dtype in expected:
        if (
            actual.name != name
            or list(actual.shape) != list(shape)
            or actual.type != dtype
        ):
            fail(
                f"runtime.session.{object_name}",
                f"name={name}, shape={list(shape)}, type={dtype}",
                f"name={actual.name}, shape={list(actual.shape)}, type={actual.type}",
                "restore the artifact tensor contract or generated model",
            )


def _validate_python_session(
    path: Path,
    tensor: Any,
    protocol: FrozenS201Protocol,
    deps: Dependencies,
    object_name: str,
) -> Mapping[str, Any]:
    provider = protocol.environment["execution_provider"]
    options = deps.ort.SessionOptions()
    options.execution_mode = deps.ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        deps.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.log_severity_level = 2
    initialization_start = time.perf_counter()
    try:
        session = deps.ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[(provider, {"use_arena": "1"})],
        )
    except Exception as error:
        fail(
            f"{object_name}.session_creation",
            f"a loadable {provider} session",
            f"{type(error).__name__}: {error}",
            "inspect model/checker output and the pinned ORT environment",
        )
    initialization_ms = (time.perf_counter() - initialization_start) * 1000.0
    if session.get_providers() != [provider]:
        fail(
            f"{object_name}.session_providers",
            repr([provider]),
            repr(session.get_providers()),
            "keep CUDA/TensorRT fallback out of CPU PTQ validation",
        )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _validate_session_metadata(inputs, outputs, protocol.model_contract)

    run_start = time.perf_counter()
    try:
        values = session.run(
            [protocol.model_contract["output_name"]],
            {protocol.model_contract["input_name"]: tensor},
        )
    except Exception as error:
        fail(
            f"{object_name}.session_run",
            "one successful finite CPU inference",
            f"{type(error).__name__}: {error}",
            "inspect calibration preprocessing, model graph, and provider",
        )
    run_ms = (time.perf_counter() - run_start) * 1000.0
    if len(values) != 1:
        fail(
            f"{object_name}.outputs",
            "one output array",
            str(len(values)),
            "restore the frozen model contract",
        )
    output = deps.np.asarray(values[0])
    if output.dtype != deps.np.float32:
        fail(
            f"{object_name}.output.dtype",
            "float32",
            str(output.dtype),
            "keep external QDQ model output in float32",
        )
    if list(output.shape) != list(protocol.model_contract["output_shape"]):
        fail(
            f"{object_name}.output.shape",
            repr(protocol.model_contract["output_shape"]),
            repr(list(output.shape)),
            "restore the frozen tensor contract",
        )
    if not deps.np.isfinite(output).all():
        fail(
            f"{object_name}.output.values",
            "all finite values",
            "NaN or Infinity",
            "inspect quantization ranges and input preprocessing",
        )
    return {
        "status": "passed",
        "model_path": str(path),
        "requested_provider": provider,
        "actual_providers": session.get_providers(),
        "execution_mode": "sequential",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": "all",
        "session_initialization_ms": initialization_ms,
        "session_run_ms": run_ms,
        "input": {
            "name": inputs[0].name,
            "type": inputs[0].type,
            "shape": list(inputs[0].shape),
        },
        "output": {
            "name": outputs[0].name,
            "type": outputs[0].type,
            "shape": list(outputs[0].shape),
            "element_count": int(output.size),
            "minimum": float(output.min()),
            "maximum": float(output.max()),
            "mean": float(output.mean(dtype=deps.np.float64)),
            "all_finite": True,
        },
    }


def _artifact_record(
    serialized_path: Path,
    reported_path: Optional[Path],
    model: Any,
    deps: Dependencies,
) -> Mapping[str, Any]:
    size_bytes = serialized_path.stat().st_size
    return {
        "path": str(reported_path) if reported_path is not None else None,
        "published": reported_path is not None,
        "sha256": sha256_file_raw(serialized_path),
        "size_bytes": size_bytes,
        "size_mebibytes": size_bytes / (1024.0 * 1024.0),
        "onnx_checker": "passed",
        "actual_metadata": _onnx_metadata(model, deps.onnx),
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_staged_report(report: Mapping[str, Any], target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            delete=False,
        )
        temporary_path = Path(handle.name)
        with handle:
            handle.write(_canonical_json_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        return temporary_path
    except (OSError, TypeError, ValueError) as error:
        fail(
            "output.report.temp",
            f"a writable temporary file beside {target}",
            str(error),
            "check output directory permissions and free space",
        )


def _publish_outputs(
    staged_model: Path,
    staged_report: Path,
    protocol: FrozenS201Protocol,
    overwrite: bool,
) -> None:
    # Recheck immediately before replacement so an ordinary concurrent writer
    # is rejected when --overwrite was not explicit.
    if not overwrite:
        ensure_output_targets_available(protocol, overwrite=False)
    sibling_model: Optional[Path] = None
    try:
        # Copy into a sibling created directly under the destination parent.
        # Moving a file out of tempfile.TemporaryDirectory can preserve that
        # directory's restrictive Windows ACL and make the published artifact
        # unreadable to the normal build/test process.
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{protocol.output_model_path.name}.",
            suffix=".tmp",
            dir=str(protocol.output_model_path.parent),
            delete=False,
        )
        sibling_model = Path(handle.name)
        with staged_model.open("rb") as source, handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        if (
            sibling_model.stat().st_size != staged_model.stat().st_size
            or sha256_file_raw(sibling_model) != sha256_file_raw(staged_model)
        ):
            fail(
                "output.model.staging_integrity",
                "a byte-identical sibling copy",
                str(sibling_model),
                "check output filesystem writes and free space",
            )
        os.replace(sibling_model, protocol.output_model_path)
        sibling_model = None
    except (OSError, S201QuantizationError) as error:
        fail(
            "output.model.publish",
            f"verified atomic replacement of {protocol.output_model_path}",
            str(error),
            "keep staging and destination on one writable filesystem",
        )
    finally:
        if sibling_model is not None and sibling_model.exists():
            sibling_model.unlink()
    try:
        os.replace(staged_report, protocol.output_report_path)
    except OSError as error:
        fail(
            "output.report.publish",
            f"atomic replacement of {protocol.output_report_path}",
            str(error),
            "inspect the already validated model output and repair report path permissions",
        )


def run_quantization(
    protocol: FrozenS201Protocol,
    overwrite: bool,
    command_arguments: Sequence[str],
) -> Mapping[str, Any]:
    ensure_output_targets_available(protocol, overwrite)
    deps = load_dependencies(protocol)
    preprocess_image, preprocess_implementation_path = load_reference_preprocess()

    for object_name, parent in (
        ("output.model_path.parent", protocol.output_model_path.parent),
        ("output.report_path.parent", protocol.output_report_path.parent),
    ):
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            fail(
                object_name,
                "a writable output directory",
                str(error),
                "correct the protocol output path or directory permissions",
            )
    try:
        temporary_context = tempfile.TemporaryDirectory(
            prefix=".s2_01_ptq_", dir=str(protocol.output_model_path.parent)
        )
    except OSError as error:
        fail(
            "quantization.temporary_directory",
            "a writable temporary directory beside the model target",
            str(error),
            "check model output directory permissions and free space",
        )
    with temporary_context as temporary_directory:
        temporary_root = Path(temporary_directory)
        preprocessed_path = temporary_root / "source.preprocessed.onnx"
        derived_path = temporary_root / "derived.int8.qdq.onnx"

        source_model = _load_and_check_model(
            protocol.source_model_path, deps, "source_model.onnx"
        )
        selected_conv_names = _validate_source_conv_nodes(source_model)
        excluded_conv_names = list(
            protocol.quantization["nodes_to_exclude"]
        )
        excluded_set = set(excluded_conv_names)
        source_ordered_exclusions = [
            name for name in selected_conv_names if name in excluded_set
        ]
        if source_ordered_exclusions != excluded_conv_names:
            fail(
                "source.graph.Conv.exclusions",
                repr(excluded_conv_names),
                repr(source_ordered_exclusions),
                "restore the exact source-graph ordered exclusion identities",
            )
        target_conv_count = (
            FROZEN_SELECTED_CONV_COUNT - len(excluded_conv_names)
        )

        preprocess_options = protocol.quantization["preprocess"]
        try:
            deps.quant_pre_process(
                input_model=str(protocol.source_model_path),
                output_model_path=str(preprocessed_path),
                skip_optimization=preprocess_options["skip_optimization"],
                skip_onnx_shape=preprocess_options["skip_onnx_shape"],
                skip_symbolic_shape=preprocess_options["skip_symbolic_shape"],
            )
        except Exception as error:
            fail(
                "quant_pre_process",
                "successful ONNX shape inference with graph optimization and "
                "symbolic shape inference disabled",
                f"{type(error).__name__}: {error}",
                "inspect source ONNX shape inference compatibility",
            )
        preprocessed_model = _load_and_check_model(
            preprocessed_path, deps, "preprocessed_model.onnx"
        )
        preprocessed_conv_names = [
            node.name
            for node in preprocessed_model.graph.node
            if node.op_type == "Conv"
        ]
        if preprocessed_conv_names != selected_conv_names:
            fail(
                "preprocessed_model.Conv.names",
                "the same ordered 64 Conv identities as the raw source",
                repr(preprocessed_conv_names),
                "keep optimization disabled so graph auditing remains source-addressable",
            )

        reader = _make_calibration_reader(
            deps, protocol, preprocess_image
        )
        quantization = protocol.quantization
        calibration_method = _resolve_calibration_method(
            deps, quantization["calibrate_method"]
        )
        activation_type = _resolve_quant_type(
            deps, quantization["activation_type"]
        )
        weight_type = _resolve_quant_type(deps, quantization["weight_type"])
        try:
            deps.quantize_static(
                model_input=str(preprocessed_path),
                model_output=str(derived_path),
                calibration_data_reader=reader,
                quant_format=deps.QuantFormat.QDQ,
                op_types_to_quantize=list(
                    quantization["op_types_to_quantize"]
                ),
                per_channel=quantization["per_channel"],
                reduce_range=quantization["reduce_range"],
                activation_type=activation_type,
                weight_type=weight_type,
                nodes_to_quantize=None,
                nodes_to_exclude=list(quantization["nodes_to_exclude"]),
                use_external_data_format=quantization[
                    "use_external_data_format"
                ],
                calibrate_method=calibration_method,
                extra_options=dict(quantization["extra_options"]),
            )
        except Exception as error:
            fail(
                "quantize_static",
                "successful frozen "
                f"{quantization['calibrate_method']} QDQ/"
                f"{quantization['activation_type']}/"
                f"{quantization['weight_type']} static PTQ",
                f"{type(error).__name__}: {error}",
                "inspect the first calibration or graph diagnostic; do not "
                "change the selected protocol in place",
            )
        if reader.consumed_sample_ids != [
            sample.sample_id for sample in protocol.calibration_samples
        ]:
            fail(
                "calibration.reader.consumed_samples",
                "all 180 manifest samples once and in manifest order",
                repr(reader.consumed_sample_ids),
                "restore the non-strided frozen calibration reader",
            )

        derived_model = _load_and_check_model(
            derived_path, deps, "derived_model.onnx"
        )
        graph_audit = _audit_qdq_graph(
            source_model,
            derived_model,
            selected_conv_names,
            excluded_conv_names,
            deps,
        )
        audit_result = graph_audit["result"]
        if (
            audit_result["quantized_conv_count"] != target_conv_count
            or audit_result["failed_conv_count"] != 0
            or audit_result["excluded_policy_violation_count"] != 0
        ):
            fail(
                "derived_model.quantized_conv_nodes",
                f"{target_conv_count} target Conv nodes with complete QDQ, "
                f"{len(excluded_conv_names)} intentional FP32 exclusions, and "
                "no failures or exclusion-policy violations",
                json.dumps(
                    {
                        "quantized_conv_count": audit_result[
                            "quantized_conv_count"
                        ],
                        "failed_conv_nodes": audit_result[
                            "failed_conv_nodes"
                        ],
                        "excluded_policy_violations": audit_result[
                            "excluded_policy_violations"
                        ],
                    },
                    ensure_ascii=False,
                ),
                "inspect target Q/DQ structure and verify every excluded Conv "
                "retains a direct FP32 weight initializer",
            )

        smoke_sample = protocol.calibration_samples[0]
        tensor, transform = preprocess_image(
            smoke_sample.image_path, protocol.model_contract["input_shape"]
        )
        if tensor.dtype != deps.np.float32 or list(tensor.shape) != list(
            protocol.model_contract["input_shape"]
        ):
            fail(
                "runtime.smoke_input",
                "contiguous float32 input matching model_contract.input_shape",
                f"dtype={tensor.dtype}, shape={list(tensor.shape)}",
                "restore the frozen reference preprocessing implementation",
            )
        if not tensor.flags.c_contiguous or not deps.np.isfinite(tensor).all():
            fail(
                "runtime.smoke_input.values",
                "contiguous finite values",
                f"contiguous={tensor.flags.c_contiguous}",
                "inspect image decode, normalization, and NCHW conversion",
            )

        source_session = _validate_python_session(
            protocol.source_model_path,
            tensor,
            protocol,
            deps,
            "runtime.source",
        )
        derived_session = _validate_python_session(
            derived_path, tensor, protocol, deps, "runtime.derived"
        )

        artifacts = {
            "source": _artifact_record(
                protocol.source_model_path,
                protocol.source_model_path,
                source_model,
                deps,
            ),
            "preprocessed": _artifact_record(
                preprocessed_path, None, preprocessed_model, deps
            ),
            "derived": _artifact_record(
                derived_path,
                protocol.output_model_path,
                derived_model,
                deps,
            ),
        }
        artifacts["preprocessed"]["scope"] = (
            "temporary quantization intermediate; SHA and metadata retained in "
            "this report, bytes not published"
        )

        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "evidence_type": "s2_01_static_ptq_artifact_card",
            "passed": True,
            "timestamp_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "command": list(command_arguments),
            "protocol": {
                "protocol_id": protocol.protocol_id,
                "path": str(protocol.declaration_path),
                "raw_sha256": sha256_file_raw(protocol.declaration_path),
                "canonical_lf_sha256": sha256_file_canonical_lf(
                    protocol.declaration_path
                ),
                "schema_version": protocol.document["schema_version"],
            },
            "artifact_contract": {
                "artifact_kind": (
                    "onnx_static_ptq_int8_qdq_"
                    f"{quantization['activation_type'].lower()}_"
                    f"{quantization['weight_type'].lower()}"
                ),
                "source_model": {
                    "path": str(protocol.source_model_path),
                    "sha256": protocol.source_model_sha256,
                    "size_bytes": protocol.source_model_size_bytes,
                },
                "derived_model_path": str(protocol.output_model_path),
                "external_model_io": dict(protocol.model_contract),
                "quantized_op_scope": {
                    "op_types_to_quantize": ["Conv"],
                    "selected_conv_count": FROZEN_SELECTED_CONV_COUNT,
                    "excluded_conv_count": len(excluded_conv_names),
                    "excluded_conv_nodes": excluded_conv_names,
                    "target_conv_count": target_conv_count,
                    "unselected_source_nodes_remain_in_declared_precision": True,
                },
                "python_runtime_legality_provider": protocol.environment[
                    "execution_provider"
                ],
                "cpp_runtime_legality_required_separately": True,
            },
            "frozen_downstream_protocol": {
                "correctness": {
                    "declaration": dict(protocol.correctness),
                    "consistency_manifest_path": str(
                        protocol.consistency_manifest_path
                    ),
                    "quality_manifest_path": str(protocol.quality_manifest_path),
                    "quality_evaluation": dict(protocol.quality_evaluation),
                    "product_matching_protocol": dict(
                        protocol.product_matching_protocol
                    ),
                    "product_matching_gates": dict(
                        protocol.product_matching_gates
                    ),
                    "quality_gates": dict(protocol.quality_gates),
                },
                "benchmark": {
                    "declaration": dict(protocol.benchmark),
                    "resolved_sample_path": str(protocol.benchmark_sample_path),
                },
                "profiling": dict(protocol.profiling),
            },
            "environment": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "packages": {
                    "onnx": deps.onnx.__version__,
                    "onnxruntime": deps.ort.__version__,
                    "numpy": deps.np.__version__,
                    "opencv": deps.cv2.__version__,
                    "onnxruntime_distributions": _distribution_version(
                        ("onnxruntime", "onnxruntime-gpu")
                    ),
                },
                "available_providers": deps.ort.get_available_providers(),
                "calibration_and_validation_provider": protocol.environment[
                    "execution_provider"
                ],
            },
            "calibration": {
                "manifest_id": protocol.calibration_manifest_id,
                "manifest_path": str(protocol.calibration_manifest_path),
                "manifest_sha256_canonical_lf": (
                    protocol.calibration_manifest_sha256_canonical_lf
                ),
                "sample_count_expected": FROZEN_CALIBRATION_SAMPLE_COUNT,
                "sample_count_hash_verified": len(
                    protocol.calibration_samples
                ),
                "sample_count_consumed": len(reader.consumed_sample_ids),
                "consumption_order": "manifest_order",
                "preprocess": dict(protocol.calibration_preprocess),
                "preprocess_implementation": {
                    "path": str(preprocess_implementation_path),
                    "raw_sha256": sha256_file_raw(
                        preprocess_implementation_path
                    ),
                },
            },
            "quantization": {
                "api": "onnxruntime.quantization.quantize_static",
                "api_signature": deps.quantize_static_signature,
                "parameters": dict(protocol.quantization),
                "calibration_method": _calibration_method_evidence(
                    protocol.quantization, deps
                ),
                "calibration_reader": "one contiguous float32 NCHW tensor per call",
            },
            "artifacts": artifacts,
            "model_size_comparison": {
                "source_fp32_size_bytes": artifacts["source"]["size_bytes"],
                "derived_int8_size_bytes": artifacts["derived"]["size_bytes"],
                "size_delta_bytes": (
                    artifacts["derived"]["size_bytes"]
                    - artifacts["source"]["size_bytes"]
                ),
                "int8_to_fp32_ratio": (
                    artifacts["derived"]["size_bytes"]
                    / artifacts["source"]["size_bytes"]
                ),
                "size_reduction_percent": (
                    100.0
                    * (
                        artifacts["source"]["size_bytes"]
                        - artifacts["derived"]["size_bytes"]
                    )
                    / artifacts["source"]["size_bytes"]
                ),
            },
            "graph_audit": graph_audit,
            "runtime_validation": {
                "smoke_sample": {
                    "sample_id": smoke_sample.sample_id,
                    "image_path": str(smoke_sample.image_path),
                    "image_sha256": smoke_sample.image_sha256,
                    "letterbox_transform": dict(transform),
                },
                "source_python_ort": source_session,
                "derived_python_ort": derived_session,
            },
            "tooling": {
                "quantize_tool_path": str(Path(__file__).resolve()),
                "quantize_tool_raw_sha256": sha256_file_raw(
                    Path(__file__).resolve()
                ),
                "protocol_tool_path": str(
                    (Path(__file__).resolve().parent / "s2_01_protocol.py")
                ),
                "protocol_tool_raw_sha256": sha256_file_raw(
                    Path(__file__).resolve().parent / "s2_01_protocol.py"
                ),
            },
            "limitations": [
                "This card proves Python ONNX checker/session legality and QDQ "
                "graph structure; the separate C++ Runtime gate remains required.",
                "Calibration uses exactly 180 frozen images and does not imply "
                "task-quality acceptance on the labeled evaluation split.",
                "The preprocessed ONNX is a temporary lineage intermediate and "
                "is reproducible only with the pinned tool/environment/protocol.",
                "Python uses CPUExecutionProvider explicitly even when the wheel "
                "also exposes CUDA or TensorRT providers.",
            ],
        }

        staged_report = _write_staged_report(report, protocol.output_report_path)
        try:
            _publish_outputs(
                derived_path, staged_report, protocol, overwrite=overwrite
            )
        finally:
            if staged_report.exists():
                staged_report.unlink()
        return report


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run declared S2-01 ONNX Runtime static QDQ INT8 PTQ and publish "
            "an audited INT8 model/card."
        )
    )
    parser.add_argument(
        "--protocol",
        required=True,
        type=Path,
        help="Path to the frozen S2-01 machine protocol JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace existing model/report targets atomically.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    command_arguments = [sys.executable, str(Path(__file__).resolve())]
    command_arguments.extend(
        sys.argv[1:] if argv is None else list(argv)
    )
    try:
        protocol = load_s2_01_protocol(arguments.protocol)
        report = run_quantization(
            protocol,
            overwrite=arguments.overwrite,
            command_arguments=command_arguments,
        )
    except (S201ProtocolError, S201QuantizationError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        "S2-01 PTQ PASS: "
        f"source_sha={report['artifacts']['source']['sha256']}; "
        f"derived_sha={report['artifacts']['derived']['sha256']}; "
        f"quantized_conv={report['graph_audit']['result']['quantized_conv_count']}/"
        f"{report['graph_audit']['selection']['target_conv_count']}; "
        f"excluded_conv="
        f"{report['graph_audit']['selection']['excluded_conv_count']}; "
        f"calibrate_method={protocol.quantization['calibrate_method']}; "
        f"activation_type={protocol.quantization['activation_type']}; "
        f"weight_type={protocol.quantization['weight_type']}; "
        f"model={protocol.output_model_path}; "
        f"report={protocol.output_report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
