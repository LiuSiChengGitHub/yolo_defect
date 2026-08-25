"""Pure tests for S2-01 ONNX Runtime trace aggregation semantics."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = CPP_INFER_ROOT / "tools" / "summarize_ort_profile.py"
SPEC = importlib.util.spec_from_file_location("summarize_ort_profile", TOOL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import profile tool from {TOOL_PATH}")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ProfileSummaryTest(unittest.TestCase):
    def summarize(self, events: list, top_n: int = 20, expected_runs: int = 1):
        session_events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 1}
            for _ in range(expected_runs)
        ]
        return profile.summarize_events(
            session_events + events,
            trace_path=Path("synthetic_trace.json"),
            model_id="synthetic",
            declared_model_sha256="A" * 64,
            precision="int8",
            expected_provider="CPUExecutionProvider",
            expected_profile_runs=expected_runs,
            top_n=top_n,
        )

    def test_filters_fence_events_and_aggregates_nodes_operators_providers(self):
        summary = self.summarize(
            [
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_0_fence_before",
                        "dur": 50,
                        "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"},
                    },
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_0_kernel_time",
                        "dur": 1000,
                        "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"},
                    },
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_0_kernel_time",
                        "dur": 3000,
                        "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"},
                    },
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "sigmoid_0_kernel_time",
                        "dur": 1000,
                        "args": {"op_name": "Sigmoid", "provider": "CPUExecutionProvider"},
                    },
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "sigmoid_0_kernel_time",
                        "dur": 1000,
                        "args": {"op_name": "Sigmoid", "provider": "CPUExecutionProvider"},
                    },
            ],
            expected_runs=2,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["trace"]["node_kernel_event_count"], 4)
        self.assertEqual(summary["trace"]["session_model_run_event_count"], 2)
        self.assertEqual(summary["trace"]["ignored_non_kernel_node_event_count"], 1)
        self.assertEqual(summary["result"]["kernel_event_total_ms"], 6.0)
        self.assertEqual(summary["result"]["unique_node_count"], 2)
        operators = summary["result"]["all_operators"]
        self.assertEqual([row["op_type"] for row in operators], ["Conv", "Sigmoid"])
        self.assertEqual(operators[0]["calls"], 2)
        self.assertEqual(operators[0]["total_ms"], 4.0)
        self.assertAlmostEqual(operators[0]["percentage"], 100.0 * 4.0 / 6.0)
        self.assertAlmostEqual(
            operators[0]["cumulative_percentage"], 100.0 * 4.0 / 6.0
        )
        self.assertEqual(operators[1]["cumulative_percentage"], 100.0)
        self.assertEqual(
            summary["result"]["providers"][0]["provider"],
            "CPUExecutionProvider",
        )
        self.assertTrue(summary["profiling_overhead"]["present"])
        self.assertFalse(summary["profiling_overhead"]["quantified"])

    def test_rejects_trace_whose_node_calls_do_not_match_declared_runs(self):
        with self.assertRaises(profile.ProfileSummaryError) as context:
            self.summarize(
                [
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_kernel_time",
                        "dur": 1,
                        "args": {
                            "op_name": "Conv",
                            "provider": "CPUExecutionProvider",
                        },
                    }
                ],
                expected_runs=2,
            )
        self.assertIn("node_call_counts", str(context.exception))

    def test_rejects_trace_whose_session_run_count_does_not_match(self):
        with self.assertRaises(profile.ProfileSummaryError) as context:
            profile.summarize_events(
                [
                    {"cat": "Session", "ph": "X", "name": "model_run", "dur": 1},
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_kernel_time",
                        "dur": 1,
                        "args": {
                            "op_name": "Conv",
                            "provider": "CPUExecutionProvider",
                        },
                    },
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_kernel_time",
                        "dur": 1,
                        "args": {
                            "op_name": "Conv",
                            "provider": "CPUExecutionProvider",
                        },
                    },
                ],
                trace_path=Path("short_session_trace.json"),
                model_id="synthetic",
                declared_model_sha256="A" * 64,
                precision="fp32",
                expected_provider="CPUExecutionProvider",
                expected_profile_runs=2,
                top_n=20,
            )
        self.assertIn("session_model_run_events", str(context.exception))

    def test_top_n_bounds_output_but_all_operator_rows_remain(self):
        events = [
            {
                "cat": "Node",
                "ph": "X",
                "name": f"node_{index}_kernel_time",
                "dur": index + 1,
                "args": {
                    "op_name": f"Op{index}",
                    "provider": "CPUExecutionProvider",
                },
            }
            for index in range(3)
        ]
        summary = self.summarize(events, top_n=1)
        self.assertEqual(len(summary["result"]["top_nodes"]), 1)
        self.assertEqual(len(summary["result"]["top_operators"]), 1)
        self.assertEqual(len(summary["result"]["all_operators"]), 3)

    def test_rejects_trace_without_positive_kernel_events(self):
        with self.assertRaises(profile.ProfileSummaryError) as context:
            self.summarize([])
        self.assertIn("kernel_events", str(context.exception))

    def test_rejects_unexpected_provider_placement(self):
        with self.assertRaises(profile.ProfileSummaryError) as context:
            self.summarize(
                [
                    {
                        "cat": "Node",
                        "ph": "X",
                        "name": "conv_kernel_time",
                        "dur": 1,
                        "args": {"op_name": "Conv", "provider": "CUDAExecutionProvider"},
                    }
                ]
            )
        self.assertIn("provider_placement", str(context.exception))

    def test_rejects_fp32_trace_bound_to_int8_artifact(self):
        events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 1}
        ] + [
            {
                "cat": "Node",
                "ph": "X",
                "name": f"{op_type}_{index}_kernel_time",
                "dur": 1,
                "args": {
                    "op_name": op_type,
                    "provider": "CPUExecutionProvider",
                },
            }
            for index, op_type in enumerate(("Conv", "Sigmoid"))
        ]
        with self.assertRaises(profile.ProfileSummaryError) as context:
            profile.summarize_events(
                events,
                trace_path=Path("fp32_trace.json"),
                model_id="int8_artifact",
                declared_model_sha256="B" * 64,
                precision="int8",
                expected_provider="CPUExecutionProvider",
                expected_profile_runs=1,
                top_n=20,
                artifact_evidence={"model_id": "int8_artifact"},
            )
        self.assertIn("precision_signature", str(context.exception))

    def test_accepts_complete_int8_operator_signature(self):
        events = [
            {"cat": "Session", "ph": "X", "name": "model_run", "dur": 1}
        ] + [
            {
                "cat": "Node",
                "ph": "X",
                "name": f"{op_type}_{index}_kernel_time",
                "dur": 1,
                "args": {
                    "op_name": op_type,
                    "provider": "CPUExecutionProvider",
                },
            }
            for index, op_type in enumerate(
                ("QLinearConv", "QuantizeLinear", "DequantizeLinear")
            )
        ]
        summary = profile.summarize_events(
            events,
            trace_path=Path("int8_trace.json"),
            model_id="int8_artifact",
            declared_model_sha256="B" * 64,
            precision="int8",
            expected_provider="CPUExecutionProvider",
            expected_profile_runs=1,
            top_n=20,
            artifact_evidence={"model_id": "int8_artifact"},
        )
        self.assertTrue(summary["model"]["trace_precision_signature"]["verified"])
        self.assertEqual(
            summary["model"]["trace_precision_signature"][
                "observed_quantization_operator_types"
            ],
            ["DequantizeLinear", "QLinearConv", "QuantizeLinear"],
        )


if __name__ == "__main__":
    unittest.main()
