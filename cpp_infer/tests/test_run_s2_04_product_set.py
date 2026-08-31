"""Focused tests for the S2-04 single-image product-set runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock


CPP_INFER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPP_INFER_ROOT / "tools"))

import run_s2_04_product_set as product_set  # noqa: E402


TRT_PROVIDER = "TensorrtExecutionProvider"


class ProductSetRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        # Python's TemporaryDirectory applies owner-only permissions that a
        # managed Windows sandbox token may be unable to traverse.  A normal
        # mkdir under the platform temp root keeps the test cross-platform.
        self.root = Path(tempfile.gettempdir()) / (
            f"yolo_defect_s2_04_product_set_{uuid.uuid4().hex}"
        )
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.cli = self.root / "yolo_defect_cpp"
        self.cli.write_bytes(b"synthetic executable")
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"synthetic frozen ONNX bytes")
        self.model_sha256 = hashlib.sha256(self.model.read_bytes()).hexdigest().upper()
        self.artifact = self.root / "artifact.txt"
        self.artifact.write_text(
            "schema_version = 1\n"
            "model_path = model.onnx\n"
            f"model_sha256 = {self.model_sha256}\n",
            encoding="utf-8",
        )
        self.config = self.root / "runtime.txt"
        self.config.write_text(
            "schema_version = 2\n"
            "artifact_spec_path = artifact.txt\n"
            "provider = tensorrt\n"
            "device_id = 0\n"
            "precision = fp16\n",
            encoding="utf-8",
        )
        self.images = self.root / "images"
        self.images.mkdir()
        self.manifest = self.root / "consistency_manifest.json"
        self.samples = []
        for index in range(product_set.FROZEN_SAMPLE_COUNT):
            sample_id = f"sample_{index:02d}"
            image_path = self.images / f"{sample_id}.jpg"
            image_path.write_bytes(f"image bytes {index}".encode("ascii"))
            self.samples.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{image_path.name}",
                    "image_sha256": hashlib.sha256(image_path.read_bytes())
                    .hexdigest()
                    .upper(),
                }
            )
        self.write_manifest()

    def write_manifest(self, manifest_id=None) -> None:
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_id": manifest_id or product_set.FROZEN_MANIFEST_ID,
                    "samples": self.samples,
                }
            ),
            encoding="utf-8",
        )

    def emit_result(
        self,
        command,
        provider=TRT_PROVIDER,
        model_sha=None,
        result_image_path=None,
    ):
        if model_sha is None:
            model_sha = self.model_sha256
        image_path = Path(command[command.index("--image") + 1])
        output_path = Path(command[command.index("--output-json") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "model": {"declared_sha256": model_sha},
                    "image": {
                        "path": str(
                            image_path
                            if result_image_path is None
                            else result_image_path
                        )
                    },
                    "runtime": {"actual_provider": provider},
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def test_runs_each_sample_once_and_accepts_bound_results(self):
        output_directory = self.root / "run_a"
        with mock.patch.object(
            product_set.subprocess,
            "run",
            side_effect=lambda command, **unused: self.emit_result(command),
        ) as run:
            written = product_set.run_product_set(
                self.cli,
                self.config,
                self.manifest,
                output_directory,
                self.model_sha256.lower(),
                TRT_PROVIDER,
            )

        self.assertEqual(len(written), product_set.FROZEN_SAMPLE_COUNT)
        self.assertEqual(run.call_count, product_set.FROZEN_SAMPLE_COUNT)
        self.assertEqual(
            [path.name for path in written],
            [f"sample_{index:02d}.detections.json" for index in range(30)],
        )
        for call in run.call_args_list:
            self.assertNotIn("env", call.kwargs)
            self.assertEqual(call.args[0][0], str(self.cli.resolve()))

    def test_all_image_hashes_are_preflighted_before_output_creation(self):
        self.samples[-1]["image_sha256"] = "0" * 64
        self.write_manifest()
        output_directory = self.root / "must_not_exist"
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("manifest.samples[29].image_sha256", str(context.exception))
        self.assertFalse(output_directory.exists())
        run.assert_not_called()

    def test_existing_output_directory_is_rejected(self):
        output_directory = self.root / "existing"
        output_directory.mkdir()
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("choose a fresh run directory", str(context.exception))
        run.assert_not_called()

    def test_wrong_actual_provider_is_rejected_on_first_result(self):
        output_directory = self.root / "provider_mismatch"
        with mock.patch.object(
            product_set.subprocess,
            "run",
            side_effect=lambda command, **unused: self.emit_result(
                command, provider="CPUExecutionProvider"
            ),
        ) as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("runtime.actual_provider", str(context.exception))
        self.assertEqual(run.call_count, 1)

    def test_wrong_model_is_rejected(self):
        output_directory = self.root / "model_mismatch"
        with mock.patch.object(
            product_set.subprocess,
            "run",
            side_effect=lambda command, **unused: self.emit_result(
                command, model_sha="F" * 64
            ),
        ):
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("model.declared_sha256", str(context.exception))

    def test_same_basename_with_wrong_image_bytes_is_rejected(self):
        wrong_directory = self.root / "wrong_images"
        wrong_directory.mkdir()
        wrong_image = wrong_directory / "sample_00.jpg"
        wrong_image.write_bytes(b"different image bytes")
        output_directory = self.root / "image_mismatch"
        with mock.patch.object(
            product_set.subprocess,
            "run",
            side_effect=lambda command, **unused: self.emit_result(
                command, result_image_path=wrong_image
            ),
        ):
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("image SHA-256", str(context.exception))

    def test_tampered_model_bytes_are_rejected_before_output_creation(self):
        self.model.write_bytes(b"tampered model bytes")
        output_directory = self.root / "tampered_model"
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("model_path SHA-256", str(context.exception))
        self.assertFalse(output_directory.exists())
        run.assert_not_called()

    def test_config_provider_must_match_expected_product_provider(self):
        self.config.write_text(
            "schema_version = 2\n"
            "artifact_spec_path = artifact.txt\n"
            "provider = cpu\n"
            "device_id = 0\n"
            "precision = fp16\n",
            encoding="utf-8",
        )
        output_directory = self.root / "provider_preflight"
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("runtime_config.provider", str(context.exception))
        self.assertFalse(output_directory.exists())
        run.assert_not_called()

    def test_tensorrt_precision_must_be_fp16(self):
        self.config.write_text(
            "schema_version = 2\n"
            "artifact_spec_path = artifact.txt\n"
            "provider = tensorrt\n"
            "device_id = 0\n"
            "precision = fp32\n",
            encoding="utf-8",
        )
        output_directory = self.root / "precision_preflight"
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    TRT_PROVIDER,
                )
        self.assertIn("runtime_config.precision", str(context.exception))
        self.assertFalse(output_directory.exists())
        run.assert_not_called()

    def configure_native(self):
        cache = self.root / "native_cache"
        cache.mkdir()
        engine = cache / "frozen.engine"
        engine.write_bytes(b"synthetic frozen TensorRT engine")
        engine_sha = hashlib.sha256(engine.read_bytes()).hexdigest().upper()
        self.config.write_text(
            "schema_version = 2\n"
            "artifact_spec_path = artifact.txt\n"
            "provider = tensorrt_native\n"
            "device_id = 0\n"
            "precision = fp16\n"
            "tensorrt_engine_cache_path = native_cache\n"
            "tensorrt_engine_path = native_cache/frozen.engine\n"
            f"tensorrt_engine_sha256 = {engine_sha}\n",
            encoding="utf-8",
        )
        self.write_manifest("neu_det_val_s2_04_native_holdout_6x5_v3")
        return engine, engine_sha

    def test_native_v3_happy_path_binds_engine_and_provider_evidence(self):
        _, engine_sha = self.configure_native()
        output_directory = self.root / "native_run"
        expected_evidence = (
            "native_tensorrt_enqueue_v3;"
            "precision_policy=fp16_dfl_softmax_fp32_else_no_tf32;"
            f"declared_engine_sha256={engine_sha};"
            f"actual_engine_sha256={engine_sha};"
            "tensorrt_runtime=10.4.0;compiled_headers=10.4.0.26;"
            "cuda_runtime=12.6;compute_capability=8.9;fallback=none"
        )

        def emit_native(command, **unused):
            completed = self.emit_result(command, provider="TensorRTNative")
            output_path = Path(command[command.index("--output-json") + 1])
            document = json.loads(output_path.read_text(encoding="utf-8"))
            document["runtime"]["provider_evidence"] = expected_evidence
            output_path.write_text(json.dumps(document), encoding="utf-8")
            return completed

        with mock.patch.object(
            product_set.subprocess, "run", side_effect=emit_native
        ) as run:
            written = product_set.run_product_set(
                self.cli,
                self.config,
                self.manifest,
                output_directory,
                self.model_sha256,
                "TensorRTNative",
                engine_sha,
            )
        self.assertEqual(len(written), product_set.FROZEN_SAMPLE_COUNT)
        self.assertEqual(run.call_count, product_set.FROZEN_SAMPLE_COUNT)

    def test_native_tampered_engine_is_rejected_before_inference(self):
        engine, engine_sha = self.configure_native()
        engine.write_bytes(b"tampered TensorRT engine")
        output_directory = self.root / "native_tampered"
        with mock.patch.object(product_set.subprocess, "run") as run:
            with self.assertRaises(product_set.ProductSetError) as context:
                product_set.run_product_set(
                    self.cli,
                    self.config,
                    self.manifest,
                    output_directory,
                    self.model_sha256,
                    "TensorRTNative",
                    engine_sha,
                )
        self.assertIn("tensorrt_engine_path SHA-256", str(context.exception))
        self.assertFalse(output_directory.exists())
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
