"""Behavioral tests using random tiny models and local token fixtures."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from mac_local_training.engine import MAX_DATASET_BYTES, TinyTokenizer, TrainingConfig, TrainingEngine, _read_data


@unittest.skipIf(mx is None, "MLX requires the local Apple Silicon environment")
class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Unequal lengths catch averaging microbatch means instead of token sums.
        self.rows = [[1, 5, 8, 13, 2], [1, 7, 2], [1, 9, 12, 11, 8, 2], [1, 4, 3, 2]]
        for name in ("train", "valid"):
            (self.root / f"{name}.jsonl").write_text("".join(
                json.dumps({"tokens": row}) + "\n" for row in self.rows))

    def tearDown(self):
        self.temp.cleanup()

    def config(self, output, **kwargs):
        return TrainingConfig(train_path=str(self.root / "train.jsonl"),
                              valid_path=str(self.root / "valid.jsonl"),
                              output_path=str(self.root / output), tiny_model=True,
                              max_seq_length=16, **kwargs)

    def copy_arrays(self, arrays):
        mx.eval(arrays)
        return {k: mx.array(v) for k, v in arrays.items()}

    def assert_arrays_close(self, a, b, tolerance=2e-6):
        self.assertEqual(set(a), set(b))
        for key in a:
            error = float(mx.max(mx.abs(a[key] - b[key])).item())
            self.assertLessEqual(error, tolerance, key)

    def test_token_weighted_accumulation_matches_large_batch(self):
        accumulated = TrainingEngine(self.config("acc", strategy="all", steps=1,
                                                accumulation_steps=2, microbatch_size=1))
        accumulated.update_once()
        actual = self.copy_arrays(accumulated.adapter_arrays())
        full = TrainingEngine(self.config("full", strategy="all", steps=1,
                                         accumulation_steps=1, microbatch_size=2))
        full.update_once()
        self.assert_arrays_close(actual, full.adapter_arrays())
        self.assertAlmostEqual(accumulated.history[0]["loss"], full.history[0]["loss"], places=5)

    def test_completion_loss_excludes_prompt_and_padding(self):
        data = [{"tokens": [1, 5, 8, 13, 2], "loss_start": 4},
                {"tokens": [1, 7, 2], "loss_start": 1}]
        for name in ("train", "valid"):
            (self.root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in data))
        engine = TrainingEngine(self.config("completion", strategy="all", steps=1))
        inputs, targets, mask = engine._batch(engine.train_data)
        self.assertEqual(mask.tolist(), [[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
        logits = engine.model(inputs).astype(mx.float32)
        losses = engine.nn.losses.cross_entropy(logits, targets, reduction="none")
        expected = losses[0, 3] + losses[1, 0] + losses[1, 1]
        self.assertAlmostEqual(engine._loss_sum(engine.model, inputs, targets, mask).item(), expected.item(), places=5)
        engine.update_once()
        self.assertEqual(engine.history[0]["tokens"], 3)

    def test_completion_rejects_missing_targets_and_context_truncation(self):
        path = self.root / "masked.jsonl"
        for start in (0, 3, True, "1"):
            path.write_text(json.dumps({"tokens": [1, 5, 2], "loss_start": start}) + "\n")
            with self.assertRaisesRegex(ValueError, "loss_start"):
                _read_data(str(path), TinyTokenizer(), 16, TinyTokenizer.vocab_size)
        path.write_text(json.dumps({"tokens": [1, 5, 8, 2], "loss_start": 3}) + "\n")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _read_data(str(path), TinyTokenizer(), 2, TinyTokenizer.vocab_size)

    def test_masked_accumulation_matches_large_batch(self):
        rows = [{"tokens": row, "loss_start": len(row) - 1 if i % 2 else 1}
                for i, row in enumerate(self.rows)]
        for name in ("train", "valid"):
            (self.root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        acc = TrainingEngine(self.config("masked-acc", strategy="all", steps=1,
                                        accumulation_steps=2, microbatch_size=1))
        acc.update_once()
        expected = self.copy_arrays(acc.adapter_arrays())
        full = TrainingEngine(self.config("masked-full", strategy="all", steps=1,
                                         accumulation_steps=1, microbatch_size=2))
        full.update_once()
        self.assert_arrays_close(expected, full.adapter_arrays())

    def test_masked_resume_preserves_target_objective(self):
        for name in ("train", "valid"):
            (self.root / f"{name}.jsonl").write_text("".join(
                json.dumps({"tokens": row, "loss_start": len(row) - 1}) + "\n" for row in self.rows))
        cfg = self.config("masked-continuous", steps=5, accumulation_steps=2)
        continuous = TrainingEngine(cfg)
        continuous.run()
        expected = self.copy_arrays(continuous.adapter_arrays())
        partial_cfg = replace(cfg, output_path=str(self.root / "masked-resumed"), steps=2)
        TrainingEngine(partial_cfg).run()
        resumed = TrainingEngine(replace(partial_cfg, steps=5, resume=True))
        report = resumed.run()
        self.assert_arrays_close(expected, resumed.adapter_arrays(), tolerance=1e-7)
        self.assertEqual(report["tokens"], 10)
        self.assertEqual(report["loss_targets"]["train_completion_rows"], 4)

    def test_rotating_freezes_base_and_moves_only_active_group(self):
        engine = TrainingEngine(self.config("groups", steps=4, accumulation_steps=1))
        base = self.copy_arrays(engine.base_arrays())
        before = self.copy_arrays(engine.adapter_arrays())
        engine.update_once()
        after = engine.adapter_arrays()
        changed = {k for k in before if not bool(mx.array_equal(before[k], after[k]).item())}
        self.assertTrue(changed)
        self.assertTrue(all("layers.0." in k for k in changed), changed)
        first_resident = engine.history[-1]["optimizer_resident_bytes"]
        engine.update_once()
        self.assertEqual(engine.active_group, 1)
        self.assertIn(0, engine.state_paths)
        self.assertTrue(engine.state_paths[0].is_file())
        state_names = [k for k, _ in engine.flatten(engine.optimizer.state)]
        self.assertFalse(any("layers.0." in k for k in state_names), state_names)
        self.assertEqual(engine.history[-1]["optimizer_resident_bytes"], first_resident)
        self.assertEqual(engine.coverage, [1, 1, 0, 0])
        self.assert_arrays_close(base, engine.base_arrays(), tolerance=0)

    def test_checkpoint_resume_matches_uninterrupted_across_group_reentry(self):
        config = self.config("continuous", steps=6, accumulation_steps=2)
        continuous = TrainingEngine(config)
        continuous.run()
        expected = self.copy_arrays(continuous.adapter_arrays())
        partial_config = replace(config, output_path=str(self.root / "resumed"), steps=3)
        partial = TrainingEngine(partial_config)
        partial.run()
        resumed = TrainingEngine(replace(partial_config, steps=6, resume=True))
        report = resumed.run()
        self.assert_arrays_close(expected, resumed.adapter_arrays(), tolerance=1e-7)
        self.assertEqual(continuous.coverage, resumed.coverage)
        self.assertEqual(continuous.cursor, resumed.cursor)
        self.assertEqual(report["microbatches"], 12)
        self.assertAlmostEqual(continuous.evaluate(), resumed.evaluate(), places=7)

    def test_changed_dataset_refuses_resume(self):
        config = self.config("changed", steps=1, accumulation_steps=1)
        TrainingEngine(config).run()
        with Path(config.train_path).open("a") as handle:
            handle.write(json.dumps({"tokens": [1, 5, 2]}) + "\n")
        with self.assertRaisesRegex(ValueError, "datasets"):
            TrainingEngine(replace(config, steps=2, resume=True))

    def test_remote_model_id_refused_before_loading(self):
        config = replace(self.config("remote"), tiny_model=False, model_path="vendor/model")
        with self.assertRaisesRegex(ValueError, "remote IDs"):
            config.validate()

    def test_corrupted_checkpoint_refuses_resume(self):
        config = self.config("corrupt", steps=1, accumulation_steps=1)
        TrainingEngine(config).run()
        manifest = json.loads((Path(config.output_path) / "latest.json").read_text())
        adapter_file = Path(config.output_path) / manifest["checkpoint"] / "adapters.safetensors"
        with adapter_file.open("ab") as handle:
            handle.write(b"invalid")
        with self.assertRaisesRegex(ValueError, "checksum"):
            TrainingEngine(replace(config, steps=2, resume=True))

    def test_causal_partition_preserves_every_next_token_target_once(self):
        original = [1, 4, 6, 8, 10, 12, 14, 16, 2]
        path = self.root / "long.jsonl"
        path.write_text(json.dumps({"tokens": original}) + "\n")
        windows = _read_data(str(path), TinyTokenizer(), 3, TinyTokenizer.vocab_size)
        self.assertEqual(windows, [[1, 4, 6, 8], [8, 10, 12, 14], [14, 16, 2]])
        self.assertEqual([target for window in windows for target in window[1:]], original[1:])
        self.assertEqual(sum(len(window) - 1 for window in windows), len(original) - 1)

    def test_memory_budget_rejects_weight_files_before_loading(self):
        model = self.root / "model"
        model.mkdir()
        # A sparse local test file exercises preflight without allocating GPU arrays.
        with (model / "model.safetensors").open("wb") as handle:
            handle.truncate(2048)
        config = replace(self.config("budget"), tiny_model=False, model_path=str(model),
                         memory_budget_gib=1024 / 1024 ** 3)
        with self.assertRaisesRegex(ValueError, "weights alone"):
            config.validate()

    def test_large_in_memory_dataset_is_refused(self):
        path = self.root / "large.jsonl"
        with path.open("wb") as handle:
            handle.truncate(MAX_DATASET_BYTES)
        config = replace(self.config("large"), train_path=str(path))
        with self.assertRaisesRegex(ValueError, "64 MiB"):
            config.validate()

    def test_nonfinite_memory_budget_is_refused(self):
        for budget in (float("nan"), float("inf"), 0, -1):
            with self.subTest(budget=budget):
                with self.assertRaisesRegex(ValueError, "memory_budget_gib"):
                    replace(self.config("nonfinite"), memory_budget_gib=budget).validate()


if __name__ == "__main__":
    unittest.main()
