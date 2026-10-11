from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import mlx.core as mx
except ImportError:
    mx = None

from mac_local_training.engine import TrainingConfig, TrainingEngine
from mac_local_training.synchronous import SynchronousTrainingEngine


@unittest.skipIf(mx is None, 'Local Apple Silicon required')
class SynchronousTests(unittest.TestCase):
    def compare(self, left, right):
        mx.eval(left, right)
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertLess(float(mx.max(mx.abs(left[name] - right[name])).item()), 2e-6, name)

    def test_matches_all_adapter_adam_with_unequal_masked_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            rows = [{'tokens': [1, 5, 8, 2], 'loss_start': 3},
                    {'tokens': [1, 7, 9, 4, 2], 'loss_start': 1},
                    {'tokens': [1, 6, 2], 'loss_start': 2}]
            data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            cfg = TrainingConfig(train_path=str(data), valid_path=str(data), output_path=str(root / 'all'),
                                 tiny_model=True, steps=5, strategy='all', accumulation_steps=2)
            all_engine = TrainingEngine(cfg)
            all_report = all_engine.run()
            expected = {name: mx.array(value) for name, value in all_engine.adapter_arrays().items()}
            mx.eval(expected)
            sync = SynchronousTrainingEngine(replace(cfg, strategy='rotating', output_path=str(root / 'sync')))
            base = {name: mx.array(value) for name, value in sync.base_arrays().items()}
            mx.eval(base)
            report = sync.run()
            self.compare(expected, sync.adapter_arrays())
            self.compare(base, sync.base_arrays())
            self.assertEqual(sync.coverage, [5] * 4)
            self.assertEqual(report['tokens'], all_report['tokens'])
            self.assertLess(report['peak_optimizer_resident_bytes'], all_report['peak_optimizer_resident_bytes'])
            self.assertAlmostEqual(report['heldout_nll_after'], all_report['heldout_nll_after'], places=5)

    def test_resume_matches_continuous_and_rejects_rotating_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            data.write_text('{"tokens":[1,5,8,2],"loss_start":3}\n')
            cfg = TrainingConfig(train_path=str(data), valid_path=str(data), output_path=str(root / 'sync'),
                                 tiny_model=True, steps=5, accumulation_steps=1)
            continuous = SynchronousTrainingEngine(cfg)
            continuous.run()
            expected = {name: mx.array(value) for name, value in continuous.adapter_arrays().items()}
            mx.eval(expected)
            partial = replace(cfg, output_path=str(root / 'resume'), steps=2)
            SynchronousTrainingEngine(partial).run()
            resumed = SynchronousTrainingEngine(replace(partial, steps=5, resume=True))
            resumed.run()
            self.compare(expected, resumed.adapter_arrays())
            with self.assertRaisesRegex(ValueError, 'Resume config'):
                TrainingEngine(replace(partial, steps=5, resume=True))
