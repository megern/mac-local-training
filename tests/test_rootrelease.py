from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from mac_local_training.endpoint import endpoint_order, cold_switch_counts
from mac_local_training.rootrelease import RootReleaseTrainingEngine
from mac_local_training.engine import TrainingConfig, TrainingEngine
from mac_local_training.offloaded import OffloadedTrainingEngine
try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'Local Apple Silicon required')
class RootReleaseTests(unittest.TestCase):
    def assert_arrays(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertLess(float(mx.max(mx.abs(left[name] - right[name])).item()), 2e-6, name)

    def test_equal_adapters_and_moments_resume_at_both_endpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            data.write_text(''.join(json.dumps(row) + '\n' for row in [
                {'tokens': [1, 5, 8, 2], 'loss_start': 3},
                {'tokens': [1, 6, 4, 9, 2], 'loss_start': 1}]))
            cfg = TrainingConfig(train_path=str(data), valid_path=str(data), output_path=str(root / 'all'),
                                 tiny_model=True, strategy='all', steps=5, accumulation_steps=3)
            reference = TrainingEngine(cfg)
            reference.run()
            expected = {k: mx.array(v) for k, v in reference.adapter_arrays().items()}
            mx.eval(expected)
            candidate_cfg = replace(cfg, strategy='rotating', output_path=str(root / 'candidate'))
            candidate = RootReleaseTrainingEngine(candidate_cfg)
            base = {k: mx.array(v) for k, v in candidate.base_arrays().items()}
            mx.eval(base)
            report = candidate.run()
            self.assert_arrays(expected, candidate.adapter_arrays())
            self.assert_arrays(base, candidate.base_arrays())
            self.assertEqual(candidate.coverage, [5] * 4)
            reference_moments = dict(reference.flatten(reference.optimizer.state))
            for group, path in candidate.state_paths.items():
                stored = mx.load(str(path))
                self.assert_arrays(stored, {k: reference_moments[k] for k in stored})
            for split in (2, 3):
                partial = replace(candidate_cfg, output_path=str(root / f'resume-{split}'), steps=split)
                RootReleaseTrainingEngine(partial).run()
                resumed = RootReleaseTrainingEngine(replace(partial, steps=5, resume=True))
                resumed.run()
                self.assert_arrays(expected, resumed.adapter_arrays())
                with self.assertRaisesRegex(ValueError, 'Resume config'):
                    OffloadedTrainingEngine(replace(partial, steps=5, resume=True))

    def test_checkpoint_survives_failed_partial_apply(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data.jsonl'
            data.write_text('{"tokens":[1,5,8,2],"loss_start":3}\n')
            cfg = TrainingConfig(train_path=str(data), valid_path=str(data), output_path=str(root / 'run'),
                                 tiny_model=True, steps=2, accumulation_steps=1)
            engine = RootReleaseTrainingEngine(cfg)
            engine.run()
            manifest = (root / 'run/latest.json').read_bytes()
            extended = replace(cfg, steps=3, resume=True)
            failing = RootReleaseTrainingEngine(extended)
            original = failing.activate_group
            calls = []
            def fail_second(group):
                calls.append(group)
                if len(calls) == 2:
                    raise OSError('injected state-load failure')
                return original(group)
            failing.activate_group = fail_second
            with self.assertRaisesRegex(OSError, 'injected'):
                failing.update_once()
            self.assertEqual(manifest, (root / 'run/latest.json').read_bytes())
            resumed = RootReleaseTrainingEngine(extended)
            resumed.run()
            continuous = RootReleaseTrainingEngine(replace(cfg, steps=3, output_path=str(root / 'continuous')))
            continuous.run()
            self.assert_arrays(continuous.adapter_arrays(), resumed.adapter_arrays())
