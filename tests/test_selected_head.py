import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from mac_local_training.engine import TrainingConfig, TrainingEngine
from mac_local_training.selected_head import SelectedHeadTrainingEngine
try:
    import mlx.core as mx
except ImportError:
    mx = None


@unittest.skipIf(mx is None, 'Apple Silicon required')
class SelectedHeadTests(unittest.TestCase):
    def test_unequal_completion_masks_and_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data = root / 'data.jsonl'
            data.write_text(''.join(json.dumps(row)+'\n' for row in [
                {'tokens':[1,5,8,2],'loss_start':3}, {'tokens':[1,6,4,9,2],'loss_start':1}]))
            cfg = TrainingConfig(train_path=str(data),valid_path=str(data),output_path=str(root/'all'),tiny_model=True,
                                 strategy='all',steps=3,microbatch_size=2,accumulation_steps=2)
            reference = TrainingEngine(cfg); a = reference.run()
            candidate = SelectedHeadTrainingEngine(replace(cfg,strategy='rotating',output_path=str(root/'selected')))
            frozen = {k:mx.array(v) for k,v in candidate.base_arrays().items()}; mx.eval(frozen)
            b = candidate.run()
            self.assertEqual(a['tokens'],b['tokens'])
            self.assertLess(abs(a['heldout_nll_after']-b['heldout_nll_after']),1e-5)
            for name,value in reference.adapter_arrays().items():
                self.assertLess(float(mx.max(mx.abs(value-candidate.adapter_arrays()[name])).item()),1e-5)
            for name,value in frozen.items(): self.assertTrue(bool(mx.all(value==candidate.base_arrays()[name]).item()))
            resumed=SelectedHeadTrainingEngine(replace(cfg,strategy='rotating',output_path=str(root/'selected'),steps=4,resume=True))
            resumed.run(); self.assertEqual(resumed.step,4)
