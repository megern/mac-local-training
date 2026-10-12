from pathlib import Path
import tempfile,json,unittest
from dataclasses import replace
from mac_local_training.engine import TrainingConfig,TrainingEngine
from mac_local_training.tiled_head import TiledHeadTrainingEngine
try:
 import mlx.core as mx
except ImportError:mx=None
@unittest.skipIf(mx is None,'Local Apple Silicon required')
class TiledHeadTests(unittest.TestCase):
 def test_sparse_long_context_matches_full_updates_and_moments(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);data=root/'data.jsonl';data.write_text(json.dumps({'tokens':[1]+[5]*63+[2],'loss_start':64})+'\n')
   cfg=TrainingConfig(train_path=str(data),valid_path=str(data),output_path=str(root/'reference'),tiny_model=True,strategy='all',steps=3,max_seq_length=64)
   a=TrainingEngine(cfg);a.run();b=TiledHeadTrainingEngine(replace(cfg,strategy='rotating',output_path=str(root/'tiled')));b.run()
   for name,value in a.adapter_arrays().items():self.assertLess(float(mx.max(mx.abs(value-b.adapter_arrays()[name])).item()),1e-5)
   expected=dict(a.flatten(a.optimizer.state))
   for path in b.state_paths.values():
    for name,value in mx.load(str(path)).items():self.assertLess(float(mx.max(mx.abs(value-expected[name])).item()),1e-5)
 def test_dense_supervision_uses_unchanged_loss(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);data=root/'data.jsonl';data.write_text(json.dumps({'tokens':[1]+[5]*31+[2],'loss_start':1})+'\n')
   e=TiledHeadTrainingEngine(TrainingConfig(train_path=str(data),valid_path=str(data),output_path=str(root/'run'),tiny_model=True,max_seq_length=32))
   batch=e._batch(e.train_data)
   self.assertEqual(float(e._loss_sum(e.model,*batch).item()),float(TrainingEngine._loss_sum(e,e.model,*batch[:3]).item()))
