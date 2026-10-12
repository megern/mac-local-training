import json,hashlib,tempfile,unittest
from pathlib import Path
from mac_local_training.study_evidence import audit_export
class StudyExportTests(unittest.TestCase):
 def build(self,root):
  proto='a'*64
  design={'recorded_protocol_sha256':proto,'runs':[{'id':'base'},{'id':'candidate'}]}
  rows=[{'run':{'id':'base','rank':4,'seed':7,'method':'endpoint'},'elapsed_seconds':2,'mlx_measured_peak_bytes':100,'local_complete_result_sha256':'b'*64}, {'run':{'id':'candidate','rank':4,'seed':7,'method':'selected'},'elapsed_seconds':1,'mlx_measured_peak_bytes':80,'local_complete_result_sha256':'c'*64}]
  analysis={'protocol_sha256':proto,'runs_audited':2,'source_result_sha256':{'base':'b'*64,'candidate':'c'*64},'all_numerical_equivalence_gates_passed':True,'primary_joint_gate_passed':True,'novelty_status':'fictional software fixture','comparisons':[{'rank':4,'paired_seed_blocks':[{'seed':7,'peak_selected_over_endpoint':.8,'elapsed_selected_over_endpoint':.5}]}]}
  for name,value in [('protocol-design.json',design),('run-measurements.json',rows),('analysis.json',analysis)]: (root/name).write_text(json.dumps(value))
  self.receipt(root,proto)
 def receipt(self,root,proto='a'*64):
  files=['protocol-design.json','run-measurements.json','analysis.json']
  (root/'export-receipt.json').write_text(json.dumps({'original_protocol_sha256':proto,'exported_files_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}))
 def test_consistency_and_resigned_ratio_contradiction(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);self.build(root);self.assertEqual(audit_export(root)['completed_runs'],2)
   data=json.loads((root/'run-measurements.json').read_text());data[1]['elapsed_seconds']=3;(root/'run-measurements.json').write_text(json.dumps(data));self.receipt(root)
   with self.assertRaises(ValueError):audit_export(root)
 def test_positive_gate_cannot_contradict_numerical_failure(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);self.build(root);data=json.loads((root/'analysis.json').read_text());data['all_numerical_equivalence_gates_passed']=False;(root/'analysis.json').write_text(json.dumps(data));self.receipt(root)
   with self.assertRaises(ValueError):audit_export(root)
