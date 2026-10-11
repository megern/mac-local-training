"""Replication integrity: exact inputs, new provenance and no overwrite."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

STUDIES = Path(__file__).resolve().parents[1] / 'studies'
sys.path.insert(0, str(STUDIES))
spec = importlib.util.spec_from_file_location('replication_preparation', STUDIES / 'prepare_replication.py')
replication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replication)


class ReplicationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        (self.package / 'studies').mkdir(parents=True)
        (self.package / 'studies/worker.py').write_text('original source\n')
        shutil.copyfile(STUDIES / 'prepare_replication.py', self.package / 'studies/prepare_replication.py')
        self.model = self.root / 'model'; self.model.mkdir()
        (self.model / 'weights.safetensors').write_bytes(b'local test bytes')
        self.data = self.root / 'data'; self.data.mkdir()
        self.design = {'recorded_protocol_sha256': 'a' * 64,
                       'export_note': 'paths sanitized', 'model_path': '<local>',
                       'model_files': {'weights.safetensors': replication.digest(self.model / 'weights.safetensors')},
                       'sources_sha256': {'studies/worker.py': replication.digest(self.package / 'studies/worker.py')},
                       'runs': [{'id': 'one'}]}
        for split in ('train', 'valid'):
            path = self.data / f'{split}.jsonl'
            path.write_text(json.dumps({'id': split, 'tokens': [1, 2], 'loss_start': 1}) + '\n')
            self.design[split] = {'path': '<local>', 'sha256': replication.digest(path), 'selected_ids': [split]}
        self.design_path = self.root / 'design.json'
        self.design_path.write_text(json.dumps(self.design))
        self.output = self.root / 'new-study'

    def prepare(self):
        with patch.object(replication, 'PACKAGE', self.package):
            return replication.prepare(self.design_path, self.model, self.data, self.output)

    def test_exact_reconstruction_creates_distinct_protocol_and_source_snapshot(self):
        result = self.prepare()
        protocol = json.loads((self.output / 'protocol.json').read_text())
        self.assertEqual(protocol['replication_of_protocol_sha256'], 'a' * 64)
        self.assertNotEqual(result['protocol_sha256'], 'a' * 64)
        self.assertEqual(protocol['train']['path'], str((self.data / 'train.jsonl').resolve()))
        self.assertEqual((self.output / 'source/studies/worker.py').read_text(), 'original source\n')
        self.assertIn('studies/prepare_replication.py', protocol['sources_sha256'])

    def test_changed_inputs_stop_before_output_creation(self):
        paths = [self.model / 'weights.safetensors', self.data / 'train.jsonl', self.package / 'studies/worker.py']
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaises(ValueError): self.prepare()
                self.assertFalse(self.output.exists())
                path.write_bytes(original)

    def test_existing_study_is_never_replaced(self):
        self.output.mkdir()
        sentinel = self.output / 'original'; sentinel.write_text('keep')
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(sentinel.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
