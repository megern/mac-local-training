import tempfile
from pathlib import Path
import unittest
from mac_local_training.report_bundle import write_bundle, audit_bundle

class BundleTests(unittest.TestCase):

    def test_html_escapes_untrusted_report_and_hashes_detect_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'review'
            write_bundle(root, {'evidence': '<script>alert(1)</script>'}, 'Untrusted <title>')
            self.assertNotIn('<script>', (root / 'report.html').read_text())
            self.assertEqual(audit_bundle(root)['status'], 'consistent')
            (root / 'report.json').write_text('{}')
            with self.assertRaises(ValueError):
                audit_bundle(root)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'review'
            root.mkdir()
            sentinel = root / 'keep'
            sentinel.write_text('keep')
            with self.assertRaises(ValueError):
                write_bundle(root, {}, 'Report')
            self.assertEqual(sentinel.read_text(), 'keep')
