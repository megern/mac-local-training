import unittest
from unittest.mock import patch
from mac_local_training.doctor import diagnose
class DoctorTests(unittest.TestCase):
    def test_planning_without_apple_or_modules(self):
        with patch('mac_local_training.doctor.platform.system',return_value='Linux'),patch('mac_local_training.doctor.find_spec',return_value=None):
            report=diagnose()
            self.assertTrue(report['planning_available'])
            self.assertFalse(report['training_candidate'])
    def test_apple_requires_all_modules(self):
        with patch('mac_local_training.doctor.platform.system',return_value='Darwin'),patch('mac_local_training.doctor.platform.machine',return_value='arm64'),patch('mac_local_training.doctor.find_spec',return_value=object()):
            self.assertTrue(diagnose()['training_candidate'])
