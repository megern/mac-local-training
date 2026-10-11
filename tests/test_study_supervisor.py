import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'studies'))
from supervise_optimizer_study import retryable, SIGNATURE


class RetryPolicyTests(unittest.TestCase):
    def test_numerical_and_source_errors_are_not_retried(self):
        for error in ['Non-finite gradients', 'Frozen source changed', 'fixed equivalence check failed', 'ValueError']:
            self.assertFalse(retryable(1, error, 1, 1))
        self.assertFalse(retryable(0, SIGNATURE, 1, 1))

    def test_only_one_device_retry_is_allowed(self):
        self.assertTrue(retryable(1, SIGNATURE, 1, 1))
        self.assertFalse(retryable(1, SIGNATURE, 2, 1))
        self.assertFalse(retryable(1, SIGNATURE, 1, 0))
