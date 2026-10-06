import unittest
from mac_local_training.planner import estimate


class PlannerTests(unittest.TestCase):
    def test_large_base_rejected_even_with_adapter_rotation(self):
        result = estimate(70, 4, 6)
        self.assertEqual(result["decision"], "reject_weights_exceed_budget")
        self.assertAlmostEqual(result["raw_weight_lower_bound_gib"], 35e9 / 2**30)

    def test_candidate_is_not_a_fit_guarantee(self):
        self.assertEqual(estimate(7, 4, 6)["decision"], "candidate_requires_measurement")

    def test_nonfinite_and_invalid_inputs_rejected(self):
        for params, bits, budget in [(float("nan"), 4, 6), (7, 3, 6), (7, 4, -1)]:
            with self.assertRaises(ValueError):
                estimate(params, bits, budget)
