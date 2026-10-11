import unittest
from mac_local_training.planner import estimate_adapter_state
class AdapterPlanTests(unittest.TestCase):
    def test_components(self):
        full=estimate_adapter_state(100); quarter=estimate_adapter_state(100,.25)
        self.assertEqual(full["modeled_total_bytes"],1600)
        self.assertEqual(quarter["components"]["resident_adapter_bytes"],400)
        self.assertEqual(quarter["components"]["active_adam_moments_bytes"],200)
        self.assertEqual(quarter["modeled_total_bytes"],700)
    def test_invalid(self):
        for count,fraction in [(True,1),(0,1),(2,0),(2,float("nan")),(2,True)]:
            with self.assertRaises(ValueError): estimate_adapter_state(count,fraction)
