import unittest
from relative_strength_lab import closed_returns,allow

class RelativeStrengthTests(unittest.TestCase):
    def test_exact_closed_points_and_no_future_dependency(self):
        d={'closed_lookup':{0:100,72000:110,86400:120,87300:999}}
        self.assertEqual(closed_returns(d,86400),(120/110-1,120/100-1))
        d['closed_lookup'][87300]=-999
        self.assertEqual(closed_returns(d,86400),(120/110-1,120/100-1))

    def test_missing_previous_day_is_not_forward_filled(self):
        self.assertIsNone(closed_returns({'closed_lookup':{1:100,72000:110,86400:120}},86400))

    def test_direction_and_no_lookahead_fallback(self):
        features=(.04,.06,.01,.02)
        self.assertTrue(allow('relative_4h',{'direction':'LONG'},features))
        self.assertFalse(allow('relative_4h',{'direction':'SHORT'},features))
        self.assertFalse(allow('relative_4h',{'direction':'LONG'},None))
        self.assertTrue(allow('baseline',{'direction':'LONG'},None))
