import unittest
from src.execution_guard import execution_quote

class ExecutionGuardTests(unittest.TestCase):
    def book(self):return {'ts':10000,'bids':[[99.99,2],[99.98,3]],'asks':[[100.01,2],[100.02,3]]}

    def test_direction_depth_and_weighted_fill(self):
        q=execution_quote(self.book(),'LONG',4,11000)
        self.assertAlmostEqual(q.average,100.015)
        self.assertEqual(q.worst,100.02)
        q=execution_quote(self.book(),'SHORT',4,11000)
        self.assertAlmostEqual(q.average,99.985)
        self.assertEqual(q.worst,99.98)

    def test_stale_future_and_insufficient_depth(self):
        for now in [14000,8000]:
            with self.assertRaises(ValueError):execution_quote(self.book(),'LONG',1,now)
        with self.assertRaises(ValueError):execution_quote(self.book(),'LONG',6,11000)

    def test_reject_wide_crossed_and_malformed_book(self):
        for asks in [[[101,10]],[[99.98,10]],[[float('nan'),10]],[[100.02,1],[100.01,1]],[[100.01,0]]]:
            b=self.book();b['asks']=asks
            with self.assertRaises(ValueError):execution_quote(b,'LONG',1,11000)

    def test_depth_slippage_rejected_even_with_tight_top_spread(self):
        b=self.book();b['asks']=[[100.01,1],[101,10]]
        with self.assertRaises(ValueError):execution_quote(b,'LONG',2,11000)
