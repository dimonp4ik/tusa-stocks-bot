import copy
import math
import unittest
from entry_hypotheses import signal_at, FAMILIES

class EntryCausalityTests(unittest.TestCase):
    def test_future_candles_cannot_change_entry_features(self):
        close=[100+.02*j+math.sin(j/3) for j in range(180)]
        close[119]=110
        bars={'close':close,'open':close[:], 'high':[x+.1 for x in close],
              'low':[x-.1 for x in close], 'time':[900*j for j in range(180)]}
        observed=[]
        for family in FAMILIES+('sweep_reclaim','exhaustion_reversal'):
            before=signal_at(bars,120,family)
            changed=copy.deepcopy(bars)
            for key in ('open','high','low','close'):
                changed[key][120:]=[999999]*60
            self.assertEqual(before,signal_at(changed,120,family))
            prefix={key:value[:120] for key,value in bars.items()}
            self.assertEqual(before,signal_at(prefix,120,family))
            if before:observed.append(before)
        self.assertTrue(observed, 'Fixture must actually produce a signal')

    def test_no_signal_with_insufficient_history(self):
        for family in FAMILIES+('sweep_reclaim','exhaustion_reversal'):self.assertIsNone(signal_at({},99,family))


class EntryCostTests(unittest.TestCase):
    def test_cost_ceiling_in_price_units(self):
        from entry_hypotheses import cost_allows
        self.assertTrue(cost_allows(100,2,.25,.1,roundtrip_cost=.0004))
        self.assertFalse(cost_allows(100,.2,.25,.1))
        self.assertTrue(cost_allows(100,1,.5,.1,roundtrip_cost=.0004))

    def test_taker_default_rejects_target_allowed_by_old_fee(self):
        from entry_hypotheses import cost_allows
        self.assertFalse(cost_allows(100,2,.25,.1))
        self.assertTrue(cost_allows(100,6,.25,.1))
        for cost in (-.001, float('nan'), float('inf')):
            self.assertFalse(cost_allows(100,6,.25,None,cost))

    def test_invalid_cost_inputs_rejected(self):
        from entry_hypotheses import cost_allows
        for risk in (0,-1,float('nan'),float('inf')):
            self.assertFalse(cost_allows(100,risk,.25,.1))


class ReversalEntryTests(unittest.TestCase):
    def fixture(self):
        close=[100+.2*math.sin(j) for j in range(130)]
        return {'close':close,'open':close[:],'high':[x+.5 for x in close],
                'low':[x-.5 for x in close]}

    def test_sweep_requires_closed_reclaim(self):
        bars=self.fixture()
        bars['low'][119]=94;bars['high'][119]=100.6
        bars['open'][119]=95;bars['close'][119]=100.5
        self.assertEqual(signal_at(bars,120,'sweep_reclaim')[0],1)
        bars['close'][119]=94.5
        self.assertIsNone(signal_at(bars,120,'sweep_reclaim'))

    def test_exhaustion_requires_reversal_not_continuing_fall(self):
        bars=self.fixture()
        bars['close'][116:120]=[100,99,95,96]
        self.assertEqual(signal_at(bars,120,'exhaustion_reversal')[0],1)
        bars['close'][119]=94
        self.assertIsNone(signal_at(bars,120,'exhaustion_reversal'))
