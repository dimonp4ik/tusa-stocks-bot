import math
import unittest
from pair_reversion_lab import net_return,replay

class PairResearchTests(unittest.TestCase):
    def test_two_legs_charge_entry_and_exit_notional(self):
        self.assertAlmostEqual(net_return(100,200,100,200,1),-.0012)
        # Half capital long A gains 5%, half short B gains 5%.
        self.assertAlmostEqual(net_return(100,200,110,180,1),.1-.0012)
        self.assertAlmostEqual(net_return(100,200,110,180,-1),-.1-.0012)

    def test_swapping_legs_and_direction_preserves_return(self):
        self.assertAlmostEqual(net_return(100,200,123,170,1),net_return(200,100,170,123,-1))

    def test_later_history_does_not_change_completed_trades(self):
        n=800
        ratio=[math.exp(.01*math.sin(j/17)+(.025 if j%100==80 else 0)) for j in range(n)]
        a={'time':[j*3600 for j in range(n)],'open':[100*x for x in ratio],
           'close':[100*x for x in ratio]}
        b={'time':a['time'][:],'open':[100]*n,'close':[100]*n}
        prefix=lambda c:{k:v[:600] for k,v in c.items()}
        earlier=replay(prefix(a),prefix(b),2)
        self.assertTrue(earlier)
        full=replay(a,b,2)
        self.assertEqual(earlier,[r for r in full if r['entry_time']<=earlier[-1]['entry_time']])
