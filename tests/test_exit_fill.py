import unittest
from unittest.mock import patch
from src.okx_trader import get_last_fill_px

class ExitFillTests(unittest.TestCase):
    def row(self,ts,px,sz,order='exit',side='sell'):
        return dict(instId='TEST',ts=str(ts),fillPx=str(px),fillSz=str(sz),ordId=order,side=side)
    def test_exit_order_vwap_excludes_entry_and_old_orders(self):
        rows=[self.row(3000,100,1),self.row(3001,110,3),self.row(3002,900,1,'entry','buy'),self.row(500,800,1,'old')]
        with patch('src.okx_trader._request',return_value=(True,rows)):
            self.assertEqual(get_last_fill_px({},'TEST',since_ts=1,side='sell'),107.5)
    def test_truncated_or_invalid_page_does_not_fabricate_price(self):
        for rows in ([self.row(3000,100,1)]*100,[self.row(3000,float('nan'),1)]):
            with patch('src.okx_trader._request',return_value=(True,rows)):
                self.assertIsNone(get_last_fill_px({},'TEST',since_ts=1,side='sell'))
