import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from src import autotrader as at

class UnprotectedPositionTests(unittest.TestCase):
    def test_failed_protection_and_close_remains_tracked(self):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('_check_threshold_cross',None),('get_bot_state',None),
                ('set_bot_state',None),('set_signal_size_mult',None),('at_all_open_positions',[]),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('ensure_leverage',(True,None)),('get_position_avg_px',100),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(True,'order')),('place_protection_oco',(False,'protection rejected')),
                ('close_position_market',(False,'still open'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(at.okx,'get_position_size',side_effect=[(True,0),(True,.01)]))
            record=stack.enter_context(patch.object(at,'at_log_position',return_value=1))
            at._open_for_user(dict(user_id=7,size_mode='fixed',size_value=10),sig,'TEST','TEST')
        record.assert_called_once()
        self.assertEqual(record.call_args.args[7],'')
        self.assertEqual(record.call_args.args[4],.01)

    def test_reconciler_closes_unprotected_position_even_with_open_signal(self):
        pos=dict(id=1,user_id=7,inst_id='TEST',signal_id=2,sl_algo_id='',sz=.01)
        with ExitStack() as stack:
            stack.enter_context(patch.object(at,'AUTOTRADE_ENABLED',True))
            stack.enter_context(patch.object(at,'at_all_open_positions',return_value=[pos]))
            stack.enter_context(patch.object(at,'at_get',return_value={'user_id':7}))
            stack.enter_context(patch.object(at,'_creds_of',return_value={'test':True}))
            stack.enter_context(patch.object(at,'_dm'))
            stack.enter_context(patch.object(at.okx,'get_position_size',return_value=(True,.01)))
            close=stack.enter_context(patch.object(at.okx,'close_position_market',return_value=(True,None)))
            record=stack.enter_context(patch.object(at,'at_close_position'))
            cancel=stack.enter_context(patch.object(at.okx,'cancel_protection'))
            at.poll_exchange_closes()
        close.assert_called_once()
        record.assert_called_once_with(1,'UNPROTECTED_CLOSED')
        cancel.assert_not_called()
