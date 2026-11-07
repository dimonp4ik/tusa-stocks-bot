import os
os.environ['PYTHON_DOTENV_DISABLED']='1'
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from src import autotrader as at

class UnprotectedPositionTests(unittest.TestCase):
    def test_lost_market_response_is_persisted_for_reconciliation(self):
        self._check_market_submission(False)

    def test_market_entry_expires_after_intent_before_submit(self):
        self._check_market_submission(True)

    def _check_market_submission(self, expired):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        if expired:
            sig["source"] = "stock_xperp_native_shadow"
            sig["strategy_entry_bar_ts"] = at.time.time() - 181
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('get_bot_state',None),
                ('set_bot_state',None),('set_signal_size_mult',None),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('validate_autotrade_account',(True,{'position_mode':'net_mode','permissions':('read_only','trade')})),('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('get_open_position_margin',(True,{'margin_usd':0,'count':0,'inst_ids':()})),
                ('ensure_leverage',(True,None)),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(False,'request failed: timed out')),
                ('get_order_by_client_id',(False,'not found'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(
                at.okx,'get_position_size',side_effect=[(True,0),(True,0)]))
            create=stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
            activate=stack.enter_context(patch.object(at,'at_activate_entry_intent',return_value=True))
            close = stack.enter_context(patch.object(at, "at_close_position"))
            stack.enter_context(patch.object(at, "_entry_skip"))
            place = at.okx.place_market_entry
            at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST','TEST')
        create.assert_called_once()
        activate.assert_not_called()
        if expired:
            place.assert_not_called()
            close.assert_called_once_with(91, "ENTRY_EXPIRED")
        else:
            place.assert_called_once()
            close.assert_not_called()

    def test_invisible_pending_entry_blocks_next_entry(self):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('get_bot_state',None),
                ('set_bot_state',None),('set_signal_size_mult',None),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('validate_autotrade_account',(True,{'position_mode':'net_mode','permissions':('read_only','trade')})),('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('get_open_position_margin',(True,{'margin_usd':0,'count':0,'inst_ids':()})),
                ('ensure_leverage',(True,None)),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(False,'request failed: timed out')),
                ('get_order_by_client_id',(False,'not found'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(
                at.okx,'get_position_size',side_effect=[(True,0),(True,0)]))
            stack.enter_context(patch.object(at, 'at_all_open_positions',
                return_value=[dict(user_id=7, inst_id='PENDING', margin_usd=4)]))
            stack.enter_context(patch.object(at, 'AUTOTRADE_MAX_ACCOUNT_POSITIONS', 12))
            stack.enter_context(patch.object(at, 'AUTOTRADE_ACCOUNT_WRITER_COUNT', 2))
            create=stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
            activate=stack.enter_context(patch.object(at,'at_activate_entry_intent',return_value=True))
            at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST','TEST')
        create.assert_not_called()
        activate.assert_not_called()

    def test_visible_pending_entry_blocks_next_entry(self):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('get_bot_state',None),
                ('set_bot_state',None),('set_signal_size_mult',None),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('validate_autotrade_account',(True,{'position_mode':'net_mode','permissions':('read_only','trade')})),('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('get_open_position_margin',(True,{'margin_usd':0,'count':1,'inst_ids':('PENDING',)})),
                ('ensure_leverage',(True,None)),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(False,'request failed: timed out')),
                ('get_order_by_client_id',(False,'not found'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(
                at.okx,'get_position_size',side_effect=[(True,0),(True,0)]))
            stack.enter_context(patch.object(at, 'at_all_open_positions',
                return_value=[dict(user_id=7, inst_id='PENDING', status='ENTRY_PENDING', margin_usd=4)]))
            stack.enter_context(patch.object(at, 'AUTOTRADE_MAX_ACCOUNT_POSITIONS', 12))
            stack.enter_context(patch.object(at, 'AUTOTRADE_ACCOUNT_WRITER_COUNT', 2))
            create=stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
            activate=stack.enter_context(patch.object(at,'at_activate_entry_intent',return_value=True))
            at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST','TEST')
        create.assert_not_called()
        activate.assert_not_called()

    def test_lost_market_response_recovers_order_and_places_protection(self):
        for mode, stop in ((None, 97), (False, 98), (True, 97)):
            with self.subTest(mode=mode):
                self._check_recovered_order_protection(mode, stop)

    def _check_recovered_order_protection(self, mode, stop):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        sig["strategy_stop_on_close"] = mode
        with ExitStack() as stack:
            stack.enter_context(patch.object(at, "STOP_CLOSE_CONFIRM", True))
            stack.enter_context(patch.object(at, "STOP_EXCHANGE_BACKSTOP_R", 1.5))
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('get_bot_state',None),('set_bot_state',None),
                ('set_signal_size_mult',None),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('validate_autotrade_account',(True,{'position_mode':'net_mode','permissions':('read_only','trade')})),('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('get_open_position_margin',(True,{'margin_usd':0,'count':0,'inst_ids':()})),
                ('ensure_leverage',(True,None)),('get_position_avg_px',100),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(False,'request failed: timed out')),
                ('get_order_by_client_id',(True,{'ordId':'42','state':'filled'})),
                ('place_protection_oco',(True,'algo'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(
                at.okx,'get_position_size',side_effect=[(True,0),(True,.01)]))
            stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
            update=stack.enter_context(patch.object(at,'at_update_entry_intent_protection'))
            activate=stack.enter_context(patch.object(at,'at_activate_entry_intent',return_value=True))
            protect=at.okx.place_protection_oco
            at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST','TEST')
        protect.assert_called_once()
        update.assert_called_once()
        self.assertEqual(update.call_args.args[:2], (91, 'algo'))
        self.assertEqual(update.call_args.args[2], stop)
        self.assertEqual(protect.call_args.args[3], stop)
        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[4], 'algo')

    def test_failed_protection_remains_tracked(self):
        for close_accepted in (True, False):
            with self.subTest(close_accepted=close_accepted):
                self._check_failed_protection_tracking(close_accepted)

    def _check_failed_protection_tracking(self, close_accepted):
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                ('at_set_balance',None),('get_bot_state',None),
                ('set_bot_state',None),('set_signal_size_mult',None),('at_all_open_positions',[]),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('validate_autotrade_account',(True,{'position_mode':'net_mode','permissions':('read_only','trade')})),('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                ('get_open_position_margin',(True,{'margin_usd':0,'count':0,'inst_ids':()})),
                ('ensure_leverage',(True,None)),('get_position_avg_px',100),
                ('get_order_book',{'ts':at.time.time()*1000,'bids':[[99.99,1000]],'asks':[[100.01,1000]]}),
                ('place_market_entry',(True,'order')),('place_protection_oco',(False,'protection rejected')),
                ('close_position_market',(close_accepted,'close response'))]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            stack.enter_context(patch.object(at.okx,'get_position_size',side_effect=[(True,0),(True,.01)]))
            stack.enter_context(patch.object(at,'at_create_entry_intent',return_value=91))
            activate=stack.enter_context(patch.object(at,'at_activate_entry_intent',return_value=True))
            retire=stack.enter_context(patch.object(at,'at_close_position'))
            at._open_for_user(dict(user_id=7,size_mode='percent',size_value=3),sig,'TEST','TEST')
        retire.assert_not_called()
        if close_accepted:
            activate.assert_not_called()
            return
        activate.assert_called_once()
        self.assertEqual(activate.call_args.args[4],'')
        self.assertEqual(activate.call_args.args[1],.01)

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

    def test_reconciler_closes_stale_pending_fill(self):
        pos=dict(id=2,user_id=7,inst_id='TEST',signal_id=3,
                 status='ENTRY_PENDING',opened_at=0,sl_algo_id='',
                 tp1_algo_id=None,sz=.01,client_order_id='entry123')
        with ExitStack() as stack:
            stack.enter_context(patch.object(at,'AUTOTRADE_ENABLED',True))
            stack.enter_context(patch.object(at,'at_all_open_positions',return_value=[pos]))
            stack.enter_context(patch.object(at,'at_get',return_value={'user_id':7}))
            stack.enter_context(patch.object(at,'_creds_of',return_value={'test':True}))
            stack.enter_context(patch.object(at,'_dm'))
            stack.enter_context(patch.object(at.okx,'get_position_size',side_effect=[(True,.01), (True,0), (True,0), (True,0)]))
            close=stack.enter_context(patch.object(at.okx,'close_position_market',return_value=(True,None)))
            record=stack.enter_context(patch.object(at,'at_close_position'))
            lookup = stack.enter_context(patch.object(
                at.okx, 'get_order_by_client_id',
                side_effect=[(True, {'state': 'partially_filled'}),
                             (True, {'state': 'filled'})]))
            at.poll_exchange_closes()
            close.assert_called_once()
            record.assert_not_called()
            lookup.assert_not_called()

            at.poll_exchange_closes()
            record.assert_not_called()

            at.poll_exchange_closes()
        record.assert_called_once_with(2, 'ENTRY_TERMINAL_FLAT')

    def test_reconciler_retires_stale_pending_nonfill(self):
        pos=dict(id=2,user_id=7,inst_id='TEST',signal_id=3,
                 status='ENTRY_PENDING',opened_at=0,sl_algo_id='',
                 tp1_algo_id=None,sz=.01,client_order_id='entry123')
        with ExitStack() as stack:
            stack.enter_context(patch.object(at,'AUTOTRADE_ENABLED',True))
            stack.enter_context(patch.object(at,'at_all_open_positions',return_value=[pos]))
            stack.enter_context(patch.object(at,'at_get',return_value={'user_id':7}))
            stack.enter_context(patch.object(at,'_creds_of',return_value={'test':True}))
            stack.enter_context(patch.object(at,'_dm'))
            size=stack.enter_context(patch.object(at.okx,'get_position_size',return_value=(True,0)))
            close=stack.enter_context(patch.object(at.okx,'close_position_market'))
            record=stack.enter_context(patch.object(at,'at_close_position'))
            lookup=stack.enter_context(patch.object(
                at.okx, 'get_order_by_client_id', return_value=(False, 'unknown')))
            at.poll_exchange_closes()
            record.assert_not_called()
            lookup.return_value = (True, {'state': 'canceled'})
            at.poll_exchange_closes()
        self.assertEqual(size.call_count,3)
        close.assert_not_called()
        record.assert_called_once_with(2,'ENTRY_TERMINAL_FLAT')
