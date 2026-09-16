import ast
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

os.environ['PYTHON_DOTENV_DISABLED'] = '1'
from src.backtest_integrity import closed_snapshot, simulate_exit
from src.live_accounting import trailing_observation, marked_r
from src.risk_limits import bounded_margin, equity_guard, position_risk
from src import okx_trader as okx


def candles(rows):
    return {k:[row[i] for row in rows] for i,k in enumerate(('open','high','low','close'))}


class CandleTests(unittest.TestCase):
    def test_no_future_htf_values(self):
        data={'time':[0,3600,7200],'close':[1,2,999]}
        self.assertEqual(closed_snapshot(data,8100,10,9,3600)['close'],[1,2])

    def test_boundary_and_before_first_close(self):
        d={'time':[0,3600],'close':[1,2]}
        self.assertEqual(closed_snapshot(d,3599,10,9,3600)['close'],[])
        self.assertEqual(closed_snapshot(d,3600,10,9,3600)['close'],[1])

    def test_daily_alignment_and_weekend_gaps(self):
        d={'time':[0,3*86400],'close':[1,999]}
        self.assertEqual(closed_snapshot(d,3*86400+1000,10,9,86400)['close'],[1])

    def run_exit(self, rows, **kw):
        params=dict(direction='LONG',entry=100,sl=90,tp1=106,tp2=120,atr=1,
                    tp1_fraction=0.,trail=True,trail_mult=1.,stop_on_close=False,
                    backstop_r=1.5,choose_trail=lambda *a:1.)
        params.update(kw)
        return simulate_exit(candles(rows),range(len(rows)),**params)

    def test_gap_stop_loses_more_than_one_r(self):
        r=self.run_exit([(85,88,80,86)])
        self.assertEqual(r.price,85)
        self.assertAlmostEqual(r.gross_r,-1.5)

    def test_stop_precedes_target_on_ambiguous_bar(self):
        r=self.run_exit([(100,125,85,110)])
        self.assertEqual(r.outcome,'SL')
        self.assertAlmostEqual(r.gross_r,-1)

    def test_trail_cannot_use_this_bar_high(self):
        r=self.run_exit([(103,107,101,106),(110,119,108,115)])
        self.assertEqual(r.outcome,'EXPIRED')
        self.assertEqual(r.price,115)

    def test_backstop_fires_even_if_close_recovers(self):
        r=self.run_exit([(100,105,84,102)],stop_on_close=True)
        self.assertEqual(r.price,85)
        self.assertEqual(r.outcome,'SL')

    def test_expiry_keeps_loss_instead_of_zero(self):
        r=self.run_exit([(100,102,94,95)])
        self.assertEqual(r.outcome,'EXPIRED')
        self.assertAlmostEqual(r.gross_r,-0.5)

    def test_partial_take_profit_and_gap_loss(self):
        r=self.run_exit([(103,107,101,106),(95,97,90,93)],tp1_fraction=0.5)
        self.assertAlmostEqual(r.gross_r,0.05)

    def test_full_tp1_banks_only_tp1(self):
        r=self.run_exit([(103,107,101,106)],tp1_fraction=1.)
        self.assertAlmostEqual(r.gross_r,0.6)

    def test_watched_entry_cannot_claim_pre_entry_high(self):
        r=self.run_exit([(105,125,99,102)],intrabar_entry=True)
        self.assertEqual(r.outcome,'EXPIRED')
        self.assertAlmostEqual(r.gross_r,0.2)

    def test_short_gap_stop(self):
        r=self.run_exit([(115,118,112,114)],direction='SHORT',sl=110,tp1=94,tp2=80)
        self.assertAlmostEqual(r.gross_r,-1.5)


class LiveAccountingTests(unittest.TestCase):
    def test_net_win_excludes_fee_losing_trail(self):
        from src.db import _row_net_r
        row=dict(status='TP1_TRAIL',realized_r=0.01,entry_price=100,sl=99,size_mult=1)
        self.assertLess(_row_net_r(row),0)

    def test_old_low_does_not_trigger_new_trail(self):
        stop,crossed=trailing_observation({'high':[112],'low':[99]},direction='LONG',
                                         entry=100,atr=2,multiple=1,quote=111)
        self.assertEqual(stop,110)
        self.assertFalse(crossed)

    def test_quote_loss_is_not_floored_to_zero(self):
        r=marked_r(direction='LONG',entry=100,exit_price=98,sl=90,tp1=106,reached_tp1=True)
        self.assertAlmostEqual(r,-0.2)

    def test_monitor_uses_quote_instead_of_old_extrema(self):
        source=Path('main.py').read_text(encoding='utf-8')
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_check_open_signals')
        frame={'time':[900],'open':[105],'high':[112],'low':[99],'close':[111],'confirmed':[False]}
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=90,tp1=106,tp2=130,
                 atr=2,opened_at=1,tp1_hit_at=800,status='TP1_PARTIAL',runner_trail_atr_mult=1)
        trader=types.SimpleNamespace(update_trailing=Mock(),mirror_transition=Mock())
        ns=dict(get_open_signals=lambda:[sig],time=types.SimpleNamespace(time=lambda:1000),
            get_klines_xperp=lambda *a,**kw:frame,get_klines=lambda *a,**kw:frame,
            _last_prices={},_slice_candles_from_open=lambda *a:frame,
            TP1_CLOSE_FRAC=0,TRAIL_RUNNER_ENABLED=True,TRAIL_ATR_MULT=1,
            STOP_CLOSE_CONFIRM=True,SIGNAL_EXPIRY_HOURS=48,SIGNAL_EXPIRY_MAX_DAYS=5,
            session_hours_between=lambda *a:1,autotrader=trader,log=Mock(),
            update_signal_status=Mock(),send_signal_update=Mock())
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'monitor_test','exec'),ns)
        ns['_check_open_signals']()
        ns['log'].warning.assert_not_called()
        ns['update_signal_status'].assert_not_called()
        trader.update_trailing.assert_called_once_with(sig,110)
        frame['high']=[135]
        frame['close']=[134]
        ns['_check_open_signals']()
        self.assertEqual(ns['update_signal_status'].call_args.args[1:3],('TP2_HIT',130))


class RiskTests(unittest.TestCase):
    def margin(self,**kw):
        args=dict(requested=100,equity=128,leverage=10,entry=100,stop=97,
                  direction='LONG',open_risk=0,trade_fraction=.0025,
                  portfolio_fraction=.0075,cost_fraction=.0014)
        args.update(kw)
        return bounded_margin(**args)

    def test_requested_margin_cannot_exceed_money_risk(self):
        self.assertAlmostEqual(self.margin()*10*(.03+.0014),.32)

    def test_open_risk_exhausted_means_no_new_position(self):
        self.assertEqual(self.margin(open_risk=1.),0)

    def test_invalid_data_refuses_entry(self):
        for kw in [dict(entry=float('nan')),dict(stop=101),dict(equity=0),dict(open_risk=-1)]:
            with self.assertRaises(ValueError):self.margin(**kw)

    def test_drawdown_pause_survives_recovery_and_restart(self):
        s,_=equity_guard({},equity=100,day='1',max_daily_loss=.01,max_drawdown=.03)
        s,r=equity_guard(s,equity=96,day='1',max_daily_loss=.01,max_drawdown=.03)
        self.assertEqual(r,'drawdown limit')
        _,r=equity_guard(s,equity=101,day='2',max_daily_loss=.01,max_drawdown=.03)
        self.assertEqual(r,'drawdown limit')

    def test_daily_pause_survives_recovery_until_next_day(self):
        s,_=equity_guard({},equity=100,day='1',max_daily_loss=.01,max_drawdown=.03)
        s,r=equity_guard(s,equity=98.5,day='1',max_daily_loss=.01,max_drawdown=.03)
        self.assertEqual(r,'daily loss limit')
        s,r=equity_guard(s,equity=100,day='1',max_daily_loss=.01,max_drawdown=.03)
        self.assertEqual(r,'daily loss limit')
        _,r=equity_guard(s,equity=100,day='2',max_daily_loss=.01,max_drawdown=.03)
        self.assertIsNone(r)

    def test_known_peak_floor_catches_predeployment_drawdown(self):
        state,reason=equity_guard({},equity=128,day='1',max_daily_loss=.01,
                                  max_drawdown=.03,peak_floor=153)
        self.assertEqual(state['peak'],153)
        self.assertEqual(reason,'drawdown limit')

    def test_peak_floor_cannot_lower_observed_peak(self):
        state,_=equity_guard({'peak':150},equity=149,day='1',max_daily_loss=.01,
                              max_drawdown=.03,peak_floor=140)
        self.assertEqual(state['peak'],150)


class ExchangeTests(unittest.TestCase):
    def test_shadow_signal_cannot_reach_real_traders(self):
        from src import autotrader as at
        sig = dict(id=17, symbol='TEST', direction='LONG', autotrade_eligible=0)
        with patch.object(at, 'AUTOTRADE_ENABLED', True), \
             patch.object(at, 'at_get_active_traders') as traders:
            at.open_positions_for_signal(sig)
        traders.assert_not_called()

    def test_missing_protection_id_is_failure(self):
        with patch.object(okx,'_request',return_value=(True,[{'algoId':''}])):
            self.assertFalse(okx.place_protection_oco({},'TEST','LONG',90,120)[0])

    def test_live_entry_uses_client_margin_without_strategy_resizing(self):
        from src import autotrader as at
        from contextlib import ExitStack
        spec=dict(ctVal=1,lotSz=.001,minSz=.001,tickSz=.01)
        sig=dict(id=1,symbol='TEST',direction='LONG',entry_price=100,sl=98,tp1=102,tp2=110)
        with ExitStack() as stack:
            for name,value in [('_creds_of',{'test':True}),('at_has_open_position',False),
                 ('at_set_balance',None),('_check_threshold_cross',None),('get_bot_state',None),
                 ('set_bot_state',None),('set_signal_size_mult',None),('at_all_open_positions',[]),('_dm',None)]:
                stack.enter_context(patch.object(at,name,return_value=value))
            for name,value in [('get_balance',(True,128)),('get_xperp_spec',spec),('get_last_price',100),
                 ('get_position_size',(True,0)),('ensure_leverage',(True,None)),
                 ('get_order_book',{'ts':at.time.time()*1000,'bids':[['99.99','1000']],
                                    'asks':[['100.01','1000']]})]:
                stack.enter_context(patch.object(at.okx,name,return_value=value))
            risk_cap=stack.enter_context(patch.object(at,'bounded_margin'))
            order=stack.enter_context(patch.object(at.okx,'place_market_entry',return_value=(False,'test refusal')))
            at._open_for_user(dict(user_id=7,size_mode='fixed',size_value=10),sig,'TEST','TEST')
            order.assert_called_once()
            for bad_book in (None, {'ts':0,'bids':[['99.99','1000']],'asks':[['100.01','1000']]},
                             {'ts':at.time.time()*1000,'bids':[['99','1000']],'asks':[['101','1000']]}):
                with patch.object(at.okx,'get_order_book',return_value=bad_book):
                    at._open_for_user(dict(user_id=7,size_mode='fixed',size_value=10),sig,'TEST','TEST')
                order.assert_called_once()  # No additional market entry on refused depth.
        order.assert_called_once()
        quantity=order.call_args.args[-1]
        self.assertEqual(quantity, okx.calc_contracts(10, at.AUTOTRADE_LEVERAGE, 100, spec))
        risk_cap.assert_not_called()

    def test_entry_order_is_market(self):
        with patch.object(okx, '_request', return_value=(True, [{'ordId': '42'}])) as request:
            self.assertEqual(okx.place_market_entry({}, 'TEST', 'LONG', 1), (True, '42'))
        self.assertEqual(request.call_args.kwargs['body']['ordType'], 'market')

    def test_order_item_error_is_failure_even_with_top_code_zero(self):
        response=Mock()
        response.json.return_value={'code':'0','data':[{'sCode':'51000','sMsg':'invalid order'}]}
        with patch.object(okx.requests,'post',return_value=response):
            success,_=okx._request(dict(api_key='test',api_secret='test',passphrase='test'),'POST','/test')
        self.assertFalse(success)

    def test_close_error_containing_position_is_not_success(self):
        with patch.object(okx,'_request',return_value=(False,'position close rejected')),patch.object(okx,'get_position_size',return_value=(True,2)):
            self.assertFalse(okx.close_position_market({},'TEST')[0])

    def test_accepted_but_unfilled_close_is_not_success(self):
        with patch.object(okx,'_request',return_value=(True,[])),patch.object(okx,'get_position_size',return_value=(True,2)):
            self.assertFalse(okx.close_position_market({},'TEST')[0])

    def test_only_confirmed_flat_is_success(self):
        with patch.object(okx,'_request',return_value=(True,[])),patch.object(okx,'get_position_size',return_value=(True,0)):
            self.assertTrue(okx.close_position_market({},'TEST')[0])

    def test_position_response_counts_all_rows(self):
        with patch.object(okx,'_request',return_value=(True,[{'pos':'0'},{'pos':'-2'}])):
            self.assertEqual(okx.get_position_size({},'TEST'),(True,2))

    def test_malformed_position_is_not_flat(self):
        with patch.object(okx,'_request',return_value=(True,[{}])):
            self.assertFalse(okx.get_position_size({},'TEST')[0])

    def test_failed_close_keeps_protection_and_tracking(self):
        from src import autotrader as at
        position=dict(id=1,user_id=7,inst_id='TEST',sl_algo_id='stop',tp1_algo_id=None)
        with patch.object(at,'AUTOTRADE_ENABLED',True),patch.object(at,'at_open_positions_for_signal',return_value=[position]),patch.object(at,'at_get',return_value={'user_id':7}),patch.object(at,'_creds_of',return_value={'test':True}),patch.object(at.okx,'close_position_market',return_value=(False,'pending')),patch.object(at.okx,'cancel_protection') as cancel,patch.object(at,'at_close_position') as closed,patch.object(at,'_dm'):
            at.mirror_transition(dict(id=5,symbol='TEST'),'SL_HIT',99)
        cancel.assert_not_called()
        closed.assert_not_called()


class DatabaseTests(unittest.TestCase):
    def test_shadow_signal_persists_as_ineligible_for_autotrade(self):
        from src import db
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'signals.db')
            with patch.object(db, 'DB_PATH', path):
                db.init_db()
                base = dict(symbol='TEST', direction='LONG', current_price=100)
                live_id = db.log_signal(base, 106, 120, 90)
                shadow_id = db.log_signal({**base, '_shadow_only': True}, 106, 120, 90)
                self.assertEqual(db.get_signal_by_id(live_id)['autotrade_eligible'], 1)
                self.assertEqual(db.get_signal_by_id(shadow_id)['autotrade_eligible'], 0)


if __name__ == '__main__':
    unittest.main()


class ReplayParameterTests(unittest.TestCase):
    def test_requested_trail_distance_changes_actual_exit(self):
        import backtest as bt
        bars={'open':[100,101,108,109], 'high':[101,108,110,110],
              'low':[99,101,107,109], 'close':[100,107,109,109],
              'time':[0,900,1800,2700], 'volume':[1]*4}
        setup={'direction':'LONG','current_price':100,'atr':1}
        with patch.object(bt,'calculate_tp_sl_local',return_value=(106,120,90)), \
             patch.object(bt,'EXIT_PROFILE','constant_research'), \
             patch.object(bt,'_TP1_CLOSE_FRAC',0):
            tight=bt.simulate_trade_direct('TEST',setup,bars,0,4,0,0,
                        exit_policy='trail',trail_atr_mult=.1)
            wide=bt.simulate_trade_direct('TEST',setup,bars,0,4,0,0,
                        exit_policy='trail',trail_atr_mult=2)
        self.assertNotEqual(tight.exit_time,wide.exit_time)
        self.assertNotEqual(tight.gross_r,wide.gross_r)
