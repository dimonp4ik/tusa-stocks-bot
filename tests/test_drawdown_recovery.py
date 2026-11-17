import copy
import json
from datetime import datetime,timezone
from unittest.mock import patch

import pytest
from src.drawdown_recovery import checked_equity_guard,failed_equity_check
from src import autotrader as at,db

NOW=datetime(2026,11,17,12,tzinfo=timezone.utc).timestamp()


def original():
    return dict(peak=128.42,day='2026-11-17',daily_start=125.52,paused=True,daily_paused=False)


def check(state,seconds=0,equity=125.52):
    return checked_equity_guard(state,equity=equity,now=NOW+seconds,
        max_daily_loss=.01,max_drawdown=.03,peak_floor=125.52)


def test_three_spaced_fresh_checks_recover_without_lowering_original_peak():
    state=original();baseline=copy.deepcopy(state)
    for seconds in (0,60):
        state,reason=check(state,seconds)
        assert state['paused'] and reason=='drawdown limit'
        state=json.loads(json.dumps(state))  # restart preserves evidence.
    state,reason=check(state,120)
    assert state['paused'] is False and reason is None
    assert state['peak']==128.42 and state['recovery_checks']==3
    assert state['last_auto_recovery_at']==NOW+120
    assert baseline==original()


def test_fast_repeated_reads_and_stale_old_evidence_cannot_recover():
    state=original()
    for seconds in (0,.1,.2,60):state,_=check(state,seconds)
    assert state['paused'] and state['recovery_checks']==2
    state,_=check(state,211)
    assert state['paused'] and state['recovery_checks']==1


def test_new_dip_or_failed_read_resets_the_fresh_check_sequence():
    state,_=check(original())
    state,_=check(state,60,124.6)  # above 3% cutoff, below recovery buffer.
    assert state['paused'] and state['recovery_checks']==0
    state,_=check(state,120)
    assert state['recovery_checks']==1
    state=failed_equity_check(state,NOW+121)
    assert state['paused'] and state['recovery_checks']==0
    assert state['peak']==128.42 and 'equity' not in state


def test_equity_remaining_below_cutoff_never_rebases_the_peak():
    state=original()
    for i in range(10):state,reason=check(state,i*60,124.)
    assert state['paused'] and state['peak']==128.42 and reason=='drawdown limit'


def test_daily_pause_is_preserved_and_only_utc_day_rollover_resets_it():
    state=dict(original(),daily_paused=True)
    for i in range(3):state,_=check(state,i*60)
    assert state['paused'] and state['recovery_status']=='daily_pause'
    for seconds in (86400,86460,86520):state,reason=check(state,seconds)
    assert state['paused'] is False and not state['daily_paused'] and reason is None


def test_recovery_does_not_disable_the_next_real_loss_limit():
    state=original()
    for seconds in (0,60,120):state,_=check(state,seconds)
    state,reason=check(state,180,124.)
    assert state['paused'] and reason=='drawdown limit' and state['recovery_checks']==0


def test_compare_and_set_does_not_overwrite_a_newer_state(tmp_path):
    with patch.object(db,'DB_PATH',str(tmp_path/'state.db')):
        db.init_db();db.set_bot_state('risk_guard:7','old')
        db.set_bot_state('risk_guard:7','newer')
        assert not db.compare_and_set_bot_state('risk_guard:7','old','cleared')
        assert db.get_bot_state('risk_guard:7')=='newer'


def test_monitor_only_reads_current_active_client_and_never_submits_orders():
    box=[json.dumps(original())];clock=[NOW]
    user=dict(user_id=7,allowed=1,active=1,last_balance=125.52)
    def cas(key,expected,value):
        assert key=='risk_guard:7' and expected==box[0]
        box[0]=value;return True
    with (patch.object(at,'AUTOTRADE_ENABLED',True),
          patch.object(at,'at_get_active_traders',return_value=[user,dict(user_id=8)]),
          patch.object(at,'at_get',side_effect=lambda uid:user if uid==7 else dict(user_id=8,allowed=1,active=0)),
          patch.object(at,'get_bot_state',side_effect=lambda key:box[0]),
          patch.object(at,'compare_and_set_bot_state',side_effect=cas),
          patch.object(at,'_creds_of',return_value={'test':True}) as creds,
          patch.object(at.okx,'get_balance',return_value=(True,125.52)) as balance,
          patch.object(at.okx,'place_market_entry') as order,
          patch.object(at.time,'time',side_effect=lambda:clock[0])):
        for seconds in (0,60,120):clock[0]=NOW+seconds;at.maintain_equity_guards()
    assert not json.loads(box[0])['paused']
    assert balance.call_count==3 and creds.call_count==3
    assert all(c.args[0]['user_id']==7 for c in creds.call_args_list)
    order.assert_not_called()
