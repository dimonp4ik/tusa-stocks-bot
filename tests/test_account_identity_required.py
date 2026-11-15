import os
os.environ['PYTHON_DOTENV_DISABLED']='1'

from unittest.mock import patch
import pytest
from src import okx_trader as okx
from src import autotrader as at


@pytest.mark.parametrize('uid',[None,'','abc','１２３'])
def test_missing_or_malformed_uid_cannot_authorize_live_trading(uid):
    raw=dict(posMode='net_mode',perm='read_only,trade',mainUid='999')
    if uid is not None:
        raw['uid']=uid
    with patch.object(okx,'_request',return_value=(True,[raw])):
        allowed,detail=okx.validate_autotrade_account({'test':True})
    assert allowed is False
    assert detail['reason']=='account_identity_missing'


def test_subaccount_identity_is_stable_across_api_keys():
    with patch.object(okx,'_request',return_value=(True,[
            dict(posMode='net_mode',perm='read_only,trade',uid='123',mainUid='999')])):
        allowed,a=okx.validate_autotrade_account({'api_key':'first'})
        allowed_b,b=okx.validate_autotrade_account({'api_key':'second'})
    assert allowed and allowed_b
    assert okx.account_key(a)==okx.account_key(b)
    assert okx.account_key(a)!=okx.account_key({'account_uid':'999'})


def test_orchestrator_rejects_unidentified_account_even_if_validator_says_yes():
    sig=dict(id=7,symbol='TEST',direction='LONG',entry_price=100,sl=99,tp1=101,tp2=102)
    with (patch.object(at,'_creds_of',return_value={'test':True}),
          patch.object(at,'at_has_open_position',return_value=False),
          patch.object(at.okx,'validate_autotrade_account',return_value=(True,{
              'position_mode':'net_mode','permissions':('read_only','trade')})),
          patch.object(at,'_entry_skip') as skipped,
          patch.object(at,'_open_for_account_locked') as entry):
        at._open_for_user(dict(user_id=1,size_mode='percent',size_value=3),sig,'TEST','TEST')
    skipped.assert_called_once()
    assert skipped.call_args.args[3]=='OKX account identity unavailable'
    entry.assert_not_called()
