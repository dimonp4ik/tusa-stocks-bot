from unittest.mock import patch
import pytest
from src import okx_trader as broker
from src.exit_summary import protective_exit_summary,format_exit_summary


def position(direction='LONG',**updates):
    result=dict(inst_id='TEST',direction=direction,entry_px=50.,entry_filled_sz=3.,sz=2.,
        client_order_id='entry1',sl_algo_id='sl',tp1_algo_id='tp1',opened_at=1.)
    result.update(updates);return result


def response(direction='LONG',*,entry_price='100',entry_size='3',exit_size='2',unrelated=True):
    entry_side='buy' if direction=='LONG' else 'sell';exit_side='sell' if direction=='LONG' else 'buy'
    def request(creds,method,path,params=None,**kwargs):
        assert method=='GET'
        if path=='/api/v5/trade/order':
            assert params==dict(instId='TEST',clOrdId='entry1')
            return True,[dict(instId='TEST',clOrdId='entry1',ordId='99',side=entry_side,
                state='filled',accFillSz=entry_size,avgPx=entry_price)]
        if path=='/api/v5/trade/orders-algo-history':
            algo=params['algoId'];kind='oco' if algo=='sl' else 'conditional'
            assert params['ordType']==kind
            return True,[dict(instId='TEST',algoId=algo,ordIdList=['22'] if algo=='sl' else ['11'])]
        assert path=='/api/v5/trade/fills'
        first=101 if direction=='LONG' else 99
        rows=[dict(instId='TEST',side=exit_side,ordId=order,ts=str(t),fillPx=str(px),fillSz=str(sz))
            for order,t,px,sz in [('11',2000,first,1),('22',3000,100,exit_size)]]
        if unrelated:rows.append(dict(instId='TEST',side=exit_side,ordId='33',ts='4000',fillPx='999',fillSz='3'))
        return True,rows
    return request


@pytest.mark.parametrize('direction',['LONG','SHORT'])
def test_exact_entry_and_all_protective_exit_fills_include_tp1_and_exclude_unrelated_orders(direction):
    with patch.object(broker,'_request',side_effect=response(direction)):
        summary=protective_exit_summary(position(direction),{},broker)
    assert summary['contracts']==3 and summary['entry_px']==100
    assert summary['price_move_pct']==pytest.approx(1/3)
    assert summary['tp1_filled']
    text=format_exit_summary(summary,lambda px:f'{px:.2f}')
    assert '+0.333%' in text and 'TP1 включён' in text and 'депозита' in text
    assert '50.00' not in text # Saved quote must not replace the confirmed entry.


@pytest.mark.parametrize('size',['1','3'])
def test_incomplete_or_excess_closing_volume_is_not_a_whole_position_result(size):
    with patch.object(broker,'_request',side_effect=response(exit_size=size)):
        assert protective_exit_summary(position(),{},broker) is None


def test_order_entry_quantity_conflicting_with_initial_snapshot_is_not_reconciled():
    with patch.object(broker,'_request',side_effect=response(entry_size='4')):
        assert protective_exit_summary(position(),{},broker) is None


def test_confirmed_exit_without_confirmed_entry_price_does_not_report_a_return():
    with patch.object(broker,'_request',side_effect=response(entry_price='nan')):
        summary=protective_exit_summary(position(),{},broker)
    assert summary['exit_px'] is not None and summary['price_move_pct'] is None
    assert 'доходность не рассчитана' in format_exit_summary(summary,str)


def test_actual_entry_order_can_recover_missing_original_quantity_without_using_remainder():
    with patch.object(broker,'_request',side_effect=response()):
        summary=protective_exit_summary(position(entry_filled_sz=None),{},broker)
    assert summary['contracts']==3


def test_read_only_history_request_requires_one_documented_order_type():
    with patch.object(broker,'_request',return_value=(True,[dict(instId='TEST',algoId='x',ordIdList=['11'])])) as request:
        assert broker.get_protection_order_ids({},'TEST','x',ord_type='conditional')==('11',)
        assert request.call_args.kwargs['params']['ordType']=='conditional'
        request.reset_mock()
        assert broker.get_protection_order_ids({},'TEST','x',ord_type='conditional,oco') is None
        request.assert_not_called()


def test_indivisible_position_fully_closed_at_tp1_needs_no_runner_fill():
    original=response(entry_size='1')
    def request(creds,method,path,params=None,**kwargs):
        ok,rows=original(creds,method,path,params,**kwargs)
        if path.endswith('orders-algo-history') and params['algoId']=='sl':rows[0]['ordIdList']=[]
        if path.endswith('/fills'):rows=[r for r in rows if r['ordId']=='11']
        return ok,rows
    with patch.object(broker,'_request',side_effect=request):
        summary=protective_exit_summary(position(entry_filled_sz=1,sz=1),{},broker)
    assert summary['contracts']==1 and summary['tp1_filled'] and summary['exit_px']==101


def test_untriggered_tp1_is_not_described_as_a_banked_partial_profit():
    original=response(exit_size='3')
    def request(creds,method,path,params=None,**kwargs):
        ok,rows=original(creds,method,path,params,**kwargs)
        if path.endswith('orders-algo-history') and params['algoId']=='tp1':rows[0]['ordIdList']=[]
        if path.endswith('/fills'):rows=[r for r in rows if r['ordId']=='22']
        return ok,rows
    with patch.object(broker,'_request',side_effect=request):
        summary=protective_exit_summary(position(),{},broker)
    assert summary['price_move_pct']==0 and not summary['tp1_filled']
    assert 'TP1 включён' not in format_exit_summary(summary,str)
