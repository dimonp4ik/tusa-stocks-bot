import sqlite3
from unittest.mock import patch
import pytest

import main
from src import db,autotrader

SOURCE='stock_xperp_native_shadow'


@pytest.fixture
def database(tmp_path):
    path=str(tmp_path/'forward.db')
    with patch.object(db,'DB_PATH',path):
        db.init_db()
        yield path


def insert(path,rr,stop=99.,closed=1.,direction='LONG'):
    with sqlite3.connect(path) as c:
        c.execute('''INSERT INTO signals
            (symbol,direction,entry_price,tp1,tp2,sl,opened_at,status,closed_at,source,realized_r)
            VALUES ('TEST',?,100,102,102,?,0,'TP2_HIT',?,?,?)''',
            (direction,stop,closed,SOURCE,rr))


def test_winning_status_without_measured_r_cannot_enable_live(database):
    for i in range(20):insert(database,None,closed=i+1)
    health=db.get_route_live_readiness(SOURCE)
    assert not health['ready'] and health['resolved']==0
    assert health['missing_outcomes']==20 and health['total_r']==0
    with (patch.object(main,'DEPLOYMENT_MODE','live'),patch.object(main,'AUTOTRADE_ENABLED',True)):
        assert 'неподтверждённых исходов 20' in main._execution_state_text()


def test_positive_r_with_negative_price_result_cannot_reach_client_accounts(database):
    for i in range(20):insert(database,2.5 if i%2==0 else -1.,stop=99.8 if i%2==0 else 98.,closed=i+1)
    health=db.get_route_live_readiness(SOURCE)
    assert health['total_r']==15.
    assert health['total_price_return']==pytest.approx(-.15)
    assert not health['ready']
    with (patch.object(autotrader,'AUTOTRADE_ENABLED',True),
          patch.object(autotrader,'at_get_active_traders') as traders):
        autotrader.open_positions_for_signal(dict(id=1,source=SOURCE,autotrade_eligible=1))
    traders.assert_not_called()


@pytest.mark.parametrize('bad',[None,float('inf')])
def test_invalid_latest_outcome_cannot_be_replaced_by_an_older_winner(database,bad):
    for i in range(20):insert(database,1.,closed=i+1)
    insert(database,bad,closed=21)
    health=db.get_route_live_readiness(SOURCE)
    assert health['observed']==20 and health['resolved']==19
    assert not health['ready'] and health['missing_outcomes']==1


def test_invalid_stop_side_refuses_ready_window(database):
    for i in range(20):insert(database,1.,stop=101. if i==19 else 99.,closed=i+1)
    health=db.get_route_live_readiness(SOURCE)
    assert health['resolved']==20 and health['missing_geometry']==1
    assert not health['ready']


def test_future_closures_are_not_current_evidence(database):
    for i in range(20):
        insert(database,-1.,closed=i+1)
        insert(database,100.,closed=2000+i)
    with patch.object(db.time_mod,'time',return_value=1000):
        health=db.get_route_live_readiness(SOURCE)
    assert not health['ready'] and health['total_r']==-20


def test_complete_fixed_risk_positive_window_remains_ready(database):
    for i in range(20):insert(database,1.,closed=i+1)
    health=db.get_route_live_readiness(SOURCE)
    assert health['ready'] and health['resolved']==20
    assert health['total_price_return']==pytest.approx(.2)


def test_floating_point_residue_is_not_a_profitable_window(database):
    for i,rr in enumerate([.1,.2,-.3]+[0.]*17):insert(database,rr,closed=i+1)
    health=db.get_route_live_readiness(SOURCE)
    assert abs(health['total_r'])<1e-10 and not health['ready']
