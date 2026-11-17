import copy
import pytest
from stock_structure_geometry_nov17 import bracket,resolve


def fixture(sign=1):
    c=dict(time=[i*900 for i in range(80)],open=[100.]*80,close=[100.]*80,
           high=[100.2]*80,low=[99.8]*80)
    c['close'][5]=101. if sign>0 else 99.
    c['high'][5]=max(c['close'][5],100.2);c['low'][5]=min(c['close'][5],99.8)
    return dict(entry_index=8,entry_time=7200,direction_n=sign,atr14_fraction=.004,audited_target_pct=2.5),c


@pytest.mark.parametrize('sign',[1,-1])
def test_future_prices_do_not_change_bracket_and_stop_is_beyond_closed_extreme(sign):
    row,c=fixture(sign);g=bracket(row,c,'swing2_impulse_origin')
    assert g['target_fraction']==pytest.approx(.01)
    assert g['risk_fraction']==pytest.approx(.003)
    assert g['stop']<min(c['low'][6:8]) if sign>0 else g['stop']>max(c['high'][6:8])
    changed=copy.deepcopy(c)
    for key in ('open','high','low','close'):changed[key][8:]=[500.]*72
    assert bracket(row,changed,'swing2_impulse_origin')==g


def test_wide_structural_risk_is_rejected_instead_of_moving_the_stop_inside_structure():
    row,c=fixture();c['low'][7]=97
    assert bracket(row,c,'swing2_original_target') is None


def test_both_level_touch_is_stop_first_and_opening_gap_preserves_the_loss():
    row,c=fixture();g=bracket(row,c,'baseline')
    c['high'][8]=104;c['low'][8]=98
    assert resolve(row,c,g)['net_r']==pytest.approx(-1.)
    c['open'][8]=98.5;c['low'][8]=98
    assert resolve(row,c,g)['net_r']==pytest.approx(-1.5)


def test_structural_target_has_half_percent_floor_and_three_percent_cap():
    row,c=fixture();c['close'][5]=100.1
    assert bracket(row,c,'swing2_impulse_origin')['target_fraction']==.005
    c['close'][5]=110
    assert bracket(row,c,'swing2_impulse_origin')['target_fraction']==.03


@pytest.mark.parametrize('sign',[1,-1])
def test_confirmation_requires_closed_signal_body_in_entry_direction(sign):
    row,c=fixture(sign)
    assert bracket(row,c,'confirmed_swing2_impulse_origin') is None
    c['open'][7]=100.-sign*.1
    assert bracket(row,c,'confirmed_swing2_impulse_origin') is not None
