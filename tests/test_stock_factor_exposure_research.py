from stock_factor_exposure_research import semiconductor_proxy_key as key
from stock_module_cluster_audit import portfolio_gate


def row(symbol,direction,start,end):
    return dict(symbol=symbol,direction=direction,entry_time=start,exit_time=end,
        signed_return=.01,net_r=1,module='test')


def test_inverse_etf_directions_map_to_same_benchmark_bet():
    assert key(row('SOXLUSDT','LONG',0,1))==key(row('SOXSUSDT','SHORT',0,1))
    assert key(row('SOXLUSDT','SHORT',0,1))==key(row('SOXSUSDT','LONG',0,1))
    assert key(row('AAPLUSDT','LONG',0,1)) is None


def test_gate_replays_replacement_and_keeps_opposite_factor_direction():
    rows=[row('SOXLUSDT','LONG',0,4000),row('SOXSUSDT','SHORT',1,4000),row('AAPLUSDT','LONG',2,4000),
          row('SOXSUSDT','LONG',1800,4000)]
    kept=portfolio_gate(rows,None,exposure_key=key,exposure_cap=1)
    assert [(r['symbol'],r['direction']) for r in kept]==[
        ('SOXLUSDT','LONG'),('AAPLUSDT','LONG'),('SOXSUSDT','LONG')]


def test_exact_exit_timestamp_does_not_free_exposure_early():
    rows=[row('SOXLUSDT','LONG',0,100),row('SOXSUSDT','SHORT',100,110),
          row('SOXSUSDT','SHORT',101,120)]
    kept=portfolio_gate(rows,None,exposure_key=key,exposure_cap=1)
    assert [r['entry_time'] for r in kept]==[0,101]
