from stock_module_cluster_audit import portfolio_gate


def test_cumulative_daily_cap_stops_entries_without_two_consecutive_losses():
    # Loss, winner, loss: the streak is only one, but the day's sum is -2R.
    rows=[dict(symbol=str(i),direction="LONG",entry_time=i*3600,exit_time=i*3600+100,
        signed_return=r*.01,net_r=r,module="test") for i,r in enumerate((-1.5,1,-1.5,2))]
    capped=portfolio_gate(rows,None,loss_streak_cap=2,daily_loss_r_cap=2)
    baseline=portfolio_gate(rows,None,loss_streak_cap=2)
    assert len(capped)==3
    assert len(baseline)==4


def test_equal_time_exit_cannot_pause_an_entry_and_next_day_resets():
    rows=[dict(symbol="a",direction="LONG",entry_time=0,exit_time=3600,
        signed_return=-.02,net_r=-2,module="test"),
        dict(symbol="b",direction="SHORT",entry_time=3600,exit_time=3700,
        signed_return=0,net_r=0,module="test"),
        dict(symbol="c",direction="LONG",entry_time=4000,exit_time=4100,
        signed_return=0,net_r=0,module="test"),
        dict(symbol="d",direction="LONG",entry_time=86400,exit_time=86500,
        signed_return=0,net_r=0,module="test")]
    kept=portfolio_gate(rows,None,daily_loss_r_cap=2)
    assert [r["symbol"] for r in kept]==["a","b","d"]
