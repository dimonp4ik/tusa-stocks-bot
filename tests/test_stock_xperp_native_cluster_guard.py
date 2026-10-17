from stock_module_cluster_audit import portfolio_gate


def row(symbol: str, entry_time: int, exit_time: int = 999999) -> dict:
    return {
        "symbol": symbol,
        "direction": "LONG",
        "entry_time": entry_time,
        "exit_time": exit_time,
        "outcome": "TP",
        "signed_return": 0.01,
        "module": "native",
    }


def test_rolling_hour_guard_reopens_exactly_at_boundary():
    rows = [row("A", 100), row("B", 200), row("C", 300), row("D", 3700)]
    accepted = portfolio_gate(rows, None, rolling_hour_cap=2)
    assert [item["symbol"] for item in accepted] == ["A", "B", "D"]


def test_rolling_hour_guard_counts_portfolio_entries_across_scans():
    rows = [row("A", 100), row("B", 1000), row("C", 2000), row("D", 3000)]
    accepted = portfolio_gate(rows, None, rolling_hour_cap=3)
    assert [item["symbol"] for item in accepted] == ["A", "B", "C"]


def test_loss_streak_cap_pauses_later_entries_same_day():
    rows = [
        {**row("A", 100, 200), "outcome": "TIME", "signed_return": -0.002},
        {**row("B", 300, 400), "outcome": "SL", "signed_return": -0.01},
        row("C", 500, 600),
    ]
    accepted = portfolio_gate(rows, None, loss_streak_cap=2)
    assert [item["symbol"] for item in accepted] == ["A", "B"]


def test_rank_key_selects_stronger_simultaneous_signal():
    rows = [
        {**row("A", 100), "cross_z": 2.1},
        {**row("B", 100), "cross_z": -3.4},
    ]
    accepted = portfolio_gate(
        rows, 1, rank_key=lambda item: -abs(item["cross_z"]))
    assert [item["symbol"] for item in accepted] == ["B"]
