import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import db
from src.result_summary import format_results_message


def _insert(path: str, status: str, realized_r: float, opened_at: float,
            size_mult: float = 1.0):
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """INSERT INTO signals
               (symbol,direction,entry_price,tp1,tp2,sl,opened_at,status,
                closed_at,realized_r,size_mult)
               VALUES ('TESTUSDT','LONG',100,101,102,99,?,?,?,?,?)""",
            (opened_at, status, opened_at + 60, realized_r, size_mult),
        )
        connection.commit()
    finally:
        connection.close()


def test_stats_explain_gross_outcomes_and_client_deposit_model():
    with TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "stats.db")
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            now = time.time()
            _insert(path, "TP2_HIT", 2.0, now - 300, size_mult=.5)
            _insert(path, "SL_HIT", -1.0, now - 200, size_mult=2.0)
            _insert(path, "EXPIRED", 0.0, now - 100)
            stats = db.get_stats(days=1, margin_pct=3, leverage=10)

    assert stats["closed"] == 3
    assert (stats["wins"], stats["losses"], stats["flat"]) == (1, 1, 1)
    assert stats["win_rate"] == 33.3
    # User results ignore legacy strategy sizing: each client uses one setting.
    assert stats["unit_total_r"] == 1.0
    # +0.6%, then -0.3%, then flat, compounded.
    assert stats["account_return_pct"] == 0.298
    assert stats["account_max_drawdown_pct"] == 0.3
    assert stats["account_model_n"] == 3
    assert stats["max_loss_streak"] == 1


def test_result_message_names_denominators_and_excludes_open_trades():
    base = {
        "since": time.time() - 86400,
        "total": 4, "closed": 3, "open": 1, "tp1_partial_open": 0,
        "wins": 2, "losses": 1, "flat": 0, "win_rate": 66.7,
        "tp1_hit": 2, "tp1_rate": 50.0,
        "unit_total_r": 2.0, "unit_r_per_trade": .667,
        "account_return_pct": .61, "account_max_drawdown_pct": .3,
        "account_model_n": 3, "account_model_missing": 0,
        "max_loss_streak": 1,
    }
    text = format_results_message(
        base, base, base, ZoneInfo("Europe/Riga"), 3, 10, configured=True)
    assert "66.7%* из 3 закрытых" in text
    assert "TP1 достигнут: 2/4" in text
    assert "Открытые сделки не входят" in text
    assert "3% маржи на сделку" in text
    assert "макс. просадка 0.30%" in text
