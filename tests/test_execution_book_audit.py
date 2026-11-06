import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

from src import db


def test_first_execution_book_snapshot_is_stable():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE setup_log (
            signal_id INTEGER,
            book_spread_bps REAL,
            book_depth_usd REAL,
            book_imbalance REAL
        )
    """)
    conn.execute("INSERT INTO setup_log(signal_id) VALUES (42)")

    @contextmanager
    def fake_conn():
        with conn:
            yield conn

    with patch.object(db, "_conn", fake_conn):
        assert db.record_signal_execution_book(
            42, spread_bps=1.5, depth_usd=12_000, imbalance=.2)
        assert not db.record_signal_execution_book(
            42, spread_bps=9, depth_usd=1, imbalance=-.9)

    row = conn.execute("SELECT * FROM setup_log").fetchone()
    assert row == (42, 1.5, 12_000, .2)


def test_invalid_execution_book_snapshot_is_not_written():
    assert not db.record_signal_execution_book(
        0, spread_bps=1, depth_usd=100, imbalance=0)
    assert not db.record_signal_execution_book(
        1, spread_bps=1, depth_usd=0, imbalance=0)
    assert not db.record_signal_execution_book(
        1, spread_bps=1, depth_usd=100, imbalance=2)
