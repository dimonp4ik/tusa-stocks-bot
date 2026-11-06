import sqlite3
from contextlib import contextmanager
from unittest.mock import patch

from src import db


def test_first_execution_book_snapshot_is_stable():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE setup_log (
            id INTEGER PRIMARY KEY,
            signal_id INTEGER,
            book_spread_bps REAL,
            book_depth_usd REAL,
            book_imbalance REAL,
            book_sample_kind TEXT,
            book_observed_ts REAL
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

    row = conn.execute(
        "SELECT signal_id, book_spread_bps, book_depth_usd, "
        "book_imbalance, book_sample_kind FROM setup_log").fetchone()
    assert row == (42, 1.5, 12_000, .2, "live_execution")


def test_decision_book_snapshot_is_labeled_and_stable():
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE setup_log (
            id INTEGER PRIMARY KEY,
            signal_id INTEGER,
            book_spread_bps REAL,
            book_depth_usd REAL,
            book_imbalance REAL,
            book_sample_kind TEXT,
            book_observed_ts REAL
        )
    """)
    conn.execute("INSERT INTO setup_log(id) VALUES (7)")

    @contextmanager
    def fake_conn():
        with conn:
            yield conn

    with patch.object(db, "_conn", fake_conn):
        assert db.record_setup_decision_book(
            7, spread_bps=2.5, depth_usd=8_000, imbalance=-.1)
        assert not db.record_setup_decision_book(
            7, spread_bps=9, depth_usd=1, imbalance=.9)

    row = conn.execute(
        "SELECT book_spread_bps, book_depth_usd, book_imbalance, "
        "book_sample_kind, book_observed_ts FROM setup_log").fetchone()
    assert row[:4] == (2.5, 8_000, -.1, "decision_snapshot")
    assert row[4] > 0


def test_init_db_migrates_book_context_columns(tmp_path, monkeypatch):
    path = tmp_path / "schema.db"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(setup_log)")}
    assert {
        "book_spread_bps", "book_depth_usd", "book_imbalance",
        "book_sample_kind", "book_observed_ts",
    } <= columns


def test_invalid_execution_book_snapshot_is_not_written():
    assert not db.record_signal_execution_book(
        0, spread_bps=1, depth_usd=100, imbalance=0)
    assert not db.record_signal_execution_book(
        1, spread_bps=1, depth_usd=0, imbalance=0)
    assert not db.record_signal_execution_book(
        1, spread_bps=1, depth_usd=100, imbalance=2)
