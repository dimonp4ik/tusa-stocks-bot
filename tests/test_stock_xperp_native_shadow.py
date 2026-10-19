import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from src import db
from src.stock_xperp_native_shadow import (
    AUDITED_SYMBOLS, BACKING_LEVEL_LOOKBACK_BARS, HOLD_BARS,
    backing_level_allows_entry, native_shadow_setups, normalized_atr14,
)
from src.telegram_notifier import bracket_for_analysis


def candles(last_close: float = 100.0, wide: bool = False,
            end: int = 39) -> dict:
    times = [index * 900 for index in range(end + 1)]
    close = [100.0] * (end + 1)
    close[-1] = last_close
    width = 2.0 if wide else .05
    return {
        "time": times,
        "open": list(close),
        "high": [value + width for value in close],
        "low": [value - width for value in close],
        "close": close,
        "volume": [1000.0] * len(close),
    }


def universe(positive: float = 102.0, negative: float | None = None) -> dict:
    symbols = sorted(AUDITED_SYMBOLS)[:20]
    result = {symbol: candles() for symbol in symbols}
    result[symbols[0]] = candles(positive)
    if negative is not None:
        result[symbols[1]] = candles(negative)
    return result


def test_native_route_uses_closed_cross_section_and_has_no_size():
    setups = native_shadow_setups(universe(), now=39 * 900 + 901)
    assert len(setups) == 1
    setup = setups[0]
    assert setup["direction"] == "SHORT"
    assert setup["target_r"] == 2.0
    assert setup["hold_bars"] == HOLD_BARS
    assert setup["universe_n"] == 20
    assert setup["backing_level_lookback_bars"] == BACKING_LEVEL_LOOKBACK_BARS
    assert setup["cross_z"] > 2
    assert "size" not in setup
    assert "size_mult" not in setup


def test_native_route_ranks_larger_extreme_first():
    setups = native_shadow_setups(
        universe(positive=103.0, negative=98.0), now=39 * 900 + 901)
    assert len(setups) == 2
    assert abs(setups[0]["cross_z"]) > abs(setups[1]["cross_z"])
    assert [row["direction"] for row in setups] == ["SHORT", "LONG"]


def test_native_route_rejects_high_atr_and_stale_signal():
    data = universe()
    first = sorted(data)[0]
    data[first] = candles(102.0, wide=True)
    assert native_shadow_setups(data, now=39 * 900 + 901) == []
    assert native_shadow_setups(universe(), now=39 * 900 + 1801) == []


def test_normalized_atr_uses_fourteen_closed_bars():
    value = normalized_atr14(candles(), 39)
    assert value == pytest.approx(.001)


def test_native_analysis_is_market_only_shadow_with_exact_bracket():
    setup = native_shadow_setups(universe(), now=39 * 900 + 901)[0]
    analysis = main._native_xperp_analysis(setup, 100.0)
    assert analysis["entry_source"] == "MARKET"
    assert analysis["_shadow_only"] is True
    assert analysis["strategy_max_bars"] == 64
    assert any("Backing level 32 bars" in value
               for value in analysis["signals"])
    assert "size" not in analysis
    assert bracket_for_analysis(analysis, 100.0) == (98.0, 98.0, 101.0)


def test_native_scan_publishes_only_paper_candidate_contract():
    symbols = sorted(AUDITED_SYMBOLS)[:10]
    setup = native_shadow_setups(universe(), now=39 * 900 + 901)[0]
    with patch.object(main, "_native_xperp_daily_loss_pause", return_value=False), \
            patch.object(main, "get_xperp_instruments",
                         return_value={symbol[:-4]: "id" for symbol in symbols}), \
            patch.object(main, "get_klines_xperp", return_value=candles()), \
            patch.object(main, "native_shadow_setups", return_value=[setup]), \
            patch.object(main, "get_open_signals", return_value=[]), \
            patch.object(main, "get_active_symbol_blocks", return_value=[]), \
            patch.object(main, "get_xperp_price", return_value=100.0), \
            patch.object(main, "log_setup_candidate_once", return_value=7), \
            patch.object(main, "_venue_signal_on_cooldown", return_value=False), \
            patch.object(main, "_publish_stock_venue_candidates",
                         return_value=1) as publish:
        main._run_native_xperp_shadow_scan()
    candidates, exposure = publish.call_args.args
    assert exposure == {"LONG": 0, "SHORT": 0}
    assert len(candidates) == 1
    analysis, setup_id = candidates[0]
    assert setup_id == 7
    assert analysis["source"] == "stock_xperp_native_shadow"
    assert analysis["_shadow_only"] is True


def test_backing_level_blocks_crossed_short_resistance():
    setup = native_shadow_setups(universe(), now=39 * 900 + 901)[0]
    assert setup["direction"] == "SHORT"
    assert backing_level_allows_entry(setup, setup["backing_level"])
    assert not backing_level_allows_entry(
        setup, setup["backing_level"] + .01)


def test_native_scan_records_but_does_not_publish_crossed_level():
    symbols = sorted(AUDITED_SYMBOLS)[:10]
    setup = native_shadow_setups(universe(), now=39 * 900 + 901)[0]
    crossed = setup["backing_level"] + .01
    with patch.object(main, "_native_xperp_daily_loss_pause", return_value=False), \
            patch.object(main, "get_xperp_instruments",
                         return_value={symbol[:-4]: "id" for symbol in symbols}), \
            patch.object(main, "get_klines_xperp", return_value=candles()), \
            patch.object(main, "native_shadow_setups", return_value=[setup]), \
            patch.object(main, "get_open_signals", return_value=[]), \
            patch.object(main, "get_active_symbol_blocks", return_value=[]), \
            patch.object(main, "get_xperp_price", return_value=crossed), \
            patch.object(main, "log_setup_candidate_once", return_value=7), \
            patch.object(main, "mark_setup_blocked") as blocked, \
            patch.object(main, "_publish_stock_venue_candidates",
                         return_value=0) as publish:
        main._run_native_xperp_shadow_scan()
    blocked.assert_called_once_with(7, "backing_level")
    assert publish.call_args.args[0] == []


def test_route_specific_loss_streak_counts_negative_timed_exit():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.db")
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            connection = sqlite3.connect(path)
            try:
                values = [
                    (100, "EXPIRED", -.2, "Native stock X-Perp shadow one"),
                    (200, "SL_HIT", -1.0, "Native stock X-Perp shadow two"),
                    (300, "SL_HIT", -1.0, "Other route"),
                ]
                for closed_at, status, realized_r, reason in values:
                    connection.execute(
                        """INSERT INTO signals
                           (symbol,direction,entry_price,tp1,tp2,sl,opened_at,
                            status,closed_at,reason,realized_r)
                           VALUES ('AAPLUSDT','LONG',100,102,102,99,1,?,?,?,?)""",
                        (status, closed_at, reason, realized_r),
                    )
                connection.commit()
            finally:
                connection.close()
            assert db.get_today_loss_streak(
                0, "Native stock X-Perp shadow") == 2
