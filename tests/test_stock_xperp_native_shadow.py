import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import main
from src import db
from src.stock_xperp_native_shadow import (
    AUDITED_SYMBOLS, BACKING_LEVEL_LOOKBACK_BARS, HOLD_BARS,
    LONG_MIN_BASKET_TREND, SHORT_MIN_BASKET_TREND,
    REQUIRED_HISTORY_BARS,
    backing_level_allows_entry, basket_trend,
    native_shadow_setups, normalized_atr14, routed_target_fraction,
)
from src.telegram_notifier import bracket_for_analysis


def candles(last_close: float = 100.0, wide: bool = False,
            end: int = 119) -> dict:
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
    setups = native_shadow_setups(universe(), now=119 * 900 + 901)
    assert len(setups) == 1
    setup = setups[0]
    assert setup["direction"] == "SHORT"
    assert setup["target_r"] == 3.0
    assert setup["hold_bars"] == HOLD_BARS
    assert setup["universe_n"] == 20
    assert setup["backing_level_lookback_bars"] == BACKING_LEVEL_LOOKBACK_BARS
    assert setup["cross_z"] > 2
    assert setup["basket_trend_96"] >= LONG_MIN_BASKET_TREND
    assert setup["basket_trend_16"] > SHORT_MIN_BASKET_TREND
    assert "basket_trend_32" in setup
    assert "size" not in setup
    assert "size_mult" not in setup


def test_native_route_ranks_larger_extreme_first():
    setups = native_shadow_setups(
        universe(positive=103.0, negative=98.0), now=119 * 900 + 901)
    assert len(setups) == 2
    assert abs(setups[0]["cross_z"]) > abs(setups[1]["cross_z"])
    assert [row["direction"] for row in setups] == ["SHORT", "LONG"]


def test_native_route_rejects_high_atr_and_stale_signal():
    data = universe()
    first = sorted(data)[0]
    data[first] = candles(102.0, wide=True)
    assert native_shadow_setups(data, now=119 * 900 + 901) == []
    # The replay enters at the next bar open.  Production may wait briefly for
    # OKX to confirm the just-closed bar, but a later five-minute retry must not
    # publish the old setup at a price the replay never traded.
    assert native_shadow_setups(universe(), now=120 * 900 + 180) != []
    assert native_shadow_setups(universe(), now=120 * 900 + 181) == []


def test_native_route_diagnostics_explain_insufficient_runtime_history():
    data = universe()
    shortened = {
        symbol: {key: values[-80:] for key, values in series.items()}
        for symbol, series in data.items()
    }
    audit = {}
    assert native_shadow_setups(
        shortened, now=119 * 900 + 901, diagnostics=audit) == []
    assert audit["result"] == "insufficient_regime_history"
    assert audit["required_history_bars"] == REQUIRED_HISTORY_BARS
    assert audit["aligned_symbols"] == 20


def test_normalized_atr_uses_fourteen_closed_bars():
    value = normalized_atr14(candles(), 119)
    assert value == pytest.approx(.001)


def test_basket_trend_detects_broad_decline_from_closed_bars_only():
    symbols = sorted(AUDITED_SYMBOLS)[:20]
    values = [100.0 * (.999 ** index) for index in range(120)]
    data = {}
    for symbol in symbols:
        data[symbol] = {
            "time": [index * 900 for index in range(120)],
            "open": list(values), "high": [x + .05 for x in values],
            "low": [x - .05 for x in values], "close": list(values),
            "volume": [1000.0] * 120,
        }
    value = basket_trend(data, 119 * 900)
    assert value is not None
    assert value < LONG_MIN_BASKET_TREND


def test_native_route_rejects_short_after_exhausted_basket_decline():
    data = universe()
    for values in data.values():
        base = values["close"][-18]
        for index in range(len(values["close"]) - 17, len(values["close"])):
            base *= .9985
            values["open"][index] = base
            values["close"][index] = base
            values["high"][index] = base + .05
            values["low"][index] = base - .05
    first = sorted(data)[0]
    data[first]["close"][-1] *= 1.02
    data[first]["open"][-1] = data[first]["close"][-1]
    data[first]["high"][-1] = data[first]["close"][-1] + .05
    data[first]["low"][-1] = data[first]["close"][-1] - .05
    assert basket_trend(data, 119 * 900, 16) <= SHORT_MIN_BASKET_TREND
    assert native_shadow_setups(data, now=119 * 900 + 901) == []


def test_target_router_uses_exhaustion_before_utc_expansion():
    assert routed_target_fraction(-.041, 8 * 3600) == .02
    assert routed_target_fraction(-.02, 8 * 3600) == .03
    assert routed_target_fraction(-.02, 14 * 3600) == .025


def test_native_analysis_is_market_only_shadow_with_exact_bracket():
    setup = native_shadow_setups(universe(), now=119 * 900 + 901)[0]
    analysis = main._native_xperp_analysis(setup, 100.0)
    assert analysis["entry_source"] == "MARKET"
    assert analysis["_shadow_only"] is True
    assert analysis["strategy_max_bars"] == 64
    assert any("Backing level 32 bars" in value
               for value in analysis["signals"])
    assert any("Basket trend 16" in value for value in analysis["signals"])
    assert any("Basket trend 32" in value for value in analysis["signals"])
    assert "size" not in analysis
    assert bracket_for_analysis(analysis, 100.0) == (97.0, 97.0, 101.0)


def test_native_scan_publishes_only_paper_candidate_contract():
    symbols = sorted(AUDITED_SYMBOLS)[:10]
    setup = native_shadow_setups(universe(), now=119 * 900 + 901)[0]
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
            patch.object(main, "_persist_native_scan_audit") as persist, \
            patch.object(main, "_publish_stock_venue_candidates",
                         return_value=1) as publish:
        main._run_native_xperp_shadow_scan()
    candidates, exposure, diagnostics = publish.call_args.args
    assert exposure == {"LONG": 0, "SHORT": 0}
    assert diagnostics == {}
    assert len(candidates) == 1
    analysis, setup_id = candidates[0]
    assert setup_id == 7
    assert analysis["source"] == "stock_xperp_native_shadow"
    assert analysis["_shadow_only"] is True
    audit = persist.call_args.args[0]
    assert audit["publish_rejects"] == {}


def test_native_status_explains_recent_decisions_and_open_rows():
    audit = {
        "ts": 2_000_000_000.0,
        "result": "qualified_but_not_published",
        "targets": 44,
        "fetched": 44,
        "raw_setups": 2,
        "publish_candidates": 0,
        "published": 0,
        "portfolio_rejects": {"duplicate_bar": 1},
        "publish_rejects": {},
    }
    recent = [{
        "symbol": "TESTUSDT", "direction": "SHORT",
        "source": "stock_xperp_native_shadow", "sent": 0,
        "block_reason": "dir_cap", "resolved": 1, "outcome": "SL",
    }]
    with patch.object(main.time, "time", return_value=2_000_000_060.0), \
            patch.object(main, "get_bot_state", return_value=json.dumps(audit)), \
            patch.object(main, "get_all_setups_since", return_value=recent), \
            patch.object(main, "get_open_signals", return_value=[{
                "symbol": "OPENUSDT", "direction": "LONG", "status": "OPEN",
                "opened_at": 2_000_000_000.0,
                "source": "stock_xperp_native_shadow",
            }]), \
            patch.object(main, "get_recent_signals", return_value=[{
                "symbol": "STOPUSDT", "direction": "SHORT",
                "status": "SL_HIT", "realized_r": -1.0,
                "opened_at": 1_999_996_400.0, "closed_at": 2_000_000_000.0,
                "strategy_runner_mode": "single_target",
                "strategy_bar_seconds": 900,
            }, {
                "symbol": "OLDUSDT", "direction": "LONG",
                "status": "EXPIRED", "realized_r": -.2,
                "opened_at": 1_999_800_000.0, "closed_at": 1_999_900_000.0,
                "strategy_runner_mode": "single_target",
                "strategy_bar_seconds": 900,
            }]):
        text = main._native_scan_status_text()
    assert "повтор свечи 1" in text
    assert "OPENUSDT LONG OPEN 0.0ч stock_xperp_native_shadow" in text
    assert "TESTUSDT SHORT dir_cap/SL" in text
    assert "Закрыто 24ч: SL_HIT 1" in text
    assert "STOPUSDT SHORT SL_HIT -1.00R 1.0ч" in text
    assert "OLDUSDT" not in text


def test_http_status_exposes_build_and_native_daily_brake():
    with patch.dict(main.os.environ, {"RAILWAY_GIT_COMMIT_SHA": "abcdef123456"}), \
            patch.object(main, "get_today_realized_r", return_value=-2.5), \
            patch.object(main, "get_today_loss_streak", return_value=0), \
            patch.object(main, "get_today_max_loss_streak", return_value=3), \
            patch.object(main, "_native_xperp_daily_loss_pause", return_value=True), \
            patch.object(main, "_native_scan_status_text", return_value="scan"):
        text, code = main.status()
    assert code == 200
    assert "Build: abcdef12" in text
    assert "Native UTC-day: -2.50R; loss streak 0; max streak 3; paused yes" in text


def test_native_scan_fetches_enough_bars_for_longest_regime_gate():
    symbols = sorted(AUDITED_SYMBOLS)[:10]
    with patch.object(main, "_native_xperp_daily_loss_pause", return_value=False), \
            patch.object(main, "get_xperp_instruments",
                         return_value={symbol[:-4]: "id" for symbol in symbols}), \
            patch.object(main, "get_klines_xperp", return_value=candles()) as fetch, \
            patch.object(main, "native_shadow_setups", return_value=[]), \
            patch.object(main, "get_open_signals", return_value=[]), \
            patch.object(main, "get_active_symbol_blocks", return_value=[]), \
            patch.object(main, "_persist_native_scan_audit"), \
            patch.object(main, "_publish_stock_venue_candidates", return_value=0):
        main._run_native_xperp_shadow_scan()
    assert fetch.call_count == len(symbols)
    assert all(call.kwargs["limit"] >= REQUIRED_HISTORY_BARS
               for call in fetch.call_args_list)


def test_backing_level_blocks_crossed_short_resistance():
    setup = native_shadow_setups(universe(), now=119 * 900 + 901)[0]
    assert setup["direction"] == "SHORT"
    assert backing_level_allows_entry(setup, setup["backing_level"])
    assert not backing_level_allows_entry(
        setup, setup["backing_level"] + .01)


def test_native_scan_records_but_does_not_publish_crossed_level():
    symbols = sorted(AUDITED_SYMBOLS)[:10]
    setup = native_shadow_setups(universe(), now=119 * 900 + 901)[0]
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
            patch.object(main, "_persist_native_scan_audit"), \
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


def test_route_specific_max_loss_streak_survives_later_winner():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.db")
        with patch.object(db, "DB_PATH", path):
            db.init_db()
            connection = sqlite3.connect(path)
            try:
                values = [
                    (100, "SL_HIT", -1.0),
                    (200, "EXPIRED", -.2),
                    (300, "TP2_HIT", 2.5),
                ]
                for closed_at, status, realized_r in values:
                    connection.execute(
                        """INSERT INTO signals
                           (symbol,direction,entry_price,tp1,tp2,sl,opened_at,
                            status,closed_at,reason,realized_r)
                           VALUES ('AAPLUSDT','LONG',100,102,102,99,1,?,?,?,?)""",
                        (status, closed_at,
                         "Native stock X-Perp shadow test", realized_r),
                    )
                connection.commit()
            finally:
                connection.close()
            assert db.get_today_loss_streak(
                0, "Native stock X-Perp shadow") == 0
            assert db.get_today_max_loss_streak(
                0, "Native stock X-Perp shadow") == 2
