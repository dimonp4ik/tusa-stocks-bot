"""
SQLite database for tracking signal performance.

Lifecycle:
  OPEN         -> signal is live, TP1 not reached yet
  TP1_PARTIAL  -> TP1 hit, 50% position closed, remaining 50% has SL moved to breakeven
  TP2_HIT      -> final target reached after TP1
  BREAKEVEN    -> TP1 hit, remaining 50% closed at entry price
  SL_HIT       -> initial stop hit before TP1
  EXPIRED      -> no TP1/SL within 24h
  TP1_EXPIRED  -> TP1 hit, then rest expired before TP2/BE
"""

import sqlite3
import time as time_mod
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DB_PATH, AUTO_BLOCK_ENABLED, AUTO_BLOCK_LOOKBACK_TRADES, AUTO_BLOCK_MIN_TRADES,
    AUTO_BLOCK_MAX_PROFIT_FACTOR, AUTO_BLOCK_MAX_WIN_RATE, AUTO_BLOCK_DAYS,
    TP1_R_MULT, TP2_R_MULT, TP1_CLOSE_FRAC, LIVE_HIST_EPOCH_TS,
)

ACTIVE_STATUSES = ("OPEN", "TP1_PARTIAL")
FINAL_STATUSES  = ("TP2_HIT", "BREAKEVEN", "SL_HIT", "EXPIRED", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")
TP1_STATUSES    = ("TP1_PARTIAL", "TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")
PROFIT_STATUSES = ("TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")


def _conn():
    """One connection per call, in WAL with a generous busy timeout.

    Nine scheduled jobs share this file — scan, position monitor, zone watch,
    shadow tracker, sweeps, digests — and none of them catch
    sqlite3.OperationalError. In the default rollback journal a writer blocks
    readers outright, and the 5-second default timeout then raises rather than
    waits: reproduced by holding a transaction for 7 seconds, at which point an
    unrelated set_bot_state() dies with "database is locked". The job it belongs
    to dies with it, and if that job is the position monitor, an open trade goes
    a cycle unwatched.

    WAL lets readers run while a writer works, which removes most of the
    contention outright; the timeout covers what is left. Both are set per
    connection — journal_mode is a property of the FILE and persists, the
    pragma is idempotent.
    """
    c = sqlite3.connect(DB_PATH, timeout=30.0)
    c.row_factory = sqlite3.Row
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        # A filesystem that cannot do WAL is still usable in the old mode —
        # slower under contention, but nothing here should fail to open.
        pass
    return c


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    """SQLite-safe migration helper."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    """Create tables if missing and migrate older DBs in place."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT NOT NULL,
                direction     TEXT NOT NULL,
                entry_price   REAL NOT NULL,
                tp1           REAL NOT NULL,
                tp2           REAL NOT NULL,
                sl            REAL NOT NULL,
                opened_at     REAL NOT NULL,
                status        TEXT NOT NULL DEFAULT 'OPEN',
                closed_at     REAL,
                exit_price    REAL,
                confidence    TEXT,
                reason        TEXT,
                tp1_hit_at    REAL,
                tp1_exit_price REAL,
                entry_low     REAL,
                entry_high    REAL,
                entry_source  TEXT,
                market_price  REAL,
                -- Strategy's own price at signal time, BEFORE it is overwritten
                -- with the live quote. Without it the drift between the bar the
                -- model fills at and the price the order actually pays cannot be
                -- measured at all — three separate live-vs-model conclusions were
                -- retracted for want of this one number.
                zone_entry_price REAL,
                mtf_score     INTEGER,
                mtf_score_max INTEGER,
                premium       INTEGER DEFAULT 0,
                atr           REAL,
                realized_r    REAL,
                runner_trail_atr_mult REAL
            )
        """)
        # Migrate older DBs
        for col, ddl in {
            "tp1_hit_at":    "REAL",
            "tp1_exit_price": "REAL",
            "entry_low":     "REAL",
            "entry_high":    "REAL",
            "entry_source":  "TEXT",
            "market_price":  "REAL",
            "zone_entry_price": "REAL",
            "mtf_score":     "INTEGER",
            "mtf_score_max": "INTEGER",
            "premium":       "INTEGER DEFAULT 0",
            "atr":           "REAL",
            "realized_r":    "REAL",
            "runner_trail_atr_mult": "REAL",
            # SL-wick diagnostic (ported 2026-07-22): on an SL_HIT, 1 = the deep
            # global feed ALSO breached the stop (real reversal), 0 = only the
            # thin X-Perp wicked to it (execution noise). NULL = not an SL /
            # check unavailable. Especially relevant for stock/commodity X-Perps
            # whose books can be thinner than crypto majors.
            "sl_xperp_only": "INTEGER",
            # Sizing context (2026-08-25). The autotrader sizes from the DB row,
            # not the in-memory setup, so a field a size rule keys on has to
            # survive the insert or the rule applies in the backtest only.
            "bos_extension_atr": "REAL",
            # 2026-08-27, same reason: OPEN_SESSION_SIZE_MULT keys on session
            # AND volume_ratio, and neither was on the row. Without them the
            # opening boost would size correctly in the backtest and do nothing
            # in production.
            "session":       "TEXT",
            "volume_ratio":  "REAL",
            # 2026-08-27: ORDERLY_SIZE_MULT keys on all three of eff_ratio,
            # vol_atr_pct and bos_extension_atr. The last was already here; the
            # other two were not, so without this the rule would size correctly
            # in the backtest and do nothing in production.
            "eff_ratio":     "REAL",
            "vol_atr_pct":   "REAL",
        }.items():
            _ensure_column(c, "signals", col, ddl)

        c.execute("""
            CREATE TABLE IF NOT EXISTS symbol_blocks (
                symbol        TEXT PRIMARY KEY,
                blocked_until REAL NOT NULL,
                reason        TEXT,
                created_at    REAL NOT NULL,
                stats_json    TEXT
            )
        """)

        # ── User tracking ────────────────────────────────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY,
                username      TEXT,
                first_name    TEXT,
                last_name     TEXT,
                first_seen    REAL NOT NULL,
                last_seen     REAL NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 1
            )
        """)

        # ── Dynamic admins (added via bot; super-admins stay in config.py) ──
        c.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                user_id    INTEGER PRIMARY KEY,
                username   TEXT,
                first_name TEXT,
                added_by   INTEGER,
                added_at   REAL NOT NULL,
                role       TEXT NOT NULL DEFAULT 'admin'
            )
        """)
        _ensure_column(c, "admins", "role", "TEXT NOT NULL DEFAULT 'admin'")

        # ── Persistent bot state (survives restarts) ─────────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # ── Claude API budget tracking ────────────────────────────────────────
        # One row per API call. Queried by summing today's spend.
        c.execute("""
            CREATE TABLE IF NOT EXISTS claude_usage (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                tier         TEXT NOT NULL,   -- 'LIGHT' | 'HEAVY'
                input_tok    INTEGER NOT NULL DEFAULT 0,
                output_tok   INTEGER NOT NULL DEFAULT 0,
                cache_write  INTEGER NOT NULL DEFAULT 0,
                cache_read   INTEGER NOT NULL DEFAULT 0,
                cost_usd     REAL NOT NULL DEFAULT 0.0,
                ok           INTEGER NOT NULL DEFAULT 1  -- 0 = failed/timeout
            )
        """)

        # ── Setup log (all setups sent to Claude, approved or rejected) ──────
        c.execute("""
            CREATE TABLE IF NOT EXISTS setup_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          REAL NOT NULL,
                symbol      TEXT NOT NULL,
                direction   TEXT NOT NULL,
                entry_price REAL,
                tp1         REAL,
                tp2         REAL,
                sl          REAL,
                mtf_score   INTEGER,
                decision    TEXT,
                confidence  TEXT,
                risk_score  INTEGER,
                reason      TEXT,
                sent        INTEGER NOT NULL DEFAULT 0,
                session     TEXT,
                entry_source TEXT,
                -- ATR the strategy measured at signal time. The live monitor and
                -- backtest both trail on THIS value; without it the shadow
                -- tracker had to estimate one from forward candles, and the two
                -- disagreed on 13% of outcomes.
                atr         REAL,
                outcome     TEXT,
                reached_tp1 INTEGER NOT NULL DEFAULT 0,
                reached_tp2 INTEGER NOT NULL DEFAULT 0,
                resolved    INTEGER NOT NULL DEFAULT 0,
                resolved_ts REAL,
                trend       TEXT
            )
        """)
        # Migrate older setup_log DBs: outcome-tracking columns (shadow tracker).
        for col, ddl in {
            "atr":          "REAL",
            "sl":           "REAL",
            "session":      "TEXT",
            "entry_source": "TEXT",
            "outcome":      "TEXT",
            "reached_tp1":  "INTEGER NOT NULL DEFAULT 0",
            "reached_tp2":  "INTEGER NOT NULL DEFAULT 0",
            "resolved":     "INTEGER NOT NULL DEFAULT 0",
            "resolved_ts":  "REAL",
            "trend":        "TEXT",
            # Open Interest shadow feature (logged, not yet acted on).
            "oi_delta_pct": "REAL",
            "oi_regime":    "TEXT",
            "oi_confirms":  "INTEGER",
            # 'live' = judged by Claude in production; 'backtest' = seeded
            # historical outcome (Claude memory prior, excluded from stats).
            "source":       "TEXT NOT NULL DEFAULT 'live'",
            # Realised R of the trade (backtest: real net_r incl. trailed
            # runner; live: left NULL, derived from bracket at read time).
            # Powers expectancy (avg R) in Claude's self-feedback block.
            "net_r":        "REAL",
            # Claude's stated strongest failure mode for the trade — collected
            # for future counter-argument-vs-outcome analysis (which of its
            # own worries actually materialize).
            "counter":      "TEXT",
            # Why a setup Claude APPROVED was still not published. Empty for
            # normal rows. Without this a capped setup is indistinguishable
            # from one Claude rejected (both land at sent=0), which understates
            # Claude's accuracy and pollutes the mirror experiment with setups
            # it actually liked. Values: 'scan_cap' (MAX_SIGNALS_PER_SCAN),
            # 'dir_cap' (same-direction correlation cap).
            "block_reason": "TEXT",
            # How many positions in THIS setup's direction were already open
            # when Claude judged it — lets us ask whether Claude's own approval
            # rate responds to a skewed book, separately from any hard cap.
            "open_same_dir": "INTEGER",
            # FK to signals.id once this setup is actually sent. Ported
            # 2026-07-31 from the crypto bot: a sent setup was independently
            # shadow-tracked on a different feed AND separately monitored for
            # real — these can disagree. A setup with a real position has an
            # authoritative outcome; it must not also run an independent
            # simulation that can contradict it.
            "signal_id":    "INTEGER",
        }.items():
            _ensure_column(c, "setup_log", col, ddl)

        # ── Autotrading: allow-listed users + their encrypted OKX keys ───────
        # allowed  — admin put the user on the list (gate for the DM button)
        # active   — onboarding finished, bot opens real positions
        # size_mode 'percent' (1-10% of balance) | 'fixed' ($ per trade)
        c.execute("""
            CREATE TABLE IF NOT EXISTS autotrade_users (
                user_id        INTEGER PRIMARY KEY,
                allowed        INTEGER NOT NULL DEFAULT 1,
                active         INTEGER NOT NULL DEFAULT 0,
                api_key_enc    TEXT,
                api_secret_enc TEXT,
                passphrase_enc TEXT,
                size_mode      TEXT,
                size_value     REAL,
                last_balance   REAL,
                mode_prompt_pending INTEGER NOT NULL DEFAULT 0,
                tp1_close_pct  REAL NOT NULL DEFAULT 0,
                added_by       INTEGER,
                added_at       REAL,
                activated_at   REAL
            )
        """)
        _ensure_column(c, "autotrade_users", "tp1_close_pct", "REAL NOT NULL DEFAULT 0")

        # ── Autotrading: one row per live position per user per signal ───────
        c.execute("""
            CREATE TABLE IF NOT EXISTS autotrade_positions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id   INTEGER NOT NULL,
                user_id     INTEGER NOT NULL,
                inst_id     TEXT NOT NULL,
                direction   TEXT NOT NULL,
                sz          REAL NOT NULL,
                entry_px    REAL,
                margin_usd  REAL,
                sl_algo_id  TEXT,
                sl_px       REAL,
                tp1_algo_id TEXT,
                tp1_sz      REAL,
                status      TEXT NOT NULL DEFAULT 'OPEN',
                opened_at   REAL NOT NULL,
                closed_at   REAL,
                close_reason TEXT,
                error       TEXT
            )
        """)
        _ensure_column(c, "autotrade_positions", "tp1_algo_id", "TEXT")
        _ensure_column(c, "autotrade_positions", "tp1_sz", "REAL")


def get_bot_state(key: str) -> str | None:
    """Read a persistent bot state value. Returns None if key not set."""
    with _conn() as c:
        row = c.execute("SELECT value FROM bot_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


def set_bot_state(key: str, value: str) -> None:
    """Write a persistent bot state value (upsert)."""
    with _conn() as c:
        c.execute("""
            INSERT INTO bot_state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))


def delete_signal(signal_id: int) -> bool:
    """Hard-delete a signal row by ID. Returns True if a row was removed."""
    with _conn() as c:
        cur = c.execute("DELETE FROM signals WHERE id = ?", (signal_id,))
        return cur.rowcount > 0


def get_recent_signals(limit: int = 20) -> list:
    """Return the most recent signals (any status) for admin review."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM signals ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_signals_count(symbol: str = None) -> int:
    """Total number of signals (optionally filtered by symbol)."""
    with _conn() as c:
        if symbol:
            row = c.execute(
                "SELECT COUNT(*) FROM signals WHERE symbol = ?", (symbol,)
            ).fetchone()
        else:
            row = c.execute("SELECT COUNT(*) FROM signals").fetchone()
        return int(row[0])


def get_signals_page(limit: int, offset: int, symbol: str = None) -> list:
    """Return a page of signals (newest first), optionally filtered by symbol."""
    with _conn() as c:
        if symbol:
            rows = c.execute(
                "SELECT * FROM signals WHERE symbol = ? "
                "ORDER BY opened_at DESC LIMIT ? OFFSET ?",
                (symbol, limit, offset),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM signals ORDER BY opened_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]


def get_distinct_signal_symbols() -> list:
    """All distinct symbols that ever appeared in the signals journal, A→Z."""
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT symbol FROM signals ORDER BY symbol ASC"
        ).fetchall()
        return [r["symbol"] for r in rows]


def log_signal(analysis: dict, tp1: float, tp2: float, sl: float) -> int:
    """Insert a new signal into DB. Status starts as OPEN. Returns its row id.

    The id matters: the send path used to re-find this row with
    get_latest_open_signal(symbol) — "newest OPEN row for this symbol" — which
    is a guess, correct only while at most one setup per symbol exists per scan.
    """
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO signals (
                symbol, direction, entry_price, tp1, tp2, sl, opened_at, status,
                confidence, reason, entry_low, entry_high, entry_source, market_price, zone_entry_price,
                mtf_score, mtf_score_max, premium, atr, bos_extension_atr,
                session, volume_ratio, eff_ratio, vol_atr_pct
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis["symbol"], analysis["direction"], analysis["current_price"],
            tp1, tp2, sl, time_mod.time(),
            analysis.get("confidence", "?"), analysis.get("reason", ""),
            analysis.get("entry_low"), analysis.get("entry_high"),
            analysis.get("entry_source"), analysis.get("market_price"),
            analysis.get("zone_entry_price"),
            analysis.get("mtf_score"), analysis.get("mtf_score"),
            1 if analysis.get("premium") else 0,
            analysis.get("atr"),
            analysis.get("bos_extension_atr"),
            analysis.get("session"),
            analysis.get("volume_ratio"),
            analysis.get("eff_ratio"),
            analysis.get("vol_atr_pct"),
        ))
        return cur.lastrowid


def get_signal_by_id(signal_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
        return dict(row) if row else None


def get_open_signals() -> list:
    """Return all signals that still need monitoring (OPEN + TP1_PARTIAL)."""
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM signals WHERE status IN ({placeholders})",
            ACTIVE_STATUSES,
        ).fetchall()
        return [dict(r) for r in rows]


def update_signal_status(signal_id: int, status: str, exit_price=None, realized_r=None,
                         runner_trail_atr_mult=None):
    """
    Update signal lifecycle.
    TP1_PARTIAL records TP1 but keeps signal active for TP2/BE monitoring.
    All other statuses close the signal.
    `realized_r` (optional) stores the actual R for variable-exit closes (trailing).
    `runner_trail_atr_mult` (optional) freezes the context-aware trail chosen at the
    TP1 candle so later monitor cycles reuse it instead of recomputing (post_tp1_v2).
    """
    now = time_mod.time()
    with _conn() as c:
        if status == "TP1_PARTIAL":
            c.execute("""
                UPDATE signals
                SET status = 'TP1_PARTIAL', tp1_hit_at = ?, tp1_exit_price = ?,
                    runner_trail_atr_mult = ?
                WHERE id = ? AND status = 'OPEN'
            """, (now, exit_price, runner_trail_atr_mult, signal_id))
        else:
            c.execute("""
                UPDATE signals SET status = ?, closed_at = ?, exit_price = ?, realized_r = ?
                WHERE id = ?
            """, (status, now, exit_price, realized_r, signal_id))


def set_sl_xperp_only(signal_id: int, xperp_only: int) -> None:
    """Record the SL-wick diagnostic on a stopped-out signal (see schema note)."""
    with _conn() as c:
        c.execute("UPDATE signals SET sl_xperp_only = ? WHERE id = ?",
                  (int(xperp_only), signal_id))


def get_sl_wick_stats(since_ts: float) -> dict:
    """Of SL closes since since_ts, how many were X-Perp-only wicks (thin-book
    noise) vs confirmed by the deep global feed (real reversals)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT sl_xperp_only FROM signals "
            "WHERE status = 'SL_HIT' AND closed_at >= ? AND sl_xperp_only IS NOT NULL",
            (since_ts,),
        ).fetchall()
    n = len(rows)
    xperp_only = sum(1 for r in rows if r["sl_xperp_only"] == 1)
    return {
        "n": n,
        "xperp_only": xperp_only,
        "confirmed": n - xperp_only,
        "xperp_only_pct": (xperp_only / n * 100) if n else 0.0,
    }


def _status_to_r(status: str) -> float:
    """Approximate R from a status alone. Delegates — do not reimplement.

    This was a second hand-written copy of the R model carrying the same dead
    50%-at-TP1 arithmetic as _status_r. It feeds get_symbol_performance and
    therefore auto_block_bad_symbols — a real decision that stops the bot
    trading a symbol. Scoring BREAKEVEN as +0.5R also made it count as a WIN,
    inflating win rate and profit factor, so bad symbols were under-blocked.
    """
    return _status_r(status)


def get_symbol_performance(symbol: str, lookback: int = None) -> dict:
    """Return recent closed-signal performance for one symbol."""
    lookback = lookback or AUTO_BLOCK_LOOKBACK_TRADES
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT status, realized_r FROM signals WHERE symbol = ? "
            f"AND status IN ({placeholders}) ORDER BY opened_at DESC LIMIT ?",
            [symbol, *FINAL_STATUSES, lookback],
        ).fetchall()

    # Use the R actually recorded on the trade, not a nominal value derived
    # from its status. This feeds auto_block_bad_symbols, which stops the
    # bot trading a symbol, and the status map scores every win as the same
    # +R regardless of what the runner really made -- on live rows here
    # those range from +0.29R to +3.27R. The crypto bot reads realized_r
    # and falls back to the map only when it is missing; this one never did.
    rs = [_row_r(r) for r in rows]
    gross_profit = sum(r for r in rs if r > 0)
    gross_loss   = abs(sum(r for r in rs if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    wins  = sum(1 for r in rs if r > 0)
    total = len(rs)
    win_rate = wins / total * 100 if total else 0.0

    return {
        "symbol":        symbol,
        "trades":        total,
        "wins":          wins,
        "win_rate":      round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "total_r":       round(sum(rs), 2),
    }


def get_recent_outcomes(symbol: str, limit: int = 8) -> list:
    """Recent final outcomes for one symbol — fuel for HEAVY coin memory.
    Includes closed_at so the prompt can show recency (a same-symbol reversal
    a few hours after a stop is a whipsaw signal Sonnet can't see otherwise)."""
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT direction, status, entry_price, exit_price, confidence, mtf_score, closed_at "
            f"FROM signals WHERE symbol = ? AND status IN ({placeholders}) "
            f"ORDER BY opened_at DESC LIMIT ?",
            [symbol, *FINAL_STATUSES, limit],
        ).fetchall()
    return [dict(r) for r in rows]


def set_symbol_block(symbol: str, days: int, reason: str, stats: dict = None) -> None:
    now   = time_mod.time()
    until = now + days * 86400
    with _conn() as c:
        c.execute("""
            INSERT INTO symbol_blocks (symbol, blocked_until, reason, created_at, stats_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                blocked_until = excluded.blocked_until,
                reason = excluded.reason,
                created_at = excluded.created_at,
                stats_json = excluded.stats_json
        """, (symbol, until, reason, now, json.dumps(stats or {}, ensure_ascii=False)))


def is_symbol_auto_blocked(symbol: str) -> bool:
    now = time_mod.time()
    with _conn() as c:
        row = c.execute(
            "SELECT blocked_until FROM symbol_blocks WHERE symbol = ?", (symbol,)
        ).fetchone()
        if not row:
            return False
        if float(row["blocked_until"]) <= now:
            c.execute("DELETE FROM symbol_blocks WHERE symbol = ?", (symbol,))
            return False
        return True


def get_active_symbol_blocks() -> list:
    now = time_mod.time()
    with _conn() as c:
        c.execute("DELETE FROM symbol_blocks WHERE blocked_until <= ?", (now,))
        rows = c.execute(
            "SELECT * FROM symbol_blocks WHERE blocked_until > ? ORDER BY blocked_until DESC",
            (now,),
        ).fetchall()
        return [dict(r) for r in rows]


def auto_block_bad_symbols() -> list:
    """Block symbols with consistently bad closed-signal stats. No API calls."""
    if not AUTO_BLOCK_ENABLED:
        return []

    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        symbols = [
            r["symbol"] for r in c.execute(
                f"SELECT DISTINCT symbol FROM signals WHERE status IN ({placeholders})",
                FINAL_STATUSES,
            ).fetchall()
        ]

    blocked = []
    for symbol in symbols:
        if is_symbol_auto_blocked(symbol):
            continue
        perf = get_symbol_performance(symbol)
        if perf["trades"] < AUTO_BLOCK_MIN_TRADES:
            continue
        if perf["profit_factor"] <= AUTO_BLOCK_MAX_PROFIT_FACTOR and \
           perf["win_rate"] <= AUTO_BLOCK_MAX_WIN_RATE:
            reason = (
                f"Auto-block {AUTO_BLOCK_DAYS}d: "
                f"PF={perf['profit_factor']} WR={perf['win_rate']}% trades={perf['trades']}"
            )
            set_symbol_block(symbol, AUTO_BLOCK_DAYS, reason, perf)
            blocked.append({"symbol": symbol, "reason": reason})
    return blocked


def unblock_symbol(symbol: str) -> None:
    """Manually remove a symbol from the block list."""
    with _conn() as c:
        c.execute("DELETE FROM symbol_blocks WHERE symbol = ?", (symbol,))


# ── User tracking ────────────────────────────────────────────────────────────

def upsert_user(user_id: int, username: str = None,
                first_name: str = None, last_name: str = None) -> None:
    """Insert or update a user record on every bot interaction."""
    now = time_mod.time()
    with _conn() as c:
        c.execute("""
            INSERT INTO users (user_id, username, first_name, last_name,
                               first_seen, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(user_id) DO UPDATE SET
                username      = COALESCE(excluded.username,    username),
                first_name    = COALESCE(excluded.first_name,  first_name),
                last_name     = COALESCE(excluded.last_name,   last_name),
                last_seen     = excluded.last_seen,
                message_count = message_count + 1
        """, (user_id, username, first_name, last_name, now, now))


def get_user_by_id(user_id: int) -> dict | None:
    """Return a single user record or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE user_id = ?",
                        (user_id,)).fetchone()
        return dict(row) if row else None


def get_all_users(limit: int = 20, offset: int = 0, query: str = "") -> list:
    """Return users sorted by most recent interaction. Supports pagination + search."""
    with _conn() as c:
        if query:
            q = f"%{query.lower()}%"
            rows = c.execute(
                "SELECT * FROM users WHERE LOWER(username) LIKE ? OR CAST(user_id AS TEXT) LIKE ? "
                "ORDER BY last_seen DESC LIMIT ? OFFSET ?",
                (q, q, limit, offset),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM users ORDER BY last_seen DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]


def get_users_count(query: str = "") -> int:
    """Total user count (optionally filtered by search query)."""
    with _conn() as c:
        if query:
            q = f"%{query.lower()}%"
            row = c.execute(
                "SELECT COUNT(*) FROM users WHERE LOWER(username) LIKE ? OR CAST(user_id AS TEXT) LIKE ?",
                (q, q),
            ).fetchone()
        else:
            row = c.execute("SELECT COUNT(*) FROM users").fetchone()
        return row[0] if row else 0


# ── Dynamic admin management ──────────────────────────────────────────────────

def add_dynamic_admin(user_id: int, username: str = None,
                      first_name: str = None, added_by: int = None,
                      role: str = "admin") -> None:
    """Add (or update) a dynamic admin/moderator entry in DB.
    role: 'admin' (full panel) | 'moderator' (monitoring + autotrade allow-list only)."""
    now = time_mod.time()
    with _conn() as c:
        c.execute("""
            INSERT INTO admins (user_id, username, first_name, added_by, added_at, role)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username   = COALESCE(excluded.username,   username),
                first_name = COALESCE(excluded.first_name, first_name),
                role       = excluded.role
        """, (user_id, username, first_name, added_by, now, role))


def remove_dynamic_admin(user_id: int) -> None:
    """Remove a dynamic admin/moderator from DB."""
    with _conn() as c:
        c.execute("DELETE FROM admins WHERE user_id = ?", (user_id,))


def get_dynamic_admins() -> list:
    """Return all dynamic admins/moderators ordered by when they were added."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM admins ORDER BY added_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def is_dynamic_admin(user_id: int) -> bool:
    """True when user_id has an entry in the admins table (admin OR moderator)."""
    with _conn() as c:
        return c.execute(
            "SELECT 1 FROM admins WHERE user_id = ?", (user_id,)
        ).fetchone() is not None


def get_dynamic_role(user_id: int) -> str | None:
    """'admin' | 'moderator' | None (not a dynamic admin/moderator)."""
    with _conn() as c:
        row = c.execute(
            "SELECT role FROM admins WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row["role"] if row else None


# ── Claude budget tracking ────────────────────────────────────────────────────

# Pricing per 1M tokens (USD) — update if Anthropic changes rates.
_CLAUDE_PRICES = {
    # model_prefix: (input, cache_write, cache_read, output)
    "claude-haiku":  (1.00, 1.25, 0.10, 5.00),
    "claude-sonnet": (3.00, 3.75, 0.30, 15.00),
}

def _model_price(model: str) -> tuple:
    for prefix, prices in _CLAUDE_PRICES.items():
        if prefix in model.lower():
            return prices
    return _CLAUDE_PRICES["claude-haiku"]   # safe default


def log_claude_call(tier: str, model: str, usage, ok: bool = True) -> float:
    """
    Record one Claude API call and return its cost in USD.
    `usage` is the anthropic Usage object (input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens).
    """
    inp  = getattr(usage, "input_tokens", 0) or 0
    out  = getattr(usage, "output_tokens", 0) or 0
    cw   = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr   = getattr(usage, "cache_read_input_tokens", 0) or 0

    p_in, p_cw, p_cr, p_out = _model_price(model)
    cost = (inp * p_in + cw * p_cw + cr * p_cr + out * p_out) / 1_000_000

    with _conn() as c:
        c.execute("""
            INSERT INTO claude_usage (ts, tier, input_tok, output_tok,
                                      cache_write, cache_read, cost_usd, ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (time_mod.time(), tier, inp, out, cw, cr, round(cost, 6), int(ok)))
    return round(cost, 6)


def get_claude_spend_today() -> float:
    """Return total Claude USD spend since midnight UTC today."""
    import time as _t
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM claude_usage WHERE ts >= ?",
            (midnight,)
        ).fetchone()
    return float(row[0])


def get_claude_spend_stats() -> dict:
    """Return spend summary: today, this week, total."""
    import time as _t
    from datetime import datetime, timezone
    now_ts = _t.time()
    now    = datetime.now(timezone.utc)
    today  = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    week   = now_ts - 7 * 86400
    with _conn() as c:
        def _sum(since):
            r = c.execute(
                "SELECT COALESCE(SUM(cost_usd),0), COUNT(*) FROM claude_usage WHERE ts >= ?",
                (since,)
            ).fetchone()
            return round(float(r[0]), 4), int(r[1])
        today_usd, today_calls = _sum(today)
        week_usd,  week_calls  = _sum(week)
        total_usd, total_calls = _sum(0)
    return {
        "today_usd": today_usd, "today_calls": today_calls,
        "week_usd":  week_usd,  "week_calls":  week_calls,
        "total_usd": total_usd, "total_calls": total_calls,
    }


def get_symbols_performance(days: int = 30, since_ts: float = None) -> list:
    """
    Per-symbol closed-signal performance over `days` days (or since_ts epoch).
    Returns list of dicts sorted by total_r descending.
    """
    cutoff = since_ts if since_ts is not None else (time_mod.time() - days * 86400)
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT symbol, status FROM signals "
            f"WHERE opened_at >= ? AND status IN ({placeholders})",
            [cutoff, *FINAL_STATUSES],
        ).fetchall()

    from collections import defaultdict
    by_sym: dict = defaultdict(list)
    for r in rows:
        by_sym[r["symbol"]].append(_status_to_r(r["status"]))

    results = []
    for sym, rs in by_sym.items():
        total   = len(rs)
        wins    = sum(1 for r in rs if r > 0)
        total_r = round(sum(rs), 2)
        results.append({
            "symbol":   sym,
            "trades":   total,
            "wins":     wins,
            "win_rate": round(wins / total * 100, 1) if total else 0.0,
            "total_r":  total_r,
        })

    results.sort(key=lambda x: x["total_r"], reverse=True)
    return results


# Banked at TP1 itself. Under the live exit profile (post_tp1_v2) TP1_CLOSE_FRAC
# is 0 — nothing closes there, TP1 only arms the trail — so this is 0.0. Derived
# rather than written down so it stays true if the profile ever changes.
_TP1_BANKED_R = float(TP1_CLOSE_FRAC) * float(TP1_R_MULT)
# The runner has no closed form. Conservative placeholder; the real value is in
# realized_r on every row the monitor closed.
_TRAIL_FALLBACK_R = float(TP1_R_MULT)


def _status_r(status: str) -> float:
    """R of a closed outcome — FALLBACK ONLY, for rows with no realized_r.

    The values here used to encode a 50%-at-TP1 model ("TP2: 50% closed at TP1
    + 50% at TP2 = 1.5R"), a geometry this bot does not run: EXIT_PROFILE is
    post_tp1_v2 with TP1_CLOSE_FRAC=0, so nothing is banked at TP1 and the whole
    position trails. TP2_HIT was reported as 1.5R when it is really 2.0R, and
    BREAKEVEN as +0.5R when nothing was banked at all.

    Ported from the crypto bot 2026-08-20, where the same three stale copies
    were found. A silent fallback that lies is worse than one that is missing.
    """
    if status == "TP2_HIT":
        return _TP1_BANKED_R + (1.0 - float(TP1_CLOSE_FRAC)) * float(TP2_R_MULT)
    if status == "TP1_TRAIL":
        return _TP1_BANKED_R + (1.0 - float(TP1_CLOSE_FRAC)) * _TRAIL_FALLBACK_R
    if status in ("TP1_HIT", "BREAKEVEN", "TP1_EXPIRED"):
        return _TP1_BANKED_R
    if status == "SL_HIT":
        return -1.00
    return 0.0


def _row_r(row) -> float:
    """Realized R for a row — prefers the stored realized_r, falls back to status R."""
    rr = row["realized_r"] if "realized_r" in row.keys() else None
    if rr is not None:
        return float(rr)
    return _status_r(row["status"])


def get_stats(days: int = 7, since_ts: float = None) -> dict:
    """Aggregate stats with R-value, direction breakdown and recent streak.

    `days`     — rolling window (last N×24h) when since_ts is None.
    `since_ts` — explicit epoch cutoff (e.g. Riga midnight for calendar 'today').
    """
    cutoff = since_ts if since_ts is not None else time_mod.time() - days * 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT status, direction, opened_at, premium, realized_r FROM signals WHERE opened_at >= ?",
            (cutoff,)
        ).fetchall()
        # Last 7 closed signals for streak (independent of days filter)
        streak_rows = c.execute(
            f"SELECT status FROM signals "
            f"WHERE status IN ({','.join('?'*len(FINAL_STATUSES))}) "
            f"ORDER BY opened_at DESC LIMIT 7",
            FINAL_STATUSES,
        ).fetchall()

    rows = [dict(r) for r in rows]

    # ── Basic counts ──────────────────────────────────────────────────────────
    total       = len(rows)
    active_open = sum(1 for r in rows if r["status"] == "OPEN")
    active_tp1  = sum(1 for r in rows if r["status"] == "TP1_PARTIAL")
    closed      = sum(1 for r in rows if r["status"] in FINAL_STATUSES)
    tp1_hit     = sum(1 for r in rows if r["status"] in TP1_STATUSES)
    tp2_hit     = sum(1 for r in rows if r["status"] == "TP2_HIT")
    breakeven   = sum(1 for r in rows if r["status"] == "BREAKEVEN")
    sl_hit      = sum(1 for r in rows if r["status"] == "SL_HIT")
    expired     = sum(1 for r in rows if r["status"] == "EXPIRED")
    tp1_expired = sum(1 for r in rows if r["status"] == "TP1_EXPIRED")
    profitable  = sum(1 for r in rows if r["status"] in PROFIT_STATUSES)

    win_rate = (profitable / closed * 100) if closed else 0.0
    tp1_rate = (tp1_hit    / total  * 100) if total  else 0.0

    # ── Total R ───────────────────────────────────────────────────────────────
    total_r = sum(_row_r(r) for r in rows if r["status"] in FINAL_STATUSES)
    r_per_trade = (total_r / closed) if closed else 0.0

    # ── Direction breakdown ───────────────────────────────────────────────────
    dir_stats = {}
    for direction in ("LONG", "SHORT"):
        dr = [r for r in rows if r.get("direction") == direction]
        dr_closed = [r for r in dr if r["status"] in FINAL_STATUSES]
        dr_wins   = sum(1 for r in dr_closed if r["status"] in PROFIT_STATUSES)
        dr_r      = sum(_row_r(r) for r in dr_closed)
        dir_stats[direction] = {
            "total":    len(dr),
            "closed":   len(dr_closed),
            "wins":     dr_wins,
            "win_rate": round(dr_wins / len(dr_closed) * 100, 1) if dr_closed else 0.0,
            "total_r":  round(dr_r, 2),
        }

    # ── Premium breakdown (💎 OB+FVG overlap + sweep setups) ──────────────────
    prem_rows   = [r for r in rows if r.get("premium")]
    prem_closed = [r for r in prem_rows if r["status"] in FINAL_STATUSES]
    prem_wins   = sum(1 for r in prem_closed if r["status"] in PROFIT_STATUSES)
    prem_r      = sum(_row_r(r) for r in prem_closed)
    premium = {
        "total":    len(prem_rows),
        "closed":   len(prem_closed),
        "wins":     prem_wins,
        "win_rate": round(prem_wins / len(prem_closed) * 100, 1) if prem_closed else 0.0,
        "total_r":  round(prem_r, 2),
    }

    # ── Recent streak (last 7 closed, newest first) ───────────────────────────
    streak = []
    for r in streak_rows:
        st = r["status"]
        if st == "TP2_HIT":
            streak.append("🏆")
        elif st in PROFIT_STATUSES:
            streak.append("✅")
        elif st == "SL_HIT":
            streak.append("❌")
        else:
            streak.append("➖")
    # Count current run (consecutive same OUTCOME GROUP from newest).
    # ✅ and 🏆 are both "win" — mix of them still counts as a streak.
    def _grp(icon): return "win" if icon in ("✅", "🏆") else ("loss" if icon == "❌" else "neutral")
    current_run = 1
    if len(streak) >= 2:
        g0 = _grp(streak[0])
        for i in range(1, len(streak)):
            if _grp(streak[i]) == g0:
                current_run += 1
            else:
                break

    return {
        "days":             days,
        "total":            total,
        "closed":           closed,
        "open":             active_open,
        "tp1_partial_open": active_tp1,
        "tp1_hit":          tp1_hit,
        "tp1_rate":         round(tp1_rate, 1),
        "tp2_hit":          tp2_hit,
        "breakeven":        breakeven,
        "tp1_expired":      tp1_expired,
        "sl_hit":           sl_hit,
        "expired":          expired,
        "win_rate":         round(win_rate, 1),
        "total_r":          round(total_r, 2),
        "r_per_trade":      round(r_per_trade, 3),
        "long":             dir_stats.get("LONG",  {}),
        "short":            dir_stats.get("SHORT", {}),
        "premium":          premium,
        "streak":           streak,
        "current_run":      current_run,
    }


# ── Setup log ─────────────────────────────────────────────────────────────────

def log_setup_candidate(analysis: dict) -> int:
    """Log a setup that reached Claude (before/after verdict). Returns row id.

    Stores the SAME final TP1/TP2/SL bracket a live trade would use (not the raw
    zone levels) so the shadow tracker can resolve every setup — sent or rejected
    — on one consistent basis. Errors in bracket calc fall back to zone levels.
    """
    price = analysis.get("current_price") or 0.0
    tp1 = analysis.get("tp1_level")
    tp2 = analysis.get("tp2_level")
    sl  = None
    try:
        from src.telegram_notifier import calculate_tp_sl  # local: avoid circular import
        tp1, tp2, sl = calculate_tp_sl(
            float(price), analysis.get("direction", ""),
            atr=float(analysis.get("atr", 0.0) or 0.0),
            recent_high=float(analysis.get("recent_high", 0.0) or 0.0),
            recent_low=float(analysis.get("recent_low", 0.0) or 0.0),
            tp1_level=analysis.get("tp1_level"),
            tp2_level=analysis.get("tp2_level"),
        )
    except Exception:
        pass
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO setup_log
                (ts, symbol, direction, entry_price, tp1, tp2, sl,
                 mtf_score, decision, confidence, risk_score, reason, sent,
                 session, entry_source, atr, trend,
                 oi_delta_pct, oi_regime, oi_confirms, counter, open_same_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time_mod.time(),
            analysis.get("symbol", ""),
            analysis.get("direction", ""),
            price,
            tp1,
            tp2,
            sl,
            analysis.get("mtf_score"),
            analysis.get("decision", "NO TRADE"),
            analysis.get("confidence", ""),
            analysis.get("risk_score"),
            analysis.get("reason", ""),
            analysis.get("session", ""),
            analysis.get("entry_source", ""),
            analysis.get("atr"),
            analysis.get("swing_trend", ""),
            analysis.get("oi_delta_pct"),
            analysis.get("oi_regime"),
            analysis.get("oi_confirms"),
            analysis.get("counter", ""),
            analysis.get("open_same_dir"),
        ))
        return cur.lastrowid


def mark_setup_blocked(setup_log_id: int, reason: str) -> None:
    """Record that a Claude-APPROVED setup was withheld by a cap, not rejected."""
    if not setup_log_id:
        return
    with _conn() as c:
        c.execute("UPDATE setup_log SET block_reason=? WHERE id=?", (reason, setup_log_id))


def get_cap_impact_stats(since_ts: float) -> dict:
    """Outcomes of setups Claude approved but never sent.

    Directly answers "did withholding it save money or cost it": the shadow
    tracker resolves these rows regardless of whether a trade was ever opened.
    Reported per reason — the two caps are deliberate risk controls,
    'send_failed' is a delivery bug and should read near-zero n.
    """
    out = {}
    with _conn() as c:
        for reason in ("dir_cap", "scan_cap", "send_failed"):
            rows = c.execute(
                """SELECT outcome, reached_tp1 FROM setup_log
                   WHERE resolved=1 AND COALESCE(outcome,'') != 'NO_FILL'
                     AND ts >= ? AND block_reason=?
                     AND COALESCE(source,'live')='live'""",
                (since_ts, reason),
            ).fetchall()
            n = len(rows)
            tp1 = sum(1 for r in rows if r["reached_tp1"])
            sl = sum(1 for r in rows if r["outcome"] == "SL")
            # Same first-order basis the mirror experiment uses: a stop avoided
            # is +1R, a TP1 missed is -TP1_R_MULT (this bot's own value).
            saved_r = sl * 1.0 - tp1 * float(TP1_R_MULT)
            out[reason] = {
                "n": n, "reached_tp1": tp1, "sl": sl,
                "tp1_pct": (tp1 / n * 100) if n else 0.0,
                "sl_pct": (sl / n * 100) if n else 0.0,
                "saved_r": saved_r,
            }
    return out


def get_skew_response_stats(since_ts: float) -> list:
    """Claude's own approval rate bucketed by how skewed the book already was."""
    with _conn() as c:
        rows = c.execute(
            """SELECT decision, open_same_dir, reached_tp1, outcome, resolved
               FROM setup_log
               WHERE ts >= ? AND open_same_dir IS NOT NULL
                 AND COALESCE(source,'live')='live'""",
            (since_ts,),
        ).fetchall()
    out = []
    for name, lo, hi in (("0-1", 0, 1), ("2-3", 2, 3), ("4+", 4, 10**6)):
        sub = [r for r in rows if lo <= int(r["open_same_dir"] or 0) <= hi]
        if not sub:
            continue
        ok = [r for r in sub if (r["decision"] or "") in ("LONG", "SHORT")]
        res = [r for r in sub if r["resolved"]]
        tp1 = sum(1 for r in res if r["reached_tp1"])
        out.append({
            "bucket": name, "n": len(sub), "approved": len(ok),
            "approve_pct": len(ok) / len(sub) * 100,
            "resolved": len(res),
            "tp1_pct": (tp1 / len(res) * 100) if res else 0.0,
        })
    return out


def get_all_setups_since(since_ts: float) -> list:
    """Every logged setup since a cutoff, raw — for the full-report CSV dump."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM setup_log WHERE ts >= ? ORDER BY ts ASC", (since_ts,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_signals_since(since_ts: float) -> list:
    """Every published signal since a cutoff, raw — for the full-report CSV dump."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM signals WHERE opened_at >= ? ORDER BY opened_at ASC",
            (since_ts,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_setup_sent(setup_log_id: int) -> None:
    """Mark a setup as actually sent to the channel."""
    if not setup_log_id:
        return
    with _conn() as c:
        c.execute("UPDATE setup_log SET sent=1 WHERE id=?", (setup_log_id,))


def get_unresolved_setups(max_age_sec: float, limit: int = 80) -> list:
    """Setups whose shadow outcome is not yet known.

    resolved=0 with a usable bracket (sl present), old enough to have at least
    one forward candle, and not older than max_age_sec (past that the window has
    expired and a final pass will mark them EXPIRED). Oldest first.
    """
    now = time_mod.time()
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM setup_log
               WHERE resolved=0 AND sl IS NOT NULL AND signal_id IS NULL
                 AND COALESCE(sent,0)=0
                 AND ts <= ? AND ts >= ?
               ORDER BY ts ASC LIMIT ?""",
            (now - 900, now - max_age_sec, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_setup_resolved(setup_id: int, outcome: str,
                        reached_tp1: int, reached_tp2: int,
                        net_r: float = None) -> None:
    """Record the outcome of a tracked setup (shadow-simulated, or copied from
    a real signal — see resolve_sent_setups_from_signals)."""
    with _conn() as c:
        c.execute(
            """UPDATE setup_log
               SET outcome=?, reached_tp1=?, reached_tp2=?, resolved=1, resolved_ts=?,
                   net_r=COALESCE(?, net_r)
               WHERE id=?""",
            (outcome, int(reached_tp1), int(reached_tp2), time_mod.time(), net_r, setup_id),
        )


def link_setup_to_signal(setup_log_id: int, signal_id: int) -> None:
    """Mark a setup_log row as backed by a real position (signals.id).

    Deliberately does NOT touch `resolved`: resolve_sent_setups_from_signals()
    re-checks every linked row and corrects any that disagree, so a stale row
    never has to be un-resolved. Un-resolving would make it vanish from
    get_setup_accuracy until the next tick succeeded — and permanently if that
    tick ever failed (hit in production on the crypto bot 2026-07-31).
    """
    if not setup_log_id or not signal_id:
        return
    with _conn() as c:
        c.execute("UPDATE setup_log SET signal_id=? WHERE id=?", (signal_id, setup_log_id))


_SIGNAL_STATUS_TO_OUTCOME = {
    "SL_HIT":      ("SL", 0, 0),
    "TP2_HIT":     ("TP2", 1, 1),
    "TP1_TRAIL":   ("TP1", 1, 0),
    "BREAKEVEN":   ("TP1", 1, 0),
    "TP1_EXPIRED": ("TP1", 1, 0),
    "EXPIRED":     ("EXPIRED", 0, 0),
}


def resolve_sent_setups_from_signals(limit: int = 80) -> int:
    """Copy the REAL outcome onto setup_log rows linked to a closed signal."""
    n = 0
    with _conn() as c:
        # Every linked row whose signal is FINAL, not just resolved=0 ones, so
        # a row already resolved with a wrong shadow-simulated outcome gets
        # corrected without ever having to be un-resolved (which would make it
        # invisible to get_setup_accuracy in the meantime).
        rows = c.execute(
            """SELECT sl.id AS setup_id, sl.resolved, sl.outcome AS cur_outcome,
                      s.status, s.realized_r
               FROM setup_log sl JOIN signals s ON s.id = sl.signal_id
               WHERE sl.signal_id IS NOT NULL
               ORDER BY sl.ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        # Write on THIS connection: mark_setup_resolved() would open a SECOND
        # connection to the same SQLite file while this one holds the read
        # transaction above, risking a silent "database is locked".
        now = time_mod.time()
        for r in rows:
            mapped = _SIGNAL_STATUS_TO_OUTCOME.get(r["status"])
            if mapped is None:
                continue
            outcome, r1, r2 = mapped
            if r["resolved"] and r["cur_outcome"] == outcome:
                continue
            c.execute(
                """UPDATE setup_log
                   SET outcome=?, reached_tp1=?, reached_tp2=?, resolved=1,
                       resolved_ts=?, net_r=COALESCE(?, net_r)
                   WHERE id=?""",
                (outcome, int(r1), int(r2), now, r["realized_r"], r["setup_id"]),
            )
            n += 1
    return n


def backfill_setup_signal_links(window_sec: float = 300) -> int:
    """One-shot, idempotent: link any already-sent setup_log row that predates
    signal_id to its real signals row, matched by symbol + direction +
    opened_at within window_sec. Self-limiting: only sent=1 AND signal_id IS
    NULL rows ever match.
    """
    n = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT id, symbol, direction, ts FROM setup_log "
            "WHERE sent=1 AND signal_id IS NULL"
        ).fetchall()
        for r in rows:
            sig = c.execute(
                """SELECT id FROM signals
                   WHERE symbol=? AND direction=?
                     AND opened_at BETWEEN ? AND ?
                   ORDER BY ABS(opened_at - ?) ASC LIMIT 1""",
                (r["symbol"], r["direction"],
                 r["ts"] - window_sec, r["ts"] + window_sec, r["ts"]),
            ).fetchone()
            if sig:
                c.execute("UPDATE setup_log SET signal_id=? WHERE id=?", (sig["id"], r["id"]))
                n += 1
    return n


def get_setup_accuracy(since_ts: float) -> dict:
    """Aggregate resolved-setup outcomes since since_ts, split by sent vs rejected.

    Returns counts and rates so the admin can see whether Claude's gate (the
    rejected bucket) actually has a worse outcome than what it let through.

    For the REJECTED bucket also computes a MIRROR estimate (shadow experiment,
    ported from the sister crypto bot 2026-07-22): what if we'd taken the
    OPPOSITE direction on each rejected setup with the levels swapped (original
    SL becomes the mirror's TP1, original TP1 becomes the mirror's stop)? Near-
    clean inverse computable from the already-resolved outcome, no new
    simulation needed:
      - original hit SL (price ran 1R against it) → mirror reaches its TP1 (+1R)
      - original reached TP1 (price ran TP1_R_MULT·R for it) → mirror hits its
        stop (-TP1_R_MULT·R)
      - original EXPIRED (no decisive move) → mirror also flat (~0)
    Mirror is +EV only when the rejected SL-rate clears TP1_R_MULT/(1+TP1_R_MULT)
    (e.g. ~41% at TP1_R_MULT=0.7, ~50% at 1.0). Watch it populate on live data
    before trusting it (small samples, first-order R est, ignores mirror
    trailing upside + live fees/slippage on a tight stop). NOT yet validated on
    stock/commodity data — the crypto bot's rejected-pool SL-rate (~54%/30d)
    happened to clear its own breakeven; this bot's numbers may differ.
    """
    out = {"sent": {}, "rejected": {}}
    with _conn() as c:
        for sent_val, key in ((1, "sent"), (0, "rejected")):
            rows = c.execute(
                # block_reason filter: a setup Claude APPROVED but a cap
                # withheld also sits at sent=0. Counting it as a rejection
                # understates Claude's accuracy and feeds the mirror experiment
                # setups Claude actually liked. Judged separately by
                # get_cap_impact_stats().
                """SELECT outcome, reached_tp1, reached_tp2 FROM setup_log
                   WHERE resolved=1 AND COALESCE(outcome,'') != 'NO_FILL'
                     AND ts >= ?
                     AND COALESCE(source,'live')='live'
                     AND (CASE WHEN UPPER(COALESCE(decision,'')) IN ('LONG','SHORT')
                               THEN 1 ELSE 0 END) = ?
                     AND (? = 0 OR sent = 1)""",
                # Split on the VERDICT, not on sent/block_reason: a setup Claude
                # APPROVED is re-logged on every scan while it waits for price to
                # return to the zone, only one copy is ever sent, and the rest sat
                # at sent=0 with nothing to tag them -- so they counted as
                # rejections. Measured on live crypto data 2026-09-01: five such
                # rows at 80% TP1 moved the rejected bucket from 53.8% to 61.1%
                # and flipped the reported gap negative. That number decides
                # whether the gate stays on.
                (since_ts, sent_val, sent_val),
            ).fetchall()
            n = len(rows)
            tp1 = sum(1 for r in rows if r["reached_tp1"])
            tp2 = sum(1 for r in rows if r["reached_tp2"])
            sl  = sum(1 for r in rows if r["outcome"] == "SL")
            exp = sum(1 for r in rows if r["outcome"] == "EXPIRED")
            out[key] = {
                "n": n, "reached_tp1": tp1, "reached_tp2": tp2, "sl": sl, "expired": exp,
                "tp1_pct": (tp1 / n * 100) if n else 0.0,
                "sl_pct":  (sl / n * 100) if n else 0.0,
            }
            if key == "rejected":
                m_win, m_loss = sl, tp1
                m_dec = m_win + m_loss
                m_r = m_win * 1.0 - m_loss * float(TP1_R_MULT)
                out[key].update({
                    "mirror_wins":   m_win,
                    "mirror_losses": m_loss,
                    "mirror_wr":     (m_win / m_dec * 100) if m_dec else 0.0,
                    "mirror_r":      round(m_r, 2),
                    "mirror_r_avg":  round(m_r / m_dec, 3) if m_dec else 0.0,
                })
    return out


def get_similar_resolved_setups(symbol: str, direction: str, mtf_score,
                                session: str = "", lookback_days: int = 30,
                                limit: int = 40, bt_limit: int = 60) -> list:
    """Resolved past setups similar to the one being judged, for AI self-feedback.

    Two tiers:
      live     — Claude-judged production setups from the last lookback_days
                 (recency matters: they reflect the current market regime);
      backtest — seeded historical outcomes (2024+) with NO time window: they
                 are priors ("how did entries like this behave historically"),
                 age is the point, not a defect.

    Coarse similarity (kept deliberately broad to avoid overfitting to noise):
    same direction, and either the same symbol OR a nearby mtf_score band.
    Newest first within each tier; каждый row carries `source` so the prompt
    builder can label live vs backtest separately.

    block_reason is excluded: _self_feedback splits these rows into "sent" vs
    "rejected" by the `sent` flag, but a setup Claude APPROVED and a cap or a
    send-failure withheld also sits at sent=0 — so Claude was being shown his
    own approvals as if he had rejected them. get_setup_accuracy got the same
    filter earlier; this call site is the one that actually feeds the prompt.

    The live tier is also floored at LIVE_HIST_EPOCH_TS: rows older than that
    came from a materially different bot and are not evidence about this one.
    """
    since = max(time_mod.time() - lookback_days * 86400, LIVE_HIST_EPOCH_TS or 0.0)
    try:
        score = int(mtf_score or 0)
    except (TypeError, ValueError):
        score = 0
    with _conn() as c:
        live = c.execute(
            """SELECT symbol, direction, mtf_score, session, entry_source,
                      decision, sent, outcome, reached_tp1, reached_tp2, ts, trend,
                      entry_price, tp1, tp2, sl, net_r,
                      COALESCE(source,'live') AS source
               FROM setup_log
               WHERE resolved=1 AND COALESCE(outcome,'') != 'NO_FILL'
                 AND ts >= ? AND direction=?
                 AND COALESCE(source,'live')='live'
                 AND COALESCE(block_reason,'')=''
                 AND (symbol=? OR ABS(COALESCE(mtf_score,0) - ?) <= 2)
               ORDER BY ts DESC LIMIT ?""",
            (since, direction, symbol, score, limit),
        ).fetchall()
        bt = c.execute(
            """SELECT symbol, direction, mtf_score, session, entry_source,
                      decision, sent, outcome, reached_tp1, reached_tp2, ts, trend,
                      entry_price, tp1, tp2, sl, net_r,
                      source
               FROM setup_log
               WHERE resolved=1 AND direction=? AND source='backtest'
                 AND (symbol=? OR ABS(COALESCE(mtf_score,0) - ?) <= 1)
               ORDER BY ts DESC LIMIT ?""",
            (direction, symbol, score, bt_limit),
        ).fetchall()
        return [dict(r) for r in live] + [dict(r) for r in bt]


def delete_backtest_seed_rows() -> int:
    """Wipe all source='backtest' setup_log rows — used when re-seeding with
    a corrected batch (e.g. a fixed BACKTEST_TP_WINDOW) so stale, understated
    priors don't sit alongside the corrected ones. Returns rows deleted.
    """
    with _conn() as c:
        cur = c.execute("DELETE FROM setup_log WHERE source='backtest'")
        return cur.rowcount


def seed_backtest_outcomes(rows: list) -> int:
    """Bulk-insert historical backtest trades as resolved setup_log rows
    (source='backtest'). These are Claude memory PRIORS — every stats consumer
    filters them out; only get_similar_resolved_setups reads them back.
    Returns inserted count. Caller gates one-shot execution via bot_state.
    """
    ins = 0
    with _conn() as c:
        for r in rows:
            try:
                outcome = str(r.get("outcome") or "")
                # TRAIL = post-TP1 trailed runner exit → TP1-class win for memory
                out_norm = "TP1" if outcome == "TRAIL" else outcome
                reached_tp1 = 1 if outcome in ("TP1", "TP2", "TRAIL") else 0
                reached_tp2 = 1 if outcome == "TP2" else 0
                ts = float(r.get("entry_time") or 0)
                if ts <= 0 or not r.get("symbol") or not r.get("direction"):
                    continue
                # Real realised R (net of costs, incl. trailed runner). Prefer
                # net_r; fall back to gross_r if a batch lacks the net column.
                try:
                    net_r = float(r.get("net_r"))
                except (TypeError, ValueError):
                    try:
                        net_r = float(r.get("gross_r"))
                    except (TypeError, ValueError):
                        net_r = None
                c.execute("""
                    INSERT INTO setup_log
                        (ts, symbol, direction, entry_price, tp1, tp2, sl,
                         mtf_score, decision, confidence, risk_score, reason, sent,
                         session, entry_source, trend,
                         outcome, reached_tp1, reached_tp2, resolved, resolved_ts,
                         source, net_r)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'BACKTEST', NULL, '', 1,
                            ?, ?, ?, ?, ?, ?, 1, ?, 'backtest', ?)
                """, (
                    ts,
                    str(r["symbol"]),
                    str(r["direction"]),
                    float(r.get("entry") or 0) or None,
                    float(r.get("tp1") or 0) or None,
                    float(r.get("tp2") or 0) or None,
                    float(r.get("sl") or 0) or None,
                    int(float(r.get("mtf_score") or 0)) or None,
                    str(r["direction"]),          # decision = filter's side
                    str(r.get("session") or ""),
                    str(r.get("entry_source") or ""),
                    str(r.get("swing_trend") or ""),
                    out_norm,
                    reached_tp1,
                    reached_tp2,
                    float(r.get("exit_time") or 0) or None,
                    net_r,
                ))
                ins += 1
            except Exception:
                continue
    return ins


def get_today_sl_streak(day_start_ts: float) -> int:
    """Consecutive SL_HIT streak among signals closed since day_start_ts,
    counted from the most recent close backwards. Any non-SL close breaks it.
    Powers the daily kill-switch."""
    with _conn() as c:
        rows = c.execute(
            "SELECT status FROM signals WHERE closed_at IS NOT NULL AND closed_at >= ? "
            "ORDER BY closed_at DESC",
            (day_start_ts,),
        ).fetchall()
    n = 0
    for r in rows:
        if r["status"] == "SL_HIT":
            n += 1
        else:
            break
    return n


def get_calibration_rows(since_ts: float, limit: int = 800) -> list:
    """Resolved live Claude-evaluated setups for the SCORECARD block: does his
    risk_score / confidence scale actually separate outcomes. Backtest priors
    excluded (they carry no Claude verdict).

    Floored at LIVE_HIST_EPOCH_TS like the other live-history reads: the 60-day
    window otherwise reaches into the pre-parity-fix bot, mixing outcomes from
    software that no longer exists into the calibration of scales he applies
    today."""
    since_ts = max(float(since_ts), LIVE_HIST_EPOCH_TS or 0.0)
    with _conn() as c:
        rows = c.execute(
            """SELECT decision, confidence, risk_score, outcome, reached_tp1,
                      entry_price, tp1, tp2, sl, net_r
               FROM setup_log
               WHERE resolved=1 AND ts >= ? AND COALESCE(source,'live')='live'
                 AND risk_score IS NOT NULL
               ORDER BY ts DESC LIMIT ?""",
            (since_ts, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def backfill_backtest_net_r(rows: list) -> int:
    """One-shot: fill net_r on already-seeded backtest rows (seeded before the
    net_r column existed). Matches on (symbol, direction, ts=entry_time).
    Returns updated count. Caller gates via bot_state so it runs once."""
    upd = 0
    with _conn() as c:
        for r in rows:
            try:
                ts = float(r.get("entry_time") or 0)
                if ts <= 0 or not r.get("symbol") or not r.get("direction"):
                    continue
                try:
                    net_r = float(r.get("net_r"))
                except (TypeError, ValueError):
                    try:
                        net_r = float(r.get("gross_r"))
                    except (TypeError, ValueError):
                        continue
                cur = c.execute(
                    """UPDATE setup_log SET net_r = ?
                       WHERE source='backtest' AND net_r IS NULL
                         AND symbol=? AND direction=? AND ts=?""",
                    (net_r, str(r["symbol"]), str(r["direction"]), ts),
                )
                upd += cur.rowcount
            except Exception:
                continue
    return upd


def get_weekly_stats() -> dict:
    """Aggregate trade + AI accuracy stats for the past 7 days."""
    from collections import defaultdict
    since = time_mod.time() - 7 * 86400
    with _conn() as c:
        sig_rows = c.execute(
            "SELECT symbol, direction, status, realized_r, trend FROM signals "
            "WHERE opened_at >= ? AND status IN ({})".format(
                ",".join("?" * len(FINAL_STATUSES))
            ),
            [since, *FINAL_STATUSES],
        ).fetchall()
        setup_rows = c.execute(
            "SELECT sent, resolved, reached_tp1, trend FROM setup_log "
            "WHERE ts >= ? AND resolved = 1 AND COALESCE(source,'live')='live'",
            (since,),
        ).fetchall()

    trades = [dict(r) for r in sig_rows]
    n_total = len(trades)
    n_tp2   = sum(1 for t in trades if t["status"] == "TP2_HIT")
    n_sl    = sum(1 for t in trades if t["status"] == "SL_HIT")
    n_exp   = sum(1 for t in trades if t["status"] in ("EXPIRED", "TP1_EXPIRED"))
    n_win   = sum(1 for t in trades if t["status"] in PROFIT_STATUSES)
    wr      = round(n_win / n_total * 100, 1) if n_total else 0.0
    total_r = round(sum(
        float(t.get("realized_r") or 0) if t.get("realized_r") is not None
        else _status_to_r(t["status"])
        for t in trades
    ), 2)

    sym_w: dict = defaultdict(int)
    sym_sl: dict = defaultdict(int)
    for t in trades:
        s = t["symbol"]
        if t["status"] in PROFIT_STATUSES: sym_w[s] += 1
        elif t["status"] == "SL_HIT": sym_sl[s] += 1
    all_syms = set(sym_w) | set(sym_sl)
    top3 = sorted(all_syms, key=lambda s: sym_w.get(s, 0) - sym_sl.get(s, 0), reverse=True)[:3]
    top3_data = [(s, sym_w.get(s, 0), sym_sl.get(s, 0)) for s in top3]

    best  = max(trades, key=lambda t: float(t.get("realized_r") or 0), default=None)
    worst = min(trades, key=lambda t: float(t.get("realized_r") or 0), default=None)

    trend_w: dict = defaultdict(int)
    trend_sl: dict = defaultdict(int)
    trend_n: dict = defaultdict(int)
    for t in trades:
        tr = (t.get("trend") or "").strip()
        if not tr:
            continue
        trend_n[tr] += 1
        if t["status"] in PROFIT_STATUSES: trend_w[tr] += 1
        elif t["status"] == "SL_HIT": trend_sl[tr] += 1
    trend_wr = {
        tr: round(trend_w[tr] / trend_n[tr] * 100, 0)
        for tr in trend_n if trend_n[tr] >= 3
    }

    setups = [dict(r) for r in setup_rows]
    sent_s = [s for s in setups if s["sent"] == 1]
    rej_s  = [s for s in setups if s["sent"] == 0]
    sent_tp1 = sum(1 for s in sent_s if s["reached_tp1"])
    rej_tp1  = sum(1 for s in rej_s  if s["reached_tp1"])

    return {
        "n_total":        n_total,
        "n_win":          n_win,
        "n_tp2":          n_tp2,
        "n_sl":           n_sl,
        "n_exp":          n_exp,
        "wr":             wr,
        "total_r":        total_r,
        "top3":           top3_data,
        "best_trade":     {"symbol": best["symbol"],  "r": float(best.get("realized_r")  or _status_to_r(best["status"]))}  if best  else None,
        "worst_trade":    {"symbol": worst["symbol"], "r": float(worst.get("realized_r") or _status_to_r(worst["status"]))} if worst else None,
        "n_sent":         len(sent_s),
        "n_rejected":     len(rej_s),
        "sent_tp1_rate":  round(sent_tp1 / len(sent_s) * 100, 1) if sent_s else 0.0,
        "rej_tp1_rate":   round(rej_tp1  / len(rej_s)  * 100, 1) if rej_s  else 0.0,
        "trend_wr":       trend_wr,
    }


def get_setups_by_date(date_str: str) -> list:
    """Return all setups for a given date. Accepts DD.MM, DD.MM.YYYY, YYYY-MM-DD.
    Timestamps stored as UTC, displayed in caller's chosen tz."""
    from datetime import datetime, timezone as _tz
    date_str = date_str.strip()
    dt = None
    for fmt in ("%d.%m.%Y", "%d.%m", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(date_str, fmt)
            if fmt == "%d.%m":
                parsed = parsed.replace(year=datetime.now().year)
            dt = parsed.replace(tzinfo=_tz.utc)
            break
        except ValueError:
            continue
    if dt is None:
        return []
    start_ts = dt.timestamp()
    end_ts   = start_ts + 86400
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM setup_log WHERE ts >= ? AND ts < ? "
            "AND COALESCE(source,'live')='live' ORDER BY ts ASC",
            (start_ts, end_ts),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Autotrading ────────────────────────────────────────────────────────────────

def at_add_allowed(user_id: int, added_by: int) -> None:
    """Admin puts a user on the autotrade allow-list (idempotent)."""
    with _conn() as c:
        c.execute("""
            INSERT INTO autotrade_users (user_id, allowed, added_by, added_at)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET allowed = 1
        """, (user_id, added_by, time_mod.time()))


def at_remove(user_id: int) -> None:
    """Admin removes a user: wipe keys, deactivate, drop from allow-list."""
    with _conn() as c:
        c.execute("""
            UPDATE autotrade_users
            SET allowed = 0, active = 0,
                api_key_enc = NULL, api_secret_enc = NULL, passphrase_enc = NULL
            WHERE user_id = ?
        """, (user_id,))


def at_get(user_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM autotrade_users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def at_all_allowed() -> list:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM autotrade_users WHERE allowed = 1 ORDER BY added_at"
        ).fetchall()
        return [dict(r) for r in rows]


def at_get_active_traders() -> list:
    """Users with finished onboarding — the ones real orders are opened for."""
    with _conn() as c:
        rows = c.execute("""
            SELECT * FROM autotrade_users
            WHERE allowed = 1 AND active = 1
              AND api_key_enc IS NOT NULL
        """).fetchall()
        return [dict(r) for r in rows]


def at_set_keys(user_id: int, api_key_enc: str, api_secret_enc: str,
                passphrase_enc: str) -> None:
    with _conn() as c:
        c.execute("""
            UPDATE autotrade_users
            SET api_key_enc = ?, api_secret_enc = ?, passphrase_enc = ?
            WHERE user_id = ?
        """, (api_key_enc, api_secret_enc, passphrase_enc, user_id))


def at_set_mode(user_id: int, size_mode: str, size_value: float) -> None:
    with _conn() as c:
        c.execute("""
            UPDATE autotrade_users
            SET size_mode = ?, size_value = ?, mode_prompt_pending = 0
            WHERE user_id = ?
        """, (size_mode, size_value, user_id))


def at_set_active(user_id: int, active: bool) -> None:
    with _conn() as c:
        c.execute("""
            UPDATE autotrade_users
            SET active = ?, activated_at = COALESCE(activated_at, ?)
            WHERE user_id = ?
        """, (1 if active else 0, time_mod.time(), user_id))


def at_set_balance(user_id: int, balance: float) -> None:
    with _conn() as c:
        c.execute("UPDATE autotrade_users SET last_balance = ? WHERE user_id = ?",
                  (balance, user_id))


def at_set_mode_prompt(user_id: int, pending: bool) -> None:
    with _conn() as c:
        c.execute("UPDATE autotrade_users SET mode_prompt_pending = ? WHERE user_id = ?",
                  (1 if pending else 0, user_id))


def at_set_tp1_close_pct(user_id: int, pct: float) -> None:
    """% of the position to market-close when TP1 first hits (0-100).
    0 = keep the full position on trailing (current default strategy)."""
    with _conn() as c:
        c.execute("UPDATE autotrade_users SET tp1_close_pct = ? WHERE user_id = ?",
                  (pct, user_id))


def at_log_position(signal_id: int, user_id: int, inst_id: str, direction: str,
                    sz: float, entry_px: float, margin_usd: float,
                    sl_algo_id: str, sl_px: float,
                    tp1_algo_id: str = None, tp1_sz: float = None) -> int:
    with _conn() as c:
        cur = c.execute("""
            INSERT INTO autotrade_positions
                (signal_id, user_id, inst_id, direction, sz, entry_px, margin_usd,
                 sl_algo_id, sl_px, tp1_algo_id, tp1_sz, status, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
        """, (signal_id, user_id, inst_id, direction, sz, entry_px, margin_usd,
              sl_algo_id, sl_px, tp1_algo_id, tp1_sz, time_mod.time()))
        return cur.lastrowid


def at_open_positions_for_signal(signal_id: int) -> list:
    with _conn() as c:
        rows = c.execute("""
            SELECT * FROM autotrade_positions
            WHERE signal_id = ? AND status = 'OPEN'
        """, (signal_id,)).fetchall()
        return [dict(r) for r in rows]


def at_has_open_position(user_id: int, inst_id: str) -> bool:
    """True if this user already holds an open position on this instrument.

    OKX allows exactly ONE closeFraction=1 TP/SL algo per position, so a second
    entry on the same instId cannot be protected: the OCO call fails with
    "You can only place 1 TP/SL order to close an entire position" and the
    caller then flattens the position, killing the FIRST (legitimate) trade
    along with it. Hit live on the sister crypto bot 2026-08-13 (PUMP) after a
    duplicate signal; ported here before it can cost a stock trade.
    """
    with _conn() as c:
        row = c.execute("""
            SELECT 1 FROM autotrade_positions
            WHERE user_id = ? AND inst_id = ? AND status != 'CLOSED'
            LIMIT 1
        """, (user_id, inst_id)).fetchone()
        return row is not None


def at_all_open_positions() -> list:
    """Every OPEN autotrade position across all users/signals — used by the
    fast exchange-side close poll (doesn't wait for the signal engine)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM autotrade_positions WHERE status = 'OPEN'"
        ).fetchall()
        return [dict(r) for r in rows]


def at_update_position_sl(pos_id: int, sl_px: float) -> None:
    with _conn() as c:
        c.execute("UPDATE autotrade_positions SET sl_px = ? WHERE id = ?",
                  (sl_px, pos_id))


def at_reduce_position_sz(pos_id: int, new_sz: float) -> None:
    """Shrink the tracked size after a partial close at TP1 — the remaining
    protection (OCO, closeFraction=1) auto-covers whatever's left on the
    exchange, this just keeps our own record in sync."""
    with _conn() as c:
        c.execute("UPDATE autotrade_positions SET sz = ? WHERE id = ?",
                  (new_sz, pos_id))


def at_close_position(pos_id: int, close_reason: str, error: str = None) -> None:
    with _conn() as c:
        c.execute("""
            UPDATE autotrade_positions
            SET status = 'CLOSED', closed_at = ?, close_reason = ?, error = ?
            WHERE id = ?
        """, (time_mod.time(), close_reason, error, pos_id))


def get_latest_open_signal(symbol: str) -> dict | None:
    """The signal row just written by log_signal (autotrade open hook)."""
    with _conn() as c:
        row = c.execute("""
            SELECT * FROM signals
            WHERE symbol = ? AND status = 'OPEN'
            ORDER BY id DESC LIMIT 1
        """, (symbol,)).fetchone()
        return dict(row) if row else None
