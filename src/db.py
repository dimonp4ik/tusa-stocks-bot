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
)

ACTIVE_STATUSES = ("OPEN", "TP1_PARTIAL")
FINAL_STATUSES  = ("TP2_HIT", "BREAKEVEN", "SL_HIT", "EXPIRED", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")
TP1_STATUSES    = ("TP1_PARTIAL", "TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")
PROFIT_STATUSES = ("TP2_HIT", "BREAKEVEN", "TP1_EXPIRED", "TP1_HIT", "TP1_TRAIL")


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
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
            "mtf_score":     "INTEGER",
            "mtf_score_max": "INTEGER",
            "premium":       "INTEGER DEFAULT 0",
            "atr":           "REAL",
            "realized_r":    "REAL",
            "runner_trail_atr_mult": "REAL",
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


def log_signal(analysis: dict, tp1: float, tp2: float, sl: float):
    """Insert a new signal into DB. Status starts as OPEN."""
    with _conn() as c:
        c.execute("""
            INSERT INTO signals (
                symbol, direction, entry_price, tp1, tp2, sl, opened_at, status,
                confidence, reason, entry_low, entry_high, entry_source, market_price,
                mtf_score, mtf_score_max, premium, atr
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis["symbol"], analysis["direction"], analysis["current_price"],
            tp1, tp2, sl, time_mod.time(),
            analysis.get("confidence", "?"), analysis.get("reason", ""),
            analysis.get("entry_low"), analysis.get("entry_high"),
            analysis.get("entry_source"), analysis.get("market_price"),
            analysis.get("mtf_score"), analysis.get("mtf_score"),
            1 if analysis.get("premium") else 0,
            analysis.get("atr"),
        ))


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


def _status_to_r(status: str) -> float:
    """Approximate R for symbol-level blocking."""
    if status == "TP2_HIT":
        return 1.5   # 50% at 1R + 50% at 2R
    if status == "TP1_TRAIL":
        return 0.75  # TP1 taken + trailed runner in profit
    if status in ("BREAKEVEN", "TP1_EXPIRED", "TP1_HIT"):
        return 0.5
    if status == "SL_HIT":
        return -1.0
    return 0.0


def get_symbol_performance(symbol: str, lookback: int = None) -> dict:
    """Return recent closed-signal performance for one symbol."""
    lookback = lookback or AUTO_BLOCK_LOOKBACK_TRADES
    placeholders = ",".join("?" for _ in FINAL_STATUSES)
    with _conn() as c:
        rows = c.execute(
            f"SELECT status FROM signals WHERE symbol = ? AND status IN ({placeholders})"
            f" ORDER BY opened_at DESC LIMIT ?",
            [symbol, *FINAL_STATUSES, lookback],
        ).fetchall()

    statuses = [r["status"] for r in rows]
    rs = [_status_to_r(s) for s in statuses]
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


def _status_r(status: str) -> float:
    """R value of a closed trade outcome (fixed R model, TP1=1.0R TP2=2.0R SL=1R).

    NOTE: for TP1_TRAIL the real R is variable and stored in the realized_r column;
    this fixed value is only a fallback when realized_r is missing.
    """
    # TP2: 50% closed at TP1 (0.5R) + 50% at TP2 (1.0R) = 1.5R
    if status == "TP2_HIT":    return  1.50
    # Trailed runner — fallback estimate (real value comes from realized_r)
    if status == "TP1_TRAIL":  return  0.75
    # TP1 only outcomes: 50% at TP1 = 0.5R
    if status in ("TP1_HIT", "BREAKEVEN", "TP1_EXPIRED"): return 0.5
    # Full SL before TP1
    if status == "SL_HIT":     return -1.00
    # Expired before any TP — no profit, small fee drag (treat as 0)
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
                 session, entry_source, trend,
                 oi_delta_pct, oi_regime, oi_confirms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
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
            analysis.get("swing_trend", ""),
            analysis.get("oi_delta_pct"),
            analysis.get("oi_regime"),
            analysis.get("oi_confirms"),
        ))
        return cur.lastrowid


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
               WHERE resolved=0 AND sl IS NOT NULL
                 AND ts <= ? AND ts >= ?
               ORDER BY ts ASC LIMIT ?""",
            (now - 900, now - max_age_sec, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def mark_setup_resolved(setup_id: int, outcome: str,
                        reached_tp1: int, reached_tp2: int) -> None:
    """Record the shadow outcome of a tracked setup."""
    with _conn() as c:
        c.execute(
            """UPDATE setup_log
               SET outcome=?, reached_tp1=?, reached_tp2=?, resolved=1, resolved_ts=?
               WHERE id=?""",
            (outcome, int(reached_tp1), int(reached_tp2), time_mod.time(), setup_id),
        )


def get_setup_accuracy(since_ts: float) -> dict:
    """Aggregate resolved-setup outcomes since since_ts, split by sent vs rejected.

    Returns counts and rates so the admin can see whether Claude's gate (the
    rejected bucket) actually has a worse outcome than what it let through.
    """
    out = {"sent": {}, "rejected": {}}
    with _conn() as c:
        for sent_val, key in ((1, "sent"), (0, "rejected")):
            rows = c.execute(
                """SELECT outcome, reached_tp1, reached_tp2 FROM setup_log
                   WHERE resolved=1 AND ts >= ? AND sent=?
                     AND COALESCE(source,'live')='live'""",
                (since_ts, sent_val),
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
    """
    since = time_mod.time() - lookback_days * 86400
    try:
        score = int(mtf_score or 0)
    except (TypeError, ValueError):
        score = 0
    with _conn() as c:
        live = c.execute(
            """SELECT symbol, direction, mtf_score, session, entry_source,
                      decision, sent, outcome, reached_tp1, reached_tp2, ts, trend,
                      COALESCE(source,'live') AS source
               FROM setup_log
               WHERE resolved=1 AND ts >= ? AND direction=?
                 AND COALESCE(source,'live')='live'
                 AND (symbol=? OR ABS(COALESCE(mtf_score,0) - ?) <= 2)
               ORDER BY ts DESC LIMIT ?""",
            (since, direction, symbol, score, limit),
        ).fetchall()
        bt = c.execute(
            """SELECT symbol, direction, mtf_score, session, entry_source,
                      decision, sent, outcome, reached_tp1, reached_tp2, ts, trend,
                      source
               FROM setup_log
               WHERE resolved=1 AND direction=? AND source='backtest'
                 AND (symbol=? OR ABS(COALESCE(mtf_score,0) - ?) <= 1)
               ORDER BY ts DESC LIMIT ?""",
            (direction, symbol, score, bt_limit),
        ).fetchall()
        return [dict(r) for r in live] + [dict(r) for r in bt]


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
                c.execute("""
                    INSERT INTO setup_log
                        (ts, symbol, direction, entry_price, tp1, tp2, sl,
                         mtf_score, decision, confidence, risk_score, reason, sent,
                         session, entry_source, trend,
                         outcome, reached_tp1, reached_tp2, resolved, resolved_ts,
                         source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'BACKTEST', NULL, '', 1,
                            ?, ?, ?, ?, ?, ?, 1, ?, 'backtest')
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
                ))
                ins += 1
            except Exception:
                continue
    return ins


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
