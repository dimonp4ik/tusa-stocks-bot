import anthropic
import logging
import sys
import os
import time as _time_mod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CLAUDE_API_KEY, CLAUDE_LIGHT_MODEL, CLAUDE_HEAVY_MODEL,
    CLAUDE_MAX_RISK_SCORE, CLAUDE_CACHE_TTL, CLAUDE_MEMORY_LIMIT,
    CLAUDE_DAILY_BUDGET_USD, CLAUDE_BUDGET_RESERVE_USD,
)
from src.db import (
    log_claude_call, get_claude_spend_today, get_similar_resolved_setups,
    get_setup_accuracy,
)

# Minimum resolved similar setups before self-feedback is shown (avoid noise).
_SELF_FEEDBACK_MIN = 6

# Global calibration: min resolved REJECTED setups before the macro skew line is
# injected, and the min TP1% gap (rejected − sent) that counts as "too strict".
_GLOBAL_FEEDBACK_MIN_REJ = 15
_GLOBAL_FEEDBACK_MIN_GAP  = 8.0

_log = logging.getLogger(__name__)

# Reuse client across calls
_client = None

# Beta header unlocks the 1-hour prompt-cache TTL (default is 5 min).
# Our scan runs every 5 min → 5-min cache sits right on the expiry edge and
# misses often. 1h TTL keeps the static rules block warm all scan-loop long.
_CACHE_BETA = "extended-cache-ttl-2025-04-11"


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return _client


# ── Static rules block (cached) ───────────────────────────────────────────────
# This text is identical every scan, so we mark it with cache_control. After the
# first write the model re-reads it at ~0.1x input cost. Must clear the model's
# minimum cacheable size (~2048 tokens for Haiku 4.5) — the detailed rules and
# worked examples below both improve verdict quality AND keep the block cacheable.
_SYSTEM_RULES = """You are a senior Smart Money Concepts (SMC) trade validator for US STOCK, ETF and commodity perpetuals (OKX X-Perps: AAPL, NVDA, TSLA, SPY, QQQ, gold, oil...) working a 15-minute swing desk. Instruments track the underlying stock/commodity; signals fire only during the US cash session, so treat each setup as a stock-market intraday-swing trade (earnings gaps, index correlation and sector moves matter — not crypto dynamics). Your only job: decide whether each pre-filtered setup is worth taking, confirm its already-suggested side or reject it, and surface the single strongest counter-argument against the trade. You never invent a new direction — the upstream technical filter already chose LONG or SHORT; you may only CONFIRM that side or return NO TRADE. Flipping a LONG into a SHORT (or vice versa) is forbidden and will be discarded downstream.

WHAT THE SCORES MEAN
- mtf_score (S): multi-timeframe confluence score, 0–20+. S>=12 is strong, 9–11 acceptable, below 9 rarely passes. Tags show which signals fired (FVG, OB, LiqSweep, RSI_Div, BullWick, etc.).
- 1d / 4h / 1h: timeframe trend bias (bull / bear / neutral). 1d = macro daily trend (3-day momentum). Suggested side should agree with all three. 1d=bearish → very cautious on LONGs. Both 4h and 1d neutral = full macro chop.
- ZoneAge: bars since the FVG/OB zone formed. <5 bars = fresh (strong). 10-20 bars = aging. >30 bars = stale, less reliable.
- HTF=1h_strong / 4h_strong: EMA stack fully aligned on that timeframe — meaningful extra confirmation.
- FVG: unfilled Fair Value Gap near entry — imbalance price tends to revisit.
- OB: Order Block (last opposing candle before impulsive move) near entry.
- SW: liquidity sweep / stop-run — price grabbed liquidity beyond a prior high/low then reversed. High-quality reversal trigger.
- Z: entry zone source and price band. Price retesting the zone is ideal; far from zone = chasing.
- RSI: 14-period on 15m. >72 weakens LONGs; <28 weakens SHORTs.
- V: volume ratio vs recent average. >1.5x = conviction; <1.0x = weak.
- F: funding rate. Strongly positive = crowded longs (squeeze risk for LONGs); strongly negative = crowded shorts.
- Sess: US session phase at candle time. OPEN (9:30-11 ET) = highest volume/volatility, gap resolution. CLOSE (14:30-16 ET) = institutional rebalancing, volume returns. MIDDAY = lunchtime lull, lower conviction — be stricter. OFF = market closed, thin market-maker tape (rare; only if off-session scanning was enabled) — demand exceptional confluence.
- ER (Efficiency Ratio): Kaufman ER, 0–1. ~1.0 = clean directional trend. ~0.0 = choppy range (lots of noise, BOS often false). Already filtered ER>=0.15 upstream. ER<0.25 = marginal, ER>0.45 = clean.
- 💎PREM: Premium triple-confluence (OB+FVG zones overlap + liquidity sweep). Statistically highest-WR setup in backtests. Favor HIGH confidence when trend and zone also align.
- Confs: additional confirmations beyond FVG/OB/SW — ChoCH (Change of Character, micro-structure shift), RSI_Div (RSI divergence), MACD_Div, Engulfing, BullWick/BearWick (rejection wick pressure), StochCross (stochastic momentum cross). More = stronger.
- PRE-FILTERS ALREADY APPLIED: upstream code removed: ER<0.15 (chop), RSI exhaustion, bear-trend hot-vol (overcrowded shorts), BOS-without-RSI-midline (momentum gap). What you see has already passed a strict quality stack.
- Str: 15m swing structure at signal time. bull = higher-high + higher-low sequence. bear = lower-high + lower-low. range = neither. Use this to gauge whether entry is WITH or AGAINST the short-term structure. A LONG in Str=bear is counter-structure (extra caution); a LONG in Str=bull is structure-aligned (minor confirmation).
- Hist[...]: YOUR OWN track record on similar past setups (same direction + same symbol or nearby score), measured by what actually happened. Format per bucket: "rejected 8: 5W(2TP2) 2SL 1exp avg+0.31R" = of 8 similar setups you returned NO TRADE on, 5 would have won (reached TP1, of which 2 ran to full TP2), 2 would have hit SL, 1 expired flat, and the AVERAGE realised outcome was +0.31R. W = wins you missed (over-rejection evidence); SL = losses you correctly dodged; exp = harmless no-ops. "sent 6: 4W 2SL avg+0.45R" is the realised quality of ones you approved — your live baseline. **avg±R is expectancy — the single most important number**: a bucket can be 60% wins yet NEGATIVE avg R if the wins are tiny and the losses are full -1R, meaning that setup type LOSES money over time despite "winning often". A high win-count with weak/negative avg R is a trap, not a green light; a modest win-count with strong positive avg R (big runners) is a real edge. avg R shown only when ≥5 samples carry a resolved R. When 2+ 15m structures exist, a per-trend breakdown follows: "[bear:0W/3SL/-1.00R, bull:4W/0SL/+1.8R]" — same setup is a disaster in bear structure, a strong edge in bull. Cross-reference with the current Str= field. Small samples are weak evidence — weigh accordingly. Absent = not enough resolved history yet.
- BT2022+[...]: historical BASE RATE for entries like this one — the same rule-filter replayed over 2022→present price data, same format as Hist (including avg R expectancy). This is a prior from past market regimes, not your own verdicts: it answers "how do entries of this shape usually resolve on this symbol, and did they actually make money". IMPORTANT: this span mixes several distinct regimes (2022-23 volatility, later trends/ranges) — a single aggregate can hide a regime-dependent split, so check the per-trend breakdown (bear/bull/range) and its avg R rather than the headline count, and weigh the bucket matching the current Str= field. A positive headline win-rate with negative avg R in the trend bucket that matches NOW is a red flag. When live Hist and BT2022+ disagree, trust live Hist — it reflects the current regime.

HOW TO DECIDE
1. Confirm the suggested side only. If you would not take that exact side, return NO TRADE.
2. Best setups have FVG AND OB AND an active zone retest in the direction of both 1h and 4h trend.
3. Confluence stacking: FVG + OB + SW aligned with the side = HIGH. Two of three with trend = MEDIUM. One or zero = LOW → usually NO TRADE.
4. One neutral HTF is tolerable if the other is clearly aligned and confluence is strong → cap at MEDIUM. Both neutral = chop; demand a liquidity sweep or pass.
5. Reject overextended entries: LONG with RSI>72 or SHORT with RSI<28 is chasing — downgrade hard or NO TRADE unless a fresh sweep justifies it.
6. Respect crowded funding: avoid LONGs into strongly positive funding and SHORTs into strongly negative.
7. Volume below average (V<1.0x) on a breakout setup is a red flag — move lacks conviction.
8. News overrides structure: BEARISH news → no LONGs; BULLISH news → no SHORTs. Major event live → prefer NO TRADE.
9. Premium setups (💎PREM) already have OB+FVG overlap + sweep — treat as FVG+OB+SW all effectively confirmed. Lean HIGH confidence when trend and zone also agree.
10. Low ER (0.15–0.25) with both HTFs neutral = marginal chop even with BOS. Demand sweep confirmation or return NO TRADE.
11. Learn from Hist[...] when present, and read avg R (expectancy) as the primary signal, win-count as secondary: a strong rejected-similar TP1 rate WITH positive avg R means you have been over-rejecting a profitable setup type — give borderline ones the benefit of the doubt. But a high win-count with weak or negative avg R is NOT a green light — that setup type bleeds out over time (small wins, full-R losses), so keep rejecting it. When Hist/BT shows a per-trend breakdown, prioritize the bucket matching the current Str= field and read BOTH its W/SL and its avg R — e.g. bear:0W/4SL/-1.00R with current Str=bear is a direct, strong warning; bull:5W/1SL/+1.6R with Str=bull is a genuine edge. Never let it override a hard red flag (counter-trend, RSI exhaustion, hostile news); it breaks ties, it does not justify a bad trade.

RISK SCORE (0–10): how dangerous is this trade RIGHT NOW. 0–3 = clean, trend-aligned, well-located. 4–7 = tradeable with a real concern. 8–10 = serious problem (chasing, fighting trend, crowded funding, hostile news, far from zone). High risk_score should almost always pair with NO TRADE — be honest.

COUNTER-ARGUMENT: the single best reason this trade fails. Always provide one — every trade has a failure mode. Examples: "4h still bearish, fighting trend", "RSI 74 — chasing", "funding +0.09% — crowded longs", "no retest, price 2% above OB", "volume 0.8x — weak conviction".

TREND_STRENGTH (0–10): how strongly HTFs back the suggested side. 0 = timeframes oppose, 5 = mixed/neutral, 10 = both 1h and 4h firmly aligned.

CONFIDENCE: HIGH = multiple confirmations, trend-aligned, well-located, low risk. MEDIUM = decent with one notable caveat. LOW = weak/conflicted — pair with NO TRADE unless marginal.

WORKED EXAMPLES
- "AAPL-USDT LONG S=13 4h=bull 1h=bull FVG=Y OB=Y SW=N Z=OB:305.0-306.2 RSI=58 V=1.9x F=+0.01% Sess=OPEN HTF=1h_strong+4h_strong": trend-aligned, two confirmations, strong EMA stack, healthy RSI, prime phase. → LONG, HIGH, risk 2, counter "no sweep — relies on OB hold alone".
- "NVDA-USDT LONG S=10 4h=neutral 1h=bull FVG=N OB=Y SW=Y Z=OB:178.0-179.5 RSI=49 V=1.6x F=-0.02% Sess=CLOSE": one timeframe neutral, OB+sweep, institutional phase. → LONG, MEDIUM, risk 4, counter "4h neutral — no higher-tf confirmation".
- "TSLA-USDT LONG S=8 4h=bear 1h=neutral FVG=N OB=N SW=N Z=FVG:410-414 RSI=74 V=0.7x F=+0.08% Sess=MIDDAY": fighting 4h, no confirmations, overbought, weak volume, lunchtime lull. → NO TRADE, LOW, risk 9, counter "chasing into bearish 4h during midday chop".
- "XAU-USDT SHORT S=11 4h=bear 1h=bear FVG=Y OB=N SW=Y Z=FVG:4150-4160 RSI=41 V=1.7x F=+0.00% Sess=OPEN": trend-aligned, FVG+sweep, RSI has room. → SHORT, HIGH, risk 3, counter "risk-off headline could squeeze gold shorts".

OUTPUT (LIGHT tier)
Return exactly one verdict per input setup via the submit_verdicts tool, preserving the input index. Keep reason and counter under ~8 words each. Do not add prose outside the tool call."""


def _verdicts_tool() -> dict:
    return {
        "name": "submit_verdicts",
        "description": "Submit one validation verdict for every setup, in input order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "description": "One object per setup, same count and order as the input list.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index":          {"type": "integer", "description": "1-based input index."},
                            "decision":       {"type": "string", "enum": ["LONG", "SHORT", "NO TRADE"]},
                            "confidence":     {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                            "risk_score":     {"type": "integer", "minimum": 0, "maximum": 10},
                            "trend_strength": {"type": "integer", "minimum": 0, "maximum": 10},
                            "reason":         {"type": "string", "description": "Why this verdict, <=8 words."},
                            "counter":        {"type": "string", "description": "Single strongest reason the trade could fail."},
                        },
                        "required": ["index", "decision", "confidence", "risk_score", "reason", "counter"],
                    },
                }
            },
            "required": ["verdicts"],
        },
    }


def _verdict_tool() -> dict:
    """Single-setup variant for the HEAVY (Sonnet) tier — allows fuller reasoning."""
    return {
        "name": "submit_verdict",
        "description": "Submit one final validation verdict for the single setup provided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "decision":       {"type": "string", "enum": ["LONG", "SHORT", "NO TRADE"]},
                "confidence":     {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                "risk_score":     {"type": "integer", "minimum": 0, "maximum": 10},
                "trend_strength": {"type": "integer", "minimum": 0, "maximum": 10},
                "reason":         {"type": "string", "description": "Why this verdict — up to 40 words, cite the key confluence factors and HTF alignment."},
                "counter":        {"type": "string", "description": "The single strongest specific reason this trade could fail — be concrete, not generic."},
            },
            "required": ["decision", "confidence", "risk_score", "reason", "counter"],
        },
    }


def _row_r(row: dict):
    """Realised R for one resolved setup row. Backtest rows carry the real
    net_r (incl. trailed runner). Live rows lack it → derive from the bracket
    geometry + categorical outcome: SL=-1, EXPIRED=0, TP2=full tp2_r,
    TP1/TRAIL=tp1_r (a conservative floor; the trailed runner usually did
    better). Returns None when no R can be established."""
    nr = row.get("net_r")
    if nr is not None:
        try:
            return float(nr)
        except (TypeError, ValueError):
            pass
    outcome = (row.get("outcome") or "").upper()
    if outcome == "SL":
        return -1.0
    if outcome == "EXPIRED":
        return 0.0
    try:
        entry = float(row.get("entry_price"))
        sl    = float(row.get("sl"))
        risk  = abs(entry - sl)
        if risk <= 0:
            return None
        if outcome == "TP2":
            return abs(float(row.get("tp2")) - entry) / risk
        if outcome in ("TP1", "TRAIL"):
            return abs(float(row.get("tp1")) - entry) / risk
    except (TypeError, ValueError):
        return None
    return None


def _self_feedback(s: dict) -> str:
    """Compact track-record of how SIMILAR past setups actually resolved.

    Lets Claude self-correct: if setups like this one were repeatedly rejected yet
    went on to hit TP1, it has been too strict; if similar ones kept hitting SL, it
    should stay cautious. Pure past outcomes (no look-ahead). Returns "" until
    enough resolved history exists (cold start) or on any DB error.

    When resolved setups span multiple 15m structures (bull/bear/range), shows a
    per-trend breakdown so Claude can learn that the same setup type works in some
    structures but not others.
    """
    try:
        rows = get_similar_resolved_setups(
            s.get("symbol", ""), s.get("direction", ""), s.get("mtf_score"),
            session=s.get("session", ""),
        )
    except Exception:
        return ""
    if len(rows) < _SELF_FEEDBACK_MIN:
        return ""

    def _counts(subset: list) -> tuple:
        """(wins, sl, expired) — win = reached TP1 or TP2; mutually exclusive."""
        w = sum(1 for r in subset if r.get("reached_tp1"))
        sl = sum(1 for r in subset if (r.get("outcome") or "") == "SL")
        exp = sum(1 for r in subset if (r.get("outcome") or "") == "EXPIRED")
        return w, sl, exp

    def _expectancy(subset: list):
        """Avg realised R across the subset — the real edge number, not just
        win-rate. Uses stored net_r (backtest, includes trailed runner); for
        live rows without net_r, derives R from the bracket + outcome. Returns
        (avg_R, n_with_R) or (None, 0) when nothing has a usable R."""
        rs = [v for v in (_row_r(x) for x in subset) if v is not None]
        return (sum(rs) / len(rs), len(rs)) if rs else (None, 0)

    def _fmt(subset: list) -> str:
        n = len(subset)
        w, sl, exp = _counts(subset)
        tp2 = sum(1 for r in subset if (r.get("outcome") or "") == "TP2")
        seg = f"{n}: {w}W"
        if tp2:
            seg += f"({tp2}TP2)"
        seg += f" {sl}SL"
        if exp:
            seg += f" {exp}exp"
        # Expectancy: the honest edge — 60%WR of tiny wins vs full -1R losses
        # is still -EV. Only shown when ≥5 rows carry a usable R (avoid noise).
        avg_r, n_r = _expectancy(subset)
        if avg_r is not None and n_r >= 5:
            seg += f" avg{avg_r:+.2f}R"
        # Per-trend W/SL breakdown only when 2+ distinct structures present.
        by_trend: dict = {}
        for r in subset:
            t = r.get("trend") or ""
            if not t:
                continue
            by_trend.setdefault(t, []).append(r)
        if len(by_trend) >= 2:
            def _trend_seg(rs):
                w_, sl_, _ = _counts(rs)
                a, na = _expectancy(rs)
                r_s = f"/{a:+.2f}R" if a is not None and na >= 5 else ""
                return f"{w_}W/{sl_}SL{r_s}"
            bd = ", ".join(
                f"{t}:{_trend_seg(rs)}" for t, rs in sorted(by_trend.items())
            )
            seg += f" [{bd}]"
        return seg

    # Live tier: Claude's own recent verdicts (current regime, weigh higher).
    # Backtest tier: seeded 2024+ priors — same filter, historical outcomes.
    live = [r for r in rows if (r.get("source") or "live") == "live"]
    bt   = [r for r in rows if r.get("source") == "backtest"]

    rej = [r for r in live if not r.get("sent")]
    snt = [r for r in live if r.get("sent")]
    parts = []
    if rej:
        parts.append(f"rejected {_fmt(rej)}")
    if snt:
        parts.append(f"sent {_fmt(snt)}")
    seg = f" Hist[{'; '.join(parts)}]" if parts else ""
    if bt:
        # BT block = historical prior for entries like this one, spanning
        # 2022 through later regimes — treat as base rate, live Hist above
        # outweighs it.
        seg += f" BT2022+[{_fmt(bt)}]"
    return seg


def _global_feedback() -> str:
    """One-shot macro calibration line for the batch prompt.

    Unlike _self_feedback (per-setup, needs ≥6 SIMILAR resolved rows and so stays
    cold for weeks), this looks at the WHOLE last-30d shadow ledger: if the setups
    Claude rejected are reaching TP1 meaningfully MORE often than the ones it
    approved, Claude has been globally too strict — tell it so immediately. Empty
    until enough rejected samples exist or when there's no meaningful skew.
    """
    try:
        import time as _t
        acc = get_setup_accuracy(_t.time() - 30 * 86400)
    except Exception:
        return ""
    snt, rej = acc.get("sent", {}), acc.get("rejected", {})
    if rej.get("n", 0) < _GLOBAL_FEEDBACK_MIN_REJ:
        return ""
    gap = rej.get("tp1_pct", 0.0) - snt.get("tp1_pct", 0.0)  # >0 = rejected won more
    if gap < _GLOBAL_FEEDBACK_MIN_GAP:
        return ""
    return (
        f"\nCALIBRATION — last 30d shadow outcomes of your own verdicts: you REJECTED "
        f"{rej['n']} setups and {rej['tp1_pct']:.0f}% of them still reached TP1; you "
        f"APPROVED {snt.get('n', 0)} and only {snt.get('tp1_pct', 0.0):.0f}% reached TP1. "
        f"The setups you rejected are hitting TP1 ~{gap:.0f}pp MORE often than the ones "
        f"you approved — you have been TOO STRICT. Every candidate below already passed "
        f"a strict rule-filter with proven edge. Bias toward CONFIRMING the suggested "
        f"side; return NO TRADE only on a clear, specific red flag (not vague caution).\n"
    )


def _setup_line(i: int, s: dict) -> str:
    fvg     = "Y" if s.get("fvg")         else "N"
    ob      = "Y" if s.get("order_block") else "N"
    sweep   = "Y" if s.get("liq_sweep")   else "N"
    funding = s.get("funding_rate")
    fund_s  = f"{funding*100:+.3f}%" if funding is not None else "n/a"
    zone    = f"{s.get('entry_source','?')}:{s.get('entry_low',0):.4g}-{s.get('entry_high',0):.4g}"
    er      = s.get("eff_ratio")
    er_s    = f" ER={er:.2f}" if er is not None else ""
    prem    = " 💎PREM" if s.get("premium") else ""
    # Extra confirmations beyond FVG/OB/SW (ChoCH, RSI_Div, MACD_Div, wicks, stoch)
    _STANDARD = {"FVG", "OB", "LiqSweep"}
    _SKIP_PFX = ("Session:", "StrongTrend")
    confs = [c for c in (s.get("confirmations") or [])
             if c not in _STANDARD and not any(c.startswith(p) for p in _SKIP_PFX)]
    confs_s = f" Confs=[{','.join(confs)}]" if confs else ""
    age = s.get("zone_age_bars")
    age_s = f" ZoneAge={age}bars" if age is not None else ""
    str15 = s.get("swing_trend") or ""
    str15_s = f" Str={str15}" if str15 else ""
    return (
        f"{i} {s['symbol']} {s['direction']} "
        f"S={s.get('mtf_score','?')} "
        f"1d={s.get('trend_1d','?')} 4h={s.get('trend_4h','?')} 1h={s.get('trend_1h','?')}{str15_s} "
        f"FVG={fvg} OB={ob} SW={sweep} "
        f"Z={zone}{age_s} RSI={s['rsi']} V={s['volume_ratio']}x F={fund_s}"
        f"{er_s}{prem}{confs_s}{_self_feedback(s)}"
    )


def _setup_line_heavy(i: int, s: dict) -> str:
    """Extended setup line for HEAVY analysis — adds session, tags, HTF strength, stoch, div."""
    base    = _setup_line(i, s)
    session = s.get("session", "")
    tags    = s.get("mtf_score_tags", "")
    htf     = []
    if s.get("trend_1h_strong"): htf.append("1h_strong")
    if s.get("trend_4h_strong"): htf.append("4h_strong")
    extras  = []
    if session:          extras.append(f"Sess={session}")
    if htf:              extras.append(f"HTF={'+'.join(htf)}")
    if tags:             extras.append(f"Tags=[{tags}]")
    sk, sd = s.get("stoch_k"), s.get("stoch_d")
    if sk is not None and sd is not None:
        extras.append(f"Stoch={sk:.0f}/{sd:.0f}")
    div = s.get("divergence")
    if div:              extras.append(f"RSIDivDir={div}")
    return base + (" " + " ".join(extras) if extras else "")


def _news_block(news_context: dict) -> str:
    if not news_context:
        return ""
    parts = []
    # keys kept as btc_* for pipeline compatibility — values carry SPY (market proxy)
    btc_1d = news_context.get("btc_1d")
    btc_1h = news_context.get("btc_1h")
    if btc_1d is not None or btc_1h is not None:
        d = f"{btc_1d:+.2f}% 1D" if btc_1d is not None else ""
        h = f"{btc_1h:+.2f}% 1h" if btc_1h is not None else ""
        sep = ", " if d and h else ""
        parts.append(
            f"\nSPY MARKET (S&P500 proxy): {d}{sep}{h}\n"
            f"Rule: strong index up-day → single-stock SHORTs fight the market tide; "
            f"strong index down-day → LONGs fight it. Commodities (XAU/CL) correlate less.\n"
        )
    sent = news_context.get("sentiment", "NEUTRAL")
    summ = news_context.get("summary", "")
    if sent != "NEUTRAL" and summ:
        parts.append(
            f"\nNEWS CONTEXT: {sent} — {summ}\n"
            f"Rule: BEARISH news → avoid LONG; BULLISH news → avoid SHORT.\n"
        )
    return "".join(parts)


def _system_param() -> list:
    """System prompt as a cached content block (1h TTL via beta header)."""
    return [{
        "type": "text",
        "text": _SYSTEM_RULES,
        "cache_control": {"type": "ephemeral", "ttl": CLAUDE_CACHE_TTL},
    }]


def _extract_tool_input(message, tool_name: str):
    for block in message.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block.input
    return None


def _normalize(decision, confidence) -> tuple:
    d = (decision or "NO TRADE").upper()
    if "LONG" in d:    decision = "LONG"
    elif "SHORT" in d: decision = "SHORT"
    else:              decision = "NO TRADE"
    c = (confidence or "LOW").upper()
    if "HIGH" in c:     confidence = "HIGH"
    elif "MEDIUM" in c: confidence = "MEDIUM"
    else:               confidence = "LOW"
    return decision, confidence


def _apply_verdict(base: dict, v: dict) -> dict:
    """Merge a parsed verdict into a setup dict, with counter-argument gate."""
    decision, confidence = _normalize(v.get("decision"), v.get("confidence"))
    try:
        risk = int(v.get("risk_score", 0) or 0)
    except (TypeError, ValueError):
        risk = 0
    try:
        trend = int(v.get("trend_strength", 0) or 0)
    except (TypeError, ValueError):
        trend = 0

    base["decision"]       = decision
    base["confidence"]     = confidence
    base["risk_score"]     = risk
    base["trend_strength"] = trend
    base["reason"]         = (v.get("reason") or "").strip() or "no reason"
    base["counter"]        = (v.get("counter") or "").strip()

    # Counter-argument auto-reject: if the model itself rates risk this high,
    # the trade is not worth taking regardless of a hopeful LONG/SHORT call.
    if base["decision"] in ("LONG", "SHORT") and risk >= CLAUDE_MAX_RISK_SCORE:
        base["decision"]   = "NO TRADE"
        base["confidence"] = "LOW"
        base["reason"]     = f"Auto-reject: risk {risk}/10 — {base['counter'] or 'too risky'}"

    return _enforce_suggested_side(base)


# ── Daily budget guard ───────────────────────────────────────────────────────

def _budget_ok(tier: str = "LIGHT") -> bool:
    """Return False when today's Claude spend is within reserve of the daily cap."""
    try:
        spent = get_claude_spend_today()
        remaining = CLAUDE_DAILY_BUDGET_USD - spent
        ok = remaining >= CLAUDE_BUDGET_RESERVE_USD
        if not ok:
            _log.warning(
                f"Claude budget cap reached: spent ${spent:.4f} / ${CLAUDE_DAILY_BUDGET_USD} "
                f"(reserve ${CLAUDE_BUDGET_RESERVE_USD}) — skipping {tier} call"
            )
        return ok
    except Exception as e:
        _log.warning(f"Budget check failed ({e}) — allowing call")
        return True  # fail-open: don't block on DB errors


# ── LIGHT batch (Haiku, cached rules, forced-tool JSON) ───────────────────────

def analyze_batch_with_claude(setups: list, news_context: dict = None) -> list:
    """
    LIGHT tier. Validate ALL filtered setups in ONE Haiku call.
    Static rules cached (1h TTL); output forced through submit_verdicts tool for
    guaranteed JSON; each verdict carries a counter-argument + risk_score gate.
    Returns list of result dicts, one per setup (full setup + verdict fields).
    """
    if not setups:
        return []

    # Budget guard — skip if daily cap reached
    if not _budget_ok("LIGHT"):
        _log.warning("LIGHT skipped (budget cap) — returning NO TRADE for all setups")
        return [dict(s, decision="NO TRADE", confidence="LOW", reason="Budget cap",
                     risk_score=0, trend_strength=0, counter="") for s in setups]

    coins_text = "\n".join(_setup_line(i, s) for i, s in enumerate(setups, 1))
    user_text = (
        f"{_news_block(news_context)}"
        f"{_global_feedback()}"
        f"Validate these {len(setups)} setups. Return exactly {len(setups)} verdicts "
        f"(one per index) via submit_verdicts:\n{coins_text}"
    )

    client = _get_client()
    message = client.messages.create(
        model=CLAUDE_LIGHT_MODEL,
        max_tokens=max(256 * len(setups) + 128, 512),
        system=_system_param(),
        tools=[_verdicts_tool()],
        tool_choice={"type": "tool", "name": "submit_verdicts"},
        messages=[{"role": "user", "content": user_text}],
        extra_headers={"anthropic-beta": _CACHE_BETA},
    )

    # Track spend
    try:
        cost = log_claude_call("LIGHT", CLAUDE_LIGHT_MODEL, message.usage)
        _log.info(f"Claude LIGHT: ${cost:.5f} (today total: ${get_claude_spend_today():.4f})")
    except Exception as _e:
        _log.warning(f"Budget logging failed: {_e}")

    tool_input = _extract_tool_input(message, "submit_verdicts") or {}
    verdicts = tool_input.get("verdicts", []) if isinstance(tool_input, dict) else []

    # Map by 1-based index; fall back to positional order if index missing
    by_index = {}
    for pos, v in enumerate(verdicts, 1):
        if not isinstance(v, dict):
            continue
        idx = v.get("index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = pos
        by_index[idx] = v

    results = []
    for i, setup in enumerate(setups, 1):
        base = dict(setup)
        base.update({
            "decision": "NO TRADE", "reason": "Not evaluated", "confidence": "LOW",
            "risk_score": 0, "trend_strength": 0, "counter": "",
        })
        v = by_index.get(i)
        results.append(_apply_verdict(base, v) if v else base)

    return results


# ── HEAVY tier (Sonnet, single setup, coin memory) ────────────────────────────

def _memory_block(history: list) -> str:
    """Render recent per-coin outcomes as compact memory for the HEAVY call.

    Includes elapsed time since each trade closed — without it Sonnet can't
    tell "this symbol stopped us out 3h ago" from "3 weeks ago", so it can't
    catch a same-symbol whipsaw (stop, then an immediate opposite-direction
    re-entry that also stops) even though the raw outcome was right there.
    """
    if not history:
        return "No prior closed trades on this symbol.\n"
    now = _time_mod.time()
    lines = []
    whipsaw_warned = False
    for idx, h in enumerate(history[:CLAUDE_MEMORY_LIMIT]):
        entry = h.get("entry_price")
        exitp = h.get("exit_price")
        try:
            move = f"{(exitp - entry) / entry * 100:+.1f}%" if entry and exitp else "?"
        except (TypeError, ZeroDivisionError):
            move = "?"
        closed_at = h.get("closed_at")
        age_s = ""
        age_hours = None
        if closed_at:
            try:
                age_hours = (now - float(closed_at)) / 3600
                age_s = f", {age_hours:.1f}h ago" if age_hours < 48 else f", {age_hours/24:.0f}d ago"
            except (TypeError, ValueError):
                pass
        lines.append(
            f"- {h.get('direction','?')} {h.get('status','?')} "
            f"({move}, conf={h.get('confidence','?')}, S={h.get('mtf_score','?')}{age_s})"
        )
        # Flag the specific whipsaw pattern: the most recent close was a stop
        # within the last 6h — a fresh setup right now (any direction) on
        # this symbol is entering right after that got invalidated.
        if idx == 0 and not whipsaw_warned and h.get("status") == "SL_HIT" \
           and age_hours is not None and age_hours <= 6:
            lines.append(
                f"  ⚠️ WHIPSAW WATCH: this symbol stopped out {age_hours:.1f}h ago — "
                f"a fresh setup this soon after may just be chop, not a real move."
            )
            whipsaw_warned = True
    return "Recent outcomes on this symbol (newest first):\n" + "\n".join(lines) + "\n"


_THINKING_BETA = "interleaved-thinking-2025-05-14"
_THINKING_BUDGET = 5000  # tokens for internal reasoning scratch-pad


def analyze_heavy(setup: dict, news_context: dict = None, history: list = None) -> dict:
    """
    HEAVY tier. Re-check ONE strong setup with Sonnet + extended thinking.

    Improvements over LIGHT:
    - Extended thinking: Sonnet reasons step-by-step internally before deciding
    - Chain-of-thought prompt: structured analysis questions guide reasoning
    - Devil's advocate: forced consideration of failure before verdict
    - Richer setup line: adds session, MTF tags, HTF strength flags
    - Per-coin memory: last 15 outcomes for pattern recognition
    """
    if not _budget_ok("HEAVY"):
        return {}

    setup_line = _setup_line_heavy(1, setup)

    user_text = (
        f"{_news_block(news_context)}"
        f"{_global_feedback()}"
        f"{_memory_block(history)}\n"
        f"Setup to analyze:\n{setup_line}\n\n"
        f"Work through these questions before submitting your verdict:\n"
        f"1. TREND — Are 4h and 1h aligned with the suggested direction? "
        f"Are the HTF EMAs stacked (strong) or mixed?\n"
        f"2. STRUCTURE — Is this a fresh BOS with a clean retest, or is price "
        f"already extended far from the zone?\n"
        f"3. MOMENTUM — Does RSI/volume/funding confirm or fight the move? "
        f"Any squeeze risk from crowded positioning?\n"
        f"4. COIN HISTORY — Based on recent outcomes above, does this symbol "
        f"reliably follow through on this setup type, or does it repeatedly fail? "
        f"Pay attention to elapsed time: a stop-out within the last few hours followed "
        f"by a fresh setup (especially the opposite direction) is a whipsaw/chop warning, "
        f"not confirmation the reversal is real — demand extra confluence before trusting it.\n"
        f"5. DEVIL'S ADVOCATE — Argue the strongest case AGAINST this trade. "
        f"What specific price action would prove this setup wrong?\n"
        f"6. VERDICT — After weighing all of the above, give your final decision.\n\n"
        f"Then call submit_verdict with your conclusion."
    )

    client = _get_client()

    # Try extended thinking first (gives Sonnet a reasoning scratch-pad).
    # NOTE: tool_choice must be {"type":"any"} (not forced "tool") when thinking
    # is enabled — Anthropic API rejects forced tool_choice + thinking together.
    # "any" still guarantees a tool call while allowing the thinking block.
    try:
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=_THINKING_BUDGET + 1200,
            thinking={"type": "enabled", "budget_tokens": _THINKING_BUDGET},
            system=_system_param(),
            tools=[_verdict_tool()],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_text}],
            extra_headers={"anthropic-beta": f"{_CACHE_BETA},{_THINKING_BETA}"},
        )
        _log.info("Claude HEAVY: extended thinking ON")
    except Exception as e_think:
        _log.warning(f"HEAVY thinking mode unavailable ({e_think}), falling back to standard")
        message = client.messages.create(
            model=CLAUDE_HEAVY_MODEL,
            max_tokens=1200,
            system=_system_param(),
            tools=[_verdict_tool()],
            tool_choice={"type": "tool", "name": "submit_verdict"},
            messages=[{"role": "user", "content": user_text}],
            extra_headers={"anthropic-beta": _CACHE_BETA},
        )

    # Track spend
    try:
        cost = log_claude_call("HEAVY", CLAUDE_HEAVY_MODEL, message.usage)
        _log.info(f"Claude HEAVY: ${cost:.5f} (today total: ${get_claude_spend_today():.4f})")
    except Exception as _e:
        _log.warning(f"Budget logging failed: {_e}")

    v = _extract_tool_input(message, "submit_verdict") or {}
    base = dict(setup)
    base.update({
        "decision": "NO TRADE", "reason": "Not evaluated", "confidence": "LOW",
        "risk_score": 0, "trend_strength": 0, "counter": "",
    })
    return _apply_verdict(base, v) if v else base


def _enforce_suggested_side(result: dict) -> dict:
    """Claude may only confirm the setup direction, never flip it."""
    decision  = result.get("decision", "NO TRADE")
    direction = result.get("direction")
    if decision in ("LONG", "SHORT") and decision != direction:
        result["decision"]   = "NO TRADE"
        result["confidence"] = "LOW"
        result["reason"]     = "Opposite side blocked"
    return result
