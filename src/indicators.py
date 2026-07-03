"""
Pure Python technical indicators + Smart Money Concepts (SMC).
No pandas, no numpy — works on any Python version.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMC_SWING_LOOKBACK, SMC_FVG_MIN_PCT, SMC_OB_LOOKBACK, ATR_PERIOD, EFF_RATIO_LOOKBACK, PD_TREND_GATE


# ── Basic indicators ──────────────────────────────────────────────────────────

def calculate_ema(values: list, period: int) -> list:
    """Exponential Moving Average."""
    k = 2 / (period + 1)
    ema = [values[0]]
    for v in values[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def calculate_atr(highs: list, lows: list, closes: list, period: int = ATR_PERIOD) -> float:
    """
    Average True Range — measures volatility.
    Returns latest ATR value (in price units).
    """
    if len(closes) < period + 1:
        return 0.0

    trs = []
    for i in range(1, len(closes)):
        h_l    = highs[i] - lows[i]
        h_pc   = abs(highs[i] - closes[i - 1])
        l_pc   = abs(lows[i]  - closes[i - 1])
        trs.append(max(h_l, h_pc, l_pc))

    # Wilder smoothing
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period

    return atr


def calculate_stoch_rsi(closes: list, rsi_period: int = 14,
                        stoch_period: int = 14,
                        smooth_k: int = 3, smooth_d: int = 3) -> tuple:
    """
    Stochastic RSI — normalises RSI into 0-100 oscillator.
    Returns (k, d):
      k < 20 and rising  → oversold reversal (LONG signal)
      k > 80 and falling → overbought reversal (SHORT signal)
      k crosses d        → momentum confirmation
    """
    needed = rsi_period + stoch_period + smooth_k + smooth_d + 2
    if len(closes) < needed:
        return 50.0, 50.0

    # Build RSI series
    rsi_values = []
    for i in range(rsi_period, len(closes)):
        rsi_values.append(calculate_rsi(closes[:i + 1], rsi_period))

    if len(rsi_values) < stoch_period:
        return 50.0, 50.0

    # Raw %K
    raw_k = []
    for i in range(stoch_period, len(rsi_values) + 1):
        w  = rsi_values[i - stoch_period:i]
        lo = min(w); hi = max(w)
        raw_k.append((rsi_values[i - 1] - lo) / (hi - lo) * 100 if hi != lo else 50.0)

    if len(raw_k) < smooth_k + smooth_d:
        return 50.0, 50.0

    # Smooth %K
    k_line = [sum(raw_k[i - smooth_k:i]) / smooth_k
              for i in range(smooth_k, len(raw_k) + 1)]

    if len(k_line) < smooth_d:
        return k_line[-1], k_line[-1]

    d = sum(k_line[-smooth_d:]) / smooth_d
    return round(k_line[-1], 2), round(d, 2)


def analyze_wicks(opens: list, highs: list, lows: list, closes: list,
                  lookback: int = 6) -> dict:
    """
    Detect buying / selling pressure via candle wicks.

    bull_pressure : lower wicks dominate last N candles (buyers rejecting lows)
    bear_pressure : upper wicks dominate (sellers rejecting highs)
    rejection     : 'bullish' | 'bearish' | None  (last candle pin-bar)
    """
    n = len(closes)
    if n < lookback + 1:
        return {"bull_pressure": False, "bear_pressure": False, "rejection": None}

    upper_wicks, lower_wicks = [], []
    for i in range(n - lookback, n):
        top    = max(opens[i], closes[i])
        bottom = min(opens[i], closes[i])
        upper_wicks.append(highs[i] - top)
        lower_wicks.append(bottom - lows[i])

    avg_u = sum(upper_wicks) / lookback
    avg_l = sum(lower_wicks) / lookback

    bull_pressure = avg_l > avg_u * 1.5
    bear_pressure = avg_u > avg_l * 1.5

    # Last-candle pin bar
    rng = highs[-1] - lows[-1]
    if rng > 0:
        u = highs[-1] - max(opens[-1], closes[-1])
        l = min(opens[-1], closes[-1]) - lows[-1]
        if   l / rng >= 0.4: rejection = "bullish"
        elif u / rng >= 0.4: rejection = "bearish"
        else:                rejection = None
    else:
        rejection = None

    return {"bull_pressure": bull_pressure,
            "bear_pressure": bear_pressure,
            "rejection":     rejection}


def detect_rsi_divergence(closes: list, highs: list, lows: list,
                          lookback: int = 30) -> str | None:
    """
    RSI divergence over last `lookback` candles.

    Bullish  : price makes lower low, RSI makes higher low → reversal UP
    Bearish  : price makes higher high, RSI makes lower high → reversal DOWN

    Returns 'bullish', 'bearish', or None.
    """
    if len(closes) < lookback + 16:
        return None

    # Build RSI series for the window
    src = closes[-(lookback + 14):]
    rsi_ser = [calculate_rsi(src[:i + 1], 14) for i in range(14, len(src))]
    if len(rsi_ser) < lookback:
        return None

    price_w = closes[-lookback:]
    high_w  = highs[-lookback:]
    low_w   = lows[-lookback:]
    rsi_w   = rsi_ser[-lookback:]

    mid = lookback // 2

    # Bearish: higher price high + lower RSI high
    ph1, ph2 = max(high_w[:mid]),  max(high_w[mid:])
    rh1, rh2 = max(rsi_w[:mid]),   max(rsi_w[mid:])
    if ph2 > ph1 * 1.001 and rh2 < rh1 * 0.985:
        return "bearish"

    # Bullish: lower price low + higher RSI low
    pl1, pl2 = min(low_w[:mid]),   min(low_w[mid:])
    rl1, rl2 = min(rsi_w[:mid]),   min(rsi_w[mid:])
    if pl2 < pl1 * 0.999 and rl2 > rl1 * 1.015:
        return "bullish"

    return None


def calculate_rsi(closes: list, period: int = 14) -> float:
    """RSI — returns only the latest value."""
    if len(closes) < period + 2:
        return 50.0

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - 100 / (1 + rs)


# ── Volatility regime ─────────────────────────────────────────────────────────

def volatility_regime(highs: list, lows: list, closes: list,
                      lookback: int = 50) -> dict:
    """
    Normalised-volatility regime via per-candle True Range as % of price.

    atr_pct : recent volatility (avg of last 3 candles' TR%) — the "now".
    ratio   : atr_pct vs its median over `lookback` candles.
              < 1  = volatility collapsing (dead/range → trades expire)
              > 1  = volatility expanding (spike/news → whipsaw stops)
    """
    n = len(closes)
    if n < lookback + 2:
        return {"atr_pct": 0.0, "median_atr_pct": 0.0, "ratio": 1.0}

    tr_pct = []
    for i in range(1, n):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        tr_pct.append(tr / (closes[i] + 1e-10))

    recent = tr_pct[-lookback:]
    srt    = sorted(recent)
    median = srt[len(srt) // 2] if srt else 0.0
    cur    = sum(tr_pct[-3:]) / 3 if len(tr_pct) >= 3 else (tr_pct[-1] if tr_pct else 0.0)
    ratio  = cur / median if median > 0 else 1.0

    return {"atr_pct": cur, "median_atr_pct": median, "ratio": ratio}


def efficiency_ratio(closes: list, lookback: int = 20) -> float:
    """
    Kaufman Efficiency Ratio — directionality of price over `lookback` bars.

      ER = abs(close[-1] - close[-1-lookback]) / sum(abs(bar-to-bar change))

    ~1.0 = clean one-way trend (net move ≈ total path travelled)
    ~0.0 = chop (price wandered a lot but went nowhere → false BOS → SL)

    Distinct from ATR/vol-regime: that measures candle *size*, this measures
    *direction*. A choppy range can be high-ATR yet near-zero ER.
    """
    n = len(closes)
    if n < lookback + 1:
        return 1.0  # not enough data → don't block
    net = abs(closes[-1] - closes[-1 - lookback])
    path = sum(abs(closes[i] - closes[i - 1]) for i in range(n - lookback, n))
    if path <= 0:
        return 0.0
    return net / path


# ── SMC: Swing Points ─────────────────────────────────────────────────────────

def find_swing_points(highs: list, lows: list, lookback: int = SMC_SWING_LOOKBACK):
    """
    Find confirmed swing highs and lows.
    A swing high/low needs `lookback` candles on each side to confirm.
    Returns: (swing_highs, swing_lows) each as list of (index, price).
    """
    swing_highs = []
    swing_lows  = []
    n = len(highs)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]

        if highs[i] == max(window_h):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


# ── SMC: Break of Structure ───────────────────────────────────────────────────

def detect_bos(closes: list, swing_highs: list, swing_lows: list,
               recent_candles: int = 10) -> str | None:
    """
    Detect Break of Structure in the last `recent_candles` candles.
    Returns 'bullish', 'bearish', or None.

    Bullish BOS: recent close breaks above a confirmed swing high.
    Bearish BOS: recent close breaks below a confirmed swing low.
    """
    if not swing_highs or not swing_lows or len(closes) < recent_candles:
        return None

    # Use last confirmed swing high/low (exclude very recent — not yet confirmed)
    last_sh = swing_highs[-1][1] if swing_highs else None
    last_sl = swing_lows[-1][1]  if swing_lows  else None

    # Check if any of the last N candles broke structure WITH strong body
    # (not just a wick poke — body must be >= 40% of candle range)
    n = len(closes)

    # Exclude last candle (index n-1) — still forming, close not final yet.
    # Only check confirmed closed candles (up to index n-2).
    for i in range(n - recent_candles, n - 1):
        if i < 0:
            continue
        c = closes[i]
        if last_sh and c > last_sh:
            return "bullish"
        if last_sl and c < last_sl:
            return "bearish"

    return None


# ── SMC: Fair Value Gap ───────────────────────────────────────────────────────

def detect_fvg(opens: list, highs: list, lows: list, closes: list,
               min_pct: float = SMC_FVG_MIN_PCT) -> dict:
    """
    Detect Fair Value Gaps in last 40 candles, active near current price.

    Bullish FVG zone: [high[i], low[i+2]]
    Bearish FVG zone: [high[i+2], low[i]]

    Returns booleans + zone tuples (low, high) for entry zone calculation.
    """
    current = closes[-1]
    n = len(closes)
    search_from = max(0, n - 40)

    bullish_zone = None
    bearish_zone = None

    for i in range(search_from, n - 2):
        # Bullish FVG
        if highs[i] < lows[i + 2]:
            gap_bot = highs[i]
            gap_top = lows[i + 2]
            size = (gap_top - gap_bot) / (gap_bot + 1e-10)
            if size >= min_pct and gap_bot * 0.999 <= current <= gap_top * 1.01:
                bullish_zone = (gap_bot, gap_top)  # keep most recent active zone

        # Bearish FVG
        elif lows[i] > highs[i + 2]:
            gap_bot = highs[i + 2]
            gap_top = lows[i]
            size = (gap_top - gap_bot) / (gap_bot + 1e-10)
            if size >= min_pct and gap_bot * 0.99 <= current <= gap_top * 1.001:
                bearish_zone = (gap_bot, gap_top)

    return {
        "bullish":      bullish_zone is not None,
        "bearish":      bearish_zone is not None,
        "bullish_zone": bullish_zone,
        "bearish_zone": bearish_zone,
    }


# ── SMC: Order Block ──────────────────────────────────────────────────────────

def detect_order_block(opens: list, highs: list, lows: list, closes: list,
                       lookback: int = SMC_OB_LOOKBACK) -> dict:
    """
    Detect Order Blocks near current price.

    Bullish OB: last bearish candle before a strong bullish impulse (3+ bull candles).
    Bearish OB: last bullish candle before a strong bearish impulse (3+ bear candles).

    Returns booleans + zone tuples (low, high) for entry zone calculation.
    """
    current = closes[-1]
    n = len(closes)
    start = max(0, n - lookback)

    bull_zone = None
    bear_zone = None

    for i in range(start, n - 4):
        # Bullish OB: bearish candle → strong bullish impulse
        if closes[i] < opens[i]:
            next3_bull = all(closes[j] > opens[j] for j in range(i + 1, min(i + 4, n)))
            if next3_bull:
                move = (closes[min(i + 3, n - 1)] - closes[i]) / (closes[i] + 1e-10)
                if move > 0.005:
                    ob_top = max(opens[i], closes[i])
                    ob_bot = min(opens[i], closes[i])
                    if ob_bot * 0.998 <= current <= ob_top * 1.005:
                        bull_zone = (ob_bot, ob_top)

        # Bearish OB: bullish candle → strong bearish impulse
        elif closes[i] > opens[i]:
            next3_bear = all(closes[j] < opens[j] for j in range(i + 1, min(i + 4, n)))
            if next3_bear:
                move = (closes[i] - closes[min(i + 3, n - 1)]) / (closes[i] + 1e-10)
                if move > 0.005:
                    ob_top = max(opens[i], closes[i])
                    ob_bot = min(opens[i], closes[i])
                    if ob_bot * 0.995 <= current <= ob_top * 1.002:
                        bear_zone = (ob_bot, ob_top)

    return {
        "bullish":      bull_zone is not None,
        "bearish":      bear_zone is not None,
        "bullish_zone": bull_zone,
        "bearish_zone": bear_zone,
    }


# ── SMC: Liquidity Sweep ──────────────────────────────────────────────────────

def detect_liquidity_sweep(highs: list, lows: list, closes: list,
                            swing_highs: list, swing_lows: list,
                            check_last: int = 4) -> dict:
    """
    Detect liquidity sweeps (stop hunts).

    Bullish sweep: candle wicked below swing low then closed above it → reversal up.
    Bearish sweep: candle wicked above swing high then closed below it → reversal down.

    Returns {'bullish': bool, 'bearish': bool}.
    """
    if not swing_highs or not swing_lows:
        return {"bullish": False, "bearish": False}

    recent_sh = [p for _, p in swing_highs[-4:]]
    recent_sl = [p for _, p in swing_lows[-4:]]

    n = len(closes)
    bull_sweep = False
    bear_sweep = False

    for i in range(max(0, n - check_last), n):
        # Bullish sweep: wick below swing low, close above it
        for level in recent_sl:
            if lows[i] < level * 0.999 and closes[i] > level:
                bull_sweep = True

        # Bearish sweep: wick above swing high, close below it
        for level in recent_sh:
            if highs[i] > level * 1.001 and closes[i] < level:
                bear_sweep = True

    return {"bullish": bull_sweep, "bearish": bear_sweep}


# ── MACD ─────────────────────────────────────────────────────────────────────

def calculate_macd(closes: list, fast: int = 12, slow: int = 26,
                   signal_p: int = 9) -> tuple:
    """
    Returns (macd_line, signal_line, histogram) — all same length as closes.
    MACD = EMA(12) - EMA(26).  Signal = EMA(9) of MACD.
    """
    if len(closes) < slow + signal_p:
        empty = [0.0] * len(closes)
        return empty, empty, empty
    ema_fast   = calculate_ema(closes, fast)
    ema_slow   = calculate_ema(closes, slow)
    macd_line  = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = calculate_ema(macd_line, signal_p)
    histogram  = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def detect_macd_divergence(closes: list, highs: list, lows: list,
                            lookback: int = 30) -> str | None:
    """
    MACD histogram divergence over last `lookback` candles.

    Bullish  : price makes lower low, MACD histogram makes higher low → reversal UP
    Bearish  : price makes higher high, MACD histogram makes lower high → reversal DOWN
    """
    if len(closes) < lookback + 40:
        return None
    _, _, histogram = calculate_macd(closes)
    if len(histogram) < lookback:
        return None

    high_w = highs[-lookback:]
    low_w  = lows[-lookback:]
    hist_w = histogram[-lookback:]
    mid    = lookback // 2

    # Bearish: price higher high + MACD histogram lower high
    ph1, ph2 = max(high_w[:mid]), max(high_w[mid:])
    mh1, mh2 = max(hist_w[:mid]), max(hist_w[mid:])
    if ph2 > ph1 * 1.001 and mh2 < mh1 - 1e-10:
        return "bearish"

    # Bullish: price lower low + MACD histogram higher (less negative) low
    pl1, pl2 = min(low_w[:mid]), min(low_w[mid:])
    ml1, ml2 = min(hist_w[:mid]), min(hist_w[mid:])
    if pl2 < pl1 * 0.999 and ml2 > ml1 + 1e-10:
        return "bullish"

    return None


# ── Change of Character ────────────────────────────────────────────────────────

def detect_choch(closes: list, highs: list, lows: list,
                 swing_lookback: int = 3, check_recent: int = 8) -> str | None:
    """
    Change of Character — faster micro-structure shift.

    Uses a shorter swing lookback (3 vs BOS's 5) to detect when price breaks
    a recent intermediate swing high/low before or alongside the main BOS.
    Acts as early confirmation of the structural reversal.

    Returns 'bullish', 'bearish', or None.
    """
    n = len(closes)
    if n < swing_lookback * 2 + check_recent + 2:
        return None
    sh, sl = find_swing_points(highs, lows, lookback=swing_lookback)
    if not sh or not sl:
        return None
    last_sh = sh[-1][1]
    last_sl = sl[-1][1]
    for i in range(max(0, n - check_recent), n - 1):
        if closes[i] > last_sh:
            return "bullish"
        if closes[i] < last_sl:
            return "bearish"
    return None


# ── Engulfing Pattern ─────────────────────────────────────────────────────────

def detect_engulfing(opens: list, closes: list, lookback: int = 4) -> str | None:
    """
    Detect bullish/bearish engulfing candle patterns in the last N closed candles.

    Bullish engulfing: previous bearish candle, current bullish candle whose body
      fully contains the previous body (close > prev open, open < prev close).
    Bearish engulfing: mirror image.
    Only closed candles checked (excludes index n-1 = still forming).
    """
    n = len(closes)
    if n < 4:
        return None
    # Check closed candles only: indices n-lookback to n-2
    for i in range(max(1, n - lookback), n - 1):
        po, pc = opens[i - 1], closes[i - 1]
        co, cc = opens[i],     closes[i]
        prev_top = max(po, pc); prev_bot = min(po, pc)
        curr_top = max(co, cc); curr_bot = min(co, cc)
        if (prev_top - prev_bot) <= 0 or (curr_top - curr_bot) <= 0:
            continue
        # Bullish: prev red, curr green, curr body wraps prev body
        if pc < po and cc > co:
            if curr_bot <= prev_bot and curr_top >= prev_top:
                return "bullish"
        # Bearish: prev green, curr red, curr body wraps prev body
        elif pc > po and cc < co:
            if curr_bot <= prev_bot and curr_top >= prev_top:
                return "bearish"
    return None


# ── Premium / Discount zones ──────────────────────────────────────────────────

def get_premium_discount(swing_highs: list, swing_lows: list,
                          closes: list) -> dict:
    """
    Premium / Discount based on recent swing range midpoint (50% Fibonacci level).

    Below 50% of the swing range = Discount (cheaper, prefer LONG entries).
    Above 50% of the swing range = Premium (expensive, prefer SHORT entries).

    Structure gate (PD_TREND_GATE, default on): "discount" is only a buy signal
    inside a bullish/neutral dealing range. In a clear down-structure (lower-high
    AND lower-low) the whole range is a bear retracement — price below midpoint is
    NOT cheap, it is mid-decline. Likewise "premium" is meaningless in a clean
    up-structure. Without this gate a LONG into a lower-high sequence wrongly
    earned a "Discount" confirmation (the 16.06 XRP loss).
    """
    if not swing_highs or not swing_lows or not closes:
        return {"in_discount": False, "in_premium": False, "midpoint": closes[-1] if closes else 0}
    hi  = max(p for _, p in swing_highs[-3:])
    lo  = min(p for _, p in swing_lows[-3:])
    if hi <= lo:
        return {"in_discount": False, "in_premium": False, "midpoint": closes[-1]}
    mid     = (hi + lo) / 2
    current = closes[-1]
    in_discount = current < mid
    in_premium  = current > mid

    if PD_TREND_GATE and len(swing_highs) >= 2 and len(swing_lows) >= 2:
        bear = swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]
        bull = swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]
        if bear:   # down-structure: "discount" is just mid-decline, not cheap
            in_discount = False
        if bull:   # up-structure: "premium" is just mid-rally, not expensive
            in_premium = False

    return {
        "in_discount": in_discount,
        "in_premium":  in_premium,
        "midpoint":    round(mid, 8),
    }


# ── 1h Trend ──────────────────────────────────────────────────────────────────

def get_1h_trend(candles_1h: dict) -> dict:
    """
    Determine 1h trend using EMA9/21/50.
    Returns dict: trend ('bullish'/'bearish'/'neutral'), strong (bool).
    strong = True when EMA9 > EMA21 > EMA50 (all aligned).
    """
    closes = candles_1h.get("close", [])
    if len(closes) < 22:
        return {"trend": "neutral", "strong": False}

    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)

    trend = "neutral"
    if ema9[-1] > ema21[-1] * 1.001:
        trend = "bullish"
    elif ema9[-1] < ema21[-1] * 0.999:
        trend = "bearish"

    # Strong trend: EMA9 > EMA21 > EMA50 (or inverse)
    strong = False
    if len(closes) >= 51:
        ema50 = calculate_ema(closes, 50)
        if trend == "bullish"  and ema21[-1] > ema50[-1]:
            strong = True
        if trend == "bearish"  and ema21[-1] < ema50[-1]:
            strong = True

    return {"trend": trend, "strong": strong}


# ── Combined SMC indicator dict ───────────────────────────────────────────────

def get_smc_indicators(candles_15m: dict, candles_1h: dict = None,
                        candles_4h: dict = None) -> dict:
    """
    Run all SMC indicators on 15m candles + optional 1h for trend.
    Returns a flat dict of all signals.
    """
    closes  = candles_15m["close"]
    opens   = candles_15m["open"]
    highs   = candles_15m["high"]
    lows    = candles_15m["low"]
    volumes = candles_15m["volume"]

    # Swing points
    swing_highs, swing_lows = find_swing_points(highs, lows)

    # BOS
    bos = detect_bos(closes, swing_highs, swing_lows)

    # FVG
    fvg = detect_fvg(opens, highs, lows, closes)

    # Order Block
    ob = detect_order_block(opens, highs, lows, closes)

    # Liquidity sweep
    sweep = detect_liquidity_sweep(highs, lows, closes, swing_highs, swing_lows)

    # Volume
    avg_vol    = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes) / len(volumes)
    vol_ratio  = volumes[-1] / (avg_vol + 1e-10)

    # RSI
    rsi = calculate_rsi(closes, 14)

    # Stochastic RSI
    stoch_k, stoch_d = calculate_stoch_rsi(closes)

    # Wick analysis
    wicks = analyze_wicks(opens, highs, lows, closes)

    # RSI divergence
    divergence = detect_rsi_divergence(closes, highs, lows)

    # MACD divergence
    macd_div = detect_macd_divergence(closes, highs, lows)

    # Change of Character (micro-structure shift, faster than BOS)
    choch = detect_choch(closes, highs, lows)

    # Engulfing candle pattern
    engulfing = detect_engulfing(opens, closes)

    # Premium / Discount zones
    prem_disc = get_premium_discount(swing_highs, swing_lows, closes)

    # ATR for stops/takes
    atr = calculate_atr(highs, lows, closes)

    # Volatility regime (dead vs spike) — quality gate
    vol_reg = volatility_regime(highs, lows, closes)

    # Efficiency ratio (chop vs trend) — quality gate
    eff_ratio = efficiency_ratio(closes, EFF_RATIO_LOOKBACK)

    # TP/SL reference levels
    recent_high = max(highs[-21:-1]) if len(highs) >= 22 else max(highs)
    recent_low  = min(lows[-21:-1])  if len(lows)  >= 22 else min(lows)

    # 1h + 4h trend (returns dict with trend + strong flag)
    t1h = get_1h_trend(candles_1h) if candles_1h else {"trend": "neutral", "strong": False}
    t4h = get_1h_trend(candles_4h) if candles_4h else {"trend": "neutral", "strong": False}

    # 1h Order Block — nested OB: stronger entry when 1h OB overlaps 15m entry zone
    ob_1h = {"bullish": False, "bearish": False, "bullish_zone": None, "bearish_zone": None}
    if candles_1h and len(candles_1h.get("close", [])) >= 10:
        try:
            ob_1h = detect_order_block(
                candles_1h["open"], candles_1h["high"],
                candles_1h["low"],  candles_1h["close"],
            )
        except Exception:
            pass

    # US session phase — use CANDLE timestamp (not live clock) so backtest is
    # accurate. OPEN/MIDDAY/CLOSE inside the NYSE session, OFF outside
    # (DST and exchange holidays handled by market_hours).
    from datetime import datetime, timezone as _tz
    from src.market_hours import session_phase
    _ts = (candles_15m.get("time") or [None])[-1]
    if _ts:
        _dt = datetime.fromtimestamp(int(_ts), tz=_tz.utc)
    else:
        _dt = datetime.now(_tz.utc)
    session = session_phase(_dt)  # "OPEN" | "MIDDAY" | "CLOSE" | "OFF"

    # BOS candle body quality: last breaking candle body >= 40% of range
    bos_body_strong = False
    if bos and len(closes) >= 2:
        i = -1  # last candle
        body   = abs(closes[i] - opens[i])
        candle_range = highs[i] - lows[i]
        bos_body_strong = (body / candle_range >= 0.4) if candle_range > 0 else False

    # ── Structural TP levels ──────────────────────────────────────────────────
    # TP1 = nearest confirmed 15m swing high/low beyond price (min 0.5% away)
    # TP2 = second 15m swing level OR nearest 1h swing level
    current_px = closes[-1]
    _bull_tps = sorted(
        [p for _, p in swing_highs if p > current_px * 1.005],
        key=lambda x: x
    )
    _bear_tps = sorted(
        [p for _, p in swing_lows if p < current_px * 0.995],
        key=lambda x: x, reverse=True
    )
    bull_tp1 = _bull_tps[0] if _bull_tps else None
    bull_tp2 = _bull_tps[1] if len(_bull_tps) > 1 else None
    bear_tp1 = _bear_tps[0] if _bear_tps else None
    bear_tp2 = _bear_tps[1] if len(_bear_tps) > 1 else None

    # 15m swing structure trend (HH+HL = bull, LH+LL = bear, else range)
    _swing_trend = "range"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        _s_bull = swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]
        _s_bear = swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]
        if _s_bull:
            _swing_trend = "bull"
        elif _s_bear:
            _swing_trend = "bear"

    # Use 1h swing levels as TP2 fallback when 15m only has one level
    if candles_1h and len(candles_1h.get("high", [])) >= 10:
        sh_1h, sl_1h = find_swing_points(
            candles_1h["high"], candles_1h["low"], lookback=3
        )
        _bull_1h = sorted([p for _, p in sh_1h if p > current_px * 1.005], key=lambda x: x)
        _bear_1h = sorted([p for _, p in sl_1h if p < current_px * 0.995], key=lambda x: x, reverse=True)
        if bull_tp2 is None and _bull_1h:
            bull_tp2 = _bull_1h[0]
        if bear_tp2 is None and _bear_1h:
            bear_tp2 = _bear_1h[0]

    return {
        "bos":              bos,
        "bos_body_strong":  bos_body_strong,
        "bullish_fvg":      fvg["bullish"],
        "bearish_fvg":      fvg["bearish"],
        "bullish_fvg_zone": fvg.get("bullish_zone"),
        "bearish_fvg_zone": fvg.get("bearish_zone"),
        "bull_ob":          ob["bullish"],
        "bear_ob":          ob["bearish"],
        "bull_ob_zone":     ob.get("bullish_zone"),
        "bear_ob_zone":     ob.get("bearish_zone"),
        "bull_sweep":       sweep["bullish"],
        "bear_sweep":       sweep["bearish"],
        "trend_1h":         t1h["trend"],
        "trend_1h_strong":  t1h["strong"],
        "trend_4h":         t4h["trend"],
        "trend_4h_strong":  t4h["strong"],
        "bull_ob_1h_zone":  ob_1h.get("bullish_zone"),
        "bear_ob_1h_zone":  ob_1h.get("bearish_zone"),
        "session":          session,
        "rsi":              round(rsi, 2),
        "stoch_k":          stoch_k,
        "stoch_d":          stoch_d,
        "wicks":            wicks,
        "divergence":       divergence,
        "macd_divergence":  macd_div,
        "choch":            choch,
        "engulfing":        engulfing,
        "in_discount":      prem_disc["in_discount"],
        "in_premium":       prem_disc["in_premium"],
        "midpoint":         prem_disc["midpoint"],
        "atr":              atr,
        "vol_atr_pct":      vol_reg["atr_pct"],
        "vol_median_pct":   vol_reg["median_atr_pct"],
        "vol_ratio_regime": vol_reg["ratio"],
        "eff_ratio":        eff_ratio,
        "volume_ratio":     round(vol_ratio, 2),
        "current_close":    closes[-1],
        "current_open":     opens[-1],
        "recent_high":      recent_high,
        "recent_low":       recent_low,
        # Structural TP targets (direction-specific swing levels)
        "bull_tp1":         bull_tp1,
        "bull_tp2":         bull_tp2,
        "bear_tp1":         bear_tp1,
        "bear_tp2":         bear_tp2,
        "swing_trend":      _swing_trend,
    }


# ── Legacy helper (kept for compatibility) ────────────────────────────────────

def get_indicators(candles: dict) -> dict:
    """Legacy indicator dict — used by old signal_filter."""
    closes  = candles["close"]
    highs   = candles["high"]
    lows    = candles["low"]
    volumes = candles["volume"]

    ema9  = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi   = calculate_rsi(closes, 14)

    avg_volume     = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else sum(volumes) / len(volumes)
    current_volume = volumes[-1]
    volume_ratio   = current_volume / avg_volume if avg_volume > 0 else 1.0

    recent_high = max(highs[-21:-1])
    recent_low  = min(lows[-21:-1])

    return {
        "ema9":          ema9[-1],
        "ema21":         ema21[-1],
        "ema9_prev":     ema9[-2],
        "ema21_prev":    ema21[-2],
        "rsi":           rsi,
        "volume_ratio":  volume_ratio,
        "recent_high":   recent_high,
        "recent_low":    recent_low,
        "current_close": closes[-1],
        "current_open":  candles["open"][-1],
    }
