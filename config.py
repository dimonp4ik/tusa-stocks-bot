import os
from dotenv import load_dotenv

load_dotenv()

# --- Required secrets (set in Render environment variables) ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# --- Admin panel: Telegram user IDs that can access /admin in DM ---
ADMIN_IDS = {671071896}  # super-admin only; others added via bot → DB

# --- Scan settings ---
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
TOP_COINS_COUNT = int(os.getenv("TOP_COINS_COUNT", "30"))  # non-crypto X-Perp pool is ~26 — take all
TIMEFRAME = "15m"          # 15m candle → swing signals, hold 2-8h
# Lookback windows MUST match backtest.py's WINDOW_15M/WINDOW_1H/WINDOW_4H
# (300/90/50) — see KLINES_1H_LIMIT below for what happened when they did not.
KLINES_LIMIT = 300         # 300 × 15m (= WINDOW_15M)

# --- Symbol quality filter ---
# ALLOWED_SYMBOLS="" (default) → auto top-volume mode, top 45 by 24h USDT volume.
# Bybit uses BTCUSDT format. BTC-USDT / BTC_USDT / BTC/USDT env values are
# accepted too and normalized at startup.
# Stock swaps turn over far less than crypto majors — $300k keeps dead
# tickers out without emptying the ~26-instrument non-crypto pool.
MIN_24H_QUOTE_VOLUME_USDT = float(os.getenv("MIN_24H_QUOTE_VOLUME_USDT", "300000"))
MAX_SPREAD_PCT            = float(os.getenv("MAX_SPREAD_PCT", "0.20"))

def _parse_symbol_list(value, default=None):
    if not value:
        return list(default or [])
    return [s.strip().upper() for s in value.split(",") if s.strip()]

def _normalize_market_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-", "").replace("_", "").replace("/", "")

ALLOWED_SYMBOLS = [_normalize_market_symbol(s) for s in _parse_symbol_list(os.getenv("ALLOWED_SYMBOLS", ""))]
BLOCKED_SYMBOLS = [_normalize_market_symbol(s) for s in _parse_symbol_list(os.getenv("BLOCKED_SYMBOLS", ""))]
# Stocks bot: commodities (XAU/XAG/CL/BZ) and index ETFs are IN the pool —
# everything non-crypto that is X-Perp tradable at 10x. No always-block list.
_ALWAYS_BLOCKED = set()
BLOCKED_SYMBOLS = list(set(BLOCKED_SYMBOLS) | _ALWAYS_BLOCKED)

# Stablecoins and fiat pairs — no trading signals
BLOCK_STABLE_BASES = {
    "USDC", "TUSD", "FDUSD", "DAI", "USDD", "USDP", "BUSD", "USTC",
    "EUR", "TRY", "BRL", "GBP", "JPY", "RUB", "UAH", "PYUSD", "USDE",
}
# Leveraged/synthetic tokens — unpredictable, not SMC-tradeable
LEVERAGED_TOKEN_SUFFIXES = ("3L", "3S", "2L", "2S", "5L", "5S", "UP", "DOWN", "BULL", "BEAR")

# --- Technical filter thresholds ---
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65
VOLUME_SPIKE_MULTIPLIER = 1.8
MIN_SIGNALS_TO_PASS = 2

# --- Signal deduplication ---
SIGNAL_COOLDOWN_HOURS = 3  # 15m swing signals hold 2-8h — 3h cooldown per coin/direction

# --- Signal expiry (no TP1/SL within this window → EXPIRED) ---
# Counted in MARKET-OPEN hours (src/market_hours.session_hours_between), not
# wall-clock. 2026-07-22: with a wall-clock clock, 25.7% of Friday entries
# expired vs 1.7-5.3% Mon-Thu — 62% of ALL expiries came from Fridays, because
# the weekend burned the budget while the underlying could not move. A
# session-gated stock bot must age positions on session time.
SIGNAL_EXPIRY_HOURS = int(os.getenv("SIGNAL_EXPIRY_HOURS", "48"))
# Hard calendar ceiling on top of the session clock. 48 session hours starting
# Friday afternoon would otherwise run ~11.8 calendar days — long enough for the
# position to sit through the ticker's own earnings report (the blackout only
# gates ENTRY, it cannot protect an already-open position) and to accumulate
# weekend gap risk. Whichever limit is hit first ends the trade.
SIGNAL_EXPIRY_MAX_DAYS = float(os.getenv("SIGNAL_EXPIRY_MAX_DAYS", "5"))

# --- KuCoin (accessible from cloud/US servers) ---
KUCOIN_BASE_URL = "https://api.kucoin.com"
QUOTE_ASSET = "USDT"
TIMEFRAME_KUCOIN = "15min"
KLINES_INTERVAL_SEC = 15 * 60

# --- 1h candles for trend direction ---
TIMEFRAME_1H_KUCOIN = "1hour"
# 50 -> 90 (= backtest WINDOW_1H) on 2026-07-31, ported from the crypto bot.
# get_1h_trend() computes its "strong" flag (EMA9>EMA21>EMA50) only when
# len(closes) >= 51 — at 50 candles that branch NEVER RAN LIVE, so
# trend_1h_strong was permanently False in production while the backtest (90)
# computed it for real. It awards +1 mtf_score, emits the StrongTrend1h
# confirmation, and adds +10 trend quality — on the crypto bot 87-89% of ALL
# backtest trades carried that bonus and ~13% sat exactly on the score gate,
# meaning live silently ran a one-point stricter filter than anything measured.
KLINES_1H_LIMIT = 90
KLINES_1H_INTERVAL_SEC = 3600

# --- 4h candles for higher timeframe bias ---
TIMEFRAME_4H_KUCOIN = "4hour"
# Matched to backtest WINDOW_4H, NOT raised past 51: trend_4h_strong stays
# False on both sides (confirmed dead in the crypto data), and raising only the
# live side would recreate the very mismatch this fixes.
KLINES_4H_LIMIT = 50
KLINES_4H_INTERVAL_SEC = 4 * 3600

# --- 1D candles for macro trend ---
TIMEFRAME_1D_KUCOIN = "1d"
KLINES_1D_LIMIT = 5
KLINES_1D_INTERVAL_SEC = 86400

# --- Trading hours filter (UTC) ---
# ⚠️ DEAD SETTINGS as of 2026-08-28. These three are imported at the top of
# main.py and referenced NOWHERE else in the codebase — no hour gate, no weekend
# gate, in either the live path or the backtest. The two agree, so there is no
# live/backtest parity gap; both simply trade around the clock. Left in place
# because the question they imply is a real and untested one (this desk holds
# perps on STOCKS, whose underlying does close, and outside the US session the
# book is thin), but do not read them as describing current behaviour. Session
# awareness in this bot lives in SIZING only — see OPEN_SESSION_SIZE_MULT.
TRADING_HOURS_START = 7    # 07:00 UTC = 10:00 Riga
TRADING_HOURS_END   = 21   # 21:00 UTC = 00:00 Riga
TRADE_WEEKENDS      = False

# --- SMC settings ---
# These three were plain literals with no os.getenv until 2026-08-28, which is
# the same "hardcoded" class as a bare number in code — it just hides better.
# Made readable so they can be swept; defaults unchanged.
#
# 🔑 SWEPT AND ALL THREE KEPT. The crypto desk moved off all three strict values
# and gained from each; NONE of that transfers here. Against base 1218tr 73.2%
# +762.17R 106.2/280.9:
#                         trades  WR      profit    prof/worst  prof/ulcer
#   swing 5 -> 3           1236  74.0%  +798.90R      99.3        331.7
#   eff-lookback 20 -> 10  1269  73.5%  +799.91R      94.7        270.8
#   FVG 0.0005 -> 0.0003   1231  72.5%  +713.75R      85.3        254.3
# The FVG threshold is the single best find of the whole crypto effort (+26-29
# trades AND lower risk in all three windows) and here it LOSES money: -6.4%
# profit, win rate down 0.7pp, both risk measures worse. Eff-lookback buys
# trades and profit but worsens both risk measures.
#
# Swing 3 looked like the one candidate (+18 trades, +4.8% profit, +0.8pp win
# rate, ulcer -11%) so it was re-tested on genuinely DISJOINT windows — capped
# at --candles 4000 (~42 days) with end-dates 2026-06-15 and 2026-08-26, because
# two different --end-date values at full depth give a window and a SUBSET of it
# (visible as identical worst-windows in both) and confirm nothing:
#   to 15.06   swing5 263tr 73.4% +172.87R ulcer-ratio 92.5
#              swing3 277tr 72.6% +177.43R ulcer-ratio 82.7
#   to 26.08   swing5 301tr 75.4% +224.69R ulcer-ratio 125.0
#              swing3 308tr 76.0% +221.29R ulcer-ratio 136.0
# The two disagree — the earlier slice loses on ulcer and win rate, the later
# one loses on profit. Not shipped.
# ⚠️ worst-windows is UNUSABLE at that sample size: it came back NEGATIVE in all
# four runs (-3.83, -0.32, -0.54, -5.44), which a drawdown cannot be. Read only
# profit, win rate and ulcer on slices this small.
#
# Reading: I had argued recognition parameters port more safely than edge claims
# because "the bot is blind" assumes little about the market. That is wrong.
# What counts as a meaningful gap is a property of the instrument, and stock
# perps gap differently from alt perps. Measure every port.
# 5 -> 4 on 2026-08-29. The test above ran on two slices, on a book selected by
# a kill-switch replay that peeked at future outcomes, and it came back
# ambiguous ("the two disagree"). Re-run on all FIVE windows of available
# history against the honest book:
#   swing    04-10     05-07     06-05     07-15     08-26    total
#     5    +112.85   +123.28   +140.26   +124.84   +185.13   +686.36
#     4    +131.90   +127.24   +152.99   +138.09   +170.78   +721.00
#     3    +130.56   +127.55   +165.63   +193.26   +127.20   +744.20
# Per window the story is monotone and clean: four windows want it SHORTER, one
# wants it longer — and that one is the most recent. 3 earns the most in total
# but gives up 31% in that latest window; 4 beats the old 5 on profit AND win
# rate in four of five with a worst case of -7.7%. Taking the value that never
# loses much rather than the one with the highest sum.
#
# Note the crypto desk sits at 3. Two desks, same parameter, different answers —
# the fifth time a VALUE has refused to port while features and tools port fine.
SMC_SWING_LOOKBACK    = int(os.getenv("SMC_SWING_LOOKBACK", "4"))
SMC_FVG_MIN_PCT       = float(os.getenv("SMC_FVG_MIN_PCT", "0.0005"))
SMC_OB_LOOKBACK       = int(os.getenv("SMC_OB_LOOKBACK", "30"))
# Minimum 3-candle impulse that qualifies an order block, as a fraction of
# price. Readable from the environment since 2026-08-29; it was a plain literal
# carried over from the crypto desk, where 0.5% in three 15m candles is an
# ordinary move. On US equities a 15m ATR runs about 0.4% of price, so the same
# number demands roughly a 1.2-ATR impulse here — a far stricter bar than it
# was ever meant to be. This parameter decides whether an order block EXISTS at
# all, so a threshold set too high does not select worse trades, it hides
# structure. Same failure mode as swing 5->3, eff-lookback 20->10 and the FVG
# threshold on the crypto side: all three were too strict.
# 0.005 -> 0.004 on 2026-08-29, measured on FIVE roughly disjoint 29-day
# windows (--candles 2800, end 04-10 / 05-07 / 06-05 / 07-15 / 08-26):
#   thr      04-10     05-07     06-05     07-15     08-26    trades
#   0.005    +86.11    +95.20   +134.15   +117.66   +163.97     995
#   0.0040   +97.94   +117.72   +140.90   +129.72   +166.99    1027
#   0.0035   +99.13   +119.58   +149.84   +131.07   +163.21    1041
# 0.004 raises profit in EVERY window, trade count in every window, and win rate
# in four of five; the ulcer ratio improves in four of five. 0.0035 earns more
# in total but gives some back in the last window, so the value that never
# loses wins — and it is the smaller move away from a number this desk never
# chose in the first place.
SMC_OB_MIN_IMPULSE = float(os.getenv("SMC_OB_MIN_IMPULSE", "0.004"))

# Zone-acceptance tolerances, as fractions of price. Readable from the
# environment since 2026-08-29; they were plain literals inherited from the
# crypto fork, and they are the parameters that decide whether price counts as
# "at" a zone at all. NEAR is how far PAST the zone still counts, APPROACH is
# how far SHORT of it counts. The approach tolerance on gaps was 1% of price —
# about 2.5 ATR on a 15m equity bar, so a gap was being accepted while price
# was still a long way from it. Same numbers make sense on crypto, where an ATR
# is several times wider.
# How far through a gap price may already be and still count as a usable zone.
# Was a module constant in src/signal_filter.py. It matters far more here than
# on the crypto desk, where the note says it only bites when a setup has no
# order block: 73% of this book's entries come from FVG (173 of 238 in the last
# window), so this gate is on the main path rather than the fallback.
# Swept 2026-08-29, its first ever, and 0.80 stands:
#   fill    06-05      07-15      08-26
#   0.50   +141.83   +128.56   +184.68
#   0.65   +140.26   +128.69   +185.74
#   0.80   +140.26   +124.84   +185.13   <- kept
#   0.90   +140.26   +124.84   +181.63
#   0.95   +140.26   +124.84   +181.63
# Tightening to 0.65 is mildly positive (+3.1% in one window, nothing in the
# other two) and loosening costs the last window. The 0.1-3% band does not earn
# a change. Note the first window reads identical across 0.65-0.95 to the last
# decimal: not the "knob never reached" failure — the env anchor was verified
# and the other windows do move — there is simply no setup there with fill in
# that range.
SMC_FVG_MAX_FILL     = float(os.getenv("SMC_FVG_MAX_FILL", "0.80"))
SMC_FVG_NEAR_TOL     = float(os.getenv("SMC_FVG_NEAR_TOL", "0.001"))
SMC_FVG_APPROACH_TOL = float(os.getenv("SMC_FVG_APPROACH_TOL", "0.01"))
SMC_OB_NEAR_TOL      = float(os.getenv("SMC_OB_NEAR_TOL", "0.002"))
# 0.005 -> 0.008 on 2026-08-29, measured on five windows. The ulcer ratio
# improves in ALL FIVE while profit holds — risk falling at constant profit is
# the opposite of the leverage signature, and it is the change the account
# actually wanted.
#   window   0.005 net / ulcer-ratio      0.008 net / ulcer-ratio
#   04-10     +97.94  32.9                +112.85  34.6
#   05-07    +117.72  39.5                +123.28  53.9   (no losing 25-stretch)
#   06-05    +140.90  67.2                +140.26  78.3
#   07-15    +129.72  40.5                +124.84  53.3   (worst ratio 20.2->38.7)
#   08-26    +166.99  50.7                +185.13  61.8   (worst ratio 38.0->125.2)
# Swept past it too: beyond 0.008 the trade count falls away (238/229/216/197 in
# the last window) and profit goes with it, so this is a peak and not a slope.
# Tightening loses on every measure. The FVG approach tolerance was swept the
# same way in both directions and KEEPS 0.01 — tightening it to 0.004 costs a
# third of the profit, so the wide tolerance there is doing real work.
SMC_OB_APPROACH_TOL  = float(os.getenv("SMC_OB_APPROACH_TOL", "0.008"))
SMC_MIN_CONFIRMATIONS = int(os.getenv("SMC_MIN_CONFIRMATIONS", "2"))
# 2026-08-28: 1.5 -> 1.3. Undocumented and never swept, and binding hard.
# Full windows (nested — one window and a subset of it):
#   min   to 30.06 tr/WR/profit  ratios       to 26.08 tr/WR/profit  ratios
#   1.3   885 73.7% +547.62R  88.9/204.0      1316 74.0% +818.08R  116.8/311.5  <- shipped
#   1.4   836 72.7% +501.51R  67.1/166.8      1256 73.2% +762.24R  102.0/268.9
#   1.5   819 73.2% +513.46R  71.5/178.2      1218 73.2% +762.17R  106.2/280.9  <- was
# Re-tested on genuinely DISJOINT slices (--candles 4000, ends 15.06 and 26.08),
# because two --end-date values at full depth are a window and a subset:
#   min   early tr/WR/profit  prof/ulcer     late tr/WR/profit  prof/ulcer
#   1.3   290 74.8% +206.76R    113.9        325 77.2% +248.78R    122.7
#   1.4   259 71.8% +164.06R     70.4        317 76.0% +236.17R    116.4
#   1.5   263 73.4% +172.87R     92.5        301 75.4% +224.69R    125.0
# 1.3 wins on trades, profit AND win rate in BOTH disjoint slices (+19.6% and
# +10.7% profit), with risk better in the early slice and flat in the late one.
#
# 🔑 The crypto desk lands on 1.4, not 1.3 — a different optimum, like every
# other cross-desk port tried this session. Do not sync these two.
#
# ⚠️ 1.4 is a DIP here: worse than BOTH 1.5 and 1.3, in the nested windows and in
# both disjoint slices. Diminishing returns cannot produce a dip, so something
# else is at work — and the obvious candidate has now been RULED OUT. The
# threshold also sets the score's +2 volume tier (max(threshold*1.35, 2.0) =
# 2.025 at 1.5, 2.000 at 1.4 and 1.3), so the 1.5 -> 1.4 step moves both. On the
# crypto desk that tier move was pinned and measured apart, and it changes
# NOTHING — the band is 1.2% wide and holds no setup. The dip here remains
# unexplained. Do not treat it as understood, and do not sync this parameter
# across desks on the strength of either sweep.
SMC_BOS_MIN_VOLUME    = float(os.getenv("SMC_BOS_MIN_VOLUME", "1.3"))
SMC_RSI_LONG_MAX      = float(os.getenv("SMC_RSI_LONG_MAX", "72"))   # skip overextended longs
SMC_RSI_SHORT_MIN     = float(os.getenv("SMC_RSI_SHORT_MIN", "28"))  # skip overextended shorts
MAX_SETUPS_TO_CLAUDE  = int(os.getenv("MAX_SETUPS_TO_CLAUDE", "7"))  # only strongest go to Claude

# --- Entry zone (FVG / Order Block) ---
# When enabled, setups without an active FVG or OB zone near price are skipped.
REQUIRE_ENTRY_ZONE       = os.getenv("REQUIRE_ENTRY_ZONE", "1") != "0"
ENTRY_ZONE_SL_BUFFER_ATR = float(os.getenv("ENTRY_ZONE_SL_BUFFER_ATR", "0.25"))

# --- Regime / retest filters (cut chop + false breakouts) ---
# REQUIRE_HTF_TREND : reject when both 1h AND 4h are neutral (no real trend = chop).
# REQUIRE_RETEST    : price must currently sit at/near the entry zone (true retest),
#                     not a far-away limit order that the backtest fills optimistically.
REQUIRE_HTF_TREND   = os.getenv("REQUIRE_HTF_TREND", "1") != "0"
REQUIRE_RETEST      = os.getenv("REQUIRE_RETEST", "1") != "0"
RETEST_MAX_DIST_PCT = float(os.getenv("RETEST_MAX_DIST_PCT", "0.006"))  # within 0.6% of zone edge (stock-scale; crypto used 1.5%)

# --- Multi-timeframe score gate (max ~15) ---
# 2026-06-11 A/B (20 sym, 2880+5760×15m, trail): scores 12-13 = WR ~20%, -6.3R.
# Raising 10→14 cut those: WR 48.9→50.7%, R/tr +17%, DD -25% on both windows.
# 14 -> 13 on 2026-08-29. The justification above is a fossil: it was measured
# 2026-06-11 on a bot whose win rate was 48.9%, against today's ~74%, so the
# "scores 12-13 = WR ~20%" bucket it was protecting against no longer exists.
# Re-swept on all five windows:
#   min    04-10     05-07     06-05     07-15     08-26    total  trades
#    14  +131.90   +127.24   +152.99   +138.09   +170.78  +721.00   1031
#    13  +150.33   +144.84   +163.16   +140.26   +172.55  +771.14   1083
#    12  +152.55   +145.13   +169.53   +144.49   +173.42  +785.12   1101
# 13 raises profit, trade count AND the ulcer ratio in EVERY window; win rate
# rises in three and slips 0.7-0.9pp in two. 12 earns a little more again but
# halves the worst-windows ratio in 07-15 (62.7 -> 33.9), so the gain there is
# bought with drawdown rather than earned.
#
# The crypto desk also sits at 13 — the first parameter this session where the
# two desks agree, and worth noting precisely because four others did not.
MTF_MIN_SCORE = int(os.getenv("MTF_MIN_SCORE", "13"))

# --- Signal-quality filters (backtested on a PINNED 20-coin / ~21-day set) ---
# №1 Volatility regime — DEFAULT ON after re-test 2026-06-05 on full context
#    momentum stack: +1.60R net, non-negative on all monthly slices, better MC p05.
#    Upper ceiling still OFF (hurt R in backtest — cuts TP2 runners).
VOL_REGIME_FILTER = os.getenv("VOL_REGIME_FILTER", "1") != "0"
# Stocks: session-gated 15m ATR on megacaps runs 0.1-0.3%; crypto's 0.15% floor
# would reject half the pool on quiet days. 0.06% still cuts truly dead tape.
VOL_MIN_ATR_PCT   = float(os.getenv("VOL_MIN_ATR_PCT", "0.0006"))  # <0.06% range = too dead
VOL_MIN_RATIO     = float(os.getenv("VOL_MIN_RATIO", "0.55"))      # cur/median below = collapsed
VOL_MAX_RATIO     = float(os.getenv("VOL_MAX_RATIO", "99"))        # ceiling OFF (hurt R in backtest)
VOL_REGIME_LOOKBACK = int(os.getenv("VOL_REGIME_LOOKBACK", "50"))

# №3 Strong BOS and №4 Structural-only confirmation were BOTH backtested and
# DROPPED (default off): each lowered win rate (37.5% → 35.0%) and Expected R
# (+0.12R → +0.03R). Strong-BOS pushed entries late (momentum spent → SL);
# structural-only cut valid reversals. Flags kept for experimentation.
REQUIRE_STRONG_BOS = os.getenv("REQUIRE_STRONG_BOS", "0") != "0"
STRONG_BOS_VOL_MULT = float(os.getenv("STRONG_BOS_VOL_MULT", "1.3"))  # x SMC_BOS_MIN_VOLUME
REQUIRE_STRONG_CONFIRM  = os.getenv("REQUIRE_STRONG_CONFIRM", "0") != "0"
MACD_CHOCH_NOISE_FILTER = os.getenv("MACD_CHOCH_NOISE_FILTER", "0") != "0"
# Crypto overlap-session guard — session concept doesn't map to stocks, OFF.
OVERLAP_BEARISH_1H_GUARD = os.getenv("OVERLAP_BEARISH_1H_GUARD", "0") != "0"

# 1D macro trend filter — skip LONG when daily candle trend is BEARISH.
# Prevents buying into a day-scale downtrend (as happened with sideways/red daily days).
DAILY_TREND_FILTER = os.getenv("DAILY_TREND_FILTER", "1") != "0"

# Double-neutral LONG block — skip LONG when BOTH 4h AND 1D are NEUTRAL.
# Two-TF neutrals = sideways/chop at macro level; longs get chopped out by range boundaries.
DOUBLE_NEUTRAL_LONG_FILTER = os.getenv("DOUBLE_NEUTRAL_LONG_FILTER", "1") != "0"

# Daily SHORT guard — mirror of DAILY_TREND_FILTER for shorts.
# Skip SHORT when daily trend is BULLISH — don't short into a day-scale uptrend.
DAILY_TREND_SHORT_FILTER = os.getenv("DAILY_TREND_SHORT_FILTER", "1") != "0"

# №A Efficiency-Ratio chop filter — DEFAULT ON (backtest-proven winner).
#    Kaufman ER over EFF_RATIO_LOOKBACK bars: ER~1 = clean trend, ER~0 = chop.
#    Skip setup if ER < EFF_RATIO_MIN. Targets the proven loss source: false BOS
#    in ranges (LINK 2W/26SL, SOL 6W/19SL). Distinct from ATR-vol (size) — ER
#    measures DIRECTION. Backtest (pinned 20 symbols, ~21d 15m), threshold sweep:
#       base 430tr 36.7% +0.08R/+33R | 0.10 341tr +0.11R/+38R | 0.12 323tr +0.12R/+39R
#       0.15 293tr 37.2% +0.14R/+41R (PEAK) | 0.20 245tr +0.13R/+31R | 0.30 151tr +0.13R/+20R
#    0.15 = clean unimodal peak: beats baseline on win%, R/trade AND total R while
#    cutting 32% junk trades. First filter to beat baseline on every axis.
EFF_RATIO_FILTER   = os.getenv("EFF_RATIO_FILTER", "1") != "0"
EFF_RATIO_LOOKBACK = int(os.getenv("EFF_RATIO_LOOKBACK", "20"))
# Re-swept 2026-08-29 on five windows because the justification above is a
# fossil (430 trades at 36.7% win rate, against today's ~74%) and because the
# crypto desk moved this same parameter 0.15 -> 0.12 the same night. Here it
# does NOT move — 0.15 is already the optimum:
#   thr    04-10     05-07     06-05     07-15     08-26     total
#   0.15  +150.33   +144.84   +163.16   +140.26   +172.55   +771.14  <- kept
#   0.12  +157.89   +144.65   +157.59   +135.91   +167.07   +763.11
#   0.10  +146.09   +154.10   +161.14   +135.72   +172.20   +769.25
# Each threshold wins a window or two and loses the rest; the totals sit within
# 1% of each other. Turning it OFF is different and clearly bad: the earliest
# window goes from no losing 25-trade stretch at all to 12.80R of worst-windows,
# with profit down 17%. The chop protection is doing real work on this book even
# though its original numbers describe a different bot.
#
# Sixth parameter this session where the two desks disagree. Crypto wants 0.12.
EFF_RATIO_MIN      = float(os.getenv("EFF_RATIO_MIN", "0.15"))
# Premium/Discount structure gate — "discount" only counts as a buy signal inside
# a bullish/neutral dealing range, "premium" only inside a bearish/neutral one. In
# a clean lower-high+lower-low down-structure, price below the range midpoint is
# mid-decline, not cheap — without this a LONG into descending swings wrongly got
# a "Discount" confirmation (the 16.06 XRP loss). Set PD_TREND_GATE=0 to disable.
PD_TREND_GATE      = os.getenv("PD_TREND_GATE", "1") != "0"
# №B Strict HTF alignment — DROPPED (default off). Backtested: 232tr +0.04R/+8R,
#    half of baseline. Cutting counter-trend also cut winners. Flag kept for experiments.
REQUIRE_STRICT_HTF = os.getenv("REQUIRE_STRICT_HTF", "0") != "0"

# --- Asymmetric bear-squeeze guard (DEFAULT ON) --------------------------------
# In crypto, full-HTF bearish shorts (BOS + 1h + 4h all bearish) with overheated
# volume attract crowded late entries → market-makers squeeze them upward.
# Skip SHORT when: bos=bearish AND trend_1h=bearish AND trend_4h=bearish AND
# vol_ratio_regime >= threshold (2.5 = 2.5× normal volume = overheated).
# Also skip "LONDON" session for full-bearish shorts (expansion attracts latecomers,
# then NYSE open reverses them).
# A/B backtest, 20 symbols, 8640×15m (~3 months), trail exit:
#   base:  2646tr  38.1% WR  +0.118R/tr  DD -68.17R
#   guard: 2344tr  39.6% WR  +0.150R/tr  DD -47.36R  (+27% R/tr, -31% DD)
# Volume-part of the guard transfers (crowded shorts get squeezed in stocks
# too); the LONDON-session skip was crypto-clock specific → no session skip.
# ON -> OFF on 2026-08-29. Two independent reasons, and they agree.
#
# The mechanism above is stated in crypto terms: market-makers squeezing crowded
# late shorts, with a LONDON-session clause. US equities do not have that
# microstructure, and this session has now watched six parameters refuse to port
# between the two desks. Its numbers are a fossil besides — 2646 trades at 38.1%
# win rate, against this book's ~74%.
#
# Measured on all five windows:
#   window   guard ON               guard OFF
#   04-10   189tr +150.33R  u62.7   199tr +158.92R  u53.6
#   05-07   202tr +144.84R  u60.8   213tr +151.80R  u61.1
#   06-05   217tr +163.16R  u102.7  236tr +185.36R  u125.1
#   07-15   232tr +140.26R  u61.6   237tr +141.99R  u59.3
#   08-26   243tr +172.55R  u63.5   250tr +179.24R  u72.7
# Profit and trade count rise in EVERY window (+6.0% and +52 trades in total),
# with the ulcer ratio better in three of five. The two windows where it slips
# are the two smallest moves in the table; 04-10 also goes from having no losing
# 25-trade stretch to a 1.92R one, which is real but tiny against 158.92R.
#
# This is a setup filter built on a story, not a risk rail — removing it does not
# widen any position. BEAR_TREND_SKIP_SESSIONS was already empty and inert.
BEAR_TREND_HOT_VOL_GUARD     = os.getenv("BEAR_TREND_HOT_VOL_GUARD", "0") != "0"
BEAR_TREND_HOT_VOL_MIN_RATIO = float(os.getenv("BEAR_TREND_HOT_VOL_MIN_RATIO", "2.5"))
BEAR_TREND_SKIP_SESSIONS     = set(_parse_symbol_list(os.getenv("BEAR_TREND_SKIP_SESSIONS", "")))

# --- Directional RSI midline confirmation (DEFAULT ON) ------------------------
# A BOS without RSI reclaiming the 50 midline (LONG) or dropping below 40
# (SHORT) = structural break without momentum confirmation → higher false-break
# rate. Distinct from the overextension caps (SMC_RSI_LONG_MAX / SHORT_MIN).
# A/B backtest, on top of bear-trend guard, same 20 symbols × 8640×15m:
#   guard:    2344tr  39.6% WR  +0.150R/tr  DD -47.36R
#   +RSI mid: 2117tr  40.1% WR  +0.175R/tr  DD -37.38R  (+17% R/tr, -21% DD)
# Re-tested 2026-08-29 (its justification is a fossil too — 2646 trades at 38.1%
# win rate). It is now close to inert: switching it OFF moves profit +1.3/+1.3/
# +1.2% across 04-10/06-05/08-26 and adds 4/14/-1 trades, with the ulcer ratio
# better in one window and worse in two. Inside the band that does not earn a
# change either way, so it stays on — but it is no longer earning its place and
# should be re-checked if the book changes again.
DIRECTIONAL_RSI_MIDLINE_FILTER = os.getenv("DIRECTIONAL_RSI_MIDLINE_FILTER", "1") != "0"
RSI_LONG_MIN_MIDLINE           = float(os.getenv("RSI_LONG_MIN_MIDLINE", "42"))  # lowered 50→42: catches zone entry earlier, same WR/R (+3 trades, +4R on 8640-bar test)
RSI_SHORT_MAX_MIDLINE          = float(os.getenv("RSI_SHORT_MAX_MIDLINE", "40"))

# --- Per-symbol / per-source / per-direction edge filters (DEFAULT ON) ----------
# Populated from loss_taxonomy analysis. Skip instruments/direction combos that
# repeatedly showed poor edge after enough backtest data.
# Edge lists start EMPTY for stocks — crypto's XMR/NEAR/AAVE findings don't
# transfer. Lists will repopulate from this bot's own loss taxonomy over time.
SYMBOL_EDGE_FILTER  = os.getenv("SYMBOL_EDGE_FILTER", "1") != "0"
LOW_EDGE_SYMBOLS    = _parse_symbol_list(os.getenv("LOW_EDGE_SYMBOLS", ""))

# 2026-06-05 A/B, 8640×15m: skipping NEAR FVG entries improved raw net (+9.99R),
# WR/R-trade, and Monte Carlo while keeping trade count high.
# QQQ: 2026-07-04 backtest (4mo, session-gated exits) — FVG entries 33tr −7.5R
# vs OB +1.5R. Index intraday is mean-reverting; FVG-imbalance momentum entries
# get chopped. OB entries (structural levels) survive. SPY unaffected (+9.8R).
SOURCE_EDGE_FILTER     = os.getenv("SOURCE_EDGE_FILTER", "1") != "0"
LOW_EDGE_FVG_SYMBOLS   = _parse_symbol_list(os.getenv("LOW_EDGE_FVG_SYMBOLS", "QQQ-USDT,QQQUSDT"))
LOW_EDGE_OB_SYMBOLS    = _parse_symbol_list(os.getenv("LOW_EDGE_OB_SYMBOLS", ""))

DIRECTION_EDGE_FILTER  = os.getenv("DIRECTION_EDGE_FILTER", "1") != "0"
LOW_EDGE_LONG_SYMBOLS  = _parse_symbol_list(os.getenv("LOW_EDGE_LONG_SYMBOLS", ""))
# QQQ shorts: 9tr −8.7R (longs +2.7R). Structural, not noise: equity-index
# return accrues overnight/up-drift (Elm "Night Moves"), dealer gamma flows
# buy dips — 15m BOS shorts on the index fight that machinery.
LOW_EDGE_SHORT_SYMBOLS = _parse_symbol_list(os.getenv("LOW_EDGE_SHORT_SYMBOLS", "QQQ-USDT,QQQUSDT"))

# --- Context momentum pack (DEFAULT ON, validated together 2026-06-05) ----------
# Weak relative strength and session-momentum mismatches → higher SL rate.
# All four proven together on 8640×15m across all monthly slices.
RELATIVE_STRENGTH_LOOKBACK_HOURS      = int(os.getenv("RELATIVE_STRENGTH_LOOKBACK_HOURS", "1"))
LONG_RELATIVE_WEAKNESS_FILTER         = os.getenv("LONG_RELATIVE_WEAKNESS_FILTER", "1") != "0"
LONG_RELATIVE_WEAKNESS_MAX_PCT        = float(os.getenv("LONG_RELATIVE_WEAKNESS_MAX_PCT", "-1.60"))

# Zone-width constant below was curve-fit on crypto vol — for stocks a 0.17%
# zone is NORMAL, not narrow → filter would fire constantly. OFF until re-tuned
# on this bot's own setup_log.
BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER  = os.getenv("BULL_NEUTRAL_LONG_NARROW_ZONE_FILTER", "0") != "0"
BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT  = float(os.getenv("BULL_NEUTRAL_LONG_MAX_ZONE_WIDTH_PCT", "0.00173509"))

LONG_NY_COIN_MOMENTUM_FILTER          = os.getenv("LONG_NY_COIN_MOMENTUM_FILTER", "1") != "0"
LONG_NY_MIN_COIN_CHANGE_1H            = float(os.getenv("LONG_NY_MIN_COIN_CHANGE_1H", "0.0"))

SHORT_FVG_COIN_MOMENTUM_FILTER        = os.getenv("SHORT_FVG_COIN_MOMENTUM_FILTER", "1") != "0"
SHORT_FVG_MAX_COIN_CHANGE_1H          = float(os.getenv("SHORT_FVG_MAX_COIN_CHANGE_1H", "0.0"))

# Crypto-only filter (London session × BTC momentum) — OFF for stocks.
FVG_LONDON_BTC_UP_FILTER  = os.getenv("FVG_LONDON_BTC_UP_FILTER", "0") != "0"
FVG_LONDON_BTC_UP_MIN_PCT = float(os.getenv("FVG_LONDON_BTC_UP_MIN_PCT", "0.29"))

# --- Risk sizing overlays (DEFAULT ON) -----------------------------------------
# Does not filter trades. Raises risk_mult for contexts that repeatedly showed
# stronger R/trade: OB entries, optimal RSI/vol, strong relative coin momentum.
QUALITY_RISK_OVERLAY    = os.getenv("QUALITY_RISK_OVERLAY", "1") != "0"
QUALITY_RISK_MULT       = float(os.getenv("QUALITY_RISK_MULT", "1.15"))
QUALITY_RISK_MAX_MULT   = float(os.getenv("QUALITY_RISK_MAX_MULT", "1.15"))
QUALITY_RISK_VOL_MIN    = float(os.getenv("QUALITY_RISK_VOL_MIN", "0.8"))
QUALITY_RISK_VOL_MAX    = float(os.getenv("QUALITY_RISK_VOL_MAX", "1.2"))
QUALITY_RISK_RSI_MIN    = float(os.getenv("QUALITY_RISK_RSI_MIN", "50"))
QUALITY_RISK_RSI_MAX    = float(os.getenv("QUALITY_RISK_RSI_MAX", "60"))
# 2026-07-22 re-validation (1811tr, 2022-2026, corrected BACKTEST_TP_WINDOW —
# see backtest_seed_stocks.csv comment): all 4 commodities + 7 stocks positive
# EVERY year sampled (INTC/NVDA had one near-flat/tiny-sample 2022) at
# +0.42-0.63R/tr, clearly above AAPL/AMZN/MSFT's +0.44-0.46R/tr and well above
# SPY's +0.14R/tr (3 of 5 years negative). GOOGL added this pass — 184tr,
# +0.528R/tr, 0 negative years, was wrongly bucketed as "mid-tier" under the
# old understated window. AAPL/AMZN/MSFT stay off: solid but not standout.
HIGH_EDGE_RISK_SYMBOLS  = _parse_symbol_list(os.getenv(
    "HIGH_EDGE_RISK_SYMBOLS",
    "XAU-USDT,XAUUSDT,XAG-USDT,XAGUSDT,CL-USDT,CLUSDT,BZ-USDT,BZUSDT,"
    "MU-USDT,MUUSDT,META-USDT,METAUSDT,INTC-USDT,INTCUSDT,"
    "TSLA-USDT,TSLAUSDT,MRVL-USDT,MRVLUSDT,NVDA-USDT,NVDAUSDT,"
    "GOOGL-USDT,GOOGLUSDT",
))
REL_STRENGTH_RISK_UP          = os.getenv("REL_STRENGTH_RISK_UP", "1") != "0"
REL_STRENGTH_RISK_UP_MIN_PCT  = float(os.getenv("REL_STRENGTH_RISK_UP_MIN_PCT", "0.5"))
REL_STRENGTH_RISK_UP_MAX_PCT  = float(os.getenv("REL_STRENGTH_RISK_UP_MAX_PCT", "2.0"))
REL_STRENGTH_RISK_UP_MULT     = float(os.getenv("REL_STRENGTH_RISK_UP_MULT", "1.15"))
REL_STRENGTH_RISK_UP_MAX_MULT = float(os.getenv("REL_STRENGTH_RISK_UP_MAX_MULT", "1.25"))
TREND_PAIR_RISK_UP            = os.getenv("TREND_PAIR_RISK_UP", "1") != "0"
TREND_PAIR_RISK_UP_1H         = os.getenv("TREND_PAIR_RISK_UP_1H", "bullish").lower()
TREND_PAIR_RISK_UP_4H         = os.getenv("TREND_PAIR_RISK_UP_4H", "bullish").lower()
TREND_PAIR_RISK_UP_MULT       = float(os.getenv("TREND_PAIR_RISK_UP_MULT", "1.15"))
TREND_PAIR_RISK_UP_MAX_MULT   = float(os.getenv("TREND_PAIR_RISK_UP_MAX_MULT", "1.25"))

# --- Adaptive market-regime filter packs (from friend's v2 — DEFAULT OFF) ------
# Graduated quality gate: requires progressively higher MTF score + structure as
# the regime worsens (clean trend → mixed → choppy), and returns a per-regime
# risk_mult for position sizing.
#
# A/B BACKTEST RESULT (10 symbols, 2880×15m, ~2 months):
#   CURRENT : 604 tr, 41.7% WR, +0.153 net R/trade, +92R total
#   ADAPTIVE: 378 tr, 41.5% WR, +0.129 net R/trade, +49R total
# Verdict: DEFENSIVE filter — helps in choppy month (May: +0.043→+0.081 R/trade)
# but cuts winners in strong-trend month (June: +0.625→+0.416). Net slightly
# WORSE for us — cuts 37% of trades without lifting win rate. KEPT OFF.
# Enable only as a conservative/range-market mode after re-validation.
ADAPTIVE_FILTER_PACKS       = os.getenv("ADAPTIVE_FILTER_PACKS", "0") != "0"
ADAPTIVE_MIXED_SCORE_BUMP   = int(os.getenv("ADAPTIVE_MIXED_SCORE_BUMP", "1"))
ADAPTIVE_CHOP_SCORE_BUMP    = int(os.getenv("ADAPTIVE_CHOP_SCORE_BUMP", "2"))
ADAPTIVE_HOT_SCORE_BUMP     = int(os.getenv("ADAPTIVE_HOT_SCORE_BUMP", "1"))
ADAPTIVE_MIXED_EFF_MIN      = float(os.getenv("ADAPTIVE_MIXED_EFF_MIN", "0.20"))
ADAPTIVE_CHOP_EFF_MIN       = float(os.getenv("ADAPTIVE_CHOP_EFF_MIN", "0.28"))
ADAPTIVE_HOT_EFF_MIN        = float(os.getenv("ADAPTIVE_HOT_EFF_MIN", "0.22"))
ADAPTIVE_CHOP_MIN_VOLUME    = float(os.getenv("ADAPTIVE_CHOP_MIN_VOLUME", "2.0"))
ADAPTIVE_HOT_MIN_VOLUME     = float(os.getenv("ADAPTIVE_HOT_MIN_VOLUME", "2.0"))
ADAPTIVE_HOT_VOL_RATIO      = float(os.getenv("ADAPTIVE_HOT_VOL_RATIO", "3.0"))
ADAPTIVE_EXTREME_VOL_RATIO  = float(os.getenv("ADAPTIVE_EXTREME_VOL_RATIO", "5.0"))
ADAPTIVE_EXTREME_ATR_PCT    = float(os.getenv("ADAPTIVE_EXTREME_ATR_PCT", "0.035"))
ADAPTIVE_MIXED_RISK_MULT    = float(os.getenv("ADAPTIVE_MIXED_RISK_MULT", "0.75"))
ADAPTIVE_CHOP_RISK_MULT     = float(os.getenv("ADAPTIVE_CHOP_RISK_MULT", "0.50"))
ADAPTIVE_HOT_RISK_MULT      = float(os.getenv("ADAPTIVE_HOT_RISK_MULT", "0.50"))
ADAPTIVE_BEAR_SQUEEZE_GUARD = os.getenv("ADAPTIVE_BEAR_SQUEEZE_GUARD", "1") != "0"
ADAPTIVE_BEAR_SKIP_NEW_YORK = os.getenv("ADAPTIVE_BEAR_SKIP_NEW_YORK", "1") != "0"
ADAPTIVE_BEAR_VOL_MIN_RATIO = float(os.getenv("ADAPTIVE_BEAR_VOL_MIN_RATIO", "0.8"))
ADAPTIVE_BEAR_VOL_MAX_RATIO = float(os.getenv("ADAPTIVE_BEAR_VOL_MAX_RATIO", "1.8"))

# --- Stability overlay: deterministic kill-switch for poorly-validated regimes -
# 2026-06-11 A/B: OVERLAP session (London+NY overlap) = WR 32%, -6.3R over 19tr.
# Skipping it: +5R total, DD -20%. Both sessions fight at overlap = chop hour.
# OVERLAP-skip was a crypto-clock finding (London+NY fight hour). Stock session
# phases are OPEN/MIDDAY/CLOSE — start with no skips, learn from own stats.
STABILITY_FILTERS_ENABLED   = os.getenv("STABILITY_FILTERS_ENABLED", "1") != "0"
STABILITY_SKIP_PACKS        = {s.lower() for s in _parse_symbol_list(os.getenv("STABILITY_SKIP_PACKS", ""))}
STABILITY_SKIP_SESSIONS     = set(_parse_symbol_list(os.getenv("STABILITY_SKIP_SESSIONS", "")))
STABILITY_MIN_EFF_RATIO     = float(os.getenv("STABILITY_MIN_EFF_RATIO", "0.0"))
STABILITY_MIN_VOLUME_RATIO  = float(os.getenv("STABILITY_MIN_VOLUME_RATIO", "0.0"))
STABILITY_MIN_QUALITY_SCORE = float(os.getenv("STABILITY_MIN_QUALITY_SCORE", "0.0"))

# --- Claude tiered analysis (cascade: cheap LIGHT gate + rare deep HEAVY) ---
# LIGHT  : Haiku validates every passed setup in ONE cached batch call (JSON via tool).
# HEAVY  : Sonnet re-checks only top setups (score >= HEAVY_MIN_SCORE) with coin memory.
# Caching: static rules block cached 1h → cheap re-reads on the 5-min scan loop.
CLAUDE_LIGHT_MODEL        = os.getenv("CLAUDE_LIGHT_MODEL", "claude-sonnet-4-5")
CLAUDE_HEAVY_MODEL        = os.getenv("CLAUDE_HEAVY_MODEL", "claude-sonnet-4-5")
CLAUDE_HEAVY_MIN_SCORE    = int(os.getenv("CLAUDE_HEAVY_MIN_SCORE", "9"))    # lowered 10→9: all survivors get Sonnet check
CLAUDE_HEAVY_MAX_PER_SCAN = int(os.getenv("CLAUDE_HEAVY_MAX_PER_SCAN", "5")) # max HEAVY checks per scan
CLAUDE_MEMORY_LIMIT       = int(os.getenv("CLAUDE_MEMORY_LIMIT", "25"))      # recent outcomes per coin (HEAVY)
CLAUDE_MAX_RISK_SCORE     = int(os.getenv("CLAUDE_MAX_RISK_SCORE", "7"))     # counter-arg auto-reject if risk >= this (7 = "real concern" per scale)
CLAUDE_CACHE_TTL          = os.getenv("CLAUDE_CACHE_TTL", "1h")              # prompt cache TTL ("5m" or "1h")
CLAUDE_DAILY_BUDGET_USD   = float(os.getenv("CLAUDE_DAILY_BUDGET_USD", "1.00"))  # hard daily cap (real Sonnet usage ~$0.3-0.5/day)
CLAUDE_BUDGET_RESERVE_USD = float(os.getenv("CLAUDE_BUDGET_RESERVE_USD", "0.05")) # stop when remaining < reserve

# Epoch for the LIVE tier of Claude's self-feedback history (unix ts, 0 = off).
# That tier looks back 30 days, which reaches into the pre-parity-fix bot: a
# filter with a 50-candle 1h lookback (Strong1h could never fire) and a stop
# that triggered on wicks. Those outcomes describe software that no longer
# exists. On the crypto bot the stale record deadlocked Claude into rejecting
# every setup. Live tier only — admin stats keep the full history.
# 1785456000 = 2026-07-31 00:00 UTC, the day the parity fixes shipped.
LIVE_HIST_EPOCH_TS = float(os.getenv("LIVE_HIST_EPOCH_TS", "1785456000"))

# --- Structure-based stops/takes (swing mode, 15m, 10x X-Perp leverage) ---
# SL sits at swing invalidation (recent swing low/high) + ATR buffer, then
# clamped to safe leverage bounds. Stocks move 3-5x less than crypto per 15m:
# megacap intraday swings are 0.3-1.5%, so crypto's 1.2-3% SL band would park
# stops far outside structure and turn every trade into a multi-day hold.
#   risk%  ~0.4–1.5% of price → on 10x = 4–15% margin at risk per stop
#   liquidation ~9% away at 10x → 1.5% max SL keeps 6x safety headroom
ATR_PERIOD    = 14
SL_ATR_BUFFER = float(os.getenv("SL_ATR_BUFFER", "0.5"))   # buffer beyond swing, in ATR
RISK_MIN_PCT  = float(os.getenv("RISK_MIN_PCT", "0.004"))  # min SL distance = 0.4%
# --- 2026-08-28: swept, and REJECTED despite looking like the biggest find yet -
# This clamp is not a rare safety net on this desk, it is the main mechanism
# placing stops: 44-47% of trades sit exactly at this ceiling and another
# 11-15% at RISK_MIN_PCT, so only ~4 in 10 get the stop structure asked for.
# Never swept before. Loosening is monotonically worse (2.0% -> -9.2% profit,
# 2.5% -> -13.0%, both risk measures down, and FEWER trades — a wider stop
# stretches the targets too, so positions live longer and hold slots).
#
# Tightening looked spectacular on the full windows:
#   ceiling  to 30.06 tr/WR/profit  ratios      to 26.08 tr/WR/profit  ratios
#   1.0%     936 76.6% +719.69R  71.4/267.9     1387 75.8% +1006.73R  99.9/366.9
#   1.2%     917 76.0% +659.25R 155.7/287.6     1366 75.2%  +928.20R 114.6/374.1
#   1.5%     885 73.7% +547.62R  88.9/204.0     1316 74.0%  +817.57R 116.7/311.8  <- KEPT
# +13.5% and +20.4% profit, more trades, higher win rate. Money units check out:
# above the 1.0% reference the sizing normalisation holds money at risk constant
# whatever the stop width, so the R gain is a real money gain, not the unit trap.
#
# It does not survive disjoint thirds, and the effect DECAYS monotonically:
#   third      profit 1.5%   profit 1.2%   change   win rate  profit/ulcer
#   to 05.06     +121.72       +149.71     +23.0%    +2.7pp      +68%
#   to 15.07     +105.53       +119.58     +13.3%    +0.4pp       +5%
#   to 26.08     +160.39       +161.77      +0.9%    -1.0pp      -14%
# In the most recent third — the best proxy for what happens next — it buys
# nothing and costs win rate and drawdown. Decay in order is a regime signature;
# noise would scatter.
#
# 🔑 METHOD NOTE, the part worth keeping: the two "windows" that made this look
# proven are NESTED. The to-26.08 window CONTAINS the to-30.06 one, so its
# +13.5% is largely the same early trades counted again. With only ~6 months of
# history this desk cannot produce independent full windows at all — disjoint
# thirds are the only honest test here, and they must be run BEFORE believing a
# result, not after.
RISK_MAX_PCT  = float(os.getenv("RISK_MAX_PCT", "0.015"))  # max SL distance = 1.5%
# Raising this was tested 2026-08-25 and REJECTED, though the diagnosis that
# suggested it was sound: 35% of trades sit pinned at the cap and do worse
# than free-stopped ones (64.6% / +0.443R against 67.2% / +0.491R). In the
# crypto bot the same signature meant the clamp was pushing stops into noise
# and raising it improved every axis at once. Here it does the opposite
# (equal-risk, 36bps + close-confirm): 0.015 +195.1 / 0.018 +135.3 /
# 0.020 +134.1 / 0.025 +120.0.
# Reading: a wide structural stop marks an UNCERTAIN setup, it does not mark
# a mis-placed one. "This group performs worse" is not the same claim as
# "loosening the constraint that binds it will help".

# Risk-normalised sizing, ported from the crypto bot 2026-08-25. Without it a
# wide stop and a tight one use the SAME margin, so they risk very different
# money while the backtest books both as 1R — which means the R totals in every
# report here do not correspond to any achievable dollar outcome, and a run of
# wide-stop losses hurts more than the drawdown figure implies.
# Measured over 2414 backtest trades: stop distance runs 0.40% to 1.50% (median
# 1.06%), so the extremes differ by 3.8x. Reference is set at the median: 53% of
# trades get trimmed, mean multiplier 0.851, tightest 0.667.
# Downward only — a tight stop never gets sized UP, because that would raise
# exposure on the trades whose stop sits closest to the noise.
# --- 2026-08-28: measured how far the R accounting drifts from money here -----
# Backtest sums outcomes in R, which assumes every trade risks the same money.
# This block makes that true only ABOVE the reference — it scales wide stops
# down and leaves tight ones alone. Both clamps bind hard: stop width spans
# exactly RISK_MIN_PCT to RISK_MAX_PCT (0.40% to 1.50%), and 42.7% of the book
# sits BELOW the 1.0% reference, risking as little as 0.40 of a reference unit.
#   stop %      share   money   R/trade   win rate
#   0-0.5      19.3%    0.416   +0.796     72.0%
#   0.5-0.7    10.9%    0.593   +0.510     69.9%
#   0.7-1.0    12.5%    0.848   +0.546     69.1%
#   1.0-1.3    10.6%    1.000   +0.846     78.6%
#   >1.3       46.7%    1.000   +0.544     68.2%
# Return per unit of MONEY at risk is +0.6026 against the flat-R +0.6213 — the
# accounting overstates by 3.0%. That is larger than the crypto desk's 1.2%
# (its money range is only 0.80-1.05 against 0.40-1.00 here) but still modest,
# and it stays modest for a reason worth noting: the tight-stop trades are
# UNDER-weighted in money and OVER-perform in R, so the two errors cancel.
#
# 📌 FOR THE ACCOUNT OWNER, not shipped. Making the normalisation two-sided
# (scaling tight stops UP to the reference instead of leaving them alone) is
# worth +3.1% of return per unit of risk, because the trades it would size up
# are the ones already beating the book. But as a standalone change it also
# raises total exposure 21% (1084 -> 1316 reference units). Getting the +3.1%
# without the extra exposure means lowering base risk per trade at the same
# time, and that dial belongs to the owner. Do not ship one half of this.
RISK_NORMALIZED_SIZING = os.getenv("RISK_NORMALIZED_SIZING", "1") != "0"
RISK_REFERENCE_PCT     = float(os.getenv("RISK_REFERENCE_PCT", "0.010"))
RISK_SIZE_MULT_MIN     = float(os.getenv("RISK_SIZE_MULT_MIN", "0.45"))

# Fresh-break trim, 2026-08-25. bos_extension_atr is how far price has run from
# the level where structure broke, in ATR. The crypto bot trims LATE entries on
# it; here the relationship is INVERTED — the least-extended entries are the
# worst — so this bot trims the opposite end:
#   <=0.71  252 сд  53.2%  +0.087R      1.33-2.13  251 сд  59.4%  +0.149R
#   0.71-1.33 253 сд 62.5% +0.279R      >2.13      251 сд  60.2%  +0.256R
#   base +0.193R
# It is not the same effect seen from the other side, it is a different one:
# the crypto bot WAITS for price to return to the zone, so a small extension
# means a clean retest. This bot enters at market, so a small extension means
# entering right at the break, before anything confirms it — and on a thin
# X-Perp book a break is often false.
# Equal-risk, both halves, six of eight tested cells pass: 0.71/x0.75 = +13.5%,
# 0.71/x0.5 = +29%. Mild chosen over maximal, as in the crypto bot.
EXTENSION_FRESH_THRESHOLD = float(os.getenv("EXTENSION_FRESH_THRESHOLD", "0.7"))
# 0.75 -> 1.0 on 2026-08-29. This trim was a fossil: it was fitted on a book
# that no longer exists, and on the current one it was cutting size on BETTER
# trades than average.
#
# The justification above claims the fresh-break bucket wins 53.2% against a
# 63.6% book. Re-measured at unit size on honestly-gated books, neither number
# reproduces — and note the BOOK is 73.2%, not 63.6%, which dates the fitting to
# before the entry-cost and execution-model fixes:
#   third   subset            rest              unit R gap
#   06-05   48tr  WR 72.9%    144tr  WR 74.3%   +0.300
#   07-15   57tr  WR 77.2%    168tr  WR 72.0%   +0.145
#   08-26   55tr  WR 76.4%    170tr  WR 73.5%   -0.033
#   pooled  160tr WR 75.6%    482tr  WR 73.2%
# Removing it improves profit AND both risk ratios in all three thirds, which is
# the opposite of the leverage signature:
#   third   net R             worst ratio      ulcer ratio
#   06-05   +124.17 -> +134.15      -           62.3 -> 63.0
#   07-15   +109.44 -> +117.66  13.4 -> 14.3    34.8 -> 36.0
#   08-26   +155.38 -> +163.97  50.4 -> 65.6    52.0 -> 54.6
# The other two multipliers were re-checked the same way and both KEEP: removing
# orderly costs 5.8-9.5% of profit and removing open_session 3.5-5.8%, with
# ratios mixed rather than better.
EXTENSION_FRESH_SIZE_MULT = float(os.getenv("EXTENSION_FRESH_SIZE_MULT", "1.0"))

# --- Opening-bell session rides bigger (2026-08-26) --------------------------
# This bot had no session multiplier at all. Split by session:
#   session  share   win rate  R/trade   t      thirds (lift over the rest)
#   OFF      78.7%    72.7%    +0.501  -0.85   +0.070 / +0.014 / -0.406
#   OPEN     12.2%    78.7%    +0.723  +2.07   +0.136 / +0.100 / +0.517
#   MIDDAY    4.9%    75.0%    +0.478  -0.28
#   CLOSE     4.2%    68.6%    +0.278  -1.18
#
# OPEN is 12.2% of the book — inside the 8-15% band where every sizing rule
# that survived today lives — significant on R, and above the rest in all
# three stretches. Same shape as the crypto bot's London finding: the bell is
# when real participation arrives, not merely a time of day.
#
# Default 1.0 = off until measured end-to-end. The per-trade lift is not the
# test; a 29%-of-book candidate with a LARGER lift than this one turned out to
# be pure leverage earlier today.
# Boosting OPEN as a whole was measured and REJECTED — the two risk measures
# move against each other, monotonically, across the sweep:
#   base   +634.62R  worst 66.6  ulcer 221.4
#   1.25   +661.74R  worst 64.7  ulcer 225.9
#   1.5    +688.87R  worst 62.8  ulcer 229.6
#   1.75   +715.99R  worst 61.0  ulcer 232.7
# The deep stretches get deeper while time-underwater shrinks. The crypto
# rules that passed had BOTH measures rising together.
#
# The session also fails the checks London passed. It is partly a volume
# proxy — average volume 4.45 against 3.15, over-represented at vol>3 (18% of
# those trades against 12% of the book), where London was UNDER-represented —
# the edge REVERSES in the middle volume band (+0.494 against +0.615), and
# breadth is 11 tickers of 19 where London was 14 of 16.
#
# What survives is the same session-plus-volume shape as the crypto bot:
#   threshold  share  trades  win rate  R/trade   thirds
#   vol>=2.5    8.0%    98     79.6%    +0.736   +0.242/+0.061/+0.444
#   vol>=3.0    6.0%    73     84.9%    +0.925   +0.412/+0.182/+0.863
#   vol>=4.0    4.0%    49     87.8%    +1.035   +0.498/+0.353/+0.796
# Tighter is stronger and thinner; at 3.0 a stretch holds 18-29 trades, which
# is where one trade moves the estimate. 2.5 keeps 8% of the book, the bottom
# of the band where today's surviving rules live, and its middle stretch is
# weak (+0.061) — recorded as the candidate's honest weakness.
# Gated on volume it passes cleanly, and the gate is what flips it:
#   base   +634.62R  worst 9.53 (66.6)  ulcer 2.87 (221.4)  MaxDD -14.18R
#   x1.5   +670.68R  worst 9.67 (69.4)  ulcer 2.84 (236.0)  MaxDD -14.02R
#   x1.75  +688.71R  worst 9.77 (70.5)  ulcer 2.84 (242.5)  MaxDD -13.93R
# Both ratios rise, profit rises, and max drawdown FALLS. Ungated, the
# worst-windows ratio fell instead — the same volume gate that rescued the
# London boost in the crypto bot rescues this one.
#
# Shipped at 1.5 although 1.75 measured better on every number. That choice is
# about evidence, not data: this bot has no historical regimes at all, so the
# whole finding rests on one window cut into thirds, its middle third is the
# weak one (+0.061), and the subset holds 98 trades. 1.75 is not better
# supported than 1.5 — it is a larger bet on the same uncertain claim. Revisit
# once there is a second regime to test against.
OPEN_VOL_MIN           = float(os.getenv("OPEN_VOL_MIN", "2.5"))  # 0 = no volume requirement
# Swept again on a PINNED window (the first sweep ran on a sliding one — see
# the note at the candle stamp in backtest.py). No turn appears through 2.0:
#   mult   profit      worst   ulcer   MaxDD
#   1.25   +733.35R    78.1    245.5   -12.79R
#   1.5    +750.71R    81.6    252.1   -12.79R
#   1.75   +767.42R    85.2    258.0   -12.79R
#   2.0    +784.12R    87.6    263.6   -12.79R
#
# Max drawdown does not move AT ALL across the sweep, so the book's deepest
# hole is made by trades outside this subset and the rule neither helps nor
# hurts it. Worst-windows does fall in absolute terms (9.39 -> 8.95), so the
# improvement is not merely profit rising against a fixed denominator.
#
# Held at 1.5 anyway. The case for restraint was never about measurement
# quality — it is about the evidence for the RULE, which is one window cut in
# thirds, with the symbol hold-out underpowered for this subset. Monotone
# improvement inside those same thin data does not thicken them; it only says
# that IF the rule is right, more is better. A larger multiplier is a larger
# bet on the same uncertain claim, which is exactly the reason 1.5 was chosen
# in the first place.
# Re-checked on five windows 2026-08-29 and this one EARNS its keep, unlike the
# orderly boost above. Removing it costs profit in all five (-2.3 to -4.2%) AND
# worsens the ulcer ratio in four of five, so the size is being paid for rather
# than merely levered. The subset is small (3-10% of the book, the right band)
# and above book in three of five windows.
OPEN_SESSION_SIZE_MULT = float(os.getenv("OPEN_SESSION_SIZE_MULT", "1.5"))
# Research handle, default 1.0 = no change. Live data 2026-08-28 showed the
# whole live/backtest gap on this desk sits in OFF-session stops filling BEYOND
# the stop level (average -1.209R instead of -1.00R, -4.88R of unmodelled excess
# over 23 stops). The backtest cannot see that — it resolves stops on 15m bars
# and books -1R — so this knob exists to price the trade-off the owner has to
# make: OFF is ~79% of the book, and trimming it costs profit the model DOES
# see while saving losses the model does NOT.
OFF_SESSION_SIZE_MULT = float(os.getenv("OFF_SESSION_SIZE_MULT", "1.0"))

# --- Volume-spike boost (2026-08-29) -----------------------------------------
# FIRST rule on this desk found by SEARCH on its own data rather than ported
# from crypto — the interaction scan had never been run here, which is why this
# sizing book had three rules against crypto's ten.
# Measured on three DISJOINT thirds (--candles 2800, ends 05.06/15.07/26.08),
# R per trade at equal size against each third's own book:
#   vol>=4.0   24tr 13% +0.714 (book +0.644)   +11%
#              21tr 10% +0.910 (book +0.510)   +78%
#              31tr 14% +1.120 (book +0.743)   +51%
# Above book in all three, win rate 71-86% against a ~70% book in all three,
# and the share sits at 10-14% — inside the 8-15% band a boost has to occupy.
# 3.5 was also tested and is weaker (+1% in the first third); 3.0 is too wide
# at 21-27% of the book.
VOLUME_SPIKE_BOOST_MIN  = float(os.getenv("VOLUME_SPIKE_BOOST_MIN", "4.0"))
# Shipped at 1.25. Measured as a live sizing rule on the same three disjoint
# thirds, profit and both risk ratios up in EVERY one:
#   mult   third1 profit  ratio   third2 profit  ratio   third3 profit  ratio
#   1.00     +121.72R     60.3      +105.53R    40.4      +160.39R     76.3
#   1.25     +126.00R     62.5      +110.31R    41.8      +168.38R     77.4
#   1.50     +127.94R     63.1      +113.73R    42.7      +172.25R     76.9
# 1.5 buys more profit but is already WORSE than 1.25 on the newest third's
# ulcer ratio (76.9 vs 77.4), which is the third that matters most for what
# happens next. Mild over maximal on a new finding, as with every other rule
# shipped here.
# 1.25 -> 1.0 on 2026-08-29. ROLLED BACK the day after shipping, and the reason
# is worth more than the rule was.
#
# Both validations that passed it were reading net_r straight out of the export,
# and this export carries NO size_mult column — every multiplier is already
# folded into net_r with no way to divide it back out. The spike subset overlaps
# the orderly and open-session boosts, so it was being credited with their 1.5x.
# Re-exported with every size multiplier forced to 1.0 and gated honestly, the
# subset stops looking special:
#   third   subset          rest            gap
#   06-05   25tr  +0.424    167tr  +0.670   -0.246   BELOW book
#   07-15   24tr  +0.809    201tr  +0.425   +0.384
#   08-26   36tr  +0.739    189tr  +0.610   +0.130
# Pooled that is 85 trades at +0.666 against +0.561, a gap of +0.105 with a
# standard error near 0.15 — indistinguishable from zero, and carried by one
# third. The symbol hold-out, the test that could settle it, comes back
# UNDERPOWERED at this book size (55 and 53 subset trades against the 60 it
# needs), which is not a negative answer but is not a positive one either.
#
# The book-level figures do improve in all three thirds (+3.0/+4.5/+5.2% profit)
# and that is what kept it alive for a day. But boosting ANY profitable slice
# adds profit mechanically; the ratios moved 1-4%, inside noise for a 200-trade
# third. Edge has to show up as the slice beating the book, and it does not.
# Removing an unjustified boost is the safe direction — see the sizing rules.
VOLUME_SPIKE_SIZE_MULT  = float(os.getenv("VOLUME_SPIKE_SIZE_MULT", "1.0"))
# ✅ SYMBOL HOLD-OUT PASSED 2026-08-29. Time thirds share a market, so they
# cannot tell a strategy property from a few lucky tickers. Splitting the 26
# symbols into halves and re-measuring answers that separately:
#   half A (13 tickers)  subset 101tr +0.816  rest 540tr +0.603  lift +0.213
#   half B (13 tickers)  subset 110tr +1.141  rest 565tr +0.572  lift +0.569
# Sign agrees on both halves. Combined with the three disjoint time thirds, this
# is as validated as anything on this desk can be given six months of history.

# --- Orderly trend rides bigger (2026-08-27) ---------------------------------
# The PAIR "clean trend on a calm tape" was rejected above: at 29% of the book
# it behaved as pure leverage. The triple search narrows the same idea to a
# workable width by adding a third condition — the break must also be a late
# one, not fresh:
#
#   eff_ratio>=0.31 & vol_atr_pct<0.0044 & bos_extension_atr>=1.271
#     1/3   87tr 74.7% lift +0.207
#     2/3   66tr 81.8% lift +0.262
#     3/3   68tr 85.3% lift +0.440
#   221 trades, 18% of the book, lift GROWING across the stretches.
#
# Note this is the MIRROR of what the crypto bot wants. There the winning
# state is LOW efficiency with HIGH volatility — busy, violent, not travelling
# cleanly. Here it is high efficiency with low volatility. Plausible: stocks
# are slower and more institutional, so ragged movement is more often news
# noise than money arriving. Two bots, opposite states, each confirmed across
# three stretches.
#
# 18% is above the 8-15% band where every surviving rule lives, and the 29%
# version of this idea already failed once, so the end-to-end measurement
# decides rather than the per-trade lift.
ORDERLY_EFF_MIN   = float(os.getenv("ORDERLY_EFF_MIN", "0.31"))
ORDERLY_ATR_MAX   = float(os.getenv("ORDERLY_ATR_MAX", "0.0044"))
ORDERLY_EXT_MIN   = float(os.getenv("ORDERLY_EXT_MIN", "1.271"))
# Swept to the turn:
#   mult   profit      worst   ulcer   MaxDD
#   1.0    +670.68R    69.4    236.0   -14.02R
#   1.25   +710.21R    77.5    246.1   -13.05R
#   1.5    +752.04R    81.7    252.4   -12.79R   <- best drawdown
#   1.75   +793.87R    83.4    254.7   -14.19R   <- ratios peak
#   2.0    +835.70R    82.0    253.9   -15.60R
#
# Shipped at 1.5, one step below the measured peak, for the same reason as
# OPEN_SESSION_SIZE_MULT: this bot has no historical regimes, so everything
# rests on one window cut into thirds. Sizing exactly onto the peak of a curve
# measured that way is the largest available bet on the weakest available
# evidence. 1.5 also holds the smallest max drawdown, and drawdown is what
# limits how large the book can be carried at all.
# 2026-08-29 re-checked on FIVE windows at unit size, and the honest answer is
# that this is a LEVERAGE dial, not an edge. The subset does not beat the book:
#   window  subset            rest      gap
#   04-10   43tr  +0.215     +0.730   -0.515   BELOW
#   05-07   35tr  +1.032     +0.434   +0.597
#   06-05   22tr  +0.630     +0.647   -0.017   BELOW
#   07-15   28tr  +0.702     +0.470   +0.232
#   08-26   41tr  +0.654     +0.696   -0.042   BELOW
# Below book in three of five, and pooled that is 169 trades at +0.625 against
# +0.593 — a gap of +0.032 with a standard error near 0.10. Indistinguishable
# from zero. An earlier three-window read looked uniformly positive, but the
# gaps there were +0.03 and the two windows added since break the pattern.
#
# KEPT AT 1.5 ANYWAY. Removing it costs profit in all five windows (-4.1/-14.6/
# -4.9/-7.9/-7.2%) while the ratios split two better, three worse, and absolute
# ulcer barely moves. So it is not free size — it is size on a slice with book-
# average edge, which is what leverage is. Raising or lowering it is the account
# owner's call, the same class as SIZE_MULT_MAX and TP1_R_MULT; 1.25 sits
# monotonically between 1.0 and 1.5 on every measure, so there is no peak to
# find. Recorded here so the decision is made on the true picture.
ORDERLY_SIZE_MULT = float(os.getenv("ORDERLY_SIZE_MULT", "1.5"))

# --- Ceiling on the stacked product (2026-08-27) -----------------------------
# This bot had NO ceiling. The crypto bot has carried SIZE_MULT_MAX=2.0 for a
# while, swept and kept there, but nothing bounded the product here — today
# ORDERLY 1.5 x OPEN_SESSION 1.5 reaches 2.25 on 8 trades, and the only reason
# it stops at 2.25 is that there are exactly two boosts. Another one and
# nothing catches it.
#
# Risk-normalised sizing is not a bound: it is guarded by `if _risk_pct >
# RISK_REFERENCE_PCT`, so it only ever scales DOWN.
#
# Set to 2.0 to match the crypto bot. The cost today is 8 trades (0.65% of the
# book) going from 2.25 to 2.0. This is a rail, not a tuning knob — its job is
# to bound a stack that does not exist yet.
SIZE_MULT_MAX = float(os.getenv("SIZE_MULT_MAX", "2.0"))

# --- Shake without participation: REJECTED ----------------------------------
# Mirror of ORDERLY above, taken from the negative side of the same triple
# search: off-session, thin volume, HIGH volatility.
#   session=OFF & volume_ratio<2.42 & vol_atr_pct>=0.0044
#   203 trades, 16.6% of the book, 6.6-11.8 points of win rate below each
#   stretch's own book, lag -0.293 to -0.409, no decay.
#
# Trimming it makes things WORSE on both measures, monotonically:
#            base        x0.75      x0.6
#   profit  +752.04R   +739.65R  +732.21R
#   worst      81.7       76.2      73.1
#   ulcer     252.4      252.0     250.4
# The absolute worst-windows figure RISES under the trim (9.20 -> 9.71 ->
# 10.02) even though size is being cut, which is the tell: the subset earns
# +0.242R per trade, so removing its contribution slows the equity curve and
# the deepest 25-trade stretches get relatively deeper.
#
# 🔑 Second confirmation, now in both bots, of the rule that decides trims:
#   thin London    +0.265 against a book of +0.286  -> trim WORKED
#   bearish 1h     +0.264 against ~+0.30            -> failed
#   shake (here)   +0.242 against +0.518            -> failed
# Lagging the book and being trimmable are different things. This subset lags
# by HALF the book and still must not be cut, because it earns. What decides is
# the subset's ABSOLUTE expectancy being near zero, not its distance from the
# average.

# --- Clean trend on a calm tape: REJECTED, kept so it is not retried --------
# Efficiency ratio >= 0.31 with ATR percent < 0.0044 — a directional move that
# is not also violent — genuinely outperforms per trade, and unlike most
# candidates its edge GROWS across the book's three consecutive stretches:
#
#            subset                  rest of that third        lift
#   1/3   132tr 75.8% +0.541       276tr 68.1% +0.371        +0.170
#   2/3   107tr 83.2% +0.742       301tr 72.8% +0.521        +0.221
#   3/3   112tr 79.5% +0.723       297tr 72.1% +0.483        +0.240
#
# Sizing it up anyway fails, and fails in the way that names itself:
#   base   1225tr +634.62R  worst 9.53 (66.6)   ulcer 2.87 (221.4)
#   x1.25  1225tr +692.48R  worst 10.91 (63.5)  ulcer 3.11 (222.9)
#   x1.5   1225tr +750.34R  worst 12.45 (60.3)  ulcer 3.38 (222.1)
# Profit climbs 9% then 18%, the worst-windows ratio falls monotonically, and
# the ulcer ratio does not move at all. A flat ratio under a rising multiplier
# IS leverage — it is what scaling the entire book looks like.
#
# The reason is size, not quality. The crypto boosts that survived this test
# cover 8-15% of their book; this subset is 29% of ours and its members cluster
# in time, because calm trending stretches arrive in runs. Scaling a third of a
# book that moves together is leverage however it is labelled.
#
# Lesson worth keeping: a per-trade lift, even a growing one, does not convert
# into money when the subset is large. Check the subset's SHARE of the book
# before believing a sizing rule.

# Live-price re-anchor sanity guard (main.py, publish-time X-Perp reprice).
# Crypto's flat 3% made sense next to a 1.2-3% SL band; here SL is 0.4-1.5%,
# so 3% drift is 2-7x a normal stop — thin overnight/open-gap ticks could
# already be past TP1/TP2 by the time this check ran and still pass. Cap
# drift at RISK_MAX_PCT itself: if the ticker moved further than the widest
# stop we'd ever set, the structural zone is stale, don't anchor to it.
LIVE_PRICE_MAX_DRIFT_PCT = float(os.getenv("LIVE_PRICE_MAX_DRIFT_PCT", "0.015"))
# 2026-07-22 stock sweep (16 sym, 2022-2026 Dukascopy), measured AFTER the
# session-aware expiry fix. Full resolution (stride=1):
#   0.7 -> WR 80.2%  net +960R  maxDD -8.92  net/DD 107.6
#   1.0 -> WR 73.9%  net +990R  maxDD -9.66  net/DD 102.5   <- current
# Net R is flat above 1.0 (1.2/1.4 land within 1%, i.e. noise), so raising TP1
# buys nothing. Below 1.0 it is a straight win-rate-for-profit trade: 0.7 gives
# +6.3pp WR for -3.2% net R.
#
# The risk-adjusted ranking is NOT stable: at stride=2 the same sweep put 1.0
# ahead (net/DD 96.7 vs 92.3), at stride=1 it puts 0.7 ahead (107.6 vs 102.5).
# That flip is sampling noise, so net/DD cannot decide this — pick on whether
# total profit or win rate matters more. 1.0 is set because it maximises profit.
#
# WHY 0.7 WAS BRIEFLY SET AND REVERTED: on PRE-FIX numbers 0.7 looked like it
# bought a 15% smaller drawdown for ~7% of profit. That was an artifact of the
# wall-clock expiry bug — the calendar timer killed long-running trades and a
# nearer TP1 partially rescued them by banking the runner sooner. Once trades
# age on session hours the drawdown gap shrinks to within noise. Do not
# re-lower TP1 for "smaller drawdown" without re-measuring.
#
# NOTE: with TP1_CLOSE_FRAC=0 nothing closes at TP1 — this level only decides
# when the runner switches to the ATR trail, and the trail floors at breakeven.
TP1_R_MULT    = float(os.getenv("TP1_R_MULT", "1.0"))      # TP1 = entry ± risk * 1.0
# 2026-08-28: 2.0 -> 3.0, same reasoning as the crypto desk and measured here.
# With TP1_CLOSE_FRAC=0 and a 0.02 ATR trail, TP2 is not the exit — it is only a
# CAP, and a near cap truncates the best trades.
#            to 30.06 profit  ratios       to 26.08 profit  ratios    worst-win
#   1.5R      +489.92R  46.9/152.5          +722.76R  69.2/239.3       10.45
#   2.0R      +497.98R  54.8/157.2          +749.66R  82.4/253.4        9.10  <- was
#   2.5R      +504.45R  64.6/171.4          +754.07R  96.5/270.7        7.81
#   3.0R      +513.46R  71.5/178.2          +762.17R 106.2/280.9        7.18  <- shipped
#   4.0R      +522.41R  72.7/182.2          +771.36R 107.4/281.8        7.18
#
# Unlike crypto, where TP2 could not move drawdown at all, here it moves it a
# lot: worst-windows falls 9.10 -> 7.18 (-21%) and ulcer 2.96 -> 2.71. Bigger
# wins offset losses inside the same bad stretch, and this book has a fatter
# win tail than the crypto one. 4.0R buys more profit but Max DD rebounds to
# -13.44R while the ratios stop improving, so 3.0 is the cut.
#
# ⚠️ These two columns are ONE window and a subset of it — this desk's history
# starts Feb-Mar 2026. The independent evidence is the crypto bot's three
# windows; this only confirms sign and size here.
TP2_R_MULT    = float(os.getenv("TP2_R_MULT", "3.0"))

# Runner exit after TP1: trail the remaining 50% by ATR instead of fixed TP2.
# Backtest (10 sym, 2880x15m): +21% net R, -27% max drawdown, same win rate vs
# fixed TP2. Trailing stop = peak ∓ TRAIL_ATR_MULT×ATR, floored at breakeven.
TRAIL_RUNNER_ENABLED = os.getenv("TRAIL_RUNNER_ENABLED", "1") != "0"
TRAIL_ATR_MULT       = float(os.getenv("TRAIL_ATR_MULT", "0.02"))  # base trail; post_tp1_v2 overrides per-context
# ⚖️ SIGNIFICANCE 2026-08-28: cannot be tested paired on THIS desk, and the
# unpaired answer is inconclusive:
#   old exit +789.00R -> new exit +817.57R, delta +28.56R, R/trade +0.024
#   p_gt_zero 0.633, 90% CI -114.27 to +164.30
# The reason is structural and worth understanding before anyone retries it.
# significance_check requires the pairing to be COMPLETE — every row in both
# files must find a partner — and here the trade COUNTS differ (1321 vs 1316):
# a different exit changes how long positions live, which changes slot
# occupancy under MAX_SAME_DIRECTION_POSITIONS, which changes which trades get
# taken at all. 1302 of them pair by entry, but the five that do not ARE part
# of the effect, so pairing and discarding them would measure the wrong thing.
# The tool falls back to unpaired, correctly, and unpaired has nowhere near the
# power for a 3.6% effect. Passing --pair-key entry does NOT force it.
#
# The same change on the crypto desk DOES pair (1266 = 1266 = 1266 there) and
# comes back p_gt_zero 1.0 with a 90% CI of +17.42 to +33.96 — every one of 5000
# runs positive. That, plus the direction agreeing here, is what this setting
# rests on. It is not independently proven on this desk and should not be
# described as if it were.
# 2026-08-28: 0.05 -> 0.02, ported from the crypto bot but measured here first.
#   mult    to 30.06 profit  worst/ulcer     to 26.08 profit  worst/ulcer
#   0.05      +493.51R        53.6/154.5       +742.90R        80.7/249.1
#   0.02      +497.98R        54.8/157.2       +749.66R        82.4/253.4  <- shipped
#   0.01      +499.35R        55.1/158.1       +751.81R        83.0/254.8
#   0.00      +500.16R        55.4/158.8       +753.51R        83.5/256.1
# Same monotone shape as crypto, same sign on every measure, but the effect is
# a third the size: +0.9% profit and ~+2% on both ratios.
#
# ⚠️ Read the two columns as ONE window, not two. Both report the SAME max
# drawdown (-12.79R) and the same worst-windows (9.20R) — the June cut is a
# subset of the August one, since this desk's history only starts Feb-Mar 2026
# when the OKX perps listed. The evidence for this change is the crypto bot's
# three independent windows; these columns only confirm the sign here.
# Not shipped at 0.00 for the same reason as there: an optimum on the edge of
# the tested range says "further", not "here", and 0.00 puts the stop on a
# level price has just traded.
# 2026-08-25: 0.25 -> 0.05, measured on the corrected trail model (BT_TRAIL_LAG).
# Monotone across the range: 0.05 +639.2R / 0.15 +624.5 / 0.25 +609.4 / 0.35
# +604.6 / 0.50 +592.9, drawdown flat at -9.6 throughout, so +5.8% at equal
# risk. Same direction as the crypto bot, smaller size (+10% there): stock
# runners give back less after the first failure to extend.

# Exit profile: "post_tp1_v2" keeps the FULL position past TP1 (TP1_CLOSE_FRAC=0)
# and trails by an ATR multiple chosen from the TP1-acceptance candle — strong
# follow-through trails wide (let it run), weak/rejected trails tight (lock).
# Validated 3 windows on our cache (90/180/365d): net R +80/+91/+124% with LOWER
# drawdown, win rate / trades / SL count UNCHANGED — it only changes how winners
# are harvested, never which trades are taken. "fixed" = legacy 50%-at-TP1 + BE.
TP1_CLOSE_FRAC = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_FRAC", "0.0"))))
EXIT_PROFILE   = os.getenv("EXIT_PROFILE", "post_tp1_v2").strip().lower()
# 2026-07-22 trail-width sweep (16 sym, 2022-2026, TP1 fixed at 1.0), tested
# (strong/weak) = 0.35/0.15, 0.50/0.25, 0.75/0.35, 1.00/0.50, 1.50/0.75:
#   net R falls MONOTONICALLY as the trail widens (+530R -> +513 -> +504 ->
#   +498 -> +498), win rate identical (74.8%) at every width, max DD identical.
# The "widen the trail to let stock winners run" idea was tested and REJECTED:
# 48% of wins cluster at 1-1.5R and a wider trail just gives that back before
# exiting — US equities mean-revert intraday enough that locking gains tight
# wins. Win rate is width-invariant because the trail floors at breakeven, so
# a trade that has touched TP1 can never revert to a loss. 0.35/0.15 is the
# measured optimum here, not just an inherited crypto default — don't re-widen
# without new evidence.
# Pinned to the base value on purpose. This split reads max(base, STRONG) /
# min(base, WEAK), so it has been INERT since the base dropped to 0.05 — the
# whole "strong trails wide, weak trails tight" feature died silently in both
# bots while the config kept advertising it. Measured on the crypto desk before
# deciding: widening the strong branch LOSES in both windows there and plain
# uniform beats every split on every measure. Kept inert deliberately now
# rather than by accident. WEAK stays 0.15 and is inert too: min(0.02,0.15).
POST_TP1_STRONG_TRAIL_ATR_MULT = float(os.getenv("POST_TP1_STRONG_TRAIL_ATR_MULT", "0.02"))
POST_TP1_WEAK_TRAIL_ATR_MULT   = float(os.getenv("POST_TP1_WEAK_TRAIL_ATR_MULT", "0.15"))
POST_TP1_STRONG_CLOSE_PROGRESS = float(os.getenv("POST_TP1_STRONG_CLOSE_PROGRESS", "0.25"))
POST_TP1_STRONG_WICK_PROGRESS  = float(os.getenv("POST_TP1_STRONG_WICK_PROGRESS", "0.55"))
POST_TP1_WEAK_CLOSE_PROGRESS   = float(os.getenv("POST_TP1_WEAK_CLOSE_PROGRESS", "-0.10"))

# --- k-NN price-shape analog risk overlay (Kronos-inspired, CPU-only) ----------
# After a setup passes, fetch a deep 15m series and match the recent price shape
# against the symbol's own past (nearest-neighbour). Score = fraction of the K
# most-similar past windows whose forward move favoured the trade direction.
# Backtest (2026-06-13, 90d, live-like 800-bar pool): score>=0.55 → WR ~68%,
# score<0.50 → WR ~59%. Used as a size multiplier (no gating) → +6% total R,
# trade frequency unchanged. Edge needs a deep pool, so a ~1000-candle fetch is
# done ONLY for symbols that already produced a setup (rare → cheap).
# OFF for stocks (2026-07-04, 1836-trade deep backtest): kNN score bands are
# FLAT here — >0.55 → +0.422R/tr vs <0.5 → +0.425R/tr, zero separation. The
# crypto price-shape-analog edge (WR 68 vs 59) does not transfer to equities;
# sizing off a non-predictive signal is noise. Flag kept for re-testing.
KNN_RISK_OVERLAY   = os.getenv("KNN_RISK_OVERLAY", "0") != "0"
KNN_DEEP_CANDLES   = int(os.getenv("KNN_DEEP_CANDLES", "1000"))   # 1 Bybit page
KNN_MAX_HISTORY    = int(os.getenv("KNN_MAX_HISTORY", "800"))     # analog pool cap
KNN_SHAPE_LEN      = int(os.getenv("KNN_SHAPE_LEN", "12"))        # query window (3h)
KNN_HORIZON        = int(os.getenv("KNN_HORIZON", "16"))          # forward bars (4h)
KNN_K              = int(os.getenv("KNN_K", "40"))                # neighbours
KNN_MIN_HISTORY    = int(os.getenv("KNN_MIN_HISTORY", "120"))     # min bars to score
KNN_HIGH_SCORE     = float(os.getenv("KNN_HIGH_SCORE", "0.55"))   # size-up threshold
KNN_HIGH_MULT      = float(os.getenv("KNN_HIGH_MULT", "1.20"))    # size-up multiplier
KNN_LOW_SCORE      = float(os.getenv("KNN_LOW_SCORE", "0.50"))    # size-down threshold
KNN_LOW_MULT       = float(os.getenv("KNN_LOW_MULT", "0.80"))     # size-down multiplier
KNN_RISK_MAX_MULT  = float(os.getenv("KNN_RISK_MAX_MULT", "1.50"))  # cap after overlays
KNN_RISK_MIN_MULT  = float(os.getenv("KNN_RISK_MIN_MULT", "0.50"))  # floor after overlays

# --- Research-validated setup cuts (2026-06-11, 20 sym, 30/60/90d backtests) ---
# RSI_Div confirmations: WR 23%, -0.21R/tr over 22tr — divergence in 15m chop = noise.
SKIP_RSI_DIV_SETUPS = os.getenv("SKIP_RSI_DIV_SETUPS", "1") != "0"
# Hour/weekday cuts — OFF by user choice (Mon-Fri 07-21 UTC full window).
# Backtest note: Monday ~0R/tr (53tr), 18-20 UTC ~+0.09R/tr (38tr) — re-enable
# via env SKIP_WEEKDAYS=0 / SKIP_UTC_HOURS=18,19,20 if WR needs a boost.
SKIP_UTC_HOURS = {h for h in os.getenv("SKIP_UTC_HOURS", "").split(",") if h.strip()}
SKIP_WEEKDAYS  = {d for d in os.getenv("SKIP_WEEKDAYS", "").split(",") if d.strip()}

# --- Market proxy correlation filter (stocks: SPY instead of BTC) ---
# Broad-market crash/pump guard: individual longs blocked when the index dumps.
# SPY tracks S&P500; pool is tech-heavy but SPY has the deepest swap liquidity.
MARKET_PROXY_SYMBOL     = os.getenv("MARKET_PROXY_SYMBOL", "SPYUSDT")
BTC_BLOCK_THRESHOLD_PCT = 1.0  # SPY ±1% intraday = genuine market-wide event

# --- Close-confirmed stop (ported 2026-07-26, DEFAULT OFF — must be measured) --
# A stop firing on a WICK touching the level exits trades that never broke.
# Requiring the 15m candle to CLOSE beyond it keeps those alive. On the sister
# crypto bot this was the one change that improved WR and profit on two
# independent windows (WR 80.8->82.8% / 78.0->80.6%, netR +4.5% / +5.7%).
#
# DEFAULT OFF here because this market is NOT the same shape, in both directions:
#   FOR  — off-session X-Perp candles are thin MM drift (see OFF_SESSION_SIGNALS
#          below, this bot's own note). Thin off-session wicks nicking a stop is
#          exactly what close-confirmation removes, so the upside may be LARGER
#          than crypto's.
#   AGAINST — stocks GAP. Positions are monitored 24/7 while signals are gated to
#          the US session, so a position can sit through a gap open. A gap blows
#          straight past the level and the close-confirmed exit then books a much
#          worse price. Crypto's worst close-stop exit was -1.9R over ~3600
#          trades; a gapping instrument can do far worse, and STOP_EXCHANGE_
#          BACKSTOP_R almost certainly needs to be wider than crypto's 2.0.
# Measure both on this bot's own backtest before enabling: run with
# STOP_CLOSE_CONFIRM=0 and =1, compare WR/netR/maxDD, and check the worst
# realised stop R to pick the backstop.
# 2026-08-25: turned ON. The live report shows 64% of stops (9 of 14 over a
# week) are X-Perp wick noise rather than a real reversal, and a level-touch
# exit is maximally exposed to exactly that.
# The verdict depends on which entry model you believe, which is why it was
# measured at three (equal-risk against the 0-bps baseline):
#   0 bps:  +413.8 vs +609.4  -> WORSE by 32%
#  18 bps:  +401.3 vs +306.0  -> better by 31%
#  36 bps:  +252.8 vs +155.0  -> better by 63%
# Live win rate (57.5% over 87 setups, 63.9% over the last 36) matches the
# costed models, not the free one, so the costed reading is the real one.
# Validated on the GATED book in both halves: +46% and +10%, win rate up in
# both (54.7->58.1, 54.1->59.5). The ungated export disagreed on the first
# half; it describes 2414 trades production never takes.
STOP_CLOSE_CONFIRM = os.getenv("STOP_CLOSE_CONFIRM", "1") != "0"
# Exchange-side stop stays in place as a disaster backstop, widened to this
# multiple of R so it cannot fire before the close confirmation. It is what
# protects the position while the bot itself is down (deploy/restart/network).
STOP_EXCHANGE_BACKSTOP_R = float(os.getenv("STOP_EXCHANGE_BACKSTOP_R", "2.5"))

# --- Concurrent same-direction exposure cap (ported 2026-07-26) ---
# Nothing else caps total open positions: only per-symbol dedup and
# MAX_SIGNALS_PER_SCAN=3, while signals live SIGNAL_EXPIRY_HOURS, so same-side
# positions accumulate. Stock X-Perps in one direction are correlated through
# SPY and sector, so N same-side positions are not N independent trades.
#
# ENABLED at 5 on 2026-08-16, after the sweep this comment demanded. Measured on
# THIS bot's 18k-candle backtest (396 trades), replaying the cap over the merged
# trade list in entry order:
#   off  396 tr  73.7%  +225.0R  DD -7.90R
#   5    384 tr  74.2%  +225.2R  DD -6.93R   <- same profit, 12% less drawdown
#   3    333 tr  75.4%  +205.7R  DD -6.47R
#
# 5 is the only value that survives a time split. Half 1: DD -6.93 against -7.90
# for -1.0R of profit. Half 2: DD -5.67, identical to no cap, profit +1.2R.
# Better in one half, neutral in the other, worse in neither.
#
# 3 was REJECTED by that same split: it looks best overall (-6.47R) but its
# drawdown gets WORSE than no cap in half 2 (-6.47 against -5.67) while costing
# profit in both halves — the overall figure is one half flattering the other.
#
# Be honest about the size: 5's benefit is modest and concentrated in one half.
# It ships because it never hurts, not because it is a large edge. The original
# warning stands — crypto's 8 is not transferable, stock correlation runs
# through SPY, not BTC.
MAX_SAME_DIRECTION_POSITIONS = int(os.getenv("MAX_SAME_DIRECTION_POSITIONS", "5"))

# --- US market session gate (see src/market_hours.py) ---
# Signals only while NYSE/Nasdaq is open — off-session X-Perp candles are thin
# MM drift. Open-position monitoring (TP/SL) stays 24/7 regardless.
# OFF_SESSION_SIGNALS=1 → scan around the clock (admin toggle, use at own risk).
OFF_SESSION_SIGNALS = os.getenv("OFF_SESSION_SIGNALS", "0") != "0"

# --- Extended session windows (measured 2026-08-20) ---
# OFF_SESSION_SIGNALS is all-or-nothing, and that turned out to be wrong: it
# lumps the London/US-pre-market window together with genuinely dead hours.
# Measured on 12 tickers, candle volume relative to the regular session:
#   регулярная 09-16 ET   100%
#   Лондон     04-09 ET    51%   <- half of session volume, real trading
#   ночь       20-04 ET    23%
#   пост-маркет 16-20      21%
#   выходные                5%   <- effectively dead
# Candle range tells the same story: 04-08 ET runs 0.8-1.2x the average bar,
# with 08:00 ET (13:00 London) WIDER than 14:00 ET inside the regular session.
# Order books were sampled live at 06:15 ET across all 26 tickers: median spread
# 3.63 bps, worst (AAPL) 10.77, and not one above the 25 bps gate. So the
# comment claiming off-session candles are "thin MM drift" is simply untrue for
# this window.
#
# Trade outcomes from a 24/7 backtest (3587 trades), under the live gates:
#   только регулярная (сейчас)   384 сд  74.2%  +256.8R  DD  -6.43R  ratio 39.9
#   + Лондон 04-09               821 сд  71.1%  +509.1R  DD -10.85R  ratio 46.9
#   + Лондон и ночь             1152 сд  70.2%  +649.8R  DD  -8.02R  ratio 81.0
#   всё кроме выходных          1215 сд  69.9%  +665.7R  DD  -8.02R  ratio 83.0
#   всё подряд                  1528 сд  68.6%  +799.4R  DD -10.62R  ratio 75.3
#
# Note the shape: London ALONE makes drawdown worse (-6.43 -> -10.85), while
# adding the night window on top brings it back to -8.02. More trades spread
# across more hours smooth the equity curve — the pair is better than either.
#
# DEFAULT OFF. This is a genuine trade-off, not a bug fix: 2.5x the profit for
# 1.25x the drawdown and 4pp of win rate. Weekends stay excluded regardless —
# 5% volume and the only window whose halves fall apart (+0.520 -> +0.202).
# Night spreads have NOT been sampled (they fall outside working hours here);
# the volume figure says thin-but-real, and our $34 notional is tiny against it,
# but the spread is unmeasured and the honest state is "London verified, night
# inferred".
#
# Format: comma-separated ET hour ranges, e.g. "4-9" or "4-9,20-24,0-4".
# ENABLED 2026-08-20 at London + night, post-market and weekends excluded.
# That combination measured 1152 сд / 70.2% WR / +649.8R / DD -8.02R (ratio
# 81.0) against 384 сд / 74.2% / +256.8R / DD -6.43R (39.9) for the regular
# session alone: 2.5x the profit for 1.25x the drawdown, win rate still at the
# 70% target. Adding the post-market window back is worth only +16R and drags
# in the single losing hour of the day (18:00 ET, 62.2% stops, -0.239R, both
# halves negative), so it stays out.
EXTENDED_SESSION_HOURS_ET = os.getenv("EXTENDED_SESSION_HOURS_ET", "4-9,20-24,0-4").strip()


def _parse_hour_windows(raw: str) -> list:
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if "-" not in part:
            continue
        a, _, b = part.partition("-")
        try:
            lo, hi = int(a), int(b)
        except ValueError:
            continue
        if 0 <= lo < hi <= 24:
            out.append((lo, hi))
    return out


EXTENDED_SESSION_WINDOWS = _parse_hour_windows(EXTENDED_SESSION_HOURS_ET)

# --- News filter (per-coin keywords) ---
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWS_BLOCK_KEYWORDS = ["hack", "exploit", "scam", "lawsuit", "sec ", "ban", "delist", "rug"]

# --- Global macro news agent (Groq free tier) ---
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
NEWS_LOOKBACK_HOURS = 2

# --- Economic calendar warning (ForexFactory weekly XML, free) ---
# Warn on a signal when a HIGH-impact macro event (CPI/FOMC/NFP) lands within
# this many hours — high whipsaw risk around scheduled releases.
EVENT_WARN_HOURS = float(os.getenv("EVENT_WARN_HOURS", "3"))

# --- Auto-block symbols with bad recent stats ---
AUTO_BLOCK_ENABLED           = os.getenv("AUTO_BLOCK_ENABLED", "1") != "0"
AUTO_BLOCK_LOOKBACK_TRADES   = int(os.getenv("AUTO_BLOCK_LOOKBACK_TRADES", "20"))
AUTO_BLOCK_MIN_TRADES        = int(os.getenv("AUTO_BLOCK_MIN_TRADES", "8"))
AUTO_BLOCK_MAX_PROFIT_FACTOR = float(os.getenv("AUTO_BLOCK_MAX_PROFIT_FACTOR", "0.80"))
AUTO_BLOCK_MAX_WIN_RATE      = float(os.getenv("AUTO_BLOCK_MAX_WIN_RATE", "35"))
AUTO_BLOCK_DAYS              = int(os.getenv("AUTO_BLOCK_DAYS", "7"))

# --- Database ---
DB_PATH = os.getenv("DB_PATH", "stocks.db")  # Railway: set DB_PATH=/data/stocks.db

# --- Backtest ---
BACKTEST_CANDLES        = int(os.getenv("BACKTEST_CANDLES", "1152"))  # 1152 × 15m ≈ 12 days
# 192 × 15m = 48h, matching live SIGNAL_EXPIRY_HOURS. Was "48" (=48 candles=12h,
# a 4x-too-short window) — same unit-mismatch bug found and fixed 2026-07-22 in
# the sister crypto bot (candles vs hours), ported here. Distinct from the
# 2026-07-04 "expiry-clock fix" in this repo (that one fixed the HTF-fetch
# candle-count divisors for equities' ~26-bars/day session, not this window).
# Not yet independently re-measured on stock/commodity data — the crypto bot's
# corrected numbers (WR 63%->73%, understated maxDD -22R->-32R) don't transfer
# directly; re-run backtests here before trusting any pre-2026-07-22 result.
BACKTEST_TP_WINDOW      = int(os.getenv("BACKTEST_TP_WINDOW", "192"))
BACKTEST_TOP_COINS      = int(os.getenv("BACKTEST_TOP_COINS", "20"))
# 0.0005 = OKX X-Perps standard-tier TAKER fee (verified 2026-07-22 against
# okx.com/en-eu/help/okx-x-perps-eea-fees-overview: maker 0.02%, taker 0.05%).
# The bot enters and exits with market orders (src/okx_trader.py: ordType
# "market", SL/TP algo triggers market-close), so taker applies both sides.
# Was 0.001 — a crypto-bot leftover, 2x the real rate. Costs are charged as a
# fraction of R, so on this bot's 0.4-1.5% stop band that error was eating
# 0.317R/trade instead of 0.211R: it understated net edge by ~21% (+826R ->
# +1003R on the 1679-trade 2022-2026 deep set). Slippage kept at 0.05%/side,
# which is conservative for liquid stock X-Perps inside the US session.
BACKTEST_FEE_RATE       = float(os.getenv("BACKTEST_FEE_RATE", "0.0005"))
BACKTEST_SLIPPAGE_RATE  = float(os.getenv("BACKTEST_SLIPPAGE_RATE", "0.0005"))
BACKTEST_USE_BTC_FILTER = os.getenv("BACKTEST_USE_BTC_FILTER", "1") != "0"

# --- Autotrading (real OKX EU orders for allow-listed users) ---
AUTOTRADE_ENABLED           = os.getenv("AUTOTRADE_ENABLED", "1") != "0"
AUTOTRADE_LEVERAGE          = int(os.getenv("AUTOTRADE_LEVERAGE", "10"))
AUTOTRADE_BALANCE_THRESHOLD = float(os.getenv("AUTOTRADE_BALANCE_THRESHOLD", "100"))
AUTOTRADE_CONTACT           = os.getenv("AUTOTRADE_CONTACT", "@sanja_tusagang")
# Fernet key for encrypting user API keys at rest — generate once:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# and set AUTOTRADE_ENC_KEY on the host. Keys are unreadable without it.

# --- Reject cooldown + kill-switch (added after 8-SL chop cluster 2026-07-10) ---
# After Claude rejects a setup, don't re-ask the same symbol+direction while
# price is still in the same zone (1 ATR) — stops "ask every scan until yes".
REJECT_COOLDOWN_HOURS = float(os.getenv("REJECT_COOLDOWN_HOURS", "3"))
# N consecutive SL among today's closed signals → pause new signals until the
# next Riga day. 0 = off.
KILL_SWITCH_SL_STREAK = int(os.getenv("KILL_SWITCH_SL_STREAK", "3"))
