"""Bounded scalar evidence from deterministic rules, without credentials/text."""
import json
import math
import re

NUMERIC=frozenset('signal_bar_ts entry entry_time entry_reference_price entry_drift_pct '
    'target_r runner_target_r hold_bars model_quality risk_pct model_risk_pct eff_ratio z '
    'cross_z signal_return cross_mean cross_std universe_n atr14_fraction backing_level '
    'backing_level_lookback_bars basket_trend_16 basket_trend_32 basket_trend_64 '
    'basket_trend_96 liquid_universe_ratio btc_fast_slow_atr btc_eff20 btc_return20_atr '
    'btc_z20 btc_atr_pct btc_slow_long_atr btc_return96_atr btc_eff96 '
    'signal_open signal_high signal_low signal_close signal_volume '
    'reference_observed_at reference_bar_ts reference_volume reference_high reference_low reference_close '
    'market_observed_at market_reference_price'.split())
LABELS=frozenset('module family direction btc_regime utc_session reference_sample_kind market_sample_kind'.split())


def forming_entry_evidence(candles, observed_at):
    """Capture a forming 15m sample with its local receipt time."""
    try:
        if isinstance(observed_at, bool):
            return {}
        stamp = float(observed_at)
        bar = float(candles['time'][-1])
        if (bool(candles['confirmed'][-1]) or not math.isfinite(stamp)
                or not bar <= stamp < bar + 900):
            return {}
        values = {name: float(candles[key][-1]) for name, key in (
            ('reference_high', 'high'), ('reference_low', 'low'), ('reference_close', 'close'))}
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            return {}
        if not values['reference_low'] <= values['reference_close'] <= values['reference_high']:
            return {}
        result = dict(
            reference_sample_kind='forming_15m_snapshot', reference_observed_at=stamp,
            reference_bar_ts=bar, market_sample_kind='forming_15m_last', market_observed_at=stamp,
            market_reference_price=values['reference_close'], **values)
        if candles.get('volume'):
            volume = float(candles['volume'][-1])
            if math.isfinite(volume) and volume >= 0:
                result['reference_volume'] = volume
        return result
    except (KeyError, IndexError, TypeError, ValueError, OverflowError):
        return {}


def context_json(value):
    if not isinstance(value,dict):
        return None
    clean={}
    for key in NUMERIC:
        item=value.get(key)
        if isinstance(item,(int,float)) and not isinstance(item,bool) and math.isfinite(item):
            clean[key]=item
    for key in LABELS:
        item=value.get(key)
        if isinstance(item,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,96}',item):
            clean[key]=item
    if not clean:
        return None
    return json.dumps(dict(schema_version=1,decision_evidence=clean),
        sort_keys=True,separators=(',',':'),allow_nan=False)
