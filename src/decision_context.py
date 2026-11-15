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
    'signal_open signal_high signal_low signal_close signal_volume'.split())
LABELS=frozenset('module family direction btc_regime utc_session'.split())


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
