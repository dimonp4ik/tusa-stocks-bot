"""Frozen-grid stock continuation/reversal comparison, never enabled live."""
import hashlib
import itertools
import json
from pathlib import Path
import pandas as pd
import numpy as np
import stock_xperp_native_cross_section_oct18_audit as feature
import stock_xperp_native_growth_forward_monitor as market
import stock_xperp_native_volatility_guard_oct18_audit as volatility
import stock_xperp_native_quality_guard_oct18_audit as quality
import stock_xperp_basket_trend_forward_oct30_audit as measure
from stock_module_cluster_audit import portfolio_gate
from stock_direction_phase_research import direction_for
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP

ROOT = Path("reports/audit_2026_09_08")
POLICIES = ("reversal", "momentum", "momentum_aligned", "mixed_strong_trend")


def metrics(frame):
    v = measure.metrics(frame, .03)
    return {k:v[k] for k in ("n","resolved","pending","return_pct","dd_pct",
        "profitable_exit_rate_pct","max_loss_streak","trades_per_active_day")}


def window(frame, start, end):
    a,b = [pd.Timestamp(x,tz="UTC").timestamp() for x in (start,end)]
    # No exit from a later interval may influence its selection score.
    return frame[frame.entry_time.ge(a) & frame.exit_time.lt(b)]


def main():
    manifest = ROOT/"stock_xperp_forward_nov14_merged/manifest.json"
    candles, sources, _ = market.load_manifest(manifest)
    all_features = feature.feature_rows(candles,2)
    # Reuse arrays instead of copying a full symbol history for every entry.
    arrays = {symbol:{key:np.asarray(values) for key,values in c.items()}
              for symbol,c in candles.items()}
    context = all_features.groupby("entry_time",as_index=False).cross_mean.first().sort_values("entry_time")
    context["basket_trend_64"] = context.cross_mean.rolling(64,min_periods=64).sum()
    entries = feature.select_entries(all_features,2,.004)
    entries = volatility.add_atr14(entries,arrays)
    entries = entries[entries.atr14_fraction.le(.01)].merge(
        context[["entry_time","basket_trend_64"]],on="entry_time",validate="many_to_one").dropna()
    last = min(c["time"][-1] for c in candles.values())
    # Keep a mature entry cohort so quick stops near the snapshot cannot be
    # compared with winners whose target window is still incomplete.
    entries = entries[entries.entry_time.le(last-63*900)].copy()
    variants, portfolios = [], {}
    for policy,target in itertools.product(POLICIES,(2.,2.5,3.)):
        part = entries.copy()
        part["direction_n"] = [direction_for(signal_return=r.signal_return,
            basket_trend_64=r.basket_trend_64,policy=policy) for r in part.itertuples()]
        part = part.dropna(subset=["direction_n"])
        resolved = pd.DataFrame(market.resolve_entries(part,arrays,
            dict(target_pct=target,stop_pct=1.,hold_bars=64)))
        resolved["module"] = policy
        rows = portfolio_gate(resolved.to_dict("records"),None,rolling_hour_cap=4,
            rolling_direction_hour_cap=2,rank_key=quality.ranker("z_high"),
            loss_streak_cap=2,direction_cap=4,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
        frame = pd.DataFrame(rows,columns=resolved.columns)
        key = f"{policy}_{target}"
        portfolios[key] = frame
        selection = {month:metrics(window(frame,f"2026-{month}-01",end))
            for month,end in (("07","2026-08-01"),("08","2026-09-01"))}
        variants.append(dict(key=key,policy=policy,target_pct=target,selection=selection))
        print(key,flush=True)
    # Rank once on July-August, before revealing September/October/November.
    eligible = [v for v in variants if all(v["selection"][m]["resolved"] >= 50 for m in ("07","08"))]
    winner = max(eligible,key=lambda v:(
        min(v["selection"][m]["return_pct"] for m in ("07","08")),
        -max(v["selection"][m]["dd_pct"] for m in ("07","08")))) if eligible else None
    report = dict(status="RESEARCH_ONLY_NO_LIVE_CHANGES",selected=winner,variants=variants,
        mature_cohort_last_entry_utc=pd.Timestamp(last-63*900,unit="s",tz="UTC").isoformat(),
        margin_fraction=.03,leverage=10,stop_pct=1,hold_bars=64,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP,
        limitations=["Native instrument history begins in mid-2026; no earlier stock X-Perp evidence",
            "Monthly scores are purged entry cohorts, not complete marked account months",
            "Historical months have been inspected in previous strategy research",
            "Standalone stock account, not shared crypto-stock ledger",
            "Gross nominal-size outcomes; no client lot rounding or fees",
            "Entry starts at bar open; real publication latency still needs separate replay",
            "New families use ATR and cross-sectional filters; current reversal-specific guards are not reused",
            "Warm-up and listings vary; July is partial coverage"],sources=sources)
    if winner:
        frame = portfolios[winner["key"]]
        report["later"] = {name:metrics(window(frame,start,end)) for name,start,end in (
            ("september","2026-09-01","2026-10-01"),
            ("october","2026-10-01","2026-11-01"),
            ("november_partial","2026-11-01","2026-12-01"),
            ("new_since_nov11","2026-11-11","2026-12-01"))}
        frame.to_pickle(ROOT/"stock_momentum_phase_nov14_selected.pkl")
        report["objective_verified"] = False
    report["sources"].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path("stock_direction_phase_research.py"),Path("stock_module_cluster_audit.py"))})
    out = ROOT/"stock_momentum_phase_nov14.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("variants","sources")},indent=2))


if __name__ == "__main__":
    main()
