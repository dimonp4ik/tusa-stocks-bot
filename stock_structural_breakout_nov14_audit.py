"""Select structural breakout geometry before revealing later months."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stock_structural_breakout_research import entries
from stock_momentum_phase_nov14_audit import ROOT, window, metrics
from stock_module_cluster_audit import portfolio_gate
import stock_xperp_native_growth_forward_monitor as market
from config import NATIVE_XPERP_DAILY_LOSS_R_CAP


def main():
    manifest = ROOT/"stock_xperp_forward_nov14_merged/manifest.json"
    candles,sources,_=market.load_manifest(manifest)
    arrays={s:{k:np.asarray(v) for k,v in c.items()} for s,c in candles.items()}
    latest=min(c["time"][-1] for c in candles.values())
    variants,frames=[],{}
    for lookback,volume,rr in itertools.product((8,20,32),(1.,1.5),(1.5,2.)):
        candidates=[]
        for symbol,candle in arrays.items():
            candidates.extend(entries(candle,symbol,lookback=lookback,minimum_volume_ratio=volume,target_r=rr))
        candidates=[r for r in candidates if r["entry_time"] <= latest-63*900]
        resolved=[]
        for row in candidates:
            value=market.resolve_row(row,arrays[row["symbol"]],row["risk_pct"]*rr*100,row["risk_pct"]*100,64)
            value["module"]="structural_breakout_research"
            resolved.append(value)
        rows=portfolio_gate(resolved,None,rolling_hour_cap=4,rolling_direction_hour_cap=2,
            rank_key=lambda r:-r["volume_ratio"],loss_streak_cap=2,direction_cap=4,
            daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP)
        frame=pd.DataFrame(rows)
        if frame.empty:
            continue
        key=f"n_{lookback}_volume_{volume}_rr_{rr}"
        selection={month:metrics(window(frame,start,end)) for month,start,end in (
            ("july","2026-07-01","2026-08-01"),("august","2026-08-01","2026-09-01"))}
        variants.append(dict(key=key,lookback=lookback,minimum_volume_ratio=volume,target_r=rr,
            candidates=len(candidates),selection=selection))
        frames[key]=frame
        print(key,flush=True)
    eligible=[v for v in variants if all(x["resolved"] >= 30 for x in v["selection"].values())]
    best=max(eligible,key=lambda v:(min(x["return_pct"] for x in v["selection"].values()),
        -max(x["dd_pct"] for x in v["selection"].values()))) if eligible else None
    report=dict(status="RESEARCH_ONLY_NOT_CONNECTED_TO_LIVE",objective_verified=False,
        selected=best,variants=variants,margin_fraction=.03,leverage=10,daily_loss_r_cap=NATIVE_XPERP_DAILY_LOSS_R_CAP,
        limitations=["Standalone nominal stock account; shared account and lot rounding not simulated",
            "No historical order-book spread or latency reconstruction",
            "Monthly selection scores are purged entry cohorts, not complete marked account months",
            "July has incomplete instrument history; earlier years unavailable for these native contracts",
            "All historical eras have been inspected in prior studies; November motivated this hypothesis",
            "No direct comparison with every production reversal guard or news blackout"],sources=sources)
    if best:
        chosen=frames[best["key"]]
        report["later"]={name:metrics(window(chosen,start,end)) for name,start,end in (
            ("september","2026-09-01","2026-10-01"),("october","2026-10-01","2026-11-01"),
            ("november_partial","2026-11-01","2026-12-01"),("since_nov11","2026-11-11","2026-12-01"))}
        chosen.to_pickle(ROOT/"stock_structural_breakout_nov14_selected.pkl")
    report["sources"].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path("stock_structural_breakout_research.py"),Path("stock_module_cluster_audit.py"))})
    (ROOT/"stock_structural_breakout_nov14.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("variants","sources")},indent=2))


if __name__ == "__main__":
    main()
