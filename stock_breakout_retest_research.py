"""First-confirmed retest of an eligible closed-bar breakout; research only."""
import numpy as np
from stock_structural_breakout_research import entries


def retest_entries(candle, symbol, *, maximum_wait=4, minimum_volume_ratio=1., target_r=2.):
    if maximum_wait not in (2,4,8):
        raise ValueError("unsupported retest expiry")
    a={k:np.asarray(candle[k]) for k in ("time","open","high","low","close","volume")}
    anchors=entries(a,symbol,lookback=20,minimum_volume_ratio=minimum_volume_ratio,target_r=target_r)
    output=[]
    seen=set()
    for anchor in anchors:
        i=anchor["entry_index"]-1
        direction=anchor["direction_n"]
        level=float((a["high"][i-20:i].max() if direction>0 else a["low"][i-20:i].min()))
        tr=np.maximum(a["high"][i-13:i+1]-a["low"][i-13:i+1],np.maximum(
            abs(a["high"][i-13:i+1]-a["close"][i-14:i]),
            abs(a["low"][i-13:i+1]-a["close"][i-14:i])))
        buffer=.1*float(tr.mean())
        for j in range(i+1,min(i+maximum_wait+1,len(a["time"])-1)):
            if a["time"][j]-a["time"][j-1] != 900:
                break
            if direction*(a["close"][j]-level) < 0:
                break
            span=a["high"][j]-a["low"][j]
            if span<=0 or a["volume"][j]<=0 or np.count_nonzero(a["volume"][j-7:j+1]>0)<6:
                continue
            location=(a["close"][j]-a["low"][j])/span
            touched=(a["low"][j]<=level+buffer if direction>0 else a["high"][j]>=level-buffer)
            confirmed=((location>=.6 if direction>0 else location<=.4)
                       and direction*(a["close"][j]-level)>0)
            if not touched or not confirmed:
                continue
            # This is the first confirmed retest. If its market open is no
            # longer admissible, never cherry-pick a later, better-priced bar.
            opened=float(a["open"][j+1])
            stop=float(a["low"][i+1:j+1].min()-buffer if direction>0
                       else a["high"][i+1:j+1].max()+buffer)
            risk=direction*(opened-stop)/opened if opened>0 else np.nan
            key=(int(a["time"][j+1]),direction)
            if (a["time"][j+1]-a["time"][j]==900 and direction*(opened-level)>0
                    and np.isfinite(risk) and .005<=risk*target_r<=.02 and risk<=.01 and key not in seen):
                seen.add(key)
                output.append(dict(anchor,symbol=symbol,entry_index=j+1,
                    entry_time=int(a["time"][j+1]),structural_stop=stop,risk_pct=risk,
                    signal_time=int(a["time"][j]),breakout_signal_time=int(a["time"][i]),
                    retest_level=level,retest_wait_bars=j-i))
            break
    return output
