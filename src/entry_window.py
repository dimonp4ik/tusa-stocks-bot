"""Remove prices that precede the decision for a native market entry."""
import math

from src.candle_evidence import bar_has_trade


class EntryWindowUnavailable(ValueError):
    pass


def _near(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-9)


def checked_window(raw, start, end, step):
    rows={}
    for r in raw:
        if not isinstance(r,(tuple,list)) or len(r)<9 or r[8]!='1':
            raise EntryWindowUnavailable('unconfirmed entry data')
        stamp=int(r[0])//1000
        if not start<=stamp<end or int(r[0])%1000 or (stamp-start)%step:
            raise EntryWindowUnavailable('entry data outside window')
        o,h,l,c,contracts,base=(float(v) for v in r[1:7])
        if (not all(math.isfinite(v) for v in (o,h,l,c,contracts,base))
                or min(o,h,l,c)<=0 or min(contracts,base)<0
                or not l<=min(o,c)<=max(o,c)<=h):
            raise EntryWindowUnavailable('invalid entry data')
        value=(stamp,o,h,l,c,contracts,base)
        if stamp in rows and rows[stamp]!=value:
            raise EntryWindowUnavailable('conflicting entry data')
        rows[stamp]=value
    if sorted(rows)!=list(range(start,end,step)):
        raise EntryWindowUnavailable('incomplete entry data')
    return [rows[t] for t in range(start,end,step)]


def _seconds(raw, start):
    return checked_window(raw, start, start+60, 1)


def _trades(raw,second,expected):
    rows=[];identities=set()
    for r in raw:
        stamp=int(r['ts']);price=float(r['px']);size=float(r['sz']);identity=int(r['tradeId'])
        if (not second*1000<=stamp<(second+1)*1000 or not math.isfinite(price)
                or not math.isfinite(size) or price<=0 or size<=0 or identity in identities):
            raise EntryWindowUnavailable('invalid boundary trades')
        identities.add(identity);rows.append((stamp,identity,price,size))
    rows.sort(key=lambda r:(r[0],r[1]))
    if not rows:
        raise EntryWindowUnavailable('missing boundary trades')
    o,h,l,c=(rows[0][2],max(r[2] for r in rows),min(r[2] for r in rows),rows[-1][2])
    if not all(_near(a,b) for a,b in zip((o,h,l,c),expected[1:5])):
        raise EntryWindowUnavailable('boundary OHLC disagrees')
    if not _near(sum(r[3] for r in rows),expected[5]):
        raise EntryWindowUnavailable('boundary trade volume disagrees')
    return rows


def clip_native_entry_minute(frame,*,opened_at,direction,sl,tp1,tp2,now,
        load_seconds,load_second_trades):
    """Only load finer data if the partial entry minute touches a bracket.

    The source minute must be closed; its traded OHLC and volume must match
    the complete 60-second window. A level touch in the fractional publication
    second additionally requires matching public trades with millisecond time.
    Failure raises rather than inventing a TP/SL or silently dropping a loss.
    """
    opened=float(opened_at);start=int(opened//60)*60
    if not math.isfinite(opened) or direction not in ('LONG','SHORT'):
        raise EntryWindowUnavailable('invalid entry window')
    positions=[i for i,t in enumerate(frame.get('time',[])) if float(t)==start]
    if not positions or opened==start:
        return frame
    index=positions[0]
    if not bar_has_trade(frame, index):
        return frame
    touch=lambda high,low:(low<=sl or high>=min(tp1,tp2)) if direction=='LONG' else (
        high>=sl or low<=max(tp1,tp2))
    if not touch(float(frame['high'][index]),float(frame['low'][index])):
        return frame
    if now<start+60 or not (frame.get('confirmed') or [False]*len(frame['time']))[index]:
        raise EntryWindowUnavailable('entry minute still forming')
    seconds=_seconds(load_seconds(start),start)
    traded=[r for r in seconds if r[5]>0]
    aggregate=((traded[0][1],max(r[2] for r in traded),min(r[3] for r in traded),traded[-1][4])
        if traded else (seconds[0][4],)*4)
    original=tuple(float(frame[k][index]) for k in ('open','high','low','close'))
    if not all(_near(a,b) for a,b in zip(aggregate,original)):
        raise EntryWindowUnavailable('entry minute and seconds disagree')
    if 'volume_contracts' in frame:
        if not _near(math.fsum(r[5] for r in seconds), frame['volume_contracts'][index]):
            raise EntryWindowUnavailable('entry minute contracts disagree')
    elif 'volume' in frame and not _near(math.fsum(r[6] for r in seconds), frame['volume'][index]):
        raise EntryWindowUnavailable('entry minute volume disagrees')
    replacements=[]
    boundary=int(math.floor(opened));part=seconds[boundary-start]
    if opened!=boundary and part[5]>0 and touch(part[2],part[3]):
        ticks=_trades(load_second_trades(boundary),boundary,part)
        # OKX trades use contracts, while the minute frame uses base units.
        # Preserve the matched trade's volume so the monitor sees its price.
        base_per_contract = part[6] / part[5]
        replacements.extend(dict(time=t/1000,open=p,high=p,low=p,close=p,
            volume=size*base_per_contract,volume_contracts=size,confirmed=1)
            for t,_,p,size in ticks if t>=math.ceil(opened*1000))
    for r in seconds:
        if r[0]>=math.ceil(opened):
            replacements.append(dict(time=r[0],open=r[1],high=r[2],low=r[3],close=r[4],
                volume=r[6],volume_contracts=r[5],confirmed=1))
    allowed={'time','open','high','low','close','volume','volume_contracts','confirmed'}
    if set(frame)-allowed or any(len(v)!=len(frame['time']) for v in frame.values()):
        raise EntryWindowUnavailable('unsupported candle schema')
    return {k:list(v[:index])+[r[k] for r in replacements]+list(v[index+1:])
        for k,v in frame.items()}


def clip_native_entry_bar(frame, *, opened_at, bar_seconds, direction, sl, tp1, tp2, now,
                          load_minutes, load_seconds, load_second_trades):
    """Replace an overlapping strategy candle with verified later minutes/trades."""
    if (not math.isfinite(opened_at) or not math.isfinite(now) or direction not in ('LONG', 'SHORT')
            or bar_seconds not in (300, 900, 3600)):
        raise EntryWindowUnavailable('invalid strategy entry window')
    bucket = int(opened_at//bar_seconds)*bar_seconds
    if opened_at == bucket or bucket not in frame.get('time', []):
        return frame
    i = frame['time'].index(bucket)
    if not bar_has_trade(frame, i):
        return frame
    lo, hi = frame['low'][i], frame['high'][i]
    touch = (lo <= sl or hi >= min(tp1,tp2)) if direction == 'LONG' else (hi >= sl or lo <= max(tp1,tp2))
    if not touch:
        return frame
    expiry = bucket+bar_seconds
    covered = min(expiry, int(now//60)*60)
    first = int(opened_at//60)*60
    if covered <= first:
        raise EntryWindowUnavailable('entry minute still forming')
    minutes = checked_window(load_minutes(bucket, covered), bucket, covered, 60)
    if covered == expiry and frame.get('confirmed', [False]*len(frame['time']))[i]:
        traded = [r for r in minutes if r[5]>0]
        if not traded:
            raise EntryWindowUnavailable('entry strategy bar has no trades')
        aggregate = traded[0][1], max(r[2] for r in traded), min(r[3] for r in traded), traded[-1][4]
        if not all(_near(val,frame[key][i]) for key,val in zip(('open','high','low','close'),aggregate,strict=True)):
            raise EntryWindowUnavailable('entry strategy bar and minutes disagree')
        if 'volume_contracts' in frame and not _near(math.fsum(r[5] for r in minutes),frame['volume_contracts'][i]):
            raise EntryWindowUnavailable('entry strategy contracts disagree')
    keys = ('time','open','high','low','close','volume_contracts','volume')
    minute_frame = {key: [r[col] for r in minutes if r[0]>=first] for col,key in enumerate(keys)}
    minute_frame['confirmed'] = [1]*len(minute_frame['time'])
    reduced = clip_native_entry_minute(minute_frame, opened_at=opened_at, direction=direction,
        sl=sl, tp1=tp1, tp2=tp2, now=now, load_seconds=load_seconds, load_second_trades=load_second_trades)
    # Close-confirmed strategy stops may inspect only the last price of a full strategy bar.
    reduced['confirmed'] = [0]*len(reduced['time'])
    if reduced['confirmed'] and covered == expiry:
        reduced['confirmed'][-1] = 1
    return {key: list(values[:i])+reduced[key]+list(values[i+1:]) for key,values in frame.items()}
