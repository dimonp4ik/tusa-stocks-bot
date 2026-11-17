"""Recover only after sustained fresh equity inside the original loss budget."""
import math
from datetime import datetime,timezone
from src.risk_limits import equity_guard

MIN_GOOD_CHECKS=3
MIN_STABLE_SECONDS=120
MIN_CHECK_SPACING=45
MAX_CHECK_GAP=150


def checked_equity_guard(state,*,equity,now,max_daily_loss,max_drawdown,peak_floor):
    if not math.isfinite(now) or now<=0:raise ValueError('invalid observation time')
    args=dict(equity=equity,day=datetime.fromtimestamp(now,timezone.utc).date().isoformat(),
              max_daily_loss=max_daily_loss,max_drawdown=max_drawdown,peak_floor=peak_floor)
    result,blocked=equity_guard(state,**args)
    if not result['paused']:return result,blocked
    threshold=result['peak']*(1-max_drawdown+min(.001,max_drawdown/4))
    result['recovery_equity_threshold']=threshold
    possible,other_block=equity_guard(dict(result,paused=False),**args)
    if equity<=threshold or other_block is not None:
        result.update(recovery_checks=0,recovery_since=None,recovery_last_good_at=None,
            recovery_status='daily_pause' if other_block=='daily loss limit' else 'below_recovery_threshold')
        return result,blocked
    since=result.get('recovery_since');last=result.get('recovery_last_good_at')
    count=result.get('recovery_checks',0)
    valid=(isinstance(count,int) and not isinstance(count,bool) and 1<=count<=MIN_GOOD_CHECKS
           and isinstance(since,(int,float)) and isinstance(last,(int,float))
           and math.isfinite(since) and math.isfinite(last) and 0<since<=last<=now
           and now-last<=MAX_CHECK_GAP)
    if not valid:since=last=now;count=1
    elif now-last>=MIN_CHECK_SPACING:last=now;count=min(MIN_GOOD_CHECKS,count+1)
    result.update(recovery_since=since,recovery_last_good_at=last,recovery_checks=count,recovery_status='checking')
    if count>=MIN_GOOD_CHECKS and now-since>=MIN_STABLE_SECONDS:
        possible.update(recovery_since=since,recovery_last_good_at=last,recovery_checks=count,
            recovery_equity_threshold=threshold,recovery_status='recovered',last_auto_recovery_at=now)
        return possible,None
    return result,blocked


def failed_equity_check(state,now):
    result=dict(state)
    result.update(recovery_checks=0,recovery_since=None,recovery_last_good_at=None,
                  recovery_status='balance_unavailable',recovery_last_attempt_at=now)
    return result
