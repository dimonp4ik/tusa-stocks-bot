"""Client-selected TP1 planning with an explicit minimum-valid-volume mode."""
from decimal import Decimal,InvalidOperation,ROUND_CEILING
from src.okx_trader import calc_partial_close


def plan_tp1_close(position_sz,close_pct,spec,*,ensure_minimum=False):
    plan=dict(calc_partial_close(position_sz,close_pct,spec),minimum_adjusted=False,full_close=False)
    if not ensure_minimum:return plan
    try:
        position=Decimal(str(position_sz));pct=Decimal(str(close_pct))
        lot=Decimal(str(spec['lotSz']));minimum=Decimal(str(spec['minSz']))
        if not all(v.is_finite() for v in (position,pct,lot,minimum)) or min(position,lot,minimum)<=0:
            raise ValueError('invalid_tp1_sizing')
        if not 0<pct<=100:return dict(plan,size=0.,actual_pct=0.)
        minimum=(max(lot,minimum)/lot).to_integral_value(rounding=ROUND_CEILING)*lot
        if position%lot or position<minimum:raise ValueError('invalid_position_grid')
        desired=(position*pct/100//lot)*lot
        size=max(minimum,desired)
        if size>=position or position-size<minimum:size=position
        return dict(size=float(size),actual_pct=float(size/position*100),
            minimum_size=float(minimum),minimum_pct=float(minimum/position*100),
            minimum_adjusted=size>desired,full_close=size==position)
    except (InvalidOperation,ValueError,TypeError,KeyError,ArithmeticError):
        return dict(plan,size=0.,actual_pct=0.)


def tp1_setting_label(user):
    pct=float(user.get('tp1_close_pct') or 0)
    if pct<=0:return 'без фиксации на TP1'
    if user.get('tp1_minimum_close'):
        return f'{pct:g}% / минимум допустимый объём'
    return f'{pct:g}%'
