"""The one place the blended post-TP1 R is computed.

This arithmetic was written out eleven times inside the monitor loop, and once
more in backtest.py under another name. Three separate bugs this session came
from exactly that: a copy that was right when it was written and wrong the
moment the original moved. parity_check.py compares this against the
backtest's version so the two cannot drift apart in silence.
"""


def blended_r(tp1_close_frac: float, tp1_r: float,
              runner_frac: float, runner_r: float) -> float:
    """R of a position that banked `tp1_close_frac` of itself at TP1 and rode
    the remainder to `runner_r`.

    Pass runner_frac=0.0, runner_r=0.0 for an exit where the runner returned
    nothing — breakeven after TP1, or a TP1 position that expired.

    Rounded to four places because that is what lands in realized_r.
    """
    return round(tp1_close_frac * tp1_r + runner_frac * runner_r, 4)
