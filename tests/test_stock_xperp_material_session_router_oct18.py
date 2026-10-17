import pandas as pd

import stock_xperp_material_session_router_oct18_audit as audit


def test_fixed_session_bucket_boundaries():
    frame = pd.DataFrame({"entry_time": [
        pd.Timestamp(value, tz="UTC").timestamp() for value in
        ("2026-10-01 14:29", "2026-10-01 14:30",
         "2026-10-01 16:30", "2026-10-01 18:30")
    ]})
    assert audit.attach_bucket(frame).session_bucket.tolist() == [
        "open", "early", "mid", "close"]


def test_bucket_gate_ignores_win_rate():
    values = {name: {"n": 5, "return_pct": 0.1, "dd_pct": 4.0,
                     "wr": 10.0}
              for name in ("development", "validation", "confirmation")}
    assert audit.bucket_ok(values)


def test_bucket_gate_requires_every_pre_fresh_period_positive():
    values = {name: {"n": 5, "return_pct": 0.1, "dd_pct": 4.0}
              for name in ("development", "validation", "confirmation")}
    values["confirmation"]["return_pct"] = -0.01
    assert not audit.bucket_ok(values)
