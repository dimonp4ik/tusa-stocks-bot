import pandas as pd

import stock_5m_stop_path_attribution_oct22_audit as audit


def test_rate_lift_uses_all_rows_as_baseline():
    frame = pd.DataFrame({"is_stop": [True, False, False, False],
                          "flag": [True, True, False, False]})

    result = audit.rate_lifts(frame, ["flag"])["flag"]

    assert result["stop_rate_pct"] == 50.0
    assert result["stop_rate_lift"] == 2.0
