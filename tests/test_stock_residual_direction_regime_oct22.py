import pandas as pd

import stock_residual_direction_regime_oct22_audit as audit


def test_policy_can_choose_different_styles_by_regime():
    rows = []
    for year in audit.DEVELOPMENT:
        for _ in range(30):
            rows.append({"year": year, "market_strength": "quiet",
                         "style": "reversal", "signed_return": .01})
            rows.append({"year": year, "market_strength": "quiet",
                         "style": "continuation", "signed_return": -.01})
            rows.append({"year": year, "market_strength": "directional",
                         "style": "reversal", "signed_return": -.01})
            rows.append({"year": year, "market_strength": "directional",
                         "style": "continuation", "signed_return": .01})

    policy = audit.learn_policy(
        pd.DataFrame(rows), ("market_strength",), 30)

    chosen = dict(zip(policy.market_strength, policy["style"]))
    assert chosen == {"directional": "continuation", "quiet": "reversal"}
