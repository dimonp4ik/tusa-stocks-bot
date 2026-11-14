# Same-benchmark ETF capacity experiment

Direxion identifies SOXL/SOXS as daily positive/inverse 3x exposures to the same semiconductor benchmark: https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs . This study maps trade direction to that benchmark direction; it does not assume exact intraday beta or hedging.

The optional research guard caps one active SOXL/SOXS position per effective benchmark direction. It runs before portfolio admission, so replacements are replayed rather than discarded from a completed-trade list. Opposite effective directions remain separate. The guard is not connected to live trading.

The current route preportfolio was rebuilt through the November-14 snapshot and its mature November-11 prefix matched the reference exactly. All current reversal backing/premarket filters and the -2R daily pause are retained. The cap removes 47 original admissions and adds 31 replacements.

It does not pass a broad improvement criterion. In the standalone September purged cohort, return changes from +28.5209% to +24.1760% and cash drawdown from 2.3578% to 2.8916%. October also worsens. Partial November improves slightly. These are nominal isolated-stock counterfactuals, not actual account-month PnL.

The crypto repository's shared_factor_account_nov14.json repeats the comparison on a single $125.64 lot-rounded account, with client margin 3% for each bot, 4/8 local slots, global limit 12, readiness from the last 20 prior shadow exits and conservative two-writer headroom. That comparison also fails to justify promotion. Its exact timing, current-specification and cohort limitations are recorded there.

Array allocation is optimized only in the research ATR/outcome helpers. Regression tests compare batch resolution with the original single-row calculation, including gap and same-bar stop-first cases. New exposure tests cover inverse direction, replacement admission and strict capacity release after an exit timestamp.
