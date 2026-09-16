# Independent entry experiments

Three prespecified families: range mean reversion (20-bar z score outside ±2, efficiency below 0.3), trend pullback (20/96 moving averages and reversal confirmation), 20-bar close breakout aligned with 96-bar mean. Stops: 2 ATR, full final targets 0.25/0.5/1/2 R, maximum 48 bars. Features use only closed bars. No partial take-profit, averaging, or martingale. Fixed final exits are research alternatives to the user's current trailing exit, not live configuration changes.

Both universes use hash-verified existing manifests (18 crypto, 26 stocks). Equal R sizing differs from legacy SMC size multipliers; raw R totals are not account returns and not directly comparable to legacy totals. Estimated round-trip fee plus slippage is 4 basis points, stress adds 10 basis points, funding omitted. No new holdout claim: this history has already been inspected. Every candidate is saved; none is automatically deployed.

Causality tests change every future candle and compare features against truncated history, with a fixture that actually produces an entry. 33 tests passed in each repository.

Next hypothesis, before testing: an entry's projected gross target must materially exceed transaction costs. Small targets during quiet stock sessions may have no plausible net edge even with a high gross hit rate. Test a prespecified cost/target ceiling, replay portfolio constraints, and compare time periods without selecting tickers from their realized profit.

Methodological reference: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf — repeated optimization raises false-discovery risk; a profitable historical winner alone is inadequate evidence.
