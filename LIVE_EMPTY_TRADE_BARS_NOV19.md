# Native no-trade OHLC must not confirm a price touch

The old monitor could announce TP/SL from an explicit zero-volume native candle,
whose OHLC may carry an older trade price. Eight regression scenarios reproduce
the old crypto behavior: long/short, fixed-BE/single-target, carried prices above
or below the bracket. This is not evidence that all client losses had this cause.

The monitor now skips explicitly empty or invalid-volume bars for barrier checks.
Trailing extrema use actual trade bars; an empty current candle cannot confirm
a new trailing exit or amend a stop. A real earlier resting TP2 touch still counts
when followed by an empty candle. Native source volume may be base or contract
volume; only positivity is used. Legacy inputs without a volume field retain
their prior behavior.

Unsent venue outcome tracking also ignores empty bars. Empty or stale terminal
counterfactuals remain unresolved rather than receiving invented touch/expiry
payoffs. Skipped bars still advance the original strategy clock. Signal deadlines
and independent exchange-side position/protection reconciliation are unchanged.
Time-exit model prices remain estimates, not verified private fills.

This change does not alter entry strategies, client margin/TP1 choices, leverage,
order types, stop/target prices, protective limits, or PAPER/LIVE authorization.
It supplies no proof of monthly profitability. The rejected research models are
not imported into the production runtime.
