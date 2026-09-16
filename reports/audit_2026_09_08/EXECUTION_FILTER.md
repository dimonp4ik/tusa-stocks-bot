# Local execution filter

Before a new market entry, request an uncached 50-level X-Perp order book after leverage setup. Reject missing, malformed, crossed, unordered, stale (>3 seconds), or materially future-dated (>1 second) books. Reject spread above 5 basis points, insufficient displayed contracts, or worst displayed fill more than 5 basis points from the best quote. Thresholds are conservative engineering defaults, not optimized evidence of an edge.

For accepted books, require the worst displayed fill to remain inside the original SL/TP1 bracket. Recalculate allowed contracts at that worst price using the existing trade and portfolio risk budgets, and never increase the prior quantity. The book estimate does not reserve liquidity: prices, queue/depth and actual execution can change before acceptance. Existing position exits and emergency closes are unaffected by this new-entry guard.

Validation: 44 tests pass in each repository, including full mocked entry flow proving unavailable/stale/wide books cannot send another market entry, both directions, VWAP across levels, insufficient volume and slippage. Public read-only probes on BTC, ETH, NVDA and XAU minimum contract sizes passed at the observation time. This snapshot proves API compatibility, not durable liquidity or strategy profitability. No live trades, deployments, commits or pushes performed.

Next task: hold SWAP-generated entry signals fixed and evaluate their execution against X-Perp candle history, rather than changing both signal source and execution simultaneously. Minute-level trailing and loss cases need further verification. Account fills and funding remain unreconciled.
