# Fee model revision, 9 September 2026

Official OKX X-Perp fee overview (updated 1 September) lists base taker 0.05% per execution, maker 0.02%, and separate periodic funding. Source: https://www.okx.com/en-gb/help/okx-x-perps-eea-fees-overview

The bots submit market entries; market exits also remove liquidity. Set default BACKTEST_FEE_RATE and audit_replay fee-rate to 0.0005 per side. Retain slippage estimate 0.0001 per side: total baseline estimate 0.12% round trip. Actual account rates/rebates must be confirmed with fills; funding is still separate and not included.

Prior reports remain immutable and labeled with their original 4 basis point assumptions. Earlier +10 basis point stress scenarios correspond to 14 basis points total and remain useful, but are not a substitute for exact account cost reconciliation. Do not represent old net R as updated-cost results. Independent entry/paired diagnostic scripts with hardcoded 4 basis point costs are legacy-cost diagnostics; repricing is required before selecting a strategy.
