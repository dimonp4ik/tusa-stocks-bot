# Hourly hypothesis experiment

Signal features use only completed hourly candles. Execution begins at the matching 15-minute open, with hard stops and full final target exits. No partial TP1, 2 ATR stop, maximum 48h holding time, estimated fees/slippage 12bps round trip. Cost must not exceed 25% of gross target distance. Data files are validated against frozen cache hashes; holding windows with missing 15-minute bars are skipped.

Three entry families and four targets were specified before execution. All results retained. This is exploration on previously inspected SWAP history, not live X-Perp performance or untouched validation. Equal R sizing differs from existing SMC risk weights. Funding remains omitted.

The longer horizon did not produce the requested combination of high win rate and durable profit. The crypto 2R breakout is a research lead with low win rate and uneven period results, not a deployment candidate. Other economically different information sources should be investigated instead of tuning small targets until a high historical win rate appears.
