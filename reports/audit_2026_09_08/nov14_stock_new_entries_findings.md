# Native stock entry experiments, November 14

No entry prototype is approved for live trading. Neither establishes +30% per month. The real-order forward-health pause is unchanged.

## Data and account assumptions

The fresh public snapshot appends 16,192 symbol-bars across 46 available instruments to the preserved November-11 history. Every overlapping candle matches exactly; the native strategy uses 44 non-index instruments. Candidate entries need a complete 64-bar outcome horizon, so the latest immature entries are excluded.

Both studies use nominal client margin 3%, leverage 10, two entries per scan, four per rolling hour, two per direction per rolling hour, four concurrent positions per direction, three-hour symbol/direction cooldown, a latched pause after two losses and the configured cumulative daily -2R pause. Thus the standalone stock book cannot exceed eight positions. This is not a replay of the common crypto-stock account.

The cumulative -2R gate was already present in the working-tree research helper but had not been committed or supplied to the first new comparisons. Its behavior is now tested, explicitly supplied by both drivers and included in their source hashes. This corrects research/live parity; it does not add another live trading restriction.

Monthly scores are purged entry cohorts: entry is in the interval and exit must be before its end. They exclude inherited positions and boundary exits, and are not full marked account-month returns. The November-11 calendar cohort starts at 00:00 UTC; it overlaps the first hours of the prior snapshot and is not an untouched forward sample. These eras have also been examined in previous research. No monthly extrapolation is made.

## Study 1: change direction on the same cross-sectional extremes

Twelve fixed combinations compare reversal, plain momentum, basket-aligned momentum and a strong-trend direction switch, each with a 2/2.5/3% target and 1% stop. Selection uses July-August only. Momentum alternatives lose already in development; reversal at 2.5% wins that comparison.

## Study 2: a different entry trigger

Twelve structural breakout combinations use a closed-bar break of an 8/20/32-bar extreme, a 20-vs-96-bar trend filter, positive volume in at least six of eight closed bars, relative volume at least 1 or 1.5, and a close near the breakout candle's extreme. Entry is at the next bar's open. The stop is beyond the prior five-bar structure plus 0.1 ATR. The proposed target is 1.5R or 2R and must represent a material 0.5-2% price move, with risk no larger than 1%.

The development winner uses a 20-bar level, volume at least its prior median and 2R. Changing the next candle's high, low, close or volume cannot change an already formed signal; regression tests exercise this explicitly.

| Purged cohort | Cross-sectional reversal control | Structural breakout |
|---|---:|---:|
| September | +27.0388% | +4.4853% |
| October | +19.8669% | +2.6903% |
| November partial | -0.6862% | -2.0264% |
| November 11 calendar onward | -2.7550% | +0.6344% |

These gross, nominal-size returns exclude lot rounding, historical spread, publication latency and fees. The reversal control omits current reversal-specific backing/premarket filters; it is not the deployed strategy's account PnL. A positive short cohort does not compensate for failure over the broader November interval. Neither winner is promoted.

## Computation and reproduction

The research drivers allocate OHLC arrays once instead of rebuilding full arrays for every candidate. Tests establish identical gap and stop-first outcomes between list and array inputs. A same-candle stop/target ambiguity counts the stop first. The input manifests and source files are hashed in the JSON reports. Existing local helper modules and market-data archives are required to rerun these drivers.
