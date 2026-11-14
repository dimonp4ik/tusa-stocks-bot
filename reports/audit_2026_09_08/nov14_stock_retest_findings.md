# First-confirmed breakout retest

Research only; the live route and its real-order health pause are unchanged.

Six variants wait 2/4/8 closed bars after an eligible 20-bar breakout and use an initial relative-volume threshold of 1 or 1.5. The first closed retest must touch the level and close strictly back on its breakout side. Closing through the level invalidates that setup. If the first confirmed retest's next market open is inadmissible, the simulation does not select a later, more favorable entry. The stop sits beyond the pullback structure plus 0.1 ATR, with a 2R target representing 0.5-2% of price.

The test retains all current research portfolio rails, including the cumulative -2R daily cap, four positions per direction and therefore at most eight stock positions. It is a standalone nominal-size stock experiment, not a shared-account or lot-rounded execution report.

The July-August development cohorts contain only 10-20 trades per month. July returns range from +0.7461% to +1.4873%; August ranges from -0.6287% to +0.0905%. No variant meets the predeclared minimum of 30 resolved trades in each development cohort, so no winner is selected and later periods are not used to rescue a sparse development result. This family cannot currently support the requested primary-bot profit or frequency. It is retained for possible future complement studies, without a claim that it works profitably.

Tests prove that the next entry candle's high/low/close/volume do not change the already confirmed setup, a close through the level invalidates the setup, and an unacceptable first market open cannot be replaced using hindsight. Inputs, boundaries, limitations and source hashes are recorded in stock_breakout_retest_nov14.json. These eras were inspected previously and November motivated the hypothesis; no pristine forward-validation claim is made.
