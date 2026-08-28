# Significance Check

Generated: 2026-08-28T13:28:40+00:00
Mode: `unpaired`
Pair key: `entry`
Risk mult: `off`
Paired rows: `1302`
Full paired rows: `1165`
Entry paired rows: `1302`

## Observed

- baseline net: `789.004488`
- candidate net: `817.565493`
- delta net: `28.561005`
- delta R/tr: `0.02397219`

## Bootstrap

- runs: `5000`
- p_gt_zero_net: `0.6332`
- p_gt_zero_rpt: `0.6488`
- p05_delta_net_r: `-114.271294`
- p50_delta_net_r: `29.210938`
- p95_delta_net_r: `164.301505`
- p05_delta_rpt: `-0.08439307`
- p50_delta_rpt: `0.02448483`
- p95_delta_rpt: `0.12696043`

## Rule

Treat weak improvements as suspicious when bootstrap lower-tail delta is near
or below zero. For risk-only overlays, full paired mode is expected.
For exit-policy experiments, entry-paired mode is expected because
the same entries can intentionally produce different exits/outcomes.
