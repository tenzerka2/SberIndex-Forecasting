# Full 2016-municipality Prophet comparison

All shards completed and all forecast pairs verified. Retrospective exploratory evaluation; not an untouched test or an organizer score.

Source: b61185ef2a01714982e57c37dcd5fd33b2a8dc5c
Workflow: https://github.com/tenzerka2/SberIndex-Forecasting/actions/runs/37585934483

| Model | MAE h1 | MAE h3 | MAE h6 | MAE h12 |
|---|---:|---:|---:|---:|
| v3_hedge | 749.21 | 1049.37 | 1243.08 | - |
| v3 | 893.52 | 1168.30 | 1318.95 | - |
| v4_diversified | 815.35 | 1099.63 | 1282.67 | - |
| prophet_auto | 1578.02 | 1859.43 | 2199.41 | 2900.64 |
| prophet_log_fourier4 | 1654.88 | 1780.45 | 2174.11 | 15017.74 |
| prophet_log_month_dummies | 1107.56 | 2162.74 | 4182.79 | 171879.33 |
| seasonal_naive | 4344.62 | 4359.59 | 4248.84 | 4703.63 |
| national_yoy_fallback | - | - | - | 1416.12 |

h12 is a single-origin short-history fallback, not V3/V4. All variants retained, including unstable predictions.
Counts: h1=20160, h3=16128, h6=10080, h12=2016. All 2016 complete series; no random sample.
The original 100-series evidence was reused only after code/data/package validation. Each reused prediction is included exactly once.
