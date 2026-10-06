# Exact contest-window benchmark (V2)

## Protocol

- target: `Все категории`, municipal cashless consumer spending
- 2,016 municipal series with complete history from 2023-01 through 2024-12
- series identities reconstructed from contiguous row blocks in the raw export, so duplicate municipality names are preserved as distinct series
- forecast origins: 2024-06, 2024-07, 2024-08, 2024-09
- horizons: 1, 2, 3 months
- 24,192 municipality-origin-horizon forecast pairs
- every forecast uses only observations available at its origin

This protocol is intended to match the public SberIndex forecasting benchmark. NOTE (audit 2026-10): the repository contains no evidence that published factor-only or seasonal-growth control numbers are reproduced; treat that statement as unverified.

## V2 forecast

The current V2 candidate is an equal-weight ensemble of two deliberately simple, leakage-free components:

1. **Local seasonal-growth component (`sng2`)**: last-year level for the target month multiplied by the municipality's mean log year-over-year growth over the last two observed months.
2. **Panel common-factor component (`factor6`)**: the municipality's current log-deviation from the cross-sectional median is held fixed, while the median panel path advances using its mean log year-over-year growth over the last six observed months.

Final prediction:

`forecast = 0.5 * sng2 + 0.5 * factor6`

The two components have different error structure: the local model reacts to municipality-specific momentum, while the panel model shrinks short-series noise toward a common trajectory. Their equal-weight average lowers MAE without fitting ensemble weights.

## Results

- **MAE:** 725.69 RUB/month
- **R² level:** 0.9915
- **R² year-over-year growth:** 0.4274
- **wMAPE:** 2.302%

By horizon:

| Horizon | MAE, RUB | R² growth |
|---:|---:|---:|
| 1 month | 598.35 | 0.578 |
| 2 months | 722.29 | 0.436 |
| 3 months | 856.42 | 0.259 |

Public reference values on the same protocol (quoted, NOT verified in this repository):

- Prophet contest baseline: ~1,428 RUB MAE
- published three-model ensemble: ~762 RUB MAE, R² growth ~0.42

If those quoted values are on identical pairs, V2 is ~49% below Prophet and ~4.8% below the 762-RUB ensemble. Our own Prophet on a 400-series sample of the same pairs: 1,627 RUB (log target) vs 699 RUB for V2.

## Important methodological note

The contest history is only 24 months, so model-selection uncertainty is material. The V2 equal-weight ensemble is intentionally simple and reproducible. Before final submission, add sensitivity checks across alternative origin windows and bootstrap results by target month; do not claim universal superiority beyond the tested contest window.

## V3 update (audit, 2026-10)

Reproduced from scratch: MAE 725.686835, R2_growth 0.427449, wMAPE 2.30244% (asserted in `benchmarks/rolling_eval.py`).

| Model | MAE exact | R² growth | wMAPE | MAE other origins (2024-04, 05, 10, 11) |
|---|---:|---:|---:|---:|
| V2 | 725.69 | 0.4274 | 2.302% | 925.1 |
| **V3 = V2 + error feedback** | **719.87** | **0.4376** | **2.284%** | 921.7 |
| V3-hedge (+ error feedback) | 730.14 | 0.4320 | 2.317% | 793.0 |

V3 by horizon: h=1 592.86, h=2 716.98, h=3 849.76 RUB.

Caveats found in the audit (details in `REPORT.md`):

- V2's window choices rank 8/175 on this window but 84/175 on other origins; its edge here comes from offsetting biases of its two components, caused by a Q1-2023/Q1-2024 level anomaly in the municipal panel that enters the 6-month common-growth window.
- The window has 6 target months. Bootstrap over target months gives a 95% CI of [−34, +32] RUB for V3-hedge minus V2, i.e. differences of this size are not significant.
- Prophet with default settings is unstable on < 2 years of history (MAE ~21k RUB on a 400-series sample); a log-target Prophet gives 1,627 RUB on the same pairs (V2: 699).
- MAE < 700 RUB was not reached by any time-safe method.

Final models, replication on the five category panels and the review of remaining risks: `FINAL_REPORT.md`.
