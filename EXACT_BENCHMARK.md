# Exact contest-window benchmark (V2)

## Protocol

- target: `Все категории`, municipal cashless consumer spending
- 2,016 municipal series with complete history from 2023-01 through 2024-12
- series identities reconstructed from contiguous row blocks in the raw export, so duplicate municipality names are preserved as distinct series
- forecast origins: 2024-06, 2024-07, 2024-08, 2024-09
- horizons: 1, 2, 3 months
- 24,192 municipality-origin-horizon forecast pairs
- every forecast uses only observations available at its origin

This protocol matches the public SberIndex forecasting benchmark. As a sanity check, the implementation reproduces published factor-only and seasonal-growth control results on the same data geometry.

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

Public reference values on the same protocol:

- Prophet contest baseline: ~1,428 RUB MAE
- published three-model ensemble: ~762 RUB MAE, R² growth ~0.42

Therefore V2 is approximately 49% lower MAE than Prophet and approximately 4.8% lower MAE than the published 762-RUB ensemble on this backtest window.

## Important methodological note

The contest history is only 24 months, so model-selection uncertainty is material. The V2 equal-weight ensemble is intentionally simple and reproducible. Before final submission, add sensitivity checks across alternative origin windows and bootstrap results by target month; do not claim universal superiority beyond the tested contest window.
