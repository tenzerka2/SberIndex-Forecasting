> Историческая версия от 6 октября 2026. Актуальные исправления, пересчитанные результаты и ограничения: [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md). Метрики старой разметки шоков не сопоставимы напрямую с новой.

# Exact contest-window benchmark

## Protocol

- target: `Все категории`, municipal cashless consumer spending (RUB);
- 2,016 municipal series with complete history 2023-01…2024-12; series identity is reconstructed from the raw export's contiguous (municipality, category) row runs, so the 107 series sharing 43 names stay distinct (`src/sbx/data.py`, `outputs/data_audit.json`);
- forecast origins 2024-06, 2024-07, 2024-08, 2024-09; horizons 1, 2, 3 months; 24,192 forecast pairs;
- every forecast uses only data available at its origin (`tests/test_time_safety.py`);
- metrics: MAE (RUB), R² of levels, R² of log year-over-year growth (base = same month one year earlier), wMAPE.

The protocol is intended to mirror the public SberIndex forecasting benchmark. We do not quote public leaderboard or baseline numbers: none of them can be verified from this repository. All comparisons below are on identical pairs computed here.

## Results (reproducible: `./reproduce.sh`, source `FINAL_METRICS.csv`)

| Model | MAE, RUB | R² level | R² growth | wMAPE |
|---|---:|---:|---:|---:|
| seasonal naive | 4,205.64 | 0.8421 | −9.278 | 13.344% |
| seasonal growth (1 month) | 976.89 | 0.9873 | 0.044 | 3.099% |
| local_sng2 | 945.61 | 0.9881 | 0.160 | 3.000% |
| panel factor (6 months) | 1,001.46 | 0.9829 | 0.149 | 3.177% |
| V2 = 0.5·local_sng2 + 0.5·panel_factor6 | 725.69 | 0.9915 | 0.4274 | 2.302% |
| **V3 = V2 + error feedback** (benchmark model) | **719.87** | **0.9917** | **0.4376** | **2.284%** |
| V3-hedge (production model) | 730.14 | 0.9917 | 0.4320 | 2.317% |

By horizon (V2 → V3): h=1 598.35 → 592.86, h=2 722.29 → 716.98, h=3 856.42 → 849.76 RUB.

`python benchmarks/exact_backtest.py` reproduces V2 alone (MAE 725.686835) with the original script; `benchmarks/rolling_eval.py` asserts the same value before computing anything else.

Prophet, on a fixed random sample of 400 series and the same pairs (`exact_sample400` rows of `FINAL_METRICS.csv`): forced yearly seasonality (not defaults) 20,997.7 RUB (unstable with < 2 years of history), log target 1,627.1 RUB; V2 699.2 and V3 694.1 RUB on the same sample.

## How much to trust this window

- 6 target months. V3 beats V2 in 6 of 6 (one-sided sign test p = 0.016); differences of ±30 RUB between other strong models are not significant month-wise.
- V2's settings are favourable to this window: of 175 nearby configurations, 8 beat V2 here but 84 beat it on origins 2024-04, 05, 10, 11 (`outputs/v2_grid.csv`).
- Robustness evidence (other origins, five untouched category panels) is in `FINAL_REPORT.md`, sections 4.2–4.3.
