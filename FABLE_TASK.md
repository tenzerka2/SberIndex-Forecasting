# Fable verification / next-iteration task

## Goal

Independently audit and improve the SberIndex forecasting + early-warning solution without data leakage.

## Current benchmark to reproduce first

- 2,016 complete municipal series
- forecast origins: 2024-06, 2024-07, 2024-08, 2024-09
- horizons: 1, 2, 3 months
- 24,192 forecast pairs
- current ensemble: MAE 725.69 RUB, R2_growth 0.4274, wMAPE 2.302%

Do not accept any improvement until this benchmark is reproduced from scratch.

## Phase 1 — audit

1. Check temporal leakage in every feature and target.
2. Verify reconstruction of 2,016 municipal series and handling of duplicated municipality names.
3. Recompute MAE, R2_level, R2_growth, wMAPE.
4. Compare on identical forecast pairs against seasonal naive, growth-naive, factor-only and Prophet.
5. Run paired significance / bootstrap across target months and municipalities.

## Phase 2 — forecasting improvements

Try only time-safe variants:
- horizon-specific blend weights learned on earlier origins only;
- residual learner over the current local/panel ensemble;
- pooled LightGBM/CatBoost using lagged local + panel features;
- robust common-factor decomposition;
- foundation time-series model benchmark if weights/runtime are available;
- conformal or quantile uncertainty intervals.

Target: MAE < 700 RUB on the exact protocol, but reject gains that disappear on additional rolling origins.

## Phase 3 — structural-change early warning

Compare:
- CUSUM / Page-Hinkley;
- BOCPD;
- ruptures / PELT;
- supervised classifier on lagged acceleration, volatility, panel deviation and external signals.

Report precision, recall, F1, PR-AUC and false alarms per 100 series-months. Optimize threshold on training/validation only.

## Phase 4 — external/news signals

If news are added, features must be constructed strictly from publications available by forecast origin. Prefer monthly event intensity, topic/sentiment and source reliability. Provide ablation against the no-news model.

## Deliverables back

- exact commands to reproduce;
- changed files or patch;
- metrics table by origin and horizon;
- evidence that no future information was used;
- ablation table;
- any failure modes found.
