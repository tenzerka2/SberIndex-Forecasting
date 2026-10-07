> Историческая версия от 6 октября 2026. Актуальные исправления, пересчитанные результаты и ограничения: [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md). Метрики старой разметки шоков не сопоставимы напрямую с новой.

# PRESENTATION_DATA — финальный source map

Каждая цифра основной презентации должна иметь воспроизводимый источник. Если цифры нет здесь, её не выносим в deck.

## Слайд 1 — headline

- 2 016 полных муниципальных рядов: `outputs/data_audit.json` → complete_series_total
- exact protocol: 24 192 forecast pairs, origins 2024-06…09, h=1–3: `FINAL_METRICS.csv`
- V3 MAE **719.87 ₽**
- R² growth **0.4376**
- wMAPE **2.284%**

Можно: «V3 даёт MAE 719.87 ₽ на exact contest-style protocol».
Нельзя: «719.87 ₽ — гарантированно лучший результат конкурса».

## Слайд 2 — данные

- history: 2023-01…2024-12, 24 месяца
- complete series: 2 016
- homonym series: 107 под 43 названиями
- exact target months: 6
- source: `outputs/data_audit.json`, `FINAL_REPORT.md`

Можно: «24 192 пары разделяют всего 6 target months».
Нельзя: «24 192 независимых испытания».

## Слайд 3 — Q1 anomaly

Figure: `figures/01_q1_anomaly.png`

- municipal median YoY Jan/Feb/Mar 2024: **5.1 / 9.8 / 11.3%**
- national nominal Jan 2024: **16.3%**
- source: `outputs/common_factor_vs_national.csv`, `outputs/common_level_gap.csv`

Можно: «в муниципальной панели есть общий level anomaly относительно национального ряда».
Нельзя: «мы доказали, что это ошибка СберИндекса».

## Слайд 4 — V3 method

- V2 = 0.5 local_sng2 + 0.5 panel_factor6
- error feedback uses 2 last known idiosyncratic one-step errors
- adjacent error correlations negative in **8/9** pairs, range **−0.20…+0.07**
- source: `src/sbx/models.py`, `outputs/v2_error_autocorr.csv`, `outputs/presentation_sources.json`

## Слайд 5 — forecasting / horizons

Exact ladder:
- seasonal naive **4205.6**
- seasonal growth1 **976.9**
- V2 **725.7**
- V3 **719.9**

Source: `FINAL_METRICS.csv`, figure `figures/03_ablation.png`.

Broad horizons, all 2 016 series:
- h=1: V3 **893.5**, V3-hedge **749.2**, TimesFM-yoy **926.1**
- h=3: V3 **1168.3**, V3-hedge **1049.4**, TimesFM-yoy **1328.3**
- h=6: V3 **1318.9**, V3-hedge **1243.1**, TimesFM-yoy **1778.9**
- h=12: V3/V3-hedge not defined on observable historical origins; seasonal naive × national YoY **1522.4**, TimesFM raw **4937.2**

Prophet log, fixed 400-series sample:
- h=1 **1610.8**
- h=3 **1736.8**
- h=6 **2096.5**

Source: `outputs/horizons_metrics.csv`.

Important: h=6 R² growth is negative for all relevant models. h=12 is not an honest V3 comparison.

## Слайд 6 — TimesFM

- model: TimesFM 2.5-200M, zero-shot, frozen official Google weights
- h=1 TimesFM-yoy MAE **926.1** vs V3 **893.5** vs V3-hedge **749.2**
- h=3 TimesFM-yoy **1328.3** vs V3 **1168.3**
- h=6 TimesFM-yoy **1778.9** vs V3 **1318.9**
- 80% interval coverage h=1: TimesFM-yoy **79.67%**, V3 **84.51%**, V3-hedge **81.43%**
- source: `outputs/horizons_metrics.csv`, `outputs/horizons_intervals.csv`

Можно: «TimesFM проиграла по point forecast, но её h=1 interval ближе к номинальным 80%».
Нельзя: «foundation models хуже в принципе».

## Слайд 7 — replication

Figure: `figures/04_category_replication.png`

- V3 better than V2: **5/5 categories** on exact
- **28/30** category × target-month cells on exact
- apr–nov: V3 better in **5/5 categories**
- rejected weekly common factor: worse than V2 in **5/5 categories** on apr–nov, median +6.2%
- source: `outputs/category_replication_tests.csv`, `PRESENTATION_DATA.md` previous audit

Можно: «V3 direction replicates in all 5 untouched categories».
Нельзя: «28/30 independent confirmations».

## Слайд 8 — structural shifts

Figures: `figures/05_shift_labels.png`, appendix `figures/06_early_warning.png`

Main rolling benchmark:
- event base rate: **1.82%**
- supervised PR-AUC: **0.383**
- lift: **21.1×**
- precision / recall / F1: **0.32 / 0.55 / 0.41**
- false alarms: **2.11 / 100 series-months**
- events detected ≤2 months: **82%**
- median delay: **1 month**
- source: `outputs/ew_v2_metrics.csv`

Важно: advantage over robust jump is not statistically established.
Нельзя: «мы предсказываем будущий шок».

## Слайд 9 — news

Corpus:
- frozen query plan: **95 requests**
- collected records: **550**
- usable after conservative date checks: **517**
- date conflicts metadata vs URL: **10**
- wrong query month excluded: **11**
- no reliable date excluded: **22**
- coverage: 2023-05…2024-11
- source: `data/news/query_plan.csv`, `outputs/news_audit.json`

Frozen ablation:
- forecast exact: V3 **719.87 → 946.84 ₽** with global news residual correction
- detection PR-AUC: **0.3682 → 0.3639**
- prediction PR-AUC: **0.0245 → 0.0269**
- prediction false alarms: **6.52 → 14.97 / 100**
- source: `outputs/news_forecast_ablation.csv`, `outputs/news_early_warning_ablation.csv`

Можно: «в нашем pre-registered sparse news corpus news features do not improve the frozen model».
Нельзя: «новости бесполезны вообще».

## Слайд 10 — real examples / reproducibility

Deterministic examples:
- stable: **Кармаскалинский муниципальный район**, run_id 13015, exact MAE **346.9 ₽**
- shift: **городской округ Североуральский**, run_id 406, shift 2024-10, exact MAE **795.1 ₽**
- failure: **Нижнеколымский муниципальный район**, run_id 2242, exact MAE **5927.0 ₽**

Source: `outputs/real_examples.csv`, generated by `benchmarks/real_examples.py`.

Figures:
- `figures/municipality_stable.png`
- `figures/municipality_shift.png`
- `figures/municipality_failure.png`

Reproducibility:
- `./reproduce.sh`
- 7 core time-safety tests: `tests/test_time_safety.py`
- 4 news-specific time-safety tests: `tests/test_news_time_safety.py`
- presentation consistency: `tests/check_presentation_numbers.py`, `tests/check_final_additions.py`

## Appendix

### Prophet
Use only «fixed 400-series sample». Never say full-panel result.

### Long horizons
Use `outputs/horizons_metrics.csv`; h=12 comparison is constrained by model definability.

### Intervals
Use `outputs/horizons_intervals.csv` and `outputs/interval_coverage_by_month.csv`.

### V3 vs V3-hedge
- exact: 719.87 vs 730.14
- other origins: 921.67 vs 792.98
- apr–nov: 806.35 vs 757.07
Source: `FINAL_METRICS.csv`.

### Key limitations
- only 24 months
- exact = 6 target months
- Q1 anomaly
- 2026 vintage of national rows may include revisions
- h>3 evidence weak
- genuine 1–3 month shift prediction not useful
- news corpus is a fixed search sample, not a complete media archive
