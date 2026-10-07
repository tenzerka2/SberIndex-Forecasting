> Историческая версия от 6 октября 2026. Актуальные исправления, пересчитанные результаты и ограничения: [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md). Метрики старой разметки шоков не сопоставимы напрямую с новой.

# Rubric checklist — финальная версия

| Критерий | Вес | Статус | Что сделано | Доказательство / слайд | Остаточный риск |
|---|---:|---|---|---|---|
| Понятность методологии | 10% | закрыто | Интерпретируемая V3: local seasonal growth + panel factor + time-safe error feedback; V3-hedge отдельно как robust вариант | `METHOD.md`, `FINAL_REPORT.md`, слайды 3–4 | схема V3 придумана после диагностики exact, это раскрыто |
| Forecasting + горизонты 1/3/6/12 | 20% | закрыто | Отдельный benchmark на одинаковых парах; V3/V3-hedge честно не определены на h=12 из-за 24 мес. истории | `benchmarks/horizons_benchmark.py`, `outputs/horizons_metrics.csv`, слайд 5 | h=6 слабый; h=12 ограничен простыми моделями |
| Structural-change methods | 20% | закрыто | Корректная panel-relative разметка; robust jump, BOCPD, PELT, CUSUM, Page-Hinkley, supervised; detection и prediction разделены | `outputs/ew_v2_metrics.csv`, слайд 8, appendix | true prediction 1–3 мес. практически не работает |
| Foundation time-series model | 15% | закрыто | TimesFM 2.5-200M zero-shot, официальные Google weights, одинаковые forecast pairs; хуже V3 по point forecast | `benchmarks/foundation_timesfm.py`, `outputs/timesfm_predictions.csv.gz`, слайд 6 | foundation model не улучшает MAE |
| Новости / event signals | 15% | закрыто | 95 pre-registered запросов, 550 статей, 517 после date checks; publication-time guard; frozen ablation forecast и early warning | `data/news/query_plan.csv`, `src/sbx/news_global.py`, `outputs/news_*ablation.csv`, слайд 9 | новости национально/поисково разрежены и не дают устойчивого локального signal |
| MAE / R² | 10% | закрыто | V3 exact MAE 719.87 ₽, R² growth 0.4376, wMAPE 2.284%; несколько baseline и out-of-design checks | `FINAL_METRICS.csv`, слайд 5 | выигрыш V3 над V2 всего 0.8% |
| Интерпретация / воспроизводимость | 10% | закрыто | 7 time-safety tests + news test; 3 алгоритмически выбранных реальных МО; one-command rebuild | `tests/`, `benchmarks/real_examples.py`, `REAL_EXAMPLES.md`, слайд 10 | Prophet/TimesFM могут использовать cached outputs, это явно отмечено |

## Итог по rubric

Все семь критериев закрыты артефактами. Финальный тезис не строится на одном leaderboard-окне: V3 проверена на других origins и пяти категориальных панелях, а модели, которые выигрывали только на exact, были отклонены.

### News result

- корпус: 550 статей, 95 заранее зафиксированных запросов, май 2023 — ноябрь 2024;
- после консервативной проверки дат используются 517 статей; 11 query-month mismatches и 22 записи без надёжной даты исключены;
- forecast exact: V3 719.87 ₽ → V3+news 946.84 ₽, то есть новости ухудшают MAE;
- detection: PR-AUC 0.368 → 0.364; prediction: 0.0245 → 0.0269, но false alarms растут 6.5 → 15.0 на 100 ряд-месяцев;
- вывод: в таком корпусе новости не дают устойчивого полезного сигнала; pipeline и temporal alignment при этом реализованы и воспроизводимы.
