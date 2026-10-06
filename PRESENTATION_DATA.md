# Данные для презентации

Каждое число ниже берётся из файла, который заново создаёт `./reproduce.sh` (исключение помечено: прогнозы Prophet кэшированы, пересчёт через `REFIT_PROPHET=1`). Формат источника: `файл` → строка/поле. Округления: MAE до рубля или копейки, доли до 0.1 п.п.

Окна: exact = origins 2024-06…09, h = 1–3 (24 192 пары); other = origins 2024-04, 05, 10, 11; apr_nov = 2024-04…11; feb_mar = 2024-02, 03.

---

## Слайд 1. Задача и главный результат

| Число | Значение | Источник |
|---|---|---|
| MAE V3, exact | 719.87 ₽ | `FINAL_METRICS.csv` → window=exact, h=1-3, model=v3, MAE = 719.8702 |
| R² growth V3 | 0.438 | там же, R2_growth = 0.4376 |
| wMAPE V3 | 2.28% | там же, wMAPE_pct = 2.2840 |
| MAE V2, exact | 725.69 ₽ | `FINAL_METRICS.csv` → model=v2_ensemble, MAE = 725.6868 |
| ΔMAE V3 − V2 | −5.82 ₽ (−0.8%) | `outputs/rolling_summary.json` → bootstrap_vs_v2.exact.v3.diff = −5.82 |
| Месяцев, где V3 лучше | 6 из 6 | там же, months_better / months_total |
| Число прогнозов | 24 192 | `FINAL_METRICS.csv` → n |

Figure: нет (ключевая цифра).

Можно утверждать: «V3 точнее V2 на конкурсном протоколе в каждом из 6 целевых месяцев»; «выигрыш небольшой, 0.8%».
Нельзя: «существенно лучше», «лучшее решение конкурса», сравнение с публичным лидербордом.

## Слайд 2. Почему задача сложная

| Число | Значение | Источник |
|---|---|---|
| Месяцев истории | 24 (2023-01…2024-12) | `outputs/data_audit.json` → periods |
| Полных рядов | 2 016 | `outputs/data_audit.json` → complete_series_total |
| Неполных рядов | 174 | там же, incomplete_series_total |
| Названий у нескольких МО | 49 | там же, names_with_several_municipalities |
| Рядов-омонимов среди полных | 107 под 43 названиями | там же, homonym_series, homonym_names |
| Целевых месяцев в exact | 6 (2024-07…2024-12) | `outputs/rolling_summary.json` → bootstrap_vs_v2.exact.v3.months_total |
| Связываемых с категориями МО | 1 909 | `outputs/data_audit.json` → category_linkable_series |

Figure: нет.

Можно: «независимых проверок по сути 6, а не 24 192»; «ряды восстановлены по структуре выгрузки и проверены автоматически».
Нельзя: «мы восстановили настоящие id МО» (id в данных нет).

## Слайд 3. Аномалия Q1 и риск переобучения

| Число | Значение | Источник |
|---|---|---|
| Рост г/г медианы МО, янв / фев / мар 2024 | 5.1% / 9.8% / 11.3% | `outputs/common_factor_vs_national.csv` → muni_median_yoy_pct |
| Национальный рост г/г, янв / фев / мар 2024 | 16.3% / 19.7% / 18.4% | там же, national_yoy_nominal_pct |
| Рост медианы МО апр–дек 2024 | 14.9–18.7% | там же |
| Отклонение уровня медианы МО от национального, фев / мар 2023 | +9.6% / +10.0% | `outputs/common_level_gap.csv` |
| То же, январь 2024 | −7.7% | там же |
| Конфигураций сетки V2 лучше V2 на exact / на other | 8 / 84 из 175 | `outputs/presentation_sources.json` → v2_grid |
| Лучшая конфигурация сетки на exact | 715.60 ₽ exact, 931.54 ₽ other | там же, best_exact_MAE_exact / best_exact_MAE_other |

Figure: `figures/01_q1_anomaly.png`.

Можно: «сдвинут общий уровень всех МО, что похоже на особенность данных»; «параметры V2 оказались удачными именно для конкурсного окна».
Нельзя: «это ошибка СберИндекса» (причину мы не знаем), «V2 был подогнан намеренно».

## Слайд 4. Метод

| Число | Значение | Источник |
|---|---|---|
| Пар соседних origins с отрицательной корреляцией ошибок V2 | 8 из 9 | `outputs/presentation_sources.json` → v2_error_autocorr_negative_pairs |
| Диапазон корреляций | от −0.20 до +0.07 | `outputs/v2_error_autocorr.csv` |
| Лагов в error feedback | 2 | `src/sbx/final.py` |

Figure: нет (формула словами).

Можно: «коэффициенты поправки оцениваются только по прошлым данным на каждую дату прогноза»; «модель полностью интерпретируема».
Нельзя: «схема V3 выбрана без взгляда на конкурсное окно» (выбор 2 лагов сделан с оглядкой на него; это раскрыто в FINAL_REPORT, review #1).

## Слайд 5. Benchmark

| Модель | MAE exact, ₽ | Источник |
|---|---:|---|
| seasonal naive | 4 205.64 | `FINAL_METRICS.csv` → exact, 1-3 |
| seasonal growth (1 мес.) | 976.89 | там же |
| local_sng2 | 945.61 | там же |
| panel factor 6m | 1 001.46 | там же |
| V2 | 725.69 | там же |
| V3 | 719.87 | там же |
| Выигрыш смеси над лучшей компонентой | −23.3% (725.69 / 945.61) | вычисление по двум строкам выше |
| Обучаемые и сложные модели | 760.40 (веса по горизонтам), 798.76 (pooled LightGBM), 833.25 (CatBoost на остатках), 846.98 (LightGBM на остатках), 884.12 (SVD ранга 4) | `outputs/ablation.csv` → MAE_exact |
| Prophet, выборка 400 рядов | 1 627.15 (лог), 20 997.68 (по умолчанию); V3 на той же выборке 694.06 | `FINAL_METRICS.csv` → window=exact_sample400 (Prophet: кэш) |

Figure: `figures/03_ablation.png`.

Можно: «сложные модели на этих данных хуже простой смеси».
Нельзя: «бустинг вообще не работает на таких задачах»; цифры стэкинга (не воспроизводятся скриптом).

## Слайд 6. Проверка устойчивости

| Число | Значение | Источник |
|---|---|---|
| MAE на other: V2 / V3 / V3-hedge | 925.12 / 921.67 / 792.98 ₽ | `FINAL_METRICS.csv` → window=other |
| ΔMAE V3 − V2 на other | −3.46 ₽, лучше в 4 из 6 месяцев | `outputs/rolling_summary.json` → bootstrap_vs_v2.other.v3 |
| Категорий, где V3 лучше V2 (exact) | 5 из 5 | `outputs/category_replication_tests.csv` → window=exact, model=v3, dMAE < 0 |
| Сочетаний «категория × месяц», где V3 лучше (exact) | 28 из 30 | там же, сумма months_better / months_total |
| Выигрыш V3 по категориям (exact) | от −0.3% до −3.4% | там же, dMAE_pct |
| Категорий, где V3 лучше V2 (apr_nov) | 5 из 5 | там же, window=apr_nov |
| Недельный фактор: категорий хуже V2 на apr_nov | 5 из 5 (медиана +6.2%) | там же, model=common_weekly2 |
| Недельный фактор на итоговой панели | 719.38 exact, 757.72 other | `outputs/ablation.csv` → common_weekly2 |

Figure: `figures/04_category_replication.png` (альтернатива: `figures/02_mae_by_origin.png`).

Можно: «знак эффекта V3 одинаков во всех 5 нетронутых категориях»; «модель, лучшая на основной панели, отклонена, потому что не воспроизвелась».
Нельзя: «30 независимых подтверждений» (месяцы общие для категорий); «V3 значимо лучше на other» (p = 0.34).

## Слайд 7. Structural shifts и detection

| Число | Значение | Источник |
|---|---|---|
| Старая разметка: доля рядов «со сдвигом», ноябрь 2023 / май 2024 | 23.4% / 0.0% | `outputs/shift_label_rates.csv` → legacy_raw_level_pct |
| Новая разметка, 2024-01…10 | 0.25–1.24% в месяц | там же, new_panel_relative_pct |
| Базовая частота позитивов (детекция) | 1.82% | `outputs/ew_v2_metrics.csv` → task=detect, method=sup_base, base_rate |
| PR-AUC детекции | 0.383 (lift 21.1×) | там же, PR_AUC, lift_vs_base_rate |
| Precision / recall / F1 | 0.32 / 0.55 / 0.41 | там же |
| Ложных тревог на 100 ряд-месяцев | 2.11 | там же, false_alarms_per_100 |
| Сдвигов поймано ≤ 2 мес. | 82% (106 событий) | там же, detected_share, events |
| Медианная задержка | 1 месяц | там же, median_delay_months |
| Тестовых месяцев | 10 (2024-01…2024-10) | там же, months |
| Legacy-классификатор на честной разметке | PR-AUC 0.024 | `outputs/shock_metrics.json` → PR_AUC |

Figure: `figures/05_shift_labels.png`.

Можно: «сдвиг обнаруживается с медианной задержкой 1 месяц»; «исходная разметка в основном отражала сезонность».
Нельзя: «детектор находит все сдвиги»; «supervised лучше простого порога» (+0.041 PR-AUC, CI [−0.003; +0.079], `outputs/ew_v2_ablation.csv`).

## Слайд 8. Почему нельзя предсказать shift за 1–3 месяца

| Число | Значение | Источник |
|---|---|---|
| Базовая частота (упреждение) | 1.39% | `outputs/ew_v2_metrics.csv` → task=predict, base_rate |
| Лучший lift упреждения | 2.53× (BOCPD), 2.50× (robust jump) | там же, lift_vs_base_rate |
| Supervised упреждение, все признаки | lift 1.97× | там же, sup_all |
| Вклад национальных / недельных признаков в упреждение | +0.0032 / +0.0016 PR-AUC, CI включает 0 | `outputs/ew_v2_ablation.csv` → task=predict |
| Вклад в детекцию: категории / национальные / недельные | −0.046 / −0.017 / −0.015 PR-AUC, CI ниже 0 | там же, task=detect |
| Проверок связи частоты сдвигов с нац./недельными сигналами | 42, минимальный p = 0.031, порог Бонферрони 0.0012 | `outputs/ew_v2_rate_correlations.csv` |
| Lead-lag категорий: до / в тот же месяц / после | lift 1.18 / 6.10 / 1.29 (256 сдвигов) | `outputs/ew_v2_leadlag.csv` |

Figure: `figures/06_early_warning.png`.

Можно: «на этих данных опережающего сигнала нет»; «категории сдвигаются вместе с итогом, а не раньше».
Нельзя: «сдвиги в принципе непредсказуемы» (только на доступных данных); «недельные данные СберИндекса бесполезны» (бесполезны для локализации сдвигов МО).

## Слайд 9. Итог

| Число | Значение | Источник |
|---|---|---|
| Время воспроизведения | ≈8 минут на 4 ядрах | `reproduce.sh` (замер сквозного прогона) |
| Тестов на утечку | 7, все проходят | `tests/test_time_safety.py` |
| Расхождение V3 и V3-hedge в прогнозе 2025 | медиана 0.38% | `outputs/presentation_sources.json` → forecast_2025_median_abs_gap_v3_vs_hedge_pct_by_h |
| Прогноз 2025 | 2 016 рядов × 12 месяцев, интервалы 80/90% | `outputs/final_forecasts_2025.csv.gz` |

Figure: нет.

Можно: «все наши числа воспроизводятся одной командой с нуля»; «Prophet пересчитывается отдельно».
Нельзя: «всё воспроизводится за 8 минут, включая Prophet».

---

## Appendix

### A1. Prophet и baselines
`FINAL_METRICS.csv` → window=exact_sample400: prophet 20 997.68 (R² growth −6 238.7), prophet_log 1 627.15, seasonal_naive 4 096.50, factor_only6 964.54, v2_ensemble 699.18, v3 694.06, v3_hedge 705.11; n = 4 800 пар (400 рядов × 12). Остальные baseline на полном exact: слайд 5. Источник Prophet: кэш `outputs/prophet_predictions.csv.gz`, выборка `np.random.default_rng(0)`, `benchmarks/prophet_baseline.py`.
Figure: нет. Нельзя: «Prophet хуже на всех рядах» (оценён на выборке).

### A2. Long horizons
`FINAL_METRICS.csv` → window=long_h:

| h | n | V2 | V3 | V3-hedge | R² growth (V3) |
|---:|---:|---:|---:|---:|---:|
| 4 | 14 112 | 1 263.64 | 1 261.26 | 1 158.55 | −0.545 |
| 5 | 12 096 | 1 275.21 | 1 274.09 | 1 184.06 | −0.560 |
| 6 | 10 080 | 1 316.51 | 1 318.95 | 1 243.08 | −0.706 |

Origins: 2024-02…2024-08 (где T + h ≤ 2024-12). h = 7–12 непроверяемы. Колонка `evidence` в `outputs/final_forecasts_2025.csv.gz`.
Можно: «для h > 3 доказательной базы нет». Нельзя: «V3-hedge лучше на длинных горизонтах» как вывод.

### A3. Time-safety tests
`tests/test_time_safety.py`: test_series_reconstruction, test_national_reader_respects_origin, test_detectors_are_online, test_category_features_online, test_intervals_use_only_observed_errors, test_final_models_ignore_future, test_forecasts_ignore_future (17 моделей × 3 origins; национальные и недельные таблицы передаются неусечёнными и испорченными после origin). Эмбарго early warning: 2 месяца (детекция), 5 месяцев (упреждение), `benchmarks/early_warning_v2.py` → EMBARGO.
Нельзя: «утечка невозможна в принципе»; можно: «тест не обнаружил зависимости прогноза от данных после даты прогноза».

### A4. Полный ablation
`ABLATION.csv` (прогноз: MAE_exact, MAE_other, MAE_apr_nov, R2_growth_exact, wMAPE_exact, categories_better_than_V2_apr_nov; early warning: PR_AUC, lift, F1, false_alarms_per_100, detected_share). Сетка V2: `outputs/v2_grid.csv`, итог в `outputs/presentation_sources.json` → v2_grid.
Figure: `figures/03_ablation.png`.

### A5. V3 vs V3-hedge
| | exact | other | apr_nov | feb_mar | Источник |
|---|---:|---:|---:|---:|---|
| V3 | 719.87 | 921.67 | 806.35 | 1 898.21 | `FINAL_METRICS.csv` |
| V3-hedge | 730.14 | 792.98 | 757.07 | 1 466.97 | там же |
| V3-hedge − V2 | +4.46 (2 из 6 мес., CI по мес. [−33.9; +32.5]) | −132.14 (5 из 6) | −54.08 (5 из 8) | | `outputs/rolling_summary.json` → bootstrap_vs_v2 |

Категории, apr_nov: V3-hedge лучше V2 в 4 из 5, от −29.6% (Маркетплейсы) до +8.0% (Продовольствие), `outputs/category_replication_tests.csv`. Интервалы 80%: V3 покрытие 93.3% (exact) / 91.8% (other), ширина 4 674 / 6 739 ₽; V3-hedge 90.2% / 92.2%, ширина 3 625 / 5 611 ₽ (`outputs/rolling_summary.json` → intervals). Расхождение прогнозов 2025: 0.38%.
Figure: `figures/02_mae_by_origin.png`.
Можно: «V3-hedge устойчивее к аномалии тренда». Нельзя: «V3-hedge лучше V2» без уточнения окна.

### A6. Definition of structural shift
Параметры: `src/sbx/early_warning.py` (KAPPA = 3.0, окна 3/3 месяца, эхо ±12, дедупликация ±2). Частоты: `outputs/shift_label_rates.csv`; кластеры 2023-04 = 14.68%, 2023-09 = 4.36%. Всего событий: `outputs/early_warning_summary.json` → events_total.
Figure: `figures/05_shift_labels.png`.
Можно: «определение операционное и защищено от календаря». Нельзя: «это истинные структурные сдвиги».

### Дополнительно: интервалы по месяцам
`outputs/interval_coverage_by_month.csv`: покрытие 80%-интервала V3 99.3% (2024-05) → 80.3% (2024-12); 90%-интервала 100.0% → 90.7%. Figure: `figures/07_interval_coverage.png`.
