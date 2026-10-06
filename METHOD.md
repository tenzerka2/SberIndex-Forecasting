# Методология — финальная версия

## 1. Задача

Прогноз месячных безналичных потребительских расходов 2 016 муниципальных образований и анализ структурных сдвигов. Основной contest-style benchmark: origins 2024-06…09, h=1–3, 24 192 пары. Отдельно проверены горизонты 1/3/6/12.

## 2. Данные и идентичность рядов

Муниципальный экспорт покрывает 2023-01…2024-12 и не содержит стабильного id. Ряд восстанавливается как непрерывный raw-block (МО, категория) с возрастающим месяцем. Полных рядов: 2 016; среди них 107 омонимов под 43 названиями. run_id выгрузки используется как технический ключ.

Внешние данные:
- национальные месячные расходы / nominal & real YoY / SA-index;
- национальные недельные категории;
- исторический news/event corpus по заранее зафиксированному query plan.

Все exogenous features выравниваются по времени и доступны только если их дата ≤ forecast origin.

## 3. Forecasting

Пусть f_t — медиана log-расходов по МО, d_it = log y_it − f_t.

- local_sng2: тот же месяц год назад × собственный средний YoY последних двух месяцев.
- panel factor: текущая позиция МО относительно медианы + общий path панели.
- V2: 50/50 смесь local и panel в рублях.
- V3: V2 + error feedback по двум последним известным idiosyncratic one-step errors. Коэффициенты оцениваются только на прошлых forecast pairs с уже наблюдаемыми target.
- V3-hedge: тот же error feedback, но общий рост panel-part = среднее municipal 6m, municipal 2m и national 2m YoY.

V3 — benchmark model. V3-hedge — robust/production alternative.

## 4. Горизонты 1 / 3 / 6 / 12

Отдельный benchmark использует одинаковые пары внутри каждого горизонта.

- h=1: 10 origins
- h=3: 8 origins
- h=6: 5 origins
- h=12: V3/V3-hedge не определены на исторических origins с наблюдаемым target, потому что им нужен observable municipal YoY на origin. Для h=12 сравниваются только модели, математически определимые на 2023 origins.

Ограничение h=12 является следствием длины данных, а не пропущенным экспериментом.

## 5. Foundation model

Запущена TimesFM 2.5-200M zero-shot с frozen official Google weights.

Два заранее заданных варианта:
- raw levels;
- log YoY context.

Fine-tuning и tuning под exact не использовались. Point forecast TimesFM хуже V3/V3-hedge на h=1/3/6, но её 80% interval на h=1 ближе к nominal coverage.

## 6. Prediction intervals

Двухчастный time-safe conformal:
- idiosyncratic normalized residual;
- отдельный band общего monthly shock.

Calibration использует только forecast errors, уже наблюдаемые на текущий origin.

## 7. Structural shifts

Операционное событие:
- panel-relative level shift ≥3 robust sigma;
- post-level устойчив 3 месяца;
- исключается 12-месячное seasonal echo;
- deduplication local maxima ±2 месяца.

Две разные задачи:
- detection: shift начался 0–2 месяца назад;
- prediction: shift начнётся через 1–3 месяца.

Сравниваются robust jump, BOCPD, PELT, CUSUM, Page-Hinkley и supervised LightGBM. Rolling retraining и embargo исключают использование незрелых labels.

## 8. News / event signals

Query plan из 95 month×topic запросов зафиксирован до полного сбора. Корпус: 550 search results за 2023-05…2024-11.

Temporal guard:
- candidate publication dates берутся из metadata и URL;
- verified publication date = наиболее поздняя достоверная candidate;
- query-month mismatches и записи без надёжной даты исключаются;
- rolling sums используют только текущий и прошлые месяцы.

Frozen ablation:
- forecasting: V3 vs V3 + news residual correction;
- early warning: base vs base + news.

Новости не улучшают финальную систему и поэтому не входят в production forecast.

## 9. Проверки качества

- exact / other / apr–nov rolling windows;
- 5 untouched category panels;
- horizon benchmark 1/3/6/12;
- TimesFM / Prophet / simple baselines;
- ablation;
- sign test по target months;
- time-safety perturbation tests;
- separate news time-safety tests;
- deterministic real examples including failure case.

## 10. Финальные headline metrics

Exact:
- V3 MAE 719.87 ₽
- R² growth 0.4376
- wMAPE 2.284%

Structural-shift detection:
- PR-AUC 0.383
- base rate 1.82%
- 82% events detected ≤2 months
- median delay 1 month

News ablation:
- forecast exact 719.87 → 946.84 ₽
- detection PR-AUC 0.3682 → 0.3639
- prediction PR-AUC 0.0245 → 0.0269, but false alarms increase sharply

## 11. Воспроизводимость

./reproduce.sh

Собственные модели, horizons, category replication, early warning, news ablation, examples и figures пересчитываются из raw CSV. Prophet / TimesFM могут использовать frozen cached predictions; полный refit documented separately.