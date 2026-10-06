# SberIndex Forecasting — final submission

Прогноз безналичных потребительских расходов 2 016 муниципальных образований, structural-shift detection и исследование внешних сигналов.

## Главный результат

- V3 exact MAE: **719.87 ₽**
- R² growth: **0.4376**
- wMAPE: **2.284%**
- V3 лучше V2 во всех 5 категориальных панелях, не использованных при выборе модели
- detection structural shifts: PR-AUC **0.383** при base rate 1.82%, median delay 1 месяц

## Что дополнительно проверено

- horizons **1 / 3 / 6 / 12**
- **TimesFM 2.5-200M** zero-shot
- Prophet sample benchmark
- 5 category panels
- national / weekly SberIndex signals
- historical news/events: **95 pre-registered queries, 550 articles**
- conformal intervals
- deterministic stable / shift / failure municipality examples
- time-safety / leakage tests

Новости и TimesFM не улучшают point forecast V3; отрицательные результаты сохранены в ablation и используются в выводах.

## Документы подачи

- FINAL_REPORT.md — полный отчёт
- METHOD.md — методология
- PRESENTATION.md — финальные 10 слайдов + appendix
- PRESENTATION_DATA.md — source map каждой цифры
- JURY_QA.md — 25 вопросов жюри
- RUBRIC_CHECKLIST.md — критерий → доказательство
- REAL_EXAMPLES.md — реальные примеры МО
- FINAL_SUBMISSION_STATUS.md — что отправлять

## Воспроизведение

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# распаковать 5 CSV СберИндекса в data/raw/
./reproduce.sh
```

reproduce.sh пересчитывает собственные forecasting results, horizon benchmark, category replication, early warning, news ablation, real examples и figures. Prophet / TimesFM могут использовать frozen cached predictions; отдельный refit описан в benchmark scripts.

## Основные файлы

- src/sbx/data.py — reconstruction 2 016 series
- src/sbx/models.py / final.py — V2, V3, V3-hedge
- src/sbx/intervals.py — time-safe conformal
- src/sbx/early_warning.py / ew_features.py — shifts / detectors
- src/sbx/news_global.py — verified publication-time news features
- benchmarks/horizons_benchmark.py — 1/3/6/12
- benchmarks/foundation_timesfm.py — TimesFM
- benchmarks/news_ablation_global.py — no-news vs news
- benchmarks/real_examples.py — deterministic examples
- tests/test_time_safety.py
- tests/test_news_time_safety.py

Исходные муниципальные CSV не коммитятся. Стабильного id МО в экспорте нет; технический ключ полного ряда — run_id raw export.