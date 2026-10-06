# SberIndex Forecasting v3 (final)

Прогноз потребительских безналичных расходов 2 016 муниципальных образований на 1–3 месяца и обнаружение структурных сдвигов. Главный документ: [`FINAL_REPORT.md`](FINAL_REPORT.md) (метод, результаты, ablation, review, ограничения, структура презентации).

## Финальные модели (заморожены в `src/sbx/final.py`)

| Модель | Роль | MAE exact | R² growth | wMAPE | MAE other origins | Категорий лучше V2 (апр–ноя) |
|---|---|---:|---:|---:|---:|---:|
| V2 (прежняя) | reference | 725.69 | 0.4274 | 2.302% | 925.1 | |
| **V3** = V2 + error feedback | benchmark | **719.87** | **0.4376** | **2.284%** | 921.7 | 5 из 5 |
| **V3-hedge** = хедж общего роста + error feedback | production | 730.14 | 0.4320 | 2.317% | **793.0** | 4 из 5 |

Exact protocol: origins 2024-06…2024-09, h = 1–3, 24 192 пары. Other: origins 2024-04, 05, 10, 11.

Early warning: детекция уже начавшегося сдвига PR-AUC 0.38 при частоте 1.8% (82% сдвигов за ≤ 2 месяца, 2.1 ложной тревоги на 100 ряд-месяцев). Упреждение за 1–3 месяца не работает (lift ≤ 2.5×), недельные и национальные данные раннего сигнала не дают.

## Воспроизведение одной командой

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw   # распакуйте 5 исходных CSV СберИндекса (маски в data/README.md)
./reproduce.sh      # ~7 минут
```

Результат: `FINAL_METRICS.csv`, `ABLATION.csv`, `figures/`, `outputs/final_forecasts_2025.csv.gz` (обе модели, h = 1–12, интервалы 80/90%), промежуточные таблицы в `outputs/`.

## Структура

- `src/sbx/data.py`: восстановление рядов по блокам (МО, категория) сырого экспорта с проверками, национальные и недельные ряды;
- `src/sbx/models.py`, `src/sbx/final.py`: все модели и замороженные финальные;
- `src/sbx/backtest.py`: rolling-origin движок, метрики, bootstrap и sign test;
- `src/sbx/intervals.py`: двухчастные conformal-интервалы;
- `src/sbx/early_warning.py`, `src/sbx/ew_features.py`: определение сдвига, детекторы, признаки;
- `benchmarks/`: `exact_backtest.py` (V2, 725.686835), `rolling_eval.py`, `category_replication.py`, `early_warning_eval.py`, `early_warning_v2.py`, `forecast_final.py`, `build_final.py`, `prophet_baseline.py`;
- `tests/test_time_safety.py`: 7 тестов на утечку и реконструкцию;
- `src/pipeline.py`: legacy pooled LightGBM для h = 1/3/6/12 (на 1 904 рядах без омонимов, не сопоставим с exact).

Исходные CSV не коммитятся. Стабильного id МО нет: ключ ряда это `run_id` выгрузки; у омонимов совпадают названия.
