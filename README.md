# SberIndex Forecasting v3

Воспроизводимое конкурсное решение: прогноз потребительских безналичных расходов на уровне муниципальных образований и раннее обнаружение структурных изменений. Полный отчёт аудита и итерации: [`REPORT.md`](REPORT.md).

## Текущий результат

Exact contest benchmark: 2 016 муниципальных рядов, origins 2024-06…2024-09, горизонты 1–3, 24 192 пары.

| Модель | MAE, ₽ | R² growth | wMAPE | MAE на других origins (04, 05, 10, 11) |
|---|---:|---:|---:|---:|
| V2: 0.5·local_sng2 + 0.5·panel_factor6 | 725.69 | 0.4274 | 2.302% | 925.1 |
| **V3: V2 + error feedback** | **719.87** | **0.4376** | **2.284%** | 921.7 |
| V3-hedge: хедж общего роста + error feedback | 730.14 | 0.4320 | 2.317% | **793.0** |

V3 лучше V2 во всех 6 целевых месяцах exact-окна (ΔMAE −5.8 ₽, 95% CI [−7.5; −3.5] по месяцам) и на дополнительных origins. V3-hedge рекомендуется для реального прогноза: V2 опирается на окно общего роста, которое захватывает аномалию I квартала в муниципальных данных, и на других origins проигрывает 130 ₽.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
# распакуйте исходные CSV СберИндекса в data/raw/
python benchmarks/exact_backtest.py       # V2 exact benchmark (725.686835)
python benchmarks/rolling_eval.py         # все модели, 10 origins, ablation, bootstrap, интервалы
python benchmarks/early_warning_eval.py   # structural-change detectors + supervised classifier
python tests/test_time_safety.py          # perturbation-тест на отсутствие утечки
python src/pipeline.py                    # pooled LightGBM на длинных горизонтах + early warning
```

## Структура

- `src/sbx/data.py`: загрузка, восстановление идентичности рядов (омонимы МО), внешние ряды с publication-time alignment;
- `src/sbx/backtest.py`: rolling-origin движок, метрики, paired bootstrap по рядам и целевым месяцам;
- `src/sbx/models.py`: baselines, V2, V3 (error feedback), варианты общего фактора, обучаемые веса, SVD, pooled/residual GBM;
- `src/sbx/intervals.py`: time-safe двухчастные conformal-интервалы;
- `src/sbx/early_warning.py`: определение структурного сдвига, CUSUM, Page-Hinkley, BOCPD, PELT, признаки классификатора;
- `benchmarks/`: exact benchmark, полная rolling-оценка, early-warning оценка, Prophet (кэш на выборке);
- `src/pipeline.py`: pooled LightGBM для горизонтов 1/3/6/12 и operational early warning;
- `tests/test_time_safety.py`: все прогнозы не меняются при порче данных после origin;
- `outputs/`: метрики, ablation, bootstrap, интервалы, early-warning таблицы;
- `METHOD.md`, `EXACT_BENCHMARK.md`, `REPORT.md`: методология, протокол, отчёт.

## Важное про данные

Исходные конкурсные CSV не коммитятся. Положите их в `data/raw/`, маски имён в `data/README.md`. В муниципальном CSV нет стабильного id; ряды восстанавливаются по непрерывным блокам (МО, категория) сырого экспорта, загрузчик проверяет результат assert'ами.
