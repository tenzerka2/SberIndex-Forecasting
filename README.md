# SberIndex Forecasting v2

Воспроизводимая конкурсная версия решения задачи прогнозирования потребительских безналичных расходов на уровне муниципальных образований и раннего предупреждения структурных изменений.

## Текущий результат

На точном конкурсном rolling-backtest:

- 2 016 полных муниципальных рядов;
- origins: 2024-06, 2024-07, 2024-08, 2024-09;
- горизонты: 1, 2, 3 месяца;
- 24 192 forecast pairs;
- **MAE 725.69 руб.**;
- **R² growth 0.4274**;
- **wMAPE 2.302%**.

Текущий прогноз — интерпретируемый equal-weight ансамбль локальной динамики муниципалитета и общего панельного фактора.

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
# распакуйте исходные CSV СберИндекса в data/raw/
python src/pipeline.py
python benchmarks/exact_backtest.py
```

## Структура

- `src/pipeline.py` — основной forecasting + early-warning pipeline;
- `benchmarks/exact_backtest.py` — точный конкурсный rolling-backtest;
- `METHOD.md` — методология;
- `EXACT_BENCHMARK.md` — протокол и результаты exact benchmark;
- `FABLE_TASK.md` — техническое задание на независимый аудит и следующую итерацию;
- `outputs/` — компактные метрики и summaries;
- `report.html` — первая версия отчёта.

## Важное про данные

Исходные конкурсные CSV не коммитятся в репозиторий. Положите их в `data/raw/`. Ожидаемые имена описаны в `data/README.md`.

## Следующие цели

1. Независимо воспроизвести MAE 725.69 без leakage.
2. Проверить результат на дополнительных rolling origins.
3. Попробовать снизить MAE ниже 700 руб. только time-safe методами.
4. Усилить early-warning structural change detector.
5. Добавить news/event признаки со строгим publication-time alignment.
6. Добавить uncertainty intervals и ablation-анализ.
