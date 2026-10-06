# Rubric checklist (статус на момент остановки работы)

| Критерий | Статус | Что сделано | Доказательство |
|---|---|---|---|
| Forecasting на h = 1/3/6/12 | сделано | Отдельный benchmark; V3/V3-hedge не определены на h=12 (нужен YoY на origin, а цель ≤ 2024-12 требует origin ≤ 2023-12) | `benchmarks/horizons_benchmark.py`, `outputs/horizons_metrics.csv`, `outputs/horizons_intervals.csv` |
| Foundation model | сделано | TimesFM 2.5-200M zero-shot (веса из официального публичного бакета Google, HF закрыт), два заранее заданных варианта; хуже V3 на всех горизонтах; смесь V3+TimesFM ухудшает exact на 80 ₽ | `benchmarks/fetch_timesfm.py`, `benchmarks/foundation_timesfm.py`, `requirements-tsfm.txt`, `outputs/timesfm_predictions.csv.gz` |
| Prophet на горизонтах | сделано | Лог-Prophet, выборка 400 рядов, h = 1..6 | `benchmarks/prophet_horizons.py`, `outputs/prophet_horizons.csv.gz` |
| News / events | частично | План 95 запросов зафиксирован в git до сбора (commit f850ac2); собрано 155 статей за 2023-05…2023-09 с датой публикации из метаданных; геопривязка МО→регион (1 756 из 2 016 рядов); найдена и описана утечка через ошибочные метаданные дат (правило: время публикации = максимум из метаданных и даты в URL). Не сделано: сбор 2023-10…2024-11, агрегация, ablation no-news vs news | `data/news/query_plan.csv`, `data/news/corpus.jsonl`, `benchmarks/news_append.py`, `src/sbx/geo.py` |
| Примеры реальных МО | не сделано | | |
| Обновление презентации под rubric | не сделано | | |
