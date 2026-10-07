# SberIndex Forecasting

Актуальный комплект от 7 октября 2026 года: [описание работы, результаты, примеры и ограничения](SUBMISSION_CURRENT.md).

На всех 2016 полных муниципальных рядах V3 Hedge снизила MAE относительно лучшего проверенного Prophet на 32,36% / 41,06% / 42,82%, а относительно лучшего проверенного TimesFM — на 19,11% / 21,04% / 30,16% на горизонтах 1/3/6 месяцев. Это ретроспективная оценка, не закрытый тест организаторов. Для годового горизонта используется отдельный national_yoy_fallback, проверенный только на одной дате.

## Быстрая проверка без Colab и исходных CSV

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.4.6 pandas==2.3.3
bash verify_submission.sh
python benchmarks/forecast_operational.py
```

Используются подготовленная панель и сохранённые прогнозы из Git. Проверка пересчитывает метрики и основной метод, воспроизводит детектор и примеры. Prophet и TimesFM повторно не запускаются. При ином имени интерпретатора: `PYTHON=python3 bash verify_submission.sh`.

Экспорт содержит 8064 исторических прогноза из декабря 2024 года на 2025 год с официальными ID, регионами и ОКТМО. Это не прогноз из текущей даты. Метод указан в каждой строке; калиброванные интервалы V3 Hedge не заявляются.

## Подтверждённые результаты

[Полный Prophet](FULL_PROPHET_RESULTS.md), [TimesFM и интервалы](TIMESFM_RESULTS.md), [официальные ID и транспортный эксперимент](GRAPH_WARNING_REPORT.md). Прогнозные пары и метрики находятся в `outputs/prophet_full_summary/` и `outputs/timesfm_summary/`. Метрики детектора и примеры — в `outputs/submission/`.

Полные тяжёлые расчёты доступны вручную через GitHub Actions: `Prophet full panel` и `TimesFM paired full panel`. Протоколы: [Prophet](FULL_PANEL_RUN.md) и [TimesFM](TIMESFM_PROTOCOL.md). Полная исследовательская цепочка с исходными файлами остаётся в `reproduce.sh`; для быстрой проверки комплекта используется `verify_submission.sh`.

## Архив предыдущих итераций

FINAL_REPORT.md, REPORT.md, PRESENTATION.md, IMPROVEMENT_REPORT.md, VERIFIED_PROPHET_RESULTS.md и прежний PDF сохранены для аудита. Они описывают разные этапы и выборки; не следует смешивать их числа с актуальной сводкой SUBMISSION_CURRENT.md. Презентация в этот комплект не входит.
