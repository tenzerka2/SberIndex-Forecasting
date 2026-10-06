# FINAL_SUBMISSION_STATUS

## Готово

- frozen forecasting models: V3 / V3-hedge;
- exact + rolling + category replication;
- horizons 1 / 3 / 6 / 12 benchmark;
- TimesFM 2.5 zero-shot benchmark;
- structural-shift detection / prediction benchmark;
- historical news corpus и time-safe news ablation;
- deterministic real municipality examples;
- conformal intervals;
- leakage/time-safety tests;
- presentation content, data map and jury Q&A.

## Финальные headline numbers

- V3 exact MAE: **719.87 ₽**
- R² growth: **0.4376**
- wMAPE: **2.284%**
- V3 better than V2 on all 5 untouched categories
- structural-shift detection: PR-AUC **0.383** in the main benchmark, event base rate 1.8%
- news ablation: no useful improvement; exact forecast MAE worsens to **946.84 ₽**
- TimesFM zero-shot is worse than V3 on h=1/3/6; its h=1 80% interval is better calibrated.

## Ограничения

1. Всего 24 месяца municipal history.
2. Exact-window содержит только 6 target months.
3. Q1 anomaly влияет на январь–март 2025.
4. h=7–12 нельзя честно валидировать для V3/V3-hedge.
5. Истинное упреждение structural shift на 1–3 месяца по имеющимся сигналам не достигнуто.
6. News corpus построен через зафиксированный search plan, но не является полным архивом всех локальных новостей России.

## Что отправлять организаторам

1. PDF-презентация (делается отдельно по `PRESENTATION.md`)
2. GitHub repository
3. `FINAL_REPORT.md`
4. `METHOD.md`
5. `FINAL_METRICS.csv`
6. `ABLATION.csv`
7. `RUBRIC_CHECKLIST.md`
8. `REAL_EXAMPLES.md`
9. `JURY_QA.md`
10. `reproduce.sh`

## Воспроизведение

`./reproduce.sh`

Собственные результаты пересчитываются из сырых CSV. Prophet / TimesFM могут использовать заранее сохранённые frozen outputs; их отдельный полный refit описан в соответствующих benchmark scripts.
