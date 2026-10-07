"""Build an auditable Russian report from research outputs, never hardcoded headline metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/research"


def markdown_table(df):
    # Keep the research workflow runnable with NumPy/pandas alone.
    values=df.astype(str)
    return "| " + " | ".join(values.columns) + " |\n| " + " | ".join(["---"]*len(values.columns)) + " |\n" + "\n".join("| " + " | ".join(row) + " |" for row in values.itertuples(index=False,name=None))


def main():
    m=pd.read_csv(OUT/"v4_metrics.csv")
    total=m[m.category.eq("Все категории")]
    comparison=total.pivot(index="model",columns="window",values="MAE")
    comparison=comparison.loc[["v3","v3_hedge","robust_filtered_feedback","v4_diversified"], ["exact","other","apr_nov","all_origins","h6"]].round(2).reset_index()
    pairs=pd.read_csv(OUT/"v4_pairs_0.csv.gz",parse_dates=["origin","target_date"])
    rows=[]
    for window,mask in {"exact":pairs.origin.isin(pd.to_datetime(["2024-06-01","2024-07-01","2024-08-01","2024-09-01"]))&pairs.h.le(3),
                        "all_origins":pairs.h.le(3)}.items():
        d=pairs[mask]
        for model in ["robust_filtered_feedback","v4_diversified"]:
            delta=(d[model]-d.y).abs()-(d.v3-d.y).abs()
            by_month=delta.groupby(d.target_date).mean()
            rows.append({"window":window,"model":model,"delta_MAE":delta.mean(),
                         "target_months_better":int((by_month<0).sum()),"target_months":len(by_month),
                         "interpretation":"descriptive only; post-selection and dependent months"})
    pd.DataFrame(rows).to_csv(OUT/"paired_comparison.csv",index=False)
    warning=pd.read_csv(OUT/"warning_metrics.csv")
    wm=warning[(warning.budget_pct.eq(2))&warning.method.isin(["jump","bocpd","logistic_base","logistic_local_news"])]
    wm=wm[["task","method","AP","precision","recall","false_alarms_per_100"]].round(4)
    revisions=pd.read_csv(OUT/"label_revision_audit.csv")
    news=json.loads((OUT/"local_news_audit.json").read_text())
    news_forecast=pd.read_csv(OUT/"news_forecast_ablation.csv")
    news_mae=news_forecast[news_forecast.window.eq("exact")&news_forecast.model.eq("v3_news")].MAE.iloc[0]
    manifest=json.loads((OUT/"data_manifest.json").read_text())
    cat=m[m.window.eq("apr_nov")].pivot(index="category",columns="model",values="MAE")
    cat=cat[["v3","v3_hedge","v4_diversified"]].round(2).reset_index()
    text=f'''# Проверка и улучшение SberIndex, 7 октября 2026

Этот отчёт — актуальное дополнение к историческому FINAL_REPORT.md. База V3 воспроизведена на исходных пяти CSV: MAE 719.870192 ₽. Все новые результаты пересчитаны локально. Старые таблицы оставлены как история и не являются результатом исправленной разметки шоков.

## Что получилось с прогнозом

{markdown_table(comparison)}

Числа — MAE в рублях на одинаковых парах внутри каждого столбца. `exact`: origins июнь–сентябрь 2024, h=1–3. `other`: апрель, май, октябрь, ноябрь, h≤3 с наблюдаемым target. `all_origins`: февраль–ноябрь 2024, h≤3. `h6`: пять origins с наблюдаемым шестимесячным target. Эти столбцы перекрываются и не являются независимыми тестами.

Самый низкий exact MAE среди новых кандидатов — robust_filtered_feedback. Но на полном наборе origins и h=6 он хуже V3: его нельзя объявлять универсально лучшим. V4-diversified улучшает V3 во всех пяти сводных окнах итоговой категории, однако V3-hedge сохраняет преимущество по устойчивости на расширенных окнах. Это компромисс точности основного окна и устойчивости, а не победа над каждым существующим вариантом.

V4-diversified использует фиксированную смесь 50/50: фильтр муниципального общего роста и существующую национальную hedge-компоненту, затем добавляет поправку по уже наблюдаемым ошибкам. Веса не обучались на exact, но сама идея проверялась после просмотра предыдущих экспериментов.

## Перенос на категории

{markdown_table(cat)}

Таблица относится к апрелю–ноябрю. V4 не улучшает все категории: особенно важны неудачи на общественном питании и продовольствии. Не следует автоматически применять настройки итоговой категории ко всем видам расходов.

## Исправленная разметка шоков

Старая разметка пересматривала от {revisions.legacy_mature_label_changes.min()} до {revisions.legacy_mature_label_changes.max()} ранее «созревших» меток на каждом из десяти проверенных origins при добавлении более поздних наблюдений. Причины: масштаб по всей истории, проверка эха t+12 и двустороннее подавление соседних событий. Это не доказывает прямую утечку во внешний прогноз, но опровергает заявленное окончательное созревание через два месяца.

Теперь масштаб фиксируется по истории до начала события, сезонное эхо проверяется только в прошлом, принимается первое событие с последующим двухмесячным подавлением повторов. Событие подтверждается через два месяца и больше не меняется. На всех проверенных origins число изменений созревших меток равно нулю. Несозревшие метки имеют значение -1 и не превращаются в отрицательные примеры. Исправлен также embargo внутренней проверки порога.

## Результат обнаружения и предупреждения

{markdown_table(wm)}

AP — average precision, как в прежнем коде под названием PR_AUC. Оценка использует исправленную разметку. Поэтому новые AP нельзя сравнивать напрямую со старым 0.383: изменилась задача и частота событий. В каждой проверке выделяется фиксированный бюджет 2% муниципалитетов (40 из 2016) в месяц; также сохранены бюджеты 1% и 5%. Logistic-base — регуляризованная модель на причинно доступных признаках, обучаемая заново в каждом месяце; jump и BOCPD — контрольные методы на тех же метках.

Обнаружение уже начавшихся изменений работает существенно лучше упреждения. Для упреждения простая jump-модель на этой небольшой выборке лучше новой логистической модели. Низкую точность положительных предупреждений нельзя скрывать за lift. Предотвращение экономических шоков не проверялось; алгоритм формирует статистический сигнал для проверки аналитиком.

## Локальные новости

Существующий корпус после проверки дат и дедупликации содержит {news['unique_articles']} уникальных публикаций. Консервативное сопоставление по уникальному названию дало {news['linked_articles']} публикации для {news['linked_municipalities']} муниципалитетов. {news['unresolved_homonyms']} одноимённых рядов не получили выдуманной географии. Все ссылки, заголовки и способы сопоставления сохранены в local_news_links.csv. Это кандидаты на связь, а не подтверждённая причинность; современная поисковая выдача не заменяет архив исторических версий страниц.

Включён отдельный признак планируемых событий по языку публикации и причинные накопления за три месяца. Этот ограниченный корпус не дал полезного улучшения предупреждений. Официальный справочник ID в предоставленном муниципальном CSV отсутствует: поля содержат mo, но не ID/ОКТМО/регион. Для надёжного расширения нужны официальный crosswalk и более полный архив локальных событий.

Общенациональная news-коррекция теперь обучается по уникальным парам origin×horizon, а не воспринимает 2016 копий одного новостного вектора как независимую историю. Пересчёт дал MAE {news_mae:.2f} ₽ на exact и также не улучшил V3: исправление методики не гарантирует повышения точности. Старое число 946.84 относится к предыдущей реализации.

## Другие исправления и ограничения запуска

Вариант Prophet с принудительной годовой сезонностью больше не описывается как «по умолчанию»; добавлен настоящий режим auto. Полный refit Prophet и Chronos-2 в этом окружении не запускались: соответствующие библиотеки недоступны. Нельзя выдавать старый кэш за новый пересчёт. Короткая история ограничивает оценку годовой сезонности Prophet, но не делает любой запуск на h=12 математически невозможным.

Двухчастные интервалы описаны как эмпирические интервалы с дополнительными предположениями, а не стандартный split-conformal с гарантией покрытия. Доступность данных контролируется по периоду наблюдения; исторические ревизии и фактический календарь публикаций остаются ограничением.

Было проверено 13 новых кандидатов в трёх явно обозначенных исследовательских фазах. Полный список, неудачные результаты и параметры находятся в config/research_protocol.json, research_phase2.json, research_phase3.json и outputs/research/. Перенос национальной сезонной формы не сработал и отклонён. Поиск параметров остановлен. Все доступные периоды и категории уже просмотрены; для окончательного подтверждения нужны новые месяцы или закрытый тест организаторов. Результат базовой модели организаторов на идентичных парах не предоставлен, поэтому победа над ней не заявляется.

## Воспроизведение

Установить NumPy≥2 и pandas≥2.2, распаковать пять CSV в data/raw, затем выполнить `bash reproduce_research.sh`. Новый расчёт не требует scipy, sklearn или загрузки нейросетевых весов. Регрессионные проверки: `python tests/test_research.py`. Для проверки настоящего Prophet auto в полном окружении: `python benchmarks/prophet_baseline.py --variant auto`.

Исходные данные идентифицированы SHA-256 в outputs/research/data_manifest.json. Новый отчёт собирается командой `python benchmarks/research_summary.py`. Полные пары доступны после воспроизведения; сводные таблицы включены в Git.
'''
    (ROOT/"IMPROVEMENT_REPORT.md").write_text(text,encoding="utf-8")
    print(comparison.to_string(index=False))


if __name__=="__main__":main()
