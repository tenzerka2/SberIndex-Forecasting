"""Municipality -> region (subject of the Russian Federation) mapping for news geo-linking.

Sources (no network access is needed at run time once fetched):
  * data/external/settlements.csv: "Населенные пункты России: численность населения и
    географические координаты" (Минздрав РФ; обработка ИНИД / АНО «ЦПУР»), GitHub
    NickyX3/russia_settlements, licence CC BY-NC-SA 4.0. Fetch: python benchmarks/fetch_geo.py
  * SPB_MO below: the 111 intra-city municipalities of St Petersburg (law No 411-68, list as in
    ru.wikipedia "Административно-территориальное деление Санкт-Петербурга", retrieved 2026-10-06).
    Every other intra-city territory in the panel is a Moscow municipal okrug or settlement.

Homonym series (107 rows sharing 43 names) are left unmapped: the export does not say which of the
same-named blocks belongs to which region.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .data import ROOT, Panel

SETTLEMENTS = ROOT / "data" / "external" / "settlements.csv"

SPB_MO = """Коломна; Сенной округ; Адмиралтейский округ; Семёновский; Измайловское; Екатерингофский; № 7; Васильевский;
Гавань; Морской; округ Морской; Остров Декабристов; Сампсониевское; Светлановское; Сосновское; Суздальское; Сергиевское;
Парнас; Шувалово-Озерки; Левашово; Парголово; Гражданка; Академическое; Финляндский округ; № 21; Пискарёвка; Северный;
Прометей; Княжево; Ульянка; Дачное; Автово; Нарвский округ; Красненькая речка; Морские ворота; Колпино; Понтонный;
Усть-Ижора; Петро-Славянка; Сапёрный; Металлострой; Полюстрово; Большая Охта; Малая Охта; Пороховые; Ржевка; Юго-Запад;
Южно-Приморский; Сосновая Поляна; Урицк; Константиновское; Горелово; Красное Село; Кронштадт; Зеленогорск; Сестрорецк;
Белоостров; Комарово; Молодёжное; Песочный; Репино; Серово; Смолячково; Солнечное; Ушково; Московская застава;
Гагаринское; Новоизмайловское; Пулковский меридиан; Звёздное; Невская застава; Ивановский; Обуховский; Рыбацкое;
Народный; № 54; Невский округ; Оккервиль; Правобережный; Введенский; Кронверкское; Посадский; Аптекарский остров;
округ Петровский; Чкаловское; Стрельна; Ломоносов; Петергоф; Лахта-Ольгино; № 65; Ланское; Чёрная речка;
Комендантский аэродром; Озеро Долгое; Юнтолово; Коломяги; Лисий Нос; Павловск; Пушкин; Шушары; Александровская;
Тярлево; Волковское; Николаевский; Купчино; Георгиевский; Александровский; Балканский; Дворцовый округ; № 78;
Литейный округ; Смольнинское; Лиговка-Ямская; Владимирский округ; № 15; № 72"""

TYPE_WORDS = [
    "внутригородская территория города федерального значения", "муниципальный район", "муниципальный округ",
    "городской округ", "сельское поселение", "городское поселение", "внутригородской район", "район", "округ",
    "кожуун", "улус", "город", "поселок", "посёлок", "поселение", "закрытое административно-территориальное образование",
    "зато", "г.", "пгт",
]


def norm(s: str) -> str:
    s = str(s).lower().replace("ё", "е").replace("\xa0", " ")
    s = re.sub(r"[«»\"()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def core(name: str) -> str:
    """Name without administrative type words: 'городской округ город Орск' -> 'орск'."""
    s = norm(name)
    for w in sorted(TYPE_WORDS, key=len, reverse=True):
        s = re.sub(rf"(^|\s){re.escape(w)}(\s|$)", " ", s)
    return re.sub(r"\s+", " ", s).strip(" -")


def adj_stem(word: str) -> str:
    """Crude stem used for morphological matching in texts: 'суджанский' -> 'суджанск'."""
    for suf in ("ский", "цкий", "ской", "цкой", "ский", "ый", "ий", "ой", "ая", "ое", "ые"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf) + (1 if suf.startswith(("ск", "цк")) else 0)] if False else word[: len(word) - len(suf) + (2 if suf in ("ский", "цкий", "ской", "цкой") else 0)]
    return word


def build_mapping(panel: Panel) -> pd.DataFrame:
    s = pd.read_csv(SETTLEMENTS, sep=";", dtype=str, usecols=["region", "municipality", "settlement", "type"])
    s["mkey"] = s["municipality"].map(core)
    by_muni = s.groupby("mkey")["region"].agg(lambda x: sorted(set(x)))
    cities = s[s["type"].isin(["г", "город"])].assign(ckey=lambda d: d["settlement"].map(core))
    by_city = cities.groupby("ckey")["region"].agg(lambda x: sorted(set(x)))
    spb = {core(x) for x in SPB_MO.replace("\n", " ").split(";")}

    rows = []
    for run_id, mo, hom in panel.meta[["run_id", "mo", "homonym"]].itertuples(index=False):
        key = core(mo)
        region, method = None, "unmatched"
        if hom:
            method = "homonym_unresolved"
        elif "внутригородская территория" in norm(mo):
            region, method = ("Санкт-Петербург", "intracity_spb_list") if key in spb else ("Москва", "intracity_default_moscow")
        else:
            cands = by_muni.get(key)
            if cands is None and key.endswith(("ский", "цкий")):  # 'ивдельский' (городской округ) vs 'ивдель'
                cands = by_city.get(key[:-4]) or by_city.get(key[:-3])
            if cands is None:
                cands = by_city.get(key)
            if cands is not None and len(cands) == 1:
                region, method = cands[0], "settlements_unique"
            elif cands is not None:
                method = "ambiguous_regions:" + "|".join(cands[:4])
        rows.append({"run_id": run_id, "mo": mo, "core": key, "region": region, "method": method})
    return pd.DataFrame(rows)
