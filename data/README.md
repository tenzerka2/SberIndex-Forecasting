# Данные

Распакуйте исходные CSV СберИндекса в `data/raw/`. Каталог `data/raw/` исключён из Git.

Ожидаются файлы, совпадающие с масками:

- `potrebitelskie-beznalicnye*.csv`
- `consumer-spending_ru*.csv`
- `consumer-spending-growth*.csv`
- `consumper-spending-index-sa*.csv`
- `ver-izmenenie-trat-po-kategoriyam*.csv`

В муниципальном CSV нет обещанного стабильного id. Для основного pipeline неоднозначные названия исключаются. Для exact contest benchmark идентичность рядов восстанавливается по contiguous row blocks исходного экспорта; это сохраняет омонимичные муниципалитеты как разные серии и даёт 2 016 полных рядов.
