# editing-product-card-images-for-MP
## Персиковый фон с тканью

```bash
pip install -r requirements.txt
python3 peach_card.py card.png output
```

Результат: `output/card_peach.png`, `output/card_peach.jpg` (q92), `output/card_peach_preview.png` (300 px).
Цвета, фактура, тень, текст — переменные в начале `peach_card.py`.

## v2: объёмный студийный фон

```bash
python3 peach_card_v2.py card.png output
```

Тональный диапазон, световое пятно, складки ткани, контактная и падающая тени, отсвет на металле,
товар ×1.08, «стекло» под «дешеддер + колтунорез», автоподбор бирюзового под WCAG.
Результат: `output/card_peach_v2.png|.jpg`, `card_peach_v2_preview.png`, `card_peach_v2_compare.png`.

## v3: исходный фон, всё крупнее, три значка

```bash
python3 card_v3.py card.png output
```

Надписи и расчёска ×1.18, «для кошек и собак», вместо плашки — три значка с подписями.
Результат: `output/card_v3.png`, сравнение `output/card_v3_compare.png`.

## v4: правки v3

```bash
python3 card_v4.py card.png output
```

Без «Видеообзора»; строка «Двусторонняя, нержавеющая сталь»; классический щит; подписи 54 Bold;
подзаголовок плотнее и ближе к заголовку. Результат: `output/card_v4.png`, сравнение с v3 `output/card_v4_compare.png`.

## v5: правки v4

```bash
python3 card_v5.py card.png output
```

Без «2 в 1»/выноски; «Двусторонняя, / нержавеющая сталь» слева у ручки; тень + контактная тень;
значки ×1.5, подписи ExtraBold. Результат: `output/card_v5.png`, сравнение с v4 `output/card_v5_compare.png`.

## v6: с исходника, две выноски и значки

```bash
python3 card_v6.py card.png output
```

Расчёска ×1.18, подзаголовок 110 px, «2 в 1» с выноской справа, выноска к лезвиям слева,
три значка (подписи 54 Black), без теней. Результат: `output/card_v6.png`, сравнение `output/card_v6_compare.png`.

## v7: левая подпись как «дешеддер + колтунорез»

```bash
python3 card_v7.py card.png output
```

Слева «нержавеющая / сталь» — Montserrat Bold, бирюзовый, размер/трекинг/шаг строк как у «дешеддер + колтунорез».
Остальное как в v6. Результат: `output/card_v7.png`, сравнение `output/card_v7_compare.png`.

## v8: короткая левая выноска

```bash
python3 card_v8.py card.png output
```

Горизонталь ровно под «нержавеющая», изгиб вниз на правом конце, «сталь» по центру. Остальное как в v7.
Результат: `output/card_v8.png`, сравнение `output/card_v8_compare.png`.

v8 с тенью: `SHADOW = True` в `card_v8.py` (тёплый серый #6B6560, 27%, размытие 18 px у ручки и 9 px у лезвий,
контактная 8 px под нижними точками). Версия без тени — `output/card_v8_noshadow.png`.

## Жёлтая расчёска по образцу v8

```bash
python3 card_yellow.py yellow.png card.png output
```

Фото не вырезается: кладётся умножением на фон образца (белый фото → фон карточки). Масштаб пропорциональный,
вписан в рамку бирюзовой расчёски. Акцент подобран по ручке и затемнён до контраста 4.5:1.
Результат: `output/card_yellow.png`, сравнение с бирюзовой `output/card_yellow_compare.png`.

Жёлтая v2: `python3 card_yellow_v2.py yellow.png card.png output` — акценты цветом ручки #F0F341 на тёмно-серых плашках,
иконки жёлтые с тёмно-серым знаком. Результат `output/card_yellow_v2.png`, сравнение `output/card_yellow_v2_compare.png`.

## Синяя расчёска

```bash
python3 card_blue.py blue.png card.png output
```

Как жёлтая v1: фото без вырезания, акцент по пикселям ручки, затемнён до 4.5:1, тень только снаружи силуэта.
Результат `output/card_blue.png`, сравнение `output/card_blue_compare.png`, проверка лезвий `output/card_blue_blades_check.png`.
