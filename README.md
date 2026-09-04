# laim-oos-oot-test

Нода мониторингового контура LAIM: тест разделимости выборок OOS/OOT.
Принимает **эталонную корзину** (`reference_umr`), **данные мониторинга**
(`monitoring_umr`) и **валидированный контракт метрики**
(`monitoring_metric`), обучает CatBoost отличать одну выборку от другой по
тексту запросов и отдаёт в агрегатор светофор с Gini, его разбросом
и диагностикой входа (`all_results`).

## Зачем нода нужна

Эталонная корзина собрана во время валидации (out-of-sample), данные
мониторинга приходят позже (out-of-time). Если стандартная модель легко
отличает их по одним только вопросам пользователей, распределение входов
агента сдвинулось, и остальные тесты читают метрики на другой популяции.
Ключевые решения:

- **Adversarial-подход без эмбеддингов и сети.** Единственный признак —
  текст запроса как текстовый признак CatBoost; Gini и есть мера дрифта.
- **Единица наблюдения — независимая группа, а не строка.** Разбиение
  train/test идёт по `reference_group_id` (диалог, сессия или строка QA):
  реплики одного диалога не оказываются по обе стороны разбиения.
- **Деградация вместо падения при малых выборках.** Меньше 50 независимых
  групп в любой выборке — серый светофор с `reason_code`, а не падение.

## Роль по методике (карточка 6.3.6)

Тест детектирует изменение распределения входных запросов относительно
эталонной выборки. Красный светофор итерации по этому тесту не формируется:
подтверждённое изменение — жёлтый сигнал и основание для внеочередной
разметки, а деградацию качества подтверждает тест динамики КМ. Пока эталонной
выборки стабильного периода ПРОМ нет (порт `reference_stable_umr` пуст), тест
сравнивает поток с корзиной первичной валидации и по методике информативен:
устойчивая разделимость корзины и потока отражает способ формирования
корзины. Вердикт выставляется по нижней границе разброса Gini по ресемплам
(`verdict_statistic = gini_ci_lower`): цвет не должен держаться на единичном
удачном разбиении; это разброс, а не доверительный интервал генеральной
совокупности.

## Место в контуре

```text
laim-baskets-adapter.reference_umr ─────────────────┐
laim-traces-dataset-converter.monitoring_umr ───────┼──► laim-oos-oot-test
laim-kriteria-selector.validated_monitoring_metric ─┘         │
        all_results      ─► laim-agg.in ◄─────────────────────┤
        test_description ─► HTML на карточке ноды (в port_wiring.json не подключён)
```

## Порты и настройки

### Входы (все обязательные)

| Порт | Тип | Что приходит с платформы |
|---|---|---|
| `reference_umr` | dataframe | Эталонная корзина `laim-umr.v2`: плоская форма `query_id`/`input_query`/`output_answer` либо упакованный `dialogue`; `main_metric` обязателен |
| `monitoring_umr` | dataframe | Выход `laim-traces-dataset-converter` в том же формате без `main_metric`; DataFrame, parquet-байты или путь к parquet |
| `monitoring_metric` | default | Контракт `laim-monitoring-metric.v2` c `assessment_mode` (`qa`, `turn_with_history`, `dialogue`); `v1` поднимается автоматически |
| `reference_stable_umr` | нет | UMR эталонной выборки стабильного периода ПРОМ (карточка 6.3.6). Если задан и не пуст — используется вместо корзины первичной валидации, `reference_source = stable_period`, результат светофорный (максимум жёлтый по таблице 15). Без него сравнение идёт с корзиной валидации: `reference_source = validation_basket`, `informative = true` — переходное положение методики, светофор итерации не формирует |

### Выходы

| Порт | Тип | Что отдаёт |
|---|---|---|
| `all_results` | default | Светофор (`calculated_traffic_lights.test_light`, `color`) и показатели теста для `laim-agg` |
| `test_description` | hidden | HTML-отчёт: цель, условия, алгоритм, критерии, таблица результатов |

### Настройки ноды

| Настройка | По умолчанию | Зачем менять |
|---|---|---|
| `yellow_threshold` | `0.4` | Нижняя граница жёлтой зоны Gini |
| `red_threshold` | `0.8` | Нижняя граница красной зоны Gini. Пороги сортируются: перепутанные местами значения дают тот же результат |
| `min_groups_per_side` | `50` | Минимум независимых групп (сессий в `dialogue`/`turn_with_history`, запросов в `qa`) в каждой из выборок; меньше — `not_computable`, `insufficient_independent_groups` |
| `resampling_iterations` | `20` | Число повторных разбиений; меньше 1 — падение `ValueError` |
| `test_size` | `0.3` | Доля групп в тестовой части каждого разбиения |
| `catboost_iterations` | `300` | Число деревьев CatBoost |
| `catboost_learning_rate` | `0.1` | Шаг обучения CatBoost |
| `catboost_depth` | `4` | Глубина деревьев CatBoost |
| `catboost_early_stopping_rounds` | `30` | Ранняя остановка по `eval_set`; действует только при train-части от 100 строк |
| `random_state` | `17` | Seed генератора, из которого выводятся независимые seed'ы для разбиений и для модели |
| `is_info` | `false` | Информативный режим: Gini считается, но светофор всегда серый |

## Как проходит прогон

**1. Контракт и drift-фреймы.** `main` сначала валидирует
`monitoring_metric`. При входном `status: not_computable` нода сразу возвращает
серый результат с исходными `reason_code` и `reason`, не читая UMR и не запуская
CatBoost. Для вычислимого контракта `prepare_drift_frames` приводит оба UMR
к единицам наблюдения по
`assessment_mode` и собирает фреймы `question`, `answer` (пустая), `target`
(`main_metric`), `reference_group_id`. В `dialogue` `question` —
JSON-список всех запросов диалога; в `turn_with_history` — история запросов
сессии плюс текущий; в `qa` — сам `input_query`. Строки эталона с пустым
`target` отбрасываются (при `missing_policy: fail` — падение).

**2. Семплер.** Эталон — класс 0 (OOS, слот `train`), мониторинг — класс 1
(OOT, слот `test`). `_group_id` берётся из `reference_group_id` с префиксом
`oos:`/`oot:`; пустая группа — `row:<номер строки>`.

**3. Guardrail.** Меньше `min_groups_per_side` независимых групп (настройка, 50)
в любой выборке — серый светофор, `status: not_computable`,
`reason_code: insufficient_independent_groups`; дальше расчёт не идёт.

**4. Нормализация префикса.** Если в одной выборке не менее 95 % запросов
начинаются с токена вида `TOKEN текст` из словаря не больше
`max(3, sqrt(n))` классов, а в другой — не более 5 %, префикс удаляется
в обеих (иначе классификатор учится на служебной разметке); факт попадает
в `input_normalization` и в WARNING.

**5. Итерации (`resampling_iterations` раз).** `GroupShuffleSplit`
по `_group_id` отдельно внутри каждого класса, доля test — `test_size`;
пересечение групп train/test проверяется. При train-части от 100 строк
до 15 % её групп отщепляются под `eval_set` для ранней остановки. CatBoost
с `auto_class_weights="Balanced"`, `eval_metric="AUC"`; на test-части
считается `Gini = max(0, 2*AUC - 1)`.

**6. Итог.** Среднее, выборочное std и 95 % CI среднего
(`1.96 * std / sqrt(n)`, обрезка в `[0, 1]`). Светофор: зелёный при
`Gini < 0.4`, жёлтый при `0.4 <= Gini < 0.8`, красный при `Gini >= 0.8`.
На выходе `yellow` переименовывается в `amber`: `laim-agg` и платформа
знают только словарь `red`/`amber`/`green`/`gray`.

### Пример лога прогона (эталон 283 диалога, мониторинг 94)

```text
INFO llm_val.sampler: Agent columns: ['question', 'answer', 'target'], размер: 94
INFO llm_val.sampler: Real columns: ['question', 'answer', 'target'], размер: 283
INFO main: Тест OOS-OOT запущен
INFO llm_val.valtest_adversarial_test: Колонки для использования: ['question']
INFO llm_val.valtest_adversarial_test: Размер OOS=283, OOT=94
INFO llm_val.valtest_adversarial_test: Итерация #0
INFO llm_val.valtest_adversarial_test: Gini iter#0: 0.2320
INFO main: {'report': {'semaphore': 'green', ...}, 'precomputed': {'status': 'computed', 'gini_value': 0.3902231237322515, 'gini_std': 0.07720653836698857, 'gini_ci_lower': np.float64(0.3563858663854292), 'gini_ci_upper': np.float64(0.4240603810790738), 'resampling_iterations': 20, 'n_oos': 283, 'n_oot': 94, 'n_oos_groups': 283, 'n_oot_groups': 94, 'input_normalization': None}}
```

Итерации 1–19 пропущены. Серый исход и нормализация видны как WARNING:

```text
WARNING llm_val.valtest_adversarial_test: Одна из выборок < 50 независимых групп: OOS=<n_oos_groups>, OOT=<n_oot_groups>. Тест неинформативен.
WARNING llm_val.valtest_adversarial_test: Перед OOS-OOT удалён sample-exclusive структурный префикс; diagnostics={'method': 'sample_exclusive_identifier_prefix', ...}
```

## Форматы выхода и контракты

Порт `all_results` — JSON-объект с плоскими полями (реальный результат
прогона выше):

```json
{
  "calculated_traffic_lights": {"test_light": "green",
    "semaphore_title": "Результат теста разделения выборок соответствует зеленому светофору"},
  "color": "green", "status": "computed", "reason_code": null, "reason": null,
  "gini_mean": 0.390223, "gini_std": 0.077207,
  "gini_ci_lower": 0.356386, "gini_ci_upper": 0.42406,
  "n_oos": 283, "n_oot": 94, "n_oos_groups": 283, "n_oot_groups": 94,
  "input_normalization": null, "test_name": "oos_oot"
}
```

Всегда заполнены: `color` и `calculated_traffic_lights.test_light`
(`green`/`amber`/`red`/`gray`), `status` (`computed` либо `not_computable`),
`n_oos`/`n_oot` (единицы наблюдения), `n_oos_groups`/`n_oot_groups`
(уникальные `reference_group_id`), `test_name`. Условно: `reason_code`
и `reason` — при `not_computable`; четыре поля `gini_*` — при `computed`
(6 знаков), `null` при сером по guardrail; `input_normalization` — `null`
либо объект `method`, `sample`, `other_sample`, `sample_rate`,
`other_sample_rate`, `prefix_classes`, `removed_oos`, `removed_oot`.
Единица наблюдения — по `assessment_mode`: `qa` — строка, `dialogue` —
сессия, `turn_with_history` — реплика с историей (группа — `session_id`).

## Падение против деградации

Нода падает с исключением (детали в логе платформы):

| Причина | Исключение |
|---|---|
| `monitoring_metric` не object, неизвестная версия, в вычислимом контракте нет `assessment_mode`, нарушен `scoring` | `MonitoringContractError` |
| UMR пуст, не DataFrame, ни плоская, ни `dialogue`-форма, смешаны обе формы, пустой `query_id`/`session_id`; `monitoring_umr` — нечитаемый parquet | `MonitoringContractError` |
| В `reference_umr` нет `main_metric`; пустой `main_metric` при `missing_policy: fail` | `MonitoringContractError` |
| `resampling_iterations < 1` | `ValueError` |

Деградации (нода возвращает результат):

| Событие | Реакция |
|---|---|
| Входной `monitoring_metric.status: not_computable` | серый, исходные `reason_code`/`reason`, вычислительный путь пропущен |
| Меньше 50 независимых групп в OOS или OOT | серый, `status: not_computable`, `reason_code: insufficient_independent_groups`, WARNING |
| `is_info = true` | серый, `status: computed`, Gini и CI заполнены |
| Sample-exclusive identifier-префикс в вопросах | префикс удалён, `input_normalization` заполнен, WARNING |
| Пустой `main_metric` в эталоне при `missing_policy` кроме `fail` | строка исключена из OOS |
| Train-часть меньше 100 строк | CatBoost без `eval_set` и ранней остановки |

## Внешние сервисы

Не применимо: нет обращений к LLM, эмбеддингам, HDFS и сети; расчёт детерминирован при фиксированном `random_state`.

## Наблюдаемость

Лог платформы (логгеры по имени модуля): размеры OOS/OOT, Gini каждой
итерации, полный словарь результата, WARNING по guardrail и нормализации.
Отдельного порта журнала нет: источник истины — `all_results` в `laim-agg`.
Триаж на сотне прогонов — по его полям: `status` и `reason_code` (сколько
серых и почему), `color`, `gini_mean` с границами CI (широкий CI — мало
итераций или малые выборки), `input_normalization` (чьи вопросы несут
технический префикс), `n_oos_groups` против `n_oos` (равны только в `qa`).

## Карта кода

```text
main.py                              порты платформы, светофор в словарь платформы, HTML-отчёт
laim_monitoring/core.py              контракт laim-monitoring-metric.v2, единицы наблюдения, prepare_drift_frames
llm_val/valtest_adversarial_test.py  guardrail, нормализация префикса, group-split, CatBoost, Gini, CI
llm_val/sampler.py                   AutoAsessorSampler: reference -> train (OOS), monitoring -> test (OOT)
llm_val/report_helper.py             semaphore_by_threshold и прочие светофоры библиотеки llm_val
llm_val/utils.py                     string_to_float для семплера
html_report_helper.py                HTML-светофоры и таблица критериев
tests/                               контракт, единицы наблюдения, drift-фреймы, main для qa/dialogue
```

## Что делать, если

- **Серый, `reason_code: insufficient_independent_groups`** — в `reason`
  указаны `n_oos_groups` и `n_oot_groups`; нужно больше независимых
  диалогов/сессий в мониторинге или корзине, а не другие пороги.
- **Серый при `status: computed`** — включён `is_info`; Gini и CI заполнены.
- **Красный при похожих по смыслу выборках** — смотрите
  `input_normalization`: если `null`, а вопросы мониторинга несут служебные
  префиксы или иное форматирование, классификатор различает форму.
- **Серый с причиной из `monitoring_metric`** — upstream не смог построить
  план измерения; чинить артефакты корзины, а не эту ноду.

## Деплой

Нода самодостаточна: никаких импортов из соседних каталогов и общих
пакетов. База — `py312-simple`; синтаксис и stdlib новее Python 3.12 не
используются. `descriptor.json` перечисляет в
`script.runConfiguration.sourceFiles` все файлы карты кода (`main.py`,
`html_report_helper.py`, пять модулей `llm_val/`, `laim_monitoring/__init__.py`
и `core.py`); точка входа — `main` в `main.py`. Автотеста соответствия
`sourceFiles` диску нет; `tests/` проверяют в дескрипторе только описание
порта `monitoring_umr`. Зависимости `requirements.txt`: `catboost`,
`scikit-learn` (`roc_auc_score`, `GroupShuffleSplit`), `pandas`, `tqdm`
(импортируется, прогресс-бар выключен), `ipython` (`IPython.display` в
`html_report_helper.py`), `jinja2` (`pandas.Styler.to_html`). Тесты:
`python3 -m pytest -q tests`.

## Глоссарий

- **OOS / OOT** — эталонная корзина валидации (out-of-sample, класс 0) /
  данные мониторинга после валидации (out-of-time, класс 1).
- **Adversarial-тест** — классификатор учится отличать OOS от OOT;
  его качество и есть мера дрифта.
- **Gini** — `2*AUC - 1`, обрезанный снизу нулём: 0 — неразличимы, 1 — разделяются идеально.
- **Независимая группа** — единица наблюдения по `assessment_mode`
  (диалог, сессия, строка) с `reference_group_id`; разбиение train/test
  группу не разрывает.
- **Sample-exclusive identifier-префикс** — технический токен в начале
  запроса почти только в одной выборке; удаляется перед обучением.
