# LAIM OOS-OOT Stability Test

Тест на стабильность данных для оценки дрифта между out-of-sample (OOS) и out-of-time (OOT) выборками.

## Описание

Тест использует adversarial approach для определения того, насколько легко можно различить две выборки данных с помощью стандартной ML модели (CatBoost). Высокое значение Gini указывает на значительный дрифт данных.

### Алгоритм работы

1. Составляется adversarial датасет из OOS и OOT данных
2. Обучается CatBoost классификатор для разделения выборок
3. Вычисляется Gini коэффициент: `Gini = 2 * AUC - 1`
4. Процедура повторяется несколько раз с разными random_state для усреднения

### Критерии светофора

| Gini | Светофор | Интерпретация |
|------|----------|---------------|
| < 0.4 | 🟢 Зеленый | Выборки похожи, дрифт минимальный |
| 0.4 - 0.8 | 🟡 Желтый | Умеренный дрифт |
| > 0.8 | 🔴 Красный | Выборки легко разделить, значительный дрифт |

## Структура проекта

```
.
├── main.py                          # Точка входа, основная функция теста
├── llm_val/
│   ├── valtest_adversarial_test.py  # Основная логика теста (Gini расчет)
│   ├── sampler.py                   # Классы для семплирования данных
│   ├── scorer.py                    # Классы для подсчета метрик
│   ├── report_helper.py             # Функции формирования светофоров
│   └── utils.py                     # Вспомогательные функции
├── test_example/
│   ├── html_report_helper.py        # HTML генерация отчетов
│   └── report_helper.py             # Дополнительные утилиты отчетов
└── descriptor.json                  # Конфигурация Docker-деплоя
```

## Модули

### llm_val/valtest_adversarial_test.py

Основная логика теста:
- `valtest_adversarial_text()` - главная функция теста
- `get_adversarial_gini_score()` - обучение CatBoost и расчет Gini
- `make_adversarial_dataset()` - создание adversarial датасета

### llm_val/sampler.py

Классы для работы с данными:
- `Sampler` - базовый абстрактный класс
- `AutoAsessorSampler` - преобразование DataFrame в формат теста

### llm_val/scorer.py

Классы для подсчета метрик:
- `Scorer` - базовый абстрактный класс
- `AutoAsessorScorer` - расчет метрик для данных автоасессора

### llm_val/report_helper.py

Функции формирования светофоров:
- `semaphore_by_threshold()` - основная функция светофора
- `worst_semaphore()` - худший светофор из списка
- `proportion_semaphore()` - пропорциональный светофор

### main.py

Точка входа:
- `main()` - основная функция, вызываемая при деплое
- `html_report_valtest_oos_oot()` - генерация HTML-отчета

## Зависимости

```
catboost
scikit-learn
pandas
tqdm
ipython
jinja2
```

## Использование

```python
from main import main

result = main(
    real_asessor_df=df_real,
    auto_asessor_df=df_agent,
    metric_selector_res={"main_metric": "metric_name"},
    metric_agg="single_mean",
    resampling_iterations=20,
    test_size=0.3,
)
```

## Конфигурация (descriptor.json)

Тест настраивается через UI параметры:
- `metric_agg` - способ агрегации метрик
- `red_threshold` - порог красного светофора
- `green_threshold` - порог зеленого светофора
- `resampling_iterations` - количество итераций ресемплинга
- `test_size` - доля тестовой выборки
- `greater_is_better` - инвертирование логики светофора
- `is_info` - сделать тест информативным (всегда серый светофор)