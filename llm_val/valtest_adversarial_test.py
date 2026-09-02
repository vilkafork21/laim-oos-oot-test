"""
Adversarial test for OOS-OOT stability detection.

Модуль содержит основную логику теста на стабильность данных.
Тест использует adversarial approach - пытается различить OOS и OOT
выборки с помощью ML модели (CatBoost). Высокая разделимость (высокий Gini)
указывает на значительный дрифт данных.
"""

import json
import logging
import re
import typing as tp

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from llm_val.report_helper import semaphore_by_threshold
from llm_val.sampler import Sampler

logger = logging.getLogger(__name__)


# Минимальное количество наблюдений в каждой из выборок для интерпретируемого Gini.
# Ниже этой границы тест автоматически возвращает gray (см. P2-4).
MIN_SAMPLES_PER_CLASS = 50
GROUP_COLUMN = "_group_id"
_STRUCTURAL_PREFIX = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s+(.+)$", re.DOTALL)
_PREFIX_SAMPLE_RATE = 0.95
_PREFIX_OTHER_SAMPLE_RATE = 0.05


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТА
# =============================================================================

def report_valtest_adversarial_text(
    gini_value: float,
    gini_std: float,
    gini_ci: tp.Tuple[float, float],
    semaphore_threshold: tp.Tuple[float, float] = (0.4, 0.8),
    is_info: bool = False,
) -> tp.Dict[str, tp.Any]:
    """
    Создание отчета по результатам теста.

    Args:
        gini_value: Значение Gini коэффициента (среднее по ресемплингам)
        gini_std: Стандартное отклонение Gini по ресемплингам
        gini_ci: 95% CI среднего Gini по повторным разбиениям
        semaphore_threshold: Пороги (yellow_low, red_low) для светофора
        is_info: Флаг информативного теста

    Returns:
        Словарь с результатами теста в формате llm_val
    """
    color = (
        "gray"
        if is_info or np.isnan(gini_value)
        else semaphore_by_threshold(
            gini_value,
            threshold=semaphore_threshold,
            greater_is_better=False,
            left_border_is_bigger=True,
            right_border_is_bigger=True,
        )
    )

    result_dataframe = pd.DataFrame(
        {
            "Значение Джини": [round(float(gini_value), 4)] if not np.isnan(gini_value) else [None],
            "Стандартное отклонение Gini": [round(float(gini_std), 4)] if not np.isnan(gini_std) else [None],
            "95% CI (нижняя)": [round(float(gini_ci[0]), 4)] if not np.isnan(gini_ci[0]) else [None],
            "95% CI (верхняя)": [round(float(gini_ci[1]), 4)] if not np.isnan(gini_ci[1]) else [None],
            "Светофор": [color],
        }
    )
    return {
        "semaphore": color,
        "result_dataframes": [result_dataframe],
        "result_plots": [],
    }


def _with_groups(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    result = frame.copy()
    namespace = f"{sample_name}:"
    if GROUP_COLUMN not in result:
        result[GROUP_COLUMN] = [f"{namespace}row:{i}" for i in range(len(result))]
    else:
        groups = []
        for index, value in enumerate(result[GROUP_COLUMN].tolist()):
            if pd.isna(value) or not str(value).strip():
                groups.append(f"{namespace}row:{index}")
                continue
            text = str(value).strip()
            groups.append(text if text.startswith(namespace) else namespace + text)
        result[GROUP_COLUMN] = groups
    return result


def _normalize_sample_prefix(
    base_questions: pd.Series,
    new_questions: pd.Series,
) -> tuple[pd.Series, pd.Series, dict | None]:
    def prefix(value: object) -> str | None:
        match = _STRUCTURAL_PREFIX.match(str(value).strip())
        return match.group(1).casefold() if match else None

    profiles = []
    for name, questions in (("oos", base_questions), ("oot", new_questions)):
        tokens = pd.Series([
            prefix(turn) for question in questions
            for turn in _question_turns(question)[0]
        ])
        profiles.append((name, tokens, len(tokens)))
    for high_name, high_tokens, high_size in profiles:
        low_name, low_tokens, low_size = next(
            profile for profile in profiles if profile[0] != high_name
        )
        vocabulary = set(high_tokens.dropna().tolist())
        max_vocabulary = max(3, int(np.sqrt(max(high_size, 1))))
        high_rate = float(high_tokens.notna().mean()) if high_size else 0.0
        low_rate = float(low_tokens.notna().mean()) if low_size else 0.0
        if (
            high_rate >= _PREFIX_SAMPLE_RATE
            and low_rate <= _PREFIX_OTHER_SAMPLE_RATE
            and 0 < len(vocabulary) <= max_vocabulary
        ):
            base_clean, removed_oos = _remove_vocabulary_prefix(base_questions, vocabulary)
            new_clean, removed_oot = _remove_vocabulary_prefix(new_questions, vocabulary)
            return base_clean, new_clean, {
                "method": "sample_exclusive_identifier_prefix",
                "sample": high_name,
                "other_sample": low_name,
                "sample_rate": high_rate,
                "other_sample_rate": low_rate,
                "prefix_classes": len(vocabulary),
                "removed_oos": removed_oos,
                "removed_oot": removed_oot,
            }
    return base_questions, new_questions, None


def _remove_vocabulary_prefix(
    questions: pd.Series,
    vocabulary: set[str],
) -> tuple[pd.Series, int]:
    values = []
    removed = 0
    for value in questions:
        turns, serialized = _question_turns(value)
        cleaned = []
        for text in turns:
            match = _STRUCTURAL_PREFIX.match(text)
            if match and match.group(1).casefold() in vocabulary:
                removed += 1
                text = match.group(2).strip()
            cleaned.append(text)
        value = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
        values.append(value if serialized else cleaned[0])
    return pd.Series(values, index=questions.index), removed


def _question_turns(value: object) -> tuple[list[str], bool]:
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [text], False
    if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
        return [item.strip() for item in parsed], True
    return [text], False


def _not_computable(
    semaphore_threshold,
    *,
    reason_code: str,
    reason: str,
    n_oos: int,
    n_oot: int,
    n_oos_groups: int,
    n_oot_groups: int,
) -> dict:
    precomputed = {
        "status": "not_computable",
        "gini_value": np.nan,
        "gini_std": np.nan,
        "gini_ci_lower": np.nan,
        "gini_ci_upper": np.nan,
        "resampling_iterations": 0,
        "n_oos": int(n_oos),
        "n_oot": int(n_oot),
        "n_oos_groups": int(n_oos_groups),
        "n_oot_groups": int(n_oot_groups),
        "reason_code": reason_code,
        "reason": reason,
    }
    return {
        "report": report_valtest_adversarial_text(
            np.nan, np.nan, (np.nan, np.nan), semaphore_threshold, is_info=True
        ),
        "precomputed": precomputed,
    }


# =============================================================================
# ОСНОВНОЙ ТЕСТ
# =============================================================================

def valtest_adversarial_text(
    sampler: Sampler,
    semaphore_threshold: tp.Tuple[float, float] = (0.4, 0.8),
    data_types: tp.Tuple[str, str] = ("train", "test"),
    resampling_iterations: int = 20,
    test_size: float = 0.3,
    catboost_iterations: tp.Optional[int] = 300,
    catboost_learning_rate: tp.Optional[float] = 0.1,
    catboost_depth: tp.Optional[int] = 4,
    catboost_early_stopping_rounds: int = 30,
    random_state: int = 17,
    use_tqdm: bool = False,
    is_info: bool = False,
    gini_value: tp.Optional[float] = None,
    **kwargs,
) -> tp.Dict[str, tp.Any]:
    """
    Тест для оценки изменения выборки OOT по сравнению с OOS
    посредством разделения выборок стандартной моделью.

    Алгоритм:
    1. Составляется adversarial датасет из OOS и OOT данных
    2. Обучается CatBoost классификатор для разделения выборок
    3. Вычисляется Gini коэффициент на тестовой части (clip снизу нулём — P1-3)
    4. Процедура повторяется несколько раз с независимыми seed'ами (P1-4)
    5. Считается среднее, стандартное отклонение и 95% CI среднего Gini
    """
    if resampling_iterations < 1:
        raise ValueError("resampling_iterations должен быть положительным")

    # Независимые RNG-потоки для split и для модели (P1-4)
    rng = np.random.default_rng(seed=random_state)
    split_seeds = rng.integers(low=0, high=10**6, size=resampling_iterations)
    model_seeds = rng.integers(low=0, high=10**6, size=resampling_iterations)

    gini_std = np.nan
    gini_ci = (np.nan, np.nan)
    n_base = len(sampler.train["X"])
    n_new = len(sampler.test["X"])
    base_groups = _with_groups(sampler.train["X"], "oos")
    new_groups = _with_groups(sampler.test["X"], "oot")
    n_base_groups = base_groups[GROUP_COLUMN].nunique()
    n_new_groups = new_groups[GROUP_COLUMN].nunique()

    if gini_value is None:
        cols_to_use = ["question"]
        logger.info(f"Колонки для использования: {cols_to_use}")

        base_columns = cols_to_use + (
            [GROUP_COLUMN] if GROUP_COLUMN in sampler.train["X"] else []
        )
        new_columns = cols_to_use + (
            [GROUP_COLUMN] if GROUP_COLUMN in sampler.test["X"] else []
        )
        base_data = _with_groups(
            sampler.train["X"][base_columns].copy(), "oos"
        )
        new_data = _with_groups(
            sampler.test["X"][new_columns].copy(), "oot"
        )

        # Guardrail на минимальный размер выборок (P2-4)
        logger.info(f"Размер OOS={n_base}, OOT={n_new}")
        if min(n_base_groups, n_new_groups) < MIN_SAMPLES_PER_CLASS:
            logger.warning(
                f"Одна из выборок < {MIN_SAMPLES_PER_CLASS} независимых групп: "
                f"OOS={n_base_groups}, OOT={n_new_groups}. Тест неинформативен."
            )
            return _not_computable(
                semaphore_threshold,
                reason_code="insufficient_independent_groups",
                reason=(
                    "Недостаточно независимых единиц наблюдения: "
                    f"min(n_oos_groups={n_base_groups}, "
                    f"n_oot_groups={n_new_groups}) < {MIN_SAMPLES_PER_CLASS}"
                ),
                n_oos=n_base,
                n_oot=n_new,
                n_oos_groups=n_base_groups,
                n_oot_groups=n_new_groups,
            )

        (
            base_data["question"],
            new_data["question"],
            normalization_diagnostics,
        ) = _normalize_sample_prefix(
            base_data["question"], new_data["question"]
        )
        if normalization_diagnostics is not None:
            logger.warning(
                "Перед OOS-OOT удалён sample-exclusive структурный префикс; "
                "diagnostics=%s",
                normalization_diagnostics,
            )

        gini_values = []
        iterator = tqdm(range(resampling_iterations)) if use_tqdm else range(resampling_iterations)
        for iteration in iterator:
            logger.info(f"Итерация #{iteration}")
            adversarial_dataset = make_adversarial_dataset(
                base_data,
                new_data,
                test_size=test_size,
                random_state=int(split_seeds[iteration]),
            )
            gini_values.append(
                get_adversarial_gini_score(
                    adversarial_dataset,
                    catboost_iterations=catboost_iterations,
                    catboost_learning_rate=catboost_learning_rate,
                    catboost_depth=catboost_depth,
                    catboost_early_stopping_rounds=catboost_early_stopping_rounds,
                    random_state=int(model_seeds[iteration]),
                )
            )
            logger.info(f"Gini iter#{iteration}: {gini_values[-1]:.4f}")

        gini_array = np.asarray(gini_values, dtype=float)
        gini_value = float(np.mean(gini_array))
        gini_std = float(np.std(gini_array, ddof=1)) if len(gini_array) > 1 else 0.0
        margin = 1.96 * gini_std / np.sqrt(len(gini_array))
        gini_ci = (
            max(0.0, gini_value - margin),
            min(1.0, gini_value + margin),
        )
    else:
        normalization_diagnostics = None

    precomputed = {
        "status": "computed",
        "gini_value": gini_value,
        "gini_std": gini_std,
        "gini_ci_lower": gini_ci[0],
        "gini_ci_upper": gini_ci[1],
        "resampling_iterations": int(resampling_iterations),
        "n_oos": int(len(sampler.train["X"])),
        "n_oot": int(len(sampler.test["X"])),
        "n_oos_groups": int(n_base_groups),
        "n_oot_groups": int(n_new_groups),
        "input_normalization": normalization_diagnostics,
    }
    report = report_valtest_adversarial_text(
        gini_value, gini_std, gini_ci, semaphore_threshold, is_info
    )
    return {"report": report, "precomputed": precomputed}


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_adversarial_gini_score(
    adversarial_dataset: tp.Dict[str, pd.DataFrame],
    catboost_iterations: tp.Optional[int] = 300,
    catboost_learning_rate: tp.Optional[float] = 0.1,
    catboost_depth: tp.Optional[int] = 4,
    catboost_early_stopping_rounds: int = 30,
    random_state: int = 17,
) -> float:
    """
    Вычисление Gini коэффициента для adversarial задачи.

    Изменения относительно baseline:
    - Явные гиперпараметры CatBoost (P1-1)
    - auto_class_weights="Balanced" против дисбаланса OOS/OOT (P1-2)
    - eval_set + early_stopping_rounds (P1-1)
    - Gini обрезается снизу нулём, т.к. отрицательный Gini ≡ AUC<0.5 ≡ шум (P1-3)
    """
    train_df = adversarial_dataset["train"]
    test_df = adversarial_dataset["test"]
    feature_columns = [
        column for column in train_df.columns
        if column not in {"target", GROUP_COLUMN}
    ]
    text_features = list(feature_columns)

    # Eval для early stopping тоже отделяется по единицам наблюдения.
    can_make_group_eval = (
        len(train_df) >= 100
        and GROUP_COLUMN in train_df
        and all(
            part[GROUP_COLUMN].nunique() >= 2
            for _, part in train_df.groupby("target", sort=True)
        )
    )
    if can_make_group_eval:
        eval_size = min(0.15, 100 / len(train_df))
        split = _group_stratified_split(
            train_df,
            test_size=eval_size,
            random_state=random_state,
        )
        train_sub, eval_sub = split["train"], split["test"]
        train_pool = Pool(train_sub[feature_columns], train_sub["target"], text_features=text_features)
        eval_pool = Pool(eval_sub[feature_columns], eval_sub["target"], text_features=text_features)
    else:
        train_pool = Pool(train_df[feature_columns], train_df["target"], text_features=text_features)
        eval_pool = None

    test_pool = Pool(test_df[feature_columns], test_df["target"], text_features=text_features)

    model = CatBoostClassifier(
        iterations=catboost_iterations,
        learning_rate=catboost_learning_rate,
        depth=catboost_depth,
        random_seed=random_state,
        logging_level="Silent",
        allow_writing_files=False,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        early_stopping_rounds=catboost_early_stopping_rounds if eval_pool is not None else None,
    )
    if eval_pool is not None:
        model.fit(train_pool, eval_set=eval_pool)
    else:
        model.fit(train_pool)

    test_preds = model.predict_proba(test_pool)[:, 1]
    auc = roc_auc_score(test_df["target"], test_preds)
    gini = 2.0 * auc - 1.0
    return max(0.0, gini)


def make_adversarial_dataset(
    base_data: pd.DataFrame,
    new_data: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 17,
) -> tp.Dict[str, pd.DataFrame]:
    """
    Создание adversarial датасета для разделения выборок.

    Объединяет base_data (target=0) и new_data (target=1). Не мутирует
    входные DataFrame (P1-5).
    """
    base_labeled = _with_groups(base_data, "oos").assign(target=0)
    new_labeled = _with_groups(new_data, "oot").assign(target=1)
    full_df = pd.concat([base_labeled, new_labeled], axis=0, ignore_index=True)

    text_cols = [
        col for col in full_df.columns if col not in {"target", GROUP_COLUMN}
    ]
    full_df[text_cols] = full_df[text_cols].fillna("")

    return _group_stratified_split(
        full_df,
        test_size=test_size,
        random_state=random_state,
    )


def _group_stratified_split(
    frame: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tp.Dict[str, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for target, part in frame.groupby("target", sort=True):
        if part[GROUP_COLUMN].nunique() < 2:
            raise ValueError(
                f"Недостаточно независимых групп для target={target!r}"
            )
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_index, test_index = next(
            splitter.split(part, groups=part[GROUP_COLUMN])
        )
        train_parts.append(part.iloc[train_index])
        test_parts.append(part.iloc[test_index])
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    overlap = set(train_df[GROUP_COLUMN]) & set(test_df[GROUP_COLUMN])
    if overlap:
        raise RuntimeError(
            "Group split допустил пересечение единиц наблюдения между train/test"
        )
    return {"train": train_df, "test": test_df}
