"""Legacy-port adapter for the OOS-OOT adversarial test."""

import logging
from ast import literal_eval
from html import escape

import pandas as pd

from llm_val.sampler import AutoAsessorSampler
from llm_val.valtest_adversarial_test import valtest_adversarial_text


_QUESTION_COLUMNS = ("question", "input_query", "agent_input_query")
_ANSWER_COLUMNS = ("answer", "output_answer", "agent_output_answer")
_GROUP_COLUMNS = ("reference_group_id", "assessment_unit_id", "session_id")
_PLATFORM_COLOR = {"yellow": "amber", "grey": "gray"}


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...], role: str) -> str:
    by_name = {str(column).casefold(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    raise ValueError(
        f"Не найдена колонка {role}; поддерживаются контрактные имена {candidates}"
    )


def _prepare_frame(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{sample_name} должен быть pandas.DataFrame")
    question = _first_column(frame, _QUESTION_COLUMNS, "вопроса")
    answer = _first_column(frame, _ANSWER_COLUMNS, "ответа")
    group = next((name for name in _GROUP_COLUMNS if name in frame), None)
    result = pd.DataFrame({
        "question": frame[question],
        "answer": frame[answer].fillna(""),
        # Marks не являются target adversarial-задачи: label 0/1 ставит сам тест.
        "target": 0.0,
    })
    if group:
        result["_group_id"] = frame[group]
    before = len(result)
    result = result.dropna(subset=["question"]).reset_index(drop=True)
    logging.info("%s: %s -> %s строк", sample_name, before, len(result))
    return result


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else round(result, 6)


def _html(precomputed: dict, color: str) -> str:
    values = [
        ("Gini (среднее)", _number(precomputed.get("gini_value"))),
        ("Стандартное отклонение", _number(precomputed.get("gini_std"))),
        ("P05 по разбиениям", _number(precomputed.get("gini_spread_lower"))),
        ("P95 по разбиениям", _number(precomputed.get("gini_spread_upper"))),
        ("Результат", color),
    ]
    normalization = precomputed.get("input_normalization")
    if normalization:
        values.insert(-1, (
            "Нормализация входа",
            "Удалён sample-exclusive identifier-префикс: "
            f"OOS={normalization['removed_oos']}, "
            f"OOT={normalization['removed_oot']}",
        ))
    if precomputed.get("reason"):
        values.insert(-1, ("Причина", precomputed["reason"]))
    rows = "".join(
        f"<tr><td>{escape(str(name))}</td><td>{escape(str(value))}</td></tr>"
        for name, value in values
    )
    iterations = precomputed.get("resampling_iterations") or "несколько"
    return f"""
<h2>Разделение выборок OOS-OOT посредством стандартной модели</h2>
<p>CatBoost отличает OOS (label 0) от OOT (label 1) только по question;
assessor marks в признаки не входят.</p>
<p>Sample-exclusive низкокардинальный identifier-префикс нейтрализуется.
Train/test делятся 70/30 по независимым dialogue/session/QA-группам.
Gini = max(0, 2*AUC - 1), расчёт повторяется {iterations} раз.</p>
<p>Границы: green &lt; 0.4; yellow 0.4–0.8; red ≥ 0.8.</p>
<table>{rows}</table>
"""


def main(
    real_asessor_df: pd.DataFrame,
    auto_asessor_df: pd.DataFrame,
    metric_selector_res: dict,
    metric_agg: str = "single_mean",
    data_types: tuple = ("train", "test"),
    yellow_threshold: float = 0.4,
    red_threshold: float = 0.8,
    is_info: bool = False,
    resampling_iterations: int = 20,
    test_size: float = 0.3,
    catboost_iterations: int = 300,
    catboost_learning_rate: float = 0.1,
    catboost_depth: int = 4,
    catboost_early_stopping_rounds: int = 30,
    random_state: int = 17,
):
    """Run OOS-OOT with the legacy workflow port names."""
    del metric_selector_res, metric_agg
    if isinstance(data_types, str):
        data_types = literal_eval(data_types)
    reference = _prepare_frame(real_asessor_df, "OOS")
    monitoring = _prepare_frame(auto_asessor_df, "OOT")
    sampler = AutoAsessorSampler(agent_df=monitoring, real_df=reference)
    if "_group_id" in reference:
        sampler.train["X"]["_group_id"] = reference["_group_id"]
    if "_group_id" in monitoring:
        sampler.test["X"]["_group_id"] = monitoring["_group_id"]

    result = valtest_adversarial_text(
        sampler=sampler,
        semaphore_threshold=(
            min(yellow_threshold, red_threshold),
            max(yellow_threshold, red_threshold),
        ),
        data_types=data_types,
        resampling_iterations=resampling_iterations,
        test_size=test_size,
        catboost_iterations=catboost_iterations,
        catboost_learning_rate=catboost_learning_rate,
        catboost_depth=catboost_depth,
        catboost_early_stopping_rounds=catboost_early_stopping_rounds,
        random_state=random_state,
        is_info=is_info,
    )
    color = result["report"]["semaphore"]
    platform_color = _PLATFORM_COLOR.get(color, color)
    precomputed = result["precomputed"]
    title = {
        "green": "Результат теста соответствует зеленому светофору",
        "yellow": "Результат теста соответствует желтому светофору",
        "red": "Результат теста соответствует красному светофору",
        "gray": "Результат теста не может быть оценен",
    }[color]
    all_results = {
        "calculated_traffic_lights": {
            "test_light": platform_color,
            "semaphore_title": title,
        },
        "color": platform_color,
        "test_name": "oos_oot",
        "status": precomputed.get("status"),
        "reason": precomputed.get("reason"),
        "gini_mean": _number(precomputed.get("gini_value")),
        "gini_std": _number(precomputed.get("gini_std")),
        "gini_spread_lower": _number(precomputed.get("gini_spread_lower")),
        "gini_spread_upper": _number(precomputed.get("gini_spread_upper")),
        "n_oos_groups": precomputed.get("n_oos_groups"),
        "n_oot_groups": precomputed.get("n_oot_groups"),
        "input_normalization": precomputed.get("input_normalization"),
    }
    return {
        "all_results": all_results,
        "test_description": _html(precomputed, platform_color),
    }
