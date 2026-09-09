"""
Main module for OOS-OOT stability test.

Этот модуль является точкой входа для теста на стабильность данных.
Тест оценивает дрифт между out-of-sample (OOS) и out-of-time (OOT) выборками
посредством обучения классификатора, пытающегося различить эти выборки.
"""

import logging
from ast import literal_eval

import pandas as pd

from llm_val.sampler import AutoAsessorSampler
from llm_val.valtest_adversarial_test import valtest_adversarial_text
from laim_monitoring import prepare_drift_frames

# Импортируем вспомогательные функции для HTML отчета
from html_report import format_report_number, render_test_report


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================




def html_report_valtest_oos_oot(res: dict, semaphore_title: str) -> str:
    pre = res["precomputed"]
    normalization = pre.get("input_normalization")
    normalization_text = "Не применялась" if "input_normalization" in pre else "Не передана в результат теста"
    if normalization:
        normalization_text = (
            "Удалён служебный префикс: "
            f"OOS — {normalization.get('removed_oos', 0)}, OOT — {normalization.get('removed_oot', 0)}"
        )
    bounds = [format_report_number(pre.get(key)) for key in ("gini_ci_lower", "gini_ci_upper")]
    interval_text = "Не рассчитано" if "Не рассчитано" in bounds else f"[{bounds[0]}; {bounds[1]}]"
    rows = [
        ("Объём OOS / OOT (независимых единиц)",
         f"{format_report_number(pre.get('n_oos_groups'), 0)} / {format_report_number(pre.get('n_oot_groups'), 0)}"),
        ("Gini — различимость выборок (среднее)", format_report_number(pre.get("gini_value"))),
        ("Число повторных разбиений", format_report_number(pre.get("resampling_iterations"), 0)),
        ("Стандартное отклонение Gini", format_report_number(pre.get("gini_std"))),
        ("95 % доверительный интервал среднего Gini",
         interval_text),
        ("Нормализация служебного префикса", normalization_text),
    ]
    return render_test_report(
        "6.3.7", "Различимость выборок OOS–OOT",
        "Оценить, насколько запросы пользователей за отчётный период (OOT) отличаются "
        "от запросов эталонной корзины (OOS), с помощью модели классификации.",
        rows, res["report"]["semaphore"],
        "Чем выше Gini, тем легче отличить запросы текущего потока от эталона. "
        "Зелёный результат означает слабую различимость, жёлтый — умеренный сдвиг, "
        "красный — выраженные различия. При сдвиге рекомендуется разобрать новые тематики "
        "и пополнить эталонную корзину. Тест не измеряет качество ответов решения. "
        "При сером результате вывод о различимости не получен.",
        "По методике: СЗ выше E и не менее 100 независимых единиц в каждой выборке. "
        "Диалоги и сессии "
        "не пересекаются между обучающей и тестовой частями.",
        "Пороги по умолчанию: зелёный — Gini < 0,4; жёлтый — 0,4 ≤ Gini < 0,8; "
        "красный — Gini ≥ 0,8. Серый: недостаточно независимых единиц, "
        "значение не рассчитано, ширина доверительного интервала больше 0,2 или информационный режим.",
        reason=pre.get("reason") or ("Оценка недоступна или выбран информационный режим."
                                    if res["report"]["semaphore"] in ("gray", "grey") else ""),
    )


# Цвет, отдаваемый ПЛАТФОРМЕ и АГРЕГАТОРУ, должен быть в их словаре
# (red/amber/green/gray). Внутри теста используется "yellow"/"grey" —
# нормализуем на границе вывода, иначе светофор на узле не отрисуется,
# а agg-master не засчитает жёлтый (он считает color == "amber").
_PLATFORM_COLOR = {"yellow": "amber", "grey": "gray"}


def report_valtest_oos_oot(res, semaphore_title):
    """
    Создание report'а для теста OOS-OOT в формате llm_val.
    """
    semaphore_color = res["report"]["semaphore"]
    platform_color = _PLATFORM_COLOR.get(semaphore_color, semaphore_color)
    html_report = html_report_valtest_oos_oot(res, semaphore_title)
    pre = res.get("precomputed", {})

    def _num(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return None if pd.isna(value) else round(value, 6)

    return {
        "all_results": {
            "calculated_traffic_lights": {
                "test_light": platform_color,
                "semaphore_title": semaphore_title,
            },
            "color": platform_color,
            "status": pre.get(
                "status", "not_computable" if platform_color == "gray" else "computed"
            ),
            "reason_code": pre.get("reason_code"),
            "reason": pre.get("reason"),
            "gini_mean": _num(pre.get("gini_value")),
            "gini_std": _num(pre.get("gini_std")),
            "gini_ci_lower": _num(pre.get("gini_ci_lower")),
            "gini_ci_upper": _num(pre.get("gini_ci_upper")),
            "n_oos": pre.get("n_oos"),
            "n_oot": pre.get("n_oot"),
            "n_oos_groups": pre.get("n_oos_groups"),
            "n_oot_groups": pre.get("n_oot_groups"),
            "input_normalization": pre.get("input_normalization"),
        },
        "hidden_port": html_report,
    }


# =============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# =============================================================================

# Defaults унифицированы с README/HTML (P0-2):
# yellow_low = 0.4, red_low = 0.8 → пороговый кортеж (0.4, 0.8) возрастает.
def main(
    reference_umr: pd.DataFrame,
    monitoring_umr: pd.DataFrame,
    monitoring_metric: dict,
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
    """
    Основная функция запуска adversarial-теста OOS↔OOT.

    Изменения относительно baseline:
    - Defaults порогов согласованы с README/HTML (P0-2)
    - random_state выведен в UI (P1-7)
    - dropna только по ключевым колонкам (P1-8)
    - Гиперпараметры CatBoost выведены в UI (P1-1)
    - Убраны неиспользуемые параметры (P2-6)
    """
    # Защитный literal_eval — поддерживает и tuple, и строку из UI (P1-6 из global, аналогично)
    if isinstance(data_types, str):
        data_types = literal_eval(data_types)

    # Корректный (возрастающий) кортеж порогов (P0-2)
    semaphore_threshold = (
        min(yellow_threshold, red_threshold),
        max(yellow_threshold, red_threshold),
    )
    reference_frame, monitoring_frame = prepare_drift_frames(
        reference_umr, monitoring_umr, monitoring_metric
    )

    sampler = AutoAsessorSampler(agent_df=monitoring_frame, real_df=reference_frame)
    sampler.train["X"]["_group_id"] = reference_frame[
        "reference_group_id"
    ].reset_index(drop=True)
    sampler.test["X"]["_group_id"] = monitoring_frame[
        "reference_group_id"
    ].reset_index(drop=True)
    logging.info("Тест OOS-OOT запущен")
    res = valtest_adversarial_text(
        sampler=sampler,
        semaphore_threshold=semaphore_threshold,
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
    logging.info(res)

    semaphore_color = res["report"]["semaphore"]

    semaphore_title = {
        "green": "Результат теста разделения выборок соответствует зеленому светофору",
        "yellow": "Результат теста разделения выборок соответствует желтому светофору",
        "red": "Результат теста разделения выборок соответствует красному светофору",
        "gray": "Результат теста разделения выборок не может быть оценен",
    }[semaphore_color]

    report_result = report_valtest_oos_oot(res, semaphore_title)
    report_result["all_results"]["test_name"] = "oos_oot"

    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
