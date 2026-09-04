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
from laim_monitoring import prepare_drift_frames, validate_monitoring_metric

# Импортируем вспомогательные функции для HTML отчета
from html_report_helper import display_semaphore, show_criteria_semaphore

logger = logging.getLogger(__name__)


# =============================================================================
# ФУНКЦИИ ФОРМИРОВАНИЯ ОТЧЕТОВ
# =============================================================================

def _table_styles():
    return [
        {"selector": "th", "props": [
            ("background-color", "#f5f5f5"),
            ("text-align", "center"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "td", "props": [
            ("text-align", "left"),
            ("border", "1px solid #ddd"),
            ("padding", "5px"),
        ]},
        {"selector": "", "props": [
            ("border-collapse", "collapse"),
            ("border", "1px solid black"),
        ]},
    ]


def html_report_valtest_oos_oot(res, semaphore_title):
    """
    Функция для сборки html-отчета по тесту OOS-OOT.
    """
    table_styles = _table_styles()

    green_criterion = "Gini < 0.4"
    yellow_criterion = "0.4 ≤ Gini < 0.8"
    red_criterion = "Gini ≥ 0.8"

    criterion_df = show_criteria_semaphore(
        green_criterion, yellow_criterion, red_criterion, table_styles
    )
    criterion_df_html = criterion_df.to_html(border=0, classes="table")

    semaphore_color = res["report"]["semaphore"]
    semaphore_html = display_semaphore(semaphore_color, return_html=True)

    pre = res["precomputed"]
    gini_value = pre.get("gini_value")
    gini_std = pre.get("gini_std")
    ci_low = pre.get("gini_ci_lower")
    ci_high = pre.get("gini_ci_upper")
    n_iter = pre.get("resampling_iterations")
    reason = pre.get("reason")
    normalization = pre.get("input_normalization")

    def _fmt(v, decimals=3):
        if v is None:
            return "n/a"
        try:
            if pd.isna(v):
                return "n/a"
        except (TypeError, ValueError):
            pass
        return f"{float(v):.{decimals}f}"

    indicators = [
        "Значение Gini (среднее)",
        "Стандартное отклонение Gini",
        "95% CI (нижняя)",
        "95% CI (верхняя)",
    ]
    values = [
        _fmt(gini_value),
        _fmt(gini_std),
        _fmt(ci_low),
        _fmt(ci_high),
    ]
    if reason:
        indicators.append("Причина невычислимости")
        values.append(str(reason))
    if normalization:
        indicators.append("Нормализация входа")
        values.append(
            "Удалён sample-exclusive identifier-префикс: "
            f"OOS={normalization['removed_oos']}, "
            f"OOT={normalization['removed_oot']}"
        )
    indicators.append("Результат теста")
    values.append(semaphore_html)
    res_df = pd.DataFrame({"Показатель": indicators, "Значение": values})

    try:
        res_df_to_html = res_df.style.hide().set_table_styles(table_styles)
    except AttributeError:
        res_df_to_html = res_df.style.hide_index().set_table_styles(table_styles)
    res_df_html = res_df_to_html.to_html(border=0, classes="table")

    html_report = f"""
<h2 style="text-align: center;">Разделение выборок OOS-OOT посредством стандартной модели</h2>
<p style="text-align: left;"><b>Цель теста</b></p>
<p style="text-align: left;">Оценить изменение выборки out-of-time по сравнению с валидационной выборкой out-of-sample посредством разделения выборок стандартной моделью.</p>
<p style="text-align: left;"><b>Условия проведения</b></p>
<ul style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li style="text-align: left;">Для СЗ &gt; E</li>
    <li style="text-align: left;">Минимум 50 независимых единиц наблюдения в каждой выборке (иначе серый светофор)</li>
    <li style="text-align: left;">Sample-exclusive низкокардинальный identifier-префикс нейтрализуется и отражается в диагностике</li>
</ul>
<p style="text-align: left;"><b>Алгоритм расчета:</b></p>
<ol style="text-align: left; margin-left: 20px; padding-left: 20px;">
    <li>Составляется набор данных путем склеивания OOS (метка 0) и OOT (метка 1).</li>
    <li>Нейтрализуется доказуемый технический identifier-префикс, если он почти эксклюзивен одной выборке.</li>
    <li>Разбиение 70/30 по независимым dialogue/session/QA-группам без пересечения train/test.</li>
    <li>Обучается CatBoost с текстовыми признаками и balanced class weights, early stopping по eval_set.</li>
    <li>Считается Gini = max(0, 2·AUC − 1) на test.</li>
    <li>Шаги 2–4 повторяются с независимыми seed'ами {n_iter if n_iter else 'несколько'} раз; в отчёт идут среднее, std и 95% CI среднего Gini.</li>
</ol>
<p style="text-align: left;"><b>Критерии выставления светофора</b></p>
<div style="text-align: left; width: 100%;">{criterion_df_html}</div><br>
<p style="text-align: left;"><b>Результаты теста</b></p>
<div style="text-align: left; width: 100%;">{res_df_html}</div><br>
"""
    return html_report


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
            # Серый — всегда not_computable: пара gray/computed для агрегатора
            # противоречие, а информационный режим не даёт вердикта.
            "status": "not_computable" if platform_color == "gray" else pre.get("status", "computed"),
            "reason_code": pre.get("reason_code") or ("informational" if platform_color == "gray" else None),
            "reason": pre.get("reason") or ("информационный режим: вердикт не выставляется"
                                            if platform_color == "gray" else None),
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
    reference_stable_umr: pd.DataFrame | None = None,
    data_types: tuple = ("train", "test"),
    yellow_threshold: float = 0.4,
    red_threshold: float = 0.8,
    min_groups_per_side: int = 50,
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
    contract = validate_monitoring_metric(monitoring_metric, require_computed=False)
    if contract["status"] == "not_computable":
        logger.warning(
            "Тест OOS-OOT не вычисляется: %s",
            contract.get("reason", "причина не указана"),
        )
        report_result = report_valtest_oos_oot(
            {
                "report": {"semaphore": "gray"},
                "precomputed": {
                    "status": "not_computable",
                    "reason_code": contract.get("reason_code"),
                    "reason": contract.get("reason"),
                },
            },
            "Результат теста разделения выборок не может быть оценен",
        )
        report_result["all_results"].update(
            reason_code=contract.get("reason_code"),
            reason=contract.get("reason"),
            test_name="oos_oot",
        )
        return {
            "all_results": report_result["all_results"],
            "test_description": report_result["hidden_port"],
        }

    # Защитный literal_eval — поддерживает и tuple, и строку из UI (P1-6 из global, аналогично)
    if isinstance(data_types, str):
        data_types = literal_eval(data_types)

    # Корректный (возрастающий) кортеж порогов (P0-2)
    semaphore_threshold = (
        min(yellow_threshold, red_threshold),
        max(yellow_threshold, red_threshold),
    )
    # Карточка 6.3.6: сравнивать поток нужно с эталонной выборкой стабильного
    # периода ПРОМ; корзина первичной валидации — переходное положение, и тогда
    # результат информативен (устойчивая разделимость отражает способ
    # формирования корзины, а не изменение потока).
    reference_source = "validation_basket"
    if reference_stable_umr is not None and len(reference_stable_umr) > 0:
        reference_umr = reference_stable_umr
        reference_source = "stable_period"
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
    logger.info("Тест OOS-OOT запущен")
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
        min_groups_per_side=min_groups_per_side,
    )
    logger.info(res)

    semaphore_color = res["report"]["semaphore"]

    semaphore_title = {
        "green": "Результат теста разделения выборок соответствует зеленому светофору",
        "yellow": "Результат теста разделения выборок соответствует желтому светофору",
        "red": "Результат теста разделения выборок соответствует красному светофору",
        "gray": "Результат теста разделения выборок не может быть оценен",
    }[semaphore_color]

    report_result = report_valtest_oos_oot(res, semaphore_title)
    report_result["all_results"].update(
        reference_source=reference_source,
        informative=reference_source == "validation_basket",
        verdict_statistic="gini_ci_lower",
        min_groups_per_side=min_groups_per_side,
    )
    report_result["all_results"]["test_name"] = "oos_oot"

    return {
        "all_results": report_result["all_results"],
        "test_description": report_result["hidden_port"],
    }
