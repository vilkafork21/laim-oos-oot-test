"""
Report helper functions for OOS-OOT stability test.

Модуль содержит функции для формирования светофоров (traffic lights)
на основе пороговых значений метрик. Используется для визуализации
результатов тестов в виде цветовой индикации.
"""

import typing as tp

import numpy as np
import pandas as pd


# =============================================================================
# ОСНОВНЫЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def semaphore_by_threshold(
    value: float,
    threshold: tp.Tuple[float, float],
    greater_is_better: bool = True,
    left_border_is_bigger: bool = True,
    right_border_is_bigger: bool = True,
) -> str:
    """
    Расчет цвета светофора в зависимости от границ.

    Args:
        value: Значение метрики
        threshold: Кортеж с порогами (красный, зеленый)
        greater_is_better: True, если чем больше метрика, тем лучше
        left_border_is_bigger: При равенстве левому порогу - зеленый
        right_border_is_bigger: При равенстве правому порогу - зеленый

    Returns:
        Цвет светофора: 'red', 'yellow' или 'green'
    """
    semaphore = {0: "red", 1: "yellow", 2: "green"}
    if (value < threshold[0]) or (not left_border_is_bigger and value == threshold[0]):
        interval = 0
    elif (value < threshold[1]) or (
        not right_border_is_bigger and value == threshold[1]
    ):
        interval = 1
    else:
        interval = 2
    if not greater_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]


def semaphore_by_threshold_without_yellow(
    value: float,
    threshold: float,
    greater_is_better: bool = True,
    border_is_bigger: bool = True,
) -> str:
    """
    Расчет цвета светофора в зависимости от границ (без желтого).

    Args:
        value: Значение метрики
        threshold: Граница светофора
        greater_is_better: True, если чем больше метрика, тем лучше
        border_is_bigger: При равенстве границе - зеленый

    Returns:
        Цвет светофора: 'red' или 'green'
    """
    semaphore = {0: "red", 1: "green"}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


def semaphore_by_threshold_without_red(
    value: float,
    threshold: float,
    greater_is_better: bool = True,
    border_is_bigger: bool = True,
) -> str:
    """
    Расчет цвета светофора в зависимости от границ (без красного).

    Args:
        value: Значение метрики
        threshold: Граница светофора
        greater_is_better: True, если чем больше метрика, тем лучше
        border_is_bigger: При равенстве границе - зеленый

    Returns:
        Цвет светофора: 'yellow' или 'green'
    """
    semaphore = {0: "yellow", 1: "green"}
    if (value < threshold) or (not border_is_bigger and value == threshold):
        interval = 0
    else:
        interval = 1
    if not greater_is_better:
        interval = abs(interval - 1)
    return semaphore[interval]


# =============================================================================
# АГРЕГИРУЮЩИЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def worst_semaphore(semaphore_list: tp.List[str]) -> str:
    """
    Расчет худшего светофора из списка.

    Args:
        semaphore_list: Список со светофорами

    Returns:
        Цвет худшего светофора
    """
    if (semaphore_list is None) or (len(semaphore_list) == 0):
        return "gray"
    semaphore_to_value = {"red": 0, "yellow": 1, "green": 2, "gray": 3}
    value_to_semaphore = {value: key for key, value in semaphore_to_value.items()}
    worst_value = min([semaphore_to_value[i] for i in semaphore_list])
    return value_to_semaphore[worst_value]


def tricky_semaphore(
    value: float,
    semaphore_list: tp.List[str],
    greater_is_better: bool = True,
    threshold_metric: tp.Tuple[float, float] = (0.4, 0.6),
    red_freq_threshold: float = 0.2,
    yellow_freq_threshold: float = 0.2,
) -> str:
    """
    Хитрый светофор - комбинирует значение метрики и частоты цветов.

    Args:
        value: Ключевая метрика качества
        semaphore_list: Список светофоров за критерии асессоров
        greater_is_better: Чем больше агрегированная метрика, тем лучше?
        threshold_metric: Границы светофора для агрегированного значения
        red_freq_threshold: Граница по относительному числу "красных" светофоров
        yellow_freq_threshold: Граница по относительному числу "желтых" светофоров

    Returns:
        Цвет светофора
    """
    color = "yellow"
    value_color = semaphore_by_threshold(value, threshold_metric, greater_is_better)
    freq_color = (
        pd.Series(semaphore_list, dtype="str").value_counts(normalize=True).to_dict()
    )
    if (value_color == "red") or (freq_color.get("red", 0) > red_freq_threshold):
        color = "red"
    elif (
        (value_color == "green")
        and (freq_color.get("yellow", 0) < yellow_freq_threshold)
        and (freq_color.get("red", 0) <= 0)
    ):
        color = "green"
    return color


# =============================================================================
# КВАНТИЛЬНЫЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def quantile_semaphore(
    value: float, threshold: tp.Tuple[float, float], inner_is_better: bool = True
) -> str:
    """
    Расчет цвета светофора в зависимости от квантиля при двусторонних границах.

    Args:
        value: Значение
        threshold: Границы светофора
        inner_is_better: True, если светофор лучше при попадании внутрь границ

    Returns:
        Цвет светофора
    """
    semaphore = {0: "red", 1: "yellow", 2: "green"}
    if (1 - threshold[0]) / 2 < value < (1 + threshold[0]) / 2:
        interval = 2
    elif (1 - threshold[1]) / 2 <= value <= (1 + threshold[1]) / 2:
        interval = 1
    else:
        interval = 0
    if not inner_is_better:
        interval = abs(interval - 2)
    return semaphore[interval]


def shuffle_ci_semaphore(
    mean_metric: float,
    std_metric: float,
    interval: tp.Tuple[float, float],
    std_threshold: float,
) -> str:
    """
    Расчет цвета светофора для теста с перемешиванием вариантов.

    Args:
        mean_metric: Среднее значение целевой метрики
        std_metric: Стандартное отклонение целевой метрики
        interval: Границы доверительного интервала
        std_threshold: Порог для светофора по стандартному отклонению

    Returns:
        Цвет светофора
    """
    if (mean_metric < interval[0]) or (mean_metric > interval[1]):
        return "red"
    if std_metric > std_threshold:
        return "yellow"
    return "green"


def quantile_semaphore_without_yellow(
    value: float, threshold: tp.Tuple[float, float], inner_is_better: bool = True
) -> str:
    """
    Расчет цвета светофора в зависимости от квантиля (без желтого).

    Args:
        value: Значение метрики
        threshold: Границы светофора
        inner_is_better: True, если светофор лучше при попадании внутрь границ

    Returns:
        Цвет светофора: 'red' или 'green'
    """
    if inner_is_better:
        if threshold[0] <= value <= threshold[1]:
            return "green"
        else:
            return "red"
    else:
        if value < threshold[0] or value > threshold[1]:
            return "green"
        else:
            return "red"


# =============================================================================
# ЧАСТОТНЫЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def local_frequency_semaphore(
    value: float, ideal_responses: float, threshold: tp.Tuple[float, float]
) -> str:
    """
    Расчет цвета светофора в зависимости от частоты элемента.

    Args:
        value: Элемент массива частот
        ideal_responses: Математическое ожидание частот
        threshold: Границы светофора

    Returns:
        Цвет светофора
    """
    color = "red"
    yellow_bound, red_bound = threshold[0], threshold[1]
    if (value >= red_bound * ideal_responses) or (value <= ideal_responses / red_bound):
        color = "red"
    elif (value >= yellow_bound * ideal_responses) or (
        value <= ideal_responses / yellow_bound
    ):
        color = "yellow"
    else:
        color = "green"
    return color


def frequency_semaphore(
    values: tp.List[float], threshold: tp.Tuple[float, float]
) -> tp.List[str]:
    """
    Расчет цветов светофора для массива частот.

    Args:
        values: Массив частот элементов
        threshold: Границы светофора

    Returns:
        Список цветов светофора для каждой частоты
    """
    semaphore_list, color = [], "red"
    ideal_responses = np.mean(values)
    for value in values:
        color = local_frequency_semaphore(
            value=value, ideal_responses=ideal_responses, threshold=threshold
        )
        semaphore_list.append(color)
    return semaphore_list


# =============================================================================
# ПРОПОРЦИОНАЛЬНЫЕ ФУНКЦИИ СВЕТОФОРА
# =============================================================================

def proportion_semaphore(
    semaphores: tp.Union[pd.Series, tp.List[str]],
    freq_threshold: tp.Tuple[float, float],
) -> str:
    """
    Расчет итогового светофора в зависимости от доли цветов.

    Args:
        semaphores: Список светофоров для каждого элемента выборки
        freq_threshold: Доля допустимых желтых и красных светофоров

    Returns:
        Цвет светофора
    """
    semaphores_series = pd.Series(semaphores)
    yellows = (semaphores_series == "yellow").sum() / len(semaphores_series)
    reds = (semaphores_series == "red").sum() / len(semaphores_series)
    if reds > freq_threshold[1]:
        color = "red"
    elif yellows + reds > freq_threshold[0]:
        color = "yellow"
    else:
        color = "green"
    return color


def quanity_semaphore(
    semaphore_list: tp.List[str], threshold: tp.Tuple[int, int]
) -> str:
    """
    Расчет итогового светофора в зависимости от количества вхождений.

    Args:
        semaphore_list: Массив с цветами частных светофоров
        threshold: Границы светофора

    Returns:
        Цвет светофора
    """
    yellow_count = sum(1 for color in semaphore_list if color == "yellow")
    red_count = sum(1 for color in semaphore_list if color == "red")
    if red_count >= threshold[0]:
        overall_semaphore = "red"
    elif yellow_count >= threshold[1]:
        overall_semaphore = "yellow"
    else:
        overall_semaphore = "green"
    return overall_semaphore