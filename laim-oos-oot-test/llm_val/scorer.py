"""
Scorer classes for OOS-OOT stability test.

Модуль содержит классы для подсчета метрик качества.
Скореры вычисляют агрегированные метрики по данным из семплеров.
"""

import typing as tp
from abc import ABC, abstractmethod

from llm_val.sampler import Sampler


# =============================================================================
# БАЗОВЫЙ КЛАСС
# =============================================================================

class Scorer(ABC):
    """
    Абстрактный класс скорера.

    Все скореры должны наследоваться от данного класса и
    реализовывать метод calc() для подсчета метрик.
    """

    def __init__(self, metrics: tp.Dict[str, tp.Any]):
        """
        Инициализация скорера.

        Args:
            metrics: Словарь с определениями метрик
        """
        self.metrics = metrics

    @abstractmethod
    def calc(
        self, sampler: Sampler, data_type: str, metric_name: str
    ) -> tp.Dict[str, float]:
        """
        Метод подсчета метрик.

        Args:
            sampler: Объект семплера с данными
            data_type: Тип данных (train, test, val, oot)
            metric_name: Название метрики

        Returns:
            Словарь с вычисленными метриками
        """
        raise NotImplementedError(f"Определите calc в {self.__class__.__name__}")


# =============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ СКОРЕРЫ
# =============================================================================

class AutoAsessorScorer(Scorer):
    """
    Скорер для данных автоасессора.

    Считает метрики по столбцу 'target'. Ключевая метрика — среднее значение target.
    """

    def calc(
        self, sampler: Sampler, data_type: str, metric_name: str | None = None
    ) -> tp.Dict[str, float]:
        """
        Подсчёт метрик для указанного типа данных.

        Args:
            sampler: Объект семплера с данными
            data_type: Тип данных (train, test, val, oot)
            metric_name: Название метрики (опционально)

        Returns:
            Словарь с вычисленными метриками
        """
        metrics = self.metrics
        data = getattr(sampler, data_type)["y"]
        if metric_name is not None:
            metrics = {metric_name: self.metrics[metric_name]}
        metric_res = {}

        for func_name, func_dict in metrics.items():
            if func_dict["is_singlecol"]:
                for name_col in data:
                    name = f"{name_col}"
                    metric_res[name] = func_dict["call"](data[name_col])
            else:
                metric_res[func_name] = func_dict["call"](data)

        return metric_res