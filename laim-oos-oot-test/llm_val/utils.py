"""
Utility functions for OOS-OOT stability test.
"""

import numpy as np


# Единое место для определения METRICS (P3-5).
# main.py импортирует отсюда.
METRICS = {
    "single_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": True},
    "multicol_mean": {"call": lambda x: float(np.mean(x.values)), "is_singlecol": False},
}


def string_to_float(value):
    """
    Преобразование строки в число с обработкой ошибок.

    Возвращает None при невозможности приведения (P3-2: ловим конкретные исключения,
    а не голый except).
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
