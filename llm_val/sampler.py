"""
Sampler classes for OOS-OOT stability test.

Модуль содержит классы для семплирования данных из DataFrame
и преобразования их в формат, пригодный для теста на дрифт.
"""

import logging
import typing as tp
from ast import literal_eval
from copy import deepcopy

import pandas as pd
from llm_val.utils import string_to_float

logger = logging.getLogger(__name__)


# =============================================================================
# БАЗОВЫЙ КЛАСС
# =============================================================================

class Sampler:
    """
    Простейший («базовый») семплер, который просто хранит переданные в него данные.
    Является абстрактной основой для специализированных семплеров.

    Слоты данных: train, val, test, oot. Для тестов дрифта OOS храним в `train`,
    OOT — в `test` (исторически). Алиасы oos/oot предоставлены для семантической
    ясности (P2-5).
    """

    def __init__(
        self,
        train: tp.Optional[tp.Any] = None,
        val: tp.Optional[tp.Any] = None,
        test: tp.Optional[tp.Any] = None,
        oot: tp.Optional[tp.Any] = None,
    ):
        self.train = train
        self.val = val
        self.test = test
        self.oot = oot

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, var):
        self._check_var(var, "train")
        self._train = deepcopy(var)

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, var):
        self._check_var(var, "val")
        self._val = deepcopy(var)

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, var):
        self._check_var(var, "test")
        self._test = deepcopy(var)

    @property
    def oot(self):
        return self._oot

    @oot.setter
    def oot(self, var):
        self._check_var(var, "oot")
        self._oot = deepcopy(var)

    # Семантические алиасы (P2-5)
    @property
    def oos(self):
        return self._train

    @oos.setter
    def oos(self, var):
        self.train = var

    def _check_var(self, var, var_name):
        if var is not None:
            if not isinstance(var, dict):
                raise AttributeError(f"{var_name} должен быть словарем")
            elif not (set(var.keys()) == {"X", "y"}):
                raise AttributeError(f"{var_name} должен содержать 2 ключа: X и y")


# =============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ СЕМПЛЕРЫ
# =============================================================================

class AutoAsessorSampler(Sampler):
    """
    Семплер для работы с данными автоасессора.

    Преобразует входной DataFrame в формат, пригодный для теста.
    Поддерживает два формата входных данных:
      1. Столбцы: 'history', 'target' - история чата
      2. Столбцы: 'question', 'answer', 'target' - готовые вопрос-ответ пары
    """

    def __init__(self, agent_df: pd.DataFrame, real_df: pd.DataFrame):
        super().__init__()

        def process_df(df, df_name):
            df = df.copy()
            df["target"] = df["target"].apply(string_to_float)
            if "question" in df.columns and "answer" in df.columns:
                df = df[["question", "answer", "target"]].copy()
            elif "history" in df.columns:
                df = self._convert_history_to_qa(df, df_name=df_name)
            else:
                raise ValueError(
                    "DataFrame должен содержать либо 'history', либо 'question' и 'answer'"
                )
            return df

        agent_df = process_df(agent_df, "agent")
        real_df = process_df(real_df, "real")
        logger.info(f"Agent columns: {list(agent_df.columns)}, размер: {len(agent_df)}")
        logger.info(f"Real columns: {list(real_df.columns)}, размер: {len(real_df)}")

        train_data = {
            "X": real_df[["question", "answer"]].reset_index(drop=True),
            "y": real_df[["target"]].reset_index(drop=True),
        }
        test_data = {
            "X": agent_df[["question", "answer"]].reset_index(drop=True),
            "y": agent_df[["target"]].reset_index(drop=True),
        }

        self.train = train_data
        self.test = test_data

    def _convert_history_to_qa(
        self, df: pd.DataFrame, df_name: str = ""
    ) -> pd.DataFrame:
        """
        Преобразует столбец 'history' в 'question' и 'answer'.
        Берет первое сообщение пользователя как вопрос,
        последнее сообщение ассистента как ответ.

        P3-3: логирует количество пропущенных строк с битым 'history'.
        """
        records = []
        skipped = 0
        for _, row in df.iterrows():
            try:
                history = literal_eval(row["history"])
            except (ValueError, SyntaxError, TypeError):
                skipped += 1
                continue

            target = row["target"]
            question = None
            answer = None
            for msg in history:
                message_type = msg.get("type", msg.get("role", None))
                if message_type in ("Пользователь", "human") and question is None:
                    question = msg.get("content", "")
                elif message_type in ("AI ассистент", "ai"):
                    answer = msg.get("content", "")  # перезаписываем — нужен последний

            records.append(
                {"question": question or "", "answer": answer or "", "target": target}
            )

        if skipped:
            logger.warning(
                f"[{df_name}] пропущено {skipped} строк из-за неразобранного 'history'"
            )
        return pd.DataFrame(records)
