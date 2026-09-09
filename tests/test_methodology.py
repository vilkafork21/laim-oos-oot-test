"""Проверки методики различимости выборок."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('oos_node', ROOT / 'main.py')
NODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE)
from llm_val import valtest_adversarial_test as drift  # noqa: E402
from llm_val.sampler import AutoAsessorSampler  # noqa: E402


def sampler(size):
    data = pd.DataFrame({'question': ['проверка вопрос номер ' + str(i) for i in range(size)],
                         'answer': '', 'target': np.nan})
    return AutoAsessorSampler(data, data)


def test_sample_minimum_and_interval_gate(monkeypatch):
    assert drift.valtest_adversarial_text(sampler(99))['report']['semaphore'] == 'gray'
    values = iter([0., 1.] * 10)
    monkeypatch.setattr(drift, 'get_adversarial_gini_score', lambda *args, **kwargs: next(values))
    result = drift.valtest_adversarial_text(sampler(100))
    assert result['report']['semaphore'] == 'gray'
    assert result['precomputed']['status'] == 'not_computable'
    assert result['precomputed']['gini_ci_upper'] - result['precomputed']['gini_ci_lower'] > .2
    for value, color in [(.399, 'green'), (.4, 'yellow'), (.8, 'red')]:
        report = drift.report_valtest_adversarial_text(value, .01, (value - .01, value + .01))
        assert report['semaphore'] == color


def test_groups_do_not_cross_and_real_gini_uses_sample_labels():
    base = pd.DataFrame({'question': ['оплата перевод карта банк'] * 100,
                         '_group_id': [f'b{i // 2}' for i in range(100)]})
    new = pd.DataFrame({'question': ['погода дождь ветер снег'] * 100,
                        '_group_id': [f'n{i // 2}' for i in range(100)]})
    split = drift.make_adversarial_dataset(base, new)
    assert set(split['train']['_group_id']).isdisjoint(split['test']['_group_id'])
    value = drift.get_adversarial_gini_score(split, catboost_iterations=15)
    assert value > .8


def test_query_frames_do_not_require_quality_labels():
    from laim_monitoring.core import _drift_frame
    data = pd.DataFrame({'query_id': ['1'], 'input_query': ['запрос'], 'output_answer': ['']})
    result = _drift_frame(data, {'assessment_mode': 'qa'}, require_target=False)
    assert len(result) == 1 and result['target'].isna().all()


def test_real_oos_oot_stable_and_shifted_distributions():
    base = pd.DataFrame({'question': ['оплата перевод карта банк'] * 100, 'answer': '', 'target': np.nan})
    for query, expected in [('оплата перевод карта банк', 'green'), ('погода дождь ветер снег', 'red')]:
        monitoring = base.assign(question=query)
        result = drift.valtest_adversarial_text(AutoAsessorSampler(monitoring, base),
                                               catboost_iterations=15)
        assert result['report']['semaphore'] == expected
        assert result['precomputed']['resampling_iterations'] == 20
        assert result['precomputed']['gini_ci_upper'] - result['precomputed']['gini_ci_lower'] <= .2
