"""Reference UMR в формате тестового датасета: packed dialogue и flat с session_id."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from laim_monitoring import MonitoringContractError, unitize


def _contract(mode: str) -> dict:
    return {
        "contract_version": "laim-monitoring-metric.v2", "umr_version": "laim-umr.v2",
        "status": "computed", "basket_id": "CI1", "name": "quality", "score_column": "main_metric",
        "assessment_mode": mode,
        "scoring": {
            "method": "identity",
            "sources": [{
                "source_id": "source_1", "column_name": "score_metric", "role": "final_score",
                "normalization": "numeric", "polarity": "direct",
            }],
            "missing_policy": "fail", "majority_denominator": None,
        },
        "aggregation": {"method": "mean", "weight_column": None},
        "baseline": {
            "value": 0.5, "scale": "ratio", "value_source": "validation_report",
            "reported_value": 0.5, "reported_scale": "ratio", "recomputed_value": 0.5,
            "reconciliation": "match",
        },
        "primary_validation": {
            "threshold": None, "comparator": None, "scale": "ratio", "verdict": None,
            "affects_monitoring": False,
        },
        "evidence": {},
    }


def test_packed_dialogue_reference_is_unitized_per_session():
    frame = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": ["[('q1', 'hi', 'hello'), ('q2', 'bye', 'see you')]", "[('q3', 'x', 'y')]"],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    units = unitize(frame, _contract("dialogue"))
    assert len(units) == 2
    assert [turn["input_query"] for turn in units["dialogue"].iloc[0]] == ["hi", "bye"]
    assert units["source_1"].tolist() == [1.0, 0.0]
    assert units["main_metric"].tolist() == [1.0, 0.0]


def test_flat_reference_with_session_id_keeps_turn_history():
    frame = pd.DataFrame({
        "session_id": ["s1", "s1", "s2"],
        "query_id": ["q1", "q2", "q3"],
        "input_query_count": [1, 1, 1],
        "input_query": ["hi", "bye", "x"],
        "output_answer": ["hello", "see you", "y"],
        "score_metric": [1.0, 0.0, 1.0],
        "main_metric": [1.0, 0.0, 1.0],
    })
    units = unitize(frame, _contract("turn_with_history"))
    assert len(units) == 3
    assert [turn["input_query"] for turn in units["assessment_context"].iloc[1]["history"]] == ["hi"]


def test_flat_reference_without_canonical_columns_is_rejected():
    with pytest.raises(MonitoringContractError):
        unitize(pd.DataFrame({"question": ["q"], "answer": ["a"]}), _contract("qa"))


def test_drift_frames_from_packed_reference_and_packed_monitoring():
    """Формы выходов adapter (packed) и TDC (packed) согласуются в drift."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("dialogue")
    reference = pd.DataFrame({
        "session_id": ["s1", "s2"],
        "dialogue": [
            "[('t1', 'вопрос один', 'ответ один'), ('t2', 'вопрос два', 'ответ два')]",
            "[('t3', 'вопрос три', 'ответ три')]",
        ],
        "input_query_count": [1, 1],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "scenario": ["a", "b"],
        "session_id": ["m1", "m2"],
        "dialogue": [
            "[('mt1', 'наблюдённый вопрос', 'наблюдённый ответ')]",
            "[('mt2', 'ещё вопрос', 'ещё ответ'), ('mt3', 'и ещё', 'и ответ')]",
        ],
        "input_query_count": [1, 1],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2  # единица drift — диалог
    assert len(mon_frame) == 2
    assert ref_frame["target"].tolist() == [1.0, 0.0]
    assert "вопрос один" in ref_frame["question"].iloc[0]


def test_drift_frames_from_flat_monitoring_with_session_id():
    """qa/turn_with_history: flat monitoring TDC без служебных колонок."""
    from laim_monitoring import prepare_drift_frames

    contract = _contract("turn_with_history")
    reference = pd.DataFrame({
        "session_id": ["s1", "s1"],
        "query_id": ["q1", "q2"],
        "input_query_count": [1, 1],
        "input_query": ["в1", "в2"],
        "output_answer": ["о1", "о2"],
        "score_metric": [1.0, 0.0],
        "main_metric": [1.0, 0.0],
    })
    monitoring = pd.DataFrame({
        "scenario": ["r", "r"],
        "session_id": ["m1", "m1"],
        "query_id": ["mq1", "mq2"],
        "input_query_count": [1, 1],
        "input_query": ["нв1", "нв2"],
        "output_answer": ["но1", "но2"],
    })

    ref_frame, mon_frame = prepare_drift_frames(reference, monitoring, contract)

    assert len(ref_frame) == 2
    assert len(mon_frame) == 2
    # История реплик сессии входит в question drift-фрейма
    assert "в1" in ref_frame["question"].iloc[1]


@pytest.mark.parametrize("mode", ["qa", "dialogue"])
def test_main_accepts_qa_and_dialogue(mode, monkeypatch):
    import main as drift

    if mode == "qa":
        reference = pd.DataFrame({
            "query_id": ["r1", "r2"],
            "input_query": ["вопрос 1", "вопрос 2"],
            "output_answer": ["ответ 1", "ответ 2"],
            "score_metric": [1.0, 0.0],
            "main_metric": [1.0, 0.0],
        })
        monitoring = reference.drop(columns=["score_metric", "main_metric"])
    else:
        reference = pd.DataFrame({
            "session_id": ["r1", "r2"],
            "dialogue": [
                repr([("r1-1", "вопрос 1", "ответ 1"), ("r1-2", "уточнение", "ответ")]),
                repr([("r2-1", "вопрос 2", "ответ 2")]),
            ],
            "input_query_count": [1, 1],
            "score_metric": [1.0, 0.0],
            "main_metric": [1.0, 0.0],
        })
        monitoring = reference.drop(columns=["score_metric", "main_metric"])

    captured = {}

    def fake_valtest(*, sampler, **_kwargs):
        captured["sizes"] = (len(sampler.train["X"]), len(sampler.test["X"]))
        captured["groups"] = (
            sampler.train["X"]["_group_id"].nunique(),
            sampler.test["X"]["_group_id"].nunique(),
        )
        return {
            "report": {"semaphore": "green"},
            "precomputed": {
                "status": "computed",
                "gini_value": 0.0,
                "gini_std": 0.0,
                "gini_ci_lower": 0.0,
                "gini_ci_upper": 0.0,
                "resampling_iterations": 1,
                "n_oos": 2,
                "n_oot": 2,
                "n_oos_groups": 2,
                "n_oot_groups": 2,
                "input_normalization": None,
            },
        }

    monkeypatch.setattr(drift, "valtest_adversarial_text", fake_valtest)

    result = drift.main(reference, monitoring, _contract(mode))

    assert captured["sizes"] == (2, 2)
    assert captured["groups"] == (2, 2)
    assert result["all_results"]["test_name"] == "oos_oot"


def test_not_computable_metric_skips_drift_computation(monkeypatch):
    import main as drift

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Вычислительный путь не должен запускаться")

    monkeypatch.setattr(drift, "prepare_drift_frames", forbidden)
    monkeypatch.setattr(drift, "AutoAsessorSampler", forbidden)
    monkeypatch.setattr(drift, "valtest_adversarial_text", forbidden)

    result = drift.main(
        object(),
        object(),
        {
            "contract_version": "laim-monitoring-metric.v2",
            "umr_version": "laim-umr.v2",
            "status": "not_computable",
            "reason_code": "ambiguous_baseline",
            "reason": "baseline нельзя определить однозначно",
        },
    )

    light = result["all_results"]
    assert light["color"] == "gray"
    assert light["status"] == "not_computable"
    assert light["reason_code"] == "ambiguous_baseline"
    assert light["reason"] == "baseline нельзя определить однозначно"
    assert light["test_name"] == "oos_oot"


def test_descriptor_describes_monitoring_umr():
    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text()
    )
    monitoring_port = next(
        port for port in descriptor["ports"] if port["name"] == "monitoring_umr"
    )
    assert "monitoring_umr" in monitoring_port["description"]
    assert "parquet_test_dataset" not in monitoring_port["description"]


def test_informational_gray_is_not_computable():
    # Серый по is_info обязан идти как not_computable с причиной: агрегатор
    # отвергает пару gray/computed как противоречие.
    import main as oos

    res = {"report": {"semaphore": "gray"},
           "precomputed": {"gini_value": 0.1, "gini_std": 0.01, "gini_ci_lower": 0.08,
                           "gini_ci_upper": 0.12, "n_oos": 100, "n_oot": 100,
                           "n_oos_groups": 60, "n_oot_groups": 60, "status": "computed"}}
    light = oos.report_valtest_oos_oot(res, "title")["all_results"]
    assert light["color"] == "gray" and light["status"] == "not_computable"
    assert light["reason"]


def test_verdict_uses_lower_bound_of_resample_spread():
    import math

    from llm_val.valtest_adversarial_test import report_valtest_adversarial_text

    # Среднее 0.45 дало бы жёлтый, но нижняя граница разброса 0.35 — зелёный.
    assert report_valtest_adversarial_text(0.45, 0.05, (0.35, 0.55))["semaphore"] == "green"
    assert report_valtest_adversarial_text(0.85, 0.02, (0.81, 0.89))["semaphore"] == "red"
    assert report_valtest_adversarial_text(0.45, 0.05, (math.nan, math.nan))["semaphore"] == "yellow"


def test_stable_reference_and_min_groups_are_passed(monkeypatch):
    import main as drift

    captured = {}

    def fake_valtest(*, sampler, **kwargs):
        captured["sizes"] = (len(sampler.train["X"]), len(sampler.test["X"]))
        captured["min_groups"] = kwargs.get("min_groups_per_side")
        return {
            "report": {"semaphore": "green"},
            "precomputed": {
                "status": "computed", "gini_value": 0.1, "gini_std": 0.0,
                "gini_ci_lower": 0.1, "gini_ci_upper": 0.1,
                "n_oos": 3, "n_oot": 2, "n_oos_groups": 3, "n_oot_groups": 2,
            },
        }

    monkeypatch.setattr(drift, "valtest_adversarial_text", fake_valtest)
    basket = pd.DataFrame({
        "query_id": ["r1", "r2"], "input_query": ["вопрос 1", "вопрос 2"],
        "output_answer": ["ответ 1", "ответ 2"], "main_metric": [1.0, 0.0],
    })
    stable = pd.DataFrame({
        "query_id": ["s1", "s2", "s3"], "input_query": ["в1", "в2", "в3"],
        "output_answer": ["о1", "о2", "о3"], "main_metric": [1.0, 1.0, 0.0],
    })
    monitoring = basket.drop(columns=["main_metric"])

    transitional = drift.main(basket, monitoring, _contract("qa"), min_groups_per_side=2)
    assert captured["sizes"] == (2, 2) and captured["min_groups"] == 2
    assert transitional["all_results"]["reference_source"] == "validation_basket"
    assert transitional["all_results"]["informative"] is True
    assert transitional["all_results"]["verdict_statistic"] == "gini_ci_lower"

    stable_run = drift.main(
        basket, monitoring, _contract("qa"), reference_stable_umr=stable, min_groups_per_side=2
    )
    assert captured["sizes"] == (3, 2)
    assert stable_run["all_results"]["reference_source"] == "stable_period"
    assert stable_run["all_results"]["informative"] is False


def test_descriptor_declares_stable_reference_port_and_min_groups():
    descriptor = json.loads(
        (Path(__file__).resolve().parents[1] / "descriptor.json").read_text()
    )
    ports = {port["name"]: port for port in descriptor["ports"]}
    assert ports["reference_stable_umr"]["required"] is False
    assert ports["reference_stable_umr"]["type"] == "dataframe"
    settings = {
        item["parameter"]: item["defaultValue"]
        for section in descriptor["ui"]["settings"]
        for component in section["components"]
        for item in component["config"]["components"]
    }
    assert settings["min_groups_per_side"] == 50
