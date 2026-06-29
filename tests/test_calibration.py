"""Tests for Stage 5 threshold calibration (doc section 4.9 / section 7)."""

from __future__ import annotations

import json
import os

from srag import (
    Controller,
    Planner,
    Stage1Pipeline,
    Verifier,
    calibrate_thresholds,
    collect_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, os.pardir, "data", "sample_corpus.jsonl")
DEVSET = os.path.join(HERE, os.pardir, "data", "dev_set.jsonl")


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_separable_points_give_perfect_abstention():
    # Answerable items score high, unanswerable low: cleanly separable.
    points = [(0.9, True), (0.8, True), (0.85, True),
              (0.1, False), (0.0, False), (0.2, False)]
    res = calibrate_thresholds(points)
    assert res.tau_answer >= res.tau_abstain
    assert res.abstention_precision == 1.0
    assert res.abstention_recall == 1.0
    # The abstain threshold separates the two clusters.
    assert 0.2 < res.tau_abstain <= 0.8


def test_thresholds_stay_ordered_when_overlapping():
    # Overlapping distributions: thresholds must still be ordered & in range.
    points = [(0.6, True), (0.55, True), (0.5, False), (0.45, True), (0.4, False)]
    res = calibrate_thresholds(points)
    assert 0.0 <= res.tau_abstain <= res.tau_answer <= 1.0001


def test_calibration_over_dev_set_abstains_on_unanswerable():
    pipe = Stage1Pipeline(planner=Planner()).index_documents(_load(CORPUS))
    ctrl = Controller(pipe, Verifier(), tau=0.55, max_corrections=3)
    points = collect_points(ctrl, _load(DEVSET))
    res = calibrate_thresholds(points)
    assert res.tau_answer >= res.tau_abstain
    # Every truly-unanswerable dev item ends up below the abstain threshold.
    assert res.abstention_recall >= 0.8


def test_calibrated_controller_abstains_on_clear_unanswerable():
    pipe = Stage1Pipeline(planner=Planner()).index_documents(_load(CORPUS))
    ctrl = Controller(pipe, Verifier(), tau=0.55, max_corrections=3)
    res = calibrate_thresholds(collect_points(ctrl, _load(DEVSET)))
    # Rebuild the controller with the calibrated thresholds.
    tuned = Controller(pipe, Verifier(), tau=res.tau_answer,
                       tau_abstain=res.tau_abstain, max_corrections=3)
    state = tuned.run("Who is the CEO of the Andromeda Mining Guild?")
    assert state.abstained is True
    assert state.final_answer == ""


def test_calibrated_controller_still_answers_answerable():
    pipe = Stage1Pipeline(planner=Planner()).index_documents(_load(CORPUS))
    ctrl = Controller(pipe, Verifier(), tau=0.55, max_corrections=3)
    res = calibrate_thresholds(collect_points(ctrl, _load(DEVSET)))
    tuned = Controller(pipe, Verifier(), tau=res.tau_answer,
                       tau_abstain=res.tau_abstain, max_corrections=3)
    state = tuned.run("Who directed Seven Samurai?")
    assert state.abstained is False
    assert "Kurosawa" in state.final_answer
