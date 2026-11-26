"""Regression fixtures for the treatment, published formulas and data boundary."""

from copy import deepcopy

import numpy as np
from rqgm.archive import Archive
from rqgm.models import UtilityRecord, WorkspaceNode

from rqstudy.data import clean
from rqstudy.policies import (
    archive_view,
    choose_parent,
    choose_thompson,
    reference_choice,
)
from rqstudy.sandbox import Worker
from rqstudy.workspace import ROOT, apply_changes


def tree():
    archive = Archive(WorkspaceNode("root", deepcopy(ROOT), None, 0))
    archive.add_node(WorkspaceNode("older", deepcopy(ROOT), "root", 1))
    archive.add_node(WorkspaceNode("latest", deepcopy(ROOT), "older", 2))
    for node, outcomes in (
        ("root", [0] * 16),
        ("older", [0] * 4),
        ("latest", [1] * 20),
    ):
        for i, outcome in enumerate(outcomes):
            archive.add_record(
                UtilityRecord(
                    f"{node}{i}", node, "reviewer", "human", outcome, {}, None, None
                )
            )
    return archive


def test_node_and_clade_evidence_differ():
    view = archive_view(tree(), 1, 40, 64)
    assert view["nodes"][0]["own_s"] == 0
    assert view["nodes"][0]["clade_s"] == 20
    # Compare decisions with identical RNG streams on deliberately mixed lineages.
    choices = [
        (
            choose_thompson(view, np.random.default_rng(s), clade=False),
            choose_thompson(view, np.random.default_rng(s), clade=True),
        )
        for s in range(500)
    ]
    assert any(a != b for a, b in choices)


def test_latest_selector_and_older_parent_inheritance():
    a = tree()
    a.nodes["latest"].workspace["selector"] = (
        "def select_parent(archive_snapshot, rng):\n    return 'older'"
    )
    selected = choose_parent(
        "editable", a, archive_view(a, 1, 40, 64), np.random.default_rng(42), Worker()
    )
    assert (selected["owner"], selected["parent"]) == ("latest", "older")
    child = apply_changes(
        a.nodes["older"].workspace, {"writer": "Focus on evidence."}, editable=True
    )
    assert child["selector"] == ROOT["selector"]
    assert child["writer"] != ROOT["writer"]


def test_failed_owner_falls_back():
    a = tree()
    a.nodes["latest"].workspace["selector"] = (
        "def select_parent(archive_snapshot, rng):\n    return 'absent'"
    )
    result = choose_parent(
        "editable", a, archive_view(a, 1, 40, 64), np.random.default_rng(2), Worker()
    )
    assert result["owner"] == "older" and result["failures"][0]["owner"] == "latest"


def test_released_and_published_penalties_can_disagree():
    v = {
        "nodes": [
            {"id": "a", "score": 0.5, "children": 6},
            {"id": "b", "score": 0.5, "children": 0},
        ]
    }
    assert reference_choice(v, 0.21, paper=False) != reference_choice(
        v, 0.21, paper=True
    )


def test_strip_outcome_headers_preserve_neighbor_text():
    assert "science" in clean(
        "'Title: science\\nPublished as a conference paper at ICLR 2024\\nSection: proof'"
    )
    assert "Published as" not in clean(
        "Published as a conference paper at ICLR 2024\nproof"
    )


def test_sandbox_blocks_private_io():
    import pytest

    with pytest.raises(ValueError):
        Worker().call(
            "def instruct(context):\n    return open('.env').read()", "instruct", {}
        )


def test_concatenated_mastheads_and_blind_review_titles():
    assert (
        clean("formula2Under review as a conference paper at ICLR 2025 appendix")
        == "formula2 appendix"
    )
    assert (
        clean("analyticalPublished as a conference paper at ICLR 2024 noise")
        == "analytical noise"
    )
    assert (
        clean(
            "Study Anonymous authors Paper under double-blind review\n\nAbstract: proof"
        )
        == "Study \n\nAbstract: proof"
    )
    assert "openreview.net" not in clean(
        "Citation https: //openreview.net/forum?id=public-id"
    )


def test_failed_worker_uniform_root_and_readonly_snapshot():
    class FailedWorker:
        def call(self, *args):
            raise ValueError("Worker timed out")

    a = tree()
    before = archive_view(a, 1, 40, 64)
    selected = choose_parent(
        "editable",
        a,
        before,
        np.random.default_rng(42),
        FailedWorker(),
        fallback_allowance=1,
    )
    assert (
        selected["trusted_root_fallback"]
        and selected["owner"] == "root"
        and len(selected["failures"]) == 2
    )
    assert selected["parent"] in a.nodes
    assert "secret" not in before
    Worker().call(
        "def instruct(context):\n    context['secret'] = 1\n    return 'ok'",
        "instruct",
        before,
    )
    assert "secret" not in before
