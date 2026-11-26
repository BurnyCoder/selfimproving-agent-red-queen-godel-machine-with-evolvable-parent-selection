"""Exercise the actual scheduler and serialization with a deterministic paper domain."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
from rqstudy.analysis import crossed_interval
from rqstudy.endpoints import score
from rqstudy.sandbox import Worker
from rqstudy.search import Study
from rqstudy.workspace import ROOT, apply_changes


class FixtureDomain:
    """Supply known reviewer judgments without loading the local model or research data."""

    def __init__(self):
        self.broker = SimpleNamespace(
            ctx=8192,
            version="fixture",
            metrics=lambda phase: {
                k: 0
                for k in (
                    "input",
                    "output",
                    "calls",
                    "actual_input",
                    "actual_output",
                    "actual_calls",
                    "service_seconds",
                    "failed_attempts",
                )
            },
        )
        self.worker = Worker()
        self.validation = [
            dict(
                id=str(i), title=str(i), label=["accept", "reject"][i % 2], text=str(i)
            )
            for i in range(24)
        ]
        self.train = [dict(p, id="t" + p["id"]) for p in self.validation]
        self.anchor = self.validation[:8]
        self.anchor_provider = SimpleNamespace(
            provider_id="test", version="1", fingerprint="test"
        )
        self.edits = []

    def edit(self, parent, lineage, view, event, seed, editable):
        self.edits.append((parent.node_id, deepcopy(parent.workspace)))
        changes = dict(
            writer="Specific revision " + event,
            reviewer="Use the parity of the fixture.",
        )
        if editable:
            changes["selector"] = (
                "def select_parent(archive_snapshot, rng):\n    return 'root'"
            )
        return apply_changes(parent.workspace, changes, editable), "fixture"

    def review(self, workspace, paper, event):
        text = paper["text"]
        choice = (
            "reject"
            if workspace["reviewer"] == ROOT["reviewer"]
            else ["accept", "reject"][int(text) % 2]
        )
        return dict(
            decision=choice,
            evidence="fixture judgment",
            route="direct",
            chunks=1,
            chars=len(text),
            covered=len(text),
        )

    def write(self, workspace, paper, event):
        return paper["title"]


def test_resume_midpoint_fork_and_erasure(tmp_path):
    reference = Study(
        tmp_path / "full", FixtureDomain(), "confirm", 1, "editable", 16
    ).run()
    interrupted = Study(
        tmp_path / "restart", FixtureDomain(), "confirm", 1, "editable", 16
    ).run(stop_at=7)
    assert interrupted.n == 7
    replay = Study(
        tmp_path / "restart", FixtureDomain(), "confirm", 1, "editable", 16
    ).run()
    for key in ("trajectory", "endpoints", "selection", "human"):
        assert replay.info[key] == reference.info[key]
    assert replay.core.epoch_vector["review"] > 1
    assert all(r.valid for r in replay.core.archive.records if r.role_id == "reviewer")
    assert any(
        not r.valid for r in replay.core.archive.records if r.role_id == "writer"
    )
    assert any(n.cached_artifacts for n in replay.core.archive.nodes.values())
    assert all(e["observed"] >= 5 for e in replay.info["endpoints"])
    assert all(s["parent"] == "root" for s in replay.info["selection"])
    assert replay.domain.edits[-1][1] == ROOT
    mid = replay.directory / "midpoint.json"
    fork = Study(
        tmp_path / "restart", FixtureDomain(), "confirm", 1, "fork-native", 16, mid
    )
    assert fork.n == 8
    assert fork.info["selection"] == [s for s in replay.info["selection"] if s["n"] < 8]
    fork.run()
    assert fork.n == replay.n


def test_minimum_anchor_and_seal_precede_scoring(tmp_path):
    domain = FixtureDomain()
    domain.anchor = domain.anchor[:4]
    with pytest.raises(RuntimeError, match="five-paper"):
        Study(tmp_path, domain, "pilot", 0, "native", 8).run()
    with pytest.raises(FileNotFoundError, match="freeze.json"):
        score(tmp_path, domain, "test", "root")
    assert not (tmp_path / "results" / "evaluations").exists()


def test_crossed_interval_and_stratification():
    labels = ["reject", "accept", "reject", "accept"]
    perfect = [[1] * 4] * 2
    assert crossed_interval(perfect, perfect, labels, 40)["low"] == 0
    diff = crossed_interval(perfect, [[1, 0, 1, 0]] * 2, labels, 40)
    assert diff["low"] == diff["high"] == diff["mean_pp"] == 50
    assert diff["per_seed_pp"] == [50, 50]


def test_unicode_reading_pass_covers_every_character():
    from rqstudy.reading import chunks

    class Codec:
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(
                ids=list(range(len(text))),
                offsets=[(n, n + 1) for n in range(len(text))],
            )

    text = "Methods\n模型 α\nResults\n📚"
    parts = chunks(text, 4, Codec())
    assert len(parts) > 1 and "".join(parts) == text
