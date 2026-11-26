"""Outcome-independent scientific regression fixtures. All papers and decisions are synthetic."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from rqstudy.analysis import summary
from rqstudy.data import canonical, clean, digest, dump, load
from rqstudy.endpoints import freeze, score
from rqstudy.search import run_name
from rqstudy.workspace import ROOT


def test_disclosures_preserve_science():
    raw = (
        "Title: Funding mechanisms\n\nAbstract: We study funding decisions.\n"
        "Section: Results\nOur estimate was 3.\n"
        "Section: Acknowledgments and Disclosure of Funding\n"
        "We thank the anonymous reviewers. This work was funded by grant XYZ.\n"
        "embeddings at layer two converge.\nSection: Appendix\nProof of convergence.\n"
        "Acknowledgements. Private donor gift.\nReferences:\nReference one."
    )
    text = clean(raw)
    for keep in (
        "Title: Funding mechanisms",
        "We study funding decisions.",
        "Our estimate was 3.",
        "embeddings at layer two converge.",
        "Section: Appendix",
        "Proof of convergence.",
        "References:",
        "Reference one.",
    ):
        assert keep in text
    for remove in (
        "Disclosure of Funding",
        "anonymous reviewers",
        "grant XYZ",
        "Private donor",
    ):
        assert remove not in text
    assert clean(text) == text


def test_standalone_funding_and_thanks_keep_adjacent_sentences():
    text = clean(
        "Accuracy increased. This work was supported by NSF grant 123. "
        "Our second experiment was smaller.\nWe thank the anonymous reviewers for feedback."
    )
    assert (
        "Accuracy increased." in text and "Our second experiment was smaller." in text
    )
    assert "NSF" not in text and "thank the anonymous reviewers" not in text
    assert clean(
        "Funding constraints affect markets. Evidence is supported by data."
    ) == ("Funding constraints affect markets. Evidence is supported by data.")
    science = "The method is supported by foundation models. Our conclusions are supported by experiments in this project."
    assert clean(science) == science
    references = "References:\nAnonymity and disclosure of funding: a review. We thank the reviewers."
    assert clean(references) == references


def test_independent_statistic_and_boundaries():
    from rqstudy.analysis import registered_verdict
    from rqstudy.audit import numerical_checks

    checks = numerical_checks()
    assert checks["independent"] == pytest.approx(checks["production"])
    assert checks["seed_only_exact_coverage"]["covered"] == 238
    assert checks["seed_only_exact_coverage"]["total"] == 256
    assert registered_verdict(0.01, 3) == "supports superiority"
    assert (
        registered_verdict(-3, -0.01)
        == registered_verdict(-3, 0)
        == registered_verdict(0, 0)
        == "evidence against superiority"
    )
    assert registered_verdict(0, 3) == registered_verdict(-2, 1) == "inconclusive"


def test_extracted_credit_caption_after_references_preserves_scientific_caption():
    raw = (
        "References:\nSmith. Disclosure of funding in science.\n"
        "Figure tab_3:\nType: table\nCaption: Jane Smith thanks A. Mentor for comments. "
        "Chris Lee is supported by a Google PhD Fellowship. This work was made possible by a gift from XYZ.\n"
        "Figure fig_4:\nType: figure\nCaption: Accuracy by sample size."
    )
    text = clean(raw)
    assert "Fellowship" not in text and "gift from" not in text and "tab_3" not in text
    assert "Smith. Disclosure of funding in science." in text
    assert "Figure fig_4:\nType: figure\nCaption: Accuracy by sample size." in text


def body_text(seed, i):
    return f"Title: Synthetic {seed}-{i}\n\nAbstract: A fixture.\nSection: Results\nFixed example."


def synthetic_rows(seed=0):
    rows = []
    for i in range(100):
        label = ("accept", "reject")[i % 2]
        rows.append(
            dict(
                id=f"p{seed}-{i:03}",
                sha256=digest(body_text(seed, i)),
                label=label,
                decision=label,
                correct=1,
                route="direct",
            )
        )
    return rows


def cohort(tmp_path, seeds):
    """Create real frozen run archives and a balanced fingerprinted test population."""
    root = tmp_path / "results"
    rows = synthetic_rows()
    ident = [(r["id"], r["sha256"], r["label"]) for r in rows]
    split = dict(
        count=100,
        ids=[r["id"] for r in rows],
        sha256=digest(canonical(ident)),
        item_sha256=[digest(canonical(x)) for x in ident],
    )
    manifest = {"splits": {"test": split}}
    dump(root / "split-manifest.json", manifest)
    dump(root / "split-manifest.2026-11-20.json", manifest)
    dump(
        root / "manuscript-audit.json",
        dict(
            manifest_sha256=digest(canonical(manifest)),
            splits={"test": {"before": split["sha256"], "after": split["sha256"]}},
        ),
    )
    dump(root / "budget.json", dict(budget=64, deadline=0))
    for seed in seeds:
        for arm in ("editable", "native"):
            name = run_name("confirm", seed, arm)
            ep = dict(
                n=64,
                node="root",
                key=digest(ROOT["reviewer"]),
                compute=dict(input=100, output=15),
            )
            dump(
                root / "runs" / (name + ".json"),
                dict(
                    name=name,
                    stage="confirm",
                    seed=seed,
                    arm=arm,
                    budget=64,
                    n=64,
                    complete=True,
                    endpoints=[ep],
                    nodes=[dict(id="root", workspace=ROOT)],
                ),
            )
    manifest = freeze(tmp_path, allow_incomplete=True)
    for seed in seeds:
        for arm in ("editable", "native"):
            name = run_name("confirm", seed, arm)
            armrows = deepcopy(rows)
            if arm == "native":
                for row in armrows:
                    row.update(decision="invalid", correct=0)
            dump(
                root / "evaluations" / f"test-{name}.json",
                dict(
                    name=name,
                    arm=arm,
                    seed=seed,
                    split="test",
                    complete=True,
                    rows=armrows,
                    key=digest(ROOT["reviewer"]),
                    checkpoint=64,
                    budget=64,
                    split_sha=split["sha256"],
                    freeze_sha=manifest["freeze_sha"],
                    frozen_at=manifest["frozen_at"],
                    started=manifest["frozen_at"],
                ),
            )
    return root


def test_false_verdict_without_joint_freeze(tmp_path):
    for seed in range(1, 9):
        for arm in ("editable", "native"):
            name = run_name("confirm", seed, arm)
            rows = synthetic_rows(seed)
            if arm == "native":
                for row in rows:
                    row.update(decision="invalid", correct=0)
            dump(
                tmp_path / "results" / "evaluations" / f"test-{name}.json",
                dict(
                    name=name,
                    arm=arm,
                    seed=seed,
                    split="test",
                    complete=True,
                    rows=rows,
                ),
            )
    with pytest.raises((ValueError, FileNotFoundError)):
        summary(tmp_path)


@pytest.mark.parametrize(
    "damage",
    [
        "other_population",
        "duplicate",
        "label",
        "bit",
        "body",
        "source",
        "checkpoint",
        "budget",
        "split_sha",
        "freeze_sha",
        "chronology",
        "short",
    ],
)
def test_inconsistent_endpoint_never_yields_an_interval(tmp_path, damage):
    root = cohort(tmp_path, [1, 2])
    path = root / "evaluations" / "test-confirm-02-editable.json"
    data = load(path)
    if damage == "other_population":
        data["rows"] = synthetic_rows(2)
    elif damage == "duplicate":
        data["rows"][1] = data["rows"][0]
    elif damage == "chronology":
        data["started"] = "2000-01-01T00:00:00+00:00"
    elif damage == "short":
        data["rows"].pop()
    elif damage in {"label", "bit", "body"}:
        field, value = {
            "label": ("label", "reject"),
            "bit": ("correct", 0),
            "body": ("sha256", digest("another paper")),
        }[damage]
        data["rows"][0][field] = value
    else:
        data[{"source": "key"}.get(damage, damage)] = "altered"
    dump(path, data)
    with pytest.raises(ValueError):
        summary(tmp_path)


def test_primary_completeness_is_per_comparison(tmp_path):
    cohort(tmp_path, range(1, 9))
    primary = summary(tmp_path)[3]["primary"]
    assert primary["interpretation"] == "supports superiority"
    assert primary["mean_pp"] == 100 and primary["missing_seeds"] == []
    (tmp_path / "results/evaluations/test-confirm-08-native.json").unlink()
    primary = summary(tmp_path)[3]["primary"]
    assert (
        primary["missing_seeds"] == [8]
        and primary["interpretation"] == "incomplete, descriptive"
    )


def test_freeze_and_resume_reject_drift_before_review(tmp_path):
    root = cohort(tmp_path, [1])
    split = load(root / "split-manifest.json")
    split["splits"]["test"]["sha256"] = digest("different")
    dump(root / "split-manifest.json", split)
    with pytest.raises(ValueError):
        freeze(tmp_path, allow_incomplete=True)
    with pytest.raises(ValueError):
        score(tmp_path, None, "test", "confirm-01-editable")


def test_scorer_resumes_valid_partial_and_rejects_bad_partial(tmp_path):
    """Exercise the real scorer with one outstanding synthetic paper and no model broker."""
    root = cohort(tmp_path, [1])
    name = "confirm-01-editable"
    path = root / "evaluations" / f"test-{name}.json"
    complete = load(path)
    partial = dict(complete, complete=False, elapsed=0.0, rows=complete["rows"][:-1])
    bad = deepcopy(partial)
    bad["rows"][0]["correct"] = 0
    dump(path, bad)
    with pytest.raises(ValueError, match="correctness"):
        score(tmp_path, None, "test", name)
    dump(path, partial)
    papers = [
        dict(
            id=r["id"],
            text=body_text(0, i),
            title=f"Synthetic 0-{i}",
            abstract="A fixture.",
            text_sha256=r["sha256"],
            label=r["label"],
        )
        for i, r in enumerate(complete["rows"])
    ]
    dump(tmp_path / ".private/data/test.json", papers)
    called = []

    def review(workspace, paper, event):
        called.append(paper["id"])
        return dict(decision="reject", route="direct", sha256=digest(paper["text"]))

    domain = SimpleNamespace(
        review=review, broker=SimpleNamespace(metrics=lambda _: {})
    )
    assert score(tmp_path, domain, "test", name)["complete"]
    assert score(tmp_path, None, "test", name)["rows"] == complete["rows"]
    assert called == [papers[-1]["id"]]


def test_fork_accounts_shared_prefix_once():
    from rqstudy.search import Study

    current = dict(
        calls=2,
        input=20,
        output=5,
        actual_calls=1,
        actual_input=10,
        actual_output=2,
        service_seconds=3,
        failed_attempts=1,
    )
    study = Study.__new__(Study)
    study.domain = SimpleNamespace(broker=SimpleNamespace(metrics=lambda _: current))
    study.info = {"compute_prefix": dict.fromkeys(current, 100)}
    actual = study.compute()
    assert (actual["calls"], actual["input"], actual["output"]) == (102, 120, 105)
    for key in (
        "actual_calls",
        "actual_input",
        "actual_output",
        "service_seconds",
        "failed_attempts",
    ):
        assert actual[key] == current[key]


def test_offline_guard_precedes_any_backend_access(tmp_path, monkeypatch):
    from rqstudy.inference import Broker

    monkeypatch.setenv("RQGM_DISABLE_MODEL", "1")
    with pytest.raises(RuntimeError, match="disabled"):
        Broker(tmp_path, "offline-test", 4096)
    assert not list(tmp_path.iterdir())
