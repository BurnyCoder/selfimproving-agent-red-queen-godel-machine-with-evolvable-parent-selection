"""Generate all empirical values from event and per-paper result files."""

from __future__ import annotations

import numpy as np
from .data import load
from .endpoints import endpoint_contract, read_freeze, roster
from .integrity import PaperPanel, registered_splits
from .search import run_name


def crossed_interval(left, right, labels, draws=10000, seed=1701):
    """Resample common seeds and common paper columns within their decision strata."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    labels = np.asarray(labels)
    if (
        left.ndim != 2
        or left.shape != right.shape
        or left.shape[0] == 0
        or labels.shape != (left.shape[1],)
        or draws < 1
        or not np.isin(left, [0, 1]).all()
        or not np.isin(right, [0, 1]).all()
        or set(labels) != {"accept", "reject"}
    ):
        raise ValueError("Expected paired binary predictions on both decision classes")
    diffs = left - right
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == label) for label in ("accept", "reject")]
    samples = []
    for start in range(0, draws, 200):
        batch = min(200, draws - start)
        rows = rng.integers(0, len(diffs), size=(batch, len(diffs)))
        cols = np.concatenate(
            [rng.choice(s, size=(batch, len(s)), replace=True) for s in strata], axis=1
        )
        samples.extend(
            diffs[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2)) * 100
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return dict(
        mean_pp=float(diffs.mean() * 100),
        low=float(low),
        high=float(high),
        pairs=diffs.shape[0],
        per_seed_pp=(diffs.mean(axis=1) * 100).tolist(),
    )


def registered_verdict(low, high):
    """Use the prospectively recorded decision boundaries, including an upper bound of zero."""
    if low > 0:
        return "supports superiority"
    if high <= 0:
        return "evidence against superiority"
    return "inconclusive"


def coverage(evals):
    """List each absent or unfinished endpoint separately from the primary comparison."""
    done = {e["name"] for e in evals if e["split"] == "test" and e["complete"]}
    arms = {}
    for seed, arm in roster():
        if run_name("confirm", seed, arm) not in done:
            arms.setdefault(arm, []).append(seed)
    missing_pairs = [
        s
        for s in range(1, 9)
        if any(s in arms.get(a, []) for a in ("editable", "native"))
    ]
    missing_tokens = [
        s
        for s in range(1, 9)
        if any(
            run_name("confirm", s, a) + "-tokens" not in done
            for a in ("editable", "native")
        )
    ]
    return dict(
        missing_seeds_by_arm=arms,
        missing_primary_pairs=missing_pairs,
        missing_token_pairs=missing_tokens,
        root_missing="root" not in done,
    )


def summary(project):
    """Include in-progress searches but count only terminal evaluations for inference."""
    runs = [load(p) for p in sorted((project / "results" / "runs").glob("*.json"))]
    evals = []
    table = []
    registrations = {}
    frozen = (
        read_freeze(project) if (project / "results/freeze.json").exists() else None
    )
    for path in sorted((project / "results" / "evaluations").glob("*.json")):
        data = load(path)
        split, name = data["split"], data["name"]
        if split not in ("pilot", "test") or path.stem != f"{split}-{name}":
            raise ValueError("Evaluation file identity mismatch")
        if split == "test" and frozen is None:
            frozen = read_freeze(project)
        if not registrations:
            registrations = registered_splits(project)
        registration = registrations[split]
        entry, identity = endpoint_contract(
            project, split, name, registration, frozen if split == "test" else None
        )
        PaperPanel(split, registration).verify(data, identity)
        evals.append(data)
        rows = data["rows"]

        def successes(items):
            return sum(r["decision"] == r["label"] for r in items)

        by_route = {}
        for route in sorted({r["route"] for r in rows}):
            sub = [r for r in rows if r["route"] == route]
            by_route[route] = dict(n=len(sub), accuracy=successes(sub) / len(sub))
        table.append(
            dict(
                name=data["name"],
                split=data["split"],
                arm=data["arm"],
                n=len(rows),
                correct=successes(rows),
                complete=data["complete"],
                routes=by_route,
                compute=data.get("compute", {}),
                checkpoint=entry["n"],
                cutoff=entry.get("cutoff"),
                logical_search_tokens=sum(
                    entry.get("compute", {}).get(k, 0) for k in ("input", "output")
                ),
            )
        )
    panels = {e["name"]: e for e in evals if e["split"] == "test" and e["complete"]}
    contrasts = {}
    for suffix in ("", "-tokens"):
        left, right, seeds = [], [], []
        labels = None
        for seed in range(1, 9):
            keys = [
                run_name("confirm", seed, a) + suffix for a in ("editable", "native")
            ]
            if not all(k in panels for k in keys):
                continue
            a, b = [sorted(panels[k]["rows"], key=lambda x: x["id"]) for k in keys]
            if [(r["id"], r["label"]) for r in a] != [(r["id"], r["label"]) for r in b]:
                raise ValueError("Mismatched paired papers")
            labels = [r["label"] for r in a]
            left.append([int(r["decision"] == r["label"]) for r in a])
            right.append([int(r["decision"] == r["label"]) for r in b])
            seeds.append(seed)
        if seeds:
            outcome = crossed_interval(left, right, labels)
            outcome["seeds"] = seeds
            outcome["missing_seeds"] = [s for s in range(1, 9) if s not in seeds]
            outcome["interpretation"] = (
                "incomplete, descriptive"
                if len(seeds) < 8
                else "secondary token-budget sensitivity"
                if suffix
                else registered_verdict(outcome["low"], outcome["high"])
            )
            contrasts["matched_tokens" if suffix else "primary"] = outcome
    return runs, evals, table, contrasts
