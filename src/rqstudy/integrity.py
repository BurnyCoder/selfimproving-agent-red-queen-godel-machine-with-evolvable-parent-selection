"""Verify public per-paper commitments without reopening sealed manuscripts for reporting."""

from collections import Counter
from datetime import datetime

from .data import SIZES, canonical, digest, load


def event_time(value):
    """Compare explicit UTC offsets at the freeze and at the start of scoring."""
    try:
        stamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid study timestamp") from exc
    if stamp.utcoffset() is None:
        raise ValueError("Missing UTC offset in study timestamp")
    return stamp


def registered_splits(project):
    """Compare today's row commitments to the amendment receipt and original ordered enrollment."""
    base = project / "results"
    current = load(base / "split-manifest.json")
    audit = load(base / "manuscript-audit.json")
    original = load(base / "split-manifest.2026-11-20.json")["splits"]
    if (
        digest(canonical(current)) != audit["manifest_sha256"]
        or current["splits"].keys() != original.keys()
    ):
        raise ValueError("Registered split manifest differs from the amendment")
    for split, panel in current["splits"].items():
        before, proof = original[split], audit["splits"][split]
        if (
            panel["ids"] != before["ids"]
            or panel["count"] != before["count"]
            or proof["before"] != before["sha256"]
            or proof["after"] != panel["sha256"]
        ):
            raise ValueError("Registered paper identities or split hash changed")
    return current["splits"]


class PaperPanel:
    """The registered ID order is shared across every seed and arm in a comparison."""

    def __init__(self, split, registration):
        self.split = split
        self.ids = registration["ids"]
        self.sha = registration["sha256"]
        commits = registration.get("item_sha256", [])
        if (
            registration["count"] != SIZES[split]
            or len(self.ids) != SIZES[split]
            or len(set(self.ids)) != len(self.ids)
            or len(commits) != len(self.ids)
        ):
            raise ValueError(f"Invalid registered paper panel for {split}")
        self.commits = dict(zip(self.ids, commits, strict=True))

    def verify(self, data, identity):
        """Check even a partial row against its committed label and manuscript hash."""
        if any(data.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Endpoint identity, checkpoint, budget, split or freeze mismatch"
            )
        if self.split == "test" and event_time(data.get("started")) < event_time(
            identity["frozen_at"]
        ):
            raise ValueError("Sealed evaluation predates its frozen endpoint")
        observed = {}
        labels = Counter()
        for row in data["rows"]:
            if (
                not isinstance(row, dict)
                or not {"id", "sha256", "label", "decision", "correct"} <= row.keys()
            ):
                raise ValueError("Endpoint row is missing a required field")
            paper, label, decision = row["id"], row["label"], row["decision"]
            if not isinstance(paper, str):
                raise ValueError("Invalid paper ID")
            triple = (paper, row["sha256"], label)
            if paper in observed or self.commits.get(paper) != digest(
                canonical(triple)
            ):
                raise ValueError(
                    "Duplicate or unregistered manuscript identity or label"
                )
            correct = int(decision == label)
            if (
                label not in ("accept", "reject")
                or decision not in ("accept", "reject", "invalid")
                or type(row["correct"]) is not int
                or row["correct"] != correct
            ):
                raise ValueError("Invalid decision or inconsistent correctness bit")
            labels[label] += 1
            observed[paper] = triple
        complete = len(observed) == len(self.ids)
        if data.get("complete") is not complete:
            raise ValueError(
                "Endpoint completion flag disagrees with scored paper count"
            )
        if complete and (
            labels != {"accept": len(self.ids) // 2, "reject": len(self.ids) // 2}
            or digest(canonical([observed[i] for i in self.ids])) != self.sha
        ):
            raise ValueError(
                "Completed endpoint differs from the balanced registered cohort"
            )
        return set(observed)
