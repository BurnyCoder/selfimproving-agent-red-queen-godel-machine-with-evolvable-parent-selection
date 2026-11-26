"""Freeze reviewer identities independently of the sealed paper outcomes."""

from __future__ import annotations

import time
from .data import canonical, digest, dump, load, read_split
from .inference import now
from .integrity import PaperPanel, registered_splits
from .search import run_name
from .workspace import ROOT


def roster():
    """Order primary pairs before secondary controls; fork is attached to seed one."""
    return (
        [(s, a) for s in range(1, 9) for a in ("editable", "native")]
        + [(s, a) for s in range(1, 6) for a in ("uniform", "handcrafted")]
        + [(1, "fork-native")]
    )


def frozen_entry(run, endpoint=None):
    """The selected source and validation checkpoint fully identify a reviewer."""
    ep = endpoint or run["endpoints"][-1]
    matches = [n["workspace"] for n in run["nodes"] if n["id"] == ep["node"]]
    if len(matches) != 1:
        raise ValueError("Frozen endpoint node is absent or ambiguous")
    workspace = matches[0]
    if digest(workspace["reviewer"]) != ep["key"]:
        raise ValueError("Endpoint program identity failed verification")
    return dict(
        name=run["name"],
        seed=run["seed"],
        arm=run["arm"],
        source=workspace["reviewer"],
        key=ep["key"],
        n=ep["n"],
        compute=ep["compute"],
    )


def roster_entries(project, budget):
    """Reconstruct eligible endpoints directly from the registered search archives."""
    if budget not in (64, 128, 256):
        raise ValueError("Unregistered confirmatory search budget")
    entries = {
        "root": dict(
            name="root",
            seed=0,
            arm="root",
            source=ROOT["reviewer"],
            key=digest(ROOT["reviewer"]),
            n=0,
        )
    }
    runs = {}
    missing = []
    for seed, arm in roster():
        name = run_name("confirm", seed, arm)
        source = project / "results" / "runs" / (name + ".json")
        data = load(source) if source.exists() else {}
        if data.get("complete"):
            expected = dict(
                name=name, seed=seed, arm=arm, stage="confirm", budget=budget, n=budget
            )
            if (
                any(data.get(k) != v for k, v in expected.items())
                or not data["endpoints"]
                or data["endpoints"][-1]["n"] != budget
            ):
                raise ValueError(
                    "Completed archive differs from its registered endpoint"
                )
            runs[name] = data
            entries[name] = frozen_entry(data)
        else:
            missing.append(name)
    # Token sensitivity uses a cutoff fixed without consulting any test predictions.
    for seed in range(1, 9):
        pair = [run_name("confirm", seed, a) for a in ("editable", "native")]
        if not all(name in runs for name in pair):
            continue

        def tokens(ep):
            return ep["compute"]["input"] + ep["compute"]["output"]

        cutoff = min(tokens(runs[name]["endpoints"][-1]) for name in pair)
        for name in pair:
            eligible = [ep for ep in runs[name]["endpoints"] if tokens(ep) <= cutoff]
            entry = (
                frozen_entry(runs[name], eligible[-1])
                if eligible
                else dict(entries["root"], seed=seed, arm=runs[name]["arm"])
            )
            entries[name + "-tokens"] = dict(
                entry, name=name + "-tokens", cutoff=cutoff
            )
    return entries, missing


def freeze_fields(project):
    """Bind the chosen budget, complete roster, sources and full registered panel together."""
    base = project / "results"
    budget = load(base / "budget.json")
    split = registered_splits(project)["test"]
    PaperPanel("test", split)
    entries, missing = roster_entries(project, budget["budget"])
    return dict(
        version=2,
        split_sha=split["sha256"],
        panel_sha=digest(canonical(split)),
        budget=budget["budget"],
        budget_sha=digest(canonical(budget)),
        missing=missing,
        entries=entries,
        entries_sha=digest(canonical(entries)),
    )


def read_freeze(project):
    """A checksum detects edits; reconstructing the selection also checks its provenance."""
    manifest = load(project / "results" / "freeze.json")
    checksum = manifest.pop("freeze_sha", None)
    if checksum != digest(canonical(manifest)):
        raise ValueError("Missing or corrupt joint freeze fingerprint")
    expected = freeze_fields(project)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Joint freeze no longer matches the budget, split or endpoint archives"
        )
    manifest["freeze_sha"] = checksum
    return manifest


def freeze(project, allow_incomplete=False):
    """Build the entire endpoint manifest before opening any test manuscript."""
    path = project / "results" / "freeze.json"
    if path.exists():
        return read_freeze(project)
    manifest = freeze_fields(project)
    if manifest["missing"]:
        if not allow_incomplete:
            raise ValueError(
                "Pending confirmatory endpoints: " + ", ".join(manifest["missing"])
            )
        budget = load(project / "results" / "budget.json")
        if (
            time.time() < budget["deadline"]
            and not (project / "results/time-stop.json").exists()
        ):
            raise ValueError(
                "An incomplete cohort requires the registered time stopping record"
            )
    manifest["frozen_at"] = now()
    manifest["freeze_sha"] = digest(canonical(manifest))
    dump(path, manifest)
    return manifest


def endpoint_contract(project, split, name, registration, manifest=None):
    """Use the same identity checks in the scorer, its resume path and the public analysis."""
    if split == "test":
        manifest = manifest or read_freeze(project)
        if name not in manifest["entries"]:
            raise ValueError("Endpoint absent from joint freeze")
        entry = manifest["entries"][name]
        budget = manifest["budget"]
    elif split == "pilot":
        budget = 64
        if name == "root":
            entry = dict(
                name=name,
                seed=0,
                arm="root",
                key=digest(ROOT["reviewer"]),
                source=ROOT["reviewer"],
                n=0,
            )
        elif name in [
            run_name("pilot", 0, a)
            for a in ("editable", "native", "uniform", "handcrafted")
        ]:
            run = load(project / "results" / "runs" / (name + ".json"))
            if (
                not run["complete"]
                or run["stage"] != "pilot"
                or run["seed"] != 0
                or run_name("pilot", run["seed"], run["arm"]) != name
                or run["name"] != name
                or run["budget"] != budget
                or run["n"] != budget
                or run["endpoints"][-1]["n"] != budget
            ):
                raise ValueError(
                    "Only completed registered pilot searches are eligible"
                )
            entry = frozen_entry(run)
        else:
            raise ValueError("Unknown pilot endpoint")
    else:
        raise ValueError("Unknown evaluation split")
    if digest(entry["source"]) != entry["key"]:
        raise ValueError("Frozen source hash mismatch")
    identity = dict(
        split=split,
        name=name,
        seed=entry["seed"],
        arm=entry["arm"],
        key=entry["key"],
        checkpoint=entry["n"],
        budget=budget,
        split_sha=registration["sha256"],
        freeze_sha=manifest["freeze_sha"] if split == "test" else None,
        frozen_at=manifest["frozen_at"] if split == "test" else None,
    )
    return entry, identity


def score(project, domain, split, name, entry=None):
    """Validate persisted rows and endpoint identity before resuming any task calls."""
    manifest = read_freeze(project) if split == "test" else None
    if split not in ("test", "pilot"):
        raise ValueError("Unknown evaluation split")
    registration = registered_splits(project)[split]
    panel = PaperPanel(split, registration)
    selected, identity = endpoint_contract(project, split, name, registration, manifest)
    if entry is not None and entry != selected:
        raise ValueError("Supplied endpoint differs from the registered source")
    path = project / "results" / "evaluations" / f"{split}-{name}.json"
    out = (
        load(path)
        if path.exists()
        else dict(
            **identity,
            rows=[],
            elapsed=0.0,
            started=now(),
            complete=False,
        )
    )
    done = panel.verify(out, identity)
    if out["complete"]:
        return out
    for paper in read_split(project, split):
        if paper["id"] in done:
            continue
        start = time.monotonic()
        result = domain.review(
            {"reviewer": selected["source"]}, paper, f"endpoint/{paper['id']}"
        )
        out["rows"].append(
            dict(
                id=paper["id"],
                label=paper["label"],
                correct=int(result["decision"] == paper["label"]),
                **result,
            )
        )
        out["elapsed"] += time.monotonic() - start
        out["compute"] = domain.broker.metrics("endpoint")
        out["complete"] = len(out["rows"]) == len(panel.ids)
        panel.verify(out, identity)
        dump(path, out)
    return out
