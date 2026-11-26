"""Run, resume and publish the registered pilot, paired confirmation and midpoint fork."""

from __future__ import annotations

from contextlib import contextmanager
import subprocess
import tempfile
import time
import zipfile
import httpx
from .data import dump, load
from .domain import Domain
from .endpoints import freeze, roster, score
from .inference import Broker, MODEL, MODEL_DIGEST, now
from .reporting import report
from .sandbox import Worker
from .search import Study, run_name

ARMS = ("editable", "native", "uniform", "handcrafted")
HF = "BurnyCoder/rqgm-editable-parent-selection"


def preflight(project):
    """Lock the largest registered GPU-resident context using only a synthetic input."""
    path = project / "results" / "runtime.json"
    if path.exists():
        return load(path)
    attempts = []
    for ctx in (16384, 8192, 4096):
        b = Broker(project, "preflight-" + str(ctx), ctx)
        try:
            b.ask(
                "hardware/synthetic",
                "preflight",
                "Return OK.",
                "Synthetic GPU test.",
                8,
            )
            processes = b.client.get("/api/ps").json()["models"]
            process = next(m for m in processes if m["digest"] == MODEL_DIGEST)
            attempts.append(
                dict(
                    ctx=ctx,
                    size=process["size"],
                    size_vram=process["size_vram"],
                    compute=b.metrics(),
                )
            )
            if (
                process["size_vram"] >= process["size"]
                and process.get("context_length", ctx) == ctx
            ):
                hardware = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip()
                runtime = dict(
                    model=MODEL,
                    digest=MODEL_DIGEST,
                    ollama=b.version,
                    ctx=ctx,
                    gpu=hardware,
                    attempts=attempts,
                    registered=now(),
                )
                dump(path, runtime)
                return runtime
        except (httpx.HTTPError, RuntimeError, KeyError, StopIteration) as exc:
            attempts.append(dict(ctx=ctx, error=str(exc)))
        finally:
            b.close()
    dump(project / ".private" / "failed-preflight.json", attempts)
    raise RuntimeError(
        "Registered contexts did not achieve full GPU residency; see the preflight log"
    )


@contextmanager
def domain_for(project, name):
    """Open and close the per-run compute ledger on all execution paths."""
    runtime = load(project / "results" / "runtime.json")
    broker = Broker(project, name, runtime["ctx"])
    try:
        if runtime["ollama"] != broker.version or runtime["digest"] != MODEL_DIGEST:
            raise ValueError("The frozen inference backend changed")
        yield Domain(project, broker, Worker())
    finally:
        broker.close()


def search(project, stage, seed, arm, budget, stop_at=None):
    """Use the same user-facing operation from the CLI and the multi-run campaign."""
    name = run_name(stage, seed, arm)
    mid = (
        project
        / ".private"
        / "runs"
        / run_name("confirm", 1, "editable")
        / "midpoint.json"
        if arm == "fork-native"
        else None
    )
    if stage == "confirm":
        registered = load(project / "results" / "budget.json")["budget"]
        if (
            budget != registered
            or (seed, arm) not in roster()
            or (project / "results" / "freeze.json").exists()
        ):
            raise ValueError(
                "Confirmatory configuration is outside its registered cohort"
            )
    elif stage != "pilot" or seed != 0 or budget != 64 or arm not in ARMS:
        raise ValueError(
            "Pilot uses seed zero, the four registered arms and 64 outcomes"
        )
    with domain_for(project, name) as d:
        return search_result(
            Study(project, d, stage, seed, arm, budget, mid).run(stop_at)
        )


def search_result(study):
    return dict(
        name=study.name,
        n=study.n,
        budget=study.info["budget"],
        epoch=study.core.epoch_vector,
    )


def evaluate(project, split, name):
    with domain_for(project, "eval-" + split + "-" + name) as d:
        out = score(project, d, split, name)
        return dict(
            name=name, split=split, papers=len(out["rows"]), complete=out["complete"]
        )


def budget_decision(project, deadline, remaining=None):
    """Price all future search and endpoint runs from the slowest pilot timings only."""
    path = project / "results" / "budget.json"
    if path.exists():
        return load(path)
    searches = [
        load(project / "results" / "runs" / (run_name("pilot", 0, a) + ".json"))
        for a in ARMS
    ]
    eps = [
        load(project / "results" / "evaluations" / ("pilot-" + r["name"] + ".json"))
        for r in searches
    ]
    eps.append(load(project / "results" / "evaluations" / "pilot-root.json"))
    if not all(r["complete"] for r in [*searches, *eps]):
        raise ValueError("Incomplete pilot block")
    per_eval = max(r["elapsed"] / r["n"] for r in searches)
    per_paper = max(e["elapsed"] / len(e["rows"]) for e in eps)
    available = deadline - time.time() if remaining is None else remaining
    search_equivalents = len(roster()) - 0.5
    endpoints = len(roster()) + 1 + 2 * 8
    test_items = load(project / "results" / "split-manifest.json")["splits"]["test"][
        "count"
    ]
    reserve = per_paper * test_items * endpoints
    projections = {
        str(b): (per_eval * b * search_equivalents + reserve) / 3600
        for b in (64, 128, 256)
    }
    fits = [b for b in (64, 128, 256) if projections[str(b)] * 3600 <= available]
    decision = dict(
        budget=max(fits, default=64),
        deadline=deadline,
        available_hours=available / 3600,
        chosen_at=now(),
        pilot_seconds_per_search_eval=per_eval,
        pilot_seconds_per_item=per_paper,
        search_equivalents=search_equivalents,
        planned_endpoints=endpoints,
        heldout_reserve_seconds=reserve,
        projections_hours=projections,
        all_projected_to_fit=bool(fits),
    )
    dump(path, decision)
    return decision


def command(project, *args):
    """Run publication tools with argument arrays and checked exit statuses."""
    return subprocess.run(list(args), cwd=project, text=True, check=True)


def publish(project, label):
    """Publish only tracked source plus committed public artifacts to both destinations."""
    report(project)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=project, text=True
    ).strip()
    if branch != "main":
        raise RuntimeError(
            "Merge the implementation PR before publishing empirical artifacts"
        )
    command(project, "git", "add", "--", "README.md", "reports", "docs", "results")
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=project
    ).returncode
    if changed == 1:
        command(project, "git", "commit", "-m", label)
    elif changed != 0:
        raise RuntimeError("Unable to check staged artifacts")
    command(project, "git", "push", "origin", "main")
    rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()
    with tempfile.TemporaryDirectory(dir=project / ".private") as temp:
        from pathlib import Path

        folder = Path(temp)
        archive = folder / "study.zip"
        command(
            project, "git", "archive", "--format=zip", "--output", str(archive), rev
        )
        with zipfile.ZipFile(archive) as source:
            source.extractall(folder)
        archive.unlink()
        body = (folder / "README.md").read_text(encoding="utf-8")
        (folder / "README.md").write_text(
            "---\nbase_model: Qwen/Qwen3.5-4B\nlibrary_name: rqgm-parent-study\ntags:\n- paper-review\n- evolutionary-search\n---\n\n"
            + body
            + "\n\nGitHub source revision "
            + rev
            + ". This repository mirrors the study sources and verified results. Evolved workspaces are published when a search has a recorded archive.\n",
            encoding="utf-8",
        )
        command(
            project,
            "hf",
            "upload",
            HF,
            str(folder),
            ".",
            "--type",
            "model",
            "--commit-message",
            label,
            "--quiet",
        )
    return rev


def campaign(project, publish_results):
    """Keep the deadline, completed milestones and resumption state local to one study."""
    ledger = project / ".private" / "campaign.json"
    state = (
        load(ledger)
        if ledger.exists()
        else dict(started=now(), deadline=time.time() + 7 * 86400, done=[])
    )
    dump(ledger, state)

    def step(tag, fn):
        if tag in state["done"]:
            return
        print(f"[{now()}] Starting {tag}", flush=True)
        fn()
        if publish_results:
            publish(project, "Record " + tag)
        else:
            report(project)
        state["done"].append(tag)
        dump(ledger, state)
        print(f"[{now()}] Completed {tag}", flush=True)

    step("GPU preflight", lambda: preflight(project))
    step("pilot root evaluation", lambda: evaluate(project, "pilot", "root"))
    for arm in ARMS:
        name = run_name("pilot", 0, arm)
        step(name + " search", lambda a=arm: search(project, "pilot", 0, a, 64))
        step(name + " evaluation", lambda n=name: evaluate(project, "pilot", n))
    step("timing-selected budget", lambda: budget_decision(project, state["deadline"]))
    chosen = load(project / "results" / "budget.json")
    if not (project / "results" / "freeze.json").exists():
        groups = [[(seed, "editable"), (seed, "native")] for seed in range(1, 9)]
        groups.extend([[item] for item in roster()[16:]])
        for group in groups:
            outstanding = [
                (s, a)
                for s, a in group
                if run_name("confirm", s, a) + " search" not in state["done"]
            ]
            if not outstanding:
                continue
            completed = [
                (s, a)
                for s, a in roster()
                if run_name("confirm", s, a) + " search" in state["done"]
            ]
            planned = completed + outstanding
            endpoints = (
                1 + len(planned) + sum(a in ("editable", "native") for s, a in planned)
            )
            paper_count = load(project / "results" / "split-manifest.json")["splits"][
                "test"
            ]["count"]
            reserve = endpoints * paper_count * chosen["pilot_seconds_per_item"]
            cost = (
                sum(0.5 if a == "fork-native" else 1 for s, a in outstanding)
                * chosen["budget"]
                * chosen["pilot_seconds_per_search_eval"]
            )
            # Always try the first primary pair. Thereafter protect time for already-earned endpoints.
            if completed and time.time() + reserve + cost > state["deadline"]:
                dump(
                    project / "results" / "time-stop.json",
                    dict(
                        at=now(),
                        next_group=outstanding,
                        projected_search_seconds=cost,
                        reserved_endpoint_seconds=reserve,
                        remaining_seconds=state["deadline"] - time.time(),
                    ),
                )
                break
            for seed, arm in outstanding:
                name = run_name("confirm", seed, arm)
                step(
                    name + " search",
                    lambda a=arm, s=seed: search(
                        project, "confirm", s, a, chosen["budget"]
                    ),
                )
    incomplete = (
        project / "results" / "time-stop.json"
    ).exists() or time.time() >= state["deadline"]
    step("joint reviewer freeze", lambda: freeze(project, allow_incomplete=incomplete))
    manifest = load(project / "results" / "freeze.json")
    for name in manifest["entries"]:
        step("sealed " + name, lambda n=name: evaluate(project, "test", n))
    return dict(completed=state["done"], missing=manifest["missing"])
