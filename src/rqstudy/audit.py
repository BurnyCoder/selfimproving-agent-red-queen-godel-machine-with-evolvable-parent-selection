"""Model-free numerical checks for the registered crossed, class-stratified analysis."""

from math import comb
from statistics import quantiles
from datetime import date
import os
import sqlite3
import subprocess
from xml.etree import ElementTree
import numpy as np

from .analysis import crossed_interval
from .data import digest, dump, load


def independent_interval(left, right, labels, draws=10000, seed=1701):
    """Weight each cell by its row and column multiplicities (Owen's pigeonhole bootstrap)."""
    diff = np.asarray(left) - np.asarray(right)
    n, m = diff.shape
    strata = [
        [i for i, label in enumerate(labels) if label == target]
        for target in ("accept", "reject")
    ]
    rng = np.random.default_rng(seed)
    values = []
    # Match the registered generator's stream in blocks. Evaluate its statistic independently.
    for start in range(0, draws, 200):
        size = min(200, draws - start)
        row_draws = rng.integers(n, size=(size, n))
        column_draws = [rng.integers(len(s), size=(size, len(s))) for s in strata]
        for i, chosen in enumerate(row_draws):
            rows = np.bincount(chosen, minlength=n)
            cols = np.zeros(m, dtype=int)
            for stratum, chosen_cols in zip(strata, column_draws, strict=True):
                cols[stratum] = np.bincount(chosen_cols[i], minlength=len(stratum))
            values.append(100 * int(rows @ diff @ cols) / (n * m))
    cuts = quantiles(values, n=40, method="inclusive")
    return dict(mean_pp=100 * int(diff.sum()) / (n * m), low=cuts[0], high=cuts[-1])


def exact_seed_coverage():
    """Enumerate the ideal percentile interval for eight independent equiprobable ±1 row effects."""
    n = 8
    covered = []
    for k in range(n + 1):
        # Conditional on k positive rows, resampled positives follow Binomial(n,k/n).
        weights = [comb(n, j) * k**j * (n - k) ** (n - j) for j in range(n + 1)]
        lower = next(j for j in range(n + 1) if 40 * sum(weights[: j + 1]) >= n**n)
        upper = next(j for j in range(n + 1) if 40 * sum(weights[: j + 1]) >= 39 * n**n)
        if lower <= n // 2 <= upper:
            covered.append(k)
    count = sum(comb(n, k) for k in covered)
    return dict(
        covered_positive_counts=covered,
        covered=count,
        total=2**n,
        fraction=count / 2**n,
    )


def numerical_checks():
    """Use deterministic synthetic correctness arrays with interacting seed and paper effects."""
    i, j = np.indices((8, 10))
    left = ((i + 2 * j + i * j) % 7 < 4).astype(int)
    right = ((3 * i + j) % 5 < 3).astype(int)
    labels = ["accept", "reject"] * 5
    result = crossed_interval(left, right, labels)
    ref = independent_interval(left, right, labels)
    if not all(abs(result[k] - ref[k]) < 1e-10 for k in ref):
        raise ValueError("Independent crossed-bootstrap calculation disagrees")
    return dict(
        kind="synthetic arithmetic check",
        draws=10000,
        seed=1701,
        paired_shape=list(left.shape),
        production={k: result[k] for k in ref},
        independent=ref,
        seed_only_exact_coverage=exact_seed_coverage(),
    )


def audit_results(project, evals, contrasts):
    """Persist independent computations, including a second calculation for each real contrast."""
    checks = numerical_checks()
    panels = {e["name"]: e for e in evals if e["split"] == "test" and e["complete"]}
    reproduced = {}
    for key, result in contrasts.items():
        suffix = "-tokens" if key == "matched_tokens" else ""
        arms = []
        for arm in ("editable", "native"):
            matrix = []
            for seed in result["seeds"]:
                rows = sorted(
                    panels[f"confirm-{seed:02d}-{arm}{suffix}"]["rows"],
                    key=lambda r: r["id"],
                )
                labels = [r["label"] for r in rows]
                matrix.append([int(r["decision"] == r["label"]) for r in rows])
            arms.append(matrix)
        ref = independent_interval(*arms, labels)
        if not all(abs(ref[k] - result[k]) < 1e-10 for k in ref):
            raise ValueError(f"{key} interval failed independent reproduction")
        reproduced[key] = ref
    data = dict(
        numerical=checks,
        reproduced_study_contrasts=reproduced,
        manuscript_evidence="manuscript-audit.json",
    )
    dump(project / "results/scientific-audit.json", data)
    return data


def record_observation(project):
    """Capture aggregate host evidence once for this review, without copying private traffic."""
    if os.getenv("RQGM_DISABLE_MODEL") != "1":
        raise ValueError("Set RQGM_DISABLE_MODEL=1 when recording the offline audit")
    private = project / ".private"
    uri = (private / "inference.sqlite").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        counts = {
            table: db.execute("SELECT COUNT(*) FROM " + table).fetchone()[0]
            for table in ("calls", "memo", "attempts")
        }
    logs = []
    for path in sorted((project / "logs").glob("preflight-*.log")):
        raw = path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        logs.append(
            dict(
                file=path.name,
                sha256=digest(raw),
                synthetic_requests=sum(
                    " hardware/synthetic preflight request " in line for line in lines
                ),
                completions=sum("] complete " in line for line in lines),
            )
        )
    tests = {}
    for suite in ("study", "upstream"):
        path = private / f"audit-{suite}.xml"
        doc = ElementTree.parse(path).find("testsuite")
        tests[suite] = {
            key: int(doc.attrib[key])
            for key in ("tests", "failures", "errors", "skipped")
        }
        tests[suite]["cases"] = [node.get("name") for node in doc.findall("testcase")]
        tests[suite]["junit_sha256"] = digest(path.read_bytes())
        if (
            tests[suite]["tests"] == 0
            or tests[suite]["errors"]
            or tests[suite]["failures"]
        ):
            raise ValueError(f"{suite} tests did not pass")
    base = project / "results"
    evidence = dict(
        date=date.today().isoformat(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
        ledger_counts=counts,
        preflight_logs=logs,
        tests=tests,
        model_access_disabled=True,
        search_files=len(list((base / "runs").glob("*.json"))),
        endpoint_rows=sum(
            len(load(p)["rows"]) for p in (base / "evaluations").glob("*.json")
        ),
    )
    dump(base / "audit-observation.json", evidence)
    return evidence
