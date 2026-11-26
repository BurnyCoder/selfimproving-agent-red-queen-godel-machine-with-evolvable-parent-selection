"""Reproduce the two attributed external result summaries from their pinned primary sources."""

import io
import json
import re
import urllib.request

from .data import digest, dump

HYPER = "https://arxiv.org/pdf/2603.19461v2"
RECONSTRUCTION = (
    "https://raw.githubusercontent.com/DreamFallenFlowers/OpenRQGM/"
    "feecb32d9fbdd1c60749ade9ecb47f971e12666e/docs/results/paper-matched-v5-summary.json"
)


def source_bytes(url):
    """Fetch publication bytes, with no executable content loaded from the remote source."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "rqgm-study-source-audit"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def verify_sources(project):
    """Extract Appendix E.5 as printed, including the authors' reported significance qualifier."""
    from pypdf import PdfReader

    raw = source_bytes(HYPER)
    page = re.sub(r"\s+", " ", PdfReader(io.BytesIO(raw)).pages[59].extract_text())
    comparisons = re.findall(r"to (0\.\d+) \(CI:\s*(0\.\d+)\s*[–−-]\s*(0\.\d+)\)", page)
    repeats = re.search(r"Each method is repeated across (\d+) runs", page)
    qualifier = re.search(r"not statistically significant \(p\s*>\s*(0\.\d+)\)", page)
    if len(comparisons) != 3 or not repeats or not qualifier:
        raise ValueError("Check HyperAgents Appendix E.5 against its original PDF")
    hyper = dict(
        url=HYPER,
        page=60,
        sha256=digest(raw),
        runs=int(repeats[1]),
        reported_p_greater_than=float(qualifier[1]),
        scores={
            arm: dict(score=float(v), low=float(lo), high=float(hi))
            for arm, (v, lo, hi) in zip(
                ("fixed uniform", "editable", "handcrafted"), comparisons, strict=True
            )
        },
    )
    raw = source_bytes(RECONSTRUCTION)
    record = json.loads(raw)
    upstream = dict(
        url=RECONSTRUCTION,
        sha256=digest(raw),
        paper_comparison_valid=record["paper_comparison_valid"],
        search={
            k: record["search"][k]
            for k in ("validation_outcomes", "evaluator_replacements", "erased_records")
        },
    )
    dump(
        project / "results/published-source-evidence.json",
        dict(hyperagents=hyper, openrqgm=upstream),
    )
