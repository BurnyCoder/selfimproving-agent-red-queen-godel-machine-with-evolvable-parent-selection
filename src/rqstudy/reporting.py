"""Rebuild tables, plots, experiment records and website data from trusted result files."""

from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analysis import coverage, summary
from .audit import audit_results
from .data import dump, load
from .integrity import registered_splits
from .paper import make_paper

TITLE = "Editable expansion-parent selection in the Red Queen Gödel Machine"
SITE = "https://burnycoder.github.io/selfimproving-agent-red-queen-godel-machine-with-evolvable-parent-selection/"
COLORS = {
    "editable": "#087d76",
    "native": "#a54c32",
    "uniform": "#536889",
    "handcrafted": "#80549b",
    "fork-native": "#895811",
    "root": "#444444",
}


def write(path: Path, text: str):
    """Use UTF-8 consistently for Markdown viewed on Windows, GitHub and Pages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def table(head, rows):
    """Render a GitHub table from the same cells consumed by the procedural report."""
    return "\n".join(
        "| "
        + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row)
        + " |"
        for row in [head, ["---"] * len(head), *rows]
    )


def evidence(run):
    """Count execution events separately from code changes and hypothetical replay draws."""
    chosen = run["selection"]
    return dict(
        nodes=len(run["nodes"]),
        edits=sum(bool(s["child"]) for s in chosen),
        invalid=sum(s["child"] is None for s in chosen),
        fallbacks=sum(len(s["failures"]) for s in chosen),
        selector_edits=sum("selector" in s["changed"] for s in chosen),
        older=sum(
            s.get("owner") is not None and s["owner"] != s["parent"] for s in chosen
        ),
        replay_diff=sum(s["published"] != s["released"] for s in chosen),
        draws=len(chosen),
        replacements=sum(bool(c["replacements"]) for c in run["checkpoints"]),
        erased=sum(c["erased_records"] for c in run["checkpoints"]),
    )


def example_rows(runs, evals):
    """Use the first recorded outputs per role/arm in file order to avoid picking highlights."""
    examples = {}
    for run in runs:
        group = examples.setdefault(run["arm"], {})
        for record in run["records"]:
            m = record["metadata"]
            group.setdefault(
                record["role_id"],
                dict(
                    id=m["paper"],
                    decision=m["decision"],
                    text=m.get("draft") or m["evidence"],
                    run=run["name"],
                ),
            )
    for e in evals:
        if e["rows"]:
            m = e["rows"][0]
            examples.setdefault(e["arm"], {}).setdefault(
                "endpoint",
                dict(
                    id=m["id"],
                    decision=m["decision"],
                    text=m["evidence"],
                    run=e["name"],
                ),
            )
    return examples


def architecture(path):
    """Draw the inheritance and scoring paths explicitly, including the moving evaluator."""
    fig, ax = plt.subplots(figsize=(10.7, 4.0), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 3.8))
    ax.axis("off")
    boxes = [
        (1.05, 2.85, "Current archive"),
        (3.55, 2.85, "Newest selector\nor fixed rule"),
        (6.1, 2.85, "Chosen parent's\nfour programs"),
        (8.7, 2.85, "Editor proposes\na child"),
        (1.05, 0.72, "Own-node\nvalidation"),
        (3.55, 0.72, "Manuscript +\nepoch reviewer"),
        (6.1, 0.72, "Human decision\nanchor"),
        (8.7, 0.72, "Checkpoint\nelection"),
    ]
    for x, y, label in boxes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            color="#14343a",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=.65", facecolor="#eff6f4", edgecolor="#5d8684"
            ),
        )
    edges = [
        ((1.85, 2.85), (2.73, 2.85)),
        ((4.43, 2.85), (5.15, 2.85)),
        ((7.02, 2.85), (7.85, 2.85)),
        ((1.05, 2.56), (1.05, 1.14)),
        ((1.8, 0.72), (2.6, 0.72)),
        ((7.06, 0.72), (7.88, 0.72)),
    ]
    for a, b in edges:
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops=dict(arrowstyle="->", color="#476363", lw=1.6),
        )
    # The reviewer task is measured on historical decisions independently of writer reward.
    ax.plot([1.05, 1.05, 6.1], [0.46, 0.12, 0.12], color="#476363", lw=1.4)
    ax.annotate(
        "",
        xy=(6.1, 0.50),
        xytext=(6.1, 0.12),
        arrowprops=dict(arrowstyle="->", color="#476363", lw=1.4),
    )
    ax.text(
        3.55,
        0.18,
        "Reviewer prediction against historical decision",
        ha="center",
        va="bottom",
        color="#14343a",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=(1.05, 3.18),
        xytext=(8.7, 3.18),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="bar,fraction=.052", color="#476363"
        ),
    )
    ax.text(4.9, 3.62, "Admit child and retain ancestry", ha="center", fontsize=9)
    ax.annotate(
        "",
        xy=(3.55, 1.16),
        xytext=(8.7, 1.16),
        arrowprops=dict(
            arrowstyle="->", connectionstyle="bar,fraction=.09", color="#476363"
        ),
    )
    ax.text(
        6.1, 1.77, "Promoted reviewer judges future drafts", ha="center", fontsize=9
    )
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plots(folder, runs, evals):
    """Show binary reward and accuracy on their full zero-to-one scale, marking epoch changes."""
    folder.mkdir(parents=True, exist_ok=True)
    architecture(folder / "architecture.png")
    for run in runs:
        name = run["name"]
        fig, (ax, tree) = plt.subplots(
            1,
            2,
            figsize=(10, 3.45),
            layout="constrained",
            gridspec_kw={"width_ratios": [1.25, 1]},
        )
        for label, s, t, color in (
            ("Reviewer accuracy", "review_success", "review_total", "#087d76"),
            ("Writer reward", "writer_success", "writer_total", "#ad5839"),
        ):
            points = [(r["n"], r[s] / r[t]) for r in run["trajectory"] if r[t]]
            if points:
                x, y = zip(*points)
                ax.step(x, y, where="post", label=label, color=color)
        for cp in run["checkpoints"]:
            if cp["replacements"]:
                ax.axvline(cp["checkpoint"], ls=":", color="#666")
        ax.set(
            xlabel="Scheduled validation evaluations",
            ylabel="Success fraction among active records",
            ylim=(-0.02, 1.02),
            title=name,
        )
        if ax.lines:
            ax.legend(fontsize=8, loc="best")
        positions = {}
        for i, node in enumerate(run["nodes"]):
            parent = node["parent"]
            depth = positions[parent][0] + 1 if parent in positions else 0
            positions[node["id"]] = (depth, i)
            if parent:
                px, py = positions[parent]
                tree.plot([px, depth], [py, i], lw=1, color="#a4b8b5")
            tree.scatter(depth, i, s=17, color=COLORS[run["arm"]])
            if len(run["nodes"]) < 22:
                tree.annotate(
                    node["id"],
                    (depth, i),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=6,
                )
        tree.set(
            xlabel="Generations from root",
            ylabel="Admission order",
            title="Workspace lineage",
        )
        tree.margins(x=0.2, y=0.08)
        fig.savefig(folder / (name + ".png"), dpi=115)
        plt.close(fig)
    complete = [e for e in evals if e["complete"] and not e["name"].endswith("-tokens")]
    if complete:
        fig, axes = plt.subplots(
            1, 2, figsize=(9, 3.5), layout="constrained", sharey=True
        )
        for ax, split in zip(axes, ("pilot", "test")):
            ax.set(
                title=split.capitalize() + " endpoint accuracy",
                ylim=(0, 1.02),
                ylabel="Correct fraction",
            )
            for j, (arm, color) in enumerate(COLORS.items()):
                values = [
                    sum(r["correct"] for r in e["rows"]) / len(e["rows"])
                    for e in complete
                    if e["split"] == split and e["arm"] == arm
                ]
                if values:
                    ax.scatter([j] * len(values), values, color=color, alpha=0.7)
            ax.set_xticks(
                range(len(COLORS)), list(COLORS), rotation=32, ha="right", fontsize=8
            )
        fig.savefig(folder / "endpoints.png", dpi=120)
        plt.close(fig)


def comparison_text(contrasts):
    """Apply the registered rule to the primary interval only after scoring."""
    result = contrasts.get("primary")
    if result is None:
        return "The primary hypothesis is untested. No complete editable–native pair has been scored on the sealed manuscripts."
    return (
        f"The mean paired difference is {result['mean_pp']:+.2f} percentage points "
        f"(95% interval {result['low']:+.2f} to {result['high']:+.2f}, {result['pairs']} seed pairs). "
        f"The registered interpretation is {result['interpretation']}."
    )


def audit_review(project, inputs, independent):
    """Keep the dated review and regenerate its numerical evidence from saved artifacts."""
    path = project / "reports/reviews/independent-scientific-audit.md"
    base = (
        path.read_text(encoding="utf-8").split("<!-- GENERATED: audit -->")[0].rstrip()
    )
    rows = []
    for split, periods in inputs["development"].items():
        for label in ("accept", "reject"):
            b, a = (periods[when][label] for when in ("before", "after"))
            counts = [
                [b[k], a[k]]
                for k in ("acknowledgments", "funding_language", "reviewer_thanks")
            ]
            rows.append(
                [
                    split,
                    label,
                    b["n"],
                    *[f"{x} → {y}" for x, y in counts],
                    f"{b['characters']['median']:,.1f} → {a['characters']['median']:,.1f}",
                ]
            )
    grid = table(
        [
            "Split",
            "Decision",
            "Papers",
            "Acknowledgments",
            "Funding phrase",
            "Reviewer thanks",
            "Median characters",
        ],
        rows,
    )
    check = independent["numerical"]
    c, sim = check["seed_only_exact_coverage"], check["independent"]
    body = (
        f"\n\n<!-- GENERATED: audit -->\n## Development and arithmetic evidence\n\n{grid}\n\n"
        "Counts are papers with at least one match, before and after the amendment. The funding diagnostic also matches sentences describing evidential support. "
        "The character-length difference persists. The input file also gives minima, maxima and all partition fingerprints. "
        "[Development evidence](../../results/manuscript-audit.json) · [Exact diagnostic definitions](../../src/rqstudy/disclosures.py)\n\n"
        f"The deterministic arithmetic fixture yielded {sim['mean_pp']:+.2f} pp with interval [{sim['low']:+.2f}, {sim['high']:+.2f}] "
        f"in both implementations using {check['draws']:,} draws and seed {check['seed']}. "
        f"In the separate equiprobable seed-only example, the ideal percentile interval covered zero in {c['covered']} of {c['total']} equiprobable samples "
        f"({100 * c['fraction']:.2f}%). This calculation illustrates finite-sample behavior. "
        "[Numerical receipt](../../results/scientific-audit.json) · [Independent arithmetic](../../src/rqstudy/audit.py)\n"
    )
    obs = project / "results/audit-observation.json"
    if obs.exists():
        o = load(obs)
        t = o["tests"]
        requests = sum(x["synthetic_requests"] for x in o["preflight_logs"])
        body += (
            f"\nOn {o['date']} the audit observed {o['search_files']} search files, {o['endpoint_rows']} endpoint predictions, "
            f"{requests} logged synthetic preflight request(s), and {o['ledger_counts']['calls']} completed broker events. "
            f"The offline suites ran {t['study']['tests']} study cases and {t['upstream']['tests']} pinned upstream cases, "
            f"with {sum(v['failures'] + v['errors'] for v in t.values())} failures or errors. "
            "[Host and test evidence](../../results/audit-observation.json)\n"
        )
    write(path, base + body)


def literature_tables(project):
    """Regenerate attributed published numbers separately from this study's outcomes."""
    data = load(project / "results/published-source-evidence.json")
    h, rq = data["hyperagents"], data["openrqgm"]
    path = project / "reports/literature-review.md"
    base = (
        path.read_text(encoding="utf-8")
        .split("<!-- GENERATED: literature -->")[0]
        .rstrip()
    )
    rows = [
        [arm, f"{s['score']:.3f}", f"{s['low']:.3f}–{s['high']:.3f}"]
        for arm, s in h["scores"].items()
    ]
    counts = rq["search"]
    write(
        path,
        base
        + "\n\n<!-- GENERATED: literature -->\n## Published comparison data\n\n"
        + f"HyperAgents reports joint paper-review and robotics test performance across {h['runs']} searches per condition. "
        + f"The authors report p > {h['reported_p_greater_than']} for differences between methods. "
        + f"[Appendix E.5]({h['url']}#page={h['page']})\n\n"
        + table(
            ["Published selector", "Joint test score", "Authors’ reported interval"],
            rows,
        )
        + f"\n\nOpenRQGM's coding reconstruction records {counts['validation_outcomes']:,} validation outcomes, "
        + f"{counts['evaluator_replacements']} evaluator replacements and {counts['erased_records']:,} erased records. "
        + f"The original record sets its direct paper-comparison validity flag to {str(rq['paper_comparison_valid']).lower()}. "
        + f"[Pinned reconstruction result]({rq['url']}) · [Extracted values and checksums](../results/published-source-evidence.json)",
    )


def report(project):
    """A single entry point rebuilds human-facing releases without calling a model."""
    registered_splits(project)
    runs, evals, scored, contrasts = summary(project)
    missing = coverage(evals)
    independent = audit_results(project, evals, contrasts)
    inputs = load(project / "results/manuscript-audit.json")
    literature_tables(project)
    reports = project / "reports"
    docs = project / "docs"
    docs.mkdir(exist_ok=True)
    examples = example_rows(runs, evals)
    completed = sum(r["complete"] for r in runs)
    complete_evals = sum(e["complete"] for e in evals)
    status = comparison_text(contrasts)
    scores = [
        [
            e["split"],
            e["name"],
            e["n"],
            f"{e['correct']}/{e['n']}",
            f"{100 * e['correct'] / e['n']:.1f}%",
            "complete" if e["complete"] else "partial",
        ]
        for e in scored
        if e["n"]
    ]
    score_table = (
        table(["Split", "Endpoint", "Papers", "Correct", "Accuracy", "Status"], scores)
        if scores
        else "No endpoint predictions have been recorded."
    )
    searches = []
    for run in runs:
        m = evidence(run)
        searches.append(
            [
                run["name"],
                f"{run['n']}/{run['budget']}",
                m["nodes"],
                m["selector_edits"],
                m["invalid"],
                m["fallbacks"],
                m["replacements"],
                m["erased"],
                f"{m['replay_diff']}/{m['draws']}",
            ]
        )
    search_table = (
        table(
            [
                "Search",
                "Validation",
                "Nodes",
                "Selector edits",
                "Invalid edits",
                "Fallback errors",
                "Replacements",
                "Erased",
                "Formula disagreements",
            ],
            searches,
        )
        if searches
        else "No search trajectory has been recorded."
    )
    computes = []
    for r in [*runs, *scored]:
        c = r.get("compute", {})
        if c:
            computes.append(
                [
                    r["name"] if "split" not in r else r["split"] + "-" + r["name"],
                    c["calls"],
                    c["actual_calls"],
                    c["input"],
                    c["output"],
                    c["actual_input"],
                    c["actual_output"],
                    f"{c['service_seconds'] / 3600:.4f}",
                    c["failed_attempts"],
                ]
            )
    compute_table = (
        table(
            [
                "Run",
                "Calls",
                "Actual calls",
                "Input",
                "Output",
                "Actual input",
                "Actual output",
                "Ollama service h",
                "Logged failed attempts",
            ],
            computes,
        )
        if computes
        else "Compute measurements begin with the first search or endpoint call."
    )
    route_rows = [
        [e["split"] + " " + e["name"], route, v["n"], f"{v['accuracy']:.3f}"]
        for e in scored
        for route, v in e["routes"].items()
    ]
    route_table = (
        table(["Endpoint", "Route", "Papers", "Accuracy"], route_rows)
        if route_rows
        else "Reading-route results are pending."
    )
    pair_rows = [
        [panel, seed, f"{pp:+.2f}", f"{value['low']:+.2f} to {value['high']:+.2f}"]
        for panel, value in contrasts.items()
        for seed, pp in zip(value["seeds"], value["per_seed_pp"])
    ]
    pair_table = (
        table(
            ["Comparison", "Seed", "Difference (pp)", "Cohort interval (pp)"], pair_rows
        )
        if pair_rows
        else "Paired effects require both frozen endpoints and a shared set of scored papers."
    )
    figures = docs / "figures"
    plots(figures, runs, evals)
    dump(
        project / "results" / "analysis.json",
        dict(
            scores=scored,
            contrasts=contrasts,
            status=status,
            coverage=missing,
            searches={r["name"]: evidence(r) for r in runs},
        ),
    )
    # The local dashboard polls a compact live snapshot. Long histories load only on demand.
    dump(
        docs / "data.json",
        dict(
            scores=scored,
            contrasts=contrasts,
            status=status,
            coverage=missing,
            development=inputs["development"],
            examples=examples,
            runs=[
                dict(
                    name=r["name"],
                    arm=r["arm"],
                    stage=r["stage"],
                    seed=r["seed"],
                    n=r["n"],
                    budget=r["budget"],
                    counts=evidence(r),
                    trajectory=r["trajectory"],
                    compute=r["compute"],
                    checkpoints=[
                        c["checkpoint"] for c in r["checkpoints"] if c["replacements"]
                    ],
                    nodes=[dict(id=v["id"], parent=v["parent"]) for v in r["nodes"]],
                    selection=[
                        {
                            k: s.get(k)
                            for k in (
                                "n",
                                "owner",
                                "parent",
                                "child",
                                "changed",
                                "error",
                                "source_hash",
                                "source",
                            )
                        }
                        for s in r["selection"]
                    ],
                )
                for r in runs
            ],
        ),
    )
    if not (docs / "live.json").exists():
        dump(docs / "live.json", {})
    summary_text = f"""# Study summary

{status}

## Endpoint accuracy

{score_table}

## Paired effects

{pair_table}

## Search and selector behavior

{search_table}

## Reading routes

{route_table}

## Measured compute

{compute_table}

Service hours sum Ollama's successful, uncached request durations. They include backend overhead. Logical tokens include broker memo hits. A fork includes its shared history in the logical comparison budget and attributes only newly incurred calls to its physical cost. Failed attempts are counted separately. [Ollama response fields](https://docs.ollama.com/api/chat)

[Registered interpretation](preregistration.md) · [Machine-readable analysis](../results/analysis.json)"""
    missing_text = " · ".join(
        f"{a} {', '.join(map(str, seeds))}"
        for a, seeds in missing["missing_seeds_by_arm"].items()
    )
    summary_text += (
        "\n\n## Sealed coverage\n\nMissing or unfinished seeds by arm — "
        + (missing_text or "none")
        + ". "
    )
    summary_text += "The sealed root is " + (
        "pending." if missing["root_missing"] else "complete."
    )
    summary_text += (
        " Missing paired token-budget endpoints by seed — "
        + (", ".join(map(str, missing["missing_token_pairs"])) or "none")
        + "."
    )
    token_rows = [
        [s["name"], s["checkpoint"], s["logical_search_tokens"], s["cutoff"]]
        for s in scored
        if s["cutoff"] is not None
    ]
    if token_rows:
        summary_text += "\n\n" + table(
            [
                "Sensitivity endpoint",
                "Checkpoint",
                "Logical search tokens",
                "Common token cap",
            ],
            token_rows,
        )
    summary_text += "\n\n[Independent audit and corpus controls](reviews/independent-scientific-audit.md)"
    write(reports / "summary.md", summary_text)
    audit_review(project, inputs, independent)
    progress = f"# Study progress\n\n{completed} completed searches and {complete_evals} completed endpoint evaluations. {status}\n\n{search_table}\n\n"
    budget_file = project / "results" / "budget.json"
    if budget_file.exists():
        b = load(budget_file)
        progress += f"The pilot timing rule selected {b['budget']} validation evaluations per search. Projected time from the pilot was {b['projections_hours'][str(b['budget'])]:.1f} hours. The measured search and evaluation timing inputs are in [the budget decision](../results/budget.json)."
    else:
        progress += "The first pilot block has a 14-hour target. The remaining-week estimate becomes available after all four pilot timings."
    write(
        reports / "progress.md",
        progress
        + "\n\n[Run matrix](planned-runs.md) · [Live dashboard](../docs/index.html)",
    )
    histories = reports / "experiments"
    index = []
    lessons = []
    for r in runs:
        name = r["name"]
        folder = histories / name
        m = evidence(r)
        index.append(f"- [{name}]({name}/experiment.md)")
        write(
            folder / "experiment.md",
            f"# {name}\n\n{r['arm']} expansion, seed {r['seed']}, frozen Qwen3.5-4B, {r['context']} tokens of context. The writer seeks acceptance by the epoch reviewer, and reviewer accuracy uses historical decisions. The endpoint follows the registered fifth-percentile rule.\n\n"
            + table(["Measure", "Observed"], [[k, v] for k, v in m.items()])
            + f"\n\n![Validation trajectory and lineage](../../../docs/figures/{name}.png)\n\n[Full raw archive, workspaces and examples](../../../results/runs/{name}.json) · [Run parameters](../../preregistration.md)",
        )
        observation = f"{name} admitted {m['edits']} edits in {m['draws']} windows. The selector source changed on {m['selector_edits']} edges, and the two child penalties chose different parents on {m['replay_diff']} snapshots. {m['replacements']} reviewer replacements erased {m['erased']} dependent records."
        lessons.append("- " + observation)
        write(
            folder / "learnings.md",
            "# Observations\n\n"
            + observation
            + "\n\nThe formula replay holds the archive and draw fixed. Its disagreements measure local decisions. The midpoint intervention measures downstream consequences.",
        )
    write(
        histories / "experiments.md",
        "# Experiment index\n\n"
        + status
        + "\n\n"
        + score_table
        + "\n\n"
        + search_table
        + "\n\n"
        + "\n".join(index)
        + "\n\n[Compute, reading routes and paired effects](../summary.md) · [Fixed settings and their rationale](../preregistration.md)",
    )
    write(
        histories / "learnings.md",
        "# Study observations\n\n"
        + (
            "\n".join(lessons)
            or "The printed and released child-count penalties differ. The planned replay compares their choices on identical archive snapshots. [Printed formula](https://arxiv.org/pdf/2603.19461v2#page=24) · [Released weights](https://github.com/facebookresearch/HyperAgents/blob/59a68f672dfb92c74aeb7e61535d776fb36e172d/utils/gl_utils.py#L571-L584)"
        )
        + "\n\n[Source audit](../state-of-the-art-review.md) · [Interpretation limits](../reviews/methods.md)",
    )
    make_paper(
        project,
        status,
        scores,
        searches,
        contrasts,
        examples,
        computes,
        route_rows,
        runs,
        inputs,
        independent,
    )
    readme = project / "README.md"
    base = (
        readme.read_text(encoding="utf-8")
        .split("<!-- GENERATED: outcomes -->")[0]
        .rstrip()
    )
    write(
        readme,
        base
        + "\n\n<!-- GENERATED: outcomes -->\n## Observed results\n\n"
        + status
        + "\n\n"
        + score_table
        + f"\n\n[Full report, compute and uncertainty](reports/summary.md) · [Examples and lineage explorer]({SITE}) · [Paper PDF](reports/paper.pdf)\n",
    )
    return dict(
        completed_searches=completed,
        completed_evaluations=complete_evals,
        primary=contrasts.get("primary"),
    )
