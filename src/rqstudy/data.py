"""Prepare immutable, paper-disjoint inputs from the published HyperAgents CSV.

Global context: only the trusted host reads labeled files. The model receives a
formatted paper, an abstract, or selected training feedback through the broker.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path
from .disclosures import development_counts, remove_disclosures

REVISION = "59a68f672dfb92c74aeb7e61535d776fb36e172d"
DATA_URL = f"https://raw.githubusercontent.com/facebookresearch/HyperAgents/{REVISION}/domains/paper_review/dataset.csv"
DATA_HASH = "efa8e77721d6d41c5d9ea69aaab8f0c16664618876e2c9dfe897a3852bb242c9"
# Mastheads recur inside converted PDFs. Match just the status phrase, including
# its venue, year and nearby spacing, to preserve adjacent scientific text.
STATUS = re.compile(
    r"(?:published\s+as|under\s+review\s+as|accepted\s+as)\s+(?:an?\s+)?(?:conference|workshop|spotlight|oral|poster)\s+paper(?:\s+(?:at|for|in|to)\s+(?:ICLR|NeurIPS|ICML|[A-Z]{2,12})(?:\s+20\d{2})?)?",
    re.I,
)
METALINE = re.compile(
    r"(?im)^[ \t]*(?:final decision|acceptance decision|decision|meta[- ]review|openreview (?:id|forum)|venue status)\s*:[^\n]*$"
)
BLIND_HEADER = re.compile(
    r"(?:anonymous\s+authors?\s*)?paper\s+under\s+double[\s-]+blind\s+review", re.I
)
LINK = re.compile(
    r"https?\s*:\s*//\s*(?:www\.)?openreview\.net/(?:forum|pdf)\?id=[\w-]+", re.I
)
MANUSCRIPT = re.compile(
    r"\ATitle:\s*(.+?)\s*\n+\s*Abstract:\s*(.+?)(?=\n+\s*Section:|\Z)", re.S | re.I
)
SIZES = {"train": 100, "validation": 100, "test": 100, "pilot": 24}


def digest(value: str | bytes) -> str:
    """Return a content identity independent of platform newlines and Python hash seeds."""
    return hashlib.sha256(
        value.encode("utf-8") if isinstance(value, str) else value
    ).hexdigest()


def canonical(value: object) -> str:
    """Use one unambiguous representation for published hashes and cache keys."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def dump(path: Path, value: object) -> None:
    """Atomically publish readable JSON so observers see whole snapshots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def load(path: Path):
    """Read a trusted JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_split(project: Path, split: str):
    """Verify manuscript contents, extracted writer inputs, labels and registered order."""
    papers = load(project / ".private" / "data" / (split + ".json"))
    manifest = load(project / "results" / "split-manifest.json")["splits"][split]
    identity = [(p["id"], digest(p["text"]), p["label"]) for p in papers]
    if (
        digest(canonical(identity)) != manifest["sha256"]
        or [p["id"] for p in papers] != manifest["ids"]
        or len(papers) != manifest["count"]
        or [digest(canonical(row)) for row in identity] != manifest["item_sha256"]
    ):
        raise ValueError(f"{split} split fingerprint mismatch")
    for paper in papers:
        parsed = MANUSCRIPT.search(paper["text"])
        if (
            paper["text_sha256"] != digest(paper["text"])
            or not parsed
            or tuple(s.strip() for s in parsed.groups())
            != (
                paper["title"],
                paper["abstract"],
            )
        ):
            raise ValueError("Writer inputs differ from the fingerprinted manuscript")
    return papers


def registered_text(raw: str) -> str:
    """Reproduce the 20 November text for the original grouping and enrollment."""
    if raw.startswith(("'", '"')):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raw = raw.strip("'\"").replace("\\n", "\n")
    raw = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n")
    return LINK.sub(
        "", METALINE.sub("", BLIND_HEADER.sub("", STATUS.sub("", raw)))
    ).strip()


def clean(raw: str) -> str:
    """Apply the registered decoding followed by the frozen disclosure amendment."""
    return remove_disclosures(registered_text(raw))


def words(text: str) -> set[str]:
    """Normalize title and abstract words for duplicate detection."""
    return set(re.findall(r"\w+", text.casefold()))


def similar(left: set[str], right: set[str], threshold: float) -> bool:
    """Jaccard overlap rejects unrelated papers sharing a short generic title."""
    return bool(left and right) and len(left & right) >= threshold * len(left | right)


def unique_papers(rows):
    """Group versions transitively by ID, title, body, or near-matching title and abstract."""
    items, groups, keys, signatures = [], [], {}, []
    missing = 0

    def root(index):
        while groups[index] != index:
            groups[index] = groups[groups[index]]
            index = groups[index]
        return index

    for row in rows:
        body = registered_text(row["paper_text"])
        match = MANUSCRIPT.search(body)
        if not match or row["outcome"] not in {"accept", "reject"}:
            missing += 1
            continue
        title, abstract = (s.strip() for s in match.groups())
        index = len(items)
        groups.append(index)
        title_set, abstract_set = words(title), words(abstract)
        for key in (
            "id:" + row["question_id"],
            "title:" + " ".join(re.findall(r"\w+", title.casefold())),
            "body:" + digest(body),
        ):
            if key in keys:
                groups[root(index)] = root(keys[key])
            keys[key] = index
        for other, (t, a) in enumerate(signatures):
            if similar(t, title_set, 0.75) and similar(a, abstract_set, 0.82):
                groups[root(index)] = root(other)
        signatures.append((title_set, abstract_set))
        items.append(
            dict(
                id=row["question_id"],
                title=title,
                abstract=abstract,
                text=body,
                label=row["outcome"],
                text_sha256=digest(body),
            )
        )
    components = {}
    for i, paper in enumerate(items):
        components.setdefault(root(i), []).append(paper)
    # Conflicting historical outcomes cannot define a unique group target.
    stable = [
        min(group, key=lambda x: x["id"])
        for group in components.values()
        if len({x["label"] for x in group}) == 1
    ]
    return stable, dict(
        parsed=len(items),
        excluded_parse_or_label=missing,
        groups=len(components),
        retained_groups=len(stable),
    )


def prepare(project: Path) -> dict:
    """Download, verify, split and fingerprint the corpus before any model calls."""
    private = project / ".private" / "data"
    private.mkdir(parents=True, exist_ok=True)
    source = private / "source.csv"
    if not source.exists():
        with urllib.request.urlopen(DATA_URL, timeout=120) as response:
            payload = response.read()
        if digest(payload) != DATA_HASH:
            raise ValueError("Released CSV checksum mismatch")
        temporary = source.with_suffix(".part")
        temporary.write_bytes(payload)
        temporary.replace(source)
    raw = source.read_bytes()
    if digest(raw) != DATA_HASH:
        raise ValueError("Local CSV checksum mismatch")
    csv.field_size_limit(16 * 1024 * 1024)
    papers, audit = unique_papers(
        csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    )
    by_label = {
        label: sorted(
            (p for p in papers if p["label"] == label),
            key=lambda p: digest("rqgm-split-v1/42/" + p["id"]),
        )
        for label in ("accept", "reject")
    }
    prior = load(project / "results" / "split-manifest.2026-11-20.json")
    if prior["source_sha256"] != DATA_HASH or prior["audit"] != audit:
        raise ValueError("Original enrollment population changed")
    manifest = dict(source=DATA_URL, source_sha256=DATA_HASH, audit=audit, splits={})
    development = {}
    prepared = {}
    offset = 0
    for name, count in SIZES.items():
        subset = sorted(
            [
                p
                for pool in by_label.values()
                for p in pool[offset : offset + count // 2]
            ],
            key=lambda p: digest("order/" + p["id"]),
        )
        if Counter(p["label"] for p in subset) != {
            "accept": count // 2,
            "reject": count // 2,
        }:
            raise ValueError(f"Insufficient eligible data for {name}")
        ident = [(p["id"], p["text_sha256"], p["label"]) for p in subset]
        previous = prior["splits"][name]
        ids = [p["id"] for p in subset]
        if ids != previous["ids"] or digest(canonical(ident)) != previous["sha256"]:
            raise ValueError(f"{name}: original IDs, labels, order or text changed")
        cleaned = []
        for paper in subset:
            body = remove_disclosures(paper["text"])
            parsed = MANUSCRIPT.search(body)
            if not parsed:
                raise ValueError(f"Unparseable amended manuscript in {name}")
            title, abstract = (s.strip() for s in parsed.groups())
            cleaned.append(
                dict(
                    paper,
                    text=body,
                    title=title,
                    abstract=abstract,
                    text_sha256=digest(body),
                )
            )
        ident = [(p["id"], p["text_sha256"], p["label"]) for p in cleaned]
        manifest["splits"][name] = dict(
            count=count,
            sha256=digest(canonical(ident)),
            ids=ids,
            item_sha256=[digest(canonical(row)) for row in ident],
        )
        if name in ("train", "validation"):
            development[name] = dict(
                before=development_counts(subset), after=development_counts(cleaned)
            )
        prepared[name] = cleaned
        offset += count // 2
    # Validate enrollment for every partition before replacing any existing split.
    freeze_path = project / "results" / "freeze.json"
    if (
        freeze_path.exists()
        and load(freeze_path)["split_sha"] != manifest["splits"]["test"]["sha256"]
    ):
        raise ValueError("Prepared text differs from the frozen study")
    current_path = project / "results/split-manifest.json"
    if any(
        any((project / "results" / kind).glob("*.json"))
        for kind in ("runs", "evaluations")
    ):
        if not current_path.exists() or load(current_path) != manifest:
            raise ValueError(
                "Cannot change a manuscript panel after a recorded search or evaluation"
            )
    registered_ids = [p["id"] for group in prepared.values() for p in group]
    if len(set(registered_ids)) != len(registered_ids):
        raise ValueError("An enrolled ID occurs more than once")
    for name, papers in prepared.items():
        dump(private / f"{name}.json", papers)
    dump(
        project / "results" / "manuscript-audit.json",
        dict(
            source_sha256=DATA_HASH,
            manifest_sha256=digest(canonical(manifest)),
            cleaner_sha256=digest(
                Path(__file__).with_name("disclosures.py").read_text(encoding="utf-8")
            ),
            development=development,
            cross_split_id_overlap=len(registered_ids) - len(set(registered_ids)),
            splits={
                s: dict(
                    count=v["count"],
                    same_ids_order_labels=True,
                    before=prior["splits"][s]["sha256"],
                    after=v["sha256"],
                )
                for s, v in manifest["splits"].items()
            },
        ),
    )
    dump(project / "results" / "split-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    print({k: v["sha256"] for k, v in prepare(Path.cwd())["splits"].items()})
