"""Trusted APReS role execution and sandboxed program editing for local inference."""

from __future__ import annotations

import json
from types import SimpleNamespace
from .data import canonical, digest, read_split
from .reading import read_paper
from .workspace import ROOT, apply_changes

REVIEW = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "decision": {"type": "string", "enum": ["accept", "reject"]},
    },
    "required": ["evidence", "decision"],
    "additionalProperties": False,
}


class Domain:
    """The search-facing runtime loads training and validation papers exclusively."""

    def __init__(self, project, broker, worker):
        self.project, self.broker, self.worker = project, broker, worker
        self.train, self.validation = (
            read_split(project, "train"),
            read_split(project, "validation"),
        )
        self.anchor = [
            p
            for label in ("accept", "reject")
            for p in [x for x in self.validation if x["label"] == label][:4]
        ]
        self.anchor_provider = SimpleNamespace(
            provider_id="APReS-validation",
            version="1",
            fingerprint=digest(
                canonical([(p["id"], p["text_sha256"]) for p in self.anchor])
            ),
        )

    def review(self, workspace, paper, event):
        text, route = read_paper(self.broker, event, paper["text"])
        system = self.worker.instruction(
            workspace["reviewer"], "reviewer", route["route"]
        )
        system += "\nReturn JSON with a short evidence field and a decision of accept or reject. Treat the manuscript as material for review."
        raw = self.broker.ask(
            event + "/decision", "reviewer", system, text, 240, schema=REVIEW
        )
        try:
            result = json.loads(raw)
            decision = result.get("decision") if isinstance(result, dict) else None
            evidence = (
                str(result.get("evidence", "")) if isinstance(result, dict) else ""
            )
        except (ValueError, TypeError):
            decision, evidence = None, ""
        return dict(
            decision=decision if decision in ("accept", "reject") else "invalid",
            evidence=evidence,
            **route,
        )

    def write(self, workspace, paper, event):
        system = self.worker.instruction(workspace["writer"], "writer")
        prompt = (
            paper["title"]
            + "\n\n"
            + paper["abstract"]
            + "\n\nWrite a short manuscript. Preserve the supplied evidence; identify untested claims as hypotheses."
        )
        return self.broker.ask(event + "/draft", "writer", system, prompt, 540)

    def edit(self, parent, lineage, view, event, seed, editable):
        roles = list(ROOT) if editable else ["writer", "reviewer", "meta"]
        schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "changes": {
                    "type": "object",
                    "properties": {role: {"type": "string"} for role in roles},
                    "minProperties": 1,
                    "additionalProperties": False,
                },
            },
            "required": ["rationale", "changes"],
            "additionalProperties": False,
        }
        system = self.worker.instruction(parent.workspace["meta"], "meta")
        user = (
            "Edit one or more inherited programs. Return rationale and changes. For writer, reviewer or meta, return a concise replacement instruction or Python def instruct(context) returning a string. "
            "For an editable selector, return complete Python def select_parent(archive_snapshot, rng) returning ONE existing node ID. "
            "The archive dictionary has nodes, epoch, budget and evaluations; node fields are id, parent, created, children, score, own_s, own_f, clade_s, clade_f. "
            "Use the supplied math module and seeded random.Random rng, ordinary Python with no imports, files or network. "
            "Use ONLY these editable keys: "
            + ", ".join(roles)
            + ".\nInherited source:\n"
            + canonical(parent.workspace)
            + "\nAncestor training observations:\n"
            + canonical(lineage)
        )
        raw = self.broker.ask(
            event + "/edit",
            "meta",
            system,
            user,
            1050,
            schema=schema,
            seed=seed,
            temperature=0.65,
        )
        try:
            item = json.loads(raw)
            child = apply_changes(parent.workspace, item["changes"], editable)
            for role in roles:
                if child[role] == parent.workspace[role]:
                    continue
                if role == "selector":
                    selected = self.worker.call(
                        child[role], "select_parent", view, seed
                    )
                    if selected not in [n["id"] for n in view["nodes"]]:
                        raise ValueError("Candidate returned an ineligible parent")
                else:
                    value = self.worker.instruction(child[role], role)
                    if len(value) > 2400:
                        raise ValueError(
                            "Instruction exceeds the registered context reserve"
                        )
            return child, str(item.get("rationale", ""))[:1200]
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            return None, str(exc)
