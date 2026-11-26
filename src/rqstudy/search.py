"""October RQGM search driven by the pinned OpenRQGM archive and snapshot format.

Algorithm 1/B.2: distinguish subtree expansion evidence from own-node validation.
HyperAgents Algorithm 5: newest admitted selector chooses the parent that is edited.
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from rqgm.config import RQGMConfig
from rqgm.engine import RQGM
from rqgm.models import (
    AnchorScore,
    CheckpointEvent,
    EvaluatorCandidate,
    EvaluatorSlot,
    RoleTask,
    UtilityRecord,
    WorkspaceNode,
)
from rqgm.persistence import restore_state, snapshot
from rqgm.scheduler import should_expand
from rqgm.statistics import best_belief

from .data import canonical, digest, dump, load
from .inference import now
from .policies import archive_view, choose_parent, choose_thompson
from .workspace import ROOT


def stable_seed(*parts) -> int:
    """Derive event-specific RNG streams so an edit cannot shift evaluation draws."""
    return int(digest(canonical(parts))[:8], 16) & 0x7FFFFFFF


def run_name(stage, seed, arm):
    return f"{stage}-{seed:02d}-{arm}"


class Study:
    """Operate the upstream core state with the v3 scheduler and a local domain."""

    def __init__(self, project, domain, stage, seed, arm, budget, fork_from=None):
        self.project, self.domain, self.arm, self.seed = project, domain, arm, seed
        self.name = run_name(stage, seed, arm)
        self.directory = project / ".private" / "runs" / self.name
        self.path = self.directory / "state.json"
        root_hash = digest(ROOT["reviewer"])
        root = EvaluatorCandidate(root_hash, "review", ROOT["reviewer"], "root", None)
        self.core = RQGM(
            deepcopy(ROOT),
            [
                RoleTask("reviewer", "human", "fixed"),
                RoleTask("writer", "manuscript", "learned", "review"),
            ],
            [EvaluatorSlot("review", "reviewer", root)],
            domain,
            RQGMConfig(
                validation_budget=budget,
                expansion_alpha=0.6,
                checkpoint_start=8,
                random_seed=seed,
            ),
            seed_node_id="root",
        )
        self.info = dict(
            stage=stage,
            seed=seed,
            arm=arm,
            budget=budget,
            context=domain.broker.ctx,
            backend=domain.broker.version,
            root_hash=root_hash,
            selection=[],
            trajectory=[],
            human={},
            endpoints=[],
            last_attempt=-1,
            compute_prefix={},
            elapsed=0.0,
            started=now(),
        )
        self.clock = time.monotonic()
        origin = self.path if self.path.exists() else fork_from
        if origin:
            restore_state(self.core, origin, require_resumable=True)
            saved = load(origin)["study"]
            for key in ("stage", "seed", "budget", "context", "backend"):
                if saved[key] != self.info[key]:
                    raise ValueError(f"Incompatible resume {key}")
            if origin == self.path and saved["arm"] != arm:
                raise ValueError("Incompatible arm")
            self.info = saved
            if origin != self.path:
                if (
                    arm != "fork-native"
                    or saved["arm"] != "editable"
                    or self.core.validation_outcomes != budget // 2
                ):
                    raise ValueError("Invalid midpoint fork")
                self.info["arm"] = arm
                self.info["compute_prefix"] = self.info["endpoints"][-1]["compute"]
        self.core.initialized = True

    @property
    def n(self):
        return self.core.validation_outcomes

    def compute(self):
        local = self.domain.broker.metrics("search")
        return {
            # The shared history counts toward the logical budget. Only new calls incur cost.
            key: val
            + (
                self.info["compute_prefix"].get(key, 0)
                if key in ("calls", "input", "output")
                else 0
            )
            for key, val in local.items()
        }

    def save(self):
        """One atomic document captures engine state, selector ledger and endpoint identities."""
        current = time.monotonic()
        self.info["elapsed"] += current - self.clock
        self.clock = current
        saved = snapshot(self.core)
        saved["study"] = self.info
        dump(self.path, saved)
        midpoint = self.directory / "midpoint.json"
        if (
            self.arm == "editable"
            and self.seed == 1
            and self.n == self.info["budget"] // 2
            and any(e["n"] == self.n for e in self.info["endpoints"])
            and not midpoint.exists()
        ):
            dump(midpoint, saved)
        self.export()

    def export(self):
        """Publish only domain outputs, programs and aggregate state; private source data stay local."""
        core = self.core
        nodes = [
            dict(
                id=n.node_id,
                parent=n.parent_id,
                step=n.created_at_step,
                workspace=n.workspace,
            )
            for n in core.archive.nodes.values()
        ]
        records = [asdict(r) for r in core.archive.records]
        result = {
            k: self.info[k]
            for k in (
                "stage",
                "arm",
                "seed",
                "budget",
                "selection",
                "endpoints",
                "trajectory",
                "elapsed",
                "started",
                "context",
                "backend",
            )
        }
        result.update(
            name=self.name,
            n=self.n,
            epoch=core.epoch_vector["review"],
            nodes=nodes,
            records=records,
            checkpoints=[asdict(e) for e in core.checkpoint_events],
            compute=self.compute(),
            complete=self.n == self.info["budget"]
            and any(e["n"] == self.n for e in self.info["endpoints"]),
        )
        dump(self.project / "results" / "runs" / (self.name + ".json"), result)
        dump(
            self.project / "docs" / "live.json",
            dict(
                name=self.name,
                updated=now(),
                n=self.n,
                budget=result["budget"],
                epoch=result["epoch"],
                nodes=len(nodes),
                trajectory=self.info["trajectory"],
                compute=result["compute"],
                selections=[
                    {
                        k: s.get(k)
                        for k in ("n", "owner", "parent", "child", "changed", "error")
                    }
                    for s in result["selection"]
                ],
            ),
        )

    def training_feedback(self, parent):
        """Only selected parents collect training feedback; its labels never enter utility counts."""
        if parent.training_feedback:
            return
        ranked = sorted(
            self.domain.train,
            key=lambda p: stable_seed(self.seed, parent.node_id, p["id"], "train"),
        )
        feedback = []
        for i, paper in enumerate(ranked[:3]):
            event = f"search/train/{parent.node_id}/{i}"
            if i == 1:
                draft = self.domain.write(parent.workspace, paper, event)
                review = self.domain.review(
                    {"reviewer": self.core.slot_map["review"].incumbent.artifact},
                    {"text": draft},
                    event + "/grade",
                )
                feedback.append(
                    dict(
                        role="writer",
                        title=paper["title"],
                        decision=review["decision"],
                        evidence=review["evidence"][:450],
                    )
                )
            else:
                review = self.domain.review(parent.workspace, paper, event)
                feedback.append(
                    dict(
                        role="reviewer",
                        title=paper["title"],
                        predicted=review["decision"],
                        target=paper["label"],
                        evidence=review["evidence"][:450],
                    )
                )
        parent.training_feedback.extend(feedback)
        self.core.archive.training_feedback[parent.node_id] = feedback

    def expand(self):
        c = self.core
        view = archive_view(
            c.archive, c.epoch_vector["review"], self.n, self.info["budget"]
        )
        selection = choose_parent(
            "native" if self.arm == "fork-native" else self.arm,
            c.archive,
            view,
            np.random.default_rng(stable_seed(self.seed, self.n, "select")),
            self.domain.worker,
            fallback_allowance=self.info["budget"]
            - sum(len(s["failures"]) for s in self.info["selection"]),
        )
        parent = c.archive.nodes[selection["parent"]]
        self.training_feedback(parent)
        lineage = []
        node = parent
        while node and len(lineage) < 4:
            lineage.append(dict(node=node.node_id, training=node.training_feedback))
            node = c.archive.nodes.get(node.parent_id)
        child, reason = self.domain.edit(
            parent,
            lineage,
            view,
            f"search/expand/{self.n}",
            stable_seed(self.seed, self.n, "edit"),
            self.arm in {"editable", "fork-native"},
        )
        selection.update(n=self.n, child=None, changed=[], error=None)
        if child is None:
            selection["error"] = reason
        else:
            child_id = f"n{len(c.archive.nodes):04d}"
            c.archive.add_node(WorkspaceNode(child_id, child, parent.node_id, self.n))
            selection.update(
                child=child_id,
                reason=reason,
                changed=[k for k in child if child[k] != parent.workspace[k]],
            )
        self.info["selection"].append(selection)
        self.info["last_attempt"] = self.n
        self.save()

    def observe_human(self, workspace, paper, event):
        """Evidence is unique by reviewer program and paper, including checkpoint reuse."""
        key = digest(workspace["reviewer"])
        book = self.info["human"].setdefault(key, {})
        if paper["id"] not in book:
            pred = self.domain.review(workspace, paper, event)
            book[paper["id"]] = dict(
                correct=int(pred["decision"] == paper["label"]), **pred
            )
        return book[paper["id"]]

    def remaining(self, node_id, task):
        used = {
            r.metadata["paper"]
            for r in self.core.archive.valid_records(
                node_ids=[node_id], task_id=task.task_id
            )
        }
        return [p for p in self.domain.validation if p["id"] not in used]

    def evaluate(self):
        c = self.core
        rng = np.random.default_rng(stable_seed(self.seed, self.n, "measure"))
        view = archive_view(
            c.archive, c.epoch_vector["review"], self.n, self.info["budget"]
        )
        eligible = {
            r["id"]: [task for task in c.tasks if self.remaining(r["id"], task)]
            for r in view["nodes"]
        }
        view["nodes"] = [row for row in view["nodes"] if eligible[row["id"]]]
        node = c.archive.nodes[choose_thompson(view, rng, False)]
        task = c.archive.least_measured_cell(node.node_id, eligible[node.node_id], rng)
        choices = self.remaining(node.node_id, task)
        paper = min(
            choices, key=lambda p: stable_seed(self.seed, task.task_id, p["id"])
        )
        event = f"search/val/{self.n}/{node.node_id}/{task.task_id}"
        slot = c.slot_map["review"]
        if task.role_id == "reviewer":
            result = self.observe_human(node.workspace, paper, event)
            bit = result["correct"]
        else:
            # Preserve the manuscript across an epoch change for on-demand re-scoring.
            key = "writer/" + paper["id"]
            if key not in node.cached_artifacts:
                node.cached_artifacts[key] = self.domain.write(
                    node.workspace, paper, event
                )
            result = self.domain.review(
                {"reviewer": slot.incumbent.artifact},
                {"text": node.cached_artifacts[key]},
                event + "/grade",
            )
            bit = int(result["decision"] == "accept")
        metadata = dict(
            paper=paper["id"],
            route=result["route"],
            decision=result["decision"],
            evidence=result["evidence"],
            chunks=result["chunks"],
            covered=result["covered"],
            chars=result["chars"],
        )
        if task.role_id == "writer":
            metadata["draft"] = node.cached_artifacts["writer/" + paper["id"]]
        c.archive.add_record(
            UtilityRecord(
                f"v{self.n:04d}",
                node.node_id,
                task.role_id,
                task.task_id,
                bit,
                c.epoch_vector,
                task.evaluator_slot,
                slot.incumbent.candidate_id if task.evaluator_slot else None,
                paper["id"],
                metadata,
            )
        )
        c.validation_outcomes += 1
        rs, rf = c.archive.counts(role_id="reviewer")
        ws, wf = c.archive.counts(role_id="writer")
        self.info["trajectory"].append(
            dict(
                n=self.n,
                epoch=slot.epoch,
                review_success=rs,
                review_total=rs + rf,
                writer_success=ws,
                writer_total=ws + wf,
                nodes=len(c.archive.nodes),
                compute=self.compute(),
            )
        )
        self.save()

    def checkpoint(self):
        """Elect a frozen incumbent on the fixed packet, then freeze a specialist by total human evidence."""
        c = self.core
        slot = c.slot_map["review"]
        old = c.epoch_vector
        owners = {}
        for node in c.archive.nodes.values():
            owners.setdefault(digest(node.workspace["reviewer"]), node)
        anchor_scores = []
        specialist = {}
        for key, node in owners.items():
            packet = [
                self.observe_human(node.workspace, p, f"search/anchor/{key}/{p['id']}")[
                    "correct"
                ]
                for p in self.domain.anchor
            ]
            if len(packet) >= 5:
                s = sum(packet)
                anchor_scores.append(
                    AnchorScore(
                        key, s, len(packet) - s, best_belief(s, len(packet) - s, 0.05)
                    )
                )
            samples = self.info["human"][key].values()
            if len(samples) >= 5:
                s = sum(r["correct"] for r in samples)
                specialist[key] = best_belief(s, len(samples) - s, 0.05)
        if not anchor_scores or not specialist:
            raise RuntimeError("No reviewer met the five-paper anchor minimum")
        winner = max(
            anchor_scores,
            key=lambda a: (
                a.best_belief,
                a.candidate_id == slot.incumbent.candidate_id,
            ),
        )
        replaced = {}
        erased = 0
        if winner.candidate_id != slot.incumbent.candidate_id:
            key = winner.candidate_id
            replaced["review"] = (slot.incumbent.candidate_id, key)
            slot.incumbent = EvaluatorCandidate(
                key, "review", owners[key].workspace["reviewer"], owners[key].node_id
            )
            slot.epoch += 1
            erased = c.archive.invalidate_slot("review", self.n)
        c.checkpoint_events.append(
            CheckpointEvent(
                self.n, old, c.epoch_vector, replaced, erased, {"review": anchor_scores}
            )
        )
        key = max(specialist, key=specialist.get)
        self.info["endpoints"].append(
            dict(
                n=self.n,
                key=key,
                node=owners[key].node_id,
                belief=specialist[key],
                observed=len(self.info["human"][key]),
                compute=self.compute(),
            )
        )
        self.save()

    def run(self, stop_at=None):
        """Resume safely even if interrupted after an evaluation and before its checkpoint."""
        budget = self.info["budget"]
        stop = budget if stop_at is None else min(stop_at, budget)
        while True:
            if (
                self.n >= 8
                and (self.n & (self.n - 1) == 0 or self.n == budget)
                and not any(e.checkpoint == self.n for e in self.core.checkpoint_events)
            ):
                self.checkpoint()
            if self.n >= stop:
                break
            if self.info["last_attempt"] != self.n and should_expand(
                self.n, len(self.core.archive.nodes), 0.6
            ):
                self.expand()
                continue
            self.evaluate()
        return self
