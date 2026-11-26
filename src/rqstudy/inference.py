"""Local streaming inference with replayable caching and per-run compute accounting.

Ollama returns prompt/evaluation token counts and nanosecond durations on the
last streaming record. https://docs.ollama.com/api/chat
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from .data import canonical, digest

MODEL = "qwen3.5:4b"
MODEL_DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"


def now() -> str:
    """Timestamp event boundaries in UTC; reports display their explicit time zone."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Broker:
    """Record full prompts and streamed outputs once for each logical call and attempt."""

    def __init__(self, project: Path, run: str, ctx: int):
        if os.getenv("RQGM_DISABLE_MODEL") == "1":
            raise RuntimeError("Model access is disabled for this offline process")
        self.project, self.run, self.ctx = project, run, ctx
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if urlparse(host).hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("This registration requires a loopback Ollama endpoint")
        self.client = httpx.Client(
            base_url=host, timeout=httpx.Timeout(1200, connect=10), trust_env=False
        )
        self.client.get("/api/version").raise_for_status()
        self.version = self.client.get("/api/version").json()["version"]
        models = self.client.get("/api/tags").json()["models"]
        if not any(
            row["name"] == MODEL and row["digest"] == MODEL_DIGEST for row in models
        ):
            raise RuntimeError("The registered Ollama digest is unavailable")
        project.joinpath(".private").mkdir(exist_ok=True)
        self.db = sqlite3.connect(project / ".private" / "inference.sqlite", timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS calls (run TEXT, event TEXT, key TEXT, phase TEXT, role TEXT, input INTEGER, output INTEGER, seconds REAL, cached INTEGER, PRIMARY KEY(run,event))"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS attempts (run TEXT,event TEXT,started TEXT,seconds REAL,error TEXT)"
        )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        project.joinpath("logs").mkdir(exist_ok=True)
        self.log = (project / "logs" / f"{run}-{stamp}.log").open(
            "a", encoding="utf-8", buffering=1
        )

    def write(self, text: str):
        """Flush verbatim model traffic to disk and the worker's terminal in real time."""
        self.log.write(text)
        self.log.flush()
        sys.stdout.write(text)
        sys.stdout.flush()

    def ask(
        self, event, role, system, user, cap, *, schema=None, seed=13, temperature=0.0
    ):
        payload = dict(
            model=MODEL,
            messages=[
                dict(role="system", content=system),
                dict(role="user", content=user),
            ],
            stream=True,
            think=False,
            keep_alive="30m",
            options=dict(
                num_ctx=self.ctx, num_predict=cap, seed=seed, temperature=temperature
            ),
        )
        if schema:
            payload["format"] = schema
        if role == "reader":
            compartment = "fixed-reader"
        elif event.startswith("search/"):
            compartment = "training" if event.startswith("search/train/") else "search"
        elif event.startswith("endpoint/"):
            compartment = "test" if self.run.startswith("eval-test-") else "pilot"
        else:
            compartment = self.run
        key = digest(canonical([compartment, MODEL_DIGEST, self.version, payload]))
        old = self.db.execute(
            "SELECT key FROM calls WHERE run=? AND event=?", (self.run, event)
        ).fetchone()
        if old and old[0] != key:
            raise RuntimeError(f"Resume produced a different request for {event}")
        saved = self.db.execute(
            "SELECT response FROM memo WHERE key=?", (key,)
        ).fetchone()
        self.write(
            f"\n[{now()}] {self.run} {event} {role} request {key}\n{canonical(payload)}\n"
        )
        if saved:
            response = json.loads(saved[0])
            self.write(f"[{now()}] cache replay\n{response['content']}\n")
        else:
            start = time.monotonic()
            parts, terminal = [], None
            try:
                # Count the complete two messages before requesting inference. Leave space
                # for the chat template, decoder and a fixed margin on every backend.
                from .reading import tokenizer

                codec = tokenizer(self.project)
                submitted = sum(
                    len(codec.encode(m["content"], add_special_tokens=False).ids)
                    for m in payload["messages"]
                )
                if submitted + cap + 128 > self.ctx:
                    raise ValueError(
                        "Complete input and reserved output exceed the frozen context"
                    )
                with self.client.stream("POST", "/api/chat", json=payload) as stream:
                    stream.raise_for_status()
                    for line in stream.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise RuntimeError(chunk["error"])
                        message = chunk.get("message", {})
                        if message.get("thinking"):
                            self.write(message["thinking"])
                        piece = message.get("content", "")
                        parts.append(piece)
                        self.write(piece)
                        if chunk.get("done"):
                            terminal = chunk
                if terminal is None:
                    raise RuntimeError("Ollama stream ended before final accounting")
                response = dict(
                    content="".join(parts),
                    input=terminal["prompt_eval_count"],
                    output=terminal.get("eval_count", 0),
                    seconds=terminal.get("total_duration", 0) / 1e9,
                    reason=terminal.get("done_reason", ""),
                )
                if response["input"] >= self.ctx - 1:
                    raise ValueError(
                        "Prompt reached the context boundary; check coverage"
                    )
            except Exception as exc:
                with self.db:
                    self.db.execute(
                        "INSERT INTO attempts VALUES(?,?,?,?,?)",
                        (self.run, event, now(), time.monotonic() - start, str(exc)),
                    )
                self.write(f"\n[{now()}] failed {exc}\n")
                raise
            self.write(
                f"\n[{now()}] complete {canonical({k: v for k, v in response.items() if k != 'content'})}\n"
            )
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO memo VALUES(?,?)", (key, canonical(response))
            )
            self.db.execute(
                "INSERT OR IGNORE INTO calls VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    self.run,
                    event,
                    key,
                    event.split("/")[0],
                    role,
                    response["input"],
                    response["output"],
                    response["seconds"],
                    int(bool(saved)),
                ),
            )
        return response["content"]

    def metrics(self, phase=None):
        """Count successful broker events, memo hits, and Ollama request duration in seconds."""
        params = [self.run]
        where = "run=?"
        if phase:
            where += " AND phase=?"
            params.append(phase)
        row = self.db.execute(
            "SELECT count(*),sum(input),sum(output),sum(1-cached),sum(input*(1-cached)),sum(output*(1-cached)),sum(seconds*(1-cached)) FROM calls WHERE "
            + where,
            params,
        ).fetchone()
        result = dict(
            zip(
                (
                    "calls",
                    "input",
                    "output",
                    "actual_calls",
                    "actual_input",
                    "actual_output",
                    "service_seconds",
                ),
                (v or 0 for v in row),
            )
        )
        result["failed_attempts"] = self.db.execute(
            "SELECT count(*) FROM attempts WHERE run=?", (self.run,)
        ).fetchone()[0]
        return result

    def close(self):
        self.db.close()
        self.client.close()
        self.log.close()
