"""Public interface for data preparation, resumable search, evaluation and reporting."""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from dotenv import load_dotenv
from . import campaign as study
from .data import prepare
from .endpoints import freeze
from .reporting import report
from .audit import record_observation
from .sources import verify_sources


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rqgm-study", description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("prepare", "preflight", "report", "selftest"):
        commands.add_parser(name)
    audit = commands.add_parser("audit")
    audit.add_argument("--record-observation", action="store_true")
    audit.add_argument("--verify-sources", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--arm", choices=(*study.ARMS, "fork-native"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--stage", choices=("pilot", "confirm"), required=True)
    run.add_argument("--budget", type=int, required=True)
    run.add_argument("--stop-at", type=int)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--split", choices=("pilot", "test"), required=True)
    ev.add_argument("--name", required=True)
    commands.add_parser("freeze").add_argument(
        "--allow-incomplete", action="store_true"
    )
    commands.add_parser("campaign").add_argument("--publish", action="store_true")
    commands.add_parser("release").add_argument("--message", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", default=8765, type=int)
    args = parser.parse_args(argv)
    project = args.project.resolve()
    load_dotenv(project / ".env")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8", errors="backslashreplace", line_buffering=True
        )
    match args.action:
        case "prepare":
            result = {k: v["sha256"] for k, v in prepare(project)["splits"].items()}
        case "preflight":
            result = study.preflight(project)
        case "run":
            result = study.search(
                project, args.stage, args.seed, args.arm, args.budget, args.stop_at
            )
        case "evaluate":
            result = study.evaluate(project, args.split, args.name)
        case "freeze":
            result = freeze(project, args.allow_incomplete)["entries_sha"]
        case "campaign":
            result = study.campaign(project, args.publish)
        case "release":
            result = study.publish(project, args.message)
        case "report":
            result = report(project)
        case "audit":
            if args.verify_sources:
                verify_sources(project)
            if args.record_observation:
                record_observation(project)
            result = report(project)
        case "selftest":
            raise SystemExit(
                subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"], cwd=project
                ).returncode
            )
        case "serve":
            handler = functools.partial(
                SimpleHTTPRequestHandler, directory=str(project / "docs")
            )
            print(f"Dashboard at http://127.0.0.1:{args.port}/", flush=True)
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                server.serve_forever()
            return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
