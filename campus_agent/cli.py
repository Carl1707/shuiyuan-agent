from __future__ import annotations

import argparse

from campus_agent.llm import load_local_env
from campus_agent.web import run_web_app


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    parser = argparse.ArgumentParser(
        prog="campus-agent",
        description="Run the read-only Shuiyuan community search Agent.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    run_web_app(host=args.host, port=args.port)
    return 0
