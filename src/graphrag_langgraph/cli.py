"""
Minimal CLI surface that mirrors the upstream GraphRAG commands but delegates to
the LangGraph-backed engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from graphrag.config.enums import IndexingMethod

from .engine import GraphRAGEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag-langgraph",
        description="GraphRAG with LangGraph orchestration",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to graphrag.yml (defaults to upstream search rules).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root; mirrors upstream CLI behaviour.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Run the LangGraph indexing pipeline.")
    index_parser.add_argument(
        "--method",
        type=str,
        default=IndexingMethod.Standard.value,
        choices=[m.value for m in IndexingMethod],
        help="Indexing method (standard/fast/update variants).",
    )
    index_parser.add_argument(
        "--update",
        action="store_true",
        help="Run in update mode (applies -update workflows).",
    )
    index_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    index_parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Enable memory profiling (if supported).",
    )

    query_parser = subparsers.add_parser(
        "query", help="Run the LangGraph query pipeline."
    )
    query_parser.add_argument("question", type=str, help="User question to answer.")
    query_parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "global", "local", "basic", "drift"],
        help="Query mode.",
    )
    query_parser.add_argument(
        "--community-level",
        type=int,
        default=None,
        help="Community level (applies to global/local/drift).",
    )
    query_parser.add_argument(
        "--dynamic-community-selection",
        action="store_true",
        help="Enable dynamic community selection for global search.",
    )
    query_parser.add_argument(
        "--response-type",
        type=str,
        default="multiple paragraphs",
        help="Desired response format string.",
    )
    query_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> Any:
    parser = _build_parser()
    args = parser.parse_args(argv)

    engine = GraphRAGEngine(config_path=args.config, root=args.root)

    if args.command == "index":
        result = engine.index(
            pipeline_method=args.method,
            is_update_run=args.update,
            verbose=args.verbose,
            memory_profile=args.memory_profile,
        )
    elif args.command == "query":
        result = engine.query(
            question=args.question,
            mode=args.mode,
            community_level=args.community_level,
            dynamic_community_selection=args.dynamic_community_selection,
            response_type=args.response_type,
            verbose=args.verbose,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return None

    print(result)
    return result


if __name__ == "__main__":
    main()
