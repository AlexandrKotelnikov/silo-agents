from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import ExperimentHarness
from .datasets import load_cases, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SiloAgents isolation benchmark")
    parser.add_argument("--corpus", type=Path, default=Path("benchmarks/corpus.jsonl"))
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/tasks.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/latest"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ExperimentHarness(load_records(args.corpus)).run(load_cases(args.cases))
    json_path, markdown_path = report.write(args.output)
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
