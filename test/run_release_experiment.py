#!/usr/bin/env python
"""Single-run launcher for the public AgentDropoutV2 release.

This wrapper intentionally avoids internal multi-endpoint fan-out. It maps a
benchmark plus a method preset to one foreground experiment command.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "release_experiments.json"

ENDPOINTS = [
    ("selector_url", "SELECTOR_URL", True),
    ("selector_model", "SELECTOR_MODEL", True),
    ("selector_key", "SELECTOR_KEY", False),
    ("reasoning_url", "REASONING_URL", True),
    ("reasoning_model", "REASONING_MODEL", True),
    ("reasoning_key", "REASONING_KEY", False),
    ("supervisor_url", "SUPERVISOR_URL", True),
    ("supervisor_model", "SUPERVISOR_MODEL", True),
    ("supervisor_key", "SUPERVISOR_KEY", False),
    ("embedding_url", "EMBEDDING_URL", True),
    ("embedding_model", "EMBEDDING_MODEL", True),
    ("embedding_key", "EMBEDDING_KEY", False),
]

SECRET_FLAGS = {"--selector_key", "--reasoning_key", "--supervisor_key", "--embedding_key"}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def resolve_config_path(record: dict[str, Any], key: str, override: str | None) -> str:
    if override:
        return resolve_path(override)
    value = record.get(key)
    env_name = record.get(f"{key}_env")
    if env_name and os.environ.get(env_name):
        value = os.environ[env_name]
    if not value:
        raise SystemExit(
            f"Missing {key}. Set {env_name} or pass the matching command-line override."
        )
    return resolve_path(value)


def append_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None or value is False:
        return
    flag = "--" + key
    if value is True:
        cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def masked_command(cmd: list[str]) -> str:
    masked = []
    skip_next = False
    for idx, part in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if part in SECRET_FLAGS and idx + 1 < len(cmd):
            masked.extend([part, "***"])
            skip_next = True
        else:
            masked.append(part)
    return shlex.join(masked)


def build_command(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    try:
        benchmark = config["benchmarks"][args.benchmark]
    except KeyError as exc:
        raise SystemExit(f"Unknown benchmark: {args.benchmark}") from exc

    try:
        method = config["method_presets"][args.method]
    except KeyError as exc:
        raise SystemExit(f"Unknown method preset: {args.method}") from exc

    pool_name = args.pool or method.get("pool") or benchmark.get("default_pool")
    if not pool_name:
        raise SystemExit("No metric pool configured for this benchmark/method.")
    try:
        pool = config["metric_pools"][pool_name]
    except KeyError as exc:
        raise SystemExit(f"Unknown metric pool: {pool_name}") from exc

    script = resolve_path(benchmark["script"])
    input_file = resolve_config_path(benchmark, "input_file", args.in_file)
    metric_pool_file = resolve_config_path(pool, "metric_pool_file", args.metric_pool_file)
    embedding_cache_file = resolve_config_path(pool, "embedding_cache_file", args.embedding_cache_file)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{args.benchmark}_{args.method}.json"
    log_file = output_dir / f"{args.benchmark}_{args.method}.log"

    cmd = [
        sys.executable,
        script,
        "--in_file",
        input_file,
        "--out_file",
        str(out_file),
        "--log_file",
        str(log_file),
        "--metric_pool_file",
        metric_pool_file,
        "--embedding_cache_file",
        embedding_cache_file,
        "--max_turns",
        str(args.max_turns if args.max_turns is not None else benchmark.get("max_turns", 7)),
    ]

    for key, env_name, required in ENDPOINTS:
        value = os.environ.get(env_name)
        if not value and not required:
            value = "EMPTY"
        if required and not value:
            if args.dry_run:
                value = f"<{env_name}>"
            else:
                raise SystemExit(f"Missing required environment variable: {env_name}")
        cmd.extend(["--" + key, value])

    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    for key, value in method.get("args", {}).items():
        append_arg(cmd, key, value)

    return cmd


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description="Run one public AgentDropoutV2 experiment preset.")
    parser.add_argument("--benchmark", choices=sorted(config["benchmarks"]), help="Benchmark id.")
    parser.add_argument("--method", choices=sorted(config["method_presets"]), help="Method preset id.")
    parser.add_argument("--model-profile", choices=sorted(config["model_profiles"]), default=None)
    parser.add_argument("--pool", choices=sorted(config["metric_pools"]), default=None)
    parser.add_argument("--output-dir", default="test/results_release")
    parser.add_argument("--in-file", default=None)
    parser.add_argument("--metric-pool-file", default=None)
    parser.add_argument("--embedding-cache-file", default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and methods.")
    args = parser.parse_args()

    if args.list:
        print("Benchmarks:")
        for name in sorted(config["benchmarks"]):
            print(f"  {name}")
        print("\nMethods:")
        for name, spec in sorted(config["method_presets"].items()):
            print(f"  {name}: {spec.get('description', '')}")
        return 0

    if not args.benchmark or not args.method:
        parser.error("--benchmark and --method are required unless --list is used")

    cmd = build_command(args, config)
    print(masked_command(cmd))
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
