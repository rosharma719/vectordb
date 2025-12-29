#!/usr/bin/env python3
"""Parallel snapshot builder for knob manifests."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from snapshot_manifest_utils import canonical_name, manifest_entries

DEFAULT_COMMAND = "cargo test --release nytimes_build_and_persist_snapshot_only -- --ignored --nocapture"

AXIS_ENV_MAP = {
    "m": ("VECTORDB_{prefix}_M", True),
    "m0": ("VECTORDB_{prefix}_M0", True),
    "ef_construct": ("VECTORDB_{prefix}_EF_CONSTRUCT", True),
    "diversity_alpha": ("VECTORDB_DIVERSITY_ALPHA", False),
    "diversity_alpha_low": ("VECTORDB_DIVERSITY_ALPHA_LOW", False),
    "diversity_alpha_high": ("VECTORDB_DIVERSITY_ALPHA_HIGH", False),
    "diversity_prune_floor": ("VECTORDB_DIVERSITY_PRUNE_FLOOR", False),
    "level_scale": ("VECTORDB_LEVEL_SCALE", False),
    "max_level_cap": ("VECTORDB_MAX_LEVEL_CAP", False),
    "neighbor_scan_cap": ("VECTORDB_NEIGHBOR_SCAN_CAP_LEVEL0", False),
}


def load_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_env_for_entry(
    entry: Dict[str, object],
    persist_path: Path,
    prefix: str,
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    env[f"VECTORDB_{prefix}_PERSIST_PATH"] = str(persist_path)
    env[f"VECTORDB_{prefix}_ALLOW_BUILD"] = "1"
    env[f"VECTORDB_{prefix}_SAVE_SNAPSHOT"] = "1"
    for axis, (template, uses_prefix) in AXIS_ENV_MAP.items():
        value = entry.get(axis)
        if value is None:
            continue
        env_name = template.format(prefix=prefix) if uses_prefix else template
        env[env_name] = str(value)
    return env


def format_env(env: Dict[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in env.items())


def acquire_lock(path: Path) -> Tuple[bool, Path]:
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False, lock_path
    os.close(fd)
    return True, lock_path


def release_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def run_build(
    header: str,
    env: Dict[str, str],
    cmds: List[str],
    path: Path,
    skip_existing: bool,
) -> Tuple[str, int]:
    print(f"# {header}")
    print(format_env(env) + " " + " ".join(cmds))
    merged_env = os.environ.copy()
    merged_env.update(env)
    if skip_existing and path.exists():
        return header, 0
    locked, lock_path = acquire_lock(path)
    if not locked:
        return header, 0
    try:
        result = subprocess.run(cmds, env=merged_env)
        return header, result.returncode
    finally:
        release_lock(lock_path)


def run_parallel_builds(
    tasks: Iterable[Tuple[str, Dict[str, object], Path]],
    env_prefix: str,
    command_template: List[str],
    skip_existing: bool,
    jobs: int,
):
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for header, entry, path in tasks:
            env = build_env_for_entry(entry, path, env_prefix)
            futures.append(executor.submit(run_build, header, env, command_template, path, skip_existing))
        for fut in as_completed(futures):
            try:
                header, returncode = fut.result()
            except Exception as exc:
                print(f"command failed for {exc}", file=sys.stderr)
                sys.exit(1)
            if returncode != 0:
                print(
                    f"command failed for {header} (exit {returncode})",
                    file=sys.stderr,
                )
                sys.exit(returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit or run cargo commands for each manifest entry."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/nyt_snapshot_builds.json"),
        help="Knob manifest to read (default docs/nyt_snapshot_builds.json)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override the dataset directory declared in the manifest",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="NYT",
        help="VectorDB env var prefix for dataset-specific knobs (default NYT)",
    )
    parser.add_argument(
        "--cargo-cmd",
        type=str,
        default=DEFAULT_COMMAND,
        help="Base cargo invocation (default builds nytimes_build_and_persist_snapshot_only)",
    )
    parser.add_argument(
        "--cargo-extra",
        nargs="*",
        default=[],
        help="Additional command-line arguments to append to the cargo invocation",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run each command instead of just printing it",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip builds when the target snapshot already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after processing N combos (useful for sampling)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent builders to run when --execute is enabled",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    base_name = manifest.get("base_name", "index")
    metric = manifest.get("metric", "cosine").lower()
    dataset_dir = Path(args.dataset_dir or manifest.get("dataset_dir", "."))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    entries = manifest_entries(manifest)
    if args.limit:
        entries = entries[: args.limit]

    command_template = shlex.split(args.cargo_cmd) + args.cargo_extra
    tasks = []
    for entry in entries:
        name = canonical_name(entry, base_name, metric)
        path = dataset_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        header = entry.get("label") or path.name
        tasks.append((header, entry, path))

    if not tasks:
        print(
            "no tasks derived from manifest (check manifest/limit/skip-existing filters)",
            file=sys.stderr,
        )
        return

    if not args.execute:
        for header, entry, path in tasks:
            env = build_env_for_entry(entry, path, args.prefix.upper())
            print(f"# {header}")
            print(format_env(env) + " " + " ".join(command_template))
        return

    run_parallel_builds(
        tasks,
        args.prefix.upper(),
        command_template,
        args.skip_existing,
        max(1, args.jobs),
    )


if __name__ == "__main__":
    main()
