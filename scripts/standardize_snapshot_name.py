#!/usr/bin/env python3
"""Generate canonical snapshot names from a manifest of build-time knobs."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from snapshot_manifest_utils import canonical_name, manifest_entries


def load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a simple JSON manifest of build knobs into canonical snapshot names."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/nyt_snapshot_builds.json"),
        help="Manifest describing datasets and builds (default docs/nyt_snapshot_builds.json)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Override the dataset directory declared inside the manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write a newline-separated manifest of canonical paths",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    base_name = manifest.get("base_name", "index")
    metric = manifest.get("metric", "cosine").lower()
    dataset_dir = Path(args.dataset_dir or manifest.get("dataset_dir", "."))
    builds = manifest_entries(manifest)

    canonical_paths: List[Path] = []
    for entry in builds:
        name = canonical_name(entry, base_name, metric)
        path = dataset_dir / name
        canonical_paths.append(path)
        label = entry.get("label")
        legacy = entry.get("legacy_names", [])
        print(path, end="")
        extras = []
        if label:
            extras.append(f"label={label}")
        if legacy:
            extras.append(f"legacy={','.join(legacy)}")
        if extras:
            print("  # " + " | ".join(extras))
        else:
            print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for path in canonical_paths:
                handle.write(str(path))
                handle.write("\n")
        print(f"Manifest written to {args.output}")


def expand_grid_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    axes = entry.get("axes", {})
    if not axes:
        return []
    axis_order = entry.get("axis_order") or list(axes.keys())
    axis_lists = []
    for axis in axis_order:
        values = axes.get(axis)
        if values is None:
            raise ValueError(f"axis '{axis}' missing values in grid entry")
        axis_lists.append(values)
    if not axis_lists:
        return []
    defaults = entry.get("defaults", {})
    prefix = entry.get("label_prefix", entry.get("label", "grid"))
    expanded = []
    for combination in product(*axis_lists):
        combo_entry = dict(defaults)
        for axis, value in zip(axis_order, combination):
            combo_entry[axis] = value
        combo_entry["label"] = build_grid_label(prefix, axis_order, combination)
        expanded.append(combo_entry)
    return expanded


def build_grid_label(prefix: str, axis_order: List[str], values: List[Any]) -> str:
    tokens = []
    for axis, value in zip(axis_order, values):
        clean_axis = axis.lower().replace(" ", "_")
        tokens.append(f"{clean_axis}{format_value(value)}")
    if tokens:
        return f"{prefix}_{'_'.join(tokens)}"
    return prefix


if __name__ == "__main__":
    main()
