from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

TOKEN_SPECS = [
    ("m", "m"),
    ("m0", "m0_"),
    ("ef_construct", "efc"),
    ("diversity_alpha", "a"),
    ("diversity_alpha_low", "alow"),
    ("diversity_alpha_high", "ahigh"),
    ("level_scale", "l"),
    ("max_level_cap", "maxlevel"),
    ("neighbor_scan_cap", "cap"),
    ("diversity_prune_floor", "prunefloor"),
]


def format_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
    else:
        text = str(value)
    return text.replace(".", "p").replace("-", "n")


def canonical_name(entry: Dict[str, Any], base_name: str, metric: str) -> str:
    tokens = [base_name, metric.lower()]
    for field, prefix in TOKEN_SPECS:
        value = entry.get(field)
        if value is None:
            continue
        tokens.append(f"{prefix}{format_value(value)}")
    extras = entry.get("extras", [])
    for extra in extras:
        tokens.append(str(extra))
    return "_".join(tokens) + ".bin"


def expand_grid_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    axes: Dict[str, List[Any]] = entry.get("axes", {})
    if not axes:
        return []
    axis_order = entry.get("axis_order") or list(axes.keys())
    axis_lists: List[List[Any]] = []
    for axis in axis_order:
        values = axes.get(axis)
        if values is None:
            raise ValueError(f"axis '{axis}' missing values in grid entry")
        axis_lists.append(values)
    if not axis_lists:
        return []
    defaults = dict(entry.get("defaults", {}))
    prefix = entry.get("label_prefix", entry.get("label", "grid"))
    expanded: List[Dict[str, Any]] = []
    for combination in product(*axis_lists):
        combo_entry = dict(defaults)
        for axis, value in zip(axis_order, combination):
            combo_entry[axis] = value
        combo_entry["label"] = build_grid_label(prefix, axis_order, combination)
        if should_skip_combo(combo_entry):
            continue
        expanded.append(combo_entry)
    return expanded


def build_grid_label(prefix: str, axis_order: List[str], values: Iterable[Any]) -> str:
    tokens = []
    for axis, value in zip(axis_order, values):
        clean_axis = axis.lower().replace(" ", "_")
        tokens.append(f"{clean_axis}{format_value(value)}")
    if tokens:
        return f"{prefix}_{'_'.join(tokens)}"
    return prefix


def should_skip_combo(entry: Dict[str, Any]) -> bool:
    m = entry.get("m")
    m0 = entry.get("m0")
    if isinstance(m, (int, float)) and isinstance(m0, (int, float)):
        if m > m0:
            return True
    return False


def manifest_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = list(manifest.get("builds", []))
    for grid in manifest.get("grids", []):
        entries.extend(expand_grid_entry(grid))
    return entries
