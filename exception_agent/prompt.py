"""Builds the per-step prompt: fixed guidance + the building's current state."""
import json
import statistics
from pathlib import Path

from .actions import ACTION_SCHEMA
from .config import GateThresholds
from .gate import pass_conditions

GUIDANCE_PATH = Path(__file__).with_name("guidance.md")


def system_prompt(thresholds: GateThresholds, guidance_path: Path = GUIDANCE_PATH) -> str:
    gate = "\n".join(f"- {line}" for line in pass_conditions(thresholds))
    return (f"{guidance_path.read_text().strip()}\n\n"
            f"## Quality gate (fixed; you cannot change it)\n{gate}\n\n"
            f"## Output JSON schema\n{json.dumps(ACTION_SCHEMA, ensure_ascii=False)}")


def dsm_summary(building: dict, cells: int = 12) -> str:
    """Height statistics plus a coarse text grid, so the LLM can see where the roof is high."""
    samples = building.get("dsm_samples") or []
    if not samples:
        return "not available (no 3D reference for this building; heights cannot be measured)"
    g = building["ground_z"]
    h = sorted(z - g for _, _, z in samples)
    q = statistics.quantiles(h, n=10) if len(h) > 1 else [h[0]] * 9
    xs, ys = [p[0] for p in building["footprint"]], [p[1] for p in building["footprint"]]
    x0, y1 = min(xs), max(ys)
    cell = max(max(xs) - x0, y1 - min(ys), 1e-6) / cells
    sums: dict[tuple[int, int], list[float]] = {}
    for x, y, z in samples:
        sums.setdefault((int((y1 - y) / cell), int((x - x0) / cell)), []).append(z - g)
    n_rows = max(r for r, _ in sums) + 1
    n_cols = max(c for _, c in sums) + 1
    rows = [" ".join(f"{sum(sums[r, c]) / len(sums[r, c]):3.0f}" if (r, c) in sums else "  ."
                     for c in range(n_cols)) for r in range(n_rows)]
    return (f"{len(h)} samples, height above ground: min {h[0]:.1f}, p10 {q[0]:.1f}, "
            f"median {statistics.median(h):.1f}, p90 {q[8]:.1f}, max {h[-1]:.1f} m\n"
            f"mean height per {cell:.1f} m cell; rows run north to south from y={y1:.1f}, "
            f"columns west to east from x={x0:.1f}; '.' is outside the building:\n" + "\n".join(rows))


def user_prompt(building: dict, prior: list[str], history: str, steps_left: int,
                last_parts: list[str] | None, attached: list[str] = ()) -> str:
    height = building.get("height_m")
    lines = [
        f"Building: {building['id']}",
        f"Footprint: {json.dumps(building['footprint'])}",
        f"Footprint bounding box: x {min(p[0] for p in building['footprint']):.1f} to "
        f"{max(p[0] for p in building['footprint']):.1f}, y {min(p[1] for p in building['footprint']):.1f} to "
        f"{max(p[1] for p in building['footprint']):.1f} (metres, +x east, +y north)",
        f"{building.get('dsm_label', 'DSM')}: {dsm_summary(building)}",
        (f"Building height: {height:g} m above ground (source: {building.get('height_source', 'unknown')})"
         if height is not None else "Building height: not given"),
        "Images attached below: " + ("; ".join(attached) or "none (roof image not available to you)"),
        "",
        "Earlier pipeline attempts:",
        *(prior or ["- none recorded"]),
        "",
        "Your steps so far:",
        history or "- none yet",
        "",
        f"Parts in your last build: {last_parts}" if last_parts else
        "You have no build yet, so fix_part is not available.",
        f"Steps left including this one: {steps_left}",
        "Answer with one JSON object.",
    ]
    return "\n".join(lines)
