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


def dsm_summary(building: dict, max_points: int = 16) -> str:
    samples = building.get("dsm_samples") or []
    if not samples:
        return "not available (no 3D reference for this building; heights cannot be measured)"
    g = building["ground_z"]
    h = sorted(z - g for _, _, z in samples)
    q = statistics.quantiles(h, n=10) if len(h) > 1 else [h[0]] * 9
    step = max(1, len(samples) // max_points)
    pts = "; ".join(f"({x:.1f}, {y:.1f}): {z - g:.1f}" for x, y, z in samples[::step][:max_points])
    return (f"{len(h)} roof samples, height above ground: min {h[0]:.1f}, p10 {q[0]:.1f}, "
            f"median {statistics.median(h):.1f}, p90 {q[8]:.1f}, max {h[-1]:.1f} m\n"
            f"sample points (x, y): height -> {pts}")


def user_prompt(building: dict, prior: list[str], history: str, steps_left: int,
                last_parts: list[str] | None) -> str:
    refs = building.get("refs") or {}
    lines = [
        f"Building: {building['id']}",
        f"Footprint: {json.dumps(building['footprint'])}",
        f"DSM: {dsm_summary(building)}",
        "Source files: " + (", ".join(f"{k}={v}" for k, v in refs.items()) or "none"),
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
