"""Read-only measurements of a mesh against the input data (DSM, footprint).

These are hints for the LLM, not gate conditions: the DSM is input data, not
ground truth, and the gate only checks topology and BRICE. A bare similarity
score says a model is wrong but not in which direction; the hints do.
"""
import math

from .mesh import Mesh


def height_errors(mesh: Mesh, building: dict) -> list[float]:
    """Mesh top minus DSM at each sample; uncovered samples compare against the ground."""
    ground = building["ground_z"]
    return [(mesh.top_z_at(x, y) or ground) - z for x, y, z in building.get("dsm_samples") or []]


def dsm_proxy_similarity(mesh: Mesh, building: dict) -> float | None:
    """Placeholder for BRICE so the loop can run. NOT BRICE: 1 / (1 + height RMSE vs DSM)."""
    errs = height_errors(mesh, building)
    if not errs:
        return None
    return round(1.0 / (1.0 + math.sqrt(sum(e * e for e in errs) / len(errs))), 3)


def hints(mesh: Mesh, building: dict) -> list[str]:
    out = []
    errs = height_errors(mesh, building)
    if errs:
        samples = building["dsm_samples"]
        mean = sum(errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        if rmse < 0.3:
            out.append(f"roof heights agree with DSM (RMSE {rmse:.1f} m)")
        else:
            w = max(range(len(errs)), key=lambda i: abs(errs[i]))
            side = "higher" if mean > 0 else "lower"
            out.append(f"roof is {abs(mean):.1f} m {side} than DSM on average (RMSE {rmse:.1f} m); "
                       f"worst {errs[w]:+.1f} m at ({samples[w][0]:.1f}, {samples[w][1]:.1f})")
    worst, at = 0.0, None
    for x, y in building["footprint"]:
        d = min((math.hypot(vx - x, vy - y) for vx, vy, _ in mesh.vertices), default=math.inf)
        if d > worst:
            worst, at = d, (x, y)
    if worst > 0.1:
        out.append(f"source footprint vertex ({at[0]}, {at[1]}) is {worst:.1f} m from the nearest model vertex")
    return out
