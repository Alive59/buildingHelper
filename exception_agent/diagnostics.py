"""Read-only measurements of a mesh against optional input data (DSM, footprint).

These are hints for the LLM, not gate conditions. The gate only checks topology
and BRICE, and many areas have no 3D reference at all: without `dsm_samples`
the only hint left is the footprint one.
"""
import math

from .mesh import Mesh


def height_errors(mesh: Mesh, building: dict, key: str = "dsm_samples") -> list[float]:
    """Mesh top minus reference at each sample; uncovered samples compare against the ground."""
    ground = building["ground_z"]
    return [(mesh.top_z_at(x, y) or ground) - z for x, y, z in building.get(key) or []]


def stub_similarity(mesh: Mesh, building: dict) -> float | None:
    """Placeholder for BRICE so the loop can run. NOT BRICE.

    BRICE scores the mesh against the roof image. This scores it against
    `_stub_reference`, a hidden height field in the demo queue entries that the
    LLM never sees: 1 / (1 + height RMSE).
    """
    errs = height_errors(mesh, building, "_stub_reference")
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
    given = building.get("height_m")
    if given is not None:
        fp = building["footprint"]
        xs, ys = [p[0] for p in fp], [p[1] for p in fp]
        n = 15
        tops = [mesh.top_z_at(min(xs) + (i + 0.5) * (max(xs) - min(xs)) / n,
                              min(ys) + (j + 0.5) * (max(ys) - min(ys)) / n)
                for i in range(n) for j in range(n)]
        tops = [t - building["ground_z"] for t in tops if t is not None]
        if tops:
            mean = sum(tops) / len(tops)
            if abs(mean - given) > max(1.0, 0.15 * given):
                out.append(f"mean roof height of your model is {mean:.1f} m; the given building height "
                           f"is {given:g} m ({building.get('height_source', 'unknown')})")
    worst, at = 0.0, None
    for x, y in building["footprint"]:
        d = min((math.hypot(vx - x, vy - y) for vx, vy, _ in mesh.vertices), default=math.inf)
        if d > worst:
            worst, at = d, (x, y)
    if worst > 0.1:
        out.append(f"source footprint vertex ({at[0]}, {at[1]}) is {worst:.1f} m from the nearest model vertex")
    return out
