"""Quality gate interface, a stand-in implementation, and the LLM-facing formatter.

The real gate lives in the bulk pipeline and must be reused unchanged: wrap it in
a class with the same `evaluate` signature that returns Check records, and the
formatter and harness work as they are. StandInGate exists only so the loop can
run end to end before that adapter is written.
"""
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .config import GateThresholds
from .mesh import Mesh


@dataclass
class Check:
    name: str
    passed: bool | None  # None = not evaluated
    value: float | None = None
    threshold: float | None = None
    data: dict = field(default_factory=dict)


@dataclass
class GateReport:
    passed: bool
    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


class Gate(Protocol):
    thresholds: GateThresholds

    def evaluate(self, mesh: Mesh, building: dict) -> GateReport: ...


class StandInGate:
    def __init__(self, thresholds: GateThresholds = GateThresholds()):
        self.thresholds = thresholds

    def evaluate(self, mesh: Mesh, building: dict) -> GateReport:
        th = self.thresholds
        checks = [Check("faces", len(mesh.faces) <= th.max_faces, len(mesh.faces), th.max_faces)]
        checks += self._topology(mesh)
        checks.append(self._roof_height(mesh, building))
        checks.append(self._footprint(mesh, building))
        checks.append(Check("bmqi", None, threshold=th.bmqi_min))
        return GateReport(all(c.passed is not False for c in checks), checks)

    @staticmethod
    def _topology(mesh: Mesh) -> list[Check]:
        undirected, directed = defaultdict(list), defaultdict(int)
        degenerate = []
        for fi, (a, b, c) in enumerate(mesh.faces):
            if len({a, b, c}) < 3:
                degenerate.append(fi)
                continue
            for u, v in ((a, b), (b, c), (c, a)):
                undirected[(min(u, v), max(u, v))].append(fi)
                directed[(u, v)] += 1
        open_faces = sorted({f for fs in undirected.values() if len(fs) == 1 for f in fs})
        n_open = sum(1 for fs in undirected.values() if len(fs) == 1)
        over = [fs for fs in undirected.values() if len(fs) > 2]
        flipped = sum(1 for n in directed.values() if n > 1)
        bad = sorted({f for fs in over for f in fs} | set(degenerate))
        return [
            Check("watertight", n_open == 0, n_open, 0, {"faces": open_faces[:6]}),
            Check("manifold", not over and not degenerate and flipped == 0,
                  len(over) + len(degenerate) + flipped, 0,
                  {"non_manifold_edges": len(over), "degenerate_faces": len(degenerate),
                   "inconsistent_edges": flipped, "faces": bad[:6]}),
        ]

    def _roof_height(self, mesh: Mesh, building: dict) -> Check:
        samples = building.get("dsm_samples") or []
        if not samples:
            return Check("roof_height", None, threshold=self.thresholds.roof_height_rmse_m)
        ground = building["ground_z"]
        # Where the mesh does not cover a DSM roof sample, its height is the ground.
        errs = [(mesh.top_z_at(x, y) or ground) - z for x, y, z in samples]
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        worst = max(range(len(errs)), key=lambda i: abs(errs[i]))
        return Check("roof_height", rmse <= self.thresholds.roof_height_rmse_m, round(rmse, 2),
                     self.thresholds.roof_height_rmse_m,
                     {"mean_error_m": round(sum(errs) / len(errs), 2),
                      "worst_error_m": round(errs[worst], 2),
                      "worst_at": [round(v, 1) for v in samples[worst][:2]]})

    def _footprint(self, mesh: Mesh, building: dict) -> Check:
        fp = building["footprint"]
        worst, at = 0.0, None
        for x, y in fp:
            d = min((math.hypot(vx - x, vy - y) for vx, vy, _ in mesh.vertices), default=math.inf)
            if d > worst:
                worst, at = d, [x, y]
        return Check("footprint_vertex", worst <= self.thresholds.footprint_vertex_err_m,
                     round(worst, 2), self.thresholds.footprint_vertex_err_m, {"at": at})


# --- Milestone 1: gate report -> short failure reasons for the LLM ---------------

def _faces(c: Check) -> str:
    faces = c.data.get("faces") or []
    return f" (e.g. faces {', '.join(map(str, faces))})" if faces else ""


def _roof_reason(c: Check) -> str:
    mean = c.data.get("mean_error_m", 0.0)
    side = "too high" if mean > 0 else "too low"
    msg = (f"roof height off by {abs(mean):.1f} m vs DSM (mesh {side} on average), "
           f"RMSE {c.value:.1f} m > {c.threshold:.1f} m")
    if "worst_at" in c.data:
        x, y = c.data["worst_at"]
        msg += f"; worst {c.data['worst_error_m']:+.1f} m at ({x}, {y})"
    return msg


def _manifold_reason(c: Check) -> str:
    bits = [f"{c.data[k]} {label}" for k, label in (
        ("non_manifold_edges", "non-manifold edges"),
        ("degenerate_faces", "degenerate faces"),
        ("inconsistent_edges", "edges with inconsistent face orientation")) if c.data.get(k)]
    return f"not manifold: {', '.join(bits) or 'failed'}{_faces(c)}"


_REASONS = {
    "faces": lambda c: f"{c.value:.0f} faces > {c.threshold:.0f}",
    "watertight": lambda c: f"not watertight: {c.value:.0f} open boundary edges{_faces(c)}",
    "manifold": _manifold_reason,
    "roof_height": _roof_reason,
    "footprint_vertex": lambda c: (
        f"footprint vertex off by {c.value:.1f} m > {c.threshold:.1f} m"
        + (f" at ({c.data['at'][0]}, {c.data['at'][1]})" if c.data.get("at") else "")),
    "bmqi": lambda c: f"BMQI {c.value:.2f} < {c.threshold:.2f}",
}


def failure_reasons(report: GateReport) -> list[str]:
    """One short, specific line per failed check."""
    out = []
    for c in report.failed:
        fmt = _REASONS.get(c.name)
        out.append(fmt(c) if fmt else f"{c.name} failed: {c.value} vs threshold {c.threshold}")
    return out


def pass_conditions(th: GateThresholds) -> list[str]:
    lines = ["watertight (no open boundary edges)", "manifold (every edge shared by exactly 2 faces)",
             f"at most {th.max_faces} triangle faces",
             f"roof height RMSE vs DSM <= {th.roof_height_rmse_m} m",
             f"every footprint vertex matched within {th.footprint_vertex_err_m} m"]
    if th.bmqi_min is not None:
        lines.append(f"BMQI >= {th.bmqi_min}")
    return lines
