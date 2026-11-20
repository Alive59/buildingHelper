"""Quality gate interface, a stand-in implementation, and the LLM-facing formatter.

The real gate lives in the bulk pipeline and must be reused unchanged: wrap it in
a class with the same `evaluate` signature that returns Check records, and the
formatter and harness work as they are. The gate checks topology and one
similarity score (BRICE); there is no geometric check against ground truth.
StandInGate's topology checks are real; its similarity score is a placeholder.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Protocol

from .config import GateThresholds
from .diagnostics import dsm_proxy_similarity
from .mesh import Mesh


@dataclass
class Check:
    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    data: dict = field(default_factory=dict)


@dataclass
class GateReport:
    passed: bool
    checks: list[Check]

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}


class Gate(Protocol):
    thresholds: GateThresholds

    def evaluate(self, mesh: Mesh, building: dict) -> GateReport: ...


class StandInGate:
    def __init__(self, thresholds: GateThresholds = GateThresholds(),
                 similarity: Callable[[Mesh, dict], float | None] = dsm_proxy_similarity):
        self.thresholds, self.similarity = thresholds, similarity

    def evaluate(self, mesh: Mesh, building: dict) -> GateReport:
        th = self.thresholds
        checks = [Check("faces", len(mesh.faces) <= th.max_faces, len(mesh.faces), th.max_faces)]
        checks += self._topology(mesh)
        score = self.similarity(mesh, building)
        # An unevaluated score fails: nothing is accepted on topology alone.
        checks.append(Check("brice", score is not None and score >= th.brice_min, score, th.brice_min))
        return GateReport(all(c.passed for c in checks), checks)

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


# --- Milestone 1: gate report -> short failure reasons for the LLM ---------------

def _faces(c: Check) -> str:
    faces = c.data.get("faces") or []
    return f" (e.g. faces {', '.join(map(str, faces))})" if faces else ""


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
    "brice": lambda c: ("BRICE could not be evaluated" if c.value is None
                        else f"BRICE {c.value:.2f} < {c.threshold:.2f} (similarity too low)"),
}


def failure_reasons(report: GateReport) -> list[str]:
    """One short, specific line per failed check."""
    out = []
    for c in report.failed:
        fmt = _REASONS.get(c.name)
        out.append(fmt(c) if fmt else f"{c.name} failed: {c.value} vs threshold {c.threshold}")
    return out


def pass_conditions(th: GateThresholds) -> list[str]:
    return ["watertight (no open boundary edges)", "manifold (every edge shared by exactly 2 faces)",
            f"at most {th.max_faces} triangle faces",
            f"BRICE similarity score >= {th.brice_min}"]
