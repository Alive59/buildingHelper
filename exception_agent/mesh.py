"""Minimal triangle mesh with OBJ I/O (stdlib only)."""
from dataclasses import dataclass, field
from pathlib import Path

Vec3 = tuple[float, float, float]
Tri = tuple[int, int, int]


@dataclass
class Mesh:
    vertices: list[Vec3] = field(default_factory=list)
    faces: list[Tri] = field(default_factory=list)

    def add(self, other: "Mesh") -> None:
        off = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend((a + off, b + off, c + off) for a, b, c in other.faces)

    def write_obj(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in self.vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces]
        path.write_text("\n".join(lines) + "\n")

    @classmethod
    def read_obj(cls, path: Path) -> "Mesh":
        mesh = cls()
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                mesh.vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "f":
                n = len(mesh.vertices)
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else n + i for i in idx]
                for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                    mesh.faces.append((idx[0], idx[k], idx[k + 1]))
        return mesh

    def top_z_at(self, x: float, y: float) -> float | None:
        """Highest surface z above (x, y), or None if the mesh does not cover it."""
        best = None
        for a, b, c in self.faces:
            (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = (
                self.vertices[a], self.vertices[b], self.vertices[c])
            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(det) < 1e-12:
                continue  # vertical or degenerate in plan view
            l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            l3 = 1.0 - l1 - l2
            if min(l1, l2, l3) < -1e-9:
                continue
            z = l1 * z1 + l2 * z2 + l3 * z3
            if best is None or z > best:
                best = z
        return best
