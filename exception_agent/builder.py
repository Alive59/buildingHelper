"""Deterministic builder: parametric building spec -> triangle mesh.

This is option (a) of the spec's open decision (section 4). The LLM never writes
vertices or faces; it describes parts, and this code makes each part a closed
solid. Parts are not boolean-unioned: overlapping parts stay separate shells.
"""
import math

from .config import Limits
from .mesh import Mesh

ROOF_TYPES = ("flat", "shed", "gable", "hip")

# JSON schema of the `geometry` field. Every key is required and unused ones are
# null, which is what strict schema-constrained decoding expects.
GEOMETRY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["parts"],
    "properties": {
        "parts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "footprint", "eave_height", "roof"],
                "properties": {
                    "name": {"type": "string"},
                    "footprint": {
                        "description": "[[x, y], ...] in metres, or null to use the source footprint",
                        "type": ["array", "null"],
                        "items": {"type": "array", "items": {"type": "number"},
                                  "minItems": 2, "maxItems": 2},
                    },
                    "eave_height": {"description": "wall height above ground, metres",
                                    "type": "number"},
                    "roof": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "rise", "ridge_axis", "upslope_azimuth_deg"],
                        "properties": {
                            "type": {"enum": list(ROOF_TYPES)},
                            "rise": {"description": "ridge height above eave, metres; 0 for flat",
                                     "type": "number"},
                            "ridge_axis": {"description": "gable/hip only: ridge along the long or short side",
                                           "enum": ["long", "short", None]},
                            "upslope_azimuth_deg": {
                                "description": "shed only: direction the roof rises toward, degrees from +y clockwise",
                                "type": ["number", "null"]},
                        },
                    },
                },
            },
        }
    },
}


class BuildError(ValueError):
    """The spec is well-formed JSON but cannot be built. Returned to the LLM."""


def build(spec: dict, source_footprint: list, ground_z: float,
          limits: Limits = Limits()) -> Mesh:
    parts = spec["parts"]
    if len(parts) > limits.max_parts:
        raise BuildError(f"{len(parts)} parts > {limits.max_parts} allowed")
    names = [p["name"] for p in parts]
    if len(set(names)) != len(names):
        raise BuildError("part names must be unique")
    mesh = Mesh()
    for part in parts:
        mesh.add(_build_part(part, source_footprint, ground_z, limits))
    return mesh


def _build_part(part: dict, source_footprint: list, ground_z: float, limits: Limits) -> Mesh:
    name = part["name"]
    poly = [(float(x), float(y)) for x, y in (part["footprint"] or source_footprint)]
    if len(poly) > 1 and poly[0] == poly[-1]:
        poly = poly[:-1]
    if len(poly) < 3:
        raise BuildError(f"part '{name}': footprint needs at least 3 vertices")
    area = _signed_area(poly)
    if abs(area) < 1e-6:
        raise BuildError(f"part '{name}': footprint has zero area")
    if area < 0:
        poly.reverse()

    eave, roof = float(part["eave_height"]), part["roof"]
    rise = float(roof["rise"])
    if not 0 < eave <= limits.max_height_m:
        raise BuildError(f"part '{name}': eave_height must be in (0, {limits.max_height_m}] m")
    if not 0 <= rise <= limits.max_height_m:
        raise BuildError(f"part '{name}': roof rise must be in [0, {limits.max_height_m}] m")
    kind = roof["type"]
    if kind in ("gable", "hip") and len(poly) != 4:
        raise BuildError(f"part '{name}': {kind} roof needs a 4-vertex footprint, got {len(poly)}; "
                         "split the building into quad parts or use flat/shed")
    if kind != "flat" and rise == 0:
        raise BuildError(f"part '{name}': {kind} roof needs rise > 0")

    z0, z1 = ground_z, ground_z + eave
    n = len(poly)
    if kind in ("gable", "hip") and _ridge_needs_rotation(poly, roof["ridge_axis"] or "long"):
        poly = poly[1:] + poly[:1]

    # Per-vertex roof-edge height (differs from z1 only for shed roofs).
    top_z = [z1] * n
    if kind == "shed":
        az = math.radians(float(roof["upslope_azimuth_deg"] or 0.0))
        d = [x * math.sin(az) + y * math.cos(az) for x, y in poly]
        lo, hi = min(d), max(d)
        top_z = [z1 + rise * (v - lo) / (hi - lo) for v in d]

    mesh = Mesh()
    mesh.vertices += [(x, y, z0) for x, y in poly]                      # 0..n-1 bottom
    mesh.vertices += [(x, y, top_z[i]) for i, (x, y) in enumerate(poly)]  # n..2n-1 top
    tris = _triangulate(poly, name)
    mesh.faces += [(c, b, a) for a, b, c in tris]  # floor faces down
    for i in range(n):
        j = (i + 1) % n
        mesh.faces += [(i, j, n + j), (i, n + j, n + i)]

    if kind in ("flat", "shed"):
        mesh.faces += [(n + a, n + b, n + c) for a, b, c in tris]
    else:
        # Ridge joins the midpoints of edges 0-1 and 2-3; hip pulls its ends inward.
        m0, m1 = _mid(poly[0], poly[1]), _mid(poly[2], poly[3])
        if kind == "hip":
            length = math.dist(m0, m1)
            inset = min(math.dist(poly[0], poly[1]) / 2, length * 0.45)
            t = inset / length
            m0, m1 = _lerp(m0, m1, t), _lerp(m1, m0, t)
        r0, r1 = 2 * n, 2 * n + 1
        mesh.vertices += [(m0[0], m0[1], z1 + rise), (m1[0], m1[1], z1 + rise)]
        t0, t1, t2, t3 = n, n + 1, n + 2, n + 3
        mesh.faces += [(t0, t1, r0), (t2, t3, r1),
                       (t1, t2, r1), (t1, r1, r0),
                       (t3, t0, r0), (t3, r0, r1)]
    return mesh


def _ridge_needs_rotation(poly: list, axis: str) -> bool:
    # Default layout puts the ridge parallel to edge 1-2.
    e01, e12 = math.dist(poly[0], poly[1]), math.dist(poly[1], poly[2])
    ridge_is_long = e12 >= e01
    return ridge_is_long != (axis == "long")


def _signed_area(poly: list) -> float:
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                     - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _triangulate(poly: list, name: str) -> list:
    """Ear clipping for a simple counter-clockwise polygon."""
    idx = list(range(len(poly)))
    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if _cross(poly[a], poly[b], poly[c]) <= 1e-12:
                continue  # reflex or collinear corner
            if any(_inside(poly[p], poly[a], poly[b], poly[c]) for p in idx if p not in (a, b, c)):
                continue
            tris.append((a, b, c))
            del idx[k]
            break
        else:
            raise BuildError(f"part '{name}': footprint is not a simple polygon")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def _inside(p, a, b, c) -> bool:
    return _cross(a, b, p) >= 0 and _cross(b, c, p) >= 0 and _cross(c, a, p) >= 0
