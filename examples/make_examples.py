"""Regenerates the 3 synthetic demo buildings and the stub script.

Run from the repo root: python examples/make_examples.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exception_agent.builder import build  # noqa: E402

HERE = Path(__file__).parent
random.seed(0)


def part(name, eave, kind, rise=0.0, axis=None, az=None, footprint=None):
    return {"name": name, "footprint": footprint, "eave_height": eave,
            "roof": {"type": kind, "rise": rise, "ridge_axis": axis, "upslope_azimuth_deg": az}}


def act(action, reason, parts=None, target=None, diagnosis=None):
    return {"action": action, "reason": reason, "target": target,
            "geometry": {"parts": parts} if parts else None, "diagnosis": diagnosis}


def dsm(truth, footprint, ground, noise=0.1, step=1.0):
    mesh = build({"parts": truth}, footprint, ground)
    xs, ys = [p[0] for p in footprint], [p[1] for p in footprint]
    out, x = [], min(xs) + step / 2
    while x < max(xs):
        y = min(ys) + step / 2
        while y < max(ys):
            z = mesh.top_z_at(x, y)
            if z is not None:
                out.append([round(x, 2), round(y, 2), round(z + random.gauss(0, noise), 2)])
            y += step
        x += step
    return out


def building(bid, footprint, ground, samples, attempts, height, source):
    # _stub_reference stands in for the roof image: only the placeholder scorer reads it.
    return {"id": bid, "footprint": footprint, "ground_z": ground, "_stub_reference": samples,
            "height_m": height, "height_source": source,
            "refs": {"roof_image": f"data/roof/{bid}.png"},
            "attempts": attempts}


rect = [[0, 0], [12, 0], [12, 8], [0, 8]]
lshape = [[0, 0], [10, 0], [10, 6], [4, 6], [4, 12], [0, 12]]
big = [[0, 0], [20, 0], [20, 15], [0, 15]]

queue = [
    # Gable house: stub starts with a flat roof, then fixes the part to a gable.
    building("B001", rect, 10.0, dsm([part("main", 6.0, "gable", 5.0, "long")], rect, 10.0), [
        {"route": "AR", "reasons": ["not watertight: 7 open boundary edges (e.g. faces 3, 9)"],
         "failed_checks": ["watertight"]},
        {"route": "rule", "reasons": ["BRICE 0.28 < 0.50 (similarity too low)"], "failed_checks": ["brice"]},
        {"route": "LoD1", "reasons": ["BRICE 0.37 < 0.50 (similarity too low)"], "failed_checks": ["brice"]}], 8.5, "lod1"),
    # L-shaped building whose wings differ by 6 m: single-part builds never pass.
    building("B002", lshape, 5.0,
             dsm([part("a", 4.0, "flat", footprint=[[0, 0], [10, 0], [10, 6], [0, 6]]),
                  part("b", 10.0, "flat", footprint=[[0, 6], [4, 6], [4, 12], [0, 12]])], lshape, 5.0), [
        {"route": "AR", "reasons": ["312 faces > 256"], "failed_checks": ["faces"]},
        {"route": "rule", "note": "plane fitting found no roof planes"},
        {"route": "LoD1", "reasons": ["BRICE 0.26 < 0.50 (similarity too low)"], "failed_checks": ["brice"]}], 5.7, "prediction"),
    # Nothing fits (e.g. roof hidden by trees in the image): the score never moves, so give up.
    building("B003", big, 0.0, [[x + 0.5, y + 0.5, round(random.uniform(2, 18), 2)]
                                for x in range(0, 20, 2) for y in range(0, 15, 2)], [
        {"route": "AR", "reasons": ["not manifold: 4 non-manifold edges (e.g. faces 12, 13, 14)"],
         "failed_checks": ["manifold"]},
        {"route": "rule", "reasons": ["BRICE 0.14 < 0.50 (similarity too low)"], "failed_checks": ["brice"]},
        {"route": "LoD1", "reasons": ["BRICE 0.17 < 0.50 (similarity too low)"], "failed_checks": ["brice"]}], 9.0, "prediction"),
]

script = {
    "B001": [
        act("rebuild_full", "start from the simplest model: a flat roof", [part("main", 8.5, "flat")]),
        act("fix_part", "flat scored 0.41; try a gable along the long side",
            [part("main", 6.0, "gable", 5.0, "long")], target="main"),
    ],
    "B002": [
        '{"action": "rebuild_full", "reason": "truncated output", "geometry": {"parts": [',
        act("rebuild_full", "start with a low flat roof", [part("main", 4.0, "flat")]),
        act("rebuild_full", "raise the flat roof", [part("main", 6.0, "flat")]),
        act("fix_part", "try a gable", [part("main", 4.0, "gable", 6.0, "long")], target="main"),
        act("fix_part", "try a shed rising to the north", [part("main", 4.0, "shed", 6.0, az=0.0)], target="main"),
    ],
    "B003": [
        act("rebuild_full", "start from the simplest model: a flat roof", [part("main", 9.0, "flat")]),
        act("fix_part", "flat scored low; try a gable", [part("main", 7.0, "gable", 4.0, "long")], target="main"),
        act("give_up", "BRICE stayed near 0.17 for both flat and gable", diagnosis=(
            "BRICE does not respond to the roof type: flat and gable both score about 0.17. "
            "The roof is probably not visible in the image (trees or shadow), or the footprint does "
            "not match the image. A human should check the roof image and its alignment.")),
    ],
}

for b in queue:
    (HERE / "queue" / f"{b['id']}.json").write_text(json.dumps(b) + "\n")
(HERE / "stub_script.json").write_text(json.dumps(script, indent=1) + "\n")
