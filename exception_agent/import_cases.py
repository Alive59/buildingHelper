"""Convert gen-model case folders into queue entries.

    python -m exception_agent.import_cases <cases_dir> <queue_dir>

Each case folder holds meta.json, footprint.geojson, crop_model.png,
ndsm_pred.npy/.png, ship_gen.obj (the failed generation) and gt_lod2.obj.

Frames. Queue entries use metres, +x east, +y north, z up, origin at the centre
of the footprint bounding box. The OBJ files are y-up with z = -north and a
tile offset, recovered here by matching the GT bounding box to the footprint.
ndsm_pred is 128 x 128, north up, with the longer bounding-box side spanning
122 px; its values in [-1, 1] are read as (v + 1) / 2 * height_m. Both were
inferred from the 12 reference cases (mask IoU 0.96-0.99 against GT).

GT is used only for `_stub_reference`, which the placeholder scorer reads and
the LLM never sees. Pass --no-gt to leave it out.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .mesh import Mesh

NDSM_SPAN_PX = 122.0


def import_case(case_dir: Path, queue_dir: Path, use_gt: bool = True, step_px: int = 3) -> dict:
    meta = json.loads((case_dir / "meta.json").read_text())
    name = meta.get("case", case_dir.name)
    rings = json.loads((case_dir / "footprint.geojson").read_text())[
        "features"][0]["geometry"]["coordinates"][0]
    ring = np.asarray(rings[0], dtype=float)[:, :2]
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    centre = (lo + hi) / 2
    pts = [[round(float(x), 2), round(float(y), 2)] for x, y in ring - centre]
    while len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    footprint = [p for i, p in enumerate(pts) if p != pts[i - 1]]

    files = queue_dir / "files" / name
    files.mkdir(parents=True, exist_ok=True)
    gt = Mesh.read_obj(case_dir / "gt_lod2.obj")
    gv = np.asarray(gt.vertices)
    off_x, off_z = lo[0] - gv[:, 0].min(), -hi[1] - gv[:, 2].min()

    def to_local(mesh: Mesh) -> Mesh:
        return Mesh([(x + off_x - centre[0], -(z + off_z) - centre[1], y) for x, y, z in mesh.vertices],
                    list(mesh.faces))

    to_local(Mesh.read_obj(case_dir / "ship_gen.obj")).write_obj(files / "gen.obj")
    refs = {}
    for key, src in (("roof_image", "crop_model.png"), ("height_map_image", "ndsm_pred.png")):
        if (case_dir / src).is_file():
            shutil.copy(case_dir / src, files / src)
            refs[key] = f"files/{name}/{src}"

    # Sample points on the nDSM pixel grid, in the local frame.
    res = float((hi - lo).max()) / NDSM_SPAN_PX
    idx = np.arange(step_px // 2, 128, step_px)
    cols, rows = np.meshgrid(idx, idx)
    xs, ys = (cols + 0.5 - 64) * res, -(rows + 0.5 - 64) * res

    height = float(meta["height_m"])
    ndsm = np.load(case_dir / "ndsm_pred.npy", allow_pickle=False)
    inside = ndsm[rows, cols] > -0.99
    pred = (ndsm[rows, cols] + 1) / 2 * height
    entry = {
        "id": name, "source_id": meta.get("id"), "footprint": footprint, "ground_z": 0.0,
        "height_m": height, "height_source": "prediction",
        "dsm_label": "Predicted height map (not ground truth)",
        "dsm_samples": [[round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                        for x, y, z in zip(xs[inside], ys[inside], pred[inside])],
        "refs": refs,
        "attempts": [{"route": "AR", "mesh_path": f"files/{name}/gen.obj"}],
        "notes": {"footprint_vertices": len(footprint), "footprint_holes_dropped": len(rings) - 1},
    }
    if use_gt:
        local_gt = to_local(gt)
        tops = [(x, y, local_gt.top_z_at(float(x), float(y))) for x, y in zip(xs.ravel(), ys.ravel())]
        entry["_stub_reference"] = [[round(float(x), 2), round(float(y), 2), round(z, 2)]
                                    for x, y, z in tops if z is not None]
    (queue_dir / f"{name}.json").write_text(json.dumps(entry) + "\n")
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cases_dir", type=Path)
    p.add_argument("queue_dir", type=Path)
    p.add_argument("--no-gt", action="store_true", help="do not derive the hidden scorer reference from GT")
    a = p.parse_args()
    cases = sorted(d for d in a.cases_dir.iterdir() if (d / "meta.json").is_file())
    for d in cases:
        e = import_case(d, a.queue_dir, use_gt=not a.no_gt)
        print(f"{e['id']}: {len(e['footprint'])} footprint vertices, {len(e['dsm_samples'])} height samples")
    print(f"{len(cases)} cases -> {a.queue_dir}")


if __name__ == "__main__":
    main()
