"""Top-down shaded render of a mesh, so a vision LLM can compare it with the roof image.

North (+y) is up, like an ortho image. Roof planes are shaded by slope direction
(light from the north-west) and brightened with height; background is white.
"""
import struct
import zlib
from pathlib import Path

import numpy as np

from .mesh import Mesh

_LIGHT = np.array([-1.0, 1.0, 2.0]) / np.sqrt(6.0)


def render_top_down(mesh: Mesh, path: Path, size: int = 384, margin: float = 0.08) -> None:
    v = np.asarray(mesh.vertices, dtype=float)
    lo, hi = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    span = max((hi - lo).max(), 1e-6)
    scale = size * (1 - 2 * margin) / span
    off = (size - (hi - lo) * scale) / 2
    px = (v[:, 0] - lo[0]) * scale + off[0]
    py = size - ((v[:, 1] - lo[1]) * scale + off[1])  # image rows grow downward
    z_lo, z_hi = v[:, 2].min(), v[:, 2].max()

    img = np.full((size, size), 255.0)
    zbuf = np.full((size, size), -np.inf)
    for a, b, c in mesh.faces:
        normal = np.cross(v[b] - v[a], v[c] - v[a])
        if normal[2] <= 1e-9:
            continue  # walls and floor are not visible from above
        x0, x1 = int(max(min(px[[a, b, c]]), 0)), int(min(max(px[[a, b, c]]) + 1, size))
        y0, y1 = int(max(min(py[[a, b, c]]), 0)), int(min(max(py[[a, b, c]]) + 1, size))
        if x0 >= x1 or y0 >= y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        det = (py[b] - py[c]) * (px[a] - px[c]) + (px[c] - px[b]) * (py[a] - py[c])
        if abs(det) < 1e-12:
            continue
        l1 = ((py[b] - py[c]) * (gx - px[c]) + (px[c] - px[b]) * (gy - py[c])) / det
        l2 = ((py[c] - py[a]) * (gx - px[c]) + (px[a] - px[c]) * (gy - py[c])) / det
        l3 = 1 - l1 - l2
        z = l1 * v[a, 2] + l2 * v[b, 2] + l3 * v[c, 2]
        win = zbuf[y0:y1, x0:x1]
        mask = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6) & (z > win)
        slope = float(np.clip(normal @ _LIGHT / np.linalg.norm(normal), 0, 1))
        height = (z - z_lo) / max(z_hi - z_lo, 1e-6)
        img[y0:y1, x0:x1][mask] = (40 + 130 * slope + 60 * height)[mask]
        win[mask] = z[mask]

    rows = b"".join(b"\x00" + row.tobytes() for row in img.astype(np.uint8))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
