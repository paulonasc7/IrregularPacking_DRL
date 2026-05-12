#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Part:
    # Axis-aligned cuboid part in grid coordinates.
    x0: int
    y0: int
    z0: int
    dx: int
    dy: int
    dz: int

    @property
    def x1(self) -> int:
        return self.x0 + self.dx

    @property
    def y1(self) -> int:
        return self.y0 + self.dy

    @property
    def z1(self) -> int:
        return self.z0 + self.dz

    @property
    def volume(self) -> int:
        return self.dx * self.dy * self.dz


def rasterize_parts(
    box_shape: Tuple[int, int, int], parts: List[Part]
) -> np.ndarray:
    nx, ny, nz = box_shape
    occ = np.zeros((nx, ny, nz), dtype=bool)

    for p in parts:
        if not (0 <= p.x0 < p.x1 <= nx and 0 <= p.y0 < p.y1 <= ny and 0 <= p.z0 < p.z1 <= nz):
            raise ValueError(f"Part out of box bounds: {p}")
        occ[p.x0 : p.x1, p.y0 : p.y1, p.z0 : p.z1] = True

    return occ


def pyramidality_from_occupancy(occ: np.ndarray) -> Tuple[float, int, int, np.ndarray]:
    # Sum occupied voxels gives parts volume.
    v_parts = int(occ.sum())

    # Heightmap: highest occupied z index + 1 for each (x, y). If empty column -> 0.
    has_occ = occ.any(axis=2)
    top_idx = occ.shape[2] - 1 - np.argmax(occ[:, :, ::-1], axis=2)
    heights = np.where(has_occ, top_idx + 1, 0)

    # Volume under top envelope over occupied XY footprint.
    # For columns with any occupancy, denominator adds full column height = top height.
    v_projected = int(heights[has_occ].sum())

    if v_projected == 0:
        return 0.0, v_parts, v_projected, heights

    return v_parts / v_projected, v_parts, v_projected, heights


def main() -> None:
    # Box dimensions in voxels: X x Y x Z.
    box_shape = (10, 8, 8)

    # Example "packed" parts with intentional vertical gaps:
    # - part A rests on floor
    # - part B starts at z=2 (gap to floor)
    # - part C starts at z=4 (larger gap)
    parts = [
        Part(x0=1, y0=1, z0=0, dx=3, dy=2, dz=2),  # floor-supported
        Part(x0=4, y0=1, z0=2, dx=2, dy=3, dz=2),  # floating by 2
        Part(x0=2, y0=4, z0=4, dx=4, dy=2, dz=2),  # floating by 4
    ]

    occ = rasterize_parts(box_shape, parts)
    p, v_parts, v_projected, heights = pyramidality_from_occupancy(occ)

    print("=== Pyramidality Example ===")
    print(f"Box shape (X,Y,Z): {box_shape}")
    print(f"Number of parts: {len(parts)}")
    print(f"Total parts volume (voxels): {v_parts}")
    print(f"Projected occupied volume along Z (voxels): {v_projected}")
    print(f"Pyramidality = V_parts / V_projected = {p:.4f}")
    print()
    print("Heightmap (top z+1 per XY column, 0 means empty column):")
    print(heights.T)  # transpose for a more natural row/column print


if __name__ == "__main__":
    main()
