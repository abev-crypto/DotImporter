from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from mathutils import Vector
from mathutils import geometry as mu_geometry
from mathutils import kdtree as mu_kdtree


@dataclass
class PlacementRegion:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    polygon: Optional[List[Vector]]
    source_object: str
    requested_custom: bool
    requested_mesh: bool
    using_mesh: bool

    @property
    def using_custom(self) -> bool:
        return self.polygon is not None

    @property
    def using_mesh_region(self) -> bool:
        return self.using_mesh

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.min_x, self.max_x, self.min_y, self.max_y

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


def parse_custom_region(json_text: str) -> Tuple[Optional[List[Vector]], str]:
    if not json_text:
        return None, ""
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None, ""

    points = data.get("points") or []
    if len(points) < 3:
        return None, data.get("object", "")

    polygon: List[Vector] = []
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x = float(pt[0])
                y = float(pt[1])
            except (TypeError, ValueError):
                continue
            polygon.append(Vector((x, y)))
    if len(polygon) < 3:
        return None, data.get("object", "")
    return polygon, data.get("object", "")


def convex_hull_2d(points: Sequence[Vector]) -> Optional[List[Vector]]:
    """Return the 2D convex hull of ``points`` projected to the XY plane."""
    unique = {(float(pt.x), float(pt.y)) for pt in points}
    if len(unique) < 3:
        return None

    sorted_pts = sorted(unique)

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return None
    return [Vector((x, y)) for x, y in hull]


def create_region(
    coords: Sequence[Vector],
    spacing: float,
    use_custom_region: bool,
    custom_region_json: str,
    mesh_polygon: Optional[Sequence[Vector]] = None,
    mesh_source_object: str = "",
    mesh_requested: bool = False,
) -> PlacementRegion:
    coords = list(coords)
    if not coords and not (mesh_requested and mesh_polygon):
        raise ValueError("Cannot create placement region without coordinates")

    polygon: Optional[List[Vector]] = None
    source_object = ""
    using_mesh_polygon = bool(
        mesh_requested and mesh_polygon and len(mesh_polygon) >= 3
    )

    if using_mesh_polygon:
        polygon = [Vector((float(pt.x), float(pt.y))) for pt in mesh_polygon]
        source_object = mesh_source_object
        bounds_points: Sequence[Vector] = polygon
    else:
        if use_custom_region:
            polygon, source_object = parse_custom_region(custom_region_json)
        bounds_points = polygon if polygon else coords

    if not bounds_points:
        raise ValueError("Cannot create placement region without coordinates")

    min_x = min(pt.x for pt in bounds_points)
    max_x = max(pt.x for pt in bounds_points)
    min_y = min(pt.y for pt in bounds_points)
    max_y = max(pt.y for pt in bounds_points)

    padding = spacing if spacing > 0 else 0.1
    if max_x - min_x < 1e-6:
        min_x -= padding
        max_x += padding
    if max_y - min_y < 1e-6:
        min_y -= padding
        max_y += padding

    return PlacementRegion(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        polygon=polygon,
        source_object=source_object,
        requested_custom=use_custom_region,
        requested_mesh=mesh_requested,
        using_mesh=using_mesh_polygon,
    )


def build_kdtree(points: Iterable[Vector]):
    points = list(points)
    if not points:
        return None
    tree = mu_kdtree.KDTree(len(points))
    for i, co in enumerate(points):
        tree.insert((co.x, co.y, co.z), i)
    tree.balance()
    return tree


def sample_point_in_polygon(poly_vectors: Sequence[Vector]) -> Optional[Vector]:
    if not poly_vectors or len(poly_vectors) < 3:
        return None
    if len(poly_vectors) == 3:
        triangles = [(0, 1, 2)]
    else:
        try:
            triangles = mu_geometry.tessellate_polygon([list(poly_vectors)])
        except Exception:
            triangles = []
    if not triangles and len(poly_vectors) >= 3:
        triangles = [(0, 1, 2)]
    if not triangles:
        return None

    tri_data: List[Tuple[float, Vector, Vector, Vector]] = []
    total_area = 0.0
    for tri in triangles:
        if len(tri) != 3:
            continue
        try:
            v0 = poly_vectors[tri[0]]
            v1 = poly_vectors[tri[1]]
            v2 = poly_vectors[tri[2]]
        except IndexError:
            continue
        area = mu_geometry.area_tri(
            Vector((v0.x, v0.y, 0.0)),
            Vector((v1.x, v1.y, 0.0)),
            Vector((v2.x, v2.y, 0.0)),
        )
        if area <= 0.0:
            continue
        tri_data.append((area, v0, v1, v2))
        total_area += area
    if total_area <= 0.0 or not tri_data:
        return None

    pick = random.uniform(0.0, total_area)
    accum = 0.0
    for area, v0, v1, v2 in tri_data:
        accum += area
        if pick <= accum:
            u = random.random()
            v = random.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            return v0 + (v1 - v0) * u + (v2 - v0) * v

    # Fallback to last triangle
    area, v0, v1, v2 = tri_data[-1]
    u = random.random()
    v = random.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    return v0 + (v1 - v0) * u + (v2 - v0) * v


def _candidate_valid(candidate: Vector, spacing: float, static_tree, new_positions: Sequence[Vector]) -> bool:
    if spacing <= 0:
        return True
    if static_tree and static_tree.find_range((candidate.x, candidate.y, candidate.z), spacing):
        return False
    for other in new_positions:
        if (candidate - other).length < spacing:
            return False
    return True


def randomize_assignments(
    target_coords: Sequence[Vector],
    region: PlacementRegion,
    spacing: float,
    static_tree,
    max_attempts: int,
) -> Tuple[List[Optional[Vector]], int]:
    assignments: List[Optional[Vector]] = []
    new_positions: List[Vector] = []
    skipped = 0

    min_x, max_x, min_y, max_y = region.bounds
    use_custom = region.using_custom
    polygon = region.polygon or []

    for co in target_coords:
        assigned: Optional[Vector] = None
        for _ in range(max_attempts):
            if use_custom:
                sample = sample_point_in_polygon(polygon)
                if sample is None:
                    break
                candidate = Vector((sample.x, sample.y, co.z))
            else:
                rx = random.uniform(min_x, max_x)
                ry = random.uniform(min_y, max_y)
                candidate = Vector((rx, ry, co.z))

            if _candidate_valid(candidate, spacing, static_tree, new_positions):
                new_positions.append(candidate)
                assigned = candidate
                break
        if assigned is None:
            skipped += 1
        assignments.append(assigned)
    return assignments, skipped


def _grid_dimensions(
    count: int,
    width: float,
    height: float,
    spacing: float,
) -> Tuple[int, int]:
    if count <= 0:
        return 0, 0

    min_spacing = max(spacing, 0.0)
    width = max(width, min_spacing if min_spacing > 0 else 1.0)
    height = max(height, min_spacing if min_spacing > 0 else 1.0)

    aspect = width / height if height > 1e-6 else 1.0

    if min_spacing > 0:
        max_cols = max(1, int(math.floor(width / min_spacing)) + 1)
        max_rows = max(1, int(math.floor(height / min_spacing)) + 1)
    else:
        max_cols = count
        max_rows = count

    max_cols = min(max_cols, count)
    max_rows = min(max_rows, count)

    best_cols = max(1, int(round(math.sqrt(count * aspect))))
    best_cols = min(best_cols, max_cols)
    best_rows = max(1, int(math.ceil(count / best_cols)))

    if best_rows > max_rows:
        best_rows = max_rows
        best_cols = max(1, int(math.ceil(count / best_rows)))
        if best_cols > max_cols:
            best_cols = max_cols

    cols, rows = best_cols, best_rows

    while cols * rows < count and (cols < max_cols or rows < max_rows):
        if cols < max_cols and (cols <= rows or rows >= max_rows):
            cols += 1
        elif rows < max_rows:
            rows += 1
        else:
            break

    cols = max(1, min(cols, max_cols))
    rows = max(1, min(rows, max_rows))
    return cols, rows


def _linspace(min_val: float, max_val: float, count: int) -> List[float]:
    if count <= 1:
        return [0.5 * (min_val + max_val)]
    step = (max_val - min_val) / (count - 1)
    return [min_val + step * i for i in range(count)]


def _points_in_polygon_filter(
    candidates: Sequence[Vector],
    polygon: Sequence[Vector],
) -> List[Vector]:
    if not polygon:
        return list(candidates)
    filtered: List[Vector] = []
    for pt in candidates:
        res = mu_geometry.point_in_polygon_2d(pt.x, pt.y, polygon, False)
        if res >= 0:
            filtered.append(pt)
    return filtered


def grid_assignments(
    target_coords: Sequence[Vector],
    region: PlacementRegion,
    spacing: float,
    static_tree,
) -> Tuple[List[Optional[Vector]], int]:
    count = len(target_coords)
    if count == 0:
        return [], 0

    cols, rows = _grid_dimensions(count, region.width, region.height, spacing)
    if cols == 0 or rows == 0:
        return [None] * count, count

    xs = _linspace(region.min_x, region.max_x, cols)
    ys = _linspace(region.min_y, region.max_y, rows)

    candidates: List[Vector] = []
    for y in ys:
        for x in xs:
            candidates.append(Vector((x, y, 0.0)))

    candidates = _points_in_polygon_filter(candidates, region.polygon or [])

    assignments: List[Optional[Vector]] = []
    new_positions: List[Vector] = []
    skipped = 0
    idx = 0

    for co in target_coords:
        assigned: Optional[Vector] = None
        while idx < len(candidates):
            candidate_xy = candidates[idx]
            idx += 1
            candidate = Vector((candidate_xy.x, candidate_xy.y, co.z))
            if _candidate_valid(candidate, spacing, static_tree, new_positions):
                new_positions.append(candidate)
                assigned = candidate
                break
        if assigned is None:
            skipped += 1
        assignments.append(assigned)

    return assignments, skipped
