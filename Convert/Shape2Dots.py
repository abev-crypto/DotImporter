"""Convert silhouette images to sampled dots along their outlines."""

import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree
from skimage.morphology import binary_erosion, disk, skeletonize

from .ColorBoundaries import color_boundaries_to_dots
from .utils import graph_to_dots


def _skeleton_to_dots(skel: np.ndarray, spacing: float) -> np.ndarray:
    """Return sampled dots from a skeletonized mask."""
    H, W = skel.shape
    coords = np.argwhere(skel)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32)

    idx_map = -np.ones((H, W), dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        idx_map[r, c] = i

    neighbors8 = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    adj = [[] for _ in range(len(coords))]
    for i, (r, c) in enumerate(coords):
        for dr, dc in neighbors8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
                j = idx_map[rr, cc]
                if j >= 0:
                    adj[i].append(j)
    deg = np.array([len(a) for a in adj])
    return graph_to_dots(coords, adj, deg, spacing)


def fill_shape(mask: np.ndarray, spacing: float, mode: str) -> np.ndarray:
    """Generate interior points within ``mask`` according to ``mode``.

    Parameters
    ----------
    mask:
        Boolean array where ``True`` indicates the region to fill.
    spacing:
        Desired spacing between points in pixels.
    mode:
        One of ``'NONE'``, ``'GRID'``, ``'SEMIGRID'``, ``'TOPOLOGY'`` or
        ``'RANDOM'``.
    """
    if spacing <= 0:
        return np.empty((0, 2), dtype=np.float32)

    H, W = mask.shape
    mode = mode.upper()

    if mode == 'NONE':
        return np.empty((0, 2), dtype=np.float32)

    if mode == 'GRID':
        ys = np.arange(0, H, spacing)
        xs = np.arange(0, W, spacing)
        pts = []
        for y in ys:
            yi = int(round(y))
            if yi >= H:
                continue
            for x in xs:
                xi = int(round(x))
                if xi < W and mask[yi, xi]:
                    pts.append((x, y))
        return np.array(pts, dtype=np.float32)

    if mode == 'SEMIGRID':
        ys = np.arange(0, H, spacing)
        pts = []
        for row, y in enumerate(ys):
            yi = int(round(y))
            if yi >= H:
                continue
            offset = (spacing / 2.0) if (row % 2) == 1 else 0.0
            xs = np.arange(offset, W, spacing)
            for x in xs:
                xi = int(round(x))
                if xi < W and mask[yi, xi]:
                    pts.append((x, y))
        return np.array(pts, dtype=np.float32)

    if mode == 'TOPOLOGY':
        pts_list = []
        current = mask.copy()
        step = max(1, int(round(spacing)))
        struct = disk(step)
        while np.any(current):
            ring = current ^ binary_erosion(current, footprint=struct)
            skel = skeletonize(ring)
            pts = _skeleton_to_dots(skel, spacing)
            if len(pts) == 0:
                break
            pts_list.append(pts)
            current = binary_erosion(current, footprint=struct)
        if pts_list:
            return np.vstack(pts_list)
        return np.empty((0, 2), dtype=np.float32)

    if mode == 'RANDOM':
        pts = []
        tree = None
        area = int(mask.sum())
        max_trials = area * 5 if area > 0 else 0
        for _ in range(max_trials):
            x = np.random.uniform(0, W)
            y = np.random.uniform(0, H)
            xi, yi = int(x), int(y)
            if xi >= W or yi >= H or not mask[yi, xi]:
                continue
            if tree is not None:
                if tree.query_ball_point((x, y), spacing):
                    continue
            pts.append((x, y))
            tree = cKDTree(pts)
        return np.array(pts, dtype=np.float32)

    return np.empty((0, 2), dtype=np.float32)


def shape_image_to_dots(
    image_path: str,
    spacing: float,
    junction_ratio: float = 0.35,
    *,
    fill_mode: str = 'NONE',
    max_points: int = 0,
    resize_to: int = 0,
    detect_color_boundary: bool = False,
    outline: bool = True,
) -> np.ndarray:
    """Sample points from a silhouette image.

    Parameters
    ----------
    image_path:
        Path to the silhouette image. The silhouette should be dark on a
        light background.
    spacing:
        Desired spacing between consecutive points in pixels.
    junction_ratio:
        Fraction of ``spacing`` used as margin around junctions.
    fill_mode:
        Strategy for generating interior points inside the silhouette.
    max_points:
        If greater than ``0``, adapt the spacing so that the number of
        sampled outline points does not exceed this value.
    resize_to:
        Resize the image so that its longer side equals this value (keeping
        aspect ratio) before processing. ``0`` disables resizing.
    detect_color_boundary:
        If ``True``, also detect and sample boundaries between color regions.
    outline:
        If ``True``, extract and sample the silhouette outline. ``False``
        skips outline extraction and only applies ``fill_shape``.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing ``(x, y)`` coordinates of the
        sampled points. ``N`` may be zero if no region is detected.
    """

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    scale = orig_w / img.size[0] if img.size[0] else 1.0

    # Binarize (silhouette = foreground True)
    arr = np.array(img)
    mask = arr < 128  # silhouette is black on white

    eff_spacing = spacing / scale

    if fill_mode == 'TOPOLOGY':
        pts = fill_shape(mask, eff_spacing, fill_mode)
        if max_points > 0 and len(pts) > max_points:
            while len(pts) > max_points and eff_spacing > 0:
                eff_spacing *= len(pts) / max_points
                pts = fill_shape(mask, eff_spacing, fill_mode)
        pts *= scale
        return pts

    if not outline:
        pts = fill_shape(mask, eff_spacing, fill_mode)
        if max_points > 0 and len(pts) > max_points:
            while len(pts) > max_points and eff_spacing > 0:
                eff_spacing *= len(pts) / max_points
                pts = fill_shape(mask, eff_spacing, fill_mode)
        pts *= scale
        return pts

    # Outline extraction (morphological gradient)
    outline = mask ^ binary_erosion(mask, footprint=disk(1))

    # Optional blur to stabilize jagged edges before skeletonization
    outline_img = Image.fromarray((outline * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=0.0)
    )
    outline = np.array(outline_img) > 0

    # Skeletonize to obtain a 1px outline
    skel = skeletonize(outline)

    outline_limit = max_points
    fill_limit = 0
    if max_points > 0 and fill_mode != 'NONE':
        outline_limit = max_points // 2 if outline else 0
        fill_limit = max_points - outline_limit

    eff_spacing_outline = eff_spacing
    pts = _skeleton_to_dots(skel, eff_spacing_outline)
    if outline_limit > 0 and len(pts) > outline_limit:
        while len(pts) > outline_limit:
            eff_spacing_outline *= len(pts) / outline_limit
            pts = _skeleton_to_dots(skel, eff_spacing_outline)

    eff_spacing_fill = eff_spacing
    interior = fill_shape(mask, eff_spacing_fill, fill_mode)
    if fill_limit > 0 and len(interior) > fill_limit:
        while len(interior) > fill_limit and eff_spacing_fill > 0:
            eff_spacing_fill *= len(interior) / fill_limit
            interior = fill_shape(mask, eff_spacing_fill, fill_mode)
    if interior.size and len(pts):
        dists, _ = cKDTree(pts).query(interior, k=1)
        min_dist = min(eff_spacing_outline, eff_spacing_fill) * 0.95
        interior = interior[dists >= min_dist]
    if interior.size:
        pts = np.vstack([pts, interior])
    pts *= scale

    if detect_color_boundary:
        color_pts = color_boundaries_to_dots(
            image_path, spacing, resize_to=resize_to
        )
        if len(color_pts):
            pts = np.vstack([pts, color_pts])

    return pts
