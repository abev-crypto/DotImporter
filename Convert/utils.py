import numpy as np
from PIL import Image, ImageFilter
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize  # type: ignore
    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully
    skeletonize = None
    _SKIMAGE_AVAILABLE = False


def check_skimage() -> tuple[bool, str]:
    """Return availability of :mod:`scikit-image`.

    Returns
    -------
    (bool, str)
        ``True`` and empty message if scikit-image is available. ``False`` and a
        guidance string otherwise.
    """

    if _SKIMAGE_AVAILABLE:
        return True, ""
    return False, "scikit-image が見つかりません。`pip install scikit-image` を実行してください。"


def load_skeleton(image_path, blur_radius=0.5, thresh_scale=0.8, resize_to=0):
    """Load an image and return its skeleton and adjacency graph.

    Parameters
    ----------
    image_path : str
        Path to the source image.
    blur_radius : float, optional
        Gaussian blur radius applied before thresholding.
    thresh_scale : float, optional
        Multiplier for the mean intensity to determine the threshold.
    resize_to : int, optional
        If greater than ``0``, resize the image so that the longer side
        equals this value while preserving the aspect ratio. Points will be
        scaled back to the original size using the returned ``scale`` factor.

    Returns
    -------
    img : PIL.Image.Image
        Resized grayscale image.
    coords : ndarray of shape (N, 2)
        Coordinates of skeleton pixels as ``(row, col)``.
    adj : list[list[int]]
        Adjacency list of the skeleton graph.
    deg : ndarray
        Degree of each node in the graph.
    scale : float
        Multiplicative factor to convert coordinates back to the original
        image scale.
    """
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
    H, W = arr.shape
    th = arr.mean() * thresh_scale
    skel = skeletonize(arr < th)
    coords = np.argwhere(skel)  # (row, col)
    idx_map = -np.ones_like(arr, dtype=np.int32)
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
    scale = orig_w / img.size[0] if img.size[0] else 1.0
    return img, coords, adj, deg, scale


def edge_paths(coords, adj, deg):
    """Extract edge-following paths from a skeleton graph."""
    visited_edges = set()
    paths = []
    for u in range(len(coords)):
        for v in adj[u]:
            if (u, v) in visited_edges:
                continue
            # start path on edge (u->v)
            path = [u]
            visited_edges.add((u, v))
            visited_edges.add((v, u))
            prev = u
            curr = v
            path.append(curr)
            while deg[curr] == 2:
                n0, n1 = adj[curr][0], adj[curr][1]
                nxt = n0 if n1 == prev else n1
                if (curr, nxt) in visited_edges:
                    break
                visited_edges.add((curr, nxt))
                visited_edges.add((nxt, curr))
                prev, curr = curr, nxt
                path.append(curr)
            paths.append(path)
    polys = []
    for p in paths:
        xy = np.array([[coords[i][1], coords[i][0]] for i in p], dtype=float)
        if len(xy) >= 2:
            mask = np.append(True, np.any(np.diff(xy, axis=0) != 0, axis=1))
            xy = xy[mask]
        if len(xy) >= 2:
            polys.append(xy)
    return polys


def sample_poly(poly, spacing, start_offset=None, end_margin=0.0):
    """Sample equally spaced points along a polyline."""
    seg = np.diff(poly, axis=0)
    seglen = np.sqrt((seg ** 2).sum(axis=1))
    L = seglen.sum()
    if L <= spacing:
        t = L * 0.5
        acc = 0.0
        j = 0
        while j < len(seglen) and acc + seglen[j] < t:
            acc += seglen[j]
            j += 1
        if j >= len(seglen):
            return poly[[-1], :]
        alpha = (t - acc) / seglen[j] if seglen[j] else 0.0
        return np.array([poly[j] + alpha * (poly[j + 1] - poly[j])])
    if start_offset is None:
        start_offset = spacing * 0.5
    lo = start_offset + end_margin
    hi = L - end_margin
    if lo >= hi:
        return np.zeros((0, 2))
    t = np.arange(lo, hi, spacing)
    out = []
    acc = 0.0
    j = 0
    for target in t:
        while j < len(seglen) and acc + seglen[j] < target:
            acc += seglen[j]
            j += 1
        if j >= len(seglen):
            out.append(poly[-1])
            break
        alpha = (target - acc) / seglen[j] if seglen[j] else 0.0
        out.append(poly[j] + alpha * (poly[j + 1] - poly[j]))
    return np.array(out)


def global_thin(points, min_dist):
    """Globally thin points so no pair is closer than ``min_dist``."""
    if len(points) == 0:
        return points
    kept = []
    tree = None
    for p in points:
        if tree is None:
            kept.append(p)
            tree = cKDTree(np.array(kept))
            continue
        d, _ = tree.query(p, k=1)
        if d >= min_dist:
            kept.append(p)
            tree = cKDTree(np.array(kept))
    return np.array(kept)
