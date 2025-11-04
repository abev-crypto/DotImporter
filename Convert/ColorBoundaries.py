"""Detect color boundaries in RGB images and sample them as dots."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

from .utils import graph_to_dots


def color_boundaries_to_dots(
    image_path: str,
    spacing: float,
    *,
    diff_thresh: float = 30.0,
    resize_to: int = 0,
) -> np.ndarray:
    """Sample points along color boundaries of an image.

    Parameters
    ----------
    image_path:
        Path to the source RGB image.
    spacing:
        Desired spacing between sampled points in pixels.
    diff_thresh:
        Threshold for color difference between adjacent pixels to detect a
        boundary. Default is ``30`` (0-441 range).
    resize_to:
        Resize the image so the longer side equals this value before
        processing. ``0`` disables resizing.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    scale = orig_w / img.size[0] if img.size[0] else 1.0

    arr = np.asarray(img, dtype=np.int16)
    H, W, _ = arr.shape

    # Compute color differences to right and bottom neighbors and keep the
    # maximum magnitude as the per-pixel gradient estimate. Using the maximum
    # helps capture boundaries regardless of the dominant direction.
    diff_r = np.zeros((H, W), dtype=np.float32)
    diff_d = np.zeros((H, W), dtype=np.float32)
    if W > 1:
        diff_r[:, :-1] = np.linalg.norm(arr[:, 1:] - arr[:, :-1], axis=2)
    if H > 1:
        diff_d[:-1, :] = np.linalg.norm(arr[1:, :] - arr[:-1, :], axis=2)
    grad = np.maximum(diff_r, diff_d)
    boundary = grad > diff_thresh

    # Save debug visualizations so the detected boundary quality can be
    # inspected outside of the pipeline.
    debug_base = Path(image_path)
    if debug_base.suffix:
        debug_base = debug_base.with_name(f"{debug_base.stem}_color_boundary{debug_base.suffix}")
    else:
        debug_base = debug_base.with_name(f"{debug_base.name}_color_boundary.png")

    try:
        boundary_img = Image.fromarray(boundary.astype(np.uint8) * 255)
        boundary_img.save(debug_base)
    except Exception:
        # If saving fails we simply continue processing without surfacing an
        # error because this output is purely for debugging convenience.
        pass

    skel = skeletonize(boundary)
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

    eff_spacing = spacing / scale
    pts = graph_to_dots(coords, adj, deg, eff_spacing)
    pts *= scale
    return pts


def grayscale_boundaries_to_dots(
    image_path: str,
    spacing: float,
    *,
    grad_thresh: float = 80.0,
    resize_to: int = 0,
) -> np.ndarray:
    """Detect edges in a grayscale image via convolution and sample them.

    Parameters
    ----------
    image_path:
        Path to the source image.
    spacing:
        Desired spacing between sampled points in pixels.
    grad_thresh:
        Threshold for gradient magnitude after applying Sobel filters.
    resize_to:
        Resize the image so the longer side equals this value before
        processing. ``0`` disables resizing.
    """
    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    scale = orig_w / img.size[0] if img.size[0] else 1.0

    arr = np.asarray(img, dtype=np.float32)
    H, W = arr.shape
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(arr, sobel_x, mode="nearest")
    gy = convolve(arr, sobel_y, mode="nearest")
    mag = np.hypot(gx, gy)
    boundary = mag > grad_thresh

    skel = skeletonize(boundary)
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

    eff_spacing = spacing / scale
    pts = graph_to_dots(coords, adj, deg, eff_spacing)
    pts *= scale
    return pts
