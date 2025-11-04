"""Combine line and shape conversions into a single set of dots."""

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .Line2Dots import line_image_to_dots
from .Shape2Dots import shape_image_to_dots


def mixed_image_to_dots(
    image_path: str,
    spacing: float,
    *,
    fill_mode: str = 'NONE',
    fill_ratio: float = 0.5,
    blur_radius: float = 0.5,
    thresh_scale: float = 0.8,
    junction_ratio: float = 0.35,
    max_points: int = 0,
    resize_to: int = 0,
    detect_color_boundary: bool = False,
    outline: bool = True,
    thickness_thresh: float = 4.0,
) -> np.ndarray:
    """Return dots from both line and shape conversion.

    Parameters
    ----------
    image_path: str
        Path to the input image.
    spacing: float
        Desired spacing between sampled points in pixels.
    fill_mode: str, optional
        Fill strategy used for :func:`shape_image_to_dots`.
    fill_ratio: float, optional
        Weight assigned to interior sampling when ``fill_mode`` is active in
        :func:`shape_image_to_dots`.
    blur_radius: float, optional
        Gaussian blur radius for the line conversion.
    thresh_scale: float, optional
        Threshold scale for the line conversion.
    junction_ratio: float, optional
        Margin ratio around junctions for both conversions.
    max_points: int, optional
        Maximum number of points for each conversion. ``0`` disables.
    resize_to: int, optional
        Resize images so the longer side equals this value before processing.
    detect_color_boundary: bool, optional
        Whether to detect color boundaries for the shape conversion.
    outline: bool, optional
        Whether to extract outline in the shape conversion.
    thickness_thresh: float, optional
        Local width (in pixels) above which points are treated as part of a
        shape rather than a line.

    Returns
    -------
    np.ndarray
        Combined array of ``(x, y)`` coordinates.
    """

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    scale = orig_w / img.size[0] if img.size[0] else 1.0

    arr = np.array(img)
    mask = arr < 128
    width_map = distance_transform_edt(mask) * 2.0 * scale

    def _widths_at(pts: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return np.empty((0,), dtype=np.float32)
        pts_scaled = pts / scale
        xs = np.clip(np.round(pts_scaled[:, 0]).astype(int), 0, width_map.shape[1] - 1)
        ys = np.clip(np.round(pts_scaled[:, 1]).astype(int), 0, width_map.shape[0] - 1)
        return width_map[ys, xs]

    line_pts = line_image_to_dots(
        image_path,
        spacing,
        blur_radius=blur_radius,
        thresh_scale=thresh_scale,
        junction_ratio=junction_ratio,
        max_points=max_points,
        resize_to=resize_to,
    )
    shape_pts = shape_image_to_dots(
        image_path,
        spacing,
        junction_ratio=junction_ratio,
        fill_mode=fill_mode,
        fill_ratio=fill_ratio,
        max_points=max_points,
        resize_to=resize_to,
        detect_color_boundary=detect_color_boundary,
        outline=outline,
    )

    if line_pts.size:
        line_pts = line_pts[_widths_at(line_pts) < thickness_thresh]
    if shape_pts.size:
        shape_pts = shape_pts[_widths_at(shape_pts) >= thickness_thresh]

    if line_pts.size and shape_pts.size:
        return np.vstack([line_pts, shape_pts])
    if line_pts.size:
        return line_pts
    return shape_pts

