"""Combine line and shape conversions into a single set of dots."""

import numpy as np

from .Line2Dots import line_image_to_dots
from .Shape2Dots import shape_image_to_dots


def mixed_image_to_dots(
    image_path: str,
    spacing: float,
    *,
    fill_mode: str = 'NONE',
    blur_radius: float = 0.5,
    thresh_scale: float = 0.8,
    junction_ratio: float = 0.35,
    max_points: int = 0,
    resize_to: int = 0,
    detect_color_boundary: bool = False,
    outline: bool = True,
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

    Returns
    -------
    np.ndarray
        Combined array of ``(x, y)`` coordinates.
    """

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
        max_points=max_points,
        resize_to=resize_to,
        detect_color_boundary=detect_color_boundary,
        outline=outline,
    )

    if line_pts.size and shape_pts.size:
        return np.vstack([line_pts, shape_pts])
    if line_pts.size:
        return line_pts
    return shape_pts

