"""Convert line drawings into sampled dot coordinates."""

import numpy as np

from .utils import graph_to_dots, load_skeleton


def line_image_to_dots(
    image_path: str,
    spacing: float,
    blur_radius: float = 0.5,
    thresh_scale: float = 0.8,
    junction_ratio: float = 0.35,
    *,
    max_points: int = 0,
    resize_to: int = 0,
) -> np.ndarray:
    """Sample points from a line drawing.

    Parameters
    ----------
    image_path:
        Path to the input image.
    spacing:
        Desired spacing between points in pixels.
    blur_radius:
        Gaussian blur radius applied before thresholding.
    thresh_scale:
        Multiplier for the mean intensity to determine the threshold.
    junction_ratio:
        Fraction of ``spacing`` used as margin around junctions.
    max_points:
        If greater than ``0``, increase the spacing so that the number of
        generated points does not exceed this value.
    resize_to:
        Resize the image so that its longer side equals this value (keeping
        aspect ratio) before processing. ``0`` disables resizing.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing ``(x, y)`` coordinates of the
        sampled points. ``N`` may be zero if no points are detected.
    """

    # Load the skeleton graph from the input image
    img, coords, adj, deg, scale = load_skeleton(
        image_path,
        blur_radius=blur_radius,
        thresh_scale=thresh_scale,
        resize_to=resize_to,
    )
    eff_spacing = spacing / scale
    pts = graph_to_dots(coords, adj, deg, eff_spacing)
    if max_points > 0 and len(pts) > max_points:
        while len(pts) > max_points:
            eff_spacing *= len(pts) / max_points
            pts = graph_to_dots(coords, adj, deg, eff_spacing)
    pts *= scale
    return pts
