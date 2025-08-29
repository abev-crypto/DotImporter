"""Convert line drawings into sampled dot coordinates."""

import numpy as np

from .utils import edge_paths, global_thin, load_skeleton, sample_poly


def line_image_to_dots(
    image_path: str,
    spacing: float,
    blur_radius: float = 0.5,
    thresh_scale: float = 0.8,
    junction_ratio: float = 0.35,
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

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing ``(x, y)`` coordinates of the
        sampled points. ``N`` may be zero if no points are detected.
    """

    # Load the skeleton graph from the input image
    img, coords, adj, deg = load_skeleton(
        image_path, blur_radius=blur_radius, thresh_scale=thresh_scale
    )
    polys = edge_paths(coords, adj, deg)

    # Sample points along each polyline and merge
    junction_margin = spacing * junction_ratio
    pts_all = []
    for poly in polys:
        pts = sample_poly(
            poly, spacing, start_offset=spacing * 0.5, end_margin=junction_margin
        )
        if len(pts):
            pts_all.append(pts)
    pts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))

    # Globally thin to remove near-duplicate points
    pts_clean = global_thin(pts, min_dist=spacing * 0.95)
    return pts_clean


def main() -> None:
    """Example usage for manual testing."""

    img_path = "/mnt/data/tori2-2431964424.jpg"
    spacing = 20.0
    pts = line_image_to_dots(img_path, spacing)

    # Optional preview and CSV export for manual verification
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    pd.DataFrame(pts, columns=["x", "y"]).to_csv(
        "/mnt/data/bird_points_clean.csv", index=False
    )

    img = Image.open(img_path).convert("L")
    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(img), cmap="gray")
    if len(pts):
        plt.scatter(pts[:, 0], pts[:, 1], s=10, c="orange")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "/mnt/data/bird_points_clean_preview.png",
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
