"""Convert silhouette images to sampled dots along their outlines."""

import numpy as np
from PIL import Image, ImageFilter
from skimage.morphology import binary_erosion, disk, skeletonize

from .utils import edge_paths, global_thin, sample_poly


def shape_image_to_dots(
    image_path: str,
    spacing: float,
    junction_ratio: float = 0.35,
    *,
    max_points: int = 0,
    resize_to: int = 0,
) -> np.ndarray:
    """Sample points along the outline of a silhouette image.

    Parameters
    ----------
    image_path:
        Path to the silhouette image. The silhouette should be dark on a
        light background.
    spacing:
        Desired spacing between consecutive points in pixels.
    junction_ratio:
        Fraction of ``spacing`` used as margin around junctions.
    max_points:
        If greater than ``0``, adapt the spacing so that the number of
        sampled points does not exceed this value.
    resize_to:
        Resize the image so that its longer side equals this value (keeping
        aspect ratio) before processing. ``0`` disables resizing.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 2)`` containing ``(x, y)`` coordinates of the
        sampled outline points. ``N`` may be zero if no outline is detected.
    """

    img = Image.open(image_path).convert("L")
    orig_w, orig_h = img.size
    if resize_to and (orig_w > resize_to or orig_h > resize_to):
        img.thumbnail((resize_to, resize_to), Image.Resampling.LANCZOS)
    scale = orig_w / img.size[0] if img.size[0] else 1.0

    # Binarize (silhouette = foreground True)
    arr = np.array(img)
    foreground = arr < 128  # silhouette is black on white

    # Outline extraction (morphological gradient)
    outline = foreground ^ binary_erosion(foreground, footprint=disk(1))

    # Optional blur to stabilize jagged edges before skeletonization
    outline_img = Image.fromarray((outline * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=0.0)
    )
    outline = np.array(outline_img) > 0

    # Skeletonize to obtain a 1px outline
    skel = skeletonize(outline)

    # Build adjacency graph of skeleton pixels
    H, W = skel.shape
    coords = np.argwhere(skel)
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

    polys = edge_paths(coords, adj, deg)

    def _sample(sp: float) -> np.ndarray:
        junction_margin = sp * junction_ratio
        pts_all = []
        for poly in polys:
            pts = sample_poly(
                poly, sp, start_offset=sp * 0.5, end_margin=junction_margin
            )
            if len(pts):
                pts_all.append(pts)
        pts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))
        return global_thin(pts, min_dist=sp * 0.95)

    eff_spacing = spacing / scale
    pts_clean = _sample(eff_spacing)
    if max_points > 0 and len(pts_clean) > max_points:
        while len(pts_clean) > max_points:
            eff_spacing *= len(pts_clean) / max_points
            pts_clean = _sample(eff_spacing)
    pts_clean *= scale
    return pts_clean


def main() -> None:
    """Example usage for manual testing."""

    img_path = "/mnt/data/person_46477-300x300-905360519.jpg"
    spacing = 15.0
    pts = shape_image_to_dots(img_path, spacing)

    # Optional preview and CSV export for manual verification
    import matplotlib.pyplot as plt
    import pandas as pd

    pd.DataFrame(pts, columns=["x", "y"]).to_csv(
        "/mnt/data/silhouette_outline_points.csv", index=False
    )

    img = Image.open(img_path).convert("L")
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(img), cmap="gray")
    if len(pts):
        plt.scatter(pts[:, 0], pts[:, 1], s=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        "/mnt/data/silhouette_outline_points_preview.png",
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
