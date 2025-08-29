"""Convert silhouette images to sampled dots along their outlines."""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from skimage.morphology import skeletonize, binary_erosion, disk
import pandas as pd

from .utils import sample_poly, global_thin, edge_paths


def main():
    # --- 1) Load silhouette image ---
    img_path = '/mnt/data/person_46477-300x300-905360519.jpg'
    img = Image.open(img_path).convert('L')

    # Binarize (silhouette = foreground True). Invert if needed based on mean.
    arr = np.array(img)
    foreground = arr < 128  # silhouette is black on white

    # --- 2) Outline extraction (morphological gradient) ---
    outline = foreground ^ binary_erosion(foreground, footprint=disk(1))

    # Optional: slight blur before skeleton to avoid broken border on jaggy edges
    outline_img = Image.fromarray((outline * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=0.0)
    )
    outline = np.array(outline_img) > 0

    # --- 3) Thin to 1px outline ---
    skel = skeletonize(outline)

    # --- 4) Build graph (8-neighborhood) ---
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

    # --- 6) Per-path sampling with junction margin & global thinning ---
    spacing = 15.0  # px spacing for this test
    junction_margin = spacing * 0.35

    pts_all = []
    for poly in polys:
        pts = sample_poly(poly, spacing, start_offset=spacing * 0.5, end_margin=junction_margin)
        if len(pts):
            pts_all.append(pts)
    pts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))
    pts_clean = global_thin(pts, min_dist=spacing * 0.95)

    # --- 7) Save results ---
    csv_path = '/mnt/data/silhouette_outline_points.csv'
    pd.DataFrame(pts_clean, columns=['x', 'y']).to_csv(csv_path, index=False)

    # Compose a visualization: original + detected outline overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(img), cmap='gray')
    plt.contour(outline, levels=[0.5], linewidths=1, linestyles='--')
    if len(pts_clean):
        plt.scatter(pts_clean[:, 0], pts_clean[:, 1], s=10)
    plt.axis('off')
    plt.tight_layout()
    png_path = '/mnt/data/silhouette_outline_points_preview.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(png_path, csv_path)


if __name__ == '__main__':
    main()
