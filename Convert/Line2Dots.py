"""Sample points from line drawings using skeleton paths."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .utils import load_skeleton, edge_paths, sample_poly, global_thin


def main():
    # Load skeleton & paths
    img_path = '/mnt/data/tori2-2431964424.jpg'
    img, coords, adj, deg = load_skeleton(img_path, blur_radius=0.5, thresh_scale=0.8)
    polys = edge_paths(coords, adj, deg)

    # Parameters
    spacing = 20.0
    junction_margin = spacing * 0.35

    # Sample + clean
    pts_all = []
    for poly in polys:
        pts = sample_poly(poly, spacing, start_offset=spacing * 0.5, end_margin=junction_margin)
        if len(pts):
            pts_all.append(pts)
    pts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))
    pts_clean = global_thin(pts, min_dist=spacing * 0.95)

    # Save & preview
    csv_path = '/mnt/data/bird_points_clean.csv'
    pd.DataFrame(pts_clean, columns=['x', 'y']).to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.array(img), cmap='gray')
    if len(pts_clean):
        plt.scatter(pts_clean[:, 0], pts_clean[:, 1], s=10, c='orange')
    plt.axis('off')
    plt.tight_layout()
    png_path = '/mnt/data/bird_points_clean_preview.png'
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    print(png_path, csv_path)


if __name__ == '__main__':
    main()
