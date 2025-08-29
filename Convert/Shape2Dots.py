# Add a preprocessing step for silhouettes: extract outline first, then run the same sampler.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from skimage.morphology import skeletonize, binary_erosion, disk
from scipy.spatial import cKDTree
import pandas as pd

# --- 1) Load silhouette image ---
img_path = '/mnt/data/person_46477-300x300-905360519.jpg'
img = Image.open(img_path).convert('L')

# Binarize (silhouette = foreground True). Invert if needed based on mean.
arr = np.array(img)
# silhouette is black on white; set threshold in the middle
foreground = arr < 128

# --- 2) Outline extraction (morphological gradient) ---
# outline = foreground XOR eroded(foreground)
outline = foreground ^ binary_erosion(foreground, footprint=disk(1))

# Optional: slight blur before skeleton to avoid broken border on jaggy edges
outline_img = Image.fromarray((outline*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=0.0))
outline = np.array(outline_img) > 0

# --- 3) Thin to 1px outline ---
skel = skeletonize(outline)

# --- 4) Build graph (8-neighborhood) ---
H, W = skel.shape
coords = np.argwhere(skel)
idx_map = -np.ones((H,W), dtype=np.int32)
for i,(r,c) in enumerate(coords):
    idx_map[r,c] = i

neighbors8=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
adj=[[] for _ in range(len(coords))]
for i,(r,c) in enumerate(coords):
    for dr,dc in neighbors8:
        rr,cc=r+dr,c+dc
        if 0<=rr<H and 0<=cc<W and skel[rr,cc]:
            j=idx_map[rr,cc]
            if j>=0: adj[i].append(j)
deg = np.array([len(a) for a in adj])

# --- 5) Edge-visited path extraction (handles loops, avoids duplicates) ---
visited_edges=set(); paths=[]
for u in range(len(coords)):
    for v in adj[u]:
        if (u,v) in visited_edges: continue
        path=[u]; visited_edges.add((u,v)); visited_edges.add((v,u))
        prev=u; curr=v; path.append(curr)
        while deg[curr]==2:
            n0,n1 = adj[curr][0], adj[curr][1]
            nxt = n0 if n1==prev else n1
            if (curr,nxt) in visited_edges: break
            visited_edges.add((curr,nxt)); visited_edges.add((nxt,curr))
            prev,curr = curr,nxt; path.append(curr)
        paths.append(path)

# Convert to xy polylines
polys=[]
for p in paths:
    xy=np.array([[coords[i][1],coords[i][0]] for i in p],dtype=float)
    if len(xy)>=2:
        mask=np.append(True, np.any(np.diff(xy,axis=0)!=0,axis=1))
        xy=xy[mask]
    if len(xy)>=2: polys.append(xy)

# --- 6) Per-path sampling with junction margin & global thinning ---
def sample_poly(poly, spacing, start_offset=None, end_margin=0.0):
    seg=np.diff(poly,axis=0)
    seglen=np.sqrt((seg**2).sum(axis=1))
    L=seglen.sum()
    if L <= spacing:
        t=L*0.5; acc=0.0; j=0
        while j < len(seglen) and acc + seglen[j] < t:
            acc += seglen[j]; j += 1
        if j >= len(seglen): return poly[[-1],:]
        alpha = (t-acc)/seglen[j] if seglen[j] else 0.0
        return np.array([poly[j] + alpha*(poly[j+1]-poly[j])])
    if start_offset is None: start_offset = spacing*0.5
    lo = start_offset + end_margin; hi = L - end_margin
    if lo >= hi: return np.zeros((0,2))
    t = np.arange(lo, hi, spacing)
    out=[]; acc=0.0; j=0
    for target in t:
        while j < len(seglen) and acc + seglen[j] < target:
            acc += seglen[j]; j += 1
        if j >= len(seglen): out.append(poly[-1]); break
        alpha = (target-acc)/seglen[j] if seglen[j] else 0.0
        out.append(poly[j] + alpha*(poly[j+1]-poly[j]))
    return np.array(out)

def global_thin(points, min_dist):
    if len(points)==0: return points
    kept=[]; tree=None
    for p in points:
        if tree is None:
            kept.append(p); tree=cKDTree(np.array(kept)); continue
        d,_=tree.query(p, k=1)
        if d >= min_dist:
            kept.append(p); tree=cKDTree(np.array(kept))
    return np.array(kept)

spacing = 15.0  # px spacing for this test
junction_margin = spacing*0.35

pts_all=[]
for poly in polys:
    pts = sample_poly(poly, spacing, start_offset=spacing*0.5, end_margin=junction_margin)
    if len(pts): pts_all.append(pts)
pts = np.vstack(pts_all) if pts_all else np.zeros((0,2))
pts_clean = global_thin(pts, min_dist=spacing*0.95)

# --- 7) Save results ---
csv_path = '/mnt/data/silhouette_outline_points.csv'
pd.DataFrame(pts_clean, columns=['x','y']).to_csv(csv_path, index=False)

# Compose a visualization: original + detected outline overlay
plt.figure(figsize=(5,5))
plt.imshow(np.array(img), cmap='gray')
plt.contour(outline, levels=[0.5], linewidths=1, linestyles='--')  # show computed outline
if len(pts_clean):
    plt.scatter(pts_clean[:,0], pts_clean[:,1], s=10)
plt.axis('off'); plt.tight_layout()
png_path = '/mnt/data/silhouette_outline_points_preview.png'
plt.savefig(png_path, dpi=200, bbox_inches='tight')
png_path, csv_path
