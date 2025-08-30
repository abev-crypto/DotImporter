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


def sample_polys(polys, spacing, junction_ratio):
    """Sample and globally thin points from multiple polylines.

    Parameters
    ----------
    polys : list[ndarray]
        Polylines to sample from.
    spacing : float
        Desired spacing between samples.
    junction_ratio : float
        Fraction of ``spacing`` used as margin around junctions.

    Returns
    -------
    ndarray
        Sampled and thinned points of shape ``(N, 2)``.
    """

    junction_margin = spacing * junction_ratio
    pts_all = []
    for poly in polys:
        pts = sample_poly(
            poly, spacing, start_offset=spacing * 0.5, end_margin=junction_margin
        )
        if len(pts):
            pts_all.append(pts)
    pts = np.vstack(pts_all) if pts_all else np.zeros((0, 2))
    return global_thin(pts, min_dist=spacing * 0.95)

def graph_from_image(image_path, blur=0.5, thresh_scale=0.8):
    img = Image.open(image_path).convert('L')
    arr = np.array(img.filter(ImageFilter.GaussianBlur(radius=blur)))
    H, W = arr.shape
    skel = skeletonize(arr < arr.mean()*thresh_scale)
    coords = np.argwhere(skel)
    idx_map = -np.ones_like(arr, dtype=np.int32)
    for i,(r,c) in enumerate(coords):
        idx_map[r,c] = i
    adj = [[] for _ in range(len(coords))]
    for i,(r,c) in enumerate(coords):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                rr,cc=r+dr,c+dc
                if 0<=rr<H and 0<=cc<W and skel[rr,cc]:
                    j=idx_map[rr,cc]
                    if j>=0: adj[i].append(j)
    deg = np.array([len(a) for a in adj])
    return img, coords, adj, deg

def extract_edge_paths(coords, adj, deg):
    visited=set(); paths=[]
    for u in range(len(coords)):
        for v in adj[u]:
            if (u,v) in visited: continue
            path=[u]; visited.add((u,v)); visited.add((v,u))
            prev=u; curr=v; path.append(curr)
            while deg[curr]==2:
                n0,n1 = adj[curr][0], adj[curr][1]
                nxt = n0 if n1==prev else n1
                if (curr,nxt) in visited: break
                visited.add((curr,nxt)); visited.add((nxt,curr))
                prev,curr = curr,nxt; path.append(curr)
            paths.append(path)
    polys_idx = paths
    polys_xy = []
    for p in polys_idx:
        xy=np.array([[coords[i][1],coords[i][0]] for i in p],dtype=float)
        if len(xy)>=2:
            mask=np.append(True, np.any(np.diff(xy,axis=0)!=0,axis=1))
            xy=xy[mask]
        if len(xy)>=2: polys_xy.append(xy)
    return polys_idx, polys_xy

def anchors_from_graph(coords, deg, adj, corner_thresh_deg=40):
    ends = np.where(deg==1)[0]
    hubs = np.where(deg>=3)[0]
    anchors_idx = set(ends.tolist() + hubs.tolist())
    # corners detection
    for i in range(len(coords)):
        if deg[i]==2:
            n0,n1 = adj[i]
            v0 = coords[n0]-coords[i]; v1=coords[n1]-coords[i]
            v0=v0/np.linalg.norm(v0); v1=v1/np.linalg.norm(v1)
            ang=np.degrees(np.arccos(np.clip(np.dot(v0,v1),-1,1)))
            if ang >= corner_thresh_deg:
                anchors_idx.add(i)
    anchors_idx = sorted(list(anchors_idx))
    anchors_xy = np.array([[coords[i][1],coords[i][0]] for i in anchors_idx], dtype=float)
    return anchors_idx, anchors_xy, set(anchors_idx)

def split_poly_at_anchors(poly_idx, anchors_set):
    out=[]; cur=[poly_idx[0]]
    for i in range(1,len(poly_idx)):
        idx=poly_idx[i]; cur.append(idx)
        if idx in anchors_set:
            if len(cur)>=2: out.append(cur)
            cur=[idx]
    if len(cur)>=2: out.append(cur)
    return out

def resample_segment(poly_xy, min_spacing):
    seg = np.diff(poly_xy, axis=0)
    seglen = np.sqrt((seg**2).sum(axis=1))
    L=float(seglen.sum())
    if L==0: return np.zeros((0,2))
    q=int(np.floor(L/min_spacing))
    if q<=0:
        return np.array([poly_xy[len(poly_xy)//2]]) # relaxed midpoint
    if q==1:
        return np.array([0.5*(poly_xy[0]+poly_xy[-1])])
    N=q-1
    s=L/(N+1)
    t=np.linspace(s,L-s,N)
    out=[]; acc=0.0; j=0
    for target in t:
        while j < len(seglen) and acc+seglen[j]<target:
            acc+=seglen[j]; j+=1
        if j>=len(seglen):
            out.append(poly_xy[-1]); break
        alpha=(target-acc)/seglen[j] if seglen[j] else 0.0
        out.append(poly_xy[j]+alpha*(poly_xy[j+1]-poly_xy[j]))
    return np.array(out)

def global_cleanup(points, min_dist):
    if len(points)==0: return points
    kept=[]; tree=None
    for p in points:
        if tree is None: kept.append(p); tree=cKDTree(np.array(kept)); continue
        d,_=tree.query(p,k=1)
        if d>=min_dist: kept.append(p); tree=cKDTree(np.array(kept))
    return np.array(kept)

def test():
    # Run pipeline on this new image
    img_path = '/mnt/data/fb2a9d5ee228473602f737150dbfb25e-2343292115.png'
    img, coords, adj, deg = graph_from_image(img_path)
    poly_idx, polys_xy = extract_edge_paths(coords, adj, deg)
    anchors_idx, anchors_xy, anchors_set = anchors_from_graph(coords, deg, adj, corner_thresh_deg=40)

    # merge anchors
    if len(anchors_xy):
        tree=cKDTree(anchors_xy); used=set(); merged=[]
        for i,p in enumerate(anchors_xy):
            if i in used: continue
            idxs=tree.query_ball_point(p,r=1.5)
            used.update(idxs)
            merged.append(np.mean(anchors_xy[idxs],axis=0))
        anchor_points=np.array(merged)
    else:
        anchor_points=np.zeros((0,2))

    min_spacing=20.0

    interior=[]
    for idxs,poly in zip(poly_idx,polys_xy):
        segs=split_poly_at_anchors(idxs,anchors_set)
        for seg_idx in segs:
            seg_xy=np.array([[coords[i][1],coords[i][0]] for i in seg_idx],dtype=float)
            pts=resample_segment(seg_xy,min_spacing)
            if len(pts): interior.append(pts)
    interior_pts=np.vstack(interior) if interior else np.zeros((0,2))

    all_pts=anchor_points.copy()
    if len(interior_pts): all_pts=np.vstack([all_pts,interior_pts])
    all_pts=global_cleanup(all_pts, min_dist=min_spacing*0.95)
