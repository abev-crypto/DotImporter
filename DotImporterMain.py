bl_info = {
    "name": "Dot Points Importer (CSV + Vertices)",
    "author": "you",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > N-panel > Dot Importer",
    "description": "Detect circular black dots from an image, export centers to CSV, and create vertices at those positions.",
    "category": "Import-Export",
}

import bpy
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    StringProperty, BoolProperty, FloatProperty, IntProperty, EnumProperty
)
from pathlib import Path
import csv
import numpy as np


# ---------- Utility: image -> grayscale & RGB (numpy) ----------
def load_image_grayscale_np(img_path: str):
    """
    Returns:
        gray : (H,W) in [0,1]
        rgb  : (H,W,3) in [0,1]
        width, height
    Note: Blender's image.pixels is bottom-to-top; we flip vertically
          so that (0,0) is top-left like typical image files.
    """
    img = bpy.data.images.load(img_path)
    w, h = img.size[0], img.size[1]
    px = np.array(img.pixels[:], dtype=np.float32)  # RGBA flat
    bpy.data.images.remove(img)  # free
    rgba = px.reshape((h, w, 4))
    rgb = rgba[..., :3]
    # Convert to grayscale with Rec.709 luma
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    # Blender pixels start at bottom row -> flip to top-left origin
    gray = np.flipud(gray)
    rgb = np.flipud(rgb)
    return gray, rgb, w, h


# ---------- Utility: connected components (4-neighborhood) ----------
def connected_components_labels(binary: np.ndarray):
    """
    Two-pass connected components (4-connectivity).
    binary: (H,W) bool/uint8 (True=foreground)
    Returns: labels (H,W) int32 with 0=background, 1..K=components
    """
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    parent = [0]  # union-find 1-indexed
    rank = [0]

    def uf_make():
        parent.append(len(parent))
        rank.append(0)
        return len(parent) - 1

    def uf_find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def uf_union(a, b):
        ra, rb = uf_find(a), uf_find(b)
        if ra == rb:
            return ra
        if rank[ra] < rank[rb]:
            parent[ra] = rb
            return rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
            return ra
        else:
            parent[rb] = ra
            rank[ra] += 1
            return ra

    # First pass
    next_label = 1
    for y in range(H):
        for x in range(W):
            if not binary[y, x]:
                continue
            # neighbors: up (y-1,x), left (y,x-1)
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up == 0 and left == 0:
                # new label
                l = uf_make()
                labels[y, x] = l
                next_label += 1
            elif up != 0 and left == 0:
                labels[y, x] = up
            elif up == 0 and left != 0:
                labels[y, x] = left
            else:
                labels[y, x] = uf_union(up, left)

    # Second pass: flatten labels
    label_map = {}
    new_id = 1
    for y in range(H):
        for x in range(W):
            l = labels[y, x]
            if l == 0:
                continue
            r = uf_find(l)
            if r not in label_map:
                label_map[r] = new_id
                new_id += 1
            labels[y, x] = label_map[r]

    return labels, (new_id - 1)


# ---------- Detection & CSV ----------
def detect_centers(gray: np.ndarray, threshold: float, invert: bool, min_area_px: int):
    """
    gray: (H,W) float [0,1]
    Returns: centers (N,2) float in pixel coordinates (x,y) with origin at top-left.
    """
    bw = gray < threshold if invert else gray > threshold
    labels, k = connected_components_labels(bw.astype(np.uint8))

    if k == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    H, W = gray.shape
    labels_flat = labels.reshape(-1)
    ys, xs = np.indices((H, W))
    xs_flat, ys_flat = xs.reshape(-1), ys.reshape(-1)

    counts = np.bincount(labels_flat, minlength=k + 1).astype(np.int32)  # index 0 = bg
    valid_ids = np.where(counts[1:] >= max(1, min_area_px))[0] + 1  # component ids

    if len(valid_ids) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)

    # sums per label
    sum_x = np.bincount(labels_flat, weights=xs_flat, minlength=k + 1)
    sum_y = np.bincount(labels_flat, weights=ys_flat, minlength=k + 1)

    cx = (sum_x[valid_ids] / counts[valid_ids]).astype(np.float32)
    cy = (sum_y[valid_ids] / counts[valid_ids]).astype(np.float32)
    centers = np.stack([cx, cy], axis=1)

    # sort by y then x
    order = np.lexsort((centers[:, 0], centers[:, 1]))
    centers = centers[order]
    areas = counts[valid_ids][order]
    return centers, areas


def save_centers_csv(csv_path: Path, centers, areas):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x", "y", "area"])
        for i, ((x, y), a) in enumerate(zip(centers, areas), start=1):
            writer.writerow([i, float(x), float(y), int(a)])

# ---------- Utility: colors at centers & CSV ----------
def sample_colors(rgb: np.ndarray, centers):
    H, W, _ = rgb.shape
    cols = []
    for x, y in centers:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = min(max(xi, 0), W - 1)
        yi = min(max(yi, 0), H - 1)
        r, g, b = rgb[yi, xi]
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)

def save_color_keys_csv(csv_path: Path, colors):
    """Save sampled RGB colors to CSV in CSV_ColorKeys format."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["CSV_ColorKeys"]
        for i in range(len(colors)):
            header += [f"vc[{i}]_R", f"vc[{i}]_G", f"vc[{i}]_B"]
        writer.writerow(header)
        row = [0]
        for r, g, b in colors:
            row += [int(r), int(g), int(b)]
        writer.writerow(row)


def apply_color_keys_action(obj, colors, frame: int = 1):
    """Create CSV_ColorKeys action and key vc[i]_* properties on the object."""
    if not obj.animation_data:
        obj.animation_data_create()
    obj.animation_data.action = bpy.data.actions.new(name="CSV_ColorKeys")

    for i, (r, g, b) in enumerate(colors):
        for ch, val in zip(("R", "G", "B"), (r, g, b)):
            key = f"vc[{i}]_{ch}"
            obj[key] = int(val)
            obj.keyframe_insert(f'["{key}"]', frame=frame)


# ---------- Blender: place vertices ----------
def pixels_to_blender_xy(x, y, w, h, unit_per_px, origin_mode, flip_y):
    if origin_mode == "center":
        X = (x - w * 0.5) * unit_per_px
        Y = ((h * 0.5 - y) if flip_y else (y - h * 0.5)) * unit_per_px
    elif origin_mode == "topleft":
        X = x * unit_per_px
        Y = ((h - y) if flip_y else y) * unit_per_px
    else:
        raise ValueError("origin_mode must be center/topleft")
    return X, Y


def create_vertices_object(name, centers_px, img_w, img_h, unit_per_px, origin_mode, flip_y, collection_name, max_points=0):
    centers = np.asarray(centers_px, dtype=np.float32)
    if max_points > 0:
        if centers.shape[0] > max_points:
            centers = centers[:max_points]
        elif centers.shape[0] < max_points:
            extra = max_points - centers.shape[0]
            xs = np.linspace(0, img_w - 1, extra, dtype=np.float32)
            ys = np.zeros(extra, dtype=np.float32)
            extra_centers = np.stack([xs, ys], axis=1)
            centers = np.vstack([centers, extra_centers])

    verts = []
    for x, y in centers:
        X, Y = pixels_to_blender_xy(x, y, img_w, img_h, unit_per_px, origin_mode, flip_y)
        verts.append((X, Y, 0.0))

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    coll = bpy.data.collections.get(collection_name)
    if not coll:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)
    coll.objects.link(obj)
    return obj, len(verts), centers


# ---------- Properties ----------
class DPIProps(PropertyGroup):
    image_path: StringProperty(
        name="Image",
        subtype='FILE_PATH',
        description="Input image file (black dots on white background recommended)",
    )
    output_dir: StringProperty(
        name="Output Dir",
        subtype='DIR_PATH',
        description="Where CSV will be saved (default: same folder as image)",
        default=""
    )
    threshold: FloatProperty(
        name="Threshold",
        description="Binarization threshold (0..1). For black dots, keep invert=True",
        default=0.5, min=0.0, max=1.0
    )
    invert: BoolProperty(
        name="Invert (black dots)",
        description="Treat darker-than-threshold as foreground",
        default=True
    )
    min_area_px: IntProperty(
        name="Min Area (px)",
        description="Ignore components smaller than this (in pixels)",
        default=20, min=1, soft_max=2000
    )
    unit_per_px: FloatProperty(
        name="Unit per Pixel",
        description="Scale factor from pixel to Blender unit",
        default=0.01, min=0.000001
    )
    origin_mode: EnumProperty(
        name="Origin",
        description="Where to place (0,0) in Blender relative to the image",
        items=[
            ('center', "Image Center", "Origin at image center"),
            ('topleft', "Top-Left", "Origin at top-left corner"),
        ],
        default='center'
    )
    flip_y: BoolProperty(
        name="Flip Y",
        description="Flip vertical axis so image Y-down becomes Blender Y-up",
        default=True
    )
    collection_name: StringProperty(
        name="Collection",
        default="DotPoints"
    )
    object_name: StringProperty(
        name="Object Name",
        default="DotPointsObj"
    )
    max_points: IntProperty(
        name="Max Points",
        description="Maximum number of vertices to create (0 for unlimited). Missing points are placed along the top edge.",
        default=0, min=0
    )
    save_csv: BoolProperty(
        name="Save CSV",
        default=True,
        description="Export detected centers to CSV",
    )


# ---------- Operator ----------
class DPI_OT_detect_and_create(Operator):
    bl_idname = "dpi.detect_and_create"
    bl_label = "Detect & Create"
    bl_description = "Detect dot centers from the image, export CSV, and create vertices"

    def execute(self, context):
        p = context.scene.dpi_props
        if not p.image_path:
            self.report({'ERROR'}, "Image not set")
            return {'CANCELLED'}

        try:
            gray, rgb, w, h = load_image_grayscale_np(bpy.path.abspath(p.image_path))
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {e}")
            return {'CANCELLED'}

        centers, areas = detect_centers(gray, p.threshold, p.invert, p.min_area_px)
        colors = sample_colors(rgb, centers)
        detected_count = len(centers)

        # Create points in Blender (may add extra vertices)
        obj, n, final_centers = create_vertices_object(
            p.object_name, centers, w, h,
            p.unit_per_px, p.origin_mode, p.flip_y, p.collection_name, p.max_points
        )

        final_len = len(final_centers)
        if areas.shape[0] > final_len:
            areas = areas[:final_len]
        elif areas.shape[0] < final_len:
            areas = np.concatenate([areas, np.zeros(final_len - areas.shape[0], dtype=np.int32)])

        if colors.shape[0] > final_len:
            colors = colors[:final_len]
        elif colors.shape[0] < final_len:
            extra = np.zeros((final_len - colors.shape[0], 3), dtype=np.uint8)
            colors = np.vstack([colors, extra])

        # Key colors on object for other addons
        apply_color_keys_action(obj, colors)

        # Save CSV files
        csv_path = None
        color_csv_path = None
        if p.save_csv:
            img_path = Path(bpy.path.abspath(p.image_path))
            out_dir = Path(bpy.path.abspath(p.output_dir)) if p.output_dir else img_path.parent
            csv_path = out_dir / f"{img_path.stem}_centers.csv"
            color_csv_path = out_dir / f"{img_path.stem}_color_keys.csv"
            try:
                save_centers_csv(csv_path, final_centers, areas)
                save_color_keys_csv(color_csv_path, colors)
            except Exception as e:
                self.report({'WARNING'}, f"CSV save failed: {e}")

        msg = f"Detected {detected_count} centers. Created {n} vertices."
        if csv_path:
            msg += f" CSV: {csv_path}"
        if color_csv_path:
            msg += f" Color CSV: {color_csv_path}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


# ---------- UI Panel ----------
class DPI_PT_panel(Panel):
    bl_label = "Dot Importer"
    bl_idname = "DPI_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Dot Importer"

    def draw(self, context):
        layout = self.layout
        p = context.scene.dpi_props

        col = layout.column(align=True)
        col.prop(p, "image_path")
        col.prop(p, "output_dir")

        box = layout.box()
        box.label(text="Detection")
        box.prop(p, "threshold")
        box.prop(p, "invert")
        box.prop(p, "min_area_px")

        box2 = layout.box()
        box2.label(text="Placement")
        box2.prop(p, "unit_per_px")
        box2.prop(p, "origin_mode")
        box2.prop(p, "flip_y")
        box2.prop(p, "collection_name")
        box2.prop(p, "object_name")
        box2.prop(p, "max_points")

        layout.prop(p, "save_csv")
        layout.operator(DPI_OT_detect_and_create.bl_idname, icon='PARTICLES')


# ---------- Register ----------
classes = (
    DPIProps,
    DPI_OT_detect_and_create,
    DPI_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.dpi_props = bpy.props.PointerProperty(type=DPIProps)

def unregister():
    del bpy.types.Scene.dpi_props
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
