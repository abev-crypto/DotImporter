bl_info = {
    "name": "Dot Points Importer (CSV + Vertices)",
    "author": "ABEYUYA",
    "version": (1, 3, 1),
    "blender": (4, 3, 0),
    "location": "View3D > N-panel > Dot Importer",
    "description": "Detect circular black dots from an image, export centers to CSV, and create vertices at those positions.",
    "category": "Import-Export",
}

import bpy
import bmesh
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    StringProperty,
    BoolProperty,
    FloatProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
)
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image, ImageDraw
from mathutils import Vector
from typing import List, Optional

from placement import (
    build_kdtree,
    convex_hull_2d,
    create_region,
    grid_assignments,
    randomize_assignments,
)

from Convert import Line2Dots, Shape2Dots, Mixed2Dots
from Convert.utils import check_skimage, sample_edge, sample_curve
from Convert.Shape2Dots import fill_shape


DEFAULT_MIN_AREA_PX = 20
DEFAULT_PIXEL_SPACING = 10.0
DEFAULT_GEOMETRY_SPACING = 1.0


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


def save_points_color_csv(csv_path: Path, centers_px, colors, img_w, img_h,
                          unit_per_px, origin_mode, flip_y,
                          range_x=0.0, range_y=0.0):
    """Save detected points with RGB colors in a CSV format.

    Columns: Time [msec], x [m], y [m], z [m], Red, Green, Blue.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time [msec]", "x [m]", "y [m]", "z [m]", "Red", "Green", "Blue"])
        unit_x, unit_y = compute_units_per_px(img_w, img_h, unit_per_px,
                                              range_x, range_y)
        for i, ((px, py), (r, g, b)) in enumerate(zip(centers_px, colors)):
            X, Y = pixels_to_blender_xy(px, py, img_w, img_h,
                                       unit_x, unit_y, origin_mode, flip_y)
            writer.writerow([i, float(X), float(Y), 0.0, int(r), int(g), int(b)])

# ---------- Utility: colors at centers ----------
def sample_colors(rgb: np.ndarray, centers):
    H, W, _ = rgb.shape
    if len(centers) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    cols = []
    for x, y in centers:
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = min(max(xi, 0), W - 1)
        yi = min(max(yi, 0), H - 1)
        r, g, b = rgb[yi, xi]
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)

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
def compute_units_per_px(img_w: float, img_h: float, unit_per_px: float,
                         range_x: float, range_y: float):
    """Return uniform scale from pixel to Blender units while keeping aspect."""
    scale_candidates = []
    if range_x > 0 and img_w > 0:
        scale_candidates.append(range_x / float(img_w))
    if range_y > 0 and img_h > 0:
        scale_candidates.append(range_y / float(img_h))

    if scale_candidates:
        scale = min(scale_candidates)
    else:
        scale = unit_per_px

    return scale, scale


def compute_image_spacing_px(props, img_w: float, img_h: float) -> float:
    unit_x, _ = compute_units_per_px(
        img_w,
        img_h,
        props.unit_per_px,
        props.output_range_x,
        props.output_range_y,
    )
    if props.vertex_spacing > 0 and unit_x > 0:
        return max(props.vertex_spacing / unit_x, 1.0)
    return DEFAULT_PIXEL_SPACING


def compute_geometry_spacing(props) -> float:
    if props.vertex_spacing > 0:
        return props.vertex_spacing
    return DEFAULT_GEOMETRY_SPACING


def pixels_to_blender_xy(x, y, w, h, unit_x, unit_y, origin_mode, flip_y):
    if origin_mode == "center":
        X = (x - w * 0.5) * unit_x
        Y = ((h * 0.5 - y) if flip_y else (y - h * 0.5)) * unit_y
    elif origin_mode == "topleft":
        X = x * unit_x
        Y = ((h - y) if flip_y else y) * unit_y
    else:
        raise ValueError("origin_mode must be center/topleft")
    return X, Y


def create_vertices_object(name, centers_px, img_w, img_h, unit_per_px, origin_mode,
                           flip_y, collection_name, max_points=0, spacing=10.0,
                           z_values=None, range_x=0.0, range_y=0.0,
                           placement_spacing=0.0, auto_adjust_max_points=False):
    centers = np.asarray(centers_px, dtype=np.float32)
    orig_count = centers.shape[0]
    unit_x, unit_y = compute_units_per_px(img_w, img_h, unit_per_px,
                                          range_x, range_y)
    if auto_adjust_max_points and max_points > 0:
        max_points = max(max_points, orig_count)
    if max_points > 0:
        if centers.shape[0] > max_points:
            centers = centers[:max_points]
            orig_count = centers.shape[0]
        elif centers.shape[0] < max_points:
            extra = max_points - centers.shape[0]
            step_x = max(spacing, 1.0)
            step_y = max(spacing, 1.0)
            if placement_spacing > 0:
                if unit_x > 0:
                    step_x = max(placement_spacing / unit_x, 1.0)
                if unit_y > 0:
                    step_y = max(placement_spacing / unit_y, 1.0)
            xs = []
            ys = []
            x = 0.0
            y = float(img_h) + step_y
            for _ in range(extra):
                xs.append(x)
                ys.append(y)
                x += step_x
                if x >= img_w:
                    x = 0.0
                    y += step_y
            extra_centers = np.stack([np.array(xs, dtype=np.float32),
                                      np.array(ys, dtype=np.float32)], axis=1)
            centers = np.vstack([centers, extra_centers])

    verts = []
    for i, (x, y) in enumerate(centers):
        X, Y = pixels_to_blender_xy(x, y, img_w, img_h, unit_x, unit_y,
                                   origin_mode, flip_y)
        Z = float(z_values[i]) if z_values is not None and i < len(z_values) else 0.0
        verts.append((X, Y, Z))

    # Normalize original vertices to fit within [-1, 1] when no explicit range is set
    if orig_count > 0 and range_x <= 0 and range_y <= 0:
        arr = np.array(verts[:orig_count], dtype=np.float32)
        max_abs = np.max(np.abs(arr[:, :2]), axis=0)
        denom = max(max_abs[0], max_abs[1])
        if denom > 0:
            scale = 1.0 / denom
            verts = [(vx * scale, vy * scale, vz) for (vx, vy, vz) in verts]

    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata(verts, [], [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    coll = bpy.data.collections.get(collection_name)
    if not coll:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)
    coll.objects.link(obj)
    return obj, len(verts), centers, verts


def create_mesh_with_faces(points, width, height):
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            faces.append((i, i + 1, i + width))
            faces.append((i + 1, i + width + 1, i + width))
    mesh = bpy.data.meshes.new("HeightMesh")
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("HeightMesh", mesh)
    return obj


def create_points_object(name, points, collection_name):
    """Create a new mesh object consisting only of ``points``."""
    mesh = bpy.data.meshes.new(name + "_Mesh")
    mesh.from_pydata([tuple(p) for p in points], [], [])
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    coll = bpy.data.collections.get(collection_name)
    if not coll:
        coll = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(coll)
    coll.objects.link(obj)
    return obj


def _mesh_region_object_poll(self, obj):
    return obj is not None and obj.type == 'MESH'


def build_mesh_region_polygon(
    context,
    target_obj: bpy.types.Object,
    mesh_obj: bpy.types.Object,
):
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh is None:
        return None
    try:
        if not mesh.polygons:
            return None
        inv_matrix = target_obj.matrix_world.inverted()
        world_matrix = eval_obj.matrix_world
        points: List[Vector] = []
        for poly in mesh.polygons:
            if len(poly.vertices) < 3:
                continue
            for vid in poly.vertices:
                co = mesh.vertices[vid].co
                world_co = world_matrix @ co
                local_co = inv_matrix @ world_co
                points.append(Vector((local_co.x, local_co.y)))
        if not points:
            return None
        return convex_hull_2d(points)
    finally:
        eval_obj.to_mesh_clear()


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
        description="Where CSV will be saved (default: same folder as .blend)",
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
    conversion_mode: EnumProperty(
        name="Conversion Mode",
        items=[('NONE', 'None', ''), ('LINE', 'Line', ''), ('SHAPE', 'Shape', ''), ('MIX', 'Mix', '')],
        description="Select conversion type for non-circular features",
        default='NONE'
    )
    fill_mode: EnumProperty(
        name="Fill Mode",
        description="Method to generate interior points for shape conversion",
        items=[
            ('NONE', 'None', ''),
            ('GRID', 'Grid', ''),
            ('SEMIGRID', 'SemiGrid', ''),
            ('TOPOLOGY', 'Topology', ''),
            ('RANDOM', 'Random', ''),
        ],
        default='NONE',
    )
    fill_ratio: FloatProperty(
        name="Fill Weight",
        description=(
            "割合 0=アウトライン優先, 1=塗り優先 (Fill Mode 使用時のみ適用)"
        ),
        default=0.5,
        min=0.0,
        max=1.0,
    )
    fill_closed: BoolProperty(
        name="Fill Closed", 
        description="Fill interior points when curve is closed", 
        default=False,
    )
    detect_color_boundary: BoolProperty(
        name="Detect Color Boundary",
        description="Include boundaries between color regions for shape conversion",
        default=False,
    )
    outline: BoolProperty(
        name="Outline",
        description="Generate outline from the silhouette for shape conversion",
        default=True,
    )
    resize_to: IntProperty(
        name="Resize Max",
        description=(
            "Resize image so the longer side equals this value before processing "
            "(0 disables resizing)"
        ),
        default=512, min=0,
    )
    blur_radius: FloatProperty(
        name="Blur Radius",
        description="Gaussian blur radius for line conversion",
        default=0.5, min=0.0,
    )
    thresh_scale: FloatProperty(
        name="Thresh Scale",
        description="Threshold scale for line conversion",
        default=0.8, min=0.0,
    )
    junction_ratio: FloatProperty(
        name="Junction Ratio",
        description="Margin ratio around junctions",
        default=0.35, min=0.0,
    )
    mix_thickness: FloatProperty(
        name="Mix Thickness",
        description="Pixel width threshold to treat regions as shapes",
        default=4.0, min=0.0,
    )
    unit_per_px: FloatProperty(
        name="Unit per Pixel",
        description="Scale factor from pixel to Blender unit",
        default=0.01, min=0.000001
    )
    output_range_x: FloatProperty(
        name="Range X [m]",
        description=(
            "Target width for generated vertices in meters (0 to disable). "
            "Aspect ratio is preserved and the result fits within the given X/Y."
        ),
        default=0.0, min=0.0
    )
    output_range_y: FloatProperty(
        name="Range Y [m]",
        description=(
            "Target height for generated vertices in meters (0 to disable). "
            "Aspect ratio is preserved and the result fits within the given X/Y."
        ),
        default=0.0, min=0.0
    )
    vertex_spacing: FloatProperty(
        name="Vertex Spacing [m]",
        description="Spacing between generated vertices in Blender units for sampling and placement (0 uses mode defaults)",
        default=0.0, min=0.0
    )
    ignore_unselected_spacing: BoolProperty(
        name="Ignore Existing Vertices",
        description="When moving selected vertices, ignore the spacing limit against unselected vertices",
        default=False,
    )
    randomize_use_mesh_region: BoolProperty(
        name="Use Mesh Region",
        description="Restrict placement to the area covered by another mesh object's faces",
        default=False,
    )
    mesh_region_object: PointerProperty(
        name="Region Mesh",
        description="Planar mesh whose faces define the placement region when enabled",
        type=bpy.types.Object,
        poll=_mesh_region_object_poll,
    )
    randomize_use_custom_region: BoolProperty(
        name="Use Custom Region",
        description="Restrict random placement to a stored custom region",
        default=False,
    )
    placement_mode: EnumProperty(
        name="Placement Mode",
        description="Strategy for repositioning selected vertices",
        items=[
            ('RANDOM', "Random", "Place vertices randomly within the region"),
            ('GRID', "Grid", "Arrange vertices along a grid topology"),
        ],
        default='RANDOM',
    )
    custom_region_json: StringProperty(
        name="Custom Region Data",
        description="Serialized polygon for custom placement region",
        default="",
        options={'HIDDEN'},
    )
    custom_region_object_name: StringProperty(
        name="Custom Region Object",
        description="Object name from which the custom region was captured",
        default="",
        options={'HIDDEN'},
    )
    custom_region_point_count: IntProperty(
        name="Custom Region Points",
        description="Number of vertices stored in the custom region",
        default=0,
        min=0,
        options={'HIDDEN'},
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
        description="Maximum number of vertices to create (0 for unlimited). Missing points are placed outside the image bounds with uniform spacing.",
        default=500, min=0
    )
    bottom_reserve_ratio: FloatProperty(
        name="Reserve Bottom %",
        description="割合 0=余白なし, 1=全ての頂点を下部に確保 (Max Points に対する割合)",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    auto_adjust_max_points: BoolProperty(
        name="Auto Expand Limit",
        description="Automatically raise the vertex limit so the requested spacing is preserved",
        default=True,
    )
    height_map_path: StringProperty(
        name="Height Map",
        subtype='FILE_PATH',
        description="Grayscale image used as height map",
    )
    auto_height: BoolProperty(
        name="Auto Height",
        description="Use height map brightness as vertex Z value",
        default=False,
    )
    save_csv: BoolProperty(
        name="Save CSV",
        default=False,
        description="Export detected points with color to CSV",
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

        img_path = bpy.path.abspath(p.image_path)
        ok, msg = check_skimage()
        if not ok:
            self.report({'ERROR'}, msg or "scikit-image が見つかりません")
            return {'CANCELLED'}
        try:
            gray, rgb, w, h = load_image_grayscale_np(img_path)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {e}")
            return {'CANCELLED'}

        reserve_ratio = min(max(p.bottom_reserve_ratio, 0.0), 1.0)
        sampling_limit = 0
        allowed_ratio = 1.0
        if not p.auto_adjust_max_points and p.max_points > 0:
            allowed_ratio = max(0.0, 1.0 - reserve_ratio)
            if allowed_ratio > 0.0:
                sampling_limit = max(1, min(p.max_points, int(p.max_points * allowed_ratio)))
            else:
                sampling_limit = 0
        pixel_spacing = compute_image_spacing_px(p, w, h)

        if p.conversion_mode == 'LINE':
            centers = Line2Dots.line_image_to_dots(
                img_path,
                pixel_spacing,
                blur_radius=p.blur_radius,
                thresh_scale=p.thresh_scale,
                junction_ratio=p.junction_ratio,
                max_points=sampling_limit,
                resize_to=p.resize_to,
            )
        elif p.conversion_mode == 'SHAPE':
            centers = Shape2Dots.shape_image_to_dots(
                img_path,
                pixel_spacing,
                junction_ratio=p.junction_ratio,
                fill_mode=p.fill_mode,
                fill_ratio=p.fill_ratio,
                max_points=sampling_limit,
                resize_to=p.resize_to,
                detect_color_boundary=p.detect_color_boundary,
                outline=p.outline,
            )
        elif p.conversion_mode == 'MIX':
            centers = Mixed2Dots.mixed_image_to_dots(
                img_path,
                pixel_spacing,
                fill_mode=p.fill_mode,
                fill_ratio=p.fill_ratio,
                blur_radius=p.blur_radius,
                thresh_scale=p.thresh_scale,
                junction_ratio=p.junction_ratio,
                max_points=sampling_limit,
                resize_to=p.resize_to,
                detect_color_boundary=p.detect_color_boundary,
                outline=p.outline,
                thickness_thresh=p.mix_thickness,
            )
        else:
            gray_proc = gray
            if p.resize_to and (w > p.resize_to or h > p.resize_to):
                if w >= h:
                    new_w = p.resize_to
                    new_h = int(round(h * p.resize_to / w))
                else:
                    new_h = p.resize_to
                    new_w = int(round(w * p.resize_to / h))
                gray_pil = Image.fromarray((gray * 255).astype(np.uint8), mode="L")
                gray_proc = np.array(
                    gray_pil.resize((new_w, new_h), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                ) / 255.0
            scale = w / gray_proc.shape[1]
            min_area_px_scaled = max(1, int(DEFAULT_MIN_AREA_PX / (scale ** 2)))
            centers, _ = detect_centers(gray_proc, p.threshold, p.invert, min_area_px_scaled)
            if scale != 1.0:
                centers *= scale

        if not p.auto_adjust_max_points and p.max_points > 0:
            if allowed_ratio <= 0.0:
                centers = np.empty((0, 2), dtype=np.float32)
            elif sampling_limit > 0 and len(centers) > sampling_limit:
                centers = centers[:sampling_limit]

        colors = sample_colors(rgb, centers)
        detected_count = len(centers)
        effective_max_points = p.max_points
        if p.auto_adjust_max_points and p.max_points > 0 and detected_count > p.max_points:
            effective_max_points = detected_count

        z_vals = None
        hm_w = hm_h = 0
        if p.auto_height and p.height_map_path:
            try:
                hgray, _, hm_w, hm_h = load_image_grayscale_np(
                    bpy.path.abspath(p.height_map_path))
                z_vals = []
                for x, y in centers:
                    xi = int(round(float(x)))
                    yi = int(round(float(y)))
                    xi = min(max(xi, 0), hm_w - 1)
                    yi = min(max(yi, 0), hm_h - 1)
                    z_vals.append(float(hgray[yi, xi]) * p.unit_per_px)
            except Exception as e:
                self.report({'WARNING'}, f"Failed to load height map: {e}")
                z_vals = None

        # Create points in Blender (may add extra vertices)
        obj, n, final_centers, verts = create_vertices_object(
            p.object_name, centers, w, h,
            p.unit_per_px, p.origin_mode, p.flip_y,
            p.collection_name, effective_max_points, pixel_spacing,
            z_values=z_vals,
            range_x=p.output_range_x,
            range_y=p.output_range_y,
            placement_spacing=p.vertex_spacing,
            auto_adjust_max_points=p.auto_adjust_max_points,
        )

        if z_vals is not None and len(verts) == hm_w * hm_h:
            mesh_obj = create_mesh_with_faces(verts, hm_w, hm_h)
            mesh_obj.name = p.object_name + "_Mesh"
            coll = bpy.data.collections.get(p.collection_name)
            if not coll:
                coll = bpy.data.collections.new(p.collection_name)
                bpy.context.scene.collection.children.link(coll)
            coll.objects.link(mesh_obj)

        final_len = len(final_centers)
        if colors.shape[0] > final_len:
            colors = colors[:final_len]
        elif colors.shape[0] < final_len:
            extra = np.zeros((final_len - colors.shape[0], 3), dtype=np.uint8)
            colors = np.vstack([colors, extra])

        # Key colors on object for other addons
        apply_color_keys_action(obj, colors)

        # Save CSV file
        csv_path = None
        if p.save_csv:
            img_path = Path(bpy.path.abspath(p.image_path))
            if p.output_dir:
                out_dir = Path(bpy.path.abspath(p.output_dir))
            else:
                out_dir = Path(bpy.path.abspath("//"))
            csv_path = out_dir / f"{img_path.stem}_points.csv"
            try:
                save_points_color_csv(csv_path, final_centers, colors, w, h,
                                      p.unit_per_px, p.origin_mode, p.flip_y,
                                      range_x=p.output_range_x,
                                      range_y=p.output_range_y)
            except Exception as e:
                self.report({'WARNING'}, f"CSV save failed: {e}")

        msg = f"Detected {detected_count} centers. Created {n} vertices."
        if csv_path:
            msg += f" CSV: {csv_path}"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class DPI_OT_mesh_to_dots(Operator):
    bl_idname = "dpi.mesh_to_dots"
    bl_label = "Mesh to Dots"
    bl_description = "Sample mesh edges and faces to generate dots"

    def execute(self, context):
        p = context.scene.dpi_props
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active mesh object required")
            return {'CANCELLED'}

        def sample_mesh(spacing: float) -> np.ndarray:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            edges = list(bm.edges)
            faces = list(bm.faces)
            mw = obj.matrix_world

            points = []
            for e in edges:
                v0 = np.array(mw @ e.verts[0].co)
                v1 = np.array(mw @ e.verts[1].co)
                pts = sample_edge(v0, v1, spacing)
                if len(pts):
                    points.extend(pts)

            for f in faces:
                verts = [np.array(mw @ v.co) for v in f.verts]
                origin = verts[0]
                normal = np.array(mw.to_3x3() @ f.normal)
                nlen = np.linalg.norm(normal)
                if nlen == 0:
                    continue
                normal /= nlen
                tangent = verts[1] - origin
                tlen = np.linalg.norm(tangent)
                if tlen == 0:
                    continue
                tangent /= tlen
                bitangent = np.cross(normal, tangent)
                blen = np.linalg.norm(bitangent)
                if blen == 0:
                    continue
                bitangent /= blen

                coords2d = []
                for v in verts:
                    p3 = v - origin
                    u = np.dot(p3, tangent)
                    v2 = np.dot(p3, bitangent)
                    coords2d.append((u, v2))
                coords2d = np.array(coords2d)
                min_uv = coords2d.min(axis=0)
                coords_shift = coords2d - min_uv
                max_uv = coords_shift.max(axis=0)
                W = max(1, int(np.floor(max_uv[0] / spacing)) + 1)
                H = max(1, int(np.floor(max_uv[1] / spacing)) + 1)
                img = Image.new('1', (W, H), 0)
                draw = ImageDraw.Draw(img)
                poly_px = [((u) / spacing, (v) / spacing) for u, v in coords_shift]
                draw.polygon(poly_px, fill=1)
                mask = np.array(img, dtype=bool)
                interior = fill_shape(mask, spacing=1.0, mode=p.fill_mode)
                for x, y in interior:
                    u = x * spacing + min_uv[0]
                    v2 = y * spacing + min_uv[1]
                    world_pt = origin + tangent * u + bitangent * v2
                    points.append(world_pt)

            bm.free()
            if not points:
                return np.empty((0, 3), dtype=float)
            return np.unique(np.round(np.array(points, dtype=float), 6), axis=0)

        spacing = compute_geometry_spacing(p)
        pts = sample_mesh(spacing)
        if pts.size == 0:
            self.report({'WARNING'}, "No points generated")
            return {'CANCELLED'}

        effective_max_points = p.max_points
        if p.auto_adjust_max_points and p.max_points > 0 and len(pts) > p.max_points:
            effective_max_points = len(pts)

        if effective_max_points > 0 and len(pts) > effective_max_points:
            if p.auto_adjust_max_points:
                pts = pts[:effective_max_points]
            else:
                eff_spacing = spacing
                while len(pts) > effective_max_points:
                    eff_spacing *= len(pts) / effective_max_points
                    pts = sample_mesh(eff_spacing)
                    if pts.size == 0:
                        break
                if len(pts) > effective_max_points:
                    pts = pts[:effective_max_points]

        create_points_object(p.object_name or "MeshDots", pts, p.collection_name)
        self.report({'INFO'}, f"Created {len(pts)} vertices from mesh")
        return {'FINISHED'}


class DPI_OT_path_to_dots(Operator):
    bl_idname = "dpi.path_to_dots"
    bl_label = "Path to Dots"
    bl_description = "Sample curve paths to generate dots"

    def execute(self, context):
        p = context.scene.dpi_props
        obj = context.active_object
        if not obj or obj.type != 'CURVE':
            self.report({'ERROR'}, "Active curve object required")
            return {'CANCELLED'}

        spacing = compute_geometry_spacing(p)
        pts = sample_curve(obj, spacing)
        points = [np.array(pt) for pt in pts]

        if p.fill_closed and any(s.use_cyclic_u for s in obj.data.splines):
            if len(points) >= 3:
                origin = points[0]
                tangent = points[1] - origin
                tlen = float(np.linalg.norm(tangent))
                if tlen == 0:
                    tangent = np.array([1.0, 0.0, 0.0])
                    tlen = 1.0
                tangent /= tlen
                normal = np.cross(tangent, points[2] - origin)
                nlen = float(np.linalg.norm(normal))
                if nlen == 0:
                    normal = np.array([0.0, 0.0, 1.0])
                    nlen = 1.0
                normal /= nlen
                bitangent = np.cross(normal, tangent)
                blen = float(np.linalg.norm(bitangent))
                if blen == 0:
                    bitangent = np.array([0.0, 1.0, 0.0])
                    blen = 1.0
                bitangent /= blen

                coords = []
                for v in points:
                    p3 = v - origin
                    u = np.dot(p3, tangent)
                    v2 = np.dot(p3, bitangent)
                    coords.append((u, v2))
                coords2d = np.array(coords)
                min_uv = coords2d.min(axis=0)
                coords_shift = coords2d - min_uv
                max_uv = coords_shift.max(axis=0)
                W = max(1, int(np.floor(max_uv[0] / spacing)) + 1)
                H = max(1, int(np.floor(max_uv[1] / spacing)) + 1)
                img = Image.new('1', (W, H), 0)
                draw = ImageDraw.Draw(img)
                poly_px = [((u) / spacing, (v) / spacing) for u, v in coords_shift]
                draw.polygon(poly_px, fill=1)
                mask = np.array(img, dtype=bool)
                interior = fill_shape(mask, spacing=1.0, mode=p.fill_mode)
                for x, y in interior:
                    u = x * spacing + min_uv[0]
                    v2 = y * spacing + min_uv[1]
                    world_pt = origin + tangent * u + bitangent * v2
                    points.append(world_pt)

        if not points:
            self.report({'WARNING'}, "No points generated")
            return {'CANCELLED'}

        pts = np.unique(np.round(np.array(points, dtype=float), 6), axis=0)
        effective_max_points = p.max_points
        if p.auto_adjust_max_points and p.max_points > 0 and len(pts) > p.max_points:
            effective_max_points = len(pts)

        if effective_max_points > 0 and len(pts) > effective_max_points:
            pts = pts[:effective_max_points]
        centers2d = [(p[0], p[1]) for p in pts]
        z_vals = [p[2] for p in pts]
        create_vertices_object(
            p.object_name or "PathDots",
            centers2d,
            1.0,
            1.0,
            1.0,
            'topleft',
            False,
            p.collection_name,
            spacing=spacing,
            z_values=z_vals,
            auto_adjust_max_points=p.auto_adjust_max_points,
        )
        self.report({'INFO'}, f"Created {len(pts)} vertices from path")
        return {'FINISHED'}


class DPI_OT_load_custom_region(Operator):
    bl_idname = "dpi.load_custom_region"
    bl_label = "Store Custom Region"
    bl_description = "Store currently selected vertices as a custom placement region"

    def execute(self, context):
        props = context.scene.dpi_props
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active mesh object required")
            return {'CANCELLED'}

        if obj.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            coords = [v.co.copy() for v in bm.verts if v.select]
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            coords = [v.co.copy() for v in bm.verts if v.select]
            bm.free()

        if len(coords) < 3:
            self.report({'WARNING'}, "Select at least 3 vertices to define a region")
            return {'CANCELLED'}

        data = {
            "object": obj.name,
            "points": [[float(c.x), float(c.y)] for c in coords],
        }
        try:
            props.custom_region_json = json.dumps(data)
        except TypeError as exc:
            self.report({'ERROR'}, f"Failed to serialize region: {exc}")
            return {'CANCELLED'}

        props.custom_region_object_name = obj.name
        props.custom_region_point_count = len(coords)
        props.randomize_use_custom_region = True
        self.report({'INFO'}, f"Stored custom region with {len(coords)} vertices from {obj.name}")
        return {'FINISHED'}


class DPI_OT_randomize_selected_vertices(Operator):
    bl_idname = "dpi.randomize_selected_vertices"
    bl_label = "Apply Placement"
    bl_description = "Reposition selected vertices using the active placement mode while respecting spacing"

    max_attempts: IntProperty(
        name="Attempts",
        description="Maximum attempts per vertex when searching for a valid random position",
        default=200,
        min=1,
        max=2000,
        options={'HIDDEN'},
    )

    def execute(self, context):
        props = context.scene.dpi_props
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active mesh object required")
            return {'CANCELLED'}

        is_edit_mode = (obj.mode == 'EDIT')
        if is_edit_mode:
            bm = bmesh.from_edit_mesh(obj.data)
        else:
            bm = bmesh.new()
            bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        selected_verts = [v for v in bm.verts if v.select]
        if not selected_verts:
            if not is_edit_mode:
                bm.free()
            self.report({'WARNING'}, "No selected vertices to randomize")
            return {'CANCELLED'}

        all_coords = [v.co.copy() for v in bm.verts]
        if not all_coords:
            if not is_edit_mode:
                bm.free()
            self.report({'WARNING'}, "Mesh contains no vertices")
            return {'CANCELLED'}

        spacing = float(props.vertex_spacing) if props.vertex_spacing > 0 else 0.0
        static_coords = [] if props.ignore_unselected_spacing else [v.co.copy() for v in bm.verts if not v.select]
        static_tree = build_kdtree(static_coords) if spacing > 0 and static_coords else None

        mesh_polygon = None
        mesh_source = ""
        mesh_warning: Optional[str] = None
        if props.randomize_use_mesh_region:
            mesh_obj = props.mesh_region_object
            if mesh_obj and mesh_obj.type == 'MESH':
                mesh_polygon = build_mesh_region_polygon(context, obj, mesh_obj)
                mesh_source = mesh_obj.name
                if mesh_polygon is None:
                    mesh_warning = "Selected mesh does not define a planar region"
            else:
                mesh_warning = "Select a mesh object for the mesh region"

        region = create_region(
            all_coords,
            spacing,
            props.randomize_use_custom_region,
            props.custom_region_json,
            mesh_polygon=mesh_polygon,
            mesh_source_object=mesh_source,
            mesh_requested=props.randomize_use_mesh_region,
        )
        if region.requested_custom and not region.using_custom and props.custom_region_json:
            self.report({'WARNING'}, "Stored custom region is invalid. Using bounding box instead.")
        if region.requested_mesh and not region.using_mesh_region:
            warn_msg = mesh_warning or "Mesh region is invalid. Using bounding box instead."
            self.report({'WARNING'}, warn_msg)

        target_coords = [vert.co.copy() for vert in selected_verts]
        if props.placement_mode == 'GRID':
            assignments, skipped = grid_assignments(
                target_coords,
                region,
                spacing,
                static_tree,
            )
            mode_label = "grid"
        else:
            assignments, skipped = randomize_assignments(
                target_coords,
                region,
                spacing,
                static_tree,
                self.max_attempts,
            )
            mode_label = "random"

        moved = 0
        for vert, new_co in zip(selected_verts, assignments):
            if new_co is None:
                continue
            vert.co = new_co
            moved += 1

        if is_edit_mode:
            bmesh.update_edit_mesh(obj.data)
        else:
            bm.to_mesh(obj.data)
            obj.data.update()
            bm.free()

        if region.using_mesh_region and region.source_object:
            if region.source_object != obj.name:
                self.report({'INFO'}, f"Mesh region from '{region.source_object}' applied to '{obj.name}'")
        elif region.using_custom and region.source_object and region.source_object != obj.name:
            self.report({'INFO'}, f"Custom region captured from '{region.source_object}' applied to '{obj.name}'")

        if skipped and not moved:
            self.report({'WARNING'}, "No valid positions found with the current spacing")
            return {'CANCELLED'}

        msg = f"Moved {moved} vertices ({mode_label})"
        if skipped:
            msg += f", skipped {skipped}"
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
        box.prop(p, "conversion_mode")
        if p.conversion_mode in {'SHAPE', 'MIX'}:
            box.prop(p, "outline", text="Outline を生成する")
        box.prop(p, "threshold")
        box.prop(p, "invert")
        if p.conversion_mode in {'SHAPE', 'MIX'}:
            box.prop(p, "fill_mode")
            if p.fill_mode != 'NONE':
                box.prop(p, "fill_ratio")
            box.prop(p, "detect_color_boundary")
        if p.conversion_mode in {'LINE', 'MIX'}:
            box.prop(p, "blur_radius")
            box.prop(p, "thresh_scale")
            box.prop(p, "junction_ratio")
        if p.conversion_mode == 'MIX':
            box.prop(p, "mix_thickness")
        box.prop(p, "resize_to")

        box2 = layout.box()
        box2.label(text="Placement")
        box2.prop(p, "unit_per_px")
        box2.prop(p, "output_range_x")
        box2.prop(p, "output_range_y")
        box2.prop(p, "vertex_spacing")
        box2.prop(p, "ignore_unselected_spacing")
        box2.prop(p, "placement_mode")
        box2.prop(p, "origin_mode")
        box2.prop(p, "flip_y")
        box2.prop(p, "collection_name")
        box2.prop(p, "object_name")
        box2.prop(p, "max_points")
        box2.prop(p, "bottom_reserve_ratio")
        box2.prop(p, "auto_adjust_max_points")
        box2.prop(p, "randomize_use_mesh_region")
        if p.randomize_use_mesh_region:
            box2.prop(p, "mesh_region_object")
        box2.prop(p, "randomize_use_custom_region")
        row = box2.row(align=True)
        row.operator(DPI_OT_load_custom_region.bl_idname, icon='EYEDROPPER')
        row.operator(DPI_OT_randomize_selected_vertices.bl_idname, icon='FORCE_LENNARDJONES')
        if p.custom_region_point_count > 0:
            info = f"{p.custom_region_point_count} pts"
            if p.custom_region_object_name:
                info += f" ({p.custom_region_object_name})"
            box2.label(text=f"Region: {info}")

        box3 = layout.box()
        box3.label(text="Height Map")
        box3.prop(p, "height_map_path")
        box3.prop(p, "auto_height")

        layout.prop(p, "save_csv")
        layout.operator(DPI_OT_detect_and_create.bl_idname, icon='PARTICLES')


class DPI_PT_mesh_panel(Panel):
    bl_label = "Mesh to Dots"
    bl_idname = "DPI_PT_mesh_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Dot Importer"

    def draw(self, context):
        layout = self.layout
        p = context.scene.dpi_props
        box = layout.box()
        box.prop(p, "vertex_spacing")
        box.prop(p, "fill_mode")
        if p.fill_mode != 'NONE':
            box.prop(p, "fill_ratio")
        box.operator(DPI_OT_mesh_to_dots.bl_idname, icon='MESH_DATA')


class DPI_PT_path_panel(Panel):
    bl_label = "Path to Dots"
    bl_idname = "DPI_PT_path_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Dot Importer"

    def draw(self, context):
        layout = self.layout
        p = context.scene.dpi_props
        box = layout.box()
        box.prop(p, "vertex_spacing")
        box.prop(p, "fill_closed")
        box.prop(p, "fill_mode")
        if p.fill_mode != 'NONE':
            box.prop(p, "fill_ratio")
        box.operator(DPI_OT_path_to_dots.bl_idname, icon='CURVE_DATA')


# ---------- Register ----------
classes = (
    DPIProps,
    DPI_OT_detect_and_create,
    DPI_OT_mesh_to_dots,
    DPI_OT_path_to_dots,
    DPI_OT_load_custom_region,
    DPI_OT_randomize_selected_vertices,
    DPI_PT_panel,
    DPI_PT_mesh_panel,
    DPI_PT_path_panel,
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
