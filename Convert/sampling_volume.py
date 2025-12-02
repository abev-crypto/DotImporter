import bpy
import bmesh
import math
import random
from mathutils import Vector
from mathutils.bvhtree import BVHTree

# =============================
# 設定
# =============================
MAX_POINTS = 500              # 使ってよいポイント最大数
POINT_OBJ_NAME = "GridPoints"  # 生成されるポイントオブジェクト名

# bbox 内部の充填率を見積もるときの粗いサンプル分解能
# 16 なら 16^3 = 4096 サンプル
FILL_SAMPLE_RES = 16


# -----------------------------
# BVH 作成まわり
# -----------------------------
def build_bvh_in_world(obj):
    """オブジェクトをワールド座標に変換したメッシュから BVH を作る。"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    eval_mesh = eval_obj.to_mesh()

    # ワールド座標に変換
    eval_mesh.transform(eval_obj.matrix_world)

    # Mesh → BMesh → BVH
    bm = bmesh.new()
    bm.from_mesh(eval_mesh)
    bm.normal_update()
    bvh = BVHTree.FromBMesh(bm)
    bm.free()

    # メモリ解放
    eval_obj.to_mesh_clear()

    return bvh


def point_inside_mesh_bvh(bvh, point: Vector) -> bool:
    """BVH を使って point がメッシュの内側か判定する。"""
    if bvh is None:
        return False

    loc, normal, index, dist = bvh.find_nearest(point)
    if loc is None or normal is None:
        return False

    v = point - loc
    # 法線の裏側なら内側とみなす（法線が外向き前提）
    return v.dot(normal) < 0.0


# -----------------------------
# 充填率推定（粗いスキャン）
# -----------------------------
def estimate_fill_fraction(bvh, min_x, max_x, min_y, max_y, min_z, max_z, res: int) -> float:
    """バウンディングボックス内に占めるメッシュ内部の割合を粗く推定する。"""
    if res <= 0:
        return 0.0

    total = 0
    inside = 0

    sx = max_x - min_x
    sy = max_y - min_y
    sz = max_z - min_z

    if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
        return 0.0

    for ix in range(res):
        x = min_x + (ix + 0.5) / res * sx
        for iy in range(res):
            y = min_y + (iy + 0.5) / res * sy
            for iz in range(res):
                z = min_z + (iz + 0.5) / res * sz
                p = Vector((x, y, z))
                total += 1
                if point_inside_mesh_bvh(bvh, p):
                    inside += 1

    if total == 0:
        return 0.0

    fill = inside / total
    print(f"Fill fraction estimate: inside={inside}, total={total}, fraction={fill:.4f}")
    return fill


# -----------------------------
# ポイントクラウドオブジェクト作成
# -----------------------------
def create_point_cloud_mesh(points, name="GridPoints"):
    """points (Vector のリスト) から頂点のみのメッシュオブジェクトを生成する。"""
    me = bpy.data.meshes.new(name)
    me.from_pydata(points, [], [])
    me.update()

    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    return obj


# -----------------------------
# メイン処理
# -----------------------------
def main():
    obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        raise RuntimeError("アクティブオブジェクトが MESH ではありません。")

    # BVH を構築
    bvh = build_bvh_in_world(obj)

    # バウンディングボックス（ワールド座標）
    world_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_x = min(c.x for c in world_corners)
    max_x = max(c.x for c in world_corners)
    min_y = min(c.y for c in world_corners)
    max_y = max(c.y for c in world_corners)
    min_z = min(c.z for c in world_corners)
    max_z = max(c.z for c in world_corners)

    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    bbox_volume = max(size_x, 0.0) * max(size_y, 0.0) * max(size_z, 0.0)

    print(f"Bounding box size: {size_x:.4f}, {size_y:.4f}, {size_z:.4f}")
    print(f"Bounding box volume: {bbox_volume:.4f}")

    if bbox_volume <= 0.0:
        raise RuntimeError("バウンディングボックスの体積が 0 です。スケールを確認してください。")

    # まず粗く充填率を推定
    fill_fraction = estimate_fill_fraction(bvh, min_x, max_x, min_y, max_y, min_z, max_z, FILL_SAMPLE_RES)

    if fill_fraction <= 0.0:
        # まったく内部が見つからない場合は適当な値にフォールバック
        # （極端に薄い板ポリなどの場合）
        fill_fraction = 0.1
        print("Fill fraction estimate is 0; falling back to 0.1")

    estimated_volume = bbox_volume * fill_fraction
    print(f"Estimated solid volume: {estimated_volume:.6f}")

    # ----------------------------
    # グリッド間隔 d を計算
    #   estimated_volume / d^3 ≒ MAX_POINTS
    # ----------------------------
    d = (estimated_volume / MAX_POINTS) ** (1.0 / 3.0)
    print(f"Grid spacing d: {d}")

    # 数値誤差対策で少しマージン
    margin = d * 0.5
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin
    min_z -= margin
    max_z += margin

    # ステップ数
    nx = max(1, int(math.ceil((max_x - min_x) / d)))
    ny = max(1, int(math.ceil((max_y - min_y) / d)))
    nz = max(1, int(math.ceil((max_z - min_z) / d)))

    print(f"Grid resolution: {nx} x {ny} x {nz}")

    points = []

    # ----------------------------
    # 本番グリッドを走査して内部点だけ集める
    # ----------------------------
    for ix in range(nx + 1):
        x = min_x + ix * d
        for iy in range(ny + 1):
            y = min_y + iy * d
            for iz in range(nz + 1):
                z = min_z + iz * d
                p = Vector((x, y, z))

                if point_inside_mesh_bvh(bvh, p):
                    points.append(p)

    print(f"Generated {len(points)} points (target: {MAX_POINTS})")

    # ----------------------------
    # 多すぎる場合はランダム間引き
    # ----------------------------
    if len(points) > MAX_POINTS:
        print(f"Too many points, randomly sampling down to {MAX_POINTS}")
        points = random.sample(points, MAX_POINTS)

    # ----------------------------
    # ポイントオブジェクトを作成
    # ----------------------------
    if points:
        create_point_cloud_mesh(points, POINT_OBJ_NAME)
    else:
        print("内部ポイントが生成されませんでした。メッシュやスケールを確認してください。")


if __name__ == "__main__":
    main()
