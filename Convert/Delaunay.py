import bpy
from mathutils import Vector
from mathutils.geometry import delaunay_2d_cdt


def delaunay_from_vertices(obj, *, epsilon: float = 0.0001):
    """
    Active メッシュの XY 投影に対して delaunay_2d_cdt を実行し、
    新しい三角形メッシュオブジェクトを返す。

    Returns:
        (Object | None, str | None): 新規オブジェクトとエラーメッセージ。
    """

    if obj is None or obj.type != 'MESH':
        return None, "アクティブオブジェクトが MESH ではありません"

    mesh = obj.data

    if len(mesh.vertices) < 3:
        return None, "頂点が3つ未満です"

    verts_2d = [v.co.to_2d() for v in mesh.vertices]

    try:
        verts2d_out, edges, faces, overts, oedges, ofaces = delaunay_2d_cdt(
            verts_2d,
            [],
            [],
            0,        # output_type: 0 で普通の三角形メッシュ
            epsilon,
        )
    except Exception as exc:  # pragma: no cover - mathutils internal
        return None, f"delaunay_2d_cdt の実行に失敗しました: {exc}"

    if not faces:
        return None, "三角形が生成されませんでした"

    verts_3d = [
        (v.x, v.y, mesh.vertices[overts[i][0]].co.z)
        for i, v in enumerate(verts2d_out)
    ]

    new_mesh = bpy.data.meshes.new(mesh.name + "_delaunay")
    new_mesh.from_pydata(verts_3d, edges, faces)
    new_mesh.update()

    new_obj = bpy.data.objects.new(obj.name + "_delaunay", new_mesh)
    new_obj.matrix_world = obj.matrix_world.copy()

    scene = bpy.context.scene
    scene.collection.objects.link(new_obj)

    bpy.context.view_layer.objects.active = new_obj
    new_obj.select_set(True)
    obj.select_set(False)

    return new_obj, None


def delaunay_from_points(points, *, epsilon: float = 0.0001, name: str = "points"):
    """
    XY 平面上のポイントリストを delaunay_2d_cdt で三角形化し、
    新しいメッシュオブジェクトを返す。

    Args:
        points (Iterable[Vector | tuple]): ワールド座標系のポイント。
        epsilon (float): 2D Delaunay 計算の許容誤差。
        name (str): 新オブジェクト/メッシュ名のベース。

    Returns:
        (Object | None, str | None): 新規オブジェクトとエラーメッセージ。
    """

    pts = [p.copy() if hasattr(p, "copy") else Vector(p) for p in points]
    if len(pts) < 3:
        return None, "ポイントが3つ未満です"

    verts_2d = [p.to_2d() for p in pts]

    try:
        verts2d_out, edges, faces, overts, oedges, ofaces = delaunay_2d_cdt(
            verts_2d,
            [],
            [],
            0,
            epsilon,
        )
    except Exception as exc:  # pragma: no cover - mathutils internal
        return None, f"delaunay_2d_cdt の実行に失敗しました: {exc}"

    if not faces:
        return None, "三角形が生成されませんでした"

    verts_3d = [
        (v.x, v.y, pts[overts[i][0]].z)
        for i, v in enumerate(verts2d_out)
    ]

    mesh_name = f"{name}_delaunay"
    new_mesh = bpy.data.meshes.new(mesh_name)
    new_mesh.from_pydata(verts_3d, edges, faces)
    new_mesh.update()

    obj_name = f"{name}_delaunay"
    new_obj = bpy.data.objects.new(obj_name, new_mesh)
    bpy.context.scene.collection.objects.link(new_obj)

    bpy.context.view_layer.objects.active = new_obj
    new_obj.select_set(True)

    return new_obj, None
