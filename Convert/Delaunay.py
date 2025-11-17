import bpy
from mathutils.geometry import delaunay_2d_cdt

def delaunay_from_vertices(obj):
    if obj is None or obj.type != 'MESH':
        print("アクティブオブジェクトが MESH ではない")
        return

    mesh = obj.data

    if len(mesh.vertices) < 3:
        print("頂点が3つ未満です")
        return

    # ---- 2D座標を用意（XY投影）----
    # もし元のメッシュがXY平面上にあるならこれでOK
    verts_2d = [v.co.to_2d() for v in mesh.vertices]

    # 制約エッジ・制約ポリゴンは無し
    input_edges = []
    input_faces = []

    # output_type と epsilon を指定して呼び出し（★ここがポイント）
    verts2d_out, edges, faces, overts, oedges, ofaces = delaunay_2d_cdt(
        verts_2d,
        input_edges,
        input_faces,
        0,        # output_type: 0 で普通の三角形メッシュ
        0.0001    # epsilon: 数値が小さすぎると数値誤差で問題が出る場合あり
    )

    # ---- 3D座標に戻す ----
    # overts[i][0] で「この出力頂点が元の何番頂点由来か」が分かるので、
    # そのZ値を使って3D座標を構築
    verts_3d = [
        (v.x,
         v.y,
         mesh.vertices[overts[i][0]].co.z)
        for i, v in enumerate(verts2d_out)
    ]

    # ---- 新しいメッシュを作成 ----
    new_mesh = bpy.data.meshes.new(mesh.name + "_delaunay")
    new_mesh.from_pydata(verts_3d, edges, faces)
    new_mesh.update()

    # ---- 新しいオブジェクトとしてシーンにリンク ----
    new_obj = bpy.data.objects.new(obj.name + "_delaunay", new_mesh)
    new_obj.matrix_world = obj.matrix_world.copy()

    scene = bpy.context.scene
    scene.collection.objects.link(new_obj)

    # 選択切り替え
    bpy.context.view_layer.objects.active = new_obj
    new_obj.select_set(True)
    obj.select_set(False)

    print("Delaunay 三角形メッシュ作成完了:", new_obj.name)


# 実行
delaunay_from_vertices(bpy.context.active_object)
