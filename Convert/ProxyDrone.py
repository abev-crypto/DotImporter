import bpy

#TODO
# マテリアルをエミッションにする
# 一時的にレンダリングReadyにするために距離チェックを無効にするオプションをGeometryNodeにつける
# GeometryNodeは使い回す

def get_or_create_material(name, base_color):
    """指定カラーのマテリアルを取得 or 作成"""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            r, g, b = base_color
            bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    return mat


def create_gn_sphere_proximity(
    obj,
    sphere_radius=0.4,  # 直径 1m -> 半径 0.5
    threshold=1.55,      # 1.5m 以下なら赤
    group_name="GN_SphereProximity",
):
    # ▼ マテリアル準備
    mat_default = get_or_create_material("Mat_Default", (1.0, 1.0, 1.0))
    mat_red = get_or_create_material("Mat_Red", (1.0, 0.0, 0.0))

    # ▼ Geometry Nodes グループ作成
    ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')

    iface = ng.interface
    iface.new_socket(
        name="Geometry",
        description="Input geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
    )
    iface.new_socket(
        name="Geometry",
        description="Output geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
    )

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    # ---- Nodes ----
    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1000, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (800, 0)

    # Mesh → Points
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-800, 0)
    mesh_to_points.mode = 'VERTICES'
    mesh_to_points.inputs["Radius"].default_value = 0.0

    # Position
    pos_node = nodes.new("GeometryNodeInputPosition")
    pos_node.location = (-800, -200)

    # Index of Nearest
    idx_nearest = nodes.new("GeometryNodeIndexOfNearest")
    idx_nearest.location = (-600, -200)
    # 4.3: data_type / domain プロパティ無し

    # Sample Index
    sample_index = nodes.new("GeometryNodeSampleIndex")
    sample_index.location = (-400, -200)
    sample_index.data_type = 'FLOAT_VECTOR'
    sample_index.domain = 'POINT'

    # Distance
    vec_math_dist = nodes.new("ShaderNodeVectorMath")
    vec_math_dist.location = (-200, -200)
    vec_math_dist.operation = 'DISTANCE'

    # Compare 距離 < threshold
    compare = nodes.new("FunctionNodeCompare")
    compare.location = (0, -200)
    compare.data_type = 'FLOAT'
    compare.operation = 'LESS_THAN'
    compare.inputs[1].default_value = threshold

    # Store Named Attribute: is_close
    store_attr = nodes.new("GeometryNodeStoreNamedAttribute")
    store_attr.location = (200, -100)
    store_attr.data_type = 'BOOLEAN'
    store_attr.domain = 'POINT'
    store_attr.inputs["Name"].default_value = "is_close"

    # UV Sphere（インスタンス用）
    uv_sphere = nodes.new("GeometryNodeMeshUVSphere")
    uv_sphere.location = (-600, 200)
    uv_sphere.inputs["Radius"].default_value = sphere_radius

    # Instance on Points
    inst_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    inst_on_points.location = (0, 200)

    # Realize Instances
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (200, 200)

    # Set Material（全体にデフォルト）
    set_mat_default = nodes.new("GeometryNodeSetMaterial")
    set_mat_default.location = (400, 200)
    set_mat_default.inputs["Material"].default_value = mat_default

    # Named Attribute（is_close を読む）
    named_attr = nodes.new("GeometryNodeInputNamedAttribute")
    named_attr.location = (400, 0)
    named_attr.data_type = 'BOOLEAN'
    named_attr.inputs["Name"].default_value = "is_close"

    # Set Material（近いものだけ赤）
    set_mat_red = nodes.new("GeometryNodeSetMaterial")
    set_mat_red.location = (600, 200)
    set_mat_red.inputs["Material"].default_value = mat_red

    # ---- Links ----

    # 入力メッシュ → Mesh to Points
    links.new(group_in.outputs["Geometry"], mesh_to_points.inputs["Mesh"])

    # 距離計算用
    # Position → Index of Nearest & Sample Index
    links.new(pos_node.outputs["Position"], idx_nearest.inputs["Position"])
    links.new(pos_node.outputs["Position"], sample_index.inputs["Value"])

    # Mesh to Points 出力を Sample Index / Store に渡す
    links.new(mesh_to_points.outputs["Points"], sample_index.inputs["Geometry"])
    links.new(mesh_to_points.outputs["Points"], store_attr.inputs["Geometry"])

    # 近傍インデックス
    links.new(idx_nearest.outputs["Index"], sample_index.inputs["Index"])

    # 距離（自分 vs 近傍）
    links.new(pos_node.outputs["Position"], vec_math_dist.inputs[0])
    links.new(sample_index.outputs["Value"], vec_math_dist.inputs[1])

    # Compare（距離 < threshold）
    links.new(vec_math_dist.outputs["Value"], compare.inputs[0])

    # Store Named Attribute: is_close
    links.new(compare.outputs["Result"], store_attr.inputs["Value"])

    # Store後の Geometry → Instance on Points の Points
    links.new(store_attr.outputs["Geometry"], inst_on_points.inputs["Points"])

    # インスタンス
    links.new(uv_sphere.outputs["Mesh"], inst_on_points.inputs["Instance"])

    # Realize
    links.new(inst_on_points.outputs["Instances"], realize.inputs["Geometry"])

    # デフォルトマテリアル
    links.new(realize.outputs["Geometry"], set_mat_default.inputs["Geometry"])

    # is_close 読み込み
    links.new(set_mat_default.outputs["Geometry"], set_mat_red.inputs["Geometry"])
    links.new(named_attr.outputs["Attribute"], set_mat_red.inputs["Selection"])

    # 出力
    links.new(set_mat_red.outputs["Geometry"], group_out.inputs["Geometry"])

    # モディファイアとして適用
    mod = obj.modifiers.new(name=group_name, type='NODES')
    mod.node_group = ng

    print(f"Geometry Nodes '{group_name}' を {obj.name} に追加しました。")


# 実行部
obj = bpy.context.active_object
if obj is None:
    raise RuntimeError("アクティブオブジェクトがありません。メッシュを選択してから実行してください。")

create_gn_sphere_proximity(obj)
