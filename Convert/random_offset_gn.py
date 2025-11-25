import bpy


def ensure_random_axis_offset_group(name="GN_RandomAxisOffset_PerVertex"):
    # 既にあればそれを使う
    node_group = bpy.data.node_groups.get(name)
    if node_group is not None and node_group.bl_idname == "GeometryNodeTree":
        return node_group

    # 無ければ新規作成
    node_group = bpy.data.node_groups.new(name=name, type="GeometryNodeTree")

    # ==== インターフェース ====
    iface = node_group.interface

    # 入力: Geometry
    iface.new_socket(
        name="Geometry",
        in_out='INPUT',
        socket_type="NodeSocketGeometry",
        description="Input geometry",
    )

    # 入力: Axis (どの方向に動かすか)
    s_axis = iface.new_socket(
        name="Axis",
        in_out='INPUT',
        socket_type="NodeSocketVector",
        description="Axis direction for offset",
    )
    s_axis.default_value = (0.0, 1.0, 0.0)  # デフォルト Z 方向

    # 入力: Min Offset (m)
    s_min = iface.new_socket(
        name="Min Offset (m)",
        in_out='INPUT',
        socket_type="NodeSocketFloat",
        description="Minimum offset amount in meters",
    )
    s_min.min_value = -1e6
    s_min.max_value = 1e6
    s_min.default_value = -0.25

    # 入力: Max Offset (m)
    s_max = iface.new_socket(
        name="Max Offset (m)",
        in_out='INPUT',
        socket_type="NodeSocketFloat",
        description="Maximum offset amount in meters",
    )
    s_max.min_value = -1e6
    s_max.max_value = 1e6
    s_max.default_value = 0.25

    # 入力: Seed
    s_seed = iface.new_socket(
        name="Seed",
        in_out='INPUT',
        socket_type="NodeSocketInt",
        description="Random seed",
    )
    s_seed.default_value = 0

    # 出力: Geometry
    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type="NodeSocketGeometry",
        description="Output geometry",
    )

    # ==== ノード ====
    nodes = node_group.nodes
    links = node_group.links
    nodes.clear()

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-600, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (400, 0)

    set_position = nodes.new("GeometryNodeSetPosition")
    set_position.location = (150, 0)

    index_node = nodes.new("GeometryNodeInputIndex")
    index_node.location = (-600, -250)

    random_value = nodes.new("FunctionNodeRandomValue")
    random_value.location = (-250, -250)
    if hasattr(random_value, "data_type"):
        random_value.data_type = 'FLOAT'

    vec_normalize = nodes.new("ShaderNodeVectorMath")
    vec_normalize.location = (-350, 100)
    vec_normalize.operation = 'NORMALIZE'

    vec_scale = nodes.new("ShaderNodeVectorMath")
    vec_scale.location = (-50, 100)
    vec_scale.operation = 'SCALE'

    # ==== 接続 ====

    # Geometry
    links.new(group_in.outputs["Geometry"], set_position.inputs["Geometry"])
    links.new(set_position.outputs["Geometry"], group_out.inputs["Geometry"])

    # Axis → Normalize → Scale.Vector
    links.new(group_in.outputs["Axis"], vec_normalize.inputs["Vector"])
    links.new(vec_normalize.outputs["Vector"], vec_scale.inputs["Vector"])

    # Min/Max → Random Value
    links.new(group_in.outputs["Min Offset (m)"], random_value.inputs["Min"])
    links.new(group_in.outputs["Max Offset (m)"], random_value.inputs["Max"])

    # Index → RandomValue.ID
    links.new(index_node.outputs["Index"], random_value.inputs["ID"])

    # Seed → RandomValue.Seed
    links.new(group_in.outputs["Seed"], random_value.inputs["Seed"])

    # RandomValue.Value → Scale.Scale
    links.new(random_value.outputs["Value"], vec_scale.inputs["Scale"])

    # Scale.Vector → SetPosition.Offset
    links.new(vec_scale.outputs["Vector"], set_position.inputs["Offset"])

    return node_group


def add_modifier_to_selected(node_group):
    """選択中の MESH オブジェクトすべてにモディファイアを追加"""
    for obj in bpy.context.selected_objects:
        if obj.type != 'MESH':
            continue
        mod = obj.modifiers.new(name="RandomAxisOffset_PerVertex", type='NODES')
        mod.node_group = node_group


def main():
    group = ensure_random_axis_offset_group()
    add_modifier_to_selected(group)


if __name__ == "__main__":
    main()
