import bpy


def create_wave_gn(name="GN_VertexWaveXYZ"):
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]

    tree = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    iface = tree.interface

    # ====== 入出力ソケット ======
    
        # 入力
    iface.new_socket(
        name="Geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
        description="Input geometry"
    )
    iface.new_socket(
        name="Amplitude",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Wave height"
    )
    iface.new_socket(
        name="Frequency",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Wave frequency"
    )
    iface.new_socket(
        name="Speed",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="Wave speed"
    )
    iface.new_socket(
        name="XStrength",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="XStrength"
    )
    iface.new_socket(
        name="YStrength",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="YStrength"
    )
    iface.new_socket(
        name="ZStrength",
        in_out='INPUT',
        socket_type='NodeSocketFloat',
        description="ZStrength"
    )
    # 出力
    iface.new_socket(
        name="Geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        description="Output geometry"
    )

    # ====== ノード ======
    group_in = tree.nodes.new("NodeGroupInput")
    group_out = tree.nodes.new("NodeGroupOutput")
    group_in.location = (-1200, 0)
    group_out.location = (400, 0)

    # デフォルト値
    for sock in group_in.outputs:
        if sock.name == "Amplitude":
            sock.default_value = 0.2
        elif sock.name == "Frequency":
            sock.default_value = 2.0
        elif sock.name == "Speed":
            sock.default_value = 1.0
        elif sock.name in ("XStrength", "YStrength", "ZStrength"):
            sock.default_value = 0.0  # 0 なら影響なし
    # ZStrength だけ標準有効にしたい場合は 1.0

    # 必要ノード
    links = tree.links

    n_setpos = tree.nodes.new("GeometryNodeSetPosition")
    n_setpos.location = (-200, 0)

    n_pos = tree.nodes.new("GeometryNodeInputPosition")
    n_pos.location = (-1500, 50)

    n_sep = tree.nodes.new("ShaderNodeSeparateXYZ")
    n_sep.location = (-1300, 50)

    n_time = tree.nodes.new("GeometryNodeInputSceneTime")
    n_time.location = (-1500, -200)

    n_x_mul_freq = tree.nodes.new("ShaderNodeMath")
    n_x_mul_freq.operation = 'MULTIPLY'
    n_x_mul_freq.location = (-1100, 50)

    n_t_mul_speed = tree.nodes.new("ShaderNodeMath")
    n_t_mul_speed.operation = 'MULTIPLY'
    n_t_mul_speed.location = (-1100, -150)

    n_phase = tree.nodes.new("ShaderNodeMath")
    n_phase.operation = 'ADD'
    n_phase.location = (-900, -50)

    n_sin = tree.nodes.new("ShaderNodeMath")
    n_sin.operation = 'SINE'
    n_sin.location = (-700, -50)

    n_amp = tree.nodes.new("ShaderNodeMath")
    n_amp.operation = 'MULTIPLY'
    n_amp.location = (-500, -50)

    # 各軸に強さを掛ける
    n_strength_x = tree.nodes.new("ShaderNodeMath")
    n_strength_x.operation = 'MULTIPLY'
    n_strength_x.location = (-300, 100)

    n_strength_y = tree.nodes.new("ShaderNodeMath")
    n_strength_y.operation = 'MULTIPLY'
    n_strength_y.location = (-300, -50)

    n_strength_z = tree.nodes.new("ShaderNodeMath")
    n_strength_z.operation = 'MULTIPLY'
    n_strength_z.location = (-300, -200)

    # Combine XYZ
    n_combine = tree.nodes.new("ShaderNodeCombineXYZ")
    n_combine.location = (-50, -50)

    # ====== リンク ======

    links.new(group_in.outputs["Geometry"], n_setpos.inputs["Geometry"])
    links.new(n_setpos.outputs["Geometry"], group_out.inputs["Geometry"])

    links.new(n_pos.outputs["Position"], n_sep.inputs["Vector"])

    links.new(n_sep.outputs["X"], n_x_mul_freq.inputs[0])
    links.new(group_in.outputs["Frequency"], n_x_mul_freq.inputs[1])

    links.new(n_time.outputs["Seconds"], n_t_mul_speed.inputs[0])
    links.new(group_in.outputs["Speed"], n_t_mul_speed.inputs[1])

    links.new(n_x_mul_freq.outputs[0], n_phase.inputs[0])
    links.new(n_t_mul_speed.outputs[0], n_phase.inputs[1])

    links.new(n_phase.outputs[0], n_sin.inputs[0])
    links.new(n_sin.outputs[0], n_amp.inputs[0])
    links.new(group_in.outputs["Amplitude"], n_amp.inputs[1])

    # 軸方向の強さ
    links.new(n_amp.outputs[0], n_strength_x.inputs[0])
    links.new(n_amp.outputs[0], n_strength_y.inputs[0])
    links.new(n_amp.outputs[0], n_strength_z.inputs[0])

    links.new(group_in.outputs["XStrength"], n_strength_x.inputs[1])
    links.new(group_in.outputs["YStrength"], n_strength_y.inputs[1])
    links.new(group_in.outputs["ZStrength"], n_strength_z.inputs[1])

    # 強さ付き出力を Combine XYZ
    links.new(n_strength_x.outputs[0], n_combine.inputs["X"])
    links.new(n_strength_y.outputs[0], n_combine.inputs["Y"])
    links.new(n_strength_z.outputs[0], n_combine.inputs["Z"])

    links.new(n_combine.outputs["Vector"], n_setpos.inputs["Offset"])

    return tree


def add_wave_modifier_to_active_object():
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        raise RuntimeError("アクティブなメッシュを選択してください。")

    node_group = create_wave_gn()

    mod = obj.modifiers.new(name="VertexWaveXYZ", type='NODES')
    mod.node_group = node_group

    print("Wave XYZ GN added to", obj.name)


if __name__ == "__main__":
    add_wave_modifier_to_active_object()
