import bpy


def _setup_emission_material(mat, color, strength):
    """Ensure the given material emits the specified color."""

    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (-200, 0)
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength

    material_output = nodes.new("ShaderNodeOutputMaterial")
    material_output.location = (0, 0)

    links.new(emission.outputs["Emission"], material_output.inputs["Surface"])


def get_or_create_emission_material(name, base_color, strength=5.0):
    """指定カラーのエミッションマテリアルを取得 or 作成"""

    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    _setup_emission_material(mat, base_color, strength)
    return mat


def _set_identifier_if_possible(socket, identifier):
    if not identifier:
        return

    prop = None
    if hasattr(socket, "bl_rna"):
        prop = socket.bl_rna.properties.get("identifier")

    if prop is not None and getattr(prop, "is_readonly", False):
        return

    try:
        socket.identifier = identifier
    except AttributeError:
        pass


def _ensure_interface_socket(interface, *, name, description, in_out, socket_type, default=None, identifier=None):
    socket = None
    for item in interface.items_tree:
        if getattr(item, "name", None) == name and getattr(item, "in_out", None) == in_out:
            socket = item
            break
    if socket is None:
        socket = interface.new_socket(
            name=name,
            description=description,
            in_out=in_out,
            socket_type=socket_type,
        )
    _set_identifier_if_possible(socket, identifier)
    if default is not None:
        socket.default_value = default
    return socket


def _build_node_group(ng, *, sphere_radius, threshold, mat_default, mat_red):
    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    iface = ng.interface

    geo_socket = _ensure_interface_socket(
        iface,
        name="Geometry",
        description="Input geometry",
        in_out='INPUT',
        socket_type='NodeSocketGeometry',
        identifier="Geometry",
    )
    enable_socket = _ensure_interface_socket(
        iface,
        name="Enable Distance Check",
        description="Disable to skip marking points as close",
        in_out='INPUT',
        socket_type='NodeSocketBool',
        default=True,
        identifier="EnableDistanceCheck",
    )
    geo_out_socket = _ensure_interface_socket(
        iface,
        name="Geometry",
        description="Output geometry",
        in_out='OUTPUT',
        socket_type='NodeSocketGeometry',
        identifier="GeometryOut",
    )

    input_sockets = [item for item in iface.items_tree if getattr(item, "in_out", None) == 'INPUT']
    output_sockets = [item for item in iface.items_tree if getattr(item, "in_out", None) == 'OUTPUT']

    geo_index = input_sockets.index(geo_socket)
    enable_index = input_sockets.index(enable_socket)
    geo_out_index = output_sockets.index(geo_out_socket)

    group_in = nodes.new("NodeGroupInput")
    group_in.location = (-1200, 0)

    group_out = nodes.new("NodeGroupOutput")
    group_out.location = (900, 0)

    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.location = (-1000, 0)
    mesh_to_points.mode = 'VERTICES'
    mesh_to_points.inputs["Radius"].default_value = 0.0

    pos_node = nodes.new("GeometryNodeInputPosition")
    pos_node.location = (-1000, -200)

    idx_nearest = nodes.new("GeometryNodeIndexOfNearest")
    idx_nearest.location = (-800, -200)

    sample_index = nodes.new("GeometryNodeSampleIndex")
    sample_index.location = (-600, -200)
    sample_index.data_type = 'FLOAT_VECTOR'
    sample_index.domain = 'POINT'

    vec_math_dist = nodes.new("ShaderNodeVectorMath")
    vec_math_dist.location = (-400, -200)
    vec_math_dist.operation = 'DISTANCE'

    compare = nodes.new("FunctionNodeCompare")
    compare.location = (-200, -200)
    compare.data_type = 'FLOAT'
    compare.operation = 'LESS_THAN'
    compare.inputs[1].default_value = threshold

    bool_math = nodes.new("FunctionNodeBooleanMath")
    bool_math.location = (0, -150)
    bool_math.operation = 'AND'

    store_attr = nodes.new("GeometryNodeStoreNamedAttribute")
    store_attr.location = (200, -100)
    store_attr.data_type = 'BOOLEAN'
    store_attr.domain = 'POINT'
    store_attr.inputs["Name"].default_value = "is_close"

    uv_sphere = nodes.new("GeometryNodeMeshUVSphere")
    uv_sphere.location = (-800, 200)
    uv_sphere.inputs["Radius"].default_value = sphere_radius

    inst_on_points = nodes.new("GeometryNodeInstanceOnPoints")
    inst_on_points.location = (-200, 200)

    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (0, 200)

    set_mat_default = nodes.new("GeometryNodeSetMaterial")
    set_mat_default.location = (200, 200)
    set_mat_default.inputs["Material"].default_value = mat_default

    named_attr = nodes.new("GeometryNodeInputNamedAttribute")
    named_attr.location = (200, 0)
    named_attr.data_type = 'BOOLEAN'
    named_attr.inputs["Name"].default_value = "is_close"

    set_mat_red = nodes.new("GeometryNodeSetMaterial")
    set_mat_red.location = (400, 200)
    set_mat_red.inputs["Material"].default_value = mat_red

    links.new(group_in.outputs[geo_index], mesh_to_points.inputs["Mesh"])
    links.new(group_in.outputs[enable_index], bool_math.inputs[1])

    links.new(mesh_to_points.outputs["Points"], sample_index.inputs["Geometry"])
    links.new(mesh_to_points.outputs["Points"], store_attr.inputs["Geometry"])

    links.new(pos_node.outputs["Position"], idx_nearest.inputs["Position"])
    links.new(pos_node.outputs["Position"], sample_index.inputs["Value"])
    links.new(idx_nearest.outputs["Index"], sample_index.inputs["Index"])

    links.new(pos_node.outputs["Position"], vec_math_dist.inputs[0])
    links.new(sample_index.outputs["Value"], vec_math_dist.inputs[1])

    links.new(vec_math_dist.outputs["Value"], compare.inputs[0])
    links.new(compare.outputs["Result"], bool_math.inputs[0])

    links.new(bool_math.outputs["Result"], store_attr.inputs["Value"])

    links.new(store_attr.outputs["Geometry"], inst_on_points.inputs["Points"])
    links.new(uv_sphere.outputs["Mesh"], inst_on_points.inputs["Instance"])
    links.new(inst_on_points.outputs["Instances"], realize.inputs["Geometry"])

    links.new(realize.outputs["Geometry"], set_mat_default.inputs["Geometry"])
    links.new(named_attr.outputs["Attribute"], set_mat_red.inputs["Selection"])
    links.new(set_mat_default.outputs["Geometry"], set_mat_red.inputs["Geometry"])

    links.new(set_mat_red.outputs["Geometry"], group_out.inputs[geo_out_index])

    return enable_socket


def create_gn_sphere_proximity(
    obj,
    *,
    sphere_radius=0.4,
    threshold=1.55,
    group_name="GN_SphereProximity",
    enable_distance_check=None,
    emission_strength=5.0,
):
    """アクティブオブジェクトにGN_SphereProximityモディファイアを作成/適用"""

    if obj is None or obj.type != 'MESH':
        raise TypeError("メッシュオブジェクトに対してのみ使用できます")

    mat_default = get_or_create_emission_material(
        "Mat_Default", (1.0, 1.0, 1.0), emission_strength
    )
    mat_red = get_or_create_emission_material(
        "Mat_Red", (1.0, 0.0, 0.0), emission_strength
    )

    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')

    enable_socket = _build_node_group(
        ng,
        sphere_radius=sphere_radius,
        threshold=threshold,
        mat_default=mat_default,
        mat_red=mat_red,
    )

    modifier = None
    for mod in obj.modifiers:
        if mod.type == 'NODES' and mod.node_group == ng:
            modifier = mod
            break
        if mod.type == 'NODES' and mod.name == group_name:
            modifier = mod
            break

    if modifier is None:
        modifier = obj.modifiers.new(name=group_name, type='NODES')

    modifier.node_group = ng

    if enable_distance_check is not None:
        identifier = getattr(enable_socket, "identifier", "EnableDistanceCheck")
        try:
            modifier[identifier] = enable_distance_check
        except Exception:
            try:
                modifier["Input_2"] = enable_distance_check
            except Exception:
                pass

    return modifier
