import bpy


NODE_GROUP_NAME = "GN_SphereProximity"


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


def _get_socket(collection, *, name=None, index=0):
    if name:
        try:
            return collection[name]
        except (KeyError, TypeError, ValueError):
            pass

    try:
        return collection[index]
    except (KeyError, IndexError, TypeError):
        pass

    for item in collection:
        return item

    return None


def _create_torus_geometry(nodes, links, *, torus_radius, torus_minor_radius, major_segments, minor_segments):
    """Return a torus mesh node, falling back to curve setup if the primitive is unavailable."""
    try:
        torus_node = nodes.new("GeometryNodeMeshTorus")
    except RuntimeError:
        path_circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        path_circle.location = (-1100, 420)
        path_circle.inputs["Radius"].default_value = torus_radius
        path_circle.inputs["Resolution"].default_value = int(major_segments)

        profile_circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
        profile_circle.location = (-1100, 240)
        profile_circle.inputs["Radius"].default_value = torus_minor_radius
        profile_circle.inputs["Resolution"].default_value = int(minor_segments)

        curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
        curve_to_mesh.location = (-900, 420)

        links.new(path_circle.outputs["Curve"], curve_to_mesh.inputs["Curve"])
        links.new(profile_circle.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
        return curve_to_mesh

    torus_node.location = (-900, 420)
    torus_node.inputs["Major Radius"].default_value = torus_radius
    torus_node.inputs["Minor Radius"].default_value = torus_minor_radius
    torus_node.inputs["Major Segments"].default_value = major_segments
    torus_node.inputs["Minor Segments"].default_value = minor_segments
    return torus_node


def _build_node_group(
    ng,
    *,
    sphere_radius,
    threshold,
    mat_default,
    mat_red,
    torus_radius,
    torus_minor_radius,
):
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
    torus_socket = _ensure_interface_socket(
        iface,
        name="Enable Torus Mesh",
        description="Disable to hide the torus helper mesh",
        in_out='INPUT',
        socket_type='NodeSocketBool',
        default=True,
        identifier="EnableTorusMesh",
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
    torus_index = input_sockets.index(torus_socket)
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
    uv_sphere.location = (-900, 200)
    uv_sphere.inputs["Radius"].default_value = sphere_radius
    uv_sphere.inputs["Segments"].default_value = 24
    uv_sphere.inputs["Rings"].default_value = 12

    torus_major_segments = 64
    torus_minor_segments = 32
    mesh_torus = _create_torus_geometry(
        nodes,
        links,
        torus_radius=torus_radius,
        torus_minor_radius=torus_minor_radius,
        major_segments=torus_major_segments,
        minor_segments=torus_minor_segments,
    )

    mesh_line = nodes.new("GeometryNodeMeshLine")
    mesh_line.location = (-1100, 420)
    mesh_line.inputs["Count"].default_value = 0

    torus_switch = nodes.new("GeometryNodeSwitch")
    torus_switch.location = (-700, 420)
    torus_switch.input_type = 'GEOMETRY'

    join_instances = nodes.new("GeometryNodeJoinGeometry")
    join_instances.location = (-500, 300)
    while len(join_instances.inputs) < 2:
        join_instances.inputs.new('NodeSocketGeometry', "Geometry")

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
    compare_result_socket = _get_socket(compare.outputs, name="Result")
    bool_input_socket = _get_socket(bool_math.inputs, index=0)
    if compare_result_socket and bool_input_socket:
        links.new(compare_result_socket, bool_input_socket)

    bool_result_socket = _get_socket(bool_math.outputs, name="Result")
    store_value_socket = _get_socket(store_attr.inputs, name="Value", index=0)
    if bool_result_socket and store_value_socket:
        links.new(bool_result_socket, store_value_socket)

    links.new(store_attr.outputs["Geometry"], inst_on_points.inputs["Points"])

    torus_mesh_socket = _get_socket(mesh_torus.outputs, name="Mesh")
    if torus_mesh_socket:
        links.new(torus_mesh_socket, torus_switch.inputs["True"])
    links.new(mesh_line.outputs["Mesh"], torus_switch.inputs["False"])
    links.new(group_in.outputs[torus_index], torus_switch.inputs["Switch"])

    links.new(uv_sphere.outputs["Mesh"], join_instances.inputs[0])
    links.new(torus_switch.outputs["Output"], join_instances.inputs[1])
    links.new(join_instances.outputs["Geometry"], inst_on_points.inputs["Instance"])
    links.new(inst_on_points.outputs["Instances"], realize.inputs["Geometry"])

    links.new(realize.outputs["Geometry"], set_mat_default.inputs["Geometry"])
    links.new(named_attr.outputs["Attribute"], set_mat_red.inputs["Selection"])
    links.new(set_mat_default.outputs["Geometry"], set_mat_red.inputs["Geometry"])

    links.new(set_mat_red.outputs["Geometry"], group_out.inputs[geo_out_index])

    ng["distance_socket_identifier"] = getattr(enable_socket, "identifier", "")
    ng["torus_socket_identifier"] = getattr(torus_socket, "identifier", "")

    return enable_socket, torus_socket


def create_gn_sphere_proximity(
    obj,
    *,
    sphere_radius=0.4,
    threshold=1.55,
    group_name=NODE_GROUP_NAME,
    enable_distance_check=None,
    enable_torus=True,
    emission_strength=5.0,
    torus_radius=0.78,
    torus_minor_radius=0.08,
):
    """アクティブオブジェクトにGN_SphereProximityモディファイアを作成/適用"""

    mat_default = get_or_create_emission_material(
        "Mat_Default", (1.0, 1.0, 1.0), emission_strength
    )
    mat_red = get_or_create_emission_material(
        "Mat_Red", (1.0, 0.0, 0.0), emission_strength
    )

    ng = bpy.data.node_groups.get(group_name)
    if ng is None:
        ng = bpy.data.node_groups.new(group_name, 'GeometryNodeTree')

    enable_socket, torus_socket = _build_node_group(
        ng,
        sphere_radius=sphere_radius,
        threshold=threshold,
        mat_default=mat_default,
        mat_red=mat_red,
        torus_radius=torus_radius,
        torus_minor_radius=torus_minor_radius,
    )

    modifier = obj.modifiers.new(name=group_name, type='NODES')
        

    modifier.node_group = ng

    set_proxy_modifier_flags(
        modifier,
        enable_distance=enable_distance_check,
        enable_torus=enable_torus,
    )

    return modifier


def set_proxy_modifier_flags(modifier, *, enable_distance=None, enable_torus=None):
    if modifier is None or modifier.type != 'NODES':
        return
    ng = modifier.node_group
    if ng is None:
        return

    def _assign(identifier, fallback_keys, value):
        if value is None:
            return
        if identifier:
            try:
                modifier[identifier] = value
                return
            except Exception:
                pass
        for key in fallback_keys:
            try:
                modifier[key] = value
                return
            except Exception:
                continue

    distance_id = ng.get("distance_socket_identifier") or "EnableDistanceCheck"
    torus_id = ng.get("torus_socket_identifier") or "EnableTorusMesh"

    _assign(distance_id, ("Input_2", "Socket_2"), enable_distance)
    _assign(torus_id, ("Input_3", "Socket_3"), enable_torus)
