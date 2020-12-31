'''The blender script for rendering the synthetic scenes into exr images. Require Blender >= 2.8
Usage:
    /path/to/blender --background --python blender_script.py -- --input-folder ... --output-folder ...
input_folder:
    description.json: description of two objects
    result.json: the resulting final state of the two objects
    camera.json: the camera views to render
output_type:
    all: all color and groundtruth images
    color: only render the color image.
    groundtruth: only render the groundtruth images (depth, normal etc).
output_folder:
    colorXXXX.exr: rendered color image
    normalXXXX.exr: rendered normal map (in world space)
    depthXXXX.exr: rendered depth image (z-depth, not real depth)
    obj1maskXXXX.exr: the visible mask of the first object
    obj2maskXXXX.exr: the visible mask of the second object
    obj1bboxXXXX.json: the coordinates of the 2D bounding box of the first object
    obj2bboxXXXX.json: the coordinates of the 2D bounding box of the second object
    cameraXXXX.exr: camera information to convert the normals to camera space
'''

import argparse
import sys
from pathlib import Path
import json

import bpy
from mathutils import Vector, Matrix
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from constant import BLENDER_TO_UNITY
import config

OBJECT_INDICES = [5, 6] # Any two different numbers other than zero will do
WALL_INDEX = 7


def enable_front_face_culling(wall_obj):
    # Enable front-face culling for the cube so that one can see through the wall
    # Reference: https://www.katsbits.com/codex/backface-culling/
    for m_slot in wall_obj.material_slots:
        mat = m_slot.material
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        pcp_node = nodes['Principled BSDF']
        output_node = nodes['Material Output']

        geo_node = nodes.new('ShaderNodeNewGeometry')
        tr_node = nodes.new('ShaderNodeBsdfTransparent')
        mix_node = nodes.new('ShaderNodeMixShader')

        links.new(geo_node.outputs['Backfacing'], mix_node.inputs['Fac'])
        links.new(pcp_node.outputs['BSDF'], mix_node.inputs[2])
        links.new(tr_node.outputs['BSDF'], mix_node.inputs[1])
        links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])


def fix_material(obj):
    '''Fix 0-dissolve surfaces and add back-face culling.
    Shapes in shapenets have overlapping faces with opposite normals.
    In most renderers this can be solved by backface culling.
    However, the blender cycles renderer still has z-depth fighting artifacts, even with backface culling on.
    Try to solve by moving all the vertices towards its vertex normal by a tiny step so that they are no longer overlapping.
    '''
    EPS = 1e-5

    # Fix overlapping faces
    mesh = obj.data
    for vidx in range(len(mesh.vertices)):
        mesh.vertices[vidx].co += mesh.vertices[vidx].normal * EPS

    for m_slot in obj.material_slots:
        mat = m_slot.material
        nodes = mat.node_tree.nodes
        # links = mat.node_tree.links
        pcp_node = nodes['Principled BSDF']
        # output_node = nodes['Material Output']

        # Fix the transparency value for ShapeNet (if it is 0, then set it to one, just like Unity does)
        if pcp_node.inputs['Alpha'].default_value == 0:
            pcp_node.inputs['Alpha'].default_value = 1


def disable_transparency(obj):
    for m_slot in obj.material_slots:
        mat = m_slot.material
        nodes = mat.node_tree.nodes
        # links = mat.node_tree.links
        pcp_node = nodes['Principled BSDF']
        pcp_node.inputs['Alpha'].default_value = 1 # Need to completely disable transparency when rendering obj masks, depths, normals etc.


# ===== bounding box computation ====
# https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file
# ===================================

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def debug_vertices(me):
    print(len(me.vertices))
    min_x = min(me.vertices[i].co[0] for i in range(len(me.vertices)))
    min_y = min(me.vertices[i].co[1] for i in range(len(me.vertices)))
    min_z = min(me.vertices[i].co[2] for i in range(len(me.vertices)))
    max_x = max(me.vertices[i].co[0] for i in range(len(me.vertices)))
    max_y = max(me.vertices[i].co[1] for i in range(len(me.vertices)))
    max_z = max(me.vertices[i].co[2] for i in range(len(me.vertices)))
    
    print(f'[{min_x}, {min_y}, {min_z}] - [{max_x}, {max_y}, {max_z}]')


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    depsgraph = bpy.context.evaluated_depsgraph_get()
    mat = cam_ob.matrix_world.normalized().inverted()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()

    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        # if min_x <= co_local.x <= max_x and min_y <= co_local.y <= max_y:
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    # min_x = clamp(min(lx), 0.0, 1.0)
    # max_x = clamp(max(lx), 0.0, 1.0)
    # min_y = clamp(min(ly), 0.0, 1.0)
    # max_y = clamp(max(ly), 0.0, 1.0)
    min_x = min(lx)
    max_x = max(lx)
    min_y = min(ly)
    max_y = max(ly)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    # if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        # return (0, 0, 0, 0)

    # Top left (u, v), bottom right (u, v)
    return (dim_y - max_y * dim_y, min_x * dim_x), \
            (dim_y - min_y * dim_y, max_x * dim_x) 

def compute_bbox(scene, obj_parts, camera):
    bbox = None
    for part in obj_parts:
        bottom_left, upper_right = camera_view_bounds_2d(scene, camera, part)
        if bbox is None:
            bbox = (bottom_left, upper_right)
        else:
            bl, ur = bbox
            bbox = ((min(bl[0], bottom_left[0]), min(bl[1], bottom_left[1])), # top left
                    (max(ur[0], upper_right[0]), max(ur[1], upper_right[1]))) # bottom right

    assert bbox is not None
    return bbox


def render(input_folder, height, width, use_gpu, output_type, output_folder, sample_count, denoising, debug):
    output_folder.mkdir(parents=True, exist_ok=True)
    assert input_folder.is_dir(), f'{input_folder} is not a directory'
    camera_json = input_folder / 'camera.json'
    desc_json = input_folder / 'description.json'
    result_json = input_folder / 'result.json'

    cdata = json.loads(camera_json.read_text())
    ddata = json.loads(desc_json.read_text())
    rdata = json.loads(result_json.read_text())

    use_color = output_type in ('all', 'color')
    use_groundtruth = output_type in ('all', 'groundtruth')

    # =========== Set up CYCLES ==============
    bpy.ops.wm.read_homefile(use_empty=True) # Requires Blender 2.79+
    context = bpy.context
    scene = context.scene
    view_layer = context.view_layer

    scene.render.engine = 'CYCLES'
    # CYCLES rendering device
    if use_gpu == 'auto':
        devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()
        use_gpu = 'yes' if devices else 'no'

    if use_gpu == 'yes':
        scene.cycles.device = 'GPU'
        scene.render.tile_x = 512
        scene.render.tile_y = 512
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    elif use_gpu == 'no':
        pass
    else:
        raise ValueError('use_gpu can only be yes, no or auto!')

    if use_gpu == 'yes':
        print('Using GPU to render')

    # ============= Set up screen =============

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    # ============ Set up two objects ===============
    # Information from description.json
    assert len(ddata['objectInfos']) == 2
    dataset1 = ddata['objectInfos'][0]['dataset']
    id1 = ddata['objectInfos'][0]['id']
    dataset2 = ddata['objectInfos'][1]['dataset']
    id2 = ddata['objectInfos'][1]['id']

    # Load the background
    bpy.ops.import_scene.obj(filepath=str(config.BACKGROUND_WALL), axis_forward='Y', axis_up='-Z')
    for obj in bpy.context.selected_objects:
        enable_front_face_culling(obj)
        obj.pass_index = WALL_INDEX

    # Load the meshes
    shape_root = Path(config.SHAPE_ROOT)
    obj1_path = shape_root / dataset1 / id1 / f'{id1}.obj'
    obj2_path = shape_root / dataset2 / id2 / f'{id2}.obj'

    obj_parts = [list(), list()]

    for i, obj_path in enumerate([obj1_path, obj2_path]):
        bpy.ops.import_scene.obj(filepath=str(obj_path))

        for obj in bpy.context.selected_objects:
            fix_material(obj)
            obj.pass_index = OBJECT_INDICES[i]
            obj_parts[i].append(obj)

            # view_layer.objects.active = obj
            # bpy.ops.object.shade_flat()
            # bpy.ops.object.editmode_toggle()
            # bpy.ops.mesh.select_all(action='SELECT')
            # bpy.ops.mesh.remove_doubles()
            # bpy.ops.mesh.split_normals()
            # bpy.ops.mesh.normals_make_consistent(inside=False)
            # bpy.ops.object.editmode_toggle()

    if use_color:
        # # ============= Set up directional lights =====
        # # Four directions
        # direction = -Vector([0, 1, -5])
        # intensity = Vector([1, 1, 1])

        # lamp_data = bpy.data.lights.new(name='Point', type='POINT')
        # lamp_data.falloff_type = 'CONSTANT'
        # lamp_data.color = intensity * 300

        # lamp_object = bpy.data.objects.new(name='Point', object_data=lamp_data)
        # lamp_object.location = - direction * 2

        # scene.collection.objects.link(lamp_object)

        # ============= Set up ambient lights ==
        world = bpy.data.worlds.new('World')
        scene.world = world
        world.use_nodes = True
        nodes = world.node_tree.nodes
        background = nodes['Background']
        background.inputs['Color'].default_value = Vector([0.1, 0.1, 0.1, 1])
        view_layer.update()
        # =============== Set up 4 directional lights ===
        coordinates = [(2.5, 4.5, 2.5), (2.5, 4.5, -2.5), (-2.5, 4.5, 2.5), (-2.5, 4.5, -2.5)]
        intensity = Vector([1, 1, 1]) * 4
        for lid, coord in enumerate(coordinates):
            lamp_data = bpy.data.lights.new(name='Point', type='POINT')
            lamp_data.color = intensity
            lamp_object = bpy.data.objects.new(name=f'Point{lid}', object_data=lamp_data)
            lamp_object.location = Vector(coord)
            scene.collection.objects.link(lamp_object)

    # ========= Set up cameras ====================
    scene.frame_start = 0
    scene.frame_end = len(cdata) - 1

    # ============ Render to files =============
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # clear default nodes
    for n in nodes:
        nodes.remove(n)


    # scene.render.filepath = combined_exr_fname
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.view_layers['View Layer'].cycles.use_denoising = denoising
    render_layers = nodes.new('CompositorNodeRLayers')
    file_output = nodes.new(type='CompositorNodeOutputFile')
    file_output.file_slots.remove(file_output.inputs['Image'])

    # ========= Color image rendering setup ===========
    if use_color:
        file_output.file_slots.new('color')
        links.new(render_layers.outputs['Image'], file_output.inputs['color'])
        scene.cycles.samples = sample_count
        render_image_or_info(cdata, rdata, output_folder, file_output, id1, id2, obj_parts, is_info=False, debug=debug)

        # Avoid overwriting
        file_output.file_slots.remove(file_output.inputs['color'])

    # ========== Groundtruth images rendering setup ======
    if use_groundtruth:
        scene.cycles.samples = 1
        scene.view_layers['View Layer'].cycles.use_denoising = False
        scene.view_layers['View Layer'].use_pass_normal = True # Blender >= 2.8
        scene.view_layers['View Layer'].use_pass_z = True
        scene.view_layers['View Layer'].use_pass_object_index = True

        file_output.file_slots.new('normal')
        file_output.file_slots.new('depth')
        file_output.file_slots.new('obj1mask')
        file_output.file_slots.new('obj2mask')
        file_output.file_slots.new('wallmask')

        # Normal
        links.new(render_layers.outputs['Normal'], file_output.inputs['normal'])
        # Depth
        links.new(render_layers.outputs['Depth'], file_output.inputs['depth'])
        # Obj mask
        id_mask1 = nodes.new('CompositorNodeIDMask')
        id_mask2 = nodes.new('CompositorNodeIDMask')
        wall_mask = nodes.new('CompositorNodeIDMask')
        id_mask1.index = OBJECT_INDICES[0]
        id_mask2.index = OBJECT_INDICES[1]
        wall_mask.index = WALL_INDEX
        links.new(render_layers.outputs['IndexOB'], id_mask1.inputs['ID value'])
        links.new(id_mask1.outputs['Alpha'], file_output.inputs['obj1mask'])
        links.new(render_layers.outputs['IndexOB'], id_mask2.inputs['ID value'])
        links.new(id_mask2.outputs['Alpha'], file_output.inputs['obj2mask'])
        links.new(render_layers.outputs['IndexOB'], wall_mask.inputs['ID value'])
        links.new(wall_mask.outputs['Alpha'], file_output.inputs['wallmask'])

        for obj_part in obj_parts:
            for obj in obj_part:
                disable_transparency(obj)

        # scene.cycles.samples = 1
        render_image_or_info(cdata, rdata, output_folder, file_output, id1, id2, obj_parts, is_info=True, debug=debug)


def render_image_or_info(cdata, rdata, output_folder, file_output, id1, id2, obj_parts, is_info=False, debug=False):
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    for frame_num, cdata_single in enumerate(cdata):
        scene.frame_set(frame_num)

        # Compute the camera matrix
        cam_origin = Vector(cdata_single['origin'])
        cam_up = Vector(cdata_single['up'])
        cam_target = Vector(cdata_single['target'])
        cam_forward = cam_target - cam_origin

        cam_z = -cam_forward.normalized()
        cam_x = -cam_z.cross(cam_up).normalized()
        cam_y = cam_z.cross(cam_x).normalized()
        euler = Matrix([cam_x, cam_y, cam_z]).transposed().to_euler()

        # Save the camera to the json file
        camera_fname = output_folder / f'camera{frame_num:04d}.json'
        camera_fname.write_text(json.dumps({
            'origin': list(cam_origin),
            'up': list(cam_y),
            'towards': list(-cam_z),
            'fov': cdata_single['fov']
        }))

        # Assign the matrix to the camera
        camera_data = bpy.data.cameras.new('Camera')
        camera_data.clip_start = cdata_single['near_clip']
        camera_data.clip_end = cdata_single['far_clip']
        camera_data.angle_x = cdata_single['fov']

        camera = bpy.data.objects.new('Camera', camera_data)
        camera.location = Vector(cam_origin)
        camera.rotation_euler = euler

        scene.collection.objects.link(camera)  # Blender >= 2.8
        scene.camera = camera
        view_layer.update()

        for state_name in rdata.keys():
            sub_output_folder = output_folder / state_name
            sub_output_folder.mkdir(parents=True, exist_ok=True)
            file_output.base_path = str(sub_output_folder)
            state_infos = rdata[state_name]['objectInfos']

            for obj_id, obj_info, obj_part in zip([id1, id2], state_infos, obj_parts):
                assert obj_info['id'] == obj_id

                # Load the transformation matrix
                transform = np.array(list(map(float, obj_info['meshTransformMatrix'].split(',')))).reshape(4, 4)

                # convert the transformation matrix from unity coordinates to blender coordinates
                transform = BLENDER_TO_UNITY.T @ transform @ BLENDER_TO_UNITY

                for obj in obj_part:
                    obj.matrix_world = transform.T

            # Save 2D bounding boxes if rendering info
            if is_info:
                bbox1 = compute_bbox(scene, obj_parts[0], camera)
                bbox2 = compute_bbox(scene, obj_parts[1], camera)
                (sub_output_folder /
                f'obj1bbox{frame_num:04d}.json').write_text(json.dumps(bbox1))
                (sub_output_folder /
                f'obj2bbox{frame_num:04d}.json').write_text(json.dumps(bbox2))

            if debug and frame_num == len(cdata) - 1:
                bpy.ops.file.pack_all() # Pack external textures for local debugging
                if is_info:
                    bpy.ops.wm.save_as_mainfile(filepath=str(sub_output_folder/ f'debug-info.blend'))
                else:
                    bpy.ops.wm.save_as_mainfile(filepath=str(sub_output_folder/ f'debug-color.blend'))

            # Render the scene
            bpy.ops.render.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--input-folder', type=Path, required=True)
    parser.add_argument('--use-gpu', default='auto', choices=['no', 'yes', 'auto'])
    parser.add_argument('--output-type', default='all', choices=['all', 'color', 'groundtruth'])
    parser.add_argument('--output-folder', default=Path('./tmp').resolve(), type=Path)
    parser.add_argument('--sample-count', type=int, default=256)
    parser.add_argument('--denoising', action='store_true', default=False)
    # parser.add_argument('--bbox-only', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])

    render(**vars(args))

