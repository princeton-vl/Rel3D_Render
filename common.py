'''Common functions that are used by multiple renderers.'''
from pathlib import Path

import config
import numpy as np
import trimesh
from constant import BLENDER_TO_UNITY

def extract_obj_paths(ddata):
    assert len(ddata['objectInfos']) == 2
    dataset1 = ddata['objectInfos'][0]['dataset']
    id1 = ddata['objectInfos'][0]['id']
    dataset2 = ddata['objectInfos'][1]['dataset']
    id2 = ddata['objectInfos'][1]['id']

    shape_root = Path(config.SHAPE_ROOT)
    obj1_path = shape_root / dataset1 / id1 / f'{id1}.obj'
    obj2_path = shape_root / dataset2 / id2 / f'{id2}.obj'

    return obj1_path, obj2_path

def extract_obj_transforms(state_info):
    transform1 = np.array(list(map(float, state_info['objectInfos'][0]['meshTransformMatrix'].split(',')))).reshape(4, 4)
    # convert the transformation matrix from unity coordinates to blender coordinates
    transform1 = BLENDER_TO_UNITY.T @ transform1 @ BLENDER_TO_UNITY

    transform2 = np.array(list(map(float, state_info['objectInfos'][1]['meshTransformMatrix'].split(',')))).reshape(4, 4)
    # convert the transformation matrix from unity coordinates to blender coordinates
    transform2 = BLENDER_TO_UNITY.T @ transform2 @ BLENDER_TO_UNITY

    return transform1, transform2

def extract_extrinsics_from_camera(cvalue):
    '''cvalue: dictionary
    '''
    # The inverse of camera extrinsic matrix:
    # R T
    # 0 1
    # where R = [cam_x', cam_y', cam_z']
    # The camera extrinsic matrix (world_to_cam):
    # R' -R'T
    # 0  1

    cam_origin = np.array(cvalue['origin'], dtype=np.float)
    cam_up = np.array(cvalue['up'], dtype=np.float)
    cam_target = np.array(cvalue['target'], dtype=np.float)
    cam_forward = cam_target - cam_origin

    cam_z = -cam_forward
    cam_x = np.cross(cam_up, cam_z)
    cam_y = np.cross(cam_z, cam_x)

    cam_x /= np.linalg.norm(cam_x)
    cam_y /= np.linalg.norm(cam_y)
    cam_z /= np.linalg.norm(cam_z)

    mat_rp = np.stack([cam_x, cam_y, cam_z], axis=0)
    t = -(mat_rp @ cam_origin)

    return mat_rp, t


def as_mesh(scene_or_mesh): # https://github.com/mikedh/trimesh/issues/507
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
    assert isinstance(mesh, trimesh.Trimesh)
    return mesh
