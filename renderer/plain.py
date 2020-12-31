'''The plain implementation of bounding box render.
'''
import numpy as np
import trimesh
import config
from pathlib import Path
from constant import BLENDER_TO_UNITY
from common import extract_obj_transforms, extract_obj_paths, extract_extrinsics_from_camera


def render_bbox(description_data, camera_data, result_data, height, width):
    '''Load the object meshes, spatial transformations from the data, and then project all the vertices
    on to the camera plane, then compute the bounding boxes in batch.
    '''

    # aliases
    ddata = description_data
    cdata = camera_data
    rdata = result_data

    obj1_path, obj2_path = extract_obj_paths(ddata)

    # Load the vertices
    scene1 = trimesh.load_mesh(str(obj1_path))
    scene2 = trimesh.load_mesh(str(obj2_path))
    if isinstance(scene1, trimesh.Scene):
        for mesh in scene1.geometry.values():
            mesh.remove_unreferenced_vertices()
        vertices1 = np.concatenate([mesh.vertices for mesh in scene1.geometry.values()], axis=0) # N_v1 x 3
    else:
        vertices1 = scene1.vertices
    
    if isinstance(scene2, trimesh.Scene):
        for mesh in scene2.geometry.values():
            mesh.remove_unreferenced_vertices()
        vertices2 = np.concatenate([mesh.vertices for mesh in scene2.geometry.values()], axis=0) # N_v2 x 3
    else:
        vertices2 = scene2.vertices

    # Add the forth dimension
    nv1 = vertices1.shape[0]
    nv2 = vertices2.shape[0]
    vertices1 = np.concatenate([vertices1, np.ones((nv1, 1), dtype=np.float)], axis=1) # N_v1 x 4
    vertices2 = np.concatenate([vertices2, np.ones((nv2, 1), dtype=np.float)], axis=1) # N_v1 x 4

    # Load the transformation matrices
    state_names = list(rdata.keys())
    n_state = len(state_names)

    obj1_trfms = np.zeros((n_state, 4, 4), dtype=np.float)
    obj2_trfms = np.zeros((n_state, 4, 4), dtype=np.float)

    for sidx, state_name in enumerate(state_names):
        transform1, transform2 = extract_obj_transforms(rdata[state_name])

        obj1_trfms[sidx, :, :] = transform1[:, :]
        obj2_trfms[sidx, :, :] = transform2[:, :]

    # Compute the camera extrinsic matrices
    n_camera = len(cdata)
    extrinsics = np.zeros((n_camera, 4, 4), dtype=np.float) # N_cam x 4 x 4
    for cidx, cvalue in enumerate(cdata):
        mat_rp, vec_t = extract_extrinsics_from_camera(cvalue)
        extrinsics[cidx, :3, :3] = mat_rp
        extrinsics[cidx, :3, 3] = vec_t
    extrinsics[:, 3, 3] = 1

    # world_to_cam @ transform to convert the vertices to camera space
    # c: camera_idx, s: state_idx, v: vertex_idx
    vert_cam1 = np.einsum('cij,sjk,vk->csvi', extrinsics, obj1_trfms, vertices1) # N_cam x N_state x N_v1 x 4
    vert_cam2 = np.einsum('cij,sjk,vk->csvi', extrinsics, obj2_trfms, vertices2) # N_cam x N_state x N_v2 x 4

    def debug_print(vertices):
        print(vertices1[..., 0].min(), vertices1[..., 0].max(), vertices1[..., 1].min(), vertices1[..., 1].max(), vertices1[..., 2].min(), vertices1[..., 2].max())


    assert np.allclose(vert_cam1[..., 3], 1)
    assert np.allclose(vert_cam2[..., 3], 1)

    # Invert the z-axis because the camera is pointing towards -z axis in its local coordinate space
    vert_cam1[..., 2] = -vert_cam1[..., 2]
    vert_cam2[..., 2] = -vert_cam2[..., 2]

    # Compute the mask of vertices that are in the frustum
    mask1 = np.zeros((n_camera, n_state, nv1), dtype=np.bool)
    mask2 = np.zeros((n_camera, n_state, nv2), dtype=np.bool)

    for cidx in range(n_camera):
        mask1[cidx, ...] = np.logical_and(vert_cam1[cidx, ..., 2] >= cdata[cidx]['near_clip'], 
                                  vert_cam1[cidx, ..., 2] <= cdata[cidx]['far_clip'])
        mask2[cidx, ...] = np.logical_and(vert_cam2[cidx, ..., 2] >= cdata[cidx]['near_clip'],
                                  vert_cam2[cidx, ..., 2] <= cdata[cidx]['far_clip'])

    # If mask[camera_id, state_id] = False, the object is outside the frustum and there is no bounding box
    bbox1_valid_mask = np.count_nonzero(mask1, axis=-1) > 0
    bbox2_valid_mask = np.count_nonzero(mask2, axis=-1) > 0

    aspect_ratio = height / width
    min_x, max_x = -0.5, 0.5
    min_y, max_y = aspect_ratio * min_x, aspect_ratio * max_x
    focal_length = 0.5 / np.tan([cdata_single['fov'] / 2 for cdata_single in cdata]) # n_camera


    # Project onto the camera screen [-0.5, 0.5]
    x_vc1 = vert_cam1[..., 0] / vert_cam1[..., 2] * focal_length.reshape((-1, 1, 1)) # N_cam x N_state x N_v1
    y_vc1 = vert_cam1[..., 1] / vert_cam1[..., 2] * focal_length.reshape((-1, 1, 1))
    x_vc2 = vert_cam2[..., 0] / vert_cam2[..., 2] * focal_length.reshape((-1, 1, 1))
    y_vc2 = vert_cam2[..., 1] / vert_cam2[..., 2] * focal_length.reshape((-1, 1, 1))

    # Rescale to [0, 1] (< 0 or > 1 is outside the screen)
    x_vc1 = (x_vc1 - min_x) / (max_x - min_x) # N_cam x N_state x N_v1
    # mask1 = np.logical_and(mask1, np.logical_and(x_vc1 >= 0, x_vc1 <= 1))

    y_vc1 = (y_vc1 - min_y) / (max_y - min_y) # N_cam x N_state x N_v1
    # mask1 = np.logical_and(mask1, np.logical_and(y_vc1 >= 0, y_vc1 <= 1))

    x_vc2 = (x_vc2 - min_x) / (max_x - min_x) # N_cam x N_state x N_v2
    # mask2 = np.logical_and(mask2, np.logical_and(x_vc2 >= 0, x_vc2 <= 1))

    y_vc2 = (y_vc2 - min_y) / (max_y - min_y) # N_cam x N_state x N_v2
    # mask2 = np.logical_and(mask2, np.logical_and(y_vc2 >= 0, y_vc2 <= 1))

    # Masking out the vertices outside the frustum
    x_vc1 = np.ma.masked_array(x_vc1, mask=np.logical_not(mask1))
    y_vc1 = np.ma.masked_array(y_vc1, mask=np.logical_not(mask1))
    x_vc2 = np.ma.masked_array(x_vc2, mask=np.logical_not(mask2))
    y_vc2 = np.ma.masked_array(y_vc2, mask=np.logical_not(mask2))

    # tl for top left, br for bottom right. Note that v and y are reverted
    tl_u1 = height * (1 - y_vc1.max(axis=-1))     # N_cam x N_state
    tl_v1 = width * x_vc1.min(axis=-1)            # N_cam x N_state
    br_u1 = height * (1 - y_vc1.min(axis=-1)) # N_cam x N_state
    br_v1 = width * x_vc1.max(axis=-1)        # N_cam x N_state

    tl_u2 = height * (1 - y_vc2.max(axis=-1))     # N_cam x N_state
    tl_v2 = width * x_vc2.min(axis=-1)            # N_cam x N_state
    br_u2 = height * (1 - y_vc2.min(axis=-1))  # N_cam x N_state
    br_v2 = width * x_vc2.max(axis=-1)        # N_cam x N_state

    # assert np.all(tl_u1 <= br_u1)
    # assert np.all(tl_v1 <= br_v1)
    # assert np.all(tl_u2 <= br_u2)
    # assert np.all(tl_v2 <= br_v2)

    bbox1 = np.stack([
        np.stack([tl_u1.data, tl_v1.data], axis=2), # N_cam x N_state x 2
        np.stack([br_u1.data, br_v1.data], axis=2) # N_cam x N_state x 2
    ], axis=2) # N_cam x N_state x 2 x 2

    bbox2 = np.stack([
        np.stack([tl_u2.data, tl_v2.data], axis=2),  # N_cam x N_state x 2
        np.stack([br_u2.data, br_v2.data], axis=2)  # N_cam x N_state x 2
    ], axis=2)  # N_cam x N_state x 2 x 2
    # Extract state -> N_cam x 2 x 2

    # Expand along the state dimension and turn it into dictionaries
    return {
        state_name: {
            'obj1bbox': bbox1[:, sidx],
            'obj2bbox': bbox2[:, sidx],
            'obj1bbox_mask': bbox1_valid_mask[:, sidx],
            'obj2bbox_mask': bbox2_valid_mask[:, sidx]
        } for sidx, state_name in enumerate(state_names)
    }
