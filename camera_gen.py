'''About generating cameras for the synthetic scene.
'''


from constant import BLENDER_TO_UNITY, ORIGINAL_CAMERA_EXTRINSICS, INTEL_SENSOR_METADATA
from pathlib import Path
from scipy.spatial.transform import Rotation
from common import as_mesh

import numpy as np


DEFAULT_FOV = 2 * np.arctan2(INTEL_SENSOR_METADATA['color_intrinsics']['width'] / 2,
                             INTEL_SENSOR_METADATA['color_intrinsics']['fx'])
DEFAULT_NEAR_CLIP = 0.01
DEFAULT_FAR_CLIP = 100


def gen_initial_camera(ddata, rdata):
    return gen_initial_camera_3elevation_4view(ddata, rdata)


def refine_camera(camera, *args, **kwargs):
    moved_back_cameras = refine_camera_move_backward_until_bbox_fit(camera, *args, **kwargs)
    return refine_camera_move_forward_until_bbox_large_enough(moved_back_cameras, *args, **kwargs)


def gen_initial_camera_single(ddata, rdata):
    '''Generate the initial camera configuration.
    First, load the default camera used in Unity when collecting the synthetic contrastive data.
    Then, lower the camera but still look at the center.
    '''
    # Do not move the camera, only rotate
    extrinsics = BLENDER_TO_UNITY.T @ ORIGINAL_CAMERA_EXTRINSICS @ BLENDER_TO_UNITY
    origin = -extrinsics[:3, :3].T @ extrinsics[:-1, -1]
    origin[1] /= 2 # Cut the camera height by half
    up = [0, 1, 0]
    target = [0, 1.5, 0] # Look roughly at the center of the scene
    return dict(
        origin=origin.tolist(), up=up, target=target,
        fov=DEFAULT_FOV, near_clip=DEFAULT_NEAR_CLIP, far_clip=DEFAULT_FAR_CLIP)


def gen_initial_camera_3elevation(ddata, rdata):
    '''Generate three initial cameras with 3 different elevations.
    '''
    extrinsics = BLENDER_TO_UNITY.T @ ORIGINAL_CAMERA_EXTRINSICS @ BLENDER_TO_UNITY
    origin = -extrinsics[:3, :3].T @ extrinsics[:-1, -1]
    origins = [origin.copy(), origin.copy(), origin.copy()]

    origins[1][1] /= 2
    origins[2][1] /= 4
    up = [0, 1, 0]
    target = [0, 1.5, 0]

    return [dict(origin=origin.tolist(), up=up, target=target, fov=DEFAULT_FOV,
                near_clip=DEFAULT_NEAR_CLIP, far_clip=DEFAULT_FAR_CLIP,
                info={'elevation_id': i}) for i, origin in enumerate(origins)]

def gen_initial_camera_3elevation_4view(ddata, rdata):
    '''Similar to gen_initial_camera_3elevation, but also from 4 azimuth angles (0, 90, 180, 270) degrees
    also height * 2, height and height / 2
    '''
    extrinsics = BLENDER_TO_UNITY.T @ ORIGINAL_CAMERA_EXTRINSICS @ BLENDER_TO_UNITY
    origin = -extrinsics[:3, :3].T @ extrinsics[:-1, -1]


    # The target is the intersection of the viewing array and the ground y=-0.5
    # viewing ray equation: t = O + k t_0
    # Plain equation: w^T t = d
    # So w^T(O + k t_0) = d  ===>   k = (d - w^T O) / (w^T t_0)
    # Replace:  O = origin, t_0 = -extrinsics[2, :3]
    #           w = [0, 1, 0], d = -0.5

    k = (-0.5 - origin[1]) / (-extrinsics[2, 1])
    target = origin - k * extrinsics[2, :3]

    up = [0, 1, 0]

    ret = []

    # Bug fix
    # For view_independent relations, only consider the views from the front
    # if rdata.get('view_independent', 'Yes') == 'No':
    if ddata['initFinalState'].get('view_independent', 'Yes') == 'No':
        all_rots = [0,]
    else:
        all_rots = [0, 90, 180, 270]

    for rot_id, angle_deg in enumerate(all_rots):
        for elev_id, elev_factor in enumerate([0.25, 1, 2]):
            new_origin = origin.copy()
            new_origin[1] *= elev_factor

            rotation_xz = Rotation.from_euler('zxy', (0, 0, angle_deg), degrees=True).as_matrix()
            new_origin = rotation_xz @ new_origin
            new_target = rotation_xz @ target
            ret.append(dict(
                origin=new_origin.tolist(), up=up, target=new_target.tolist(), fov=DEFAULT_FOV,
                near_clip=DEFAULT_NEAR_CLIP, far_clip=DEFAULT_FAR_CLIP,
                info={'elevation_id': elev_id, 'rotation_id': rot_id}
            ))

    assert len(ret) in {12, 3}, len(ret) # 3 elevations * 4 rotations
    return ret


def refine_camera_move_backward_until_bbox_fit(cameras, ddata, rdata, f_get_info):
    '''Move the camera towards its z axis until the bounding boxes of all objects are inside the camera frame
    :param camera: initial camera
    :param f_get_info: a function that takes a list of cameras as input and output the rendered bounding boxes and foreground masks of both objects.
    '''
    TOTAL_SEARCH_STEP = 5
    MAX_STEP_SIZE = 5

    refined_cameras = []

    for camera in cameras:
        # first, get the camera lookat ray
        target = np.array(camera['target'], dtype=np.float)
        origin = np.array(camera['origin'], dtype=np.float)
        up = np.array(camera['up'], dtype=np.float)
        lookat = target - origin
        lookat /= np.sqrt(np.sum(lookat ** 2))

        cameras_to_test = []

        for search_step in range(TOTAL_SEARCH_STEP+1):
            cameras_to_test.append(dict(
                origin=(origin - lookat * search_step / TOTAL_SEARCH_STEP * MAX_STEP_SIZE).tolist(), # Move the camera along its z axis
                target=target.tolist(),
                up=up.tolist(),
                fov=camera['fov'],
                near_clip=camera['near_clip'],
                far_clip=camera['far_clip'],
                info=camera['info']
            ))

        (height, width), data = f_get_info(cameras_to_test)

        for batch_idx, camera_to_test in enumerate(cameras_to_test):
            camera_ok = True
            for _state_name, state_data in data.items():
                obj1bbox = state_data['obj1bbox'][batch_idx] # 2x2
                obj2bbox = state_data['obj2bbox'][batch_idx] # 2x2
                bbox1valid = state_data['obj1bbox_mask'][batch_idx]
                bbox2valid = state_data['obj2bbox_mask'][batch_idx]
                both_bboxes = np.stack([obj1bbox, obj2bbox], axis=0) # b x 2 x 2
                most_top_left = both_bboxes[:, 0, :].min(axis=0) # 2
                most_bottom_right = both_bboxes[:, 1, :].max(axis=0) # 2
                if most_top_left[0] < 0 or most_top_left[1] < 0 or most_bottom_right[0] > height-1 or most_bottom_right[1] > width-1 \
                    or not (bbox1valid and bbox2valid):
                    # some part of the object is out of the viewing frame in some state, or the whole object is out of frame
                    camera_ok = False
                    break

            if camera_ok:
                break

        if not camera_ok:
            print('Warning: not able to find a camera that fits all bounding boxes')

        camera_to_test['info']['search_step'] = batch_idx
        refined_cameras.append(camera_to_test)

    assert len(refined_cameras) == len(cameras)

    return refined_cameras


def refine_camera_look_at_center_of_both_objects(camera, ddata, rdata):
    '''Change the camera target so that it looks at the center of both objects in "initialState".
    '''
    import trimesh
    import config

    # Load shape objects transformations
    shape_root = Path(config.SHAPE_ROOT)
    dataset1 = ddata['objectInfos'][0]['dataset']
    id1 = ddata['objectInfos'][0]['id']
    dataset2 = ddata['objectInfos'][1]['dataset']
    id2 = ddata['objectInfos'][1]['id']

    obj1_path = shape_root / dataset1 / id1 / f'{id1}.obj'
    obj2_path = shape_root / dataset2 / id2 / f'{id2}.obj'

    obj1 = as_mesh(trimesh.load_mesh(str(obj1_path)))
    obj2 = as_mesh(trimesh.load_mesh(str(obj2_path)))

    # Compute the center of the two objects
    obj1_center = obj1.centroid
    obj2_center = obj2.centroid

    # convert the transformation matrix
    # for state in ('initialState', 'finalState'):
    state = next(iter(rdata.keys()))
    state_infos = rdata[state]['objectInfos']

    transform1 = np.array(
        list(map(float, state_infos[0]['meshTransformMatrix'].split(',')))).reshape(4, 4)
    transform1 = BLENDER_TO_UNITY.T @ transform1 @ BLENDER_TO_UNITY

    transform2 = np.array(
        list(map(float, state_infos[1]['meshTransformMatrix'].split(',')))).reshape(4, 4)
    transform2 = BLENDER_TO_UNITY.T @ transform2 @ BLENDER_TO_UNITY

    # transform the center
    obj1_center = transform1 @ np.concatenate([obj1_center, [1]])
    obj2_center = transform2 @ np.concatenate([obj2_center, [1]])
    obj1_center = obj1_center[:3] / obj1_center[3]
    obj2_center = obj2_center[:3] / obj2_center[3]

    # Look at the center of the two objects
    target = (obj1_center + obj2_center) / 2

    return dict(origin=camera['origin'], target=target.tolist(), up=camera['up'],
                fov=camera['fov'], near_clip=camera['near_clip'], far_clip=camera['far_clip'])


def refine_camera_move_forward_until_bbox_large_enough(cameras, ddata, rdata, f_get_info):
    '''Move the camera towards its z axis until the bounding boxes of all objects are inside the camera frame
    :param camera: initial camera
    :param f_get_info: a function that takes a list of cameras as input and output the rendered bounding boxes and foreground masks of both objects.
    '''
    TOTAL_SEARCH_STEP = 20 # Linearly separate 20 segments from camera origin to camera target
    MAX_STEP_SIZE = 1 # from camera origin to camera target, camera target is on the ground
    RATIO = 0.0025      # If the smalled bounding box for all objects in all states reach 0.01 screen area, then stop moving back
    MAX_ONSCREEN_RATIO = 0.9 # atmost 50% of the object can be cropped for all objects in all states

    refined_cameras=[]

    for camera in cameras:
        # first, get the camera lookat ray
        target = np.array(camera['target'], dtype=np.float)
        origin = np.array(camera['origin'], dtype=np.float)
        up = np.array(camera['up'], dtype=np.float)
        lookat = target - origin
        # lookat /= np.sqrt(np.sum(lookat ** 2))

        cameras_to_test = []

        for search_step in range(TOTAL_SEARCH_STEP):
            cameras_to_test.append(dict(
                origin=(origin + lookat * search_step / TOTAL_SEARCH_STEP * \
                        MAX_STEP_SIZE).tolist(),  # Move the camera along its -z axis
                target=(target + lookat * search_step / TOTAL_SEARCH_STEP * \
                        MAX_STEP_SIZE).tolist(),  # Also move the target, otherwise the camera origin may move over the target
                up=up.tolist(),
                fov=camera['fov'],
                near_clip=camera['near_clip'],
                far_clip=camera['far_clip'],
                info=camera.get('info', {})
            ))

        (height, width), data= f_get_info(cameras_to_test)

        last_min_area = 0
        last_camera = None
        for batch_idx, camera_to_test in enumerate(cameras_to_test):
            # Compute the min area of all the bounding boxes
            min_area = height * width # Max possible = height * width
            onscreen_ratio = 1 # Max possible = 1
            for _state_name, state_data in data.items():
                obj1bbox = state_data['obj1bbox'][batch_idx]  # 2x2, ((top, left), (bottom, right))
                obj2bbox = state_data['obj2bbox'][batch_idx]  # 2x2, ((top, left), (bottom, right))
                bbox1valid = state_data['obj1bbox_mask'][batch_idx]
                bbox2valid = state_data['obj2bbox_mask'][batch_idx]

                # Compute the original area of the bounding box without screenspace cropping
                if bbox1valid:
                    original_obj1_area = (obj1bbox[1, :] - obj1bbox[0, :]).prod()
                else:
                    original_obj1_area = 1
                if bbox2valid:
                    original_obj2_area = (obj2bbox[1, :] - obj2bbox[0, :]).prod()
                else:
                    original_obj2_area = 1

                # Screen-space cropping
                obj1bbox[:, 0] = np.maximum(obj1bbox[:, 0], 0)
                obj1bbox[:, 0] = np.minimum(obj1bbox[:, 0], height)
                obj1bbox[:, 1] = np.maximum(obj1bbox[:, 1], 0)
                obj1bbox[:, 1] = np.minimum(obj1bbox[:, 1], width)

                obj2bbox[:, 0] = np.maximum(obj2bbox[:, 0], 0)
                obj2bbox[:, 0] = np.minimum(obj2bbox[:, 0], height)
                obj2bbox[:, 1] = np.maximum(obj2bbox[:, 1], 0)
                obj2bbox[:, 1] = np.minimum(obj2bbox[:, 1], width)


                if bbox1valid:
                    obj1_area = (obj1bbox[1, :] - obj1bbox[0, :]).prod()
                else:
                    obj1_area = 0

                if bbox2valid:
                    obj2_area = (obj2bbox[1, :] - obj2bbox[0, :]).prod()
                else:
                    obj2_area = 0

                assert obj1_area >= 0 and original_obj1_area >= 0
                assert obj2_area >= 0 and original_obj2_area >= 0

                min_area = min(min_area, min(obj1_area, obj2_area))

                # Compute the minimum onscreen_ratio for all objects and all states
                ratio1 = obj1_area / original_obj1_area; ratio2 = obj2_area / original_obj2_area
                onscreen_ratio = min(onscreen_ratio, min(ratio1, ratio2))

            camera_to_test['info']['search_step'] = batch_idx
            camera_to_test['info']['obj1_bbox_area'] = obj1_area
            camera_to_test['info']['obj2_bbox_area'] = obj2_area

            found = True
            if onscreen_ratio < MAX_ONSCREEN_RATIO: # The objects are cropped too much
                print(f'Onscreen ratio {onscreen_ratio} is below the threshold {MAX_ONSCREEN_RATIO}')
                if last_camera != None:
                    refined_cameras.append(last_camera)
                else:
                    refined_cameras.append(camera_to_test)
                break

            if min_area > RATIO * height * width: # Reach the criterion, this camera is good
                print(f'Reach min_area > RATIO * height * width at step {batch_idx}/{TOTAL_SEARCH_STEP+1}')
                refined_cameras.append(camera_to_test)
                break

            if min_area < last_min_area: # Starts getting smaller, return the last one (biggest area)
                print(f'Starting decreasing at step {batch_idx}/{TOTAL_SEARCH_STEP+1}, from {last_min_area} to {min_area}')
                refined_cameras.append(last_camera)
                break
            found = False

            last_camera = camera_to_test
            last_min_area = min_area

        if not found:
            print('min_area keeps increasing. Choose the closest one')
            refined_cameras.append(last_camera) # Area keeps increasing, so append the last one (best one)

    assert len(refined_cameras) == len(cameras)

    return refined_cameras
