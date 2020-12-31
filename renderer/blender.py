import numpy as np
import subprocess
import json
import config
from pathlib import Path
import cv2


def from_linear(linear):
    srgb = linear.copy()
    less = linear <= 0.0031308
    srgb[less] = linear[less] * 12.92
    srgb[~less] = 1.055 * np.power(linear[~less], 1.0 / 2.4) - 0.055
    return srgb


def render(description_data, camera_data, result_data, height, width, cache_dir, use_gpu, sample_count, denoising, output_type=None, debug=False):
    '''A python wrapper to call the blender binary to run `blender_script.py` and load the rendered exr images back to numpy arrays.'''
    assert use_gpu in ['yes', 'no', 'auto']
    assert output_type in ['all', 'color', 'groundtruth', None]
    use_color = not (output_type == 'groundtruth')
    use_groundtruth = not (output_type == 'color')

    cache_dir.mkdir(exist_ok=True, parents=True)

    # valid_states = valid_state_from_result(result_data)
    # result_data_clean = {key: result_data[key] for key in valid_states}

    (cache_dir / 'description.json').write_text(json.dumps(description_data))
    (cache_dir / 'result.json').write_text(json.dumps(result_data))
    (cache_dir / 'camera.json').write_text(json.dumps(camera_data))

    render_args = ['blender', '--background',
                   '--python', str((Path(__file__).parent / 'blender_script.py').resolve()),
                   '--',
                   '--height', str(height),
                   '--width', str(width),
                   '--input-folder', str(cache_dir),
                   '--output-folder', str(cache_dir),
                   '--use-gpu', use_gpu,
                   '--sample-count', str(sample_count)
                   ]
    if denoising:
        render_args.append('--denoising')
    if output_type:
        render_args.append('--output-type')
        render_args.append(output_type)
    if debug:
        render_args.append('--debug')

    # if debug:
    print(' '.join(render_args))

    # blocking
    if not debug:
        subprocess.run(render_args, cwd=cache_dir, env=config.BLENDER_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    else:
        subprocess.run(render_args, cwd=cache_dir, env=config.BLENDER_ENV, check=True)

    ret = dict()
    for state_name in result_data:
        batch_color = []
        batch_normal = []
        batch_depth = []
        batch_obj1mask = []
        batch_obj2mask = []
        batch_wallmask = []
        batch_obj1bbox = []
        batch_obj2bbox = []

        for frame_num in range(len(camera_data)):
            color_exr_fname = cache_dir / \
                state_name / f'color{frame_num:04d}.exr'
            normal_exr_fname = cache_dir / \
                state_name / f'normal{frame_num:04d}.exr'
            depth_exr_fname = cache_dir / \
                state_name / f'depth{frame_num:04d}.exr'
            obj1mask_exr_fname = cache_dir / \
                state_name / f'obj1mask{frame_num:04d}.exr'
            obj2mask_exr_fname = cache_dir / \
                state_name / f'obj2mask{frame_num:04d}.exr'
            wallmask_exr_fname = cache_dir / \
                state_name / f'wallmask{frame_num:04d}.exr'
            obj1bbox_fname = cache_dir / state_name / \
                f'obj1bbox{frame_num:04d}.json'
            obj2bbox_fname = cache_dir / state_name / \
                f'obj2bbox{frame_num:04d}.json'
            camera_fname = cache_dir / f'camera{frame_num:04d}.json'

            # All these files should exist
            to_checks = []
            if use_color:
                to_checks.append(color_exr_fname)
            if use_groundtruth:
                to_checks += [normal_exr_fname, depth_exr_fname,
                              obj1mask_exr_fname, obj2mask_exr_fname, camera_fname]

            for to_check in to_checks:
                if not to_check.is_file():
                    raise RuntimeError(
                        f'file {to_check} does not exist, rendering error might have occured')

            if use_groundtruth:
                obj1mask = cv2.imread(str(obj1mask_exr_fname), cv2.IMREAD_UNCHANGED)
                obj1mask = np.logical_or.reduce(obj1mask, axis=2, keepdims=True)  # pylint: disable=no-member
                obj2mask = cv2.imread(str(obj2mask_exr_fname), cv2.IMREAD_UNCHANGED)
                obj2mask = np.logical_or.reduce(obj2mask, axis=2, keepdims=True)  # pylint: disable=no-member
                wallmask = cv2.imread(str(wallmask_exr_fname), cv2.IMREAD_UNCHANGED)
                wallmask = np.logical_or.reduce(wallmask, axis=2, keepdims=True)  # pylint: disable=no-member

                normal = cv2.imread(str(normal_exr_fname), cv2.IMREAD_UNCHANGED)
                depth = cv2.imread(str(depth_exr_fname), cv2.IMREAD_ANYDEPTH)[..., np.newaxis]

            if use_color:
                color = cv2.imread(str(color_exr_fname), cv2.IMREAD_UNCHANGED)
                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                color = from_linear(color)  # Gamma correction
                color = np.clip(color.astype(np.float32) * 255, 0, 255)
                batch_color.append(color)

            if use_groundtruth:
                # Compute the normals in camera space
                # Cycles outputs normals in world coordinates, with the order of (z, y, x)
                # Our output: x: screen, from left to right; y: forward looking direction
                # z: up direction
                # pylint: disable=pointless-string-statement
                '''
                    z  y
                    | /
                    |/
                    ------- x
                '''
                camera = json.loads(camera_fname.read_text())

                cam_up = np.array(camera['up'], dtype=np.float32)
                cam_up /= np.sqrt((cam_up * cam_up).sum())
                cam_towards = np.array(camera['towards'], dtype=np.float32)
                cam_towards /= np.sqrt((cam_towards * cam_towards).sum())
                cam_right = np.cross(cam_towards, cam_up)
                world_to_cam = np.stack((cam_right, cam_towards, cam_up))

                normal = np.nan_to_num(normal)
                normal = normal[:, :, ::-1]          # (z, y, x) to (x, y, z)
                # transform from world to cam coordinates
                normal = np.einsum('ij,rcj->rci', world_to_cam, normal)

                umask = ~(obj1mask | obj2mask | wallmask)

                depth[umask] = 0
                normal[np.tile(umask, (1, 1, 3))] = 0

                # Compute the real depth in camera space using z-depth (distance)
                assert depth.shape == (height, width, 1)
                aspect_ratio = height / width
                u, v = np.mgrid[:height, :width]
                x = v / (width - 1) - 0.5
                y = (0.5 - u / (height - 1)) * aspect_ratio
                focal_length = 0.5 / np.tan(camera['fov'] / 2)
                cos_theta = focal_length / np.sqrt((x ** 2 + y ** 2 + focal_length ** 2))
                depth *= cos_theta.reshape(depth.shape) # From distance to z

                # Read bounding boxes
                obj1bbox = np.array(json.loads(obj1bbox_fname.read_text()))
                obj2bbox = np.array(json.loads(obj2bbox_fname.read_text()))

                batch_normal.append(normal)
                batch_depth.append(depth)
                batch_obj1mask.append(obj1mask)
                batch_obj2mask.append(obj2mask)
                batch_wallmask.append(wallmask)
                batch_obj1bbox.append(obj1bbox)
                batch_obj2bbox.append(obj2bbox)

        ret[state_name] = dict()
        if use_color:
            ret[state_name]['color'] = np.stack(batch_color, axis=0)  # BHW3
        if use_groundtruth:
            ret[state_name].update(**{
                'normal': np.stack(batch_normal, axis=0),  # BHW3
                'depth': np.stack(batch_depth, axis=0),   # BHW1
                'obj1mask': np.stack(batch_obj1mask, axis=0),  # BHW1
                'obj2mask': np.stack(batch_obj2mask, axis=0),  # BHW1
                'wallmask': np.stack(batch_wallmask, axis=0),  # BHW1
                'obj1bbox': np.stack(batch_obj1bbox, axis=0),  # B22
                'obj2bbox': np.stack(batch_obj2bbox, axis=0),  # B22
                # B (object array of dictionaries)
                'camera': np.array(camera_data),
                'n_camera': len(camera_data)
            })

    return ret
