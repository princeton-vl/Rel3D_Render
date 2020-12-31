# pylint: disable=unsubscriptable-object

import os
import json
import cv2
import numpy as np
import config
import argparse
import uuid
import pdb
import shutil
from pathlib import Path
from imgproc import depth_colormap, draw_bbox_with_label, draw_front_up_axes
from collections import defaultdict, OrderedDict
from PIL import Image
from renderer import blender, plain
import csv_parse, transforms
import pickle as pkl


def visualize(color, camera, normal, depth, obj1mask, obj2mask, wallmask, obj1bbox, obj2bbox, label1, label2, ddata_objectinfo, rdata_objectinfo, **_kwargs):
    '''Generate an image of b x h x 4w x 3 that visualizes color, normal, depth, bboxes and object masks with annotated labels.'''
    COLOR1 = (64, 255, 64)
    COLOR2 = (64, 64, 255)
    batch_size, height, width = color.shape[:3]

    visualization = np.zeros((batch_size, height, 4 * width, 3), dtype=np.uint8)

    for idx in range(batch_size):
        color_idx = color[idx]
        color_idx = draw_bbox_with_label(color_idx, obj1bbox[idx], label1, COLOR1)
        color_idx = draw_bbox_with_label(color_idx, obj2bbox[idx], label2, COLOR2)
        visualization[idx, :height, :width] = color_idx

        nx, nz, ny = normal[idx, ..., 0], -normal[idx, ..., 1], -normal[idx, ..., 2]
        nxyz = np.stack([nx, ny, nz], axis=2).astype(np.float32)
        transform_vector = transforms.get_transform_vector(camera[idx], ddata_objectinfo, rdata_objectinfo)
        visualization[idx, :, width:2*width] = draw_front_up_axes((nxyz + 1) / 2 * 255, camera[idx], transform_vector)
        rows, cols = np.nonzero(obj1mask[idx][..., 0])
        visualization[idx, rows, 2*width+cols , :] = COLOR1
        rows, cols = np.nonzero(obj2mask[idx][..., 0])
        visualization[idx, rows, 2*width+cols , :] = COLOR2

        mask = (obj1mask[idx] | obj2mask[idx] | wallmask[idx])[..., 0]
        visualization[idx, :, 3*width:] = (depth_colormap(depth[idx][..., 0], mask=mask))

    return visualization


def render_scene(description_data, result_data, height, width, cache_dir, use_gpu, sample_count, denoising, output_type=None, debug=False,
                 cidx=None, states=None):
    '''Given a single row of data (description, result), generate and refine a list of cameras, and the use the cameras
    to call the blender.render to get the images.'''
    from camera_gen import gen_initial_camera, refine_camera
    init_cameras = gen_initial_camera(description_data, result_data)
    if not cidx is None:
        init_cameras = [init_cameras[cidx]]

    valid_states = csv_parse.valid_state_from_result(result_data)
    if states is None:
        result_data_clean = {key: result_data[key] for key in valid_states}
    else:
        # states should be a subset of valid_states
        for state in states:
            assert state in valid_states
        result_data_clean = {key: result_data[key] for key in states}

    def f_get_info(cameras):
        '''The function called inside `refine_camera` to obtain the information (such as mask, bounding boxes) of the both objects
        '''
        # data = blender.render(description_data, cameras, result_data_clean, height, width, cache_dir, use_gpu, sample_count, denoising, output_type='groundtruth', debug=debug)
        data = plain.render_bbox(description_data, cameras, result_data_clean, height, width)
        return (height, width), data

    camera_data = refine_camera(init_cameras, description_data, result_data_clean, f_get_info)

    return blender.render(description_data, camera_data, result_data_clean, height, width, cache_dir, use_gpu, sample_count, denoising, output_type, debug)

def get_uid_data(task_csv):
    '''A dictionary that maps an uid to the description and result data
    '''
    uid_data_path = f'./data/uid_data.pkl'
    if os.path.exists(uid_data_path):
        with open(uid_data_path, 'rb') as file:
            uid_data = pkl.load(file)
    else:
        uid_data = {}
        tasks = csv_parse.all_valid_tasks_multiple(task_csv)
        for task in tasks:
            descpt_data, result_data = csv_parse.load_row(task)
            valid_states = csv_parse.valid_state_from_result(result_data)
            uids = [csv_parse.uid_from_description(descpt_data, state_name) for state_name in valid_states]
            for uid in uids:
                uid_data[uid] = (descpt_data, result_data)

        with open(uid_data_path, 'wb') as file:
            pkl.dump(uid_data, file)

    return uid_data

def get_uid_cidx(img_name):
    """
    :param img_name: format output_path / f'{uid} cam{cidx} rgb.png'
    """
    img_name = img_name.split("/")[-1]
    assert img_name[-8:] == " rgb.png"
    img_name = img_name[:-8]

    import re
    m = re.search(r'\d+$', img_name)
    assert not m is None
    cidx = int(m.group())

    img_name = img_name[:-len(str(cidx))]
    assert img_name[-4:] == " cam"
    uid = img_name[0:-4]

    return uid, cidx

def get_state(uid):
    """
    :param uid: ' - '.join([relation, obj1_uid, obj2_uid, state_name])
    """
    state = uid.split(' - ')[-1]
    return state

def render_scene_one(task_csv, output_folder, img_name, height, width, cache_dir,use_gpu, sample_count, denoising, skip, debug=False):
    '''Returns data for a particular scene and camera which is parsed from the img_name
    '''
    def save_rgb_depth(render_result, rgb_fname, depth_fname):
        """
        Save depth and rgb images
        """
        cv2.imwrite(str(rgb_fname), cv2.cvtColor(render_result['color'], cv2.COLOR_RGB2BGR))
        depth = render_result['depth']
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth_image = Image.fromarray(np.squeeze(depth))
        depth_image.save(depth_fname)

    def cleanup_result(render_result, state):
        render_result = render_result[state]
        for x in render_result:
            if not isinstance(render_result[x], int):
                render_result[x] = render_result[x][0]
        return render_result

    uid, cidx = get_uid_cidx(img_name)
    state = get_state(uid)

    rgb_fname = output_folder / f'{uid} cam{cidx} rgb.png'
    depth_fname = output_folder / f'{uid} cam{cidx} depth.tiff'
    output_fname = output_folder / f'{uid} cam{cidx}.pkl'

    skip_renderer = False
    if output_fname.exists() and skip:
        try:
            with open(output_fname, 'rb') as file:
                pkl.load(file)
        except (pickle.UnpicklingError, ImportError, EOFError) as e:
            print(f'Corrupted file: {output_fname}')
            print(e)
        else:
            print(f'Skipping file: {output_fname}')
            skip_renderer = True

    if not skip_renderer:
        uid_data = get_uid_data(task_csv)
        descpt_data, result_data = uid_data[uid]
        render_result = render_scene(descpt_data, result_data, height, width, cache_dir, use_gpu, sample_count, denoising,
                                     output_type='all', debug=debug, cidx=cidx, states=[state])
        render_result = cleanup_result(render_result, state)

        save_rgb_depth(render_result, rgb_fname, depth_fname)
        with open(output_fname, 'wb') as file:
            pkl.dump(render_result, file)

def render_scene_all(task_csv, output_folder, json_data, start, end, array_index, array_total, height, width, use_gpu, sample_count, denoising, skip, debug=False):
    '''Batch render scenes. Need to render the examples from index `start` to index `end`.
    But the job is submitted in batch, so as the number `array_index` job of the whole `array_total` jobs,
    it only needs to render a subset of [start, end). This subset range is computed through the `chunk` function.
    '''

    with open(json_data) as file:
        data = json.load(file)
    data = data['train'] + data['test']
    max_idx = len(data)

    def chunk(task_id, task_total, start, end):
        '''Chunk samples for tasks.'''
        total = end - start
        start_idx = int(task_id / task_total * total) + start
        end_idx = int((task_id + 1) / task_total * total) + start
        return start_idx, end_idx

    if start is None: start = 0
    if end is None: end = max_idx
    start_idx, end_idx = chunk(array_index, array_total, start, end)
    print(f'[{start_idx}, {end_idx}) <- [{start}, {end})')

    assert start <= start_idx < end
    assert start < end_idx <= end

    # Temporary directory to store the intermediate stuff
    if not debug:
        cache_dir = Path(config.TMP_DIR) / uuid.uuid4().hex
    else: # Debugging mode, use tmp under the workdir
        cache_dir = Path(__file__).parent.resolve() / 'tmp_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

    output_folder.mkdir(parents=True, exist_ok=True)
    for idx in range(start_idx, end_idx):
        sample = data[idx]
        img_name = sample['rgb']
        render_scene_one(task_csv, output_folder, img_name, height, width, cache_dir, use_gpu, sample_count, denoising, skip, debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', default=720, type=int)
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--task-csv', default=config.TASK_CSV)
    parser.add_argument('--json-data', default=config.JSON_DATA)
    parser.add_argument('--output-folder', type=Path, required=True)

    subparsers = parser.add_subparsers()
    render_parser = subparsers.add_parser('render') # Render the images

    render_parser.add_argument('--use-gpu', default='auto')
    render_parser.add_argument('--sample-count', default=512)
    render_parser.add_argument('--denoising', action='store_true', default=False)
    render_parser.add_argument('--debug', action='store_true', default=False)
    render_parser.add_argument('--array-index', type=int, default=0)
    render_parser.add_argument('--array-total', type=int, default=1)
    render_parser.add_argument('--start', type=int)
    render_parser.add_argument('--end', type=int)
    render_parser.add_argument('--skip', action='store_true', default=False)

    args = parser.parse_args()

    render_scene_all(**vars(args))
