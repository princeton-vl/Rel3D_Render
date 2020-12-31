import argparse
import pickle
import trimesh
import fcl # flexible collision detection
import numpy as np
from pathlib import Path
import config
from constant import BLENDER_TO_UNITY
from scipy.spatial.transform import Rotation
from typing import List, Dict
from common import as_mesh, extract_obj_paths, extract_obj_transforms, extract_extrinsics_from_camera
import render_scene as rs

class CacheDict:
    def __init__(self, path):
        if isinstance(path, str):
            path = Path(path)
        else:
            assert isinstance(path, Path)

        self.path = path
        self.data = None

    def lazy_init(self):
        if self.data is None:
            if self.path.is_file():
                self.data = pickle.loads(self.path.read_bytes())
            else:
                self.data = dict()

    def flush(self):
        if self.data:
            self.path.parent.mkdir(exist_ok=True, parents=True)
            self.path.write_bytes(pickle.dumps(self.data))
            self.path.chmod(0o664)


    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]

        value = self._fetch(key)
        self.data[key] = value

        return value


    def _fetch(self, key):
        # Do the computationally intensive stuff here
        raise NotImplementedError


class CuboidInfoCacheDict(CacheDict):
    def _fetch(self, key):
        # First, find the object mesh model
        dataset, full_id = key
        shape_root = Path(config.SHAPE_ROOT)
        obj_path = shape_root / dataset / full_id / f'{full_id}.obj'

        # Then load the mesh vertices using the TriMesh library
        scene_or_mesh = trimesh.load_mesh(str(obj_path))
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

        assert(isinstance(mesh, trimesh.Trimesh))

        return tuple(mesh.extents), tuple(mesh.centroid)


__cuboid_info = CuboidInfoCacheDict(Path(config.CACHE_DIR) / 'cuboid_info.bin')

def flush_cache():
    __cuboid_info.flush()


def get_transform_vector(camera_info: Dict, descpt_objectinfo: List, result_objectinfo: List) -> List[float]:
    '''
    :return: a dictionary with keys ['raw_absolute', 'aligned_absolute', 'aligned_relative']
    Raw absolute: a 24-d vector of object A's centroid + object A's rotation + object A's aligning rotation
                  + object A's sizes
                  object B's centroid + object B's rotation + object B's aligning rotation
                  + object B's sizes
                  all in camera space except for sizes
                  the reference frame is the world frame.
    Aligned absolute: a 18-d vector of object A's centroid + object A's combined rotation + object A's sizes in
                       (front/back, left/right, top/bottom)
                      + same for object B
                      all in camera space except for sizes
    Aligned relative: a 9-d vector of object B's centroid + object B's combined rotation in object A's reference frame
                      (only consider translation and rotation, no scaling or shearing) + object B's sizes (in front/back,
                      left/right, top/bottom) / object A's sizes (same format).
    '''
    raw_absolute = []
    aligned_absolute = []
    aligned_relative = []

    # The cached map
    __cuboid_info.lazy_init()

    cuboid_sizes = []
    centroids = []

    # Camera transformations
    cam_r, cam_t = extract_extrinsics_from_camera(camera_info)

    # Get the sizes and the centers for the two objects
    init_rots = []
    for i in range(2):
        key = (descpt_objectinfo[i]['dataset'], descpt_objectinfo[i]['id'])
        cuboid_size, centroid = __cuboid_info[key]
        cuboid_sizes.append(np.array(cuboid_size))
        centroids.append(np.array(centroid + (1,)))

        vi = np.array(descpt_objectinfo[i]['front'])
        vk = np.array(descpt_objectinfo[i]['up'])
        vj = np.cross(vk, vi)
        init_rots.append(np.stack([vi, vj, vk], axis=1))

    # Get the translations and rotations for the two objects
    translations = []
    rotations = []
    transforms = []
    scales = []
    for i in range(2):
        transform = np.array(list(map(float, result_objectinfo[i]['meshTransformMatrix'].split(',')))).reshape(4, 4)
        # convert the transformation matrix from unity coordinates to blender coordinates
        transform = BLENDER_TO_UNITY.T @ transform @ BLENDER_TO_UNITY
        transforms.append(transform)

        # Reference: https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati/417813
        # See doc.png
        translations.append(transform[:3, 3])
        scale = np.sqrt(np.sum(transform[:3, :3] ** 2, axis=1))
        scales.append(scale)
        rotations.append(transform[:3, :3] / scale.reshape(1, 3))

    # The vector from object 0's centroid to object 1's centroid is M_1 @ c_1 - M_0 @ c_0
    # assume front=i_0, up=k_0 in the object0's local space, and in a reference frame
    # where x=i_0, y=j_0, z=k_0, the object2's centroid is loc, then:
    # R_0 @ (i_0, j_0, k_0) @ loc = M_1 @ c_1 - M_0 @ c_0
    loc = init_rots[0].T @ rotations[0].T @ (transforms[1] @ centroids[1] - transforms[0] @ centroids[0])[:3]

    # The pose vector represented using rotation
    pose_mat = init_rots[0].T @ rotations[0].T @ rotations[1] @ init_rots[1]
    pose_euler = Rotation.from_matrix(pose_mat).as_euler('zxy', degrees=True)

    # The relative sizes on front-back/left-right/top-bottom
    rel_sizes = np.abs((init_rots[1].T @ (cuboid_sizes[1] * scales[1])) / (init_rots[0].T @ (cuboid_sizes[0] * scales[0])))

    # Not adding the relative sizes as we are adding the absolute sizes of the objects
    aligned_relative += loc.tolist() + pose_euler.tolist()

    for i in range(2):
        raw_absolute += (cam_r @ (transforms[i] @ centroids[i])[:3] + cam_t).tolist() # Centroid in camera coordinates
        raw_absolute += Rotation.from_matrix(cam_r @ rotations[i]).as_euler('zxy', degrees=True).tolist() # Extracted rotation
        raw_absolute += Rotation.from_matrix(init_rots[i]).as_euler('zxy', degrees=True).tolist() # Aligning rotation
        raw_absolute += (cuboid_sizes[i] * scales[i]).tolist() # Sizes in xyz in the reference frame of itself
        aligned_absolute += (cam_r @ (transforms[i] @ centroids[i])[:3] + cam_t).tolist() # Centroid in camera coordinates
        aligned_absolute += Rotation.from_matrix(cam_r @ rotations[i] @ init_rots[i]).as_euler('zxy', degrees=True).tolist() # Combined rotation
        aligned_absolute += np.abs(init_rots[i].T @ (cuboid_sizes[i] * scales[i])).tolist() # Sizes in (front/back, left/right, top/bottom)

    # Per Ankit's request, also add the absolute scales into the relative vector
    aligned_relative += np.abs(init_rots[0].T @ (cuboid_sizes[0] * scales[0])).tolist() + np.abs(init_rots[1].T @ (cuboid_sizes[1] * scales[1])).tolist()

    assert len(raw_absolute) == 24
    assert len(aligned_absolute) == 18
    assert len(aligned_relative) == 12

    return dict(raw_absolute=raw_absolute, aligned_absolute=aligned_absolute, aligned_relative=aligned_relative)


def compute_mesh_distance(descpt_data, result_data):
    '''In each state in result_data, compute the distance
    between mesh A and mesh B. This information
    can be used further to decide A touching B  / A far away from B.
    '''
    obj1_path, obj2_path = extract_obj_paths(descpt_data)
    obj1 = as_mesh(trimesh.load(str(obj1_path)))
    obj2 = as_mesh(trimesh.load(str(obj2_path)))

    ret = {}
    # Our transformations have scaling, but in fcl there seems only rotation and translation
    # So we transform the vertices beforehand

    for state_name, state_data in result_data.items():
        transform1, transform2 = extract_obj_transforms(state_data)

        m1 = fcl.BVHModel(); m2 = fcl.BVHModel()
        m1.beginModel(len(obj1.vertices), len(obj1.faces))
        m1.addSubModel(obj1.vertices @ transform1[:3, :3].T + transform1[:3, 3:].T, obj1.faces)
        m1.endModel()
        m2.beginModel(len(obj2.vertices), len(obj2.faces))
        m2.addSubModel(obj2.vertices @ transform2[:3, :3].T + transform2[:3, 3:].T, obj2.faces)
        m2.endModel()

        c1 = fcl.CollisionObject(m1, fcl.Transform(np.eye(3), np.zeros(3)))
        c2 = fcl.CollisionObject(m2, fcl.Transform(np.eye(3), np.zeros(3)))

        request = fcl.DistanceRequest()
        result = fcl.DistanceResult()

        dist = fcl.distance(c1, c2, request, result)

        ret[state_name] = dist

    return ret


def get_transform(task_csv, output_folder, img_path):
    '''Returns data for a particular scene and camera which is parsed from the img_name
    '''
    print(img_path)
    uid, cidx = rs.get_uid_cidx(img_path)
    state = rs.get_state(uid)
    output_fname = output_folder / f'{uid} cam{cidx}.pkl'

    uid_data = rs.get_uid_data(task_csv)
    descpt_data, result_data = uid_data[uid]

    print(output_fname)
    with open(output_fname, 'rb') as file:
        result = pickle.load(file)

    transform_vector = get_transform_vector(result['camera'], descpt_data['objectInfos'], result_data[state]['objectInfos'])
    return transform_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-csv', default=config.TASK_CSV)
    parser.add_argument('--output-folder', type=Path, required=True)
    parser.add_argument('--img-path', required=True)

    args = parser.parse_args()
    print(get_transform(**vars(args)))
