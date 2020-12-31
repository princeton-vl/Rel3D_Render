import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def depth_colormap(depth, vmin=None, vmax=None, mask=None, eps=0.0001):
    ''' Colorize a depth map. If mask is provided, only colorize the masked region.
    :param depth: h x w numpy array.
    :param mask: h x w numpy array or None
    '''

    assert depth.shape == mask.shape
    assert depth.ndim == 2

    mask = mask.astype(np.bool)

    if mask is not None:
        if mask.max() == True:
            depth[~mask] = depth[mask].min()

    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = max(depth.max(), depth.min() + eps)

    depth = np.uint8((depth - vmin) / (vmax - vmin) * 255)

    color = cv2.applyColorMap(depth, cv2.COLORMAP_HOT)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    if mask is not None: # Grey out the unmasked region 
        color[~np.stack((mask,) * 3, axis=-1)] = 128

    return color


def draw_bbox_with_label(image, bbox, label, color):
    '''
    :param bbox: np.array([[u1, v1], [u2, v2]])
    '''
    image = cv2.rectangle(image.copy(), tuple(bbox[0][::-1].astype(np.int32)),
                                 tuple(bbox[1][::-1].astype(np.int32)), color, 1)
    text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    center = bbox.mean(axis=0)
    top_left = (int(center[1] - text_size[0] / 2), int(center[0] + text_size[1] / 2))
    cv2.putText(image, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def draw_front_up_axes(image, camera, transform_vector):
    '''
    :param image: h x w x 3, uint8
    :param camera: 10-d vector, origin/towards/up and fov
    '''

    height, width = image.shape[:2]
    fov = camera['fov']
    focal_length = width / np.tan(fov / 2) / 2

    def project(point_in_cam): # point: size 3
        x = point_in_cam[0] / -point_in_cam[2]
        y = point_in_cam[1] / -point_in_cam[2]
        v = round(float(x * focal_length + width / 2))
        u = round(float(height / 2 - y * focal_length))
        return (v, u)

    # Extract the object information
    aligned_absolute = transform_vector['aligned_absolute']
    centroid1 = aligned_absolute[0: 3]; init_rot1 = aligned_absolute[ 3: 6]; size1 = aligned_absolute[ 6: 9]
    centroid2 = aligned_absolute[9:12]; init_rot2 = aligned_absolute[12:15]; size2 = aligned_absolute[15:18]

    # Convert from Euler angles to rotation matrices
    init_rot1 = Rotation.from_euler('zxy', init_rot1, degrees=True).as_matrix()
    init_rot2 = Rotation.from_euler('zxy', init_rot2, degrees=True).as_matrix()

    # Extract front/up axes from rotation matrices
    front1 = init_rot1[:, 0]; up1 = init_rot1[:, 2]
    front2 = init_rot2[:, 0]; up2 = init_rot2[:, 2]

    centroid1_2d = project(centroid1); centroid2_2d = project(centroid2)
    front1_2d = project(centroid1 + front1 * size1[0])
    front2_2d = project(centroid2 + front2 * size2[0])
    up1_2d = project(centroid1 + up1 * size1[2])
    up2_2d = project(centroid2 + up2 * size2[2])

    image = cv2.arrowedLine(image, centroid1_2d, front1_2d, (0, 0, 0))
    image = cv2.arrowedLine(image, centroid1_2d, up1_2d, (255, 255, 255))
    image = cv2.arrowedLine(image, centroid2_2d, front2_2d, (0, 0, 0))
    image = cv2.arrowedLine(image, centroid2_2d, up2_2d, (255, 255, 255))

    return image
