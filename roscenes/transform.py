#    RoScenes
#    Copyright (C) 2024  Alibaba Cloud
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
from scipy.spatial.transform import Rotation


def xyzwlhq2corners(xyzwlhq: np.ndarray) -> np.ndarray:
    """
    ```
                             up z    x front (yaw=0)
                                ^   ^
                                |  /
                                | /
    (yaw=0.5*pi) left y <------ 0

          1 -front-- 0
         /|         /|
        2 --back-- 3 . h
        | |        | |
        . 5 -------. 4
        |/   bot   |/ l
        6 -------- 7
            w
    ```


    Args:
        xyzwlhq (`NDArray[float64]`): `[N, 10+]` array of [x, y, z], [w, l, h], and a quaternion [4], and other values not used here.

    Returns:
        `NDArray[float64]`: `[N, 8, 3]` corner coordinates.

    """
    xyzwlhq = xyzwlhq.copy()

    def _createCenteredBox(wlh):
        wlh = wlh / 2
        w, l, h = wlh[:, 0], wlh[:, 1], wlh[:, 2]
        # [N, 4, 3]
        bottom = np.stack([
            np.stack([ l, -w, -h], -1), # bottom head right
            np.stack([ l,  w, -h], -1), # bottom head left
            np.stack([-l,  w, -h], -1), # bottom tail left
            np.stack([-l, -w, -h], -1), # botoom tail right
        ], -2)
        top = bottom.copy()
        top[..., 2] *= -1
        # [N, 8, 3]
        corners = np.concatenate([top, bottom], -2)
        return corners

    # the box centered by (0, 0, 0), [N, 8, 3]
    centeredBox = _createCenteredBox(xyzwlhq[:, 3:6]).copy()
    # rotate, then translate
    rotatedBox = Rotation.from_quat(np.broadcast_to(xyzwlhq[:, 6:].copy()[:, None, :], [len(centeredBox), 8, 4]).reshape(-1, 4)).apply(centeredBox.reshape(-1, 3)).reshape(-1, 8, 3)
    # [N, 1, 3]
    xyz = xyzwlhq[:, None, :3]
    # [N, 8, 3]
    result = rotatedBox + xyz
    return result

def corners2xyzwlhq(corners3d: np.ndarray) -> np.ndarray:
    """
    ```
                             up z    x front (yaw=0)
                                ^   ^
                                |  /
                                | /
    (yaw=0.5*pi) left y <------ 0

          1 -front-- 0
         /|         /|
        2 --back-- 3 . h
        | |        | |
        . 5 -------. 4
        |/   bot   |/ l
        6 -------- 7
            w
    ```

    Args:
        corners: (N, 8, 3) [x0, y0, z0, ..., x7, y7, z7], (x, y, z) in lidar coords

    Returns:
        array: (N, 3 + 3 + 4), [x, y, z], [w, l, h], [q1, q2, q3, q4].
    """
    corners3d = corners3d.copy()
    def _transformMatrix(realCoord):
        # [3, 3]
        alignBasis = np.eye(3)
        transformMatrix = np.einsum("ji,njk->nik", alignBasis, realCoord)
        return Rotation.from_matrix(transformMatrix).as_quat()


    width = np.linalg.norm(corners3d[..., [4, 7, 0, 3], :] - corners3d[..., [5, 6, 1, 2], :], axis=-1).mean(-1)
    length = np.linalg.norm(corners3d[..., [4, 5, 0, 1], :] - corners3d[..., [7, 6, 3, 2], :], axis=-1).mean(-1)
    height = np.linalg.norm(corners3d[..., [4, 5, 6, 7], :] - corners3d[..., [0, 1, 2, 3], :], axis=-1).mean(-1)

    # calculate xyz on the bottom plane. top plane is parallel to bottom.
    # [N, 3]
    xReal = (corners3d[..., [4, 5], :] - corners3d[..., [7, 6], :]).sum(-2)
    xReal /= np.linalg.norm(xReal, axis=-1, keepdims=True)
    yReal = (corners3d[..., [6, 5], :] - corners3d[..., [7, 4], :]).sum(-2)
    yReal /= np.linalg.norm(yReal, axis=-1, keepdims=True)
    zReal = np.cross(xReal, yReal)
    zReal /= np.linalg.norm(zReal, axis=-1, keepdims=True)
    # [N, 3, 3]
    realCoord = np.stack([xReal, yReal, zReal], -1)

    # [N, 3, 3]
    quat = _transformMatrix(realCoord)

    # [N, 3]
    center_point = corners3d.mean(-2)
    # [N, 3 + 3 + 4]
    rectified = np.concatenate([center_point, np.stack([width, length, height], -1), quat], -1)

    return rectified

def xyzwlhq2kitti(xyzwlhq: np.ndarray) -> np.ndarray:
    """
    ```
                             up z    x front (yaw=0)
                                ^   ^
                                |  /
                                | /
    (yaw=0.5*pi) left y <------ 0

          1 -front-- 0
         /|         /|
        2 --back-- 3 . h
        | |        | |
        . 5 -------. 4
        |/   bot   |/ l
        6 -------- 7
            w
    ```

    Args:
        corners: (N, 8, 3) [x0, y0, z0, ..., x7, y7, z7], (x, y, z) in lidar coords

    Returns:
        kitti box: (7,) [x, y, z, w, l, h, r] in lidar coords, origin: (0.5, 0.5, 0.5)
    """
    xyzwlhq = xyzwlhq.copy()
    q = xyzwlhq[:, 6:10]
    # [N, 1]
    # This is confirmed by visualization
    yaw = np.pi / 2 + Rotation.from_quat(q).as_euler('zyx')[:, :1]
    return np.concatenate([xyzwlhq[:, :6], yaw], -1)

def corners2kitti(corners3d: np.ndarray) -> np.ndarray:
    """
    ```
                             up z    x front (yaw=0)
                                ^   ^
                                |  /
                                | /
    (yaw=0.5*pi) left y <------ 0

          1 -front-- 0
         /|         /|
        2 --back-- 3 . h
        | |        | |
        . 5 -------. 4
        |/   bot   |/ l
        6 -------- 7
            w
    ```

    Args:
        xyzwlhq: (N, 3 + 3 + 4), [x, y, z], [w, l, h], [q1, q2, q3, q4].

    Returns:
        kitti box: (7,) [x, y, z, w, l, h, r] in lidar coords, origin: (0.5, 0.5, 0.5)
    """
    return xyzwlhq2kitti(corners2xyzwlhq(corners3d))

def kitti2xyzwlhq(kitti: np.ndarray) -> np.ndarray:
    """
    See `xyzwlhq2kitti`, this is the reverse transform.
    """
    xyzwlhr = kitti[:, :7].copy()
    yaws = xyzwlhr[:, 6:].copy()
    # See xyzwlhq2kitti
    ypr = np.concatenate([yaws - np.pi / 2, np.zeros_like(yaws), np.zeros_like(yaws)], -1)
    # [N, 4]
    q = Rotation.from_euler("zyx", ypr).as_quat()
    # [N, 10]
    return np.concatenate([xyzwlhr[:, :6], q], -1)

def kitti2corners(kitti: np.ndarray) -> np.ndarray:
    """
    ```
                             up z    x front (yaw=0)
                                ^   ^
                                |  /
                                | /
    (yaw=0.5*pi) left y <------ 0

          1 -front-- 0
         /|         /|
        2 --back-- 3 . h
        | |        | |
        . 5 -------. 4
        |/   bot   |/ l
        6 -------- 7
            w
    ```

    Args:
        kitti box: (7,) [x, y, z, w, l, h, r] in lidar coords, origin: (0.5, 0.5, 0.5)

    Returns:
        corners: (N, 8, 3).
    """
    xyzwlhr = kitti[:, :7].copy()

    def _createCenteredBox(wlh):
        wlh = wlh / 2
        w, l, h = wlh[:, 0], wlh[:, 1], wlh[:, 2]
        # [N, 4, 3]
        bottom = np.stack([
            np.stack([ l, -w, -h], -1), # bottom head right
            np.stack([ l,  w, -h], -1), # bottom head left
            np.stack([-l,  w, -h], -1), # bottom tail left
            np.stack([-l, -w, -h], -1), # botoom tail right
        ], -2)
        top = bottom.copy()
        top[..., 2] *= -1
        # [N, 8, 3]
        corners = np.concatenate([top, bottom], -2)
        return corners

    # the box centered by (0, 0, 0), [N, 8, 3]
    centeredBox = _createCenteredBox(xyzwlhr[:, 3:6]).copy()
    # rotate, then translate
    # [N, 3]
    # See xyzwlhq2kitti
    yaws = xyzwlhr[:, 6:].copy()
    ypr = np.concatenate([yaws - np.pi / 2, np.zeros_like(yaws), np.zeros_like(yaws)], -1)
    rotatedBox = Rotation.from_euler("zyx", np.broadcast_to(ypr[:, None, :], [len(centeredBox), 8, 3]).reshape(-1, 3)).apply(centeredBox.reshape(-1, 3)).reshape(-1, 8, 3)
    # [N, 1, 3]
    xyz = xyzwlhr[:, None, :3]
    # [N, 8, 3]
    result = rotatedBox + xyz
    return result


def yaw2quat(yaws: np.ndarray) -> np.ndarray:
    """Converts yaw to quanternion.
    ```
            BEV
        Y
        ^
        |  /
        | /
        |/ yaw
        0--------> X

    Args:
        yaws: (N) or (N, 1)

    Returns:
        quats: (N, 4).
    ```
    """
    if len(yaws.shape) == 1:
        yaws = yaws[:, None]
    elif len(yaws.shape) > 2:
        raise RuntimeError(f"yaws shape mismatch: {yaws.shape}. Expected a [N, 1]")
    elif yaws.shape[-1] != 1:
        raise RuntimeError(f"yaws shape mismatch: {yaws.shape}. Expected a [N, 1]")

    # See xyzwlhq2kitti
    # [N, 3]
    ypr = np.concatenate([yaws.copy() - np.pi / 2, np.zeros_like(yaws), np.zeros_like(yaws)], -1)
    # [N, 4]
    rotation = Rotation.from_euler("zyx", ypr).as_quat()
    return rotation


def quat2yaw(quats: np.ndarray) -> np.ndarray:
    """Converts quanternion to yaw. pitch and roll are set to 0.
    ```
            BEV
        Y
        ^
        |  /
        | /
        |/ yaw
        0--------> X

    Args:
        quats: (N, 4).

    Returns:
        yaws: (N, 1)
    ```
    """
    if len(quats.shape) != 2:
        raise RuntimeError(f"quats shape mismatch: {quats.shape}. Expected a [N, 4]")
    elif quats.shape[-1] != 1:
        raise RuntimeError(f"quats shape mismatch: {quats.shape}. Expected a [N, 4]")

    return Rotation.from_quat(quats).as_euler('zyx')[:, :1]

def xyzwlhq2bevbox(boxes: np.ndarray) -> np.ndarray:
    """Converts 3D boxes to bev rotated 2D boxes.
    ```
            BEV
        Y
        ^
        |  /
        | /
        |/ yaw
        0--------> X

    Args:
        boxes: (N, 10+) in xyzwlhq format.

    Returns:
        bev boxes: (N, 4, 2).
    ```
    """
    yaw = quat2yaw(boxes[:, 6:10])
    # [N, 2]
    xy = boxes[:, :2]
    # [N, 2]
    halfLW = (boxes[:, 3:5] / 2)[:, ::-1]
    # [N, 4, 2]
    normalBox = np.array([
        [halfLW[:, 0], -halfLW[:, 1]],
        [halfLW[:, 0], halfLW[:, 1]],
        [-halfLW[:, 0], halfLW[:, 1]],
        [-halfLW[:, 0], -halfLW[:, 1]]
    ]).transpose(2, 0, 1)
    # [N, 2, 2]
    rotate = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ]).transpose(2, 1, 0)
    rotateBox = np.matmul(normalBox, rotate)
    final = rotateBox + xy[:, None, :]
    return final