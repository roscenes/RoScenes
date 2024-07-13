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
from __future__ import annotations

import os
from pathlib import Path
from typing import OrderedDict, TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from roscenes.data.clip import Clip


@dataclass
class Frame:
    """All captured images by cameras at specific timestamp, in a clip."""
    timeStamp: int
    """The Unix timestamp, 13-digit."""
    imagePaths: OrderedDict[str, Path]
    """Key is camera token, value is image path (this is path relative to root directory, use `Frame.images` instead to get path joint with root directory)."""
    index: int
    """The absolute index in the clip sequence."""
    parent: Clip = field(init=False)

    @property
    def previous(self) -> Frame | None:
        raise NotImplementedError
    @property
    def next(self) -> Frame | None:
        raise NotImplementedError

    boxes3D: npt.NDArray[np.float64]
    """`[N, 3+3+4]`. All 3d boxes, each column is `X, Y, Z, w, l, h, q1, q2, q3, q4`. Coordinate system is:

        ```
                    up Y    X front         rotated from BEV:
                       ^   ^    (yaw=0)           ^ Y
                       |  /                       |
        (yaw=0.5*pi)   | /                        |        X
        left Z <------ 0                        Z .------>

              1 -front-- 0
             /|         /|
            2 --back-- 3 h
            | |        | |
            . 5 -------. 4
            |/         |l
            6 ---w---- 7
        ```
        The quaternion rotates the standard basis to the box basis, which is box-centered, X points to front, Y points to left, Z points to top.
    """
    velocities: npt.NDArray[np.float64]
    """`[N, 2]`. `[vx, vy]` of each box in m/s. `vz` is ignored."""
    labels: npt.NDArray[np.int64]
    """`[N]`. Integer label of each box. Check `consts.labels` for details."""
    instancesIDs: npt.NDArray[np.int64]
    """`[N]`. Integer tracking ID of each box. This value is unique across clip sequence."""

    visibleBoundingBoxes: OrderedDict[str, npt.NDArray[np.float64]]
    """Key is camera token, value is `[num_visible, 4]` array. Each row is a `[xmin, ymin, xmax, ymax]` 2d bounding box of a visible object in image coordinates under this camera. Visible object ID see `Frame.visibleInstanceIDs`."""
    visibleProjected3DBoxes: OrderedDict[str, npt.NDArray[np.float64]]
    """Key is camera token, value is `[num_visible, 8, 3]` array. Each row is 8 3d corners representing a 3D box of a visible object in camera coordinates under this camera. Visible object ID see `Frame.visibleInstanceIDs`."""
    visibleObjectOcclusions: OrderedDict[str, npt.NDArray[np.float64]]
    """Key is camera token, value is `[num_visible]` array. Each row is occlusion rate (0~1) of a visible object under this camera. Visible object ID see `Frame.visibleInstanceIDs`."""
    visibleInstanceIDs: OrderedDict[str, npt.NDArray[np.int64]]
    """Key is camera token, value is `[num_visible]` array. Each row is tracking ID of visible object under this camera."""
    visibleLabels: OrderedDict[str, npt.NDArray[np.int64]]
    """Key is camera token, value is `[num_visible]` array. Each row is label of visible object under this camera. Visible object ID see `Frame.visibleInstanceIDs`."""
    # behindStillObject: OrderedDict[str, npt.NDArray[np.bool_]]
    # """Key is camera token, value is `[num_visible]` array. Each row is indicator of visible object behinds still objects under this camera. Visible object ID see `Frame.visibleInstanceIDs`."""

    # The unique identifier
    token: str
    """The unique identifier of frame."""

    @property
    def images(self) -> OrderedDict[str, str]:
        """The image path with root dir joined.

        Returns:
            OrderedDict[str, str]: Key is camera token, value is joined image paths.
        """
        return OrderedDict((key, os.path.join(self.parent.parent.rootDir, "images", value)) for key, value in self.imagePaths.items())

    @property
    def extrinsics(self) -> OrderedDict[str, np.ndarray]:
        """All cameras' extrinsic.

        Returns:
            OrderedDict[str, np.ndarray]: Key is camera token, value is extrinsic. `extrinsic @ (X, Y, Z, 1) = (x, y, z, _)`.
        """
        return OrderedDict((key, value.extrinsic.copy()) for key, value in self.parent.cameras.items())

    @property
    def intrinsics(self) -> OrderedDict[str, np.ndarray]:
        """All cameras' intrinsic.

        Returns:
            OrderedDict[str, np.ndarray]: Key is camera token, value is intrinsic. `intrinsic @ (x, y, z, 1) = (u0, v0, d, _)`. Then `(u, v) = (u0, v0) / d`.
        """
        return OrderedDict((key, value.intrinsic.copy()) for key, value in self.parent.cameras.items())

    @property
    def world2images(self) -> OrderedDict[str, np.ndarray]:
        """All cameras' world to image transforms.

        Returns:
            OrderedDict[str, np.ndarray]: Key is camera token, value is transform matrix from world coords to image coords [4, 4].
        """
        return OrderedDict((key, value.world2image.copy()) for key, value in self.parent.cameras.items())

    @property
    def instanceOcc(self) -> np.ndarray:
        # [N, num_cams], 0 -> not occluded, 1 -> totally occluded, -1 -> not visible
        occs = np.full([len(self.instancesIDs), len(self.imagePaths)], -1.0)
        for i, (instanceIDThisView, occ) in enumerate(zip(self.visibleInstanceIDs.values(), self.visibleObjectOcclusions.values())):
            # [N, n]
            thisViewToGlobalIDMapping = self.instancesIDs[:, None] == instanceIDThisView
            # this row is all 0 -> not visible
            notvisibleIDs = thisViewToGlobalIDMapping.sum(-1) < 1
            # [N, n] @ [n] -> [N], bool assign to int64 array becomes 1
            occs[:, i] = thisViewToGlobalIDMapping @ occ
            # assign not visibles to -1
            occs[:, i][notvisibleIDs] = -1
        # num of visible cameras
        visibleCount = (occs >= 0).sum(-1)
        instanceOcc = occs.copy()
        instanceOcc[instanceOcc < 0] = 0
        instanceOcc = instanceOcc.sum(-1) / visibleCount
        # all invisble is -1
        instanceOcc[visibleCount < 1] = -1
        # instance level occ, -1=invisible, 0=no occlusion, 1=total occlusion
        return instanceOcc