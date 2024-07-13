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
import pickle
from io import BytesIO
from typing import OrderedDict, TYPE_CHECKING
from dataclasses import dataclass, field
from functools import wraps

import lmdb
import numpy as np
import numpy.typing as npt

from roscenes.data.camera import Camera
from roscenes.data.frame import Frame
from roscenes.typing import Indexing

if TYPE_CHECKING:
    from roscenes.data.scene import Scene


# For lmdb environment lazy loading
def _lazyLoadLMDB(func):
    @wraps(func)
    def _decorator(self: Clip, *args, **kwargs):
        if self._env is None:
            # 512MiB is enough for a clip.
            # disable readahead to enhance random read performance.
            # disable lock since we use in a read-only scenario.
            self._env = lmdb.Environment(os.path.join(self.parent.rootDir, "database", self.token), map_size=1024*1024*512, readonly=True, readahead=False, lock=False)
            # enable buffer to enhance performance
            self._txn = self._env.begin(buffers=True)
        return func(self, *args, **kwargs)
    return _decorator


@dataclass
class Clip:
    """A sequence of captured images and annotations."""
    cameras: OrderedDict[str, Camera]
    """Key is camera token, value is a single Camera."""
    startTimeStamp: int
    """The 13-digit Unix timestamp represents start frame."""
    endTimeStamp: int
    """The 13-digit Unix timestamp represents end frame."""
    sequence: list[str]
    """All frame's token from start to end."""

    bound: npt.NDArray[np.float64]
    """The `[6]` cuboid that covers all 3D annotations. Ordered by `[xmin, ymin, zmin, xmax, ymax, zmax]`."""

    token: str
    """The unique identifier of clip."""

    parent: Scene = field(init=False)
    """The ancestor."""

    # these private attributes are used for lmdb
    _env: lmdb.Environment = None
    _txn: lmdb.Transaction = None

    def _postInitialize(self):
        # update camera's parent
        for cam in self.cameras.values():
            cam.parent = self

    def _tryClose(self):
        if self._env is not None:
            if self._txn is not None:
                del self._txn
            self._env.close()
            del self._env
        self._env = None
        self._txn = None

    # def projection(self, corners3d: np.ndarray, W=1920, H=1080) -> list[tuple[np.ndarray, np.ndarray]]:
    #     """Project unnormalized 3d points to images in each camera.

    #     Args:
    #         corners3d (`NDArray[float64]`): `[..., 8, 3]` array of unnormalized 3D corner boxes.

    #     Returns:
    #         `list[tuple[NDArray[float64], NDArray[bool]]]`: Length == cameras. each is a (`[..., 8, 3]`, `[...]`): projected corners boxes with depths and visibilities.
    #     """
    #     results = list()
    #     # [N, 8, 3] -> [N, 8, 4], [x, y, z, 1]
    #     corners3d = np.concatenate([corners3d, np.ones([corners3d.shape[0], corners3d.shape[1], 1], dtype=corners3d.dtype)], -1)

    #     for cam in self.cameras.values():
    #         boxUnderImage = corners3d @ cam.world2image.T

    #         # [N, 8, 3]
    #         boxUnderImage = boxUnderImage[..., :3]

    #         # scale
    #         # [N, 8, 3]
    #         boxUnderImage[..., :2] /= (boxUnderImage[..., 2:] + 1e-6)

    #         # [N, 8]
    #         reasonable = (boxUnderImage[..., 0] < W) * (boxUnderImage[..., 0] > 0) * (boxUnderImage[..., 1] > 0) * (boxUnderImage[..., 1] < H)

    #         # check any point is in image, [N]
    #         reasonable = reasonable.sum(-1).astype(bool)
    #         # in front of camera and in image area [N]
    #         inImage = (boxUnderImage.mean(-2)[..., 2] > 0) * reasonable

    #         results.append((boxUnderImage, inImage))
    #     return results

    # ignore lmdb objects when pickling
    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop('_env', None)
        d.pop('_txn', None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


    def __del__(self):
        if self._env is not None:
            if self._txn is not None:
                del self._txn
            self._env.close()
        self._env = None
        self._txn = None

    @_lazyLoadLMDB
    def __iter__(self):
        for v in self.sequence:
            buffer = self._txn.get(v.encode())
            if buffer is None:
                raise KeyError(v)
            frame: Frame = pickle.load(BytesIO(buffer))
            frame.parent = self
            yield frame

    @_lazyLoadLMDB
    def __getitem__(self, idx: Indexing) -> Frame:
        viewTokens = self.sequence[idx]
        # A single frame
        if isinstance(viewTokens, str):
            buffer = self._txn.get(viewTokens.encode())
            if buffer is None:
                raise KeyError(viewTokens)
            frame: Frame = pickle.load(BytesIO(buffer))
            frame.parent = self
            return frame

        result = list()
        for viewToken in viewTokens:
            buffer = self._txn.get(viewToken.encode())
            if buffer is None:
                raise KeyError(viewToken)
            frame: Frame = pickle.load(BytesIO(buffer))
            frame.parent = self
            result.append(frame)
        return result

    # def getByTimeStamp(self, timeStamp: int):
    #     allTimeStamps = np.array([v.timeStamp for v in self])
    #     delta = np.abs(timeStamp - allTimeStamps)
    #     nearest = np.argmin(delta)
    #     return self[nearest]


    def __len__(self):
        return len(self.sequence)
