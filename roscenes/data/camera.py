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

from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from roscenes.data.clip import Clip



@dataclass
class Camera:
    """A camera in the clip."""
    name: str
    """The 5-char camera name."""
    extrinsic: npt.NDArray[np.float64]
    """The `[4, 4]` camera extrinsic transforms World coord to Camera coord. `extrinsic @ (X, Y, Z, 1) = (x, y, z, _)`."""
    intrinsic: npt.NDArray[np.float64]
    """The `[4, 4]` camera intrinsic transforms Camera coord to Image coord. `intrinsic @ (x, y, z, 1) = (u0, v0, d, _)`. Then `(u, v) = (u0, v0) / d`."""
    # depthRange: npt.NDArray[np.float64]
    # """The camera depth in shape `[2]`, ordered by `[near, far]`."""

    @property
    def world2image(self) -> npt.NDArray[np.float64]:
        return (self.intrinsic @ self.extrinsic).copy()

    # parent clip
    parent: Clip = field(init=False)

    # The unique identifier
    token: str
    """The unique identifier of camera."""

    # @property
    # def focal(self) -> tuple[float, float]:
    #     return float(self.intrinsic[0, 0]), float(self.intrinsic[1, 1])
