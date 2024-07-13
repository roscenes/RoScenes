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
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Prediction:
    timeStamp: int
    boxes3D: npt.NDArray[np.float64]
    labels: npt.NDArray[np.float64]
    scores: npt.NDArray[np.float64]
    token: str
    velocities: npt.NDArray[np.float64]

    def sort(self, maxPredictionPerSample: int):
        """Sort boxes by score."""
        if len(self.boxes3D) > maxPredictionPerSample:
            raise ValueError(f"A prediction has too much boxes ({len(self.boxes3D)} boxes), which exceeds `maxBoxesPerSample = {maxPredictionPerSample}`. (token: {self.token}).")
        # [N] bigger is higher
        indices = np.argsort(-self.scores)
        self.boxes3D = self.boxes3D[indices]
        self.scores = self.scores[indices]
        self.labels = self.labels[indices]
        self.velocities = self.velocities[indices]
