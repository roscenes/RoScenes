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
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from roscenes.evaluation.detection.result import APResult, TPResult
from roscenes.misc import Registry


class HandlerReg(Registry):
    pass


class Handler(ABC):
    name: str
    @abstractmethod
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], gtVelocities: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predVelocities: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64]):
        """Calculate evalutaion metrics using groundtruth and prediction.

        Args:
            gtBoxes (np.ndarray): [N, 10+] array, which at least contains xyz, wlh, q1q2q3q4.
            gtLabels (np.ndarray): [N], int label
            predBoxes (np.ndarray): [n, 10+] array
            predLabels (np.ndarray): [n], int label
            predScores (np.ndarray): [n] scores
            assignResult (np.ndarray): [n] for each prediction, the matched gt index. -1 if no assignment available
        """
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any):
        return self.handle(*args, **kwds)


class RanklistHandler(Handler):
    pass

class TruePositiveHandler(Handler):
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], gtVelocities: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predVelocities: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64]):
        # NOTE: assignResult < 0 means a false one
        if np.sum(assignResult >= 0) < 1:
            # If no true positives, return all 1 (max error) TPResult.
            return TPResult(np.ones_like(predScores), predScores)
        else:
            return None

@HandlerReg.register
class PrecisionRecall(RanklistHandler):
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64], **_):
        # NOTE: assignResult < 0 means a false one
        matches = assignResult >= 0
        tp = np.cumsum(matches.astype(np.float64))
        fp = np.cumsum((~matches).astype(np.float64))
        prec = tp / (fp + tp)
        recall = tp / float(len(gtBoxes))

        return APResult(prec, recall, predScores.copy())

@HandlerReg.register
class TranslationError(TruePositiveHandler):
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64], **kwargs):
        check = super().handle(gtBoxes=gtBoxes, predBoxes=predBoxes, predScores=predScores, assignResult=assignResult, **kwargs)
        if check is not None:
            return check
        # NOTE: assignResult < 0 means a false one
        mask = assignResult >= 0
        # [#tp, 10+]
        findedBoxes = gtBoxes[assignResult[mask]]
        predBoxes = predBoxes[mask]
        # [#tp]
        centerDistance = np.sqrt(((findedBoxes[:, :2] - predBoxes[:, :2]) ** 2).sum(-1))
        score = predScores[mask]

        return TPResult(centerDistance, score)

@HandlerReg.register
class ScaleError(TruePositiveHandler):
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64], **kwargs):
        check = super().handle(gtBoxes=gtBoxes, predBoxes=predBoxes, predScores=predScores, assignResult=assignResult, **kwargs)
        if check is not None:
            return check
        # NOTE: assignResult < 0 means a false one
        mask = assignResult >= 0
        # [#tp, 10+]
        findedSize = gtBoxes[assignResult[mask]][:, 3:6]
        predSize = predBoxes[mask][:, 3:6]

        findedVolume = np.prod(findedSize, -1)
        predVolume = np.prod(predSize, -1)
        minVolume = np.prod(np.min(np.stack([findedSize, predSize], 1), 1), -1)

        # [#tp]
        iou = minVolume / (findedVolume + predVolume - minVolume)
        score = predScores[mask]

        return TPResult(1 - iou, score)

@HandlerReg.register
class VelocityError(TruePositiveHandler):
    def handle(self, *, gtVelocities: npt.NDArray[np.float64], predVelocities: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64], **kwargs):
        check = super().handle(gtVelocities=gtVelocities, predVelocities=predVelocities, predScores=predScores, assignResult=assignResult, **kwargs)
        if check is not None:
            return check
        # NOTE: assignResult < 0 means a false one
        mask = assignResult >= 0
        # [#tp, 2]
        findedV = gtVelocities[assignResult[mask]]
        predV = predVelocities[mask]
        # [#tp, 2]
        velocityDiff = np.sqrt(((findedV - predV) ** 2).sum(-1))
        score = predScores[mask]

        return TPResult(velocityDiff, score)

@HandlerReg.register
class OrientationError(TruePositiveHandler):
    def handle(self, *, gtBoxes: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64], assignResult: npt.NDArray[np.float64], **kwargs):
        check = super().handle(gtBoxes=gtBoxes, predBoxes=predBoxes, predScores=predScores, assignResult=assignResult, **kwargs)
        if check is not None:
            return check
        # NOTE: assignResult < 0 means a false one
        mask = assignResult >= 0
        # [#tp, 4]
        findedRotation = gtBoxes[assignResult[mask]][:, 6:10]
        predRotation = predBoxes[mask][:, 6:10]
        # [#tp]
        # assume gt = diff * pred
        # so diff = gt * (pred)^-1
        # magnitude = rad diff angle
        # range = [0, pi]
        rotationMagnitude = (Rotation.from_quat(findedRotation) * Rotation.from_quat(predRotation).inv()).magnitude()
        score = predScores[mask]

        return TPResult(rotationMagnitude, score)


if __name__ == "__main__":
    mask = np.random.rand(25)>0.5
    recall = np.cumsum(mask)
    recall = recall / recall[-1]
    recall_interp = np.linspace(0, 1, 101)
    conf = np.sort(np.random.rand(25))[::-1]
    value = np.sort(np.random.rand(25))
    conf_interp=np.interp(recall_interp, recall, conf, right=0)

    value_interp1 = np.interp(conf_interp[::-1], conf[mask][::-1], value[mask][::-1])[::-1]
    value_interp2 = np.interp(recall_interp, recall[mask], value[mask])

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(recall_interp, conf_interp)
    ax2.plot(recall_interp, value_interp1, c="r")
    ax2.plot(recall_interp, value_interp2, c="b")
    plt.savefig("interpolate.png")