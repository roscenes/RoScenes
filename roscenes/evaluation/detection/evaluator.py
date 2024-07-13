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

import warnings
from itertools import product

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

from roscenes.data.scene import Scene
from roscenes.evaluation.detection.prediction import Prediction
from roscenes.evaluation.detection.result import ClassWiseDetectionResult, DetectionResult, ThresholdDetectionResult, TPResult
from roscenes.evaluation.detection.handlers import RanklistHandler, TruePositiveHandler
from roscenes.evaluation.detection.config import DetectionEvaluationConfig, ThresholdMetric
from roscenes.evaluation.detection.aggregator import ClasswiseAggregator
from roscenes.consts import richProgress
from roscenes.misc import progressedJoblib


class MultiView3DEvaluator:
    def __init__(self, config: DetectionEvaluationConfig):
        if config.thresholdMetric == ThresholdMetric.IOU:
            raise NotImplementedError
        self.config = config
        self.handlers = config.createHandlers()
        self._labelIDMapping = {c: i for i, c in enumerate(self.config.classes)}
        self.progress = richProgress

    def __call__(self, clip: Scene, prediction: list[Prediction]) -> ClassWiseDetectionResult:
        with self.progress:
            return self.collectResult(clip, prediction)

    def _checkCorrespondence(self, clip: Scene, prediction: list[Prediction]):
        if len(clip) != len(prediction):
            warnings.warn(f"The given prediction length mismatch. Expected (gt) length: {len(clip)}. Got length: {len(prediction)}")
        for v, p in zip(clip, prediction):
            if v.token != p.token:
                raise ValueError(f"Clip and prediction tokens are not matched. Clip token: {v.token}, prediction token: {p.token}.")
            if v.timeStamp != p.timeStamp:
                raise ValueError(f"Clip and prediction are not time aligned at {v.token} (timestamp: {v.timeStamp}) and {p.token} (timestamp: {p.timeStamp}).")

    def _filterGT(self, gtBoxes: npt.NDArray[np.float64], gtLabels: npt.NDArray[np.float64], gtVelocity: npt.NDArray[np.float64], gtViewIdx: npt.NDArray[np.float64]):
        x, y, z = gtBoxes[:, 0], gtBoxes[:, 1], gtBoxes[:, 2]
        xmin, ymin, zmin, xmax, ymax, zmax = self.config.rangeFilter
        mask = (x >= xmin) * (x <= xmax) * (y >= ymin) * (y <= ymax) * (z >= zmin) * (z <= zmax)
        return gtBoxes[mask], gtLabels[mask], gtVelocity[mask], gtViewIdx[mask]

    def _filterPred(self, predBoxes: npt.NDArray[np.float64], predLabels: npt.NDArray[np.float64], predVelocity: npt.NDArray[np.float64], predViewIdx: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64]):
        x, y, z = predBoxes[:, 0], predBoxes[:, 1], predBoxes[:, 2]
        xmin, ymin, zmin, xmax, ymax, zmax = self.config.rangeFilter
        mask = (x >= xmin) * (x <= xmax) * (y >= ymin) * (y <= ymax) * (z >= zmin) * (z <= zmax)
        mask *= predScores >= self.config.scoreFilter
        # just filter labels that out of range
        mask *= (predLabels < len(self.config.classes)) * (predLabels >= 0)
        return predBoxes[mask], predLabels[mask], predVelocity[mask], predViewIdx[mask], predScores[mask]

    def collectResult(self, clip: Scene, prediction: list[Prediction]) -> ClassWiseDetectionResult:
        self._checkCorrespondence(clip, prediction)

        task = self.progress.add_task('Collecting groundtruth', total=len(clip))

        # NOTE: need to join all views' boxes to a single one for evaluation
        gtBoxes, gtLabels, gtVelocity, gtViewIdx = list(), list(), list(), list()
        for i, views in enumerate(clip):
            gtBoxes.append(views.boxes3D)
            gtLabels.append(views.labels)
            gtVelocity.append(views.velocities)
            gtViewIdx.append(np.full_like(views.labels, i, dtype=np.int64))
            self.progress.update(task, advance=1)

        self.progress.remove_task(task)

        gtBoxes, gtLabels, gtVelocity, gtViewIdx = map(np.concatenate, [gtBoxes, gtLabels, gtVelocity, gtViewIdx])

        task = self.progress.add_task('Collecting prediction', total=len(clip))

        predBoxes, predLabels, predVelocity, predViewIdx, predScores = list(), list(), list(), list(), list()
        for i, pred in enumerate(prediction):
            pred.sortAndPrune(self.config.maxBoxesPerSample)
            predBoxes.append(pred.boxes3D)
            predLabels.append(pred.labels)
            predVelocity.append(pred.velocities)
            predViewIdx.append(np.full_like(pred.labels, i, dtype=np.int64))
            predScores.append(pred.scores)
            self.progress.update(task, advance=1)

        self.progress.remove_task(task)

        predBoxes, predLabels, predVelocity, predViewIdx, predViewIdx, predScores = map(np.concatenate, [predBoxes, predLabels, predVelocity, predViewIdx, predViewIdx, predScores])

        gtBoxes, gtLabels, gtVelocity, gtViewIdx = self._filterGT(gtBoxes, gtLabels, gtVelocity, gtViewIdx)
        predBoxes, predLabels, predVelocity, predViewIdx, predScores = self._filterPred(predBoxes, predLabels, predVelocity, predViewIdx, predScores)

        # finally, sort all predictions by score, descending
        predSortIdx = np.argsort(-predScores)
        predBoxes, predLabels, predVelocity, predViewIdx, predScores = predBoxes[predSortIdx], predLabels[predSortIdx], predVelocity[predSortIdx], predViewIdx[predSortIdx], predScores[predSortIdx]

        def _parallelRun(label, threshold):
            label = self._labelIDMapping[label]
            gtMask = gtLabels == label
            if gtMask.sum() < 1:
                return label, threshold, None
            gtBoxesOfLabel = gtBoxes[gtMask].copy()
            gtVelocityOfLabel = gtVelocity[gtMask].copy()
            gtViewIdxOfLabel = gtViewIdx[gtMask].copy()

            predMask = predLabels == label
            if predMask.sum() < 1:
                return label, threshold, DetectionResult.nothing(0 if threshold != self.config.tpThreshold else len([x for x in self.handlers if not isinstance(x, RanklistHandler)]))
            # already sorted by score
            predBoxesOfLabel = predBoxes[predMask].copy()
            predScoresOfLabel = predScores[predMask].copy()
            predVelocityOfLabel = predVelocity[predMask].copy()
            predViewIdxOfLabel = predViewIdx[predMask].copy()
            return label, threshold, self._labelResult(threshold, gtBoxesOfLabel, gtVelocityOfLabel, gtViewIdxOfLabel, predBoxesOfLabel, predVelocityOfLabel, predViewIdxOfLabel, predScoresOfLabel)

        params = list(product(self.config.classes, self.config.matchingThreshold))
        with progressedJoblib(self.progress, 'Collecting result', total=len(params)):
            dispatcher = Parallel(-1)
            returnedValue = dispatcher(delayed(_parallelRun)(*p) for p in params)

        results = dict()
        for label, threshold, v in returnedValue:
            if v is None:
                results[label] = ThresholdDetectionResult.ignored(label)
                continue
            if label not in results:
                results[label] = dict()
            results[label][threshold] = v

        for label in results:
            if isinstance(results[label], dict):
                results[label] = ThresholdDetectionResult(label, results[label])

        return ClasswiseAggregator(list(handler.name for handler in self.handlers if isinstance(handler, TruePositiveHandler)), {value: key for key, value in self._labelIDMapping.items()})(results)

    def _labelResult(self, threshold: float, gtBoxes: npt.NDArray[np.float64], gtVelocities: npt.NDArray[np.float64], gtViewIdxOfLabel: npt.NDArray[np.float64], predBoxes: npt.NDArray[np.float64], predVelocities: npt.NDArray[np.float64], predViewIdxOfLabel: npt.NDArray[np.float64], predScores: npt.NDArray[np.float64]) -> DetectionResult:
        # total assign result (which prediction is matched with which groundtruth) used for metric
        assignResult = -np.ones([len(predViewIdxOfLabel)], dtype=np.int64)
        allViewsIdx = np.unique(gtViewIdxOfLabel)

        for idx in allViewsIdx:
            filteredGT = gtViewIdxOfLabel == idx
            filteredPred = predViewIdxOfLabel == idx

            # [#num_filtered_gt] maps index of gt[filteredGT] to global
            gtLocalIndexMapping = np.argwhere(filteredGT).squeeze(-1)

            if self.config.thresholdMetric == ThresholdMetric.IOU:
                raise NotImplementedError
                # use pytorch3d calculate this, not tested.
                # [n, N], prediction to gt distances, pair-wise
                _, distance = box3d_overlap(torch.from_numpy(xyzwlhq2corners(predBoxes[filteredPred]).astype(np.float32)), torch.from_numpy(xyzwlhq2corners(gtBoxes[filteredGT]).astype(np.float32)))
            else:
                # NOTE: nuScenes only consider xy for distance computation
                distance = np.sqrt(((predBoxes[filteredPred, None, :2] - gtBoxes[filteredGT, :2]) ** 2).sum(-1))

            # [n, N], pred to gt distance, from nearest to farthest
            sortedDistance = np.sort(distance, -1)
            # [n, N], pred to gt indices, from nearest to farthest
            sortedIdx = np.argsort(distance, -1)
            # [n] int. >0 -> tp with a matching gt idx, -1 -> fp
            # indicating the ith prediction matches the assigned[i] gt
            assigned = _assignPoints(sortedDistance, sortedIdx, threshold)
            # remap to global gt index
            assigned[assigned >= 0] = gtLocalIndexMapping[assigned[assigned >= 0]]
            # fill them back to total assign result
            assignResult[filteredPred] = assigned

        apResult = None
        tpResults = list()
        for handler in self.handlers:
            if threshold != self.config.tpThreshold and not isinstance(handler, RanklistHandler):
                continue
            r = handler(gtBoxes=gtBoxes, gtVelocities=gtVelocities, predBoxes=predBoxes, predVelocities=predVelocities, predScores=predScores, assignResult=assignResult)
            if isinstance(r, TPResult):
                tpResults.append(r)
            elif apResult is not None:
                raise RuntimeError("Duplicated AP result.")
            else:
                apResult = r

        return DetectionResult(apResult, tpResults) # ThresholdDetectionResult(label, {t: v[0] for t, v in result.items()}, {t: v[1] for t, v in result.items()})

# N1 = 1000, N2 = 3000, run 20 times uses 15s.
def _assignPoints(sortedDis, sortedIdx, threshold):
    N1, N2 = sortedDis.shape
    unAssigned = set(list(range(N2)))
    matched = sortedDis <= threshold

    result = -np.ones([N1], dtype=np.int64)
    for i in range(N1):
        # [?]
        matchedPoints = sortedIdx[i][matched[i]]
        if len(matchedPoints) < 1:
            # All of points are not in threshold
            result[i] = -1
            continue
        remaining = np.intersect1d(list(unAssigned), matchedPoints, assume_unique=True)
        if len(remaining) == 0:
            # This row is a duplicate detection that one row before it has assigned its nearest gt.
            result[i] = -1
            continue
        # The nearest, unassigned point index
        remaining = np.in1d(matchedPoints, remaining)
        finded = matchedPoints[remaining][0]
        unAssigned.remove(finded)
        result[i] = finded

        if len(unAssigned) < 1:
            # no points left
            break

    return result
