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
import os
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from dashtable import data2rst

@dataclass
class APResult:
    precision: npt.NDArray[np.float64]
    recall: npt.NDArray[np.float64]
    score: npt.NDArray[np.float64]

@dataclass
class TPResult:
    value: npt.NDArray[np.float64]
    score: npt.NDArray[np.float64]

    def __post_init__(self):
        self.value = np.cumsum(self.value) / (np.arange(len(self.value)) + 1)

@dataclass
class DetectionResult:
    """Result for a single frame."""
    ap: APResult
    tps: list[TPResult]

    @staticmethod
    def nothing(tpLength):
        return DetectionResult(APResult(np.zeros([10]), np.zeros([10]), np.linspace(1.0, 0.0, 10)), [TPResult(np.ones([10]), np.linspace(1.0, 0.0, 10)) for _ in range(tpLength)])

    @property
    def hasTP(self):
        return self.tps is not None and len(self.tps) > 0

    @property
    def apWeight(self):
        return len(self.tps)

    @property
    def result(self):
        precision, recall, score = self.ap.precision.copy(), self.ap.recall.copy(), self.ap.score.copy()
        # check if self is nothing
        if sum(self.ap.recall) < 1e-6:
            # return zero
            if len(self.tps) < 1:
                return 0.0, None
            return 0.0, [1.0 for _ in range(len(self.tps))]

        # remap recall to [0, 1] 101 points.
        recallInterp = np.linspace(0, 1, 101)
        precisionInterp = np.interp(recallInterp, recall, precision, right=0)
        scoreInterp = np.interp(recallInterp, recall, score, right=0)

        # clip recall > 0.1, precision > 0.1
        precisionInterp = precisionInterp[recallInterp > 0.1]
        precisionInterp -= 0.1
        precisionInterp = np.clip(precisionInterp, 0.0, 1.0)
        ap = np.mean(precisionInterp) / (1.0 - 0.1)

        if len(self.tps) < 1:
            return ap, None

        tpResults = list()
        for tp in self.tps:
            value, score = tp.value.copy(), tp.score.copy()
            # make interpolation sequence to be ascending
            valueInterp = np.interp(scoreInterp[::-1], score[::-1], value[::-1])[::-1]
            # clip, recall > 0.1, score > 0
            # this should be left True, right False
            recallMask = recallInterp > 0.1
            # this should be right True, left False
            scoreMask = scoreInterp > 0
            # add one more True to the right-most mask
            rightMost = min(len(scoreMask) - 1, np.argwhere(scoreMask)[-1][0] + 1)
            scoreMask[rightMost] = True
            valueInterp = valueInterp[recallMask * scoreMask]
            # NOTE: replcace nan tp result to one (max error).
            tpResults.append(np.nan_to_num(np.mean(valueInterp), nan=1., posinf=1., neginf=1.))
        return ap, tpResults

@dataclass
class ThresholdDetectionResult:
    label: int
    results: dict[float, DetectionResult]
    isIgnored: bool = False
    apWeight: float = field(init=False)

    def __post_init__(self):
        if self.isIgnored:
            self.apWeight = np.nan
            return
        apWeights = [r.apWeight for r in self.results.values() if r.hasTP]
        if not np.allclose(apWeights, np.mean(apWeights)):
            raise RuntimeError
        self.apWeight = apWeights[0]

    def ap(self, threshold: float):
        if self.isIgnored:
            raise RuntimeError("Ignored `ThresholdDetectionResult` does not have value.")
        return self.results[threshold].result[0]

    def mAP(self):
        if self.isIgnored:
            raise RuntimeError("Ignored `ThresholdDetectionResult` does not have value.")
        return np.mean(list(x.result[0] for x in self.results.values()))

    def tps(self):
        if self.isIgnored:
            raise RuntimeError("Ignored `ThresholdDetectionResult` does not have value.")
        res = [x.result for x in self.results.values()]
        tps = [r[1] for r in res if r[1] is not None]
        tps = np.clip(np.mean(np.float64(tps), 0), 0, 1)
        return tps

    @property
    def result(self) -> tuple[float, np.ndarray]:
        if self.isIgnored:
            raise RuntimeError("Ignored `ThresholdDetectionResult` does not have value.")
        res = [x.result for x in self.results.values()]
        aps = [r[0] for r in res if r[0] is not None]
        tps = [r[1] for r in res if r[1] is not None]
        mAP = np.mean(aps)
        # [#num_tp]
        tps = np.mean(np.float64(tps), 0)
        return mAP, tps

    @staticmethod
    def ignored(label: int):
        return ThresholdDetectionResult(label, None, True)

@dataclass
class ClassWiseDetectionResult:
    values: dict[str, tuple[float, npt.NDArray[np.float64]]]
    apWeight: float
    raw: dict[str, ThresholdDetectionResult]
    tpNames: list[str]

    def __str__(self):
        table = list()
        header = list()
        header.append("Class")
        header.append("NDS")
        header.append("mAP")
        for tpName in self.tpNames:
            header.append("m" + tpName)
        table.append(header)

        # First table
        table.append(["All", f"{self.result:.4f}", f"{self.mAP:.4f}"] + list(f"{x:.4f}" for x in self.allTP))

        result = data2rst(table, use_headers=True, center_cells=True, center_headers=True)

        table = list()
        header = list()
        subHeader = list()
        spans = list()
        header.append("Class")
        subHeader.append("")
        spans.append([[0, 0], [1, 0]])
        header.append("NDS")
        subHeader.append("")
        spans.append([[0, 1], [1, 1]])
        header.append("AP")

        dists = (list(x for x in self.raw.values() if not x.isIgnored)[0].results.keys())

        spans.append([[0, x] for x in range(len(header) - 1, len(header) + len(dists) - 1)])
        for d in dists:
            header.append("")
            subHeader.append(f"{d:.1f}m")
        header.pop()
        for tpName in self.tpNames:
            header.append(tpName)
            subHeader.append("")
            spans.append([[0, len(header) - 1], [1, len(header) - 1]])

        table.append(header)
        table.append(subHeader)

        for name in self.raw.keys():
            raw = self.raw[name]
            if raw.isIgnored:
                table.append([name, "Ignored"] + [""] * (len(self.tpNames) + len(dists)))
                spans.append([[len(table) - 1, x] for x in range(1, len(self.tpNames) + len(dists) + 1 + 1)])
                continue
            row = [name]
            tps = raw.tps()
            nds = (raw.mAP() * raw.apWeight + np.sum(1 - tps)) / (2 * raw.apWeight)
            row.append(f"{nds:.4f}")
            for dist in self.raw[name].results.keys():
                row.append(f"{raw.ap(dist):.4f}")
            for i, tpName in enumerate(self.tpNames):
                row.append(f"{tps[i]:.4f}")
            table.append(row)

        result += os.linesep
        result += data2rst(table, spans=spans, use_headers=True, center_cells=True, center_headers=True)
        return result


    @property
    def mAP(self):
        # [n_class]
        aps = list(v[0] for v in self.values.values())
        return np.mean(aps)

    @property
    def allTP(self):
        # [n_class, n_tp]
        tps = np.stack(list(v[1] for v in self.values.values()))
        return np.clip(np.mean(tps, 0), 0, 1)

    @property
    def result(self) -> float:
        # [n_class]
        aps = list(v[0] for v in self.values.values())
        # [n_class, n_tp]
        tps = np.stack(list(v[1] for v in self.values.values()))
        mAP, sumTP = np.mean(aps), np.sum(1 - np.clip(np.mean(tps, 0), 0, 1))
        return ((mAP * self.apWeight) + sumTP) / (2 * self.apWeight)

    @property
    def summary(self) -> dict[str, float]:
        detail = dict()

        metric_prefix = f'RoScenes'

        for name in self.values.keys():
            raw = self.raw[name]
            if raw.isIgnored:
                continue
            for dist in self.raw[name].results.keys():
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, dist)] = f"{raw.ap(dist):.4f}"
            tps = raw.tps()
            for i, tpName in enumerate(self.tpNames):
                detail['{}/{}_{}'.format(metric_prefix, name, tpName)] = f"{tps[i]:.4f}"
        # [#tp]
        allTPs = self.allTP
        for i, tpName in enumerate(self.tpNames):
            detail['{}/m{}'.format(metric_prefix, tpName)] = allTPs[i]

        detail['{}/NDS'.format(metric_prefix)] = self.result
        detail['{}/mAP'.format(metric_prefix)] = self.mAP
        return detail

