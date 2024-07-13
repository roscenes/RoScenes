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

import numpy as np

from roscenes.evaluation.detection.result import ThresholdDetectionResult, ClassWiseDetectionResult


class ClasswiseAggregator:
    def __init__(self, tpNames: list[str], idLabelMapping: dict[int, str]):
        self.tpNames = tpNames
        self.idLabelMapping = idLabelMapping
    def __call__(self, results: dict[int, ThresholdDetectionResult]) -> ClassWiseDetectionResult:
        apWeights = np.array(list(v.apWeight for v in results.values() if not v.isIgnored))
        if not np.allclose(apWeights, np.mean(apWeights)):
            raise ValueError("ap weights.")
        res = dict()
        for label, result in results.items():
            if result.isIgnored:
                continue
            res[self.idLabelMapping[label]] = result.result
        return ClassWiseDetectionResult(res, result.apWeight, {self.idLabelMapping[key]: value for key, value in results.items()}, self.tpNames)