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
from dataclasses import dataclass, field
from enum import Enum

from roscenes.consts import strLabels

from .handlers import HandlerReg, Handler, RanklistHandler


class ThresholdMetric(Enum):
    CenterDistance = 1
    IOU = 2 # NOTE: not implemented


@dataclass
class DetectionEvaluationConfig:
    classes: list[str]
    matchingThreshold: list[float]
    tpThreshold: float
    thresholdMetric: ThresholdMetric
    maxBoxesPerSample: int
    scoreFilter: float
    rangeFilter: tuple[float, float, float, float, float, float]
    handlers: list[str]

    _handlerInstances: list[Handler] = field(init=False)

    def __post_init__(self):
        # Check parameters are valid.
        if len(self.handlers) < 1:
            raise ValueError(f'None of metrics are given. Availabe metrics: {HandlerReg.summary()}.')
        if any(x <= 0. for x in self.matchingThreshold):
            raise ValueError('The given matching threshold has one threshold lower than 0.')
        if self.tpThreshold not in self.matchingThreshold:
            raise KeyError("The given tp theshold should be one of matching threshold.")
        if not 0.0 <= self.scoreFilter < 1:
            raise ValueError("The givem score based prediction filter should be in [0, 1).")
        if len (self.rangeFilter) != 6 or not (self.rangeFilter[0] < self.rangeFilter[3]) and (self.rangeFilter[1] < self.rangeFilter[4]) and (self.rangeFilter[2] < self.rangeFilter[5]):
            raise ValueError("The given range based detection filter has wrong bound (length should be 6, min should < max).")
        if not self.maxBoxesPerSample > 0:
            raise ValueError("The given maximum predicted boxes filter should be larger than 0.")

        self._handlerInstances = [HandlerReg.get(x)() for x in self.handlers]
        if not any(isinstance(x, RanklistHandler) for x in self._handlerInstances):
            raise ValueError('You must provide at least one ranklist-based metric (for example, `PrecisionRecall`).')


defaultEvaluationConfig = DetectionEvaluationConfig(
    classes=list(strLabels.keys()),
    matchingThreshold=[0.5, 1., 2., 4.],
    tpThreshold=2.,
    thresholdMetric=ThresholdMetric.CenterDistance,
    maxBoxesPerSample=500,
    scoreFilter=0.0,
    rangeFilter=[-400., -40., 0., 400., 40., 6.],
    handlers=[
        'PrecisionRecall',
        'TranslationError',
        'ScaleError',
        'OrientationError',
        'VelocityError' # TBD.
    ]
)