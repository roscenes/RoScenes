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

from dataclasses import dataclass
import os

import cv2
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from shapely.geometry import Polygon
from pillow_heif import register_heif_opener

register_heif_opener()

from roscenes.typing import StrPath
from roscenes.data.scene import Scene
from roscenes.data.frame import Frame
from roscenes.data.camera import Camera
from roscenes.evaluation.detection.prediction import Prediction

_trackingPalette = (np.float32(sns.color_palette('husl', 32)) * 255).astype(np.uint8).tolist()
_cv2TrackingPalette = (np.float32(sns.color_palette('husl', 32)) * 255).astype(np.uint8)[..., ::-1].tolist()

@dataclass
class VisualizerConfig:
    predictionVisualizeThreshold: float = 0.3
    palette: tuple[tuple[int]] = ( # RGB uint8
        (  0,   0,   0),
        (255,   0,   0), # truck
        (  0,   0, 255), # bus
        (  0, 255,   0), # van
        (255, 255,   0), # car
    )
    cv2Palette: tuple[tuple[int]] = ( # BGR uint8
        (  0,   0,   0),
        (  0,   0, 255), # truck
        (255,   0,   0), # bus
        (0,   255,   0), # van
        (  0, 255, 255), # car
    )
    split: bool = False
    tracking: bool = False
    trackingPalette = _trackingPalette
    cv2TrackingPalette = _cv2TrackingPalette


class Visualizer:
    def __init__(self, scene: Scene, config: VisualizerConfig, prediction: list[Prediction] = None):
        self.scene = scene
        self.config = config
        self.prediction = prediction

    def _plotBox(self, image: StrPath, corners: np.ndarray, label: np.ndarray, instanceID: np.ndarray, shift: int = 4) -> np.ndarray:
        # sub-pixel sampling, pass shift to cv2 functions
        pointMultiplier = 2 ** shift

        # sort by depth
        sortIds = np.argsort(-np.mean(corners[..., -1], -1))
        # [N, 8, 2]
        corners = corners[sortIds, ..., :2]
        label = label[sortIds]
        instanceID = instanceID[sortIds]

        # NOTE: use pillow to load heic file
        img = cv2.cvtColor(np.asarray(Image.open(image)), cv2.COLOR_RGB2BGR)
        cleanImg = img.copy()

        for singleCorner, singleLabel, singleID in zip(corners, label, instanceID):
            # sub-pixel sampling, pass shift to cv2 functions
            singleCorner = singleCorner * pointMultiplier

            # crop the clean object region
            # paste to current image
            # then draw line
            objectPoly = Polygon(singleCorner)
            objectPoly = np.array(objectPoly.convex_hull.exterior.coords, dtype=np.int32)
            mask = np.zeros_like(cleanImg[..., 0])
            cv2.fillPoly(mask, [objectPoly], (255, 255, 255), cv2.LINE_AA, shift)
            fg = cv2.bitwise_and(cleanImg, cleanImg, mask=mask)
            bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
            img = cv2.add(fg, bg)

            lineColor = self.config.cv2Palette[singleLabel] if not self.config.tracking else self.config.cv2TrackingPalette[singleID % len(self.config.cv2TrackingPalette)]

            cv2.polylines(img, [singleCorner[[2, 3, 7, 6]].astype(int)], True, lineColor, 2, cv2.LINE_AA, shift)
            # NOTE: hard-coded heading face
            cv2.polylines(img, [singleCorner[:4].astype(int)], True, lineColor, 2, cv2.LINE_AA, shift)
            cv2.polylines(img, [singleCorner[4:].astype(int)], True, lineColor, 2, cv2.LINE_AA, shift)
            cv2.polylines(img, [singleCorner[[0, 1, 5, 4]].astype(int)], True, [255, 255, 255], 2, cv2.LINE_AA, shift)
        return img

    def _plotBEV(self, box):
        pass


    def _visualize(self, frames: list[Frame], predictions: list[Prediction] = None):
        results = list()
        for frame in frames:
            frameResult = list()
            # sort by camera location and focal length
            for token, camera in sorted(frame.parent.cameras.items(), key=lambda x: (np.linalg.inv(x[1].extrinsic)[0, -1], x[1].intrinsic[0, 0])):
                image = frame.images[token]
                # has GT (not test set)
                if frame.visibleProjected3DBoxes is not None:
                    # [n, 8, 3]
                    visibleBoxes = frame.visibleProjected3DBoxes[token]
                    # [n, 8, 4]
                    imageBoxes = (np.concatenate([visibleBoxes, np.ones_like(visibleBoxes[..., -1:])], -1) @ camera.intrinsic.T)[..., :3]
                    imageBoxes[..., :2] /= imageBoxes[..., 2:]
                    frameResult.append((camera.name, self._plotBox(image, imageBoxes[..., :3], frame.visibleLabels[token], frame.visibleInstanceIDs[token], 4)))
                else:
                    frameResult.append((camera.name, cv2.imread(image)))
            results.append((frame.token, frameResult))
        return results

    def __getitem__(self, idx):
        frames = self.scene[idx]
        if not isinstance(frames, list):
            frames = [frames]
        result = self._visualize(frames, )
        return result


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    tgt = sys.argv[2]
    tracking = bool(int(sys.argv[3]))
    visualizer = Visualizer(Scene.load(path), VisualizerConfig(tracking=tracking))

    print(f'Start to visualize {path}...')

    os.makedirs(tgt)

    i = 0
    while i < len(visualizer):
        result = visualizer[i]
        for frame_idx, frame_content in result:
            os.makedirs(os.path.join(tgt, f'{frame_idx}'))
            for camera_id, img in frame_content:
                cv2.imwrite(os.path.join(tgt, frame_idx, f'{camera_id}.jpg'), img)

        print(f'{i}-th frame visualized, continue? [y/n]: ')
        while True:
            response = input()
            if response.lower() == 'y':
                break
            elif response.lower() == 'n':
                print('bye!')
                exit()
            else:
                print('Please enter y or n: ')
        i += 1