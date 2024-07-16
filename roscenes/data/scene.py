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
import glob
import logging
import itertools
from dataclasses import dataclass, field
from typing import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from roscenes.misc import configLogging
from roscenes.data.metadata import Metadata
from roscenes.data.clip import Clip
from roscenes.data.frame import Frame
from roscenes.consts import logger
from roscenes.typing import StrPath, Indexing

__all__ = [
    "Scene",
    "ConcatScene"
]

@dataclass
class Scene:
    """The top wrapper for collection of clips"""
    name: str
    """The user-friendly name of this scene."""
    metadata: Metadata
    """A brief top-level information of this scene."""
    # used for frame image path generation.
    rootDir: Path
    """The path where this scene locates."""
    clips: list[Clip]
    """All clips this scene has."""

    # used for indexing, [N, 2], N = length of all frames, first is clipIdx, second is offset
    # see __getitem__
    _indexing: npt.NDArray[np.int64]

    def __str__(self) -> str:
        return f'"{self.name}" ({self.metadata.split}): Ambience: `{self.metadata.ambience}`, difficulty: `{self.metadata.difficulty}`, created at {self.metadata.creation.strftime("%Y-%m-%d %H:%M")}'

    def concat(self, another: Scene | Iterable[Scene] | ConcatScene) -> ConcatScene:
        """Concatenate this scene with another scene(s).

        Args:
            another (Scene | Iterable[Scene] | ConcatScene): A single scene or a bunch of scenes.

        Returns:
            ConcatScene: The concatenated scenes.
        """
        if isinstance(another, Scene):
            return ConcatScene([self, another])
        elif isinstance(another, ConcatScene):
            return ConcatScene(list(itertools.chain([self], another.scenes)))
        return ConcatScene(list(itertools.chain([self], another)))

    @staticmethod
    def load(rootDir: StrPath | Iterable[StrPath], disableLogging: bool = False) -> Scene | ConcatScene:
        """Load scene data by given path(s).

        Args:
            rootDir (StrPath | Iterable[StrPath]): A single path, or a bunch of paths. Each path can be a glob string to retrieve directories that contain scene data.
            disableLogging (bool, optional): Whether to disable logging to all returned scene(s). Defaults to False.

        Raises:
            ValueError: If the given rootDir is neither a dir path nor a valid glob.

        Returns:
            Scene | ConcatScene: A scene if a single path is given, or concatenated scenes if glob or paths are given.
        """
        logger = configLogging(logging.ERROR if disableLogging else logging.INFO)
        if isinstance(rootDir, (str, Path)):
            # normal loading
            if os.path.isdir(rootDir):
                databasePath = os.path.join(rootDir, "database")
                with open(os.path.join(databasePath, "scene.pkl"), "rb") as fp:
                    scene: Scene = pickle.load(fp)
                # update dynamic attributes
                scene.rootDir = rootDir
                scene.logger = logger
                for clip in scene.clips:
                    clip.parent = scene
                    clip._postInitialize()
                logger.info('Load %s, %s frames.', scene, len(scene))
                return scene
            # try fetch glob list, then load from list
            elif len(glob.glob(str(rootDir))) > 0:
                rootDir = sorted(glob.glob(str(rootDir)))
            else:
                raise ValueError('The given rootDir is neither a dir path nor a valid glob.')
        # load from list of dirs
        scenes = list()
        for root in rootDir:
            scenes.append(Scene.load(root, disableLogging))
        return ConcatScene(scenes)

    def __getitem__(self, idx: Indexing) -> Frame | list(Frame):
        indices = self._indexing[idx]
        # a single frame
        if len(indices.shape) < 2:
            clipIdx, offset = indices
            return self.clips[clipIdx][offset]
        # slice or iterable indexing returns list of frames
        result = list()
        for clipIdx, offset in indices:
            result.append(self.clips[clipIdx][offset])
        return result

    def __iter__(self):
        for clip in self.clips:
            for v in clip:
                yield v

    def __len__(self):
        return len(self._indexing)

    def __del__(self):
        if not hasattr(self, "clips"):
            return
        for clip in self.clips:
            clip.parent = None
            del clip

@dataclass
class ConcatScene:
    scenes: list[Scene]

    # [N, 3], scene idx, clip idx, offset
    _indexing: np.ndarray = field(init=False)

    def concat(self, another: Scene | Iterable[Scene] | ConcatScene) -> ConcatScene:
        """Concatenate another scene(s) to itself.

        Args:
            another (Scene | Iterable[Scene] | ConcatScene): The given scene(s).

        Returns:
            ConcatScene: A new ConcatScene contains current data plus given new data.
        """
        if isinstance(another, Scene):
            return ConcatScene(self.scenes + [another])
        if isinstance(another, ConcatScene):
            return ConcatScene(self.scenes + another.scenes)
        return ConcatScene(list(itertools.chain(self.scenes, another)))

    def __post_init__(self):
        # [N, 2]
        indexing = np.concatenate([x._indexing for x in self.scenes], 0)
        # [N, 3]
        indexing = np.concatenate([np.concatenate([np.full([len(x), 1], i, dtype=np.int64) for i, x in enumerate(self.scenes)]), indexing], -1)
        self._indexing = indexing
        logger.info('ConcatScene created of total frames: %s.', len(self))

    def __getitem__(self, idx: Indexing) -> Frame | list(Frame):
        # normal indexing
        if isinstance(idx, int):
            sceneIdx, clipIdx, offset = self._indexing[idx]
            return self.scenes[sceneIdx].clips[clipIdx][offset]

        # slice or iterable indexing returns list of frames
        indices = self._indexing[idx]
        result = list()
        for sceneIdx, clipIdx, offset in indices:
            result.append(self.scenes[sceneIdx][clipIdx][offset])
        return result

    def __iter__(self):
        for scene in self.scenes:
            for v in scene:
                yield v

    def __len__(self):
        return len(self._indexing)

    def __del__(self):
        if not hasattr(self, "scenes"):
            return
        for scene in self.scenes:
            del scene
