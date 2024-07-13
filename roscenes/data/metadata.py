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
from datetime import datetime
from enum import Enum

class _Enum(Enum):
    def __str__(self):
        return self.name

class Difficulty(_Enum):
    unknown = 0
    easy = 1
    hard = 2
    mixed = 3

class Ambience(_Enum):
    unknown = 0
    day = 1
    night = 2
    mixed = 3

class Weather(_Enum):
    unknown = 0
    clear = 1
    dirty = 2
    mixed = 3

class Split(_Enum):
    unknown = 0
    train = 1
    validation = 2
    test = 3

@dataclass
class Metadata:
    difficulty: Difficulty
    ambience: Ambience
    # weather: Weather
    split: Split
    creation: datetime
    additional: str
