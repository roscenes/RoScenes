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
import logging
from collections import OrderedDict
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TimeRemainingColumn, MofNCompleteColumn


richProgress = Progress("[i blue]{task.description}[/]", MofNCompleteColumn(), TimeElapsedColumn(), BarColumn(None), TimeRemainingColumn(), refresh_per_second=6, transient=True, expand=True)

logger = logging.getLogger('roscenes')

strLabels = OrderedDict(
    [
        (0, 'other'),
        (1, 'truck'),
        (2, 'bus'),
        (3, 'van'),
        (4, 'car')
    ]
)

intLabels = OrderedDict(
    [
        ('other', 0),
        ('truck', 1),
        ('bus', 2),
        ('van', 3),
        ('car', 4)
    ]
)