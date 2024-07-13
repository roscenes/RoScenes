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
####################################################
####### https://github.com/VL-Group/vlutils ########
####################################################
from __future__ import annotations

import re
import os
import inspect
import functools
import abc
import sys
import logging
import logging.config
import multiprocessing
import time
import contextlib
from io import StringIO
from typing import Optional, ClassVar, Dict, Generic, TypeVar

import joblib
import yaml
import rich.logging
from rich.console import ConsoleRenderable
from rich.text import Text
from rich.progress import Progress

from roscenes.consts import logger


T = TypeVar("T")


__all__ = [
    'WaitingBar',
    'LoggingDisabler',
    'configLogging',
    'readableSize',
    'DecoratorContextManager',
    'Registry'
]


# https://github.com/pytorch/pytorch/blob/671ee71ad4b6f507218d1cad278a8e743780b716/torch/autograd/grad_mode.py#L16
class DecoratorContextManager(abc.ABC):
    """Allow a context manager to be used as a decorator

    Example:
    ```python
        class Foo(DecoratorContextManager):
            ...

            def __enter__(self):
                ...

            def __exit__(self, exc_type, exc_val, exc_tb):
                ...

        # normal usecase
        def add(x, y):
            return a + b

        # Normal usecase
        with Foo():
            add(3, 4)


        # Equivalent
        @Foo()
        def addD(x, y):
            return a + b

        addD(3, 4)
    ```
    """

    def __call__(self, func):
        if inspect.isgeneratorfunction(func):
            return self._wrap_generator(func)

        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorate_context

    def _wrap_generator(self, func):
        """Wrap each generator invocation with the context manager"""
        @functools.wraps(func)
        def generator_context(*args, **kwargs):
            gen = func(*args, **kwargs)
            while True:
                try:
                    with self:
                        x = next(gen)
                    yield x
                except StopIteration:
                    break
        return generator_context

    @abc.abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError


def readableSize(size: int, floating: int = 2, binary: bool = True) -> str:
    """Convert bytes to human-readable string (like `-h` option in POSIX).

    Args:
        size (int): Total bytes.
        floating (int, optional): Floating point length. Defaults to 2.
        binary (bool, optional): Format as X or Xi. Defaults to True.

    Returns:
        str: Human-readable string of size.
    """
    size = float(size)
    unit = "B"
    if binary:
        for unit in ['', 'ki', 'Mi', 'Gi', 'Ti', 'PiB']:
            if size < 1024.0 or unit == 'Pi':
                break
            size /= 1024.0
        return f"{size:.{floating}f}{unit}"
    for unit in ['', 'k', 'M', 'G', 'T', 'P']:
        if size < 1000.0 or unit == 'P':
            break
        size /= 1000.0
    return f"{size:.{floating}f}{unit}"


class WaitingBar(DecoratorContextManager):
    """A CLI tool for printing waiting bar.

    Example:
    ```python
        @WaitingBar("msg")
        def longTime():
            # Long time operation
            ...

        with WaitingBar("msg"):
            # Long time operation
            ...
    ```

    Args:
        msg (str): Addtional message shows after bar.
        ncols (int): Total columns of bar.
    """
    def __init__(self, msg: str, ncols: int = 10):
        if ncols <= 8:
            raise ValueError("ncols must greater than 8, got %d", ncols)
        self._msg = msg
        self._ticker = None
        self._stillRunning = None
        self._ncols = ncols
        self.animation = list()
        # "       =       "
        template = (" " * (ncols + 1) + "=" * (ncols - 8) + " " * (ncols + 1))
        for i in range(2 * (ncols - 2)):
            start = 2 * (ncols - 2) - i
            end = 3 * (ncols - 2) - i
            self.animation.append("[" + template[start:end] + "]" + r" %s")

    def __enter__(self):
        self._stillRunning = multiprocessing.Value("b", True)
        self._ticker = multiprocessing.Process(name="waitingBarTicker", target=self._print, args=[self._stillRunning])
        self._ticker.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stillRunning.value = False
        self._ticker.join()
        print(" " * (len(self._msg) + self._ncols + 1), end="\r", file=sys.stderr)

    def _print(self, stillRunning: multiprocessing.Value):
        i = 0
        while bool(stillRunning.value):
            print(self.animation[i % len(self.animation)] % self._msg, end='\r', file=sys.stderr)
            time.sleep(.06)
            i += 1


class LoggingDisabler:
    """Disable or enable logging temporarily.

    Example:
    ```python
        # True -> disable logging, False -> enable logging
        with LoggingDisabler(logger, True):
            # Some operations
            ...
    ```

    Args:
        logger (logging.Logger): The target logger to interpolate.
        disable (bool): Whether to disable logging.
    """
    def __init__(self, logger: logging.Logger, disable: bool):
        self._logger = logger
        self._disable = disable
        self._previous_status = False

    def __enter__(self):
        if self._disable:
            self._previous_status = self._logger.disabled
            self._logger.disabled = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._disable:
            self._logger.disabled = self._previous_status


class KeywordRichHandler(rich.logging.RichHandler):
    KEYWORDS: ClassVar[Optional[list[str]]] = [
        r"(?P<green>\b([gG]ood|[bB]etter|[bB]est|[sS]uccess(|ful|fully))\b)",
        r"(?P<magenta>\b([bB]ase|[cC]all(|s|ed|ing)|[Mm]ount(|s|ed|ing))\b)",
        r"(?P<cyan>\b([mM]aster|nccl|NCCL|[mM]ain|···|[tT]otal|[tT]rain(|s|ed|ing)|[vV]alidate(|s|d)|[vV]alidat(|ing|ion)|[tT]est(|s|ed|ing))\b)",
        r"(?P<yellow>\b([lL]atest|[lL]ast|[sS]tart(|s|ed|ing)|[bB]egin(|s|ning)|[bB]egun|[cC]reate(|s|d|ing)|[gG]et(|s|ting)|[gG]ot|)\b)",
        r"(?P<red>\b([eE]nd(|s|ed|ing)|[fF]inish(|es|ed|ing)|[kK]ill(|s|ed|ing)|[iI]terrupt(|s|ed|ting)|[qQ]uit|QUIT|[eE]xit|EXIT|[bB]ad|[wW]orse|[sS]low(|er))\b)",
        r"(?P<italic>\b([aA]ll|[aA]ny|[nN]one)\b)"
    ]

    def render_message(self, record: logging.LogRecord, message: str) -> ConsoleRenderable:
        use_markup = getattr(record, "markup", self.markup)
        message_text = Text.from_markup(message) if use_markup else Text(message)

        highlighter = getattr(record, "highlighter", self.highlighter)

        if self.KEYWORDS:
            for keyword in self.KEYWORDS:
                message_text.highlight_regex(keyword)
                # message_text.highlight_words(value, key, case_sensitive=False)

        if highlighter:
            message_text = highlighter(message_text)

        return message_text


def configLogging(level: str | int = logging.INFO) -> logging.Logger:
    logging_config = {
        "version": 1,
        "formatters": {
            "full": {
                "format": r"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "roscenes.misc.KeywordRichHandler",
                "level": level,
                "rich_tracebacks": True,
                "tracebacks_show_locals": False,
                "log_time_format": r"%m/%d %H:%M",
                "markup": False,
                "enable_link_path": False
            }
        },
        "loggers": {
            'roscenes': {
                "propagate": True,
                "level": level,
                "handlers": [
                    "console"
                ]
            }
        }
    }
    logging.config.dictConfig(logging_config)
    return logging.getLogger('roscenes')



def _alignYAML(str, pad=0, aligned_colons=False):
    props = re.findall(r'^\s*[\S]+:', str, re.MULTILINE)
    if not props:
        return str
    longest = max([len(i) for i in props]) + pad
    if aligned_colons:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]): \s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))-1-pad) + ':'.ljust(pad+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])
    else:
        return ''.join([i+'\n' for i in map(
                    lambda str: re.sub(r'^(\s*.+?[^:#]: )\s*(.*)',
                        lambda m: m.group(1) + ''.ljust(longest-len(m.group(1))+1) + m.group(2), str, re.MULTILINE),
                    str.split('\n'))])


def pPrint(d: dict) -> str:
    """Print dict prettier.

    Args:
        d (dict): The input dict.

    Returns:
        str: Resulting string.
    """
    with StringIO() as stream:
        yaml.safe_dump(d, stream, default_flow_style=False)
        return _alignYAML(stream.getvalue(), pad=1, aligned_colons=True)

class Registry(Generic[T]):
    """A registry. Inherit from it to create a lots of factories.

    Example:
    ```python
        # Inherit to make a factory.
        class Geometry(Registry):
            ...

        # Register with auto-key "Foo"
        @Geometry.register
        class Foo:
            ...

        # Register with manual-key "Bar"
        @Geometry.register("Bar")
        class Bar:
            ...

        instance = Geometry.get("Foo")()
        assert isinstance(instance, Foo)

        instance = Geometry["Bar"]()
        assert isinstance(instance, Bar)
    ```
    """
    _map: Dict[str, T]
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._map: Dict[str, T] = dict()

    @classmethod
    def register(cls, key):
        """Decorator for register anything into registry.

        Args:
            key (str): The key for registering an object.
        """
        if isinstance(key, str):
            def insert(value):
                cls._map[key] = value
                return value
            return insert
        else:
            cls._map[key.__name__] = key
            return key

    @classmethod
    def get(cls, key: str, logger = logger) -> T:
        """Get an object from registry.

        Args:
            key (str): The key for the registered object.
        """
        result = cls._map.get(key)
        if result is None:
            raise KeyError(f"No entry for {cls.__name__}. Avaliable entries are: {os.linesep + cls.summary()}.")
        elif isinstance(result, functools.partial):
            logger.debug("Get <%s.%s> from \"%s\".", result.func.__module__, result.func.__qualname__, cls.__name__)
        else:
            logger.debug("Get <%s.%s> from \"%s\".", result.__module__, result.__qualname__, cls.__name__)
        return result

    @classmethod
    def values(cls):
        """Get all registered objects."""
        return cls._map.values()

    @classmethod
    def keys(cls):
        """Get all registered keys."""
        return cls._map.keys()

    @classmethod
    def items(cls):
        """Get all registered key-value pairs."""
        return cls._map.items()

    @classmethod
    def summary(cls) -> str:
        """Get registry summary.
        """
        return pPrint({
            k: v.__module__ + "." + v.__name__ for k, v in cls._map.items()
        })


@contextlib.contextmanager
def progressedJoblib(progress: Progress, desc: str, total: int):
    """Context manager to patch joblib to report progress bar update."""
    if not progress.live.is_started:
        raise RuntimeError('You must pass a live progress. This context manager does not handle progress lifecycle.')
    task = progress.add_task(desc, total=total)
    class _batchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            progress.update(task, advance=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _batchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        progress.remove_task(task)
