"""
Script to benchmark and test the various functions in one neat package. Mostly for running on a pi.
"""

from main import IronAcer
from main import LoadWebcam

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import time
from typing import Any, Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer(ContextDecorator):

    timers: ClassVar[Dict[str, list]] = dict()
    name: Optional[str] = None
    text: str = "Time Elapsed: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, [])

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name].append(elapsed_time)
        return elapsed_time

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class IronTimer(IronAcer):
    def __init__(self, inference=False, motion_detector=True):
        super(IronTimer, self).__init__(inference=inference, motion_detection=motion_detector, surveillance_mode=True)

    @Timer('Inference', text='Inference: {:0.4f} seconds')
    def time_inference(self, frame):
        self.inferencer(frame)

    @Timer('Motion', text='Motion: {:0.4f} seconds')
    def time_motion_detector(self, frame):
        self.motion_detectoriser(frame)

    def main(self):
        i = 0
        with LoadWebcam(pipe=self.source, output_img_size=self.imgsz) as stream:
            for frame in stream:
                if self.motion_detection:
                    self.time_motion_detector(frame)
                if self.inference:
                    self.time_inference(frame)
                i += 1
                if i == 10:
                    break
            print(Timer.timers)

if __name__ == '__main__':
    I = IronTimer(inference=True)
    I.main()

