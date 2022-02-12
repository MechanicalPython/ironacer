"""
Script to benchmark and test the various functions in one neat package. Mostly for running on a pi.
"""

from main import IronAcer
from main import LoadWebcam

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import time
from typing import Any, Callable, ClassVar, Dict, Optional


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
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
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

    def total_duration(self):
        total = {}
        for n, l in self.timers.items():
            total.update({n: sum(l)})
            if self.logger:
                self.logger(f'Total time for {n}: {sum(l)}')
        return total

    def average_duration(self):
        avg = {}
        for n, l in self.timers.items():
            avg.update({n: sum(l) / len(l)})
            if self.logger:
                self.logger(f'Mean time for {n}: {sum(l) / len(l)}')
        return avg


class IronTimer(IronAcer):
    def __init__(self, interations=10, inference=False, motion_detector=True):
        super(IronTimer, self).__init__(inference=inference, motion_detection=motion_detector, surveillance_mode=True)
        self.interations =interations

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
                if i == self.interations:
                    break
            Timer().total_duration()
            Timer().average_duration()


if __name__ == '__main__':
    I = IronTimer(inference=True)
    I.main()

