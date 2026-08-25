from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DebugFrame:
    sequence: int
    captured_at: float
    image: np.ndarray


class DebugFrameStore:
    """Single-producer/latest-frame store for the Qt debug panel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._frame: Optional[DebugFrame] = None

    def publish(self, image: np.ndarray, captured_at: Optional[float] = None) -> int:
        if image is None:
            return 0
        now = time.perf_counter() if captured_at is None else float(captured_at)
        # The control thread must never mutate a published frame afterwards.
        image.setflags(write=False)
        with self._lock:
            self._sequence += 1
            self._frame = DebugFrame(self._sequence, now, image)
            return self._sequence

    def latest(self) -> Optional[DebugFrame]:
        with self._lock:
            return self._frame
