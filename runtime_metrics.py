from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Optional


class RollingFps:
    """FPS measured from completed/presented events, never from loop iterations."""

    def __init__(self, window_seconds: float = 1.0) -> None:
        self.window_seconds = max(0.25, float(window_seconds))
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def record(self, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        cutoff = now - self.window_seconds
        with self._lock:
            self._events.append(now)
            while self._events and self._events[0] < cutoff:
                self._events.popleft()

    def value(self, timestamp: Optional[float] = None) -> float:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        cutoff = now - self.window_seconds
        with self._lock:
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            if len(self._events) < 2:
                return 0.0
            span = self._events[-1] - self._events[0]
            if span <= 0.0:
                return 0.0
            return float((len(self._events) - 1) / span)


@dataclass(frozen=True)
class ControlMetricsSnapshot:
    fps: float
    last_tick_at: float
    last_duration_ms: float


class ControlMetrics:
    def __init__(self, window_seconds: float = 1.0) -> None:
        self._fps = RollingFps(window_seconds)
        self._lock = threading.Lock()
        self._last_tick_at = 0.0
        self._last_duration_ms = 0.0

    def record(self, duration_seconds: float, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._fps.record(now)
        with self._lock:
            self._last_tick_at = now
            self._last_duration_ms = max(0.0, float(duration_seconds) * 1000.0)

    def snapshot(self) -> ControlMetricsSnapshot:
        with self._lock:
            last_tick = self._last_tick_at
            last_duration = self._last_duration_ms
        return ControlMetricsSnapshot(
            fps=self._fps.value(),
            last_tick_at=last_tick,
            last_duration_ms=last_duration,
        )


class DisplayMetrics:
    """Counts a frame only when the Qt debug label paints a newly submitted frame."""

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._fps = RollingFps(window_seconds)
        self._lock = threading.Lock()
        self._last_presented_at = 0.0

    def record_presented(self, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._fps.record(now)
        with self._lock:
            self._last_presented_at = now

    def fps(self) -> float:
        return self._fps.value()

    def last_presented_at(self) -> float:
        with self._lock:
            return self._last_presented_at


@dataclass(frozen=True)
class InferenceMetricSnapshot:
    fps: float
    last_completed_at: float
    last_duration_ms: float


class InferenceMetrics:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self._fps = RollingFps(window_seconds)
        self._lock = threading.Lock()
        self._last_completed_at = 0.0
        self._last_duration_ms = 0.0

    def record(self, duration_seconds: float, timestamp: Optional[float] = None) -> None:
        now = time.perf_counter() if timestamp is None else float(timestamp)
        self._fps.record(now)
        with self._lock:
            self._last_completed_at = now
            self._last_duration_ms = max(0.0, float(duration_seconds) * 1000.0)

    def snapshot(self) -> InferenceMetricSnapshot:
        with self._lock:
            completed = self._last_completed_at
            duration = self._last_duration_ms
        return InferenceMetricSnapshot(
            fps=self._fps.value(),
            last_completed_at=completed,
            last_duration_ms=duration,
        )
