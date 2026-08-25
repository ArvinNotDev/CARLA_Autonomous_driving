from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Optional

import numpy as np

from runtime_metrics import InferenceMetrics


@dataclass(frozen=True)
class TrajectoryResult:
    request_id: int
    source_frame_id: int
    command: str
    submitted_at: float
    completed_at: float
    steer: Optional[float]
    prediction: Optional[np.ndarray]
    valid: bool
    error: Optional[str] = None

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.completed_at)


@dataclass(frozen=True)
class VisionResult:
    request_id: int
    source_frame_id: int
    submitted_at: float
    completed_at: float
    result: Optional[dict[str, Any]]
    valid: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class IntersectionResult:
    request_id: int
    source_frame_id: int
    submitted_at: float
    completed_at: float
    is_intersection: bool
    valid: bool
    error: Optional[str] = None


@dataclass
class _Task:
    request_id: int
    source_frame_id: int
    submitted_at: float
    kind: str
    frame_a: Optional[np.ndarray]
    frame_b: Optional[np.ndarray]
    command: Optional[str]


class InferenceWorker:
    """
    Single model-execution thread.

    Vision, trajectory, and intersection inference are serialized on purpose.
    Each task type keeps only its newest pending request. This prevents model
    contention and eliminates stale queued frames from building up.
    """

    def __init__(
        self,
        *,
        vision_processor=None,
        trajectory_agent=None,
        intersection_model=None,
    ) -> None:
        self.vision_processor = vision_processor
        self.trajectory_agent = trajectory_agent
        self.intersection_model = intersection_model

        self._condition = threading.Condition()
        self._pending: dict[str, Optional[_Task]] = {
            "vision": None,
            "trajectory": None,
            "intersection": None,
        }
        self._next_request_id = 0
        self._latest_submitted = {"vision": 0, "trajectory": 0, "intersection": 0}
        self._latest_completed = {"vision": 0, "trajectory": 0, "intersection": 0}
        self._latest_trajectory: Optional[TrajectoryResult] = None
        self._latest_vision: Optional[VisionResult] = None
        self._latest_intersection: Optional[IntersectionResult] = None
        self._busy_kind: Optional[str] = None
        self._last_error: dict[str, Optional[str]] = {
            "vision": None,
            "trajectory": None,
            "intersection": None,
        }
        self._stop = False
        self._started = False

        self.metrics = {
            "vision": InferenceMetrics(),
            "trajectory": InferenceMetrics(),
            "intersection": InferenceMetrics(),
        }
        self._thread = threading.Thread(
            target=self._run,
            name="carla-inference-worker",
            daemon=True,
        )

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._thread.start()

    def _submit(
        self,
        kind: str,
        *,
        source_frame_id: int,
        frame_a: Optional[np.ndarray],
        frame_b: Optional[np.ndarray],
        command: Optional[str] = None,
    ) -> int:
        with self._condition:
            if self._stop:
                return 0
            self._next_request_id += 1
            request_id = self._next_request_id
            task = _Task(
                request_id=request_id,
                source_frame_id=int(source_frame_id),
                submitted_at=time.perf_counter(),
                kind=kind,
                frame_a=frame_a,
                frame_b=frame_b,
                command=command,
            )
            self._pending[kind] = task
            self._latest_submitted[kind] = request_id
            self._condition.notify()
            return request_id

    def submit_vision(
        self,
        *,
        source_frame_id: int,
        semantic_frame: Optional[np.ndarray],
        rgb_frame: Optional[np.ndarray],
    ) -> int:
        return self._submit(
            "vision",
            source_frame_id=source_frame_id,
            frame_a=semantic_frame,
            frame_b=rgb_frame,
        )

    def submit_trajectory(
        self,
        *,
        source_frame_id: int,
        bgr_frame: np.ndarray,
        command: str,
    ) -> int:
        return self._submit(
            "trajectory",
            source_frame_id=source_frame_id,
            frame_a=bgr_frame,
            frame_b=None,
            command=command,
        )

    def submit_intersection(
        self,
        *,
        source_frame_id: int,
        frame_bgr: np.ndarray,
    ) -> int:
        return self._submit(
            "intersection",
            source_frame_id=source_frame_id,
            frame_a=frame_bgr,
            frame_b=None,
        )

    def _take_task(self) -> Optional[_Task]:
        # Alternate tasks when multiple types are pending so one model cannot
        # starve the others. Preference still follows oldest submission time.
        pending = [task for task in self._pending.values() if task is not None]
        if not pending:
            return None
        task = min(pending, key=lambda item: item.submitted_at)
        self._pending[task.kind] = None
        self._busy_kind = task.kind
        return task

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._stop and all(
                    task is None for task in self._pending.values()
                ):
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                task = self._take_task()

            if task is None:
                continue

            started = time.perf_counter()
            try:
                if task.kind == "trajectory":
                    self._run_trajectory(task)
                elif task.kind == "vision":
                    self._run_vision(task)
                elif task.kind == "intersection":
                    self._run_intersection(task)
            finally:
                duration = time.perf_counter() - started
                self.metrics[task.kind].record(duration)
                with self._condition:
                    self._busy_kind = None

    def _run_trajectory(self, task: _Task) -> None:
        completed = time.perf_counter()
        command = str(task.command or "LANE_FOLLOW")
        try:
            if self.trajectory_agent is None or task.frame_a is None:
                raise RuntimeError("Trajectory model is unavailable")

            # Mutable agent state is accessed only from this worker.
            if getattr(self.trajectory_agent, "_worker_command", None) != command:
                self.trajectory_agent.prev_steer = 0.0
                self.trajectory_agent._worker_command = command

            # CameraManager exposes OpenCV BGR frames. Convert exactly once here.
            frame_bgr = task.frame_a
            steer, pred = self.trajectory_agent.get_steering_and_pred(
                frame_bgr,
                command_name=command,
            )
            pred = np.asarray(pred, dtype=np.float32).copy()
            pred.setflags(write=False)
            completed = time.perf_counter()
            result = TrajectoryResult(
                request_id=task.request_id,
                source_frame_id=task.source_frame_id,
                command=command,
                submitted_at=task.submitted_at,
                completed_at=completed,
                steer=float(steer),
                prediction=pred,
                valid=True,
            )
            with self._condition:
                # Do not let an older task overwrite a result already completed
                # for a newer request.
                if task.request_id >= self._latest_completed["trajectory"]:
                    self._latest_completed["trajectory"] = task.request_id
                    self._latest_trajectory = result
                    self._last_error["trajectory"] = None
        except Exception as exc:
            completed = time.perf_counter()
            with self._condition:
                self._last_error["trajectory"] = str(exc)
                # Deliberately retain the previous valid result on failure.

    def _run_vision(self, task: _Task) -> None:
        completed = time.perf_counter()
        try:
            if self.vision_processor is None:
                raise RuntimeError("Vision model is unavailable")

            result = self.vision_processor.detect(task.frame_a, task.frame_b)
            if not isinstance(result, dict):
                raise TypeError("Vision processor returned a non-dict result")

            result = dict(result)
            result["_source_frame_id"] = task.source_frame_id
            result["_completed_at"] = time.perf_counter()
            completed = result["_completed_at"]

            envelope = VisionResult(
                request_id=task.request_id,
                source_frame_id=task.source_frame_id,
                submitted_at=task.submitted_at,
                completed_at=completed,
                result=result,
                valid=True,
            )
            with self._condition:
                if task.request_id >= self._latest_completed["vision"]:
                    self._latest_completed["vision"] = task.request_id
                    self._latest_vision = envelope
                    self._last_error["vision"] = None
        except Exception as exc:
            with self._condition:
                self._last_error["vision"] = str(exc)
                # Retain last valid vision result.

    def _run_intersection(self, task: _Task) -> None:
        try:
            if self.intersection_model is None or task.frame_a is None:
                raise RuntimeError("Intersection model is unavailable")
            value = bool(self.intersection_model.is_intersection_ahead(task.frame_a))
            completed = time.perf_counter()
            result = IntersectionResult(
                request_id=task.request_id,
                source_frame_id=task.source_frame_id,
                submitted_at=task.submitted_at,
                completed_at=completed,
                is_intersection=value,
                valid=True,
            )
            with self._condition:
                if task.request_id >= self._latest_completed["intersection"]:
                    self._latest_completed["intersection"] = task.request_id
                    self._latest_intersection = result
                    self._last_error["intersection"] = None
        except Exception as exc:
            with self._condition:
                self._last_error["intersection"] = str(exc)
                # Retain last valid intersection result.

    def trajectory_snapshot(self) -> tuple[Optional[TrajectoryResult], bool, Optional[str], int]:
        with self._condition:
            return (
                self._latest_trajectory,
                self._busy_kind == "trajectory" or self._pending["trajectory"] is not None,
                self._last_error["trajectory"],
                self._latest_submitted["trajectory"],
            )

    def vision_snapshot(self) -> tuple[Optional[VisionResult], bool, Optional[str], int]:
        with self._condition:
            return (
                self._latest_vision,
                self._busy_kind == "vision" or self._pending["vision"] is not None,
                self._last_error["vision"],
                self._latest_submitted["vision"],
            )

    def intersection_snapshot(self) -> tuple[Optional[IntersectionResult], bool, Optional[str], int]:
        with self._condition:
            return (
                self._latest_intersection,
                self._busy_kind == "intersection" or self._pending["intersection"] is not None,
                self._last_error["intersection"],
                self._latest_submitted["intersection"],
            )

    def stop(self, join_timeout: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            for key in self._pending:
                self._pending[key] = None
            self._condition.notify_all()
        if self._started and self._thread.is_alive():
            self._thread.join(timeout=max(0.0, float(join_timeout)))
