from __future__ import annotations


class ContinuousConditionGate:
    """Confirm that a condition has stayed continuously true for a duration."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def reset(self) -> None:
        self._started_at = None

    def update(
        self,
        condition: bool,
        timestamp: float,
        required_seconds: float,
    ) -> bool:
        """Return True only after condition has remained true long enough."""
        if not condition:
            self.reset()
            return False

        if self._started_at is None:
            self._started_at = float(timestamp)

        duration = max(0.0, float(required_seconds))
        return float(timestamp) - self._started_at >= duration
