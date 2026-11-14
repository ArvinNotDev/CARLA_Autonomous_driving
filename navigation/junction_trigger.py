from __future__ import annotations

from typing import Optional


class JunctionTriggerLatch:
    """Keep a junction observation pending until the stop sequence accepts it.

    A detector can see a crosswalk before the route planner's entry-distance
    threshold is reached. The observation therefore remains pending while the
    evidence is still present, allowing the control loop to retry planning.
    After a successful plan, the latch rearms only once the evidence has been
    absent for the configured clear interval and the junction manager is idle.
    """

    def __init__(self, clear_duration: float = 1.0) -> None:
        self.clear_duration = max(0.0, float(clear_duration))
        self.armed = True
        self._pending = False
        self._clear_since: Optional[float] = None

    @property
    def pending(self) -> bool:
        return self._pending

    def observe(
        self,
        *,
        detected: bool,
        evidence_present: bool,
        timestamp: float,
        manager_idle: bool,
    ) -> bool:
        try:
            observed_at = float(timestamp)
        except (TypeError, ValueError):
            return self._pending

        if evidence_present:
            self._clear_since = None
        elif self._clear_since is None:
            self._clear_since = observed_at

        if (
            not evidence_present
            and manager_idle
            and self._clear_since is not None
            and observed_at - self._clear_since >= self.clear_duration
        ):
            self.armed = True
            self._pending = False
            self._clear_since = None

        if detected and evidence_present and self.armed:
            self._pending = True

        return self._pending

    def acknowledge(self) -> None:
        """Consume the event only after the manager has accepted a real plan."""
        self._pending = False
        self.armed = False
        self._clear_since = None
