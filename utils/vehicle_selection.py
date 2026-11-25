from __future__ import annotations

from collections.abc import Iterable


def vehicle_spawn_candidates(
    available_vehicles: Iterable[tuple[str, str]],
    preferred_id: str | None,
    fallback_id: str = "vehicle.tesla.model3",
) -> list[str]:
    """Prioritize the last-used vehicle, then Tesla, then remaining options."""
    available_ids = list(dict.fromkeys(
        str(item[0]) for item in available_vehicles
        if isinstance(item, (tuple, list)) and item and item[0]
    ))
    available = set(available_ids)
    candidates: list[str] = []
    for candidate in (preferred_id, fallback_id, *available_ids):
        if candidate and candidate in available and candidate not in candidates:
            candidates.append(candidate)
    return candidates
