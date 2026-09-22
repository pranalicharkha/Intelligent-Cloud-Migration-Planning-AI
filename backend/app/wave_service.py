from __future__ import annotations

from collections import defaultdict

from fastapi import HTTPException, status

from .data_loader import get_applications
from .models import MigrationWave, MigrationWavesResponse


def get_migration_waves(application_ids: list[str] | None) -> MigrationWavesResponse:
    try:
        applications = get_applications()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Application dataset is missing or unreadable.",
        ) from exc

    if application_ids is None:
        selected_ids = {application.id for application in applications}
    else:
        selected_ids = {str(item).strip() for item in application_ids if str(item).strip()}

    valid_ids = {application.id for application in applications}
    unknown_ids = sorted(selected_ids - valid_ids)
    if unknown_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Applications not found: {unknown_ids}",
        )

    app_map = {application.id: application for application in applications}
    dependency_map: dict[str, list[str]] = {app_id: [] for app_id in selected_ids}
    for app_id in selected_ids:
        app = app_map[app_id]
        dependency_map[app_id] = [dep for dep in app.dependencies if dep in selected_ids]

    wave_applications: dict[int, list[str]] = defaultdict(list)
    seen: set[str] = set()
    wave_number = 1
    remaining = set(selected_ids)

    while remaining:
        ready = sorted(app_id for app_id in remaining if all(dep in seen for dep in dependency_map[app_id]))
        if not ready:
            ready = sorted(remaining)
        wave_applications[wave_number] = ready
        seen.update(ready)
        remaining -= set(ready)
        wave_number += 1

    waves: list[MigrationWave] = []
    for index, app_ids in sorted(wave_applications.items(), key=lambda item: item[0]):
        risk = "Low"
        if len(app_ids) > 3:
            risk = "Medium"
        if any(app_map[app_id].criticality == "High" for app_id in app_ids):
            risk = "High"
        waves.append(MigrationWave(wave=index, applications=app_ids, risk=risk))

    return MigrationWavesResponse(waves=waves)
