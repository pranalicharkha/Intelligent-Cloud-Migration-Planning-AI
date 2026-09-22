from __future__ import annotations

from fastapi import HTTPException, status

from .data_loader import get_application_by_id
from .models import CostRiskResponse


def get_cost_risk(application_id: str) -> CostRiskResponse:
    try:
        application = get_application_by_id(application_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Application not found: {application_id}",
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Application dataset is missing or unreadable.",
        ) from exc

    criticality_factor = {"Low": 1.0, "Medium": 1.4, "High": 1.8}[application.criticality]
    monthly_cost = (
        (application.cpu_usage or 0) * 280
        + (application.memory_usage or 0) * 210
        + (application.age_years or 0) * 8
        + application.dependency_count * 35
    ) * criticality_factor

    lower = round(monthly_cost * 0.8, 2)
    upper = round(monthly_cost * 1.25, 2)
    risk_score = min(
        100.0,
        round((application.dependency_count * 8) + ((application.age_years or 0) * 1.2) + (criticality_factor * 15), 2),
    )

    return CostRiskResponse(
        app_id=application.id,
        application_id=application.id,
        monthly_aws_cost=round(monthly_cost, 2),
        cost_range={"lower": lower, "upper": upper},
        risk_score=risk_score,
    )
