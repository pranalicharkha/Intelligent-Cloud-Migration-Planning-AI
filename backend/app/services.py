from fastapi import HTTPException, status

from .data import MOCK_APPLICATIONS, MOCK_COSTS, MOCK_RECOMMENDATIONS
from .models import (
    CopilotResponse,
    CostRiskResponse,
    MigrationWavesResponse,
    RecommendationResponse,
)


def _application_ids() -> set[str]:
    return {application.id for application in MOCK_APPLICATIONS}


def get_applications():
    return MOCK_APPLICATIONS


def get_recommendation(application_id: str) -> RecommendationResponse:
    if application_id not in _application_ids():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    recommendation, confidence, explanation = MOCK_RECOMMENDATIONS[application_id]
    return RecommendationResponse(
        application_id=application_id,
        recommendation=recommendation,
        confidence=confidence,
        explanation=explanation,
    )


def get_migration_waves(application_ids: list[str] | None) -> MigrationWavesResponse:
    selected_ids = application_ids or list(_application_ids())
    unknown_ids = set(selected_ids) - _application_ids()
    if unknown_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Applications not found: {sorted(unknown_ids)}",
        )

    selected = set(selected_ids)
    waves = [
        {"wave": 1, "applications": ["app-002"] if "app-002" in selected else [], "risk": "Low"},
        {"wave": 2, "applications": ["app-001", "app-004"] if selected else [], "risk": "Medium"},
        {"wave": 3, "applications": ["app-003"] if "app-003" in selected else [], "risk": "High"},
    ]
    return MigrationWavesResponse(waves=[wave for wave in waves if wave["applications"]])


def answer_copilot(question: str) -> CopilotResponse:
    return CopilotResponse(
        question=question,
        answer="Start with low-risk, loosely coupled applications, then migrate dependent and business-critical services in later waves.",
        sources=["AWS Migration Strategies", "Mock Migration Runbook"],
    )


def get_cost_risk(application_id: str) -> CostRiskResponse:
    if application_id not in _application_ids():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application not found")
    monthly_cost, lower, upper, risk_score = MOCK_COSTS[application_id]
    return CostRiskResponse(
        application_id=application_id,
        monthly_aws_cost=monthly_cost,
        cost_range={"lower": lower, "upper": upper},
        risk_score=risk_score,
    )
