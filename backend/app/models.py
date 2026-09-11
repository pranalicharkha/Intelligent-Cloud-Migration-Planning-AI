from typing import Literal

from pydantic import BaseModel, Field


SixRStrategy = Literal[
    "Rehost",
    "Replatform",
    "Repurchase",
    "Refactor",
    "Retire",
    "Retain",
]


class Application(BaseModel):
    id: str
    name: str
    owner: str
    technology: str
    criticality: Literal["Low", "Medium", "High"]
    dependencies: list[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    application_id: str = Field(min_length=1)


class RecommendationResponse(BaseModel):
    application_id: str
    recommendation: SixRStrategy
    confidence: float = Field(ge=0, le=1)
    explanation: str


class MigrationWavesRequest(BaseModel):
    application_ids: list[str] | None = None


class MigrationWave(BaseModel):
    wave: int = Field(ge=1)
    applications: list[str]
    risk: Literal["Low", "Medium", "High"]


class MigrationWavesResponse(BaseModel):
    waves: list[MigrationWave]


class CopilotRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class CopilotResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class CostRiskRequest(BaseModel):
    application_id: str = Field(min_length=1)


class CostRiskResponse(BaseModel):
    application_id: str
    monthly_aws_cost: float = Field(ge=0)
    cost_range: dict[str, float]
    risk_score: float = Field(ge=0, le=100)
