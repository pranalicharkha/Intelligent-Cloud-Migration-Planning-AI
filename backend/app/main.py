from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    Application,
    CopilotRequest,
    CopilotResponse,
    CostRiskRequest,
    CostRiskResponse,
    MigrationWavesRequest,
    MigrationWavesResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from .services import (
    answer_copilot,
    get_applications,
    get_cost_risk,
    get_migration_waves,
    get_recommendation,
)

app = FastAPI(
    title="Intelligent Cloud Migration Planning API",
    version="0.1.0",
    description="Local mock integration API for the migration planning prototype.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/applications", response_model=list[Application])
def applications() -> list[Application]:
    return get_applications()


@app.post("/recommendation", response_model=RecommendationResponse)
def recommendation(request: RecommendationRequest) -> RecommendationResponse:
    return get_recommendation(request.application_id)


@app.post("/migration-waves", response_model=MigrationWavesResponse)
def migration_waves(request: MigrationWavesRequest) -> MigrationWavesResponse:
    return get_migration_waves(request.application_ids)


@app.post("/copilot", response_model=CopilotResponse)
def copilot(request: CopilotRequest) -> CopilotResponse:
    return answer_copilot(request.question)


@app.post("/cost-risk", response_model=CostRiskResponse)
def cost_risk(request: CostRiskRequest) -> CostRiskResponse:
    return get_cost_risk(request.application_id)
