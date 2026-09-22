from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_applications_endpoint() -> None:
    response = client.get("/applications")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_application_by_id() -> None:
    response = client.get("/applications/APP001")
    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "APP001" or payload["application_id"] == "APP001"


def test_recommendation_endpoint() -> None:
    response = client.post("/recommendation", json={"application_id": "APP001"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_id"] == "APP001" or payload["application_id"] == "APP001"
    assert payload["recommendation"] in {
        "Rehost",
        "Replatform",
        "Repurchase",
        "Refactor",
        "Retire",
        "Retain",
    }
    assert 0 <= float(payload["confidence"]) <= 1


def test_validation_errors() -> None:
    response = client.post("/recommendation", json={})
    assert response.status_code == 422


def test_missing_application() -> None:
    response = client.post("/recommendation", json={"application_id": "APP2000"})
    assert response.status_code == 404


def test_service_error() -> None:
    response = client.post("/cost-risk", json={"application_id": "APP2000"})
    assert response.status_code == 404


def test_swagger_and_openapi() -> None:
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200
    assert "Intelligent Cloud Migration Planning API" in openapi_response.json()["info"]["title"]
