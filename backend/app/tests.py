"""Small endpoint smoke tests runnable with the standard library only."""

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_endpoints() -> None:
    assert client.get("/applications").status_code == 200
    assert client.post("/recommendation", json={"application_id": "app-001"}).status_code == 200
    assert client.post("/migration-waves", json={}).status_code == 200
    assert client.post("/copilot", json={"question": "What should we migrate first?"}).status_code == 200
    assert client.post("/cost-risk", json={"application_id": "app-001"}).status_code == 200
