from fastapi.testclient import TestClient

from src.realtime_ai_platform.api.main import app


def test_health_reports_model_state():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert "model_exists" in response.json()
