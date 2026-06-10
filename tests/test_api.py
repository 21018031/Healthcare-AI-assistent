import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, settings

# Ensure we are in mock mode for testing
settings.use_mock = True

client = TestClient(app)


def test_api_health_and_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_stats() -> None:
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert data["mock_mode"] is True


def test_api_ingest_and_ask() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        doc_path = Path(tmpdir) / "clinical_guideline.txt"
        doc_path.write_text(
            "Medication errors should be reported within 24 hours to the compliance officer.",
            encoding="utf-8",
        )

        # Ingest
        response = client.post("/ingest", json={"directory": tmpdir})
        assert response.status_code == 200
        data = response.json()
        assert data["documents_loaded"] == 1
        assert data["chunks_indexed"] > 0

        # Ask
        ask_response = client.post(
            "/ask",
            json={"question": "Where should medication errors be reported?", "top_k": 3},
        )
        assert ask_response.status_code == 200
        ask_data = ask_response.json()
        assert "compliance officer" in ask_data["answer"]
        assert len(ask_data["sources"]) > 0
        assert "document" in ask_data["sources"][0]
        assert "chunk" in ask_data["sources"][0]
        assert ask_data["confidence"] in ["high", "medium", "low"]
        assert len(ask_data["steps_executed"]) >= 1

        # Test booking ask via API
        booking_response = client.post(
            "/ask",
            json={"question": "Can I book a pediatrics slot on Tuesday?"}
        )
        assert booking_response.status_code == 200
        booking_data = booking_response.json()
        assert "mock appointment availability" in booking_data["answer"]
        assert "Pediatrics" in booking_data["answer"]
        assert "Tuesday" in booking_data["answer"]
        assert len(booking_data["sources"]) == 0
        assert booking_data["confidence"] == "high"
        assert len(booking_data["steps_executed"]) >= 2
