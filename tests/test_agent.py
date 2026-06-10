from pathlib import Path
from app.config import Settings
from app.agent import HealthcareAssistantAgent


def test_agent_mock_ingest_and_answer(tmp_path: Path) -> None:
    # Setup temporary directories for testing
    kb_dir = tmp_path / "data"
    kb_dir.mkdir()
    db_dir = tmp_path / "vector_store"

    # Write a test medical document
    doc_path = kb_dir / "asthma.txt"
    doc_path.write_text(
        "Asthma is a chronic disease that affects your airways. "
        "Symptoms include coughing, wheezing, and shortness of breath.",
        encoding="utf-8",
    )

    settings = Settings(
        use_mock=True,
        knowledge_base_dir=kb_dir,
        vector_db_dir=db_dir,
        collection_name="test_agent_collection",
        openai_api_key=None,
    )

    agent = HealthcareAssistantAgent(settings)

    # 1. Test Ingestion
    docs_loaded, chunks_indexed = agent.ingest()
    assert docs_loaded == 1
    assert chunks_indexed > 0

    # 2. Test Ask Relevant Question (Should match and cite)
    response = agent.answer_question("What is asthma?")
    assert "chronic disease" in response["answer"]
    assert len(response["sources"]) > 0
    assert response["sources"][0]["document"] is not None
    assert response["sources"][0]["chunk"] is not None
    assert response["confidence"] in ["high", "medium", "low"]
    assert len(response["steps_executed"]) >= 1
    assert response["steps_executed"][0]["action"] == "retrieve_healthcare_context"

    # 3. Test Ask Irrelevant Question (Should refuse to answer)
    refusal_response = agent.answer_question("What is the capital of France?")
    assert "I could not find this information in the provided documents." in refusal_response["answer"]
    assert len(refusal_response["sources"]) == 0
    assert refusal_response["confidence"] == "none"

    # 4. Test Ask Booking Query (Should route to check_available_slots)
    booking_response = agent.answer_question("Can I book a cardiology appointment for Monday?")
    assert "mock appointment availability" in booking_response["answer"]
    assert "Cardiology" in booking_response["answer"]
    assert "Monday" in booking_response["answer"]
    assert len(booking_response["sources"]) == 0
    assert booking_response["confidence"] == "high"
    assert any(step["action"] == "check_available_slots" for step in booking_response["steps_executed"])
