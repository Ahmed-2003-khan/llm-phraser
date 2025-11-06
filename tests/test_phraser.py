# Purpose: Unit tests for the LLM Phraser (MS 5).

import pytest
from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI app
from app.schemas import PhraserInput
import os

# Use FastAPI's TestClient
# This client handles the app startup/shutdown (lifespan)
client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """
    This fixture automatically runs for every test.
    It mocks the environment variable *before* the app lifespan starts.
    This prevents the TestClient from crashing when it tries to init the Groq client.
    """
    monkeypatch.setenv("GROQ_API_KEY", "test_key_12345")

def test_health_check():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "llm-phraser"}

def test_generate_phrase_success(mocker):
    """
    Tests the /phrase endpoint with a mocked LLM call.
    'mocker' is a fixture from pytest-mock.
    """
    
    # 1. ARRANGE
    
    # This is the fake response we want our mock LLM to return
    mock_response_text = "This is a successful mock counter-offer at $48,000."
    
    # This is the function we want to mock.
    # We patch it *where it is imported* (in app.main)
    mock_llm_call = mocker.patch(
        "app.main.generate_llm_response",  
        return_value=mock_response_text
    )

    # This is the JSON payload we will send
    test_payload = {
        "action": "COUNTER",
        "response_key": "STANDARD_COUNTER",
        "counter_price": 48000.0,
        "policy_type": "rule-based",
        "policy_version": "1.1.0"
    }

    # 2. ACT
    # Send the POST request to the TestClient
    response = client.post("/phrase", json=test_payload)

    # 3. ASSERT
    
    # Check for a successful response
    assert response.status_code == 200
    assert response.json() == {"response_text": mock_response_text}

    # Check that our mock function was called exactly once
    mock_llm_call.assert_called_once()
    
    # Get the arguments it was called with
    call_args = mock_llm_call.call_args[0] # Get positional args
    
    # Check that the first argument was a PhraserInput object
    # This confirms our main.py logic correctly parsed the JSON
    assert isinstance(call_args[0], PhraserInput)
    assert call_args[0].response_key == "STANDARD_COUNTER"
    assert call_args[0].counter_price == 48000.0

def test_generate_phrase_llm_error(mocker):
    """
    Tests how the API behaves if the LLM call fails.
    """
    # 1. ARRANGE
    # This time, we mock the function to raise an exception
    mock_llm_call = mocker.patch(
        "app.main.generate_llm_response",
        side_effect=Exception("Simulated Groq API Error")
    )

    test_payload = {
        "action": "REJECT",
        "response_key": "REJECT_LOWBALL",
        "policy_type": "rule-based",
        "policy_version": "1.1.0"
    }

    # 2. ACT
    response = client.post("/phrase", json=test_payload)

    # 3. ASSERT
    # We should get a 500 Internal Server Error
    assert response.status_code == 500
    assert "An internal server error occurred" in response.json()["detail"]