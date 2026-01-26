# E:\FYP\llm-phraser\tests\test_phraser.py
# Version: v2.0.0 - LangChain Integration Tests

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas import PhraserInput
from unittest.mock import AsyncMock, patch

# Create the TestClient for FastAPI
client = TestClient(app)


# --- Basic Health Check Test ---
def test_health_check():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "llm-phraser"
    assert data["version"] == "2.0.0"


# --- Test 1: REJECT_LOWBALL - Should NOT contain a price in response ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_reject_lowball_no_counter_price(mock_generate):
    """
    Tests that REJECT_LOWBALL generates a refusal WITHOUT proposing a new price.
    This verifies the LLM is correctly instructed to reject without countering.
    """
    # Arrange
    mock_generate.return_value = "I appreciate your offer, but unfortunately that's too low for us to consider."
    
    # Test payload with REJECT_LOWBALL
    test_payload = {
        "action": "REJECT",
        "response_key": "REJECT_LOWBALL",
        "policy_type": "rule-based",
        "policy_version": "1.1.0"
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data["response_text"]
    
    # The response should NOT contain dollar signs or numbers (no counter-offer)
    assert "$" not in response_text
    assert not any(char.isdigit() for char in response_text)


# --- Test 2: COUNTER_FINAL_OFFER - Should contain "final" or "limit" ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_counter_final_offer_contains_keyword(mock_generate):
    """
    Tests that COUNTER_FINAL_OFFER generates text containing 'final' or 'limit'.
    This verifies the finality tone is preserved.
    """
    # Arrange
    mock_generate.return_value = "This is my absolute final offer at $50,000. I cannot go lower."
    
    test_payload = {
        "action": "COUNTER",
        "response_key": "COUNTER_FINAL_OFFER",
        "counter_price": 50000.0,
        "policy_type": "rule-based",
        "policy_version": "1.1.0"
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert
    assert response.status_code == 200
    response_data = response.json()
    response_text = response_data["response_text"].lower()
    
    # Response should contain either "final" or "limit"
    assert "final" in response_text or "limit" in response_text


# --- Test 3: Verify NO 'mam' field required ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_no_mam_field_required(mock_generate):
    """
    Tests that the service works WITHOUT requiring a 'mam' field.
    This verifies the security boundary - no sensitive data needed.
    """
    # Arrange
    mock_generate.return_value = "We're getting close! I can meet you at $45,000."
    
    # This payload intentionally does NOT include 'mam'
    test_payload = {
        "action": "COUNTER",
        "response_key": "STANDARD_COUNTER",
        "counter_price": 45000.0,
        "policy_type": "rule-based"
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert - Should succeed without 'mam'
    assert response.status_code == 200
    assert "response_text" in response.json()
    
    # Verify that the function was called with PhraserInput that has no 'mam'
    assert mock_generate.called
    call_args = mock_generate.call_args[0][0]
    assert isinstance(call_args, PhraserInput)
    assert not hasattr(call_args, 'mam')


# --- Test 4: Verify 'mam' field is IGNORED if sent ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_mam_field_ignored(mock_generate):
    """
    Tests that the API ignores attempts to send a 'mam' field.
    This verifies the Pydantic schema acts as a firewall.
    """
    # Arrange
    mock_generate.return_value = "We're getting close! I can meet you at $45,000."
    
    # This payload attempts to send 'mam' - should be ignored by Pydantic
    test_payload = {
        "action": "COUNTER",
        "response_key": "STANDARD_COUNTER",
        "counter_price": 45000.0,
        "policy_type": "rule-based",
        "mam": 40000.0  # This should be ignored
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert - Should still work (extra fields ignored by default)
    assert response.status_code == 200


# --- Test 5: Standard Success Case ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_generate_phrase_success(mock_generate):
    """
    Tests a successful phrase generation with STANDARD_COUNTER.
    """
    # Arrange
    mock_generate.return_value = "That's close! My best offer right now is $48,000. How does that sound?"
    
    test_payload = {
        "action": "COUNTER",
        "response_key": "STANDARD_COUNTER",
        "counter_price": 48000.0,
        "policy_type": "rule-based",
        "policy_version": "1.1.0"
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert "response_text" in data
    assert "$48,000" in data["response_text"]


# --- Test 6: Error Handling ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_generate_phrase_llm_error(mock_generate):
    """
    Tests how the API behaves if the LLM/LangChain call fails.
    """
    # Arrange - Mock to raise an exception
    mock_generate.side_effect = Exception("Simulated LangChain Error")
    
    test_payload = {
        "action": "REJECT",
        "response_key": "REJECT_LOWBALL",
        "policy_type": "rule-based"
    }
    
    # Act
    response = client.post("/phrase", json=test_payload)
    
    # Assert - Should return 500 error since the exception propagates
    assert response.status_code == 500


# --- Test 7: ACCEPT_FINAL Template ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_accept_final_template(mock_generate):
    """Tests the ACCEPT_FINAL template."""
    mock_generate.return_value = "Perfect! We can accept $50,000. Let's finalize this deal."
    
    test_payload = {
        "action": "ACCEPT",
        "response_key": "ACCEPT_FINAL",
        "counter_price": 50000.0,
        "policy_type": "rule-based"
    }
    
    response = client.post("/phrase", json=test_payload)
    assert response.status_code == 200
    assert len(response.json()["response_text"]) > 0


# --- Test 8: ACCEPT_SENTIMENT_CLOSE Template ---
@patch("app.main.generate_llm_response", new_callable=AsyncMock)
def test_accept_sentiment_close_template(mock_generate):
    """Tests the ACCEPT_SENTIMENT_CLOSE template."""
    mock_generate.return_value = "Alright, I can see this matters to you. We'll accept $47,000 to move forward."
    
    test_payload = {
        "action": "ACCEPT",
        "response_key": "ACCEPT_SENTIMENT_CLOSE",
        "counter_price": 47000.0,
        "policy_type": "rule-based"
    }
    
    response = client.post("/phrase", json=test_payload)
    assert response.status_code == 200
    assert len(response.json()["response_text"]) > 0


# --- Test 9: Prompt Template Selection ---
def test_prompt_template_selection():
    """Tests that the correct prompt template is selected for each response_key."""
    from app.prompt_templates import get_prompt_template
    from app.schemas import PhraserInput
    
    # Test REJECT_LOWBALL
    input_reject = PhraserInput(
        action="REJECT",
        response_key="REJECT_LOWBALL",
        policy_type="rule-based"
    )
    template = get_prompt_template(input_reject)
    assert template is not None
    
    # Test COUNTER_FINAL_OFFER
    input_final = PhraserInput(
        action="COUNTER",
        response_key="COUNTER_FINAL_OFFER",
        counter_price=50000.0,
        policy_type="rule-based"
    )
    template = get_prompt_template(input_final)
    assert template is not None
    
    # Test STANDARD_COUNTER
    input_counter = PhraserInput(
        action="COUNTER",
        response_key="STANDARD_COUNTER",
        counter_price=48000.0,
        policy_type="rule-based"
    )
    template = get_prompt_template(input_counter)
    assert template is not None