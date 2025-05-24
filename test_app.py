import pytest
from unittest.mock import patch
from app import app

@pytest.fixture
def client():
    return app.test_client()

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Claude Conversational AI Assistant" in response.data

def test_chat_valid_input(client):
    mock_response = {
        "body": '{"completion": "Hello, human!"}'
    }

    class MockBody:
        def read(self):
            return mock_response["body"].encode()

    with patch("app.client.invoke_model") as mock_invoke:
        mock_invoke.return_value = {"body": MockBody()}
        response = client.post("/chat", json={"user_input": "Hi"})
        assert response.status_code == 200
        data = response.get_json()
        assert "response" in data
        assert data["response"] == "Hello, human!"

def test_chat_missing_user_input(client):
    response = client.post("/chat", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "Missing user input"

def test_chat_invoke_model_exception(client):
    with patch("app.client.invoke_model", side_effect=Exception("Mock failure")):
        response = client.post("/chat", json={"user_input": "Hi"})
        assert response.status_code == 500
        data = response.get_json()
        assert data["error"] == "Failed to invoke model"
        assert "Mock failure" in data["detail"]
