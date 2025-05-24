from app import app

def test_health_check():
    tester = app.test_client()
    response = tester.get("/")
    assert response.status_code == 200
    assert b"Claude Conversational AI Assistant" in response.data
