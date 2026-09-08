"""Integration and API tests for AskYourDoc FastAPI endpoints."""

import pytest
from starlette.testclient import TestClient
from src.api.app import create_app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_index_page(client):
    """Test that the homepage renders successfully with HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AskYourDoc" in response.text
    assert "Upload Document" in response.text


def test_get_documents(client):
    """Test retrieving indexed documents list."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_upload_invalid_file_extension(client):
    """Test that non-PDF uploads are rejected with 400 Bad Request."""
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", b"some plain text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only valid .pdf files are accepted" in response.json()["detail"]