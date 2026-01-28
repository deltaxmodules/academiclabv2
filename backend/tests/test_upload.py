from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_upload_csv_initial_flow():
    csv_content = "id,price,target\n1,10,0\n2,20,1\n3,,0\n"
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "session_id" in data
    assert "problems_detected" in data
