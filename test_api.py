import pytest
from fastapi.testclient import TestClient
from api import app
import os
from datetime import datetime, timedelta
import json

client = TestClient(app)

def test_login_success():
    response = client.post(
        "/api/auth/login",
        data={
            "email": os.getenv("DEMO_PATIENT_EMAIL", "demo@example.com"),
            "password": os.getenv("DEMO_PATIENT_PASSWORD", "demo123")
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "user_id" in data
    assert "role" in data

def test_login_invalid_credentials():
    response = client.post(
        "/api/auth/login",
        data={
            "email": "invalid@example.com",
            "password": "wrongpassword"
        }
    )
    assert response.status_code == 401

def test_upload_record_unauthorized():
    response = client.post(
        "/api/records/upload",
        files={"file": ("test.pdf", b"test content", "application/pdf")},
        data={"patient_id": "test_user", "record_type": "Lab Report"}
    )
    assert response.status_code == 401

def test_upload_record_success():
    # First login to get token
    login_response = client.post(
        "/api/auth/login",
        data={
            "email": os.getenv("DEMO_PATIENT_EMAIL", "demo@example.com"),
            "password": os.getenv("DEMO_PATIENT_PASSWORD", "demo123")
        }
    )
    token = login_response.json()["access_token"]

    # Then try to upload a file
    response = client.post(
        "/api/records/upload",
        files={"file": ("test.pdf", b"test content", "application/pdf")},
        data={"patient_id": "demo_user", "record_type": "Lab Report"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "File uploaded successfully"

def test_upload_invalid_file_type():
    # First login to get token
    login_response = client.post(
        "/api/auth/login",
        data={
            "email": os.getenv("DEMO_PATIENT_EMAIL", "demo@example.com"),
            "password": os.getenv("DEMO_PATIENT_PASSWORD", "demo123")
        }
    )
    token = login_response.json()["access_token"]

    # Try to upload an invalid file type
    response = client.post(
        "/api/records/upload",
        files={"file": ("test.txt", b"test content", "text/plain")},
        data={"patient_id": "demo_user", "record_type": "Lab Report"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 400

def test_get_records_unauthorized():
    response = client.get("/api/records/demo_user")
    assert response.status_code == 401

def test_get_records_success():
    # First login to get token
    login_response = client.post(
        "/api/auth/login",
        data={
            "email": os.getenv("DEMO_PATIENT_EMAIL", "demo@example.com"),
            "password": os.getenv("DEMO_PATIENT_PASSWORD", "demo123")
        }
    )
    token = login_response.json()["access_token"]

    # Then try to get records
    response = client.get(
        "/api/records/demo_user",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_health_metric_validation():
    # Test valid metric
    valid_metric = {
        "metric_id": "test_metric",
        "patient_id": "demo_user",
        "metric_type": "blood_pressure",
        "value": 120,
        "unit": "mmHg",
        "timestamp": datetime.now().isoformat(),
        "is_critical": False
    }
    response = client.post(
        "/api/patients/demo_user/metrics",
        json=valid_metric,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200

    # Test invalid metric type
    invalid_metric_type = {
        "metric_id": "test_metric",
        "patient_id": "demo_user",
        "metric_type": "invalid_type",
        "value": 120,
        "unit": "mmHg",
        "timestamp": datetime.now().isoformat(),
        "is_critical": False
    }
    response = client.post(
        "/api/patients/demo_user/metrics",
        json=invalid_metric_type,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422

    # Test invalid value
    invalid_value = {
        "metric_id": "test_metric",
        "patient_id": "demo_user",
        "metric_type": "blood_pressure",
        "value": 400,  # Invalid blood pressure value
        "unit": "mmHg",
        "timestamp": datetime.now().isoformat(),
        "is_critical": False
    }
    response = client.post(
        "/api/patients/demo_user/metrics",
        json=invalid_value,
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 422 