from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import json
from pydantic import BaseModel, validator
import utils
from fastapi.responses import FileResponse
import os
import base64
from utils import (
    verify_password, create_access_token, get_current_user,
    encrypt_sensitive_data, decrypt_sensitive_data, log_phi_access,
    validate_record_access, mask_sensitive_data,
    SecurityManager,
    HIPAACompliance,
    AccessControl
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Patient Health Records API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add CORS middleware with specific origins
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize security components
security_manager = SecurityManager()
hipaa = HIPAACompliance()
access_control = AccessControl()

# Load demo credentials from environment variables
DEMO_USERS = {
    "demo@example.com": {
        "password": "demo123",
        "role": "patient",
        "user_id": "demo_user",
        "name": "Demo Patient"
    },
    "doctor@example.com": {
        "password": "doctor123",
        "role": "doctor",
        "user_id": "demo_doctor",
        "name": "Demo Doctor"
    }
}

# Demo data storage
DEMO_APPOINTMENTS = {
    "appt_1": {
        "appointment_id": "appt_1",
        "patient_id": "demo_user",
        "doctor_id": "demo_doctor",
        "date": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "notes": "Regular checkup",
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": None,
        "updated_by": None
    }
}

# Constants for file upload
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png"
}

# Create a directory for storing files if it doesn't exist
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Demo data storage with actual files
DEMO_FILES = {}

# Add demo patient profiles with medical information
DEMO_PATIENT_PROFILES = {
    "demo_user": {
        "user_id": "demo_user",
        "name": "Demo Patient",
        "age": 35,
        "gender": "Male",
        "medical_conditions": ["Hypertension", "Type 2 Diabetes"],
        "allergies": ["Penicillin"],
        "medications": ["Metformin", "Lisinopril"],
        "last_checkup": "2024-01-15",
        "blood_type": "O+",
        "emergency_contact": "123-456-7890"
    }
}

# Add new models for notifications
class Notification(BaseModel):
    notification_id: str
    user_id: str
    message: str
    created_at: datetime
    read: bool = False

# Demo data storage for notifications
DEMO_NOTIFICATIONS = {}

# Add new models for health data
class HealthMetric(BaseModel):
    metric_id: str
    patient_id: str
    metric_type: str
    value: float
    unit: str
    timestamp: datetime
    is_critical: bool = False

    @validator('metric_type')
    def validate_metric_type(cls, v):
        valid_types = ['blood_pressure', 'heart_rate', 'blood_sugar', 'temperature', 'weight']
        if v not in valid_types:
            raise ValueError(f'metric_type must be one of {valid_types}')
        return v

    @validator('value')
    def validate_value(cls, v, values):
        if 'metric_type' in values:
            if values['metric_type'] == 'blood_pressure':
                if not isinstance(v, (int, float)) or v < 0 or v > 300:
                    raise ValueError('Blood pressure must be between 0 and 300')
            elif values['metric_type'] == 'heart_rate':
                if not isinstance(v, (int, float)) or v < 0 or v > 250:
                    raise ValueError('Heart rate must be between 0 and 250')
            elif values['metric_type'] == 'blood_sugar':
                if not isinstance(v, (int, float)) or v < 0 or v > 1000:
                    raise ValueError('Blood sugar must be between 0 and 1000')
        return v

class HealthAlert(BaseModel):
    alert_id: str
    patient_id: str
    metric_type: str
    value: float
    threshold: float
    severity: str  # "critical", "warning", "info"
    timestamp: datetime
    message: str
    is_resolved: bool = False

# Demo data storage for health metrics and alerts
DEMO_HEALTH_METRICS = {}
DEMO_HEALTH_ALERTS = {}

# Pydantic models for request/response validation
class User(BaseModel):
    user_id: str
    role: str
    email: str
    name: str

class HealthRecord(BaseModel):
    record_id: str
    patient_id: str
    record_type: str
    upload_date: datetime
    metadata: dict
    access_granted_to: List[str]

class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    doctor_id: str
    date: datetime
    status: str
    notes: Optional[str]

# Authentication endpoints
@app.post("/api/auth/register")
async def register_user(email: str = Form(...), password: str = Form(...), role: str = Form(...)):
    try:
        # In a real application, this would create a user in the database
        hashed_password = utils.hash_password(password)
        user_id = f"user_{datetime.utcnow().timestamp()}"
        
        return {
            "user_id": user_id,
            "email": email,
            "role": role,
            "message": "User registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        print(f"Login attempt for email: {email}")  # Debug log
        # Check demo credentials
        if email in DEMO_USERS:
            print(f"User found in DEMO_USERS")  # Debug log
            if password == DEMO_USERS[email]["password"]:
                print(f"Password matches")  # Debug log
                user = DEMO_USERS[email]
                token = utils.generate_jwt_token(user["user_id"], user["role"])
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "user_id": user["user_id"],
                    "role": user["role"]
                }
            else:
                print(f"Password mismatch. Expected: {DEMO_USERS[email]['password']}, Got: {password}")  # Debug log
        else:
            print(f"User not found in DEMO_USERS")  # Debug log
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        print(f"Login error: {str(e)}")  # Debug log
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Health record endpoints
@app.post("/api/records/upload")
async def upload_record(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    record_type: str = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Validate file type
        if file.content_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES.keys())}"
            )

        # Validate file size
        file_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
                )
        await file.seek(0)

        # Validate user access
        user = await get_current_user(token)
        if not validate_record_access(user, patient_id):
            raise HTTPException(status_code=403, detail="Access denied")

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = ALLOWED_FILE_TYPES[file.content_type]
        filename = f"{patient_id}_{timestamp}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Log access
        log_phi_access(user["user_id"], "upload", f"Record uploaded for patient {patient_id}")

        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "record_type": record_type
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/records/{patient_id}")
async def get_record(patient_id: str, token: str = Depends(oauth2_scheme)):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Check if user is authorized to view these records
        if user_data["role"] == "patient" and user_data["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Not authorized to view these records")
        
        # Get all records for this patient
        records = []
        
        # Add uploaded records for this patient
        for record_id, file_data in DEMO_FILES.items():
            if record_id.startswith("record_"):
                records.append({
                    "record_id": record_id,
                    "patient_id": patient_id,
                    "record_type": file_data.get("record_type", "Uploaded Record"),
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        "type": file_data["type"].split("/")[-1],
                        "size": len(base64.b64decode(file_data["content"])),
                        "upload_date": datetime.now(timezone.utc).isoformat()
                    },
                    "file_name": file_data["filename"],
                    "file_type": file_data["type"]
                })
        
        return records
    except Exception as e:
        raise HTTPException(status_code=404, detail="Records not found")

@app.get("/api/records/download/{record_id}")
async def download_record(record_id: str, token: str = Depends(oauth2_scheme)):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Check if record exists
        if record_id not in DEMO_FILES:
            raise HTTPException(status_code=404, detail="Record not found")
        
        # Get file data
        file_data = DEMO_FILES[record_id]
        file_content = base64.b64decode(file_data["content"])
        
        # Create a temporary file
        file_path = os.path.join(UPLOAD_DIR, file_data["filename"])
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Return the file with proper headers
        return FileResponse(
            path=file_path,
            filename=file_data["filename"],
            media_type=file_data["type"],
            headers={
                "Content-Disposition": f'attachment; filename="{file_data["filename"]}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Appointment endpoints
@app.post("/api/appointments")
async def create_appointment(
    patient_id: str = Form(...),
    date: datetime = Form(...),
    medical_condition: str = Form(...),
    notes: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Verify that the user is a patient and matches the patient_id
        if user_data["role"] != "patient" or user_data["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Only patients can create their own appointments")
        
        # Generate appointment ID
        appointment_id = f"appt_{datetime.utcnow().timestamp()}"
        
        # Create appointment record
        appointment = {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "date": date.isoformat(),
            "status": "pending",
            "medical_condition": medical_condition,
            "notes": notes,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "updated_by": None
        }
        
        # Store in DEMO_APPOINTMENTS
        DEMO_APPOINTMENTS[appointment_id] = appointment
        
        return appointment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/appointments")
async def get_appointments(
    user_id: str,
    role: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Filter appointments based on role and user_id
        if role == "doctor":
            # Return appointments where doctor_id matches
            return [apt for apt in DEMO_APPOINTMENTS.values() if apt["doctor_id"] == user_id]
        else:
            # Return appointments where patient_id matches
            return [apt for apt in DEMO_APPOINTMENTS.values() if apt["patient_id"] == user_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/appointments/{appointment_id}")
async def update_appointment_status(
    appointment_id: str,
    status: dict,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can update appointment status")
        
        # Check if appointment exists
        if appointment_id not in DEMO_APPOINTMENTS:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        # Update appointment status and metadata
        DEMO_APPOINTMENTS[appointment_id]["status"] = status["status"]
        DEMO_APPOINTMENTS[appointment_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        DEMO_APPOINTMENTS[appointment_id]["updated_by"] = user_data["user_id"]
        
        return {
            "appointment_id": appointment_id,
            "status": status["status"],
            "updated_at": DEMO_APPOINTMENTS[appointment_id]["updated_at"],
            "message": "Appointment status updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add new search endpoint
@app.get("/api/patients/search")
async def search_patients(
    query: str = None,
    condition: str = None,
    age_min: int = None,
    age_max: int = None,
    gender: str = None,
    blood_type: str = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can search patients")
        
        # Get all patients
        patients = []
        for email, user in DEMO_USERS.items():
            if user["role"] == "patient":
                patient_id = user["user_id"]
                if patient_id in DEMO_PATIENT_PROFILES:
                    profile = DEMO_PATIENT_PROFILES[patient_id]
                    patients.append({
                        "user_id": patient_id,
                        "name": user["name"],
                        "email": email,
                        "profile": profile
                    })
        
        # Apply filters
        filtered_patients = patients
        
        if query:
            filtered_patients = [
                p for p in filtered_patients
                if query.lower() in p["name"].lower() or
                   query.lower() in p["email"].lower() or
                   any(query.lower() in cond.lower() for cond in p["profile"]["medical_conditions"])
            ]
        
        if condition:
            filtered_patients = [
                p for p in filtered_patients
                if condition.lower() in [c.lower() for c in p["profile"]["medical_conditions"]]
            ]
        
        if age_min is not None:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"]["age"] >= age_min
            ]
        
        if age_max is not None:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"]["age"] <= age_max
            ]
        
        if gender:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"]["gender"].lower() == gender.lower()
            ]
        
        if blood_type:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"]["blood_type"].lower() == blood_type.lower()
            ]
        
        return filtered_patients
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/profile")
async def get_patient_profile(
    patient_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient profiles")
        
        # Check if patient exists
        if patient_id not in DEMO_PATIENT_PROFILES:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get patient profile
        profile = DEMO_PATIENT_PROFILES[patient_id]
        user_info = next((user for user in DEMO_USERS.values() if user["user_id"] == patient_id), None)
        
        if not user_info:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return {
            "user_id": patient_id,
            "name": user_info["name"],
            "email": next(email for email, user in DEMO_USERS.items() if user["user_id"] == patient_id),
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add new endpoint for doctor to schedule appointment
@app.post("/api/appointments/schedule")
async def schedule_appointment(
    patient_id: str = Form(...),
    date: datetime = Form(...),
    notes: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can schedule appointments")
        
        # Generate appointment ID
        appointment_id = f"appt_{datetime.utcnow().timestamp()}"
        
        # Create appointment record
        appointment = {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "doctor_id": user_data["user_id"],
            "date": date.isoformat(),
            "status": "scheduled",
            "notes": notes,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": user_data["user_id"]
        }
        
        # Store in DEMO_APPOINTMENTS
        DEMO_APPOINTMENTS[appointment_id] = appointment
        
        # Create notification for patient
        notification_id = f"notif_{datetime.utcnow().timestamp()}"
        notification = {
            "notification_id": notification_id,
            "user_id": patient_id,
            "message": f"New appointment scheduled with Dr. {user_data['name']} on {date.strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "read": False
        }
        DEMO_NOTIFICATIONS[notification_id] = notification
        
        return {
            "appointment": appointment,
            "notification": notification
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add endpoint to get notifications
@app.get("/api/notifications")
async def get_notifications(
    user_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to view these notifications")
        
        # Get notifications for user
        notifications = [
            notif for notif in DEMO_NOTIFICATIONS.values()
            if notif["user_id"] == user_id
        ]
        
        return notifications
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add endpoint to mark notification as read
@app.put("/api/notifications/{notification_id}")
async def mark_notification_read(
    notification_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Check if notification exists
        if notification_id not in DEMO_NOTIFICATIONS:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        # Check if user owns the notification
        if DEMO_NOTIFICATIONS[notification_id]["user_id"] != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to modify this notification")
        
        # Mark as read
        DEMO_NOTIFICATIONS[notification_id]["read"] = True
        
        return {"message": "Notification marked as read"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add new endpoints for health data analysis
@app.get("/api/patients/{patient_id}/metrics")
async def get_patient_metrics(
    patient_id: str,
    metric_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient metrics")
        
        # Get metrics for patient
        metrics = [
            metric for metric in DEMO_HEALTH_METRICS.values()
            if metric["patient_id"] == patient_id
        ]
        
        # Apply filters
        if metric_type:
            metrics = [m for m in metrics if m["metric_type"] == metric_type]
        if start_date:
            metrics = [m for m in metrics if datetime.fromisoformat(m["timestamp"]) >= start_date]
        if end_date:
            metrics = [m for m in metrics if datetime.fromisoformat(m["timestamp"]) <= end_date]
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/alerts")
async def get_patient_alerts(
    patient_id: str,
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient alerts")
        
        # Get alerts for patient
        alerts = [
            alert for alert in DEMO_HEALTH_ALERTS.values()
            if alert["patient_id"] == patient_id
        ]
        
        # Apply filters
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a["is_resolved"] == resolved]
        
        return alerts
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/trends")
async def get_patient_trends(
    patient_id: str,
    metric_type: str,
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient trends")
        
        # Calculate date range based on period
        end_date = datetime.now(timezone.utc)
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            raise HTTPException(status_code=400, detail="Invalid period")
        
        # Get metrics for patient
        metrics = [
            metric for metric in DEMO_HEALTH_METRICS.values()
            if metric["patient_id"] == patient_id
            and metric["metric_type"] == metric_type
            and datetime.fromisoformat(metric["timestamp"]) >= start_date
            and datetime.fromisoformat(metric["timestamp"]) <= end_date
        ]
        
        # Calculate statistics
        if metrics:
            values = [m["value"] for m in metrics]
            stats = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1],
                "trend": "increasing" if values[-1] > values[0] else "decreasing",
                "data_points": len(values)
            }
        else:
            stats = {
                "min": None,
                "max": None,
                "avg": None,
                "latest": None,
                "trend": None,
                "data_points": 0
            }
        
        return {
            "metric_type": metric_type,
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "statistics": stats,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/patients/{patient_id}/metrics")
async def add_patient_metric(
    patient_id: str,
    metric_type: str = Form(...),
    value: str = Form(...),  # Changed from float to str to handle blood pressure format
    unit: str = Form(...),
    timestamp: datetime = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        
        # Generate metric ID
        metric_id = f"metric_{datetime.utcnow().timestamp()}"
        
        # Check for critical values
        is_critical = False
        if metric_type == "blood_pressure":
            systolic, diastolic = map(float, value.split("/"))
            if systolic > 140 or diastolic > 90:
                is_critical = True
        else:
            value_float = float(value)
            if metric_type == "heart_rate":
                if value_float > 100 or value_float < 60:
                    is_critical = True
            elif metric_type == "blood_sugar":
                if value_float > 140 or value_float < 70:
                    is_critical = True
        
        # Create metric record
        metric = {
            "metric_id": metric_id,
            "patient_id": patient_id,
            "metric_type": metric_type,
            "value": value,
            "unit": unit,
            "timestamp": timestamp.isoformat(),
            "is_critical": is_critical
        }
        
        # Store in DEMO_HEALTH_METRICS
        DEMO_HEALTH_METRICS[metric_id] = metric
        
        # Create alert if critical
        if is_critical:
            alert_id = f"alert_{datetime.utcnow().timestamp()}"
            alert = {
                "alert_id": alert_id,
                "patient_id": patient_id,
                "metric_type": metric_type,
                "value": value,
                "threshold": "140/90" if metric_type == "blood_pressure" else (100 if metric_type == "heart_rate" else 140),
                "severity": "critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Critical {metric_type} value: {value} {unit}",
                "is_resolved": False
            }
            DEMO_HEALTH_ALERTS[alert_id] = alert
        
        return metric
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/alerts/{alert_id}")
async def update_alert_status(
    alert_id: str,
    is_resolved: bool = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Verify token and permissions
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can update alerts")
        
        # Check if alert exists
        if alert_id not in DEMO_HEALTH_ALERTS:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update alert status
        DEMO_HEALTH_ALERTS[alert_id]["is_resolved"] = is_resolved
        
        return {"message": "Alert status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 