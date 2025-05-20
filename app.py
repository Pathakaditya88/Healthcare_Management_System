import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
from typing import Dict, List
import os
from dotenv import load_dotenv
import time
import numpy as np

# Load environment variables
load_dotenv()

# API configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
if os.getenv('SPACE_ID'):  # Running on Hugging Face Spaces
    API_URL = f"https://{os.getenv('SPACE_ID')}.hf.space"

# Session state initialization
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'token_expiry' not in st.session_state:
    st.session_state.token_expiry = None

def check_session():
    """Check if the session is valid and refresh token if needed."""
    if not st.session_state.token:
        return False
    
    if st.session_state.token_expiry and datetime.now() > st.session_state.token_expiry:
        st.session_state.token = None
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.session_state.token_expiry = None
        return False
    
    return True

def refresh_token():
    """Refresh the authentication token."""
    try:
        response = requests.post(
            f"{API_URL}/api/auth/refresh",
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.token_expiry = datetime.now() + timedelta(minutes=30)
            return True
    except Exception as e:
        st.error(f"Error refreshing token: {str(e)}")
    return False

def login_page():
    st.title("Patient Health Records System")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            try:
                # Create form data
                form_data = {
                    "email": email,
                    "password": password
                }
                
                # Make the request with proper headers
                response = requests.post(
                    f"{API_URL}/api/auth/login",
                    data=form_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.user_role = data["role"]
                    st.session_state.user_id = data["user_id"]
                    st.session_state.token_expiry = datetime.now() + timedelta(minutes=30)
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

def make_authenticated_request(method, endpoint, **kwargs):
    """Make an authenticated request to the API with error handling."""
    if not check_session():
        st.error("Your session has expired. Please log in again.")
        st.rerun()
        return None

    try:
        headers = kwargs.pop('headers', {})
        headers["Authorization"] = f"Bearer {st.session_state.token}"
        
        response = requests.request(
            method,
            f"{API_URL}{endpoint}",
            headers=headers,
            **kwargs
        )
        
        if response.status_code == 401:
            if refresh_token():
                return make_authenticated_request(method, endpoint, **kwargs)
            else:
                st.error("Session expired. Please log in again.")
                st.rerun()
                return None
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

def patient_dashboard():
    st.title("Patient Dashboard")
    
    # Add logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.rerun()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Upload Records", "View Records", "Appointments"]
    )
    
    if page == "Upload Records":
        upload_records_page()
    elif page == "View Records":
        view_records_page()
    elif page == "Appointments":
        appointments_page()

def upload_records_page():
    st.header("Upload Health Records")
    
    with st.form("upload_form"):
        file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png"])
        record_type = st.selectbox(
            "Record Type",
            ["Lab Report", "Prescription", "X-Ray", "Other"]
        )
        submit = st.form_submit_button("Upload")
        
        if submit and file:
            try:
                # Prepare the file for upload
                files = {
                    "file": (file.name, file.getvalue(), file.type)
                }
                
                # Prepare form data
                data = {
                    "patient_id": st.session_state.user_id,
                    "record_type": record_type
                }
                
                response = make_authenticated_request(
                    "POST",
                    "/api/records/upload",
                    files=files,
                    data=data
                )
                
                if response:
                    st.success("Record uploaded successfully")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def display_record_details(record):
    st.write(f"**Record ID:** {record.get('record_id', 'N/A')}")
    st.write(f"**Type:** {record.get('record_type', 'N/A')}")
    st.write(f"**Upload Date:** {format_datetime(record.get('upload_date', 'N/A'))}")
    
    # Display metadata if available
    metadata = record.get('metadata', {})
    if metadata:
        st.write("**Metadata:**")
        for key, value in metadata.items():
            st.write(f"- {key}: {value}")
    
    # Add download button
    if record.get('file_name'):
        st.write(f"**File:** {record['file_name']}")
        try:
            response = requests.get(
                f"{API_URL}/api/records/download/{record['record_id']}",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                stream=True
            )
            
            if response.status_code == 200:
                # Create a download button with the file content
                st.download_button(
                    label="Download File",
                    data=response.content,
                    file_name=record['file_name'],
                    mime=record.get('file_type', 'application/octet-stream')
                )
            else:
                st.error("Failed to download file")
        except Exception as e:
            st.error(f"Error downloading file: {str(e)}")

def view_records_page():
    st.header("Health Records")
    
    response = make_authenticated_request(
        "GET",
        f"/api/records/{st.session_state.user_id}"
    )
    
    if response:
        records = response
        if not records:
            st.info("No health records found.")
            return
            
        # Display each record
        for record in records:
            with st.expander(f"{record.get('record_type', 'Unknown Type')} - {format_datetime(record.get('upload_date', 'No Date'))}"):
                display_record_details(record)

def format_datetime(dt_str):
    if not dt_str:
        return "N/A"
    try:
        # Parse the ISO format string to datetime
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        # Convert to local timezone
        local_dt = dt.astimezone()
        # Format with timezone information
        return local_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        return f"Invalid date: {dt_str}"

def display_appointment_details(appointment):
    st.write(f"**Appointment ID:** {appointment['appointment_id']}")
    st.write(f"**Date:** {format_datetime(appointment['date'])}")
    st.write(f"**Status:** {appointment['status'].title()}")
    if appointment.get('notes'):
        st.write(f"**Notes:** {appointment['notes']}")
    
    # Show request and update details
    st.write("**Request Details:**")
    st.write(f"- Requested at: {format_datetime(appointment.get('requested_at'))}")
    
    if appointment.get('updated_at'):
        st.write("**Response Details:**")
        st.write(f"- Updated at: {format_datetime(appointment['updated_at'])}")
        st.write(f"- Updated by: {appointment['updated_by']}")

def appointment_requests_page():
    st.header("Appointment Requests")
    
    try:
        response = requests.get(
            f"{API_URL}/api/appointments",
            params={
                "user_id": st.session_state.user_id,
                "role": st.session_state.user_role
            },
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            appointments = response.json()
            if not appointments:
                st.info("No appointment requests found.")
                return
            
            # Group appointments by status
            pending_appointments = [apt for apt in appointments if apt['status'] == 'pending']
            other_appointments = [apt for apt in appointments if apt['status'] != 'pending']
            
            # Display pending appointments first
            if pending_appointments:
                st.subheader("Pending Requests")
                for appointment in pending_appointments:
                    with st.expander(f"Request from Patient - {format_datetime(appointment['date'])}"):
                        display_appointment_details(appointment)
                        
                        if st.session_state.user_role == "doctor":
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("Accept", key=f"accept_{appointment['appointment_id']}"):
                                    update_appointment_status(appointment['appointment_id'], 'accepted')
                            with col2:
                                if st.button("Decline", key=f"decline_{appointment['appointment_id']}"):
                                    update_appointment_status(appointment['appointment_id'], 'declined')
            
            # Display other appointments
            if other_appointments:
                st.subheader("Other Appointments")
                for appointment in other_appointments:
                    with st.expander(f"Appointment - {format_datetime(appointment['date'])}"):
                        display_appointment_details(appointment)
        else:
            st.error("Failed to fetch appointment requests")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def create_appointment(date: datetime, medical_condition: str, notes: str = None):
    try:
        response = requests.post(
            f"{API_URL}/api/appointments",
            data={
                "patient_id": st.session_state.user_id,
                "date": date.isoformat(),
                "medical_condition": medical_condition,
                "notes": notes
            },
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            st.success("Appointment request created successfully")
            time.sleep(1)  # Add a small delay to show the success message
            st.rerun()
        else:
            st.error(f"Failed to create appointment: {response.text}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def appointments_page():
    st.header("My Appointments")
    
    # Add form for creating new appointments
    if st.session_state.user_role == "patient":
        st.subheader("Request New Appointment")
        with st.form("new_appointment_form"):
            # Medical condition input
            medical_condition = st.text_area("Describe your medical condition or reason for appointment")
            
            appointment_date = st.date_input("Appointment Date")
            appointment_time = st.time_input("Appointment Time")
            notes = st.text_area("Additional Notes (Optional)")
            
            submit = st.form_submit_button("Request Appointment")
            
            if submit:
                if not medical_condition:
                    st.error("Please describe your medical condition")
                else:
                    # Combine date and time
                    appointment_datetime = datetime.combine(appointment_date, appointment_time)
                    create_appointment(appointment_datetime, medical_condition, notes)
    
    # Display existing appointments
    try:
        response = requests.get(
            f"{API_URL}/api/appointments",
            params={
                "user_id": st.session_state.user_id,
                "role": st.session_state.user_role
            },
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            appointments = response.json()
            if not appointments:
                st.info("No appointments found.")
                return
            
            # Group appointments by status
            pending_appointments = [apt for apt in appointments if apt['status'] == 'pending']
            accepted_appointments = [apt for apt in appointments if apt['status'] == 'accepted']
            declined_appointments = [apt for apt in appointments if apt['status'] == 'declined']
            
            # Display appointments by status
            if pending_appointments:
                st.subheader("Pending Appointments")
                for appointment in pending_appointments:
                    with st.expander(f"Appointment - {format_datetime(appointment['date'])}"):
                        display_appointment_details(appointment)
            
            if accepted_appointments:
                st.subheader("Accepted Appointments")
                for appointment in accepted_appointments:
                    with st.expander(f"Appointment - {format_datetime(appointment['date'])}"):
                        display_appointment_details(appointment)
            
            if declined_appointments:
                st.subheader("Declined Appointments")
                for appointment in declined_appointments:
                    with st.expander(f"Appointment - {format_datetime(appointment['date'])}"):
                        display_appointment_details(appointment)
        else:
            st.error("Failed to fetch appointments")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def doctor_dashboard():
    st.title("Doctor Dashboard")
    
    # Add logout button in sidebar
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.user_role = None
        st.session_state.user_id = None
        st.rerun()
    
    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Patient Search", "View Records", "Appointment Requests"]
    )
    
    if page == "Patient Search":
        patient_search_page()
    elif page == "View Records":
        view_records_page()
    elif page == "Appointment Requests":
        appointment_requests_page()

def patient_search_page():
    st.header("Patient Search")
    
    # Initialize selected_patient in session state if not exists
    if 'selected_patient' not in st.session_state:
        st.session_state.selected_patient = None
    
    # If a patient is selected, show their records
    if st.session_state.selected_patient:
        view_patient_records(st.session_state.selected_patient)
        if st.button("Back to Search", key="back_to_search_btn"):
            st.session_state.selected_patient = None
            st.rerun()
        return
    
    # Create two columns for filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic search
        search_query = st.text_input("Search by name, email, or condition", key="search_query_page")
        
        # Medical condition filter
        condition = st.text_input("Filter by medical condition", key="condition_filter_page")
        
        # Age range filter
        age_min = st.number_input("Minimum age", min_value=0, max_value=120, value=None, key="age_min_page")
        age_max = st.number_input("Maximum age", min_value=0, max_value=120, value=None, key="age_max_page")
    
    with col2:
        # Gender filter
        gender = st.selectbox("Gender", options=["", "Male", "Female", "Other"], key="gender_filter_page")
        
        # Blood type filter
        blood_type = st.selectbox("Blood Type", options=["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key="blood_type_filter_page")
    
    # Search button
    if st.button("Search Patients", key="search_patients_page_btn"):
        try:
            # Build query parameters
            params = {}
            if search_query:
                params["query"] = search_query
            if condition:
                params["condition"] = condition
            if age_min is not None:
                params["age_min"] = age_min
            if age_max is not None:
                params["age_max"] = age_max
            if gender:
                params["gender"] = gender
            if blood_type:
                params["blood_type"] = blood_type
            
            # Make API request
            response = requests.get(
                f"{API_URL}/api/patients/search",
                params=params,
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            
            if response.status_code == 200:
                patients = response.json()
                if not patients:
                    st.info("No patients found matching your criteria.")
                else:
                    st.success(f"Found {len(patients)} patients")
                    
                    # Display patient cards
                    for patient in patients:
                        with st.expander(f"{patient['name']} ({patient['email']})"):
                            profile = patient['profile']
                            
                            # Basic Info
                            st.subheader("Basic Information")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Age:** {profile['age']}")
                                st.write(f"**Gender:** {profile['gender']}")
                                st.write(f"**Blood Type:** {profile['blood_type']}")
                            with col2:
                                st.write(f"**Last Checkup:** {profile['last_checkup']}")
                                st.write(f"**Emergency Contact:** {profile['emergency_contact']}")
                            
                            # Medical Information
                            st.subheader("Medical Information")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Medical Conditions:**")
                                for condition in profile['medical_conditions']:
                                    st.write(f"- {condition}")
                                st.write("**Allergies:**")
                                for allergy in profile['allergies']:
                                    st.write(f"- {allergy}")
                            with col2:
                                st.write("**Current Medications:**")
                                for med in profile['medications']:
                                    st.write(f"- {med}")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("View Records", key=f"view_records_{patient['user_id']}"):
                                    st.session_state.selected_patient = patient['user_id']
                                    st.rerun()
                            with col2:
                                if st.button("Schedule Appointment", key=f"schedule_{patient['user_id']}"):
                                    schedule_appointment(patient['user_id'])
            else:
                st.error("Failed to search patients. Please try again.")
        except Exception as e:
            st.error(f"Error searching patients: {str(e)}")

def view_patient_records(patient_id: str):
    st.header(f"Patient Records - {patient_id}")
    
    # Add tabs for different views
    tab1, tab2, tab3 = st.tabs(["Health Records", "Health Metrics", "Health Alerts"])
    
    with tab1:
        try:
            response = requests.get(
                f"{API_URL}/api/records/{patient_id}",
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            
            if response.status_code == 200:
                records = response.json()
                if not records:
                    st.info("No health records found for this patient.")
                else:
                    # Display each record
                    for record in records:
                        with st.expander(f"{record.get('record_type', 'Unknown Type')} - {format_datetime(record.get('upload_date', 'No Date'))}"):
                            display_record_details(record)
            else:
                st.error("Failed to fetch patient records")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab2:
        display_health_metrics(patient_id)
    
    with tab3:
        display_health_alerts(patient_id)

def display_health_metrics(patient_id: str):
    st.subheader("Health Metrics")
    
    # Create columns for the form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Metric type selection
        metric_type = st.selectbox(
            "Select Metric Type",
            ["blood_pressure", "heart_rate", "blood_sugar", "temperature", "weight"]
        )
        
        # Value input based on metric type
        if metric_type == "blood_pressure":
            bp_col1, bp_col2 = st.columns(2)
            with bp_col1:
                systolic = st.number_input("Systolic", min_value=60, max_value=200, value=120)
            with bp_col2:
                diastolic = st.number_input("Diastolic", min_value=40, max_value=120, value=80)
            value = f"{systolic}/{diastolic}"
            unit = "mmHg"
        else:
            value = st.number_input(
                "Value",
                min_value=0.0,
                max_value=300.0 if metric_type == "blood_sugar" else 250.0,
                value=70.0 if metric_type == "heart_rate" else 98.6 if metric_type == "temperature" else 70.0
            )
            unit = {
                "heart_rate": "bpm",
                "blood_sugar": "mg/dL",
                "temperature": "°F",
                "weight": "kg"
            }.get(metric_type, "")
    
    with col2:
        # Timestamp input
        timestamp = st.date_input("Date", value=datetime.now().date())
        time = st.time_input("Time", value=datetime.now().time())
        timestamp = datetime.combine(timestamp, time)
        
        # Submit button
        if st.button("Save Metric", type="primary"):
            try:
                response = requests.post(
                    f"{API_URL}/api/patients/{patient_id}/metrics",
                    data={
                        "metric_type": metric_type,
                        "value": value,
                        "unit": unit,
                        "timestamp": timestamp.isoformat()
                    },
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    st.success("✅ Metric saved successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to save metric: {response.text}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display existing metrics
    try:
        response = requests.get(
            f"{API_URL}/api/patients/{patient_id}/metrics",
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            metrics = response.json()
            if not metrics:
                st.info("No health metrics found for this patient.")
                return
            
            # Group metrics by type
            metric_types = set(m["metric_type"] for m in metrics)
            for metric_type in metric_types:
                type_metrics = [m for m in metrics if m["metric_type"] == metric_type]
                
                with st.expander(f"{metric_type.replace('_', ' ').title()} ({len(type_metrics)} records)"):
                    # Create a DataFrame for plotting
                    df = pd.DataFrame([
                        {
                            "timestamp": datetime.fromisoformat(m["timestamp"]),
                            "value": float(m["value"].split("/")[0]) if "/" in str(m["value"]) else float(m["value"]),
                            "unit": m["unit"]
                        }
                        for m in type_metrics
                    ])
                    
                    if not df.empty:
                        # Plot trend
                        fig = px.line(
                            df,
                            x="timestamp",
                            y="value",
                            title=f"{metric_type.replace('_', ' ').title()} Trend"
                        )
                        st.plotly_chart(fig)
                        
                        # Display latest value
                        latest = type_metrics[-1]
                        st.write(f"**Latest Value:** {latest['value']} {latest['unit']}")
                        if latest.get("is_critical"):
                            st.warning("⚠️ Critical value detected!")
        
        else:
            st.error("Failed to fetch health metrics")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def display_health_alerts(patient_id: str):
    st.subheader("Health Alerts")
    
    try:
        response = requests.get(
            f"{API_URL}/api/patients/{patient_id}/alerts",
            params={"resolved": False},
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            alerts = response.json()
            if not alerts:
                st.info("No active health alerts for this patient.")
                return
            
            # Group alerts by severity
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            warning_alerts = [a for a in alerts if a["severity"] == "warning"]
            
            if critical_alerts:
                st.error("🚨 Critical Alerts")
                for alert in critical_alerts:
                    with st.expander(f"{alert['metric_type'].replace('_', ' ').title()} - {alert['value']} {alert.get('unit', '')}"):
                        st.write(f"**Message:** {alert['message']}")
                        st.write(f"**Threshold:** {alert['threshold']}")
                        st.write(f"**Time:** {format_datetime(alert['timestamp'])}")
                        
                        if st.button("Mark as Resolved", key=f"resolve_{alert['alert_id']}"):
                            try:
                                response = requests.put(
                                    f"{API_URL}/api/alerts/{alert['alert_id']}",
                                    data={"is_resolved": True},
                                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                                )
                                
                                if response.status_code == 200:
                                    st.success("Alert marked as resolved")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to update alert status")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            
            if warning_alerts:
                st.warning("⚠️ Warning Alerts")
                for alert in warning_alerts:
                    with st.expander(f"{alert['metric_type'].replace('_', ' ').title()} - {alert['value']} {alert.get('unit', '')}"):
                        st.write(f"**Message:** {alert['message']}")
                        st.write(f"**Threshold:** {alert['threshold']}")
                        st.write(f"**Time:** {format_datetime(alert['timestamp'])}")
                        
                        if st.button("Mark as Resolved", key=f"resolve_{alert['alert_id']}"):
                            try:
                                response = requests.put(
                                    f"{API_URL}/api/alerts/{alert['alert_id']}",
                                    data={"is_resolved": True},
                                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                                )
                                
                                if response.status_code == 200:
                                    st.success("Alert marked as resolved")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to update alert status")
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        else:
            st.error("Failed to fetch health alerts")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def update_appointment_status(appointment_id: str, status: str):
    try:
        response = requests.put(
            f"{API_URL}/api/appointments/{appointment_id}",
            json={"status": status},
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            st.success(f"Appointment {status} successfully")
            time.sleep(1)  # Add a small delay to show the success message
            st.rerun()
        else:
            st.error("Failed to update appointment status")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add notification functions
def get_notifications():
    try:
        response = requests.get(
            f"{API_URL}/api/notifications",
            params={"user_id": st.session_state.user_id},
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching notifications: {str(e)}")
        return []

def mark_notification_read(notification_id):
    try:
        response = requests.put(
            f"{API_URL}/api/notifications/{notification_id}",
            headers={"Authorization": f"Bearer {st.session_state.token}"}
        )
        
        if response.status_code == 200:
            st.rerun()
        else:
            st.error("Failed to mark notification as read")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def display_notifications():
    notifications = get_notifications()
    unread_count = sum(1 for n in notifications if not n["read"])
    
    if unread_count > 0:
        st.sidebar.markdown(f"🔔 **{unread_count} new notifications**")
    
    if notifications:
        with st.sidebar.expander("Notifications", expanded=unread_count > 0):
            for notification in sorted(notifications, key=lambda x: x["created_at"], reverse=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(notification["message"])
                    st.write(f"*{format_datetime(notification['created_at'])}*")
                with col2:
                    if not notification["read"]:
                        if st.button("✓", key=f"read_{notification['notification_id']}"):
                            mark_notification_read(notification["notification_id"])

def schedule_appointment(patient_id: str):
    st.subheader("Schedule Appointment")
    
    with st.form("schedule_appointment_form"):
        appointment_date = st.date_input("Appointment Date")
        appointment_time = st.time_input("Appointment Time")
        notes = st.text_area("Notes (Optional)")
        
        submit = st.form_submit_button("Schedule Appointment")
        
        if submit:
            try:
                # Combine date and time
                appointment_datetime = datetime.combine(appointment_date, appointment_time)
                
                response = requests.post(
                    f"{API_URL}/api/appointments/schedule",
                    data={
                        "patient_id": patient_id,
                        "date": appointment_datetime.isoformat(),
                        "notes": notes
                    },
                    headers={"Authorization": f"Bearer {st.session_state.token}"}
                )
                
                if response.status_code == 200:
                    st.success("Appointment scheduled successfully")
                    time.sleep(1)  # Add a small delay to show the success message
                    st.rerun()
                else:
                    st.error(f"Failed to schedule appointment: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

def main():
    if not check_session():
        login_page()
    else:
        # Display notifications in sidebar
        display_notifications()
        
        if st.session_state.user_role == "patient":
            patient_dashboard()
        elif st.session_state.user_role == "doctor":
            doctor_dashboard()
        else:
            st.error("Invalid user role")
            st.session_state.token = None
            st.session_state.user_role = None
            st.session_state.user_id = None
            st.rerun()

if __name__ == "__main__":
    main() 