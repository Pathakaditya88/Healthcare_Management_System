import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from typing import Dict, List, Optional, Any
import pandas as pd
import json
from PIL import Image
import io
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from passlib.context import CryptContext
import numpy as np
from sklearn.preprocessing import StandardScaler
from fastapi import UploadFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)

# Demo secret key for JWT
DEMO_JWT_SECRET = os.getenv("JWT_SECRET_KEY", "demo-secret-key-for-jwt-token-generation")

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", DEMO_JWT_SECRET)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Encryption setup
def generate_encryption_key() -> bytes:
    """Generate a new encryption key."""
    return Fernet.generate_key()

def get_encryption_key() -> bytes:
    """Get the encryption key from environment variable or generate a new one."""
    key = os.getenv('ENCRYPTION_KEY')
    if not key:
        # Generate a new key and encode it properly
        key = base64.urlsafe_b64encode(os.urandom(32))
        return key
    # Ensure the key is properly formatted
    if len(key) != 32:
        key = base64.urlsafe_b64encode(key.encode()[:32])
    return key.encode()

def encrypt_data(data: str) -> bytes:
    """Encrypt sensitive data."""
    try:
        f = Fernet(get_encryption_key())
        return f.encrypt(data.encode())
    except Exception as e:
        print(f"Encryption error: {str(e)}")
        return data.encode()  # Return unencrypted data if encryption fails

def decrypt_data(encrypted_data: bytes) -> str:
    """Decrypt sensitive data."""
    try:
        f = Fernet(get_encryption_key())
        return f.decrypt(encrypted_data).decode()
    except Exception as e:
        print(f"Decryption error: {str(e)}")
        return encrypted_data.decode()  # Return encrypted data if decryption fails

# Authentication helpers
def hash_password(password: str) -> bytes:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password: str, hashed: bytes) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode(), hashed)

def generate_jwt_token(user_id: str, role: str) -> str:
    """Generate a JWT token for authentication."""
    payload = {
        'user_id': user_id,
        'role': role,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_jwt_token(token: str) -> Dict:
    """Verify and decode a JWT token."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")

def create_access_token(user_id: str, role: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "sub": user_id,
        "role": role,
        "exp": expire
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# File handling
def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file"""
    allowed_extensions = {".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx"}
    file_ext = os.path.splitext(file.filename)[1].lower()
    return file_ext in allowed_extensions

def extract_metadata(file_data: bytes, file_type: str) -> Dict:
    """Extract basic metadata from uploaded file."""
    metadata = {
        'size': len(file_data),
        'type': file_type,
        'upload_date': datetime.utcnow().isoformat()
    }
    
    if file_type.lower() in ['jpg', 'jpeg', 'png']:
        try:
            image = Image.open(io.BytesIO(file_data))
            metadata.update({
                'dimensions': image.size,
                'format': image.format
            })
        except Exception:
            pass
    
    return metadata

# Health data processing
def analyze_health_data(data: List[Dict]) -> Dict:
    """Analyze health data and generate trends."""
    if not data:
        return {}
    
    df = pd.DataFrame(data)
    analysis = {}
    
    # Calculate trends for numerical values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        analysis[col] = {
            'mean': df[col].mean(),
            'trend': df[col].diff().mean(),
            'latest': df[col].iloc[-1]
        }
    
    return analysis

def check_critical_values(data: Dict) -> List[str]:
    """Check for critical health values and generate alerts."""
    alerts = []
    
    # Define critical thresholds
    thresholds = {
        'blood_pressure_systolic': {'min': 90, 'max': 140},
        'blood_pressure_diastolic': {'min': 60, 'max': 90},
        'heart_rate': {'min': 60, 'max': 100},
        'temperature': {'min': 36.1, 'max': 37.2}
    }
    
    for key, value in data.items():
        if key in thresholds:
            if value < thresholds[key]['min']:
                alerts.append(f"Low {key.replace('_', ' ')}: {value}")
            elif value > thresholds[key]['max']:
                alerts.append(f"High {key.replace('_', ' ')}: {value}")
    
    return alerts

# Access control
def check_access_permission(user_id: str, record_id: str, user_role: str) -> bool:
    """Check if a user has permission to access a record."""
    if user_role == 'admin':
        return True
    
    # This would typically query a database to check permissions
    # For now, return a placeholder implementation
    return True

def log_access(user_id: str, record_id: str, action: str) -> None:
    """Log record access for audit purposes."""
    log_entry = {
        'user_id': user_id,
        'record_id': record_id,
        'action': action,
        'timestamp': datetime.utcnow().isoformat()
    }
    # This would typically write to a database
    print(f"Access log: {json.dumps(log_entry)}")

class SecurityManager:
    def __init__(self):
        self.key = self._generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def _generate_key(self) -> bytes:
        """Generate encryption key using PBKDF2"""
        salt = b'health_records_salt'  # In production, use a secure random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(SECRET_KEY.encode()))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption error: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decrypted_data = self.cipher_suite.decrypt(base64.urlsafe_b64decode(encrypted_data))
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption error: {str(e)}")
            raise

class HIPAACompliance:
    def __init__(self):
        self.access_log = []
    
    def log_access(self, user_id: str, patient_id: str, action: str):
        """Log access to PHI for HIPAA compliance"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "patient_id": patient_id,
            "action": action
        }
        self.access_log.append(log_entry)
        logger.info(f"PHI Access: {log_entry}")
    
    def validate_consent(self, patient_id: str, purpose: str) -> bool:
        """Validate patient consent for data access"""
        # In production, implement proper consent management
        return True
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive PHI data"""
        masked_data = data.copy()
        sensitive_fields = ["ssn", "phone", "email", "address"]
        
        for field in sensitive_fields:
            if field in masked_data:
                masked_data[field] = "***"
        
        return masked_data

class AccessControl:
    def __init__(self):
        self.access_matrix = {}
    
    def check_access(self, user_id: str, resource_id: str, action: str) -> bool:
        """Check if user has access to resource"""
        # In production, implement proper RBAC
        if user_id.startswith("doctor"):
            return True
        return user_id == resource_id
    
    def grant_access(self, user_id: str, resource_id: str, action: str):
        """Grant access to resource"""
        if resource_id not in self.access_matrix:
            self.access_matrix[resource_id] = {}
        if user_id not in self.access_matrix[resource_id]:
            self.access_matrix[resource_id][user_id] = set()
        self.access_matrix[resource_id][user_id].add(action)
    
    def revoke_access(self, user_id: str, resource_id: str, action: str):
        """Revoke access to resource"""
        if resource_id in self.access_matrix and user_id in self.access_matrix[resource_id]:
            self.access_matrix[resource_id][user_id].discard(action)

# Initialize security components
security_manager = SecurityManager()
hipaa_compliance = HIPAACompliance()
access_control = AccessControl()

def encrypt_sensitive_data(data):
    """Helper function to encrypt sensitive data"""
    return security_manager.encrypt_data(data)

def decrypt_sensitive_data(encrypted_data):
    """Helper function to decrypt sensitive data"""
    return security_manager.decrypt_data(encrypted_data)

def log_phi_access(user_id, action, record_id=None):
    """Helper function to log PHI access"""
    return hipaa_compliance.log_access(user_id, action, record_id)

def validate_record_access(patient_id, doctor_id, record_id):
    """Helper function to validate record access"""
    return hipaa_compliance.validate_consent(patient_id, record_id)

def mask_sensitive_data(data):
    """Helper function to mask sensitive data"""
    return hipaa_compliance.mask_sensitive_data(data)

def get_current_user(token: str) -> dict:
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["sub"], "role": payload["role"]}
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.JWTError:
        raise ValueError("Invalid token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

# ... existing code ... 