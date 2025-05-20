# Healthcare Management System

A modern healthcare management system that helps patients and doctors manage medical records, appointments, and health data efficiently. Built with FastAPI and Streamlit, this application provides a secure and user-friendly interface for healthcare management.

## What's Inside?

- 🔐 **Secure Authentication**: JWT-based authentication system with role-based access control
- 📋 **Medical Records**: Easy upload, download, and management of medical documents
- 📅 **Smart Scheduling**: Intuitive appointment scheduling system for patients and doctors
- 📊 **Health Analytics**: Visualize and analyze health metrics and trends
- 🔒 **Data Protection**: HIPAA-compliant security measures for sensitive health information
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## Getting Started

### Prerequisites

Before you begin, make sure you have:
- Python 3.8 or newer
- Git installed
- A code editor (VS Code recommended)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd healthcare-management-system
```

2. Set up a virtual environment:
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Configuration

1. Create a `.env` file in the project root:
```env
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

2. Create the uploads directory:
```bash
mkdir uploads
```

## Running the Application

The application has two main components:

### Backend Server (FastAPI)
```bash
uvicorn api:app --reload
```
Access the API at: http://localhost:8000

### Frontend Interface (Streamlit)
```bash
streamlit run app.py
```
Access the web interface at: http://localhost:8501

## Security Features

We take security seriously. Here's what we've implemented:
- Password hashing with bcrypt
- JWT token authentication
- Role-based access control
- Secure file upload validation
- Environment variable protection
- Input validation and sanitization

## API Documentation

Once the backend is running, you can explore the API:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## Project Structure

```
├── api.py              # FastAPI backend
├── app.py             # Streamlit frontend
├── utils.py           # Helper functions
├── requirements.txt   # Dependencies
├── .env              # Environment config
├── uploads/          # File storage
└── README.md         # Documentation
```

## Key Dependencies

### Backend
- FastAPI - Modern web framework
- Uvicorn - ASGI server
- Python-multipart - File handling
- Python-jose - JWT tokens
- Passlib - Password security
- Python-dotenv - Environment management

### Frontend
- Streamlit - Web interface
- Plotly - Data visualization
- Pandas - Data processing
- NumPy - Numerical operations
- Scikit-learn - ML utilities

### Development
- Pytest - Testing
- Black - Code formatting
- Flake8 - Code linting

## How to Use

### Authentication
1. Register a new account
2. Log in with your credentials
3. The system will manage your access token automatically

### Medical Records
1. Upload your medical documents (PDF, images)
2. View and download your records
3. Track your medical history

### Appointments
1. Schedule new appointments
2. View your upcoming appointments
3. Cancel or reschedule as needed

### Health Analytics
1. View your health trends
2. Generate health reports
3. Analyze your health metrics

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Need Help?

If you run into any issues or have questions:
- Open an issue in the GitHub repository
- Contact the development team
- Check the documentation

## Acknowledgments

Thanks to all contributors and the open-source community for their support and feedback. 