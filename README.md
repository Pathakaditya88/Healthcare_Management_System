# Healthcare Management System

Hey there! This is a healthcare management system I built to help patients and doctors manage medical records, appointments, and health data more easily. It's got a FastAPI backend for the API stuff and a Streamlit frontend for the user interface.

## What This Does

Basically, it's a system where:
- Patients can upload their medical documents, request appointments, and view their health records
- Doctors can search through patients, review records, schedule appointments, and track health metrics
- Everything is secured with JWT tokens and proper authentication
- File uploads are handled safely with size and type validation
- Health data can be analyzed and trends can be visualized

## Quick Start

If you just want to get this running locally, here's the fastest way:

```bash
# Set up your environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create uploads folder
mkdir uploads

# Run the API (in one terminal)
uvicorn api:app --reload

# Run the UI (in another terminal)
streamlit run app.py
```

Then open http://localhost:8000/docs for the API docs and http://localhost:8501 for the web interface.

## Setting Up Environment Variables

Create a `.env` file in your project root:

```env
SECRET_KEY=your-super-secret-key-here
ALLOWED_ORIGINS=http://localhost:8501
```

The SECRET_KEY is used for JWT token generation, so make it something secure and random.

## Demo Users

I've included some demo accounts so you can test things out right away:

- **Patient**: demo@example.com / demo123
- **Doctor**: doctor@example.com / doctor123

Just log in through the Streamlit interface and you'll be able to explore the system.

## Project Structure

Here's how the code is organized:

```
healthcare-management-system/
│
├── api.py                 # Main FastAPI app for local development
├── api/
│   └── index.py          # Serverless entrypoint for Vercel deployment
├── app.py                 # Streamlit user interface
├── utils.py               # Security utilities, JWT helpers, etc.
├── requirements.txt       # Python package dependencies
├── vercel.json            # Vercel configuration
├── uploads/               # Where uploaded files get stored
└── README.md              # This file
```

## Deploying to Vercel

The API part is ready to deploy on Vercel as a serverless function. Here's how:

1. Push your code to GitHub or GitLab
2. Import the project into Vercel (choose "Other" as the framework)
3. Set these environment variables in Vercel:
   - SECRET_KEY (required)
   - ALLOWED_ORIGINS (optional, for CORS)
   - UPLOAD_DIR (optional, defaults to "uploads")
4. Deploy

Your API will be available at your Vercel URL under `/api/...` routes.

Note: The Streamlit app can't run on Vercel, so you'll need to host that separately (like on Streamlit Community Cloud) and point it to your Vercel API URL.

## Main API Endpoints

Here are the key endpoints you'll probably use:

- `POST /api/auth/login` - Log in and get a token
- `POST /api/records/upload` - Upload a medical document
- `GET /api/records/{patient_id}` - Get records for a patient
- `POST /api/appointments` - Create an appointment request
- `GET /api/appointments` - List appointments
- `GET /api/patients/search` - Search for patients (doctors only)

You can explore all the endpoints interactively at `/docs` when the API is running.

## Common Issues and Solutions

**Getting 401 errors?** Make sure you're sending the Authorization header with your JWT token: `Authorization: Bearer <your-token>`

**CORS problems?** Check that your ALLOWED_ORIGINS includes the domain where your frontend is running

**File uploads failing?** Keep files under 10MB and make sure you're using multipart/form-data

**Can't connect to the API?** Double-check your API_URL setting in the Streamlit app

## Dependencies

The main packages this project uses:

**Backend**: FastAPI, Uvicorn, Python-multipart, PyJWT, Passlib
**Frontend**: Streamlit, Plotly, Pandas, NumPy
**Security**: bcrypt, cryptography

## Contributing

Feel free to contribute! Just keep your changes focused and include a clear description of what you're doing. Small improvements are always welcome.

## License

This is open source under the MIT License, so you can use it however you want.

## Getting Help

If you run into problems:
- Check the API docs at `/docs` for endpoint details
- Look at the error messages in your terminal
- Make sure all environment variables are set correctly
- Verify that your virtual environment is activated

That's about it! The system is pretty straightforward once you get it running. Let me know if you need help with anything specific. 