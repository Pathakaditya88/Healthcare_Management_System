# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run FastAPI with uvicorn (use $PORT if provided by platform)
CMD ["uvicorn", "index:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
