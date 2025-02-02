FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy only necessary files first
COPY requirements.txt .
COPY setup.py .
COPY sigma_llm ./sigma_llm
COPY api ./api

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Set the working directory to the api folder
WORKDIR /app/api

# The port should match what your Flask app is using
EXPOSE 8090

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8090", "app:app"] 