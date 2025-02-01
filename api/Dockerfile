FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy the entire project
COPY . .

# Install the project and its dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .

# Set the working directory to the api folder
WORKDIR /app/api

EXPOSE 5000

CMD ["python", "app.py"] 