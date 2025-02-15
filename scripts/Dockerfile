FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY setup.py .
COPY sigma_llm ./sigma_llm
COPY scripts ./scripts
COPY config ./config
COPY utils ./utils
COPY query_rule_pairs.json .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e .
RUN pip install google-cloud-storage rapidfuzz

# Set environment variables for debug logging
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=DEBUG

# Set the working directory to the scripts folder
WORKDIR /app/scripts

# Verify file structure on startup
RUN echo "Verifying file structure..." && \
    ls -la /app && \
    echo "Contents of config directory:" && \
    ls -la /app/config && \
    echo "Contents of scripts directory:" && \
    ls -la /app/scripts

# Command to run experiments with verbose logging and debug prints
CMD ["sh", "-c", "pwd && ls -la && python -v run_experiments.py"]