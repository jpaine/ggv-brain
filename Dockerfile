FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY workflow_1_crunchbase_daily_monitor.py .
COPY phase5_pitch_quality_features.py .
COPY v11_5_phase5_model_20251105_170709.pkl .
COPY v11_5_phase5_scaler_20251105_170709.pkl .
COPY v11_5_phase5_config_20251105_170709.json .

# Run workflow
CMD ["python", "workflow_1_crunchbase_daily_monitor.py"]

