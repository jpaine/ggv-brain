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
COPY phase2_real_feature_calculations.py .
COPY v11_6_1_model_20251112_170640.pkl .
COPY v11_6_1_scaler_20251112_170640.pkl .

# Run workflow
CMD ["python", "workflow_1_crunchbase_daily_monitor.py"]

