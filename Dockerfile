# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Create directories for static and media files
RUN mkdir -p staticfiles media

# Create entrypoint script with error handling
RUN echo '#!/bin/bash\n\
set -e\n\
echo "========================================"\n\
echo "Starting Link-hash application..."\n\
echo "========================================"\n\
\n\
echo "Step 1: Collecting static files..."\n\
python manage.py collectstatic --noinput || echo "Warning: collectstatic failed, continuing..."\n\
\n\
echo "Step 2: Creating migrations..."\n\
python manage.py makemigrations || echo "Warning: makemigrations failed, continuing..."\n\
\n\
echo "Step 3: Running migrations..."\n\
python manage.py migrate --noinput || echo "Warning: migrate failed, continuing..."\n\
\n\
echo "Step 4: Starting Daphne server on 0.0.0.0:8000..."\n\
exec daphne -b 0.0.0.0 -p 8000 config.asgi:application\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

# Expose port
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
