#!/bin/bash
"""
Startup script for REINVENT4 Streamlit application in Docker container
"""

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Create necessary directories
mkdir -p /app/data /app/logs /app/outputs

# Set permissions
chmod 755 /app/data /app/logs /app/outputs

# Check dependencies
echo "🔍 Checking dependencies..."
python check_dependencies.py

# Start the Streamlit application
echo "🚀 Starting REINVENT4 Streamlit application..."
exec streamlit run streamlit_app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.maxUploadSize=200
