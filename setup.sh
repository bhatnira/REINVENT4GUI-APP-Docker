#!/bin/bash

# Setup script for Streamlit deployment
echo "Setting up GenChem for deployment..."

# Create necessary directories
mkdir -p .streamlit
mkdir -p data
mkdir -p temp
mkdir -p results

# Set environment variables for deployment
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_PORT=${PORT:-8501}

echo "GenChem deployment setup complete!"
echo "Main app file: streamlit_app/app.py"
echo "Run with: streamlit run streamlit_app/app.py"
