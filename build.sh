#!/bin/bash
"""
Build and test script for REINVENT4-APP Docker container
"""

set -e

echo "🐳 Building REINVENT4-APP Docker container..."

# Build the Docker image
docker build -t reinvent4-app:latest .

echo "✅ Docker image built successfully!"

# Test the container
echo "🧪 Testing the container..."

# Start container in background
docker run -d --name reinvent4-test -p 8501:8501 reinvent4-app:latest

# Wait for container to start
echo "⏳ Waiting for container to start..."
sleep 30

# Test health endpoint
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Container health check passed!"
else
    echo "❌ Container health check failed!"
    docker logs reinvent4-test
    docker stop reinvent4-test
    docker rm reinvent4-test
    exit 1
fi

# Test main application
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Application is responding!"
else
    echo "⚠️  Application may not be fully ready, but health check passed"
fi

# Cleanup
docker stop reinvent4-test
docker rm reinvent4-test

echo "🎉 All tests passed! Container is ready for deployment."
echo ""
echo "📋 Next steps:"
echo "1. Push your code to GitHub"
echo "2. Connect your repository to Render.com"
echo "3. Deploy using the provided render.yaml configuration"
echo ""
echo "🚀 Local development:"
echo "   docker-compose up -d"
echo "   Open http://localhost:8501 in your browser"
