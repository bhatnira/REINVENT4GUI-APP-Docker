# REINVENT4-APP Docker Deployment Guide

## 🐳 Docker Containerization

This document provides instructions for containerizing the REINVENT4-APP and deploying it to Render.com.

## 📋 Prerequisites

- Docker installed on your system
- Git repository access
- Render.com account (for cloud deployment)

## 🚀 Quick Start

### Local Development with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t reinvent4-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 reinvent4-app
   ```

3. **Access the application:**
   Open your browser and navigate to `http://localhost:8501`

### Using Docker Compose (Recommended for Development)

1. **Start the application:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

## 🌐 Render.com Deployment

### Automatic Deployment

1. **Fork the repository** to your GitHub account

2. **Create a new Web Service** on Render.com:
   - Connect your GitHub repository
   - Select the repository: `REINVENT4-APP`
   - Choose `Docker` as the environment
   - Use the provided `render.yaml` configuration

3. **Environment Variables** (automatically set by render.yaml):
   ```
   PORT=8501
   PYTHONUNBUFFERED=1
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_SERVER_ENABLE_CORS=false
   STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
   ```

4. **Deploy** - Render will automatically build and deploy your application

### Manual Deployment

If you prefer manual configuration:

1. **Create New Web Service:**
   - Environment: Docker
   - Repository: Your forked REINVENT4-APP repo
   - Branch: main
   - Dockerfile Path: `./Dockerfile`

2. **Configure Settings:**
   - Instance Type: Starter (can upgrade later)
   - Region: Choose your preferred region
   - Health Check Path: `/_stcore/health`

3. **Add Environment Variables:**
   ```
   PORT = 8501
   PYTHONUNBUFFERED = 1
   STREAMLIT_SERVER_HEADLESS = true
   STREAMLIT_SERVER_ENABLE_CORS = false
   STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION = false
   ```

4. **Deploy**

## 📁 File Structure

```
REINVENT4-APP/
├── Dockerfile                 # Docker container configuration
├── docker-compose.yml        # Local development setup
├── render.yaml               # Render.com deployment config
├── .dockerignore             # Files to exclude from Docker build
├── docker-requirements.txt   # Simplified requirements for container
├── start.sh                  # Container startup script
├── check_dependencies.py     # Dependency validation script
└── streamlit_app/            # Streamlit application
    ├── app.py                # Main application
    └── requirements.txt      # Original requirements
```

## ⚙️ Configuration

### Docker Image Details

- **Base Image:** `python:3.10-slim`
- **Exposed Port:** 8501
- **Working Directory:** `/app`
- **Health Check:** `/_stcore/health`

### Dependencies

The Docker version uses a simplified dependency set:
- ✅ **Included:** RDKit, Streamlit, PyTorch, pandas, plotly
- ❌ **Excluded:** OpenEye toolkits (requires license)
- ❌ **Excluded:** ChemProp (large dependencies)

### Storage

- **Persistent Storage:** 20GB disk mounted at `/app/data`
- **Logs:** Stored in `/app/logs`
- **Outputs:** Stored in `/app/outputs`

## 🔧 Customization

### Resource Requirements

For production use, consider upgrading:

```yaml
# In render.yaml
plan: standard  # or pro
numInstances: 2  # for high availability
disk:
  sizeGB: 50    # for larger datasets
```

### Custom Domain

Add to render.yaml:
```yaml
domains:
  - reinvent4.yourdomain.com
```

### Environment Variables

Add custom variables in Render dashboard or render.yaml:
```yaml
envVars:
  - key: CUSTOM_CONFIG
    value: production
```

## 🐛 Troubleshooting

### Common Issues

1. **Application not starting:**
   - Check logs: `docker logs <container_id>`
   - Verify dependencies: `python check_dependencies.py`

2. **Memory issues:**
   - Upgrade Render plan to Standard or Pro
   - Optimize PyTorch installation

3. **Slow loading:**
   - Enable caching in Streamlit
   - Use CDN for static assets

### Debug Mode

Run with debug logging:
```bash
docker run -e STREAMLIT_LOGGER_LEVEL=debug -p 8501:8501 reinvent4-app
```

## 📊 Monitoring

### Health Checks

The application includes health checks at:
- **Local:** `http://localhost:8501/_stcore/health`
- **Render:** Automatically configured

### Logs

Access logs through:
- **Docker:** `docker logs reinvent4-app`
- **Render:** Dashboard logs section

## 🔐 Security

### Environment Variables

Never commit sensitive data. Use Render's environment variables for:
- API keys
- Database credentials
- License keys

### Network Security

The application runs on port 8501 with:
- CORS disabled for security
- XSRF protection disabled (appropriate for containerized deployment)
- Headless mode enabled

## 🚀 Performance Optimization

### For Production

1. **Use multi-stage builds** for smaller images
2. **Enable caching** for Streamlit components
3. **Optimize PyTorch** installation (CPU-only if no GPU needed)
4. **Use nginx** as reverse proxy for static files

### Scaling

Render.com supports:
- **Horizontal scaling:** Multiple instances
- **Vertical scaling:** Larger instance types
- **Auto-scaling:** Based on CPU/memory usage

## 📞 Support

For issues related to:
- **Docker:** Check container logs and dependency status
- **Render.com:** Consult Render documentation
- **REINVENT4:** Refer to original repository documentation

## 🔄 Updates

To update the deployed application:
1. Push changes to your repository
2. Render will automatically rebuild and deploy
3. Monitor deployment logs for any issues

---

**Note:** This containerized version excludes proprietary dependencies like OpenEye toolkits. Some advanced features may have limited functionality. For full feature access, consider a self-hosted deployment with proper licenses.
