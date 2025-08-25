# REINVENT4 Docker Deployment

A complete Docker containerization of the REINVENT4 molecular design platform, ready for deployment on Render.com and other cloud platforms.

## 🚀 Quick Start

### Local Development

1. **Start the application**:
   ```bash
   docker-compose up -d
   ```

2. **Access the application**:
   - Open your browser to: http://localhost:8504
   - Health check: http://localhost:8504/_stcore/health

3. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Production Deployment (Render.com)

The application is pre-configured for automatic deployment on Render.com:

1. Connect your repository to Render.com
2. The `render.yaml` file will automatically configure the deployment
3. Push your code to trigger deployment

## 📁 Project Structure

```
REINVENT4-APP/
├── streamlit_app/           # Streamlit web application
├── reinvent/               # REINVENT4 core modules
├── configs/                # Configuration files
├── docker-requirements.txt # Containerized dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Local development setup
├── render.yaml             # Render.com deployment config
├── start.sh                # Container startup script
└── README_DOCKER.md        # This file
```

## 🐳 Docker Configuration

### Key Files

- **`Dockerfile`**: Multi-stage build with optimized layers
- **`docker-requirements.txt`**: Simplified dependencies for containerization
- **`start.sh`**: Startup script with dependency checking
- **`docker-compose.yml`**: Local development environment

### Dependencies

**Core Stack (✅ Working)**:
- Python 3.10
- Streamlit 1.48.1
- PyTorch 2.8.0+cpu
- TorchVision 0.23.0
- RDKit 2025.03.5
- Pandas 2.3.2
- NumPy 2.2.6

**REINVENT4 Status**:
- ✅ Basic validation modules working
- ⚠️ Advanced modules require `mmpdblib` (optional)
- ✅ Web interface fully functional

## 🔧 Development Commands

### Building and Testing

```bash
# Build the Docker image
docker build -t reinvent4-app:latest .

# Run with docker-compose
docker-compose up -d --build

# Run tests
python3 test_full_deployment.py

# Check logs
docker-compose logs

# Access container shell
docker exec -it reinvent4-app-reinvent4-app-1 bash
```

### Debugging

```bash
# Check container status
docker ps

# View real-time logs
docker-compose logs -f

# Test dependencies inside container
docker exec reinvent4-app-reinvent4-app-1 python -c "import torch, rdkit; print('OK')"
```

## 🌐 Deployment Configuration

### Render.com Settings

The `render.yaml` file configures:
- **Service Type**: Web Service
- **Environment**: Docker
- **Port**: 8504
- **Health Check**: `/_stcore/health`
- **Auto Deploy**: On git push
- **Resources**: 1GB RAM, 0.5 CPU

### Environment Variables

No sensitive environment variables required for basic functionality.

### Health Monitoring

- **Health Endpoint**: `/_stcore/health`
- **Expected Response**: `ok`
- **Startup Time**: ~30-60 seconds

## 📊 Testing and Validation

### Automated Tests

```bash
# Run complete deployment test
python3 test_full_deployment.py
```

**Test Coverage**:
- ✅ Container health
- ✅ Web application accessibility
- ✅ Core dependency imports
- ✅ REINVENT4 basic functionality

### Manual Testing

1. **Web Interface**: Visit http://localhost:8504
2. **Health Check**: `curl http://localhost:8504/_stcore/health`
3. **Dependency Check**: Check container logs for startup messages

## 🔄 Continuous Deployment

### Git Workflow

1. **Development**:
   ```bash
   # Make changes
   git add .
   git commit -m "Update feature"
   ```

2. **Local Testing**:
   ```bash
   docker-compose up -d --build
   python3 test_full_deployment.py
   ```

3. **Deploy**:
   ```bash
   git push origin main  # Triggers Render.com deployment
   ```

### Deployment Pipeline

1. **Git Push** → Render.com detects changes
2. **Build** → Docker image built from Dockerfile
3. **Deploy** → New version goes live automatically
4. **Health Check** → Automatic validation

## 🛠️ Troubleshooting

### Common Issues

**Container won't start**:
```bash
docker-compose logs  # Check error messages
docker-compose down && docker-compose up -d --build
```

**Import errors**:
```bash
# Check which dependencies are missing
docker exec reinvent4-app-reinvent4-app-1 pip list
```

**Performance issues**:
- Increase memory allocation in `render.yaml`
- Use CPU-only PyTorch build (already configured)

### Known Limitations

- **OpenEye**: Proprietary chemistry toolkit not included (use RDKit)
- **GPU Support**: CPU-only PyTorch for cloud deployment
- **MMP Analysis**: Some features require additional setup

## 🎯 Production Considerations

### Security
- No sensitive data in container
- Uses official Python base image
- Minimal attack surface

### Performance
- Optimized for CPU-only inference
- Streamlined dependencies
- Efficient Docker layers

### Monitoring
- Built-in health checks
- Structured logging
- Error handling

## 📝 Next Steps

1. **Optional Enhancements**:
   - Add authentication if needed
   - Configure custom domain
   - Set up monitoring/alerting

2. **Advanced Features**:
   - Add GPU support for heavy workloads
   - Integrate with external databases
   - Add API endpoints

3. **Scaling**:
   - Configure horizontal scaling on Render.com
   - Add load balancing if needed
   - Implement caching strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test locally with Docker
4. Submit a pull request

## 📄 License

[Include your license information here]

---

**Status**: ✅ Production Ready  
**Last Updated**: $(date)  
**Docker Version**: 24.0+  
**Python Version**: 3.10  
**Streamlit Version**: 1.48.1
