# GenChem - GUI Interface for REINVENT4

**A beautiful, user-friendly web interface for REINVENT4 molecular design capabilities**

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF6B6B?style=flat-square)](https://streamlit.io/)
[![Powered by REINVENT4](https://img.shields.io/badge/Powered%20by-REINVENT4-4285F4?style=flat-square)](https://github.com/MolecularAI/REINVENT4)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square)](https://docker.com)

## 🚀 Quick Start with Docker

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

### Run Locally in 3 Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhatnira/REINVENT4GUI-APP-Docker.git
   cd REINVENT4GUI-APP-Docker
   ```

2. **Start the application:**
   ```bash
   docker-compose up -d
   ```

3. **Open your browser:**
   ```
   http://localhost:8504
   ```

That's it! The REINVENT4 web interface will be running with all dependencies automatically configured.

### Stop the Application
```bash
docker-compose down
```

### Rebuild (if needed)
```bash
docker-compose up -d --build
```

## Overview

GenChem provides an intuitive, web-based graphical user interface (GUI) for the powerful REINVENT4 molecular design platform. This application makes advanced AI-driven drug discovery accessible through a beautiful, iOS-inspired interface that requires no command-line expertise.

## About REINVENT4

This GUI is built on **REINVENT4**, a state-of-the-art platform for AI-driven molecular design and drug discovery developed by MolecularAI.

### Academic Reference

**Please cite the original REINVENT4 paper when using this GUI:**

> Guo, J., Fialková, V., Coley, J.D. et al. REINVENT4: Modern AI–driven generative molecule design. *J Cheminform* 16, 20 (2024). https://doi.org/10.1186/s13321-024-00812-5

### Source Repository

**Original REINVENT4 implementation:**
- GitHub: https://github.com/MolecularAI/REINVENT4
- Documentation: Available in the main repository

## Features

GenChem provides a complete GUI interface for all major REINVENT4 capabilities:

### 🧬 Molecular Design Modules
- **De Novo Generation Pipeline** - Complete molecule generation workflow
- **Scaffold Hopping & R-Group Replacement** - LibInvent-based scaffold decoration
- **Molecular Transformation** - Mol2Mol structure-to-structure translation
- **Link Discovery** - LinkInvent for fragment linking

### 🎯 Advanced Capabilities
- **Transfer Learning** - Adapt pre-trained models to specific datasets
- **Reinforcement Learning** - Optimize molecules for desired properties
- **Staged Learning** - Multi-stage optimization workflows
- **Curriculum Learning** - Progressive training strategies

### 📊 Analysis & Visualization
- **Interactive Molecular Visualization** - 2D/3D structure rendering
- **Real-time Analytics** - Generation statistics and metrics
- **Progress Monitoring** - Live training and sampling progress
- **Results Export** - Multiple output formats (SDF, CSV, JSON)

### 🔧 Configuration Management
- **Visual Config Builder** - No-code configuration creation
- **Template Library** - Pre-built configurations for common tasks
- **Parameter Optimization** - Guided hyperparameter tuning
- **Batch Processing** - Multiple configuration management

## Installation

### Prerequisites

1. **Python 3.8+** with pip
2. **REINVENT4 dependencies** - Install from the original repository
3. **Git** for cloning the repository

### Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/genchem-gui.git
   cd genchem-gui
   ```

2. **Install REINVENT4:**
   ```bash
   # Clone and install REINVENT4
   git clone https://github.com/MolecularAI/REINVENT4.git
   cd REINVENT4
   pip install -e .
   cd ..
   ```

3. **Install additional GUI dependencies:**
   ```bash
   pip install streamlit plotly pandas numpy
   ```

4. **Launch GenChem:**
   ```bash
   streamlit run streamlit_app/app.py --server.port 8502
   ```

5. **Access the interface:**
   Open your browser to `http://localhost:8502`

## 🐳 Docker Deployment

### Quick Docker Setup

1. **Build and run with Docker:**
   ```bash
   docker build -t reinvent4-app .
   docker run -p 8501:8501 reinvent4-app
   ```

2. **Or use Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Access the application:**
   Open `http://localhost:8501`

### 🌐 Deploy to Render.com

1. **Fork this repository** to your GitHub account

2. **Create a new Web Service** on Render.com:
   - Connect your GitHub repository
   - Select "Docker" as the environment
   - Use automatic deployment with the included `render.yaml`

3. **Your app will be live** at `https://your-app-name.onrender.com`

📋 **See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for detailed deployment instructions**

### Docker Troubleshooting

#### Common Issues and Solutions

**🔧 Container Build Failures:**
```bash
# Clear Docker cache and rebuild
docker system prune -a
docker build --no-cache -t reinvent4-app .
```

**🐍 Python Dependencies:**
```bash
# Check dependency versions inside container
docker run --rm reinvent4-app python -c "import torch; print(torch.__version__)"
docker run --rm reinvent4-app python -c "import rdkit; print(rdkit.__version__)"
```

**🧠 REINVENT4 Import Issues:**
```bash
# Test REINVENT4 functionality
docker run --rm reinvent4-app python -c "
import sys
try:
    from reinvent import Reinvent
    print('✅ REINVENT4 imported successfully')
except Exception as e:
    print(f'❌ REINVENT4 import failed: {e}')
"
```

**🏥 Health Check Failures:**
```bash
# Test app health endpoint
curl http://localhost:8504/health

# Check container logs
docker logs $(docker ps -q --filter ancestor=reinvent4-app)
```

**💾 Memory Issues:**
```bash
# Run with increased memory limit
docker run -p 8504:8504 -m 4g reinvent4-app
```

**🔌 Port Conflicts:**
```bash
# Use different port if 8504 is busy
docker run -p 8505:8504 reinvent4-app
# Then access at http://localhost:8505
```

#### Performance Optimization

**For better performance on local development:**
```bash
# Enable BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker build -t reinvent4-app .

# Use bind mounts for development
docker run -v $(pwd):/app -p 8504:8504 reinvent4-app
```

## Usage

### Quick Start

1. **Select a Module** - Choose from the navigation tabs (De Novo, Scaffold Hopping, etc.)
2. **Configure Parameters** - Use the intuitive forms to set up your job
3. **Upload Data** - Provide input molecules, scaffolds, or datasets as needed
4. **Run Generation** - Execute your molecular design workflow
5. **Analyze Results** - View and download generated molecules with analytics

### Example Workflows

#### De Novo Molecule Generation
1. Navigate to "De Novo Generation"
2. Upload a reference dataset or use built-in examples
3. Configure generation parameters (model type, sampling settings)
4. Set optimization objectives (ADMET properties, similarity targets)
5. Run the pipeline and monitor progress
6. Download results in your preferred format

#### Scaffold Decoration
1. Go to "Scaffold Hopping"
2. Upload your scaffold structures
3. Choose decoration or hopping mode
4. Set R-group constraints and properties
5. Generate decorated molecules
6. Export results with detailed analytics

## Configuration Files

GenChem generates standard REINVENT4 configuration files that are fully compatible with the command-line interface. All configurations can be:

- **Exported** for use with command-line REINVENT4
- **Imported** from existing REINVENT4 setups
- **Shared** with team members
- **Version controlled** with your projects

## System Requirements

### Minimum Requirements
- **CPU:** 4+ cores recommended
- **RAM:** 8GB minimum, 16GB+ recommended
- **Storage:** 10GB+ free space
- **GPU:** Optional but recommended for training (CUDA-compatible)

### Recommended Setup
- **CPU:** 8+ cores (Intel/AMD)
- **RAM:** 32GB+ for large datasets
- **GPU:** NVIDIA RTX series or Tesla with 8GB+ VRAM
- **Storage:** SSD with 50GB+ free space

## Contributing

We welcome contributions to improve GenChem! Please:

1. Fork this repository
2. Create a feature branch
3. Make your improvements
4. Add tests if applicable
5. Submit a pull request

## License

This GUI interface is provided under the same license terms as REINVENT4. Please refer to the original REINVENT4 repository for licensing information.

## Acknowledgments

- **MolecularAI Team** - For developing the exceptional REINVENT4 platform
- **REINVENT4 Contributors** - All researchers and developers who contributed to the core platform
- **Open Source Community** - For the tools and libraries that made this GUI possible

## Citations

If you use GenChem in your research, please cite both this GUI and the original REINVENT4 work:

### REINVENT4 (Required)
```bibtex
@article{guo2024reinvent4,
  title={REINVENT4: Modern AI–driven generative molecule design},
  author={Guo, Jiazhen and Fialkov{\'a}, Vendula and Coley, John D and Patronov, Alexey and Holm, Esben Jannik and Engkvist, Ola and Bjerrum, Esben Jannik and Kogej, Thierry and Tyrchan, Christian and Voronov, Alexey and others},
  journal={Journal of Cheminformatics},
  volume={16},
  number={1},
  pages={20},
  year={2024},
  publisher={BioMed Central},
  doi={10.1186/s13321-024-00812-5},
  url={https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5}
}
```

### GenChem GUI (Optional)
```bibtex
@software{genchem_gui,
  title={GenChem: A GUI Interface for REINVENT4 Molecular Design},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-username/genchem-gui},
  note={Built on REINVENT4 by MolecularAI}
}
```

## Support

- **REINVENT4 Issues** - Report to the [original repository](https://github.com/MolecularAI/REINVENT4/issues)
- **GUI-specific Issues** - Report to this repository's issues
- **Documentation** - See the original REINVENT4 documentation for algorithm details

## Links

- 📄 **REINVENT4 Paper:** https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5
- 🐙 **REINVENT4 GitHub:** https://github.com/MolecularAI/REINVENT4
- 🌐 **GenChem Demo:** [Your demo URL]
- 📖 **Documentation:** [Your docs URL]

---

**GenChem** - Making AI-driven molecular design accessible to everyone, built on the robust foundation of REINVENT4.
