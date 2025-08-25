# 🧪 REINVENT4 Web Interface

A comprehensive Streamlit-based web application for molecular design using REINVENT4.

## 🚀 Quick Start

```bash
# Navigate to the streamlit app directory
cd streamlit_app

# Install dependencies
pip install -r requirements.txt

# Launch the web interface
streamlit run app.py --server.port 8502
```

Then open your browser to: **http://localhost:8502**

## 📋 Features

### 🔬 **Generation Modes**
- **De Novo Generation**: Create completely new molecules from scratch
- **Scaffold Hopping**: Decorate scaffolds with R-groups
- **Linker Design**: Connect molecular fragments with optimal linkers
- **R-Group Replacement**: Replace specific functional groups

### 📈 **Optimization**
- **Molecule Optimization**: Improve existing molecules using RL
- **Transfer Learning**: Fine-tune models for specific tasks
- **Reinforcement Learning**: Multi-objective optimization
- **Library Design**: Generate focused molecular libraries

### 🎯 **Analysis & Utilities**
- **Scoring Functions**: Multi-component scoring systems
- **Results Visualization**: Interactive plots and analysis
- **Configuration Manager**: Save and load experiment configurations
- **File Manager**: Comprehensive file handling and downloads

## 🏗️ Architecture

```
streamlit_app/
├── app.py              # Main application (5900+ lines)
├── requirements.txt    # Python dependencies
├── README.md          # Detailed documentation
├── QUICKSTART.md      # Quick setup guide
├── launch.py          # Alternative launcher
└── .streamlit/        # Streamlit configuration
```

## 📊 Capabilities

### **Input Methods**
- Text input for SMILES strings
- File upload (SDF, CSV, SMI formats)
- Integration with previous results
- Example molecules and templates

### **Output Formats**
- CSV with molecular properties
- JSON for data exchange
- SDF for structure files
- Configuration files for reproducibility

### **Visualization**
- Property distribution plots
- Optimization trajectories
- Similarity analysis
- Interactive molecular tables

## 🛠️ Technical Details

### **Dependencies**
- **Streamlit**: Web framework
- **REINVENT4**: Core molecular generation
- **RDKit**: Chemical informatics
- **Plotly**: Interactive visualizations
- **Pandas/NumPy**: Data processing

### **Models Supported**
- **Reinvent**: De novo generation
- **LibInvent**: Scaffold decoration
- **LinkInvent**: Linker design
- **Mol2Mol**: Molecule optimization

## 🎮 Usage Examples

### **De Novo Generation**
1. Select "🔬 De Novo Generation"
2. Configure model file: `priors/reinvent.prior`
3. Set number of molecules to generate
4. Click "🚀 Generate Molecules"
5. Download results in multiple formats

### **Molecule Optimization**
1. Select "📈 Molecule Optimization"
2. Input starting molecules (SMILES)
3. Configure scoring functions
4. Set optimization parameters
5. Monitor real-time progress

### **Scaffold Decoration**
1. Select "🧬 Scaffold Hopping"
2. Input scaffolds with [*] attachment points
3. Choose decoration mode
4. Generate and analyze results

## 📈 Status

- ✅ **Fully Functional**: All modules implemented
- ✅ **Production Ready**: Comprehensive error handling
- ✅ **Real REINVENT4**: Integration with actual models
- ✅ **Interactive GUI**: User-friendly interface
- ✅ **Multi-format Export**: Flexible output options

## 🔧 Configuration

The app automatically detects:
- Available CUDA devices
- Installed REINVENT4 models
- System capabilities

Configuration options include:
- Model file paths
- Output directories
- Compute devices (CPU/GPU)
- Sampling parameters

## 🚀 Deployment

### **Local Development**
```bash
streamlit run app.py --server.port 8502
```

### **Production Deployment**
```bash
streamlit run app.py --server.port 8502 --server.headless true
```

### **Docker Deployment**
```dockerfile
FROM python:3.9
COPY streamlit_app/ /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.headless=true"]
```

## 📝 Documentation

- `README.md`: Comprehensive setup guide
- `QUICKSTART.md`: Fast setup instructions
- `OVERVIEW.md`: Feature overview
- `STATUS.md`: Development status

## 🤝 Contributing

This is a production-ready implementation of REINVENT4 web interface. The codebase includes:

- Complete module implementations
- Error handling and validation
- Progress monitoring
- Result visualization
- File management system

## 📞 Support

For issues with:
- **REINVENT4 Core**: Check the main REINVENT4 repository
- **Web Interface**: Review the streamlit_app documentation
- **Dependencies**: Ensure all requirements are installed

---

**🧪 Ready for molecular design workflows!** 🚀
