# REINVENT4 Streamlit Web Application

A comprehensive web-based graphical user interface for REINVENT4, providing intuitive access to all molecular generation and optimization capabilities.

## Features

### 🔬 Molecule Generation
- **De Novo Generation**: Create completely new molecules from scratch using trained REINVENT models
- **Scaffold Hopping**: Find alternative scaffolds while maintaining key molecular features
- **Linker Design**: Connect molecular fragments with optimal linkers using LinkInvent
- **R-Group Replacement**: Modify specific positions in molecules while preserving the core structure

### 📈 Optimization Strategies
- **Transfer Learning**: Fine-tune pre-trained models on specific molecular datasets
- **Reinforcement Learning**: Optimize molecules toward custom scoring functions
- **Curriculum Learning**: Multi-stage optimization with progressively refined objectives
- **Multi-objective Optimization**: Balance multiple molecular properties simultaneously

### 🎯 Scoring & Analysis
- **Custom Scoring Functions**: Build multi-component scoring functions with transforms
- **Real-time Visualization**: Monitor optimization progress with interactive plots
- **Property Analysis**: Analyze molecular properties and drug-likeness
- **Batch Processing**: Handle large-scale molecular generation and optimization

### 💾 Data Management
- **Multiple Input Formats**: Support for SMILES, SDF, CSV files
- **Export Options**: Download results in various formats (CSV, SDF, JSON)
- **Configuration Management**: Save and reuse optimization configurations
- **Progress Tracking**: Monitor long-running jobs with progress indicators

## Installation

### Prerequisites
- Python 3.10 or higher
- REINVENT4 installed and configured
- CUDA-compatible GPU (recommended for large-scale generation)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/MolecularAI/REINVENT4.git
   cd REINVENT4
   ```

2. **Install the web application dependencies**:
   ```bash
   cd streamlit_app
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   python launch.py
   ```

   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8502`

### Advanced Installation

For development or customization:

```bash
# Install with development dependencies
python launch.py --install-deps --dev

# Run on a custom port
python launch.py --port 8080

# Run on all interfaces
python launch.py --host 0.0.0.0
```

## Usage Guide

### Getting Started

1. **Home Page**: Overview of REINVENT4 capabilities and system status
2. **Navigation**: Use the sidebar to access different modules
3. **Configuration**: Each module provides intuitive parameter configuration
4. **Execution**: Run generation or optimization with real-time progress tracking
5. **Results**: Analyze and download results in multiple formats

### Module Overview

#### 🔬 De Novo Generation
- Select a trained REINVENT model
- Configure generation parameters (number of molecules, temperature, etc.)
- Generate novel molecules and analyze their properties
- Export results for further analysis

#### 🧬 Scaffold Hopping
- Input scaffold molecules with attachment points marked as [*]
- Choose between scaffold decoration and scaffold replacement
- Configure chemical constraints and property filters
- Generate and evaluate scaffold variants

#### 🔗 Linker Design
- Input fragment pairs separated by "|"
- Set linker length and complexity constraints
- Generate linkers connecting the fragments
- Analyze linker properties and drug-likeness

#### ⚗️ R-Group Replacement
- Input molecules with R-group positions marked
- Define allowed elements and size constraints
- Generate R-group variants with property analysis
- Filter results by drug-likeness criteria

#### 📈 Molecule Optimization
- Choose optimization strategy (RL, TL, Curriculum Learning)
- Define target properties and scoring functions
- Configure learning parameters
- Monitor optimization progress in real-time

### Configuration

The application uses several configuration files:

- **`.streamlit/config.toml`**: Streamlit server and UI configuration
- **`requirements.txt`**: Python dependencies
- **Generated configs**: REINVENT4 configuration files created automatically

### Input Formats

The application supports multiple input formats:

- **SMILES files**: Text files with one SMILES per line
- **SDF files**: Structure-data files with molecular structures
- **CSV files**: Comma-separated files with molecular data
- **Text input**: Direct input of SMILES strings in the web interface

### Output Formats

Results can be downloaded in various formats:

- **CSV**: Tabular data with molecular properties
- **SDF**: Structure files for molecular visualization
- **JSON**: Machine-readable results with metadata
- **Configuration files**: TOML/JSON files for reproducing runs

## Advanced Features

### Custom Scoring Functions

Build complex scoring functions by combining:
- **ADMET properties**: Solubility, permeability, metabolism
- **Physicochemical properties**: Molecular weight, LogP, TPSA
- **Activity models**: QSAR models, docking scores
- **Similarity metrics**: Tanimoto, pharmacophore similarity
- **Synthetic accessibility**: RA score, synthesis difficulty

### Batch Processing

Handle large datasets efficiently:
- Upload files with hundreds of starting molecules
- Configure parallel processing parameters
- Monitor progress with real-time updates
- Export comprehensive results

### Integration with External Tools

The application can integrate with:
- **ChEMBL**: Query bioactivity databases
- **PubChem**: Access chemical structure databases
- **Molecular visualization**: RDKit, PyMOL integration
- **QSAR modeling**: ChemProp, scikit-learn models

## Troubleshooting

### Common Issues

1. **REINVENT4 not found**:
   - Ensure REINVENT4 is properly installed
   - Check that the parent directory contains the reinvent package

2. **CUDA errors**:
   - Verify CUDA installation and compatibility
   - Use CPU mode if GPU is unavailable

3. **Memory issues**:
   - Reduce batch sizes for large generations
   - Close other applications to free memory

4. **Port conflicts**:
   - Use a different port: `python launch.py --port 8080`
   - Check for other running applications

### Performance Optimization

- **Use GPU**: Enable CUDA for faster generation
- **Batch size**: Optimize based on available memory
- **Parallel processing**: Enable for scoring functions
- **Memory management**: Monitor usage during long runs

## Development

### Contributing

To contribute to the web application:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Structure

```
streamlit_app/
├── app.py              # Main application file
├── launch.py           # Launch script
├── requirements.txt    # Dependencies
├── .streamlit/         # Streamlit configuration
│   └── config.toml
└── README.md          # This file
```

### Adding New Features

To add new functionality:

1. **Create new pages**: Add functions like `show_new_feature_page()`
2. **Update navigation**: Add entries to the sidebar menu
3. **Implement backend**: Connect to REINVENT4 functionality
4. **Add visualization**: Use Plotly for interactive charts
5. **Update documentation**: Keep README and help text current

## License

This application is part of REINVENT4 and follows the same license terms.

## Support

For support and questions:

- **Issues**: GitHub Issues for bug reports and feature requests
- **Documentation**: REINVENT4 official documentation
- **Community**: Join the molecular AI community discussions

## Acknowledgments

- **REINVENT4 Team**: Core molecular generation algorithms
- **Streamlit**: Web application framework
- **RDKit**: Chemical structure handling
- **Plotly**: Interactive visualizations

---

**Happy Molecular Generation!** 🧪✨
