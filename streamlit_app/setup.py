#!/usr/bin/env python3
"""
Setup script for REINVENT4 Streamlit Web Application
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 10:
        print(f"✅ Python {major}.{minor} is compatible")
        return True
    else:
        print(f"❌ Python {major}.{minor} is not compatible. Need Python 3.10 or higher.")
        return False

def check_reinvent_installation():
    """Check if REINVENT4 is installed"""
    try:
        # Try to import from parent directory
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from reinvent import version
        print(f"✅ REINVENT4 found (version {version.__version__})")
        return True
    except ImportError:
        print("⚠️ REINVENT4 not found in parent directory")
        
        # Try system-wide installation
        try:
            import reinvent
            print("✅ REINVENT4 found (system installation)")
            return True
        except ImportError:
            print("❌ REINVENT4 not found")
            return False

def install_dependencies():
    """Install Python dependencies"""
    app_dir = Path(__file__).parent
    requirements_file = app_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r {requirements_file}",
        "Installing Python dependencies"
    )

def create_directories():
    """Create necessary directories"""
    app_dir = Path(__file__).parent
    directories = [
        app_dir / "logs",
        app_dir / "uploads",
        app_dir / "downloads", 
        app_dir / "configs",
        app_dir / "sample_data"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_sample_configs():
    """Create sample configuration files"""
    app_dir = Path(__file__).parent
    configs_dir = app_dir / "configs"
    
    sample_configs = {
        "sampling_example.toml": '''# REINVENT4 sampling configuration example
run_type = "sampling"
device = "cuda:0"

[parameters]
model_file = "priors/reinvent.prior"
output_file = "sampling_results.csv"
num_smiles = 100
unique_molecules = true
randomize_smiles = true
''',
        
        "scoring_example.toml": '''# REINVENT4 scoring configuration example
run_type = "scoring"

[parameters]
smiles_file = "compounds.smi"
output_csv = "scoring_results.csv"

[scoring]
type = "geometric_mean"
parallel = false

[[scoring.component]]
[scoring.component.molecular_weight]
[[scoring.component.molecular_weight.endpoint]]
name = "MW"
weight = 1.0
transform.type = "step"
transform.high = 500.0
transform.low = 200.0
''',
        
        "transfer_learning_example.toml": '''# REINVENT4 transfer learning configuration example
run_type = "transfer_learning"
device = "cuda:0"
tb_logdir = "tb_TL"

[parameters]
num_epochs = 10
batch_size = 64
input_model_file = "priors/reinvent.prior"
smiles_file = "training_compounds.smi"
output_model_file = "TL_model.ckpt"
validation_smiles_file = "validation_compounds.smi"
'''
    }
    
    for filename, content in sample_configs.items():
        config_file = configs_dir / filename
        with open(config_file, 'w') as f:
            f.write(content)
        print(f"✅ Created sample config: {config_file}")
    
    return True

def setup_environment():
    """Set up environment variables"""
    app_dir = Path(__file__).parent
    env_file = app_dir / ".env.example"
    
    env_content = """# REINVENT4 Web Application Environment Variables

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8502
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# REINVENT4 Configuration
REINVENT_DEVICE=cuda:0
REINVENT_LOG_LEVEL=info

# Optional: External service endpoints
# CHEMBL_URL=https://www.ebi.ac.uk/chembl/api
# PUBCHEM_URL=https://pubchem.ncbi.nlm.nih.gov/rest

# Optional: GPU memory settings
# CUDA_VISIBLE_DEVICES=0
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created environment template: {env_file}")
    print("   Copy to .env and customize as needed")
    
    return True

def test_installation():
    """Test if everything is working"""
    print("\n🧪 Testing Installation...")
    
    # Test imports
    try:
        import streamlit
        print("✅ Streamlit import successful")
    except ImportError:
        print("❌ Streamlit import failed")
        return False
    
    try:
        import pandas
        print("✅ Pandas import successful")
    except ImportError:
        print("❌ Pandas import failed")
        return False
    
    try:
        import plotly
        print("✅ Plotly import successful")
    except ImportError:
        print("❌ Plotly import failed")
        return False
    
    # Test app file
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"
    
    if app_file.exists():
        print("✅ Main application file found")
    else:
        print("❌ Main application file not found")
        return False
    
    print("✅ Installation test completed successfully")
    return True

def print_next_steps():
    """Print instructions for next steps"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Ensure REINVENT4 models are available in the 'priors' directory")
    print("2. Customize configuration in .env file if needed")
    print("3. Launch the application:")
    print("   python launch.py")
    print("   or")
    print("   streamlit run app.py")
    print("\n🌐 The application will be available at http://localhost:8502")
    print("\n📚 See README.md for detailed usage instructions")

def main():
    """Main setup function"""
    print("🚀 REINVENT4 Streamlit Web Application Setup")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        return 1
    
    # Check REINVENT4
    reinvent_available = check_reinvent_installation()
    if not reinvent_available:
        print("⚠️ REINVENT4 not found. Make sure it's installed in the parent directory.")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return 1
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        return 1
    
    # Create sample configs
    if not create_sample_configs():
        print("❌ Failed to create sample configurations")
        return 1
    
    # Setup environment
    if not setup_environment():
        print("❌ Failed to setup environment")
        return 1
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return 1
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
