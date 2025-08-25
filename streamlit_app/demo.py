#!/usr/bin/env python3
"""
Demo script to test REINVENT4 Streamlit Web Application
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        print("✅ Streamlit is installed")
        return True
    except ImportError:
        print("❌ Streamlit is not installed")
        return False

def check_reinvent_available():
    """Check if REINVENT4 is available"""
    try:
        # Add parent directory to path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        from reinvent import version
        print(f"✅ REINVENT4 is available (version {version.__version__})")
        return True
    except ImportError:
        print("❌ REINVENT4 is not available")
        return False

def create_sample_data():
    """Create sample data files for testing"""
    app_dir = Path(__file__).parent
    data_dir = app_dir / "sample_data"
    data_dir.mkdir(exist_ok=True)
    
    # Sample SMILES files
    samples = {
        "molecules.smi": [
            "CCO",
            "c1ccccc1",
            "CC(=O)O",
            "CCN(CC)CC",
            "c1ccncc1"
        ],
        "scaffolds.smi": [
            "c1ccc([*])cc1",
            "c1ccnc([*])c1",
            "CC([*])C=O"
        ],
        "fragments.smi": [
            "c1ccccc1|CCO",
            "c1ccncc1|CC(=O)O",
            "CCN|c1ccccc1"
        ]
    }
    
    for filename, smiles_list in samples.items():
        filepath = data_dir / filename
        with open(filepath, 'w') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")
        print(f"✅ Created sample file: {filepath}")
    
    return data_dir

def test_app_launch():
    """Test launching the Streamlit app"""
    app_dir = Path(__file__).parent
    app_file = app_dir / "app.py"
    
    if not app_file.exists():
        print("❌ app.py not found")
        return False
    
    print("🚀 Testing app launch (will run for 10 seconds)...")
    
    try:
        # Launch the app in background
        process = subprocess.Popen([
            "streamlit", "run", str(app_file),
            "--server.port", "8502",  # Use different port for testing
            "--server.headless", "true"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Test if the app responds
        try:
            response = requests.get("http://localhost:8502", timeout=5)
            if response.status_code == 200:
                print("✅ App launched successfully and is responding")
                success = True
            else:
                print(f"⚠️ App responded with status code: {response.status_code}")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"❌ App is not responding: {e}")
            success = False
        
        # Terminate the process
        process.terminate()
        process.wait(timeout=5)
        
        return success
        
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

def main():
    """Run demo tests"""
    print("🧪 REINVENT4 Streamlit Web Application Demo")
    print("=" * 50)
    
    # Check prerequisites
    print("\n📋 Checking Prerequisites...")
    streamlit_ok = check_streamlit_installed()
    reinvent_ok = check_reinvent_available()
    
    if not streamlit_ok:
        print("\n💡 To install Streamlit: pip install streamlit")
    
    if not reinvent_ok:
        print("\n💡 Make sure REINVENT4 is properly installed")
    
    # Create sample data
    print("\n📁 Creating Sample Data...")
    sample_dir = create_sample_data()
    
    # Test app launch
    if streamlit_ok:
        print("\n🚀 Testing Application...")
        app_works = test_app_launch()
        
        if app_works:
            print("\n🎉 Demo completed successfully!")
            print(f"\n📂 Sample data created in: {sample_dir}")
            print("\n🌐 To launch the full application:")
            print("   python launch.py")
            print("   or")
            print("   streamlit run app.py")
        else:
            print("\n⚠️ Demo completed with issues. Check the error messages above.")
    else:
        print("\n⏭️ Skipping app test (Streamlit not installed)")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
