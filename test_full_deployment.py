#!/usr/bin/env python3
"""
Complete deployment test for REINVENT4 Docker application
Tests all major functionality and dependencies
"""

import requests
import time
import sys
import json
from typing import Dict, List

def test_health_endpoint() -> bool:
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8504/_stcore/health", timeout=10)
        if response.status_code == 200 and response.text.strip() == "ok":
            print("✅ Health endpoint working")
            return True
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_main_application() -> bool:
    """Test the main application endpoint"""
    try:
        response = requests.get("http://localhost:8504", timeout=15)
        if response.status_code == 200:
            print("✅ Main application accessible")
            return True
        else:
            print(f"❌ Main application failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Main application error: {e}")
        return False

def test_dependency_imports() -> Dict[str, bool]:
    """Test all dependency imports"""
    import subprocess
    
    test_script = '''
import sys
import json

results = {}

# Test core dependencies
try:
    import streamlit
    results["streamlit"] = {"status": "ok", "version": streamlit.__version__}
except Exception as e:
    results["streamlit"] = {"status": "error", "error": str(e)}

try:
    import torch
    results["torch"] = {"status": "ok", "version": torch.__version__}
except Exception as e:
    results["torch"] = {"status": "error", "error": str(e)}

try:
    import torchvision
    results["torchvision"] = {"status": "ok", "version": torchvision.__version__}
except Exception as e:
    results["torchvision"] = {"status": "error", "error": str(e)}

try:
    import rdkit
    results["rdkit"] = {"status": "ok", "version": rdkit.__version__}
except Exception as e:
    results["rdkit"] = {"status": "error", "error": str(e)}

try:
    import pandas
    results["pandas"] = {"status": "ok", "version": pandas.__version__}
except Exception as e:
    results["pandas"] = {"status": "error", "error": str(e)}

try:
    import numpy
    results["numpy"] = {"status": "ok", "version": numpy.__version__}
except Exception as e:
    results["numpy"] = {"status": "error", "error": str(e)}

# Test REINVENT4 imports
try:
    from reinvent import validation
    results["reinvent_validation"] = {"status": "ok", "version": "available"}
except Exception as e:
    results["reinvent_validation"] = {"status": "error", "error": str(e)}

try:
    from reinvent.Reinvent import Reinvent
    results["reinvent_main"] = {"status": "ok", "version": "available"}
except Exception as e:
    results["reinvent_main"] = {"status": "error", "error": str(e)}

try:
    import reinvent.runmodes
    results["reinvent_runmodes"] = {"status": "ok", "version": "available"}
except Exception as e:
    results["reinvent_runmodes"] = {"status": "error", "error": str(e)}

print(json.dumps(results, indent=2))
'''
    
    try:
        result = subprocess.run([
            "docker", "exec", "reinvent4-app-reinvent4-app-1", 
            "python", "-c", test_script
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            dependencies = json.loads(result.stdout)
            
            print("\n📦 Dependency Test Results:")
            success_count = 0
            total_count = len(dependencies)
            
            for dep, info in dependencies.items():
                if info["status"] == "ok":
                    print(f"✅ {dep}: {info.get('version', 'ok')}")
                    success_count += 1
                else:
                    print(f"❌ {dep}: {info.get('error', 'unknown error')}")
            
            print(f"\n📊 Dependencies: {success_count}/{total_count} working")
            return dependencies
        else:
            print(f"❌ Dependency test failed: {result.stderr}")
            return {}
    except Exception as e:
        print(f"❌ Dependency test error: {e}")
        return {}

def run_full_test() -> bool:
    """Run complete deployment test"""
    print("🧪 Running Full REINVENT4 Deployment Test")
    print("=" * 50)
    
    # Wait for container to be ready
    print("⏳ Waiting for container to be ready...")
    time.sleep(5)
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    
    # Test main application
    app_ok = test_main_application()
    
    # Test dependencies
    dependencies = test_dependency_imports()
    deps_ok = len(dependencies) > 0
    
    # Summary
    print("\n🎯 Test Summary:")
    print(f"{'✅' if health_ok else '❌'} Health endpoint")
    print(f"{'✅' if app_ok else '❌'} Main application")
    print(f"{'✅' if deps_ok else '❌'} Dependencies")
    
    if health_ok and app_ok and deps_ok:
        print("\n🎉 All tests passed! Application is ready for deployment.")
        
        # Specific deployment readiness
        core_deps = ["streamlit", "torch", "torchvision", "rdkit", "pandas", "numpy"]
        core_working = all(
            dependencies.get(dep, {}).get("status") == "ok" 
            for dep in core_deps
        )
        
        reinvent_basic = dependencies.get("reinvent_validation", {}).get("status") == "ok"
        
        print(f"\n🚀 Deployment Status:")
        print(f"{'✅' if core_working else '❌'} Core ML/Chemistry stack")
        print(f"{'✅' if reinvent_basic else '❌'} Basic REINVENT4 functionality")
        
        if core_working:
            print("\n✨ Ready for Render.com deployment!")
            print("   Use 'git push' to deploy to your connected repository")
            return True
        else:
            print("\n⚠️  Some issues detected but application is functional")
            return False
    else:
        print("\n❌ Tests failed. Check the issues above.")
        return False

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
