#!/usr/bin/env python3
"""
Dependency check script for REINVENT4 Docker deployment
This script checks for missing proprietary dependencies and provides alternatives
"""

import importlib
import sys
import warnings

def check_dependency(module_name, alternative=None, required=True):
    """Check if a dependency is available and suggest alternatives if not"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is available")
        return True
    except ImportError:
        if required:
            print(f"❌ {module_name} is missing")
            if alternative:
                print(f"   💡 Alternative: {alternative}")
        else:
            print(f"⚠️  {module_name} is optional and not available")
            if alternative:
                print(f"   💡 Alternative: {alternative}")
        return False

def main():
    """Main dependency check"""
    print("🔍 Checking REINVENT4 dependencies for Docker deployment...\n")
    
    # Core dependencies that should be available
    core_deps = {
        'streamlit': None,
        'pandas': None,
        'numpy': None,
        'torch': None,
        'rdkit': None,
    }
    
    # Optional/proprietary dependencies with alternatives
    optional_deps = {
        'openeye': 'Use RDKit for structure handling and visualization',
        'chemprop': 'Use alternative property prediction models',
        'mmpdb': 'Use RDKit for matched molecular pairs analysis',
    }
    
    print("Core Dependencies:")
    core_available = 0
    for dep, alt in core_deps.items():
        if check_dependency(dep, alt, required=True):
            core_available += 1
    
    print(f"\nCore dependencies available: {core_available}/{len(core_deps)}")
    
    print("\nOptional Dependencies:")
    for dep, alt in optional_deps.items():
        check_dependency(dep, alt, required=False)
    
    print("\n" + "="*60)
    if core_available == len(core_deps):
        print("✅ All core dependencies are available!")
        print("🚀 Ready for Docker deployment")
    else:
        print("❌ Some core dependencies are missing")
        print("⚠️  Docker deployment may have limited functionality")
    
    print("\n📝 Notes for Render.com deployment:")
    print("- OpenEye toolkits require a license and are not included")
    print("- RDKit provides open-source alternatives for most functionality")
    print("- Some advanced features may be disabled without proprietary tools")

if __name__ == "__main__":
    main()
