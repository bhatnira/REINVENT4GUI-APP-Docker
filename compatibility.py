#!/usr/bin/env python3
"""
Compatibility layer for REINVENT4-APP Docker deployment
This module provides fallbacks for missing dependencies
"""

import warnings
import sys
from typing import Optional, Any

class DependencyHandler:
    """Handles missing dependencies gracefully"""
    
    def __init__(self):
        self.missing_deps = []
        self.alternatives = {}
    
    def check_and_import(self, module_name: str, alternative: Optional[str] = None):
        """Check if module is available and import it, or use alternative"""
        try:
            module = __import__(module_name)
            return module
        except ImportError:
            self.missing_deps.append(module_name)
            if alternative:
                self.alternatives[module_name] = alternative
                warnings.warn(f"Module {module_name} not available, using {alternative}")
            else:
                warnings.warn(f"Module {module_name} not available, some features may be disabled")
            return None
    
    def get_missing_deps(self):
        """Return list of missing dependencies"""
        return self.missing_deps
    
    def get_alternatives(self):
        """Return dict of alternatives for missing dependencies"""
        return self.alternatives

# Global dependency handler
dep_handler = DependencyHandler()

# Check for optional dependencies
OPENEYE_AVAILABLE = dep_handler.check_and_import('openeye', 'rdkit') is not None
CHEMPROP_AVAILABLE = dep_handler.check_and_import('chemprop') is not None
MMPDB_AVAILABLE = dep_handler.check_and_import('mmpdb') is not None

def get_molecule_drawer():
    """Get the best available molecule drawing library"""
    if OPENEYE_AVAILABLE:
        try:
            from openeye import oedepict  # type: ignore
            return 'openeye'
        except ImportError:
            pass
    
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Draw  # type: ignore
        return 'rdkit'
    except ImportError:
        return None

def get_property_predictor():
    """Get the best available property prediction library"""
    if CHEMPROP_AVAILABLE:
        return 'chemprop'
    else:
        return 'rdkit_descriptors'

def show_compatibility_info():
    """Display compatibility information"""
    print("🔧 REINVENT4-APP Compatibility Information")
    print("=" * 50)
    
    missing = dep_handler.get_missing_deps()
    alternatives = dep_handler.get_alternatives()
    
    if not missing:
        print("✅ All dependencies are available!")
    else:
        print("⚠️  Some optional dependencies are missing:")
        for dep in missing:
            alt = alternatives.get(dep, "Feature disabled")
            print(f"   - {dep}: {alt}")
    
    print("\n📋 Available features:")
    print(f"   - Molecule drawing: {get_molecule_drawer() or 'Limited'}")
    print(f"   - Property prediction: {get_property_predictor()}")
    print(f"   - OpenEye tools: {'Yes' if OPENEYE_AVAILABLE else 'No'}")
    print(f"   - ChemProp: {'Yes' if CHEMPROP_AVAILABLE else 'No'}")

if __name__ == "__main__":
    show_compatibility_info()
