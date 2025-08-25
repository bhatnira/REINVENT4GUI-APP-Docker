#!/usr/bin/env python3
"""
Simple test script to verify REINVENT4-APP is working in Docker
"""

import requests
import time
import sys

def test_health_endpoint(url="http://localhost:8501"):
    """Test if the health endpoint is responding"""
    try:
        response = requests.get(f"{url}/_stcore/health", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

def test_main_app(url="http://localhost:8501"):
    """Test if the main application is responding"""
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

def main():
    """Run all tests"""
    print("🧪 Testing REINVENT4-APP Docker deployment...")
    
    # Test health endpoint
    print("Testing health endpoint...")
    if test_health_endpoint():
        print("✅ Health endpoint is responding")
    else:
        print("❌ Health endpoint is not responding")
        return False
    
    # Test main application
    print("Testing main application...")
    if test_main_app():
        print("✅ Main application is responding")
    else:
        print("⚠️  Main application may still be starting")
    
    print("🎉 Basic tests completed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
