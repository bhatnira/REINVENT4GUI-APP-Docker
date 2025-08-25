#!/usr/bin/env python3
"""
Quick port configuration examples for REINVENT4 Streamlit Web Application
"""

import os
import subprocess
import sys

def show_port_options():
    """Show different ways to run the application on different ports"""
    
    print("🧪 REINVENT4 Streamlit Web Application - Port Configuration")
    print("=" * 60)
    
    print("\n📋 Available Options:")
    print("\n1. Default Port (8502):")
    print("   python launch.py")
    print("   streamlit run app.py")
    print("   🌐 http://localhost:8502")
    
    print("\n2. Custom Port (8080):")
    print("   python launch.py --port 8080")
    print("   streamlit run app.py --server.port 8080")
    print("   🌐 http://localhost:8080")
    
    print("\n3. Custom Port (9000):")
    print("   python launch.py --port 9000")
    print("   streamlit run app.py --server.port 9000")
    print("   🌐 http://localhost:9000")
    
    print("\n4. All Interfaces (accessible from other machines):")
    print("   python launch.py --host 0.0.0.0 --port 8502")
    print("   streamlit run app.py --server.address 0.0.0.0 --server.port 8502")
    print("   🌐 http://your-ip-address:8502")
    
    print("\n5. Development Mode with Auto-reload:")
    print("   python launch.py --dev --port 8503")
    print("   🌐 http://localhost:8503")
    
    print("\n🔧 Configuration Files:")
    print("   .streamlit/config.toml - Default port: 8502")
    print("   Command line arguments override config file settings")
    
    print("\n💡 Tips:")
    print("   - Use ports 8000-9999 for development")
    print("   - Avoid common ports like 80, 443, 22, 3000")
    print("   - Check if port is available: netstat -an | grep :PORT")

def launch_on_port(port):
    """Launch the application on a specific port"""
    
    print(f"🚀 Launching REINVENT4 Web Interface on port {port}")
    
    try:
        subprocess.run([
            "python", "launch.py", "--port", str(port)
        ])
    except KeyboardInterrupt:
        print(f"\n⏹️ Stopped application on port {port}")
    except Exception as e:
        print(f"❌ Error launching on port {port}: {e}")

def main():
    """Main function"""
    
    if len(sys.argv) == 1:
        show_port_options()
    elif len(sys.argv) == 2:
        try:
            port = int(sys.argv[1])
            if 1024 <= port <= 65535:
                launch_on_port(port)
            else:
                print("❌ Port must be between 1024 and 65535")
        except ValueError:
            print("❌ Invalid port number")
    else:
        print("Usage: python port_config.py [PORT]")
        print("       python port_config.py        (show options)")
        print("       python port_config.py 8080   (launch on port 8080)")

if __name__ == "__main__":
    main()
