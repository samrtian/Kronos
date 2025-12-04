#!/usr/bin/env python3
"""
Kronos Web UI startup script
"""

import os
import sys
import subprocess
import webbrowser
import time
import socket



def find_free_port(start_port=7070, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

def check_dependencies():
    """Check if dependencies are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import plotly
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installation completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Dependencies installation failed")
        return False

def main():
    """Main function"""
    print("🚀 Starting Kronos Web UI...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\nAuto-install dependencies? (y/n): ", end="")
        if input().lower() == 'y':
            if not install_dependencies():
                return
        else:
            print("Please manually install dependencies and retry")
            return
    
    # Check model availability
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model import Kronos, KronosTokenizer, KronosPredictor
        print("✅ Kronos model library available")
        model_available = True
    except ImportError:
        print("⚠️  Kronos model library not available, will use simulated prediction")
        model_available = False
    
    # Start Flask application
    print("\n🌐 Starting Web server...")
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Find a free port
    print("\n🔍 Finding available port...")
    port = find_free_port(start_port=7070, max_attempts=20)

    if port is None:
        print("❌ No available ports found in range 7070-8100")
        print("Please close some applications and try again")
        return

    print(f"✅ Found available port: {port}")

    # Start server
    try:
        from app import app
        print("\n🌐 Starting Web server...")
        print(f"✅ Web server started successfully!")
        print(f"🌐 Access URL: http://localhost:{port}")
        print("💡 Tip: Press Ctrl+C to stop server")

        # Auto-open browser
        time.sleep(1)
        webbrowser.open(f'http://localhost:{port}')

        # Start Flask application
        app.run(debug=True, host='127.0.0.1', port=port, use_reloader=False)

    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPlease check if port {port} is accessible or try running as administrator")

if __name__ == "__main__":
    main()
