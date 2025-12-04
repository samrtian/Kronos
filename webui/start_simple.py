#!/usr/bin/env python3
"""
Simple Kronos Web UI startup script
Direct startup without auto port detection
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Main function"""
    print("=" * 60)
    print("🚀 Kronos Web UI - Simple Startup")
    print("=" * 60)

    try:
        from app import app

        print("\n✅ App loaded successfully")
        print("🌐 Starting server on http://127.0.0.1:8080")
        print("💡 Tip: Press Ctrl+C to stop server\n")

        # Start Flask application with minimal configuration
        app.run(
            host='127.0.0.1',      # Only localhost
            port=8080,              # Fixed port
            debug=False,            # Disable debug mode
            use_reloader=False,     # Disable reloader
            threaded=True           # Enable threading
        )

    except ImportError as e:
        print(f"\n❌ Failed to import app: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        return 1

    except OSError as e:
        if "address already in use" in str(e).lower() or "access" in str(e).lower():
            print(f"\n❌ Port 8080 is not accessible: {e}")
            print("\nTry these solutions:")
            print("  1. Run as Administrator (right-click -> Run as administrator)")
            print("  2. Try a different port by editing this file (line 23)")
            print("  3. Check Windows Firewall settings")
            print("  4. Use: netstat -ano | findstr :8080  to find what's using the port")
        else:
            print(f"\n❌ Network error: {e}")
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main() or 0)
