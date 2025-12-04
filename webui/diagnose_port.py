#!/usr/bin/env python3
"""
Port diagnostic tool for Kronos WebUI
Helps identify port access issues
"""

import socket
import subprocess
import sys

def check_port(port):
    """Check if a port is available"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', port))
            return True, "Available"
    except PermissionError:
        return False, "Permission denied (may need administrator rights)"
    except OSError as e:
        if "address already in use" in str(e).lower():
            return False, "Already in use"
        else:
            return False, f"Error: {e}"


def find_process_on_port(port):
    """Find which process is using a port"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=5
        )

        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    return pid
        return None
    except Exception as e:
        return f"Error: {e}"

def main():
    """Main diagnostic function"""
    print("=" * 70)
    print("Kronos WebUI Port Diagnostic Tool")
    print("=" * 70)

    # Test common ports
    test_ports = [5000, 7070, 8080, 8000, 8888, 3000]

    print("\n📊 Testing common ports:\n")
    print(f"{'Port':<10} {'Status':<15} {'Details':<40}")
    print("-" * 70)

    available_ports = []
    for port in test_ports:
        available, details = check_port(port)
        status = "✅ Available" if available else "❌ Not Available"
        print(f"{port:<10} {status:<15} {details:<40}")

        if available:
            available_ports.append(port)
        elif "in use" in details:
            pid = find_process_on_port(port)
            if pid:
                print(f"           → Process ID: {pid}")

    # Recommendations
    print("\n" + "=" * 70)
    print("📋 Recommendations:")
    print("=" * 70)

    if available_ports:
        print(f"\n✅ Available ports found: {', '.join(map(str, available_ports))}")
        print(f"\nYou can use one of these ports:")
        print(f"  python start_simple.py")
        print(f"  (Edit start_simple.py and change port to {available_ports[0]})")
    else:
        print("\n⚠️  No available ports found!")
        print("\nPossible solutions:")
        print("  1. Run this script as Administrator")
        print("  2. Check Windows Firewall settings")
        print("  3. Try using a different port range")

    # Windows Firewall check
    print("\n" + "=" * 70)
    print("🔥 Windows Firewall Quick Check:")
    print("=" * 70)
    print("If ports show 'Permission denied', try:")
    print("  1. Run Command Prompt as Administrator")
    print("  2. Run: python start_simple.py")
    print("  3. Allow access when Windows Firewall prompts")

    # Additional info
    print("\n" + "=" * 70)
    print("📖 Additional Information:")
    print("=" * 70)
    print("To find what's using a port:")
    print("  netstat -ano | findstr :<PORT>")
    print("\nTo kill a process by PID:")
    print("  taskkill /PID <PID> /F")
    print("\nTo temporarily disable firewall (not recommended):")
    print("  netsh advfirewall set allprofiles state off")

    return 0 if available_ports else 1

if __name__ == "__main__":
    exit(main())
