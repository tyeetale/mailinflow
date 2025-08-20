#!/usr/bin/env python3
"""
Mailinflow - Simple Email Triage System

Main entry point that launches the Streamlit interface for manual email triage.
"""

import subprocess
import sys
import os
import socket
from pathlib import Path

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return start_port  # Fallback to original port

def main():
    """Launch the Streamlit interface."""
    print("📧 Mailinflow - Email Triage System")
    print("=" * 40)
    
    # Check if Streamlit is available
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Find available port
    port = find_available_port()
    print(f"🔌 Using port: {port}")
    
    # Launch the interface
    print("🚀 Launching Streamlit interface...")
    print(f"📱 Opening in browser at http://localhost:{port}")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_triage.py",
            "--server.port", str(port)
        ])
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
