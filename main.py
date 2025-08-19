#!/usr/bin/env python3
"""
Mailinflow - Simple Email Triage System

Main entry point that launches the Streamlit interface for manual email triage.
"""

import subprocess
import sys
import os
from pathlib import Path

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
    
    # Launch the interface
    print("🚀 Launching Streamlit interface...")
    print("📱 Opening in browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_triage.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
