#!/usr/bin/env python3
"""
Mailinflow Manual Triage Interface

Streamlit-based interface for manually triaging difficult emails.
This interface allows users to classify emails that the ML system
couldn't confidently categorize and stores these decisions as training data.
"""

import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os

# Add the current directory to Python path to import config_manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import config_manager, get_available_accounts, get_system_config
from email_triage import CLASSIFICATION_LABELS, MANUAL_TRIAGE_LABELS, FOLDER_MAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Mailinflow Manual Triage",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_training_data() -> Dict[str, Any]:
    """Load existing training data."""
    data_path = Path("email_labels_zeroinbox.jsonl")
    if not data_path.exists():
        return {}
    
    data = {}
    try:
        with data_path.open("r") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'uid' in record:
                        data[record['uid']] = record
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        st.error(f"Error loading training data: {e}")
    
    return data

def save_training_data(data: Dict[str, Any]):
    """Save training data to file."""
    data_path = Path("email_labels_zeroinbox.jsonl")
    try:
        with data_path.open("w") as f:
            for record in data.values():
                f.write(json.dumps(record) + "\n")
        st.success(f"Saved {len(data)} training records")
    except Exception as e:
        st.error(f"Error saving training data: {e}")

def get_email_preview(email_data: Dict[str, Any]) -> str:
    """Generate a preview of email content."""
    subject = email_data.get('subject', 'No Subject')
    sender = email_data.get('from', 'Unknown Sender')
    date = email_data.get('date', 'Unknown Date')
    text = email_data.get('text', '')[:500]  # First 500 characters
    
    preview = f"""
**From:** {sender}
**Subject:** {subject}
**Date:** {date}
**Content:** {text}...
"""
    return preview

def main():
    st.title("📧 Mailinflow Manual Triage Interface")
    st.markdown("---")
    
    # Sidebar for configuration and controls
    with st.sidebar:
        st.header("Configuration")
        
        # Configuration status
        st.success("✅ Environment-based Configuration")
        st.info("Using .env file for configuration")
        
        # Configuration refresh
        if st.button("🔄 Refresh Configuration"):
            config_manager.refresh_config()
            st.rerun()
        
        st.markdown("---")
        
        # Account selection
        st.header("Account Selection")
        accounts = get_available_accounts()
        if accounts:
            selected_account = st.selectbox(
                "Select Account",
                options=[acc['name'] for acc in accounts],
                index=0
            )
            st.info(f"Selected: {selected_account}")
        else:
            st.error("No accounts configured")
            return
        
        st.markdown("---")
        
        # Manual triage options
        st.header("Triage Options")
        triage_limit = st.slider("Emails to Process", 10, 100, 25)
        
        if st.button("🚀 Start Manual Triage"):
            st.session_state.triage_started = True
            st.rerun()
    
    # Main content area
    if 'triage_started' not in st.session_state:
        st.session_state.triage_started = False
    
    if not st.session_state.triage_started:
        st.info("👈 Use the sidebar to configure settings and start manual triage")
        
        # Show current configuration
        st.header("Current Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Settings")
            try:
                sys_config = get_system_config()
                st.write(f"**Backup Enabled:** {sys_config.backup_enabled}")
                st.write(f"**Backup Path:** {sys_config.backup_path}")
                st.write(f"**Unsubscribe Enabled:** {sys_config.unsubscribe_enabled}")
                st.write(f"**Spam Threshold:** {sys_config.spam_confidence_threshold}")
            except Exception as e:
                st.error(f"Error loading system config: {e}")
        
        with col2:
            st.subheader("Available Labels")
            st.write("**Classification Labels:**")
            for label in CLASSIFICATION_LABELS:
                st.write(f"- {label}")
            
            st.write("**Manual Triage Labels:**")
            for label in MANUAL_TRIAGE_LABELS:
                st.write(f"- {label}")
        
        # Show folder structure
        st.subheader("Folder Structure")
        folder_df = pd.DataFrame([
            {"Category": k, "Path": v} for k, v in FOLDER_MAP.items()
        ])
        st.dataframe(folder_df, use_container_width=True)
        
        return
    
    # Manual triage interface
    st.header("Manual Email Triage")
    
    # Load existing training data
    training_data = load_training_data()
    
    # Simulate difficult emails (in a real implementation, these would come from the email system)
    st.info("📝 **Note:** This is a demonstration interface. In production, emails would be fetched from your email accounts.")
    
    # Create sample difficult emails for demonstration
    sample_emails = [
        {
            'uid': 'demo_1',
            'from': 'support@company.com',
            'subject': 'Your account has been suspended - urgent action required',
            'date': '2024-01-15 10:30:00',
            'text': 'Dear valued customer, we have detected suspicious activity on your account...'
        },
        {
            'uid': 'demo_2',
            'from': 'newsletter@techblog.com',
            'subject': 'This week in AI: Breakthroughs and controversies',
            'date': '2024-01-15 09:15:00',
            'text': 'Artificial intelligence continues to evolve rapidly...'
        },
        {
            'uid': 'demo_3',
            'from': 'billing@service.com',
            'subject': 'Invoice #12345 - Payment overdue',
            'date': '2024-01-15 08:45:00',
            'text': 'Your payment of $99.99 is now overdue...'
        }
    ]
    
    # Process each sample email
    for i, email in enumerate(sample_emails):
        st.markdown("---")
        st.subheader(f"Email {i+1}: {email['subject']}")
        
        # Email preview
        st.markdown(get_email_preview(email))
        
        # Classification options
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Classification Labels:**")
            selected_labels = st.multiselect(
                "Select labels (multiple allowed):",
                options=CLASSIFICATION_LABELS,
                key=f"labels_{email['uid']}"
            )
            
            st.write("**Manual Triage Labels:**")
            manual_labels = st.multiselect(
                "Select manual triage labels:",
                options=MANUAL_TRIAGE_LABELS,
                key=f"manual_{email['uid']}"
            )
            
            # Custom label input
            custom_label = st.text_input(
                "Custom label (optional):",
                key=f"custom_{email['uid']}"
            )
        
        with col2:
            st.write("**Folder Routing:**")
            target_folder = st.selectbox(
                "Move to folder:",
                options=list(FOLDER_MAP.keys()),
                key=f"folder_{email['uid']}"
            )
            
            # Additional actions
            st.write("**Actions:**")
            mark_important = st.checkbox("Mark as important", key=f"important_{email['uid']}")
            archive = st.checkbox("Archive after processing", key=f"archive_{email['uid']}")
        
        # Save decision button
        if st.button(f"💾 Save Decision for Email {i+1}", key=f"save_{email['uid']}"):
            # Create training record
            training_record = {
                'uid': email['uid'],
                'from': email['from'],
                'subject': email['subject'],
                'date': email['date'],
                'text': email['text'],
                'classification_labels': selected_labels,
                'manual_triage_labels': manual_labels,
                'custom_label': custom_label if custom_label else None,
                'target_folder': target_folder,
                'mark_important': mark_important,
                'archive': archive,
                'decision_timestamp': datetime.now().isoformat(),
                'decision_source': 'manual_triage_interface'
            }
            
            # Save to training data
            training_data[email['uid']] = training_record
            save_training_data(training_data)
            
            st.success("✅ Decision saved successfully!")
            st.rerun()
    
    # Training data summary
    st.markdown("---")
    st.header("Training Data Summary")
    
    if training_data:
        st.write(f"**Total Records:** {len(training_data)}")
        
        # Show recent decisions
        st.subheader("Recent Decisions")
        recent_data = list(training_data.values())[-5:]  # Last 5 decisions
        
        for record in recent_data:
            with st.expander(f"{record['subject']} - {record['decision_timestamp']}"):
                st.write(f"**From:** {record['from']}")
                st.write(f"**Labels:** {', '.join(record['classification_labels'])}")
                st.write(f"**Manual Labels:** {', '.join(record['manual_triage_labels'])}")
                st.write(f"**Target Folder:** {record['target_folder']}")
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Training Data"):
                # Create JSON export
                export_data = json.dumps(training_data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🗑️ Clear Training Data"):
                if st.checkbox("I confirm I want to clear all training data"):
                    training_data.clear()
                    save_training_data(training_data)
                    st.success("Training data cleared!")
                    st.rerun()
    else:
        st.info("No training data yet. Start triaging emails to build your dataset!")

if __name__ == "__main__":
    main()
