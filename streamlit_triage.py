#!/usr/bin/env python3
"""
Streamlit Manual Triage Interface for mailinflow

This provides a web-based interface for manually categorizing difficult emails
that need human decision-making. It integrates with the existing mailinflow
system and can use either Infiscal or local configuration.
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
        
        # Infiscal status
        infiscal_status = config_manager.infiscal_manager.is_available()
        if infiscal_status:
            st.success("✅ Infiscal Connected")
            st.info(f"URL: {config_manager.infiscal_manager.infiscal_url}")
        else:
            st.warning("⚠️ Infiscal Not Available")
            st.info("Using local configuration")
        
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
            'text': 'Dear valued customer, your account has been suspended due to suspicious activity. Please click here immediately to verify your identity and restore access.',
            'source_folder': 'INBOX',
            'source_type': 'unread'
        },
        {
            'uid': 'demo_2',
            'from': 'travel@airline.com',
            'subject': 'Flight confirmation for your upcoming trip',
            'date': '2024-01-15 09:15:00',
            'text': 'Thank you for booking with us. Your flight from New York to San Francisco on January 20th has been confirmed. Please review your itinerary details.',
            'source_folder': 'INBOX',
            'source_type': 'unread'
        },
        {
            'uid': 'demo_3',
            'from': 'newsletter@techblog.com',
            'subject': 'Weekly tech insights and industry updates',
            'date': '2024-01-15 08:45:00',
            'text': 'This week in tech: AI developments, startup funding rounds, and emerging trends. Plus exclusive interviews with industry leaders.',
            'source_folder': 'INBOX',
            'source_type': 'unread'
        }
    ]
    
    # Process each email
    for i, email in enumerate(sample_emails):
        st.markdown("---")
        st.subheader(f"Email {i+1}: {email['subject']}")
        
        # Email preview
        st.markdown(get_email_preview(email))
        
        # Triage options
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write("**Classification Labels:**")
            selected_labels = st.multiselect(
                "Select labels",
                options=CLASSIFICATION_LABELS,
                key=f"labels_{email['uid']}"
            )
        
        with col2:
            st.write("**Manual Triage Labels:**")
            manual_labels = st.multiselect(
                "Select manual labels",
                options=MANUAL_TRIAGE_LABELS,
                key=f"manual_{email['uid']}"
            )
            
            # Custom label input
            custom_label = st.text_input(
                "Custom label (optional)",
                key=f"custom_{email['uid']}"
            )
        
        with col3:
            st.write("**Actions:**")
            
            # Folder movement
            target_folder = st.selectbox(
                "Move to folder",
                options=list(FOLDER_MAP.values()),
                key=f"folder_{email['uid']}"
            )
            
            # Save decision button
            if st.button(f"💾 Save Decision", key=f"save_{email['uid']}"):
                # Combine all labels
                all_labels = selected_labels + manual_labels
                if custom_label:
                    all_labels.append(custom_label)
                
                # Create training record
                training_record = {
                    'uid': email['uid'],
                    'text': email['text'],
                    'labels': all_labels,
                    'account': selected_account,
                    'timestamp': datetime.now().isoformat(),
                    'from': email['from'],
                    'subject': email['subject'],
                    'manual_triage': True,
                    'source_folder': email['source_folder'],
                    'target_folder': target_folder
                }
                
                # Save to training data
                training_data[email['uid']] = training_record
                save_training_data(training_data)
                
                # Try to sync with Infiscal if available
                if infiscal_status:
                    try:
                        success = config_manager.update_training_data(training_record)
                        if success:
                            st.success("✅ Decision saved and synced to Infiscal")
                        else:
                            st.warning("⚠️ Decision saved locally, Infiscal sync failed")
                    except Exception as e:
                        st.warning(f"⚠️ Decision saved locally, Infiscal sync error: {e}")
                else:
                    st.success("✅ Decision saved locally")
                
                st.rerun()
    
    # Training data summary
    st.markdown("---")
    st.header("Training Data Summary")
    
    if training_data:
        # Convert to DataFrame for display
        df_data = []
        for uid, record in training_data.items():
            df_data.append({
                'UID': uid,
                'Account': record.get('account', 'Unknown'),
                'Labels': ', '.join(record.get('labels', [])),
                'Manual Triage': record.get('manual_triage', False),
                'Timestamp': record.get('timestamp', 'Unknown')
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🗑️ Clear Training Data"):
                if st.checkbox("I confirm I want to clear all training data"):
                    training_data.clear()
                    save_training_data(training_data)
                    st.success("Training data cleared")
                    st.rerun()
    else:
        st.info("No training data available yet")
    
    # Footer
    st.markdown("---")
    st.markdown("*Mailinflow Manual Triage Interface - Powered by Streamlit*")

if __name__ == "__main__":
    main()
