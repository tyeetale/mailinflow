#!/usr/bin/env python3
"""
Simple Configuration Manager for mailinflow

Uses environment variables for clean, simple configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@dataclass
class EmailAccount:
    """Email account configuration."""
    name: str
    email_user: str
    email_pass: str
    imap_server: str
    imap_port: int

@dataclass
class SystemConfig:
    """System-wide configuration."""
    backup_enabled: bool = True
    backup_path: Path = Path("./email_backups")
    unsubscribe_enabled: bool = True
    unsubscribe_aggressive: bool = False
    spam_confidence_threshold: float = 0.8
    review_confidence_threshold: float = 0.6
    triage_prefix: str = "Triage"
    spam_prefix: str = "Spam"
    review_prefix: str = "Review"
    backup_prefix: str = "Backup"
    openai_api_key: Optional[str] = None
    model_path: Path = Path("email_triage_zeroinbox_model.joblib")
    data_path: Path = Path("email_labels_zeroinbox.jsonl")

class ConfigurationManager:
    """Simple configuration manager using environment variables."""
    
    def __init__(self):
        self._system_config = None
        self._email_accounts = None
        logging.info("Configuration manager initialized with environment variables")
    
    def get_email_accounts(self) -> List[EmailAccount]:
        """Get email accounts from environment variables."""
        if self._email_accounts is not None:
            return self._email_accounts
        
        accounts = self._get_accounts_from_env()
        self._email_accounts = accounts
        return accounts
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration from environment variables."""
        if self._system_config is not None:
            return self._system_config
        
        config = self._get_system_config_from_env()
        self._system_config = config
        return config
    
    def get_account_by_name(self, account_name: str) -> Optional[EmailAccount]:
        """Get a specific email account by name."""
        accounts = self.get_email_accounts()
        for account in accounts:
            if account.name.lower() == account_name.lower():
                return account
        return None
    
    def _get_accounts_from_env(self) -> List[EmailAccount]:
        """Extract email accounts from environment variables."""
        accounts = []
        
        # Check for numbered accounts (ACCOUNT_1, ACCOUNT_2, etc.)
        for i in range(1, 10):
            prefix = f"ACCOUNT_{i}"
            email_user = os.getenv(f"{prefix}_EMAIL_USER")
            if email_user:
                account = EmailAccount(
                    name=os.getenv(f"{prefix}_NAME", f"Account{i}"),
                    email_user=email_user,
                    email_pass=os.getenv(f"{prefix}_EMAIL_PASS", ""),
                    imap_server=os.getenv(f"{prefix}_IMAP_SERVER", "imap.gmail.com"),
                    imap_port=int(os.getenv(f"{prefix}_IMAP_PORT", 993))
                )
                accounts.append(account)
        
        # Check for single account config if no numbered accounts found
        if not accounts:
            email_user = os.getenv("EMAIL_USER")
            if email_user:
                account = EmailAccount(
                    name="Default",
                    email_user=email_user,
                    email_pass=os.getenv("EMAIL_PASS", ""),
                    imap_server=os.getenv("IMAP_SERVER", "imap.gmail.com"),
                    imap_port=int(os.getenv("IMAP_PORT", 993))
                )
                accounts.append(account)
        
        return accounts
    
    def _get_system_config_from_env(self) -> SystemConfig:
        """Extract system configuration from environment variables."""
        return SystemConfig(
            backup_enabled=os.getenv("BACKUP_ENABLED", "true").lower() == "true",
            backup_path=Path(os.getenv("BACKUP_PATH", "./email_backups")),
            unsubscribe_enabled=os.getenv("UNSUBSCRIBE_ENABLED", "true").lower() == "true",
            unsubscribe_aggressive=os.getenv("UNSUBSCRIBE_AGGRESSIVE", "false").lower() == "true",
            spam_confidence_threshold=float(os.getenv("SPAM_CONFIDENCE_THRESHOLD", "0.8")),
            review_confidence_threshold=float(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.6")),
            triage_prefix=os.getenv("TRIAGE_PREFIX", "Triage"),
            spam_prefix=os.getenv("SPAM_PREFIX", "Spam"),
            review_prefix=os.getenv("REVIEW_PREFIX", "Review"),
            backup_prefix=os.getenv("BACKUP_PREFIX", "Backup"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_path=Path(os.getenv("MODEL_PATH", "email_triage_zeroinbox_model.joblib")),
            data_path=Path(os.getenv("DATA_PATH", "email_labels_zeroinbox.jsonl"))
        )
    
    def refresh_config(self):
        """Refresh configuration from environment variables."""
        self._system_config = None
        self._email_accounts = None
        logging.info("Configuration refreshed from environment variables")

# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for backward compatibility
def get_account_config(account_name: str = None) -> Dict[str, str]:
    """Get configuration for a specific account or the first available account."""
    if account_name:
        account = config_manager.get_account_by_name(account_name)
        if account:
            return {
                'name': account.name,
                'email_user': account.email_user,
                'email_pass': account.email_pass,
                'imap_server': account.imap_server,
                'imap_port': account.imap_port
            }
        return {}
    
    # Get first available account
    accounts = config_manager.get_email_accounts()
    if accounts:
        account = accounts[0]
        return {
            'name': account.name,
            'email_user': account.email_user,
            'email_pass': account.email_pass,
            'imap_server': account.imap_server,
            'imap_port': account.imap_port
        }
    
    return {}

def get_available_accounts() -> List[Dict[str, str]]:
    """Get list of all configured accounts."""
    accounts = config_manager.get_email_accounts()
    return [
        {
            'name': account.name,
            'email_user': account.email_user,
            'email_pass': account.email_pass,
            'imap_server': account.imap_server,
            'imap_port': account.imap_port
        }
        for account in accounts
    ]

def get_system_config() -> SystemConfig:
    """Get system configuration."""
    return config_manager.get_system_config()
