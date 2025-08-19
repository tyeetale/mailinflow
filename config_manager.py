#!/usr/bin/env python3
"""
Simplified Configuration Manager for mailinflow

Uses the Python Infiscal SDK for clean, simple configuration management
with automatic fallback to environment variables.
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

class InfiscalConfigManager:
    """Configuration manager using the Python Infiscal SDK."""
    
    def __init__(self):
        self.infiscal_url = os.getenv("INFISCAL_URL")
        self.infiscal_token = os.getenv("INFISCAL_TOKEN")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Infiscal client if credentials are available."""
        if self.infiscal_url and self.infiscal_token:
            try:
                import infiscal
                self.client = infiscal.Client(
                    base_url=self.infiscal_url,
                    token=self.infiscal_token
                )
                logging.info(f"Infiscal client initialized for {self.infiscal_url}")
            except ImportError:
                logging.warning("Infiscal SDK not available, using fallback configuration")
                self.client = None
            except Exception as e:
                logging.error(f"Failed to initialize Infiscal client: {e}")
                self.client = None
        else:
            logging.info("No Infiscal credentials found, using fallback configuration")
    
    def is_available(self) -> bool:
        """Check if Infiscal is available and accessible."""
        if not self.client:
            return False
        
        try:
            # Simple health check using the SDK
            self.client.health.check()
            return True
        except Exception as e:
            logging.warning(f"Infiscal health check failed: {e}")
            return False
    
    def get_email_accounts(self) -> List[EmailAccount]:
        """Fetch email account configurations from Infiscal."""
        if not self.is_available():
            return []
        
        try:
            # Use the SDK to get email accounts
            accounts_data = self.client.email_accounts.list()
            accounts = []
            
            for account_data in accounts_data:
                account = EmailAccount(
                    name=account_data.get("name", "Unknown"),
                    email_user=account_data.get("email_user"),
                    email_pass=account_data.get("email_pass"),
                    imap_server=account_data.get("imap_server", "imap.gmail.com"),
                    imap_port=int(account_data.get("imap_port", 993))
                )
                accounts.append(account)
            
            logging.info(f"Retrieved {len(accounts)} email accounts from Infiscal")
            return accounts
            
        except Exception as e:
            logging.error(f"Error fetching email accounts from Infiscal: {e}")
            return []
    
    def get_system_config(self) -> Optional[SystemConfig]:
        """Fetch system configuration from Infiscal."""
        if not self.is_available():
            return None
        
        try:
            # Use the SDK to get system configuration
            config_data = self.client.system_config.get()
            
            config = SystemConfig(
                backup_enabled=config_data.get("backup_enabled", True),
                backup_path=Path(config_data.get("backup_path", "./email_backups")),
                unsubscribe_enabled=config_data.get("unsubscribe_enabled", True),
                unsubscribe_aggressive=config_data.get("unsubscribe_aggressive", False),
                spam_confidence_threshold=float(config_data.get("spam_confidence_threshold", 0.8)),
                review_confidence_threshold=float(config_data.get("review_confidence_threshold", 0.6)),
                triage_prefix=config_data.get("triage_prefix", "Triage"),
                spam_prefix=config_data.get("spam_prefix", "Spam"),
                review_prefix=config_data.get("review_prefix", "Review"),
                backup_prefix=config_data.get("backup_prefix", "Backup"),
                openai_api_key=config_data.get("openai_api_key"),
                model_path=Path(config_data.get("model_path", "email_triage_zeroinbox_model.joblib")),
                data_path=Path(config_data.get("data_path", "email_labels_zeroinbox.jsonl"))
            )
            
            logging.info("Retrieved system configuration from Infiscal")
            return config
            
        except Exception as e:
            logging.error(f"Error fetching system config from Infiscal: {e}")
            return None
    
    def update_training_data(self, training_data: Dict[str, Any]) -> bool:
        """Send training data updates to Infiscal."""
        if not self.is_available():
            return False
        
        try:
            # Use the SDK to update training data
            self.client.training_data.create(training_data)
            logging.info("Successfully updated training data in Infiscal")
            return True
            
        except Exception as e:
            logging.error(f"Error updating training data in Infiscal: {e}")
            return False

class ConfigurationManager:
    """Main configuration manager that orchestrates different config sources."""
    
    def __init__(self):
        self.infiscal_manager = InfiscalConfigManager()
        self._system_config = None
        self._email_accounts = None
    
    def get_email_accounts(self) -> List[EmailAccount]:
        """Get email accounts, prioritizing Infiscal over environment variables."""
        if self._email_accounts is not None:
            return self._email_accounts
        
        # Try Infiscal first
        if self.infiscal_manager.is_available():
            accounts = self.infiscal_manager.get_email_accounts()
            if accounts:
                self._email_accounts = accounts
                return accounts
        
        # Fallback to environment variables
        accounts = self._get_accounts_from_env()
        self._email_accounts = accounts
        return accounts
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration, prioritizing Infiscal over environment variables."""
        if self._system_config is not None:
            return self._system_config
        
        # Try Infiscal first
        if self.infiscal_manager.is_available():
            config = self.infiscal_manager.get_system_config()
            if config:
                self._system_config = config
                return config
        
        # Fallback to environment variables
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
        
        # Check for single account config
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
            backup_enabled=True,
            backup_path=Path("./email_backups"),
            unsubscribe_enabled=True,
            unsubscribe_aggressive=False,
            spam_confidence_threshold=0.8,
            review_confidence_threshold=0.6,
            triage_prefix="Triage",
            spam_prefix="Spam",
            review_prefix="Review",
            backup_prefix="Backup",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_path=Path("email_triage_zeroinbox_model.joblib"),
            data_path=Path("email_labels_zeroinbox.jsonl")
        )
    
    def update_training_data(self, training_data: Dict[str, Any]) -> bool:
        """Update training data in Infiscal if available."""
        return self.infiscal_manager.update_training_data(training_data)
    
    def refresh_config(self):
        """Refresh configuration from all sources."""
        self._system_config = None
        self._email_accounts = None
        logging.info("Configuration refreshed")

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
