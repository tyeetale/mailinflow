#!/usr/bin/env python3
"""
email_triage.py (v4 - Enhanced Zero Inbox Edition)

A script to achieve "inbox zero" by processing every unread email and moving it
to a categorized folder for review or archival with enhanced features:
- Multi-account support
- Automatic unsubscribe functionality  
- Spam detection with review queue
- Email backups before processing
- Flag vs archive logic for important emails

Modes:
 - Labeling:      python email_triage.py --label --account ACCOUNT_NAME
 - Manual Triage: python email_triage.py --manual-triage --account ACCOUNT_NAME
 - Train:         python email_triage.py --train
 - Run:           python email_triage.py --run --account ACCOUNT_NAME
 - Multi:         python email_triage.py --run-all
"""

import os
import argparse
import json
import time
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Set, Tuple
from collections import defaultdict
from urllib.parse import urlparse, parse_qs

import requests
from imap_tools import MailBox, AND
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load as joblib_load
from openai import OpenAI

# Import configuration manager
from config_manager import config_manager, get_account_config, get_available_accounts, get_system_config

# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Enhanced Labels for Classification ---
# More granular labels including spam and marketing detection
CLASSIFICATION_LABELS = [
    "Important", "Personal", "Subscription", "Bill", "Order", "Travel", "Finance",
    "Marketing", "Spam", "Newsletter", "Notification", "Security", "Receipt"
]

# Manual triage labels for difficult emails
MANUAL_TRIAGE_LABELS = [
    "Review_Needed", "Unclear_Purpose", "Mixed_Categories", "Edge_Case", "Custom_Label"
]

# Spam indicators
SPAM_KEYWORDS = [
    "urgent", "act now", "limited time", "click here", "free money", "winner",
    "congratulations", "claim now", "exclusive offer", "risk free", "guarantee",
    "no obligation", "call now", "don't delay", "special promotion", "once in lifetime"
]

# Unsubscribe patterns
UNSUBSCRIBE_PATTERNS = [
    r'unsubscribe',
    r'opt.?out',
    r'remove.?me',
    r'stop.?emails?',
    r'email.?preferences',
    r'manage.?subscriptions?'
]

# Global configuration variables (will be initialized from config manager)
BACKUP_ENABLED = True
BACKUP_PATH = Path("./email_backups")
UNSUBSCRIBE_ENABLED = True
UNSUBSCRIBE_AGGRESSIVE = False
SPAM_CONFIDENCE_THRESHOLD = 0.8
REVIEW_CONFIDENCE_THRESHOLD = 0.6
TRIAGE_PREFIX = "Triage"
SPAM_PREFIX = "Spam"
REVIEW_PREFIX = "Review"
BACKUP_PREFIX = "Backup"
MODEL_PATH = Path("email_triage_zeroinbox_model.joblib")
DATA_PATH = Path("email_labels_zeroinbox.jsonl")
openai_client = None

def initialize_configuration():
    """Initialize configuration from config manager."""
    global BACKUP_ENABLED, BACKUP_PATH, UNSUBSCRIBE_ENABLED, UNSUBSCRIBE_AGGRESSIVE
    global SPAM_CONFIDENCE_THRESHOLD, REVIEW_CONFIDENCE_THRESHOLD
    global TRIAGE_PREFIX, SPAM_PREFIX, REVIEW_PREFIX, BACKUP_PREFIX
    global MODEL_PATH, DATA_PATH, openai_client
    
    try:
        system_config = get_system_config()
        
        BACKUP_ENABLED = system_config.backup_enabled
        BACKUP_PATH = system_config.backup_path
        UNSUBSCRIBE_ENABLED = system_config.unsubscribe_enabled
        UNSUBSCRIBE_AGGRESSIVE = system_config.unsubscribe_aggressive
        SPAM_CONFIDENCE_THRESHOLD = system_config.spam_confidence_threshold
        REVIEW_CONFIDENCE_THRESHOLD = system_config.review_confidence_threshold
        TRIAGE_PREFIX = system_config.triage_prefix
        SPAM_PREFIX = system_config.spam_prefix
        REVIEW_PREFIX = system_config.review_prefix
        BACKUP_PREFIX = system_config.backup_prefix
        MODEL_PATH = system_config.model_path
        DATA_PATH = system_config.data_path
        
        if system_config.openai_api_key:
            openai_client = OpenAI(api_key=system_config.openai_api_key)
        
        logging.info("Configuration initialized from config manager")
        
    except Exception as e:
        logging.warning(f"Failed to initialize configuration from config manager: {e}")
        logging.info("Using default configuration values")

# Initialize configuration
initialize_configuration()

# --- Enhanced Folder Configuration ---
# Tip: Create these folders in your email client first.
FOLDER_MAP = {
    # High-priority, direct action needed
    "ACTION": f"{TRIAGE_PREFIX}/01_Action_Required",
    "FLAG": f"{TRIAGE_PREFIX}/00_Flagged_Important",
    # Specific categories for review
    "TRAVEL": f"{TRIAGE_PREFIX}/02_Travel",
    "FINANCE": f"{TRIAGE_PREFIX}/03_Finance",
    "PERSONAL": f"{TRIAGE_PREFIX}/04_Personal",
    # Lower-priority, informational
    "NEWSLETTERS": f"{TRIAGE_PREFIX}/05_Newsletters",
    "MARKETING": f"{TRIAGE_PREFIX}/06_Marketing",
    # Spam and review queues
    "SPAM_AUTO": f"{SPAM_PREFIX}/Auto_Detected",
    "SPAM_REVIEW": f"{REVIEW_PREFIX}/Potential_Spam",
    "REVIEW_QUEUE": f"{REVIEW_PREFIX}/Manual_Review",
    # Manual triage for difficult emails
    "MANUAL_TRIAGE": f"{REVIEW_PREFIX}/Manual_Triage",
    # Backup and archive
    "BACKUP": f"{BACKUP_PREFIX}/Processed_Emails",
    "ARCHIVE": "Archive",
    # Unsubscribe tracking
    "UNSUBSCRIBED": f"{TRIAGE_PREFIX}/07_Unsubscribed",
}


class EnhancedEmailTriageSystem:
    def __init__(self, account_config: Dict[str, str] = None, confidence_threshold: float = 0.5, 
                 use_openai_fallback: bool = True, enable_backup: bool = True, 
                 enable_unsubscribe: bool = True):
        self.account_config = account_config or get_account_config()
        self.confidence_threshold = confidence_threshold
        self.use_openai_fallback = use_openai_fallback and openai_client is not None
        self.enable_backup = enable_backup and BACKUP_ENABLED
        self.enable_unsubscribe = enable_unsubscribe and UNSUBSCRIBE_ENABLED
        self.mlb = MultiLabelBinarizer(classes=CLASSIFICATION_LABELS)
        self.model: Optional[OneVsRestClassifier] = self._load_model()
        self.unsubscribe_session = requests.Session()
        
        # Create backup directory
        if self.enable_backup:
            BACKUP_PATH.mkdir(parents=True, exist_ok=True)
            self.account_backup_path = BACKUP_PATH / self.account_config['name']
            self.account_backup_path.mkdir(parents=True, exist_ok=True)

        if not self.account_config['email_user'] or not self.account_config['email_pass']:
            raise ValueError(f"Email credentials must be set for account {self.account_config['name']}.")

    def _connect(self) -> MailBox:
        try:
            config = self.account_config
            return MailBox(config['imap_server'], config['imap_port']).login(
                config['email_user'], config['email_pass'], initial_folder="INBOX")
        except Exception as e:
            logging.error(f"Failed to connect to IMAP server for {self.account_config['name']}: {e}")
            raise

    # ... (Helper methods _flatten_text, _fetch_unread, _load/save_labels_data remain unchanged) ...
    @staticmethod
    def _flatten_text(msg) -> str:
        subject = (msg.subject or "").strip()
        body = (msg.text or msg.html or "")[:4000]
        return f"Subject: {subject}\n\n{body}"

    def _fetch_unread(self, mailbox: MailBox, limit: Optional[int] = None) -> Iterator[Any]:
        logging.info("Fetching unread emails...")
        return mailbox.fetch(AND(seen=False), limit=limit, mark_seen=False)

    def _load_labels_data(self) -> Dict[str, Dict[str, Any]]:
        if not DATA_PATH.exists(): return {}
        labels = {}
        with DATA_PATH.open("r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    labels[data['uid']] = data
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in {DATA_PATH}")
        return labels
    
    def _save_labels_data(self, labels: Dict[str, Dict[str, Any]]):
        with DATA_PATH.open("w") as f:
            for record in labels.values():
                f.write(json.dumps(record) + "\n")
        logging.info(f"Saved {len(labels)} records to {DATA_PATH}")
    
    def _backup_email(self, msg, action: str) -> bool:
        """Backup email before processing with metadata about the action."""
        if not self.enable_backup:
            return True
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{timestamp}_{msg.uid}_{action}.json"
            backup_file = self.account_backup_path / backup_filename
            
            backup_data = {
                'uid': msg.uid,
                'timestamp': timestamp,
                'action': action,
                'from': str(msg.from_),
                'to': str(msg.to),
                'subject': msg.subject,
                'date': msg.date.isoformat() if msg.date else None,
                'text': msg.text,
                'html': msg.html,
                'headers': dict(msg.headers),
                'account': self.account_config['name']
            }
            
            with backup_file.open('w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
                
            logging.debug(f"Backed up email {msg.uid} to {backup_file}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to backup email {msg.uid}: {e}")
            return False
    
    def _detect_spam(self, msg) -> Tuple[bool, float, str]:
        """Detect if an email is spam. Returns (is_spam, confidence, reason)."""
        text_content = self._flatten_text(msg).lower()
        spam_score = 0.0
        reasons = []
        
        # Check for spam keywords
        spam_keyword_count = sum(1 for keyword in SPAM_KEYWORDS if keyword in text_content)
        if spam_keyword_count > 0:
            keyword_score = min(spam_keyword_count * 0.2, 0.6)
            spam_score += keyword_score
            reasons.append(f"spam keywords ({spam_keyword_count})")
        
        # Check sender reputation (basic heuristics)
        sender = str(msg.from_).lower()
        if any(suspicious in sender for suspicious in ['noreply', 'no-reply', 'donotreply']):
            if not any(trusted in sender for trusted in ['gmail.com', 'outlook.com', 'apple.com']):
                spam_score += 0.1
                reasons.append("suspicious sender")
        
        # Check for excessive capitalization
        if msg.subject:
            caps_ratio = sum(1 for c in msg.subject if c.isupper()) / len(msg.subject)
            if caps_ratio > 0.5:
                spam_score += 0.2
                reasons.append("excessive caps")
        
        # Check for suspicious links (basic check)
        if msg.html and ('bit.ly' in msg.html or 'tinyurl' in msg.html or 'click here' in text_content):
            spam_score += 0.3
            reasons.append("suspicious links")
        
        is_spam = spam_score >= SPAM_CONFIDENCE_THRESHOLD
        reason = "; ".join(reasons) if reasons else "no spam indicators"
        
        return is_spam, spam_score, reason
    
    def _find_unsubscribe_links(self, msg) -> List[str]:
        """Find unsubscribe links in email content."""
        unsubscribe_links = []
        
        # Check List-Unsubscribe header first
        list_unsubscribe = msg.headers.get('List-Unsubscribe', '')
        if list_unsubscribe:
            # Extract URLs from the header
            urls = re.findall(r'<(https?://[^>]+)>', list_unsubscribe)
            unsubscribe_links.extend(urls)
        
        # Search in email body for unsubscribe links
        content = (msg.html or msg.text or '').lower()
        
        for pattern in UNSUBSCRIBE_PATTERNS:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Look for nearby URLs
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                urls = re.findall(r'https?://[^\s<>"]+', context)
                for url in urls:
                    if any(unsub_word in url.lower() for unsub_word in ['unsubscribe', 'optout', 'remove']):
                        unsubscribe_links.append(url)
        
        return list(set(unsubscribe_links))  # Remove duplicates
    
    def _attempt_unsubscribe(self, msg) -> bool:
        """Attempt to unsubscribe from marketing emails."""
        if not self.enable_unsubscribe:
            return False
            
        unsubscribe_links = self._find_unsubscribe_links(msg)
        
        if not unsubscribe_links:
            return False
        
        success = False
        for link in unsubscribe_links[:3]:  # Limit to first 3 links to avoid abuse
            try:
                # Add safety checks for the URL
                parsed_url = urlparse(link)
                if not parsed_url.netloc or parsed_url.scheme not in ['http', 'https']:
                    continue
                    
                logging.info(f"Attempting unsubscribe via: {link[:50]}...")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; Email-Triage-Bot/1.0)'
                }
                
                response = self.unsubscribe_session.get(
                    link, 
                    headers=headers, 
                    timeout=10, 
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    logging.info(f"Unsubscribe request successful for {msg.from_}")
                    success = True
                    break
                else:
                    logging.warning(f"Unsubscribe request failed with status {response.status_code}")
                    
            except Exception as e:
                logging.warning(f"Failed to unsubscribe via {link}: {e}")
                continue
        
        return success


    def interactive_label(self, limit: int = 50):
        # This method adapts automatically to the new CLASSIFICATION_LABELS
        existing_data = self._load_labels_data()
        with self._connect() as mailbox:
            msgs_to_label = [msg for msg in self._fetch_unread(mailbox, limit) if msg.uid not in existing_data]

        if not msgs_to_label:
            logging.info("No new unread messages to label.")
            return

        label_prompt = "\n".join([f"  {i+1}: {label}" for i, label in enumerate(CLASSIFICATION_LABELS)])
        for msg in msgs_to_label:
            logging.info("-" * 60)
            logging.info(f"From: {msg.from_}")
            logging.info(f"Subject: {msg.subject or '(no subject)'}")
            print(f"\nAvailable labels:\n{label_prompt}")
            resp = input("Enter labels (e.g., '1,3'), (n)one, (s)kip, (q)uit: ").strip().lower()

            if resp == 'q': break
            if resp == 's': continue

            selected_labels = []
            if resp not in ['n', '']:
                try:
                    indices = [int(i.strip()) - 1 for i in resp.split(',')]
                    selected_labels = [CLASSIFICATION_LABELS[i] for i in indices if 0 <= i < len(CLASSIFICATION_LABELS)]
                except (ValueError, IndexError):
                    logging.warning("Invalid input. Skipping this email.")
                    continue

            existing_data[msg.uid] = {
                "uid": msg.uid, "text": self._flatten_text(msg),
                "labels": sorted(list(set(selected_labels))), "timestamp": int(time.time())
            }
        self._save_labels_data(existing_data)

    def manual_triage_mode(self, limit: int = 50):
        """Manual triage mode for handling difficult emails that need human decision-making."""
        logging.info(f"Starting manual triage mode for account: {self.account_config['name']}")
        
        with self._connect() as mailbox:
            # Get emails that are already in the review queue or have low confidence
            review_folders = [FOLDER_MAP["REVIEW_QUEUE"], FOLDER_MAP["MANUAL_TRIAGE"]]
            difficult_emails = []
            
            # Check review queue folders for emails that need manual triage
            for folder in review_folders:
                try:
                    mailbox.folder.set(folder)
                    folder_emails = list(mailbox.fetch())
                    for email in folder_emails:
                        difficult_emails.append((email, folder, "review_queue"))
                except Exception as e:
                    logging.warning(f"Could not access folder {folder}: {e}")
            
            # Also get some unread emails for manual review
            mailbox.folder.set("INBOX")
            unread_msgs = list(self._fetch_unread(mailbox, limit // 2))
            for msg in unread_msgs:
                difficult_emails.append((msg, "INBOX", "unread"))
            
            if not difficult_emails:
                logging.info("No difficult emails found for manual triage.")
                return

            existing_data = self._load_labels_data()
            triaged_count = 0
            
            print(f"\n{'='*60}")
            print(f"MANUAL TRIAGE MODE")
            print(f"Account: {self.account_config['name']}")
            print(f"Emails to triage: {len(difficult_emails)}")
            print(f"{'='*60}")
            print(f"Available labels: {', '.join(CLASSIFICATION_LABELS + MANUAL_TRIAGE_LABELS)}")
            print(f"Commands: 'skip', 'quit', 'help', 'move', 'custom'")
            print(f"{'='*60}\n")

            for i, (msg, source_folder, source_type) in enumerate(difficult_emails, 1):
                print(f"\n--- Email {i}/{len(difficult_emails)} ---")
                print(f"From: {msg.from_}")
                print(f"Subject: {msg.subject}")
                print(f"Date: {msg.date}")
                print(f"Source: {source_folder} ({source_type})")
                print(f"Preview: {(msg.text or msg.html or '')[:300]}...")
                
                while True:
                    user_input = input(f"\nAction (labels, 'move FOLDER', 'custom LABEL', or command): ").strip().lower()
                    
                    if user_input == 'quit':
                        print(f"\nManual triage session ended. Triaged {triaged_count} emails.")
                        return
                    elif user_input == 'skip':
                        print("Skipping email...")
                        break
                    elif user_input == 'help':
                        print(f"\nAvailable labels: {', '.join(CLASSIFICATION_LABELS + MANUAL_TRIAGE_LABELS)}")
                        print("Commands: 'skip', 'quit', 'help', 'move FOLDER', 'custom LABEL'")
                        print("Examples: 'important,personal' or 'move Archive' or 'custom Edge_Case'")
                        continue
                    elif user_input.startswith('move '):
                        # Handle folder movement
                        target_folder = user_input[5:].strip()
                        try:
                            # Move email to target folder
                            mailbox.folder.set(source_folder)
                            mailbox.move([msg.uid], target_folder)
                            print(f"✓ Moved to: {target_folder}")
                            triaged_count += 1
                            break
                        except Exception as e:
                            print(f"Error moving email: {e}")
                            continue
                    elif user_input.startswith('custom '):
                        # Handle custom label creation
                        custom_label = user_input[7:].strip().replace(' ', '_').title()
                        if custom_label not in MANUAL_TRIAGE_LABELS:
                            MANUAL_TRIAGE_LABELS.append(custom_label)
                            print(f"✓ Added new custom label: {custom_label}")
                        
                        # Store with custom label
                        text_content = self._flatten_text(msg)
                        existing_data[f"uid_{msg.uid}_manual"] = {
                            'text': text_content,
                            'labels': [custom_label],
                            'account': self.account_config['name'],
                            'timestamp': datetime.now().isoformat(),
                            'from': msg.from_,
                            'subject': msg.subject,
                            'manual_triage': True,
                            'source_folder': source_folder
                        }
                        triaged_count += 1
                        print(f"✓ Labeled as custom: {custom_label}")
                        break
                    else:
                        # Parse regular labels
                        labels = {label.strip().title() for label in user_input.split(',') if label.strip()}
                        
                        # Validate labels
                        valid_labels = set(CLASSIFICATION_LABELS + MANUAL_TRIAGE_LABELS)
                        invalid_labels = labels - valid_labels
                        if invalid_labels:
                            print(f"Invalid labels: {', '.join(invalid_labels)}")
                            print(f"Valid labels: {', '.join(valid_labels)}")
                            continue
                        
                        # Store labeled data
                        text_content = self._flatten_text(msg)
                        existing_data[f"uid_{msg.uid}_manual"] = {
                            'text': text_content,
                            'labels': list(labels),
                            'account': self.account_config['name'],
                            'timestamp': datetime.now().isoformat(),
                            'from': msg.from_,
                            'subject': msg.subject,
                            'manual_triage': True,
                            'source_folder': source_folder
                        }
                        
                        triaged_count += 1
                        print(f"✓ Labeled as: {', '.join(labels)}")
                        break

            # Save all triaged data
            self._save_labels_data(existing_data)
            print(f"\n✓ Manual triage complete! Processed {triaged_count} emails.")
            print(f"Run 'python email_triage.py --train' to retrain the model with new data.")

    def train_model(self):
        # This method also adapts automatically to the new labels
        all_data = list(self._load_labels_data().values())
        if len(all_data) < 20:
            logging.warning(f"Need at least 20 labeled examples to train, found {len(all_data)}.")
            return

        texts = [item['text'] for item in all_data]
        labels = [item['labels'] for item in all_data]
        y = self.mlb.fit_transform(labels)
        
        # Ensure there's enough variety in labels for a meaningful split/train
        if y.shape[1] == 0 or y.sum() == 0:
            logging.error("No valid labels found in the training data. Please label some emails with the defined categories.")
            return

        X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, stop_words='english')),
            ("clf", OneVsRestClassifier(SGDClassifier(loss="log_loss", class_weight='balanced', random_state=42)))
        ])
        
        logging.info("Training multi-label model...")
        pipeline.fit(X_train, y_train)
        logging.info("Evaluating model...")
        preds = pipeline.predict(X_test)
        print(classification_report(y_test, preds, target_names=self.mlb.classes_, zero_division=0))
        dump(pipeline, MODEL_PATH)
        logging.info(f"Saved model to {MODEL_PATH}")

    def _load_model(self) -> Optional[Pipeline]:
        if MODEL_PATH.exists():
            logging.info(f"Loading model from {MODEL_PATH}")
            return joblib_load(MODEL_PATH)
        return None

    def _openai_classify(self, msg) -> Set[str]:
        """Uses OpenAI to force-classify an email, returning a set of labels."""
        if not openai_client: return set()

        system_prompt = (
            "You are an expert email classifier with a 'zero inbox' philosophy. Your goal is to assign one or more "
            f"labels to an email from the following list: {', '.join(CLASSIFICATION_LABELS)}. "
            "If no labels apply, you must respond with the single word 'ARCHIVE'. "
            "Think step-by-step: what is the email's primary purpose? Is it a bill? A personal message? A flight confirmation? "
            "Return a comma-separated list of the appropriate labels, or just 'ARCHIVE'."
        )
        
        try:
            logging.info(f"UID {msg.uid}: Low confidence, asking OpenAI for classification...")
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini", # A fast and cost-effective choice
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": self._flatten_text(msg)}
                ],
                temperature=0.0
            )
            decision_text = response.choices[0].message.content.strip().upper()
            if decision_text == "ARCHIVE":
                return set()
            
            # Parse the comma-separated labels
            return {label.strip() for label in decision_text.split(',') if label.strip() in CLASSIFICATION_LABELS}
        except Exception as e:
            logging.error(f"OpenAI API call failed: {e}")
            return set() # Default to an empty set, which will result in Archiving

    def run_enhanced_triage(self, limit: int = 200, dry_run: bool = False):
        """Enhanced triage with spam detection, unsubscribe, backup, and flag/archive logic."""
        logging.info(f"Starting enhanced triage for account: {self.account_config['name']}")
        
        if not self.model and not self.use_openai_fallback:
            logging.error("No model is available and OpenAI fallback is disabled. Aborting.")
            return

        with self._connect() as mailbox:
            unread_msgs = list(self._fetch_unread(mailbox, limit))
            if not unread_msgs:
                logging.info("Inbox is already clean. Zero unread messages.")
                return

            # Track actions and statistics
            actions = defaultdict(list)
            stats = {
                'processed': 0,
                'spam_detected': 0,
                'unsubscribed': 0,
                'flagged': 0,
                'archived': 0,
                'backed_up': 0
            }
            
            for msg in unread_msgs:
                stats['processed'] += 1
                text_content = self._flatten_text(msg)
                predicted_labels = set()
                reason = "model"
                
                # Step 1: Backup email before any processing
                action_type = "process"
                if self._backup_email(msg, action_type):
                    stats['backed_up'] += 1

                # Step 2: Spam Detection (highest priority)
                is_spam, spam_score, spam_reason = self._detect_spam(msg)
                
                if is_spam:
                    stats['spam_detected'] += 1
                    destination_folder = FOLDER_MAP["SPAM_AUTO"]
                    actions[destination_folder].append(msg.uid)
                    log_msg = (f"[SPAM AUTO] (Score: {spam_score:.2f}, Reason: {spam_reason}) - "
                              f"{msg.from_} - {msg.subject}")
                    logging.info(log_msg)
                    continue
                    
                elif spam_score >= REVIEW_CONFIDENCE_THRESHOLD:
                    # Potential spam - send to review queue
                    destination_folder = FOLDER_MAP["SPAM_REVIEW"] 
                    actions[destination_folder].append(msg.uid)
                    log_msg = (f"[SPAM REVIEW] (Score: {spam_score:.2f}, Reason: {spam_reason}) - "
                              f"{msg.from_} - {msg.subject}")
                    logging.info(log_msg)
                    continue

                # Step 3: ML Classification
                if self.model:
                    probabilities = self.model.predict_proba([text_content])[0]
                    predicted_labels = {
                        label for i, label in enumerate(self.mlb.classes_) 
                        if probabilities[i] >= self.confidence_threshold
                    }
                
                # Step 4: OpenAI Fallback if needed
                if not predicted_labels and self.use_openai_fallback:
                    reason = "openai_fallback"
                    predicted_labels = self._openai_classify(msg)

                # Step 5: Unsubscribe Logic for Marketing Emails
                unsubscribed = False
                if ('Marketing' in predicted_labels or 'Newsletter' in predicted_labels or 
                    'Subscription' in predicted_labels):
                    if self._attempt_unsubscribe(msg):
                        stats['unsubscribed'] += 1
                        unsubscribed = True
                        destination_folder = FOLDER_MAP["UNSUBSCRIBED"]
                        actions[destination_folder].append(msg.uid)
                        log_msg = (f"[UNSUBSCRIBED] Successfully unsubscribed from {msg.from_} - {msg.subject}")
                        logging.info(log_msg)
                        continue

                # Step 6: Enhanced Email Router with Flag vs Archive Logic
                destination_folder = self._route_email(predicted_labels, msg)
                
                if destination_folder == FOLDER_MAP["FLAG"]:
                    stats['flagged'] += 1
                elif destination_folder == FOLDER_MAP["ARCHIVE"]:
                    stats['archived'] += 1

                actions[destination_folder].append(msg.uid)
                
                # Enhanced logging with more details
                importance_indicators = []
                if 'Important' in predicted_labels:
                    importance_indicators.append("IMPORTANT")
                if 'Security' in predicted_labels:
                    importance_indicators.append("SECURITY")
                if 'Bill' in predicted_labels or 'Receipt' in predicted_labels:
                    importance_indicators.append("FINANCIAL")
                
                indicators_str = f" [{', '.join(importance_indicators)}]" if importance_indicators else ""
                
                log_msg = (f"[MOVE -> {destination_folder.split('/')[-1]}]{indicators_str} "
                          f"(Labels: {predicted_labels or {'None'}}, Reason: {reason}, "
                          f"Spam Score: {spam_score:.2f}) - {msg.from_} - {msg.subject}")
                logging.info(log_msg)

            # Summary statistics
            logging.info(f"\n=== TRIAGE SUMMARY for {self.account_config['name']} ===")
            logging.info(f"Processed: {stats['processed']} emails")
            logging.info(f"Spam detected: {stats['spam_detected']}")
            logging.info(f"Unsubscribed: {stats['unsubscribed']}")
            logging.info(f"Flagged (important): {stats['flagged']}")
            logging.info(f"Archived: {stats['archived']}")
            logging.info(f"Backed up: {stats['backed_up']}")

            if dry_run:
                logging.warning("[DRY RUN] No actual changes will be made.")
                for folder, uids in actions.items():
                    if uids: 
                        logging.info(f"Would move {len(uids)} message(s) to '{folder}'")
                return

            # Execute moves
            for folder, uids in actions.items():
                if uids:
                    logging.info(f"Moving {len(uids)} message(s) to '{folder}'...")
                    try:
                        mailbox.move(uids, folder)
                    except Exception as e:
                        logging.error(f"Failed to move messages to {folder}: {e}. They may remain in inbox.")
            
            logging.info(f"Enhanced inbox triage complete for {self.account_config['name']}.")
    
    def _route_email(self, predicted_labels: Set[str], msg) -> str:
        """Enhanced routing logic with flag vs archive for important emails."""
        
        # Highest Priority - Flag for immediate attention
        if ('Important' in predicted_labels and 
            ('Security' in predicted_labels or 'Bill' in predicted_labels or 
             'Receipt' in predicted_labels)):
            return FOLDER_MAP["FLAG"]
        
        # High Priority - Action Required
        if ('Bill' in predicted_labels or 'Order' in predicted_labels or 
            'Receipt' in predicted_labels):
            return FOLDER_MAP["ACTION"]
        
        # Important but not urgent - Flag for review
        if 'Important' in predicted_labels:
            return FOLDER_MAP["FLAG"]
            
        # Security notifications - Flag regardless
        if 'Security' in predicted_labels:
            return FOLDER_MAP["FLAG"]
            
        # Specific categories
        if 'Travel' in predicted_labels:
            return FOLDER_MAP["TRAVEL"]
        elif 'Finance' in predicted_labels:
            return FOLDER_MAP["FINANCE"]
        elif 'Personal' in predicted_labels:
            return FOLDER_MAP["PERSONAL"]
        elif ('Newsletter' in predicted_labels or 'Subscription' in predicted_labels):
            return FOLDER_MAP["NEWSLETTERS"]
        elif 'Marketing' in predicted_labels:
            return FOLDER_MAP["MARKETING"]
        elif ('Notification' in predicted_labels and not predicted_labels - {'Notification'}):
            # Pure notifications with no other classification
            return FOLDER_MAP["ARCHIVE"]
        
        # Low confidence - send to review queue
        if not predicted_labels:
            return FOLDER_MAP["REVIEW_QUEUE"]
            
        # Default fallback
        return FOLDER_MAP["ARCHIVE"]


# get_available_accounts function is now imported from config_manager

def run_multi_account_triage(limit: int, dry_run: bool, confidence: float, use_openai_fallback: bool):
    """Run triage across all configured accounts."""
    accounts = get_available_accounts()
    
    if not accounts:
        logging.error("No email accounts configured. Please set up accounts in .env file.")
        return
    
    logging.info(f"Found {len(accounts)} configured account(s)")
    
    overall_stats = {
        'total_accounts': len(accounts),
        'successful_accounts': 0,
        'total_processed': 0,
        'total_spam': 0,
        'total_unsubscribed': 0,
        'total_flagged': 0
    }
    
    for account in accounts:
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing account: {account['name']} ({account['email_user']})")
            logging.info(f"{'='*60}")
            
            system = EnhancedEmailTriageSystem(
                account_config=account,
                confidence_threshold=confidence,
                use_openai_fallback=use_openai_fallback
            )
            
            system.run_enhanced_triage(limit=limit, dry_run=dry_run)
            overall_stats['successful_accounts'] += 1
            
        except Exception as e:
            logging.error(f"Failed to process account {account['name']}: {e}")
            continue
    
    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"MULTI-ACCOUNT TRIAGE COMPLETE")
    logging.info(f"{'='*60}")
    logging.info(f"Total accounts configured: {overall_stats['total_accounts']}")
    logging.info(f"Successfully processed: {overall_stats['successful_accounts']}")
    logging.info(f"Failed accounts: {overall_stats['total_accounts'] - overall_stats['successful_accounts']}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced 'Zero Inbox' email triaging script with multi-account support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Infiscal Configuration
  python email_triage.py --infiscal-url https://infiscal.example.com --infiscal-token YOUR_TOKEN
  python email_triage.py --refresh-config
  
  # Label emails for a specific account
  python email_triage.py --label --account Primary
  
  # Manual triage of difficult emails
  python email_triage.py --manual-triage --account Primary
  
  # Train model from all labeled data
  python email_triage.py --train
  
  # Run triage on specific account
  python email_triage.py --run --account Work --dry-run
  
  # Run triage on all configured accounts
  python email_triage.py --run-all --dry-run
  
  # Process with custom settings
  python email_triage.py --run-all --limit 50 --confidence 0.7 --no-backup
        """
    )
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", action="store_true", 
                      help="Interactive labeling mode")
    group.add_argument("--manual-triage", action="store_true", 
                      help="Manual triage mode for difficult emails")
    group.add_argument("--train", action="store_true", 
                      help="Train model from labels")
    group.add_argument("--run", action="store_true", 
                      help="Run triage on specified account")
    group.add_argument("--run-all", action="store_true", 
                      help="Run triage on all configured accounts")
    group.add_argument("--list-accounts", action="store_true", 
                      help="List all configured accounts")

    # Account selection
    parser.add_argument("--account", type=str, 
                       help="Account name to process (required for --label and --run)")
    
    # Infiscal configuration
    parser.add_argument("--infiscal-url", type=str,
                       help="Infiscal server URL (overrides INFISCAL_URL env var)")
    parser.add_argument("--infiscal-token", type=str,
                       help="Infiscal authentication token (overrides INFISCAL_TOKEN env var)")
    parser.add_argument("--refresh-config", action="store_true",
                       help="Refresh configuration from Infiscal")
    
    # Processing options
    parser.add_argument("--limit", type=int, default=100, 
                       help="Max number of messages to fetch per account")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Confidence threshold for ML model decisions")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    
    # Feature toggles
    parser.add_argument("--no-fallback", action="store_true", 
                       help="Disable OpenAI fallback for low-confidence predictions")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Disable email backup before processing")
    parser.add_argument("--no-unsubscribe", action="store_true", 
                       help="Disable automatic unsubscribe attempts")
    
    args = parser.parse_args()

    # Handle Infiscal configuration
    if args.infiscal_url or args.infiscal_token:
        if args.infiscal_url:
            config_manager.infiscal_manager.infiscal_url = args.infiscal_url
        if args.infiscal_token:
            config_manager.infiscal_manager.infiscal_token = args.infiscal_token
            config_manager.infiscal_manager.session.headers.update({
                "Authorization": f"Bearer {args.infiscal_token}",
                "Content-Type": "application/json"
            })
        logging.info("Infiscal configuration updated from command line arguments")
    
    # Handle config refresh
    if args.refresh_config:
        config_manager.refresh_config()
        initialize_configuration()
        logging.info("Configuration refreshed from Infiscal")
        return

    # Handle list accounts
    if args.list_accounts:
        accounts = get_available_accounts()
        if accounts:
            print("Configured accounts:")
            for account in accounts:
                print(f"  - {account['name']}: {account['email_user']} ({account['imap_server']})")
        else:
            print("No accounts configured. Please set up accounts in Infiscal or .env file.")
        return

    # Handle run-all mode
    if args.run_all:
        run_multi_account_triage(
            limit=args.limit,
            dry_run=args.dry_run,
            confidence=args.confidence,
            use_openai_fallback=not args.no_fallback
        )
        return

    # For single account operations, require account specification
    if args.run or args.label:
        if not args.account:
            # Try to get the first available account
            accounts = get_available_accounts()
            if len(accounts) == 1:
                account_config = accounts[0]
                logging.info(f"Using single configured account: {account_config['name']}")
            else:
                parser.error("--account is required when multiple accounts are configured. "
                           "Use --list-accounts to see available accounts.")
        else:
            account_config = get_account_config(args.account)
            if not account_config['email_user']:
                parser.error(f"Account '{args.account}' not found. Use --list-accounts to see available accounts.")
    else:
        account_config = None

    # Create system instance
    if account_config:
        system = EnhancedEmailTriageSystem(
            account_config=account_config,
            confidence_threshold=args.confidence,
            use_openai_fallback=not args.no_fallback,
            enable_backup=not args.no_backup,
            enable_unsubscribe=not args.no_unsubscribe
        )

    # Execute requested operation
    if args.label:
        system.interactive_label(limit=args.limit)
    elif args.manual_triage:
        system.manual_triage_mode(limit=args.limit)
    elif args.train:
        # Training uses global data, no account needed
        temp_system = EnhancedEmailTriageSystem()
        temp_system.train_model()
    elif args.run:
        system.run_enhanced_triage(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()