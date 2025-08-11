#!/usr/bin/env python3
"""
email_triage.py (v3 - Zero Inbox Edition)

A script to achieve "inbox zero" by processing every unread email and moving it
to a categorized folder for review or archival.

Modes:
 - Labeling:  python email_triage.py --label
 - Train:     python email_triage.py --train
 - Run:       python email_triage.py --run
"""

import os
import argparse
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Set
from collections import defaultdict

from imap_tools import MailBox, AND
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load as joblib_load
from openai import OpenAI

# -------------------------
# Configuration
# -------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Configuration ---
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.gmail.com")
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS") # Use an "App Password"

# --- Pre-defined Labels for Classification ---
# More granular labels for better sorting.
CLASSIFICATION_LABELS = [
    "Important", "Personal", "Subscription", "Bill", "Order", "Travel", "Finance"
]

# --- Folder Configuration (The Zero Inbox System) ---
# Tip: Create these folders in your email client first.
TRIAGE_PREFIX = os.getenv("TRIAGE_PREFIX", "Triage")
FOLDER_MAP = {
    # High-priority, direct action needed
    "ACTION": f"{TRIAGE_PREFIX}/01_Action_Required",
    # Specific categories for review
    "TRAVEL": f"{TRIAGE_PREFIX}/02_Travel",
    "FINANCE": f"{TRIAGE_PREFIX}/03_Finance",
    "PERSONAL": f"{TRIAGE_PREFIX}/04_Personal",
    # Lower-priority, informational
    "NEWSLETTERS": f"{TRIAGE_PREFIX}/05_Newsletters",
    # Catch-all for everything else unimportant
    "ARCHIVE": "Archive",
}

# --- Model & Data Paths ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", "email_triage_zeroinbox_model.joblib"))
DATA_PATH = Path(os.getenv("DATA_PATH", "email_labels_zeroinbox.jsonl"))

# Optional OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class EmailTriageSystem:
    def __init__(self, confidence_threshold: float = 0.5, use_openai_fallback: bool = True):
        self.confidence_threshold = confidence_threshold
        self.use_openai_fallback = use_openai_fallback and openai_client is not None
        self.mlb = MultiLabelBinarizer(classes=CLASSIFICATION_LABELS)
        self.model: Optional[OneVsRestClassifier] = self._load_model()

        if not EMAIL_USER or not EMAIL_PASS:
            raise ValueError("EMAIL_USER and EMAIL_PASS must be set in the environment.")

    def _connect(self) -> MailBox:
        try:
            return MailBox(IMAP_SERVER).login(EMAIL_USER, EMAIL_PASS, initial_folder="INBOX")
        except Exception as e:
            logging.error(f"Failed to connect to IMAP server: {e}")
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

    def run_triage(self, limit: int = 200, dry_run: bool = False):
        """Processes every unread email and moves it to a destination folder."""
        if not self.model and not self.use_openai_fallback:
            logging.error("No model is available and OpenAI fallback is disabled. Aborting.")
            return

        with self._connect() as mailbox:
            unread_msgs = list(self._fetch_unread(mailbox, limit))
            if not unread_msgs:
                logging.info("Inbox is already clean. Zero unread messages.")
                return

            # Use defaultdict to simplify adding UIDs
            actions = defaultdict(list)
            
            for msg in unread_msgs:
                text_content = self._flatten_text(msg)
                predicted_labels = set()
                reason = "model"

                if self.model:
                    probabilities = self.model.predict_proba([text_content])[0]
                    predicted_labels = {
                        label for i, label in enumerate(self.mlb.classes_) 
                        if probabilities[i] >= self.confidence_threshold
                    }
                
                # If model is unsure OR doesn't exist, and fallback is on, use AI
                if not predicted_labels and self.use_openai_fallback:
                    reason = "openai_fallback"
                    predicted_labels = self._openai_classify(msg)

                # --- The Email Router (Prioritized Action Logic) ---
                destination_folder = FOLDER_MAP["ARCHIVE"] # Default to Archive
                if 'Bill' in predicted_labels or 'Order' in predicted_labels:
                    destination_folder = FOLDER_MAP["ACTION"]
                elif 'Travel' in predicted_labels:
                    destination_folder = FOLDER_MAP["TRAVEL"]
                elif 'Finance' in predicted_labels:
                    destination_folder = FOLDER_MAP["FINANCE"]
                elif 'Personal' in predicted_labels:
                    destination_folder = FOLDER_MAP["PERSONAL"]
                elif 'Important' in predicted_labels: # General important catch-all
                    destination_folder = FOLDER_MAP["ACTION"]
                elif 'Subscription' in predicted_labels:
                    destination_folder = FOLDER_MAP["NEWSLETTERS"]

                actions[destination_folder].append(msg.uid)
                log_msg = (f"[MOVE -> {destination_folder}] (Labels: {predicted_labels or {'None'}}, "
                           f"Reason: {reason}) - {msg.subject}")
                logging.info(log_msg)

            if dry_run:
                logging.warning("[DRY RUN] No actual changes will be made.")
                for folder, uids in actions.items():
                    if uids: logging.info(f"Would move {len(uids)} message(s) to '{folder}'")
                return

            for folder, uids in actions.items():
                if uids:
                    logging.info(f"Moving {len(uids)} message(s) to '{folder}'...")
                    try:
                        mailbox.move(uids, folder)
                    except Exception as e:
                        logging.error(f"Failed to move messages to {folder}: {e}. They may remain in inbox.")
            
            logging.info("Inbox triage complete.")


def main():
    parser = argparse.ArgumentParser(description="A 'Zero Inbox' email triaging script.")
    # ... (parser arguments remain the same) ...
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", action="store_true", help="Interactive labeling mode")
    group.add_argument("--train", action="store_true", help="Train model from labels")
    group.add_argument("--run", action="store_true", help="Run triage on unread messages")

    parser.add_argument("--limit", type=int, default=100, help="Max number of messages to fetch")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for the local model's decision")
    parser.add_argument("--no-fallback", action="store_true", help="Disable OpenAI fallback for low-confidence predictions")
    parser.add_argument("--dry-run", action="store_true", help="In run mode, show what would be done without making changes")
    args = parser.parse_args()

    system = EmailTriageSystem(
        confidence_threshold=args.confidence,
        use_openai_fallback=not args.no_fallback
    )

    if args.label:
        system.interactive_label(limit=args.limit)
    elif args.train:
        system.train_model()
    elif args.run:
        system.run_triage(limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()