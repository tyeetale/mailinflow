# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the mailinflow repository.

## Project Overview

**mailinflow** is an advanced ML-based email triage system designed to achieve "inbox zero" by automatically categorizing and filing unread emails into organized folders. It's a comprehensive, free alternative to commercial email management services with enterprise-grade features:

### Core Capabilities

- **Multi-account support** for processing multiple email accounts simultaneously
- **Intelligent spam detection** with auto-detection and review queues
- **Automatic unsubscribe functionality** for marketing emails
- **Email backup system** preserving processed emails with metadata
- **Flag vs Archive logic** for important email prioritization
- **Enhanced classification** with 13+ categories including security alerts
- **Manual triage mode** for handling difficult emails and building training data
- **Machine learning pipeline** with scikit-learn and optional OpenAI GPT fallback

## Key Architecture

The main application is `email-triage.py` - a comprehensive script built around:

### EnhancedEmailTriageSystem Class

Core system supporting:

- Multi-account processing with individual configurations
- Spam detection with configurable thresholds
- Backup functionality with account-specific organization
- Unsubscribe logic with pattern matching
- Smart routing with priority-based folder assignment
- Comprehensive statistics and enhanced logging

### Operational Modes

- `--label --account ACCOUNT`: Interactive labeling for training data collection
- `--manual-triage --account ACCOUNT`: Manual triage for difficult emails
- `--train`: Train ML model from all labeled data (including manual triage data)
- `--run --account ACCOUNT`: Execute triage on single account
- `--run-all`: Process all configured accounts
- `--list-accounts`: Show configured accounts

## Enhanced Email Processing Flow

1. **Account Selection**: Processes single account or all configured accounts
2. **Email Backup**: Backs up emails with metadata before processing
3. **Spam Detection**: Multi-factor spam scoring with auto-detection and review queues
4. **ML Classification**: 13+ category classification with confidence scoring
5. **Unsubscribe Logic**: Automatic unsubscribe attempts for marketing emails
6. **Smart Routing**: Flag vs archive logic with priority-based folder assignment
7. **Manual Triage Integration**: Difficult emails routed to manual review
8. **Statistics & Logging**: Comprehensive processing statistics and enhanced logging

## Enhanced Classification & Routing

### High Priority (Flagged)

- **Important + Security/Bills** → `Triage/00_Flagged_Important`
- **Action Required** → `Triage/01_Action_Required`

### Category-Specific Folders

- **Travel** → `Triage/02_Travel`
- **Finance** → `Triage/03_Finance`
- **Personal** → `Triage/04_Personal`
- **Newsletters** → `Triage/05_Newsletters`
- **Marketing** → `Triage/06_Marketing`
- **Unsubscribed** → `Triage/07_Unsubscribed`

### Spam & Review Handling

- **Auto-detected spam** → `Spam/Auto_Detected`
- **Potential spam** → `Review/Potential_Spam`
- **Manual review queue** → `Review/Manual_Review`
- **Manual triage** → `Review/Manual_Triage`

### Default Actions

- **Archive**: Low-priority, informational emails
- **Backup**: All processed emails preserved with metadata

## Manual Triage Mode

### Purpose

The manual triage mode addresses the need for human decision-making on emails that are difficult to classify automatically. This mode serves two critical functions:

1. **Immediate Email Management**: Handle difficult emails in real-time
2. **Training Data Generation**: Build a comprehensive dataset of human decisions for ML model improvement

### Features

- **Review Queue Integration**: Processes emails from review folders and inbox
- **Custom Label Creation**: Allows creation of new classification categories
- **Folder Movement**: Direct email routing to appropriate folders
- **Training Data Storage**: All decisions stored with metadata for model retraining
- **Flexible Input**: Supports labels, folder moves, and custom categories

### Usage

```bash
# Manual triage for specific account
python email-triage.py --manual-triage --account Primary

# Manual triage with custom limit
python email-triage.py --manual-triage --account Work --limit 100
```

### Manual Triage Labels

- **Review_Needed**: Emails requiring further analysis
- **Unclear_Purpose**: Ambiguous emails
- **Mixed_Categories**: Emails spanning multiple categories
- **Edge_Case**: Unusual email types
- **Custom_Label**: User-defined categories

## Development Commands

The project uses uv for dependency management with Python 3.11+ required.

### Environment Setup

```bash
# Install dependencies
uv sync

# Copy and configure environment template
cp .env.template .env
# Edit .env with your account credentials

# List configured accounts
python3 email-triage.py --list-accounts
```

### Multi-Account Operations

```bash
# Interactive labeling for specific account
python3 email-triage.py --label --account Primary

# Manual triage of difficult emails
python3 email-triage.py --manual-triage --account Work

# Train ML model from all labeled data
python3 email-triage.py --train

# Test single account (dry run)
python3 email-triage.py --run --account Work --dry-run

# Process all accounts (dry run)
python3 email-triage.py --run-all --dry-run

# Live processing with custom settings
python3 email-triage.py --run-all --limit 50 --confidence 0.7
```

### Feature Control

```bash
# Disable specific features
python3 email-triage.py --run-all --no-backup --no-unsubscribe --no-fallback
```

## Required Configuration

The system supports multiple configuration sources with Infiscal as the primary option:

### Option 1: Infiscal Integration (Recommended)

Connect directly to Infiscal for centralized configuration management:

```bash
# Set environment variables
export INFISCAL_URL="https://your-infiscal-server.com"
export INFISCAL_TOKEN="your-authentication-token"

# Or use command line arguments
python email-triage.py --infiscal-url https://infiscal-server.com --infiscal-token YOUR_TOKEN

# Refresh configuration from Infiscal
python email-triage.py --refresh-config
```

**Expected Infiscal API Endpoints:**

- `GET /health` - Health check
- `GET /api/email-accounts` - Email account configurations
- `GET /api/system-config` - System-wide settings
- `POST /api/training-data` - Training data updates

### Option 2: Environment Variables (Fallback)

Create `.env` file from template for local configuration:

```bash
# Multi-Account Setup
ACCOUNT_1_NAME="Primary"
ACCOUNT_1_EMAIL_USER="your-email@example.com"
ACCOUNT_1_EMAIL_PASS="your-app-password"
ACCOUNT_1_IMAP_SERVER="imap.gmail.com"
ACCOUNT_1_IMAP_PORT=993

# Additional accounts follow same pattern (ACCOUNT_2_, ACCOUNT_3_, etc.)
```

### Global Settings

- `OPENAI_API_KEY`: Optional, for AI fallback classification
- `BACKUP_ENABLED`: Enable email backup (default: true)
- `BACKUP_PATH`: Backup directory path (default: ./email_backups)
- `UNSUBSCRIBE_ENABLED`: Enable auto-unsubscribe (default: true)
- `SPAM_CONFIDENCE_THRESHOLD`: Spam auto-detection threshold (default: 0.8)
- `REVIEW_CONFIDENCE_THRESHOLD`: Spam review threshold (default: 0.6)

## Data Files & Structure

### Training Data

- `email_labels_zeroinbox.jsonl`: Training data from interactive labeling and manual triage
- `email_triage_zeroinbox_model.joblib`: Trained ML model

### Backup Structure

- `email_backups/[account_name]/`: Account-specific email backups with metadata
- Organized by date and processing type

### Configuration

- **Infiscal**: Primary configuration source for accounts and system settings
- `.env`: Fallback configuration (from .env.template)
- `infiscal_config.template`: Infiscal API specification and setup guide
- `pyproject.toml`: Dependencies including requests, openai, imap-tools

## Project Intentions & Philosophy

### Zero Inbox Philosophy

The system is built around the principle that every email should have a clear purpose and destination. No email should remain in the inbox indefinitely.

### Continuous Learning

The manual triage mode creates a feedback loop where human decisions improve the ML model over time, leading to better automatic classification.

### Data Preservation

All emails are backed up before processing, ensuring no data loss while maintaining a clean inbox.

### User Empowerment

Users maintain full control over email organization while leveraging ML assistance for routine categorization.

### Scalability

Multi-account support allows users to manage multiple email accounts with consistent organization and learning across all accounts.

## Recent Enhancements

### Manual Triage Integration

- Added `--manual-triage` mode for handling difficult emails
- Integrated with review queue system
- Custom label creation capabilities
- Training data generation from human decisions

### Environment File Cleanup

- Removed duplicate `.env.example` file
- Consolidated configuration in `.env.template`
- Streamlined setup process

### Infiscal Integration

- Added direct Infiscal connectivity for configuration management
- Centralized account and system configuration
- Automatic fallback to environment variables
- Command-line configuration options
- Training data synchronization with Infiscal

### Enhanced Training Pipeline

- Manual triage data integrated into training process
- Improved model accuracy through human feedback
- Custom label support in classification system

## Future Development Directions

### Planned Features

- Email thread analysis and grouping
- Calendar integration for travel/event emails
- Advanced unsubscribe pattern detection
- Email importance scoring based on sender history
- Integration with task management systems

### Performance Optimizations

- Batch processing for large email volumes
- Caching for frequently accessed patterns
- Parallel processing for multi-account operations

### User Experience

- Web interface for manual triage
- Email preview improvements
- Bulk operations for similar emails
- Advanced filtering and search capabilities
