# 📧 Mailinflow

Simple, intelligent email triage system with environment-based configuration.

## 🚀 Quick Start

### 1. Setup Configuration

```bash
cp .env.template .env
# Edit .env with your email account details
```

### 2. Launch Interface

```bash
python main.py
```

The Streamlit interface will open in your browser at `http://localhost:8501`.

## 📁 Project Structure

```
mailinflow/
├── main.py                 # 🚀 Simple launcher (just run: python main.py)
├── streamlit_triage.py     # 🌐 Web interface for manual triage
├── config_manager.py       # ⚙️ Simple environment-based configuration
├── email-triage.py         # 📧 Core email processing logic
├── .env.template           # 🔑 Configuration template
├── pyproject.toml          # 📦 Dependencies
├── README.md               # 📖 Simple getting started guide
└── instructions/           # 📚 All documentation organized here
```

## ⚙️ Configuration

### Simple .env Setup

```bash
# Single Account
EMAIL_USER="your-email@example.com"
EMAIL_PASS="your-password"
IMAP_SERVER="imap.gmail.com"
IMAP_PORT=993

# Multiple Accounts (optional)
ACCOUNT_1_NAME="Primary"
ACCOUNT_1_EMAIL_USER="primary@example.com"
ACCOUNT_1_EMAIL_PASS="primary-password"

# System Settings
BACKUP_ENABLED=true
UNSUBSCRIBE_ENABLED=true
SPAM_CONFIDENCE_THRESHOLD=0.8

# Optional
OPENAI_API_KEY="sk-..."
```

## 🔧 Development

### Install Dependencies

```bash
pip install -e .
```

### Run Tests

```bash
python -m pytest
```

## 📚 Documentation

See the `instructions/` folder for detailed guides and documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Need Help?** Check the `instructions/` folder for comprehensive documentation.
