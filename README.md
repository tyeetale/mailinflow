# 📧 Mailinflow

Simple, intelligent email triage system with Infiscal integration.

## 🚀 Quick Start

### 1. Setup Configuration

```bash
cp .env.template .env
# Edit .env with your Infiscal credentials and email settings
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
├── config_manager.py       # ⚙️ Clean Infiscal SDK integration
├── email-triage.py         # 📧 Core email processing logic
├── .env.template           # 🔑 Minimal configuration template
├── pyproject.toml          # 📦 Dependencies (including infiscal SDK)
├── README.md               # 📖 Simple getting started guide
└── instructions/           # 📚 All documentation organized here
    ├── CLAUDE.md           # Complete project documentation
    ├── STREAMLIT_README.md # Interface usage guide
    └── infiscal_config.template # API specification
```

## ⚙️ Configuration

### Minimal .env Setup

```bash
# Infiscal (Primary)
INFISCAL_URL="https://your-server.com"
INFISCAL_TOKEN="your-token"

# Fallback Email
EMAIL_USER="your-email@example.com"
EMAIL_PASS="your-password"
IMAP_SERVER="imap.gmail.com"
IMAP_PORT=993

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

See the `instructions/` folder for detailed guides:

- **CLAUDE.md** - Complete project documentation
- **STREAMLIT_README.md** - Interface usage guide
- **infiscal_config.template** - API specification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Need Help?** Check the `instructions/` folder for comprehensive documentation.
