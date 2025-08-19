# 📧 Mailinflow Streamlit Triage Interface

A modern web-based interface for manually categorizing difficult emails in the mailinflow system.

## 🚀 Quick Start

### Option 1: Use the Launcher Script (Recommended)

```bash
python3 launch_triage_ui.py
```

### Option 2: Direct Streamlit Launch

```bash
streamlit run streamlit_triage.py
```

The interface will automatically open in your default web browser at `http://localhost:8501`.

## 🎯 Features

### 📊 **Configuration Dashboard**

- **Infiscal Status**: Shows connection status and server URL
- **Account Selection**: Choose which email account to work with
- **System Settings**: View current configuration values
- **Label Overview**: See all available classification categories

### ✉️ **Manual Email Triage**

- **Email Preview**: View sender, subject, date, and content
- **Multi-Label Selection**: Choose from standard classification labels
- **Custom Labels**: Create new categories for edge cases
- **Folder Routing**: Direct emails to appropriate folders
- **Decision Tracking**: All choices saved as training data

### 📈 **Training Data Management**

- **Real-time Updates**: See training data as you work
- **Export Options**: Download data as CSV for analysis
- **Data Cleanup**: Clear training data when needed
- **Infiscal Sync**: Automatic synchronization with central system

## ⚙️ Configuration

### Environment Variables

The interface uses the same configuration as the main mailinflow system:

```bash
# Infiscal (Primary)
INFISCAL_URL="https://your-infiscal-server.com"
INFISCAL_TOKEN="your-token"

# Fallback Configuration
ACCOUNT_1_NAME="Primary"
ACCOUNT_1_EMAIL_USER="your-email@example.com"
ACCOUNT_1_EMAIL_PASS="your-password"
# ... other settings
```

### Configuration Priority

1. **Infiscal** (if available and configured)
2. **Environment Variables** (fallback)
3. **Default Values** (last resort)

## 🎨 Interface Layout

### Sidebar

- **Configuration Status**: Infiscal connection and account selection
- **Triage Options**: Set processing limits and start triage
- **Refresh Controls**: Update configuration from Infiscal

### Main Content

- **Configuration Overview**: System settings and available labels
- **Email Processing**: Individual email triage interface
- **Training Summary**: Data export and management tools

## 📱 Usage Workflow

### 1. **Setup & Configuration**

- Check Infiscal connection status
- Select email account to work with
- Review system configuration

### 2. **Start Triage Session**

- Click "🚀 Start Manual Triage"
- Set processing limits
- Begin email categorization

### 3. **Process Emails**

- Review email content and metadata
- Select appropriate classification labels
- Choose target folder for routing
- Save decision to training data

### 4. **Review & Export**

- View accumulated training data
- Export data for analysis
- Sync with Infiscal if available

## 🔧 Technical Details

### Dependencies

- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation and display
- **Config Manager**: Integration with mailinflow configuration
- **Email Triage**: Core classification logic

### File Structure

```
mailinflow/
├── streamlit_triage.py      # Main Streamlit interface
├── launch_triage_ui.py      # Launcher script
├── config_manager.py        # Configuration management
├── email-triage.py          # Core email processing
├── .env.template            # Configuration template
└── requirements.txt         # Python dependencies
```

### Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support
- **Mobile**: Responsive design

## 🚨 Troubleshooting

### Common Issues

#### "Module not found" errors

```bash
pip install -r requirements.txt
```

#### Configuration not loading

- Check `.env` file exists and is properly formatted
- Verify Infiscal credentials if using Infiscal
- Run `python email-triage.py --refresh-config`

#### Streamlit not launching

```bash
# Check if Streamlit is installed
pip show streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

#### Port conflicts

```bash
# Use different port
streamlit run streamlit_triage.py --server.port 8502
```

### Debug Mode

```bash
# Enable debug logging
export STREAMLIT_LOG_LEVEL=debug
streamlit run streamlit_triage.py
```

## 🔄 Integration with Main System

### Training Data Flow

1. **Manual Decisions** → Saved to local file
2. **Local File** → Synced to Infiscal (if available)
3. **Infiscal Data** → Used for ML model training
4. **Improved Model** → Better automatic classification

### Command Line Integration

The Streamlit interface works alongside the command-line tools:

- **Streamlit**: Manual triage and data review
- **Command Line**: Automated processing and training
- **Shared Data**: Both use the same training data files

## 📚 Advanced Usage

### Custom Labels

Create new classification categories for your specific needs:

- **Business-specific**: Industry, department, priority
- **Workflow-based**: Action required, follow-up needed
- **Temporal**: Urgent, scheduled, archive

### Batch Processing

- Process multiple emails with similar patterns
- Use consistent labeling for better ML training
- Export decisions for team review

### Data Analysis

- Review labeling patterns over time
- Identify common email types
- Optimize folder structure

## 🤝 Contributing

### Development Setup

```bash
# Clone repository
git clone <your-repo>
cd mailinflow

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run streamlit_triage.py --server.runOnSave true
```

### Customization

- Modify `streamlit_triage.py` for interface changes
- Update `config_manager.py` for new configuration sources
- Extend `email-triage.py` for new processing features

---

**Need Help?** Check the main `CLAUDE.md` file for comprehensive project documentation.
