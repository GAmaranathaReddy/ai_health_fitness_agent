# Environment Variables Setup Guide

## Overview
This guide explains how to set up and use environment variables in your AI Health & Fitness Agent application.

## What are Environment Variables?
Environment variables are key-value pairs that store configuration data outside of your code. They're useful for:
- Keeping sensitive information (like API keys, passwords) secure
- Configuring different settings for development, testing, and production
- Making your application more flexible and portable

## Setting up Environment Variables

### Method 1: Using a .env file (Recommended)

1. **Create a .env file** in your project root directory:
```bash
touch .env
```

2. **Add your environment variables** to the .env file:
```bash
# Hugging Face Configuration (for Ollama deployed on HF Spaces)
HF_API_KEY=hf_your_api_key_here
HF_SPACE_URL=https://your-username-space-name.hf.space

# LLM Configuration
DEFAULT_MODEL=llama3.2:1b
OLLAMA_HOST=http://localhost:11434

# App Configuration
APP_PORT=8501
APP_HOST=0.0.0.0
DEBUG=false
```

3. **Add .env to .gitignore** (IMPORTANT for security):
```bash
echo ".env" >> .gitignore
```

### Method 2: System Environment Variables

You can also set environment variables at the system level:

**On macOS/Linux:**
```bash
export HF_API_KEY="hf_your_api_key_here"
export HF_SPACE_URL="https://your-username-space-name.hf.space"
export DEFAULT_MODEL="llama3.2:1b"
```

**On Windows:**
```cmd
set HF_API_KEY=hf_your_api_key_here
set HF_SPACE_URL=https://your-username-space-name.hf.space
set DEFAULT_MODEL=llama3.2:1b
```

### Method 3: Streamlit Secrets (For Streamlit Cloud)

If deploying to Streamlit Cloud, create a `secrets.toml` file:
```toml
[huggingface]
api_key = "hf_your_api_key_here"
space_url = "https://your-username-space-name.hf.space"

[ollama]
host = "http://localhost:11434"
default_model = "llama3.2:1b"
```

## Environment Variables Used

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HF_API_KEY` | Hugging Face API key for authentication | None | No |
| `HF_SPACE_URL` | Hugging Face Space URL for Ollama | None | No |
| `OLLAMA_HOST` | Ollama server host | http://localhost:11434 | No |
| `DEFAULT_MODEL` | Default LLM model to use | llama3.2:1b | No |
| `APP_PORT` | Port for Streamlit app | 8501 | No |
| `APP_HOST` | Host for Streamlit app | 0.0.0.0 | No |
| `DEBUG` | Enable debug mode | false | No |

## Using Environment Variables in Code

The application uses a configuration module (`config.py`) that automatically loads environment variables:

```python
from .config import config

# Access environment variables
model_name = config.DEFAULT_MODEL
host = config.OLLAMA_HOST
debug_mode = config.DEBUG

# Check if Hugging Face is configured
if config.is_huggingface_configured():
    print("Using Hugging Face Spaces for Ollama")
else:
    print("Using local Ollama instance")
```

## Security Best Practices

1. **Never commit .env files** to version control
2. **Use different .env files** for different environments (dev, prod, test)
3. **Keep sensitive data** in environment variables, not in code
4. **Use strong passwords** and rotate them regularly
5. **Limit access** to production environment variables

## Troubleshooting

### Common Issues:

1. **"Environment variable not found"**
   - Check if .env file exists and is in the correct location
   - Verify variable names match exactly (case-sensitive)
   - Ensure python-dotenv is installed: `pip install python-dotenv`

2. **"Connection refused to Ollama"**
   - Check if OLLAMA_HOST is correct
   - Verify Ollama is running on the specified host
   - Test ngrok configuration if using remote access

3. **"Module not found: config"**
   - Ensure config.py is in the correct directory
   - Check import path in your main file

### Debug Configuration:
Enable debug mode to see current configuration:
```python
from .config import config
config.print_config_status()
```

## Example Usage Scenarios

### Scenario 1: Local Development
```bash
# .env
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=llama3.2:1b
DEBUG=true
```

### Scenario 2: Hugging Face Spaces
```bash
# .env
HF_API_KEY=hf_abcd1234...
HF_SPACE_URL=https://myusername-ollama.hf.space
DEFAULT_MODEL=llama3.2:3b
```

### Scenario 3: Production Deployment
```bash
# .env
HF_API_KEY=hf_prod_key...
HF_SPACE_URL=https://company-ai-backend.hf.space
DEFAULT_MODEL=llama3.2:7b
APP_PORT=8080
DEBUG=false
```

## Installation Requirements

Make sure you have the required packages:
```bash
pip install python-dotenv streamlit phidata ollama
```

Or update your requirements.txt:
```
streamlit
phidata
ollama
pytest
python-dotenv
```
