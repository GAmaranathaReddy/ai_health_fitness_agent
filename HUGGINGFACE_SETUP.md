# Hugging Face Spaces Integration Guide

## Overview
This guide explains how to configure your AI Health & Fitness Agent to work with Ollama deployed on Hugging Face Spaces.

## Prerequisites
1. A Hugging Face account
2. An Ollama server deployed on Hugging Face Spaces
3. A Hugging Face API token

## Step 1: Get Your Hugging Face Credentials

### 1.1 Get Your API Token
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with appropriate permissions
3. Copy the token (it starts with `hf_...`)

### 1.2 Get Your Space URL
1. Go to your Hugging Face Space where Ollama is deployed
2. Copy the Space URL (format: `https://your-username-space-name.hf.space`)

## Step 2: Configure Environment Variables

### Option A: Using .env file (Recommended)

Create a `.env` file in your project root:

```bash
# Hugging Face Configuration
HF_API_KEY=hf_your_actual_api_key_here
HF_SPACE_URL=https://your-username-space-name.hf.space

# Optional: Model Configuration
DEFAULT_MODEL=llama3.2:1b
```

### Option B: System Environment Variables

```bash
export HF_API_KEY="hf_your_actual_api_key_here"
export HF_SPACE_URL="https://your-username-space-name.hf.space"
export DEFAULT_MODEL="llama3.2:1b"
```

### Option C: Streamlit Secrets (for Streamlit Cloud)

Create `.streamlit/secrets.toml`:

```toml
[huggingface]
api_key = "hf_your_actual_api_key_here"
space_url = "https://your-username-space-name.hf.space"

[ollama]
default_model = "llama3.2:1b"
```

## Step 3: Verify Configuration

Run the following command to check your configuration:

```python
from ai_health_fitness_agent.config import config
config.print_config_status()
```

You should see:
```
=== AI Health & Fitness Agent Configuration ===
Default Model: llama3.2:1b
...
Connection Priority:
✅ 1. Hugging Face Spaces (Active)
   URL: https://your-username-space-name.hf.space
   API Key: ********...xyz1
...
```

## Step 4: Common Issues and Troubleshooting

### Issue 1: "Connection refused" or "Unauthorized"
**Solution:**
- Verify your HF_API_KEY is correct and active
- Ensure your Space is running and publicly accessible
- Check if your Space URL is correct (no trailing slash)

### Issue 2: "Model not found"
**Solution:**
- Verify the model name exists in your Ollama deployment
- Check available models by visiting your Space URL in browser
- Update DEFAULT_MODEL environment variable

### Issue 3: "Authentication failed"
**Solution:**
- Regenerate your Hugging Face API token
- Ensure the token has appropriate permissions
- Check if the token is correctly set in environment variables

## Step 5: Testing Your Setup

### Test 1: Basic Connection
```python
from ollama import Client
from ai_health_fitness_agent.config import config

# Test connection
client_config = config.get_ollama_config()
client = Client(host=client_config["host"])

# Try to list models
try:
    models = client.list()
    print("✅ Connection successful!")
    print(f"Available models: {[m.model for m in models.models]}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Test 2: Model Inference
```python
from ai_health_fitness_agent.config import config
from phi.model.ollama import Ollama

# Test model inference
try:
    llm = Ollama(id=config.DEFAULT_MODEL, client=client)
    response = llm.invoke("Hello! Can you help with fitness advice?")
    print("✅ Model inference successful!")
    print(f"Response: {response}")
except Exception as e:
    print(f"❌ Model inference failed: {e}")
```

## Step 6: Security Best Practices

1. **Never commit your .env file** to version control
2. **Use environment variables** in production
3. **Rotate your API keys** regularly
4. **Limit API key permissions** to minimum required
5. **Monitor your Space usage** on Hugging Face

## Step 7: Deployment Options

### Local Development
```bash
# .env
HF_API_KEY=hf_your_token
HF_SPACE_URL=https://your-space.hf.space
DEBUG=true
```

### Streamlit Cloud
Use Streamlit's secrets management through the web interface.

### Docker Deployment
```dockerfile
ENV HF_API_KEY=hf_your_token
ENV HF_SPACE_URL=https://your-space.hf.space
```

## Example Hugging Face Space URLs

Common formats:
- `https://username-spacename.hf.space`
- `https://username-ollama-server.hf.space`
- `https://yourcompany-ai-backend.hf.space`

## Fallback Strategy

The application uses a priority system:
1. **Primary**: Hugging Face Spaces (if configured)
2. **Secondary**: Ngrok with Basic Auth (if configured)
3. **Fallback**: Local Ollama server

This ensures your app works even if one service is unavailable.

## Support

If you encounter issues:
1. Check the configuration status using `config.print_config_status()`
2. Verify your Hugging Face Space is running
3. Test your API key at https://huggingface.co/settings/tokens
4. Check Hugging Face Space logs for errors
