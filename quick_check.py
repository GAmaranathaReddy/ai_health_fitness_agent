#!/usr/bin/env python3
"""
Simple diagnostic script to check your setup
"""
import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent / "ai_health_fitness_agent"))

try:
    from config import config
    print("✅ Configuration loaded successfully")
    config.print_config_status()

    if config.is_huggingface_configured():
        print(f"\n🤗 HF Space URL: {config.HF_SPACE_URL}")
        print("💡 Possible slow response causes:")
        print("   - Hugging Face Space is 'sleeping' and needs to cold start")
        print("   - Space is under heavy load")
        print("   - Network latency to Hugging Face servers")
        print("   - Large model taking time to respond")
        print("\n🚀 Solutions:")
        print("   1. Wait 30-60 seconds for first response (cold start)")
        print("   2. Try refreshing the page")
        print("   3. Use a smaller model (e.g., llama3.2:1b instead of 3b)")
        print("   4. Switch to local Ollama for development")
    else:
        print("\n🔧 Using local Ollama")
        print("💡 Check if Ollama is running: ollama serve")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Please install requirements: pip install python-dotenv ollama")
