#!/usr/bin/env python3
"""
Test script for Hugging Face Spaces integration
Run this script to verify your configuration before running the main app.
"""

import sys
import os
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent / "ai_health_fitness_agent"))

from config import config
from ollama import Client

def test_configuration():
    """Test the configuration setup"""
    print("🧪 Testing Configuration...")
    config.print_config_status()

    # Test environment variables
    if config.is_huggingface_configured():
        print("\n✅ Hugging Face configuration detected")
        print(f"Space URL: {config.HF_SPACE_URL}")
        print(f"API Key: {'*' * 8}...{config.HF_API_KEY[-4:] if config.HF_API_KEY else 'None'}")
    else:
        print("\n❌ Hugging Face not configured")
        if not config.HF_API_KEY:
            print("   - Missing HF_API_KEY")
        if not config.HF_SPACE_URL:
            print("   - Missing HF_SPACE_URL")

def test_connection():
    """Test connection to Ollama server"""
    print("\n🔗 Testing Connection...")

    try:
        ollama_config = config.get_ollama_config()
        print(f"Connecting to: {ollama_config['host']}")

        client = Client(host=ollama_config["host"])

        # Test basic connection
        models = client.list()
        print("✅ Connection successful!")
        print(f"Available models: {[m.model for m in models.models]}")

        return client
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Please check:")
        print("  - Your Space is running")
        print("  - API key is correct")
        print("  - Space URL is accessible")
        return None

def test_model_inference(client):
    """Test model inference"""
    print(f"\n🤖 Testing Model Inference with {config.DEFAULT_MODEL}...")

    try:
        # Simple test message
        response = client.generate(
            model=config.DEFAULT_MODEL,
            prompt="Hello! Please respond with 'Health and fitness assistant ready!'"
        )

        print("✅ Model inference successful!")
        print(f"Response: {response['response'][:100]}...")
        return True

    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        print("Possible issues:")
        print(f"  - Model '{config.DEFAULT_MODEL}' not available")
        print("  - Insufficient API permissions")
        print("  - Space resource limits")
        return False

def main():
    """Run all tests"""
    print("🏋️‍♂️ AI Health & Fitness Agent - Configuration Test")
    print("=" * 55)

    # Test 1: Configuration
    test_configuration()

    # Test 2: Connection
    client = test_connection()
    if not client:
        print("\n❌ Cannot proceed with model testing due to connection issues")
        sys.exit(1)

    # Test 3: Model inference
    if test_model_inference(client):
        print("\n🎉 All tests passed! Your configuration is ready.")
        print("You can now run: python run.py")
    else:
        print("\n⚠️ Model inference failed, but connection works.")
        print("Check your model name and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
