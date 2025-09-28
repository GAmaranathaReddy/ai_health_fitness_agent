"""
Environment configuration module for AI Health & Fitness Agent
"""
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class to handle all environment variables"""

    # Hugging Face Configuration (Primary)
    HF_API_KEY: Optional[str] = os.environ.get("HF_API_KEY")
    HF_SPACE_URL: Optional[str] = os.environ.get("HF_SPACE_URL")

    # Ollama Configuration
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "llama3.2:1b")

    # App Configuration
    APP_PORT: int = int(os.environ.get("APP_PORT", "8501"))
    APP_HOST: str = os.environ.get("APP_HOST", "0.0.0.0")

    # Debug mode
    DEBUG: bool = os.environ.get("DEBUG", "False").lower() == "true"

    @classmethod
    def is_huggingface_configured(cls) -> bool:
        """Check if Hugging Face credentials are properly configured"""
        return all([cls.HF_API_KEY, cls.HF_SPACE_URL])

    @classmethod
    def get_ollama_config(cls) -> dict:
        """Get Ollama configuration based on available credentials"""
        # Priority 1: Hugging Face Spaces
        if cls.is_huggingface_configured():
            return {
                "host": cls.HF_SPACE_URL
            }
        # Priority 2: Local Ollama
        else:
            return {
                "host": cls.OLLAMA_HOST
            }

    @classmethod
    def print_config_status(cls):
        """Print current configuration status for debugging"""
        print("=== AI Health & Fitness Agent Configuration ===")
        print(f"Default Model: {cls.DEFAULT_MODEL}")
        print(f"App Port: {cls.APP_PORT}")
        print(f"App Host: {cls.APP_HOST}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("")
        print("Connection Priority:")
        if cls.is_huggingface_configured():
            print("✅ 1. Hugging Face Spaces (Active)")
            print(f"   URL: {cls.HF_SPACE_URL}")
            api_key_display = ('*' * 8 + '...' + cls.HF_API_KEY[-4:]
                              if cls.HF_API_KEY else 'None')
            print(f"   API Key: {api_key_display}")
        else:
            print("❌ 1. Hugging Face Spaces (Not configured)")

        fallback_status = ("✅ Active" if not cls.is_huggingface_configured()
                          else "⏸️ Fallback")
        print(f"🔧 2. Local Ollama ({fallback_status})")
        print(f"   Host: {cls.OLLAMA_HOST}")
        print("=" * 47)


# Create a global config instance
config = Config()
