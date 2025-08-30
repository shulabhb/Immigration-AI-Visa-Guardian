#!/usr/bin/env python3
"""
Startup script for Immigration Guardian RAG Chatbot
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_ollama():
    """Start Ollama if not running"""
    print("🔄 Starting Ollama...")
    try:
        # Check if llama3.1:8b is available
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not any("llama3.1:8b" in model.get("name", "") for model in models):
                print("📥 Downloading llama3.1:8b model...")
                subprocess.run(["ollama", "pull", "llama3.1:8b"], check=True)
                print("✅ Model downloaded successfully!")
            else:
                print("✅ Ollama is running and model is available!")
            return True
    except:
        print("❌ Ollama is not running. Please start Ollama first:")
        print("   ollama serve")
        return False

def main():
    print("🤖 Immigration Guardian RAG Chatbot")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app/rag_chatbot.py").exists():
        print("❌ Please run this script from the visa_guardian directory")
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        if not start_ollama():
            sys.exit(1)
    
    print("\n🚀 Starting the chatbot...")
    print("📱 Web interface will be available at: http://localhost:8000/ui")
    print("🔌 API will be available at: http://localhost:8000")
    print("💬 CLI interface available at: http://localhost:8000/cli")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start the FastAPI server
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
