#!/bin/bash
# AI Short-Video Creator - Development Setup Script

set -e

echo "🚀 AI Short-Video Creator - Development Setup"
echo "============================================="

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>/dev/null | cut -d' ' -f2 | cut -d'.' -f1,2)
if ! command -v python3 &> /dev/null || [[ $(echo "$python_version < 3.8" | bc -l) == 1 ]]; then
    echo "❌ Python 3.8+ is required but not found"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "✅ Python $python_version found"

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg not found. Installing..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg python3-dev build-essential
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg python3-devel gcc
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg python3 base-devel
        else
            echo "Please install FFmpeg manually for your Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "Please install Homebrew and run: brew install ffmpeg"
            exit 1
        fi
    else
        echo "Please install FFmpeg manually for your operating system"
        exit 1
    fi
else
    echo "✅ FFmpeg found"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for development)
echo "🧠 Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download required models
echo "🤖 Downloading AI models..."
python -c "
import nltk
try:
    nltk.download('punkt')
    print('✅ NLTK data downloaded')
except Exception as e:
    print(f'⚠️  NLTK download failed: {e}')

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print('📥 Downloading BART model...')
    AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
    print('✅ BART model downloaded')
except Exception as e:
    print(f'⚠️  BART download failed: {e}')

try:
    from TTS.api import TTS
    print('📥 Downloading TTS model...')
    tts = TTS('tts_models/en/ljspeech/tacotron2-DDC', progress_bar=False)
    print('✅ TTS model downloaded')
except Exception as e:
    print(f'⚠️  TTS download failed: {e}')
"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp docker/.env.example .env
    echo "Please edit .env file to add your API keys"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p assets/samples assets/outputs models

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the application: streamlit run app/main.py"
echo ""
echo "Or use Docker:"
echo "docker-compose -f docker/docker-compose.yml up --build"
echo ""
echo "📖 Check README.md for detailed usage instructions"