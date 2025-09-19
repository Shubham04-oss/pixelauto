# 🎬 AI Short-Video Creator

An open-source tool that automatically converts articles, transcripts, or any text content into engaging short-form videos (30-60 seconds) optimized for social media platforms like Instagram Reels, YouTube Shorts, and TikTok.

## ✨ Features

- **🤖 AI-Powered Summarization**: Automatically extracts key points from long-form content
- **🎙️ Text-to-Speech**: Generate natural-sounding narration with multiple voice options
- **📝 Smart Captions**: Auto-generated captions with perfect timing and formatting
- **🖼️ Visual Generation**: AI-generated images or stock photo integration for visual appeal
- **📱 Social Media Ready**: Export in 9:16 format optimized for mobile viewing
- **🎨 Customizable Styles**: Multiple video templates and styling options
- **🚀 Zero-Cost Stack**: Powered entirely by free and open-source technologies

## 🎯 Use Cases

- Convert blog articles into shareable video content
- Transform podcast transcripts into social media clips
- Create educational micro-content from research papers
- Generate marketing videos from product descriptions
- Turn meeting notes into summary videos

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Summarization** | HuggingFace Transformers (BART/T5) | Extract key points from text |
| **Text-to-Speech** | Coqui TTS | Generate natural narration |
| **Captions** | Forced alignment + FFmpeg | Sync text with audio |
| **Visuals** | Stable Diffusion / Pexels API | Generate/fetch relevant images |
| **Video Editing** | MoviePy + FFmpeg | Combine all elements |
| **UI** | Streamlit | User-friendly web interface |
| **Deployment** | Docker | Consistent environment |

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-short-video-creator.git
cd ai-short-video-creator

# Start with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Access the app at http://localhost:8501
```

### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/yourusername/ai-short-video-creator.git
cd ai-short-video-creator

# Run setup script
chmod +x docker/setup.sh
./docker/setup.sh

# Activate virtual environment
source venv/bin/activate

# Start the application
streamlit run app/main.py
```

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or later
- **FFmpeg**: Latest version
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for models

### Platform Support
- ✅ Linux (Ubuntu 20.04+, Debian 11+)
- ✅ macOS (10.15+)
- ✅ Windows 10/11 (with WSL2 recommended)

## 🔧 Installation Details

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg python3-dev build-essential
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
- Install [FFmpeg](https://ffmpeg.org/download.html)
- Install [Python 3.8+](https://python.org/downloads/)

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

### 3. Download AI Models

The application will automatically download required models on first run:
- **BART** (400MB) - Text summarization
- **TTS Models** (200MB) - Text-to-speech
- **NLTK Data** (50MB) - Text processing

## 📱 Usage

### Basic Workflow

1. **Input Text**: Upload a file or paste text content
2. **Configure Settings**: Choose video length, style, and voice
3. **Generate Summary**: AI extracts key points automatically
4. **Create Video**: Process combines narration, visuals, and captions
5. **Download**: Get your social-ready video in 9:16 format

### Supported Input Formats

- **Text Files**: `.txt`, `.md`, `.rtf`
- **Documents**: `.pdf`, `.docx`
- **Web Content**: URLs (article extraction)
- **Audio**: `.mp3`, `.wav` (with transcription)

### Output Specifications

- **Format**: MP4 (H.264)
- **Resolution**: 1080x1920 (9:16 aspect ratio)
- **Duration**: 30-60 seconds
- **Audio**: 44.1kHz, stereo
- **Captions**: Burned-in + separate SRT file

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Optional API Keys
PEXELS_API_KEY=your_pexels_key_here
HUGGINGFACE_HUB_TOKEN=your_hf_token_here

# Model Configuration
SUMMARIZER_MODEL=facebook/bart-large-cnn
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
MAX_SUMMARY_LENGTH=150
VIDEO_DURATION=45

# Performance Settings
USE_GPU=false
BATCH_SIZE=1
NUM_WORKERS=2
```

### Advanced Settings

Edit `app/config.py` for detailed customization:

```python
# Video settings
VIDEO_CONFIG = {
    'width': 1080,
    'height': 1920,
    'fps': 30,
    'bitrate': '2M'
}

# Text settings
TEXT_CONFIG = {
    'max_words': 100,
    'font_size': 48,
    'font_family': 'Arial Bold'
}
```

## 📊 Performance Optimization

### Speed Improvements

1. **Use GPU**: Enable CUDA for 3-5x faster processing
2. **Model Caching**: Models download once and cache locally
3. **Batch Processing**: Process multiple videos efficiently
4. **Optimized Models**: Use quantized models for faster inference

### Resource Usage

| Component | CPU Usage | Memory | Processing Time |
|-----------|-----------|--------|-----------------|
| Summarization | 60-80% | 2GB | 10-30s |
| TTS Generation | 40-60% | 1GB | 15-45s |
| Video Creation | 80-100% | 1GB | 20-60s |
| **Total** | - | **4GB** | **45-135s** |

## 🎨 Customization

### Video Styles

Choose from predefined styles or create custom ones:

- **Minimal**: Clean text on solid backgrounds
- **Dynamic**: Animated text with transitions
- **Corporate**: Professional styling with logos
- **Social**: Trendy fonts and vibrant colors

### Voice Options

Multiple TTS voices available:
- **Default**: Neutral, clear pronunciation
- **Energetic**: Upbeat, engaging tone
- **Professional**: Formal, authoritative
- **Casual**: Friendly, conversational

## 🔍 Troubleshooting

### Common Issues

**1. FFmpeg Not Found**
```bash
# Linux
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

**2. Model Download Fails**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/bart-large-cnn')"
```

**3. Out of Memory**
```bash
# Reduce batch size in config.py
BATCH_SIZE = 1

# Use CPU instead of GPU
USE_GPU = false
```

**4. Slow Processing**
- Enable GPU acceleration if available
- Use smaller models for faster processing
- Reduce video duration or resolution

### Debug Mode

Enable detailed logging:

```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
streamlit run app/main.py
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork and clone your fork
git clone https://github.com/yourusername/ai-short-video-creator.git
cd ai-short-video-creator

# Create feature branch
git checkout -b feature/amazing-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
black app/ tests/
flake8 app/ tests/
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8, use Black formatter
2. **Testing**: Add tests for new features
3. **Documentation**: Update README and docstrings
4. **Commits**: Use conventional commit messages
5. **Pull Requests**: Include description and test results

### Areas for Contribution

- 🎨 New video templates and styles
- 🗣️ Additional TTS voices and languages
- 📊 Performance optimizations
- 🧪 Test coverage improvements
- 📚 Documentation enhancements
- 🐛 Bug fixes and stability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- [HuggingFace](https://huggingface.co/) for transformer models
- [Coqui TTS](https://github.com/coqui-ai/TTS) for text-to-speech
- [MoviePy](https://zulko.github.io/moviepy/) for video processing
- [Streamlit](https://streamlit.io/) for the web interface
- [FFmpeg](https://ffmpeg.org/) for multimedia processing

## 📞 Support

- 📖 **Documentation**: Check this README and code comments
- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/ai-short-video-creator/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-short-video-creator/discussions)
- 📧 **Contact**: your.email@example.com

## 🚀 What's Next?

### Planned Features

- [ ] Multi-language support
- [ ] Custom voice cloning
- [ ] Advanced video effects
- [ ] Batch processing UI
- [ ] API endpoint for integration
- [ ] Mobile app companion

### Version Roadmap

- **v1.0**: Core functionality (current)
- **v1.1**: Performance optimizations
- **v1.2**: Advanced styling options
- **v2.0**: Multi-language and voice cloning

---

**Made with ❤️ by the open-source community**

*Transform your text into engaging videos in minutes, not hours!*