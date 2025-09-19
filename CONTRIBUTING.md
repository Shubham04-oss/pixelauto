# Contributing to AI Short-Video Creator

Thank you for your interest in contributing to the AI Short-Video Creator! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ai-short-video-creator.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## 🛠️ Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-short-video-creator.git
cd ai-short-video-creator

# Run setup script
chmod +x docker/setup.sh
./docker/setup.sh

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## 📋 Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting: `black app/ tests/`
- Use flake8 for linting: `flake8 app/ tests/`
- Add type hints where appropriate
- Write docstrings for all functions and classes

## 🧪 Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Run tests before submitting: `pytest tests/`
- Include integration tests for AI components

## 📝 Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Follow conventional commit messages
6. Request review from maintainers

## 🐛 Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error logs if applicable

## 💡 Feature Requests

Include:
- Clear description of the feature
- Use case and benefits
- Proposed implementation approach
- Any relevant examples or mockups

## 🎯 Areas for Contribution

- New video templates and styles
- Additional TTS voices and languages
- Performance optimizations
- Documentation improvements
- Bug fixes and stability improvements
- Test coverage expansion

## 📖 Documentation

- Update README.md for user-facing changes
- Add docstrings to all new functions
- Include code examples in documentation
- Update API documentation if applicable

## 🌐 Community Guidelines

- Be respectful and inclusive
- Help others learn and contribute
- Follow the code of conduct
- Participate in discussions constructively

Thank you for contributing! 🎉