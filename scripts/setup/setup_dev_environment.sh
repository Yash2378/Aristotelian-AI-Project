#!/bin/bash

# Setup development environment for Multilingual Aristotelian AI

set -e

echo "🚀 Setting up Multilingual Aristotelian AI development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements/development.txt

# Download spaCy models
echo "🌍 Downloading language models..."
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download it_core_news_sm

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/philosophical_texts/{en,es,fr,de,it}
mkdir -p data/processed
mkdir -p data/evaluation
mkdir -p data/reference_translations

# Copy example environment file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your actual API keys and configuration"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker detected. You can use 'make docker-compose-up' to start services"
else
    echo "⚠️ Docker not detected. Please install Docker for full development experience"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Start services: make docker-compose-up"
echo "3. Run the application: make run-dev"
echo "4. Run tests: make test"
echo ""
echo "For more commands, run: make help"