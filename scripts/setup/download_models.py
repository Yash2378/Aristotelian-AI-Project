#!/usr/bin/env python3
"""
Download required language models for the Multilingual Aristotelian AI system
"""

import subprocess
import sys
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Required spaCy models
SPACY_MODELS = [
    "en_core_web_sm",  # English
    "es_core_news_sm", # Spanish
    "fr_core_news_sm", # French
    "de_core_news_sm", # German
    "it_core_news_sm", # Italian
]

# Optional models (will try to download but won't fail if unavailable)
OPTIONAL_MODELS = [
    "xx_ent_wiki_sm",  # Multi-language
]

def run_command(command: List[str]) -> bool:
    """Run a command and return success status"""
    try:
        logger.info(f"Running: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {e.stderr.strip()}")
        return False

def download_spacy_models():
    """Download required spaCy models"""
    logger.info("Downloading spaCy models...")
    
    success_count = 0
    total_count = len(SPACY_MODELS)
    
    for model in SPACY_MODELS:
        logger.info(f"Downloading {model}...")
        if run_command([sys.executable, "-m", "spacy", "download", model]):
            success_count += 1
        else:
            logger.error(f"Failed to download {model}")
    
    # Try optional models
    for model in OPTIONAL_MODELS:
        logger.info(f"Trying to download optional model {model}...")
        run_command([sys.executable, "-m", "spacy", "download", model])
    
    logger.info(f"Successfully downloaded {success_count}/{total_count} required models")
    
    if success_count < total_count:
        logger.warning("Some models failed to download. The system may have reduced functionality.")
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_downloads = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'vader_lexicon'
        ]
        
        for item in nltk_downloads:
            try:
                logger.info(f"Downloading NLTK data: {item}")
                nltk.download(item, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK data {item}: {e}")
        
        logger.info("NLTK data download completed")
        return True
        
    except ImportError:
        logger.error("NLTK not installed. Please install requirements first.")
        return False

def verify_downloads():
    """Verify that models were downloaded correctly"""
    logger.info("Verifying model downloads...")
    
    # Test spaCy models
    try:
        import spacy
        
        for model in SPACY_MODELS:
            try:
                nlp = spacy.load(model)
                # Test with a simple sentence
                doc = nlp("This is a test sentence.")
                logger.info(f"✅ {model} loaded successfully")
            except OSError:
                logger.error(f"❌ {model} failed to load")
                
    except ImportError:
        logger.error("spaCy not installed. Please install requirements first.")
    
    # Test NLTK
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        
        # Test tokenization
        tokens = word_tokenize("This is a test sentence.")
        logger.info("✅ NLTK working correctly")
        
    except Exception as e:
        logger.error(f"❌ NLTK test failed: {e}")

def main():
    """Main function"""
    logger.info("Starting model download process...")
    
    success = True
    
    # Download spaCy models
    if not download_spacy_models():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    # Verify downloads
    verify_downloads()
    
    if success:
        logger.info("🎉 All models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("⚠️ Some downloads failed. Check logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()