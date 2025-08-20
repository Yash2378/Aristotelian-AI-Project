# scripts/setup/initialize_database.py
#!/usr/bin/env python3
"""
Initialize database for the Multilingual Aristotelian AI system
"""

import asyncio
import logging
import os
from pathlib import Path

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine

from src.utils.config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQL schema
SCHEMA_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSONB DEFAULT '{}'::jsonb
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    language VARCHAR(10) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    message_type VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    language VARCHAR(10),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- Evaluation results table
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    evaluation_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL,
    language VARCHAR(10),
    category VARCHAR(50),
    details JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System performance table
CREATE TABLE IF NOT EXISTS system_performance (
    id SERIAL PRIMARY KEY,
    evaluation_date DATE NOT NULL,
    overall_score FLOAT,
    philosophical_accuracy FLOAT,
    cultural_appropriateness FLOAT,
    educational_effectiveness FLOAT,
    cross_lingual_consistency FLOAT,
    bleu_score_avg FLOAT,
    user_count INTEGER,
    interaction_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255),
    satisfaction_score INTEGER CHECK (satisfaction_score BETWEEN 1 AND 5),
    feedback_text TEXT,
    categories TEXT[],
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE SET NULL
);

-- User interactions table
CREATE TABLE IF NOT EXISTS user_interactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    language VARCHAR(10) NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    processing_time FLOAT,
    confidence FLOAT,
    philosophical_concepts TEXT[],
    cultural_notes TEXT,
    evaluation_scores JSONB DEFAULT '{}'::jsonb,
    session_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Philosophical concepts table
CREATE TABLE IF NOT EXISTS philosophical_concepts (
    id SERIAL PRIMARY KEY,
    concept_name VARCHAR(255) NOT NULL,
    language VARCHAR(10) NOT NULL,
    definition TEXT NOT NULL,
    related_concepts TEXT[],
    cultural_notes JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concept_name, language)
);

-- Expert validations table
CREATE TABLE IF NOT EXISTS expert_validations (
    id SERIAL PRIMARY KEY,
    response_id VARCHAR(255) NOT NULL,
    expert_id VARCHAR(255) NOT NULL,
    validation_score FLOAT NOT NULL,
    comments TEXT,
    categories TEXT[],
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_evaluation_id ON evaluation_results(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_timestamp ON evaluation_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_interactions_language ON user_interactions(language);
CREATE INDEX IF NOT EXISTS idx_system_performance_date ON system_performance(evaluation_date);
CREATE INDEX IF NOT EXISTS idx_philosophical_concepts_language ON philosophical_concepts(language);

-- Insert initial philosophical concepts
INSERT INTO philosophical_concepts (concept_name, language, definition, related_concepts, cultural_notes) VALUES
('virtue', 'en', 'Excellence of character; a disposition to act in ways that promote human flourishing', ARRAY['eudaimonia', 'phronesis', 'temperance'], '{"communication_style": "direct", "examples": ["courage in facing challenges", "honesty in business dealings"]}'),
('virtud', 'es', 'Excelencia del carácter; una disposición para actuar de maneras que promuevan el florecimiento humano', ARRAY['eudaimonia', 'phronesis', 'templanza'], '{"communication_style": "expressive", "examples": ["valor en enfrentar desafíos", "honestidad en los negocios", "respeto hacia los mayores"]}'),
('vertu', 'fr', 'Excellence du caractère; une disposition à agir de manière à promouvoir l''épanouissement humain', ARRAY['eudaimonia', 'phronesis', 'tempérance'], '{"communication_style": "intellectual", "examples": ["courage face aux défis", "honnêteté dans les affaires", "raffinement culturel"]}'),
('tugend', 'de', 'Charakterexzellenz; eine Disposition, auf eine Weise zu handeln, die menschliches Gedeihen fördert', ARRAY['eudaimonia', 'phronesis', 'mäßigung'], '{"communication_style": "systematic", "examples": ["Mut bei Herausforderungen", "Ehrlichkeit im Geschäft", "systematisches Denken"]}'),
('virtù', 'it', 'Eccellenza del carattere; una disposizione ad agire in modi che promuovano il fiorire umano', ARRAY['eudaimonia', 'phronesis', 'temperanza'], '{"communication_style": "passionate", "examples": ["coraggio nelle sfide", "onestà negli affari", "bellezza nell''arte"]}')
ON CONFLICT (concept_name, language) DO NOTHING;

-- Insert sample cultural contexts
INSERT INTO philosophical_concepts (concept_name, language, definition, related_concepts, cultural_notes) VALUES
('eudaimonia', 'en', 'The highest human good; often translated as happiness or flourishing', ARRAY['virtue', 'contemplation', 'friendship'], '{"regional_adaptations": {"north_america": "individual self-actualization", "europe": "social contribution"}}'),
('eudaimonia', 'es', 'El bien humano más alto; a menudo traducido como felicidad o florecimiento', ARRAY['virtud', 'contemplación', 'amistad'], '{"regional_adaptations": {"latin_america": "bienestar familiar y comunitario", "spain": "equilibrio entre tradición y progreso"}}'),
('eudaimonia', 'fr', 'Le bien humain le plus élevé; souvent traduit par bonheur ou épanouissement', ARRAY['vertu', 'contemplation', 'amitié'], '{"regional_adaptations": {"france": "excellence intellectuelle et culturelle", "canada": "harmonie entre cultures"}}'),
('eudaimonia', 'de', 'Das höchste menschliche Gut; oft als Glück oder Gedeihen übersetzt', ARRAY['tugend', 'kontemplation', 'freundschaft'], '{"regional_adaptations": {"germany": "systematische Selbstverwirklichung", "austria": "kulturelle Besinnung"}}'),
('eudaimonia', 'it', 'Il bene umano più alto; spesso tradotto come felicità o fioritura', ARRAY['virtù', 'contemplazione', 'amicizia'], '{"regional_adaptations": {"italy": "bellezza nella vita quotidiana", "artistic_expression": "realizzazione attraverso l''arte"}}')
ON CONFLICT (concept_name, language) DO NOTHING;
"""

async def create_database_if_not_exists():
    """Create database if it doesn't exist"""
    settings = get_settings()
    
    # Parse database URL to get connection info
    import urllib.parse
    parsed = urllib.parse.urlparse(settings.DATABASE_URL)
    
    db_name = parsed.path[1:]  # Remove leading slash
    host = parsed.hostname
    port = parsed.port or 5432
    username = parsed.username
    password = parsed.password
    
    # Connect to postgres database to create our database
    postgres_url = f"postgresql://{username}:{password}@{host}:{port}/postgres"
    
    try:
        conn = await asyncpg.connect(postgres_url)
        
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", db_name
        )
        
        if not exists:
            logger.info(f"Creating database: {db_name}")
            await conn.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database {db_name} created successfully")
        else:
            logger.info(f"Database {db_name} already exists")
        
        await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        raise

async def initialize_schema():
    """Initialize database schema"""
    settings = get_settings()
    
    try:
        logger.info("Connecting to database...")
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        logger.info("Creating tables and indexes...")
        await conn.execute(SCHEMA_SQL)
        
        logger.info("Database schema initialized successfully")
        await conn.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize schema: {e}")
        raise

async def verify_database():
    """Verify database setup"""
    settings = get_settings()
    
    try:
        conn = await asyncpg.connect(settings.DATABASE_URL)
        
        # Check tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        
        table_names = [row['table_name'] for row in tables]
        expected_tables = [
            'users', 'conversations', 'messages', 'evaluation_results',
            'system_performance', 'user_feedback', 'user_interactions',
            'philosophical_concepts', 'expert_validations'
        ]
        
        missing_tables = set(expected_tables) - set(table_names)
        if missing_tables:
            logger.error(f"Missing tables: {missing_tables}")
            return False
        
        # Check philosophical concepts
        concept_count = await conn.fetchval(
            "SELECT COUNT(*) FROM philosophical_concepts"
        )
        
        logger.info(f"✅ Database verification passed")
        logger.info(f"✅ Found {len(table_names)} tables")
        logger.info(f"✅ Found {concept_count} philosophical concepts")
        
        await conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False

async def main():
    """Main initialization function"""
    logger.info("🚀 Starting database initialization...")
    
    try:
        # Create database if it doesn't exist
        await create_database_if_not_exists()
        
        # Initialize schema
        await initialize_schema()
        
        # Verify setup
        if await verify_database():
            logger.info("🎉 Database initialization completed successfully!")
        else:
            logger.error("❌ Database verification failed")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)