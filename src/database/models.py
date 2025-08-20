# src/database/models.py
"""
SQLAlchemy models for the Aristotelian AI system
Supports multilingual content and evaluation tracking
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, 
    ForeignKey, JSON, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, ARRAY
import uuid

Base = declarative_base()

class ProcessedText(Base):
    """Stores processed philosophical texts from PDFs"""
    
    __tablename__ = 'processed_texts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(String(50), unique=True, nullable=False)
    title = Column(String(500), nullable=False)
    author = Column(String(200), nullable=False)
    language = Column(String(5), nullable=False)
    total_pages = Column(Integer, nullable=False)
    total_words = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    philosophical_concepts = Column(ARRAY(String), nullable=True)
    difficulty_score = Column(Float, nullable=False)
    segment_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("ProcessedText", back_populates="segments")
    
    # Indexes
    __table_args__ = (
        Index('idx_segments_document', 'document_id'),
        Index('idx_segments_concepts', 'philosophical_concepts', postgresql_using='gin'),
        Index('idx_segments_difficulty', 'difficulty_score'),
        Index('idx_segments_page', 'page_number'),
        UniqueConstraint('document_id', 'segment_id', name='uq_document_segment'),
    )

class User(Base):
    """User accounts for the philosophical AI system"""
    
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), unique=True, nullable=False)  # External user ID
    email = Column(String(255), unique=True, nullable=True)
    preferred_language = Column(String(5), nullable=False, default='en')
    cultural_context = Column(JSON, nullable=True)
    learning_preferences = Column(JSON, nullable=True)
    subscription_tier = Column(String(20), default='free')  # free, premium, academic
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    feedback = relationship("UserFeedback", back_populates="user", cascade="all, delete-orphan")
    evaluations = relationship("UserEvaluation", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_language', 'preferred_language'),
        Index('idx_users_active', 'is_active'),
        Index('idx_users_last_activity', 'last_activity'),
    )

class Conversation(Base):
    """Conversation sessions between users and the AI"""
    
    __tablename__ = 'conversations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    source_document_id = Column(UUID(as_uuid=True), ForeignKey('processed_texts.id'), nullable=True)
    language = Column(String(5), nullable=False)
    cultural_context = Column(JSON, nullable=True)
    total_messages = Column(Integer, default=0)
    avg_response_time = Column(Float, nullable=True)
    avg_philosophical_accuracy = Column(Float, nullable=True)
    avg_cultural_appropriateness = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    last_message_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    source_document = relationship("ProcessedText", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_conversations_user', 'user_id'),
        Index('idx_conversations_language', 'language'),
        Index('idx_conversations_active', 'is_active'),
        Index('idx_conversations_started', 'started_at'),
    )

class Message(Base):
    """Individual messages within conversations"""
    
    __tablename__ = 'messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id'), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    language = Column(String(5), nullable=False)
    philosophical_concepts = Column(ARRAY(String), nullable=True)
    cultural_notes = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    evaluation_scores = Column(JSON, nullable=True)
    message_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_messages_conversation', 'conversation_id'),
        Index('idx_messages_role', 'role'),
        Index('idx_messages_concepts', 'philosophical_concepts', postgresql_using='gin'),
        Index('idx_messages_created', 'created_at'),
    )

class PhilosophicalConcept(Base):
    """Master list of philosophical concepts across languages"""
    
    __tablename__ = 'philosophical_concepts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    concept_key = Column(String(100), unique=True, nullable=False)  # virtue_ethics, eudaimonia
    english_term = Column(String(200), nullable=False)
    translations = Column(JSON, nullable=False)  # {lang: [terms]}
    definition = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)  # ethics, metaphysics, logic
    difficulty_level = Column(String(20), nullable=False)  # beginner, intermediate, advanced
    related_concepts = Column(ARRAY(String), nullable=True)
    cultural_variations = Column(JSON, nullable=True)
    aristotelian_source = Column(String(200), nullable=True)  # Nicomachean Ethics, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_concepts_category', 'category'),
        Index('idx_concepts_difficulty', 'difficulty_level'),
        Index('idx_concepts_related', 'related_concepts', postgresql_using='gin'),
    )

class CulturalAdaptation(Base):
    """Cultural adaptation rules and examples"""
    
    __tablename__ = 'cultural_adaptations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cultural_context = Column(String(100), nullable=False)  # latin_america, mediterranean
    language = Column(String(5), nullable=False)
    concept_key = Column(String(100), nullable=False)
    adaptation_type = Column(String(50), nullable=False)  # metaphor, example, explanation
    original_content = Column(Text, nullable=False)
    adapted_content = Column(Text, nullable=False)
    cultural_values = Column(ARRAY(String), nullable=True)
    effectiveness_score = Column(Float, nullable=True)
    expert_validated = Column(Boolean, default=False)
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_adaptations_context', 'cultural_context'),
        Index('idx_adaptations_language', 'language'),
        Index('idx_adaptations_concept', 'concept_key'),
        Index('idx_adaptations_validated', 'expert_validated'),
    )

class UserFeedback(Base):
    """User feedback on AI responses and system performance"""
    
    __tablename__ = 'user_feedback'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    conversation_id = Column(String(100), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id'), nullable=True)
    satisfaction_score = Column(Integer, nullable=False)  # 1-5
    feedback_categories = Column(ARRAY(String), nullable=True)
    feedback_text = Column(Text, nullable=True)
    philosophical_accuracy_rating = Column(Integer, nullable=True)  # 1-5
    cultural_appropriateness_rating = Column(Integer, nullable=True)  # 1-5
    educational_value_rating = Column(Integer, nullable=True)  # 1-5
    suggested_improvements = Column(Text, nullable=True)
    language = Column(String(5), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="feedback")
    message = relationship("Message")
    
    # Indexes
    __table_args__ = (
        Index('idx_feedback_user', 'user_id'),
        Index('idx_feedback_satisfaction', 'satisfaction_score'),
        Index('idx_feedback_categories', 'feedback_categories', postgresql_using='gin'),
        Index('idx_feedback_created', 'created_at'),
    )

class EvaluationResult(Base):
    """Results from comprehensive evaluations"""
    
    __tablename__ = 'evaluation_results'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    evaluation_id = Column(String(100), unique=True, nullable=False)
    evaluation_type = Column(String(50), nullable=False)  # comprehensive, philosophical_accuracy, etc.
    model_version = Column(String(100), nullable=False)
    test_language = Column(String(5), nullable=False)
    cultural_context = Column(String(100), nullable=True)
    overall_score = Column(Float, nullable=False)
    philosophical_accuracy = Column(Float, nullable=True)
    cultural_appropriateness = Column(Float, nullable=True)
    educational_effectiveness = Column(Float, nullable=True)
    cross_lingual_consistency = Column(Float, nullable=True)
    bleu_score = Column(Float, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    test_samples_count = Column(Integer, nullable=False)
    expert_validation_score = Column(Float, nullable=True)
    recommendations = Column(JSON, nullable=True)
    evaluation_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_evaluations_type', 'evaluation_type'),
        Index('idx_evaluations_language', 'test_language'),
        Index('idx_evaluations_model', 'model_version'),
        Index('idx_evaluations_score', 'overall_score'),
        Index('idx_evaluations_created', 'created_at'),
    )

class UserEvaluation(Base):
    """Real-time evaluation results for individual user interactions"""
    
    __tablename__ = 'user_evaluations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    conversation_id = Column(String(100), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id'), nullable=False)
    philosophical_accuracy = Column(Float, nullable=False)
    cultural_appropriateness = Column(Float, nullable=False)
    educational_effectiveness = Column(Float, nullable=False)
    response_relevance = Column(Float, nullable=False)
    concept_coverage = Column(Float, nullable=False)
    language_quality = Column(Float, nullable=False)
    overall_quality = Column(Float, nullable=False)
    evaluation_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="evaluations")
    message = relationship("Message")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_eval_user', 'user_id'),
        Index('idx_user_eval_conversation', 'conversation_id'),
        Index('idx_user_eval_overall', 'overall_quality'),
        Index('idx_user_eval_created', 'created_at'),
    )

class ExpertValidator(Base):
    """Expert validators for philosophical content"""
    
    __tablename__ = 'expert_validators'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    institution = Column(String(300), nullable=True)
    specialization = Column(ARRAY(String), nullable=False)  # virtue_ethics, ancient_philosophy
    languages = Column(ARRAY(String), nullable=False)
    cultural_expertise = Column(ARRAY(String), nullable=True)
    credentials = Column(JSON, nullable=True)
    validation_count = Column(Integer, default=0)
    avg_validation_score = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_validation = Column(DateTime, nullable=True)
    
    # Relationships
    validations = relationship("ExpertValidation", back_populates="expert", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_experts_specialization', 'specialization', postgresql_using='gin'),
        Index('idx_experts_languages', 'languages', postgresql_using='gin'),
        Index('idx_experts_active', 'is_active'),
    )

class ExpertValidation(Base):
    """Individual expert validations of AI responses"""
    
    __tablename__ = 'expert_validations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    validation_id = Column(String(100), unique=True, nullable=False)
    expert_id = Column(UUID(as_uuid=True), ForeignKey('expert_validators.id'), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id'), nullable=True)
    evaluation_id = Column(UUID(as_uuid=True), ForeignKey('evaluation_results.id'), nullable=True)
    content_to_validate = Column(Text, nullable=False)
    philosophical_concept = Column(String(100), nullable=False)
    language = Column(String(5), nullable=False)
    cultural_context = Column(String(100), nullable=True)
    validation_score = Column(Float, nullable=False)  # 0-1
    accuracy_rating = Column(Integer, nullable=False)  # 1-5
    cultural_sensitivity_rating = Column(Integer, nullable=False)  # 1-5
    educational_value_rating = Column(Integer, nullable=False)  # 1-5
    detailed_feedback = Column(Text, nullable=True)
    suggested_improvements = Column(Text, nullable=True)
    validation_confidence = Column(Float, nullable=False)  # Expert's confidence in their validation
    time_spent_minutes = Column(Integer, nullable=True)
    validated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    expert = relationship("ExpertValidator", back_populates="validations")
    message = relationship("Message")
    evaluation = relationship("EvaluationResult")
    
    # Indexes
    __table_args__ = (
        Index('idx_validations_expert', 'expert_id'),
        Index('idx_validations_concept', 'philosophical_concept'),
        Index('idx_validations_language', 'language'),
        Index('idx_validations_score', 'validation_score'),
        Index('idx_validations_validated', 'validated_at'),
    )

class SystemMetrics(Base):
    """System-wide performance metrics"""
    
    __tablename__ = 'system_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_date = Column(DateTime, nullable=False)
    total_users_active = Column(Integer, nullable=False)
    total_conversations = Column(Integer, nullable=False)
    total_messages = Column(Integer, nullable=False)
    avg_philosophical_accuracy = Column(Float, nullable=False)
    avg_cultural_appropriateness = Column(Float, nullable=False)
    avg_educational_effectiveness = Column(Float, nullable=False)
    avg_response_time = Column(Float, nullable=False)
    language_distribution = Column(JSON, nullable=False)
    concept_frequency = Column(JSON, nullable=False)
    user_satisfaction_avg = Column(Float, nullable=False)
    expert_validation_avg = Column(Float, nullable=True)
    bleu_scores_by_language = Column(JSON, nullable=True)
    engagement_improvement = Column(Float, nullable=True)
    api_costs_total = Column(Float, nullable=True)
    api_calls_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_date', 'metric_date'),
        Index('idx_metrics_accuracy', 'avg_philosophical_accuracy'),
        Index('idx_metrics_satisfaction', 'user_satisfaction_avg'),
    )

class TrainingDataset(Base):
    """Training datasets for model fine-tuning"""
    
    __tablename__ = 'training_datasets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    version = Column(String(20), nullable=False)
    total_samples = Column(Integer, nullable=False)
    language_distribution = Column(JSON, nullable=False)
    concept_distribution = Column(JSON, nullable=False)
    difficulty_distribution = Column(JSON, nullable=False)
    cultural_distribution = Column(JSON, nullable=False)
    source_documents = Column(ARRAY(String), nullable=False)
    training_statistics = Column(JSON, nullable=True)
    file_path = Column(String(1000), nullable=False)
    validation_file_path = Column(String(1000), nullable=True)
    cohere_dataset_id = Column(String(100), nullable=True)
    status = Column(String(50), default='created')  # created, uploaded, training, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    fine_tuning_jobs = relationship("FineTuningJob", back_populates="dataset", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_datasets_status', 'status'),
        Index('idx_datasets_version', 'version'),
        Index('idx_datasets_created', 'created_at'),
    )

class FineTuningJob(Base):
    """Fine-tuning job tracking"""
    
    __tablename__ = 'fine_tuning_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(100), unique=True, nullable=False)
    cohere_finetune_id = Column(String(100), unique=True, nullable=False)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey('training_datasets.id'), nullable=False)
    model_name = Column(String(200), nullable=False)
    base_model = Column(String(100), nullable=False, default='command-r-plus')
    hyperparameters = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False)  # queued, training, completed, failed
    progress_percentage = Column(Float, default=0.0)
    training_metrics = Column(JSON, nullable=True)
    validation_metrics = Column(JSON, nullable=True)
    final_model_id = Column(String(100), nullable=True)
    training_cost = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    dataset = relationship("TrainingDataset", back_populates="fine_tuning_jobs")
    
    # Indexes
    __table_args__ = (
        Index('idx_finetuning_status', 'status'),
        Index('idx_finetuning_model', 'model_name'),
        Index('idx_finetuning_started', 'started_at'),
        Index('idx_finetuning_completed', 'completed_at'),
    )

# src/database/connection.py
"""
Database connection and session management
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from src.utils.config import get_settings, get_database_config
from src.database.models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self):
        self.settings = get_settings()
        self.db_config = get_database_config()
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
    
    def initialize(self):
        """Initialize database connections"""
        try:
            # Synchronous engine for migrations
            self.engine = create_engine(
                self.db_config['url'],
                poolclass=QueuePool,
                pool_size=self.db_config['pool_size'],
                max_overflow=self.db_config['max_overflow'],
                pool_timeout=self.db_config['pool_timeout'],
                pool_recycle=self.db_config['pool_recycle'],
                echo=self.settings.ENVIRONMENT == 'development'
            )
            
            # Async engine for API operations
            async_url = self.db_config['url'].replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(
                async_url,
                poolclass=QueuePool,
                pool_size=self.db_config['pool_size'],
                max_overflow=self.db_config['max_overflow'],
                pool_timeout=self.db_config['pool_timeout'],
                pool_recycle=self.db_config['pool_recycle'],
                echo=self.settings.ENVIRONMENT == 'development'
            )
            
            # Session makers
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            self.AsyncSessionLocal = sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                autocommit=False,
                autoflush=False
            )
            
            logger.info("Database connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def create_tables(self):
        """Create all tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    @asynccontextmanager
    async def get_async_session(self):
        """Get async database session"""
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    def get_session(self):
        """Get synchronous database session"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

# Global database manager instance
db_manager = DatabaseManager()

def get_database_pool():
    """Dependency to get database session"""
    return db_manager.get_session()

def get_async_database_pool():
    """Dependency to get async database session"""
    return db_manager.get_async_session()

def initialize_database():
    """Initialize database on startup"""
    db_manager.initialize()
    db_manager.create_tables()

# src/database/repositories.py
"""
Repository pattern for database operations
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

from src.database.models import (
    User, Conversation, Message, ProcessedText, TextSegment,
    PhilosophicalConcept, CulturalAdaptation, UserFeedback,
    EvaluationResult, ExpertValidation, SystemMetrics
)

logger = logging.getLogger(__name__)

class UserRepository:
    """Repository for user operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        preferred_language: str = 'en',
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user"""
        user = User(
            user_id=user_id,
            email=email,
            preferred_language=preferred_language,
            cultural_context=cultural_context or {}
        )
        
        self.session.add(user)
        await self.session.flush()
        return user
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by external user ID"""
        result = await self.session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def update_user_activity(self, user_id: str):
        """Update user's last activity timestamp"""
        await self.session.execute(
            update(User)
            .where(User.user_id == user_id)
            .values(last_activity=datetime.utcnow())
        )
    
    async def get_active_users(self, hours: int = 24) -> List[User]:
        """Get users active within specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(User).where(
                and_(User.is_active == True, User.last_activity >= cutoff)
            )
        )
        return result.scalars().all()

class ConversationRepository:
    """Repository for conversation operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_conversation(
        self,
        conversation_id: str,
        user_id: UUID,
        language: str,
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            language=language,
            cultural_context=cultural_context or {}
        )
        
        self.session.add(conversation)
        await self.session.flush()
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation with messages"""
        result = await self.session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.conversation_id == conversation_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_conversations(
        self,
        user_id: UUID,
        limit: int = 10,
        offset: int = 0
    ) -> List[Conversation]:
        """Get user's conversations with pagination"""
        result = await self.session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(desc(Conversation.last_message_at))
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()
    
    async def update_conversation_metrics(
        self,
        conversation_id: str,
        avg_response_time: Optional[float] = None,
        avg_philosophical_accuracy: Optional[float] = None,
        avg_cultural_appropriateness: Optional[float] = None,
        engagement_score: Optional[float] = None
    ):
        """Update conversation metrics"""
        updates = {'last_message_at': datetime.utcnow()}
        
        if avg_response_time is not None:
            updates['avg_response_time'] = avg_response_time
        if avg_philosophical_accuracy is not None:
            updates['avg_philosophical_accuracy'] = avg_philosophical_accuracy
        if avg_cultural_appropriateness is not None:
            updates['avg_cultural_appropriateness'] = avg_cultural_appropriateness
        if engagement_score is not None:
            updates['engagement_score'] = engagement_score
        
        await self.session.execute(
            update(Conversation)
            .where(Conversation.conversation_id == conversation_id)
            .values(**updates)
        )

class MessageRepository:
    """Repository for message operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        language: str,
        philosophical_concepts: Optional[List[str]] = None,
        cultural_notes: Optional[str] = None,
        confidence_score: Optional[float] = None,
        processing_time: Optional[float] = None,
        evaluation_scores: Optional[Dict[str, float]] = None
    ) -> Message:
        """Create a new message"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            language=language,
            philosophical_concepts=philosophical_concepts or [],
            cultural_notes=cultural_notes,
            confidence_score=confidence_score,
            processing_time=processing_time,
            evaluation_scores=evaluation_scores or {}
        )
        
        self.session.add(message)
        await self.session.flush()
        return message
    
    async def get_conversation_messages(
        self,
        conversation_id: UUID,
        limit: int = 50
    ) -> List[Message]:
        """Get messages for a conversation"""
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(asc(Message.created_at))
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_messages_by_concept(
        self,
        philosophical_concept: str,
        language: Optional[str] = None,
        limit: int = 100
    ) -> List[Message]:
        """Get messages containing specific philosophical concept"""
        query = select(Message).where(
            Message.philosophical_concepts.contains([philosophical_concept])
        )
        
        if language:
            query = query.where(Message.language == language)
        
        query = query.order_by(desc(Message.created_at)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()

class EvaluationRepository:
    """Repository for evaluation operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_evaluation_result(
        self,
        evaluation_id: str,
        evaluation_type: str,
        model_version: str,
        test_language: str,
        overall_score: float,
        detailed_results: Dict[str, Any],
        **kwargs
    ) -> EvaluationResult:
        """Create evaluation result"""
        evaluation = EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=evaluation_type,
            model_version=model_version,
            test_language=test_language,
            overall_score=overall_score,
            detailed_results=detailed_results,
            **kwargs
        )
        
        self.session.add(evaluation)
        await self.session.flush()
        return evaluation
    
    async def get_latest_evaluations(
        self,
        evaluation_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10
    ) -> List[EvaluationResult]:
        """Get latest evaluation results"""
        query = select(EvaluationResult)
        
        if evaluation_type:
            query = query.where(EvaluationResult.evaluation_type == evaluation_type)
        if language:
            query = query.where(EvaluationResult.test_language == language)
        
        query = query.order_by(desc(EvaluationResult.created_at)).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_evaluation_trends(
        self,
        days: int = 30,
        evaluation_type: str = 'comprehensive'
    ) -> List[Dict[str, Any]]:
        """Get evaluation score trends over time"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(
                func.date(EvaluationResult.created_at).label('date'),
                func.avg(EvaluationResult.overall_score).label('avg_score'),
                func.avg(EvaluationResult.philosophical_accuracy).label('avg_accuracy'),
                func.avg(EvaluationResult.cultural_appropriateness).label('avg_cultural'),
                func.count(EvaluationResult.id).label('evaluation_count')
            )
            .where(
                and_(
                    EvaluationResult.evaluation_type == evaluation_type,
                    EvaluationResult.created_at >= cutoff
                )
            )
            .group_by(func.date(EvaluationResult.created_at))
            .order_by(func.date(EvaluationResult.created_at))
        )
        
        return [
            {
                'date': row.date,
                'avg_score': float(row.avg_score) if row.avg_score else 0,
                'avg_accuracy': float(row.avg_accuracy) if row.avg_accuracy else 0,
                'avg_cultural': float(row.avg_cultural) if row.avg_cultural else 0,
                'evaluation_count': row.evaluation_count
            }
            for row in result
        ]

class MetricsRepository:
    """Repository for system metrics operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_daily_metrics(
        self,
        metric_date: datetime,
        metrics_data: Dict[str, Any]
    ) -> SystemMetrics:
        """Create daily system metrics"""
        metrics = SystemMetrics(
            metric_date=metric_date,
            **metrics_data
        )
        
        self.session.add(metrics)
        await self.session.flush()
        return metrics
    
    async def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics"""
        result = await self.session.execute(
            select(SystemMetrics)
            .order_by(desc(SystemMetrics.metric_date))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_metrics_trend(
        self,
        days: int = 30
    ) -> List[SystemMetrics]:
        """Get metrics trend over specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        result = await self.session.execute(
            select(SystemMetrics)
            .where(SystemMetrics.metric_date >= cutoff)
            .order_by(asc(SystemMetrics.metric_date))
        )
        return result.scalars().all()
    
    async def calculate_current_metrics(self) -> Dict[str, Any]:
        """Calculate current system metrics from live data"""
        today = datetime.utcnow().date()
        
        # Active users (last 24 hours)
        active_users = await self.session.execute(
            select(func.count(User.id))
            .where(User.last_activity >= datetime.utcnow() - timedelta(hours=24))
        )
        
        # Total conversations today
        conversations_today = await self.session.execute(
            select(func.count(Conversation.id))
            .where(func.date(Conversation.started_at) == today)
        )
        
        # Messages today
        messages_today = await self.session.execute(
            select(func.count(Message.id))
            .where(func.date(Message.created_at) == today)
        )
        
        # Average scores today
        avg_scores = await self.session.execute(
            select(
                func.avg(Message.evaluation_scores['philosophical_accuracy'].astext.cast(func.float)).label('avg_accuracy'),
                func.avg(Message.evaluation_scores['cultural_appropriateness'].astext.cast(func.float)).label('avg_cultural'),
                func.avg(Message.evaluation_scores['educational_effectiveness'].astext.cast(func.float)).label('avg_educational'),
                func.avg(Message.processing_time).label('avg_response_time')
            )
            .where(
                and_(
                    func.date(Message.created_at) == today,
                    Message.role == 'assistant'
                )
            )
        )
        
        # Language distribution today
        lang_dist = await self.session.execute(
            select(
                Message.language,
                func.count(Message.id).label('count')
            )
            .where(func.date(Message.created_at) == today)
            .group_by(Message.language)
        )
        
        # User satisfaction today
        user_satisfaction = await self.session.execute(
            select(func.avg(UserFeedback.satisfaction_score))
            .where(func.date(UserFeedback.created_at) == today)
        )
        
        scores_row = avg_scores.first()
        lang_distribution = {row.language: row.count for row in lang_dist}
        
        return {
            'total_users_active': active_users.scalar() or 0,
            'total_conversations': conversations_today.scalar() or 0,
            'total_messages': messages_today.scalar() or 0,
            'avg_philosophical_accuracy': float(scores_row.avg_accuracy) if scores_row.avg_accuracy else 0.0,
            'avg_cultural_appropriateness': float(scores_row.avg_cultural) if scores_row.avg_cultural else 0.0,
            'avg_educational_effectiveness': float(scores_row.avg_educational) if scores_row.avg_educational else 0.0,
            'avg_response_time': float(scores_row.avg_response_time) if scores_row.avg_response_time else 0.0,
            'language_distribution': lang_distribution,
            'concept_frequency': {},  # Would need additional calculation
            'user_satisfaction_avg': float(user_satisfaction.scalar()) if user_satisfaction.scalar() else 0.0
        }

# src/database/migrations.py
"""
Database migration utilities
"""

import logging
from pathlib import Path
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.environment import EnvironmentContext

from src.utils.config import get_settings

logger = logging.getLogger(__name__)

class MigrationManager:
    """Manages database migrations using Alembic"""
    
    def __init__(self):
        self.settings = get_settings()
        self.alembic_cfg = Config()
        
        # Set alembic configuration
        self.alembic_cfg.set_main_option('script_location', 'alembic')
        self.alembic_cfg.set_main_option('sqlalchemy.url', self.settings.DATABASE_URL)
    
    def create_migration(self, message: str):
        """Create a new migration"""
        try:
            command.revision(self.alembic_cfg, autogenerate=True, message=message)
            logger.info(f"Created migration: {message}")
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def run_migrations(self):
        """Run pending migrations"""
        try:
            command.upgrade(self.alembic_cfg, 'head')
            logger.info("Migrations completed successfully")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def rollback_migration(self, revision: str = '-1'):
        """Rollback to specific revision"""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Rolled back to revision: {revision}")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    def get_current_revision(self) -> str:
        """Get current database revision"""
        try:
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            def get_revision(rev, context):
                return rev
            
            with EnvironmentContext(
                self.alembic_cfg,
                script,
                fn=get_revision,
                as_sql=False,
                starting_rev=None,
                destination_rev='head'
            ):
                pass
                
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return "unknown"

# Global migration manager
migration_manager = MigrationManager()

def run_migrations():
    """Run database migrations on startup"""
    migration_manager.run_migrations()

def create_migration(message: str):
    """Create a new migration"""
    migration_manager.create_migration(message)True)
    difficulty_distribution = Column(JSON, nullable=True)
    processing_metadata = Column(JSON, nullable=True)
    source_file_path = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    segments = relationship("TextSegment", back_populates="document", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="source_document")
    
    # Indexes
    __table_args__ = (
        Index('idx_processed_texts_language', 'language'),
        Index('idx_processed_texts_concepts', 'philosophical_concepts', postgresql_using='gin'),
        Index('idx_processed_texts_created', 'created_at'),
    )

class TextSegment(Base):
    """Individual segments of processed texts"""
    
    __tablename__ = 'text_segments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    segment_id = Column(String(100), nullable=False)  # page_X_para_Y format
    document_id = Column(UUID(as_uuid=True), ForeignKey('processed_texts.id'), nullable=False)
    content = Column(Text, nullable=False)
    language = Column(String(5), nullable=False)
    section_type = Column(String(50), nullable=False)  # chapter, paragraph, quote, etc.
    page_number = Column(Integer, nullable=False)
    word_count = Column(Integer, nullable=False)
    sentence_count = Column(Integer, nullable=False)
    philosophical_concepts = Column(ARRAY(String), nullable=