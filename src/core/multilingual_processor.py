# src/core/multilingual_processor.py
"""
Core multilingual processing system for Aristotelian philosophical texts
Handles language detection, cultural adaptation, and cross-lingual consistency
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

import cohere
import spacy
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import get_settings
from src.utils.exceptions import ModelException, ValidationException
from src.database.models import PhilosophicalConcept, CulturalAdaptation

logger = logging.getLogger(__name__)

class MultilingualPhilosophicalProcessor:
    """
    Core processor for multilingual philosophical dialogue
    Integrates Cohere's Command R+ with specialized philosophical knowledge
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cohere_client = None
        self.language_models = {}
        self.concept_embeddings = {}
        self.cultural_adaptations = {}
        
        # Supported languages with their spaCy models
        self.supported_languages = {
            'en': 'en_core_web_lg',
            'es': 'es_core_news_lg', 
            'fr': 'fr_core_news_lg',
            'de': 'de_core_news_lg',
            'it': 'it_core_news_lg',
            'ar': 'ar_core_news_lg',
            'el': 'el_core_news_lg'
        }
        
        # Philosophical concept mappings across languages
        self.concept_mappings = {
            'virtue': {
                'en': ['virtue', 'excellence', 'moral goodness'],
                'es': ['virtud', 'excelencia moral', 'bondad'],
                'fr': ['vertu', 'excellence morale', 'bonté'],
                'de': ['Tugend', 'moralische Exzellenz', 'Güte'],
                'it': ['virtù', 'eccellenza morale', 'bontà'],
                'ar': ['فضيلة', 'التميز الأخلاقي', 'الخير'],
                'el': ['αρετή', 'ηθική αριστεία', 'αγαθότητα']
            },
            'eudaimonia': {
                'en': ['happiness', 'flourishing', 'well-being', 'eudaimonia'],
                'es': ['felicidad', 'florecimiento', 'bienestar', 'eudaimonía'],
                'fr': ['bonheur', 'épanouissement', 'bien-être', 'eudémonia'],
                'de': ['Glück', 'Gedeihen', 'Wohlbefinden', 'Eudämonie'],
                'it': ['felicità', 'fioritura', 'benessere', 'eudaimonia'],
                'ar': ['السعادة', 'الازدهار', 'الرفاهية', 'اليودايمونيا'],
                'el': ['ευδαιμονία', 'ευημερία', 'ευεξία']
            },
            'phronesis': {
                'en': ['practical wisdom', 'prudence', 'phronesis'],
                'es': ['sabiduría práctica', 'prudencia', 'frónesis'],
                'fr': ['sagesse pratique', 'prudence', 'phronèse'],
                'de': ['praktische Weisheit', 'Klugheit', 'Phronesis'],
                'it': ['saggezza pratica', 'prudenza', 'phronesis'],
                'ar': ['الحكمة العملية', 'التعقل', 'الفرونيسيس'],
                'el': ['φρόνησις', 'πρακτική σοφία']
            }
        }
        
        # Cultural context templates for adaptations
        self.cultural_contexts = {
            'latin_america': {
                'values': ['family', 'community', 'honor', 'tradition'],
                'examples': ['extended family decisions', 'community festivals', 'elders wisdom'],
                'metaphors': ['river flowing to sea', 'tree with deep roots']
            },
            'mediterranean': {
                'values': ['hospitality', 'passion', 'beauty', 'history'],
                'examples': ['dinner gatherings', 'artistic expression', 'ancient wisdom'],
                'metaphors': ['olive tree growth', 'wine aging']
            },
            'northern_europe': {
                'values': ['efficiency', 'equality', 'nature', 'innovation'],
                'examples': ['work-life balance', 'social systems', 'environmental care'],
                'metaphors': ['forest ecosystem', 'seasonal cycles']
            },
            'middle_east': {
                'values': ['hospitality', 'wisdom', 'justice', 'community'],
                'examples': ['guest welcoming', 'scholarly debate', 'fair trade'],
                'metaphors': ['desert oasis', 'ancient library']
            }
        }
    
    async def initialize(self):
        """Initialize all required models and resources"""
        try:
            # Initialize Cohere client
            self.cohere_client = cohere.AsyncClient(
                api_key=self.config.get('cohere_api_key')
            )
            
            # Load language models
            await self._load_language_models()
            
            # Load philosophical concept embeddings
            await self._load_concept_embeddings()
            
            # Load cultural adaptation rules
            await self._load_cultural_adaptations()
            
            logger.info("Multilingual processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize multilingual processor: {e}")
            raise ModelException(
                message=f"Initialization failed: {str(e)}",
                model_id="multilingual_processor"
            )
    
    async def _load_language_models(self):
        """Load spaCy models for each supported language"""
        for lang, model_name in self.supported_languages.items():
            try:
                self.language_models[lang] = spacy.load(model_name)
                logger.debug(f"Loaded {model_name} for {lang}")
            except OSError:
                logger.warning(f"Could not load {model_name}, using basic model")
                self.language_models[lang] = spacy.load(f"{lang}_core_web_sm")
    
    async def _load_concept_embeddings(self):
        """Load pre-computed embeddings for philosophical concepts"""
        # Initialize sentence transformer for concept embeddings
        self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Compute embeddings for all concepts in all languages
        for concept, translations in self.concept_mappings.items():
            self.concept_embeddings[concept] = {}
            for lang, terms in translations.items():
                embeddings = self.sentence_transformer.encode(terms)
                self.concept_embeddings[concept][lang] = {
                    'terms': terms,
                    'embeddings': embeddings,
                    'mean_embedding': np.mean(embeddings, axis=0)
                }
    
    async def _load_cultural_adaptations(self):
        """Load cultural adaptation rules and templates"""
        # In production, this would load from database
        # For now, using predefined templates
        self.cultural_adaptations = self.cultural_contexts.copy()
    
    async def detect_language(self, text: str) -> str:
        """Detect the language of input text"""
        try:
            # Use a simple language detection approach
            # In production, use proper language detection library
            text_lower = text.lower()
            
            # Simple keyword-based detection (replace with proper library)
            spanish_indicators = ['qué', 'cómo', 'por qué', 'dónde', 'la', 'el', 'de', 'que']
            french_indicators = ['qu\'est', 'comment', 'pourquoi', 'où', 'le', 'la', 'de', 'que']
            german_indicators = ['was', 'wie', 'warum', 'wo', 'der', 'die', 'das', 'und']
            italian_indicators = ['che', 'come', 'perché', 'dove', 'il', 'la', 'di', 'che']
            
            if any(indicator in text_lower for indicator in spanish_indicators):
                return 'es'
            elif any(indicator in text_lower for indicator in french_indicators):
                return 'fr'
            elif any(indicator in text_lower for indicator in german_indicators):
                return 'de'
            elif any(indicator in text_lower for indicator in italian_indicators):
                return 'it'
            else:
                return 'en'  # Default to English
                
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'en'  # Default fallback
    
    async def extract_philosophical_concepts(
        self, 
        text: str, 
        language: str = 'en'
    ) -> List[str]:
        """Extract philosophical concepts from text using semantic similarity"""
        try:
            # Get text embedding
            text_embedding = self.sentence_transformer.encode([text])[0]
            
            identified_concepts = []
            
            # Compare with concept embeddings
            for concept, lang_data in self.concept_embeddings.items():
                if language in lang_data:
                    concept_embedding = lang_data[language]['mean_embedding']
                    similarity = cosine_similarity(
                        [text_embedding], 
                        [concept_embedding]
                    )[0][0]
                    
                    # Threshold for concept identification
                    if similarity > 0.3:
                        identified_concepts.append(concept)
            
            return identified_concepts
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return []
    
    async def generate_response(
        self,
        user_input: str,
        language: str,
        cultural_context: Optional[Dict[str, Any]] = None,
        conversation_history: List[Dict] = None,
        philosophical_concepts: List[str] = None
    ) -> Dict[str, Any]:
        """Generate culturally-adapted philosophical response"""
        try:
            # Build context-aware prompt
            prompt = await self._build_philosophical_prompt(
                user_input=user_input,
                language=language,
                cultural_context=cultural_context,
                conversation_history=conversation_history or [],
                philosophical_concepts=philosophical_concepts or []
            )
            
            # Generate response using Cohere
            response = await self.cohere_client.generate(
                model=self.config.get('fine_tuned_model_id', 'command-r-plus'),
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                p=0.75,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            generated_text = response.generations[0].text.strip()
            
            # Apply cultural adaptations
            adapted_response = await self._apply_cultural_adaptations(
                response=generated_text,
                language=language,
                cultural_context=cultural_context
            )
            
            # Generate cultural notes
            cultural_notes = await self._generate_cultural_notes(
                response=adapted_response,
                language=language,
                cultural_context=cultural_context
            )
            
            return {
                'response': adapted_response,
                'cultural_notes': cultural_notes,
                'confidence': 0.8,  # Calculate based on concept matching
                'philosophical_concepts': philosophical_concepts or []
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise ModelException(
                message=f"Failed to generate response: {str(e)}",
                model_id=self.config.get('fine_tuned_model_id', 'command-r-plus')
            )
    
    async def generate_streaming_response(
        self,
        user_input: str,
        language: str,
        cultural_context: Optional[Dict[str, Any]] = None,
        conversation_history: List[Dict] = None
    ):
        """Generate streaming response for real-time interaction"""
        try:
            prompt = await self._build_philosophical_prompt(
                user_input=user_input,
                language=language,
                cultural_context=cultural_context,
                conversation_history=conversation_history or []
            )
            
            # Use Cohere's streaming API
            stream = await self.cohere_client.generate_stream(
                model=self.config.get('fine_tuned_model_id', 'command-r-plus'),
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                k=0,
                p=0.75
            )
            
            async for token in stream:
                if token.event_type == 'text-generation':
                    yield token.text
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"
    
    async def _build_philosophical_prompt(
        self,
        user_input: str,
        language: str,
        cultural_context: Optional[Dict[str, Any]] = None,
        conversation_history: List[Dict] = None,
        philosophical_concepts: List[str] = None
    ) -> str:
        """Build context-aware prompt for philosophical dialogue"""
        
        # Language-specific instruction templates
        instruction_templates = {
            'en': "You are a knowledgeable philosophy teacher specializing in Aristotelian ethics. Respond thoughtfully and educationally.",
            'es': "Eres un profesor de filosofía especializado en la ética aristotélica. Responde de manera reflexiva y educativa.",
            'fr': "Vous êtes un professeur de philosophie spécialisé dans l'éthique aristotélicienne. Répondez de manière réfléchie et éducative.",
            'de': "Sie sind ein Philosophielehrer, der sich auf die aristotelische Ethik spezialisiert hat. Antworten Sie nachdenklich und lehrreich.",
            'it': "Sei un insegnante di filosofia specializzato nell'etica aristotelica. Rispondi in modo riflessivo ed educativo."
        }
        
        instruction = instruction_templates.get(language, instruction_templates['en'])
        
        # Add cultural context if available
        cultural_adaptation = ""
        if cultural_context and cultural_context.get('region'):
            region = cultural_context['region']
            if region in self.cultural_adaptations:
                context = self.cultural_adaptations[region]
                cultural_adaptation = f"\nAdapt your response for {region} cultural context, incorporating values like {', '.join(context['values'][:2])} and using relatable examples."
        
        # Add conversation history
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-4:]  # Last 2 exchanges
            history_context = "\nPrevious conversation:\n"
            for msg in recent_history:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                history_context += f"{role}: {msg['content'][:100]}...\n"
        
        # Add philosophical concepts context
        concepts_context = ""
        if philosophical_concepts:
            concepts_context = f"\nFocus on these philosophical concepts: {', '.join(philosophical_concepts)}"
        
        prompt = f"""{instruction}{cultural_adaptation}{history_context}{concepts_context}