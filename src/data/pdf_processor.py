# src/data/pdf_processor.py
"""
Advanced PDF processing pipeline for converting Aristotelian texts
Handles 15M+ tokens with structure preservation and metadata extraction
"""

import asyncio
import logging
import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass, asdict

import PyPDF2
import pdfplumber
import pandas as pd
from transformers import AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from langdetect import detect
import textstat

from src.utils.config import get_settings
from src.utils.exceptions import ValidationException
from src.database.models import ProcessedText, PhilosophicalConcept

logger = logging.getLogger(__name__)

@dataclass
class TextSegment:
    """Structured representation of a text segment"""
    id: str
    content: str
    language: str
    section_type: str  # chapter, paragraph, quote, etc.
    page_number: int
    word_count: int
    sentence_count: int
    philosophical_concepts: List[str]
    difficulty_score: float
    metadata: Dict[str, Any]
    timestamp: str

@dataclass
class ProcessedDocument:
    """Complete processed document structure"""
    document_id: str
    title: str
    author: str
    language: str
    total_pages: int
    total_words: int
    total_tokens: int
    segments: List[TextSegment]
    philosophical_concepts: List[str]
    difficulty_distribution: Dict[str, float]
    processing_metadata: Dict[str, Any]
    created_at: str

class AristotelianPDFProcessor:
    """
    Advanced PDF processor specifically designed for philosophical texts
    Handles multilingual content with structure preservation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it', 'ar', 'el', 'la']
        self.tokenizer = None
        self.nlp_models = {}
        
        # Aristotelian concept patterns for extraction
        self.aristotelian_patterns = {
            'virtue_ethics': [
                r'\b(virtue|virtues|arete|arête|virtud|virtudes|vertu|vertus|tugend|virtù)\b',
                r'\b(moral\s+excellence|excelencia\s+moral|excellence\s+morale|moralische\s+exzellenz)\b',
                r'\b(character|carácter|caractère|charakter|carattere)\b'
            ],
            'eudaimonia': [
                r'\b(eudaimonia|eudaimonic|happiness|felicidad|bonheur|glück|felicità)\b',
                r'\b(flourishing|florecimiento|épanouissement|gedeihen|fioritura)\b',
                r'\b(well.?being|bienestar|bien.?être|wohlbefinden|benessere)\b'
            ],
            'practical_wisdom': [
                r'\b(phronesis|practical\s+wisdom|sabiduría\s+práctica|sagesse\s+pratique)\b',
                r'\b(prudence|prudencia|prudence|klugheit|prudenza)\b',
                r'\b(judgment|juicio|jugement|urteil|giudizio)\b'
            ],
            'mean_doctrine': [
                r'\b(golden\s+mean|mean|medio|moyenne|mittelmaß|mezzo)\b',
                r'\b(moderation|moderación|modération|mäßigung|moderazione)\b',
                r'\b(extremes|extremos|extrêmes|extreme|estremi)\b'
            ]
        }
        
        # Section type patterns
        self.section_patterns = {
            'chapter': r'^(chapter|capítulo|chapitre|kapitel|capitolo)\s+\d+',
            'book': r'^(book|libro|livre|buch|libro)\s+[ivxlcdm]+',
            'section': r'^(section|sección|section|abschnitt|sezione)\s+\d+',
            'paragraph': r'^\d+\.\s',
            'quote': r'^["\'«]',
            'footnote': r'^\d+\s'
        }
    
    async def initialize(self):
        """Initialize tokenizer and NLP models"""
        try:
            # Initialize tokenizer for token counting
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            
            # Load spaCy models for each language
            language_models = {
                'en': 'en_core_web_lg',
                'es': 'es_core_news_lg',
                'fr': 'fr_core_news_lg',
                'de': 'de_core_news_lg',
                'it': 'it_core_news_lg'
            }
            
            for lang, model_name in language_models.items():
                try:
                    self.nlp_models[lang] = spacy.load(model_name)
                except OSError:
                    logger.warning(f"Could not load {model_name}, using basic model")
                    self.nlp_models[lang] = spacy.load(f"{lang}_core_web_sm")
            
            logger.info("PDF processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PDF processor: {e}")
            raise
    
    async def process_pdf_batch(
        self, 
        pdf_paths: List[Path],
        output_dir: Path,
        max_workers: int = 4
    ) -> List[ProcessedDocument]:
        """Process multiple PDFs in parallel"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_documents = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for pdf_path in pdf_paths:
                future = executor.submit(self._process_single_pdf, pdf_path)
                futures.append((future, pdf_path))
            
            for future, pdf_path in futures:
                try:
                    processed_doc = await asyncio.wrap_future(future)
                    processed_documents.append(processed_doc)
                    
                    # Save individual document
                    output_file = output_dir / f"{processed_doc.document_id}.json"
                    await self._save_processed_document(processed_doc, output_file)
                    
                    logger.info(f"Processed {pdf_path.name}: {processed_doc.total_tokens} tokens")
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
        
        # Save combined dataset
        await self._save_combined_dataset(processed_documents, output_dir)
        
        return processed_documents
    
    def _process_single_pdf(self, pdf_path: Path) -> ProcessedDocument:
        """Process a single PDF file"""
        try:
            # Extract text and metadata
            raw_text, metadata = self._extract_text_with_metadata(pdf_path)
            
            # Detect language
            language = self._detect_document_language(raw_text)
            
            # Segment text into meaningful chunks
            segments = self._segment_text(raw_text, language, metadata)
            
            # Extract philosophical concepts
            philosophical_concepts = self._extract_philosophical_concepts(segments, language)
            
            # Calculate statistics
            total_words = sum(segment.word_count for segment in segments)
            total_tokens = self._count_tokens(raw_text)
            difficulty_distribution = self._analyze_difficulty_distribution(segments)
            
            # Create document ID
            document_id = hashlib.md5(pdf_path.name.encode()).hexdigest()[:12]
            
            processed_doc = ProcessedDocument(
                document_id=document_id,
                title=metadata.get('title', pdf_path.stem),
                author=metadata.get('author', 'Unknown'),
                language=language,
                total_pages=metadata.get('pages', 0),
                total_words=total_words,
                total_tokens=total_tokens,
                segments=segments,
                philosophical_concepts=philosophical_concepts,
                difficulty_distribution=difficulty_distribution,
                processing_metadata={
                    'source_file': str(pdf_path),
                    'processing_date': datetime.utcnow().isoformat(),
                    'processor_version': '2.0.0',
                    'language_confidence': 0.8  # Would be calculated
                },
                created_at=datetime.utcnow().isoformat()
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise
    
    def _extract_text_with_metadata(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        
        text_content = ""
        metadata = {}
        
        try:
            # Use pdfplumber for better text extraction
            with pdfplumber.open(pdf_path) as pdf:
                metadata['pages'] = len(pdf.pages)
                
                # Extract metadata
                if pdf.metadata:
                    metadata['title'] = pdf.metadata.get('Title', '')
                    metadata['author'] = pdf.metadata.get('Author', '')
                    metadata['creator'] = pdf.metadata.get('Creator', '')
                    metadata['creation_date'] = pdf.metadata.get('CreationDate', '')
                
                # Extract text page by page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean and normalize text
                        cleaned_text = self._clean_text(page_text)
                        text_content += f"\n[PAGE_{page_num}]\n{cleaned_text}\n"
            
        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    metadata['pages'] = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_text = self._clean_text(page_text)
                            text_content += f"\n[PAGE_{page_num}]\n{cleaned_text}\n"
                            
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {pdf_path}: {e2}")
                raise ValidationException(
                    message=f"Could not extract text from PDF: {pdf_path}",
                    field="pdf_file",
                    value=str(pdf_path)
                )
        
        return text_content, metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)  # Rejoin hyphenated words
        text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1 \2', text)  # Fix sentence breaks
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'^.*?Copyright.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Page numbers
        
        # Normalize quotes
        text = re.sub(r'[""''``]', '"', text)
        
        return text.strip()
    
    def _detect_document_language(self, text: str) -> str:
        """Detect the primary language of the document"""
        
        try:
            # Use first 1000 characters for detection
            sample_text = text[:1000]
            detected = detect(sample_text)
            
            if detected in self.supported_languages:
                return detected
            else:
                logger.warning(f"Detected unsupported language: {detected}, defaulting to English")
                return 'en'
                
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def _segment_text(
        self, 
        text: str, 
        language: str, 
        metadata: Dict[str, Any]
    ) -> List[TextSegment]:
        """Segment text into meaningful chunks with structure preservation"""
        
        segments = []
        current_page = 1
        
        # Split by pages first
        page_sections = re.split(r'\[PAGE_(\d+)\]', text)
        
        for i in range(1, len(page_sections), 2):
            page_num = int(page_sections[i])
            page_content = page_sections[i + 1] if i + 1 < len(page_sections) else ""
            
            if not page_content.strip():
                continue
            
            # Further segment each page
            page_segments = self._segment_page_content(
                page_content, 
                language, 
                page_num
            )
            segments.extend(page_segments)
        
        return segments
    
    def _segment_page_content(
        self, 
        content: str, 
        language: str, 
        page_num: int
    ) -> List[TextSegment]:
        """Segment content within a single page"""
        
        segments = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for para_idx, paragraph in enumerate(paragraphs):
            if len(paragraph) < 50:  # Skip very short paragraphs
                continue
            
            # Determine section type
            section_type = self._classify_section_type(paragraph)
            
            # Count words and sentences
            word_count = len(paragraph.split())
            sentences = sent_tokenize(paragraph)
            sentence_count = len(sentences)
            
            # Extract concepts from this segment
            concepts = self._extract_concepts_from_text(paragraph, language)
            
            # Calculate difficulty score
            difficulty_score = self._calculate_difficulty_score(paragraph, language)
            
            # Create segment ID
            segment_id = f"page_{page_num}_para_{para_idx}"
            
            segment = TextSegment(
                id=segment_id,
                content=paragraph,
                language=language,
                section_type=section_type,
                page_number=page_num,
                word_count=word_count,
                sentence_count=sentence_count,
                philosophical_concepts=concepts,
                difficulty_score=difficulty_score,
                metadata={
                    'paragraph_index': para_idx,
                    'avg_sentence_length': word_count / sentence_count if sentence_count > 0 else 0,
                    'has_quotes': '"' in paragraph,
                    'has_references': bool(re.search(r'\d{4}|\bpp?\.\s*\d+', paragraph))
                },
                timestamp=datetime.utcnow().isoformat()
            )
            
            segments.append(segment)
        
        return segments
    
    def _classify_section_type(self, text: str) -> str:
        """Classify the type of text section"""
        
        text_lower = text.lower().strip()
        
        for section_type, pattern in self.section_patterns.items():
            if re.match(pattern, text_lower):
                return section_type
        
        # Additional heuristics
        if text.startswith('"') or text.startswith('"'):
            return 'quote'
        elif len(text.split()) < 20:
            return 'heading'
        elif text.count('.') / len(text.split()) > 0.3:
            return 'paragraph'
        else:
            return 'text_block'
    
    def _extract_concepts_from_text(self, text: str, language: str) -> List[str]:
        """Extract philosophical concepts from text segment"""
        
        concepts = []
        text_lower = text.lower()
        
        for concept_category, patterns in self.aristotelian_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    concepts.append(concept_category)
                    break  # Only add each category once per segment
        
        return concepts
    
    def _calculate_difficulty_score(self, text: str, language: str) -> float:
        """Calculate text difficulty score (0-1, higher = more difficult)"""
        
        try:
            # Base score from readability
            if language == 'en':
                flesch_score = textstat.flesch_reading_ease(text)
                base_score = max(0, (100 - flesch_score) / 100)
            else:
                # Simple heuristic for non-English
                avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
                avg_sentence_length = len(text.split()) / text.count('.')
                base_score = min(1.0, (avg_word_length + avg_sentence_length) / 20)
            
            # Adjust for philosophical complexity
            philosophical_terms = sum(1 for concept in self.aristotelian_patterns.keys() 
                                    if any(re.search(pattern, text.lower()) 
                                          for pattern in self.aristotelian_patterns[concept]))
            
            complexity_bonus = min(0.3, philosophical_terms * 0.1)
            
            return min(1.0, base_score + complexity_bonus)
            
        except:
            return 0.5  # Default moderate difficulty
    
    def _extract_philosophical_concepts(
        self, 
        segments: List[TextSegment], 
        language: str
    ) -> List[str]:
        """Extract all unique philosophical concepts from document"""
        
        all_concepts = set()
        for segment in segments:
            all_concepts.update(segment.philosophical_concepts)
        
        return list(all_concepts)
    
    def _analyze_difficulty_distribution(self, segments: List[TextSegment]) -> Dict[str, float]:
        """Analyze the distribution of difficulty levels"""
        
        if not segments:
            return {}
        
        difficulties = [segment.difficulty_score for segment in segments]
        
        return {
            'easy': sum(1 for d in difficulties if d < 0.3) / len(difficulties),
            'medium': sum(1 for d in difficulties if 0.3 <= d < 0.7) / len(difficulties),
            'hard': sum(1 for d in difficulties if d >= 0.7) / len(difficulties),
            'average': sum(difficulties) / len(difficulties),
            'std_dev': (sum((d - sum(difficulties)/len(difficulties))**2 for d in difficulties) / len(difficulties))**0.5
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the configured tokenizer"""
        try:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                return len(tokens)
            else:
                # Fallback: approximate 4 characters per token
                return len(text) // 4
        except:
            return len(text) // 4
    
    async def _save_processed_document(
        self, 
        document: ProcessedDocument, 
        output_path: Path
    ):
        """Save processed document to JSON file"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(document), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved processed document: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save document {output_path}: {e}")
            raise
    
    async def _save_combined_dataset(
        self, 
        documents: List[ProcessedDocument], 
        output_dir: Path
    ):
        """Save combined dataset with metadata"""
        
        # Create dataset summary
        total_tokens = sum(doc.total_tokens for doc in documents)
        total_words = sum(doc.total_words for doc in documents)
        language_distribution = {}
        concept_frequency = {}
        
        for doc in documents:
            # Language distribution
            lang = doc.language
            language_distribution[lang] = language_distribution.get(lang, 0) + 1
            
            # Concept frequency
            for concept in doc.philosophical_concepts:
                concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
        
        dataset_metadata = {
            'dataset_id': f"aristotelian_corpus_{datetime.utcnow().strftime('%Y%m%d')}",
            'total_documents': len(documents),
            'total_tokens': total_tokens,
            'total_words': total_words,
            'language_distribution': language_distribution,
            'concept_frequency': concept_frequency,
            'difficulty_distribution': self._calculate_overall_difficulty(documents),
            'creation_date': datetime.utcnow().isoformat(),
            'documents': [doc.document_id for doc in documents]
        }
        
        # Save metadata
        metadata_path = output_dir / 'dataset_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)
        
        # Save document list
        document_list = [asdict(doc) for doc in documents]
        dataset_path = output_dir / 'complete_dataset.json'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(document_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved complete dataset: {total_tokens:,} tokens across {len(documents)} documents")
    
    def _calculate_overall_difficulty(self, documents: List[ProcessedDocument]) -> Dict[str, float]:
        """Calculate difficulty distribution across all documents"""
        
        all_scores = []
        for doc in documents:
            for segment in doc.segments:
                all_scores.append(segment.difficulty_score)
        
        if not all_scores:
            return {}
        
        return {
            'easy': sum(1 for d in all_scores if d < 0.3) / len(all_scores),
            'medium': sum(1 for d in all_scores if 0.3 <= d < 0.7) / len(all_scores),
            'hard': sum(1 for d in all_scores if d >= 0.7) / len(all_scores),
            'average': sum(all_scores) / len(all_scores)
        }

# src/data/fine_tuning_pipeline.py
"""
Fine-tuning pipeline for Cohere Command R+ with Aristotelian corpus
Implements cross-lingual training with cultural adaptation
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

import cohere
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.pdf_processor import ProcessedDocument, TextSegment
from src.utils.config import get_settings
from src.utils.exceptions import ModelException

logger = logging.getLogger(__name__)

class CohereFineTuningPipeline:
    """
    Advanced fine-tuning pipeline for philosophical dialogue
    Handles multilingual training with cultural context preservation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cohere_client = None
        self.tokenizer = None
        
        # Training parameters
        self.max_tokens = config.get('max_tokens', 15000000)
        self.batch_size = config.get('training_batch_size', 32)
        self.supported_languages = ['en', 'es', 'fr', 'de', 'it']
        
        # Cultural context mappings for training data augmentation
        self.cultural_contexts = {
            'latin_america': ['es'],
            'mediterranean': ['es', 'fr', 'it'],
            'northern_europe': ['de', 'en'],
            'western_europe': ['fr', 'de', 'en'],
            'global_english': ['en']
        }
    
    async def initialize(self):
        """Initialize Cohere client and tokenizer"""
        try:
            self.cohere_client = cohere.AsyncClient(
                api_key=self.config.get('cohere_api_key')
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
            
            logger.info("Fine-tuning pipeline initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize fine-tuning pipeline: {e}")
            raise
    
    async def prepare_training_data(
        self,
        processed_documents: List[ProcessedDocument],
        output_dir: Path,
        target_samples: int = 50000
    ) -> Dict[str, Any]:
        """
        Prepare training data from processed documents
        Creates question-answer pairs with cultural context
        """
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_samples = []
        validation_samples = []
        
        # Generate training samples from documents
        for doc in processed_documents:
            doc_samples = await self._generate_qa_pairs_from_document(doc)
            training_samples.extend(doc_samples)
        
        # Apply data augmentation for cultural contexts
        augmented_samples = await self._apply_cultural_augmentation(training_samples)
        training_samples.extend(augmented_samples)
        
        # Limit to target number of samples
        if len(training_samples) > target_samples:
            training_samples = training_samples[:target_samples]
        
        # Split train/validation
        train_data, val_data = train_test_split(
            training_samples, 
            test_size=0.1, 
            random_state=42,
            stratify=[sample['language'] for sample in training_samples]
        )
        
        # Format for Cohere fine-tuning
        formatted_train = await self._format_for_cohere(train_data)
        formatted_val = await self._format_for_cohere(val_data)
        
        # Save datasets
        train_path = output_dir / 'training_data.jsonl'
        val_path = output_dir / 'validation_data.jsonl'
        
        await self._save_jsonl(formatted_train, train_path)
        await self._save_jsonl(formatted_val, val_path)
        
        # Generate statistics
        stats = self._generate_training_statistics(train_data, val_data)
        stats_path = output_dir / 'training_statistics.json'
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Prepared {len(train_data)} training samples, {len(val_data)} validation samples")
        
        return {
            'training_path': str(train_path),
            'validation_path': str(val_path),
            'statistics': stats
        }
    
    async def _generate_qa_pairs_from_document(
        self, 
        document: ProcessedDocument
    ) -> List[Dict[str, Any]]:
        """Generate question-answer pairs from a document"""
        
        qa_pairs = []
        
        for segment in document.segments:
            if segment.word_count < 50 or segment.section_type in ['footnote', 'heading']:
                continue
            
            # Generate different types of questions
            questions = await self._generate_questions_for_segment(segment, document.language)
            
            for question in questions:
                qa_pair = {
                    'question': question,
                    'answer': segment.content,
                    'language': document.language,
                    'concepts': segment.philosophical_concepts,
                    'difficulty': segment.difficulty_score,
                    'cultural_context': self._infer_cultural_context(document.language),
                    'source_document': document.document_id,
                    'segment_id': segment.id
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    async def _generate_questions_for_segment(
        self, 
        segment: TextSegment, 
        language: str
    ) -> List[str]:
        """Generate appropriate questions for a text segment"""
        
        questions = []
        
        # Question templates by language
        question_templates = {
            'en': [
                "What does Aristotle say about {}?",
                "How does {} relate to virtue ethics?",
                "Explain the concept of {} according to Aristotle.",
                "What is the Aristotelian view on {}?"
            ],
            'es': [
                "¿Qué dice Aristóteles sobre {}?",
                "¿Cómo se relaciona {} con la ética de las virtudes?",
                "Explica el concepto de {} según Aristóteles.",
                "¿Cuál es la visión aristotélica sobre {}?"
            ],
            'fr': [
                "Que dit Aristote à propos de {} ?",
                "Comment {} se rapporte-t-il à l'éthique des vertus ?",
                "Expliquez le concept de {} selon Aristote.",
                "Quelle est la vision aristotélicienne de {} ?"
            ],
            'de': [
                "Was sagt Aristoteles über {}?",
                "Wie bezieht sich {} auf die Tugendethik?",
                "Erklären Sie das Konzept von {} nach Aristoteles.",
                "Was ist die aristotelische Sicht auf {}?"
            ],
            'it': [
                "Cosa dice Aristotele riguardo a {}?",
                "Come si collega {} all'etica delle virtù?",
                "Spiega il concetto di {} secondo Aristotele.",
                "Qual è la visione aristotelica su {}?"
            ]
        }
        
        templates = question_templates.get(language, question_templates['en'])
        
        # Generate questions based on identified concepts
        for concept in segment.philosophical_concepts:
            concept_terms = {
                'virtue_ethics': {'en': 'virtue', 'es': 'virtud', 'fr': 'vertu', 'de': 'Tugend', 'it': 'virtù'},
                'eudaimonia': {'en': 'happiness', 'es': 'felicidad', 'fr': 'bonheur', 'de': 'Glück', 'it': 'felicità'},
                'practical_wisdom': {'en': 'practical wisdom', 'es': 'sabiduría práctica', 'fr': 'sagesse pratique', 'de': 'praktische Weisheit', 'it': 'saggezza pratica'},
                'mean_doctrine': {'en': 'the golden mean', 'es': 'el término medio', 'fr': 'le juste milieu', 'de': 'die goldene Mitte', 'it': 'la via di mezzo'}
            }
            
            if concept in concept_terms and language in concept_terms[concept]:
                term = concept_terms[concept][language]
                for template in templates[:2]:  # Use first 2 templates
                    question = template.format(term)
                    questions.append(question)
        
        # Add general questions if no specific concepts
        if not questions:
            general_templates = {
                'en': ["Can you explain this philosophical passage?", "What is the main idea here?"],
                'es': ["¿Puedes explicar este pasaje filosófico?", "¿Cuál es la idea principal aquí?"],
                'fr': ["Pouvez-vous expliquer ce passage philosophique ?", "Quelle est l'idée principale ici ?"],
                'de': ["Können Sie diese philosophische Passage erklären?", "Was ist die Hauptidee hier?"],
                'it': ["Puoi spiegare questo passaggio filosofico?", "Qual è l'idea principale qui?"]
            }
            questions = general_templates.get(language, general_templates['en'])
        
        return questions[:3]  # Limit to 3 questions per segment
    
    async def _apply_cultural_augmentation(
        self, 
        training_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply cultural context augmentation to training samples"""
        
        augmented_samples = []
        
        for sample in training_samples:
            language = sample['language']
            
            # Find applicable cultural contexts for this language
            applicable_contexts = []
            for context, languages in self.cultural_contexts.items():
                if language in languages:
                    applicable_contexts.append(context)
            
            # Create variations with different cultural contexts
            for context in applicable_contexts:
                if context != sample['cultural_context']:
                    augmented_sample = sample.copy()
                    augmented_sample['cultural_context'] = context
                    
                    # Modify the answer slightly for cultural adaptation
                    augmented_sample['answer'] = await self._adapt_answer_for_culture(
                        sample['answer'], 
                        context, 
                        language
                    )
                    
                    augmented_samples.append(augmented_sample)
        
        # Limit augmentation to avoid overwhelming the dataset
        max_augmented = len(training_samples) // 2
        return augmented_samples[:max_augmented]
    
    async def _adapt_answer_for_culture(
        self, 
        answer: str, 
        cultural_context: str, 
        language: str
    ) -> str:
        """Adapt answer text for specific cultural context"""
        
        # Cultural adaptation mappings
        cultural_adaptations = {
            'latin_america': {
                'additions': [
                    "En el contexto latinoamericano, esto se relaciona con la importancia de la familia y la comunidad.",
                    "Considering Latin American values of family and community, this concept takes on additional meaning."
                ]
            },
            'mediterranean': {
                'additions': [
                    "Dans la tradition méditerranéenne, cela évoque l'importance de la convivialité et de la sagesse ancienne.",
                    "In Mediterranean culture, this connects to values of hospitality and ancient wisdom."
                ]
            },
            'northern_europe': {
                'additions': [
                    "In Northern European context, this aligns with values of social responsibility and balance.",
                    "Dies resoniert mit nordeuropäischen Werten der sozialen Verantwortung."
                ]
            }
        }
        
        if cultural_context in cultural_adaptations:
            adaptations = cultural_adaptations[cultural_context]['additions']
            
            # Add a culturally relevant note (10% chance to avoid over-augmentation)
            if np.random.random() < 0.1:
                adaptation = np.random.choice(adaptations)
                return f"{answer}\n\n{adaptation}"
        
        return answer
    
    def _infer_cultural_context(self, language: str) -> str:
        """Infer cultural context from language"""
        
        context_mapping = {
            'es': 'latin_america',
            'fr': 'mediterranean', 
            'it': 'mediterranean',
            'de': 'northern_europe',
            'en': 'global_english'
        }
        
        return context_mapping.get(language, 'global_english')
    
    async def _format_for_cohere(
        self, 
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Format samples for Cohere fine-tuning API"""
        
        formatted_samples = []
        
        for sample in samples:
            # Create prompt-completion pair
            prompt = f"Question: {sample['question']}\nContext: {sample.get('cultural_context', 'general')}\nAnswer:"
            completion = sample['answer']
            
            formatted_sample = {
                'prompt': prompt,
                'completion': completion,
                'metadata': {
                    'language': sample['language'],
                    'concepts': sample['concepts'],
                    'difficulty': sample['difficulty']
                }
            }
            
            formatted_samples.append(formatted_sample)
        
        return formatted_samples
    
    async def _save_jsonl(self, data: List[Dict], path: Path):
        """Save data in JSONL format"""
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _generate_training_statistics(
        self, 
        train_data: List[Dict[str, Any]], 
        val_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive training statistics"""
        
        all_data = train_data + val_data
        
        # Language distribution
        language_dist = {}
        for sample in all_data:
            lang = sample['language']
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        # Concept distribution
        concept_dist = {}
        for sample in all_data:
            for concept in sample['concepts']:
                concept_dist[concept] = concept_dist.get(concept, 0) + 1
        
        # Difficulty distribution
        difficulties = [sample['difficulty'] for sample in all_data]
        difficulty_stats = {
            'easy': sum(1 for d in difficulties if d < 0.3) / len(difficulties),
            'medium': sum(1 for d in difficulties if 0.3 <= d < 0.7) / len(difficulties),
            'hard': sum(1 for d in difficulties if d >= 0.7) / len(difficulties),
            'average': sum(difficulties) / len(difficulties)
        }
        
        # Cultural context distribution
        cultural_dist = {}
        for sample in all_data:
            context = sample['cultural_context']
            cultural_dist[context] = cultural_dist.get(context, 0) + 1
        
        return {
            'total_samples': len(all_data),
            'training_samples': len(train_data),
            'validation_samples': len(val_data),
            'language_distribution': language_dist,
            'concept_distribution': concept_dist,
            'difficulty_distribution': difficulty_stats,
            'cultural_distribution': cultural_dist,
            'avg_question_length': sum(len(s['question'].split()) for s in all_data) / len(all_data),
            'avg_answer_length': sum(len(s['answer'].split()) for s in all_data) / len(all_data),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def start_fine_tuning(
        self,
        training_data_path: str,
        validation_data_path: str,
        model_name: str = "aristotelian-multilingual-v1"
    ) -> Dict[str, Any]:
        """Start Cohere fine-tuning job"""
        
        try:
            # Upload training data
            train_dataset = await self.cohere_client.datasets.create(
                name=f"{model_name}-training",
                data=open(training_data_path, 'rb'),
                dataset_type='fine-tuning'
            )
            
            # Upload validation data
            val_dataset = await self.cohere_client.datasets.create(
                name=f"{model_name}-validation", 
                data=open(validation_data_path, 'rb'),
                dataset_type='fine-tuning'
            )
            
            # Start fine-tuning
            finetune = await self.cohere_client.finetunes.create(
                request={
                    'name': model_name,
                    'settings': {
                        'base_model': 'command-r-plus',
                        'dataset_id': train_dataset.id,
                        'hyperparameters': {
                            'early_stopping_patience': 10,
                            'early_stopping_threshold': 0.001,
                            'train_batch_size': self.batch_size,
                            'learning_rate': 0.01
                        }
                    },
                    'validation_data': val_dataset.id
                }
            )
            
            logger.info(f"Started fine-tuning job: {finetune.id}")
            
            return {
                'finetune_id': finetune.id,
                'model_name': model_name,
                'status': 'started',
                'training_dataset_id': train_dataset.id,
                'validation_dataset_id': val_dataset.id
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise ModelException(
                message=f"Failed to start fine-tuning: {str(e)}",
                model_id=model_name
            )
    
    async def monitor_fine_tuning(self, finetune_id: str) -> Dict[str, Any]:
        """Monitor fine-tuning progress"""
        
        try:
            finetune = await self.cohere_client.finetunes.get(finetune_id)
            
            return {
                'id': finetune.id,
                'status': finetune.status,
                'progress': getattr(finetune, 'progress', 0),
                'metrics': getattr(finetune, 'metrics', {}),
                'created_at': finetune.created_at,
                'updated_at': getattr(finetune, 'updated_at', None)
            }
            
        except Exception as e:
            logger.error(f"Failed to get fine-tuning status: {e}")
            raise ModelException(
                message=f"Failed to monitor fine-tuning: {str(e)}",
                model_id=finetune_id
            )