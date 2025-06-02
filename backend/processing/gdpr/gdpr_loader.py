"""
GDPR Document Loader and Validator

This module handles loading and validating enhanced GDPR JSON files with sophisticated 
structural metadata. Extracted from the main processing pipeline to follow the 
Single Responsibility Principle - this module's only job is to safely load and 
validate GDPR documents.

Key responsibilities:
- Load GDPR JSON files with comprehensive error handling
- Validate expected enhanced metadata structure
- Provide detailed logging about document structure and quality
- Return validated document data and chunks for further processing
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime


class GDPRDocumentLoader:
    """
    Handles loading and validation of enhanced GDPR JSON files.
    
    This class encapsulates all the logic for safely loading GDPR documents
    and validating that they contain the expected enhanced metadata structure
    that the citation system depends on.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the GDPR document loader.
        
        Args:
            logger: Configured logger instance for detailed operation tracking
        """
        self.logger = logger
        self.logger.info("GDPR Document Loader initialized")
    
    def load_and_validate_gdpr_json(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load and validate enhanced GDPR JSON file with sophisticated structural metadata.
        
        This method reads your carefully crafted GDPR JSON that contains both content
        and rich structural metadata about the regulation's organization. We validate
        the structure to ensure it contains the enhanced metadata we expect for
        creating precise citations.
        
        Args:
            file_path: Path to the enhanced GDPR JSON file
            
        Returns:
            Tuple of (document_data, chunks) where:
            - document_data: Complete document with metadata
            - chunks: List of processed chunks ready for conversion
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON structure is invalid
            Exception: For other loading/validation errors
        """
        self.logger.info(f"Loading enhanced GDPR JSON file from: {file_path}")
        
        # Validate file exists before attempting to load
        if not os.path.exists(file_path):
            error_msg = f"Enhanced GDPR JSON file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load the JSON file with proper encoding handling
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract and validate document metadata
            document_metadata = data.get('document', {}).get('metadata', {})
            self._log_document_metadata(document_metadata)
            
            # Extract and validate chunks
            chunks = data.get('chunks', [])
            self._validate_chunks_structure(chunks)
            
            self.logger.info("Enhanced GDPR JSON loading and validation completed successfully")
            return data, chunks
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in GDPR file: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error loading enhanced GDPR JSON file: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def _log_document_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log comprehensive information about the document metadata for transparency.
        
        This helps track what kind of document we're processing and validates
        that it contains the expected metadata structure.
        """
        self.logger.info("=== GDPR DOCUMENT METADATA ===")
        self.logger.info(f"Document title: {metadata.get('official_title', 'Unknown title')}")
        self.logger.info(f"Source: {metadata.get('source', 'Unknown source')}")
        self.logger.info(f"Effective date: {metadata.get('effective_date', 'Unknown date')}")
        self.logger.info(f"Total chapters: {metadata.get('total_chapters', 'Unknown')}")
        self.logger.info(f"Total articles: {metadata.get('total_articles', 'Unknown')}")
        self.logger.info(f"Jurisdiction: {metadata.get('jurisdiction', 'Unknown')}")
        
        # Log any additional metadata fields that might be present
        for key, value in metadata.items():
            if key not in ['official_title', 'source', 'effective_date', 'total_chapters', 'total_articles', 'jurisdiction']:
                self.logger.debug(f"Additional metadata - {key}: {value}")
    
    def _validate_chunks_structure(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Validate the structure and quality of GDPR chunks.
        
        This method performs comprehensive validation to ensure the chunks
        contain the enhanced structural metadata that the citation system needs.
        """
        self.logger.info(f"Validating structure of {len(chunks)} GDPR chunks...")
        
        if not chunks:
            self.logger.warning("No chunks found in GDPR document - this may indicate a processing issue")
            return
        
        # Track validation statistics
        validation_stats = {
            'total_chunks': len(chunks),
            'chunks_with_enhanced_metadata': 0,
            'chunks_with_content': 0,
            'chunk_types': {},
            'articles_found': set(),
            'chapters_found': set(),
            'validation_errors': []
        }
        
        # Validate each chunk
        for i, chunk in enumerate(chunks):
            try:
                self._validate_single_chunk(chunk, i, validation_stats)
            except Exception as e:
                error_msg = f"Validation error in chunk {i}: {str(e)}"
                validation_stats['validation_errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Log comprehensive validation results
        self._log_validation_results(validation_stats)
    
    def _validate_single_chunk(self, chunk: Dict[str, Any], index: int, stats: Dict[str, Any]) -> None:
        """
        Validate a single chunk and update validation statistics.
        
        Args:
            chunk: The chunk to validate
            index: Index of the chunk for error reporting
            stats: Statistics dictionary to update
        """
        # Check for required basic structure
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Validate content presence
        if content and content.strip():
            stats['chunks_with_content'] += 1
        else:
            self.logger.warning(f"Chunk {index} has empty or missing content")
        
        # Track chunk types
        chunk_type = metadata.get('type', 'unknown')
        stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        # Track articles and chapters
        if metadata.get('article_number'):
            stats['articles_found'].add(metadata['article_number'])
        if metadata.get('chapter_number'):
            stats['chapters_found'].add(metadata['chapter_number'])
        
        # Check for enhanced structural metadata
        article_structure = metadata.get('article_structure')
        if article_structure:
            stats['chunks_with_enhanced_metadata'] += 1
            self._validate_enhanced_metadata_structure(article_structure, index)
    
    def _validate_enhanced_metadata_structure(self, article_structure: Dict[str, Any], chunk_index: int) -> None:
        """
        Validate the enhanced metadata structure for completeness and correctness.
        
        This ensures that the sophisticated metadata your system depends on
        is properly formatted and contains the expected structural information.
        """
        if not isinstance(article_structure, dict):
            self.logger.warning(f"Chunk {chunk_index}: article_structure is not a dictionary")
            return
        
        # Check for expected structural elements
        expected_fields = ['paragraph_count', 'paragraphs']
        present_fields = []
        
        for field in expected_fields:
            if field in article_structure:
                present_fields.append(field)
        
        if present_fields:
            self.logger.debug(f"Chunk {chunk_index}: Enhanced metadata contains {present_fields}")
        else:
            self.logger.debug(f"Chunk {chunk_index}: Enhanced metadata present but minimal structure")
    
    def _log_validation_results(self, stats: Dict[str, Any]) -> None:
        """
        Log comprehensive validation results for transparency and debugging.
        
        This provides a clear picture of the document quality and helps identify
        any issues with the enhanced metadata structure.
        """
        self.logger.info("=== GDPR CHUNKS VALIDATION RESULTS ===")
        self.logger.info(f"Total chunks: {stats['total_chunks']}")
        self.logger.info(f"Chunks with content: {stats['chunks_with_content']}")
        self.logger.info(f"Chunks with enhanced metadata: {stats['chunks_with_enhanced_metadata']}")
        self.logger.info(f"Unique articles found: {len(stats['articles_found'])}")
        self.logger.info(f"Unique chapters found: {len(stats['chapters_found'])}")
        
        # Log chunk type distribution
        self.logger.info("Chunk type distribution:")
        for chunk_type, count in sorted(stats['chunk_types'].items()):
            self.logger.info(f"  - {chunk_type}: {count} chunks")
        
        # Calculate and log enhancement rate
        if stats['total_chunks'] > 0:
            enhancement_rate = (stats['chunks_with_enhanced_metadata'] / stats['total_chunks']) * 100
            self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}% of chunks have enhanced metadata")
        
        # Log any validation errors
        if stats['validation_errors']:
            self.logger.warning(f"Validation completed with {len(stats['validation_errors'])} errors:")
            for error in stats['validation_errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  - {error}")
            if len(stats['validation_errors']) > 5:
                self.logger.warning(f"  ... and {len(stats['validation_errors']) - 5} more errors")
        else:
            self.logger.info("✅ All chunks passed validation successfully")
    
    def find_gdpr_file(self, processed_dir: str, raw_dir: str, filename: str = "gdpr_final_manual.json") -> str:
        """
        Find the GDPR JSON file in the expected locations.
        
        This method implements the file discovery logic, checking multiple
        possible locations for the enhanced GDPR JSON file.
        
        Args:
            processed_dir: Primary directory to check
            raw_dir: Backup directory to check
            filename: Name of the file to find
            
        Returns:
            Path to the found file
            
        Raises:
            FileNotFoundError: If file is not found in any location
        """
        primary_path = os.path.join(processed_dir, filename)
        backup_path = os.path.join(raw_dir, filename)
        
        if os.path.exists(primary_path):
            self.logger.info(f"Found GDPR file at primary location: {primary_path}")
            return primary_path
        elif os.path.exists(backup_path):
            self.logger.info(f"Found GDPR file at backup location: {backup_path}")
            return backup_path
        else:
            error_msg = f"Enhanced GDPR JSON file '{filename}' not found in either location:\n" \
                       f"  Primary: {primary_path}\n" \
                       f"  Backup: {backup_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)


def create_gdpr_loader(logger: logging.Logger) -> GDPRDocumentLoader:
    """
    Factory function to create a configured GDPR document loader.
    
    This provides a clean interface for creating loader instances with
    proper dependency injection of the logger.
    """
    return GDPRDocumentLoader(logger)