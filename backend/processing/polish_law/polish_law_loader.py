"""
Polish Law Document Loader and Validator

This module handles loading and validating enhanced Polish law JSON files with 
sophisticated structural metadata. Following the same proven pattern as the GDPR loader,
this module's only responsibility is to safely load and validate Polish law documents.

Key responsibilities:
- Load Polish law JSON files with comprehensive error handling
- Validate expected enhanced metadata structure specific to Polish legal documents
- Provide detailed logging about document structure and quality
- Return validated document data and chunks for further processing

The validation logic is adapted for Polish law document patterns while maintaining
the same robust error handling and logging approach established in the GDPR system.
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime


class PolishLawDocumentLoader:
    """
    Handles loading and validation of enhanced Polish law JSON files.
    
    This class encapsulates all the logic for safely loading Polish law documents
    and validating that they contain the expected enhanced metadata structure
    that the citation system depends on. The validation is specifically tailored
    for Polish legal document patterns and organizational structure.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law document loader.
        
        Args:
            logger: Configured logger instance for detailed operation tracking
        """
        self.logger = logger
        self.logger.info("Polish Law Document Loader initialized")
    
    def load_and_validate_polish_law_json(self, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load and validate enhanced Polish law JSON file with sophisticated structural metadata.
        
        This method reads your carefully crafted Polish law JSON that contains both content
        and rich structural metadata about the law's organization. We validate the structure
        to ensure it contains the enhanced metadata we expect for creating precise citations
        in the Polish legal context.
        
        Args:
            file_path: Path to the enhanced Polish law JSON file
            
        Returns:
            Tuple of (document_data, chunks) where:
            - document_data: Complete document with metadata
            - chunks: List of processed chunks ready for conversion
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If the JSON structure is invalid
            Exception: For other loading/validation errors
        """
        self.logger.info(f"Loading enhanced Polish law JSON file from: {file_path}")
        
        # Validate file exists before attempting to load
        if not os.path.exists(file_path):
            error_msg = f"Enhanced Polish law JSON file not found: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            # Load the JSON file with proper encoding handling for Polish characters
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract and validate document metadata specific to Polish law
            document_metadata = data.get('document', {}).get('metadata', {})
            self._log_document_metadata(document_metadata)
            
            # Extract and validate chunks with Polish law-specific patterns
            chunks = data.get('chunks', [])
            self._validate_chunks_structure(chunks)
            
            self.logger.info("Enhanced Polish law JSON loading and validation completed successfully")
            return data, chunks
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format in Polish law file: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Error loading enhanced Polish law JSON file: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def _log_document_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log comprehensive information about the Polish law document metadata.
        
        This helps track what kind of document we're processing and validates
        that it contains the expected metadata structure for Polish legal documents.
        The logging is tailored to highlight Polish law-specific attributes.
        """
        self.logger.info("=== POLISH LAW DOCUMENT METADATA ===")
        self.logger.info(f"Document title: {metadata.get('official_title', 'Unknown title')}")
        self.logger.info(f"Source: {metadata.get('source', 'Unknown source')}")
        self.logger.info(f"Effective date: {metadata.get('effective_date', 'Unknown date')}")
        self.logger.info(f"Total chapters: {metadata.get('total_chapters', 'Unknown')}")
        self.logger.info(f"Total articles: {metadata.get('total_articles', 'Unknown')}")
        self.logger.info(f"Jurisdiction: {metadata.get('jurisdiction', 'Unknown')}")
        
        # Log Polish law-specific metadata fields
        if metadata.get('law_type'):
            self.logger.info(f"Law type: {metadata['law_type']}")
        if metadata.get('parliament_session'):
            self.logger.info(f"Parliament session: {metadata['parliament_session']}")
        if metadata.get('gazette_reference'):
            self.logger.info(f"Official gazette reference: {metadata['gazette_reference']}")
        
        # Log any additional metadata fields that might be present
        for key, value in metadata.items():
            if key not in ['official_title', 'source', 'effective_date', 'total_chapters', 
                          'total_articles', 'jurisdiction', 'law_type', 'parliament_session', 
                          'gazette_reference']:
                self.logger.debug(f"Additional metadata - {key}: {value}")
    
    def _validate_chunks_structure(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Validate the structure and quality of Polish law chunks.
        
        This method performs comprehensive validation to ensure the chunks contain
        the enhanced structural metadata that the citation system needs. The validation
        is specifically adapted for Polish legal document structure and patterns.
        """
        self.logger.info(f"Validating structure of {len(chunks)} Polish law chunks...")
        
        if not chunks:
            self.logger.warning("No chunks found in Polish law document - this may indicate a processing issue")
            return
        
        # Track validation statistics specific to Polish law documents
        validation_stats = {
            'total_chunks': len(chunks),
            'chunks_with_enhanced_metadata': 0,
            'chunks_with_content': 0,
            'chunk_types': {},
            'articles_found': set(),
            'chapters_found': set(),
            'sections_found': set(),  # Polish law often has sections
            'validation_errors': []
        }
        
        # Validate each chunk with Polish law-specific checks
        for i, chunk in enumerate(chunks):
            try:
                self._validate_single_chunk(chunk, i, validation_stats)
            except Exception as e:
                error_msg = f"Validation error in Polish law chunk {i}: {str(e)}"
                validation_stats['validation_errors'].append(error_msg)
                self.logger.warning(error_msg)
        
        # Log comprehensive validation results
        self._log_validation_results(validation_stats)
    
    def _validate_single_chunk(self, chunk: Dict[str, Any], index: int, stats: Dict[str, Any]) -> None:
        """
        Validate a single chunk and update validation statistics.
        
        This method checks for Polish law-specific structural elements and patterns
        while maintaining the same validation rigor as the GDPR system.
        
        Args:
            chunk: The chunk to validate
            index: Index of the chunk for error reporting
            stats: Statistics dictionary to update
        """
        # Check for required basic structure
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Validate content presence and quality
        if content and content.strip():
            stats['chunks_with_content'] += 1
            
            # Check for Polish-specific content indicators
            if any(polish_indicator in content.lower() for polish_indicator in 
                   ['ustawa', 'artykuł', 'rozdział', 'przepis']):
                self.logger.debug(f"Chunk {index} contains Polish legal terminology")
        else:
            self.logger.warning(f"Polish law chunk {index} has empty or missing content")
        
        # Track chunk types with Polish law-specific categories
        chunk_type = metadata.get('type', 'unknown')
        stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        # Track Polish law organizational elements
        if metadata.get('article_number'):
            stats['articles_found'].add(metadata['article_number'])
        if metadata.get('chapter_number'):
            stats['chapters_found'].add(metadata['chapter_number'])
        if metadata.get('section_number'):  # Polish law often uses sections
            stats['sections_found'].add(metadata['section_number'])
        
        # Check for enhanced structural metadata specific to Polish law
        article_structure = metadata.get('article_structure')
        if article_structure:
            stats['chunks_with_enhanced_metadata'] += 1
            self._validate_enhanced_metadata_structure(article_structure, index)
    
    def _validate_enhanced_metadata_structure(self, article_structure: Dict[str, Any], chunk_index: int) -> None:
        """
        Validate the enhanced metadata structure for Polish law documents.
        
        This ensures that the sophisticated metadata your system depends on
        is properly formatted and contains the expected structural information
        specific to Polish legal document organization.
        """
        if not isinstance(article_structure, dict):
            self.logger.warning(f"Polish law chunk {chunk_index}: article_structure is not a dictionary")
            return
        
        # Check for expected structural elements in Polish law
        expected_fields = ['paragraph_count', 'paragraphs']
        present_fields = []
        
        for field in expected_fields:
            if field in article_structure:
                present_fields.append(field)
        
        # Check for Polish law-specific structural patterns
        if 'paragraphs' in article_structure:
            paragraphs = article_structure['paragraphs']
            if isinstance(paragraphs, dict):
                # Look for Polish numbering patterns
                polish_numbering_patterns = ['1)', '2)', 'a)', 'b)']
                found_patterns = []
                
                for para_key, para_data in paragraphs.items():
                    if isinstance(para_data, dict):
                        numbering_style = para_data.get('numbering_style', '')
                        if numbering_style and numbering_style not in found_patterns:
                            found_patterns.append(numbering_style)
                
                if found_patterns:
                    self.logger.debug(f"Polish law chunk {chunk_index}: Found numbering patterns: {found_patterns}")
        
        if present_fields:
            self.logger.debug(f"Polish law chunk {chunk_index}: Enhanced metadata contains {present_fields}")
        else:
            self.logger.debug(f"Polish law chunk {chunk_index}: Enhanced metadata present but minimal structure")
    
    def _log_validation_results(self, stats: Dict[str, Any]) -> None:
        """
        Log comprehensive validation results for Polish law documents.
        
        This provides a clear picture of the document quality and helps identify
        any issues with the enhanced metadata structure specific to Polish law.
        """
        self.logger.info("=== POLISH LAW CHUNKS VALIDATION RESULTS ===")
        self.logger.info(f"Total chunks: {stats['total_chunks']}")
        self.logger.info(f"Chunks with content: {stats['chunks_with_content']}")
        self.logger.info(f"Chunks with enhanced metadata: {stats['chunks_with_enhanced_metadata']}")
        self.logger.info(f"Unique articles found: {len(stats['articles_found'])}")
        self.logger.info(f"Unique chapters found: {len(stats['chapters_found'])}")
        self.logger.info(f"Unique sections found: {len(stats['sections_found'])}")
        
        # Log chunk type distribution for Polish law
        self.logger.info("Polish law chunk type distribution:")
        for chunk_type, count in sorted(stats['chunk_types'].items()):
            self.logger.info(f"  - {chunk_type}: {count} chunks")
        
        # Calculate and log enhancement rate
        if stats['total_chunks'] > 0:
            enhancement_rate = (stats['chunks_with_enhanced_metadata'] / stats['total_chunks']) * 100
            self.logger.info(f"Enhancement rate: {enhancement_rate:.1f}% of chunks have enhanced metadata")
        
        # Log any validation errors encountered
        if stats['validation_errors']:
            self.logger.warning(f"Validation completed with {len(stats['validation_errors'])} errors:")
            for error in stats['validation_errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  - {error}")
            if len(stats['validation_errors']) > 5:
                self.logger.warning(f"  ... and {len(stats['validation_errors']) - 5} more errors")
        else:
            self.logger.info("✅ All Polish law chunks passed validation successfully")
    
    def find_polish_law_file(self, processed_dir: str, raw_dir: str, 
                           filename: str = "polish_law_final_manual.json") -> str:
        """
        Find the Polish law JSON file in the expected locations.
        
        This method implements the file discovery logic, checking multiple
        possible locations for the enhanced Polish law JSON file. It follows
        the same reliable pattern established in the GDPR system.
        
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
            self.logger.info(f"Found Polish law file at primary location: {primary_path}")
            return primary_path
        elif os.path.exists(backup_path):
            self.logger.info(f"Found Polish law file at backup location: {backup_path}")
            return backup_path
        else:
            error_msg = f"Enhanced Polish law JSON file '{filename}' not found in either location:\n" \
                       f"  Primary: {primary_path}\n" \
                       f"  Backup: {backup_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)


def create_polish_law_loader(logger: logging.Logger) -> PolishLawDocumentLoader:
    """
    Factory function to create a configured Polish law document loader.
    
    This provides a clean interface for creating loader instances with
    proper dependency injection of the logger. The factory pattern ensures
    consistent initialization across the application.
    """
    return PolishLawDocumentLoader(logger)