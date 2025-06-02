"""
Polish Law Document Converter with Intelligent Metadata Integration

This module handles the conversion of enhanced Polish law JSON chunks into LangChain 
Document objects while applying the sophisticated metadata flattening approach.

Following the same proven pattern as the GDPR converter, this creates "bilingual documents" 
that work efficiently with vector databases while preserving all the information needed 
for precise legal citations. The converter adapts the universal conversion process to 
Polish law-specific document patterns and organizational structures.

This demonstrates a key architectural principle: the same conversion framework can be 
adapted to different domains by changing the domain-specific processing logic while 
maintaining the same overall structure and error handling patterns.
"""

import os
import logging
from typing import Dict, List, Any
from datetime import datetime
from langchain.docstore.document import Document

from polish_law_metadata_flattener import PolishLawMetadataFlattener


class PolishLawDocumentConverter:
    """
    Converts enhanced Polish law JSON chunks to LangChain Documents with intelligent metadata flattening.
    
    This class represents the core solution to the vector database constraint challenge
    for Polish law documents. We transform sophisticated nested metadata into a format 
    that Chroma can store while preserving all the information the enhanced agent needs 
    for precise citations.
    
    The conversion process creates documents that can "speak" both the vector database
    language (simple key-value pairs) and the sophisticated legal analysis language
    (complex nested structures) simultaneously. This dual capability is what makes
    the citation system so powerful and precise.
    """
    
    def __init__(self, metadata_flattener: PolishLawMetadataFlattener, logger: logging.Logger):
        """
        Initialize the Polish law document converter.
        
        The dependency injection pattern used here is crucial for maintainability.
        By injecting the metadata flattener rather than creating it internally,
        we make the converter testable and flexible. This also demonstrates the
        Single Responsibility Principle - this class handles conversion, while
        the flattener handles the complex metadata transformation logic.
        
        Args:
            metadata_flattener: Configured metadata flattener for processing complex structures
            logger: Configured logger for tracking conversion operations
        """
        self.metadata_flattener = metadata_flattener
        self.logger = logger
        self.logger.info("Polish Law Document Converter initialized with intelligent metadata flattening")
        
        # Track conversion statistics to monitor system performance
        # These statistics help identify patterns and optimize the conversion process
        self.conversion_stats = {
            'total_chunks': 0,
            'successful_conversions': 0,
            'enhanced_structure_count': 0,
            'chunk_types': {},
            'complexity_levels': {},
            'errors': 0,
            'polish_specific_patterns': {}  # Track Polish law-specific patterns
        }
    
    def convert_chunks_to_documents(self, chunks: List[Dict[str, Any]], 
                                  source_metadata: Dict[str, Any],
                                  processing_timestamp: str) -> List[Document]:
        """
        Convert enhanced Polish law JSON chunks to LangChain Document objects with intelligent metadata flattening.
        
        This function represents the core solution to the vector database constraint challenge
        for Polish law documents. We transform sophisticated nested metadata into a format 
        that Chroma can store while preserving all the information your enhanced agent needs 
        for precise citations.
        
        The process demonstrates how complex transformations can be broken down into manageable,
        well-tested steps. Each chunk goes through validation, metadata enhancement, flattening,
        and final document creation, with comprehensive error handling at each stage.
        
        Args:
            chunks: List of enhanced JSON chunks from Polish law processing
            source_metadata: Document-level metadata for context
            processing_timestamp: Timestamp for tracking processing sessions
            
        Returns:
            List of LangChain Document objects ready for embedding
        """
        self.logger.info("Starting conversion of enhanced Polish law JSON chunks to LangChain Document objects...")
        self.logger.info("Implementing intelligent metadata flattening for vector database compatibility...")
        
        self.conversion_stats['total_chunks'] = len(chunks)
        docs = []
        
        # Process each chunk individually with comprehensive error handling
        # This approach ensures that a problem with one chunk doesn't break the entire conversion
        for i, chunk in enumerate(chunks):
            try:
                document = self._convert_single_chunk(chunk, source_metadata, processing_timestamp, i)
                if document:
                    docs.append(document)
                    self.conversion_stats['successful_conversions'] += 1
                    
            except Exception as e:
                self.conversion_stats['errors'] += 1
                self.logger.error(f"Error converting Polish law chunk {i}: {str(e)}")
                continue  # Continue processing other chunks even if one fails
        
        # Log comprehensive conversion statistics
        self._log_conversion_results()
        
        return docs
    
    def _convert_single_chunk(self, chunk: Dict[str, Any], source_metadata: Dict[str, Any],
                             processing_timestamp: str, chunk_index: int) -> Document:
        """
        Convert a single chunk to a LangChain Document with enhanced metadata.
        
        This method handles the detailed work of creating a properly formatted document
        with both flattened metadata for database compatibility and preserved complex
        structure for sophisticated analysis. The step-by-step approach makes the
        conversion process transparent and debuggable.
        
        The method demonstrates how complex processes can be broken down into clear,
        manageable steps that each have a single, well-defined responsibility.
        """
        content = chunk.get('content', '')
        metadata = chunk.get('metadata', {})
        
        # Step 1: Validate content quality before proceeding
        # This early validation prevents creating empty or invalid documents
        if not self._validate_chunk_content(content, chunk_index):
            return None
        
        # Step 2: Build the enhanced but flattened metadata structure
        # This creates the foundation metadata that includes both Polish law-specific
        # information and processing context needed for sophisticated analysis
        enhanced_metadata = self._build_enhanced_metadata(metadata, source_metadata, 
                                                         processing_timestamp, chunk_index)
        
        # Step 3: Apply intelligent metadata flattening for complex structures
        # This is where the magic happens - complex nested structures get transformed
        # into simple key-value pairs while preserving all the information
        self._apply_metadata_flattening(metadata, enhanced_metadata, chunk_index)
        
        # Step 4: Track statistics for this conversion
        # This helps us understand patterns and optimize the conversion process
        self._update_conversion_statistics(metadata, enhanced_metadata)
        
        # Step 5: Create the final document with flattened but complete metadata
        document = Document(
            page_content=content.strip(),
            metadata=enhanced_metadata
        )
        
        # Log sample conversion details for the first few documents
        # This provides visibility into how the conversion process is working
        if chunk_index < 3:
            self._log_sample_conversion(enhanced_metadata, content, chunk_index)
        
        return document
    
    def _validate_chunk_content(self, content: str, chunk_index: int) -> bool:
        """
        Validate that the chunk content is suitable for processing.
        
        This ensures we don't create empty or invalid documents that would cause 
        issues in the vector database. Early validation saves time and prevents 
        problems downstream in the processing pipeline.
        
        Content validation is particularly important for Polish law documents because
        they may contain special characters, formatting, or encoding issues that
        need to be caught early in the process.
        """
        if not content or not content.strip():
            self.logger.warning(f"Empty content in Polish law chunk {chunk_index}, skipping...")
            return False
        
        # Additional validation for Polish law documents
        # Check for minimum content length to ensure meaningful documents
        if len(content.strip()) < 10:
            self.logger.warning(f"Polish law chunk {chunk_index} content too short ({len(content)} chars), skipping...")
            return False
        
        return True
    
    def _build_enhanced_metadata(self, chunk_metadata: Dict[str, Any], 
                                source_metadata: Dict[str, Any],
                                processing_timestamp: str, chunk_index: int) -> Dict[str, Any]:
        """
        Build the enhanced metadata structure that preserves all essential information.
        
        This creates the foundation metadata that includes both Polish law-specific information
        and processing context needed for sophisticated analysis. The metadata structure
        is designed to be both comprehensive and database-compatible.
        
        The metadata design demonstrates how to balance completeness with simplicity,
        ensuring that the citation system has all the information it needs while
        maintaining compatibility with vector database constraints.
        """
        # Track chunk type for statistical analysis and processing optimization
        chunk_type = chunk_metadata.get('type', 'unknown')
        
        # Create the comprehensive metadata structure for Polish law documents
        # This structure includes both standard legal document metadata and
        # Polish law-specific organizational information
        enhanced_metadata = {
            # Basic Polish law document structure (always simple values for database compatibility)
            'type': chunk_type,
            'chapter_number': chunk_metadata.get('chapter_number', ''),
            'chapter_title': chunk_metadata.get('chapter_title', ''),
            'section_number': chunk_metadata.get('section_number', ''),  # Polish law often uses sections
            'section_title': chunk_metadata.get('section_title', ''),
            'article_number': chunk_metadata.get('article_number', ''),
            'article_title': chunk_metadata.get('article_title', ''),
            'page': chunk_metadata.get('page', ''),
            
            # Polish law-specific context for legal research
            # These fields help the citation system understand the Polish legal context
            'law': 'polish_data_protection',
            'source': source_metadata.get('source', ''),
            'official_title': source_metadata.get('official_title', ''),
            'effective_date': source_metadata.get('effective_date', ''),
            'jurisdiction': source_metadata.get('jurisdiction', 'Poland'),
            'law_type': source_metadata.get('law_type', 'national_law'),
            
            # Polish legal system-specific metadata
            'parliament_session': source_metadata.get('parliament_session', ''),
            'gazette_reference': source_metadata.get('gazette_reference', ''),
            'amendment_info': source_metadata.get('amendment_info', ''),
            
            # Processing metadata for debugging and optimization
            # This information helps track the processing pipeline and debug issues
            'chunk_index': chunk_index,
            'processing_timestamp': processing_timestamp
        }
        
        return enhanced_metadata
    
    def _apply_metadata_flattening(self, chunk_metadata: Dict[str, Any], 
                                  enhanced_metadata: Dict[str, Any], chunk_index: int) -> None:
        """
        Apply the sophisticated metadata flattening to complex article structures.
        
        This is where we take sophisticated nested metadata and apply the flattening algorithm
        to make it compatible with vector databases while preserving all the information needed 
        for precise citations. The flattening process is the core innovation that makes the
        entire system possible.
        
        The process demonstrates how complex technical challenges can be solved through
        intelligent design - we transform the problem (complex metadata vs. simple database)
        into a solution that satisfies both requirements simultaneously.
        """
        article_structure = chunk_metadata.get('article_structure', {})
        
        if article_structure:
            # Apply our sophisticated flattening algorithm adapted for Polish law
            # The flattener understands Polish law-specific patterns and preserves them
            flattened_structure = self.metadata_flattener.flatten_article_structure(article_structure)
            
            # Merge flattened structure into the document metadata
            # This creates the "bilingual" metadata that works with both simple and complex systems
            enhanced_metadata.update(flattened_structure)
            
            # Track that this chunk has enhanced structure for statistics
            self.conversion_stats['enhanced_structure_count'] += 1
            
            self.logger.debug(f"Enhanced Polish law chunk {chunk_index}: Article {enhanced_metadata.get('article_number', 'N/A')} "
                           f"with {flattened_structure.get('paragraph_count', 0)} paragraphs, "
                           f"complexity: {flattened_structure.get('complexity_level', 'unknown')}")
        else:
            # No enhanced structure - set basic indicators for Polish law compatibility
            # This ensures consistent metadata structure across all documents
            enhanced_metadata.update({
                'has_enhanced_structure': False,
                'paragraph_count': 0,
                'has_sub_paragraphs': False,
                'numbering_style': '',
                'complexity_level': 'simple',
                'article_structure_json': ''
            })
    
    def _update_conversion_statistics(self, chunk_metadata: Dict[str, Any], 
                                    enhanced_metadata: Dict[str, Any]) -> None:
        """
        Update conversion statistics for monitoring and reporting.
        
        These statistics help track the effectiveness of the conversion process and
        identify patterns in the processed documents. The data is invaluable for
        optimizing the system and understanding the characteristics of the Polish
        law document collection being processed.
        
        Statistics collection is a best practice in data processing systems because
        it provides visibility into system performance and helps identify issues
        before they become serious problems.
        """
        # Track chunk types to understand document composition
        chunk_type = chunk_metadata.get('type', 'unknown')
        self.conversion_stats['chunk_types'][chunk_type] = \
            self.conversion_stats['chunk_types'].get(chunk_type, 0) + 1
        
        # Track complexity levels if enhanced structure is present
        if enhanced_metadata.get('has_enhanced_structure', False):
            complexity = enhanced_metadata.get('complexity_level', 'unknown')
            self.conversion_stats['complexity_levels'][complexity] = \
                self.conversion_stats['complexity_levels'].get(complexity, 0) + 1
        
        # Track Polish law-specific patterns
        if enhanced_metadata.get('section_number'):
            self.conversion_stats['polish_specific_patterns']['sections_found'] = \
                self.conversion_stats['polish_specific_patterns'].get('sections_found', 0) + 1
        
        if enhanced_metadata.get('gazette_reference'):
            self.conversion_stats['polish_specific_patterns']['gazette_references'] = \
                self.conversion_stats['polish_specific_patterns'].get('gazette_references', 0) + 1
    
    def _log_sample_conversion(self, enhanced_metadata: Dict[str, Any], 
                              content: str, chunk_index: int) -> None:
        """
        Log details about sample conversions for verification and debugging.
        
        This provides visibility into how the conversion process is working for the 
        first few documents, helping identify any issues early. Sample logging is 
        a practical compromise between having complete visibility and avoiding 
        information overload in the logs.
        
        The logging demonstrates how to provide just enough information for debugging
        without overwhelming the system with excessive detail.
        """
        has_structure = enhanced_metadata.get('has_enhanced_structure', False)
        chunk_type = enhanced_metadata.get('type', 'unknown')
        article_number = enhanced_metadata.get('article_number', 'N/A')
        complexity = enhanced_metadata.get('complexity_level', 'unknown')
        
        self.logger.info(f"Sample Polish law chunk {chunk_index}: {chunk_type} - "
                      f"Article {article_number} - "
                      f"Enhanced: {'✓' if has_structure else '✗'} - "
                      f"Complexity: {complexity} - "
                      f"Content: {len(content)} chars")
        
        # Log Polish law-specific information
        if enhanced_metadata.get('section_number'):
            self.logger.info(f"  Section: {enhanced_metadata['section_number']} - {enhanced_metadata.get('section_title', 'N/A')}")
        
        if enhanced_metadata.get('gazette_reference'):
            self.logger.info(f"  Gazette reference: {enhanced_metadata['gazette_reference']}")
    
    def _log_conversion_results(self) -> None:
        """
        Log comprehensive conversion results for transparency and monitoring.
        
        This provides a complete picture of how well the conversion process worked
        and helps identify any issues that need attention. Comprehensive reporting
        is essential for maintaining system quality and identifying optimization
        opportunities.
        
        The logging structure demonstrates how to present complex statistical information
        in a clear, actionable format that helps both developers and system administrators
        understand system performance.
        """
        stats = self.conversion_stats
        
        self.logger.info("=" * 60)
        self.logger.info("ENHANCED POLISH LAW METADATA CONVERSION COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Total chunks processed: {stats['total_chunks']}")
        self.logger.info(f"Successful conversions: {stats['successful_conversions']}")
        self.logger.info(f"Enhanced structures: {stats['enhanced_structure_count']}")
        self.logger.info(f"Conversion errors: {stats['errors']}")
        
        # Log chunk type distribution for Polish law documents
        if stats['chunk_types']:
            self.logger.info("Polish law chunk type distribution:")
            for chunk_type, count in sorted(stats['chunk_types'].items()):
                self.logger.info(f"  - {chunk_type}: {count} chunks")
        
        # Log complexity distribution
        if stats['complexity_levels']:
            self.logger.info("Polish law complexity level distribution:")
            for complexity, count in sorted(stats['complexity_levels'].items()):
                self.logger.info(f"  - {complexity}: {count} chunks")
        
        # Log Polish law-specific patterns found
        if stats['polish_specific_patterns']:
            self.logger.info("Polish law-specific patterns found:")
            for pattern, count in sorted(stats['polish_specific_patterns'].items()):
                self.logger.info(f"  - {pattern}: {count} instances")
        
        # Calculate and log enhancement rate
        if stats['successful_conversions'] > 0:
            enhancement_rate = (stats['enhanced_structure_count'] / stats['successful_conversions']) * 100
            self.logger.info(f"Polish law enhanced structure rate: {enhancement_rate:.1f}%")
        
        # Log metadata flattening summary from the flattener component
        self.metadata_flattener.log_flattening_summary()
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the conversion process.
        
        This method provides access to all the statistics collected during the
        conversion process, making it possible for other components to incorporate
        this information into their own reporting and analysis.
        
        Returns:
            Dictionary containing detailed conversion statistics
        """
        stats = dict(self.conversion_stats)
        
        # Add metadata flattening statistics from the flattener component
        # This demonstrates how modular components can share information
        flattening_stats = self.metadata_flattener.get_flattening_statistics()
        stats['metadata_flattening'] = flattening_stats
        
        return stats


def create_polish_law_document_converter(metadata_flattener: PolishLawMetadataFlattener, 
                                       logger: logging.Logger) -> PolishLawDocumentConverter:
    """
    Factory function to create a configured Polish law document converter.
    
    This provides a clean interface for creating converter instances with proper 
    dependency injection of the metadata flattener and logger. The factory pattern 
    ensures consistent initialization and makes it easy to modify the creation 
    process if needed in the future.
    
    The factory pattern is particularly valuable in complex systems because it
    centralizes object creation logic and makes it easier to manage dependencies
    and configuration across the entire application.
    """
    return PolishLawDocumentConverter(metadata_flattener, logger)