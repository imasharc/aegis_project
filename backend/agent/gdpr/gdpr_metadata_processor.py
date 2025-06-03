"""
GDPR Metadata Processor

This module reconstructs sophisticated structural information from the flattened metadata
created by the processing pipeline. It serves as the "reverse" of the GDPRMetadataFlattener,
taking simple key-value pairs and rebuilding the complex nested structures needed for
precise citation creation.

Think of this as the "decompression" half of your metadata system:
- Processing Pipeline: Complex structures → Flattened metadata (compression)
- Agent Pipeline: Flattened metadata → Reconstructed structures (decompression)

This approach demonstrates how architectural consistency creates powerful synergies
between different parts of your system.
"""

import json
import logging
from typing import Dict, List, Any, Optional


class GDPRMetadataProcessor:
    """
    Processes and reconstructs flattened GDPR metadata for sophisticated citation creation.
    
    This class performs the inverse operation of the GDPRMetadataFlattener, taking the
    simple key-value pairs stored in the vector database and reconstructing the complex
    structural information needed for precise legal citations.
    
    The processor demonstrates how your flattened metadata approach enables sophisticated
    functionality while maintaining vector database compatibility.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the GDPR metadata processor.
        
        Args:
            logger: Configured logger for tracking metadata processing operations
        """
        self.logger = logger
        self.logger.info("GDPR Metadata Processor initialized")
        
        # Track processing statistics across all operations
        self.processing_stats = {
            'total_metadata_processed': 0,
            'enhanced_structures_reconstructed': 0,
            'json_deserialization_successes': 0,
            'json_deserialization_failures': 0,
            'fallback_to_indicators': 0,
            'processing_errors': 0
        }
    
    def extract_and_reconstruct_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and reconstruct sophisticated structural information from flattened metadata.
        
        This method takes the simple key-value pairs created by your processing pipeline
        and rebuilds the complex structural information that your citation system needs.
        It demonstrates how the flattened metadata approach preserves all functionality
        while working within vector database constraints.
        
        Args:
            metadata: Flattened metadata dictionary from vector store document
            
        Returns:
            Reconstructed metadata with both quick indicators and full structure
        """
        self.logger.debug("Processing flattened GDPR metadata for structural reconstruction")
        self.processing_stats['total_metadata_processed'] += 1
        
        # Initialize comprehensive metadata structure with safe defaults
        reconstructed_info = self._create_default_metadata_structure(metadata)
        
        # Extract quick structural indicators for efficient processing
        if reconstructed_info['has_enhanced_structure']:
            self._extract_quick_indicators(metadata, reconstructed_info)
            
            # Attempt to reconstruct complete structure from preserved JSON
            success = self._reconstruct_complete_structure(metadata, reconstructed_info)
            
            if success:
                self.processing_stats['enhanced_structures_reconstructed'] += 1
                self.logger.debug(f"Successfully reconstructed enhanced structure: "
                                f"{reconstructed_info['quick_indicators']['paragraph_count']} paragraphs, "
                                f"complexity: {reconstructed_info['quick_indicators']['complexity_level']}")
            else:
                self.processing_stats['fallback_to_indicators'] += 1
                self.logger.debug("Using quick indicators only - complete structure unavailable")
        else:
            self.logger.debug(f"Basic metadata processed: Article {reconstructed_info['article_number']} "
                            f"(no enhanced structure)")
        
        return reconstructed_info
    
    def _create_default_metadata_structure(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the default reconstructed metadata structure.
        
        This provides a consistent foundation that works whether we have enhanced
        metadata or just basic document information. The structure mirrors what
        the original agent expected but now comes from processed metadata.
        """
        return {
            # Basic document identifiers (always available from processing pipeline)
            'article_number': metadata.get('article_number', ''),
            'chapter_number': metadata.get('chapter_number', ''),
            'chapter_title': metadata.get('chapter_title', ''),
            'section_number': metadata.get('section_number', ''),
            'section_title': metadata.get('section_title', ''),
            'article_title': metadata.get('article_title', ''),
            
            # Enhancement indicators from flattened metadata
            'has_enhanced_structure': metadata.get('has_enhanced_structure', False),
            
            # Containers for reconstructed information
            'quick_indicators': {},
            'full_structure': None,
            
            # Processing context
            'reconstruction_successful': False,
            'reconstruction_method': 'none'
        }
    
    def _extract_quick_indicators(self, metadata: Dict[str, Any], 
                                 reconstructed_info: Dict[str, Any]) -> None:
        """
        Extract quick structural indicators from flattened metadata.
        
        These indicators provide immediate access to essential structural information
        without requiring JSON deserialization. This demonstrates how your flattening
        approach creates multiple levels of access to the same information.
        """
        quick_indicators = {
            'paragraph_count': metadata.get('paragraph_count', 0),
            'has_sub_paragraphs': metadata.get('has_sub_paragraphs', False),
            'numbering_style': metadata.get('numbering_style', ''),
            'complexity_level': metadata.get('complexity_level', 'simple')
        }
        
        reconstructed_info['quick_indicators'] = quick_indicators
        reconstructed_info['reconstruction_method'] = 'quick_indicators'
        
        self.logger.debug(f"Extracted quick indicators: {quick_indicators['paragraph_count']} paragraphs, "
                        f"complexity: {quick_indicators['complexity_level']}, "
                        f"sub-paragraphs: {quick_indicators['has_sub_paragraphs']}")
    
    def _reconstruct_complete_structure(self, metadata: Dict[str, Any], 
                                       reconstructed_info: Dict[str, Any]) -> bool:
        """
        Reconstruct complete structural information from preserved JSON.
        
        This method demonstrates the power of your flattened metadata approach.
        The complete structure was preserved as a JSON string during processing,
        and now we can deserialize it to access all the sophisticated structural
        information your citation system needs.
        """
        json_str = metadata.get('article_structure_json', '')
        
        if not json_str:
            self.logger.debug("No preserved JSON structure available")
            return False
        
        try:
            # Deserialize the complete structure that was preserved during processing
            full_structure = json.loads(json_str)
            reconstructed_info['full_structure'] = full_structure
            reconstructed_info['reconstruction_successful'] = True
            reconstructed_info['reconstruction_method'] = 'full_json_reconstruction'
            
            self.processing_stats['json_deserialization_successes'] += 1
            
            self.logger.debug("Successfully reconstructed complete structure from preserved JSON")
            
            # Validate the reconstructed structure for consistency
            self._validate_reconstructed_structure(full_structure, reconstructed_info)
            
            return True
            
        except json.JSONDecodeError as e:
            self.processing_stats['json_deserialization_failures'] += 1
            self.logger.warning(f"Failed to deserialize preserved JSON structure: {e}")
            return False
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.warning(f"Error during structure reconstruction: {e}")
            return False
    
    def _validate_reconstructed_structure(self, full_structure: Dict[str, Any], 
                                         reconstructed_info: Dict[str, Any]) -> None:
        """
        Validate that the reconstructed structure is consistent with quick indicators.
        
        This validation ensures that the flattening and reconstruction process
        maintained data integrity throughout the pipeline. Any inconsistencies
        could indicate issues with the processing pipeline that need attention.
        """
        if not isinstance(full_structure, dict):
            self.logger.warning("Reconstructed structure is not a dictionary")
            return
        
        # Compare quick indicators with reconstructed structure for consistency
        quick_count = reconstructed_info['quick_indicators']['paragraph_count']
        reconstructed_count = full_structure.get('paragraph_count', 0)
        
        if quick_count != reconstructed_count:
            self.logger.warning(f"Paragraph count inconsistency: quick={quick_count}, "
                              f"reconstructed={reconstructed_count}")
        
        # Validate paragraph structure if present
        paragraphs = full_structure.get('paragraphs', {})
        if paragraphs and isinstance(paragraphs, dict):
            actual_paragraph_count = len(paragraphs)
            if actual_paragraph_count != reconstructed_count:
                self.logger.warning(f"Paragraph structure count mismatch: "
                                  f"declared={reconstructed_count}, actual={actual_paragraph_count}")
        
        self.logger.debug("Structure validation completed")
    
    def create_processing_hints(self, reconstructed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create processing hints for content analysis based on reconstructed metadata.
        
        This method transforms the reconstructed structural information into hints
        that can guide content parsing. It demonstrates how metadata processing
        enables intelligent, guided analysis rather than blind text parsing.
        
        Args:
            reconstructed_info: Reconstructed metadata information
            
        Returns:
            Dictionary of processing hints for content analysis
        """
        if not reconstructed_info['has_enhanced_structure']:
            return {
                'has_hints': False,
                'use_guided_parsing': False,
                'parsing_strategy': 'simple'
            }
        
        quick_indicators = reconstructed_info['quick_indicators']
        
        # Create comprehensive hints based on available information
        hints = {
            'has_hints': True,
            'use_guided_parsing': True,
            'parsing_strategy': 'enhanced' if reconstructed_info['reconstruction_successful'] else 'indicator_guided',
            
            # Direct indicators for parsing guidance
            'paragraph_count': quick_indicators['paragraph_count'],
            'has_sub_paragraphs': quick_indicators['has_sub_paragraphs'],
            'numbering_style': quick_indicators['numbering_style'],
            'complexity_level': quick_indicators['complexity_level'],
            
            # Parsing recommendations based on structure
            'recommended_parser': self._recommend_parser_strategy(quick_indicators),
            'expected_patterns': self._identify_expected_patterns(quick_indicators),
            
            # Full structure availability
            'full_structure_available': reconstructed_info['reconstruction_successful'],
            'reconstruction_method': reconstructed_info['reconstruction_method']
        }
        
        self.logger.debug(f"Created processing hints: {hints['parsing_strategy']} strategy, "
                        f"parser: {hints['recommended_parser']}")
        
        return hints
    
    def _recommend_parser_strategy(self, quick_indicators: Dict[str, Any]) -> str:
        """
        Recommend the best parsing strategy based on structural indicators.
        
        This method analyzes the available metadata to determine which parsing
        approach will be most effective for the specific document structure.
        """
        complexity = quick_indicators['complexity_level']
        has_sub_paragraphs = quick_indicators['has_sub_paragraphs']
        paragraph_count = quick_indicators['paragraph_count']
        
        if complexity == 'complex' and has_sub_paragraphs:
            return 'sophisticated_with_sub_paragraphs'
        elif has_sub_paragraphs:
            return 'guided_with_sub_paragraphs'
        elif paragraph_count > 1:
            return 'multi_paragraph'
        else:
            return 'simple_single_paragraph'
    
    def _identify_expected_patterns(self, quick_indicators: Dict[str, Any]) -> List[str]:
        """
        Identify expected structural patterns based on metadata indicators.
        
        This helps the content analyzer know what patterns to look for,
        making parsing more reliable and efficient.
        """
        patterns = []
        
        numbering_style = quick_indicators['numbering_style']
        has_sub_paragraphs = quick_indicators['has_sub_paragraphs']
        
        # Add pattern expectations based on metadata
        if has_sub_paragraphs:
            if numbering_style == 'alphabetical':
                patterns.append('alphabetical_sub_paragraphs')  # (a), (b), (c)
            elif numbering_style == 'number_closing_paren':
                patterns.append('numeric_sub_paragraphs')  # (1), (2), (3)
            else:
                patterns.append('mixed_sub_paragraphs')
        
        # Add GDPR-specific patterns
        patterns.append('gdpr_article_structure')
        
        if quick_indicators['paragraph_count'] > 1:
            patterns.append('multi_paragraph_article')
        
        return patterns
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about metadata processing operations.
        
        This provides insights into how well the reconstruction process is working
        and helps identify any issues with the flattened metadata approach.
        """
        stats = dict(self.processing_stats)
        
        # Calculate success rates
        if stats['total_metadata_processed'] > 0:
            enhancement_rate = (stats['enhanced_structures_reconstructed'] / stats['total_metadata_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
            
            if stats['enhanced_structures_reconstructed'] > 0:
                json_success_rate = (stats['json_deserialization_successes'] / stats['enhanced_structures_reconstructed']) * 100
                stats['json_success_rate_percent'] = round(json_success_rate, 1)
            else:
                stats['json_success_rate_percent'] = 0
        else:
            stats['enhancement_rate_percent'] = 0
            stats['json_success_rate_percent'] = 0
        
        return stats
    
    def log_processing_summary(self) -> None:
        """
        Log a comprehensive summary of all metadata processing operations.
        
        This provides visibility into how well the metadata reconstruction
        process is working across all processed documents.
        """
        stats = self.get_processing_statistics()
        
        self.logger.info("=== GDPR METADATA PROCESSING SUMMARY ===")
        self.logger.info(f"Total metadata processed: {stats['total_metadata_processed']}")
        self.logger.info(f"Enhanced structures reconstructed: {stats['enhanced_structures_reconstructed']}")
        self.logger.info(f"Enhancement rate: {stats['enhancement_rate_percent']}%")
        self.logger.info(f"JSON reconstruction successes: {stats['json_deserialization_successes']}")
        self.logger.info(f"JSON success rate: {stats['json_success_rate_percent']}%")
        self.logger.info(f"Fallback to indicators: {stats['fallback_to_indicators']}")
        self.logger.info(f"Processing errors: {stats['processing_errors']}")


def create_gdpr_metadata_processor(logger: logging.Logger) -> GDPRMetadataProcessor:
    """
    Factory function to create a configured GDPR metadata processor.
    
    This provides a clean interface for creating processor instances with
    proper dependency injection of the logger.
    """
    return GDPRMetadataProcessor(logger)