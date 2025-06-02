"""
GDPR Metadata Flattening Engine

This module contains the core innovation of your system - the ability to take complex,
nested metadata structures and "flatten" them into simple key-value pairs that vector
databases can handle, while preserving all the sophisticated structural information
for later reconstruction.

Think of this as a "compression and decompression" system for legal document structure:
- Compression: Complex nested metadata → Simple flat key-value pairs
- Decompression: Simple metadata + JSON string → Full structural reconstruction

This solves the fundamental challenge of representing sophisticated legal document 
organization within technical constraints.
"""

import json
import logging
from typing import Dict, Any, Set


class GDPRMetadataFlattener:
    """
    Handles the intelligent flattening of complex GDPR article structure metadata.
    
    This class encapsulates the sophisticated logic for transforming nested metadata
    structures into vector database-compatible formats while preserving all the
    information needed for precise citation creation.
    
    The flattening approach is "lossless" - no information is discarded, it's just
    reorganized for technical compatibility.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the metadata flattener.
        
        Args:
            logger: Configured logger for tracking flattening operations
        """
        self.logger = logger
        self.logger.info("GDPR Metadata Flattener initialized")
        
        # Track flattening statistics across all operations
        self.flattening_stats = {
            'total_processed': 0,
            'enhanced_structures_found': 0,
            'complexity_distribution': {},
            'numbering_styles_found': set(),
            'flattening_errors': 0
        }
    
    def flatten_article_structure(self, article_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently flatten complex GDPR article structure metadata for vector database compatibility.
        
        This function applies the "lossless flattening" approach. We extract essential information 
        as simple key-value pairs while preserving the complete structure for full access when needed. 
        This solves the core challenge of representing sophisticated legal document organization 
        within technical constraints.
        
        The beauty of this approach is that we don't lose any information - we just transform
        it into a format that works within vector database limitations while maintaining
        all the sophisticated capabilities your citation system needs.
        
        Args:
            article_structure: Complex nested metadata structure from GDPR processing
            
        Returns:
            Flattened metadata dictionary with both simple indicators and preserved structure
        """
        self.logger.debug("Starting GDPR article structure flattening process")
        self.flattening_stats['total_processed'] += 1
        
        # Initialize flattened structure with safe defaults
        flattened = self._create_default_flattened_structure()
        
        # Handle empty or invalid input gracefully
        if not self._validate_input_structure(article_structure):
            self.logger.debug("No complex GDPR structure to flatten")
            return flattened
        
        try:
            # Extract and flatten the structural information
            self._extract_basic_indicators(article_structure, flattened)
            self._analyze_paragraph_complexity(article_structure, flattened)
            self._preserve_complete_structure(article_structure, flattened)
            
            # Update statistics
            self._update_flattening_statistics(flattened)
            
            self.logger.debug(f"Successfully flattened GDPR structure: {flattened['paragraph_count']} paragraphs, "
                            f"complexity: {flattened['complexity_level']}")
            
            return flattened
            
        except Exception as e:
            self.flattening_stats['flattening_errors'] += 1
            self.logger.warning(f"Error flattening GDPR article structure: {e}")
            # Return minimal structure to ensure processing continues
            flattened['article_structure_json'] = json.dumps(article_structure) if article_structure else ''
            return flattened
    
    def _create_default_flattened_structure(self) -> Dict[str, Any]:
        """
        Create the default flattened structure template.
        
        This provides a consistent structure that all flattening operations return,
        ensuring predictable behavior throughout the system.
        """
        return {
            'has_enhanced_structure': False,
            'paragraph_count': 0,
            'has_sub_paragraphs': False,
            'numbering_style': '',
            'complexity_level': 'simple',  # simple, mixed, complex
            'article_structure_json': ''   # Complete structure preserved as string
        }
    
    def _validate_input_structure(self, article_structure: Any) -> bool:
        """
        Validate that the input structure is suitable for flattening.
        
        This ensures we only attempt to flatten valid, non-empty structures.
        """
        return article_structure and isinstance(article_structure, dict)
    
    def _extract_basic_indicators(self, article_structure: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Extract basic structural indicators from the complex metadata.
        
        These indicators provide quick access to essential information without
        requiring deserialization of the complete structure.
        """
        flattened['has_enhanced_structure'] = True
        flattened['paragraph_count'] = article_structure.get('paragraph_count', 0)
        
        self.logger.debug(f"Extracted basic indicators: {flattened['paragraph_count']} paragraphs")
    
    def _analyze_paragraph_complexity(self, article_structure: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Analyze paragraph structure to understand GDPR-specific patterns and complexity.
        
        GDPR often uses specific patterns like alphabetical sub-paragraphs (a), (b), (c)
        that are different from other legal systems. This analysis captures those patterns
        for efficient processing later.
        """
        # Get paragraph information, handling null values gracefully
        paragraphs_info = article_structure.get('paragraphs', {})
        if paragraphs_info is None:
            paragraphs_info = {}
        
        # Initialize analysis variables
        has_any_sub_paragraphs = False
        numbering_styles = set()
        complexity_indicators = []
        
        # Analyze each paragraph for GDPR-specific patterns
        for para_key, para_info in paragraphs_info.items():
            if isinstance(para_info, dict):
                self._analyze_single_paragraph(para_info, has_any_sub_paragraphs, 
                                             numbering_styles, complexity_indicators)
        
        # Store the analysis results
        flattened['has_sub_paragraphs'] = has_any_sub_paragraphs
        flattened['numbering_style'] = list(numbering_styles)[0] if numbering_styles else ''
        flattened['complexity_level'] = self._determine_complexity_level(complexity_indicators)
        
        # Update global statistics
        if numbering_styles:
            self.flattening_stats['numbering_styles_found'].update(numbering_styles)
    
    def _analyze_single_paragraph(self, para_info: Dict[str, Any], has_any_sub_paragraphs: bool,
                                 numbering_styles: Set[str], complexity_indicators: list) -> None:
        """
        Analyze a single paragraph for complexity indicators.
        
        This method identifies patterns specific to GDPR structure, such as
        alphabetical sub-paragraphs and various numbering schemes.
        """
        # Check if this paragraph has sub-paragraphs
        if para_info.get('has_sub_paragraphs', False):
            has_any_sub_paragraphs = True
            complexity_indicators.append('sub_paragraphs')
            
            # Collect GDPR-specific numbering styles (alphabetical vs numeric)
            style = para_info.get('numbering_style', '')
            if style:
                numbering_styles.add(style)
            
            # Track sub-paragraph complexity for GDPR provisions
            sub_count = para_info.get('sub_paragraph_count', 0)
            if sub_count > 4:  # GDPR often has many sub-provisions
                complexity_indicators.append('many_sub_paragraphs')
    
    def _determine_complexity_level(self, complexity_indicators: list) -> str:
        """
        Determine the overall complexity level based on identified indicators.
        
        This classification helps agents quickly filter and prioritize documents
        based on their structural complexity.
        """
        if len(complexity_indicators) == 0:
            return 'simple'
        elif len(complexity_indicators) <= 2:
            return 'mixed'
        else:
            return 'complex'
    
    def _preserve_complete_structure(self, article_structure: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Preserve the complete structure as a JSON string for full reconstruction.
        
        This is the most important part of the flattening process - ensuring that
        no information is lost. The complete structure is serialized and stored
        as a string, which can be deserialized later when precise citations are needed.
        """
        try:
            flattened['article_structure_json'] = json.dumps(article_structure)
            self.logger.debug("Complete structure preserved as JSON string")
        except Exception as e:
            self.logger.warning(f"Failed to serialize complete structure: {e}")
            flattened['article_structure_json'] = ''
    
    def _update_flattening_statistics(self, flattened: Dict[str, Any]) -> None:
        """
        Update global statistics about the flattening process.
        
        These statistics help monitor the effectiveness of the flattening approach
        and identify patterns in the processed documents.
        """
        if flattened['has_enhanced_structure']:
            self.flattening_stats['enhanced_structures_found'] += 1
            
            complexity = flattened['complexity_level']
            self.flattening_stats['complexity_distribution'][complexity] = \
                self.flattening_stats['complexity_distribution'].get(complexity, 0) + 1
    
    def get_flattening_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the flattening operations performed.
        
        This provides insights into the quality and patterns of the processed documents.
        """
        stats = dict(self.flattening_stats)
        
        # Convert set to list for JSON serialization
        stats['numbering_styles_found'] = list(self.flattening_stats['numbering_styles_found'])
        
        # Calculate enhancement rate
        if stats['total_processed'] > 0:
            enhancement_rate = (stats['enhanced_structures_found'] / stats['total_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
        else:
            stats['enhancement_rate_percent'] = 0
        
        return stats
    
    def log_flattening_summary(self) -> None:
        """
        Log a comprehensive summary of all flattening operations.
        
        This provides visibility into how well the flattening process worked
        across all processed documents.
        """
        stats = self.get_flattening_statistics()
        
        self.logger.info("=== GDPR METADATA FLATTENING SUMMARY ===")
        self.logger.info(f"Total structures processed: {stats['total_processed']}")
        self.logger.info(f"Enhanced structures found: {stats['enhanced_structures_found']}")
        self.logger.info(f"Enhancement rate: {stats['enhancement_rate_percent']}%")
        self.logger.info(f"Flattening errors: {stats['flattening_errors']}")
        
        if stats['complexity_distribution']:
            self.logger.info("Complexity distribution:")
            for complexity, count in sorted(stats['complexity_distribution'].items()):
                self.logger.info(f"  - {complexity}: {count} structures")
        
        if stats['numbering_styles_found']:
            self.logger.info(f"Numbering styles found: {', '.join(stats['numbering_styles_found'])}")


def create_gdpr_metadata_flattener(logger: logging.Logger) -> GDPRMetadataFlattener:
    """
    Factory function to create a configured GDPR metadata flattener.
    
    This provides a clean interface for creating flattener instances with
    proper dependency injection.
    """
    return GDPRMetadataFlattener(logger)