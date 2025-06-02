"""
Polish Law Metadata Flattening Engine

This module contains the same core innovation as the GDPR flattener - the ability to take 
complex, nested metadata structures and "flatten" them into simple key-value pairs that 
vector databases can handle, while preserving all the sophisticated structural information 
for later reconstruction.

The fascinating aspect of this implementation is how it adapts the universal flattening 
principle to Polish law's specific structural patterns. While GDPR often uses alphabetical 
sub-paragraphs like (a), (b), (c), Polish law frequently uses numeric patterns like 1), 2), 3) 
and has different organizational hierarchies. This demonstrates the flexibility and 
adaptability of the flattening approach across different legal systems.

Think of this as a "universal translator" for legal document structures - the same core 
algorithm works across different legal traditions by adapting to their specific patterns.
"""

import json
import logging
from typing import Dict, Any, Set


class PolishLawMetadataFlattener:
    """
    Handles the intelligent flattening of complex Polish law article structure metadata.
    
    This class encapsulates the sophisticated logic for transforming nested metadata
    structures into vector database-compatible formats while preserving all the
    information needed for precise citation creation in the Polish legal context.
    
    The flattening approach is "lossless" - no information is discarded, it's just
    reorganized for technical compatibility while respecting Polish law's unique
    structural patterns and organizational principles.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law metadata flattener.
        
        Args:
            logger: Configured logger for tracking flattening operations
        """
        self.logger = logger
        self.logger.info("Polish Law Metadata Flattener initialized")
        
        # Track flattening statistics across all operations
        # This helps us understand the patterns and complexity in Polish law documents
        self.flattening_stats = {
            'total_processed': 0,
            'enhanced_structures_found': 0,
            'complexity_distribution': {},
            'numbering_styles_found': set(),
            'polish_specific_patterns': set(),  # Track Polish law-specific patterns
            'flattening_errors': 0
        }
    
    def flatten_article_structure(self, article_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligently flatten complex Polish law article structure metadata for vector database compatibility.
        
        This function applies the same "lossless flattening" approach as the GDPR system but is 
        specifically adapted for Polish law structural patterns. Polish legal documents often have 
        different organizational hierarchies and numbering schemes compared to EU regulations, and 
        this implementation recognizes and preserves those patterns.
        
        The beauty of this approach is that we maintain the same conceptual framework (extract 
        simple indicators while preserving complete structure) but adapt the pattern recognition 
        to Polish legal document conventions. This demonstrates how good architectural patterns 
        can be successfully adapted across different domains.
        
        Args:
            article_structure: Complex nested metadata structure from Polish law processing
            
        Returns:
            Flattened metadata dictionary with both simple indicators and preserved structure
        """
        self.logger.debug("Starting Polish law article structure flattening process")
        self.flattening_stats['total_processed'] += 1
        
        # Initialize flattened structure with safe defaults
        flattened = self._create_default_flattened_structure()
        
        # Handle empty or invalid input gracefully
        if not self._validate_input_structure(article_structure):
            self.logger.debug("No complex Polish law structure to flatten")
            return flattened
        
        try:
            # Extract and flatten the structural information using Polish law patterns
            self._extract_basic_indicators(article_structure, flattened)
            self._analyze_polish_paragraph_complexity(article_structure, flattened)
            self._preserve_complete_structure(article_structure, flattened)
            
            # Update statistics for monitoring and optimization
            self._update_flattening_statistics(flattened)
            
            self.logger.debug(f"Successfully flattened Polish law structure: {flattened['paragraph_count']} paragraphs, "
                            f"complexity: {flattened['complexity_level']}")
            
            return flattened
            
        except Exception as e:
            self.flattening_stats['flattening_errors'] += 1
            self.logger.warning(f"Error flattening Polish law article structure: {e}")
            # Return minimal structure to ensure processing continues gracefully
            flattened['article_structure_json'] = json.dumps(article_structure) if article_structure else ''
            return flattened
    
    def _create_default_flattened_structure(self) -> Dict[str, Any]:
        """
        Create the default flattened structure template for Polish law documents.
        
        This provides a consistent structure that all flattening operations return,
        ensuring predictable behavior throughout the system. The structure is identical
        to the GDPR system, demonstrating how the same data model works across different
        legal systems while the processing logic adapts to domain-specific patterns.
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
        
        This ensures we only attempt to flatten valid, non-empty structures,
        preventing errors and providing consistent behavior across all processing scenarios.
        """
        return article_structure and isinstance(article_structure, dict)
    
    def _extract_basic_indicators(self, article_structure: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Extract basic structural indicators from the complex Polish law metadata.
        
        These indicators provide quick access to essential information without requiring 
        deserialization of the complete structure. This optimization allows the citation 
        system to make quick decisions about document complexity and processing approach.
        """
        flattened['has_enhanced_structure'] = True
        flattened['paragraph_count'] = article_structure.get('paragraph_count', 0)
        
        self.logger.debug(f"Extracted basic Polish law indicators: {flattened['paragraph_count']} paragraphs")
    
    def _analyze_polish_paragraph_complexity(self, article_structure: Dict[str, Any], flattened: Dict[str, Any]) -> None:
        """
        Analyze paragraph structure to understand Polish law-specific patterns and complexity.
        
        Polish law has distinct structural patterns that differ from EU regulations. For example,
        Polish law often uses numeric sub-paragraphs like 1), 2), 3) rather than alphabetical 
        ones, and may have different hierarchical organizations. This analysis captures those 
        patterns for efficient processing later while maintaining compatibility with the 
        universal citation system.
        
        This method demonstrates how domain expertise can be encoded into the flattening process
        without breaking the overall architectural pattern.
        """
        # Get paragraph information, handling null values gracefully
        paragraphs_info = article_structure.get('paragraphs', {})
        if paragraphs_info is None:
            paragraphs_info = {}
        
        # Initialize analysis variables for Polish law patterns
        has_any_sub_paragraphs = False
        numbering_styles = set()
        complexity_indicators = []
        polish_specific_indicators = []
        
        # Analyze each paragraph for Polish law-specific patterns
        for para_key, para_info in paragraphs_info.items():
            if isinstance(para_info, dict):
                self._analyze_single_polish_paragraph(
                    para_info, has_any_sub_paragraphs, numbering_styles, 
                    complexity_indicators, polish_specific_indicators
                )
        
        # Store the analysis results with Polish law adaptations
        flattened['has_sub_paragraphs'] = has_any_sub_paragraphs
        flattened['numbering_style'] = list(numbering_styles)[0] if numbering_styles else ''
        flattened['complexity_level'] = self._determine_complexity_level(complexity_indicators)
        
        # Update global statistics with Polish law-specific patterns
        if numbering_styles:
            self.flattening_stats['numbering_styles_found'].update(numbering_styles)
        if polish_specific_indicators:
            self.flattening_stats['polish_specific_patterns'].update(polish_specific_indicators)
            
        self.logger.debug(f"Polish law complexity analysis: {len(complexity_indicators)} indicators, "
                        f"numbering styles: {list(numbering_styles)}")
    
    def _analyze_single_polish_paragraph(self, para_info: Dict[str, Any], has_any_sub_paragraphs: bool,
                                       numbering_styles: Set[str], complexity_indicators: list,
                                       polish_specific_indicators: list) -> None:
        """
        Analyze a single paragraph for Polish law-specific complexity indicators.
        
        This method identifies patterns specific to Polish legal document structure, such as
        the common use of numeric sub-paragraphs and specific hierarchical organizations
        that are characteristic of Polish legal drafting conventions.
        
        Understanding these patterns is crucial for creating precise citations that follow
        Polish legal citation standards while maintaining compatibility with the universal
        citation system architecture.
        """
        # Check if this paragraph has sub-paragraphs (common in Polish law)
        if para_info.get('has_sub_paragraphs', False):
            has_any_sub_paragraphs = True
            complexity_indicators.append('sub_paragraphs')
            
            # Collect Polish law-specific numbering styles
            style = para_info.get('numbering_style', '')
            if style:
                numbering_styles.add(style)
                
                # Identify Polish law-specific patterns
                if style == 'number_closing_paren':  # Common in Polish law: 1), 2), 3)
                    polish_specific_indicators.append('numeric_sub_paragraphs')
                elif style == 'letter_closing_paren':  # Also used: a), b), c)
                    polish_specific_indicators.append('alphabetic_sub_paragraphs')
            
            # Track sub-paragraph complexity specific to Polish law provisions
            sub_count = para_info.get('sub_paragraph_count', 0)
            if sub_count > 5:  # Polish law articles sometimes have many sub-provisions
                complexity_indicators.append('many_sub_paragraphs')
            
            # Check for Polish law-specific structural elements
            if para_info.get('has_nested_provisions', False):
                polish_specific_indicators.append('nested_provisions')
                complexity_indicators.append('nested_structure')
            
            # Look for Polish legal terminology patterns in structure
            if para_info.get('contains_references', False):
                polish_specific_indicators.append('cross_references')
                complexity_indicators.append('complex_references')
    
    def _determine_complexity_level(self, complexity_indicators: list) -> str:
        """
        Determine the overall complexity level based on identified indicators.
        
        This classification helps citation agents quickly filter and prioritize documents
        based on their structural complexity, enabling more efficient processing strategies
        for documents of different complexity levels.
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
        
        This is the most critical part of the flattening process - ensuring that no information 
        is lost. The complete structure is serialized and stored as a string, which can be 
        deserialized later when precise citations are needed. This approach allows us to have 
        the best of both worlds: simple metadata for database compatibility and complete 
        structure for sophisticated analysis.
        
        This demonstrates a key principle in software architecture: when faced with competing 
        constraints (database simplicity vs. data richness), find a solution that satisfies 
        both rather than compromising on either.
        """
        try:
            flattened['article_structure_json'] = json.dumps(article_structure)
            self.logger.debug("Complete Polish law structure preserved as JSON string")
        except Exception as e:
            self.logger.warning(f"Failed to serialize complete Polish law structure: {e}")
            flattened['article_structure_json'] = ''
    
    def _update_flattening_statistics(self, flattened: Dict[str, Any]) -> None:
        """
        Update global statistics about the flattening process.
        
        These statistics help monitor the effectiveness of the flattening approach
        and identify patterns in the processed Polish law documents. This data can
        inform optimizations and help understand the characteristics of the document
        collection being processed.
        """
        if flattened['has_enhanced_structure']:
            self.flattening_stats['enhanced_structures_found'] += 1
            
            complexity = flattened['complexity_level']
            self.flattening_stats['complexity_distribution'][complexity] = \
                self.flattening_stats['complexity_distribution'].get(complexity, 0) + 1
    
    def get_flattening_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the flattening operations performed.
        
        This provides insights into the quality and patterns of the processed Polish law
        documents, helping understand how well the flattening approach is working and
        what types of structural patterns are most common in the document collection.
        """
        stats = dict(self.flattening_stats)
        
        # Convert sets to lists for JSON serialization
        stats['numbering_styles_found'] = list(self.flattening_stats['numbering_styles_found'])
        stats['polish_specific_patterns'] = list(self.flattening_stats['polish_specific_patterns'])
        
        # Calculate enhancement rate for Polish law documents
        if stats['total_processed'] > 0:
            enhancement_rate = (stats['enhanced_structures_found'] / stats['total_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
        else:
            stats['enhancement_rate_percent'] = 0
        
        return stats
    
    def log_flattening_summary(self) -> None:
        """
        Log a comprehensive summary of all Polish law flattening operations.
        
        This provides visibility into how well the flattening process worked across all 
        processed documents, highlighting any Polish law-specific patterns that were 
        discovered and preserved. This information is valuable for both debugging and 
        understanding the characteristics of the document collection.
        """
        stats = self.get_flattening_statistics()
        
        self.logger.info("=== POLISH LAW METADATA FLATTENING SUMMARY ===")
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
            
        if stats['polish_specific_patterns']:
            self.logger.info(f"Polish law-specific patterns: {', '.join(stats['polish_specific_patterns'])}")


def create_polish_law_metadata_flattener(logger: logging.Logger) -> PolishLawMetadataFlattener:
    """
    Factory function to create a configured Polish law metadata flattener.
    
    This provides a clean interface for creating flattener instances with proper 
    dependency injection. The factory pattern ensures consistent initialization 
    and makes it easy to modify the creation process if needed in the future.
    """
    return PolishLawMetadataFlattener(logger)