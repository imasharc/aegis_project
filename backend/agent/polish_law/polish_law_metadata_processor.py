"""
Polish Law Metadata Processor

This module reconstructs sophisticated structural information from the flattened metadata
created by the Polish law processing pipeline. It serves as the "reverse" of the 
PolishLawMetadataFlattener, taking simple key-value pairs and rebuilding the complex 
nested structures needed for precise Polish legal citation creation.

Polish Law Specific Features:
- Section-aware metadata reconstruction (unique to Polish legal structure)
- Gazette reference processing for legal authenticity validation
- Parliament session and amendment information handling
- Polish legal numbering pattern recognition and processing
- Enhanced support for Polish legal document organizational patterns

Think of this as the "decompression" half of your Polish law metadata system:
- Processing Pipeline: Complex structures → Flattened metadata (compression)
- Agent Pipeline: Flattened metadata → Reconstructed structures (decompression)

This approach demonstrates how architectural consistency creates powerful synergies
between different parts of your system while respecting Polish legal document conventions.
"""

import json
import logging
from typing import Dict, List, Any, Optional


class PolishLawMetadataProcessor:
    """
    Processes and reconstructs flattened Polish law metadata for sophisticated citation creation.
    
    This class performs the inverse operation of the PolishLawMetadataFlattener, taking the
    simple key-value pairs stored in the vector database and reconstructing the complex
    structural information needed for precise Polish legal citations.
    
    The processor demonstrates how your flattened metadata approach enables sophisticated
    functionality while maintaining vector database compatibility, specifically adapted
    for Polish legal document patterns including sections, gazette references, and
    Polish legal organizational structures.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law metadata processor.
        
        Args:
            logger: Configured logger for tracking metadata processing operations
        """
        self.logger = logger
        self.logger.info("Polish Law Metadata Processor initialized")
        
        # Track processing statistics across all operations with Polish law specifics
        self.processing_stats = {
            'total_metadata_processed': 0,
            'enhanced_structures_reconstructed': 0,
            'json_deserialization_successes': 0,
            'json_deserialization_failures': 0,
            'fallback_to_indicators': 0,
            'processing_errors': 0,
            'sections_processed': 0,  # Unique to Polish law
            'gazette_references_processed': 0,  # Important for Polish law authenticity
            'parliament_sessions_found': 0  # Polish law-specific metadata
        }
    
    def extract_and_reconstruct_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and reconstruct sophisticated structural information from flattened Polish law metadata.
        
        This method takes the simple key-value pairs created by your Polish law processing pipeline
        and rebuilds the complex structural information that your citation system needs for Polish
        legal documents. It demonstrates how the flattened metadata approach preserves all functionality
        while working within vector database constraints, with special handling for Polish law features.
        
        Args:
            metadata: Flattened metadata dictionary from vector store document
            
        Returns:
            Reconstructed metadata with both quick indicators and full structure for Polish law
        """
        self.logger.debug("Processing flattened Polish law metadata for structural reconstruction")
        self.processing_stats['total_metadata_processed'] += 1
        
        # Initialize comprehensive metadata structure with safe defaults for Polish law
        reconstructed_info = self._create_default_polish_law_metadata_structure(metadata)
        
        # Extract quick structural indicators for efficient processing
        if reconstructed_info['has_enhanced_structure']:
            self._extract_polish_law_quick_indicators(metadata, reconstructed_info)
            
            # Attempt to reconstruct complete structure from preserved JSON
            success = self._reconstruct_complete_polish_law_structure(metadata, reconstructed_info)
            
            if success:
                self.processing_stats['enhanced_structures_reconstructed'] += 1
                self.logger.debug(f"Successfully reconstructed Polish law enhanced structure: "
                                f"{reconstructed_info['quick_indicators']['paragraph_count']} paragraphs, "
                                f"complexity: {reconstructed_info['quick_indicators']['complexity_level']}")
            else:
                self.processing_stats['fallback_to_indicators'] += 1
                self.logger.debug("Using quick indicators only - complete Polish law structure unavailable")
        else:
            self.logger.debug(f"Basic Polish law metadata processed: Article {reconstructed_info['article_number']} "
                            f"(no enhanced structure)")
        
        # Process Polish law-specific metadata elements
        self._process_polish_law_specific_metadata(metadata, reconstructed_info)
        
        return reconstructed_info
    
    def _create_default_polish_law_metadata_structure(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the default reconstructed metadata structure for Polish law documents.
        
        This provides a consistent foundation that works whether we have enhanced
        metadata or just basic document information, with Polish law-specific fields
        that differ from GDPR structure.
        """
        return {
            # Basic document identifiers (always available from Polish law processing pipeline)
            'article_number': metadata.get('article_number', ''),
            'chapter_number': metadata.get('chapter_number', ''),
            'chapter_title': metadata.get('chapter_title', ''),
            'section_number': metadata.get('section_number', ''),  # Unique to Polish law structure
            'section_title': metadata.get('section_title', ''),    # Polish law organizational element
            'article_title': metadata.get('article_title', ''),
            
            # Polish law-specific identifiers and metadata
            'law_type': metadata.get('law_type', 'national_law'),
            'jurisdiction': metadata.get('jurisdiction', 'Poland'),
            'gazette_reference': metadata.get('gazette_reference', ''),    # Critical for Polish law authenticity
            'parliament_session': metadata.get('parliament_session', ''), # Polish law-specific provenance
            'amendment_info': metadata.get('amendment_info', ''),         # Important for Polish law evolution
            'effective_date': metadata.get('effective_date', ''),
            
            # Enhancement indicators from flattened metadata
            'has_enhanced_structure': metadata.get('has_enhanced_structure', False),
            
            # Containers for reconstructed information
            'quick_indicators': {},
            'full_structure': None,
            'polish_law_specifics': {},  # Container for Polish law-unique elements
            
            # Processing context
            'reconstruction_successful': False,
            'reconstruction_method': 'none'
        }
    
    def _extract_polish_law_quick_indicators(self, metadata: Dict[str, Any], 
                                           reconstructed_info: Dict[str, Any]) -> None:
        """
        Extract quick structural indicators from flattened Polish law metadata.
        
        These indicators provide immediate access to essential structural information
        without requiring JSON deserialization, adapted for Polish legal document patterns
        including section organization and Polish numbering conventions.
        """
        quick_indicators = {
            'paragraph_count': metadata.get('paragraph_count', 0),
            'has_sub_paragraphs': metadata.get('has_sub_paragraphs', False),
            'numbering_style': metadata.get('numbering_style', ''),
            'complexity_level': metadata.get('complexity_level', 'simple'),
            
            # Polish law-specific quick indicators
            'has_sections': bool(metadata.get('section_number', '')),  # Polish law organizational feature
            'has_gazette_reference': bool(metadata.get('gazette_reference', '')),  # Legal authenticity indicator
            'has_amendment_info': bool(metadata.get('amendment_info', ''))  # Polish law evolution tracking
        }
        
        reconstructed_info['quick_indicators'] = quick_indicators
        reconstructed_info['reconstruction_method'] = 'polish_law_quick_indicators'
        
        self.logger.debug(f"Extracted Polish law quick indicators: {quick_indicators['paragraph_count']} paragraphs, "
                        f"complexity: {quick_indicators['complexity_level']}, "
                        f"sub-paragraphs: {quick_indicators['has_sub_paragraphs']}, "
                        f"sections: {quick_indicators['has_sections']}")
    
    def _reconstruct_complete_polish_law_structure(self, metadata: Dict[str, Any], 
                                                 reconstructed_info: Dict[str, Any]) -> bool:
        """
        Reconstruct complete structural information from preserved JSON for Polish law documents.
        
        This method demonstrates the power of your flattened metadata approach applied to Polish law.
        The complete structure was preserved as a JSON string during processing, and now we can 
        deserialize it to access all the sophisticated structural information your Polish law 
        citation system needs.
        """
        json_str = metadata.get('article_structure_json', '')
        
        if not json_str:
            self.logger.debug("No preserved JSON structure available for Polish law document")
            return False
        
        try:
            # Deserialize the complete structure that was preserved during Polish law processing
            full_structure = json.loads(json_str)
            reconstructed_info['full_structure'] = full_structure
            reconstructed_info['reconstruction_successful'] = True
            reconstructed_info['reconstruction_method'] = 'polish_law_full_json_reconstruction'
            
            self.processing_stats['json_deserialization_successes'] += 1
            
            self.logger.debug("Successfully reconstructed complete Polish law structure from preserved JSON")
            
            # Validate the reconstructed structure for consistency with Polish law patterns
            self._validate_reconstructed_polish_law_structure(full_structure, reconstructed_info)
            
            return True
            
        except json.JSONDecodeError as e:
            self.processing_stats['json_deserialization_failures'] += 1
            self.logger.warning(f"Failed to deserialize preserved Polish law JSON structure: {e}")
            return False
        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.warning(f"Error during Polish law structure reconstruction: {e}")
            return False
    
    def _validate_reconstructed_polish_law_structure(self, full_structure: Dict[str, Any], 
                                                   reconstructed_info: Dict[str, Any]) -> None:
        """
        Validate that the reconstructed Polish law structure is consistent with quick indicators.
        
        This validation ensures that the flattening and reconstruction process
        maintained data integrity throughout the Polish law processing pipeline. Any inconsistencies
        could indicate issues with the processing pipeline that need attention.
        """
        if not isinstance(full_structure, dict):
            self.logger.warning("Reconstructed Polish law structure is not a dictionary")
            return
        
        # Compare quick indicators with reconstructed structure for consistency
        quick_count = reconstructed_info['quick_indicators']['paragraph_count']
        reconstructed_count = full_structure.get('paragraph_count', 0)
        
        if quick_count != reconstructed_count:
            self.logger.warning(f"Polish law paragraph count inconsistency: quick={quick_count}, "
                              f"reconstructed={reconstructed_count}")
        
        # Validate paragraph structure if present
        paragraphs = full_structure.get('paragraphs', {})
        if paragraphs and isinstance(paragraphs, dict):
            actual_paragraph_count = len(paragraphs)
            if actual_paragraph_count != reconstructed_count:
                self.logger.warning(f"Polish law paragraph structure count mismatch: "
                                  f"declared={reconstructed_count}, actual={actual_paragraph_count}")
        
        # Validate Polish law-specific structural elements
        self._validate_polish_law_specific_structure_elements(full_structure)
        
        self.logger.debug("Polish law structure validation completed")
    
    def _validate_polish_law_specific_structure_elements(self, full_structure: Dict[str, Any]) -> None:
        """
        Validate Polish law-specific structural elements in the reconstructed structure.
        
        This ensures that Polish law-specific patterns like section organization
        and numbering conventions are properly preserved and reconstructed.
        """
        # Check for Polish law-specific numbering patterns
        if 'numbering_patterns' in full_structure:
            patterns = full_structure['numbering_patterns']
            if 'numeric_sub_paragraphs' in patterns:
                self.logger.debug("Polish law numeric sub-paragraph pattern validated")
            if 'section_organization' in patterns:
                self.logger.debug("Polish law section organization pattern validated")
        
        # Validate cross-reference preservation (common in Polish legal documents)
        if 'cross_references' in full_structure:
            cross_refs = full_structure['cross_references']
            if cross_refs:
                self.logger.debug(f"Polish law cross-references preserved: {len(cross_refs)} references")
    
    def _process_polish_law_specific_metadata(self, metadata: Dict[str, Any], 
                                            reconstructed_info: Dict[str, Any]) -> None:
        """
        Process Polish law-specific metadata elements that are unique to Polish legal documents.
        
        This method handles the special metadata elements that are important for Polish law
        citations but don't exist in other legal systems, such as gazette references,
        parliament sessions, and Polish legal organizational structures.
        """
        polish_specifics = {}
        
        # Process section information (unique organizational element in Polish law)
        section_number = metadata.get('section_number', '')
        section_title = metadata.get('section_title', '')
        
        if section_number:
            self.processing_stats['sections_processed'] += 1
            polish_specifics['section_info'] = {
                'number': section_number,
                'title': section_title,
                'has_section': True
            }
            self.logger.debug(f"Processed Polish law section: {section_number} - {section_title}")
        else:
            polish_specifics['section_info'] = {'has_section': False}
        
        # Process gazette reference (critical for Polish law authenticity)
        gazette_ref = metadata.get('gazette_reference', '')
        if gazette_ref:
            self.processing_stats['gazette_references_processed'] += 1
            polish_specifics['gazette_info'] = {
                'reference': gazette_ref,
                'has_gazette_reference': True,
                'authenticity_verified': True  # Presence indicates official publication
            }
            self.logger.debug(f"Processed Polish law gazette reference: {gazette_ref}")
        else:
            polish_specifics['gazette_info'] = {'has_gazette_reference': False}
        
        # Process parliament session information (Polish law-specific provenance)
        parliament_session = metadata.get('parliament_session', '')
        if parliament_session:
            self.processing_stats['parliament_sessions_found'] += 1
            polish_specifics['parliament_info'] = {
                'session': parliament_session,
                'has_parliament_info': True
            }
            self.logger.debug(f"Processed Polish law parliament session: {parliament_session}")
        else:
            polish_specifics['parliament_info'] = {'has_parliament_info': False}
        
        # Process amendment information (important for Polish law evolution tracking)
        amendment_info = metadata.get('amendment_info', '')
        if amendment_info:
            polish_specifics['amendment_info'] = {
                'details': amendment_info,
                'has_amendments': True
            }
            self.logger.debug(f"Processed Polish law amendment info: {amendment_info}")
        else:
            polish_specifics['amendment_info'] = {'has_amendments': False}
        
        # Store all Polish law-specific information
        reconstructed_info['polish_law_specifics'] = polish_specifics
    
    def create_polish_law_processing_hints(self, reconstructed_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create processing hints for content analysis based on reconstructed Polish law metadata.
        
        This method transforms the reconstructed structural information into hints
        that can guide content parsing for Polish legal documents. It demonstrates how metadata 
        processing enables intelligent, guided analysis rather than blind text parsing,
        with special consideration for Polish legal document patterns.
        
        Args:
            reconstructed_info: Reconstructed metadata information for Polish law
            
        Returns:
            Dictionary of processing hints for Polish law content analysis
        """
        if not reconstructed_info['has_enhanced_structure']:
            return {
                'has_hints': False,
                'use_guided_parsing': False,
                'parsing_strategy': 'simple',
                'legal_system': 'polish_law'
            }
        
        quick_indicators = reconstructed_info['quick_indicators']
        polish_specifics = reconstructed_info.get('polish_law_specifics', {})
        
        # Create comprehensive hints based on available Polish law information
        hints = {
            'has_hints': True,
            'use_guided_parsing': True,
            'legal_system': 'polish_law',
            'parsing_strategy': 'enhanced' if reconstructed_info['reconstruction_successful'] else 'indicator_guided',
            
            # Direct indicators for parsing guidance
            'paragraph_count': quick_indicators['paragraph_count'],
            'has_sub_paragraphs': quick_indicators['has_sub_paragraphs'],
            'numbering_style': quick_indicators['numbering_style'],
            'complexity_level': quick_indicators['complexity_level'],
            
            # Polish law-specific parsing guidance
            'has_sections': quick_indicators.get('has_sections', False),
            'section_aware_parsing': polish_specifics.get('section_info', {}).get('has_section', False),
            'expect_polish_terminology': True,  # Always expect Polish legal terms
            
            # Parsing recommendations based on Polish law structure
            'recommended_parser': self._recommend_polish_law_parser_strategy(quick_indicators, polish_specifics),
            'expected_patterns': self._identify_polish_law_expected_patterns(quick_indicators, polish_specifics),
            
            # Full structure availability
            'full_structure_available': reconstructed_info['reconstruction_successful'],
            'reconstruction_method': reconstructed_info['reconstruction_method']
        }
        
        self.logger.debug(f"Created Polish law processing hints: {hints['parsing_strategy']} strategy, "
                        f"parser: {hints['recommended_parser']}, "
                        f"section-aware: {hints['section_aware_parsing']}")
        
        return hints
    
    def _recommend_polish_law_parser_strategy(self, quick_indicators: Dict[str, Any], 
                                            polish_specifics: Dict[str, Any]) -> str:
        """
        Recommend the best parsing strategy based on Polish law structural indicators.
        
        This method analyzes the available metadata to determine which parsing
        approach will be most effective for the specific Polish law document structure.
        """
        complexity = quick_indicators['complexity_level']
        has_sub_paragraphs = quick_indicators['has_sub_paragraphs']
        paragraph_count = quick_indicators['paragraph_count']
        has_sections = quick_indicators.get('has_sections', False)
        
        # Polish law-specific parsing strategy recommendations
        if has_sections and complexity == 'complex' and has_sub_paragraphs:
            return 'polish_law_sophisticated_with_sections_and_sub_paragraphs'
        elif has_sections and has_sub_paragraphs:
            return 'polish_law_guided_with_sections_and_sub_paragraphs'
        elif has_sections:
            return 'polish_law_section_aware_parsing'
        elif complexity == 'complex' and has_sub_paragraphs:
            return 'polish_law_sophisticated_with_sub_paragraphs'
        elif has_sub_paragraphs:
            return 'polish_law_guided_with_sub_paragraphs'
        elif paragraph_count > 1:
            return 'polish_law_multi_paragraph'
        else:
            return 'polish_law_simple_single_paragraph'
    
    def _identify_polish_law_expected_patterns(self, quick_indicators: Dict[str, Any], 
                                             polish_specifics: Dict[str, Any]) -> List[str]:
        """
        Identify expected structural patterns based on Polish law metadata indicators.
        
        This helps the content analyzer know what patterns to look for,
        making parsing more reliable and efficient for Polish legal documents.
        """
        patterns = []
        
        numbering_style = quick_indicators['numbering_style']
        has_sub_paragraphs = quick_indicators['has_sub_paragraphs']
        has_sections = quick_indicators.get('has_sections', False)
        
        # Add pattern expectations based on Polish law metadata
        if has_sections:
            patterns.append('polish_law_section_organization')  # Unique to Polish law
        
        if has_sub_paragraphs:
            if numbering_style == 'number_closing_paren':
                patterns.append('numeric_sub_paragraphs')  # Common in Polish law: 1), 2), 3)
            elif numbering_style == 'alphabetical':
                patterns.append('alphabetical_sub_paragraphs')  # Also used: a), b), c)
            else:
                patterns.append('mixed_sub_paragraphs')
        
        # Add Polish law-specific patterns
        patterns.append('polish_legal_document_structure')
        patterns.append('polish_legal_terminology')  # Expect Polish terms like "ustawa", "artykuł"
        
        if quick_indicators['paragraph_count'] > 1:
            patterns.append('multi_paragraph_polish_article')
        
        # Add patterns based on Polish law-specific metadata
        if polish_specifics.get('gazette_info', {}).get('has_gazette_reference', False):
            patterns.append('gazette_reference_context')
        
        if polish_specifics.get('amendment_info', {}).get('has_amendments', False):
            patterns.append('amendment_aware_parsing')
        
        return patterns
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about Polish law metadata processing operations.
        
        This provides insights into how well the reconstruction process is working
        and helps identify any issues with the flattened metadata approach for Polish law.
        """
        stats = dict(self.processing_stats)
        
        # Calculate success rates for Polish law processing
        if stats['total_metadata_processed'] > 0:
            enhancement_rate = (stats['enhanced_structures_reconstructed'] / stats['total_metadata_processed']) * 100
            stats['enhancement_rate_percent'] = round(enhancement_rate, 1)
            
            section_processing_rate = (stats['sections_processed'] / stats['total_metadata_processed']) * 100
            stats['section_processing_rate_percent'] = round(section_processing_rate, 1)
            
            gazette_processing_rate = (stats['gazette_references_processed'] / stats['total_metadata_processed']) * 100
            stats['gazette_processing_rate_percent'] = round(gazette_processing_rate, 1)
            
            if stats['enhanced_structures_reconstructed'] > 0:
                json_success_rate = (stats['json_deserialization_successes'] / stats['enhanced_structures_reconstructed']) * 100
                stats['json_success_rate_percent'] = round(json_success_rate, 1)
            else:
                stats['json_success_rate_percent'] = 0
        else:
            stats['enhancement_rate_percent'] = 0
            stats['section_processing_rate_percent'] = 0
            stats['gazette_processing_rate_percent'] = 0
            stats['json_success_rate_percent'] = 0
        
        return stats
    
    def log_processing_summary(self) -> None:
        """
        Log a comprehensive summary of all Polish law metadata processing operations.
        
        This provides visibility into how well the metadata reconstruction
        process is working across all processed Polish law documents.
        """
        stats = self.get_processing_statistics()
        
        self.logger.info("=== POLISH LAW METADATA PROCESSING SUMMARY ===")
        self.logger.info(f"Total metadata processed: {stats['total_metadata_processed']}")
        self.logger.info(f"Enhanced structures reconstructed: {stats['enhanced_structures_reconstructed']}")
        self.logger.info(f"Enhancement rate: {stats['enhancement_rate_percent']}%")
        self.logger.info(f"JSON reconstruction successes: {stats['json_deserialization_successes']}")
        self.logger.info(f"JSON success rate: {stats['json_success_rate_percent']}%")
        self.logger.info(f"Fallback to indicators: {stats['fallback_to_indicators']}")
        self.logger.info(f"Processing errors: {stats['processing_errors']}")
        
        # Log Polish law-specific processing statistics
        self.logger.info("Polish law-specific processing metrics:")
        self.logger.info(f"  - Sections processed: {stats['sections_processed']} ({stats['section_processing_rate_percent']}%)")
        self.logger.info(f"  - Gazette references processed: {stats['gazette_references_processed']} ({stats['gazette_processing_rate_percent']}%)")
        self.logger.info(f"  - Parliament sessions found: {stats['parliament_sessions_found']}")


def create_polish_law_metadata_processor(logger: logging.Logger) -> PolishLawMetadataProcessor:
    """
    Factory function to create a configured Polish law metadata processor.
    
    This provides a clean interface for creating processor instances with
    proper dependency injection of the logger.
    """
    return PolishLawMetadataProcessor(logger)