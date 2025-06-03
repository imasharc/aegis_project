"""
GDPR Citation Builder

This module creates precise legal citations by combining information from all the previous
components in the pipeline. It represents the culmination of your sophisticated architectural
approach, taking the structural analysis and creating the precise references that make your
system stand out.

The citation builder demonstrates how architectural layers build upon each other:
- Vector Store Connector retrieves documents with metadata
- Metadata Processor reconstructs structural information
- Content Analyzer identifies precise locations within structure  
- Citation Builder synthesizes everything into perfect citations

This is where all the sophisticated metadata preservation and analysis pays off,
creating citations like "Article 6, paragraph 1(a) (Chapter II: Principles)"
that provide exact legal reference points.
"""

import logging
from typing import Dict, List, Any, Optional


class GDPRCitationBuilder:
    """
    Creates precise GDPR citations using structural analysis and metadata reconstruction.
    
    This class represents the sophisticated endpoint of your refactored architecture.
    It takes the structural analysis from the content analyzer and the reconstructed
    metadata from the metadata processor to create the most precise citations possible.
    
    The citation builder demonstrates how your flattened metadata approach enables
    precise functionality while maintaining vector database compatibility. All the
    complex work done by previous components enables this class to create citations
    that would be impossible with basic document retrieval approaches.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the GDPR citation builder.
        
        Args:
            logger: Configured logger for tracking citation building operations
        """
        self.logger = logger
        self.logger.info("GDPR Citation Builder initialized")
        
        # Track citation building statistics across all operations
        self.citation_stats = {
            'total_citations_built': 0,
            'maximum_precision_citations': 0,  # Article + paragraph + sub-paragraph
            'paragraph_precision_citations': 0,  # Article + paragraph
            'article_precision_citations': 0,    # Article only
            'fallback_citations': 0,
            'citation_errors': 0,
            'precision_levels': {
                'maximum': 0,
                'paragraph': 0, 
                'article': 0,
                'fallback': 0
            }
        }
    
    def create_precise_citation(self, reconstructed_metadata: Dict[str, Any], 
                               content: str, quote: str, 
                               content_analysis: Dict[str, Any]) -> str:
        """
        Create a precise GDPR citation using all available structural information.
        
        This method represents the culmination of your sophisticated approach. It combines
        the reconstructed metadata from the metadata processor with the structural analysis
        from the content analyzer to create the most precise citation possible.
        
        The method demonstrates how architectural layers work together to achieve
        sophisticated functionality that would be impossible with any single component.
        
        Args:
            reconstructed_metadata: Metadata reconstructed by the metadata processor
            content: Original document content for verification
            quote: Specific quote to locate within the structure
            content_analysis: Structure analysis from the content analyzer
            
        Returns:
            Precise citation string with maximum available detail
        """
        self.logger.info("Creating precise GDPR citation using comprehensive structural analysis")
        self.citation_stats['total_citations_built'] += 1
        
        try:
            # Step 1: Attempt to locate the quote within the analyzed structure
            quote_location = self._locate_quote_with_analysis(quote, content_analysis)
            
            # Step 2: Build the citation using all available information
            citation_parts = self._build_citation_components(
                reconstructed_metadata, quote_location, content_analysis
            )
            
            # Step 3: Assemble the final citation with appropriate precision level
            final_citation = self._assemble_final_citation(citation_parts, quote_location)
            
            # Step 4: Log the citation creation success with precision analysis
            self._log_citation_success(final_citation, quote_location, citation_parts)
            
            return final_citation
            
        except Exception as e:
            self.citation_stats['citation_errors'] += 1
            self.logger.error(f"Error creating precise GDPR citation: {e}")
            # Return fallback citation to ensure system continues operating
            return self._create_fallback_citation(reconstructed_metadata)
    
    def _locate_quote_with_analysis(self, quote: str, content_analysis: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate the quote using the sophisticated content analysis results.
        
        This method leverages the structural analysis performed by the content analyzer
        to find the exact location of a quote within the document structure. This is
        where the guided parsing approach pays off with precise location identification.
        """
        if not content_analysis.get('parsing_successful', False):
            self.logger.debug("Content analysis was not successful - cannot locate quote precisely")
            return None
        
        try:
            # Use the content analyzer's structural map to find the quote
            paragraphs = content_analysis.get('paragraphs', {})
            
            if not paragraphs:
                self.logger.debug("No paragraph structure available for quote location")
                return None
            
            # Search through the analyzed structure
            location_result = self._search_analyzed_structure(quote, paragraphs)
            
            if location_result:
                self.logger.debug(f"Quote located using content analysis: {location_result}")
            else:
                self.logger.debug("Quote not found in analyzed structure")
            
            return location_result
            
        except Exception as e:
            self.logger.warning(f"Error locating quote with analysis: {e}")
            return None
    
    def _search_analyzed_structure(self, quote: str, paragraphs: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search the analyzed paragraph structure for the specific quote.
        
        This method performs hierarchical search through the structure identified
        by the content analyzer, providing maximum precision when possible.
        """
        # Normalize quote for consistent matching
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.debug("Quote too short for reliable structure-based location")
            return None
        
        # Search through analyzed paragraphs
        for para_num, para_data in paragraphs.items():
            # Check main paragraph text
            para_text = ' '.join(para_data.get('full_text', '').split()).lower()
            
            if clean_quote in para_text:
                # Check sub-paragraphs for maximum precision
                sub_paragraphs = para_data.get('sub_paragraphs', {})
                
                for sub_para_key, sub_para_data in sub_paragraphs.items():
                    sub_para_text = ' '.join(sub_para_data.get('text', '').split()).lower()
                    
                    if clean_quote in sub_para_text:
                        # Maximum precision: found in specific sub-paragraph
                        return {
                            'paragraph': para_num,
                            'sub_paragraph': sub_para_key,
                            'location_type': 'sub_paragraph',
                            'confidence': 'high',
                            'numbering_style': sub_para_data.get('numbering_style', 'unknown')
                        }
                
                # Medium precision: found in main paragraph
                return {
                    'paragraph': para_num,
                    'sub_paragraph': None,
                    'location_type': 'main_paragraph',
                    'confidence': 'medium'
                }
        
        return None
    
    def _build_citation_components(self, reconstructed_metadata: Dict[str, Any], 
                                  quote_location: Optional[Dict[str, str]],
                                  content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build all the components needed for a comprehensive citation.
        
        This method extracts and organizes all the available structural information
        to prepare for creating the most detailed citation possible. It demonstrates
        how the metadata reconstruction enables sophisticated citation creation.
        """
        components = {
            'article_number': reconstructed_metadata.get('article_number', ''),
            'chapter_number': reconstructed_metadata.get('chapter_number', ''),
            'chapter_title': reconstructed_metadata.get('chapter_title', ''),
            'section_number': reconstructed_metadata.get('section_number', ''),
            'section_title': reconstructed_metadata.get('section_title', ''),
            'article_title': reconstructed_metadata.get('article_title', ''),
            'has_quote_location': quote_location is not None,
            'quote_location': quote_location,
            'analysis_method': content_analysis.get('analysis_method', 'unknown'),
            'patterns_detected': content_analysis.get('patterns_detected', [])
        }
        
        # Determine the highest precision level possible
        if quote_location:
            if quote_location.get('sub_paragraph'):
                components['precision_level'] = 'maximum'
            elif quote_location.get('paragraph'):
                components['precision_level'] = 'paragraph'
            else:
                components['precision_level'] = 'article'
        else:
            components['precision_level'] = 'article'
        
        self.logger.debug(f"Built citation components with {components['precision_level']} precision level")
        
        return components
    
    def _assemble_final_citation(self, components: Dict[str, Any], 
                                quote_location: Optional[Dict[str, str]]) -> str:
        """
        Assemble the final citation string using all available components.
        
        This method creates the precise citation format that your system is known for,
        combining article information with structural details when available. The
        method demonstrates how all the sophisticated processing enables precise output.
        """
        citation_parts = []
        precision_level = components['precision_level']
        
        # Build the core article reference
        article_num = components['article_number']
        if article_num:
            if quote_location:
                # Add structural information based on location analysis
                para_num = quote_location.get('paragraph')
                sub_para_key = quote_location.get('sub_paragraph')
                
                if sub_para_key and precision_level == 'maximum':
                    # Maximum precision: Article, paragraph, and sub-paragraph
                    citation_parts.append(f"Article {article_num}, paragraph {para_num}({sub_para_key})")
                    self.citation_stats['maximum_precision_citations'] += 1
                    self.citation_stats['precision_levels']['maximum'] += 1
                    self.logger.debug(f"Created maximum precision citation: Article {article_num}, paragraph {para_num}({sub_para_key})")
                    
                elif para_num and precision_level in ['paragraph', 'maximum']:
                    # Paragraph precision: Article and paragraph
                    citation_parts.append(f"Article {article_num}, paragraph {para_num}")
                    self.citation_stats['paragraph_precision_citations'] += 1
                    self.citation_stats['precision_levels']['paragraph'] += 1
                    self.logger.debug(f"Created paragraph precision citation: Article {article_num}, paragraph {para_num}")
                    
                else:
                    # Article precision only
                    citation_parts.append(f"Article {article_num}")
                    self.citation_stats['article_precision_citations'] += 1
                    self.citation_stats['precision_levels']['article'] += 1
                    self.logger.debug(f"Created article precision citation: Article {article_num}")
            else:
                # No location information - article only
                citation_parts.append(f"Article {article_num}")
                self.citation_stats['article_precision_citations'] += 1
                self.citation_stats['precision_levels']['article'] += 1
        
        # Add chapter information for complete context
        chapter_num = components['chapter_number']
        chapter_title = components['chapter_title']
        
        if chapter_num and chapter_title:
            # Convert to Roman numerals for GDPR chapter references
            try:
                chapter_roman = self._convert_to_roman(int(chapter_num))
                chapter_info = f"Chapter {chapter_roman}: {chapter_title}"
            except (ValueError, TypeError):
                chapter_info = f"Chapter {chapter_num}: {chapter_title}"
            
            citation_parts.append(f"({chapter_info})")
            self.logger.debug(f"Added GDPR chapter context: {chapter_info}")
        
        # Combine all parts into the final citation
        final_citation = " ".join(citation_parts) if citation_parts else f"GDPR - Article {article_num or 'Unknown'}"
        
        return final_citation
    
    def _convert_to_roman(self, num: int) -> str:
        """
        Convert an integer to Roman numerals for GDPR chapter references.
        
        GDPR uses Roman numerals for chapter numbering, so this method ensures
        that citations follow the proper legal formatting conventions.
        """
        if num <= 0:
            return str(num)
        
        values = [100, 90, 50, 40, 10, 9, 5, 4, 1]
        numerals = ['C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        result = ''
        
        for i, value in enumerate(values):
            count = num // value
            if count:
                result += numerals[i] * count
                num -= value * count
        
        return result
    
    def _create_fallback_citation(self, reconstructed_metadata: Dict[str, Any]) -> str:
        """
        Create a fallback citation when sophisticated processing fails.
        
        This method ensures that the system continues to function even when
        the advanced features encounter issues, demonstrating graceful degradation
        while maintaining basic functionality.
        """
        self.citation_stats['fallback_citations'] += 1
        self.citation_stats['precision_levels']['fallback'] += 1
        
        article_num = reconstructed_metadata.get('article_number', 'Unknown')
        chapter_num = reconstructed_metadata.get('chapter_number', '')
        chapter_title = reconstructed_metadata.get('chapter_title', '')
        
        fallback_citation = f"GDPR - Article {article_num}"
        
        if chapter_num and chapter_title:
            try:
                chapter_roman = self._convert_to_roman(int(chapter_num))
                fallback_citation += f" (Chapter {chapter_roman}: {chapter_title})"
            except (ValueError, TypeError):
                fallback_citation += f" (Chapter {chapter_num}: {chapter_title})"
        
        self.logger.info(f"Created fallback citation: {fallback_citation}")
        
        return fallback_citation
    
    def _log_citation_success(self, final_citation: str, quote_location: Optional[Dict[str, str]], 
                             components: Dict[str, Any]) -> None:
        """
        Log successful citation creation with detailed information for monitoring.
        
        This logging provides visibility into how well the citation building process
        is working and helps identify the precision levels being achieved across
        different types of documents and queries.
        """
        precision_level = components['precision_level']
        analysis_method = components['analysis_method']
        
        self.logger.info(f"Successfully created {precision_level} precision GDPR citation: {final_citation}")
        self.logger.debug(f"Citation built using {analysis_method} analysis method")
        
        if quote_location:
            location_type = quote_location['location_type']
            confidence = quote_location['confidence']
            self.logger.debug(f"Quote location: {location_type} with {confidence} confidence")
        
        # Log patterns that contributed to success
        patterns = components.get('patterns_detected', [])
        if patterns:
            self.logger.debug(f"Structural patterns detected: {patterns}")
    
    def create_basic_citation_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Create a basic citation directly from metadata when full processing is not available.
        
        This method provides a simplified path for creating citations when the full
        sophisticated pipeline is not needed or available. It demonstrates how the
        architecture supports both advanced and basic use cases.
        """
        self.citation_stats['total_citations_built'] += 1
        self.citation_stats['article_precision_citations'] += 1
        self.citation_stats['precision_levels']['article'] += 1
        
        article_num = metadata.get('article_number', 'Unknown')
        chapter_num = metadata.get('chapter_number', '')
        chapter_title = metadata.get('chapter_title', '')
        
        basic_citation = f"GDPR - Article {article_num}"
        
        if chapter_num and chapter_title:
            try:
                chapter_roman = self._convert_to_roman(int(chapter_num))
                basic_citation += f" (Chapter {chapter_roman}: {chapter_title})"
            except (ValueError, TypeError):
                basic_citation += f" (Chapter {chapter_num}: {chapter_title})"
        
        self.logger.debug(f"Created basic citation from metadata: {basic_citation}")
        
        return basic_citation
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about citation building operations.
        
        Returns:
            Dictionary containing detailed citation building statistics and precision metrics
        """
        stats = dict(self.citation_stats)
        
        # Calculate precision distribution percentages
        if stats['total_citations_built'] > 0:
            for precision_type in ['maximum', 'paragraph', 'article', 'fallback']:
                count = stats['precision_levels'][precision_type]
                percentage = (count / stats['total_citations_built']) * 100
                stats[f'{precision_type}_precision_percentage'] = round(percentage, 1)
            
            # Calculate overall precision score (weighted by precision level)
            weights = {'maximum': 3, 'paragraph': 2, 'article': 1, 'fallback': 0}
            total_weighted_score = sum(stats['precision_levels'][level] * weight 
                                     for level, weight in weights.items())
            max_possible_score = stats['total_citations_built'] * 3
            
            if max_possible_score > 0:
                overall_precision_score = (total_weighted_score / max_possible_score) * 100
                stats['overall_precision_score'] = round(overall_precision_score, 1)
            else:
                stats['overall_precision_score'] = 0
        
        return stats
    
    def log_citation_summary(self) -> None:
        """
        Log a comprehensive summary of all citation building operations.
        
        This provides insights into the effectiveness of the sophisticated citation
        system and helps identify patterns in precision achievement across different
        document types and analysis scenarios.
        """
        stats = self.get_citation_statistics()
        
        self.logger.info("=== GDPR CITATION BUILDING SUMMARY ===")
        self.logger.info(f"Total citations built: {stats['total_citations_built']}")
        self.logger.info(f"Maximum precision citations: {stats['maximum_precision_citations']} ({stats.get('maximum_precision_percentage', 0)}%)")
        self.logger.info(f"Paragraph precision citations: {stats['paragraph_precision_citations']} ({stats.get('paragraph_precision_percentage', 0)}%)")
        self.logger.info(f"Article precision citations: {stats['article_precision_citations']} ({stats.get('article_precision_percentage', 0)}%)")
        self.logger.info(f"Fallback citations: {stats['fallback_citations']} ({stats.get('fallback_precision_percentage', 0)}%)")
        self.logger.info(f"Overall precision score: {stats.get('overall_precision_score', 0)}%")
        self.logger.info(f"Citation errors: {stats['citation_errors']}")
        
        # Provide interpretation of the precision score
        precision_score = stats.get('overall_precision_score', 0)
        if precision_score >= 80:
            self.logger.info("Excellent citation precision - sophisticated analysis working optimally")
        elif precision_score >= 60:
            self.logger.info("Good citation precision - most documents analyzed successfully")
        elif precision_score >= 40:
            self.logger.info("Moderate citation precision - consider optimizing metadata quality")
        else:
            self.logger.info("Low citation precision - review processing pipeline and metadata structure")


def create_gdpr_citation_builder(logger: logging.Logger) -> GDPRCitationBuilder:
    """
    Factory function to create a configured GDPR citation builder.
    
    This provides a clean interface for creating citation builder instances with
    proper dependency injection of the logger.
    """
    return GDPRCitationBuilder(logger)