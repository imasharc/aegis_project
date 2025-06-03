"""
Polish Law Citation Builder

This module creates precise Polish legal citations by combining information from all the previous
components in the pipeline. It represents the culmination of your sophisticated architectural
approach applied to Polish law, taking the structural analysis and creating the precise references 
that follow Polish legal citation conventions.

Polish Law Specific Features:
- Section-aware citation formatting (unique to Polish legal structure)
- Gazette reference integration for legal authenticity
- Polish legal numbering pattern recognition (1), 2), 3) vs (a), (b), (c))
- Parliament session and amendment information in citations
- Polish legal citation formatting standards and conventions

The citation builder demonstrates how architectural layers build upon each other:
- Polish Law Vector Store Connector retrieves documents with metadata
- Polish Law Metadata Processor reconstructs structural information
- Polish Law Content Analyzer identifies precise locations within structure  
- Polish Law Citation Builder synthesizes everything into perfect Polish legal citations

This is where all the sophisticated Polish law metadata preservation and analysis pays off,
creating citations like "Article 6, paragraph 1, point 2), Section III: Data Processing Principles
(Polish Data Protection Law, Dz.U. 2018 poz. 1000)" that provide exact legal reference points
following Polish legal citation standards.
"""

import logging
from typing import Dict, List, Any, Optional


class PolishLawCitationBuilder:
    """
    Creates precise Polish law citations using structural analysis and metadata reconstruction.
    
    This class represents the sophisticated endpoint of your refactored architecture applied
    to Polish law. It takes the structural analysis from the content analyzer and the reconstructed
    metadata from the metadata processor to create the most precise citations possible while
    following Polish legal citation conventions.
    
    The citation builder demonstrates how your flattened metadata approach enables
    precise functionality while maintaining vector database compatibility. All the
    complex work done by previous components enables this class to create citations
    that would be impossible with basic document retrieval approaches, specifically
    adapted for Polish legal document structure and citation requirements.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law citation builder.
        
        Args:
            logger: Configured logger for tracking citation building operations
        """
        self.logger = logger
        self.logger.info("Polish Law Citation Builder initialized")
        
        # Track citation building statistics across all operations with Polish law specifics
        self.citation_stats = {
            'total_citations_built': 0,
            'maximum_precision_citations': 0,  # Article + section + paragraph + sub-paragraph
            'section_precision_citations': 0,  # Article + section + paragraph
            'paragraph_precision_citations': 0,  # Article + paragraph
            'article_precision_citations': 0,    # Article only
            'fallback_citations': 0,
            'citation_errors': 0,
            'precision_levels': {
                'maximum': 0,      # Full structural detail
                'section': 0,      # Section-aware (unique to Polish law)
                'paragraph': 0,    # Paragraph-level
                'article': 0,      # Article-level only
                'fallback': 0      # Basic fallback
            },
            'polish_law_features': {
                'gazette_references_included': 0,
                'section_citations_created': 0,
                'parliament_info_included': 0,
                'amendment_context_included': 0
            }
        }
    
    def create_precise_citation(self, reconstructed_metadata: Dict[str, Any], 
                               content: str, quote: str, 
                               content_analysis: Dict[str, Any]) -> str:
        """
        Create a precise Polish law citation using all available structural information.
        
        This method represents the culmination of your sophisticated approach applied to Polish law.
        It combines the reconstructed metadata from the metadata processor with the structural analysis
        from the content analyzer to create the most precise citation possible while following
        Polish legal citation conventions and organizational patterns.
        
        The method demonstrates how architectural layers work together to achieve
        sophisticated functionality that would be impossible with any single component,
        specifically adapted for Polish legal document citation requirements.
        
        Args:
            reconstructed_metadata: Metadata reconstructed by the Polish law metadata processor
            content: Original document content for verification
            quote: Specific quote to locate within the structure
            content_analysis: Structure analysis from the Polish law content analyzer
            
        Returns:
            Precise citation string with maximum available detail following Polish law conventions
        """
        self.logger.info("Creating precise Polish law citation using comprehensive structural analysis")
        self.citation_stats['total_citations_built'] += 1
        
        try:
            # Step 1: Attempt to locate the quote within the analyzed Polish law structure
            quote_location = self._locate_quote_with_polish_law_analysis(quote, content_analysis)
            
            # Step 2: Build the citation using all available Polish law information
            citation_parts = self._build_polish_law_citation_components(
                reconstructed_metadata, quote_location, content_analysis
            )
            
            # Step 3: Assemble the final citation with appropriate precision level and Polish law formatting
            final_citation = self._assemble_final_polish_law_citation(citation_parts, quote_location)
            
            # Step 4: Log the citation creation success with precision analysis
            self._log_polish_law_citation_success(final_citation, quote_location, citation_parts)
            
            return final_citation
            
        except Exception as e:
            self.citation_stats['citation_errors'] += 1
            self.logger.error(f"Error creating precise Polish law citation: {e}")
            # Return fallback citation to ensure system continues operating
            return self._create_polish_law_fallback_citation(reconstructed_metadata)
    
    def _locate_quote_with_polish_law_analysis(self, quote: str, content_analysis: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate the quote using the sophisticated Polish law content analysis results.
        
        This method leverages the structural analysis performed by the Polish law content analyzer
        to find the exact location of a quote within the document structure, including Polish law-specific
        organizational elements like sections. This is where the guided parsing approach pays off 
        with precise location identification adapted for Polish legal document patterns.
        """
        if not content_analysis.get('parsing_successful', False):
            self.logger.debug("Polish law content analysis was not successful - cannot locate quote precisely")
            return None
        
        try:
            # Use the content analyzer's structural map to find the quote in Polish law structure
            paragraphs = content_analysis.get('paragraphs', {})
            sections = content_analysis.get('sections', {})  # Unique to Polish law
            
            if not paragraphs and not sections:
                self.logger.debug("No paragraph or section structure available for quote location")
                return None
            
            # Search through the analyzed Polish law structure with section awareness
            location_result = self._search_analyzed_polish_law_structure(quote, paragraphs, sections)
            
            if location_result:
                self.logger.debug(f"Quote located using Polish law content analysis: {location_result}")
            else:
                self.logger.debug("Quote not found in analyzed Polish law structure")
            
            return location_result
            
        except Exception as e:
            self.logger.warning(f"Error locating quote with Polish law analysis: {e}")
            return None
    
    def _search_analyzed_polish_law_structure(self, quote: str, paragraphs: Dict[str, Any], 
                                            sections: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search the analyzed Polish law structure for the specific quote.
        
        This method performs hierarchical search through the structure identified
        by the content analyzer, providing maximum precision when possible and
        taking advantage of Polish law's unique section organization.
        """
        # Normalize quote for consistent matching
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.debug("Quote too short for reliable structure-based location")
            return None
        
        # First search within sections if they exist (unique to Polish law structure)
        if sections:
            for section_num, section_data in sections.items():
                section_paragraphs = section_data.get('paragraphs', {})
                
                # Search within section's paragraphs
                for para_num, para_data in section_paragraphs.items():
                    para_text = ' '.join(para_data.get('full_text', '').split()).lower()
                    
                    if clean_quote in para_text:
                        # Check sub-paragraphs for maximum precision
                        sub_paragraphs = para_data.get('sub_paragraphs', {})
                        
                        for sub_para_key, sub_para_data in sub_paragraphs.items():
                            sub_para_text = ' '.join(sub_para_data.get('text', '').split()).lower()
                            
                            if clean_quote in sub_para_text:
                                # Maximum precision: section + paragraph + sub-paragraph
                                return {
                                    'section': section_num,
                                    'paragraph': para_num,
                                    'sub_paragraph': sub_para_key,
                                    'location_type': 'section_paragraph_sub_paragraph',
                                    'confidence': 'maximum',
                                    'numbering_style': sub_para_data.get('numbering_style', 'unknown')
                                }
                        
                        # High precision: section + paragraph
                        return {
                            'section': section_num,
                            'paragraph': para_num,
                            'sub_paragraph': None,
                            'location_type': 'section_paragraph',
                            'confidence': 'high'
                        }
        
        # Search through paragraphs not in sections or when no sections exist
        for para_num, para_data in paragraphs.items():
            # Skip if this paragraph is already processed in a section
            if para_data.get('section'):
                continue
            
            para_text = ' '.join(para_data.get('full_text', '').split()).lower()
            
            if clean_quote in para_text:
                # Check sub-paragraphs for maximum precision
                sub_paragraphs = para_data.get('sub_paragraphs', {})
                
                for sub_para_key, sub_para_data in sub_paragraphs.items():
                    sub_para_text = ' '.join(sub_para_data.get('text', '').split()).lower()
                    
                    if clean_quote in sub_para_text:
                        # Medium-high precision: paragraph + sub-paragraph
                        return {
                            'section': None,
                            'paragraph': para_num,
                            'sub_paragraph': sub_para_key,
                            'location_type': 'paragraph_sub_paragraph',
                            'confidence': 'medium-high',
                            'numbering_style': sub_para_data.get('numbering_style', 'unknown')
                        }
                
                # Medium precision: paragraph only
                return {
                    'section': None,
                    'paragraph': para_num,
                    'sub_paragraph': None,
                    'location_type': 'main_paragraph',
                    'confidence': 'medium'
                }
        
        return None
    
    def _build_polish_law_citation_components(self, reconstructed_metadata: Dict[str, Any], 
                                            quote_location: Optional[Dict[str, str]],
                                            content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build all the components needed for a comprehensive Polish law citation.
        
        This method extracts and organizes all the available structural information
        to prepare for creating the most detailed citation possible while respecting
        Polish legal citation conventions and organizational patterns.
        """
        components = {
            # Basic Polish law document identifiers
            'article_number': reconstructed_metadata.get('article_number', ''),
            'chapter_number': reconstructed_metadata.get('chapter_number', ''),
            'chapter_title': reconstructed_metadata.get('chapter_title', ''),
            'section_number': reconstructed_metadata.get('section_number', ''),  # Unique to Polish law
            'section_title': reconstructed_metadata.get('section_title', ''),    # Polish law organizational element
            'article_title': reconstructed_metadata.get('article_title', ''),
            
            # Polish law-specific metadata for citations
            'law_type': reconstructed_metadata.get('law_type', 'national_law'),
            'jurisdiction': reconstructed_metadata.get('jurisdiction', 'Poland'),
            'gazette_reference': reconstructed_metadata.get('gazette_reference', ''),    # Critical for authenticity
            'parliament_session': reconstructed_metadata.get('parliament_session', ''), # Polish law provenance
            'amendment_info': reconstructed_metadata.get('amendment_info', ''),         # Legal evolution context
            'effective_date': reconstructed_metadata.get('effective_date', ''),
            
            # Quote location and analysis information
            'has_quote_location': quote_location is not None,
            'quote_location': quote_location,
            'analysis_method': content_analysis.get('analysis_method', 'unknown'),
            'patterns_detected': content_analysis.get('patterns_detected', []),
            'polish_law_features': content_analysis.get('polish_law_features', {}),
            
            # Polish law-specific analysis results
            'sections_found_in_analysis': content_analysis.get('polish_law_features', {}).get('sections_found', 0),
            'polish_terms_detected': content_analysis.get('polish_law_features', {}).get('polish_terms_detected', []),
            'gazette_refs_in_content': content_analysis.get('polish_law_features', {}).get('gazette_references', [])
        }
        
        # Determine the highest precision level possible for Polish law
        if quote_location:
            if quote_location.get('section') and quote_location.get('sub_paragraph'):
                components['precision_level'] = 'maximum'  # Section + paragraph + sub-paragraph
            elif quote_location.get('section'):
                components['precision_level'] = 'section'  # Section + paragraph
            elif quote_location.get('sub_paragraph'):
                components['precision_level'] = 'paragraph'  # Paragraph + sub-paragraph
            elif quote_location.get('paragraph'):
                components['precision_level'] = 'paragraph'  # Paragraph only
            else:
                components['precision_level'] = 'article'
        else:
            components['precision_level'] = 'article'
        
        self.logger.debug(f"Built Polish law citation components with {components['precision_level']} precision level")
        
        return components
    
    def _assemble_final_polish_law_citation(self, components: Dict[str, Any], 
                                          quote_location: Optional[Dict[str, str]]) -> str:
        """
        Assemble the final citation string using all available components with Polish law formatting.
        
        This method creates the precise citation format following Polish legal citation conventions,
        combining article information with structural details when available. The method demonstrates 
        how all the sophisticated processing enables precise output while respecting Polish legal 
        citation standards and organizational patterns.
        """
        citation_parts = []
        precision_level = components['precision_level']
        
        # Build the core article reference following Polish law conventions
        article_num = components['article_number']
        if article_num:
            if quote_location:
                # Add structural information based on location analysis
                section_num = quote_location.get('section')
                para_num = quote_location.get('paragraph')
                sub_para_key = quote_location.get('sub_paragraph')
                
                if section_num and para_num and sub_para_key and precision_level == 'maximum':
                    # Maximum precision: Article, section, paragraph, and sub-paragraph
                    section_title = components.get('section_title', '')
                    if section_title:
                        citation_parts.append(f"Article {article_num}, Section {section_num}: {section_title}, "
                                            f"paragraph {para_num}, point {sub_para_key})")
                    else:
                        citation_parts.append(f"Article {article_num}, Section {section_num}, "
                                            f"paragraph {para_num}, point {sub_para_key})")
                    
                    self.citation_stats['maximum_precision_citations'] += 1
                    self.citation_stats['precision_levels']['maximum'] += 1
                    self.citation_stats['polish_law_features']['section_citations_created'] += 1
                    self.logger.debug(f"Created maximum precision Polish law citation with section")
                    
                elif section_num and para_num and precision_level in ['section', 'maximum']:
                    # Section precision: Article, section, and paragraph
                    section_title = components.get('section_title', '')
                    if section_title:
                        citation_parts.append(f"Article {article_num}, Section {section_num}: {section_title}, "
                                            f"paragraph {para_num}")
                    else:
                        citation_parts.append(f"Article {article_num}, Section {section_num}, paragraph {para_num}")
                    
                    self.citation_stats['section_precision_citations'] += 1
                    self.citation_stats['precision_levels']['section'] += 1
                    self.citation_stats['polish_law_features']['section_citations_created'] += 1
                    self.logger.debug(f"Created section precision Polish law citation")
                    
                elif para_num and sub_para_key and precision_level in ['paragraph', 'maximum']:
                    # Paragraph precision with sub-paragraph: Article, paragraph, and sub-paragraph
                    citation_parts.append(f"Article {article_num}, paragraph {para_num}, point {sub_para_key})")
                    self.citation_stats['paragraph_precision_citations'] += 1
                    self.citation_stats['precision_levels']['paragraph'] += 1
                    self.logger.debug(f"Created paragraph precision Polish law citation with sub-paragraph")
                    
                elif para_num:
                    # Paragraph precision: Article and paragraph
                    citation_parts.append(f"Article {article_num}, paragraph {para_num}")
                    self.citation_stats['paragraph_precision_citations'] += 1
                    self.citation_stats['precision_levels']['paragraph'] += 1
                    self.logger.debug(f"Created paragraph precision Polish law citation")
                    
                else:
                    # Article precision only
                    citation_parts.append(f"Article {article_num}")
                    self.citation_stats['article_precision_citations'] += 1
                    self.citation_stats['precision_levels']['article'] += 1
            else:
                # No location information - article only
                citation_parts.append(f"Article {article_num}")
                self.citation_stats['article_precision_citations'] += 1
                self.citation_stats['precision_levels']['article'] += 1
        
        # Add Polish law document context
        law_context = self._build_polish_law_context(components)
        if law_context:
            citation_parts.append(f"({law_context})")
        
        # Add gazette reference for legal authenticity (critical for Polish law)
        gazette_ref = components.get('gazette_reference', '')
        if gazette_ref:
            citation_parts.append(f"[Dz.U. {gazette_ref}]")
            self.citation_stats['polish_law_features']['gazette_references_included'] += 1
            self.logger.debug(f"Added gazette reference to Polish law citation: {gazette_ref}")
        
        # Combine all parts into the final citation
        if citation_parts:
            final_citation = " ".join(citation_parts)
        else:
            final_citation = f"Polish Data Protection Law - Article {article_num or 'Unknown'}"
        
        return final_citation
    
    def _build_polish_law_context(self, components: Dict[str, Any]) -> str:
        """
        Build the legal context string for Polish law citations.
        
        This creates the contextual information that identifies the specific Polish law
        and provides additional context for legal research and verification.
        """
        context_parts = []
        
        # Add basic law identification
        context_parts.append("Polish Data Protection Law")
        
        # Add chapter information if available
        chapter_num = components.get('chapter_number', '')
        chapter_title = components.get('chapter_title', '')
        
        if chapter_num and chapter_title:
            context_parts.append(f"Chapter {chapter_num}: {chapter_title}")
        
        # Add effective date if available
        effective_date = components.get('effective_date', '')
        if effective_date:
            context_parts.append(f"effective {effective_date}")
        
        # Add parliament session information if available (Polish law-specific)
        parliament_session = components.get('parliament_session', '')
        if parliament_session:
            context_parts.append(f"Parliament Session {parliament_session}")
            self.citation_stats['polish_law_features']['parliament_info_included'] += 1
        
        # Add amendment context if available
        amendment_info = components.get('amendment_info', '')
        if amendment_info:
            context_parts.append(f"as amended: {amendment_info}")
            self.citation_stats['polish_law_features']['amendment_context_included'] += 1
        
        return ", ".join(context_parts) if context_parts else ""
    
    def _create_polish_law_fallback_citation(self, reconstructed_metadata: Dict[str, Any]) -> str:
        """
        Create a fallback citation when sophisticated processing fails for Polish law.
        
        This method ensures that the system continues to function even when
        the advanced features encounter issues, demonstrating graceful degradation
        while maintaining basic functionality and Polish law citation conventions.
        """
        self.citation_stats['fallback_citations'] += 1
        self.citation_stats['precision_levels']['fallback'] += 1
        
        article_num = reconstructed_metadata.get('article_number', 'Unknown')
        chapter_num = reconstructed_metadata.get('chapter_number', '')
        chapter_title = reconstructed_metadata.get('chapter_title', '')
        section_num = reconstructed_metadata.get('section_number', '')
        gazette_ref = reconstructed_metadata.get('gazette_reference', '')
        
        fallback_citation = f"Polish Data Protection Law - Article {article_num}"
        
        # Add section if available (unique to Polish law)
        if section_num:
            section_title = reconstructed_metadata.get('section_title', '')
            if section_title:
                fallback_citation += f", Section {section_num}: {section_title}"
            else:
                fallback_citation += f", Section {section_num}"
        
        # Add chapter context
        if chapter_num and chapter_title:
            fallback_citation += f" (Chapter {chapter_num}: {chapter_title})"
        
        # Add gazette reference if available
        if gazette_ref:
            fallback_citation += f" [Dz.U. {gazette_ref}]"
        
        self.logger.info(f"Created Polish law fallback citation: {fallback_citation}")
        
        return fallback_citation
    
    def _log_polish_law_citation_success(self, final_citation: str, quote_location: Optional[Dict[str, str]], 
                                       components: Dict[str, Any]) -> None:
        """
        Log successful citation creation with detailed information for monitoring.
        
        This logging provides visibility into how well the citation building process
        is working for Polish law documents and helps identify the precision levels being 
        achieved across different types of documents and queries.
        """
        precision_level = components['precision_level']
        analysis_method = components['analysis_method']
        
        self.logger.info(f"Successfully created {precision_level} precision Polish law citation: {final_citation}")
        self.logger.debug(f"Citation built using {analysis_method} analysis method")
        
        if quote_location:
            location_type = quote_location['location_type']
            confidence = quote_location['confidence']
            section = quote_location.get('section')
            
            if section:
                self.logger.debug(f"Quote location includes Polish law section: {section} with {confidence} confidence")
            else:
                self.logger.debug(f"Quote location: {location_type} with {confidence} confidence")
        
        # Log Polish law-specific features that contributed to citation
        polish_features = components.get('polish_law_features', {})
        if polish_features.get('sections_found', 0) > 0:
            self.logger.debug(f"Polish law sections identified: {polish_features['sections_found']}")
        
        # Log patterns that contributed to success
        patterns = components.get('patterns_detected', [])
        if patterns:
            polish_patterns = [p for p in patterns if 'polish' in p.lower() or 'section' in p.lower()]
            if polish_patterns:
                self.logger.debug(f"Polish law-specific patterns detected: {polish_patterns}")
    
    def create_basic_citation_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Create a basic citation directly from metadata when full processing is not available.
        
        This method provides a simplified path for creating citations when the full
        sophisticated pipeline is not needed or available, adapted for Polish law
        citation conventions and organizational patterns.
        """
        self.citation_stats['total_citations_built'] += 1
        self.citation_stats['article_precision_citations'] += 1
        self.citation_stats['precision_levels']['article'] += 1
        
        article_num = metadata.get('article_number', 'Unknown')
        chapter_num = metadata.get('chapter_number', '')
        chapter_title = metadata.get('chapter_title', '')
        section_num = metadata.get('section_number', '')
        section_title = metadata.get('section_title', '')
        gazette_ref = metadata.get('gazette_reference', '')
        
        basic_citation = f"Polish Data Protection Law - Article {article_num}"
        
        # Add section information if available (unique to Polish law)
        if section_num:
            if section_title:
                basic_citation += f", Section {section_num}: {section_title}"
            else:
                basic_citation += f", Section {section_num}"
        
        # Add chapter context
        if chapter_num and chapter_title:
            basic_citation += f" (Chapter {chapter_num}: {chapter_title})"
        
        # Add gazette reference for authenticity
        if gazette_ref:
            basic_citation += f" [Dz.U. {gazette_ref}]"
        
        self.logger.debug(f"Created basic Polish law citation from metadata: {basic_citation}")
        
        return basic_citation
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about Polish law citation building operations.
        
        Returns:
            Dictionary containing detailed citation building statistics and precision metrics for Polish law
        """
        stats = dict(self.citation_stats)
        
        # Calculate precision distribution percentages
        if stats['total_citations_built'] > 0:
            for precision_type in ['maximum', 'section', 'paragraph', 'article', 'fallback']:
                count = stats['precision_levels'][precision_type]
                percentage = (count / stats['total_citations_built']) * 100
                stats[f'{precision_type}_precision_percentage'] = round(percentage, 1)
            
            # Calculate overall precision score (weighted by precision level) with Polish law adjustments
            weights = {'maximum': 4, 'section': 3, 'paragraph': 2, 'article': 1, 'fallback': 0}
            total_weighted_score = sum(stats['precision_levels'][level] * weight 
                                     for level, weight in weights.items())
            max_possible_score = stats['total_citations_built'] * 4
            
            if max_possible_score > 0:
                overall_precision_score = (total_weighted_score / max_possible_score) * 100
                stats['overall_precision_score'] = round(overall_precision_score, 1)
            else:
                stats['overall_precision_score'] = 0
                
            # Calculate Polish law-specific enhancement rates
            polish_features = stats['polish_law_features']
            stats['gazette_reference_rate'] = round((polish_features['gazette_references_included'] / stats['total_citations_built']) * 100, 1)
            stats['section_citation_rate'] = round((polish_features['section_citations_created'] / stats['total_citations_built']) * 100, 1)
        
        return stats
    
    def log_citation_summary(self) -> None:
        """
        Log a comprehensive summary of all Polish law citation building operations.
        
        This provides insights into the effectiveness of the sophisticated citation
        system for Polish law documents and helps identify patterns in precision achievement 
        across different document types and analysis scenarios.
        """
        stats = self.get_citation_statistics()
        
        self.logger.info("=== POLISH LAW CITATION BUILDING SUMMARY ===")
        self.logger.info(f"Total citations built: {stats['total_citations_built']}")
        self.logger.info(f"Maximum precision citations: {stats['maximum_precision_citations']} ({stats.get('maximum_precision_percentage', 0)}%)")
        self.logger.info(f"Section precision citations: {stats['section_precision_citations']} ({stats.get('section_precision_percentage', 0)}%)")
        self.logger.info(f"Paragraph precision citations: {stats['paragraph_precision_citations']} ({stats.get('paragraph_precision_percentage', 0)}%)")
        self.logger.info(f"Article precision citations: {stats['article_precision_citations']} ({stats.get('article_precision_percentage', 0)}%)")
        self.logger.info(f"Fallback citations: {stats['fallback_citations']} ({stats.get('fallback_precision_percentage', 0)}%)")
        self.logger.info(f"Overall precision score: {stats.get('overall_precision_score', 0)}%")
        self.logger.info(f"Citation errors: {stats['citation_errors']}")
        
        # Log Polish law-specific metrics
        self.logger.info("Polish law-specific citation features:")
        self.logger.info(f"  - Gazette references included: {stats['polish_law_features']['gazette_references_included']} ({stats.get('gazette_reference_rate', 0)}%)")
        self.logger.info(f"  - Section-aware citations: {stats['polish_law_features']['section_citations_created']} ({stats.get('section_citation_rate', 0)}%)")
        self.logger.info(f"  - Parliament info included: {stats['polish_law_features']['parliament_info_included']}")
        self.logger.info(f"  - Amendment context included: {stats['polish_law_features']['amendment_context_included']}")
        
        # Provide interpretation of the precision score
        precision_score = stats.get('overall_precision_score', 0)
        if precision_score >= 80:
            self.logger.info("Excellent Polish law citation precision - sophisticated analysis working optimally")
        elif precision_score >= 60:
            self.logger.info("Good Polish law citation precision - most documents analyzed successfully")
        elif precision_score >= 40:
            self.logger.info("Moderate Polish law citation precision - consider optimizing metadata quality")
        else:
            self.logger.info("Low Polish law citation precision - review processing pipeline and metadata structure")


def create_polish_law_citation_builder(logger: logging.Logger) -> PolishLawCitationBuilder:
    """
    Factory function to create a configured Polish law citation builder.
    
    This provides a clean interface for creating citation builder instances with
    proper dependency injection of the logger.
    """
    return PolishLawCitationBuilder(logger)