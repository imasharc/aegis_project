"""
Polish Law Response Parser

This module handles the complex task of parsing LLM responses and converting them into
structured citations using all the sophisticated Polish law components we've built. It serves as
the "orchestration layer" between the LLM's natural language output and the precise
citation system your architecture enables for Polish legal documents.

Polish Law Specific Features:
- Section-aware citation enhancement (unique to Polish legal structure)
- Polish legal terminology validation and processing
- Gazette reference integration for citation authenticity
- Polish legal numbering pattern recognition and enhancement
- Parliament session and amendment context integration

The response parser demonstrates how architectural sophistication enables reliable
AI integration for Polish law. Instead of hoping the LLM gives us perfect output, we use our
sophisticated analysis capabilities to validate and enhance whatever the LLM provides,
specifically adapted for Polish legal document patterns and citation requirements.

This approach shows how good architecture makes AI more reliable rather than just
hoping the AI will be perfect on its own, while respecting Polish legal citation conventions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from .polish_law_metadata_processor import PolishLawMetadataProcessor
from .polish_law_content_analyzer import PolishLawContentAnalyzer
from .polish_law_citation_builder import PolishLawCitationBuilder


class PolishLawResponseParser:
    """
    Parses LLM responses and creates structured citations using sophisticated Polish law analysis.
    
    This class represents the integration layer between AI responses and your precise
    citation system for Polish law. Instead of accepting whatever the LLM produces, it uses all the
    sophisticated components we've built to validate, enhance, and perfect the citations
    while following Polish legal citation conventions and organizational patterns.
    
    The parser demonstrates how architectural sophistication makes AI more reliable for Polish law.
    Your metadata processing, content analysis, and citation building capabilities
    enable this class to take imperfect LLM output and create precise, verifiable citations
    that follow Polish legal standards and include Polish law-specific elements.
    """
    
    def __init__(self, metadata_processor: PolishLawMetadataProcessor,
                 content_analyzer: PolishLawContentAnalyzer,
                 citation_builder: PolishLawCitationBuilder,
                 logger: logging.Logger):
        """
        Initialize the Polish law response parser with all required components.
        
        Args:
            metadata_processor: Configured metadata processor for Polish law structure reconstruction
            content_analyzer: Configured content analyzer for intelligent Polish law parsing
            citation_builder: Configured citation builder for precise Polish legal reference creation
            logger: Configured logger for tracking parsing operations
        """
        self.metadata_processor = metadata_processor
        self.content_analyzer = content_analyzer
        self.citation_builder = citation_builder
        self.logger = logger
        
        self.logger.info("Polish Law Response Parser initialized with sophisticated analysis components")
        
        # Track parsing statistics across all operations with Polish law specifics
        self.parsing_stats = {
            'total_responses_parsed': 0,
            'successful_extractions': 0,
            'enhanced_citations_created': 0,
            'fallback_citations_created': 0,
            'parsing_errors': 0,
            'citation_blocks_processed': 0,
            'citation_blocks_successful': 0,
            'llm_format_compliance_rate': 0.0,
            'polish_law_features': {
                'section_enhanced_citations': 0,
                'gazette_reference_validations': 0,
                'polish_terminology_validated': 0,
                'parliament_context_added': 0
            }
        }
    
    def parse_llm_response_to_citations(self, llm_response: str, 
                                       document_metadata_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse LLM response and create precisely formatted Polish law citations using sophisticated analysis.
        
        This method represents the culmination of your architectural approach applied to Polish law. It takes the
        natural language output from the LLM and uses all your sophisticated components to
        create the most precise citations possible while following Polish legal citation conventions
        and incorporating Polish law-specific elements like sections and gazette references.
        
        Args:
            llm_response: Raw response text from the LLM
            document_metadata_list: List of document metadata from vector store retrieval
            
        Returns:
            List of structured citation dictionaries with enhanced precision for Polish law
        """
        self.logger.info("Starting sophisticated LLM response parsing with enhanced Polish law citation creation")
        self.parsing_stats['total_responses_parsed'] += 1
        
        try:
            # Step 1: Extract citation blocks from LLM response
            citation_blocks = self._extract_citation_blocks(llm_response)
            
            if not citation_blocks:
                self.logger.warning("No citation blocks found in LLM response - creating fallback Polish law citations")
                return self._create_fallback_citations_from_metadata(document_metadata_list)
            
            # Step 2: Process each citation block with sophisticated Polish law analysis
            citations = self._process_polish_law_citation_blocks(citation_blocks, document_metadata_list)
            
            # Step 3: Validate and enhance the final citation set with Polish law specifics
            validated_citations = self._validate_and_enhance_polish_law_citations(citations)
            
            # Step 4: Log parsing success and statistics
            self._log_polish_law_parsing_success(validated_citations, citation_blocks)
            
            return validated_citations
            
        except Exception as e:
            self.parsing_stats['parsing_errors'] += 1
            self.logger.error(f"Error during sophisticated Polish law LLM response parsing: {e}")
            # Return fallback citations to ensure system continues operating
            return self._create_fallback_citations_from_metadata(document_metadata_list)
    
    def _extract_citation_blocks(self, llm_response: str) -> List[Dict[str, str]]:
        """
        Extract structured citation blocks from the LLM response.
        
        This method parses the LLM's natural language response to identify the
        citation information it attempted to provide. The method is robust to
        variations in LLM output formatting while extracting the essential components
        for Polish law citation creation.
        """
        self.logger.debug("Extracting citation blocks from LLM response for Polish law processing")
        
        # Split response into potential citation blocks
        raw_blocks = llm_response.split("CITATION ")[1:]  # Skip everything before first "CITATION"
        self.parsing_stats['citation_blocks_processed'] += len(raw_blocks)
        
        citation_blocks = []
        
        for block_index, block in enumerate(raw_blocks):
            try:
                citation_info = self._parse_single_citation_block(block, block_index)
                
                if citation_info:
                    citation_blocks.append(citation_info)
                    self.parsing_stats['citation_blocks_successful'] += 1
                    self.logger.debug(f"Successfully extracted Polish law citation block {block_index + 1}")
                else:
                    self.logger.warning(f"Failed to extract useful information from Polish law citation block {block_index + 1}")
                    
            except Exception as e:
                self.logger.warning(f"Error processing Polish law citation block {block_index + 1}: {e}")
                continue
        
        # Calculate format compliance rate
        if len(raw_blocks) > 0:
            compliance_rate = (len(citation_blocks) / len(raw_blocks)) * 100
            self.parsing_stats['llm_format_compliance_rate'] = compliance_rate
            self.logger.debug(f"LLM format compliance rate for Polish law: {compliance_rate:.1f}%")
        
        return citation_blocks
    
    def _parse_single_citation_block(self, block: str, block_index: int) -> Optional[Dict[str, str]]:
        """
        Parse a single citation block to extract the essential components.
        
        This method handles the detailed work of extracting article information,
        quotes, and explanations from the LLM's response format. It's designed
        to be robust to variations in how the LLM formats its responses while
        preparing for Polish law-specific enhancements.
        """
        lines = block.strip().split('\n')
        citation_info = {
            'block_index': block_index,
            'article_info': '',
            'quote': '',
            'explanation': '',
            'raw_block': block
        }
        
        # Extract information from each line
        for line in lines:
            line = line.strip()
            
            if line.startswith("- Article:"):
                citation_info['article_info'] = line.replace("- Article:", "").strip()
            elif line.startswith("- Quote:"):
                quote = line.replace("- Quote:", "").strip()
                # Clean up quote formatting (remove quotes if present)
                if quote.startswith('"') and quote.endswith('"'):
                    quote = quote[1:-1]
                citation_info['quote'] = quote
            elif line.startswith("- Explanation:"):
                citation_info['explanation'] = line.replace("- Explanation:", "").strip()
        
        # Validate that we extracted the essential components
        if citation_info['quote'] and citation_info['explanation']:
            self.logger.debug(f"Polish law block {block_index}: extracted quote '{citation_info['quote'][:50]}...' "
                            f"with explanation '{citation_info['explanation'][:50]}...'")
            return citation_info
        else:
            self.logger.debug(f"Polish law block {block_index}: missing essential components (quote or explanation)")
            return None
    
    def _process_polish_law_citation_blocks(self, citation_blocks: List[Dict[str, str]], 
                                          document_metadata_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process citation blocks using sophisticated document matching and Polish law analysis.
        
        This method demonstrates how your architectural approach enables intelligent
        processing for Polish law. Instead of just accepting what the LLM said, we use our sophisticated
        analysis capabilities to find the best matching document and create the most
        precise citation possible while following Polish legal citation conventions.
        """
        self.logger.debug(f"Processing {len(citation_blocks)} citation blocks with sophisticated Polish law analysis")
        
        processed_citations = []
        
        for citation_info in citation_blocks:
            try:
                # Step 1: Find the best matching document using content analysis
                best_match = self._find_best_polish_law_document_match(citation_info, document_metadata_list)
                
                if best_match:
                    # Step 2: Create sophisticated citation using all available Polish law components
                    enhanced_citation = self._create_enhanced_polish_law_citation(citation_info, best_match)
                    
                    if enhanced_citation:
                        processed_citations.append(enhanced_citation)
                        self.parsing_stats['enhanced_citations_created'] += 1
                        self.logger.debug(f"Created enhanced Polish law citation from block {citation_info['block_index']}")
                    else:
                        # Fall back to basic citation if enhancement fails
                        basic_citation = self._create_basic_polish_law_citation(citation_info, best_match)
                        processed_citations.append(basic_citation)
                        self.parsing_stats['fallback_citations_created'] += 1
                        self.logger.debug(f"Created basic Polish law citation from block {citation_info['block_index']}")
                else:
                    # No good document match - create minimal citation
                    minimal_citation = self._create_minimal_polish_law_citation(citation_info)
                    processed_citations.append(minimal_citation)
                    self.parsing_stats['fallback_citations_created'] += 1
                    self.logger.warning(f"No document match for Polish law block {citation_info['block_index']} - created minimal citation")
                    
            except Exception as e:
                self.logger.warning(f"Error processing Polish law citation block {citation_info['block_index']}: {e}")
                # Continue processing other blocks
                continue
        
        return processed_citations
    
    def _find_best_polish_law_document_match(self, citation_info: Dict[str, str], 
                                           document_metadata_list: List[Dict]) -> Optional[Dict]:
        """
        Find the best matching document for a citation using sophisticated analysis.
        
        This method uses content matching and metadata quality assessment to identify
        which document the LLM was likely referring to, with special consideration for
        Polish law-specific features and terminology.
        """
        quote = citation_info['quote']
        
        if not quote or len(quote) < 20:  # Need substantial quote for reliable matching
            self.logger.debug("Quote too short for reliable Polish law document matching")
            return None
        
        best_match = None
        best_score = 0
        
        # Score each document based on content match and metadata quality with Polish law considerations
        for doc_metadata in document_metadata_list:
            score = self._calculate_polish_law_document_match_score(quote, doc_metadata)
            
            if score > best_score:
                best_score = score
                best_match = doc_metadata
        
        # Only return match if score meets minimum threshold
        if best_score >= 10:  # Minimum viable match score
            self.logger.debug(f"Found Polish law document match with score {best_score} for quote '{quote[:50]}...'")
            return best_match
        else:
            self.logger.debug(f"No adequate Polish law document match found (best score: {best_score})")
            return None
    
    def _calculate_polish_law_document_match_score(self, quote: str, doc_metadata: Dict) -> float:
        """
        Calculate a matching score between a quote and a Polish law document.
        
        This scoring system considers both content matching and metadata quality
        to identify the most appropriate document for sophisticated analysis,
        with bonuses for Polish law-specific features.
        """
        score = 0
        content = doc_metadata.get('content', '')
        metadata = doc_metadata.get('metadata', {})
        
        # Content matching (primary factor)
        quote_words = set(quote.lower().split())
        content_words = set(content.lower().split())
        
        if quote_words and content_words:
            # Calculate word overlap
            common_words = quote_words.intersection(content_words)
            word_overlap_rate = len(common_words) / len(quote_words)
            score += word_overlap_rate * 10  # Up to 10 points for word overlap
        
        # Direct substring match (strong indicator)
        if len(quote) > 30 and quote[:30].lower() in content.lower():
            score += 15  # Strong bonus for direct substring match
        
        # Metadata quality bonus (enables better analysis)
        if metadata.get('has_enhanced_structure', False):
            score += 5  # Bonus for enhanced metadata
        
        # Polish law-specific metadata bonuses
        if metadata.get('section_number'):  # Section organization bonus
            score += 3
        if metadata.get('gazette_reference'):  # Gazette reference authenticity bonus
            score += 2
        if metadata.get('jurisdiction') == 'Poland':  # Jurisdiction confirmation
            score += 2
        
        # Polish legal terminology bonus
        polish_terms = ['ustawa', 'artykuł', 'rozdział', 'przepis']
        content_lower = content.lower()
        polish_term_matches = sum(1 for term in polish_terms if term in content_lower)
        if polish_term_matches > 0:
            score += polish_term_matches * 1  # Bonus for Polish legal terminology
        
        # Article-specific matching
        if quote.lower() in content.lower():
            score += 20  # Strong bonus for full quote match
        
        return score
    
    def _create_enhanced_polish_law_citation(self, citation_info: Dict[str, str], 
                                           document_match: Dict) -> Optional[Dict[str, Any]]:
        """
        Create an enhanced citation using all sophisticated Polish law analysis components.
        
        This method demonstrates the power of your architectural approach applied to Polish law. We use
        the metadata processor, content analyzer, and citation builder together
        to create the most precise citation possible from the available information,
        while following Polish legal citation conventions.
        """
        try:
            # Step 1: Reconstruct metadata using the Polish law metadata processor
            reconstructed_metadata = self.metadata_processor.extract_and_reconstruct_metadata(
                document_match['metadata']
            )
            
            # Step 2: Create processing hints for Polish law content analysis
            processing_hints = self.metadata_processor.create_polish_law_processing_hints(reconstructed_metadata)
            
            # Step 3: Analyze content structure using the Polish law content analyzer
            content_analysis = self.content_analyzer.analyze_content_structure(
                document_match['content'], processing_hints
            )
            
            # Step 4: Build precise citation using the Polish law citation builder
            precise_citation = self.citation_builder.create_precise_citation(
                reconstructed_metadata, document_match['content'], 
                citation_info['quote'], content_analysis
            )
            
            # Step 5: Create the structured citation object with Polish law enhancements
            enhanced_citation = {
                'article': precise_citation,
                'quote': citation_info['quote'],
                'explanation': citation_info['explanation'],
                'enhancement_level': 'sophisticated',
                'analysis_metadata': {
                    'reconstruction_successful': reconstructed_metadata.get('reconstruction_successful', False),
                    'analysis_method': content_analysis.get('analysis_method', 'unknown'),
                    'parsing_successful': content_analysis.get('parsing_successful', False),
                    'document_match_quality': 'high',
                    'polish_law_features': {
                        'section_aware': bool(reconstructed_metadata.get('polish_law_specifics', {}).get('section_info', {}).get('has_section', False)),
                        'gazette_reference': bool(reconstructed_metadata.get('polish_law_specifics', {}).get('gazette_info', {}).get('has_gazette_reference', False)),
                        'parliament_info': bool(reconstructed_metadata.get('polish_law_specifics', {}).get('parliament_info', {}).get('has_parliament_info', False))
                    }
                }
            }
            
            # Track Polish law-specific enhancements
            if enhanced_citation['analysis_metadata']['polish_law_features']['section_aware']:
                self.parsing_stats['polish_law_features']['section_enhanced_citations'] += 1
            
            if enhanced_citation['analysis_metadata']['polish_law_features']['gazette_reference']:
                self.parsing_stats['polish_law_features']['gazette_reference_validations'] += 1
            
            if enhanced_citation['analysis_metadata']['polish_law_features']['parliament_info']:
                self.parsing_stats['polish_law_features']['parliament_context_added'] += 1
            
            # Check for Polish terminology validation
            polish_features = content_analysis.get('polish_law_features', {})
            if polish_features.get('polish_terms_detected'):
                self.parsing_stats['polish_law_features']['polish_terminology_validated'] += 1
            
            self.logger.debug(f"Created enhanced Polish law citation: {precise_citation}")
            return enhanced_citation
            
        except Exception as e:
            self.logger.warning(f"Error creating enhanced Polish law citation: {e}")
            return None
    
    def _create_basic_polish_law_citation(self, citation_info: Dict[str, str], 
                                        document_match: Dict) -> Dict[str, Any]:
        """
        Create a basic citation when enhanced processing is not possible for Polish law.
        
        This method provides reliable citation creation using just the basic
        metadata, ensuring the system continues to function even when the
        sophisticated features encounter issues, while still following Polish law conventions.
        """
        # Use the citation builder for basic Polish law citation creation
        basic_citation = self.citation_builder.create_basic_citation_from_metadata(
            document_match['metadata']
        )
        
        return {
            'article': basic_citation,
            'quote': citation_info['quote'],
            'explanation': citation_info['explanation'],
            'enhancement_level': 'basic',
            'analysis_metadata': {
                'document_match_quality': 'medium',
                'polish_law_features': {
                    'section_aware': bool(document_match['metadata'].get('section_number')),
                    'gazette_reference': bool(document_match['metadata'].get('gazette_reference')),
                    'parliament_info': bool(document_match['metadata'].get('parliament_session'))
                }
            }
        }
    
    def _create_minimal_polish_law_citation(self, citation_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Create a minimal citation when no good document match is available for Polish law.
        
        This ensures the system provides some citation even when document
        matching fails, demonstrating graceful degradation while maintaining
        Polish law citation conventions.
        """
        article_info = citation_info.get('article_info', 'Unknown Article')
        
        minimal_citation = f"Polish Data Protection Law - {article_info}" if article_info else "Polish Data Protection Law - Article Unknown"
        
        return {
            'article': minimal_citation,
            'quote': citation_info['quote'],
            'explanation': citation_info['explanation'],
            'enhancement_level': 'minimal',
            'analysis_metadata': {
                'document_match_quality': 'none',
                'polish_law_features': {
                    'section_aware': False,
                    'gazette_reference': False,
                    'parliament_info': False
                }
            }
        }
    
    def _validate_and_enhance_polish_law_citations(self, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and enhance the final citation set for quality and consistency with Polish law standards.
        
        This method performs final quality checks and enhancements to ensure
        the citation set meets the high standards your system is known for,
        with special attention to Polish legal citation requirements.
        """
        self.logger.debug(f"Validating and enhancing {len(citations)} Polish law citations")
        
        validated_citations = []
        
        for citation in citations:
            try:
                # Validate essential components
                if self._validate_citation_components(citation):
                    # Apply final enhancements with Polish law specifics
                    enhanced_citation = self._apply_final_polish_law_enhancements(citation)
                    validated_citations.append(enhanced_citation)
                    self.parsing_stats['successful_extractions'] += 1
                else:
                    self.logger.warning(f"Polish law citation failed validation: {citation.get('article', 'Unknown')}")
                    
            except Exception as e:
                self.logger.warning(f"Error validating Polish law citation: {e}")
                continue
        
        return validated_citations
    
    def _validate_citation_components(self, citation: Dict[str, Any]) -> bool:
        """
        Validate that a citation contains all essential components.
        
        This ensures that every citation meets the minimum quality standards
        before being included in the final result set for Polish law.
        """
        required_fields = ['article', 'quote', 'explanation']
        
        for field in required_fields:
            value = citation.get(field, '')
            if not value or not isinstance(value, str) or len(value.strip()) < 5:
                self.logger.debug(f"Polish law citation validation failed: {field} is missing or too short")
                return False
        
        return True
    
    def _apply_final_polish_law_enhancements(self, citation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply final enhancements to ensure citation quality and consistency with Polish law standards.
        
        This method adds any final touches needed to make the citation meet
        the high standards your system is known for while following Polish legal conventions.
        """
        enhanced_citation = citation.copy()
        
        # Ensure article reference has proper Polish law formatting
        article_ref = enhanced_citation['article']
        if not article_ref.startswith('Polish') and not article_ref.startswith('Article'):
            enhanced_citation['article'] = f"Polish Data Protection Law - {article_ref}"
        
        # Clean up quote formatting
        quote = enhanced_citation['quote'].strip()
        if not quote.endswith('.') and not quote.endswith('!') and not quote.endswith('?'):
            if len(quote) > 50:  # Only add period to substantial quotes
                enhanced_citation['quote'] = quote + '.'
        
        # Add enhancement metadata if not present
        if 'analysis_metadata' not in enhanced_citation:
            enhanced_citation['analysis_metadata'] = {
                'enhancement_level': 'basic',
                'polish_law_features': {
                    'section_aware': False,
                    'gazette_reference': False,
                    'parliament_info': False
                }
            }
        
        # Enhance explanation with Polish law context if not already present
        explanation = enhanced_citation['explanation']
        if 'polish' not in explanation.lower() and 'poland' not in explanation.lower():
            enhanced_citation['explanation'] = f"{explanation} (Polish data protection law context)"
        
        return enhanced_citation
    
    def _create_fallback_citations_from_metadata(self, document_metadata_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create fallback citations when LLM response parsing fails completely for Polish law.
        
        This method ensures the system always provides some citations even when
        the LLM response is unusable, demonstrating robust error handling while
        maintaining Polish law citation conventions.
        """
        self.logger.warning("Creating fallback citations from Polish law document metadata")
        
        fallback_citations = []
        
        # Create citations from the top 3 documents
        for doc_metadata in document_metadata_list[:3]:
            try:
                basic_citation = self.citation_builder.create_basic_citation_from_metadata(
                    doc_metadata['metadata']
                )
                
                content = doc_metadata['content']
                quote = content[:200] + "..." if len(content) > 200 else content
                
                fallback_citation = {
                    'article': basic_citation,
                    'quote': quote,
                    'explanation': 'Retrieved relevant content from this Polish data protection law provision',
                    'enhancement_level': 'fallback',
                    'analysis_metadata': {
                        'document_match_quality': 'automatic',
                        'polish_law_features': {
                            'section_aware': bool(doc_metadata['metadata'].get('section_number')),
                            'gazette_reference': bool(doc_metadata['metadata'].get('gazette_reference')),
                            'parliament_info': bool(doc_metadata['metadata'].get('parliament_session'))
                        }
                    }
                }
                
                fallback_citations.append(fallback_citation)
                self.parsing_stats['fallback_citations_created'] += 1
                
            except Exception as e:
                self.logger.warning(f"Error creating fallback Polish law citation: {e}")
                continue
        
        # Ensure we have at least one citation
        if not fallback_citations:
            fallback_citations.append({
                'article': 'Polish Data Protection Law',
                'quote': 'Multiple relevant provisions found in Polish data protection law',
                'explanation': 'Retrieved content related to Polish data protection requirements',
                'enhancement_level': 'minimal',
                'analysis_metadata': {
                    'document_match_quality': 'none',
                    'polish_law_features': {
                        'section_aware': False,
                        'gazette_reference': False,
                        'parliament_info': False
                    }
                }
            })
        
        return fallback_citations
    
    def _log_polish_law_parsing_success(self, citations: List[Dict[str, Any]], citation_blocks: List[Dict[str, str]]) -> None:
        """
        Log successful parsing with detailed statistics for monitoring and optimization.
        
        This logging provides insights into how well the sophisticated parsing
        approach is working for Polish law and helps identify areas for improvement.
        """
        enhanced_count = sum(1 for cite in citations if cite.get('enhancement_level') == 'sophisticated')
        basic_count = sum(1 for cite in citations if cite.get('enhancement_level') == 'basic')
        section_enhanced = self.parsing_stats['polish_law_features']['section_enhanced_citations']
        gazette_validated = self.parsing_stats['polish_law_features']['gazette_reference_validations']
        
        self.logger.info(f"Successfully parsed LLM response for Polish law: {len(citations)} citations created")
        self.logger.info(f"  - Enhanced citations: {enhanced_count}")
        self.logger.info(f"  - Basic citations: {basic_count}")
        self.logger.info(f"  - Citation blocks processed: {len(citation_blocks)}")
        self.logger.info(f"  - Format compliance rate: {self.parsing_stats['llm_format_compliance_rate']:.1f}%")
        self.logger.info(f"Polish law-specific enhancements:")
        self.logger.info(f"  - Section-enhanced citations: {section_enhanced}")
        self.logger.info(f"  - Gazette reference validations: {gazette_validated}")
        self.logger.info(f"  - Polish terminology validated: {self.parsing_stats['polish_law_features']['polish_terminology_validated']}")
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about Polish law response parsing operations.
        
        Returns:
            Dictionary containing detailed parsing statistics and performance metrics for Polish law
        """
        stats = dict(self.parsing_stats)
        
        # Calculate success rates
        if stats['total_responses_parsed'] > 0:
            success_rate = (stats['successful_extractions'] / stats['total_responses_parsed']) * 100
            stats['overall_success_rate_percent'] = round(success_rate, 1)
        else:
            stats['overall_success_rate_percent'] = 0
        
        if stats['citation_blocks_processed'] > 0:
            block_success_rate = (stats['citation_blocks_successful'] / stats['citation_blocks_processed']) * 100
            stats['block_success_rate_percent'] = round(block_success_rate, 1)
        else:
            stats['block_success_rate_percent'] = 0
        
        # Calculate Polish law-specific enhancement rates
        if stats['enhanced_citations_created'] > 0:
            section_enhancement_rate = (stats['polish_law_features']['section_enhanced_citations'] / stats['enhanced_citations_created']) * 100
            stats['section_enhancement_rate_percent'] = round(section_enhancement_rate, 1)
            
            gazette_validation_rate = (stats['polish_law_features']['gazette_reference_validations'] / stats['enhanced_citations_created']) * 100
            stats['gazette_validation_rate_percent'] = round(gazette_validation_rate, 1)
        else:
            stats['section_enhancement_rate_percent'] = 0
            stats['gazette_validation_rate_percent'] = 0
        
        return stats
    
    def log_parsing_summary(self) -> None:
        """
        Log a comprehensive summary of all Polish law response parsing operations.
        
        This provides insights into the effectiveness of the sophisticated parsing
        approach and helps identify patterns in LLM response quality and processing success for Polish law.
        """
        stats = self.get_parsing_statistics()
        
        self.logger.info("=== POLISH LAW RESPONSE PARSING SUMMARY ===")
        self.logger.info(f"Total responses parsed: {stats['total_responses_parsed']}")
        self.logger.info(f"Overall success rate: {stats['overall_success_rate_percent']}%")
        self.logger.info(f"Enhanced citations created: {stats['enhanced_citations_created']}")
        self.logger.info(f"Fallback citations created: {stats['fallback_citations_created']}")
        self.logger.info(f"Citation blocks processed: {stats['citation_blocks_processed']}")
        self.logger.info(f"Block success rate: {stats['block_success_rate_percent']}%")
        self.logger.info(f"Average format compliance: {stats['llm_format_compliance_rate']:.1f}%")
        self.logger.info(f"Parsing errors: {stats['parsing_errors']}")
        
        # Log Polish law-specific parsing metrics
        self.logger.info("Polish law-specific parsing performance:")
        self.logger.info(f"  - Section-enhanced citations: {stats['polish_law_features']['section_enhanced_citations']} ({stats['section_enhancement_rate_percent']}%)")
        self.logger.info(f"  - Gazette reference validations: {stats['polish_law_features']['gazette_reference_validations']} ({stats['gazette_validation_rate_percent']}%)")
        self.logger.info(f"  - Polish terminology validated: {stats['polish_law_features']['polish_terminology_validated']}")
        self.logger.info(f"  - Parliament context added: {stats['polish_law_features']['parliament_context_added']}")


def create_polish_law_response_parser(metadata_processor: PolishLawMetadataProcessor,
                                     content_analyzer: PolishLawContentAnalyzer,
                                     citation_builder: PolishLawCitationBuilder,
                                     logger: logging.Logger) -> PolishLawResponseParser:
    """
    Factory function to create a configured Polish law response parser.
    
    This provides a clean interface for creating parser instances with
    proper dependency injection of all required components.
    """
    return PolishLawResponseParser(metadata_processor, content_analyzer, citation_builder, logger)