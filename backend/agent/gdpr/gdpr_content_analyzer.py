"""
GDPR Content Analyzer

This module performs intelligent analysis of GDPR document content using the structural
hints provided by the metadata processor. Instead of blind text parsing, this analyzer
uses the reconstructed metadata to guide its analysis, making it much more accurate
and efficient.

Think of this as having a "blueprint" while exploring a complex building - the metadata
tells us what to expect, making navigation much more reliable than wandering randomly.

The analyzer demonstrates how your architectural approach creates powerful synergies:
- Processing Pipeline creates flattened metadata
- Metadata Processor reconstructs structural hints  
- Content Analyzer uses hints for intelligent parsing
- Citation Builder uses parsed structure for precise references
"""

import re
import logging
from typing import Dict, List, Any, Optional


class GDPRContentAnalyzer:
    """
    Analyzes GDPR document content using intelligent, metadata-guided parsing.
    
    This class demonstrates how your flattened metadata approach enables sophisticated
    content analysis. Instead of generic text parsing, we use the structural hints
    from the metadata processor to guide our analysis, making it much more accurate
    and reliable than traditional approaches.
    
    The analyzer adapts its strategy based on the available metadata, using the most
    sophisticated approach possible while gracefully falling back when needed.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the GDPR content analyzer.
        
        Args:
            logger: Configured logger for tracking content analysis operations
        """
        self.logger = logger
        self.logger.info("GDPR Content Analyzer initialized")
        
        # Track analysis statistics across all operations
        self.analysis_stats = {
            'total_content_analyzed': 0,
            'guided_analysis_used': 0,
            'simple_analysis_used': 0,
            'parsing_successes': 0,
            'parsing_failures': 0,
            'quote_locations_found': 0,
            'quote_locations_failed': 0
        }
    
    def analyze_content_structure(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze document content structure using intelligent, metadata-guided parsing.
        
        This method demonstrates the power of your architectural approach. Instead of
        parsing content blindly, we use the structural hints from the metadata processor
        to guide our analysis, making it much more accurate and efficient.
        
        Args:
            content: Document content to analyze
            processing_hints: Structural hints from metadata processor
            
        Returns:
            Dictionary containing parsed content structure and analysis metadata
        """
        self.logger.debug("Starting intelligent content structure analysis")
        self.analysis_stats['total_content_analyzed'] += 1
        
        # Choose analysis strategy based on available hints
        if processing_hints.get('has_hints', False) and processing_hints.get('use_guided_parsing', False):
            return self._perform_guided_analysis(content, processing_hints)
        else:
            return self._perform_simple_analysis(content)
    
    def _perform_guided_analysis(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform guided content analysis using metadata hints.
        
        This method demonstrates how your flattened metadata creates a competitive
        advantage. Instead of generic parsing, we use the specific structural
        information preserved from the original processing to guide our analysis.
        """
        self.logger.debug("Using guided analysis with metadata hints")
        self.analysis_stats['guided_analysis_used'] += 1
        
        # Initialize content map with guided parsing approach
        content_map = {
            'paragraphs': {},
            'parsing_successful': False,
            'analysis_method': 'guided',
            'hints_used': processing_hints.copy(),
            'patterns_detected': []
        }
        
        # Extract guidance parameters from hints
        paragraph_count = processing_hints.get('paragraph_count', 0)
        has_sub_paragraphs = processing_hints.get('has_sub_paragraphs', False)
        numbering_style = processing_hints.get('numbering_style', '')
        expected_patterns = processing_hints.get('expected_patterns', [])
        
        self.logger.debug(f"Analysis guidance: {paragraph_count} paragraphs expected, "
                        f"sub-paragraphs: {has_sub_paragraphs}, style: {numbering_style}")
        
        try:
            # Use specialized parsing based on expected structure
            if has_sub_paragraphs:
                success = self._parse_with_sub_paragraph_guidance(
                    content, content_map, paragraph_count, numbering_style, expected_patterns
                )
            else:
                success = self._parse_with_paragraph_guidance(
                    content, content_map, paragraph_count, expected_patterns
                )
            
            if success:
                content_map['parsing_successful'] = True
                self.analysis_stats['parsing_successes'] += 1
                self.logger.debug(f"Guided analysis successful: found {len(content_map['paragraphs'])} paragraphs")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Guided analysis failed - structure not as expected")
                # Fall back to simple analysis
                return self._perform_simple_analysis(content)
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in guided analysis: {e}")
            # Fall back to simple analysis on error
            return self._perform_simple_analysis(content)
        
        return content_map
    
    def _parse_with_sub_paragraph_guidance(self, content: str, content_map: Dict[str, Any],
                                          paragraph_count: int, numbering_style: str,
                                          expected_patterns: List[str]) -> bool:
        """
        Parse content with sub-paragraph guidance from metadata hints.
        
        This method uses the specific numbering style information preserved in your
        flattened metadata to look for the right patterns, making parsing much more
        reliable than generic approaches.
        """
        self.logger.debug(f"Parsing with sub-paragraph guidance: style={numbering_style}")
        
        lines = content.split('\n')
        current_paragraph = None
        current_sub_paragraph = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for main paragraph markers using expected count for validation
            paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
            if paragraph_match:
                para_num = paragraph_match.group(1)
                para_text_start = paragraph_match.group(2)
                
                # Validate against expected count from metadata
                if int(para_num) <= paragraph_count:
                    content_map['paragraphs'][para_num] = {
                        'start_line': line_idx,
                        'start_text': para_text_start,
                        'full_text': line,
                        'sub_paragraphs': {}
                    }
                    current_paragraph = para_num
                    current_sub_paragraph = None
                    
                    content_map['patterns_detected'].append(f'main_paragraph_{para_num}')
                    self.logger.debug(f"Found expected main paragraph {para_num}")
                    continue
            
            # Look for sub-paragraphs using numbering style guidance
            if current_paragraph:
                sub_para_location = self._detect_sub_paragraph_with_style(
                    line, numbering_style, expected_patterns
                )
                
                if sub_para_location:
                    sub_para_key = sub_para_location['key']
                    sub_para_text = sub_para_location['text']
                    
                    content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                        'start_line': line_idx,
                        'text': sub_para_text,
                        'full_line': line,
                        'numbering_style': numbering_style
                    }
                    current_sub_paragraph = sub_para_key
                    
                    content_map['patterns_detected'].append(f'sub_paragraph_{current_paragraph}_{sub_para_key}')
                    self.logger.debug(f"Found sub-paragraph {current_paragraph}({sub_para_key})")
                    continue
                
                # Handle continuation text
                if current_sub_paragraph:
                    content_map['paragraphs'][current_paragraph]['sub_paragraphs'][current_sub_paragraph]['text'] += ' ' + line
                else:
                    content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
        
        # Validate that we found the expected structure
        found_paragraphs = len(content_map['paragraphs'])
        expected_paragraphs = paragraph_count
        
        if found_paragraphs > 0:
            self.logger.debug(f"Guided sub-paragraph parsing: found {found_paragraphs} paragraphs "
                            f"(expected {expected_paragraphs})")
            return True
        else:
            self.logger.warning("Guided sub-paragraph parsing found no structure")
            return False
    
    def _detect_sub_paragraph_with_style(self, line: str, numbering_style: str,
                                        expected_patterns: List[str]) -> Optional[Dict[str, str]]:
        """
        Detect sub-paragraphs using the specific numbering style from metadata.
        
        This method demonstrates how your metadata preservation enables precise
        pattern matching. We know exactly what style to look for based on the
        structural information preserved during processing.
        """
        # Handle GDPR's common alphabetical sub-paragraphs (a), (b), (c)
        if numbering_style == 'alphabetical' or 'alphabetical_sub_paragraphs' in expected_patterns:
            alpha_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
            if alpha_match:
                return {
                    'key': alpha_match.group(1),
                    'text': alpha_match.group(2),
                    'style': 'alphabetical'
                }
        
        # Handle numeric sub-paragraphs (1), (2), (3)
        if numbering_style == 'number_closing_paren' or 'numeric_sub_paragraphs' in expected_patterns:
            numeric_match = re.match(r'^\((\d+)\)\s+(.+)', line)
            if numeric_match:
                return {
                    'key': numeric_match.group(1),
                    'text': numeric_match.group(2),
                    'style': 'numeric'
                }
        
        # Handle mixed or unknown patterns by trying both
        if not numbering_style or 'mixed_sub_paragraphs' in expected_patterns:
            # Try alphabetical first (more common in GDPR)
            alpha_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
            if alpha_match:
                return {
                    'key': alpha_match.group(1),
                    'text': alpha_match.group(2),
                    'style': 'alphabetical_fallback'
                }
            
            # Try numeric as backup
            numeric_match = re.match(r'^\((\d+)\)\s+(.+)', line)
            if numeric_match:
                return {
                    'key': numeric_match.group(1),
                    'text': numeric_match.group(2),
                    'style': 'numeric_fallback'
                }
        
        return None
    
    def _parse_with_paragraph_guidance(self, content: str, content_map: Dict[str, Any],
                                      paragraph_count: int, expected_patterns: List[str]) -> bool:
        """
        Parse content with paragraph-level guidance for simpler structures.
        
        This method handles documents that have multiple paragraphs but no sub-paragraphs,
        using the expected paragraph count to validate the parsing results.
        """
        self.logger.debug(f"Parsing with paragraph guidance: expecting {paragraph_count} paragraphs")
        
        lines = content.split('\n')
        current_paragraph = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for paragraph markers
            paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
            if paragraph_match:
                para_num = paragraph_match.group(1)
                para_text = paragraph_match.group(2)
                
                # Validate against expected count
                if int(para_num) <= paragraph_count:
                    content_map['paragraphs'][para_num] = {
                        'start_line': line_idx,
                        'full_text': line,
                        'sub_paragraphs': {}
                    }
                    current_paragraph = para_num
                    
                    content_map['patterns_detected'].append(f'paragraph_{para_num}')
                    self.logger.debug(f"Found paragraph {para_num}")
                    continue
            
            # Handle continuation text
            if current_paragraph:
                content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
        
        found_paragraphs = len(content_map['paragraphs'])
        
        if found_paragraphs > 0:
            self.logger.debug(f"Guided paragraph parsing: found {found_paragraphs} paragraphs")
            return True
        else:
            self.logger.warning("Guided paragraph parsing found no structure")
            return False
    
    def _perform_simple_analysis(self, content: str) -> Dict[str, Any]:
        """
        Perform simple content analysis without metadata guidance.
        
        This method provides reliable parsing even when enhanced metadata is not
        available, ensuring the system works gracefully across all document types.
        It demonstrates how your architecture gracefully degrades while maintaining
        functionality.
        """
        self.logger.debug("Using simple analysis without metadata guidance")
        self.analysis_stats['simple_analysis_used'] += 1
        
        content_map = {
            'paragraphs': {},
            'parsing_successful': False,
            'analysis_method': 'simple',
            'hints_used': {},
            'patterns_detected': []
        }
        
        try:
            lines = content.split('\n')
            current_paragraph = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for basic paragraph patterns without guidance
                paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
                if paragraph_match:
                    para_num = paragraph_match.group(1)
                    para_text = paragraph_match.group(2)
                    
                    content_map['paragraphs'][para_num] = {
                        'start_line': line_idx,
                        'full_text': line,
                        'sub_paragraphs': {}
                    }
                    current_paragraph = para_num
                    content_map['patterns_detected'].append(f'simple_paragraph_{para_num}')
                    continue
                
                # Look for sub-paragraphs using general patterns
                if current_paragraph:
                    # Try both alphabetical and numeric patterns
                    alpha_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
                    numeric_match = re.match(r'^\((\d+)\)\s+(.+)', line)
                    
                    if alpha_match:
                        sub_para_key = alpha_match.group(1)
                        sub_para_text = alpha_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line,
                            'numbering_style': 'alphabetical_detected'
                        }
                        content_map['patterns_detected'].append(f'simple_sub_para_{current_paragraph}_{sub_para_key}')
                        continue
                    elif numeric_match:
                        sub_para_key = numeric_match.group(1)
                        sub_para_text = numeric_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line,
                            'numbering_style': 'numeric_detected'
                        }
                        content_map['patterns_detected'].append(f'simple_sub_para_{current_paragraph}_{sub_para_key}')
                        continue
                
                # Handle continuation text
                if current_paragraph:
                    content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
            
            if len(content_map['paragraphs']) > 0:
                content_map['parsing_successful'] = True
                self.analysis_stats['parsing_successes'] += 1
                self.logger.debug(f"Simple analysis successful: found {len(content_map['paragraphs'])} paragraphs")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Simple analysis found no paragraph structure")
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in simple content analysis: {e}")
        
        return content_map
    
    def locate_quote_in_structure(self, quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate a specific quote within the analyzed document structure.
        
        This method demonstrates how the structured analysis enables precise quote
        location, which is essential for creating accurate citations. The method
        provides different levels of precision based on what the analysis discovered.
        """
        self.logger.debug(f"Locating quote in analyzed structure: '{quote[:50]}...'")
        
        if not content_map.get('parsing_successful', False):
            self.logger.warning("Cannot locate quote - content analysis was not successful")
            return None
        
        # Clean and normalize quote for better matching
        clean_quote = self._normalize_quote_for_matching(quote)
        if not clean_quote:
            return None
        
        try:
            # Search through the analyzed structure with detailed logging
            location_result = self._search_structure_for_quote(clean_quote, content_map)
            
            if location_result:
                self.analysis_stats['quote_locations_found'] += 1
                self._log_quote_location_success(location_result, quote)
            else:
                self.analysis_stats['quote_locations_failed'] += 1
                self.logger.warning("Could not locate quote in analyzed document structure")
            
            return location_result
            
        except Exception as e:
            self.analysis_stats['quote_locations_failed'] += 1
            self.logger.warning(f"Error locating quote in structure: {e}")
            return None
    
    def _normalize_quote_for_matching(self, quote: str) -> Optional[str]:
        """
        Normalize a quote for reliable matching within the document structure.
        
        This method cleans up the quote text to improve matching reliability
        while maintaining enough content for accurate identification.
        """
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.warning(f"Quote too short for reliable matching: '{quote}'")
            return None
        
        return clean_quote
    
    def _search_structure_for_quote(self, clean_quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search through the analyzed structure to find the quote location.
        
        This method provides hierarchical search, checking sub-paragraphs first
        for maximum precision, then falling back to paragraph-level location.
        """
        for para_num, para_data in content_map['paragraphs'].items():
            # Check main paragraph text
            para_text = ' '.join(para_data.get('full_text', '').split()).lower()
            
            if clean_quote in para_text:
                # Found in paragraph - check for sub-paragraph specificity
                sub_paragraph_location = self._check_sub_paragraphs_for_quote(
                    clean_quote, para_data, para_num
                )
                
                if sub_paragraph_location:
                    return sub_paragraph_location
                
                # Quote found in main paragraph but not in specific sub-paragraph
                self.logger.debug(f"Quote located in main paragraph {para_num}")
                return {
                    'paragraph': para_num,
                    'sub_paragraph': None,
                    'location_type': 'main_paragraph',
                    'confidence': 'medium'
                }
        
        return None
    
    def _check_sub_paragraphs_for_quote(self, clean_quote: str, para_data: Dict[str, Any],
                                       para_num: str) -> Optional[Dict[str, str]]:
        """
        Check sub-paragraphs for quote location to achieve maximum precision.
        
        This method searches within sub-paragraphs to provide the most precise
        citation possible, enabling references like "Article 6, paragraph 1(a)".
        """
        sub_paragraphs = para_data.get('sub_paragraphs', {})
        
        for sub_para_key, sub_para_data in sub_paragraphs.items():
            sub_para_text = ' '.join(sub_para_data.get('text', '').split()).lower()
            
            if clean_quote in sub_para_text:
                self.logger.debug(f"Quote precisely located: paragraph {para_num}, sub-paragraph {sub_para_key}")
                return {
                    'paragraph': para_num,
                    'sub_paragraph': sub_para_key,
                    'location_type': 'sub_paragraph',
                    'confidence': 'high',
                    'numbering_style': sub_para_data.get('numbering_style', 'unknown')
                }
        
        return None
    
    def _log_quote_location_success(self, location_result: Dict[str, str], original_quote: str) -> None:
        """
        Log successful quote location with details for debugging and verification.
        
        This logging helps track the effectiveness of the structure analysis
        and quote location process for optimization and debugging purposes.
        """
        location_type = location_result['location_type']
        confidence = location_result['confidence']
        paragraph = location_result['paragraph']
        sub_paragraph = location_result.get('sub_paragraph')
        
        if sub_paragraph:
            self.logger.info(f"Quote located with {confidence} confidence: "
                           f"paragraph {paragraph}, sub-paragraph {sub_paragraph}")
        else:
            self.logger.info(f"Quote located with {confidence} confidence: paragraph {paragraph}")
        
        # Log quote preview for verification
        quote_preview = original_quote[:100] + "..." if len(original_quote) > 100 else original_quote
        self.logger.debug(f"Located quote: '{quote_preview}'")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about content analysis operations.
        
        Returns:
            Dictionary containing detailed analysis statistics and performance metrics
        """
        stats = dict(self.analysis_stats)
        
        # Calculate success rates and performance metrics
        if stats['total_content_analyzed'] > 0:
            guided_rate = (stats['guided_analysis_used'] / stats['total_content_analyzed']) * 100
            stats['guided_analysis_rate_percent'] = round(guided_rate, 1)
            
            parsing_success_rate = (stats['parsing_successes'] / stats['total_content_analyzed']) * 100
            stats['parsing_success_rate_percent'] = round(parsing_success_rate, 1)
        else:
            stats['guided_analysis_rate_percent'] = 0
            stats['parsing_success_rate_percent'] = 0
        
        # Calculate quote location success rate
        total_quote_attempts = stats['quote_locations_found'] + stats['quote_locations_failed']
        if total_quote_attempts > 0:
            quote_success_rate = (stats['quote_locations_found'] / total_quote_attempts) * 100
            stats['quote_location_success_rate_percent'] = round(quote_success_rate, 1)
        else:
            stats['quote_location_success_rate_percent'] = 0
        
        return stats
    
    def log_analysis_summary(self) -> None:
        """
        Log a comprehensive summary of all content analysis operations.
        
        This provides visibility into how well the guided analysis approach
        is working and helps identify opportunities for optimization.
        """
        stats = self.get_analysis_statistics()
        
        self.logger.info("=== GDPR CONTENT ANALYSIS SUMMARY ===")
        self.logger.info(f"Total content analyzed: {stats['total_content_analyzed']}")
        self.logger.info(f"Guided analysis used: {stats['guided_analysis_used']} ({stats['guided_analysis_rate_percent']}%)")
        self.logger.info(f"Simple analysis used: {stats['simple_analysis_used']}")
        self.logger.info(f"Parsing success rate: {stats['parsing_success_rate_percent']}%")
        self.logger.info(f"Quote location success rate: {stats['quote_location_success_rate_percent']}%")
        self.logger.info(f"Quote locations found: {stats['quote_locations_found']}")
        self.logger.info(f"Analysis method effectiveness demonstrates metadata guidance value")


def create_gdpr_content_analyzer(logger: logging.Logger) -> GDPRContentAnalyzer:
    """
    Factory function to create a configured GDPR content analyzer.
    
    This provides a clean interface for creating analyzer instances with
    proper dependency injection of the logger.
    """
    return GDPRContentAnalyzer(logger)