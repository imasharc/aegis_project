"""
Polish Law Content Analyzer

This module performs intelligent analysis of Polish law document content using the structural
hints provided by the Polish law metadata processor. Instead of blind text parsing, this analyzer
uses the reconstructed metadata to guide its analysis, making it much more accurate
and efficient for Polish legal document patterns.

Polish Law Specific Features:
- Section-aware content analysis (unique to Polish legal structure)
- Polish legal numbering pattern recognition (1), 2), 3) vs (a), (b), (c))
- Polish legal terminology detection and processing
- Gazette reference context awareness
- Polish legal citation pattern recognition

Think of this as having a "blueprint" while exploring a Polish legal building - the metadata
tells us what to expect based on Polish legal document conventions, making navigation much 
more reliable than wandering randomly through the text.

The analyzer demonstrates how your architectural approach creates powerful synergies:
- Polish Law Processing Pipeline creates flattened metadata with Polish legal patterns
- Polish Law Metadata Processor reconstructs structural hints with section awareness
- Polish Law Content Analyzer uses hints for intelligent Polish legal document parsing
- Polish Law Citation Builder uses parsed structure for precise Polish legal references
"""

import re
import logging
from typing import Dict, List, Any, Optional


class PolishLawContentAnalyzer:
    """
    Analyzes Polish law document content using intelligent, metadata-guided parsing.
    
    This class demonstrates how your flattened metadata approach enables sophisticated
    content analysis specifically adapted for Polish legal documents. Instead of generic 
    text parsing, we use the structural hints from the metadata processor to guide our 
    analysis, making it much more accurate and reliable than traditional approaches.
    
    The analyzer adapts its strategy based on the available metadata, using the most
    sophisticated approach possible while gracefully falling back when needed, with
    special recognition of Polish legal document patterns and organizational structures.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the Polish law content analyzer.
        
        Args:
            logger: Configured logger for tracking content analysis operations
        """
        self.logger = logger
        self.logger.info("Polish Law Content Analyzer initialized")
        
        # Track analysis statistics across all operations with Polish law specifics
        self.analysis_stats = {
            'total_content_analyzed': 0,
            'guided_analysis_used': 0,
            'simple_analysis_used': 0,
            'parsing_successes': 0,
            'parsing_failures': 0,
            'quote_locations_found': 0,
            'quote_locations_failed': 0,
            'section_aware_parsing_used': 0,  # Unique to Polish law
            'polish_terminology_detected': 0,  # Polish legal language patterns
            'gazette_context_detected': 0     # Polish law authenticity markers
        }
    
    def analyze_content_structure(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Polish law document content structure using intelligent, metadata-guided parsing.
        
        This method demonstrates the power of your architectural approach applied to Polish law.
        Instead of parsing content blindly, we use the structural hints from the metadata processor
        to guide our analysis, making it much more accurate and efficient for Polish legal documents.
        
        Args:
            content: Polish law document content to analyze
            processing_hints: Structural hints from Polish law metadata processor
            
        Returns:
            Dictionary containing parsed content structure and analysis metadata for Polish law
        """
        self.logger.debug("Starting intelligent Polish law content structure analysis")
        self.analysis_stats['total_content_analyzed'] += 1
        
        # Choose analysis strategy based on available hints and Polish law patterns
        if processing_hints.get('has_hints', False) and processing_hints.get('use_guided_parsing', False):
            return self._perform_polish_law_guided_analysis(content, processing_hints)
        else:
            return self._perform_simple_polish_law_analysis(content)
    
    def _perform_polish_law_guided_analysis(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform guided content analysis using Polish law metadata hints.
        
        This method demonstrates how your flattened metadata creates a competitive
        advantage for Polish law documents. Instead of generic parsing, we use the specific 
        structural information preserved from the original processing to guide our analysis,
        including Polish law-specific organizational patterns like sections.
        """
        self.logger.debug("Using guided analysis with Polish law metadata hints")
        self.analysis_stats['guided_analysis_used'] += 1
        
        # Initialize content map with guided parsing approach for Polish law
        content_map = {
            'paragraphs': {},
            'sections': {},  # Unique to Polish law structure
            'parsing_successful': False,
            'analysis_method': 'polish_law_guided',
            'hints_used': processing_hints.copy(),
            'patterns_detected': [],
            'polish_law_features': {
                'sections_found': 0,
                'polish_terms_detected': [],
                'gazette_references': [],
                'numbering_patterns': []
            }
        }
        
        # Extract guidance parameters from hints with Polish law specifics
        paragraph_count = processing_hints.get('paragraph_count', 0)
        has_sub_paragraphs = processing_hints.get('has_sub_paragraphs', False)
        numbering_style = processing_hints.get('numbering_style', '')
        expected_patterns = processing_hints.get('expected_patterns', [])
        has_sections = processing_hints.get('has_sections', False)
        section_aware_parsing = processing_hints.get('section_aware_parsing', False)
        
        self.logger.debug(f"Polish law analysis guidance: {paragraph_count} paragraphs expected, "
                        f"sub-paragraphs: {has_sub_paragraphs}, style: {numbering_style}, "
                        f"sections: {has_sections}")
        
        try:
            # Detect Polish legal terminology first for validation
            self._detect_polish_legal_terminology(content, content_map)
            
            # Use specialized parsing based on expected Polish law structure
            if section_aware_parsing and has_sections:
                success = self._parse_with_section_and_paragraph_guidance(
                    content, content_map, paragraph_count, numbering_style, expected_patterns
                )
                self.analysis_stats['section_aware_parsing_used'] += 1
            elif has_sub_paragraphs:
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
                self.logger.debug(f"Polish law guided analysis successful: found {len(content_map['paragraphs'])} paragraphs, "
                                f"{content_map['polish_law_features']['sections_found']} sections")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Polish law guided analysis failed - structure not as expected")
                # Fall back to simple analysis
                return self._perform_simple_polish_law_analysis(content)
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in Polish law guided analysis: {e}")
            # Fall back to simple analysis on error
            return self._perform_simple_polish_law_analysis(content)
        
        return content_map
    
    def _detect_polish_legal_terminology(self, content: str, content_map: Dict[str, Any]) -> None:
        """
        Detect Polish legal terminology and patterns in the content.
        
        This validation ensures we're working with authentic Polish legal content
        and helps identify the linguistic patterns that can guide our parsing.
        """
        polish_legal_terms = {
            'ustawa': 'law/statute',
            'artykuł': 'article', 
            'rozdział': 'chapter',
            'przepis': 'provision',
            'dziennik ustaw': 'official gazette',
            'parlament': 'parliament',
            'nowelizacja': 'amendment',
            'uchylenie': 'repeal',
            'wejście w życie': 'entry into force'
        }
        
        content_lower = content.lower()
        detected_terms = []
        
        for term, translation in polish_legal_terms.items():
            if term in content_lower:
                detected_terms.append({'term': term, 'translation': translation})
                self.analysis_stats['polish_terminology_detected'] += 1
        
        content_map['polish_law_features']['polish_terms_detected'] = detected_terms
        
        if detected_terms:
            self.logger.debug(f"Polish legal terminology detected: {[t['term'] for t in detected_terms]}")
            content_map['patterns_detected'].append('authentic_polish_legal_content')
        else:
            self.logger.warning("No Polish legal terminology detected - content may not be Polish law")
        
        # Detect gazette references (critical for Polish law authenticity)
        gazette_patterns = [r'dz\.?\s*u\.?\s*\d+', r'dziennik\s+ustaw\s+\d+']
        gazette_refs = []
        
        for pattern in gazette_patterns:
            matches = re.finditer(pattern, content_lower)
            for match in matches:
                gazette_refs.append(match.group())
                self.analysis_stats['gazette_context_detected'] += 1
        
        if gazette_refs:
            content_map['polish_law_features']['gazette_references'] = gazette_refs
            content_map['patterns_detected'].append('gazette_references_found')
            self.logger.debug(f"Gazette references detected: {gazette_refs}")
    
    def _parse_with_section_and_paragraph_guidance(self, content: str, content_map: Dict[str, Any],
                                                  paragraph_count: int, numbering_style: str,
                                                  expected_patterns: List[str]) -> bool:
        """
        Parse content with both section and paragraph guidance from Polish law metadata hints.
        
        This is the most sophisticated parsing method, using both section organization
        (unique to Polish law) and paragraph structure information to achieve maximum
        precision in content analysis.
        """
        self.logger.debug(f"Parsing with Polish law section and paragraph guidance: style={numbering_style}")
        
        lines = content.split('\n')
        current_section = None
        current_paragraph = None
        current_sub_paragraph = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for section markers (unique to Polish law structure)
            section_match = re.match(r'^(?:Rozdział|Section)\s+([IVXLC]+|\d+)[\.\:]?\s*(.+)', line, re.IGNORECASE)
            if section_match:
                section_num = section_match.group(1)
                section_title = section_match.group(2)
                
                content_map['sections'][section_num] = {
                    'start_line': line_idx,
                    'title': section_title,
                    'full_text': line,
                    'paragraphs': {}
                }
                current_section = section_num
                current_paragraph = None
                current_sub_paragraph = None
                
                content_map['polish_law_features']['sections_found'] += 1
                content_map['patterns_detected'].append(f'section_{section_num}')
                self.logger.debug(f"Found Polish law section {section_num}: {section_title}")
                continue
            
            # Look for main paragraph markers using expected count for validation
            paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
            if paragraph_match:
                para_num = paragraph_match.group(1)
                para_text_start = paragraph_match.group(2)
                
                # Validate against expected count from metadata
                if int(para_num) <= paragraph_count:
                    paragraph_data = {
                        'start_line': line_idx,
                        'start_text': para_text_start,
                        'full_text': line,
                        'sub_paragraphs': {},
                        'section': current_section
                    }
                    
                    content_map['paragraphs'][para_num] = paragraph_data
                    
                    # Also add to section if we're in one
                    if current_section and current_section in content_map['sections']:
                        content_map['sections'][current_section]['paragraphs'][para_num] = paragraph_data
                    
                    current_paragraph = para_num
                    current_sub_paragraph = None
                    
                    content_map['patterns_detected'].append(f'main_paragraph_{para_num}')
                    self.logger.debug(f"Found expected main paragraph {para_num} in section {current_section}")
                    continue
            
            # Look for sub-paragraphs using Polish law numbering style guidance
            if current_paragraph:
                sub_para_location = self._detect_sub_paragraph_with_polish_style(
                    line, numbering_style, expected_patterns
                )
                
                if sub_para_location:
                    sub_para_key = sub_para_location['key']
                    sub_para_text = sub_para_location['text']
                    
                    content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                        'start_line': line_idx,
                        'text': sub_para_text,
                        'full_line': line,
                        'numbering_style': numbering_style,
                        'section': current_section
                    }
                    current_sub_paragraph = sub_para_key
                    
                    content_map['patterns_detected'].append(f'sub_paragraph_{current_paragraph}_{sub_para_key}')
                    content_map['polish_law_features']['numbering_patterns'].append(sub_para_location['style'])
                    self.logger.debug(f"Found sub-paragraph {current_paragraph}({sub_para_key}) in section {current_section}")
                    continue
                
                # Handle continuation text
                if current_sub_paragraph:
                    content_map['paragraphs'][current_paragraph]['sub_paragraphs'][current_sub_paragraph]['text'] += ' ' + line
                else:
                    content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
        
        # Validate that we found the expected structure
        found_paragraphs = len(content_map['paragraphs'])
        found_sections = len(content_map['sections'])
        expected_paragraphs = paragraph_count
        
        if found_paragraphs > 0 or found_sections > 0:
            self.logger.debug(f"Polish law section-aware parsing: found {found_paragraphs} paragraphs, "
                            f"{found_sections} sections (expected {expected_paragraphs} paragraphs)")
            return True
        else:
            self.logger.warning("Polish law section-aware parsing found no structure")
            return False
    
    def _detect_sub_paragraph_with_polish_style(self, line: str, numbering_style: str,
                                               expected_patterns: List[str]) -> Optional[Dict[str, str]]:
        """
        Detect sub-paragraphs using Polish law-specific numbering style from metadata.
        
        This method demonstrates how your metadata preservation enables precise
        pattern matching for Polish legal documents. We know exactly what style to look 
        for based on the structural information preserved during processing.
        """
        # Handle Polish law's numeric sub-paragraphs 1), 2), 3) (more common than in GDPR)
        if numbering_style == 'number_closing_paren' or 'numeric_sub_paragraphs' in expected_patterns:
            numeric_match = re.match(r'^(\d+)\)\s+(.+)', line)
            if numeric_match:
                return {
                    'key': numeric_match.group(1),
                    'text': numeric_match.group(2),
                    'style': 'polish_numeric'
                }
        
        # Handle alphabetical sub-paragraphs (a), (b), (c) - also used in Polish law
        if numbering_style == 'alphabetical' or 'alphabetical_sub_paragraphs' in expected_patterns:
            # Polish law sometimes uses both (a) and a) formats
            alpha_paren_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
            alpha_close_match = re.match(r'^([a-z])\)\s+(.+)', line)
            
            if alpha_paren_match:
                return {
                    'key': alpha_paren_match.group(1),
                    'text': alpha_paren_match.group(2),
                    'style': 'polish_alphabetical_paren'
                }
            elif alpha_close_match:
                return {
                    'key': alpha_close_match.group(1),
                    'text': alpha_close_match.group(2),
                    'style': 'polish_alphabetical_close'
                }
        
        # Handle mixed or unknown patterns by trying Polish law common patterns
        if not numbering_style or 'mixed_sub_paragraphs' in expected_patterns:
            # Try numeric first (more common in Polish law)
            numeric_match = re.match(r'^(\d+)\)\s+(.+)', line)
            if numeric_match:
                return {
                    'key': numeric_match.group(1),
                    'text': numeric_match.group(2),
                    'style': 'polish_numeric_fallback'
                }
            
            # Try alphabetical as backup
            alpha_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
            if alpha_match:
                return {
                    'key': alpha_match.group(1),
                    'text': alpha_match.group(2),
                    'style': 'polish_alphabetical_fallback'
                }
        
        return None
    
    def _parse_with_sub_paragraph_guidance(self, content: str, content_map: Dict[str, Any],
                                          paragraph_count: int, numbering_style: str,
                                          expected_patterns: List[str]) -> bool:
        """
        Parse content with sub-paragraph guidance for Polish law documents.
        
        This method handles documents that have multiple paragraphs with sub-paragraphs
        but no section organization, using Polish law-specific numbering patterns.
        """
        self.logger.debug(f"Parsing with Polish law sub-paragraph guidance: style={numbering_style}")
        
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
            
            # Look for sub-paragraphs using Polish law numbering style guidance
            if current_paragraph:
                sub_para_location = self._detect_sub_paragraph_with_polish_style(
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
                    content_map['polish_law_features']['numbering_patterns'].append(sub_para_location['style'])
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
            self.logger.debug(f"Polish law guided sub-paragraph parsing: found {found_paragraphs} paragraphs "
                            f"(expected {expected_paragraphs})")
            return True
        else:
            self.logger.warning("Polish law guided sub-paragraph parsing found no structure")
            return False
    
    def _parse_with_paragraph_guidance(self, content: str, content_map: Dict[str, Any],
                                      paragraph_count: int, expected_patterns: List[str]) -> bool:
        """
        Parse content with paragraph-level guidance for simpler Polish law structures.
        
        This method handles documents that have multiple paragraphs but no sub-paragraphs,
        using the expected paragraph count to validate the parsing results.
        """
        self.logger.debug(f"Parsing with Polish law paragraph guidance: expecting {paragraph_count} paragraphs")
        
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
            self.logger.debug(f"Polish law guided paragraph parsing: found {found_paragraphs} paragraphs")
            return True
        else:
            self.logger.warning("Polish law guided paragraph parsing found no structure")
            return False
    
    def _perform_simple_polish_law_analysis(self, content: str) -> Dict[str, Any]:
        """
        Perform simple content analysis without metadata guidance for Polish law documents.
        
        This method provides reliable parsing even when enhanced metadata is not
        available, ensuring the system works gracefully across all document types
        while still applying Polish law-specific recognition patterns.
        """
        self.logger.debug("Using simple analysis without metadata guidance for Polish law")
        self.analysis_stats['simple_analysis_used'] += 1
        
        content_map = {
            'paragraphs': {},
            'sections': {},
            'parsing_successful': False,
            'analysis_method': 'polish_law_simple',
            'hints_used': {},
            'patterns_detected': [],
            'polish_law_features': {
                'sections_found': 0,
                'polish_terms_detected': [],
                'gazette_references': [],
                'numbering_patterns': []
            }
        }
        
        try:
            # Detect Polish legal terminology even in simple analysis
            self._detect_polish_legal_terminology(content, content_map)
            
            lines = content.split('\n')
            current_paragraph = None
            current_section = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for section markers (even in simple analysis)
                section_match = re.match(r'^(?:Rozdział|Section)\s+([IVXLC]+|\d+)[\.\:]?\s*(.+)', line, re.IGNORECASE)
                if section_match:
                    section_num = section_match.group(1)
                    section_title = section_match.group(2)
                    
                    content_map['sections'][section_num] = {
                        'start_line': line_idx,
                        'title': section_title,
                        'full_text': line,
                        'paragraphs': {}
                    }
                    current_section = section_num
                    content_map['polish_law_features']['sections_found'] += 1
                    content_map['patterns_detected'].append(f'simple_section_{section_num}')
                    continue
                
                # Look for basic paragraph patterns without guidance
                paragraph_match = re.match(r'^(\d+)\.\s+(.+)', line)
                if paragraph_match:
                    para_num = paragraph_match.group(1)
                    para_text = paragraph_match.group(2)
                    
                    content_map['paragraphs'][para_num] = {
                        'start_line': line_idx,
                        'full_text': line,
                        'sub_paragraphs': {},
                        'section': current_section
                    }
                    current_paragraph = para_num
                    content_map['patterns_detected'].append(f'simple_paragraph_{para_num}')
                    continue
                
                # Look for sub-paragraphs using Polish law general patterns
                if current_paragraph:
                    # Try both numeric and alphabetical patterns common in Polish law
                    numeric_match = re.match(r'^(\d+)\)\s+(.+)', line)
                    alpha_paren_match = re.match(r'^\(([a-z])\)\s+(.+)', line)
                    alpha_close_match = re.match(r'^([a-z])\)\s+(.+)', line)
                    
                    if numeric_match:
                        sub_para_key = numeric_match.group(1)
                        sub_para_text = numeric_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line,
                            'numbering_style': 'numeric_detected'
                        }
                        content_map['patterns_detected'].append(f'simple_sub_para_{current_paragraph}_{sub_para_key}')
                        content_map['polish_law_features']['numbering_patterns'].append('numeric_simple')
                        continue
                    elif alpha_paren_match:
                        sub_para_key = alpha_paren_match.group(1)
                        sub_para_text = alpha_paren_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line,
                            'numbering_style': 'alphabetical_paren_detected'
                        }
                        content_map['patterns_detected'].append(f'simple_sub_para_{current_paragraph}_{sub_para_key}')
                        content_map['polish_law_features']['numbering_patterns'].append('alphabetical_paren_simple')
                        continue
                    elif alpha_close_match:
                        sub_para_key = alpha_close_match.group(1)
                        sub_para_text = alpha_close_match.group(2)
                        
                        content_map['paragraphs'][current_paragraph]['sub_paragraphs'][sub_para_key] = {
                            'start_line': line_idx,
                            'text': sub_para_text,
                            'full_line': line,
                            'numbering_style': 'alphabetical_close_detected'
                        }
                        content_map['patterns_detected'].append(f'simple_sub_para_{current_paragraph}_{sub_para_key}')
                        content_map['polish_law_features']['numbering_patterns'].append('alphabetical_close_simple')
                        continue
                
                # Handle continuation text
                if current_paragraph:
                    content_map['paragraphs'][current_paragraph]['full_text'] += ' ' + line
            
            if len(content_map['paragraphs']) > 0 or len(content_map['sections']) > 0:
                content_map['parsing_successful'] = True
                self.analysis_stats['parsing_successes'] += 1
                self.logger.debug(f"Polish law simple analysis successful: found {len(content_map['paragraphs'])} paragraphs, "
                                f"{len(content_map['sections'])} sections")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Polish law simple analysis found no paragraph or section structure")
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in simple Polish law content analysis: {e}")
        
        return content_map
    
    def locate_quote_in_structure(self, quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate a specific quote within the analyzed Polish law document structure.
        
        This method demonstrates how the structured analysis enables precise quote
        location for Polish legal documents, which is essential for creating accurate citations
        with Polish law-specific structural references including sections.
        """
        self.logger.debug(f"Locating quote in analyzed Polish law structure: '{quote[:50]}...'")
        
        if not content_map.get('parsing_successful', False):
            self.logger.warning("Cannot locate quote - Polish law content analysis was not successful")
            return None
        
        # Clean and normalize quote for better matching
        clean_quote = self._normalize_quote_for_matching(quote)
        if not clean_quote:
            return None
        
        try:
            # Search through the analyzed structure with detailed logging
            location_result = self._search_polish_law_structure_for_quote(clean_quote, content_map)
            
            if location_result:
                self.analysis_stats['quote_locations_found'] += 1
                self._log_quote_location_success(location_result, quote)
            else:
                self.analysis_stats['quote_locations_failed'] += 1
                self.logger.warning("Could not locate quote in analyzed Polish law document structure")
            
            return location_result
            
        except Exception as e:
            self.analysis_stats['quote_locations_failed'] += 1
            self.logger.warning(f"Error locating quote in Polish law structure: {e}")
            return None
    
    def _search_polish_law_structure_for_quote(self, clean_quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search through the analyzed Polish law structure to find the quote location.
        
        This method provides hierarchical search, checking sub-paragraphs first
        for maximum precision, then falling back to paragraph-level location,
        with special handling for Polish law section organization.
        """
        # First search within sections if they exist (unique to Polish law)
        sections = content_map.get('sections', {})
        if sections:
            for section_num, section_data in sections.items():
                section_paragraphs = section_data.get('paragraphs', {})
                
                for para_num, para_data in section_paragraphs.items():
                    para_text = ' '.join(para_data.get('full_text', '').split()).lower()
                    
                    if clean_quote in para_text:
                        # Found in paragraph within section - check for sub-paragraph specificity
                        sub_paragraph_location = self._check_sub_paragraphs_for_quote(
                            clean_quote, para_data, para_num
                        )
                        
                        if sub_paragraph_location:
                            # Maximum precision: section, paragraph, and sub-paragraph
                            sub_paragraph_location['section'] = section_num
                            sub_paragraph_location['location_type'] = 'section_paragraph_sub_paragraph'
                            sub_paragraph_location['confidence'] = 'maximum'
                            return sub_paragraph_location
                        
                        # High precision: section and paragraph
                        self.logger.debug(f"Quote located in Polish law section {section_num}, paragraph {para_num}")
                        return {
                            'section': section_num,
                            'paragraph': para_num,
                            'sub_paragraph': None,
                            'location_type': 'section_paragraph',
                            'confidence': 'high'
                        }
        
        # Search in paragraphs not associated with sections
        for para_num, para_data in content_map['paragraphs'].items():
            # Skip if this paragraph is already in a section (avoid duplicates)
            if para_data.get('section'):
                continue
            
            para_text = ' '.join(para_data.get('full_text', '').split()).lower()
            
            if clean_quote in para_text:
                # Found in paragraph - check for sub-paragraph specificity
                sub_paragraph_location = self._check_sub_paragraphs_for_quote(
                    clean_quote, para_data, para_num
                )
                
                if sub_paragraph_location:
                    return sub_paragraph_location
                
                # Quote found in main paragraph but not in specific sub-paragraph
                self.logger.debug(f"Quote located in Polish law main paragraph {para_num}")
                return {
                    'paragraph': para_num,
                    'sub_paragraph': None,
                    'section': None,
                    'location_type': 'main_paragraph',
                    'confidence': 'medium'
                }
        
        return None
    
    def _check_sub_paragraphs_for_quote(self, clean_quote: str, para_data: Dict[str, Any],
                                       para_num: str) -> Optional[Dict[str, str]]:
        """
        Check sub-paragraphs for quote location to achieve maximum precision for Polish law.
        
        This method searches within sub-paragraphs to provide the most precise
        citation possible, enabling references like "Article 6, paragraph 1, point 2)"
        following Polish legal citation conventions.
        """
        sub_paragraphs = para_data.get('sub_paragraphs', {})
        
        for sub_para_key, sub_para_data in sub_paragraphs.items():
            sub_para_text = ' '.join(sub_para_data.get('text', '').split()).lower()
            
            if clean_quote in sub_para_text:
                section = para_data.get('section')
                self.logger.debug(f"Quote precisely located: paragraph {para_num}, sub-paragraph {sub_para_key}"
                                f"{f', section {section}' if section else ''}")
                
                return {
                    'paragraph': para_num,
                    'sub_paragraph': sub_para_key,
                    'section': section,
                    'location_type': 'sub_paragraph',
                    'confidence': 'high',
                    'numbering_style': sub_para_data.get('numbering_style', 'unknown')
                }
        
        return None
    
    def _normalize_quote_for_matching(self, quote: str) -> Optional[str]:
        """
        Normalize a quote for reliable matching within Polish law document structure.
        
        This method cleans up the quote text to improve matching reliability
        while maintaining enough content for accurate identification.
        """
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.warning(f"Quote too short for reliable matching: '{quote}'")
            return None
        
        return clean_quote
    
    def _log_quote_location_success(self, location_result: Dict[str, str], original_quote: str) -> None:
        """
        Log successful quote location with details for debugging and verification.
        
        This logging helps track the effectiveness of the structure analysis
        and quote location process for optimization and debugging purposes,
        with Polish law-specific context information.
        """
        location_type = location_result['location_type']
        confidence = location_result['confidence']
        paragraph = location_result.get('paragraph')
        sub_paragraph = location_result.get('sub_paragraph')
        section = location_result.get('section')
        
        # Build location description with Polish law specifics
        location_desc = []
        if section:
            location_desc.append(f"section {section}")
        if paragraph:
            location_desc.append(f"paragraph {paragraph}")
        if sub_paragraph:
            location_desc.append(f"sub-paragraph {sub_paragraph}")
        
        location_str = ", ".join(location_desc) if location_desc else "unknown location"
        
        self.logger.info(f"Quote located with {confidence} confidence in Polish law document: {location_str}")
        
        # Log quote preview for verification
        quote_preview = original_quote[:100] + "..." if len(original_quote) > 100 else original_quote
        self.logger.debug(f"Located quote: '{quote_preview}'")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about Polish law content analysis operations.
        
        Returns:
            Dictionary containing detailed analysis statistics and performance metrics for Polish law
        """
        stats = dict(self.analysis_stats)
        
        # Calculate success rates and performance metrics for Polish law
        if stats['total_content_analyzed'] > 0:
            guided_rate = (stats['guided_analysis_used'] / stats['total_content_analyzed']) * 100
            stats['guided_analysis_rate_percent'] = round(guided_rate, 1)
            
            parsing_success_rate = (stats['parsing_successes'] / stats['total_content_analyzed']) * 100
            stats['parsing_success_rate_percent'] = round(parsing_success_rate, 1)
            
            section_aware_rate = (stats['section_aware_parsing_used'] / stats['total_content_analyzed']) * 100
            stats['section_aware_parsing_rate_percent'] = round(section_aware_rate, 1)
        else:
            stats['guided_analysis_rate_percent'] = 0
            stats['parsing_success_rate_percent'] = 0
            stats['section_aware_parsing_rate_percent'] = 0
        
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
        Log a comprehensive summary of all Polish law content analysis operations.
        
        This provides visibility into how well the guided analysis approach
        is working for Polish law documents and helps identify opportunities for optimization.
        """
        stats = self.get_analysis_statistics()
        
        self.logger.info("=== POLISH LAW CONTENT ANALYSIS SUMMARY ===")
        self.logger.info(f"Total content analyzed: {stats['total_content_analyzed']}")
        self.logger.info(f"Guided analysis used: {stats['guided_analysis_used']} ({stats['guided_analysis_rate_percent']}%)")
        self.logger.info(f"Simple analysis used: {stats['simple_analysis_used']}")
        self.logger.info(f"Parsing success rate: {stats['parsing_success_rate_percent']}%")
        self.logger.info(f"Quote location success rate: {stats['quote_location_success_rate_percent']}%")
        self.logger.info(f"Quote locations found: {stats['quote_locations_found']}")
        
        # Log Polish law-specific metrics
        self.logger.info("Polish law-specific analysis metrics:")
        self.logger.info(f"  - Section-aware parsing used: {stats['section_aware_parsing_used']} ({stats['section_aware_parsing_rate_percent']}%)")
        self.logger.info(f"  - Polish terminology detected: {stats['polish_terminology_detected']} instances")
        self.logger.info(f"  - Gazette context detected: {stats['gazette_context_detected']} references")
        
        self.logger.info(f"Analysis method effectiveness demonstrates Polish law metadata guidance value")


def create_polish_law_content_analyzer(logger: logging.Logger) -> PolishLawContentAnalyzer:
    """
    Factory function to create a configured Polish law content analyzer.
    
    This provides a clean interface for creating analyzer instances with
    proper dependency injection of the logger.
    """
    return PolishLawContentAnalyzer(logger)