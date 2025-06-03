"""
Internal Security Content Analyzer

This module performs intelligent analysis of security procedure document content using the structural
hints provided by the metadata processor. Instead of blind text parsing, this analyzer
uses the reconstructed procedural metadata to guide its analysis, making it much more accurate
and efficient for security procedure implementation workflows.

Think of this as having a "blueprint" while exploring a complex security implementation - the metadata
tells us what implementation steps to expect, making navigation much more reliable than parsing randomly.

The analyzer demonstrates how your architectural approach creates powerful synergies:
- Processing Pipeline creates flattened procedural metadata
- Metadata Processor reconstructs structural hints  
- Content Analyzer uses hints for intelligent parsing
- Citation Builder uses parsed structure for precise references

This approach is specifically adapted for security procedures, understanding implementation steps,
configuration requirements, tool dependencies, and workflow patterns.
"""

import re
import logging
from typing import Dict, List, Any, Optional


class InternalSecurityContentAnalyzer:
    """
    Analyzes security procedure document content using intelligent, metadata-guided parsing.
    
    This class demonstrates how your flattened procedural metadata approach enables sophisticated
    content analysis for security procedures. Instead of generic text parsing, we use the structural hints
    from the metadata processor to guide our analysis, making it much more accurate
    and reliable than traditional approaches for implementation workflows.
    
    The analyzer adapts its strategy based on the available procedural metadata, using the most
    sophisticated approach possible while gracefully falling back when needed for various
    complexity levels of security procedures.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security content analyzer.
        
        Args:
            logger: Configured logger for tracking content analysis operations
        """
        self.logger = logger
        self.logger.info("Internal Security Content Analyzer initialized")
        
        # Track analysis statistics across all operations
        self.analysis_stats = {
            'total_content_analyzed': 0,
            'guided_analysis_used': 0,
            'simple_analysis_used': 0,
            'parsing_successes': 0,
            'parsing_failures': 0,
            'step_locations_found': 0,
            'step_locations_failed': 0,
            'workflow_patterns_detected': {}
        }
    
    def analyze_procedural_content_structure(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze security procedure content structure using intelligent, metadata-guided parsing.
        
        This method demonstrates the power of your architectural approach for security procedures. Instead of
        parsing content blindly, we use the structural hints from the metadata processor
        to guide our analysis, making it much more accurate and efficient for implementation
        workflows than generic text parsing approaches.
        
        Args:
            content: Security procedure document content to analyze
            processing_hints: Structural hints from metadata processor
            
        Returns:
            Dictionary containing parsed content structure and analysis metadata
        """
        self.logger.debug("Starting intelligent procedural content structure analysis")
        self.analysis_stats['total_content_analyzed'] += 1
        
        # Choose analysis strategy based on available hints for security procedures
        if processing_hints.get('has_hints', False) and processing_hints.get('use_guided_parsing', False):
            return self._perform_guided_procedural_analysis(content, processing_hints)
        else:
            return self._perform_simple_procedural_analysis(content)
    
    def _perform_guided_procedural_analysis(self, content: str, processing_hints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform guided content analysis using procedural metadata hints.
        
        This method demonstrates how your flattened procedural metadata creates a competitive
        advantage for security procedures. Instead of generic parsing, we use the specific implementation
        information preserved from the original processing to guide our analysis for security workflows.
        """
        self.logger.debug("Using guided analysis with procedural metadata hints")
        self.analysis_stats['guided_analysis_used'] += 1
        
        # Initialize content map with guided parsing approach
        content_map = {
            'implementation_steps': {},
            'parsing_successful': False,
            'analysis_method': 'guided',
            'hints_used': processing_hints.copy(),
            'patterns_detected': [],
            'workflow_type': processing_hints.get('workflow_type', 'sequential')
        }
        
        # Extract guidance parameters from hints
        step_count = processing_hints.get('implementation_step_count', 0)
        has_sub_steps = processing_hints.get('has_sub_steps', False)
        workflow_type = processing_hints.get('workflow_type', 'sequential')
        expected_patterns = processing_hints.get('expected_patterns', [])
        
        self.logger.debug(f"Analysis guidance: {step_count} implementation steps expected, "
                        f"sub-steps: {has_sub_steps}, workflow: {workflow_type}")
        
        try:
            # Use specialized parsing based on expected procedural structure
            if has_sub_steps and workflow_type in ['conditional', 'parallel']:
                success = self._parse_with_complex_workflow_guidance(
                    content, content_map, step_count, workflow_type, expected_patterns
                )
            elif has_sub_steps:
                success = self._parse_with_sub_step_guidance(
                    content, content_map, step_count, expected_patterns
                )
            else:
                success = self._parse_with_step_guidance(
                    content, content_map, step_count, expected_patterns
                )
            
            if success:
                content_map['parsing_successful'] = True
                self.analysis_stats['parsing_successes'] += 1
                self.logger.debug(f"Guided procedural analysis successful: found {len(content_map['implementation_steps'])} steps")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Guided procedural analysis failed - structure not as expected")
                # Fall back to simple analysis
                return self._perform_simple_procedural_analysis(content)
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in guided procedural analysis: {e}")
            # Fall back to simple analysis on error
            return self._perform_simple_procedural_analysis(content)
        
        return content_map
    
    def _parse_with_complex_workflow_guidance(self, content: str, content_map: Dict[str, Any],
                                            step_count: int, workflow_type: str,
                                            expected_patterns: List[str]) -> bool:
        """
        Parse content with complex workflow guidance from procedural metadata hints.
        
        This method uses the specific workflow type information preserved in your
        flattened metadata to look for conditional logic, parallel execution paths,
        and other complex implementation patterns specific to security procedures.
        """
        self.logger.debug(f"Parsing with complex workflow guidance: type={workflow_type}")
        
        lines = content.split('\n')
        current_step = None
        current_sub_step = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for main implementation step markers using expected count for validation
            step_match = re.match(r'^(?:Step\s+)?(\d+)[:.]?\s+(.+)', line, re.IGNORECASE)
            if step_match:
                step_num = step_match.group(1)
                step_description = step_match.group(2)
                
                # Validate against expected count from metadata
                if int(step_num) <= step_count:
                    content_map['implementation_steps'][step_num] = {
                        'start_line': line_idx,
                        'description': step_description,
                        'full_text': line,
                        'sub_steps': {},
                        'workflow_indicators': []
                    }
                    current_step = step_num
                    current_sub_step = None
                    
                    content_map['patterns_detected'].append(f'main_step_{step_num}')
                    self.logger.debug(f"Found expected main step {step_num}")
                    continue
            
            # Look for workflow-specific patterns when in a step context
            if current_step:
                workflow_pattern = self._detect_workflow_patterns_with_type(
                    line, workflow_type, expected_patterns
                )
                
                if workflow_pattern:
                    pattern_type = workflow_pattern['type']
                    pattern_description = workflow_pattern['description']
                    
                    content_map['implementation_steps'][current_step]['workflow_indicators'].append({
                        'type': pattern_type,
                        'description': pattern_description,
                        'line': line_idx
                    })
                    
                    content_map['patterns_detected'].append(f'workflow_{current_step}_{pattern_type}')
                    self.logger.debug(f"Found workflow pattern {pattern_type} in step {current_step}")
                    continue
                
                # Look for sub-steps using security procedure patterns
                sub_step_location = self._detect_security_sub_steps(
                    line, expected_patterns
                )
                
                if sub_step_location:
                    sub_step_key = sub_step_location['key']
                    sub_step_text = sub_step_location['text']
                    
                    content_map['implementation_steps'][current_step]['sub_steps'][sub_step_key] = {
                        'start_line': line_idx,
                        'text': sub_step_text,
                        'full_line': line,
                        'pattern_type': sub_step_location.get('pattern_type', 'unknown')
                    }
                    current_sub_step = sub_step_key
                    
                    content_map['patterns_detected'].append(f'sub_step_{current_step}_{sub_step_key}')
                    self.logger.debug(f"Found sub-step {current_step}.{sub_step_key}")
                    continue
                
                # Handle continuation text
                if current_sub_step:
                    content_map['implementation_steps'][current_step]['sub_steps'][current_sub_step]['text'] += ' ' + line
                else:
                    content_map['implementation_steps'][current_step]['full_text'] += ' ' + line
        
        # Validate that we found the expected procedural structure
        found_steps = len(content_map['implementation_steps'])
        expected_steps = step_count
        
        # Track workflow patterns in statistics
        self.analysis_stats['workflow_patterns_detected'][workflow_type] = \
            self.analysis_stats['workflow_patterns_detected'].get(workflow_type, 0) + 1
        
        if found_steps > 0:
            self.logger.debug(f"Guided complex workflow parsing: found {found_steps} steps "
                            f"(expected {expected_steps})")
            return True
        else:
            self.logger.warning("Guided complex workflow parsing found no structure")
            return False
    
    def _detect_workflow_patterns_with_type(self, line: str, workflow_type: str,
                                          expected_patterns: List[str]) -> Optional[Dict[str, str]]:
        """
        Detect workflow patterns using the specific workflow type from procedural metadata.
        
        This method demonstrates how your metadata preservation enables precise
        pattern matching for security procedures. We know exactly what workflow type to look for based on the
        implementation information preserved during processing.
        """
        line_lower = line.lower()
        
        # Handle conditional workflow patterns
        if workflow_type == 'conditional' or 'conditional_sub_steps' in expected_patterns:
            # Look for if-then-else logic
            if re.search(r'\b(if|when|unless|provided that)\b', line_lower):
                return {
                    'type': 'conditional_start',
                    'description': 'Conditional logic entry point'
                }
            elif re.search(r'\b(then|otherwise|else|alternatively)\b', line_lower):
                return {
                    'type': 'conditional_branch',
                    'description': 'Conditional logic branch'
                }
        
        # Handle parallel workflow patterns
        if workflow_type == 'parallel' or 'parallel_sub_steps' in expected_patterns:
            if re.search(r'\b(simultaneously|concurrently|in parallel|at the same time)\b', line_lower):
                return {
                    'type': 'parallel_execution',
                    'description': 'Parallel execution indicator'
                }
        
        # Handle automation workflow patterns
        if workflow_type == 'automated' or 'automation_workflow' in expected_patterns:
            if re.search(r'\b(automated|script|scheduled|automatic)\b', line_lower):
                return {
                    'type': 'automation_step',
                    'description': 'Automated execution step'
                }
        
        return None
    
    def _detect_security_sub_steps(self, line: str, expected_patterns: List[str]) -> Optional[Dict[str, str]]:
        """
        Detect sub-steps using security procedure-specific patterns.
        
        This method identifies patterns that are characteristic of security procedure
        implementation, such as configuration steps, validation requirements, and
        verification processes that are essential for security workflows.
        """
        # Security procedure configuration patterns
        config_match = re.match(r'^(Configure|Set|Enable|Disable|Create|Deploy|Install)\s+(.+)', line, re.IGNORECASE)
        if config_match:
            action = config_match.group(1)
            description = config_match.group(2)
            return {
                'key': f"{action.lower()}_{hash(description) % 1000}",
                'text': description,
                'pattern_type': 'configuration'
            }
        
        # Security verification patterns
        verify_match = re.match(r'^(Verify|Validate|Check|Confirm|Test)\s+(.+)', line, re.IGNORECASE)
        if verify_match:
            action = verify_match.group(1)
            description = verify_match.group(2)
            return {
                'key': f"{action.lower()}_{hash(description) % 1000}",
                'text': description,
                'pattern_type': 'verification'
            }
        
        # Security monitoring patterns
        monitor_match = re.match(r'^(Monitor|Log|Alert|Report|Audit)\s+(.+)', line, re.IGNORECASE)
        if monitor_match:
            action = monitor_match.group(1)
            description = monitor_match.group(2)
            return {
                'key': f"{action.lower()}_{hash(description) % 1000}",
                'text': description,
                'pattern_type': 'monitoring'
            }
        
        # Generic sub-step patterns with enumeration
        enum_match = re.match(r'^([a-z])\)\s+(.+)', line, re.IGNORECASE)
        if enum_match:
            enum_key = enum_match.group(1).lower()
            description = enum_match.group(2)
            return {
                'key': enum_key,
                'text': description,
                'pattern_type': 'enumerated'
            }
        
        # Numbered sub-step patterns
        num_match = re.match(r'^(\d+)\.\s+(.+)', line)
        if num_match:
            num_key = num_match.group(1)
            description = num_match.group(2)
            return {
                'key': f"substep_{num_key}",
                'text': description,
                'pattern_type': 'numbered'
            }
        
        return None
    
    def _parse_with_sub_step_guidance(self, content: str, content_map: Dict[str, Any],
                                    step_count: int, expected_patterns: List[str]) -> bool:
        """
        Parse content with sub-step guidance for moderate complexity security procedures.
        
        This method handles security procedures that have sub-steps but simpler workflow patterns,
        using the expected step count to validate the parsing results for implementation accuracy.
        """
        self.logger.debug(f"Parsing with sub-step guidance: expecting {step_count} steps with sub-steps")
        
        lines = content.split('\n')
        current_step = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for main implementation step markers
            step_match = re.match(r'^(?:Step\s+)?(\d+)[:.]?\s+(.+)', line, re.IGNORECASE)
            if step_match:
                step_num = step_match.group(1)
                step_description = step_match.group(2)
                
                # Validate against expected count
                if int(step_num) <= step_count:
                    content_map['implementation_steps'][step_num] = {
                        'start_line': line_idx,
                        'description': step_description,
                        'full_text': line,
                        'sub_steps': {}
                    }
                    current_step = step_num
                    
                    content_map['patterns_detected'].append(f'step_{step_num}')
                    self.logger.debug(f"Found step {step_num}")
                    continue
            
            # Handle sub-step detection when in step context
            if current_step:
                sub_step_location = self._detect_security_sub_steps(line, expected_patterns)
                
                if sub_step_location:
                    sub_step_key = sub_step_location['key']
                    sub_step_text = sub_step_location['text']
                    
                    content_map['implementation_steps'][current_step]['sub_steps'][sub_step_key] = {
                        'start_line': line_idx,
                        'text': sub_step_text,
                        'full_line': line,
                        'pattern_type': sub_step_location.get('pattern_type', 'unknown')
                    }
                    
                    content_map['patterns_detected'].append(f'sub_step_{current_step}_{sub_step_key}')
                    self.logger.debug(f"Found sub-step {current_step}.{sub_step_key}")
                    continue
                
                # Handle continuation text
                content_map['implementation_steps'][current_step]['full_text'] += ' ' + line
        
        found_steps = len(content_map['implementation_steps'])
        
        if found_steps > 0:
            self.logger.debug(f"Guided sub-step parsing: found {found_steps} steps")
            return True
        else:
            self.logger.warning("Guided sub-step parsing found no structure")
            return False
    
    def _parse_with_step_guidance(self, content: str, content_map: Dict[str, Any],
                                step_count: int, expected_patterns: List[str]) -> bool:
        """
        Parse content with step-level guidance for simpler security procedures.
        
        This method handles security procedures that have multiple steps but no sub-steps,
        using the expected step count to validate the parsing results for basic implementation guidance.
        """
        self.logger.debug(f"Parsing with step guidance: expecting {step_count} implementation steps")
        
        lines = content.split('\n')
        current_step = None
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Look for implementation step markers
            step_match = re.match(r'^(?:Step\s+)?(\d+)[:.]?\s+(.+)', line, re.IGNORECASE)
            if step_match:
                step_num = step_match.group(1)
                step_description = step_match.group(2)
                
                # Validate against expected count
                if int(step_num) <= step_count:
                    content_map['implementation_steps'][step_num] = {
                        'start_line': line_idx,
                        'description': step_description,
                        'full_text': line,
                        'sub_steps': {}
                    }
                    current_step = step_num
                    
                    content_map['patterns_detected'].append(f'step_{step_num}')
                    self.logger.debug(f"Found step {step_num}")
                    continue
            
            # Handle continuation text
            if current_step:
                content_map['implementation_steps'][current_step]['full_text'] += ' ' + line
        
        found_steps = len(content_map['implementation_steps'])
        
        if found_steps > 0:
            self.logger.debug(f"Guided step parsing: found {found_steps} steps")
            return True
        else:
            self.logger.warning("Guided step parsing found no structure")
            return False
    
    def _perform_simple_procedural_analysis(self, content: str) -> Dict[str, Any]:
        """
        Perform simple procedural analysis without metadata guidance.
        
        This method provides reliable parsing even when enhanced procedural metadata is not
        available, ensuring the system works gracefully across all security procedure document types.
        It demonstrates how your architecture gracefully degrades while maintaining
        functionality for basic implementation guidance.
        """
        self.logger.debug("Using simple analysis without procedural metadata guidance")
        self.analysis_stats['simple_analysis_used'] += 1
        
        content_map = {
            'implementation_steps': {},
            'parsing_successful': False,
            'analysis_method': 'simple',
            'hints_used': {},
            'patterns_detected': []
        }
        
        try:
            lines = content.split('\n')
            current_step = None
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for basic step patterns without guidance
                step_match = re.match(r'^(?:Step\s+)?(\d+)[:.]?\s+(.+)', line, re.IGNORECASE)
                if step_match:
                    step_num = step_match.group(1)
                    step_description = step_match.group(2)
                    
                    content_map['implementation_steps'][step_num] = {
                        'start_line': line_idx,
                        'description': step_description,
                        'full_text': line,
                        'sub_steps': {}
                    }
                    current_step = step_num
                    content_map['patterns_detected'].append(f'simple_step_{step_num}')
                    continue
                
                # Look for sub-steps using general security patterns
                if current_step:
                    sub_step_location = self._detect_security_sub_steps(line, [])
                    
                    if sub_step_location:
                        sub_step_key = sub_step_location['key']
                        sub_step_text = sub_step_location['text']
                        
                        content_map['implementation_steps'][current_step]['sub_steps'][sub_step_key] = {
                            'start_line': line_idx,
                            'text': sub_step_text,
                            'full_line': line,
                            'pattern_type': sub_step_location.get('pattern_type', 'detected')
                        }
                        content_map['patterns_detected'].append(f'simple_sub_step_{current_step}_{sub_step_key}')
                        continue
                
                # Handle continuation text
                if current_step:
                    content_map['implementation_steps'][current_step]['full_text'] += ' ' + line
            
            if len(content_map['implementation_steps']) > 0:
                content_map['parsing_successful'] = True
                self.analysis_stats['parsing_successes'] += 1
                self.logger.debug(f"Simple procedural analysis successful: found {len(content_map['implementation_steps'])} steps")
            else:
                self.analysis_stats['parsing_failures'] += 1
                self.logger.warning("Simple procedural analysis found no step structure")
            
        except Exception as e:
            self.analysis_stats['parsing_failures'] += 1
            self.logger.warning(f"Error in simple procedural content analysis: {e}")
        
        return content_map
    
    def locate_quote_in_procedural_structure(self, quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate a specific quote within the analyzed security procedure structure.
        
        This method demonstrates how the structured analysis enables precise quote
        location for security procedures, which is essential for creating accurate implementation citations. The method
        provides different levels of precision based on what the analysis discovered.
        """
        self.logger.debug(f"Locating quote in analyzed procedural structure: '{quote[:50]}...'")
        
        if not content_map.get('parsing_successful', False):
            self.logger.warning("Cannot locate quote - procedural content analysis was not successful")
            return None
        
        # Clean and normalize quote for better matching
        clean_quote = self._normalize_quote_for_matching(quote)
        if not clean_quote:
            return None
        
        try:
            # Search through the analyzed structure with detailed logging
            location_result = self._search_procedural_structure_for_quote(clean_quote, content_map)
            
            if location_result:
                self.analysis_stats['step_locations_found'] += 1
                self._log_quote_location_success(location_result, quote)
            else:
                self.analysis_stats['step_locations_failed'] += 1
                self.logger.warning("Could not locate quote in analyzed procedural document structure")
            
            return location_result
            
        except Exception as e:
            self.analysis_stats['step_locations_failed'] += 1
            self.logger.warning(f"Error locating quote in procedural structure: {e}")
            return None
    
    def _normalize_quote_for_matching(self, quote: str) -> Optional[str]:
        """
        Normalize a quote for reliable matching within the security procedure structure.
        
        This method cleans up the quote text to improve matching reliability
        while maintaining enough content for accurate identification in implementation contexts.
        """
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.warning(f"Quote too short for reliable procedural matching: '{quote}'")
            return None
        
        return clean_quote
    
    def _search_procedural_structure_for_quote(self, clean_quote: str, content_map: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search through the analyzed procedural structure to find the quote location.
        
        This method provides hierarchical search, checking sub-steps first
        for maximum precision, then falling back to step-level location for implementation guidance.
        """
        for step_num, step_data in content_map['implementation_steps'].items():
            # Check main step text
            step_text = ' '.join(step_data.get('full_text', '').split()).lower()
            
            if clean_quote in step_text:
                # Found in step - check for sub-step specificity
                sub_step_location = self._check_sub_steps_for_quote(
                    clean_quote, step_data, step_num
                )
                
                if sub_step_location:
                    return sub_step_location
                
                # Quote found in main step but not in specific sub-step
                self.logger.debug(f"Quote located in main implementation step {step_num}")
                return {
                    'step': step_num,
                    'sub_step': None,
                    'location_type': 'main_step',
                    'confidence': 'medium'
                }
        
        return None
    
    def _check_sub_steps_for_quote(self, clean_quote: str, step_data: Dict[str, Any],
                                  step_num: str) -> Optional[Dict[str, str]]:
        """
        Check sub-steps for quote location to achieve maximum precision.
        
        This method searches within sub-steps to provide the most precise
        citation possible for security procedures, enabling references like "Procedure 3.1, Step 2, Configure Access Controls".
        """
        sub_steps = step_data.get('sub_steps', {})
        
        for sub_step_key, sub_step_data in sub_steps.items():
            sub_step_text = ' '.join(sub_step_data.get('text', '').split()).lower()
            
            if clean_quote in sub_step_text:
                self.logger.debug(f"Quote precisely located: step {step_num}, sub-step {sub_step_key}")
                return {
                    'step': step_num,
                    'sub_step': sub_step_key,
                    'location_type': 'sub_step',
                    'confidence': 'high',
                    'pattern_type': sub_step_data.get('pattern_type', 'unknown')
                }
        
        return None
    
    def _log_quote_location_success(self, location_result: Dict[str, str], original_quote: str) -> None:
        """
        Log successful quote location with details for debugging and verification.
        
        This logging helps track the effectiveness of the procedural structure analysis
        and quote location process for optimization and debugging purposes.
        """
        location_type = location_result['location_type']
        confidence = location_result['confidence']
        step = location_result['step']
        sub_step = location_result.get('sub_step')
        
        if sub_step:
            self.logger.info(f"Quote located with {confidence} confidence: "
                           f"step {step}, sub-step {sub_step}")
        else:
            self.logger.info(f"Quote located with {confidence} confidence: step {step}")
        
        # Log quote preview for verification
        quote_preview = original_quote[:100] + "..." if len(original_quote) > 100 else original_quote
        self.logger.debug(f"Located quote: '{quote_preview}'")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about procedural content analysis operations.
        
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
        total_quote_attempts = stats['step_locations_found'] + stats['step_locations_failed']
        if total_quote_attempts > 0:
            quote_success_rate = (stats['step_locations_found'] / total_quote_attempts) * 100
            stats['quote_location_success_rate_percent'] = round(quote_success_rate, 1)
        else:
            stats['quote_location_success_rate_percent'] = 0
        
        return stats
    
    def log_analysis_summary(self) -> None:
        """
        Log a comprehensive summary of all procedural content analysis operations.
        
        This provides visibility into how well the guided analysis approach
        is working for security procedures and helps identify opportunities for optimization.
        """
        stats = self.get_analysis_statistics()
        
        self.logger.info("=== INTERNAL SECURITY PROCEDURAL CONTENT ANALYSIS SUMMARY ===")
        self.logger.info(f"Total content analyzed: {stats['total_content_analyzed']}")
        self.logger.info(f"Guided analysis used: {stats['guided_analysis_used']} ({stats['guided_analysis_rate_percent']}%)")
        self.logger.info(f"Simple analysis used: {stats['simple_analysis_used']}")
        self.logger.info(f"Parsing success rate: {stats['parsing_success_rate_percent']}%")
        self.logger.info(f"Quote location success rate: {stats['quote_location_success_rate_percent']}%")
        self.logger.info(f"Step locations found: {stats['step_locations_found']}")
        self.logger.info(f"Analysis method effectiveness demonstrates procedural metadata guidance value")
        
        # Log workflow patterns detected
        if stats['workflow_patterns_detected']:
            self.logger.info("Workflow patterns detected:")
            for workflow_type, count in stats['workflow_patterns_detected'].items():
                self.logger.info(f"  - {workflow_type}: {count} instances")


def create_internal_security_content_analyzer(logger: logging.Logger) -> InternalSecurityContentAnalyzer:
    """
    Factory function to create a configured internal security content analyzer.
    
    This provides a clean interface for creating analyzer instances with
    proper dependency injection of the logger.
    """
    return InternalSecurityContentAnalyzer(logger)