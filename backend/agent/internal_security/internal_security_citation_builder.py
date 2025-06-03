"""
Internal Security Citation Builder

This module creates precise security procedure citations by combining information from all the previous
components in the pipeline. It represents the culmination of your sophisticated architectural
approach, taking the structural analysis and creating the precise references that make your
system stand out for security procedure implementation guidance.

The citation builder demonstrates how architectural layers build upon each other:
- Vector Store Connector retrieves documents with metadata
- Metadata Processor reconstructs procedural information
- Content Analyzer identifies precise locations within structure  
- Citation Builder synthesizes everything into perfect citations

This is where all the sophisticated procedural metadata preservation and analysis pays off,
creating citations like "Procedure 3.1: User Account Management Process, Step 2 - Configure Access Controls"
that provide exact implementation reference points for security procedures.
"""

import logging
from typing import Dict, List, Any, Optional


class InternalSecurityCitationBuilder:
    """
    Creates precise security procedure citations using structural analysis and metadata reconstruction.
    
    This class represents the sophisticated endpoint of your refactored architecture for security procedures.
    It takes the structural analysis from the content analyzer and the reconstructed
    procedural metadata from the metadata processor to create the most precise citations possible
    for implementation guidance and security workflow references.
    
    The citation builder demonstrates how your flattened procedural metadata approach enables
    precise functionality while maintaining vector database compatibility. All the
    complex work done by previous components enables this class to create citations
    that would be impossible with basic document retrieval approaches for security procedures.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize the internal security citation builder.
        
        Args:
            logger: Configured logger for tracking citation building operations
        """
        self.logger = logger
        self.logger.info("Internal Security Citation Builder initialized")
        
        # Track citation building statistics across all operations
        self.citation_stats = {
            'total_citations_built': 0,
            'maximum_precision_citations': 0,  # Procedure + step + sub-step
            'step_precision_citations': 0,     # Procedure + step
            'procedure_precision_citations': 0, # Procedure only
            'fallback_citations': 0,
            'citation_errors': 0,
            'precision_levels': {
                'maximum': 0,
                'step': 0,
                'procedure': 0,
                'fallback': 0
            },
            'workflow_types_handled': {}
        }
    
    def create_precise_procedure_citation(self, reconstructed_metadata: Dict[str, Any], 
                                        content: str, quote: str, 
                                        content_analysis: Dict[str, Any]) -> str:
        """
        Create a precise security procedure citation using all available structural information.
        
        This method represents the culmination of your sophisticated approach for security procedures. It combines
        the reconstructed procedural metadata from the metadata processor with the structural analysis
        from the content analyzer to create the most precise citation possible for implementation guidance.
        
        The method demonstrates how architectural layers work together to achieve
        sophisticated functionality that would be impossible with any single component for security workflows.
        
        Args:
            reconstructed_metadata: Metadata reconstructed by the metadata processor
            content: Original document content for verification
            quote: Specific quote to locate within the structure
            content_analysis: Structure analysis from the content analyzer
            
        Returns:
            Precise citation string with maximum available detail for implementation reference
        """
        self.logger.info("Creating precise security procedure citation using comprehensive structural analysis")
        self.citation_stats['total_citations_built'] += 1
        
        try:
            # Step 1: Attempt to locate the quote within the analyzed structure
            quote_location = self._locate_quote_with_analysis(quote, content_analysis)
            
            # Step 2: Build the citation using all available information
            citation_components = self._build_citation_components(
                reconstructed_metadata, quote_location, content_analysis
            )
            
            # Step 3: Assemble the final citation with appropriate precision level
            final_citation = self._assemble_final_procedure_citation(citation_components, quote_location)
            
            # Step 4: Log the citation creation success with precision analysis
            self._log_citation_success(final_citation, quote_location, citation_components)
            
            return final_citation
            
        except Exception as e:
            self.citation_stats['citation_errors'] += 1
            self.logger.error(f"Error creating precise security procedure citation: {e}")
            # Return fallback citation to ensure system continues operating
            return self._create_fallback_procedure_citation(reconstructed_metadata)
    
    def _locate_quote_with_analysis(self, quote: str, content_analysis: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Locate the quote using the sophisticated content analysis results.
        
        This method leverages the structural analysis performed by the content analyzer
        to find the exact location of a quote within the security procedure structure. This is
        where the guided parsing approach pays off with precise location identification for implementation steps.
        """
        if not content_analysis.get('parsing_successful', False):
            self.logger.debug("Content analysis was not successful - cannot locate quote precisely")
            return None
        
        try:
            # Use the content analyzer's structural map to find the quote
            implementation_steps = content_analysis.get('implementation_steps', {})
            
            if not implementation_steps:
                self.logger.debug("No implementation step structure available for quote location")
                return None
            
            # Search through the analyzed structure
            location_result = self._search_analyzed_procedure_structure(quote, implementation_steps)
            
            if location_result:
                self.logger.debug(f"Quote located using content analysis: {location_result}")
            else:
                self.logger.debug("Quote not found in analyzed procedure structure")
            
            return location_result
            
        except Exception as e:
            self.logger.warning(f"Error locating quote with analysis: {e}")
            return None
    
    def _search_analyzed_procedure_structure(self, quote: str, implementation_steps: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Search the analyzed implementation step structure for the specific quote.
        
        This method performs hierarchical search through the structure identified
        by the content analyzer, providing maximum precision when possible for security procedure references.
        """
        # Normalize quote for consistent matching
        clean_quote = ' '.join(quote.split()).lower()
        
        if len(clean_quote) < 10:
            self.logger.debug("Quote too short for reliable structure-based location")
            return None
        
        # Search through analyzed implementation steps
        for step_num, step_data in implementation_steps.items():
            # Check main step text
            step_text = ' '.join(step_data.get('full_text', '').split()).lower()
            
            if clean_quote in step_text:
                # Check sub-steps for maximum precision
                sub_steps = step_data.get('sub_steps', {})
                
                for sub_step_key, sub_step_data in sub_steps.items():
                    sub_step_text = ' '.join(sub_step_data.get('text', '').split()).lower()
                    
                    if clean_quote in sub_step_text:
                        # Maximum precision: found in specific sub-step
                        return {
                            'step': step_num,
                            'sub_step': sub_step_key,
                            'location_type': 'sub_step',
                            'confidence': 'high',
                            'pattern_type': sub_step_data.get('pattern_type', 'unknown')
                        }
                
                # Medium precision: found in main step
                return {
                    'step': step_num,
                    'sub_step': None,
                    'location_type': 'main_step',
                    'confidence': 'medium'
                }
        
        return None
    
    def _build_citation_components(self, reconstructed_metadata: Dict[str, Any], 
                                  quote_location: Optional[Dict[str, str]],
                                  content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build all the components needed for a comprehensive security procedure citation.
        
        This method extracts and organizes all the available procedural information
        to prepare for creating the most detailed citation possible. It demonstrates
        how the metadata reconstruction enables sophisticated citation creation for security procedures.
        """
        components = {
            'procedure_number': reconstructed_metadata.get('procedure_number', ''),
            'procedure_title': reconstructed_metadata.get('procedure_title', ''),
            'section_number': reconstructed_metadata.get('section_number', ''),
            'section_title': reconstructed_metadata.get('section_title', ''),
            'policy_reference': reconstructed_metadata.get('policy_reference', ''),
            'classification_level': reconstructed_metadata.get('classification_level', ''),
            'has_quote_location': quote_location is not None,
            'quote_location': quote_location,
            'analysis_method': content_analysis.get('analysis_method', 'unknown'),
            'patterns_detected': content_analysis.get('patterns_detected', []),
            'workflow_type': content_analysis.get('workflow_type', 'sequential')
        }
        
        # Determine the highest precision level possible
        if quote_location:
            if quote_location.get('sub_step'):
                components['precision_level'] = 'maximum'
            elif quote_location.get('step'):
                components['precision_level'] = 'step'
            else:
                components['precision_level'] = 'procedure'
        else:
            components['precision_level'] = 'procedure'
        
        self.logger.debug(f"Built citation components with {components['precision_level']} precision level")
        
        return components
    
    def _assemble_final_procedure_citation(self, components: Dict[str, Any], 
                                         quote_location: Optional[Dict[str, str]]) -> str:
        """
        Assemble the final citation string using all available components.
        
        This method creates the precise citation format that your system is known for,
        combining procedure information with structural details when available. The
        method demonstrates how all the sophisticated processing enables precise output for security procedures.
        """
        citation_parts = []
        precision_level = components['precision_level']
        
        # Build the core procedure reference
        procedure_num = components['procedure_number']
        procedure_title = components['procedure_title']
        
        if procedure_num:
            if quote_location:
                # Add structural information based on location analysis
                step_num = quote_location.get('step')
                sub_step_key = quote_location.get('sub_step')
                pattern_type = quote_location.get('pattern_type', '')
                
                if sub_step_key and precision_level == 'maximum':
                    # Maximum precision: Procedure, step, and sub-step with pattern type
                    if pattern_type and pattern_type != 'unknown':
                        if procedure_title:
                            citation_parts.append(f"Procedure {procedure_num}: {procedure_title}, Step {step_num} - {pattern_type.title()} ({sub_step_key})")
                        else:
                            citation_parts.append(f"Procedure {procedure_num}, Step {step_num} - {pattern_type.title()} ({sub_step_key})")
                    else:
                        if procedure_title:
                            citation_parts.append(f"Procedure {procedure_num}: {procedure_title}, Step {step_num} - {sub_step_key}")
                        else:
                            citation_parts.append(f"Procedure {procedure_num}, Step {step_num} - {sub_step_key}")
                    
                    self.citation_stats['maximum_precision_citations'] += 1
                    self.citation_stats['precision_levels']['maximum'] += 1
                    self.logger.debug(f"Created maximum precision citation: Procedure {procedure_num}, Step {step_num}, Sub-step {sub_step_key}")
                    
                elif step_num and precision_level in ['step', 'maximum']:
                    # Step precision: Procedure and step
                    if procedure_title:
                        citation_parts.append(f"Procedure {procedure_num}: {procedure_title}, Step {step_num}")
                    else:
                        citation_parts.append(f"Procedure {procedure_num}, Step {step_num}")
                    
                    self.citation_stats['step_precision_citations'] += 1
                    self.citation_stats['precision_levels']['step'] += 1
                    self.logger.debug(f"Created step precision citation: Procedure {procedure_num}, Step {step_num}")
                    
                else:
                    # Procedure precision only
                    if procedure_title:
                        citation_parts.append(f"Procedure {procedure_num}: {procedure_title}")
                    else:
                        citation_parts.append(f"Procedure {procedure_num}")
                    
                    self.citation_stats['procedure_precision_citations'] += 1
                    self.citation_stats['precision_levels']['procedure'] += 1
                    self.logger.debug(f"Created procedure precision citation: Procedure {procedure_num}")
            else:
                # No location information - procedure only
                if procedure_title:
                    citation_parts.append(f"Procedure {procedure_num}: {procedure_title}")
                else:
                    citation_parts.append(f"Procedure {procedure_num}")
                
                self.citation_stats['procedure_precision_citations'] += 1
                self.citation_stats['precision_levels']['procedure'] += 1
        
        # Add section information for complete context
        section_num = components['section_number']
        section_title = components['section_title']
        
        if section_num and section_title:
            section_info = f"Section {section_num}: {section_title}"
            citation_parts.append(f"({section_info})")
            self.logger.debug(f"Added section context: {section_info}")
        
        # Add policy reference if available
        policy_ref = components['policy_reference']
        if policy_ref:
            citation_parts.append(f"[Policy: {policy_ref}]")
            self.logger.debug(f"Added policy reference: {policy_ref}")
        
        # Add classification level for security procedures if available
        classification = components['classification_level']
        if classification and classification.lower() != 'unclassified':
            citation_parts.append(f"[{classification}]")
            self.logger.debug(f"Added classification level: {classification}")
        
        # Track workflow types handled
        workflow_type = components['workflow_type']
        if workflow_type:
            self.citation_stats['workflow_types_handled'][workflow_type] = \
                self.citation_stats['workflow_types_handled'].get(workflow_type, 0) + 1
        
        # Combine all parts into the final citation
        final_citation = " ".join(citation_parts) if citation_parts else f"Internal Security Procedures - Procedure {procedure_num or 'Unknown'}"
        
        return final_citation
    
    def _create_fallback_procedure_citation(self, reconstructed_metadata: Dict[str, Any]) -> str:
        """
        Create a fallback citation when sophisticated processing fails.
        
        This method ensures that the system continues to function even when
        the advanced features encounter issues, demonstrating graceful degradation
        while maintaining basic functionality for security procedures.
        """
        self.citation_stats['fallback_citations'] += 1
        self.citation_stats['precision_levels']['fallback'] += 1
        
        procedure_num = reconstructed_metadata.get('procedure_number', 'Unknown')
        procedure_title = reconstructed_metadata.get('procedure_title', '')
        section_num = reconstructed_metadata.get('section_number', '')
        section_title = reconstructed_metadata.get('section_title', '')
        
        fallback_citation = f"Internal Security Procedures - Procedure {procedure_num}"
        
        if procedure_title:
            fallback_citation += f": {procedure_title}"
        
        if section_num and section_title:
            fallback_citation += f" (Section {section_num}: {section_title})"
        
        self.logger.info(f"Created fallback procedure citation: {fallback_citation}")
        
        return fallback_citation
    
    def _log_citation_success(self, final_citation: str, quote_location: Optional[Dict[str, str]], 
                             components: Dict[str, Any]) -> None:
        """
        Log successful citation creation with detailed information for monitoring.
        
        This logging provides visibility into how well the citation building process
        is working and helps identify the precision levels being achieved across
        different types of security procedure documents and queries.
        """
        precision_level = components['precision_level']
        analysis_method = components['analysis_method']
        workflow_type = components['workflow_type']
        
        self.logger.info(f"Successfully created {precision_level} precision security procedure citation: {final_citation}")
        self.logger.debug(f"Citation built using {analysis_method} analysis method")
        self.logger.debug(f"Workflow type: {workflow_type}")
        
        if quote_location:
            location_type = quote_location['location_type']
            confidence = quote_location['confidence']
            pattern_type = quote_location.get('pattern_type', 'unknown')
            self.logger.debug(f"Quote location: {location_type} with {confidence} confidence, pattern: {pattern_type}")
        
        # Log patterns that contributed to success
        patterns = components.get('patterns_detected', [])
        if patterns:
            self.logger.debug(f"Structural patterns detected: {patterns[:3]}")  # Show first 3 patterns
    
    def create_basic_procedure_citation_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Create a basic citation directly from metadata when full processing is not available.
        
        This method provides a simplified path for creating citations when the full
        sophisticated pipeline is not needed or available. It demonstrates how the
        architecture supports both advanced and basic use cases for security procedures.
        """
        self.citation_stats['total_citations_built'] += 1
        self.citation_stats['procedure_precision_citations'] += 1
        self.citation_stats['precision_levels']['procedure'] += 1
        
        procedure_num = metadata.get('procedure_number', 'Unknown')
        procedure_title = metadata.get('procedure_title', '')
        section_num = metadata.get('section_number', '')
        section_title = metadata.get('section_title', '')
        classification = metadata.get('classification_level', '')
        
        basic_citation = f"Internal Security Procedures - Procedure {procedure_num}"
        
        if procedure_title:
            basic_citation += f": {procedure_title}"
        
        if section_num and section_title:
            basic_citation += f" (Section {section_num}: {section_title})"
        
        if classification and classification.lower() != 'unclassified':
            basic_citation += f" [{classification}]"
        
        self.logger.debug(f"Created basic procedure citation from metadata: {basic_citation}")
        
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
            for precision_type in ['maximum', 'step', 'procedure', 'fallback']:
                count = stats['precision_levels'][precision_type]
                percentage = (count / stats['total_citations_built']) * 100
                stats[f'{precision_type}_precision_percentage'] = round(percentage, 1)
            
            # Calculate overall precision score (weighted by precision level)
            weights = {'maximum': 3, 'step': 2, 'procedure': 1, 'fallback': 0}
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
        system for security procedures and helps identify patterns in precision achievement across different
        document types and analysis scenarios.
        """
        stats = self.get_citation_statistics()
        
        self.logger.info("=== INTERNAL SECURITY PROCEDURE CITATION BUILDING SUMMARY ===")
        self.logger.info(f"Total citations built: {stats['total_citations_built']}")
        self.logger.info(f"Maximum precision citations: {stats['maximum_precision_citations']} ({stats.get('maximum_precision_percentage', 0)}%)")
        self.logger.info(f"Step precision citations: {stats['step_precision_citations']} ({stats.get('step_precision_percentage', 0)}%)")
        self.logger.info(f"Procedure precision citations: {stats['procedure_precision_citations']} ({stats.get('procedure_precision_percentage', 0)}%)")
        self.logger.info(f"Fallback citations: {stats['fallback_citations']} ({stats.get('fallback_precision_percentage', 0)}%)")
        self.logger.info(f"Overall precision score: {stats.get('overall_precision_score', 0)}%")
        self.logger.info(f"Citation errors: {stats['citation_errors']}")
        
        # Log workflow types handled
        if stats['workflow_types_handled']:
            self.logger.info("Workflow types processed:")
            for workflow_type, count in sorted(stats['workflow_types_handled'].items()):
                self.logger.info(f"  - {workflow_type}: {count} citations")
        
        # Provide interpretation of the precision score
        precision_score = stats.get('overall_precision_score', 0)
        if precision_score >= 80:
            self.logger.info("Excellent citation precision - sophisticated analysis working optimally for security procedures")
        elif precision_score >= 60:
            self.logger.info("Good citation precision - most security procedure documents analyzed successfully")
        elif precision_score >= 40:
            self.logger.info("Moderate citation precision - consider optimizing procedural metadata quality")
        else:
            self.logger.info("Low citation precision - review processing pipeline and procedural metadata structure")


def create_internal_security_citation_builder(logger: logging.Logger) -> InternalSecurityCitationBuilder:
    """
    Factory function to create a configured internal security citation builder.
    
    This provides a clean interface for creating citation builder instances with
    proper dependency injection of the logger.
    """
    return InternalSecurityCitationBuilder(logger)